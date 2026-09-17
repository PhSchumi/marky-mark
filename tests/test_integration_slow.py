"""Integration tests: need network (first run) and a few minutes of CPU. ``pytest -m slow``."""

import math

import pytest
import torch

pytestmark = pytest.mark.slow

MODEL = "facebook/opt-125m"


@pytest.fixture(scope="module")
def tiny():
    from wmbench.bootstrap import ensure_markllm
    from wmbench.models import PromptFormatter, load_causal_lm

    ensure_markllm()
    model, tok = load_causal_lm(MODEL, "cpu", torch.float32)
    return model, tok, PromptFormatter(tok)


@pytest.mark.parametrize("name", ["none", "KGW", "MorphMark", "PF", "SynthID", "SIR"])
def test_variant_generate_and_detect(tiny, name):
    from wmbench.watermarks import build_variant

    model, tok, fmt = tiny
    v = build_variant(name, model, tok, "cpu", 32)
    ids = fmt.encode("The history of the Roman Empire is")
    torch.manual_seed(0)
    out = v.generate(ids, "The history of the Roman Empire is")
    assert 0 < len(out.continuation_ids) <= 32
    assert out.continuation.strip()
    if v.is_watermarked:
        d = v.detect(out.continuation)
        assert d.error is None and math.isfinite(d.score) and d.n_tokens > 0
        short = v.detect("Hi")
        assert math.isnan(short.score)


def test_synthid_state_reset(tiny):
    """A second generation must equal the first one from a fresh instance under the same seed."""
    from wmbench.watermarks import build_variant

    model, tok, fmt = tiny
    ids = fmt.encode("Once upon a time")
    v = build_variant("SynthID", model, tok, "cpu", 24)
    torch.manual_seed(1)
    first = v.generate(ids).continuation_ids
    torch.manual_seed(1)
    second = v.generate(ids).continuation_ids
    fresh = build_variant("SynthID", model, tok, "cpu", 24)
    torch.manual_seed(1)
    third = fresh.generate(ids).continuation_ids
    assert first == second == third


def test_pf_score_is_pvalue_based(tiny):
    from wmbench.watermarks import build_variant

    model, tok, fmt = tiny
    v = build_variant("PF", model, tok, "cpu", 40)
    out = v.generate(fmt.encode("In the beginning"))
    d = v.detect(out.continuation)
    assert "p_value" in d.raw and 0 <= d.raw["p_value"] <= 1
    assert d.score == pytest.approx(-math.log10(max(d.raw["p_value"], 1e-300)))


def test_smoke_pipeline(tmp_path):
    """All stages on 3 C4 prompts and 2 translation prompts with the tiny model."""
    from wmbench.config import BenchmarkConfig
    from wmbench.evaluate import Runner

    cfg = BenchmarkConfig.smoke_test(run_name="pytest-smoke", device="cpu", results_root=str(tmp_path),
                                     variants=("none", "KGW", "PF"), datasets=("c4", "wmt16_de_en"),
                                     n_samples={"c4": 3, "wmt16_de_en": 2, "cnn_dailymail": 1, "humaneval": 1},
                                     max_new_tokens={"c4": 24, "wmt16_de_en": 16, "cnn_dailymail": 16, "humaneval": 16},
                                     min_detect_tokens=4, run_robustness=True, attacks=("word_deletion",),
                                     attack_similarity_model=None)
    r = Runner(cfg, show_progress=False)
    r.run()
    out = r.stage_report()
    assert (cfg.run_dir / "summary.md").exists()
    assert set(out["main"].variant) >= {"none", "KGW", "PF"}
    # length control: every text kind of a sample was scored at the same common token budget
    import json

    n_det_rows = {}
    for variant in ("KGW", "PF"):
        path = r.run_dir / "detection" / "c4" / f"{variant}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        n_det_rows[variant] = len(rows)
        by: dict = {}
        for row in rows:
            by.setdefault(row["sample_id"], {})[row["text_kind"]] = row
        assert len(by) == 3
        for sid, kinds in by.items():
            assert set(kinds) == {"watermarked", "human", "unwatermarked", "attacked:word_deletion"}, sid
            assert len({kinds[k]["detect_tokens"] for k in kinds}) == 1, (sid, kinds)
            clean = ("watermarked", "human", "unwatermarked")
            assert {kinds[k]["n_tokens"] for k in clean} == {kinds["watermarked"]["detect_tokens"]}, (sid, kinds)
            assert kinds["attacked:word_deletion"]["n_tokens"] <= kinds["watermarked"]["detect_tokens"]
        att = r.run_dir / "attacks" / "c4" / "word_deletion" / f"{variant}.jsonl"
        assert len(att.read_text().strip().splitlines()) == 3
    assert "attacked_tpr_word_deletion" in out["main"].columns
    # resume: nothing is regenerated or re-scored
    r2 = Runner(cfg, show_progress=False)
    r2.stage_generate()
    r2.stage_attack()
    r2.stage_detect()
    gens = r2.run_dir / "generations" / "c4" / "KGW.jsonl"
    assert len(gens.read_text().strip().splitlines()) == 3
    det = r2.run_dir / "detection" / "c4" / "KGW.jsonl"
    assert len(det.read_text().strip().splitlines()) == n_det_rows["KGW"]


def test_chat_template_single_bos_in_ppl_and_semstamp(tiny):
    """With a chat template that emits BOS, PPL context and SemStamp ids must match PromptFormatter.encode."""
    from wmbench import metrics
    from wmbench.models import PromptFormatter

    model, tok, _ = tiny
    tok.chat_template = "{{ bos_token }}{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}<assistant>"
    try:
        fmt = PromptFormatter(tok, use_chat_template=True)
        ids = fmt.encode("hello")
        text = fmt.format("hello")
        assert ids.count(tok.bos_token_id) == 1
        assert list(tok(text, add_special_tokens=False)["input_ids"]) == ids
        nll_a, n_a = metrics.continuation_nll(model, tok, text, " world", "cpu", prompt_add_special_tokens=False)
        nll_b, n_b = metrics.continuation_nll(model, tok, text, " world", "cpu", prompt_add_special_tokens=True)
        assert n_a == n_b and nll_a != nll_b  # the double BOS changes the context
    finally:
        tok.chat_template = None


def test_semstamp_uses_shared_prompt_ids_and_budget(tiny):
    from wmbench.watermarks import build_variant

    model, tok, fmt = tiny
    v = build_variant("SemStamp", model, tok, "cpu", 24, semstamp_n_max=2)
    ids = fmt.encode("The history of the Roman Empire is a long one.")
    seen = {}
    orig = model.generate

    def spy(input_ids, *a, **kw):
        seen.setdefault("first", input_ids[0].tolist())
        return orig(input_ids, *a, **kw)

    model.generate = spy
    try:
        out = v.generate(ids, "The history of the Roman Empire is a long one.")
    finally:
        model.generate = orig
    assert seen["first"] == ids
    assert len(out.continuation_ids) <= 24


def _tiny_t5(tok):
    """Randomly initialised T5 over the OPT vocabulary: exercises the real seq2seq generate path offline."""
    from transformers import T5Config, T5ForConditionalGeneration

    torch.manual_seed(0)
    cfg = T5Config(vocab_size=len(tok), d_model=32, d_kv=8, d_ff=64, num_layers=1, num_decoder_layers=1,
                   num_heads=2, decoder_start_token_id=tok.pad_token_id, pad_token_id=tok.pad_token_id,
                   eos_token_id=tok.eos_token_id)
    return T5ForConditionalGeneration(cfg)


def test_dipper_attacker_through_attack_and_detect_stages(tiny, tmp_path):
    from wmbench.attacks import DipperAttacker
    from wmbench.config import BenchmarkConfig
    from wmbench.evaluate import attack_records, detect_records
    from wmbench.generation import JsonlStore
    from wmbench.watermarks import build_variant

    model, tok, fmt = tiny
    attacker = DipperAttacker(_tiny_t5(tok), tok, model_name="tiny-random-t5", tokenizer_name="opt",
                              gen_kwargs={"max_new_tokens": 6})
    prompt = "The history of the Roman Empire is"
    ids = fmt.encode(prompt)
    kgw, none = build_variant("KGW", model, tok, "cpu", 32), build_variant("none", model, tok, "cpu", 32)
    torch.manual_seed(0)
    wm = kgw.generate(ids, prompt).continuation
    torch.manual_seed(1)
    base = none.generate(ids, prompt).continuation
    human = "Rome grew from a small town on the Tiber into an empire. Its legions marched far. Trade followed them."
    recs = [{"sample_id": "s0", "continuation": wm, "human_text": human, "prompt": prompt, "error": None}]
    none_by_id = {"s0": {"sample_id": "s0", "continuation": base, "error": None}}
    cfg = BenchmarkConfig.smoke_test(run_name="dipper", results_root=str(tmp_path), run_robustness=True,
                                     attacks=("dipper",), attack_similarity_model=None,
                                     max_new_tokens={"c4": 32, "wmt16_de_en": 8, "cnn_dailymail": 8, "humaneval": 8})
    env = {"device_type": "cpu", "dtype": "float32"}
    att_store = JsonlStore(tmp_path / "attacks.jsonl")
    out = attack_records(attacker, tok, "c4", "KGW", recs, none_by_id, cfg, env, att_store, show_progress=False)
    row = att_store.read()[-1]
    assert out["errors"] == 0 and row["error"] is None, row
    assert row["text"] is not None and 0 < row["n_tokens"] <= row["detect_tokens"]
    assert 0 <= row["bleu_vs_source"] <= 100 and row["len_ratio"] > 0 and row["sim_embed"] is None
    # sampling is seeded: the same seed reproduces the paraphrase
    src = " ".join(wm.split())
    assert attacker.attack(src, prompt, row["seed"]) == attacker.attack(src, prompt, row["seed"])
    det_store = JsonlStore(tmp_path / "detection.jsonl")
    detect_records(kgw, tok, "c4", recs, none_by_id, cfg, env, det_store, attacks={"dipper": {"s0": row}},
                   show_progress=False)
    kinds = {r["text_kind"]: r for r in det_store.read()}
    assert set(kinds) == {"watermarked", "human", "unwatermarked", "attacked:dipper"}
    assert kinds["attacked:dipper"]["detect_tokens"] == kinds["watermarked"]["detect_tokens"]
    assert kinds["attacked:dipper"]["attack_config_hash"] == row["attack_config_hash"]


def test_similarity_scorer_orders_paraphrase_above_unrelated_text():
    from wmbench.attacks import load_similarity_scorer

    score = load_similarity_scorer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "cpu")
    if score is None:
        pytest.skip("similarity model not in the local HF cache")
    src = "The cat sat on the mat."
    assert score(src, "A cat was sitting on the mat.") > score(src, "Stock markets fell sharply on Monday.")
