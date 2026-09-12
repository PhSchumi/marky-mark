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
                                     min_detect_tokens=4)
    r = Runner(cfg, show_progress=False)
    r.run()
    out = r.stage_report()
    assert (cfg.run_dir / "summary.md").exists()
    assert set(out["main"].variant) >= {"none", "KGW", "PF"}
    # resume: nothing is regenerated
    r2 = Runner(cfg, show_progress=False)
    r2.stage_generate()
    gens = r2.run_dir / "generations" / "c4" / "KGW.jsonl"
    assert len(gens.read_text().strip().splitlines()) == 3


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
