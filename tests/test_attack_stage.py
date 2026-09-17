"""Stage ``attack``: attacked texts of the common-budget watermarked text, cached with strength metrics."""

from __future__ import annotations

import math

import pytest

from fakes import FakeTokenizer, words
from wmbench.attacks import attack_config_digest
from wmbench.config import BenchmarkConfig
from wmbench.evaluate import STAGES, attack_path, attack_records
from wmbench.generation import JsonlStore, stable_seed

ENV = {"device_type": "cpu", "dtype": "float32"}


class UpperAttacker:
    """Deterministic fake: upper-cases and drops every 4th word; records what it was asked to attack."""

    name = "upper"

    def __init__(self):
        self.calls = []

    def config(self):
        return {"attack": "upper", "version": 1}

    def attack(self, text, prompt, seed):
        self.calls.append((text, prompt, seed))
        return " ".join(w.upper() for i, w in enumerate(text.split()) if i % 4 != 3)

    def close(self):
        pass


class LongerAttacker(UpperAttacker):
    name = "longer"

    def attack(self, text, prompt, seed):
        return text + " extra1 extra2 extra3"


class BrokenAttacker(UpperAttacker):
    name = "broken"

    def attack(self, text, prompt, seed):
        raise RuntimeError("paraphraser exploded")


def _cfg(tmp_path, **kw):
    base = dict(run_name="att", results_root=str(tmp_path), run_robustness=True,
                max_new_tokens={"c4": 200, "wmt16_de_en": 8, "cnn_dailymail": 8, "humaneval": 8})
    return BenchmarkConfig.smoke_test(**{**base, **kw})


def _rec(sid, cont, human, error=None, prompt="the prompt"):
    return {"sample_id": sid, "continuation": cont, "human_text": human, "error": error, "prompt": prompt}


def _rows(store):
    return {r["sample_id"]: r for r in store.read()}


def test_attack_stage_is_part_of_the_pipeline_and_paths_are_per_attack(tmp_path):
    assert STAGES == ("generate", "attack", "detect", "quality", "report")
    assert attack_path(tmp_path, "c4", "dipper", "KGW") == tmp_path / "attacks" / "c4" / "dipper" / "KGW.jsonl"


def test_config_defaults_and_hash_independence():
    cfg = BenchmarkConfig()
    assert cfg.attacks == ("word_deletion", "dipper")
    assert cfg.dipper_model == "kalpeshk2011/dipper-paraphraser-xxl" and cfg.dipper_tokenizer == "google/t5-v1_1-xxl"
    assert (cfg.dipper_lex_diversity, cfg.dipper_order_diversity, cfg.dipper_sent_interval) == (60, 0, 1)
    assert BenchmarkConfig.smoke_test().attacks == ("word_deletion",)
    assert cfg.config_hash() == cfg.with_updates(dipper_lex_diversity=80, attacks=("dipper",),
                                                 attack_similarity_model=None).config_hash()


def test_attack_records_attacks_the_common_budget_text_with_a_per_sample_seed(tmp_path):
    tok, cfg, a = FakeTokenizer(), _cfg(tmp_path), UpperAttacker()
    recs = [_rec("s0", words(30, "w"), words(20, "h"), prompt="P0")]
    none = {"s0": _rec("s0", words(40, "n"), None)}
    store = JsonlStore(attack_path(cfg.run_dir, "c4", a.name, "KGW"))
    out = attack_records(a, tok, "c4", "KGW", recs, none, cfg, ENV, store, show_progress=False)
    assert out["attacked"] == 1
    assert a.calls == [(words(20, "w"), "P0", stable_seed(cfg.seed, "attack", "upper", "c4", "s0"))]
    r = _rows(store)["s0"]
    assert (r["dataset"], r["variant"], r["attack"]) == ("c4", "KGW", "upper")
    assert r["attack_config_hash"] == attack_config_digest(a) and r["error"] is None
    assert r["detect_tokens"] == 20 and r["n_tokens"] == 15 and r["n_tokens_uncapped"] == 15
    assert r["len_ratio"] == pytest.approx(0.75)
    assert r["text"] == " ".join(w.upper() for i, w in enumerate(words(20, "w").split()) if i % 4 != 3)
    assert 0 <= r["bleu_vs_source"] <= 100 and r["sim_embed"] is None
    assert r["seed"] == stable_seed(cfg.seed, "attack", "upper", "c4", "s0")


def test_attack_records_caps_the_attacked_text_at_the_budget(tmp_path):
    tok, cfg, a = FakeTokenizer(), _cfg(tmp_path), LongerAttacker()
    recs = [_rec("s0", words(10, "w"), words(50, "h"))]
    none = {"s0": _rec("s0", words(50, "n"), None)}
    store = JsonlStore(attack_path(cfg.run_dir, "c4", a.name, "KGW"))
    attack_records(a, tok, "c4", "KGW", recs, none, cfg, ENV, store, show_progress=False)
    r = _rows(store)["s0"]
    assert r["detect_tokens"] == 10 and r["n_tokens"] == 10 and r["n_tokens_uncapped"] == 13
    assert r["len_ratio"] == pytest.approx(1.3) and r["text"] == words(10, "w")


def test_attack_records_uses_the_similarity_scorer_when_given(tmp_path):
    tok, cfg, a = FakeTokenizer(), _cfg(tmp_path), UpperAttacker()
    recs = [_rec("s0", words(10, "w"), words(50, "h"))]
    none = {"s0": _rec("s0", words(50, "n"), None)}
    store = JsonlStore(attack_path(cfg.run_dir, "c4", a.name, "KGW"))
    seen = []

    def sim(source, attacked):
        seen.append((source, attacked))
        return 0.42

    attack_records(a, tok, "c4", "KGW", recs, none, cfg, ENV, store, similarity=sim, show_progress=False)
    assert _rows(store)["s0"]["sim_embed"] == pytest.approx(0.42)
    assert seen == [(words(10, "w"), _rows(store)["s0"]["text"])]


def test_attack_records_resumes_and_reattacks_when_the_budget_changes(tmp_path):
    tok, cfg, a = FakeTokenizer(), _cfg(tmp_path), UpperAttacker()
    recs = [_rec("s0", words(12, "w"), words(12, "h"))]
    none = {"s0": _rec("s0", words(12, "n"), None)}
    store = JsonlStore(attack_path(cfg.run_dir, "c4", a.name, "KGW"))
    attack_records(a, tok, "c4", "KGW", recs, none, cfg, ENV, store, show_progress=False)
    assert attack_records(a, tok, "c4", "KGW", recs, none, cfg, ENV, store, show_progress=False)["attacked"] == 0
    assert len(store.read()) == 1
    none["s0"]["continuation"] = words(7, "n")
    assert attack_records(a, tok, "c4", "KGW", recs, none, cfg, ENV, store, show_progress=False)["attacked"] == 1
    assert _rows(store)["s0"]["detect_tokens"] == 7  # last row wins


def test_attack_records_skips_samples_without_none_or_with_an_empty_budget(tmp_path):
    tok, cfg, a = FakeTokenizer(), _cfg(tmp_path), UpperAttacker()
    recs = [_rec("s0", words(9, "w"), words(9, "h")), _rec("s1", words(9, "w"), words(9, "h")),
            _rec("s2", "", words(9, "h")), _rec("s3", words(9, "w"), words(9, "h"), error="boom")]
    none = {"s0": _rec("s0", words(9, "n"), None), "s2": _rec("s2", words(9, "n"), None)}
    store = JsonlStore(attack_path(cfg.run_dir, "c4", a.name, "KGW"))
    out = attack_records(a, tok, "c4", "KGW", recs, none, cfg, ENV, store, show_progress=False)
    assert set(_rows(store)) == {"s0"}
    assert out == {"attacked": 1, "cached": 0, "skipped_no_none": 1, "skipped_empty": 1, "errors": 0}


def test_attack_records_records_attacker_errors_and_continues(tmp_path):
    tok, cfg, a = FakeTokenizer(), _cfg(tmp_path), BrokenAttacker()
    recs = [_rec("s0", words(9, "w"), words(9, "h")), _rec("s1", words(9, "w"), words(9, "h"))]
    none = {"s0": _rec("s0", words(9, "n"), None), "s1": _rec("s1", words(9, "n"), None)}
    store = JsonlStore(attack_path(cfg.run_dir, "c4", a.name, "KGW"))
    out = attack_records(a, tok, "c4", "KGW", recs, none, cfg, ENV, store, show_progress=False)
    rows = _rows(store)
    assert out["errors"] == 2 and all("paraphraser exploded" in rows[s]["error"] for s in ("s0", "s1"))
    assert all(rows[s]["text"] is None and math.isnan(rows[s]["len_ratio"]) for s in ("s0", "s1"))
    # error rows are retried on resume
    assert attack_records(a, tok, "c4", "KGW", recs, none, cfg, ENV, store, show_progress=False)["errors"] == 2
