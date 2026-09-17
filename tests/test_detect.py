"""Length-controlled detection: every text kind of a sample is scored at one common token budget."""

from __future__ import annotations

import json
import math

import pytest

from wmbench.config import BenchmarkConfig
from wmbench.evaluate import common_token_budget, detect_records
from wmbench.generation import JsonlStore
from fakes import FakeTokenizer, FakeVariant, words as _words


# ----------------------------------------------------------------------------- common_token_budget


def test_common_token_budget_truncates_every_text_to_the_shortest():
    tok = FakeTokenizer()
    t, out = common_token_budget(tok, {"watermarked": _words(10, "w"), "human": _words(6, "h"),
                                       "unwatermarked": _words(8, "n")}, budget=7)
    assert t == 6
    assert {k: len(tok(v)["input_ids"]) for k, v in out.items()} == {"watermarked": 6, "human": 6, "unwatermarked": 6}
    assert out["watermarked"] == _words(6, "w") and out["unwatermarked"] == _words(6, "n")


def test_common_token_budget_is_capped_by_the_generation_budget():
    tok = FakeTokenizer()
    t, out = common_token_budget(tok, {"a": _words(10, "a"), "b": _words(10, "b")}, budget=4)
    assert t == 4 and all(len(tok(v)["input_ids"]) == 4 for v in out.values())


def test_common_token_budget_returns_short_texts_untouched():
    tok = FakeTokenizer()
    human = _words(6, "h")
    t, out = common_token_budget(tok, {"watermarked": _words(9, "w"), "human": human}, budget=200)
    assert t == 6 and out["human"] is human


def test_common_token_budget_empty_or_missing_text_gives_zero():
    tok = FakeTokenizer()
    t, out = common_token_budget(tok, {"watermarked": _words(5, "w"), "human": None}, budget=200)
    assert t == 0 and out == {"watermarked": "", "human": ""}


# ----------------------------------------------------------------------------- detect_records


def _cfg(tmp_path, **kw):
    base = dict(run_name="det", results_root=str(tmp_path), min_detect_tokens=2, run_robustness=True,
                max_new_tokens={"c4": 20, "wmt16_de_en": 8, "cnn_dailymail": 8, "humaneval": 8})
    return BenchmarkConfig.smoke_test(**{**base, **kw})


def _rec(sid, cont, human, error=None):
    return {"sample_id": sid, "continuation": cont, "human_text": human, "error": error}


def _rows_by(store):
    out = {}
    for r in store.read():
        out.setdefault(r["sample_id"], {})[r["text_kind"]] = r
    return out


def test_detect_records_scores_watermarked_human_and_none_at_one_length(tmp_path):
    tok, cfg = FakeTokenizer(), _cfg(tmp_path)
    v = FakeVariant(tok)
    # sample 0: watermarked is the shortest; sample 1: none is the shortest; sample 2: budget binds
    recs = [_rec("s0", _words(5, "w"), _words(20, "h")), _rec("s1", _words(20, "w"), _words(20, "h")),
            _rec("s2", _words(40, "w"), _words(40, "h"))]
    none = {"s0": _rec("s0", _words(20, "n"), None), "s1": _rec("s1", _words(3, "n"), None),
            "s2": _rec("s2", _words(40, "n"), None)}
    store = JsonlStore(tmp_path / "KGW.jsonl")
    detect_records(v, tok, "c4", recs, none, cfg, {"device_type": "cpu", "dtype": "float32"}, store,
                   show_progress=False)
    rows = _rows_by(store)
    expected = {"s0": 5, "s1": 3, "s2": 20}
    for sid, t in expected.items():
        kinds = rows[sid]
        assert set(kinds) == {"watermarked", "human", "unwatermarked"}
        assert {kinds[k]["detect_tokens"] for k in kinds} == {t}
        assert {kinds[k]["n_tokens"] for k in ("watermarked", "human", "unwatermarked")} == {t}, sid
        assert kinds["watermarked"]["variant"] == "KGW" and kinds["watermarked"]["dataset"] == "c4"
        assert kinds["watermarked"]["error"] is None and kinds["watermarked"]["score"] == t


def _attack_row(sid, t, text, name="upper", ahash="a1", error=None):
    return {"sample_id": sid, "attack": name, "attack_config_hash": ahash, "detect_tokens": t, "text": text,
            "error": error}


def test_detect_records_scores_attack_rows_as_attacked_kinds_at_the_sample_budget(tmp_path):
    tok, cfg = FakeTokenizer(), _cfg(tmp_path)
    v = FakeVariant(tok)
    recs = [_rec("s0", _words(30, "w"), _words(30, "h")), _rec("s1", _words(30, "w"), _words(30, "h"))]
    none = {"s0": _rec("s0", _words(10, "n"), None), "s1": _rec("s1", _words(10, "n"), None)}
    attacks = {"upper": {"s0": _attack_row("s0", 10, _words(7, "W"))},          # same budget -> scored
               "dipper": {"s0": _attack_row("s0", 64, _words(9, "D"), name="dipper", ahash="d1"),  # stale budget
                          "s1": _attack_row("s1", 10, None, name="dipper", ahash="d1", error="boom")}}
    store = JsonlStore(tmp_path / "KGW.jsonl")
    out = detect_records(v, tok, "c4", recs, none, cfg, {"device_type": "cpu", "dtype": "float32"}, store,
                         attacks=attacks, show_progress=False)
    rows = _rows_by(store)
    assert set(rows["s0"]) == {"watermarked", "human", "unwatermarked", "attacked:upper"}
    assert set(rows["s1"]) == {"watermarked", "human", "unwatermarked"}
    att = rows["s0"]["attacked:upper"]
    assert att["detect_tokens"] == 10 and att["n_tokens"] == 7 and att["attack_config_hash"] == "a1"
    assert att["raw"]["text"] == _words(7, "W")
    assert rows["s0"]["watermarked"]["attack_config_hash"] is None
    assert out["skipped_stale_attack"] == 2 and out["scored"] == 7


def test_detect_records_rescores_attacked_rows_when_the_attack_hash_changes(tmp_path):
    tok, cfg = FakeTokenizer(), _cfg(tmp_path)
    v = FakeVariant(tok)
    env = {"device_type": "cpu", "dtype": "float32"}
    recs = [_rec("s0", _words(12, "w"), _words(12, "h"))]
    none = {"s0": _rec("s0", _words(12, "n"), None)}
    attacks = {"upper": {"s0": _attack_row("s0", 12, _words(8, "W"), ahash="a1")}}
    store = JsonlStore(tmp_path / "KGW.jsonl")
    assert detect_records(v, tok, "c4", recs, none, cfg, env, store, attacks=attacks, show_progress=False)["scored"] == 4
    assert detect_records(v, tok, "c4", recs, none, cfg, env, store, attacks=attacks, show_progress=False)["scored"] == 0
    attacks["upper"]["s0"] = _attack_row("s0", 12, _words(6, "V"), ahash="a2")
    assert detect_records(v, tok, "c4", recs, none, cfg, env, store, attacks=attacks, show_progress=False)["scored"] == 1
    assert _rows_by(store)["s0"]["attacked:upper"]["attack_config_hash"] == "a2"


def test_detect_records_skips_samples_without_a_none_generation(tmp_path):
    tok, cfg = FakeTokenizer(), _cfg(tmp_path)
    v = FakeVariant(tok)
    recs = [_rec("s0", _words(9, "w"), _words(9, "h")), _rec("s1", _words(9, "w"), _words(9, "h")),
            _rec("s2", _words(9, "w"), _words(9, "h")), _rec("s3", _words(9, "w"), _words(9, "h"), error="boom")]
    none = {"s0": _rec("s0", _words(9, "n"), None), "s1": _rec("s1", None, None, error="boom")}
    store = JsonlStore(tmp_path / "KGW.jsonl")
    summary = detect_records(v, tok, "c4", recs, none, cfg, {"device_type": "cpu", "dtype": "float32"}, store,
                             show_progress=False)
    assert set(_rows_by(store)) == {"s0"}
    assert summary["scored"] == 3 and summary["skipped_no_none"] == 2 and summary["skipped_empty"] == 0


def test_detect_records_resumes_and_rescores_when_the_common_budget_changes(tmp_path):
    tok, cfg = FakeTokenizer(), _cfg(tmp_path)
    v = FakeVariant(tok)
    env = {"device_type": "cpu", "dtype": "float32"}
    recs = [_rec("s0", _words(12, "w"), _words(12, "h"))]
    none = {"s0": _rec("s0", _words(12, "n"), None)}
    store = JsonlStore(tmp_path / "KGW.jsonl")
    detect_records(v, tok, "c4", recs, none, cfg, env, store, show_progress=False)
    n_lines = len(store.read())
    assert n_lines == 3
    assert detect_records(v, tok, "c4", recs, none, cfg, env, store, show_progress=False)["scored"] == 0
    assert len(store.read()) == n_lines  # nothing appended on resume
    none["s0"]["continuation"] = _words(7, "n")  # `none` regenerated shorter: every kind is re-scored at T=7
    assert detect_records(v, tok, "c4", recs, none, cfg, env, store, show_progress=False)["scored"] == 3
    latest = _rows_by(store)["s0"]
    assert {latest[k]["detect_tokens"] for k in latest} == {7}


def test_detect_records_rescores_rows_written_before_length_control(tmp_path):
    tok, cfg = FakeTokenizer(), _cfg(tmp_path)
    v = FakeVariant(tok)
    env = {"device_type": "cpu", "dtype": "float32"}
    recs = [_rec("s0", _words(12, "w"), _words(12, "h"))]
    none = {"s0": _rec("s0", _words(12, "n"), None)}
    store = JsonlStore(tmp_path / "KGW.jsonl")
    detect_records(v, tok, "c4", recs, none, cfg, env, store, show_progress=False)
    legacy = [{k: val for k, val in r.items() if k != "detect_tokens"} for r in store.read()]
    store.rewrite(legacy)
    assert detect_records(v, tok, "c4", recs, none, cfg, env, store, show_progress=False)["scored"] == 3


def test_detect_records_without_attack_rows_writes_no_attacked_kinds(tmp_path):
    tok, cfg = FakeTokenizer(), _cfg(tmp_path)
    v = FakeVariant(tok)
    recs = [_rec("s0", _words(12, "w"), _words(12, "h"))]
    none = {"s0": _rec("s0", _words(12, "n"), None)}
    store = JsonlStore(tmp_path / "KGW.jsonl")
    detect_records(v, tok, "c4", recs, none, cfg, {"device_type": "cpu", "dtype": "float32"}, store,
                   attacks=None, show_progress=False)
    assert set(_rows_by(store)["s0"]) == {"watermarked", "human", "unwatermarked"}


def test_detect_records_empty_common_budget_is_skipped_not_scored(tmp_path):
    tok, cfg = FakeTokenizer(), _cfg(tmp_path)
    v = FakeVariant(tok)
    recs = [_rec("s0", "", _words(12, "h"))]
    none = {"s0": _rec("s0", _words(12, "n"), None)}
    store = JsonlStore(tmp_path / "KGW.jsonl")
    summary = detect_records(v, tok, "c4", recs, none, cfg, {"device_type": "cpu", "dtype": "float32"}, store,
                             show_progress=False)
    assert summary["skipped_empty"] == 1 and summary["scored"] == 0 and not v.seen
    assert not store.path.exists() or store.read() == []
