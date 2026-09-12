import json
import os

import pytest

from wmbench.config import SAMPLING, BenchmarkConfig, load_config_file
from wmbench.data import PromptTemplates, postprocess
from wmbench.generation import JsonlStore, latest_by_sample, stable_seed


def test_presets_and_hash_stability():
    full, smoke = BenchmarkConfig.full(), BenchmarkConfig.smoke_test()
    assert smoke.smoke and not full.smoke
    assert smoke.model_name == "facebook/opt-125m"
    assert full.config_hash() != smoke.config_hash()
    # reporting-only knobs and names do not change the data hash
    assert full.config_hash() == full.with_updates(run_name="x", results_root="/tmp", target_fprs=(0.1,)).config_hash()
    # work selection is not part of the hash: array tasks with different variants share a results dir
    assert full.config_hash() == full.with_updates(variants=("KGW",), datasets=("c4",), n_samples={"c4": 3}).config_hash()
    assert full.config_hash() != full.with_updates(max_new_tokens={"c4": 10}).config_hash()
    assert full.config_hash() != full.with_updates(seed=3).config_hash()
    assert full.config_hash() != full.with_updates(watermark_overrides={"KGW": {"delta": 1.0}}).config_hash()


def test_sampling_is_plain_multinomial():
    assert SAMPLING == {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0}


def test_from_env(monkeypatch):
    monkeypatch.setenv("WMBENCH_SMOKE_TEST", "1")
    monkeypatch.setenv("WMBENCH_MODEL", "org/model")
    monkeypatch.setenv("WMBENCH_VARIANTS", "none, KGW")
    monkeypatch.setenv("WMBENCH_N_SAMPLES", '{"c4": 3}')
    monkeypatch.setenv("WMBENCH_ORACLE", "none")
    cfg = BenchmarkConfig.from_env(run_name="r")
    assert cfg.smoke and cfg.model_name == "org/model" and cfg.variants == ("none", "KGW")
    assert cfg.n_samples["c4"] == 3 and cfg.n_samples["wmt16_de_en"] == 6
    assert cfg.oracle_model_name is None and cfg.run_name == "r"


def test_config_file_roundtrip(tmp_path):
    cfg = BenchmarkConfig.smoke_test(run_name="rt")
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg.as_dict()))
    assert load_config_file(p) == cfg


def test_prompt_templates_base_vs_chat():
    base, chat = PromptTemplates(chat=False, translation_shots=1), PromptTemplates(chat=True)
    t = base.translation("Hallo Welt", [("Guten Tag", "Good day"), ("x", "y")])
    assert t.startswith("German: Guten Tag\nEnglish: Good day") and t.endswith("German: Hallo Welt\nEnglish:")
    assert "x" not in t  # shots capped by translation_shots
    assert chat.translation("Hallo Welt", []).endswith("Hallo Welt")
    assert base.summarization("w " * 10).endswith("TL;DR:")
    assert len(PromptTemplates(article_max_words=3).summarization("a b c d e").split()) == 3 + 1


def test_postprocess():
    assert postprocess("translation", " Hello there.\nGerman: nein\nEnglish: no") == "Hello there."
    assert postprocess("translation", "Hello German: x") == "Hello"
    assert postprocess("summarization", "One.\n\nTwo.") == "One."
    assert postprocess("code", "```python\n    return 1\n```") == "    return 1"
    assert postprocess("completion", " raw ") == " raw "


def test_jsonl_store_resume_and_truncated_line(tmp_path):
    store = JsonlStore(tmp_path / "a" / "b.jsonl")
    store.write({"sample_id": "1", "error": "boom"})
    store.write({"sample_id": "2", "error": None})
    store.write({"sample_id": "1", "error": None})  # retried
    with open(store.path, "a") as f:
        f.write('{"sample_id": "3", "err')  # truncated last line
    rows = store.read()
    assert [r["sample_id"] for r in rows] == ["1", "2", "1"]
    latest = latest_by_sample(rows, ok_only=True)
    assert set(latest) == {"1", "2"} and latest["1"]["error"] is None


def test_stable_seed_is_deterministic_and_distinct():
    assert stable_seed(0, "c4", "c4-1", "KGW") == stable_seed(0, "c4", "c4-1", "KGW")
    assert stable_seed(0, "c4", "c4-1", "KGW") != stable_seed(0, "c4", "c4-1", "PF")
    assert stable_seed(1, "c4", "c4-1", "KGW") != stable_seed(0, "c4", "c4-1", "KGW")
