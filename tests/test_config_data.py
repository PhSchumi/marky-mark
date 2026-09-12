import json
import os
from types import SimpleNamespace

import pytest

from wmbench import config as config_mod
from wmbench.config import SAMPLING, BenchmarkConfig, load_config_file, watermark_config_digest
from wmbench.data import PromptTemplates, load_task, postprocess
from wmbench.generation import CONTEXT_OVERFLOW, JsonlStore, latest_by_sample, stable_seed
from wmbench.models import neutralize_generation_config, stop_token_ids


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
    assert SAMPLING == {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
                        "repetition_penalty": 1.0, "no_repeat_ngram_size": 0}


def test_hash_covers_watermark_json_files(tmp_path, monkeypatch):
    d = tmp_path / "wm"
    d.mkdir()
    (d / "KGW.json").write_text('{"algorithm_name": "KGW", "delta": 2.0}')
    monkeypatch.setattr(config_mod, "CONFIG_DIR", d)
    h1 = BenchmarkConfig.full().config_hash()
    (d / "KGW.json").write_text('{\n  "delta": 2.0,\n  "algorithm_name": "KGW"\n}')  # whitespace/order only
    assert BenchmarkConfig.full().config_hash() == h1
    (d / "KGW.json").write_text('{"algorithm_name": "KGW", "delta": 1.0}')
    assert BenchmarkConfig.full().config_hash() != h1
    assert set(watermark_config_digest(d)) == {"KGW.json"}


def test_device_preference_not_in_hash():
    full = BenchmarkConfig.full()
    assert full.config_hash() == full.with_updates(device="cuda:0", dtype="float16").config_hash()


def test_from_env_composes_with_preset(monkeypatch):
    monkeypatch.setenv("WMBENCH_RESULTS", "/scratch/x")
    monkeypatch.setenv("WMBENCH_SEED", "7")
    monkeypatch.setenv("WMBENCH_SMOKE_TEST", "1")
    cfg = BenchmarkConfig.from_env(preset="full")
    assert not cfg.smoke and cfg.results_root == "/scratch/x" and cfg.seed == 7
    assert BenchmarkConfig.from_env().smoke
    with pytest.raises(ValueError):
        BenchmarkConfig.from_env(preset="bogus")


def test_generation_config_neutralised_and_stop_ids():
    from transformers import GenerationConfig

    gc = GenerationConfig(repetition_penalty=1.3, no_repeat_ngram_size=3, eos_token_id=[7, 9], bos_token_id=1)
    model = SimpleNamespace(generation_config=gc)
    tok = SimpleNamespace(eos_token_id=9, pad_token_id=0, bos_token_id=1)
    dropped = neutralize_generation_config(model, tok)
    assert dropped["repetition_penalty"] == 1.3 and dropped["no_repeat_ngram_size"] == 3
    diff = model.generation_config.to_diff_dict()
    assert set(diff) - {"transformers_version"} <= {"bos_token_id", "eos_token_id", "pad_token_id"}
    assert stop_token_ids(model, tok) == {7, 9, 0}


def test_context_overflow_rows_are_final():
    rows = [{"sample_id": "a", "error": f"{CONTEXT_OVERFLOW}: too long"}, {"sample_id": "b", "error": "boom"}]
    done = latest_by_sample(rows, ok_only=True)
    assert set(done) == {"a"}


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


def test_postprocess_code_keeps_indentation():
    from wmbench import metrics

    assert postprocess("code", "    return a + b\n") == "    return a + b"
    assert postprocess("code", "\n    return a + b\n\n\ndef other(): pass") == "    return a + b"
    body = postprocess("code", "    return a + b\n")
    assert metrics.humaneval_passes("def add(a, b):\n", body, "def check(f):\n    assert f(1, 2) == 3\n", "add")


def test_translation_shots_independent_of_n_samples(tmp_path):
    root = tmp_path / "MarkLLM"
    (root / "dataset" / "wmt16_de_en").mkdir(parents=True)
    rows = [{"de": f"de{i}", "en": f"en{i}"} for i in range(10)]
    (root / "dataset" / "wmt16_de_en" / "validation.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows))
    t = PromptTemplates(chat=False, translation_shots=2)
    a = load_task("wmt16_de_en", 3, t, markllm_root=root)
    b = load_task("wmt16_de_en", 6, t, markllm_root=root)
    assert [s.task_prompt for s in a] == [s.task_prompt for s in b[:3]]
    assert "de8" in a[0].task_prompt and "de9" in a[0].task_prompt  # demos come from the reserved tail
    with pytest.raises(ValueError):
        load_task("wmt16_de_en", 9, t, markllm_root=root)  # 10 rows - 2 reserved


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
