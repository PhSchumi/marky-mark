"""Report building on synthetic JSONL results (no models involved)."""

import json
import math
import random

import pytest

import pandas as pd

from wmbench.config import BenchmarkConfig
from wmbench.report import build_report, detection_table, df_to_markdown, fpr_col, load_run, quality_table


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def synthetic_run(tmp_path):
    cfg = BenchmarkConfig.smoke_test(run_name="synth", results_root=str(tmp_path), variants=("none", "KGW", "PF"),
                                     datasets=("c4", "wmt16_de_en"), min_detect_tokens=5, run_robustness=True)
    rng = random.Random(0)
    run = cfg.run_dir
    (run / "manifest.json").parent.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"config_hash": cfg.config_hash(), "markllm_commit": "abc",
                                                   "environment": {"device_type": "cpu"}}))
    n = 30
    for variant in ("none", "KGW", "PF"):
        gens = [{"sample_id": f"c4-{i}", "dataset": "c4", "variant": variant, "error": None,
                 "continuation": "w " * 40, "continuation_tokens": 40, "gen_seconds": 1.0,
                 "prompt_text": "p", "human_text": "h " * 60, "answer": None, "reference": None}
                for i in range(n)]
        gens[3]["error"] = "boom"
        _write(run / "generations" / "c4" / f"{variant}.jsonl", gens)
        wmt = [{"sample_id": f"wmt16-{i}", "dataset": "wmt16_de_en", "variant": variant, "error": None,
                "continuation": "x", "continuation_tokens": 5, "gen_seconds": 0.5, "prompt_text": "p",
                "answer": "the cat sat on the mat today" if variant != "PF" else "a cat sat on a mat", "reference": "the cat sat on the mat today", "human_text": None}
               for i in range(10)]
        _write(run / "generations" / "wmt16_de_en" / f"{variant}.jsonl", wmt)
        qual = [{"sample_id": f"c4-{i}", "dataset": "c4", "variant": variant, "error": None,
                 "nll_sum": 40 * math.log(10 if variant == "none" else 12), "n_tokens": 40,
                 "ppl": 10 if variant == "none" else 12, "rep_2": 0.1, "rep_3": 0.05, "rep_4": 0.02,
                 "diversity": 0.83, "log_diversity_markllm": 6.0, "oracle_model": "m"} for i in range(n) if i != 3]
        _write(run / "quality" / "c4" / f"{variant}.jsonl", qual)
        _write(run / "quality" / "wmt16_de_en" / f"{variant}.jsonl",
               [{"sample_id": f"wmt16-{i}", "dataset": "wmt16_de_en", "variant": variant, "error": None,
                 "bleu_sentence": 100.0 if variant != "PF" else 50.0, "answer_words": 3} for i in range(10)])
        if variant != "none":
            det = []
            for i in range(n):
                if i == 3:
                    continue
                for kind, mu in (("watermarked", 5.0), ("human", 0.0), ("unwatermarked", 0.2), ("attacked", 3.0)):
                    det.append({"sample_id": f"c4-{i}", "dataset": "c4", "variant": variant, "text_kind": kind,
                                "score": rng.gauss(mu, 1.0), "is_watermarked": mu > 2, "n_tokens": 40,
                                "error": None, "raw": {}})
            det[0]["score"] = float("nan")  # one detector failure
            _write(run / "detection" / "c4" / f"{variant}.jsonl", det)
    _write(run / "quality" / "c4" / "human.jsonl",
           [{"sample_id": f"c4-{i}", "dataset": "c4", "variant": "human", "error": None, "nll_sum": 40 * math.log(8),
             "n_tokens": 40, "ppl": 8, "rep_2": 0.05, "rep_3": 0.01, "rep_4": 0.0, "diversity": 0.94,
             "log_diversity_markllm": 7.0, "oracle_model": "m"} for i in range(n)])
    return cfg


def test_build_report_end_to_end(synthetic_run):
    cfg = synthetic_run
    out = build_report(cfg, plots=True)
    main = out["main"].set_index("variant")
    assert set(main.index) == {"none", "KGW", "PF", "human"}
    assert main.loc["none", "ppl_corpus"] == pytest.approx(10.0)
    assert main.loc["KGW", "ppl_corpus"] == pytest.approx(12.0)
    assert main.loc["human", "ppl_corpus"] == pytest.approx(8.0)
    assert main.loc["KGW", "auroc_vs_human"] > 0.95
    assert main.loc["KGW", "tpr@1%fpr_vs_human"] > 0.5
    assert main.loc["none", "bleu"] == pytest.approx(100.0)
    assert main.loc["PF", "bleu"] < 100.0
    assert 0 <= main.loc["KGW", "attacked_tpr"] <= 1
    det = out["detection"].set_index("variant")
    assert det.loc["KGW", "n_pos"] == 28  # 29 kept minus one NaN
    assert det.loc["KGW", "fpr_warning_human"]  # fewer than 100 negatives
    deltas = out["paired_deltas"]
    row = deltas[(deltas.variant == "PF") & (deltas.metric == "bleu_sentence")].iloc[0]
    assert row.delta_vs_none == pytest.approx(-50.0) and row.n_pairs == 10
    for name in ("summary.md", "summary.csv", "detection_summary.csv", "quality_summary.csv", "paired_deltas.csv",
                 "results.json"):
        assert (cfg.run_dir / name).exists()
    assert (cfg.run_dir / "plots" / "roc_vs_human.png").exists()
    assert "SMOKE TEST" in (cfg.run_dir / "summary.md").read_text()
    frames = load_run(cfg.run_dir)
    assert len(frames["generations"]) == 3 * (30 + 10)

    def _no_const(c):
        raise ValueError(c)

    json.loads((cfg.run_dir / "results.json").read_text(), parse_constant=_no_const)  # no bare NaN tokens


def test_report_survives_all_error_quality_rows_and_missing_quality(synthetic_run):
    cfg = synthetic_run
    _write(cfg.run_dir / "quality" / "wmt16_de_en" / "KGW.jsonl",
           [{"sample_id": f"wmt16-{i}", "dataset": "wmt16_de_en", "variant": "KGW", "error": "boom"} for i in range(10)])
    out = build_report(cfg, plots=True)
    q = out["quality"]
    row = q[(q.dataset == "wmt16_de_en") & (q.variant == "KGW")].iloc[0]
    assert row.n_err_quality == 10 and "bleu_corpus" not in row.dropna()
    # detection rows but no c4 quality rows at all -> plots must not abort the report
    import shutil

    shutil.rmtree(cfg.run_dir / "quality" / "c4")
    out = build_report(cfg, plots=True)
    assert (cfg.run_dir / "summary.md").exists() and "auroc_vs_human" in out["main"].columns


def test_fpr_column_naming_and_all_targets(synthetic_run):
    cfg = synthetic_run.with_updates(target_fprs=(0.001, 0.02, 0.05))
    frames = load_run(cfg.run_dir)
    dt = detection_table(frames["detection"], cfg, frames["generations"])
    assert fpr_col(0.001, "human") == "tpr@0.1%fpr_human"
    for col in ("tpr@0.1%fpr_human", "tpr@2%fpr_human", "tpr@5%fpr_human", "tpr@0.1%fpr_human_ci"):
        assert col in dt.columns


def test_latest_prefers_usable_row(synthetic_run):
    cfg = synthetic_run
    path = cfg.run_dir / "generations" / "wmt16_de_en" / "KGW.jsonl"
    with open(path, "a") as f:  # a stale trailing error row must not hide the good generation
        f.write(json.dumps({"sample_id": "wmt16-0", "dataset": "wmt16_de_en", "variant": "KGW", "error": "late boom"}) + "\n")
    frames = load_run(cfg.run_dir)
    g = frames["generations"]
    r = g[(g.variant == "KGW") & (g.sample_id == "wmt16-0") & (g.dataset == "wmt16_de_en")].iloc[0]
    assert r.error is None or pd.isna(r.error)
    out = build_report(cfg, plots=False)
    assert out["quality"][(out["quality"].dataset == "wmt16_de_en") & (out["quality"].variant == "KGW")].iloc[0].n_bleu == 10


def test_df_to_markdown_handles_nan_and_empty():
    import pandas as pd

    assert df_to_markdown(pd.DataFrame()) == "_(no rows)_"
    md = df_to_markdown(pd.DataFrame([{"a": float("nan"), "b": "x", "c": 1.23456}]))
    assert "| - | x | 1.235 |" in md
