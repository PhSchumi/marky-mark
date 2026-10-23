"""Report building on synthetic JSONL results (no models involved)."""

import json
import math
import random

import pytest

import pandas as pd

from wmbench.config import BenchmarkConfig
from wmbench.report import build_report, detection_table, df_to_markdown, fpr_col, load_run, quality_table


SHORT_SAMPLES = (5, 6)  # detection budget below min_detect_tokens=5 -> excluded from every detection number


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
                t = 3 if i in SHORT_SAMPLES else 40  # common detection budget of the sample (all kinds share it)
                for kind, mu in (("watermarked", 5.0), ("human", 0.0), ("unwatermarked", 0.2),
                                 ("attacked:word_deletion", 3.0), ("attacked:dipper", 1.5)):
                    attacked = kind.startswith("attacked:")
                    det.append({"sample_id": f"c4-{i}", "dataset": "c4", "variant": variant, "text_kind": kind,
                                "score": rng.gauss(mu, 1.0), "is_watermarked": mu > 2, "detect_tokens": t,
                                "n_tokens": t - 1 if attacked else t, "error": None, "raw": {},
                                "attack_config_hash": "ah" if attacked else None})
            det[0]["score"] = float("nan")  # one detector failure
            _write(run / "detection" / "c4" / f"{variant}.jsonl", det)
            for attack, sim in (("word_deletion", None), ("dipper", 0.8)):
                _write(run / "attacks" / "c4" / attack / f"{variant}.jsonl",
                       [{"sample_id": f"c4-{i}", "dataset": "c4", "variant": variant, "attack": attack,
                         "attack_config_hash": "ah", "detect_tokens": 3 if i in SHORT_SAMPLES else 40,
                         "text": "x", "n_tokens": 39, "n_tokens_uncapped": 44, "len_ratio": 1.1,
                         "bleu_vs_source": 25.0, "sim_embed": sim, "error": None} for i in range(n) if i != 3])
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
    assert 0 <= main.loc["KGW", "attacked_tpr_word_deletion"] <= 1
    assert main.loc["KGW", "attacked_tpr_dipper"] <= main.loc["KGW", "attacked_tpr_word_deletion"]
    det = out["detection"].set_index("variant")
    for attack in ("word_deletion", "dipper"):
        assert det.loc["KGW", f"attacked_n_{attack}"] == 27
        assert 0 <= det.loc["KGW", f"attacked_auroc_human_{attack}"] <= 1
        assert 0 <= det.loc["KGW", f"attacked_fixed_thr_tpr_{attack}"] <= 1
        assert det.loc["KGW", f"attacked_tokens_mean_{attack}"] == 39
        assert det.loc["KGW", f"attack_len_ratio_{attack}"] == pytest.approx(1.1)
        assert det.loc["KGW", f"attack_bleu_vs_source_{attack}"] == pytest.approx(25.0)
    assert det.loc["KGW", "attack_sim_embed_dipper"] == pytest.approx(0.8)
    assert pd.isna(det.loc["KGW", "attack_sim_embed_word_deletion"])
    assert det.loc["KGW", "attacks"] == "word_deletion, dipper"
    assert det.loc["KGW", "n_gen_ok"] == 29  # 30 generations minus the error row
    assert det.loc["KGW", "n_samples_kept"] == 27  # minus the two samples below min_detect_tokens
    assert det.loc["KGW", "n_pos"] == 26  # 27 kept minus one NaN
    assert det.loc["KGW", "n_neg_human"] == 27 and det.loc["KGW", "n_neg_none"] == 27
    assert det.loc["KGW", "detect_tokens_mean"] == 40
    assert det.loc["KGW", "pos_tokens_mean"] == det.loc["KGW", "neg_tokens_mean_human"] == det.loc["KGW", "neg_tokens_mean_none"] == 40
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


def test_gsm8k_accuracy_and_degradation_table(synthetic_run):
    cfg = synthetic_run.with_updates(datasets=("c4", "wmt16_de_en", "gsm8k"))
    n_correct = {"none": 8, "KGW": 6, "PF": 8}
    for variant, k in n_correct.items():
        _write(cfg.run_dir / "generations" / "gsm8k" / f"{variant}.jsonl",
               [{"sample_id": f"gsm8k-{i}", "dataset": "gsm8k", "variant": variant, "error": None,
                 "continuation": "x", "continuation_tokens": 100, "gen_seconds": 1.0, "prompt_text": "p",
                 "answer": "1", "reference": "1", "human_text": None} for i in range(10)])
        _write(cfg.run_dir / "quality" / "gsm8k" / f"{variant}.jsonl",
               [{"sample_id": f"gsm8k-{i}", "dataset": "gsm8k", "variant": variant, "error": None,
                 "correct": i < k} for i in range(10)])
    out = build_report(cfg, plots=False)
    main = out["main"].set_index("variant")
    assert main.loc["none", "gsm8k_acc"] == pytest.approx(0.8)
    assert main.loc["KGW", "gsm8k_acc"] == pytest.approx(0.6)
    deg = out["degradation"].set_index(["dataset", "variant", "metric"])
    kgw = deg.loc[("gsm8k", "KGW", "accuracy")]
    assert kgw.none == pytest.approx(0.8) and kgw.watermarked == pytest.approx(0.6)
    assert kgw.change_pct == pytest.approx(-25.0)
    assert deg.loc[("gsm8k", "PF", "accuracy")].change_pct == pytest.approx(0.0)
    ppl = deg.loc[("c4", "KGW", "perplexity (oracle)")]
    assert ppl.none == pytest.approx(10.0) and ppl.watermarked == pytest.approx(12.0)
    assert ppl.change_pct == pytest.approx(20.0) and ppl.better == "lower"
    assert deg.loc[("wmt16_de_en", "PF", "BLEU (sentence mean)")].change_pct == pytest.approx(-50.0)
    assert (cfg.run_dir / "degradation.csv").exists()
    assert "Quality degradation vs. unwatermarked" in (cfg.run_dir / "summary.md").read_text()


def test_gsm8k_repeats_are_averaged_per_question(synthetic_run):
    # 4 questions, 2 answers each. none: 3 + 2 of 4 questions right -> 5/8; KGW: 2 + 1 -> 3/8.
    cfg = synthetic_run.with_updates(datasets=("c4", "gsm8k"), variants=("none", "KGW"), gsm8k_repeats=2)
    correct = {"none": [[True, True, True, False], [True, True, False, False]],
               "KGW": [[True, True, False, False], [True, False, False, False]]}
    for variant, rounds in correct.items():
        rows = []
        for repeat, answers in enumerate(rounds):
            for i, ok in enumerate(answers):
                sample_id = f"gsm8k-{i}" if repeat == 0 else f"gsm8k-{i}-r{repeat}"
                rows.append({"sample_id": sample_id, "dataset": "gsm8k", "variant": variant, "error": None,
                             "correct": ok})
        _write(cfg.run_dir / "quality" / "gsm8k" / f"{variant}.jsonl", rows)
    out = build_report(cfg, plots=False)
    qt = out["quality"].set_index(["dataset", "variant"])
    assert qt.loc[("gsm8k", "none"), "accuracy"] == pytest.approx(5 / 8)
    assert qt.loc[("gsm8k", "none"), "n_questions"] == 4
    assert qt.loc[("gsm8k", "none"), "answers_per_question"] == pytest.approx(2.0)
    kgw = out["paired_deltas"].set_index(["dataset", "variant", "metric"]).loc[("gsm8k", "KGW", "accuracy")]
    assert kgw.n_pairs == 4  # questions, not answers
    assert kgw.delta_vs_none == pytest.approx(3 / 8 - 5 / 8)
    assert kgw.rel_change_pct == pytest.approx(-40.0)


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


def test_min_detect_tokens_filters_on_the_common_detection_budget(synthetic_run):
    frames = load_run(synthetic_run.run_dir)
    lenient = detection_table(frames["detection"], synthetic_run.with_updates(min_detect_tokens=3), frames["generations"])
    strict = detection_table(frames["detection"], synthetic_run.with_updates(min_detect_tokens=41), frames["generations"])
    assert set(lenient.set_index("variant").n_samples_kept) == {29}
    assert strict.empty or set(strict.set_index("variant").n_samples_kept) == {0}


def test_report_rejects_detection_rows_written_without_length_control(synthetic_run):
    cfg = synthetic_run
    path = cfg.run_dir / "detection" / "c4" / "PF.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    _write(path, [{k: v for k, v in r.items() if k != "detect_tokens"} for r in rows])
    with pytest.raises(RuntimeError, match="detect_tokens"):
        build_report(cfg, plots=False)


def test_report_ignores_stray_legacy_rows_when_the_variant_was_rescored(synthetic_run, caplog):
    """Legacy rows of samples that stage `detect` skipped must not block a re-scored run."""
    cfg = synthetic_run
    path = cfg.run_dir / "detection" / "c4" / "KGW.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    legacy_ids = {"c4-10", "c4-11"}
    _write(path, [{k: v for k, v in r.items() if k != "detect_tokens"} if r["sample_id"] in legacy_ids else r
                  for r in rows])
    with caplog.at_level("WARNING", logger="wmbench.report"):
        out = build_report(cfg, plots=False)
    det = out["detection"].set_index("variant")
    assert det.loc["KGW", "n_samples_kept"] == 25  # 27 minus the two samples that only have legacy rows
    assert det.loc["PF", "n_samples_kept"] == 27
    assert any("before length control" in m for m in caplog.messages)


def _det_rows(cfg, variant):
    path = cfg.run_dir / "detection" / "c4" / f"{variant}.jsonl"
    return path, [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_detection_uses_the_last_row_even_when_it_is_an_error(synthetic_run):
    """A re-score at a smaller common budget that fails must supersede the stale good row."""
    cfg = synthetic_run
    path, rows = _det_rows(cfg, "KGW")
    for kind in ("watermarked", "human", "unwatermarked", "attacked:word_deletion", "attacked:dipper"):
        rows.append({"sample_id": "c4-0", "dataset": "c4", "variant": "KGW", "text_kind": kind, "detect_tokens": 3,
                     "n_tokens": 3, "score": float("nan"), "is_watermarked": None, "error": "too short", "raw": {}})
    _write(path, rows)
    det = build_report(cfg, plots=False)["detection"].set_index("variant")
    assert det.loc["KGW", "n_samples_kept"] == 26  # c4-0 now sits at T=3 < min_detect_tokens
    assert det.loc["PF", "n_samples_kept"] == 27


def test_negatives_must_share_the_budget_of_the_watermarked_row(synthetic_run, caplog):
    """A negative scored at another length than the sample's watermarked text is dropped, not compared."""
    cfg = synthetic_run
    path, rows = _det_rows(cfg, "KGW")
    for r in rows:
        if r["sample_id"] == "c4-1" and r["text_kind"] == "human":
            r["detect_tokens"] = r["n_tokens"] = 20
    _write(path, rows)
    with caplog.at_level("WARNING", logger="wmbench.report"):
        det = build_report(cfg, plots=False)["detection"].set_index("variant")
    assert det.loc["KGW", "n_samples_kept"] == 27
    assert det.loc["KGW", "n_neg_human"] == 26 and det.loc["KGW", "n_neg_none"] == 27
    assert any("budget" in m and "c4/KGW" in m for m in caplog.messages)


def test_report_without_attack_rows_has_no_attack_columns(synthetic_run):
    import shutil

    cfg = synthetic_run
    shutil.rmtree(cfg.run_dir / "attacks")
    for variant in ("KGW", "PF"):
        path = cfg.run_dir / "detection" / "c4" / f"{variant}.jsonl"
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        _write(path, [r for r in rows if not r["text_kind"].startswith("attacked:")])
    out = build_report(cfg, plots=True)
    assert not [c for c in out["detection"].columns if c.startswith("attack")]
    assert not [c for c in out["main"].columns if c.startswith("attacked_tpr")]
