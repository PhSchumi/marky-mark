"""Summaries, tables and plots from the JSONL results (pure pandas; no models are loaded)."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from wmbench import metrics
from wmbench.config import DETECTION_DATASETS, BenchmarkConfig
from wmbench.data import TASKS
from wmbench.generation import is_final_error
from wmbench.watermarks import VARIANTS

log = logging.getLogger("wmbench.report")

QUALITY_COLUMNS = {
    "completion": ("nll_sum", "n_tokens", "ppl", "rep_2", "rep_3", "rep_4", "diversity", "log_diversity_markllm", "oracle_model"),
    "translation": ("bleu_sentence", "answer_words"),
    "summarization": ("rouge1", "rouge2", "rougeL", "answer_words"),
    "code": ("passed",),
    "math": ("correct",),
}

# Headline metric per task for the degradation table: (dataset, paired_deltas metric) -> (label, better).
DEGRADATION_METRICS = {
    ("c4", "log_ppl"): ("perplexity (oracle)", "lower"),
    ("c4", "diversity"): ("diversity", "higher"),
    ("wmt16_de_en", "bleu_sentence"): ("BLEU (sentence mean)", "higher"),
    ("cnn_dailymail", "rougeL"): ("ROUGE-L", "higher"),
    ("gsm8k", "accuracy"): ("accuracy", "higher"),
    ("humaneval", "pass"): ("pass@1", "higher"),
}


# ----------------------------------------------------------------------------- loading


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _latest(rows: list[dict], keys=("sample_id",)) -> list[dict]:
    """Last row per key, preferring the last *usable* row (no error, or a final error) when one exists.

    Mirrors ``generation.latest_by_sample(ok_only=True)``: a stale trailing error row must not
    hide a good record that the resume logic itself treats as done.
    """
    best: dict[tuple, dict] = {}
    for r in rows:
        k = tuple(r.get(x) for x in keys)
        prev = best.get(k)
        usable = not r.get("error") or is_final_error(r.get("error"))
        if prev is None or usable or (prev.get("error") and not is_final_error(prev.get("error"))):
            best[k] = r
    return list(best.values())


def _last(rows: list[dict], keys=("sample_id", "text_kind")) -> list[dict]:
    """Plain last row per key, for detection files.

    Stage ``detect`` appends a row only when a (sample, kind) has to be (re-)scored, e.g. at a new
    common budget, so the newest row is the truth even when it is an error: a stale good row may
    sit at another length and must not mask it.
    """
    best: dict[tuple, dict] = {}
    for r in rows:
        best[tuple(r.get(x) for x in keys)] = r
    return list(best.values())


def load_run(run_dir: Path) -> dict[str, pd.DataFrame | dict]:
    """Load manifest + all generation / detection / quality rows of a run into DataFrames."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text()) if (run_dir / "manifest.json").exists() else {}
    frames = {}
    for kind, keys in (("generations", ("sample_id",)), ("detection", ("sample_id", "text_kind")),
                       ("quality", ("sample_id",)), ("attacks", ("sample_id",))):
        rows = []
        pick = _last if kind in ("detection", "attacks") else _latest
        pattern = "*/*/*.jsonl" if kind == "attacks" else "*/*.jsonl"  # attacks/<dataset>/<attack>/<variant>
        for path in sorted((run_dir / kind).glob(pattern)):
            rows.extend(pick(_read_jsonl(path), keys))
        df = pd.DataFrame(rows)
        if not df.empty and "error" not in df.columns:
            df["error"] = None
        frames[kind] = df
    frames["manifest"] = manifest
    return frames


# ----------------------------------------------------------------------------- helpers


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _ci(lo, hi, nd=2):
    return "-" if not (math.isfinite(lo) and math.isfinite(hi)) else f"[{lo:.{nd}f}, {hi:.{nd}f}]"


def fpr_col(fpr: float, tag: str) -> str:
    """Column name for TPR at a target FPR: 0.01 -> ``tpr@1%fpr_<tag>``, 0.001 -> ``tpr@0.1%fpr_<tag>``."""
    return f"tpr@{fpr * 100:g}%fpr_{tag}"


def kept_sample_ids(det: pd.DataFrame, cfg: BenchmarkConfig, *, warn: bool = True) -> set:
    """Samples of *one variant* whose common detection budget reaches ``min_detect_tokens``.

    ``detect_tokens`` is the per-sample length at which stage ``detect`` scored the watermarked,
    the human and the ``none`` text alike, so the filter keeps the three classes on the same
    samples at the same lengths. Rows without the field were scored before length control and
    are refused rather than silently mixed in.
    """
    if det.empty or "text_kind" not in det:
        return set()
    legacy = det.detect_tokens.isna() if "detect_tokens" in det.columns else pd.Series(True, index=det.index)
    if legacy.all():
        raise RuntimeError("detection rows without `detect_tokens` found: they were scored before detection was "
                           "length-controlled (schema 2). Re-run stage `detect` for this run.")
    if legacy.any():
        if warn:
            log.warning("%s: ignoring %d detection rows scored before length control (stage `detect` skipped their "
                        "samples: no `none` generation or an empty common budget)",
                        "/".join(str(x) for x in det[["dataset", "variant"]].iloc[0]), int(legacy.sum()))
        det = det[~legacy]
    wm = det[det.text_kind == "watermarked"].groupby("sample_id")["detect_tokens"].min()
    return set(wm[wm >= cfg.min_detect_tokens].index)


def _same_budget_as_watermarked(dv: pd.DataFrame, label: str, *, warn: bool = True) -> pd.DataFrame:
    """Keep only rows scored at the budget of their sample's watermarked row (the equal-length invariant).

    Stage ``detect`` writes all kinds of a sample at one ``detect_tokens``; a mismatch means the
    stage was interrupted between kinds (resume completes it) and such a negative must not be
    compared with a positive of another length.
    """
    pos = dv[dv.text_kind == "watermarked"]
    other = dv[dv.text_kind != "watermarked"]
    if other.empty:
        return dv
    budget = pos.drop_duplicates("sample_id").set_index("sample_id").detect_tokens
    bad = other.detect_tokens.values != budget.reindex(other.sample_id).values
    if not bad.any():
        return dv
    if warn:
        log.warning("%s: dropping %d detection rows whose budget differs from their sample's watermarked row "
                    "(stage `detect` interrupted? re-run it)", label, int(bad.sum()))
    return pd.concat([pos, other[~bad]])


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame rows as JSON-safe dicts (NaN/NaT -> null)."""
    if df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


# ----------------------------------------------------------------------------- summaries


def _attack_names(kinds, cfg: BenchmarkConfig) -> list[str]:
    """Attack names present as ``attacked:<name>`` kinds, configured attacks first."""
    names = {str(k).split(":", 1)[1] for k in kinds if str(k).startswith("attacked:")}
    return [a for a in cfg.attacks if a in names] + sorted(a for a in names if a not in cfg.attacks)


def detection_table(det: pd.DataFrame, cfg: BenchmarkConfig, gens: pd.DataFrame,
                    attacks: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per variant: AUROC / TPR@FPR / best-F1 / fixed-threshold TPR+FPR against human and 'none' negatives.

    With attacked rows (``attacked:<name>`` kinds) every attack adds ``attacked_*_<name>`` columns: TPR at the
    threshold fixed on the clean human negatives, AUROC vs the clean human negatives, TPR at the shipped
    threshold, token count, and the attack strength from the ``attacks`` frame (length ratio, BLEU vs the
    source text, embedding similarity).
    """
    rows = []
    if det.empty:
        return pd.DataFrame(rows)
    primary = cfg.target_fprs[0]
    for dataset in DETECTION_DATASETS:
        d = det[det.dataset == dataset]
        if d.empty:
            continue
        g = gens[gens.dataset == dataset] if not gens.empty and "dataset" in gens else pd.DataFrame()
        for variant in [v for v in cfg.variants if v != "none" and v in set(d.variant)]:
            dvar = d[d.variant == variant]
            keep = kept_sample_ids(dvar, cfg)  # per variant: the common budget depends on its own lengths
            dv = _same_budget_as_watermarked(dvar[dvar.sample_id.isin(keep) & dvar.detect_tokens.notna()],
                                             f"{dataset}/{variant}")
            pos = dv[dv.text_kind == "watermarked"]
            gv = g[g.variant == variant] if not g.empty else pd.DataFrame()
            row = {"dataset": dataset, "variant": variant, "score": VARIANTS[variant].score_name,
                   "n_gen_ok": int(gv.error.isna().sum()) if not gv.empty else None,
                   "n_samples_kept": len(keep), "n_pos": int(pos.score.notna().sum()),
                   "n_pos_nan": int(pos.score.isna().sum()), "pos_mean": float(pos.score.mean()),
                   "detect_tokens_mean": float(pos.detect_tokens.mean()) if len(pos) else float("nan"),
                   "pos_tokens_mean": float(pos.n_tokens.mean()) if len(pos) else float("nan")}
            for neg_kind, tag in (("human", "human"), ("unwatermarked", "none")):
                neg = dv[dv.text_kind == neg_kind]
                if neg.empty:
                    continue
                s = metrics.detection_summary(pos.score, neg.score, pos.is_watermarked, neg.is_watermarked)
                boot = metrics.detection_bootstrap(pos.score, neg.score, primary, seed=cfg.seed)
                row.update({f"n_neg_{tag}": s.n_neg, f"neg_tokens_mean_{tag}": float(neg.n_tokens.mean()),
                            f"neg_mean_{tag}": s.neg_mean, f"auroc_{tag}": s.auroc,
                            f"auroc_{tag}_ci": _ci(*boot["auroc"])})
                for f in cfg.target_fprs:
                    row[fpr_col(f, tag)] = metrics.tpr_at_fpr(pos.score, neg.score, f)[0]
                row[fpr_col(primary, tag) + "_ci"] = _ci(*boot["tpr_at_fpr"])
                row.update({f"best_f1_{tag}": s.best_f1,
                            f"fixed_thr_tpr_{tag}": s.fixed_threshold_tpr, f"fixed_thr_fpr_{tag}": s.fixed_threshold_fpr,
                            f"fpr_warning_{tag}": bool(s.n_neg < 1 / primary)})
            hum = dv[dv.text_kind == "human"]
            names = _attack_names(dv.text_kind.unique(), cfg)
            if names and not hum.empty:
                _, thr = metrics.tpr_at_fpr(pos.score, hum.score, primary)
                row["attacks"] = ", ".join(names)
                for name in names:
                    att = dv[dv.text_kind == f"attacked:{name}"]
                    a = metrics._clean(att.score)
                    flags = att.is_watermarked.dropna()
                    row.update({
                        f"attacked_n_{name}": int(len(a)),
                        f"attacked_tpr_at_clean_thr_{name}": float(np.mean(a > thr)) if len(a) else float("nan"),
                        f"attacked_auroc_human_{name}": metrics.auroc(att.score, hum.score),
                        f"attacked_fixed_thr_tpr_{name}": float(flags.astype(bool).mean()) if len(flags) else float("nan"),
                        f"attacked_tokens_mean_{name}": float(att.n_tokens.mean()) if len(att) else float("nan"),
                    })
                    if attacks is not None and not attacks.empty and "attack" in attacks:
                        ar = attacks[(attacks.dataset == dataset) & (attacks.variant == variant)
                                     & (attacks.attack == name) & attacks.sample_id.isin(keep) & attacks.error.isna()]
                        if not ar.empty:
                            sim = pd.to_numeric(ar.sim_embed, errors="coerce") if "sim_embed" in ar else pd.Series(dtype=float)
                            row.update({f"attack_len_ratio_{name}": float(ar.len_ratio.mean()),
                                        f"attack_bleu_vs_source_{name}": float(ar.bleu_vs_source.mean()),
                                        f"attack_sim_embed_{name}": float(sim.mean()) if sim.notna().any() else float("nan")})
            row["fixed_threshold"] = VARIANTS[variant].fixed_threshold
            row["nominal_fpr"] = VARIANTS[variant].nominal_fpr
            rows.append(row)
    return pd.DataFrame(rows)


def quality_table(qual: pd.DataFrame, gens: pd.DataFrame, cfg: BenchmarkConfig) -> pd.DataFrame:
    """Per (dataset, variant): PPL (corpus + median), diversity, BLEU, ROUGE, pass@1, speed, errors."""
    rows = []
    variants = list(cfg.variants) + ["human"]
    for dataset in cfg.datasets:
        task = TASKS[dataset]
        for variant in variants:
            q = qual[(qual.dataset == dataset) & (qual.variant == variant)] if not qual.empty else pd.DataFrame()
            g = gens[(gens.dataset == dataset) & (gens.variant == variant)] if not gens.empty else pd.DataFrame()
            if q.empty and g.empty:
                continue
            row = {"dataset": dataset, "variant": variant,
                   "n_ok": int(g.error.isna().sum()) if not g.empty else 0,
                   "n_err": int(g.error.notna().sum()) if not g.empty else 0,
                   "n_ok_quality": int(q.error.isna().sum()) if not q.empty else 0,
                   "n_err_quality": int(q.error.notna().sum()) if not q.empty else 0}
            if not g.empty and variant != "human":
                ok = g[g.error.isna()]
                row.update(cont_tokens_mean=float(ok.continuation_tokens.mean()) if len(ok) else float("nan"),
                           gen_seconds_mean=float(ok.gen_seconds.mean()) if len(ok) else float("nan"),
                           tokens_per_second=float(ok.continuation_tokens.sum() / ok.gen_seconds.sum())
                           if len(ok) and ok.gen_seconds.sum() else float("nan"))
            qok = q[q.error.isna()] if not q.empty else pd.DataFrame()
            if qok.empty:
                rows.append(row)
                continue
            qok = qok.reindex(columns=qok.columns.union(QUALITY_COLUMNS[task.kind]))
            if task.kind == "completion":
                oracles = set(qok.oracle_model.dropna())
                if len(oracles) > 1:
                    log.warning("%s/%s: quality rows come from several oracles %s; re-run stage quality", dataset,
                                variant, sorted(oracles))
                row.update(ppl_corpus=metrics.corpus_perplexity(qok.nll_sum, qok.n_tokens),
                           ppl_median=float(qok.ppl.median()), ppl_mean=float(qok.ppl.mean()),
                           ppl_ci=_ci(*metrics.bootstrap_ci(qok.ppl, np.median, seed=cfg.seed)),
                           diversity=float(qok.diversity.mean()), rep_2=float(qok.rep_2.mean()),
                           log_diversity_markllm=float(qok.log_diversity_markllm.mean()),
                           oracle=" | ".join(sorted(oracles)) if oracles else None)
            elif task.kind == "translation":
                hyp = g[g.error.isna()].set_index("sample_id") if not g.empty else pd.DataFrame()
                ids = [s for s in qok.sample_id if s in hyp.index]
                qb = qok.set_index("sample_id").loc[ids] if ids else qok.iloc[0:0]
                row.update(bleu_corpus=metrics.corpus_bleu(list(hyp.loc[ids].answer), list(hyp.loc[ids].reference))
                           if ids else float("nan"),
                           bleu_sentence_mean=float(qok.bleu_sentence.mean()),
                           bleu_sentence_ci=_ci(*metrics.bootstrap_ci(qok.bleu_sentence, seed=cfg.seed)),
                           answer_words_mean=float(qok.answer_words.mean()), n_bleu=len(ids))
            elif task.kind == "summarization":
                row.update(rouge1=float(qok.rouge1.mean()), rouge2=float(qok.rouge2.mean()),
                           rougeL=float(qok.rougeL.mean()),
                           rougeL_ci=_ci(*metrics.bootstrap_ci(qok.rougeL, seed=cfg.seed)),
                           answer_words_mean=float(qok.answer_words.mean()))
            elif task.kind == "code":
                row.update(pass_at_1=float(qok.passed.astype(float).mean()),
                           pass_at_1_ci=_ci(*metrics.bootstrap_ci(qok.passed.astype(float), seed=cfg.seed)))
            elif task.kind == "math":
                correct = qok.correct.astype(float)
                row.update(accuracy=float(correct.mean()),
                           accuracy_ci=_ci(*metrics.bootstrap_ci(correct, seed=cfg.seed)))
            rows.append(row)
    return pd.DataFrame(rows)


def paired_deltas(qual: pd.DataFrame, cfg: BenchmarkConfig) -> pd.DataFrame:
    """Per-sample paired differences variant - none (same prompts) with bootstrap CIs.

    Besides the absolute difference, every row has the relative change in percent
    (``rel_change_pct`` = mean(variant) / mean(none) - 1, with a paired bootstrap CI). For ``log_ppl``
    the means are geometric-mean perplexities and the change is ``exp(delta) - 1``.
    """
    rows = []
    if qual.empty:
        return pd.DataFrame(rows)
    specs = {"c4": [("log_ppl", "ppl", np.log), ("diversity", "diversity", None)],
             "wmt16_de_en": [("bleu_sentence", "bleu_sentence", None)],
             "cnn_dailymail": [("rougeL", "rougeL", None)],
             "humaneval": [("pass", "passed", None)],
             "gsm8k": [("accuracy", "correct", None)]}
    for dataset, items in specs.items():
        base = qual[(qual.dataset == dataset) & (qual.variant == "none") & qual.error.isna()].set_index("sample_id")
        if base.empty:
            continue
        for variant in [v for v in cfg.variants if v != "none"]:
            q = qual[(qual.dataset == dataset) & (qual.variant == variant) & qual.error.isna()].set_index("sample_id")
            common = base.index.intersection(q.index)
            if len(common) < 2:
                continue
            for metric, col, fn in items:
                if col not in q.columns or col not in base.columns:
                    continue
                a, b = q.loc[common, col].astype(float), base.loc[common, col].astype(float)
                if fn is not None:
                    a, b = fn(a), fn(b)
                lo, hi = metrics.paired_bootstrap_ci(a, b, seed=cfg.seed)
                delta = float(np.nanmean(a - b))
                if fn is np.log:  # back on the perplexity scale: geometric means and their ratio
                    base_mean = float(np.exp(np.nanmean(b)))
                    variant_mean = float(np.exp(np.nanmean(a)))
                    rel = (math.exp(delta) - 1) * 100
                    rel_lo = (math.exp(lo) - 1) * 100 if math.isfinite(lo) else float("nan")
                    rel_hi = (math.exp(hi) - 1) * 100 if math.isfinite(hi) else float("nan")
                else:
                    base_mean = float(np.nanmean(b))
                    variant_mean = float(np.nanmean(a))
                    rel = (variant_mean / base_mean - 1) * 100 if base_mean != 0 else float("nan")
                    rel_lo, rel_hi = metrics.paired_bootstrap_rel_ci(a, b, seed=cfg.seed)
                rows.append({"dataset": dataset, "variant": variant, "metric": metric, "n_pairs": len(common),
                             "delta_vs_none": delta, "ci_low": lo, "ci_high": hi,
                             "base_mean": base_mean, "variant_mean": variant_mean,
                             "rel_change_pct": rel, "rel_ci_low": rel_lo, "rel_ci_high": rel_hi})
    return pd.DataFrame(rows)


def degradation_table(pdlt: pd.DataFrame) -> pd.DataFrame:
    """Headline quality metric per task and watermark: unwatermarked value, watermarked value, % change.

    Built from :func:`paired_deltas` (same prompts, paired bootstrap 95% CI). ``better`` says which
    direction is an improvement, so a negative change is a loss for "higher" metrics and a gain for "lower".
    """
    rows = []
    if pdlt.empty:
        return pd.DataFrame(rows)
    for _, r in pdlt.iterrows():
        key = (r.dataset, r.metric)
        if key not in DEGRADATION_METRICS:
            continue
        label, better = DEGRADATION_METRICS[key]
        rows.append({"dataset": r.dataset, "metric": label, "better": better, "variant": r.variant,
                     "n_pairs": r.n_pairs, "none": r.base_mean, "watermarked": r.variant_mean,
                     "change_pct": r.rel_change_pct, "change_pct_ci": _ci(r.rel_ci_low, r.rel_ci_high, nd=1)})
    return pd.DataFrame(rows)


def main_table(qt: pd.DataFrame, dt: pd.DataFrame, cfg: BenchmarkConfig) -> pd.DataFrame:
    """One row per variant with the headline numbers from every track."""
    order = ["none"] + [v for v in cfg.variants if v != "none"] + ["human"]
    primary = cfg.target_fprs[0]
    rows = []
    for variant in order:
        row = {"variant": variant}
        if not qt.empty:
            c4 = qt[(qt.dataset == "c4") & (qt.variant == variant)]
            if not c4.empty:
                r = c4.iloc[0]
                row.update({"ppl_corpus": r.get("ppl_corpus"), "ppl_median": r.get("ppl_median"),
                            "diversity": r.get("diversity"), "tok/s": r.get("tokens_per_second"),
                            "n_err": r.get("n_err")})
            wmt = qt[(qt.dataset == "wmt16_de_en") & (qt.variant == variant)]
            if not wmt.empty:
                row["bleu"] = wmt.iloc[0].get("bleu_corpus")
            cnn = qt[(qt.dataset == "cnn_dailymail") & (qt.variant == variant)]
            if not cnn.empty:
                row["rougeL"] = cnn.iloc[0].get("rougeL")
            he = qt[(qt.dataset == "humaneval") & (qt.variant == variant)]
            if not he.empty:
                row["pass@1"] = he.iloc[0].get("pass_at_1")
            gsm = qt[(qt.dataset == "gsm8k") & (qt.variant == variant)]
            if not gsm.empty:
                row["gsm8k_acc"] = gsm.iloc[0].get("accuracy")
        if not dt.empty:
            d = dt[dt.variant == variant]
            if not d.empty:
                r = d.iloc[0]
                row.update({"auroc_vs_human": r.get("auroc_human"),
                            fpr_col(primary, "vs_human"): r.get(fpr_col(primary, "human")),
                            "auroc_vs_none": r.get("auroc_none"),
                            fpr_col(primary, "vs_none"): r.get(fpr_col(primary, "none")),
                            "fixed_thr_tpr": r.get("fixed_thr_tpr_human"), "fixed_thr_fpr": r.get("fixed_thr_fpr_human")})
                prefix = "attacked_tpr_at_clean_thr_"
                for col in [c for c in d.columns if c.startswith(prefix)]:
                    row["attacked_tpr_" + col[len(prefix):]] = r.get(col)
        if len(row) > 1:
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- markdown / plots


def df_to_markdown(df: pd.DataFrame, nd: int = 3) -> str:
    if df.empty:
        return "_(no rows)_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c], nd) if not isinstance(r[c], str) else r[c] for c in cols) + " |")
    return "\n".join(lines)


def make_plots(frames: dict, cfg: BenchmarkConfig, out_dir: Path, dt: pd.DataFrame, qt: pd.DataFrame) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    det = frames["detection"]
    variants = [v for v in cfg.variants if v != "none"]
    primary = cfg.target_fprs[0]
    if not det.empty and "dataset" in det:
        d0 = det[det.dataset == DETECTION_DATASETS[0]]
        present = [v for v in variants if v in set(d0.variant)]
        kept = {v: kept_sample_ids(d0[d0.variant == v], cfg, warn=False) for v in present}
        d0 = pd.concat([_same_budget_as_watermarked(
            d0[(d0.variant == v) & d0.sample_id.isin(kept[v]) & d0.detect_tokens.notna()], v, warn=False)
            for v in present]) if present else d0
        if present:
            fig, axes = plt.subplots(1, len(present), figsize=(4 * len(present), 3.2), squeeze=False)
            for ax, variant in zip(axes[0], present):
                d = d0[d0.variant == variant]
                colors = {"human": "tab:gray", "unwatermarked": "tab:blue", "watermarked": "tab:red"}
                extra = ("tab:orange", "tab:green", "tab:purple", "tab:brown")
                for i, name in enumerate(_attack_names(d.text_kind.unique(), cfg)):
                    colors[f"attacked:{name}"] = extra[i % len(extra)]
                for kind, color in colors.items():
                    s = d[d.text_kind == kind].score.dropna()
                    if len(s):
                        ax.hist(s, bins=20, alpha=0.5, label=kind, color=color)
                thr = VARIANTS[variant].fixed_threshold
                if thr is not None:
                    ax.axvline(thr, color="k", ls="--", lw=1, label="shipped threshold")
                ax.set_title(f"{variant}: {VARIANTS[variant].score_name} (n kept={len(kept[variant])})", fontsize=8)
                ax.legend(fontsize=7)
            fig.tight_layout()
            p = out_dir / "score_histograms.png"
            fig.savefig(p, dpi=120)
            plt.close(fig)
            paths.append(p)

            from sklearn.metrics import roc_curve

            fig, ax = plt.subplots(figsize=(4.5, 4))
            for variant in present:
                d = d0[d0.variant == variant]
                pos, neg = d[d.text_kind == "watermarked"].score.dropna(), d[d.text_kind == "human"].score.dropna()
                if len(pos) and len(neg):
                    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
                    fpr, tpr, _ = roc_curve(y, np.r_[pos, neg])
                    ax.plot(fpr, tpr, label=f"{variant} (AUC {metrics.auroc(pos, neg):.3f})")
            ax.plot([0, 1], [0, 1], "k:", lw=0.8)
            ax.set_xlabel("FPR (human text)")
            ax.set_ylabel("TPR (watermarked)")
            ax.set_xscale("symlog", linthresh=0.01)
            ax.legend(fontsize=7)
            fig.tight_layout()
            p = out_dir / "roc_vs_human.png"
            fig.savefig(p, dpi=120)
            plt.close(fig)
            paths.append(p)
    if not qt.empty and not dt.empty and "ppl_corpus" in qt.columns:
        c4 = qt[qt.dataset == "c4"].set_index("variant")
        col = fpr_col(primary, "human")
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        for _, r in dt.iterrows():
            if r.variant in c4.index and col in r and pd.notna(c4.loc[r.variant, "ppl_corpus"]) and pd.notna(r[col]):
                ax.scatter(c4.loc[r.variant, "ppl_corpus"], r[col])
                ax.annotate(r.variant, (c4.loc[r.variant, "ppl_corpus"], r[col]), fontsize=8)
        if "none" in c4.index and pd.notna(c4.loc["none", "ppl_corpus"]):
            ax.axvline(c4.loc["none", "ppl_corpus"], color="k", ls="--", lw=1, label="none (PPL)")
            ax.legend(fontsize=7)
        ax.set_xlabel("corpus PPL (oracle)")
        ax.set_ylabel(f"TPR @ {primary * 100:g}% FPR (vs human)")
        fig.tight_layout()
        p = out_dir / "ppl_vs_tpr.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)
    if not qt.empty:
        bars = []
        for dataset, col in (("wmt16_de_en", "bleu_corpus"), ("cnn_dailymail", "rougeL"), ("humaneval", "pass_at_1"),
                             ("gsm8k", "accuracy")):
            t = qt[(qt.dataset == dataset) & (qt.variant != "human")]
            if not t.empty and col in t and t[col].notna().any():
                bars.append((f"{dataset}: {col}", t.set_index("variant")[col]))
        if bars:
            fig, axes = plt.subplots(1, len(bars), figsize=(4 * len(bars), 3.2), squeeze=False)
            for ax, (title, series) in zip(axes[0], bars):
                series.plot.bar(ax=ax)
                ax.set_title(title, fontsize=9)
                ax.tick_params(axis="x", labelsize=8)
            fig.tight_layout()
            p = out_dir / "task_quality.png"
            fig.savefig(p, dpi=120)
            plt.close(fig)
            paths.append(p)
    return paths


def build_report(cfg: BenchmarkConfig, *, plots: bool = True) -> dict:
    """Write summary.csv / summary.md / results.json (+ plots) for a run and return the tables."""
    run_dir = cfg.run_dir
    frames = load_run(run_dir)
    gens, det, qual = frames["generations"], frames["detection"], frames["quality"]
    dt = detection_table(det, cfg, gens, frames.get("attacks"))
    qt = quality_table(qual, gens, cfg)
    pdlt = paired_deltas(qual, cfg)
    dg = degradation_table(pdlt)
    mt = main_table(qt, dt, cfg)
    mt.to_csv(run_dir / "summary.csv", index=False)
    dt.to_csv(run_dir / "detection_summary.csv", index=False)
    qt.to_csv(run_dir / "quality_summary.csv", index=False)
    pdlt.to_csv(run_dir / "paired_deltas.csv", index=False)
    dg.to_csv(run_dir / "degradation.csv", index=False)
    plot_paths: list[Path] = []
    if plots:
        try:
            plot_paths = make_plots(frames, cfg, run_dir / "plots", dt, qt)
        except Exception as exc:  # noqa: BLE001 - a plotting failure must not lose the tables
            log.warning("plotting failed: %s", exc)

    primary = cfg.target_fprs[0]
    md = [f"# wmbench results: `{cfg.run_name}`", ""]
    if cfg.smoke:
        md += ["> **SMOKE TEST** - tiny model, a handful of samples, short generations. These numbers only",
               "> validate that the pipeline runs; they say nothing about the watermarks.", ""]
    env = frames["manifest"].get("environment", {})
    md += [f"Model: `{cfg.model_name}` | oracle: `{cfg.oracle_model_name or cfg.model_name}` | "
           f"device: {env.get('device_type')} ({env.get('dtype')}) | "
           f"MarkLLM: `{frames['manifest'].get('markllm_commit', '')[:8]}` | config hash: `{cfg.config_hash()}`", ""]
    md += ["## Main table", "", df_to_markdown(mt), ""]
    md += [f"Detection columns use C4 continuations; negatives are human continuations (`vs_human`) or the `none` "
           f"variant's generations (`vs_none`). Per sample and variant the watermarked, human and `none` texts are "
           f"scored at one common token budget (the shortest of the three, at most `max_new_tokens`); samples whose "
           f"budget is below `min_detect_tokens` ({cfg.min_detect_tokens}) are excluded (`n_gen_ok` vs "
           f"`n_samples_kept` in detection_summary.csv). `fixed_thr_*` use each algorithm's shipped threshold. "
           f"`attacked_tpr_<attack>` is the TPR of the attacked watermarked texts at the threshold fixed on the clean "
           f"human negatives; attack strength is in `attack_len_ratio_*`, `attack_bleu_vs_source_*` and "
           f"`attack_sim_embed_*` of detection_summary.csv. "
           f"`tpr@{primary * 100:g}%fpr` is unreliable when fewer than {int(1 / primary)} negatives are available "
           f"(see `fpr_warning_*` in detection_summary.csv).", ""]
    md += ["## Quality degradation vs. unwatermarked (%)", "",
           "Same prompts for every variant. `change_pct` = watermarked / unwatermarked - 1, with a paired "
           "bootstrap 95% CI. `better` says which direction is an improvement (perplexity: lower).", "",
           df_to_markdown(dg), ""]
    md += ["## Detection", "", df_to_markdown(dt), "", "## Quality per dataset", "", df_to_markdown(qt), "",
           "## Paired differences vs `none` (same prompts, bootstrap 95% CI)", "", df_to_markdown(pdlt), ""]
    if plot_paths:
        md += ["## Plots", ""] + [f"![{p.stem}](plots/{p.name})" for p in plot_paths] + [""]
    (run_dir / "summary.md").write_text("\n".join(md))
    (run_dir / "results.json").write_text(json.dumps({
        "main": _records(mt), "detection": _records(dt), "quality": _records(qt), "paired_deltas": _records(pdlt),
        "degradation": _records(dg)},
        indent=2, allow_nan=False))
    log.info("report written to %s", run_dir / "summary.md")
    return {"main": mt, "detection": dt, "quality": qt, "paired_deltas": pdlt, "degradation": dg,
            "plots": plot_paths, "frames": frames}
