#!/usr/bin/env python
"""Paper-ready LaTeX tables and PDF figures for one wmbench run.

    python scripts/paper_assets.py --run run_results/qwen15b-20261020-142513 --out paper --compile

Writes ``<out>/tables/*.tex`` (booktabs floats with draft captions), ``<out>/figures/*.pdf`` (vector,
TrueType fonts embedded) each with a ``.tex`` float snippet and a ``.png`` preview, ``<out>/numbers.json``
(every printed value, full precision) and ``<out>/preview.tex``: a two-column gallery (3.3 in columns,
6.9 in text width) that shows every option at its printed size. ``--compile`` builds it with tectonic.
Runs with an attack stage also get ``<out>/numbers_robustness.json`` (see ``compute_robustness``).

Numbers come from the run's JSONL records through ``wmbench.report``'s loaders and filters, so they match
summary.md (checked on load). Computed here in addition: paired bootstrap CIs of the corpus-PPL ratio and
of the corpus-BLEU difference (the statistics the tables print), CIs of the absolute metrics,
standardized detection scores and the calibration of the analytic null distributions. For attacked text:
CIs of TPR@1% FPR and AUROC, paired differences between methods, retained detection margin, the share of
attacked tokens whose watermark context survives (needs the generator tokenizer), the effective FPR of the
mean-score detectors at the attacked length, and how often SIR's context exceeded its 512-token embedder.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from wmbench import metrics, report  # noqa: E402
from wmbench.config import CONFIG_DIR, BenchmarkConfig  # noqa: E402
from wmbench.watermarks import VARIANTS  # noqa: E402

log = logging.getLogger("paper_assets")

# ----------------------------------------------------------------------------- style

METHOD_ORDER = ("KGW", "MorphMark", "PF", "SynthID", "SIR")
# Categorical slots 1-5 of the dataviz reference palette in fixed order, checked with its validator on a
# white surface: adjacent pairs pass; slots 1-3 also pass all-pairs (the calibration plot shows exactly
# those three). Slots 3-5 sit below 3:1 contrast on white, so every figure names methods in text as well.
COLOR = {"KGW": "#2a78d6", "MorphMark": "#eb6834", "PF": "#1baf7a", "SynthID": "#eda100", "SIR": "#e87ba4",
         "none": "#52514e", "human": "#898781"}
MARKER = {"KGW": "o", "MorphMark": "s", "PF": "D", "SynthID": "^", "SIR": "v", "none": "X", "human": "*"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
COL_W, TEXT_W = 3.3, 6.9  # inches; column widths of two-column venues: ACL 3.03, ICML/AAAI 3.25-3.3, IEEE 3.5
N_BOOT = 2000
KINDS = ("watermarked", "human", "unwatermarked")
NAME = {"none": "Unwatermarked", "human": "Human"}
SCORE_LABEL = {"KGW": "z-score", "MorphMark": "z-score", "PF": r"$-\log_{10}\,p$", "SynthID": "Mean g-value",
               "SIR": "Mean bias similarity"}
SCORE_TEX = {"KGW": "$z$-score", "MorphMark": "$z$-score", "PF": r"$-\log_{10} p$", "SynthID": r"mean $g$",
             "SIR": "mean bias sim."}


ATTACK_SHORT = {"word_deletion": "Del.", "dipper": "DIPPER"}
ATTACK_LONG = {"word_deletion": "Word deletion", "dipper": "DIPPER paraphrase"}


def display(v: str) -> str:
    return NAME.get(v, v)


def attack_phrase(run: Run, a: str) -> str:
    """The attack and its strength setting, for captions."""
    if a == "word_deletion":
        return f"word deletion (Del.; each word dropped with probability {run.cfg.word_deletion_ratio:g})"
    if a == "dipper":
        return (f"DIPPER paraphrasing (lexical diversity {run.cfg.dipper_lex_diversity}, "
                f"order diversity {run.cfg.dipper_order_diversity})")
    return a.replace("_", " ")


def context_label(m: str) -> str:
    h = context_width(m)
    return "semantic" if h is None else f"$h={h}$"


# ----------------------------------------------------------------------------- loading


@dataclasses.dataclass
class Run:
    cfg: BenchmarkConfig
    manifest: dict
    frames: dict
    dt: pd.DataFrame  # report.detection_table
    qt: pd.DataFrame  # report.quality_table
    det: dict[str, pd.DataFrame]  # C4 detection rows per method: kept samples, common budget (as in the report)
    methods: list[str]
    rob: dict | None = None  # compute_robustness, for runs with an attack stage


def load(run_dir: Path) -> Run:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    c = dict(manifest["config"])
    for k in ("variants", "datasets", "target_fprs", "attacks"):
        if isinstance(c.get(k), list):
            c[k] = tuple(c[k])
    known = {f.name for f in dataclasses.fields(BenchmarkConfig)}
    c = {k: v for k, v in c.items() if k in known}
    c.update(results_root=str(run_dir.parent), run_name=run_dir.name)  # the manifest holds the cluster path
    cfg = BenchmarkConfig(**c)
    frames = report.load_run(run_dir)
    dt = report.detection_table(frames["detection"], cfg, frames["generations"], frames.get("attacks"))
    qt = report.quality_table(frames["quality"], frames["generations"], cfg)
    methods = [m for m in METHOD_ORDER if m in cfg.variants and m in set(dt.variant)]
    d = frames["detection"]
    det = {}
    for m in methods:
        dv = d[(d.dataset == "c4") & (d.variant == m)]
        keep = report.kept_sample_ids(dv, cfg, warn=False)
        det[m] = report._same_budget_as_watermarked(dv[dv.sample_id.isin(keep) & dv.detect_tokens.notna()], m,
                                                    warn=False)
    return Run(cfg, manifest, frames, dt, qt, det, methods)


def scores(run: Run, m: str, kind: str) -> np.ndarray:
    d = run.det[m]
    return d.loc[d.text_kind == kind, "score"].astype(float).dropna().to_numpy()


def _quality_rows(run: Run, dataset: str, variant: str) -> pd.DataFrame:
    q = run.frames["quality"]
    return q[(q.dataset == dataset) & (q.variant == variant) & q.error.isna()].set_index("sample_id")


def _generation_rows(run: Run, dataset: str, variant: str) -> pd.DataFrame:
    g = run.frames["generations"]
    return g[(g.dataset == dataset) & (g.variant == variant) & g.error.isna()].set_index("sample_id")


# ----------------------------------------------------------------------------- statistics


def _nll(q: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    nll, n = q.nll_sum.to_numpy(float), q.n_tokens.to_numpy(float)
    ok = np.isfinite(nll) & (n > 0)  # empty continuations carry no tokens (same rule as metrics.corpus_perplexity)
    return nll[ok], n[ok]


def corpus_ppl_ci(q: pd.DataFrame, seed: int) -> tuple[float, float, float]:
    nll, n = _nll(q)
    idx = np.random.default_rng(seed).integers(0, len(nll), (N_BOOT, len(nll)))
    boot = np.exp(nll[idx].sum(1) / n[idx].sum(1))
    return math.exp(nll.sum() / n.sum()), *np.quantile(boot, [0.025, 0.975])


def paired_ppl_change(a: pd.DataFrame, b: pd.DataFrame, seed: int) -> tuple[float, float, float]:
    """Corpus-PPL ratio a/b - 1 in percent on the prompts both have, paired bootstrap over prompts."""
    ids = a.index.intersection(b.index)
    a, b = a.loc[ids], b.loc[ids]
    ok = (np.isfinite(a.nll_sum.to_numpy(float)) & (a.n_tokens.to_numpy(float) > 0)
          & np.isfinite(b.nll_sum.to_numpy(float)) & (b.n_tokens.to_numpy(float) > 0))
    an, at = a.nll_sum.to_numpy(float)[ok], a.n_tokens.to_numpy(float)[ok]
    bn, bt = b.nll_sum.to_numpy(float)[ok], b.n_tokens.to_numpy(float)[ok]
    est = math.exp(an.sum() / at.sum() - bn.sum() / bt.sum())
    idx = np.random.default_rng(seed).integers(0, len(an), (N_BOOT, len(an)))
    boot = np.exp(an[idx].sum(1) / at[idx].sum(1) - bn[idx].sum(1) / bt[idx].sum(1))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return 100 * (est - 1), 100 * (lo - 1), 100 * (hi - 1)


def paired_mean_delta(a: pd.Series, b: pd.Series, seed: int, scale: float = 1.0) -> tuple[float, float, float]:
    ids = a.index.intersection(b.index)
    x, y = a.loc[ids].to_numpy(float), b.loc[ids].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    lo, hi = metrics.paired_bootstrap_ci(x, y, n_boot=N_BOOT, seed=seed)
    return scale * float(np.mean(x[ok] - y[ok])), scale * lo, scale * hi


class CorpusBleu:
    """sacreBLEU corpus BLEU from per-sentence sufficient statistics, so bootstrap resamples are cheap."""

    def __init__(self):
        from sacrebleu.metrics import BLEU

        self.bleu = BLEU()

    def stats(self, g: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.bleu._extract_corpus_statistics(list(g.answer), [list(g.reference)]), dtype=float)

    def score(self, summed) -> float:
        return float(self.bleu._compute_score_from_stats([int(round(x)) for x in summed]).score)

    def ci(self, g: pd.DataFrame, seed: int) -> tuple[float, float, float]:
        s = self.stats(g)
        idx = np.random.default_rng(seed).integers(0, len(s), (N_BOOT, len(s)))
        boot = [self.score(s[i].sum(0)) for i in idx]
        return self.score(s.sum(0)), *np.quantile(boot, [0.025, 0.975])

    def paired_delta(self, a: pd.DataFrame, b: pd.DataFrame, seed: int) -> tuple[float, float, float]:
        ids = a.index.intersection(b.index)
        sa, sb = self.stats(a.loc[ids]), self.stats(b.loc[ids])
        idx = np.random.default_rng(seed).integers(0, len(ids), (N_BOOT, len(ids)))
        boot = [self.score(sa[i].sum(0)) - self.score(sb[i].sum(0)) for i in idx]
        return self.score(sa.sum(0)) - self.score(sb.sum(0)), *np.quantile(boot, [0.025, 0.975])


def reported_scores(run: Run, m: str, kind: str) -> pd.Series:
    """The detector's reported scores of one text kind, indexed by sample id."""
    rows = run.det[m]
    return rows[rows.text_kind == kind].set_index("sample_id").score.astype(float).dropna()


def linear_scores(run: Run, m: str, kind: str) -> pd.Series:
    """Detector scores of one text kind on a roughly linear scale, indexed by sample id.

    PF's reported score is a log p-value; its normal approximation ``(S - k) / sqrt(k)`` of the
    Gamma(k, 1) test on the summed score ``S`` over ``k`` scored tokens is used instead.
    """
    rows = run.det[m]
    rows = rows[rows.text_kind == kind].set_index("sample_id")
    if m == "PF":
        s = rows.raw.apply(lambda r: (r["raw_sum"] - r["n_scored"]) / math.sqrt(r["n_scored"])
                           if isinstance(r, dict) and r.get("n_scored") else np.nan)
    else:
        s = rows.score
    return s.astype(float).dropna()


def standardized_scores(run: Run, m: str, kinds=KINDS) -> dict[str, np.ndarray]:
    """Scores in units of the detector's spread on human text: ``(s - mean_human) / sd_human``.

    All detectors are thereby on a comparable scale (d' in signal-detection terms); PF enters through
    the normal approximation of ``linear_scores``.
    """
    h = linear_scores(run, m, "human").to_numpy()
    mu, sd = float(h.mean()), float(h.std(ddof=1))
    return {k: (linear_scores(run, m, k).to_numpy() - mu) / sd for k in kinds}


def null_pvalues(run: Run, m: str) -> np.ndarray | None:
    """p-values of the non-watermarked texts (human + unwatermarked) under the method's analytic null."""
    from scipy.stats import norm

    d = run.det[m]
    neg = d[d.text_kind.isin(("human", "unwatermarked"))]
    if m == "PF":
        p = neg.raw.apply(lambda r: r.get("p_value") if isinstance(r, dict) else np.nan)
    elif VARIANTS[m].score_name.startswith("z-score"):
        p = pd.Series(norm.sf(neg.score.astype(float).to_numpy()), index=neg.index)
    else:
        return None  # SynthID's mean g-value and SIR's mean similarity ship without a null distribution
    return p.astype(float).dropna().to_numpy()


def _close(a, b, what: str, tol: float = 1e-9) -> None:
    if not (np.isfinite(a) and np.isfinite(b) and abs(a - b) <= tol * max(1.0, abs(b))):
        raise AssertionError(f"{what}: recomputed {a!r} != report {b!r}")


def compute_numbers(run: Run) -> pd.DataFrame:
    """Every number the tables and figures print, one row per variant (none, methods..., human)."""
    seed = run.cfg.seed
    qt = run.qt.set_index(["dataset", "variant"])
    dt = run.dt.set_index("variant")
    bleu = CorpusBleu()

    def q(ds, v, col):
        try:
            x = qt.loc[(ds, v), col]
        except KeyError:
            return float("nan")
        return float(x) if x is not None and pd.notna(x) else float("nan")

    base_c4, base_wmt, base_cnn = (_quality_rows(run, "c4", "none"), _generation_rows(run, "wmt16_de_en", "none"),
                                   _quality_rows(run, "cnn_dailymail", "none"))
    rows = {}
    for v in ("none", *run.methods, "human"):
        r: dict = {}
        c4 = _quality_rows(run, "c4", v)
        r["ppl"], r["ppl_lo"], r["ppl_hi"] = corpus_ppl_ci(c4, seed)
        _close(r["ppl"], q("c4", v, "ppl_corpus"), f"{v} corpus PPL")
        r["ppl_median"] = q("c4", v, "ppl_median")
        r["ppl_median_lo"], r["ppl_median_hi"] = metrics.bootstrap_ci(c4.ppl, np.median, n_boot=N_BOOT, seed=seed)
        r["diversity"] = 100 * q("c4", v, "diversity")
        lo, hi = metrics.bootstrap_ci(c4.diversity, n_boot=N_BOOT, seed=seed)
        r["diversity_lo"], r["diversity_hi"] = 100 * lo, 100 * hi
        r["rep2"] = 100 * q("c4", v, "rep_2")
        r["n_c4"] = len(c4)
        if v != "human":
            wmt = _generation_rows(run, "wmt16_de_en", v)
            r["bleu"], r["bleu_lo"], r["bleu_hi"] = bleu.ci(wmt, seed)
            _close(r["bleu"], q("wmt16_de_en", v, "bleu_corpus"), f"{v} corpus BLEU", tol=1e-6)
            r["sbleu"] = q("wmt16_de_en", v, "bleu_sentence_mean")
            r["mt_words"] = q("wmt16_de_en", v, "answer_words_mean")
            cnn = _quality_rows(run, "cnn_dailymail", v)
            r["r1"], r["r2"], r["rl"] = (100 * q("cnn_dailymail", v, c) for c in ("rouge1", "rouge2", "rougeL"))
            lo, hi = metrics.bootstrap_ci(cnn.rougeL, n_boot=N_BOOT, seed=seed)
            r["rl_lo"], r["rl_hi"] = 100 * lo, 100 * hi
            r["sum_words"] = q("cnn_dailymail", v, "answer_words_mean")
            for ds, key in (("c4", "c4"), ("wmt16_de_en", "wmt"), ("cnn_dailymail", "cnn")):
                r[f"tps_{key}"] = q(ds, v, "tokens_per_second")
        if v in run.methods:
            r["d_ppl"], r["d_ppl_lo"], r["d_ppl_hi"] = paired_ppl_change(c4, base_c4, seed)
            r["d_div"], r["d_div_lo"], r["d_div_hi"] = paired_mean_delta(c4.diversity, base_c4.diversity, seed, 100)
            r["d_bleu"], r["d_bleu_lo"], r["d_bleu_hi"] = bleu.paired_delta(wmt, base_wmt, seed)
            r["d_rl"], r["d_rl_lo"], r["d_rl_hi"] = paired_mean_delta(cnn.rougeL, base_cnn.rougeL, seed, 100)
            pos, hum, unw = scores(run, v, "watermarked"), scores(run, v, "human"), scores(run, v, "unwatermarked")
            r["n_det"] = len(pos)
            r["auroc"] = metrics.auroc(pos, hum)
            _close(r["auroc"], dt.loc[v, "auroc_human"], f"{v} AUROC")
            r["tpr1"] = metrics.tpr_at_fpr(pos, hum, 0.01)[0]
            r["tpr5"] = metrics.tpr_at_fpr(pos, hum, 0.05)[0]
            _close(r["tpr1"], dt.loc[v, "tpr@1%fpr_human"], f"{v} TPR@1%")
            boot = metrics.detection_bootstrap(pos, hum, 0.01, n_boot=N_BOOT, seed=seed)
            (r["auroc_lo"], r["auroc_hi"]), (r["tpr1_lo"], r["tpr1_hi"]) = boot["auroc"], boot["tpr_at_fpr"]
            r["auroc_unwm"] = metrics.auroc(pos, unw)
            r["tpr1_unwm"] = metrics.tpr_at_fpr(pos, unw, 0.01)[0]
            r["thr"] = VARIANTS[v].fixed_threshold
            r["nominal_fpr"] = VARIANTS[v].nominal_fpr
            r["thr_tpr"] = float(dt.loc[v, "fixed_thr_tpr_human"])
            r["thr_fpr_human"] = float(dt.loc[v, "fixed_thr_fpr_human"])
            r["thr_fpr_unwm"] = float(dt.loc[v, "fixed_thr_fpr_none"])
            r["detect_tokens_mean"] = float(dt.loc[v, "detect_tokens_mean"])
            neg = np.r_[hum, unw]  # every non-watermarked text of the kept samples
            r["n_neg_pooled"] = len(neg)
            r["tpr_zero_fp"] = float(np.mean(pos > neg.max()))  # TPR when not a single false positive is allowed
            r["separation_gap"] = float(pos.min() - neg.max())  # > 0: perfect separation
            p = null_pvalues(run, v)
            if p is not None:  # observed FPR at nominal 1% / 5% under the analytic null (figure F9)
                r["calib_fpr_1"], r["calib_fpr_5"] = float(np.mean(p <= 0.01)), float(np.mean(p <= 0.05))
            wmt_q, base_wmt_q = _quality_rows(run, "wmt16_de_en", v), _quality_rows(run, "wmt16_de_en", "none")
            r["d_sbleu"], r["d_sbleu_lo"], r["d_sbleu_hi"] = paired_mean_delta(wmt_q.bleu_sentence,
                                                                               base_wmt_q.bleu_sentence, seed)
            z = standardized_scores(run, v)["watermarked"]
            r["margin_median"], r["margin_p01"] = float(np.median(z)), float(np.percentile(z, 1))
        rows[v] = r
    n = pd.DataFrame.from_dict(rows, orient="index")
    for key in ("tps_c4", "tps_wmt", "tps_cnn"):
        n[f"rel_{key}"] = 100 * n[key] / n.loc["none", key]
    return n


# ----------------------------------------------------------------------------- robustness statistics

SUM_TYPE = {"KGW", "MorphMark", "PF"}  # scores that grow with sqrt(length); SynthID and SIR average per token


def context_width(m: str) -> int | None:
    """Preceding tokens the method's key depends on (None: SIR keys on a semantic embedding)."""
    c = json.loads((CONFIG_DIR / f"{m}.json").read_text())
    if m in ("KGW", "MorphMark"):
        return int(c["prefix_length"])
    if m == "SynthID":
        return int(c["ngram_len"]) - 1  # an n-gram of ngram_len = ngram_len - 1 context tokens + the token
    if m == "PF":
        return int(c["ngram"])
    return None


def _fast_auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUROC (ties count one half), identical to sklearn's roc_auc_score."""
    from scipy.stats import rankdata

    r = rankdata(np.r_[pos, neg])
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _paired_method_delta(a_pos, a_neg, b_pos, b_neg, fpr: float, seed: int) -> dict:
    """Method a minus method b on the prompts both have: TPR@fpr and AUROC, paired bootstrap over prompts.

    Each method keeps its own human negatives (scored by its own detector); a resample draws the same
    prompts for the positives and negatives of both methods.
    """
    ids = a_pos.index.intersection(a_neg.index).intersection(b_pos.index).intersection(b_neg.index)
    ap, an, bp, bn = (s.loc[ids].to_numpy() for s in (a_pos, a_neg, b_pos, b_neg))

    def stats(i):
        return (metrics.tpr_at_fpr(ap[i], an[i], fpr)[0] - metrics.tpr_at_fpr(bp[i], bn[i], fpr)[0],
                _fast_auroc(ap[i], an[i]) - _fast_auroc(bp[i], bn[i]))

    d_tpr, d_auc = stats(np.arange(len(ids)))
    idx = np.random.default_rng(seed).integers(0, len(ids), (N_BOOT, len(ids)))
    boot = np.array([stats(i) for i in idx])
    (t_lo, a_lo), (t_hi, a_hi) = np.quantile(boot, [0.025, 0.975], axis=0)
    return {"n_common": int(len(ids)), "d_tpr1": d_tpr, "d_tpr1_lo": t_lo, "d_tpr1_hi": t_hi,
            "d_auroc": d_auc, "d_auroc_lo": a_lo, "d_auroc_hi": a_hi}


def context_survival(src: list[int], att: list[int], h: int) -> tuple[int, int]:
    """(surviving, total) attacked-text positions whose (h+1)-gram (h context tokens + the token) occurs in src."""
    grams = {tuple(src[i - h:i + 1]) for i in range(h, len(src))}
    total = max(0, len(att) - h)
    return sum(tuple(att[i - h:i + 1]) in grams for i in range(h, len(att))), total


def load_tokenizers(run: Run):
    """(generator tokenizer, SIR embedder tokenizer); None for one that cannot be loaded (e.g. offline)."""
    from wmbench.bootstrap import SIR_EMBEDDER_REPO
    from wmbench.models import load_tokenizer

    toks = []
    for name in (run.cfg.tokenizer_name or run.cfg.model_name, SIR_EMBEDDER_REPO):
        try:
            toks.append(load_tokenizer(name))
        except Exception as exc:  # noqa: BLE001 - the tokenizer-based statistics are optional
            log.warning("tokenizer %s unavailable (%s); skipping the statistics that need it", name, exc)
            toks.append(None)
    return tuple(toks)


def compute_robustness(run: Run, tokenizer=None, sir_tokenizer=None) -> dict:
    """Robustness numbers: per attack and method, between methods, and the context-width mechanism.

    ``tokenizer`` is the generator's (for n-gram survival), ``sir_tokenizer`` SIR's embedder tokenizer
    (for the context-truncation check); sections whose tokenizer is missing are skipped.
    """
    from scipy.stats import norm, spearmanr

    seed, fpr = run.cfg.seed, 0.01
    dt = run.dt.set_index("variant")
    kinds = set().union(*(set(run.det[m].text_kind) for m in run.methods))
    attacks = report._attack_names(kinds, run.cfg)
    out: dict = {"attacks": attacks, "context_width": {m: context_width(m) for m in run.methods},
                 "per_method": {}, "pairs": {}, "rank_correlation": {}, "survival": {}, "length_mismatch": {}}
    a_rows = run.frames.get("attacks")
    gens = run.frames["generations"]
    for a in attacks:
        per = {}
        # Rank statistics (TPR, AUROC) use the detectors' reported scores, as the report does; margins use
        # the linear scale of ``standardized_scores``.
        rep = {m: (reported_scores(run, m, f"attacked:{a}"), reported_scores(run, m, "human")) for m in run.methods}
        for m in run.methods:
            att, hum = rep[m]
            r = {"n": int(len(att)), "tpr1": metrics.tpr_at_fpr(att, hum, fpr)[0],
                 "tpr5": metrics.tpr_at_fpr(att, hum, 0.05)[0], "auroc": metrics.auroc(att, hum)}
            _close(r["tpr1"], dt.loc[m, f"attacked_tpr_at_clean_thr_{a}"], f"{m} {a} TPR@1%")
            _close(r["auroc"], dt.loc[m, f"attacked_auroc_human_{a}"], f"{m} {a} AUROC")
            boot = metrics.detection_bootstrap(att, hum, fpr, n_boot=N_BOOT, seed=seed)
            (r["auroc_lo"], r["auroc_hi"]), (r["tpr1_lo"], r["tpr1_hi"]) = boot["auroc"], boot["tpr_at_fpr"]
            z = standardized_scores(run, m, ("watermarked", f"attacked:{a}"))
            r["margin_clean_median"] = float(np.median(z["watermarked"]))
            r["margin_median"] = float(np.median(z[f"attacked:{a}"]))
            r["margin_retained"] = r["margin_median"] / r["margin_clean_median"]
            per[m] = r
        out["per_method"][a] = per
        out["pairs"][a] = {f"{x}-{y}": _paired_method_delta(*rep[x], *rep[y], fpr, seed)
                           for i, x in enumerate(run.methods) for y in run.methods[i + 1:]}
        clean = [per[m]["margin_clean_median"] for m in run.methods]
        rho, p = spearmanr(clean, [per[m]["tpr1"] for m in run.methods])
        out["rank_correlation"][a] = {"clean_margin_vs_tpr1_rho": float(rho), "p": float(p), "n": len(clean)}

        # Effective FPR of the clean 1% threshold at the attacked length, for the per-token mean detectors:
        # their spread on human text scales as 1/sqrt(length) (normal approximation).
        out["length_mismatch"][a] = {}
        for m in (x for x in run.methods if x not in SUM_TYPE):
            d = run.det[m]
            hum_rows, att_rows = d[d.text_kind == "human"], d[d.text_kind == f"attacked:{a}"]
            h = hum_rows.score.astype(float)
            _, thr = metrics.tpr_at_fpr(att_rows.score, h, fpr)
            l_h, l_a = float(hum_rows.n_tokens.mean()), float(att_rows.n_tokens.mean())
            sd = float(h.std(ddof=1)) * math.sqrt(l_h / l_a)
            out["length_mismatch"][a][m] = {"tokens_human": l_h, "tokens_attacked": l_a,
                                            "effective_fpr": float(norm.sf((thr - h.mean()) / sd))}

        if tokenizer is None or a_rows is None or a_rows.empty:
            continue
        surv = {}
        for m in run.methods:
            keep = set(run.det[m].sample_id)
            ar = a_rows[(a_rows.dataset == "c4") & (a_rows.variant == m) & (a_rows.attack == a)
                        & a_rows.error.isna() & a_rows.sample_id.isin(keep)].drop_duplicates("sample_id", keep="last")
            src = gens[(gens.dataset == "c4") & (gens.variant == m)].drop_duplicates("sample_id", keep="last") \
                .set_index("sample_id").continuation
            per_text: dict[int, list[float]] = {1: [], 4: [], 8: []}
            pooled = {h: [0, 0] for h in per_text}
            for row in ar.itertuples():
                s_ids = tokenizer(src.loc[row.sample_id] or "", add_special_tokens=False)["input_ids"][: int(row.detect_tokens)]
                a_ids = tokenizer(row.text or "", add_special_tokens=False)["input_ids"]
                for h in per_text:
                    k, n_ = context_survival(s_ids, a_ids, h)
                    per_text[h].append(k / n_ if n_ else np.nan)
                    pooled[h][0] += k
                    pooled[h][1] += n_
            s = {f"h{h}": pooled[h][0] / pooled[h][1] for h in per_text}
            s["len_ratio_scored"] = float((ar.n_tokens / ar.detect_tokens).mean())
            hw = context_width(m)
            if hw in per_text:
                retained = per[m]["margin_retained"]
                s["implied_survival"] = retained / math.sqrt(s["len_ratio_scored"]) if m in SUM_TYPE else retained
                s["measured_survival"] = s[f"h{hw}"]
                att = linear_scores(run, m, f"attacked:{a}")
                frac = pd.Series(per_text[hw], index=ar.sample_id.to_numpy())
                ids = att.index.intersection(frac.dropna().index)
                rho, p = spearmanr(att.loc[ids], frac.loc[ids])
                s["score_vs_survival_rho"], s["score_vs_survival_p"] = float(rho), float(p)
            surv[m] = s
        out["survival"][a] = surv

    if sir_tokenizer is not None and "SIR" in run.methods:
        # SIR embeds the whole decoded context (prompt + text so far), truncated at 512 embedder tokens:
        # a sample exceeded the limit during generation if its full context is longer at the last step.
        out["sir_context_truncated"] = {}
        for ds in run.cfg.datasets:
            g = gens[(gens.dataset == ds) & (gens.variant == "SIR") & gens.error.isna()]
            if g.empty:
                continue
            n_emb = [len(sir_tokenizer((p or "") + (c or ""), add_special_tokens=True)["input_ids"])
                     for p, c in zip(g.prompt_text, g.continuation)]
            out["sir_context_truncated"][ds] = {"n": len(n_emb), "n_over_512": int(sum(x > 512 for x in n_emb))}
    return out


def significant(n: pd.DataFrame, v: str, key: str) -> bool:
    lo, hi = n.loc[v, f"{key}_lo"], n.loc[v, f"{key}_hi"]
    return bool(lo > 0 or hi < 0)


# ----------------------------------------------------------------------------- LaTeX helpers


def num(x, nd: int = 1, *, signed: bool = False, negative_zero: bool = False) -> str:
    """Fixed-point number for LaTeX text mode; '--' for missing; a real minus sign.

    A value that rounds to zero prints unsigned, except with ``negative_zero`` (CI bounds): an upper
    bound of -0.02 prints as '-0.0', so a significant interval never reads as touching zero.
    """
    if x is None or not np.isfinite(x):
        return "--"
    s = f"{abs(float(x)):.{nd}f}"
    if float(s) == 0 and not (negative_zero and x < 0):
        return s
    if x < 0:
        return "$-$" + s
    return "$+$" + s if signed else s


def ci(lo, hi, nd: int = 1) -> str:
    return rf"[{num(lo, nd, negative_zero=True)},\,{num(hi, nd, negative_zero=True)}]"


def pct_small(x: float, nd: int = 1) -> str:
    """A rate given as a fraction, printed in percent; tiny values keep one significant digit (0.003, not 0.0)."""
    if x is None or not np.isfinite(x):
        return "--"
    p = 100 * x
    if 0 < p < 0.05:
        return f"{p:.{-int(math.floor(math.log10(p)))}f}"
    return num(p, nd)


_TEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
                "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
_TEX_UNICODE = {"\u2019": "'", "\u2018": "`", "\u201c": "``", "\u201d": "''", "\u2014": "---", "\u2013": "--",
                "\u2026": r"\ldots{}", "\u00a0": "~"}


def tex_text(s: str) -> str:
    """Arbitrary text as pdfLaTeX-safe LaTeX (special characters escaped, Unicode punctuation mapped)."""
    s = "".join(_TEX_SPECIAL.get(ch, ch) for ch in s)
    parts = s.split('"')
    s = "".join(p + ("``" if i % 2 == 0 else "''") for i, p in enumerate(parts[:-1])) + parts[-1]
    for k, v in _TEX_UNICODE.items():
        s = s.replace(k, v)
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def _header(run: Run, title: str, packages: str = "booktabs") -> str:
    return (f"% {title}\n% Generated by scripts/paper_assets.py from run '{run.cfg.run_name}' "
            f"(config {run.manifest.get('config_hash')}); regenerate rather than edit the numbers.\n"
            f"% Requires \\usepackage{{{packages}}}.\n")


def _float(env: str, body: str, caption: str, label: str, size: str = r"\small") -> str:
    return (f"\\begin{{{env}}}[t]\n\\centering\n{size}\n{body}"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{{env}}}\n")


STACK = r"\providecommand{\wmbstack}[2]{\begin{tabular}[c]{@{}c@{}}#1\\[-2pt]{\scriptsize #2}\end{tabular}}" + "\n"


def _row(cells: list[str]) -> str:
    return " & ".join(cells) + r" \\" + "\n"


def _rows(run: Run) -> list[str]:
    return ["none", *run.methods, "human"]


def _name_cell(v: str) -> str:
    return rf"\textit{{{display(v)}}}" if v in NAME else v


# ----------------------------------------------------------------------------- tables


def _captions(run: Run, n: pd.DataFrame) -> dict[str, str]:
    cfg = run.cfg
    nd = n.loc[run.methods, "n_det"]
    n_tex = (f"$n={int(nd.min())}$" if nd.min() == nd.max()  # the en dash must stay outside math mode
             else f"$n={int(nd.min())}$--${int(nd.max())}$")
    oracle = (cfg.oracle_model_name or cfg.model_name).split("/")[-1].upper()  # facebook/opt-2.7b -> OPT-2.7B
    model = cfg.model_name.split("/")[-1].upper()
    return {
        "n": n_tex,
        "oracle": oracle,
        "model": model,
        "detect": (f"Detection on C4 continuations against human-written continuations of the same prompts "
                   f"({n_tex} per method; watermarked and human texts cut to a common budget of at most "
                   f"{cfg.max_new_tokens['c4']} tokens, mean {n.loc[run.methods, 'detect_tokens_mean'].mean():.0f})"),
        "tasks": (f"PPL: corpus perplexity of the C4 continuations under {oracle}; Div.: SimCTG diversity (\\%); "
                  f"BLEU: sacreBLEU on WMT16 de$\\to$en ($n={cfg.n_samples['wmt16_de_en']}$); R-L: ROUGE-L F1 "
                  f"(\\%) on CNN/DailyMail ($n={cfg.n_samples['cnn_dailymail']}$)"),
        "speed": "tok/s: generation throughput on C4 (one NVIDIA RTX A4000, fp16, batch size 1)",
        "boot": f"95\\% bootstrap CIs over prompts ($B={N_BOOT}$)",
    }


def tab_main_compact(run: Run, n: pd.DataFrame, cap: dict) -> str:
    def dag(v, key):
        return r"\rlap{$^\dagger$}" if v in run.methods and significant(n, v, f"d_{key}") else ""

    body = [r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
            r" & \multicolumn{2}{c}{Detection} & \multicolumn{2}{c}{C4} & WMT & CNN \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-6}\cmidrule(lr){7-7}",
            r"Method & AUC$\uparrow$ & TPR$\uparrow$ & PPL$\downarrow$ & Div.$\uparrow$ & BLEU$\uparrow$ "
            r"& R-L$\uparrow$ \\", r"\midrule"]
    for v in _rows(run):
        r = n.loc[v]
        det = v in run.methods
        task = v != "human"
        if v == "human":
            body.append(r"\midrule")
        body.append(_row([_name_cell(v),
                          num(r.auroc, 3) if det else "--", num(100 * r.tpr1, 1) if det else "--",
                          num(r.ppl, 1) + dag(v, "ppl"), num(r.diversity, 1) + dag(v, "div"),
                          num(r.bleu, 1) + dag(v, "bleu") if task else "--",
                          num(r.rl, 1) + dag(v, "rl") if task else "--"]).rstrip("\n"))
        if v == "none":
            body.append(r"\midrule")
    body += [r"\bottomrule", r"\end{tabular}"]
    caption = (f"Main results for {cap['model']}. {cap['detect']}: AUC, area under the ROC curve; TPR, "
               f"true-positive rate at 1\\% FPR (\\%). "
               f"{cap['tasks']}. $^\\dagger$: the 95\\% paired bootstrap CI of the difference to "
               f"unwatermarked sampling excludes zero.")
    return _header(run, "Main results, single column") + _float("table", "\n".join(body) + "\n", caption,
                                                                  "tab:main-compact", r"\footnotesize")


def tab_main_wide(run: Run, n: pd.DataFrame, cap: dict) -> str:
    def delta(v, key, text):
        if v not in run.methods:
            return ""
        mark = r"$^\dagger$" if significant(n, v, f"d_{key}") else r"\phantom{$^\dagger$}"
        return rf" {{\scriptsize({text}){mark}}}"

    attacks = run.rob["attacks"] if run.rob else []
    k = len(attacks)
    det_cols = 2 + k
    head_det = (rf"\multicolumn{{2}}{{c}}{{Detection (C4)}} & \multicolumn{{{k}}}{{c}}{{Attacked}}" if k
                else r"\multicolumn{2}{c}{Detection (C4)}")
    rules = r"\cmidrule(lr){2-3}" + (rf"\cmidrule(lr){{4-{3 + k}}}" if k else "")
    rules += "".join(rf"\cmidrule(lr){{{c}-{c + 1}}}" for c in (det_cols + 2, det_cols + 4))
    rules += rf"\cmidrule(lr){{{det_cols + 6}-{det_cols + 6}}}"
    body = [rf"\setlength{{\tabcolsep}}{{{1.8 if k else 3.4}pt}}", rf"\begin{{tabular}}{{@{{}}l{'c' * (7 + k)}@{{}}}}",
            r"\toprule",
            rf" & {head_det} & \multicolumn{{2}}{{c}}{{Open-ended text (C4)}} & "
            r"\multicolumn{2}{c}{Downstream tasks} & Speed \\",
            rules,
            r"Method & AUROC$\uparrow$ & TPR$_{1\%}\uparrow$ & "
            + "".join(f"{ATTACK_SHORT.get(a, a)}$\\uparrow$ & " for a in attacks)
            + r"PPL$\downarrow$ & Div.$\uparrow$ & BLEU$\uparrow$ & R-L$\uparrow$ & tok/s$\uparrow$ \\",
            r"\midrule"]
    for v in _rows(run):
        r = n.loc[v]
        det, task = v in run.methods, v != "human"
        if v == "human":
            body.append(r"\midrule")
        body.append(_row([
            _name_cell(v),
            num(r.auroc, 3) if det else "--", num(100 * r.tpr1, 1) if det else "--",
            *(num(100 * run.rob["per_method"][a][v]["tpr1"], 1) if det else "--" for a in attacks),
            num(r.ppl, 1) + (delta(v, "ppl", num(r.d_ppl, 0, signed=True) + r"\%") if det else ""),
            num(r.diversity, 1) + (delta(v, "div", num(r.d_div, 1, signed=True)) if det else ""),
            (num(r.bleu, 1) + (delta(v, "bleu", num(r.d_bleu, 1, signed=True)) if det else "")) if task else "--",
            (num(r.rl, 1) + (delta(v, "rl", num(r.d_rl, 1, signed=True)) if det else "")) if task else "--",
            num(r.tps_c4, 1) if task else "--"]).rstrip("\n"))
        if v == "none":
            body.append(r"\midrule")
    body += [r"\bottomrule", r"\end{tabular}"]
    attacked = (f" Attacked: TPR (\\%) after {' and after '.join(attack_phrase(run, a) for a in attacks)}, at the "
                f"threshold set to 1\\% FPR on the clean human-written texts." if k else "")
    caption = (f"Detection, text quality and cost of five watermarks on {cap['model']}. {cap['detect']}; "
               f"TPR$_{{1\\%}}$: true-positive rate (\\%) at 1\\% FPR.{attacked} "
               f"{cap['tasks']}. In parentheses: difference to unwatermarked sampling on the same prompts "
               f"(PPL as relative change); $^\\dagger$: its 95\\% paired bootstrap CI excludes zero. "
               f"{cap['speed']}.")
    return _header(run, "Main results, full width, with differences") + _float("table*", "\n".join(body) + "\n",
                                                                               caption, "tab:main-wide",
                                                                               r"\footnotesize")


def tab_main_ci(run: Run, n: pd.DataFrame, cap: dict) -> str:
    body = [STACK.rstrip("\n"), r"\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
            r" & \multicolumn{2}{c}{Detection (C4)} & \multicolumn{2}{c}{Open-ended text (C4)} & "
            r"\multicolumn{1}{c}{WMT16} & \multicolumn{1}{c}{CNN/DM} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-6}\cmidrule(lr){7-7}",
            r"Method & AUROC$\uparrow$ & TPR@1\%FPR$\uparrow$ & PPL$\downarrow$ & Div.\ (\%)$\uparrow$ & BLEU$\uparrow$ "
            r"& ROUGE-L$\uparrow$ \\", r"\midrule"]
    for v in _rows(run):
        r = n.loc[v]
        det, task = v in run.methods, v != "human"

        def cell(x, lo, hi, nd, scale=1.0, show=True):
            if not show:
                return "--"
            return rf"\wmbstack{{{num(scale * x, nd)}}}{{{ci(scale * lo, scale * hi, nd)}}}"

        if v == "human":
            body.append(r"\midrule")
        body.append(_row([
            _name_cell(v),
            cell(r.auroc, r.get("auroc_lo"), r.get("auroc_hi"), 3, show=det),
            cell(r.tpr1, r.get("tpr1_lo"), r.get("tpr1_hi"), 1, 100, show=det),
            cell(r.ppl, r.ppl_lo, r.ppl_hi, 1), cell(r.diversity, r.diversity_lo, r.diversity_hi, 1),
            cell(r.get("bleu"), r.get("bleu_lo"), r.get("bleu_hi"), 1, show=task),
            cell(r.get("rl"), r.get("rl_lo"), r.get("rl_hi"), 1, show=task)]).rstrip("\n"))
        if v == "none":
            body.append(r"\midrule")
    body += [r"\bottomrule", r"\end{tabular}"]
    caption = (f"Main results with {cap['boot']} in brackets. {cap['detect']}. {cap['tasks']}.")
    return _header(run, "Main results, full width, with 95% CIs") + _float("table*", "\n".join(body) + "\n", caption,
                                                                           "tab:main-ci")


def tab_detection_wide(run: Run, n: pd.DataFrame, cap: dict) -> str:
    body = [STACK.rstrip("\n"), r"\setlength{\tabcolsep}{3.4pt}", r"\begin{tabular}{@{}llccccccccc@{}}", r"\toprule",
            r" & & & & & & \multicolumn{5}{c}{At the shipped threshold $\tau$} \\", r"\cmidrule(lr){7-11}",
            r"Method & Statistic & $n$ & AUROC & TPR@1\% & TPR@5\% & $\tau$ & TPR & FPR$_\mathrm{H}$ & "
            r"FPR$_\mathrm{U}$ & nominal \\", r"\midrule"]
    for m in run.methods:
        r = n.loc[m]
        body.append(_row([
            m, SCORE_TEX[m], str(int(r.n_det)),
            rf"\wmbstack{{{num(r.auroc, 3)}}}{{{ci(r.auroc_lo, r.auroc_hi, 3)}}}",
            rf"\wmbstack{{{num(100 * r.tpr1, 1)}}}{{{ci(100 * r.tpr1_lo, 100 * r.tpr1_hi, 1)}}}",
            num(100 * r.tpr5, 1), f"{r.thr:g}", num(100 * r.thr_tpr, 1), num(100 * r.thr_fpr_human, 1),
            num(100 * r.thr_fpr_unwm, 1), pct_small(r.nominal_fpr) if pd.notna(r.nominal_fpr) else "--"]).rstrip("\n"))
    body += [r"\bottomrule", r"\end{tabular}"]
    caption = (f"Detection performance. {cap['detect']}; {cap['boot']} in brackets. TPR and FPR in \\%; "
               f"FPR$_\\mathrm{{H}}$ and FPR$_\\mathrm{{U}}$: false-positive rate at the shipped threshold on human-written "
               f"and on unwatermarked continuations; nominal: FPR implied by $\\tau$ under the detector's null "
               f"distribution (none is defined for SynthID's and SIR's mean scores).")
    return _header(run, "Detection details, full width") + _float("table*", "\n".join(body) + "\n", caption,
                                                                  "tab:detection-wide")


def tab_detection_compact(run: Run, n: pd.DataFrame, cap: dict) -> str:
    body = [r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
            r" & & \multicolumn{2}{c}{TPR (\%)} & \multicolumn{2}{c}{FPR@$\tau$ (\%)} \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
            r"Method & AUROC & @1\% & @5\% & obs. & nom. \\", r"\midrule"]
    for m in run.methods:
        r = n.loc[m]
        body.append(_row([m, num(r.auroc, 3), num(100 * r.tpr1, 1), num(100 * r.tpr5, 1),
                          num(100 * r.thr_fpr_human, 1),
                          pct_small(r.nominal_fpr) if pd.notna(r.nominal_fpr) else "--"]).rstrip("\n"))
    body += [r"\bottomrule", r"\end{tabular}"]
    caption = (f"Detection on C4 ({cap['n']} per method; negatives: human-written continuations). Observed "
               f"(obs.) FPR at each method's shipped threshold $\\tau$ versus the nominal (nom.) FPR its null "
               f"distribution implies (none is defined for SynthID and SIR).")
    return _header(run, "Detection, single column") + _float("table", "\n".join(body) + "\n", caption,
                                                             "tab:detection-compact")


def tab_robustness(run: Run, n: pd.DataFrame, cap: dict) -> str:
    """Per attack: its strength on each method's texts, then AUROC and TPR@1% FPR with CIs."""
    dt = run.dt.set_index("variant")
    body = [STACK.rstrip("\n"), r"\setlength{\tabcolsep}{2.1pt}", r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
            r" & & \multicolumn{2}{c}{Strength} & \multicolumn{2}{c}{Detection} \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
            r"Method & $h$ & BLEU & Sim. & AUROC & TPR@1\% \\"]
    for a in run.rob["attacks"]:
        body += [r"\midrule", rf"\multicolumn{{6}}{{@{{}}l}}{{\textit{{{ATTACK_LONG.get(a, a)}}}}} \\"]
        for m in run.methods:
            r, d = run.rob["per_method"][a][m], dt.loc[m]
            body.append(_row([
                m, "sem." if context_width(m) is None else str(context_width(m)),
                num(d[f"attack_bleu_vs_source_{a}"], 1),
                num(d[f"attack_sim_embed_{a}"], 2),
                rf"\wmbstack{{{num(r['auroc'], 3)}}}{{{ci(r['auroc_lo'], r['auroc_hi'], 3)}}}",
                rf"\wmbstack{{{num(100 * r['tpr1'], 1)}}}{{{ci(100 * r['tpr1_lo'], 100 * r['tpr1_hi'], 1)}}}",
            ]).rstrip("\n"))
    body += [r"\bottomrule", r"\end{tabular}"]
    sim = run.cfg.attack_similarity_model.split("/")[-1] if getattr(run.cfg, "attack_similarity_model", None) else None
    caption = (f"Robustness on C4 ({cap['n']} per method). Attacks: "
               f"{'; '.join(attack_phrase(run, a) for a in run.rob['attacks'])}. $h$: preceding tokens the "
               f"watermark key depends on (sem.: a semantic embedding of the context). Strength, measured "
               f"against the clean watermarked text: BLEU, sentence BLEU"
               + (f"; Sim., cosine similarity of {sim} embeddings" if sim else "; Sim., embedding similarity")
               + ". Token-length ratio to the clean text before capping at its length: "
               + "; ".join(f"{ATTACK_LONG.get(a, a).lower()} {dt[f'attack_len_ratio_{a}'].min():.2f}--"
                           f"{dt[f'attack_len_ratio_{a}'].max():.2f}" for a in run.rob["attacks"])
               + f". Detection against the clean human-written texts; TPR (\\%) at the threshold set to 1\\% FPR "
               f"on them; {cap['boot']} below each value.")
    return _header(run, "Robustness, single column") + _float("table", "\n".join(body) + "\n", caption,
                                                              "tab:robustness", r"\footnotesize")


def tab_quality_deltas(run: Run, n: pd.DataFrame, cap: dict) -> str:
    body = [STACK.rstrip("\n"), r"\setlength{\tabcolsep}{2.1pt}", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
            r" & \multicolumn{2}{c}{C4} & WMT16 & CNN/DM \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-5}",
            r"Method & PPL (\%) & Div.\ (pts) & BLEU & ROUGE-L \\", r"\midrule"]
    for m in run.methods:
        cells = [m]
        for key in ("d_ppl", "d_div", "d_bleu", "d_rl"):
            x, lo, hi = n.loc[m, key], n.loc[m, key + "_lo"], n.loc[m, key + "_hi"]
            nd = 0 if key == "d_ppl" else 1  # relative PPL change in whole percent
            s = num(x, nd, signed=True)
            if significant(n, m, key):
                s = rf"\textbf{{{s}}}"
            cells.append(rf"\wmbstack{{{s}}}{{{ci(lo, hi, nd)}}}")
        body.append(_row(cells).rstrip("\n"))
    body += [r"\bottomrule", r"\end{tabular}"]
    caption = (f"Change in quality relative to unwatermarked sampling on the same prompts, with 95\\% paired "
               f"bootstrap CIs ($B={N_BOOT}$). PPL: relative change of the corpus perplexity; Div.\\ and ROUGE-L in "
               f"percentage points; BLEU in sacreBLEU points. Bold: CI excludes zero.")
    return _header(run, "Quality differences to unwatermarked, single column") + _float(
        "table", "\n".join(body) + "\n", caption, "tab:quality-deltas", r"\footnotesize")


def tab_quality_full(run: Run, n: pd.DataFrame, cap: dict) -> str:
    body = [STACK.rstrip("\n"), r"\setlength{\tabcolsep}{3.3pt}", r"\begin{tabular}{@{}lccccccccccc@{}}",
            r"\toprule",
            r" & \multicolumn{4}{c}{C4 completion} & \multicolumn{3}{c}{WMT16 de$\to$en} & "
            r"\multicolumn{4}{c}{CNN/DailyMail} \\",
            r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}\cmidrule(lr){9-12}",
            r"Method & PPL & PPL$_\mathrm{med}$ & Div. & Rep-2 & BLEU & sBLEU & words & R-1 & R-2 & R-L & words \\",
            r"\midrule"]
    for v in _rows(run):
        r = n.loc[v]
        task = v != "human"
        if v == "human":
            body.append(r"\midrule")
        body.append(_row([
            _name_cell(v),
            rf"\wmbstack{{{num(r.ppl, 1)}}}{{{ci(r.ppl_lo, r.ppl_hi, 1)}}}",
            rf"\wmbstack{{{num(r.ppl_median, 1)}}}{{{ci(r.ppl_median_lo, r.ppl_median_hi, 1)}}}",
            rf"\wmbstack{{{num(r.diversity, 1)}}}{{{ci(r.diversity_lo, r.diversity_hi, 1)}}}", num(r.rep2, 1),
            rf"\wmbstack{{{num(r.bleu, 1)}}}{{{ci(r.bleu_lo, r.bleu_hi, 1)}}}" if task else "--",
            num(r.get("sbleu"), 1) if task else "--", num(r.get("mt_words"), 1) if task else "--",
            num(r.get("r1"), 1) if task else "--", num(r.get("r2"), 1) if task else "--",
            rf"\wmbstack{{{num(r.rl, 1)}}}{{{ci(r.rl_lo, r.rl_hi, 1)}}}" if task else "--",
            num(r.get("sum_words"), 1) if task else "--"]).rstrip("\n"))
        if v == "none":
            body.append(r"\midrule")
    body += [r"\bottomrule", r"\end{tabular}"]
    caption = (f"Quality metrics per task with {cap['boot']}. C4: corpus and median per-sample perplexity under "
               f"{cap['oracle']}, SimCTG diversity and repeated-bigram rate (\\%). WMT16: corpus sacreBLEU, mean "
               f"sentence BLEU and answer length in words. CNN/DailyMail: ROUGE F1 (\\%) and summary length.")
    return _header(run, "Quality details, full width") + _float("table*", "\n".join(body) + "\n", caption,
                                                                "tab:quality-full")


def tab_efficiency(run: Run, n: pd.DataFrame, cap: dict) -> str:
    body = [r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
            r"Method & C4 & WMT16 & CNN/DM \\", r"\midrule"]
    for v in ["none", *run.methods]:
        r = n.loc[v]
        cells = [_name_cell(v)]
        for key in ("c4", "wmt", "cnn"):
            tps = num(r[f"tps_{key}"], 1)
            cells.append(tps if v == "none" else rf"{tps} {{\scriptsize({num(r[f'rel_tps_{key}'], 0)}\%)}}")
        body.append(_row(cells).rstrip("\n"))
        if v == "none":
            body.append(r"\midrule")
    body += [r"\bottomrule", r"\end{tabular}"]
    lengths = ", ".join(f"{label} $\\le${run.cfg.max_new_tokens[d]}"
                        for d, label in (("c4", "C4"), ("wmt16_de_en", "WMT16"), ("cnn_dailymail", "CNN/DM")))
    caption = ("Generation throughput in tokens per second (one NVIDIA RTX A4000, fp16, batch size 1); "
               f"in parentheses relative to unwatermarked sampling. Output lengths: {lengths} new tokens.")
    return _header(run, "Throughput, single column") + _float("table", "\n".join(body) + "\n", caption,
                                                              "tab:efficiency")


def tab_setup(run: Run, n: pd.DataFrame, cap: dict) -> str:
    cfg = run.cfg
    rows = [("Completion", "C4 (RealNewsLike)", "c4", "PPL, Div., detection"),
            ("Translation", r"WMT16 de$\to$en", "wmt16_de_en", "sacreBLEU"),
            ("Summarization", "CNN/DailyMail", "cnn_dailymail", "ROUGE-1/2/L")]
    stack = (r"\providecommand{\wmbstackl}[2]{\begin{tabular}[c]{@{}l@{}}#1\\[-2pt]{\scriptsize #2}\end{tabular}}")
    body = [stack, r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lrrl@{}}", r"\toprule",
            r"Task (data) & $n$ & Tok. & Metrics \\", r"\midrule"]
    for task, data, key, metric in rows:
        body.append(_row([rf"\wmbstackl{{{task}}}{{{data}}}", str(cfg.n_samples[key]), str(cfg.max_new_tokens[key]),
                          metric]).rstrip("\n"))
    body += [r"\bottomrule", r"\end{tabular}"]
    caption = (f"Evaluation tasks. $n$: prompts per method; Tok.: maximum new tokens. All methods share "
               f"{cap['model']}, the prompt token ids and multinomial sampling at temperature 1 without top-$k$ or "
               f"top-$p$ truncation; perplexity is measured with {cap['oracle']}. Prompts ({cap['model']} has no chat "
               f"template): the raw news prefix (completion); {cfg.translation_shots} German--English example pairs, then "
               f"\\texttt{{German:}}~source \\texttt{{English:}} (translation); the article cut to "
               f"{cfg.article_max_words} words, then \\texttt{{TL;DR:}} (summarization).")
    return _header(run, "Tasks and data, single column") + _float("table", "\n".join(body) + "\n", caption,
                                                                  "tab:setup", r"\footnotesize")


def tab_methods(run: Run, n: pd.DataFrame, cap: dict) -> str:
    cfgs = {m: json.loads((CONFIG_DIR / VARIANTS[m].config_file).read_text()) for m in run.methods}

    def hp(m: str) -> str:
        c = cfgs[m]
        if m == "KGW":
            return rf"$\gamma={c['gamma']}$, $\delta={c['delta']}$, {c['prefix_length']}-token context"
        if m == "MorphMark":
            k = c.get(f"k_{c['type']}")
            return (rf"$\gamma={c['gamma']}$, {c['type']} morphing ($k={k}$), $p_0={c['p_0']}$, "
                    rf"{c['prefix_length']}-token context")
        if m == "PF":
            return rf"{c['ngram']}-token context, {c['seeding']} seeding"
        if m == "SynthID":
            return rf"{len(c['keys'])} tournament layers, {c['ngram_len']}-gram context, {c['watermark_mode']}"
        if m == "SIR":
            return rf"$\delta={c['delta']}$, compositional-BERT context embedding"
        return "--"

    mech = {"KGW": "green-list logit bias", "MorphMark": "adaptive green-list shift",
            "PF": "keyed Permute-and-Flip", "SynthID": "tournament sampling", "SIR": "semantic logit bias"}
    body = [r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}llp{0.29\textwidth}lcc@{}}", r"\toprule",
            r"Method & Mechanism & Configuration & Statistic & $\tau$ & Nominal FPR \\", r"\midrule"]
    for m in run.methods:
        r = n.loc[m]
        body.append(_row([m, mech.get(m, VARIANTS[m].family), r"\raggedright " + hp(m), SCORE_TEX[m], f"{r.thr:g}",
                          (pct_small(r.nominal_fpr) + r"\%") if pd.notna(r.nominal_fpr) else "--"]).rstrip("\n"))
    body += [r"\bottomrule", r"\end{tabular}"]
    caption = ("Watermarking methods as configured (MarkLLM implementations with their default configurations). "
               "$\\tau$: shipped detection threshold; nominal FPR: false-positive rate implied by $\\tau$ under the "
               "detector's null distribution (none is defined for SynthID's and SIR's mean scores).")
    return _header(run, "Methods and hyperparameters, full width") + _float("table*", "\n".join(body) + "\n",
                                                                            caption, "tab:methods")


EXAMPLE_SCORE = {"KGW": "$z={:.1f}$", "MorphMark": "$z={:.1f}$", "PF": r"$-\log_{{10}}p={:.0f}$",
                 "SynthID": r"$\bar g={:.3f}$", "SIR": r"$\bar s={:.2f}$"}


def tab_examples(run: Run, n: pd.DataFrame, cap: dict, sample_id: str = "c4-39", words: int = 42) -> str:
    gens = run.frames["generations"]
    g = gens[(gens.dataset == "c4") & (gens.sample_id == sample_id)].set_index("variant")

    def clip(s: str) -> str:
        w = s.split()
        return tex_text(" ".join(w[:words])) + (r"\ldots{}" if len(w) > words else "")

    def row(label: str, text: str) -> str:  # \raggedright in the last column: end the row with \tabularnewline
        return rf"\raggedright {label} & \raggedright {text} \tabularnewline"

    budget = {}
    body = [r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.15}",
            r"\begin{tabular}{@{}p{0.13\textwidth}p{0.85\textwidth}@{}}", r"\toprule",
            row(r"\textit{Prompt}", rf"\textit{{{clip(g.loc['none', 'prompt'])}}}"), r"\midrule",
            row("Human", clip(g.loc["none", "human_text"])),
            row("Unwatermarked", clip(g.loc["none", "continuation"])), r"\midrule"]
    for m in run.methods:
        d = run.det[m]
        hit = d[(d.sample_id == sample_id) & (d.text_kind == "watermarked")]
        score = EXAMPLE_SCORE[m].format(float(hit.score.iloc[0])) if len(hit) else "--"
        budget[m] = int(hit.detect_tokens.iloc[0]) if len(hit) else None
        body.append(row(rf"{m}\newline{{\scriptsize {score}}}", clip(g.loc[m, "continuation"])))
    body += [r"\bottomrule", r"\end{tabular}"]
    t = sorted({b for b in budget.values() if b})
    flagged = all(bool(run.det[m][(run.det[m].sample_id == sample_id) & (run.det[m].text_kind == "watermarked")]
                       .score.gt(VARIANTS[m].fixed_threshold).all()) for m in run.methods)
    caption = (f"Continuations of one C4 prompt (first {words} words; sample \\texttt{{{sample_id}}}). Under each "
               f"method: the detection score of its text under its own detector at "
               f"{t[0] if len(t) == 1 else ', '.join(map(str, t))} tokens ($z$: $z$-score; $\\bar g$: mean $g$-value; "
               f"$\\bar s$: mean bias similarity)"
               + ("; every score exceeds the method's threshold $\\tau$ (Table~\\ref{tab:methods})."
                  if flagged else "."))
    return _header(run, "Qualitative example, full width") + _float("table*", "\n".join(body) + "\n", caption,
                                                                    "tab:examples", r"\footnotesize")


# ----------------------------------------------------------------------------- figures


def setup_matplotlib(font: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    if font == "serif":
        family = next((f for f in ("STIXGeneral", "DejaVu Serif") if f in names), "serif")
        math_rc = {"mathtext.fontset": "stix"}
    else:  # Arial ships as a single TrueType file and embeds cleanly (Type 42)
        family = next((f for f in ("Arial", "Helvetica", "DejaVu Sans") if f in names), "sans-serif")
        math_rc = {"mathtext.fontset": "stixsans"}  # STIX sans: a Greek tau that does not read as a T
    plt.rcParams.update({
        "font.family": family, **math_rc,
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "text.color": INK, "axes.labelcolor": INK,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlepad": 3,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2,
        "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "lines.linewidth": 1.25, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.handlelength": 1.4, "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.0, "legend.borderaxespad": 0.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
    })
    return plt


class FigureWriter:
    def __init__(self, plt, out: Path):
        self.plt, self.out = plt, out
        out.mkdir(parents=True, exist_ok=True)

    def save(self, fig, name: str) -> Path:
        fig.savefig(self.out / f"{name}.pdf", metadata={"CreationDate": None, "Creator": None, "Producer": None})
        fig.savefig(self.out / f"{name}.png", dpi=220)
        self.plt.close(fig)
        return self.out / f"{name}.pdf"


def _ring(m: str, ms: float = 5.0, filled: bool = True) -> dict:
    """Marker style: filled with a 0.8 pt surface ring, or hollow in the series colour."""
    c = COLOR[m]
    if filled:
        return dict(marker=MARKER[m], ms=ms, mfc=c, mec=SURFACE, mew=0.8, ls="none")
    return dict(marker=MARKER[m], ms=ms, mfc=SURFACE, mec=c, mew=1.0, ls="none")


def fig_score_distributions(fw: FigureWriter, run: Run, layout: str) -> Path:
    import matplotlib.ticker as mticker
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt, ms = fw.plt, run.methods
    if layout == "wide":
        fig, axes = plt.subplots(1, len(ms), figsize=(TEXT_W, 1.62), layout="constrained")
        axes, legend_ax = list(np.atleast_1d(axes)), None
    else:
        nrow = math.ceil((len(ms) + 1) / 2)
        fig, grid = plt.subplots(nrow, 2, figsize=(COL_W, 1.02 * nrow + 0.05), layout="constrained")
        axes = list(grid.ravel())
        legend_ax = axes[len(ms)]
        for ax in axes[len(ms):]:
            ax.axis("off")
    for ax, m in zip(axes, ms):
        s = {k: scores(run, m, k) for k in KINDS}
        allv = np.concatenate(list(s.values()))
        if m == "PF":  # -log10 p spans 0..150: symmetric log axis, linear below 1
            ax.set_xscale("symlog", linthresh=1.0, linscale=0.45)
            tr = ax.xaxis.get_transform()
            lo, hi = tr.transform(np.array([[0.0], [allv.max()]])).ravel()
            edges = tr.inverted().transform(np.linspace(lo, hi, 44).reshape(-1, 1)).ravel()
            ax.set_xticks([0, 1, 10, 100], labels=["0", "1", "10", "100"])
            ax.xaxis.set_minor_locator(mticker.NullLocator())
        else:
            edges = np.linspace(allv.min(), allv.max(), 44)
        ax.hist(s["human"], bins=edges, histtype="stepfilled", color=MUTED, alpha=0.28, lw=0)
        ax.hist(s["unwatermarked"], bins=edges, histtype="step", color=INK2, lw=0.8)
        ax.hist(s["watermarked"], bins=edges, histtype="stepfilled", color=COLOR[m], alpha=0.22, lw=0)
        ax.hist(s["watermarked"], bins=edges, histtype="step", color=COLOR[m], lw=1.0)
        thr = VARIANTS[m].fixed_threshold
        if thr is not None:
            ax.axvline(thr, color=INK, lw=0.7)
            ax.text(thr, 1.0, r" $\tau$", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=7,
                    color=INK)
        ax.set_title(m)
        ax.set_xlabel(SCORE_LABEL[m])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_ylim(bottom=0)
        ax.margins(x=0.02)
    handles = [tuple(Patch(facecolor=COLOR[m], alpha=0.55, lw=0) for m in ms),
               Patch(facecolor=MUTED, alpha=0.28, lw=0), Line2D([], [], color=INK2, lw=0.8),
               Line2D([], [], color=INK, lw=0.7)]
    labels = ["Watermarked", "Human", "Unwatermarked", r"Threshold $\tau$"]
    kw = dict(handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})
    if legend_ax is None:
        fig.legend(handles, labels, loc="outside upper center", ncols=4, **kw)
    else:
        legend_ax.legend(handles, labels, loc="center", ncols=1, handlelength=2.2, **kw)
    return fw.save(fig, f"fig_score_distributions_{layout}")


def fig_detection_margin(fw: FigureWriter, run: Run) -> Path:
    import matplotlib.transforms as mtransforms

    plt, ms = fw.plt, run.methods
    fig, ax = plt.subplots(figsize=(COL_W, 1.95), layout="constrained")
    rng = np.random.default_rng(run.cfg.seed)
    z = {m: standardized_scores(run, m) for m in ms}
    human = np.concatenate([z[m]["human"] for m in ms])
    lo, hi = np.percentile(human, [0.5, 99.5])
    ax.axvspan(lo, hi, color=MUTED, alpha=0.2, lw=0, zorder=0)
    ax.text(hi + 0.4, -0.72, "Human text (central 99%)", fontsize=6.5, color=INK2, ha="left", va="center")
    right = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(1.0, -0.72, "median", transform=right, fontsize=6.5, color=INK2, ha="right", va="center")
    for y, m in enumerate(ms):
        v = z[m]["watermarked"]
        ax.scatter(v, y + rng.uniform(-0.27, 0.27, len(v)), s=2.2, color=COLOR[m], alpha=0.45, lw=0, zorder=2)
        med = float(np.median(v))
        ax.plot([med, med], [y - 0.36, y + 0.36], color=INK, lw=1.3, solid_capstyle="butt", zorder=3)
        ax.text(1.0, y, f"{med:.1f}", transform=right, fontsize=7, color=INK2, ha="right", va="center")
    ax.set_yticks(range(len(ms)), labels=ms)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(ms) - 0.5, -1.05)
    xmax = max(float(np.max(z[m]["watermarked"])) for m in ms)
    ax.set_xlim(min(lo, -3) - 0.5, xmax * 1.13 + 2)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("Detection score in SDs of human-text scores")
    return fw.save(fig, "fig_detection_margin")


def _repel(y: dict[str, float], gap: float) -> dict[str, float]:
    """Label positions (log10 units) at least ``gap`` apart, as close to ``y`` as a top-down sweep allows."""
    order = sorted(y, key=y.get)
    pos = {k: y[k] for k in order}
    for lo, hi in zip(order, order[1:]):
        pos[hi] = max(pos[hi], pos[lo] + gap)
    shift = (sum(pos.values()) - sum(y.values())) / len(y)  # re-centre the block on the data
    return {k: v - shift for k, v in pos.items()}


def fig_robustness_margin(fw: FigureWriter, run: Run) -> Path:
    """Median detection margin d' per method, clean and after each attack (log axis), labelled by context width."""
    plt = fw.plt
    attacks = run.rob["attacks"]
    conds = ["Clean", *(ATTACK_LONG.get(a, a).replace(" paraphrase", "") for a in attacks)]
    # Order by context width: semantic first, then increasing h
    ms = sorted(run.methods, key=lambda m: -1 if context_width(m) is None else context_width(m))
    vals = {m: [run.rob["per_method"][attacks[0]][m]["margin_clean_median"],
                *(run.rob["per_method"][a][m]["margin_median"] for a in attacks)] for m in ms}
    fig, ax = plt.subplots(figsize=(COL_W, 2.15), layout="constrained")
    x = np.arange(len(conds))
    for m in ms:
        ax.plot(x, vals[m], color=COLOR[m], lw=1.3, zorder=2)
        ax.plot(x, vals[m], **_ring(m, ms=5), zorder=3)
    ax.set_yscale("log")
    lo, hi = min(min(v) for v in vals.values()), max(max(v) for v in vals.values())
    ax.set_ylim(lo / 1.5, hi * 1.5)
    gap = 0.115 * math.log10(hi * 1.5 / (lo / 1.5)) / 2.0  # about one 7 pt line on this axis height
    for side, i, ha, dx in (("left", 0, "right", -0.12), ("right", len(conds) - 1, "left", 0.12)):
        pos = _repel({m: math.log10(vals[m][i]) for m in ms}, gap)
        for m in ms:
            label = f"{m} ({context_label(m)})" if side == "right" else f"{vals[m][i]:.1f}"
            ax.text(i + dx, 10 ** pos[m], label, color=COLOR[m] if side == "right" else INK2, fontsize=7,
                    ha=ha, va="center")
    ax.set_xticks(x, labels=conds)
    ax.set_xlim(-0.75, len(conds) - 1 + 1.45)
    ax.spines["bottom"].set_bounds(0, len(conds) - 1)
    ax.grid(axis="y", which="major")
    ax.set_axisbelow(True)
    from matplotlib.ticker import FuncFormatter

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylabel(r"Median margin $d'$ (SDs of human text)")
    return fw.save(fig, "fig_robustness_margin")


def fig_roc(fw: FigureWriter, run: Run) -> Path:
    from sklearn.metrics import roc_curve

    plt, ms = fw.plt, run.methods
    xmin = 1e-3
    curves, at_zero = {}, {}
    for m in ms:
        pos, neg = scores(run, m, "watermarked"), scores(run, m, "human")
        fpr, tpr, _ = roc_curve(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg], drop_intermediate=False)
        at_zero[m] = float(tpr[fpr == 0].max())  # TPR reached without a single false positive
        curves[m] = (np.clip(fpr, xmin, 1)[1:], tpr[1:])  # drop roc_curve's (0, 0) start; FPR 0 -> left edge
    ymin = math.floor(100 * min(at_zero.values()) - 0.5) / 100
    fig, ax = plt.subplots(figsize=(COL_W, 2.15), layout="constrained")
    n_neg = min(len(scores(run, m, "human")) for m in ms)
    ax.axvspan(xmin, 1 / n_neg, color=MUTED, alpha=0.18, lw=0, zorder=0)
    ax.text(xmin * 1.12, ymin + 0.0008, "below\n1/n", fontsize=6.5, color=INK2, ha="left", va="bottom")
    ax.axvline(0.01, color=INK2, lw=0.6, zorder=1)
    ax.text(0.0112, ymin + 0.0008, "1% FPR", fontsize=6.5, color=INK2, ha="left", va="bottom")
    for m in ms:
        ax.plot(*curves[m], color=COLOR[m], lw=1.25, label=m, zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(xmin, 1)
    ax.set_ylim(ymin, 1.002)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1], labels=["0.1%", "1%", "10%", "100%"])
    step = 0.01 if 1 - ymin <= 0.06 else 0.02
    yt = np.round(np.arange(1.0, ymin - 1e-9, -step)[::-1], 4)
    ax.set_yticks(yt, labels=[f"{100 * t:.0f}%" for t in yt])
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_xlabel("False-positive rate on human text (log scale)")
    ax.set_ylabel("True-positive rate")
    ax.legend(loc="lower right", ncols=1)
    return fw.save(fig, "fig_roc_lowfpr")


FOREST_PANELS = (("d_ppl", "C4 perplexity", "Change (%)"), ("d_div", "C4 diversity", "\u0394 (points)"),
                 ("d_bleu", "WMT16 BLEU", "\u0394 BLEU"), ("d_rl", "CNN/DM ROUGE-L", "\u0394 (points)"))


def fig_quality_forest(fw: FigureWriter, run: Run, n: pd.DataFrame, layout: str) -> Path:
    from matplotlib.lines import Line2D

    plt, ms = fw.plt, run.methods
    if layout == "col":
        fig, axes = plt.subplots(2, 2, figsize=(COL_W, 2.75), layout="constrained", sharey=True)
    else:
        fig, axes = plt.subplots(1, 4, figsize=(TEXT_W, 1.5), layout="constrained", sharey=True)
    ys = np.arange(len(ms))
    for ax, (key, title, xlabel) in zip(axes.ravel(), FOREST_PANELS):
        ax.axvline(0, color=INK2, lw=0.6, zorder=1)
        for y, m in zip(ys, ms):
            est, lo, hi = n.loc[m, key], n.loc[m, key + "_lo"], n.loc[m, key + "_hi"]
            sig = significant(n, m, key)
            ax.plot([lo, hi], [y, y], color=COLOR[m], lw=1.25, alpha=1.0 if sig else 0.45, zorder=2)
            ax.plot([est], [y], zorder=3, **_ring(m, 5.5, filled=sig))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        lim = max(abs(n.loc[ms, key + "_lo"]).max(), abs(n.loc[ms, key + "_hi"]).max()) * 1.08
        ax.set_xlim(-lim, lim)  # symmetric around zero, so the sign of an effect reads at a glance
    axes.ravel()[0].set_yticks(ys, labels=ms)
    axes.ravel()[0].invert_yaxis()
    handles = [Line2D([], [], marker="o", ms=4.5, mfc=INK2, mec=SURFACE, mew=0.8, ls="none"),
               Line2D([], [], marker="o", ms=4.5, mfc=SURFACE, mec=INK2, mew=1.0, ls="none")]
    fig.legend(handles, ["95% CI excludes 0", "95% CI includes 0"], loc="outside upper center", ncols=2)
    return fw.save(fig, f"fig_quality_forest_{layout}")


_UR, _UL, _LR, _LL = ((4, 3), "left", "bottom"), ((-4, 3), "right", "bottom"), ((4, -3), "left", "top"), \
    ((-4, -3), "right", "top")
# Label placement per point: a diagonal quadrant, clear of the horizontal and vertical CI arms (tuned for this run)
_SCATTER_LABEL = {"human": _UR, "none": _UL, "KGW": _UL, "MorphMark": _LR, "PF": _LR, "SynthID": _UR, "SIR": _LR}


def fig_ppl_vs_diversity(fw: FigureWriter, run: Run, n: pd.DataFrame) -> Path:
    plt = fw.plt
    fig, ax = plt.subplots(figsize=(COL_W, 2.3), layout="constrained")
    for v in ("human", "none", *run.methods):
        r = n.loc[v]
        x, y = r.ppl, r["diversity"]
        ax.errorbar(x, y, xerr=[[x - r.ppl_lo], [r.ppl_hi - x]], yerr=[[y - r.diversity_lo], [r.diversity_hi - y]],
                    fmt="none", ecolor=COLOR[v], elinewidth=0.8, capsize=0, zorder=2)
        ax.plot([r.ppl], [r["diversity"]], zorder=3, **_ring(v, 6.5 if v == "human" else 5.0))
        (dx, dy), ha, va = _SCATTER_LABEL.get(v, ((6, 0), "left", "center"))
        ax.annotate(display(v), (r.ppl, r["diversity"]), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                    fontsize=7, color=INK2)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("Perplexity under OPT-2.7B (corpus level)")
    ax.set_ylabel("Diversity (%)")
    ax.margins(x=0.12, y=0.14)
    return fw.save(fig, "fig_ppl_vs_diversity")


def fig_ppl_distributions(fw: FigureWriter, run: Run) -> Path:
    from matplotlib.colors import to_rgba

    plt = fw.plt
    order = ["human", "none", *run.methods]
    data = [_quality_rows(run, "c4", v).ppl.astype(float).dropna().to_numpy() for v in order]
    fig, ax = plt.subplots(figsize=(COL_W, 2.2), layout="constrained")
    bp = ax.boxplot(data, orientation="horizontal", whis=(5, 95), widths=0.55, patch_artist=True,
                    medianprops=dict(color=INK, lw=1.2), flierprops=dict(marker="o", ms=1.6, mec="none", alpha=0.5))
    for i, v in enumerate(order):
        c = COLOR[v]
        bp["boxes"][i].set(facecolor=to_rgba(c, 0.16), edgecolor=c, lw=0.9)
        for part in ("whiskers", "caps"):
            for line in bp[part][2 * i: 2 * i + 2]:
                line.set(color=c, lw=0.9)
        bp["fliers"][i].set(markerfacecolor=c)
    ax.set_xscale("log")
    ticks = [t for t in (5, 10, 20, 50, 100, 200) if min(map(np.min, data)) <= t <= max(map(np.max, data))]
    ax.set_xticks(ticks, labels=[str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks(range(1, len(order) + 1), labels=[display(v) for v in order])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("Per-sample perplexity under OPT-2.7B (log scale)")
    return fw.save(fig, "fig_ppl_distributions")


_COST_LABEL = {"none": _UR, "KGW": ((0, 6), "center", "bottom"), "MorphMark": _LL, "PF": ((0, 6), "center", "bottom"),
               "SynthID": _LR, "SIR": ((0, 6), "center", "bottom")}


def fig_cost_tradeoff(fw: FigureWriter, run: Run, n: pd.DataFrame) -> Path:
    plt = fw.plt
    fig, ax = plt.subplots(figsize=(COL_W, 2.3), layout="constrained")
    ax.axvline(0, color=INK2, lw=0.6, zorder=1)
    ax.axhline(100, color=INK2, lw=0.6, zorder=1)
    ax.plot([0], [100], zorder=3, **_ring("none", 5.5))
    (dx, dy), ha, va = _COST_LABEL["none"]
    ax.annotate(display("none"), (0, 100), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=7,
                color=INK2)
    for m in run.methods:
        r = n.loc[m]
        ax.errorbar(r.d_ppl, r.rel_tps_c4, xerr=[[r.d_ppl - r.d_ppl_lo], [r.d_ppl_hi - r.d_ppl]], fmt="none",
                    ecolor=COLOR[m], elinewidth=0.9, capsize=0, zorder=2)
        ax.plot([r.d_ppl], [r.rel_tps_c4], zorder=3, **_ring(m, 5.5))
        (dx, dy), ha, va = _COST_LABEL.get(m, ((6, 0), "left", "center"))
        ax.annotate(m, (r.d_ppl, r.rel_tps_c4), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                    fontsize=7, color=INK2)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("Perplexity change vs. unwatermarked (%)")
    ax.set_ylabel("Throughput (% of unwatermarked)")
    ax.set_ylim(0, 112)
    ax.margins(x=0.1)
    return fw.save(fig, "fig_cost_tradeoff")


def fig_throughput(fw: FigureWriter, run: Run, n: pd.DataFrame) -> Path:
    plt, ms = fw.plt, run.methods
    fig, axes = plt.subplots(1, 3, figsize=(COL_W, 1.55), layout="constrained", sharey=True)
    ys = np.arange(len(ms))
    for ax, (key, title) in zip(axes, (("c4", "C4"), ("wmt", "WMT16"), ("cnn", "CNN/DM"))):
        ax.axvline(100, color=INK2, lw=0.6, zorder=1)
        for y, m in zip(ys, ms):
            x = n.loc[m, f"rel_tps_{key}"]
            ax.plot([0, x], [y, y], color=COLOR[m], lw=1.0, solid_capstyle="butt", zorder=2)
            ax.plot([x], [y], zorder=3, **_ring(m, 4.5))
        ax.set_title(title)
        ax.set_xlim(0, 118)
        ax.set_xticks([0, 50, 100])
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(ys, labels=ms)
    axes[0].invert_yaxis()
    fig.supxlabel("Throughput (% of unwatermarked)", fontsize=8, color=INK)
    return fw.save(fig, "fig_throughput")


def fig_calibration(fw: FigureWriter, run: Run) -> Path:
    from scipy.stats import binom

    plt = fw.plt
    ms = [m for m in run.methods if null_pvalues(run, m) is not None]
    fig, ax = plt.subplots(figsize=(COL_W, 2.3), layout="constrained")
    alphas = np.logspace(-3, np.log10(0.5), 240)
    n_min = min(len(null_pvalues(run, m)) for m in ms)
    band_lo = binom.ppf(0.025, n_min, alphas) / n_min
    band_hi = binom.ppf(0.975, n_min, alphas) / n_min
    ax.fill_between(alphas, np.maximum(band_lo, 1e-4), band_hi, color=MUTED, alpha=0.2, lw=0, zorder=0,
                    label="95% range if calibrated")
    ax.plot(alphas, alphas, color=INK2, lw=0.7, zorder=1, label="Nominal = observed")
    for m in ms:
        p = np.sort(null_pvalues(run, m))
        emp = np.searchsorted(p, alphas, side="right") / len(p)
        emp = np.where(emp > 0, emp, np.nan)
        ax.plot(alphas, emp, color=COLOR[m], lw=1.25, zorder=2, label=m)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 0.5)
    ax.set_ylim(1e-3, 0.7)
    ticks, labels = [1e-3, 1e-2, 1e-1, 0.5], ["0.1%", "1%", "10%", "50%"]
    ax.set_xticks(ticks, labels=labels)
    ax.set_yticks(ticks, labels=labels)
    ax.grid(True, which="major")
    ax.set_axisbelow(True)
    ax.set_xlabel("Nominal FPR \u03b1 (p-value threshold)")
    ax.set_ylabel("Observed FPR")
    ax.legend(loc="upper left", ncols=1)
    return fw.save(fig, "fig_calibration")


def figure_snippet(run: Run, name: str, env: str, caption: str, label: str) -> str:
    width = r"\columnwidth" if env == "figure" else r"\textwidth"
    return (f"% Generated by scripts/paper_assets.py from run '{run.cfg.run_name}'. Requires \\usepackage{{graphicx}}.\n"
            f"\\begin{{{env}}}[t]\n\\centering\n\\includegraphics[width={width}]{{figures/{name}.pdf}}\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{{env}}}\n")


# ----------------------------------------------------------------------------- gallery

GALLERY_PREAMBLE = r"""\documentclass[10pt,twocolumn]{article}
\usepackage[letterpaper,textwidth=6.9in,textheight=9.2in,columnsep=0.3in,centering]{geometry}
\usepackage[T1]{fontenc}
\usepackage{newtxtext,newtxmath}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage[font=small,labelfont=bf,skip=4pt]{caption}
\usepackage{xcolor}
\setlength{\parindent}{0pt}
% Gallery only: floats become boxes set where they are defined, at the width they would have in a paper.
\makeatletter
\renewenvironment{table}[1][]{\par\medskip\noindent\begin{minipage}{\columnwidth}\def\@captype{table}}{\end{minipage}\par\bigskip}
\renewenvironment{figure}[1][]{\par\medskip\noindent\begin{minipage}{\columnwidth}\def\@captype{figure}}{\end{minipage}\par\bigskip}
\renewenvironment{table*}[1][]{\noindent\begin{minipage}{\textwidth}\def\@captype{table}}{\end{minipage}\par\bigskip}
\renewenvironment{figure*}[1][]{\noindent\begin{minipage}{\textwidth}\def\@captype{figure}}{\end{minipage}\par\bigskip}
\makeatother
\newcommand{\option}[3]{\par\smallskip{\raggedright{\sffamily\bfseries\color{black!75}#1}\hfill{\sffamily\scriptsize\color{black!55}#2}\\[-1pt]{\sffamily\scriptsize\color{black!55}#3}\par}\smallskip}
"""


@dataclasses.dataclass
class Option:
    code: str
    path: str  # relative to the gallery, without extension for figures' .tex snippet
    wide: bool
    note: str


def write_gallery(out: Path, run: Run, options: list[Option]) -> Path:
    index = "\n".join(rf"\item[\textbf{{{o.code}}}] {tex_text(o.note)} {{\scriptsize\texttt{{{tex_text(o.path)}}}}}"
                      for o in options)
    parts = [GALLERY_PREAMBLE, r"\begin{document}",
             r"\twocolumn[{\section*{Paper assets: " + tex_text(run.cfg.run_name) + r" --- gallery of options}"
             r"{\small Every option below is set at its printed width in a two-column layout "
             r"(3.3\,in columns, 6.9\,in text width, Times). Single-column items sit in a column, wide items span "
             r"the page. Include a table with \texttt{\textbackslash input\{tables/<name>\}} and a figure with "
             r"\texttt{\textbackslash input\{figures/<name>\}} (both need \texttt{booktabs}/\texttt{graphicx}).}"
             r"\begin{description}\itemsep0pt\small " + index + r"\end{description}\bigskip}]"]
    for o in options:
        head = rf"\option{{{o.code}}}{{{'full width' if o.wide else 'single column'}}}{{{tex_text(o.note)} "
        head += rf"--- \texttt{{{tex_text(o.path)}.tex}}}}"
        if o.wide:
            parts.append(r"\twocolumn[{" + head + "\n" + rf"\input{{{o.path}}}" + "}]")
        else:
            parts.append(head + "\n" + rf"\input{{{o.path}}}")
    parts.append(r"\end{document}")
    path = out / "preview.tex"
    path.write_text("\n".join(parts) + "\n")
    return path


def compile_gallery(tex: Path) -> bool:
    exe = shutil.which("tectonic")
    if exe is None:
        log.warning("tectonic not found; compile %s with any LaTeX engine", tex)
        return False
    proc = subprocess.run([exe, "--keep-logs", tex.name], cwd=tex.parent, capture_output=True, text=True)
    if proc.returncode != 0:
        log.error("tectonic failed:\n%s", proc.stderr[-4000:])
        return False
    for line in (tex.with_suffix(".log")).read_text(errors="ignore").splitlines():
        if line.startswith(("Overfull", "Underfull \\hbox")):
            log.warning("preview.tex: %s", line)
    return True


# ----------------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, default=REPO / "run_results" / "qwen15b-20261020-142513", help="results directory of one run")
    ap.add_argument("--out", type=Path, default=REPO / "paper", help="output directory")
    ap.add_argument("--font", choices=("sans", "serif"), default="sans", help="figure font (sans: Arial)")
    ap.add_argument("--compile", action="store_true", help="build preview.pdf with tectonic")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("wmbench").setLevel(logging.WARNING)

    run = load(args.run.resolve())
    n = compute_numbers(run)
    out = args.out.resolve()
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "numbers.json").write_text(json.dumps(json.loads(n.to_json(orient="index")), indent=2))
    if run.cfg.attacks:
        run.rob = compute_robustness(run, *load_tokenizers(run))
        (out / "numbers_robustness.json").write_text(json.dumps(run.rob, indent=2, default=float))
        if not run.rob["attacks"]:
            run.rob = None  # attacks configured but never run
    cap = _captions(run, n)

    tables = [
        ("T1a", "tab_main_compact", tab_main_compact, False, "Main results, compact; daggers mark significant quality changes"),
        ("T1b", "tab_main_wide", tab_main_wide, True, "Main results with differences to unwatermarked inline"),
        ("T1c", "tab_main_ci", tab_main_ci, True, "Main results with 95% CIs under every value"),
        ("T2a", "tab_detection_compact", tab_detection_compact, False, "Detection: AUROC, TPR at 1%/5% FPR, threshold FPR"),
        ("T2b", "tab_detection_wide", tab_detection_wide, True, "Detection details with CIs and both negative sets"),
        ("T3", "tab_quality_deltas", tab_quality_deltas, False, "Quality change vs. unwatermarked with paired CIs"),
        ("T4", "tab_quality_full", tab_quality_full, True, "All quality metrics per task (appendix)"),
        ("T5", "tab_efficiency", tab_efficiency, False, "Throughput per task"),
        ("T6", "tab_setup", tab_setup, False, "Tasks, data and generation budgets"),
        ("T7", "tab_methods", tab_methods, True, "Methods, mechanisms and hyperparameters (appendix)"),
        ("T8", "tab_examples", tab_examples, True, "One prompt, every method's continuation (appendix)"),
    ]
    if run.rob:
        tables.append(("T9", "tab_robustness", tab_robustness, False,
                       "Robustness: attack strength, AUROC and TPR at 1% FPR per attack"))
    options: list[Option] = []
    for code, name, fn, wide, note in tables:
        (out / "tables" / f"{name}.tex").write_text(fn(run, n, cap))
        options.append(Option(code, f"tables/{name}", wide, note))

    plt = setup_matplotlib(args.font)
    fw = FigureWriter(plt, out / "figures")
    nd = cap["n"]
    pf_greedier = "PF" in run.methods and n.loc["PF", "d_ppl_hi"] < 0 and n.loc["PF", "d_div_hi"] < 0
    budget = run.cfg.max_new_tokens["c4"]
    figures = [
        ("F1a", lambda: fig_score_distributions(fw, run, "wide"), "figure*", "fig:score-distributions",
         "Score distributions per detector, one row", True,
         f"Detection scores on C4 continuations ({nd} per method): watermarked texts (colored), human-written "
         f"continuations of the same prompts (gray fill) and unwatermarked generations (gray line), all cut to a "
         f"common budget of at most {budget} tokens. The vertical line marks each method's shipped threshold "
         f"$\\tau$. PF's $-\\log_{{10}} p$ is shown on a symmetric log axis."),
        ("F1b", lambda: fig_score_distributions(fw, run, "col"), "figure", "fig:score-distributions-col",
         "Score distributions per detector, grid", False,
         f"Detection scores on C4 ({nd} per method) for watermarked, human-written and unwatermarked texts at "
         f"a common token budget; vertical line: shipped threshold $\\tau$. PF on a symmetric log axis."),
        ("F2", lambda: fig_detection_margin(fw, run), "figure", "fig:detection-margin",
         "Detection margin on one common scale", False,
         "Detection margin. Each dot is one watermarked C4 continuation, its score expressed in standard deviations "
         "of the same detector's scores on human-written continuations, $(s-\\mu_\\mathrm{H})/\\sigma_\\mathrm{H}$; "
         "the gray band holds the central 99\\% of human texts. Numbers: median. PF enters through the normal "
         "approximation $(S-k)/\\sqrt{k}$ of its Gamma test."),
        ("F3", lambda: fig_roc(fw, run), "figure", "fig:roc",
         "ROC with logarithmic FPR axis", False,
         f"ROC curves against human-written negatives ({nd}) with a logarithmic FPR axis and a zoomed TPR axis. "
         f"FPRs below $1/n$ (shaded) cannot be resolved; curves start at the TPR reached without any false positive."),
        ("F4a", lambda: fig_quality_forest(fw, run, n, "col"), "figure", "fig:quality-forest",
         "Quality change vs. unwatermarked (forest plot), 2x2", False,
         f"Effect of each watermark on text quality relative to unwatermarked sampling on the same prompts: estimate "
         f"and 95\\% paired bootstrap CI ($B={N_BOOT}$). Perplexity: relative change of the corpus perplexity under "
         f"OPT-2.7B; diversity and ROUGE-L in percentage points; BLEU in sacreBLEU points."),
        ("F4b", lambda: fig_quality_forest(fw, run, n, "wide"), "figure*", "fig:quality-forest-wide",
         "Quality change vs. unwatermarked (forest plot), one row", True,
         f"Effect of each watermark on text quality relative to unwatermarked sampling on the same prompts (estimate "
         f"and 95\\% paired bootstrap CI, $B={N_BOOT}$). Perplexity: relative change of the corpus perplexity under "
         f"OPT-2.7B; diversity and ROUGE-L in percentage points; BLEU in sacreBLEU points."),
        ("F5", lambda: fig_ppl_vs_diversity(fw, run, n), "figure", "fig:ppl-diversity",
         "Perplexity vs. diversity", False,
         "Perplexity under OPT-2.7B against SimCTG diversity of the C4 continuations (95\\% bootstrap CIs)."
         + (" PF lowers perplexity together with diversity, the signature of more deterministic decoding rather than "
            "better text." if pf_greedier else "")),
        ("F6", lambda: fig_ppl_distributions(fw, run), "figure", "fig:ppl-distributions",
         "Per-sample perplexity distributions", False,
         "Per-sample perplexity of C4 continuations under OPT-2.7B (log scale). Boxes: quartiles and median; "
         "whiskers: 5th to 95th percentile."),
        ("F7", lambda: fig_cost_tradeoff(fw, run, n), "figure", "fig:cost",
         "Cost map: perplexity change vs. throughput", False,
         "Cost of watermarking relative to unwatermarked sampling: change of the corpus perplexity (95\\% paired "
         "bootstrap CI) against generation throughput on C4. Detection is not shown because every method reaches "
         f"AUROC $\\geq {math.floor(1000 * n.loc[run.methods, 'auroc'].min()) / 1000:.3f}$."),  # floor: a bound, not a rounding
        ("F8", lambda: fig_throughput(fw, run, n), "figure", "fig:throughput",
         "Throughput per task", False,
         "Generation throughput relative to unwatermarked sampling per task (batch size 1, one NVIDIA RTX A4000, fp16)."),
        ("F9", lambda: fig_calibration(fw, run), "figure", "fig:calibration",
         "Calibration of the analytic null distributions", False,
         "Calibration of the detectors' null distributions on non-watermarked text (human-written and unwatermarked "
         "continuations pooled): fraction of texts with $p\\le\\alpha$ against the nominal level $\\alpha$. The band "
         "is the 95\\% range expected under perfect calibration. SynthID and SIR ship without a null distribution."),
    ]
    if run.rob:
        from scipy.stats import spearmanr

        last = run.rob["per_method"][run.rob["attacks"][-1]]
        width = [-1 if context_width(m) is None else context_width(m) for m in run.methods]
        follows_h = spearmanr(width, [last[m]["margin_retained"] for m in run.methods])[0] <= -0.8
        figures.append((
            "F10", lambda: fig_robustness_margin(fw, run), "figure", "fig:robustness-margin",
            "Detection margin clean vs. under attack", False,
            "Median detection margin $d'$ of the watermarked C4 texts, clean and after each attack, in standard "
            "deviations of the same detector's scores on clean human-written text (log axis; left: clean values). "
            "Labels give the context width $h$ the watermark key depends on."
            + (" The share of the margin that survives falls with $h$, independently of the clean margin."
               if follows_h else "")))
    for code, make, env, label, note, wide, caption in figures:
        pdf = make()
        (pdf.with_suffix(".tex")).write_text(figure_snippet(run, pdf.stem, env, caption, label))
        options.append(Option(code, f"figures/{pdf.stem}", wide, note))

    ordered = [o for o in options if not o.wide] + [o for o in options if o.wide]
    tex = write_gallery(out, run, ordered)
    log.info("wrote %d tables and %d figures to %s", len(tables), len(figures), out)
    if args.compile and compile_gallery(tex):
        log.info("gallery: %s", tex.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
