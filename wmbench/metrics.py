"""Quality, detection and code-correctness metrics.

Implemented natively (sacrebleu, rouge_score, scikit-learn, torch) rather than via
MarkLLM's evaluation pipelines so that per-sample records, timing and resumability
are available and the OpenAI/translate import chain of MarkLLM's pipelines is avoided.
Definitions follow the MarkLLM / KGW papers and are documented per function.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

import numpy as np
import torch

# --------------------------------------------------------------------------- quality


@torch.no_grad()
def continuation_nll(model, tokenizer, prompt: str, continuation: str, device: str,
                     max_length: int = 2048) -> tuple[float, int]:
    """Summed negative log-likelihood of ``continuation`` given ``prompt`` and the token count.

    Prompt and continuation are tokenized separately and concatenated, so the loss
    is taken exactly over the continuation tokens (the KGW paper convention; MarkLLM's
    ``PPLCalculator`` instead scores the continuation without any context). The prompt
    is left-truncated if the sequence exceeds ``max_length``. Returns ``(nan, 0)`` for an
    empty continuation.
    """
    p_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    c_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    if len(c_ids) == 0:
        return float("nan"), 0
    ids = (p_ids + c_ids)[-max_length:]
    n_ctx = max(len(ids) - len(c_ids), 1)  # at least one context token to predict from
    input_ids = torch.tensor([ids], device=device)
    logits = model(input_ids).logits[0].float()
    # position t predicts token t+1; score predictions for continuation tokens only
    pred = logits[n_ctx - 1:-1]
    target = input_ids[0, n_ctx:]
    nll = torch.nn.functional.cross_entropy(pred, target, reduction="sum")
    return float(nll), int(target.numel())


def conditional_perplexity(model, tokenizer, prompt: str, continuation: str, device: str,
                           max_length: int = 2048) -> tuple[float, int]:
    """``exp(mean NLL)`` of the continuation given the prompt; see :func:`continuation_nll`."""
    nll, n = continuation_nll(model, tokenizer, prompt, continuation, device, max_length)
    return (math.exp(nll / n) if n else float("nan")), n


def corpus_perplexity(nll_sums, token_counts) -> float:
    """Token-weighted PPL ``exp(sum NLL / sum tokens)`` over many samples (robust to outliers)."""
    nll = float(np.nansum([x for x in nll_sums if x is not None]))
    n = int(sum(int(t) for t, x in zip(token_counts, nll_sums) if x is not None and math.isfinite(x)))
    return math.exp(nll / n) if n else float("nan")


def _ngram_repetition(tokens: list[str], n: int) -> float | None:
    total = len(tokens) - n + 1
    if total <= 0:
        return None
    unique = len({tuple(tokens[i:i + n]) for i in range(total)})
    return 1.0 - unique / total


def diversity_metrics(text: str) -> dict[str, float]:
    """Repetition and diversity of one text on whitespace tokens.

    * ``rep_n`` (n = 2, 3, 4): fraction of repeated n-grams, ``1 - unique/total``.
    * ``diversity``: ``prod_n (1 - rep_n)`` in [0, 1] (Su et al., 2022, SimCTG). Higher = more diverse.
    * ``log_diversity_markllm``: MarkLLM's ``LogDiversityAnalyzer`` formula, reproduced exactly:
      ``-log(1 - prod_n (1 - rep_n / 100))`` clipped at 20. Its ``/100`` treats the fractional
      ``rep_n`` as a percentage, so the value is roughly ``-log(sum rep_n) + log(100)``; it is
      reported only for comparability with MarkLLM outputs. Texts shorter than a 4-gram give ``nan``.
    """
    tokens = text.split()
    reps = [_ngram_repetition(tokens, n) for n in (2, 3, 4)]
    if any(r is None for r in reps):
        return {"rep_2": float("nan"), "rep_3": float("nan"), "rep_4": float("nan"),
                "diversity": float("nan"), "log_diversity_markllm": float("nan")}
    diversity = 1.0
    markllm = 1.0
    for r in reps:
        diversity *= 1.0 - r
        markllm *= 1.0 - r / 100
    return {
        "rep_2": reps[0], "rep_3": reps[1], "rep_4": reps[2],
        "diversity": diversity,
        "log_diversity_markllm": -math.log(max(1.0 - markllm, math.exp(-20))),
    }


def log_diversity(text: str) -> float:
    """MarkLLM-compatible log diversity (see :func:`diversity_metrics`)."""
    return diversity_metrics(text)["log_diversity_markllm"]


def sentence_bleu(hypothesis: str, reference: str) -> float:
    """sacreBLEU sentence BLEU (0-100), the per-sample score MarkLLM's ``BLEUCalculator`` averages."""
    import sacrebleu

    return float(sacrebleu.sentence_bleu(hypothesis, [reference]).score)


def corpus_bleu(hypotheses: list[str], references: list[str]) -> float:
    """sacreBLEU corpus BLEU (0-100) over all samples of one variant; the headline MT number."""
    import sacrebleu

    if not hypotheses:
        return float("nan")
    return float(sacrebleu.corpus_bleu(hypotheses, [references]).score)


_rouge_scorer = None


def rouge_scores(hypothesis: str, reference: str) -> dict[str, float]:
    """ROUGE-1/2/L F-measure with stemming (same settings as MarkLLM's ROUGE calculators)."""
    global _rouge_scorer
    if _rouge_scorer is None:
        from rouge_score import rouge_scorer

        _rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    s = _rouge_scorer.score(reference, hypothesis)
    return {"rouge1": s["rouge1"].fmeasure, "rouge2": s["rouge2"].fmeasure, "rougeL": s["rougeL"].fmeasure}


# --------------------------------------------------------------------------- HumanEval

_HUMANEVAL_RUNNER = """
import sys, io
src = sys.stdin.read()
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()
g = {}
exec(src, g)
"""


def humaneval_passes(prompt: str, completion: str, test: str, entry_point: str, timeout: float = 10.0) -> bool:
    """Run one HumanEval task in a fresh subprocess with a timeout; True iff ``check`` passes.

    Model-written code is executed. MarkLLM's ``PassOrNotJudger`` uses ``exec`` in-process;
    here it is isolated in a subprocess with the working directory set to a temp dir.
    """
    program = f"{prompt}\n{completion}\n{test}\n\ncheck({entry_point})\n"
    with tempfile.TemporaryDirectory() as tmp:
        try:
            proc = subprocess.run([sys.executable, "-c", _HUMANEVAL_RUNNER], input=program, text=True,
                                  capture_output=True, timeout=timeout, cwd=tmp,
                                  env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"})
        except subprocess.TimeoutExpired:
            return False
    return proc.returncode == 0


def clean_code_completion(text: str) -> str:
    """Keep the first function body (MarkLLM's ``CodeGenerationTextEditor`` rule)."""
    text = text.lstrip("\n")
    return text.split("\n\n")[0]


# --------------------------------------------------------------------------- detection


@dataclass
class DetectionSummary:
    n_pos: int
    n_neg: int
    auroc: float
    tpr_at_1pct_fpr: float
    tpr_at_5pct_fpr: float
    best_f1: float
    best_f1_threshold: float
    fixed_threshold_tpr: float | None = None
    fixed_threshold_fpr: float | None = None
    fixed_threshold_f1: float | None = None
    fixed_threshold_acc: float | None = None
    pos_mean: float = float("nan")
    neg_mean: float = float("nan")
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "extra"}
        d.update(self.extra)
        return d


def _clean(scores) -> np.ndarray:
    a = np.asarray([float(s) for s in scores], dtype=float)
    return a[np.isfinite(a)]


def auroc(pos_scores, neg_scores) -> float:
    from sklearn.metrics import roc_auc_score

    pos, neg = _clean(pos_scores), _clean(neg_scores)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return float(roc_auc_score(y, np.concatenate([pos, neg])))


def tpr_at_fpr(pos_scores, neg_scores, target_fpr: float) -> tuple[float, float]:
    """Empirical TPR when the threshold is set so that at most ``target_fpr`` of negatives are flagged.

    With ``n_neg`` negatives at most ``floor(target_fpr * n_neg)`` may score above the
    threshold; with fewer than ``1/target_fpr`` negatives this means *no* negative may
    exceed it, i.e. the threshold is the maximum negative score. Returns ``(tpr, threshold)``.
    Scores strictly above the threshold count as detected.
    """
    pos, neg = _clean(pos_scores), _clean(neg_scores)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    k = int(math.floor(target_fpr * len(neg)))
    neg_desc = np.sort(neg)[::-1]
    threshold = float(neg_desc[k]) if k < len(neg_desc) else float(neg_desc[-1]) - 1e-9
    return float(np.mean(pos > threshold)), threshold


def best_f1(pos_scores, neg_scores) -> tuple[float, float]:
    """Max F1 over all midpoint thresholds (MarkLLM's ``rule='best'``)."""
    pos, neg = _clean(pos_scores), _clean(neg_scores)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(scores)
    scores, labels = scores[order], labels[order]
    best, best_thr = -1.0, float("nan")
    candidates = [(scores[i] + scores[i + 1]) / 2 for i in range(len(scores) - 1)] or [scores[0]]
    for thr in candidates:
        pred = scores >= thr
        tp = float(np.sum(pred & (labels == 1)))
        fp = float(np.sum(pred & (labels == 0)))
        fn = float(np.sum(~pred & (labels == 1)))
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best:
            best, best_thr = f1, float(thr)
    return best, best_thr


def fixed_threshold_metrics(pos_flags, neg_flags) -> dict[str, float]:
    """TPR/FPR/F1/ACC from each algorithm's own ``is_watermarked`` decision."""
    pos = np.asarray([bool(x) for x in pos_flags if x is not None])
    neg = np.asarray([bool(x) for x in neg_flags if x is not None])
    if len(pos) == 0 or len(neg) == 0:
        return {"tpr": float("nan"), "fpr": float("nan"), "f1": float("nan"), "acc": float("nan")}
    tp, fn = int(pos.sum()), int((~pos).sum())
    fp, tn = int(neg.sum()), int((~neg).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {"tpr": tp / len(pos), "fpr": fp / len(neg), "f1": f1, "acc": (tp + tn) / (len(pos) + len(neg))}


def detection_summary(pos_scores, neg_scores, pos_flags=None, neg_flags=None) -> DetectionSummary:
    pos, neg = _clean(pos_scores), _clean(neg_scores)
    tpr1, _ = tpr_at_fpr(pos, neg, 0.01)
    tpr5, _ = tpr_at_fpr(pos, neg, 0.05)
    f1, thr = best_f1(pos, neg)
    s = DetectionSummary(
        n_pos=len(pos), n_neg=len(neg), auroc=auroc(pos, neg),
        tpr_at_1pct_fpr=tpr1, tpr_at_5pct_fpr=tpr5, best_f1=f1, best_f1_threshold=thr,
        pos_mean=float(np.mean(pos)) if len(pos) else float("nan"),
        neg_mean=float(np.mean(neg)) if len(neg) else float("nan"),
    )
    if pos_flags is not None and neg_flags is not None:
        fixed = fixed_threshold_metrics(pos_flags, neg_flags)
        s.fixed_threshold_tpr, s.fixed_threshold_fpr = fixed["tpr"], fixed["fpr"]
        s.fixed_threshold_f1, s.fixed_threshold_acc = fixed["f1"], fixed["acc"]
    return s


# --------------------------------------------------------------------------- bootstrap


def bootstrap_ci(values, stat=np.mean, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of ``stat`` over ``values`` (NaNs dropped)."""
    v = _clean(values)
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    stats = np.asarray([stat(v[i]) for i in idx])
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def paired_bootstrap_ci(a, b, stat=np.mean, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """CI of ``stat(a) - stat(b)`` resampling matched pairs (same sample ids in ``a`` and ``b``)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    deltas = np.asarray([stat(a[i]) - stat(b[i]) for i in idx])
    return float(np.quantile(deltas, alpha / 2)), float(np.quantile(deltas, 1 - alpha / 2))


def detection_bootstrap(pos_scores, neg_scores, target_fpr: float = 0.01, n_boot: int = 500, seed: int = 0
                        ) -> dict[str, tuple[float, float]]:
    """Bootstrap CIs for AUROC and TPR@target FPR, resampling positives and negatives separately."""
    pos, neg = _clean(pos_scores), _clean(neg_scores)
    if len(pos) < 2 or len(neg) < 2:
        return {"auroc": (float("nan"), float("nan")), "tpr_at_fpr": (float("nan"), float("nan"))}
    rng = np.random.default_rng(seed)
    aucs, tprs = [], []
    for _ in range(n_boot):
        p = pos[rng.integers(0, len(pos), len(pos))]
        n = neg[rng.integers(0, len(neg), len(neg))]
        aucs.append(auroc(p, n))
        tprs.append(tpr_at_fpr(p, n, target_fpr)[0])
    return {"auroc": (float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975))),
            "tpr_at_fpr": (float(np.quantile(tprs, 0.025)), float(np.quantile(tprs, 0.975)))}
