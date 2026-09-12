import math

import numpy as np
import pytest

from wmbench import metrics
from wmbench.models import extract_continuation


def test_diversity_repetitive_text_is_low():
    diverse = "the quick brown fox jumps over the lazy dog while the cat sleeps on a warm mat"
    repetitive = "the the the the the the the the the the the the"
    d, r = metrics.diversity_metrics(diverse), metrics.diversity_metrics(repetitive)
    assert d["diversity"] > r["diversity"]
    assert d["log_diversity_markllm"] > r["log_diversity_markllm"]
    assert math.isnan(metrics.log_diversity("too short"))


def test_diversity_hand_computed():
    # tokens: a b a b a  -> bigrams: ab, ba, ab, ba (2 unique / 4) rep_2 = 0.5
    # trigrams: aba, bab, aba (2/3) rep_3 = 1/3 ; 4-grams: abab, baba (2/2) rep_4 = 0
    d = metrics.diversity_metrics("a b a b a")
    assert d["rep_2"] == pytest.approx(0.5) and d["rep_3"] == pytest.approx(1 / 3) and d["rep_4"] == pytest.approx(0.0)
    assert d["diversity"] == pytest.approx(0.5 * (2 / 3) * 1.0)
    markllm = -math.log(1 - (1 - 0.5 / 100) * (1 - (1 / 3) / 100) * 1.0)
    assert d["log_diversity_markllm"] == pytest.approx(markllm)


def test_log_diversity_no_repetition_is_capped():
    text = " ".join(str(i) for i in range(50))  # no repeated n-grams
    assert metrics.log_diversity(text) == pytest.approx(20.0)
    assert metrics.diversity_metrics(text)["diversity"] == pytest.approx(1.0)


def test_corpus_perplexity_and_bootstrap():
    assert metrics.corpus_perplexity([math.log(2) * 10, math.log(2) * 5], [10, 5]) == pytest.approx(2.0)
    assert math.isnan(metrics.corpus_perplexity([], []))
    lo, hi = metrics.bootstrap_ci([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert lo <= 5.5 <= hi
    lo, hi = metrics.paired_bootstrap_ci([2, 3, 4, 5], [1, 2, 3, 4])
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)
    b = metrics.detection_bootstrap([5, 6, 7, 8], [1, 2, 3, 4], n_boot=50)
    assert b["auroc"][0] == pytest.approx(1.0)


def test_bleu_and_rouge_identity():
    assert metrics.sentence_bleu("the cat sat on the mat", "the cat sat on the mat") == pytest.approx(100.0)
    assert metrics.corpus_bleu(["a b c d"], ["a b c d"]) == pytest.approx(100.0)
    r = metrics.rouge_scores("the cat sat on the mat", "the cat sat on the mat")
    assert r["rouge1"] == pytest.approx(1.0) and r["rougeL"] == pytest.approx(1.0)
    assert math.isnan(metrics.corpus_bleu([], []))


def test_auroc_separable_and_random():
    assert metrics.auroc([5, 6, 7], [1, 2, 3]) == pytest.approx(1.0)
    assert metrics.auroc([1, 2, 3], [1, 2, 3]) == pytest.approx(0.5)
    assert math.isnan(metrics.auroc([], [1.0]))


def test_tpr_at_fpr_threshold_is_max_negative_for_small_n():
    neg = [0.0, 1.0, 2.0, 3.0]  # 4 negatives, 1% FPR -> zero false positives allowed
    pos = [2.5, 3.0, 3.5, 4.0]
    tpr, thr = metrics.tpr_at_fpr(pos, neg, 0.01)
    assert thr == 3.0
    assert tpr == pytest.approx(0.5)  # only 3.5 and 4.0 are strictly above 3.0


def test_tpr_at_fpr_allows_k_false_positives():
    neg = list(range(100))  # 100 negatives, 5% -> 5 may be above threshold
    pos = [96.5] * 10
    tpr, thr = metrics.tpr_at_fpr(pos, neg, 0.05)
    assert thr == 94  # 95..99 (five values) are above
    assert tpr == pytest.approx(1.0)
    assert np.mean(np.asarray(neg) > thr) <= 0.05


def test_best_f1_perfect_separation():
    f1, thr = metrics.best_f1([4, 5, 6], [1, 2, 3])
    assert f1 == pytest.approx(1.0)
    assert 3 < thr < 4


def test_fixed_threshold_metrics():
    m = metrics.fixed_threshold_metrics([True, True, False], [False, False, True])
    assert m["tpr"] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(1 / 3)
    assert m["acc"] == pytest.approx(4 / 6)


def test_detection_summary_ignores_nan_scores():
    s = metrics.detection_summary([5, 6, float("nan")], [1, 2], [True, True, None], [False, False])
    assert s.n_pos == 2 and s.n_neg == 2
    assert s.auroc == pytest.approx(1.0)
    assert s.fixed_threshold_tpr == pytest.approx(1.0)


def test_humaneval_runner_pass_and_fail():
    prompt = "def add(a, b):\n"
    good = "    return a + b\n"
    bad = "    return a - b\n"
    test = "def check(f):\n    assert f(1, 2) == 3\n"
    assert metrics.humaneval_passes(prompt, good, test, "add") is True
    assert metrics.humaneval_passes(prompt, bad, test, "add") is False
    assert metrics.humaneval_passes(prompt, "    import time; time.sleep(30)\n", test, "add", timeout=1) is False


def test_clean_code_completion():
    assert metrics.clean_code_completion("\n    return 1\n\n\ndef other():") == "    return 1"


def test_extract_continuation():
    assert extract_continuation("Hello world, this is new", "Hello world,") == " this is new"
    assert extract_continuation("Hello world, this is new", " Hello world,") == " this is new"
    # tokenization changed whitespace: fall back to word-count truncation
    assert extract_continuation("Hello  world, this is new", "Hello world,") == "this is new"
