"""Unit tests for evaluation metrics.

Source of Truth: SPEC.md Section 6 & User Request PART C.
Tests:
- Hand-computed tiny examples for AUROC, AUPRC, FPR@95%TPR, TPR@1%FPR.
- Scikit-learn equivalence tests.
- Perfect detector yields AUROC == 1.0, AUPRC == 1.0, FPR@95%TPR == 0.0, TPR@1%FPR == 1.0.
- Random detector yields AUROC ~ 0.5 (within sampling tolerance).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from eval.metrics import (
    compute_all_metrics,
    compute_auprc,
    compute_auroc,
    compute_fpr_at_tpr,
    compute_tpr_at_fpr,
)


def test_hand_computed_tiny_example():
    """Verify metrics on a hand-calculated 4-sample test case.
    
    y_true  = [0,   0,   1,   1]
    y_score = [0.1, 0.4, 0.35, 0.8]
    
    Rankings by score (high to low):
    1. Score 0.8  -> True label 1 (TP=1, FP=0) -> TPR=0.5,  FPR=0.0, Precision=1/1=1.0
    2. Score 0.4  -> True label 0 (TP=1, FP=1) -> TPR=0.5,  FPR=0.5, Precision=1/2=0.5
    3. Score 0.35 -> True label 1 (TP=2, FP=1) -> TPR=1.0,  FPR=0.5, Precision=2/3=0.667
    4. Score 0.1  -> True label 0 (TP=2, FP=2) -> TPR=1.0,  FPR=1.0, Precision=2/4=0.5
    
    - Hand AUROC:
      Pairs (positive, negative):
      pos=0.8 vs neg=0.1 -> 1 win
      pos=0.8 vs neg=0.4 -> 1 win
      pos=0.35 vs neg=0.1 -> 1 win
      pos=0.35 vs neg=0.4 -> 0 win (0.35 < 0.4)
      Total positive wins = 3 out of 4 pairs -> AUROC = 3 / 4 = 0.75.
    
    - Hand FPR@95%TPR:
      At score 0.35, TPR reaches 1.0 (>= 0.95), where FP=1 out of 2 negatives -> FPR = 1/2 = 0.5.
      
    - Hand TPR@1%FPR:
      Target FPR <= 0.01. Only operating point with FPR <= 0.01 is threshold > 0.8, where FPR=0, TPR=0,
      or score 0.8 where FPR=0, TP=1/2 = 0.5.
    """
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])

    # 1. AUROC matches hand calculation (0.75) and sklearn
    auroc = compute_auroc(y_true, y_score)
    assert auroc == 0.75
    assert np.isclose(auroc, roc_auc_score(y_true, y_score))

    # 2. AUPRC matches sklearn average precision
    auprc = compute_auprc(y_true, y_score)
    expected_auprc = average_precision_score(y_true, y_score)
    assert np.isclose(auprc, expected_auprc)

    # 3. FPR@95%TPR matches hand calculation (0.5)
    fpr_95 = compute_fpr_at_tpr(y_true, y_score, target_tpr=0.95)
    assert fpr_95 == 0.5

    # 4. TPR@1%FPR matches hand calculation (0.5)
    tpr_1 = compute_tpr_at_fpr(y_true, y_score, target_fpr=0.01)
    assert tpr_1 == 0.5


def test_perfect_scorer():
    """Verify that a perfectly separating detector produces optimal theoretical values."""
    y_true = np.array([0] * 100 + [1] * 100)
    y_score = np.array([0.1] * 100 + [0.9] * 100)

    metrics = compute_all_metrics(y_true, y_score)

    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["fpr_at_95_tpr"] == 0.0
    assert metrics["tpr_at_1_fpr"] == 1.0


def test_random_scorer_auroc_near_half():
    """Verify that random scores yield AUROC approximately equal to 0.5."""
    rng = np.random.default_rng(12345)
    n = 10000
    y_true = np.array([0] * n + [1] * n)
    y_score = rng.uniform(0.0, 1.0, size=2 * n)

    auroc = compute_auroc(y_true, y_score)
    assert 0.48 <= auroc <= 0.52, f"Random AUROC {auroc} was not near 0.5"
