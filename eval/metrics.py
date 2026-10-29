"""Evaluation metrics for threshold-free and fixed-operating-point anomaly detection.

Source of Truth: SPEC.md Section 6 & User Request PART C.
Metrics implemented:
- AUROC (Area under the Receiver Operating Characteristic curve)
- AUPRC (Area under the Precision-Recall curve / Average Precision)
- FPR@95%TPR (False Positive Rate at True Positive Rate >= 0.95)
- TPR@1%FPR (True Positive Rate at False Positive Rate <= 0.01)

Convention:
- Binary labels: 0 = Benign (negative), 1 = Attack / Anomaly (positive).
- Anomaly scores: Higher values indicate higher anomaly likelihood.
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _ensure_numpy_arrays(
    y_true: Union[np.ndarray, pd.Series, list],
    y_score: Union[np.ndarray, pd.Series, list],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert inputs to 1D float/int numpy arrays."""
    y_true_arr = np.asarray(y_true, dtype=np.int64).ravel()
    y_score_arr = np.asarray(y_score, dtype=np.float64).ravel()

    if len(y_true_arr) != len(y_score_arr):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_true_arr)} != len(y_score)={len(y_score_arr)}"
        )
    if len(y_true_arr) == 0:
        raise ValueError("Inputs y_true and y_score must not be empty.")

    unique_labels = np.unique(y_true_arr)
    if len(unique_labels) < 2:
        # Edge case: single class in evaluation set
        pass

    return y_true_arr, y_score_arr


def compute_auroc(
    y_true: Union[np.ndarray, pd.Series, list],
    y_score: Union[np.ndarray, pd.Series, list],
) -> float:
    """Compute Area Under the Receiver Operating Characteristic (AUROC)."""
    y_t, y_s = _ensure_numpy_arrays(y_true, y_score)
    if len(np.unique(y_t)) < 2:
        return float("nan")
    return float(roc_auc_score(y_t, y_s))


def compute_auprc(
    y_true: Union[np.ndarray, pd.Series, list],
    y_score: Union[np.ndarray, pd.Series, list],
) -> float:
    """Compute Area Under the Precision-Recall Curve (Average Precision)."""
    y_t, y_s = _ensure_numpy_arrays(y_true, y_score)
    if len(np.unique(y_t)) < 2:
        return float("nan")
    return float(average_precision_score(y_t, y_s))


def compute_fpr_at_tpr(
    y_true: Union[np.ndarray, pd.Series, list],
    y_score: Union[np.ndarray, pd.Series, list],
    target_tpr: float = 0.95,
) -> float:
    """Compute the minimum False Positive Rate (FPR) at which True Positive Rate (TPR) >= target_tpr.
    
    A lower FPR@95%TPR indicates a more discriminative detector that produces fewer false alarms
    while capturing 95% of attacks.
    """
    y_t, y_s = _ensure_numpy_arrays(y_true, y_score)
    if len(np.unique(y_t)) < 2:
        return float("nan")

    fpr, tpr, _ = roc_curve(y_t, y_s)
    # Find points where TPR >= target_tpr
    eligible = fpr[tpr >= target_tpr]
    if len(eligible) == 0:
        return 1.0
    return float(np.min(eligible))


def compute_tpr_at_fpr(
    y_true: Union[np.ndarray, pd.Series, list],
    y_score: Union[np.ndarray, pd.Series, list],
    target_fpr: float = 0.01,
) -> float:
    """Compute the maximum True Positive Rate (TPR) achievable at False Positive Rate (FPR) <= target_fpr.
    
    A higher TPR@1%FPR indicates superior sensitivity under strict false alarm budgets.
    """
    y_t, y_s = _ensure_numpy_arrays(y_true, y_score)
    if len(np.unique(y_t)) < 2:
        return float("nan")

    fpr, tpr, _ = roc_curve(y_t, y_s)
    # Find points where FPR <= target_fpr
    eligible = tpr[fpr <= target_fpr]
    if len(eligible) == 0:
        return 0.0
    return float(np.max(eligible))


def compute_all_metrics(
    y_true: Union[np.ndarray, pd.Series, list],
    y_score: Union[np.ndarray, pd.Series, list],
) -> Dict[str, float]:
    """Compute complete benchmark metric dictionary."""
    return {
        "auroc": compute_auroc(y_true, y_score),
        "auprc": compute_auprc(y_true, y_score),
        "fpr_at_95_tpr": compute_fpr_at_tpr(y_true, y_score, target_tpr=0.95),
        "tpr_at_1_fpr": compute_tpr_at_fpr(y_true, y_score, target_fpr=0.01),
    }
