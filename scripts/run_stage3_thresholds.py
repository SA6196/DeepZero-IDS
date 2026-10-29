"""Stage 3: Label-Free Threshold Calibration and Evaluation.

DEV DATA, NOT FOR REPORTED RESULTS
Uses saved scores from Stages 1 and 2 (no retraining).
Thresholds fit strictly on VAL-BENIGN scores:
1. POT (q=0.01, u=98th percentile)
2. POT (q=0.001, u=98th percentile)
3. Empirical 99.0% percentile
4. Empirical 99.9% percentile
5. Mean + 3 sigma
6. ORACLE (Best F1 on test, reference only)
Outputs:
- results/final_p1_thresholds.csv
- results/final_p2_thresholds.csv
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from thresholds.pot import fit_pot_threshold

STATUS_TAG = "DEV DATA, NOT FOR REPORTED RESULTS"


def compute_oracle_best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float, float, float]:
    """Find oracle threshold that maximizes F1 score on test set (reference only)."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    denom = precisions + recalls
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = np.where(denom > 0, 2 * (precisions * recalls) / denom, 0.0)

    best_idx = int(np.argmax(f1_scores))
    best_f1 = float(f1_scores[best_idx])
    best_prec = float(precisions[best_idx])
    best_rec = float(recalls[best_idx])

    if best_idx < len(thresholds):
        best_thresh = float(thresholds[best_idx])
    else:
        best_thresh = float(thresholds[-1]) if len(thresholds) > 0 else 0.0

    return best_thresh, best_f1, best_prec, best_rec


def evaluate_threshold(
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    y_test: np.ndarray,
    method_name: str,
    target_fpr: Optional[float],
) -> Dict[str, Any]:
    """Compute threshold on val_scores and evaluate realized metrics on test_scores."""
    if method_name == "POT (q=0.01)":
        thresh = fit_pot_threshold(val_scores, q=0.01, percentile_u=98.0)
    elif method_name == "POT (q=0.001)":
        thresh = fit_pot_threshold(val_scores, q=0.001, percentile_u=98.0)
    elif method_name == "Empirical 99%":
        thresh = float(np.percentile(val_scores, 99.0))
    elif method_name == "Empirical 99.9%":
        thresh = float(np.percentile(val_scores, 99.9))
    elif method_name == "Mean + 3 sigma":
        thresh = float(np.mean(val_scores) + 3.0 * np.std(val_scores))
    elif method_name == "ORACLE, USES TEST LABELS, REFERENCE ONLY":
        thresh, _, _, _ = compute_oracle_best_f1_threshold(y_test, test_scores)
    else:
        raise ValueError(f"Unknown threshold method: {method_name}")

    pred = test_scores > thresh
    benign_mask = (y_test == 0)
    attack_mask = (y_test == 1)

    realized_fpr = float(np.mean(pred[benign_mask])) if np.any(benign_mask) else 0.0
    realized_tpr = float(np.mean(pred[attack_mask])) if np.any(attack_mask) else 0.0

    tp = int(np.sum(pred & attack_mask))
    fp = int(np.sum(pred & benign_mask))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(2 * precision * realized_tpr / (precision + realized_tpr)) if (precision + realized_tpr) > 0 else 0.0

    rel_error = None
    if target_fpr is not None and target_fpr > 0:
        rel_error = round(abs(realized_fpr - target_fpr) / target_fpr, 4)

    return {
        "method": method_name,
        "threshold": round(float(thresh), 6),
        "target_fpr": target_fpr,
        "realized_fpr": round(realized_fpr, 6),
        "relative_error_fpr": rel_error,
        "realized_tpr": round(realized_tpr, 6),
        "precision": round(precision, 6),
        "f1": round(f1, 6),
    }


def run_stage3():
    print("=" * 105)
    print(f"[{STATUS_TAG}]")
    print("STAGE 3: LABEL-FREE THRESHOLD EVALUATION (NO RETRAINING)")
    print("Methods: POT (q=0.01, q=0.001), Empirical (99%, 99.9%), Mean+3sigma, Oracle Best-F1")
    print("=" * 105)

    scores_dir = REPO_ROOT / "results" / "scores"
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    p1_csv_path = results_dir / "final_p1_thresholds.csv"
    p2_csv_path = results_dir / "final_p2_thresholds.csv"

    methods_info = [
        ("POT (q=0.01)", 0.01),
        ("POT (q=0.001)", 0.001),
        ("Empirical 99%", 0.01),
        ("Empirical 99.9%", 0.001),
        ("Mean + 3 sigma", 0.00135),
        ("ORACLE, USES TEST LABELS, REFERENCE ONLY", None),
    ]

    # --- P1 Threshold Evaluation ---
    p1_files = sorted(list(scores_dir.glob("final_p1_*.npz")))
    print(f"Found {len(p1_files)} Stage 1 score files.")

    p1_records: List[Dict[str, Any]] = []
    for f in p1_files:
        # Format: final_p1_{model}_{family}_s{seed}.npz
        stem = f.stem[len("final_p1_") :]
        parts = stem.split("_s")
        seed = int(parts[-1])
        model_and_fam = parts[0]

        # Parse model and family
        # Valid models: IsolationForest, OneClassSVM, LOF, PCA, MLP_AE, VAE, Trimmed_AE
        known_models = ["IsolationForest", "OneClassSVM", "LOF", "PCA", "MLP_AE", "VAE", "Trimmed_AE"]
        model_slug = None
        for km in known_models:
            if model_and_fam.startswith(km):
                model_slug = km
                fam_slug = model_and_fam[len(km) + 1 :]
                break

        if model_slug is None:
            continue

        model_name = model_slug.replace("_", " ")
        family_name = fam_slug.replace("_", "/")

        data = np.load(f)
        val_scores = data["val_scores"]
        test_scores = data["test_scores"]
        y_test = data["y_test"]

        for m_name, target_fpr in methods_info:
            res = evaluate_threshold(val_scores, test_scores, y_test, m_name, target_fpr)
            p1_records.append({
                "protocol": "P1",
                "family": family_name,
                "seed": seed,
                "model": model_name,
                **res,
            })

    if p1_records:
        pd.DataFrame(p1_records).to_csv(p1_csv_path, index=False)
        print(f"Saved {len(p1_records)} P1 threshold records to {p1_csv_path}")

    # --- P2 Threshold Evaluation ---
    p2_files = sorted(list(scores_dir.glob("final_p2_*.npz")))
    print(f"Found {len(p2_files)} Stage 2 score files.")

    p2_records: List[Dict[str, Any]] = []
    for f in p2_files:
        # Format: final_p2_{model}_s{seed}.npz
        stem = f.stem[len("final_p2_") :]
        parts = stem.split("_s")
        seed = int(parts[-1])
        model_slug = parts[0]
        model_name = model_slug.replace("_", " ")

        data = np.load(f)
        val_scores = data["val_scores"]
        test_scores = data["test_scores"]
        y_test = data["y_test"]

        for m_name, target_fpr in methods_info:
            res = evaluate_threshold(val_scores, test_scores, y_test, m_name, target_fpr)
            p2_records.append({
                "protocol": "P2",
                "seed": seed,
                "model": model_name,
                **res,
            })

    if p2_records:
        pd.DataFrame(p2_records).to_csv(p2_csv_path, index=False)
        print(f"Saved {len(p2_records)} P2 threshold records to {p2_csv_path}")

    print("\nStage 3 finished successfully.")


if __name__ == "__main__":
    run_stage3()
