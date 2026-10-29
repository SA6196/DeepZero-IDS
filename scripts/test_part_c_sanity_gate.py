"""Part C Sanity Gate: Re-run P4 Plain AE at c = 0 for BruteForce, PortScan, WebAttack (seed 0).

DEV DATA, NOT FOR REPORTED RESULTS
Rule:
Compare with refreshed P1 Plain AE.
If any differs by > 0.03 AUROC, STOP, report why, and do nothing else.
If all are within 0.03, continue to Part D.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

torch.set_num_threads(4)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from data.preprocess import TabularPreprocessor
from eval.metrics import compute_all_metrics
from models.autoencoders import MLPAutoencoder, train_autoencoder
from splits.protocols import generate_p4_contamination_splits


def main():
    seed = 0
    status_tag = "DEV DATA, NOT FOR REPORTED RESULTS"
    max_threads = 4

    print("=" * 95)
    print(f"[{status_tag}]")
    print("PART C: SANITY GATE (P4 Plain AE at c=0 vs Refreshed P1 Plain AE)")
    print(f"Seed: {seed} | Max Threads: {max_threads}")
    print("=" * 95)

    data_path = REPO_ROOT / "data" / "processed" / "cicids2017_cleaned_dev.parquet"
    print(f"Loading cleaned dataset from {data_path} ...")
    t0 = time.time()
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} rows in {time.time() - t0:.2f}s.\n")

    feature_cols = [
        c for c in df.columns
        if c not in ["Label", "Attack_Family", "Day", "Destination Port"]
    ]

    target_families = ["BruteForce", "PortScan", "WebAttack"]

    # Refreshed P1 reference values from Part B
    ref_p1_path = REPO_ROOT / "results" / f"p1_refreshed_pca_ae_seed{seed}.csv"
    p1_df = pd.read_csv(ref_p1_path)

    p1_references = {}
    for fam in target_families:
        p1_val = p1_df[(p1_df["family"] == fam) & (p1_df["model"] == "Plain AE")]["auroc"].values[0]
        p1_references[fam] = p1_val

    results = []
    any_divergence = False

    for family in target_families:
        print(f"\n--- Running P4 Plain AE at c=0 for Family: {family} ---")
        split = generate_p4_contamination_splits(
            df,
            seed=seed,
            lofo_family=family,
            contamination_rate=0.0,
            n_train_total=200000,
            include_test_family=False,
            val_benign_ratio=0.2,
            test_benign_ratio=0.2,
        )

        # Fit preprocessor on contaminated TRAIN only (here c=0, 200k clean rows)
        X_train_df = df.loc[split.train_indices, feature_cols]
        preprocessor = TabularPreprocessor(skew_threshold=3.0, clip_value=10.0)
        preprocessor.fit(X_train_df)

        X_train = preprocessor.transform(X_train_df)
        X_val = preprocessor.transform(df.loc[split.val_indices, feature_cols])

        benign_test_indices = df.index[
            (df["Attack_Family"] == "Benign") & (df.index.isin(split.test_indices))
        ].to_numpy()
        attack_test_indices = df.index[
            (df["Attack_Family"] == family) & (df.index.isin(split.test_indices))
        ].to_numpy()

        eval_indices = np.sort(np.concatenate([benign_test_indices, attack_test_indices]))
        X_test_df = df.loc[eval_indices, feature_cols]
        y_test = (df.loc[eval_indices, "Attack_Family"] == family).astype(int).to_numpy()
        X_test = preprocessor.transform(X_test_df)

        plain_ae = MLPAutoencoder(
            input_dim=X_train.shape[1],
            hidden_dims=(64, 32, 16),
            bottleneck_dim=16,
            seed=seed,
        )

        t_start = time.time()
        trained_ae, history = train_autoencoder(
            plain_ae,
            X_train=X_train,
            X_val_benign=X_val,
            seed=seed,
            max_epochs=30,
            batch_size=256,
            lr=0.001,
            patience=3,
            trim_frac=0.0,
            warmup_epochs=0,
            verbose=False,
        )
        scores_ae = trained_ae.compute_reconstruction_error(X_test)
        runtime = time.time() - t_start

        m_ae = compute_all_metrics(y_test, scores_ae)
        p4_auroc = m_ae["auroc"]
        p1_auroc = p1_references[family]
        diff = abs(p4_auroc - p1_auroc)

        status = "PASSED (<= 0.03)" if diff <= 0.03 else "DIVERGENCE (> 0.03)"
        if diff > 0.03:
            any_divergence = True

        print(
            f"  [DONE] Family: {family:<10} | P4 (c=0, 200k): {p4_auroc:.4f} | "
            f"Refreshed P1 (500k): {p1_auroc:.4f} | |Delta|: {diff:.4f} | Status: {status} | Time: {runtime:.2f}s"
        )

        results.append({
            "Family": family,
            "P4 (c=0, 200k)": round(p4_auroc, 4),
            "Refreshed P1 (500k)": round(p1_auroc, 4),
            "|Delta AUROC|": round(diff, 4),
            "FPR@95%TPR": round(m_ae["fpr_at_95_tpr"], 4),
            "Status": status,
        })

    print("\n" + "=" * 95)
    print(f"[{status_tag}]")
    print("PART C: SANITY GATE SUMMARY TABLE")
    print("=" * 95)
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))

    if any_divergence:
        print("\n[GATE BLOCKED] At least one family diverged by > 0.03 AUROC. Investigation required.")
        sys.exit(1)
    else:
        print("\n[GATE PASSED] All families within 0.03 AUROC of refreshed P1. Proceeding to Part D.")
        sys.exit(0)


if __name__ == "__main__":
    main()
