"""Stage 6: Ablation Studies (Reported Only, No Selection).

DEV DATA, NOT FOR REPORTED RESULTS
6a: AE Bottleneck Dimension Ablation:
- Bottleneck b in {8, 32} (layers: input -> 64 -> 32 -> b -> 32 -> 64 -> input).
- b = 16 reused from Stage 1 (results/final_p1_seeds012.csv).
- Protocol P1, families: DoS/DDoS, PortScan, WebAttack.
- Seeds: 0, 1, 2.
- Output: results/final_ablation_bottleneck.csv

6b: Trimmed AE Trim Fraction Ablation:
- Trim fraction in {0.10, 0.20}.
- trim = 0.05 reused from existing Setting B results (results/p4_redo_seeds012.csv).
- Setting B, families: PortScan and DoS/DDoS, c = 5%.
- Seeds: 0, 1, 2.
- Output: results/final_ablation_trim.csv
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(4)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from data.preprocess import TabularPreprocessor
from eval.metrics import compute_all_metrics
from models.autoencoders import MLPAutoencoder, TrimmedAutoencoder, train_autoencoder
from splits.protocols import generate_p1_lofo_splits, generate_p4_setting_b_splits

STATUS_TAG = "DEV DATA, NOT FOR REPORTED RESULTS"


def load_existing_results(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def save_single_result(record: Dict[str, Any], csv_path: Path):
    rec_df = pd.DataFrame([record])
    if not csv_path.exists():
        rec_df.to_csv(csv_path, index=False)
    else:
        rec_df.to_csv(csv_path, mode="a", header=False, index=False)


def run_stage6():
    print("=" * 105)
    print(f"[{STATUS_TAG}]")
    print("STAGE 6: ABLATIONS (6a: BOTTLENECK DIMENSION, 6b: TRIM FRACTION)")
    print("=" * 105)

    data_path = REPO_ROOT / "data" / "processed" / "cicids2017_cleaned_dev.parquet"
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_6a = results_dir / "final_ablation_bottleneck.csv"
    csv_6b = results_dir / "final_ablation_trim.csv"
    p1_csv = results_dir / "final_p1_seeds012.csv"
    redo_csv = results_dir / "p4_redo_seeds012.csv"

    print(f"Loading cleaned dataset from {data_path} ...")
    df = pd.read_parquet(data_path)
    feature_cols = [c for c in df.columns if c not in ["Label", "Attack_Family", "Day", "Destination Port"]]

    # ==========================================
    # 6A: AE BOTTLENECK ABLATION (b in {8, 16, 32})
    # ==========================================
    print("\n" + "=" * 90)
    print("STAGE 6a: AE BOTTLENECK ABLATION (b in {8, 32}, reusing b=16)")
    print("=" * 90)

    cached_6a = load_existing_results(csv_6a)
    families_6a = ["DoS/DDoS", "PortScan", "WebAttack"]
    seeds = [0, 1, 2]

    # First, import b=16 from Stage 1 if available
    if p1_csv.exists():
        p1_df = pd.read_csv(p1_csv)
        sub_16 = p1_df[
            (p1_df["model"] == "MLP AE")
            & (p1_df["family"].isin(families_6a))
        ]
        for _, r in sub_16.iterrows():
            if cached_6a.empty or not (
                (cached_6a["family"] == r["family"])
                & (cached_6a["seed"] == int(r["seed"]))
                & (cached_6a["bottleneck"] == 16)
            ).any():
                rec_16 = {
                    "family": r["family"],
                    "seed": int(r["seed"]),
                    "bottleneck": 16,
                    "auroc": float(r["auroc"]),
                    "fpr_at_95_tpr": float(r["fpr_at_95_tpr"]),
                    "tpr_at_1_fpr": float(r["tpr_at_1_fpr"]),
                    "train_time_seconds": float(r.get("train_time_seconds", 0.0)),
                }
                save_single_result(rec_16, csv_6a)
        cached_6a = load_existing_results(csv_6a)

    # Train for b in {8, 32}
    for seed in seeds:
        splits = generate_p1_lofo_splits(
            df,
            seed=seed,
            val_benign_ratio=0.2,
            test_benign_ratio=0.2,
            target_families=families_6a,
        )
        ref_split = splits[families_6a[0]]
        train_idx = ref_split.train_indices[: min(len(ref_split.train_indices), 500000)]
        val_idx = ref_split.val_indices

        X_train_df = df.loc[train_idx, feature_cols]
        prep = TabularPreprocessor(skew_threshold=3.0, clip_value=10.0)
        prep.fit(X_train_df)

        X_train = prep.transform(X_train_df)
        X_val = prep.transform(df.loc[val_idx, feature_cols])

        X_tests = {}
        y_tests = {}
        for fam in families_6a:
            sp = splits[fam]
            X_tests[fam] = prep.transform(df.loc[sp.test_indices, feature_cols])
            y_tests[fam] = (df.loc[sp.test_indices, "Attack_Family"] == fam).astype(int).to_numpy()

        for b in [8, 32]:
            all_cached = all(
                not cached_6a.empty
                and (
                    (cached_6a["family"] == fam)
                    & (cached_6a["seed"] == seed)
                    & (cached_6a["bottleneck"] == b)
                ).any()
                for fam in families_6a
            )
            if all_cached:
                print(f"  [6a CACHED] s={seed} | b={b}")
                continue

            print(f"\n--- Training AE with Bottleneck b={b} (seed={seed}) ---")
            t0 = time.time()
            ae = MLPAutoencoder(
                input_dim=X_train.shape[1],
                hidden_dims=(64, 32, b),
                bottleneck_dim=b,
                seed=seed,
            )
            trained_ae, _ = train_autoencoder(
                ae,
                X_train=X_train,
                X_val_benign=X_val,
                seed=seed,
                max_epochs=30,
                batch_size=1024,
                lr=0.001,
                patience=3,
                trim_frac=0.0,
                warmup_epochs=0,
                verbose=False,
            )
            train_t = round(time.time() - t0, 2)

            for fam in families_6a:
                t_scores = trained_ae.compute_reconstruction_error(X_tests[fam])
                m = compute_all_metrics(y_tests[fam], t_scores)
                rec = {
                    "family": fam,
                    "seed": seed,
                    "bottleneck": b,
                    "auroc": round(float(m["auroc"]), 6),
                    "fpr_at_95_tpr": round(float(m["fpr_at_95_tpr"]), 6),
                    "tpr_at_1_fpr": round(float(m["tpr_at_1_fpr"]), 6),
                    "train_time_seconds": train_t,
                }
                save_single_result(rec, csv_6a)
                cached_6a = load_existing_results(csv_6a)
                print(f"  [6a DONE] s={seed} | b={b:<2} | {fam:<11} | AUROC: {m['auroc']:.4f} | FPR@95: {m['fpr_at_95_tpr']:.4f}")

    # ==========================================
    # 6B: TRIM FRACTION ABLATION (trim in {0.10, 0.20}, reusing 0.05)
    # ==========================================
    print("\n" + "=" * 90)
    print("STAGE 6b: TRIMMED AE FRACTION ABLATION (trim in {0.10, 0.20}, reusing 0.05)")
    print("=" * 90)

    cached_6b = load_existing_results(csv_6b)
    families_6b = ["PortScan", "DoS/DDoS"]

    # First, import trim=0.05 from results/p4_redo_seeds012.csv if available
    if redo_csv.exists():
        redo_df = pd.read_csv(redo_csv)
        sub_05 = redo_df[
            (redo_df["setting"] == "Setting B")
            & (redo_df["model"] == "Trimmed AE")
            & (redo_df["c"] == 5)
            & (redo_df["family"].isin(families_6b))
        ]
        for _, r in sub_05.iterrows():
            if cached_6b.empty or not (
                (cached_6b["family"] == r["family"])
                & (cached_6b["seed"] == int(r["seed"]))
                & (cached_6b["trim_fraction"] == 0.05)
            ).any():
                rec_05 = {
                    "family": r["family"],
                    "seed": int(r["seed"]),
                    "trim_fraction": 0.05,
                    "c": 5,
                    "auroc": float(r["auroc"]),
                    "fpr_at_95_tpr": float(r["fpr_at_95_tpr"]),
                    "tpr_at_1_fpr": float(r["tpr_at_1_fpr"]),
                    "train_time_seconds": float(r.get("runtime", 0.0)),
                }
                save_single_result(rec_05, csv_6b)
        cached_6b = load_existing_results(csv_6b)

    # Train trim fractions {0.10, 0.20}
    for trim_frac in [0.10, 0.20]:
        for fam in families_6b:
            for seed in seeds:
                if not cached_6b.empty and (
                    (cached_6b["family"] == fam)
                    & (cached_6b["seed"] == seed)
                    & (cached_6b["trim_fraction"] == trim_frac)
                ).any():
                    print(f"  [6b CACHED] trim={trim_frac:.2f} | {fam:<9} | s={seed}")
                    continue

                split = generate_p4_setting_b_splits(
                    df,
                    seed=seed,
                    family=fam,
                    contamination_rate=0.05,
                    n_train_total=200000,
                    val_benign_ratio=0.2,
                    test_benign_ratio=0.2,
                )

                X_train_df = df.loc[split.train_indices, feature_cols]
                prep = TabularPreprocessor(skew_threshold=3.0, clip_value=10.0)
                prep.fit(X_train_df)

                X_train = prep.transform(X_train_df)
                X_val = prep.transform(df.loc[split.val_indices, feature_cols])
                X_test = prep.transform(df.loc[split.test_indices, feature_cols])
                y_test = (df.loc[split.test_indices, "Attack_Family"] == fam).astype(int).to_numpy()

                t0 = time.time()
                ae = TrimmedAutoencoder(
                    input_dim=X_train.shape[1],
                    hidden_dims=(64, 32, 16),
                    bottleneck_dim=16,
                    seed=seed,
                )
                trained_ae, _ = train_autoencoder(
                    ae,
                    X_train=X_train,
                    X_val_benign=X_val,
                    seed=seed,
                    max_epochs=30,
                    batch_size=256,
                    lr=0.001,
                    patience=3,
                    trim_frac=trim_frac,
                    warmup_epochs=2,
                    verbose=False,
                )
                train_t = round(time.time() - t0, 2)
                t_scores = trained_ae.compute_reconstruction_error(X_test)
                m = compute_all_metrics(y_test, t_scores)

                rec = {
                    "family": fam,
                    "seed": seed,
                    "trim_fraction": trim_frac,
                    "c": 5,
                    "auroc": round(float(m["auroc"]), 6),
                    "fpr_at_95_tpr": round(float(m["fpr_at_95_tpr"]), 6),
                    "tpr_at_1_fpr": round(float(m["tpr_at_1_fpr"]), 6),
                    "train_time_seconds": train_t,
                }
                save_single_result(rec, csv_6b)
                cached_6b = load_existing_results(csv_6b)
                print(
                    f"  [6b DONE] trim={trim_frac:.2f} | {fam:<9} | s={seed} | "
                    f"AUROC: {m['auroc']:.4f} | FPR@95: {m['fpr_at_95_tpr']:.4f} | Time: {train_t}s"
                )

    print("\nStage 6 finished successfully.")


if __name__ == "__main__":
    run_stage6()
