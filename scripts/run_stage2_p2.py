"""Stage 2: Final Protocol P2 Grid (Temporal Split).

DEV DATA, NOT FOR REPORTED RESULTS
Train on Monday benign (with train caps), validate on Monday held-out benign,
Test on Tuesday-Friday (benign + all attack families).
Models: IsolationForest, OneClassSVM, LOF, PCA, MLP AE, VAE, Trimmed AE (trim 5%).
Seeds: 0, 1, 2.
Outputs:
- results/final_p2_seeds012.csv
- results/scores/final_p2_{model}_s{seed}.npz
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

from baselines.classical import (
    IsolationForestBaseline,
    LocalOutlierFactorBaseline,
    OneClassSVMBaseline,
    PCAReconstructionBaseline,
)
from data.preprocess import TabularPreprocessor
from eval.metrics import compute_all_metrics, compute_auroc
from models.autoencoders import (
    MLPAutoencoder,
    TrimmedAutoencoder,
    VariationalAutoencoder,
    train_autoencoder,
)
from splits.protocols import generate_p2_temporal_splits

STATUS_TAG = "DEV DATA, NOT FOR REPORTED RESULTS"


def load_existing_results(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def is_result_cached(df: pd.DataFrame, seed: int, model: str) -> bool:
    if df.empty:
        return False
    required_cols = {"seed", "model"}
    if not required_cols.issubset(df.columns):
        return False
    match = df[
        (df["seed"] == seed)
        & (df["model"] == model)
    ]
    return not match.empty


def save_single_result(record: Dict[str, Any], csv_path: Path):
    rec_df = pd.DataFrame([record])
    if not csv_path.exists():
        rec_df.to_csv(csv_path, index=False)
    else:
        rec_df.to_csv(csv_path, mode="a", header=False, index=False)


def run_stage2():
    print("=" * 105)
    print(f"[{STATUS_TAG}]")
    print("STAGE 2: FINAL PROTOCOL P2 GRID (TEMPORAL: MONDAY TRAIN -> TUE-FRI TEST)")
    print("Models: IsolationForest, OneClassSVM, LOF, PCA, MLP AE, VAE, Trimmed AE | Seeds: {0, 1, 2}")
    print("=" * 105)

    data_path = REPO_ROOT / "data" / "processed" / "cicids2017_cleaned_dev.parquet"
    results_dir = REPO_ROOT / "results"
    scores_dir = results_dir / "scores"
    results_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "final_p2_seeds012.csv"
    cached_df = load_existing_results(csv_path)

    print(f"Loading cleaned dataset from {data_path} ...")
    df = pd.read_parquet(data_path)
    feature_cols = [c for c in df.columns if c not in ["Label", "Attack_Family", "Day", "Destination Port"]]

    models = ["IsolationForest", "OneClassSVM", "LOF", "PCA", "MLP AE", "VAE", "Trimmed AE"]
    seeds = [0, 1, 2]

    # Monday benign pool size is ~420k rows, so caps apply safely
    caps = {
        "OneClassSVM": 20000,
        "LOF": 50000,
        "IsolationForest": 200000,
        "PCA": 200000,
        "MLP AE": 500000,
        "VAE": 500000,
        "Trimmed AE": 500000,
    }

    total_runs = len(seeds) * len(models)
    completed_counter = 0

    for seed in seeds:
        print(f"\n{'='*40} SEED {seed} {'='*40}")
        split = generate_p2_temporal_splits(
            df,
            seed=seed,
            val_benign_ratio=0.2,
            train_days=["Monday"],
            test_days=["Tuesday", "Wednesday", "Thursday", "Friday"],
        )

        train_idx_all = split.train_indices
        val_idx = split.val_indices
        test_idx = split.test_indices

        # Fit preprocessor on full Monday train pool (capped at max 500k)
        max_train_cap = 500000
        rng = np.random.default_rng(seed)
        shuffled_train_idx = rng.permutation(train_idx_all)
        train_idx_500k = shuffled_train_idx[: min(len(shuffled_train_idx), max_train_cap)]

        X_train_df_500k = df.loc[train_idx_500k, feature_cols]
        preprocessor = TabularPreprocessor(skew_threshold=3.0, clip_value=10.0)
        preprocessor.fit(X_train_df_500k)

        # Precompute transformed arrays
        X_train_500k = preprocessor.transform(X_train_df_500k)
        X_val = preprocessor.transform(df.loc[val_idx, feature_cols])
        X_test = preprocessor.transform(df.loc[test_idx, feature_cols])

        # Test metadata
        test_families = df.loc[test_idx, "Attack_Family"].to_numpy()
        is_attack_test = (test_families != "Benign")
        y_test_binary = is_attack_test.astype(int)
        attack_prev = float(np.mean(y_test_binary))

        attack_fams_present = [f for f in np.unique(test_families) if f != "Benign"]

        for model_name in models:
            completed_counter += 1
            if is_result_cached(cached_df, seed, model_name):
                print(f"  [{completed_counter}/{total_runs}] [SKIPPED - CACHED] s={seed} | {model_name}")
                continue

            cap = caps[model_name]
            X_tr = X_train_500k[: min(len(X_train_500k), cap)]

            print(f"\n--- Training P2 {model_name} (cap={len(X_tr):,}, seed={seed}) ---")
            t0 = time.time()
            best_ep = None
            stop_ep = None

            if model_name == "IsolationForest":
                clf = IsolationForestBaseline(n_estimators=100, random_state=seed, n_jobs=4)
                clf.fit(X_tr)
                val_scores = clf.score(X_val)
                test_scores = clf.score(X_test)

            elif model_name == "OneClassSVM":
                clf = OneClassSVMBaseline(kernel="rbf", gamma="scale", nu=0.05)
                clf.fit(X_tr)
                val_scores = clf.score(X_val)
                test_scores = clf.score(X_test)

            elif model_name == "LOF":
                clf = LocalOutlierFactorBaseline(n_neighbors=20, novelty=True, n_jobs=4)
                clf.fit(X_tr)
                val_scores = clf.score(X_val)
                test_scores = clf.score(X_test)

            elif model_name == "PCA":
                clf = PCAReconstructionBaseline(n_components=10, random_state=seed)
                clf.fit(X_tr)
                val_scores = clf.score(X_val)
                test_scores = clf.score(X_test)

            elif model_name == "MLP AE":
                ae = MLPAutoencoder(
                    input_dim=X_tr.shape[1],
                    hidden_dims=(64, 32, 16),
                    bottleneck_dim=16,
                    seed=seed,
                )
                trained_ae, history = train_autoencoder(
                    ae,
                    X_train=X_tr,
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
                val_scores = trained_ae.compute_reconstruction_error(X_val)
                test_scores = trained_ae.compute_reconstruction_error(X_test)
                best_ep = min(history, key=lambda x: x["val_recon_loss"])["epoch"] if history else None
                stop_ep = len(history) if history else None

            elif model_name == "Trimmed AE":
                ae = TrimmedAutoencoder(
                    input_dim=X_tr.shape[1],
                    hidden_dims=(64, 32, 16),
                    bottleneck_dim=16,
                    seed=seed,
                )
                trained_ae, history = train_autoencoder(
                    ae,
                    X_train=X_tr,
                    X_val_benign=X_val,
                    seed=seed,
                    max_epochs=30,
                    batch_size=1024,
                    lr=0.001,
                    patience=3,
                    trim_frac=0.05,
                    warmup_epochs=2,
                    verbose=False,
                )
                val_scores = trained_ae.compute_reconstruction_error(X_val)
                test_scores = trained_ae.compute_reconstruction_error(X_test)
                best_ep = min(history, key=lambda x: x["val_recon_loss"])["epoch"] if history else None
                stop_ep = len(history) if history else None

            elif model_name == "VAE":
                vae = VariationalAutoencoder(
                    input_dim=X_tr.shape[1],
                    hidden_dims=(64, 32),
                    latent_dim=16,
                    seed=seed,
                )
                trained_vae, history = train_autoencoder(
                    vae,
                    X_train=X_tr,
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
                val_scores = trained_vae.compute_reconstruction_error(X_val)
                test_scores = trained_vae.compute_reconstruction_error(X_test)
                best_ep = min(history, key=lambda x: x["val_recon_loss"])["epoch"] if history else None
                stop_ep = len(history) if history else None

            train_time = round(time.time() - t0, 2)

            # Overall metrics
            metrics = compute_all_metrics(y_test_binary, test_scores)

            # Per-family AUROC (benign test vs each family present)
            benign_test_mask = (test_families == "Benign")
            per_fam_auroc = {}
            for fam in attack_fams_present:
                fam_mask = (test_families == fam)
                eval_mask = benign_test_mask | fam_mask
                y_sub = fam_mask[eval_mask].astype(int)
                s_sub = test_scores[eval_mask]
                per_fam_auroc[f"auroc_{fam.replace('/', '_')}"] = round(float(compute_auroc(y_sub, s_sub)), 6)

            # Drift check: median score of VAL-BENIGN (Monday) vs median score of benign test rows (Tue-Fri)
            val_benign_med = float(np.median(val_scores))
            test_benign_med = float(np.median(test_scores[benign_test_mask]))
            drift_ratio = round(test_benign_med / max(1e-12, val_benign_med), 4)

            # Save scores NPZ (float32)
            model_slug = model_name.replace(" ", "_")
            npz_path = scores_dir / f"final_p2_{model_slug}_s{seed}.npz"
            np.savez_compressed(
                npz_path,
                val_scores=val_scores.astype(np.float32),
                test_scores=test_scores.astype(np.float32),
                y_test=y_test_binary.astype(np.int32),
                test_families=test_families,
            )

            rec = {
                "protocol": "P2",
                "seed": seed,
                "model": model_name,
                "overall_auroc": round(float(metrics["auroc"]), 6),
                "overall_fpr_at_95_tpr": round(float(metrics["fpr_at_95_tpr"]), 6),
                "overall_tpr_at_1_fpr": round(float(metrics["tpr_at_1_fpr"]), 6),
                "overall_auprc": round(float(metrics["auprc"]), 6),
                "attack_prevalence": round(attack_prev, 8),
                "val_benign_median": round(val_benign_med, 6),
                "test_benign_median": round(test_benign_med, 6),
                "drift_ratio": drift_ratio,
                "train_time_seconds": train_time,
                "best_epoch": best_ep,
                "stopped_epoch": stop_ep,
                **per_fam_auroc,
            }

            save_single_result(rec, csv_path)
            cached_df = load_existing_results(csv_path)

            print(
                f"  [{completed_counter}/{total_runs}] [DONE] s={seed} | {model_name:<15} | "
                f"Overall AUROC: {metrics['auroc']:.4f} | FPR@95: {metrics['fpr_at_95_tpr']:.4f} | "
                f"Drift Ratio (TestBen/ValBen): {drift_ratio:.2f}x | Train: {train_time}s"
            )

    print("\nStage 2 finished successfully.")


if __name__ == "__main__":
    run_stage2()
