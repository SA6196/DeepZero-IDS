"""Stage 1: Final Protocol P1 Grid (LOFO).

DEV DATA, NOT FOR REPORTED RESULTS
Held-out families: DoS/DDoS, PortScan, BruteForce, WebAttack, Botnet.
(Infiltration excluded: 36 rows, unreliable statistical sample).
Models: IsolationForest, OneClassSVM, LOF, PCA, MLP AE, VAE, Trimmed AE (trim 5%).
Seeds: 0, 1, 2.
Train Caps: OCSVM=20k, LOF=50k, IF=200k, PCA=200k, AE/VAE/Trimmed AE=500k.
Outputs:
- results/final_p1_seeds012.csv
- results/scores/final_p1_{model}_{family}_s{seed}.npz
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
from eval.metrics import compute_all_metrics
from models.autoencoders import (
    MLPAutoencoder,
    TrimmedAutoencoder,
    VariationalAutoencoder,
    train_autoencoder,
)
from splits.protocols import generate_p1_lofo_splits

STATUS_TAG = "DEV DATA, NOT FOR REPORTED RESULTS"


def load_existing_results(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def is_result_cached(df: pd.DataFrame, family: str, seed: int, model: str) -> bool:
    if df.empty:
        return False
    required_cols = {"family", "seed", "model"}
    if not required_cols.issubset(df.columns):
        return False
    match = df[
        (df["family"] == family)
        & (df["seed"] == seed)
        & (df["model"] == model)
    ]
    return not match.empty


def save_single_result(record: Dict[str, Any], csv_path: Path):
    rec_df = pd.DataFrame([record])
    if not csv_path.exists():
        rec_df.to_csv(csv_path, index=False)
    else:
        rec_df.to_csv(csv_path, mode="a", header=False, index=False)


def run_stage1():
    print("=" * 105)
    print(f"[{STATUS_TAG}]")
    print("STAGE 1: FINAL PROTOCOL P1 GRID (LOFO)")
    print("Held-out Families: DoS/DDoS, PortScan, BruteForce, WebAttack, Botnet (Infiltration excluded)")
    print("Models: IsolationForest, OneClassSVM, LOF, PCA, MLP AE, VAE, Trimmed AE | Seeds: {0, 1, 2}")
    print("=" * 105)

    data_path = REPO_ROOT / "data" / "processed" / "cicids2017_cleaned_dev.parquet"
    results_dir = REPO_ROOT / "results"
    scores_dir = results_dir / "scores"
    results_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "final_p1_seeds012.csv"
    cached_df = load_existing_results(csv_path)

    print(f"Loading cleaned dataset from {data_path} ...")
    df = pd.read_parquet(data_path)
    feature_cols = [c for c in df.columns if c not in ["Label", "Attack_Family", "Day", "Destination Port"]]

    families = ["DoS/DDoS", "PortScan", "BruteForce", "WebAttack", "Botnet"]
    models = ["IsolationForest", "OneClassSVM", "LOF", "PCA", "MLP AE", "VAE", "Trimmed AE"]
    seeds = [0, 1, 2]

    # Speed caps (nested subsampling on clean benign train)
    caps = {
        "OneClassSVM": 20000,
        "LOF": 50000,
        "IsolationForest": 200000,
        "PCA": 200000,
        "MLP AE": 500000,
        "VAE": 500000,
        "Trimmed AE": 500000,
    }

    total_runs = len(seeds) * len(models) * len(families)
    completed_counter = 0

    for seed in seeds:
        print(f"\n{'='*40} SEED {seed} {'='*40}")
        # Generate P1 LOFO splits for this seed
        splits = generate_p1_lofo_splits(
            df,
            seed=seed,
            val_benign_ratio=0.2,
            test_benign_ratio=0.2,
            target_families=families,
        )

        # Invariant: benign_train_indices and benign_val_indices are identical across families
        ref_split = splits[families[0]]
        train_idx_all = ref_split.train_indices
        val_idx = ref_split.val_indices

        # Fit preprocessor on full 500k train cap
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

        # Pre-transform TEST for each family
        X_tests = {}
        y_tests = {}
        for fam in families:
            sp = splits[fam]
            X_test_raw = df.loc[sp.test_indices, feature_cols]
            X_tests[fam] = preprocessor.transform(X_test_raw)
            y_tests[fam] = (df.loc[sp.test_indices, "Attack_Family"] == fam).astype(int).to_numpy()

        # Model loop
        for model_name in models:
            # Check if all families for this model & seed are already cached
            all_cached = all(
                is_result_cached(cached_df, fam, seed, model_name)
                for fam in families
            )
            if all_cached:
                for fam in families:
                    completed_counter += 1
                    print(f"  [{completed_counter}/{total_runs}] [SKIPPED - CACHED] s={seed} | {model_name:<15} | {fam}")
                continue

            # Model-specific train cap (nested within 500k)
            cap = caps[model_name]
            X_tr = X_train_500k[: min(len(X_train_500k), cap)]

            print(f"\n--- Training {model_name} (cap={len(X_tr):,}, seed={seed}) ---")
            t0 = time.time()
            best_ep = None
            stop_ep = None

            if model_name == "IsolationForest":
                clf = IsolationForestBaseline(n_estimators=100, random_state=seed, n_jobs=4)
                clf.fit(X_tr)
                val_scores = clf.score(X_val)
                score_fn = clf.score

            elif model_name == "OneClassSVM":
                clf = OneClassSVMBaseline(kernel="rbf", gamma="scale", nu=0.05)
                clf.fit(X_tr)
                val_scores = clf.score(X_val)
                score_fn = clf.score

            elif model_name == "LOF":
                clf = LocalOutlierFactorBaseline(n_neighbors=20, novelty=True, n_jobs=4)
                clf.fit(X_tr)
                val_scores = clf.score(X_val)
                score_fn = clf.score

            elif model_name == "PCA":
                clf = PCAReconstructionBaseline(n_components=10, random_state=seed)
                clf.fit(X_tr)
                val_scores = clf.score(X_val)
                score_fn = clf.score

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
                score_fn = trained_ae.compute_reconstruction_error
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
                score_fn = trained_ae.compute_reconstruction_error
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
                score_fn = trained_vae.compute_reconstruction_error
                best_ep = min(history, key=lambda x: x["val_recon_loss"])["epoch"] if history else None
                stop_ep = len(history) if history else None

            train_time = round(time.time() - t0, 2)

            # Evaluate on all families
            for fam in families:
                completed_counter += 1
                if is_result_cached(cached_df, fam, seed, model_name):
                    print(f"  [{completed_counter}/{total_runs}] [SKIPPED - CACHED] s={seed} | {model_name:<15} | {fam}")
                    continue

                test_scores = score_fn(X_tests[fam])
                y_test = y_tests[fam]
                metrics = compute_all_metrics(y_test, test_scores)
                attack_prev = float(np.mean(y_test))

                # Save score NPZ (float32)
                model_slug = model_name.replace(" ", "_")
                fam_slug = fam.replace("/", "_")
                npz_path = scores_dir / f"final_p1_{model_slug}_{fam_slug}_s{seed}.npz"
                np.savez_compressed(
                    npz_path,
                    val_scores=val_scores.astype(np.float32),
                    test_scores=test_scores.astype(np.float32),
                    y_test=y_test.astype(np.int32),
                )

                rec = {
                    "protocol": "P1",
                    "family": fam,
                    "seed": seed,
                    "model": model_name,
                    "auroc": round(float(metrics["auroc"]), 6),
                    "fpr_at_95_tpr": round(float(metrics["fpr_at_95_tpr"]), 6),
                    "tpr_at_1_fpr": round(float(metrics["tpr_at_1_fpr"]), 6),
                    "auprc": round(float(metrics["auprc"]), 6),
                    "attack_prevalence": round(attack_prev, 8),
                    "train_time_seconds": train_time,
                    "best_epoch": best_ep,
                    "stopped_epoch": stop_ep,
                }
                save_single_result(rec, csv_path)
                cached_df = load_existing_results(csv_path)

                print(
                    f"  [{completed_counter}/{total_runs}] [DONE] s={seed} | {model_name:<15} | {fam:<11} | "
                    f"AUROC: {metrics['auroc']:.4f} | FPR@95: {metrics['fpr_at_95_tpr']:.4f} | AUPRC: {metrics['auprc']:.4f}"
                )

    print("\nStage 1 finished successfully.")


if __name__ == "__main__":
    run_stage1()
