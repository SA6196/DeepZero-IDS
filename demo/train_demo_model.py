"""Train demo MLP Autoencoder on benign network flows.

DEV DATA, NOT FOR REPORTED RESULTS

This script trains ONE final autoencoder (MLP AE: input -> 64 -> 32 -> 16 -> 32 -> 64 -> input)
on benign-only TRAIN data from data/processed/cicids2017_cleaned_dev.parquet.
- Preprocessing: fit on TRAIN only (skew_threshold=3.0, clip_value=10.0).
- Early stopping on VAL-BENIGN reconstruction loss (seed 0).
- Training rows capped at 200,000.
- Alarm threshold calibrated with thresholds/pot.py (q = 1e-2) using VAL-BENIGN scores only.
- Saves to demo/artifacts/: model weights, fitted preprocessor, threshold.json.
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

torch.set_num_threads(4)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from data.preprocess import TabularPreprocessor
from models.autoencoders import MLPAutoencoder, train_autoencoder
from splits.protocols import generate_p1_lofo_splits
from thresholds.pot import fit_pot_threshold

STATUS_TAG = "DEV DATA, NOT FOR REPORTED RESULTS"


def train_demo_model(
    data_path: Path = REPO_ROOT / "data" / "processed" / "cicids2017_cleaned_dev.parquet",
    output_dir: Path = REPO_ROOT / "demo" / "artifacts",
    max_train_rows: int = 200_000,
    seed: int = 0,
    q: float = 0.01,
) -> None:
    print("=" * 80)
    print(f"[{STATUS_TAG}]")
    print("DEMO AUTOENCODER TRAINING")
    print(f"Seed: {seed} | Max train rows: {max_train_rows:,} | Target risk q: {q}")
    print("=" * 80)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Please verify the path before running."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading cleaned dataset from {data_path} ...")
    t0 = time.time()
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} flows with {len(df.columns)} columns in {time.time() - t0:.2f}s")

    feature_cols = [
        c for c in df.columns if c not in ["Label", "Attack_Family", "Day", "Destination Port"]
    ]
    print(f"Identified {len(feature_cols)} raw feature columns.")

    print("Generating Protocol P1 benign splits (seed=0, val=0.2, test=0.2) ...")
    splits = generate_p1_lofo_splits(
        df,
        seed=seed,
        val_benign_ratio=0.2,
        test_benign_ratio=0.2,
        target_families=["DoS/DDoS"],
    )
    ref_split = splits["DoS/DDoS"]
    train_idx_all = ref_split.train_indices
    val_idx = ref_split.val_indices

    print(f"Available benign TRAIN rows: {len(train_idx_all):,}")
    print(f"Available benign VAL rows: {len(val_idx):,}")

    rng = np.random.default_rng(seed)
    shuffled_train_idx = rng.permutation(train_idx_all)
    train_idx_capped = shuffled_train_idx[: min(len(shuffled_train_idx), max_train_rows)]
    print(f"Capping TRAIN rows to: {len(train_idx_capped):,}")

    # Fit preprocessor strictly on TRAIN rows only
    print("\nFitting TabularPreprocessor on TRAIN data only ...")
    X_train_df = df.loc[train_idx_capped, feature_cols]
    preprocessor = TabularPreprocessor(skew_threshold=3.0, clip_value=10.0)
    preprocessor.fit(X_train_df)

    X_train = preprocessor.transform(X_train_df)
    X_val_benign = preprocessor.transform(df.loc[val_idx, feature_cols])
    input_dim = X_train.shape[1]
    print(f"Preprocessor fitted. Retained features: {input_dim}")

    # Initialize accepted MLP Autoencoder architecture
    print("\nInitializing MLP Autoencoder architecture (input -> 64 -> 32 -> 16 -> 32 -> 64 -> input) ...")
    model = MLPAutoencoder(
        input_dim=input_dim,
        hidden_dims=(64, 32),
        bottleneck_dim=16,
        seed=seed,
    )

    print("\nStarting training (max 30 epochs, batch=1024, lr=1e-3, patience=3 on VAL-BENIGN) ...")
    trained_model, history = train_autoencoder(
        model=model,
        X_train=X_train,
        X_val_benign=X_val_benign,
        seed=seed,
        max_epochs=30,
        batch_size=1024,
        lr=1e-3,
        patience=3,
        trim_frac=0.0,
        warmup_epochs=0,
        verbose=True,
    )

    # Calibrate alarm threshold via Peaks-Over-Threshold (POT) on VAL-BENIGN scores only
    print("\nScoring VAL-BENIGN flows for threshold calibration ...")
    val_scores = trained_model.compute_reconstruction_error(X_val_benign)
    n_calib = int(len(val_scores))
    print(f"Val scores count: {n_calib:,} | min={val_scores.min():.5f}, median={np.median(val_scores):.5f}, max={val_scores.max():.5f}")

    print(f"Calibrating POT threshold with target risk q = {q} (q = 1e-2) ...")
    threshold_val = fit_pot_threshold(val_scores, q=q, percentile_u=90.0)
    print(f"Calibrated POT Threshold: {threshold_val:.6f}")

    # Save artifacts to demo/artifacts/
    model_weights_path = output_dir / "model_weights.pt"
    torch.save(trained_model.state_dict(), model_weights_path)
    print(f"Saved model weights to {model_weights_path}")

    preprocessor_path = output_dir / "preprocessor.joblib"
    preprocessor.save(preprocessor_path)
    print(f"Saved fitted preprocessor to {preprocessor_path}")

    threshold_info = {
        "status": STATUS_TAG,
        "threshold": float(threshold_val),
        "q": float(q),
        "calibration_rows": n_calib,
        "number_of_calibration_rows": n_calib,
        "model_architecture": "MLPAutoencoder(input_dim, 64, 32, 16, 32, 64, input_dim)",
        "input_dim": input_dim,
        "seed": seed,
        "max_train_rows": max_train_rows,
        "feature_cols": preprocessor.feature_names_in_,
    }
    threshold_json_path = output_dir / "threshold.json"
    with open(threshold_json_path, "w", encoding="utf-8") as f:
        json.dump(threshold_info, f, indent=2)
    print(f"Saved threshold metadata to {threshold_json_path}")

    # Also save model config for quick model reconstruction
    config_info = {
        "status": STATUS_TAG,
        "input_dim": input_dim,
        "hidden_dims": [64, 32],
        "bottleneck_dim": 16,
    }
    config_path = output_dir / "model_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[{STATUS_TAG}] Training complete. Artifacts successfully saved.")
    print("=" * 80)


if __name__ == "__main__":
    train_demo_model()
