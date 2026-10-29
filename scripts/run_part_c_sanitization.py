"""Part C & D: Seed-Guided Sanitization Experiment and Reporting.

DEV DATA, NOT FOR REPORTED RESULTS
Setting B, Families: PortScan and DoS/DDoS.
c in {0, 2, 5, 10}%, n_seed = 10,000, seeds 0, 1, 2.
Seed-size ablation at c = 5% only: n_seed in {1,000, 10,000}.
Models: Sanitized AE, Sanitized PCA.
Reuses existing Plain AE, Trimmed AE, and Plain PCA results from results/p4_redo_seeds012.csv.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(4)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from baselines.classical import PCAReconstructionBaseline
from data.preprocess import TabularPreprocessor
from eval.metrics import compute_all_metrics
from models.autoencoders import MLPAutoencoder, train_autoencoder
from robustness.sanitization import draw_trusted_seed_set, filter_pool_with_seed
from splits.protocols import generate_p4_setting_b_splits


STATUS_TAG = "DEV DATA, NOT FOR REPORTED RESULTS"


def load_existing_results(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def save_single_result(record: Dict[str, Any], csv_path: Path, json_path: Path):
    rec_df = pd.DataFrame([record])
    if not csv_path.exists():
        rec_df.to_csv(csv_path, index=False)
    else:
        rec_df.to_csv(csv_path, mode="a", header=False, index=False)

    all_records = []
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                all_records = json.load(f)
        except Exception:
            all_records = []
    all_records.append(record)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)


def is_result_cached(df: pd.DataFrame, setting: str, family: str, c: int, seed: int, n_seed: int, model: str) -> bool:
    if df.empty:
        return False
    required_cols = {"setting", "family", "c", "seed", "n_seed", "model"}
    if not required_cols.issubset(df.columns):
        return False
    match = df[
        (df["setting"] == setting)
        & (df["family"] == family)
        & (df["c"] == c)
        & (df["seed"] == seed)
        & (df["n_seed"] == n_seed)
        & (df["model"] == model)
    ]
    return not match.empty


def run_sanitized_pipeline(
    df: pd.DataFrame,
    feature_cols: List[str],
    family: str,
    c_pct: int,
    seed: int,
    n_seed: int,
    model_type: str,  # 'ae' or 'pca'
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute the two-stage seed-guided sanitization pipeline."""
    c = c_pct / 100.0
    t0 = time.time()

    # 1. Generate P4 Setting B split
    split = generate_p4_setting_b_splits(
        df,
        seed=seed,
        family=family,
        contamination_rate=c,
        n_train_total=200000,
        val_benign_ratio=0.2,
        test_benign_ratio=0.2,
    )

    # 2. Draw trusted clean seed set S strictly disjoint from pool, VAL and TEST
    S_indices, S_fit, S_cal = draw_trusted_seed_set(
        df,
        seed=seed,
        n_seed=n_seed,
        n_train_max=200000,
        val_benign_ratio=0.2,
        test_benign_ratio=0.2,
    )

    # Invariant checks
    pool_idx_set = set(split.train_indices)
    val_idx_set = set(split.val_indices)
    test_idx_set = set(split.test_indices)
    seed_idx_set = set(S_indices)
    assert pool_idx_set.isdisjoint(seed_idx_set), "CRITICAL: Seed set overlaps with train pool!"
    assert val_idx_set.isdisjoint(seed_idx_set), "CRITICAL: Seed set overlaps with VAL-BENIGN!"
    assert test_idx_set.isdisjoint(seed_idx_set), "CRITICAL: Seed set overlaps with TEST!"

    # Unlabeled feature dataframes (NO labels passed into filter_pool_with_seed)
    X_pool_raw = df.loc[split.train_indices, feature_cols]
    X_seed_fit_raw = df.loc[S_fit, feature_cols]
    X_seed_cal_raw = df.loc[S_cal, feature_cols]

    # Stage 1: Fit preprocessor on S_fit only, train stage-1 model, POT threshold on S_cal, filter pool
    keep_mask, drop_mask, pot_threshold, stage1_model, prep_stage1 = filter_pool_with_seed(
        X_pool_raw=X_pool_raw,
        X_seed_fit_raw=X_seed_fit_raw,
        X_seed_cal_raw=X_seed_cal_raw,
        model_type=model_type,
        q=0.01,
        seed=seed,
        skew_threshold=3.0,
        clip_value=10.0,
        max_epochs=30,
        batch_size=256,
        lr=0.001,
        patience=3,
        verbose=False,
    )

    # Label mask used ONLY downstream for reporting removal statistics
    train_families = df.loc[split.train_indices, "Attack_Family"].to_numpy()
    is_attack_pool = (train_families == family)
    is_benign_pool = ~is_attack_pool
    n_attack_pool = int(np.sum(is_attack_pool))
    n_benign_pool = int(np.sum(is_benign_pool))

    n_attack_removed = int(np.sum(drop_mask & is_attack_pool))
    n_benign_removed = int(np.sum(drop_mask & is_benign_pool))

    frac_attack_removed = float(n_attack_removed / n_attack_pool) if n_attack_pool > 0 else 0.0
    frac_benign_removed = float(n_benign_removed / n_benign_pool) if n_benign_pool > 0 else 0.0

    kept_pool_indices = split.train_indices[keep_mask]
    n_kept = len(kept_pool_indices)

    # Stage 2:
    # Prepare test set and val set
    X_val_df = df.loc[split.val_indices, feature_cols]
    X_test_df = df.loc[split.test_indices, feature_cols]
    y_test = (df.loc[split.test_indices, "Attack_Family"] == family).astype(int).to_numpy()

    best_ep: Optional[int] = None
    stop_ep: Optional[int] = None

    if model_type == "ae":
        # Fit new preprocessor on (kept pool rows + S)
        stage2_train_indices = np.concatenate([kept_pool_indices, S_indices])
        X_stage2_raw = df.loc[stage2_train_indices, feature_cols]

        prep_stage2 = TabularPreprocessor(skew_threshold=3.0, clip_value=10.0)
        prep_stage2.fit(X_stage2_raw)

        X_stage2_train = prep_stage2.transform(X_stage2_raw)
        X_stage2_val = prep_stage2.transform(X_val_df)
        X_stage2_test = prep_stage2.transform(X_test_df)

        final_ae = MLPAutoencoder(
            input_dim=X_stage2_train.shape[1],
            hidden_dims=(64, 32, 16),
            bottleneck_dim=16,
            seed=seed,
        )
        trained_ae, history = train_autoencoder(
            final_ae,
            X_train=X_stage2_train,
            X_val_benign=X_stage2_val,
            seed=seed,
            max_epochs=30,
            batch_size=256,
            lr=0.001,
            patience=3,
            trim_frac=0.0,
            warmup_epochs=0,
            verbose=False,
        )
        test_scores = trained_ae.compute_reconstruction_error(X_stage2_test)
        best_ep = min(history, key=lambda x: x["val_recon_loss"])["epoch"] if history else None
        stop_ep = len(history) if history else None
        model_name = "Sanitized AE"

    elif model_type == "pca":
        # Fit new preprocessor on kept pool rows only, and train final PCA on kept pool rows
        X_stage2_raw = df.loc[kept_pool_indices, feature_cols]

        prep_stage2 = TabularPreprocessor(skew_threshold=3.0, clip_value=10.0)
        prep_stage2.fit(X_stage2_raw)

        X_stage2_train = prep_stage2.transform(X_stage2_raw)
        X_stage2_test = prep_stage2.transform(X_test_df)

        final_pca = PCAReconstructionBaseline(
            n_components=10,
            random_state=seed,
        )
        final_pca.fit(X_stage2_train)
        test_scores = final_pca.score(X_stage2_test)
        model_name = "Sanitized PCA"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    t_run = time.time() - t0
    metrics = compute_all_metrics(y_test, test_scores)
    attack_prev = float(np.mean(y_test))

    record = {
        "setting": "Setting B",
        "family": family,
        "c": c_pct,
        "seed": seed,
        "n_seed": n_seed,
        "model": model_name,
        "auroc": round(float(metrics["auroc"]), 6),
        "fpr_at_95_tpr": round(float(metrics["fpr_at_95_tpr"]), 6),
        "tpr_at_1_fpr": round(float(metrics["tpr_at_1_fpr"]), 6),
        "attack_prevalence": round(attack_prev, 8),
        "best_epoch": best_ep,
        "stopped_epoch": stop_ep,
        "runtime": round(t_run, 2),
        "pot_threshold": round(pot_threshold, 6),
        "pool_size": len(split.train_indices),
        "n_kept": n_kept,
        "n_dropped": int(np.sum(drop_mask)),
        "n_attack_pool": n_attack_pool,
        "n_benign_pool": n_benign_pool,
        "n_attack_removed": n_attack_removed,
        "n_benign_removed": n_benign_removed,
        "frac_attack_removed": round(frac_attack_removed, 6),
        "frac_benign_removed": round(frac_benign_removed, 6),
    }

    return record, metrics


def main():
    print("=" * 115)
    print(f"[{STATUS_TAG}]")
    print("PART C: SEED-GUIDED SANITIZATION EXPERIMENT")
    print("Setting B | Families: PortScan, DoS/DDoS | c in {0, 2, 5, 10}% | Seeds: {0, 1, 2}")
    print("Seed Size: n_seed = 10,000 (Ablation at c=5%: n_seed in {1,000, 10,000}) | Max Threads: 4")
    print("=" * 115)

    data_path = REPO_ROOT / "data" / "processed" / "cicids2017_cleaned_dev.parquet"
    results_dir = REPO_ROOT / "results"
    figures_dir = REPO_ROOT / "results" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "p4_sanitization_results.csv"
    json_path = results_dir / "p4_sanitization_results.json"

    cached_df = load_existing_results(csv_path)

    print(f"Loading cleaned dataset from {data_path} ...")
    df = pd.read_parquet(data_path)
    feature_cols = [c for c in df.columns if c not in ["Label", "Attack_Family", "Day", "Destination Port"]]

    families = ["PortScan", "DoS/DDoS"]
    seeds = [0, 1, 2]
    c_levels = [0, 2, 5, 10]
    models = ["ae", "pca"]

    # Build execution matrix
    # Main grid: n_seed = 10,000, c in {0, 2, 5, 10}
    # Ablation: n_seed = 1,000, c = 5%
    jobs = []
    for s in seeds:
        for fam in families:
            for c_pct in c_levels:
                for m in models:
                    jobs.append((fam, c_pct, s, 10000, m))
            # Seed ablation at c = 5%
            for m in models:
                jobs.append((fam, 5, s, 1000, m))

    total_jobs = len(jobs)
    print(f"Total sanitization runs planned: {total_jobs}")

    for idx, (fam, c_pct, s, n_seed, m_type) in enumerate(jobs, 1):
        m_name = "Sanitized AE" if m_type == "ae" else "Sanitized PCA"
        if is_result_cached(cached_df, "Setting B", fam, c_pct, s, n_seed, m_name):
            print(f"  [{idx}/{total_jobs}] [SKIPPED - CACHED] {fam:<9} | c={c_pct:02d}% | s={s} | n_seed={n_seed:<5} | Model: {m_name}")
            continue

        record, metrics = run_sanitized_pipeline(
            df=df,
            feature_cols=feature_cols,
            family=fam,
            c_pct=c_pct,
            seed=s,
            n_seed=n_seed,
            model_type=m_type,
        )

        save_single_result(record, csv_path, json_path)
        cached_df = load_existing_results(csv_path)

        print(
            f"  [{idx}/{total_jobs}] [PROGRESS] {fam:<9} | c={c_pct:02d}% | s={s} | n_seed={n_seed:<5} | "
            f"Model: {record['model']:<13} | AUROC: {record['auroc']:.4f} | FPR@95: {record['fpr_at_95_tpr']:.4f} | "
            f"Att Removed: {record['frac_attack_removed']*100:.1f}% | Ben Removed: {record['frac_benign_removed']*100:.1f}% | "
            f"Time: {record['runtime']:.2f}s"
        )

    print("\nAll sanitization runs completed.")


if __name__ == "__main__":
    main()
