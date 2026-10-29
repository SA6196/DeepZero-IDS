"""Protocol P1 Refresh for PCA and Plain AE only with Fixed Preprocessing.

DEV DATA, NOT FOR REPORTED RESULTS
Rules:
- 500k TRAIN, all six families
- Preprocessor fit strictly on TRAIN (500k rows) with post-standardization clip [-10, 10]
- max_threads = 4 enforced everywhere
- Print progress line after every finished run
- Compare old vs new AUROC and FPR@95%TPR, flag changes > 0.03
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
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import yaml

torch.set_num_threads(4)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from baselines.classical import PCAReconstructionBaseline
from data.preprocess import TabularPreprocessor
from eval.metrics import compute_all_metrics
from models.autoencoders import MLPAutoencoder, train_autoencoder
from splits.protocols import generate_p1_lofo_splits


def main():
    config_path = REPO_ROOT / "configs" / "baselines_p1_cicids2017.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["experiment"].get("seed", 0))
    status_tag = cfg["experiment"].get("status", "DEV DATA, NOT FOR REPORTED RESULTS")
    max_threads = int(cfg["experiment"].get("max_threads", 4))

    print("=" * 100)
    print(f"[{status_tag}]")
    print(f"Protocol P1 REFRESH: PCA & Plain AE (Fixed Preprocessing with [-10, 10] Clip)")
    print(f"Seed: {seed} | Max Threads: {max_threads}")
    print("=" * 100)

    data_path = REPO_ROOT / cfg["dataset"]["cleaned_path"]
    print(f"Loading cleaned dataset from {data_path} ...")
    t0 = time.time()
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} rows in {time.time() - t0:.2f}s.\n")

    feature_cols = [
        c for c in df.columns
        if c not in ["Label", "Attack_Family", "Day", "Destination Port"]
    ]

    target_families = cfg["protocol"].get(
        "target_families",
        ["DoS/DDoS", "PortScan", "BruteForce", "WebAttack", "Botnet", "Infiltration"],
    )

    cap_pca = int(cfg["speed_caps"]["train_rows"].get("pca", 200000))
    cap_ae = int(cfg["speed_caps"]["train_rows"].get("autoencoder", 500000))
    skew_thresh = float(cfg["preprocessing"].get("skew_threshold", 3.0))

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating Protocol P1 LOFO splits (seed={seed})...")
    lofo_splits = generate_p1_lofo_splits(
        df,
        seed=seed,
        val_benign_ratio=float(cfg["protocol"].get("val_benign_ratio", 0.2)),
        test_benign_ratio=float(cfg["protocol"].get("test_benign_ratio", 0.2)),
        target_families=target_families,
    )

    refreshed_records: List[Dict[str, Any]] = []

    total_jobs = len(target_families) * 2
    job_idx = 0

    for family in target_families:
        split = lofo_splits[family]
        print("\n" + "=" * 90)
        print(f"Processing Held-Out Family: {family}")
        print("=" * 90)

        # 1. Nest benign training indices
        rng_train = np.random.default_rng(seed)
        shuffled_train = rng_train.permutation(split.train_indices)

        train_idx_500k = shuffled_train[: min(cap_ae, len(shuffled_train))]
        train_idx_200k = shuffled_train[: min(cap_pca, len(shuffled_train))]

        print(f"  Fitting Preprocessor on TRAIN (500k rows, clip_value=10.0, skew_thresh={skew_thresh}) ...")
        t_prep0 = time.time()
        X_train_df = df.loc[train_idx_500k, feature_cols]
        preprocessor = TabularPreprocessor(skew_threshold=skew_thresh, clip_value=10.0)
        preprocessor.fit(X_train_df)
        print(f"  Preprocessor fit in {time.time() - t_prep0:.2f}s.")

        X_train_500k = preprocessor.transform(X_train_df)
        X_train_200k = X_train_500k[: len(train_idx_200k)]

        # Sample clean VAL-BENIGN (capped at 100k for efficiency)
        rng_val = np.random.default_rng(seed + 500)
        val_sub_indices = rng_val.choice(
            split.val_indices,
            size=min(100000, len(split.val_indices)),
            replace=False,
        )
        X_val_benign = preprocessor.transform(df.loc[val_sub_indices, feature_cols])

        # Prepare TEST: all benign test + all held-out family test
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

        n_benign_test = len(benign_test_indices)
        n_attack_test = len(attack_test_indices)
        attack_prev = n_attack_test / (n_benign_test + n_attack_test)

        # -------------------------------------------------------------
        # Model 1: PCA Reconstruction
        # -------------------------------------------------------------
        job_idx += 1
        t_start = time.time()
        pca = PCAReconstructionBaseline(
            n_components=cfg["models"]["pca"]["n_components"],
            random_state=cfg["models"]["pca"]["random_state"],
        )
        pca.fit(X_train_200k)
        scores_pca = pca.score(X_test)
        t_pca = time.time() - t_start
        m_pca = compute_all_metrics(y_test, scores_pca)

        print(
            f"  [{job_idx}/{total_jobs}] [DONE] Family: {family:<12} | Model: {'PCA':<10} | "
            f"AUROC: {m_pca['auroc']:.4f} | FPR@95%: {m_pca['fpr_at_95_tpr']:.4f} | Time: {t_pca:.2f}s"
        )

        refreshed_records.append({
            "family": family,
            "model": "PCA",
            "auroc": round(m_pca["auroc"], 6),
            "fpr_at_95_tpr": round(m_pca["fpr_at_95_tpr"], 6),
            "tpr_at_1_fpr": round(m_pca["tpr_at_1_fpr"], 6),
            "runtime": round(t_pca, 2),
        })

        # -------------------------------------------------------------
        # Model 2: Plain MLP Autoencoder
        # -------------------------------------------------------------
        job_idx += 1
        t_start = time.time()
        ae_cfg = cfg["models"]["autoencoder"]
        ae_model = MLPAutoencoder(
            input_dim=X_train_500k.shape[1],
            hidden_dims=tuple(ae_cfg["hidden_dims"]),
            bottleneck_dim=ae_cfg["bottleneck_dim"],
            seed=seed,
        )
        log_ae_path = results_dir / f"loss_refreshed_ae_p1_{family}_seed{seed}.csv"
        trained_ae, history = train_autoencoder(
            ae_model,
            X_train=X_train_500k,
            X_val_benign=X_val_benign,
            seed=seed,
            max_epochs=ae_cfg["max_epochs"],
            batch_size=ae_cfg["batch_size"],
            lr=ae_cfg["lr"],
            patience=ae_cfg["patience"],
            trim_frac=0.0,
            warmup_epochs=0,
            log_csv_path=log_ae_path,
            verbose=False,
        )
        scores_ae = trained_ae.compute_reconstruction_error(X_test)
        t_ae = time.time() - t_start
        m_ae = compute_all_metrics(y_test, scores_ae)

        print(
            f"  [{job_idx}/{total_jobs}] [DONE] Family: {family:<12} | Model: {'Plain AE':<10} | "
            f"AUROC: {m_ae['auroc']:.4f} | FPR@95%: {m_ae['fpr_at_95_tpr']:.4f} | Time: {t_ae:.2f}s"
        )

        refreshed_records.append({
            "family": family,
            "model": "Plain AE",
            "auroc": round(m_ae["auroc"], 6),
            "fpr_at_95_tpr": round(m_ae["fpr_at_95_tpr"], 6),
            "tpr_at_1_fpr": round(m_ae["tpr_at_1_fpr"], 6),
            "runtime": round(t_ae, 2),
        })

    # Save refreshed P1 results
    csv_out = results_dir / f"p1_refreshed_pca_ae_seed{seed}.csv"
    json_out = results_dir / f"p1_refreshed_pca_ae_seed{seed}.json"
    res_df = pd.DataFrame(refreshed_records)
    res_df.to_csv(csv_out, index=False)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(refreshed_records, f, indent=2)

    # Load old P1 results for comparison
    old_csv = results_dir / f"p1_baselines_and_ae_seed{seed}.csv"
    old_df = pd.read_csv(old_csv) if old_csv.exists() else None

    print("\n" + "=" * 105)
    print(f"[{status_tag}]")
    print("PROTOCOL P1 REFRESH COMPARISON: Old vs New (Fixed Preprocessing)")
    print("=" * 105)

    comp_rows = []
    for fam in target_families:
        for m_name in ["PCA", "Plain AE"]:
            old_model_name = "PCA" if m_name == "PCA" else "Autoencoder"
            new_sub = res_df[(res_df["family"] == fam) & (res_df["model"] == m_name)]
            new_auroc = new_sub["auroc"].values[0]
            new_fpr = new_sub["fpr_at_95_tpr"].values[0]

            if old_df is not None:
                old_sub = old_df[(old_df["family"] == fam) & (old_df["model"] == old_model_name)]
                old_auroc = old_sub["auroc"].values[0]
                old_fpr = old_sub["fpr_at_95_tpr"].values[0]
            else:
                old_auroc, old_fpr = np.nan, np.nan

            delta_auroc = new_auroc - old_auroc
            delta_fpr = new_fpr - old_fpr
            flag = "FLAG (|Delta| > 0.03)" if abs(delta_auroc) > 0.03 else "OK (<= 0.03)"

            comp_rows.append({
                "Family": fam,
                "Model": m_name,
                "Old AUROC": f"{old_auroc:.4f}",
                "New AUROC": f"{new_auroc:.4f}",
                "Delta AUROC": f"{delta_auroc:+.4f}",
                "Old FPR@95%": f"{old_fpr:.4f}",
                "New FPR@95%": f"{new_fpr:.4f}",
                "Delta FPR": f"{delta_fpr:+.4f}",
                "Status": flag,
            })

    print(pd.DataFrame(comp_rows).to_string(index=False))


if __name__ == "__main__":
    main()
