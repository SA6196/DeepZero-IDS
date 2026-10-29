"""Stage 4: Sanitization Diagnostic and Sensitivity Analysis.

DEV DATA, NOT FOR REPORTED RESULTS
4a (diagnostic, no tuning):
- Setting B, PortScan and DoS/DDoS, c = 5%, seeds 0-2, n_seed = 10,000.
- Stage-1 AE and Stage-1 PCA fit on S_fit.
- Report:
  (i) Fraction of injected attack rows whose score exceeds 50th, 90th, 95th, 99th percentile of S_cal scores.
  (ii) AUROC of stage-1 score separating injected-attack rows from benign pool rows.
  (iii) Median attack score / median S_cal score.
- Output: results/final_stage4a_diagnostic.csv

4b (sensitivity, NOT tuning):
- POT q in {0.03, 0.05} (u = 90th percentile of S_cal; q = 0.01 reused from results/p4_sanitization_results.csv).
- c in {0, 5}%, PortScan and DoS/DDoS, Sanitized AE and Sanitized PCA, seeds 0-2, n_seed = 10,000.
- Report: AUROC, FPR@95%TPR, attack rows removed, benign rows removed.
- Output: results/final_stage4b_sensitivity.csv
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

from baselines.classical import PCAReconstructionBaseline
from data.preprocess import TabularPreprocessor
from eval.metrics import compute_all_metrics, compute_auroc
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


def save_single_result(record: Dict[str, Any], csv_path: Path):
    rec_df = pd.DataFrame([record])
    if not csv_path.exists():
        rec_df.to_csv(csv_path, index=False)
    else:
        rec_df.to_csv(csv_path, mode="a", header=False, index=False)


def run_stage4():
    print("=" * 105)
    print(f"[{STATUS_TAG}]")
    print("STAGE 4: SANITIZATION DIAGNOSTIC (4a) AND SENSITIVITY (4b)")
    print("=" * 105)

    data_path = REPO_ROOT / "data" / "processed" / "cicids2017_cleaned_dev.parquet"
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_4a = results_dir / "final_stage4a_diagnostic.csv"
    csv_4b = results_dir / "final_stage4b_sensitivity.csv"
    existing_sanit_csv = results_dir / "p4_sanitization_results.csv"

    print(f"Loading cleaned dataset from {data_path} ...")
    df = pd.read_parquet(data_path)
    feature_cols = [c for c in df.columns if c not in ["Label", "Attack_Family", "Day", "Destination Port"]]

    families = ["PortScan", "DoS/DDoS"]
    seeds = [0, 1, 2]
    n_seed = 10000

    # ==========================================
    # 4A: DIAGNOSTIC (c = 5%)
    # ==========================================
    print("\n" + "=" * 90)
    print("STAGE 4a: STAGE-1 DETECTOR RECONSTRUCTION DIAGNOSTIC (c = 5%)")
    print("=" * 90)
    cached_4a = load_existing_results(csv_4a)

    for fam in families:
        for seed in seeds:
            for model_type in ["ae", "pca"]:
                m_label = "Stage-1 AE" if model_type == "ae" else "Stage-1 PCA"
                if not cached_4a.empty and "family" in cached_4a.columns:
                    match = cached_4a[
                        (cached_4a["family"] == fam)
                        & (cached_4a["seed"] == seed)
                        & (cached_4a["model"] == m_label)
                    ]
                    if not match.empty:
                        print(f"  [4a CACHED] {fam:<9} | s={seed} | {m_label}")
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

                S_indices, S_fit, S_cal = draw_trusted_seed_set(
                    df,
                    seed=seed,
                    n_seed=n_seed,
                    n_train_max=200000,
                    val_benign_ratio=0.2,
                    test_benign_ratio=0.2,
                )

                X_pool_raw = df.loc[split.train_indices, feature_cols]
                X_seed_fit_raw = df.loc[S_fit, feature_cols]
                X_seed_cal_raw = df.loc[S_cal, feature_cols]

                # Preprocessor fit on S_fit ONLY
                prep = TabularPreprocessor(skew_threshold=3.0, clip_value=10.0)
                prep.fit(X_seed_fit_raw)

                X_s_fit = prep.transform(X_seed_fit_raw)
                X_s_cal = prep.transform(X_seed_cal_raw)
                X_pool = prep.transform(X_pool_raw)

                if model_type == "ae":
                    stg1_ae = MLPAutoencoder(
                        input_dim=X_s_fit.shape[1],
                        hidden_dims=(64, 32, 16),
                        bottleneck_dim=16,
                        seed=seed,
                    )
                    trained_m, _ = train_autoencoder(
                        stg1_ae,
                        X_train=X_s_fit,
                        X_val_benign=X_s_cal,
                        seed=seed,
                        max_epochs=30,
                        batch_size=256,
                        lr=0.001,
                        patience=3,
                        trim_frac=0.0,
                        warmup_epochs=0,
                        verbose=False,
                    )
                    s_cal_scores = trained_m.compute_reconstruction_error(X_s_cal)
                    pool_scores = trained_m.compute_reconstruction_error(X_pool)
                else:
                    stg1_pca = PCAReconstructionBaseline(n_components=10, random_state=seed)
                    stg1_pca.fit(X_s_fit)
                    s_cal_scores = stg1_pca.score(X_s_cal)
                    pool_scores = stg1_pca.score(X_pool)

                train_fams = df.loc[split.train_indices, "Attack_Family"].to_numpy()
                is_attack_pool = (train_fams == fam)
                attack_scores = pool_scores[is_attack_pool]
                benign_pool_scores = pool_scores[~is_attack_pool]

                # Percentiles of S_cal
                p50 = float(np.percentile(s_cal_scores, 50.0))
                p90 = float(np.percentile(s_cal_scores, 90.0))
                p95 = float(np.percentile(s_cal_scores, 95.0))
                p99 = float(np.percentile(s_cal_scores, 99.0))

                frac_gt_p50 = float(np.mean(attack_scores > p50))
                frac_gt_p90 = float(np.mean(attack_scores > p90))
                frac_gt_p95 = float(np.mean(attack_scores > p95))
                frac_gt_p99 = float(np.mean(attack_scores > p99))

                pool_labels_binary = is_attack_pool.astype(int)
                sep_auroc = float(compute_auroc(pool_labels_binary, pool_scores))

                med_att = float(np.median(attack_scores))
                med_scal = float(np.median(s_cal_scores))
                med_ratio = round(med_att / max(1e-12, med_scal), 4)

                rec_4a = {
                    "family": fam,
                    "seed": seed,
                    "model": m_label,
                    "c": 5,
                    "n_seed": n_seed,
                    "frac_gt_p50": round(frac_gt_p50, 6),
                    "frac_gt_p90": round(frac_gt_p90, 6),
                    "frac_gt_p95": round(frac_gt_p95, 6),
                    "frac_gt_p99": round(frac_gt_p99, 6),
                    "pool_separation_auroc": round(sep_auroc, 6),
                    "median_attack_score": round(med_att, 6),
                    "median_scal_score": round(med_scal, 6),
                    "median_ratio_attack_to_scal": med_ratio,
                }
                save_single_result(rec_4a, csv_4a)
                cached_4a = load_existing_results(csv_4a)

                print(
                    f"  [4a DONE] {fam:<9} | s={seed} | {m_label:<11} | "
                    f"Sep AUROC: {sep_auroc:.4f} | Att > p90: {frac_gt_p90*100:.1f}% | "
                    f"Att > p99: {frac_gt_p99*100:.1f}% | Ratio: {med_ratio:.2f}x"
                )

    # ==========================================
    # 4B: SENSITIVITY (q in {0.03, 0.05}, reuse q=0.01)
    # ==========================================
    print("\n" + "=" * 90)
    print("STAGE 4b: POT RISK PARAMETER SENSITIVITY (q in {0.03, 0.05}, reusing q=0.01)")
    print("=" * 90)

    # First, import existing q=0.01 rows if present
    cached_4b = load_existing_results(csv_4b)
    if existing_sanit_csv.exists():
        existing_df = pd.read_csv(existing_sanit_csv)
        sub_01 = existing_df[
            (existing_df["setting"] == "Setting B")
            & (existing_df["c"].isin([0, 5]))
            & (existing_df["n_seed"] == 10000)
            & (existing_df["family"].isin(families))
        ].copy()

        for _, r in sub_01.iterrows():
            m_name = r["model"]
            if cached_4b.empty or not (
                (cached_4b["family"] == r["family"])
                & (cached_4b["c"] == int(r["c"]))
                & (cached_4b["seed"] == int(r["seed"]))
                & (cached_4b["model"] == m_name)
                & (cached_4b["q"] == 0.01)
            ).any():
                rec_01 = {
                    "family": r["family"],
                    "c": int(r["c"]),
                    "seed": int(r["seed"]),
                    "model": m_name,
                    "q": 0.01,
                    "auroc": float(r["auroc"]),
                    "fpr_at_95_tpr": float(r["fpr_at_95_tpr"]),
                    "frac_attack_removed": float(r.get("frac_attack_removed", 0.0)),
                    "frac_benign_removed": float(r.get("frac_benign_removed", 0.0)),
                }
                save_single_result(rec_01, csv_4b)
        cached_4b = load_existing_results(csv_4b)

    # Now evaluate q in {0.03, 0.05}
    q_levels = [0.03, 0.05]
    c_levels = [0, 5]

    for q in q_levels:
        for fam in families:
            for c_pct in c_levels:
                c_val = c_pct / 100.0
                for seed in seeds:
                    for model_type in ["ae", "pca"]:
                        m_name = "Sanitized AE" if model_type == "ae" else "Sanitized PCA"
                        if not cached_4b.empty and "q" in cached_4b.columns:
                            match = cached_4b[
                                (cached_4b["family"] == fam)
                                & (cached_4b["c"] == c_pct)
                                & (cached_4b["seed"] == seed)
                                & (cached_4b["model"] == m_name)
                                & (cached_4b["q"] == q)
                            ]
                            if not match.empty:
                                print(f"  [4b CACHED] q={q:.2f} | {fam:<9} | c={c_pct:02d}% | s={seed} | {m_name}")
                                continue

                        split = generate_p4_setting_b_splits(
                            df,
                            seed=seed,
                            family=fam,
                            contamination_rate=c_val,
                            n_train_total=200000,
                            val_benign_ratio=0.2,
                            test_benign_ratio=0.2,
                        )

                        S_indices, S_fit, S_cal = draw_trusted_seed_set(
                            df,
                            seed=seed,
                            n_seed=n_seed,
                            n_train_max=200000,
                            val_benign_ratio=0.2,
                            test_benign_ratio=0.2,
                        )

                        X_pool_raw = df.loc[split.train_indices, feature_cols]
                        X_seed_fit_raw = df.loc[S_fit, feature_cols]
                        X_seed_cal_raw = df.loc[S_cal, feature_cols]

                        keep_mask, drop_mask, pot_th, _, _ = filter_pool_with_seed(
                            X_pool_raw=X_pool_raw,
                            X_seed_fit_raw=X_seed_fit_raw,
                            X_seed_cal_raw=X_seed_cal_raw,
                            model_type=model_type,
                            q=q,
                            seed=seed,
                            skew_threshold=3.0,
                            clip_value=10.0,
                            max_epochs=30,
                            batch_size=256,
                            lr=0.001,
                            patience=3,
                            verbose=False,
                        )

                        train_fams = df.loc[split.train_indices, "Attack_Family"].to_numpy()
                        is_attack_pool = (train_fams == fam)
                        is_benign_pool = ~is_attack_pool
                        n_att = int(np.sum(is_attack_pool))
                        n_ben = int(np.sum(is_benign_pool))

                        frac_att_rem = float(np.sum(drop_mask & is_attack_pool) / n_att) if n_att > 0 else 0.0
                        frac_ben_rem = float(np.sum(drop_mask & is_benign_pool) / n_ben) if n_ben > 0 else 0.0

                        kept_pool_idx = split.train_indices[keep_mask]
                        X_val_df = df.loc[split.val_indices, feature_cols]
                        X_test_df = df.loc[split.test_indices, feature_cols]
                        y_test = (df.loc[split.test_indices, "Attack_Family"] == fam).astype(int).to_numpy()

                        if model_type == "ae":
                            stg2_idx = np.concatenate([kept_pool_idx, S_indices])
                            X_tr_raw = df.loc[stg2_idx, feature_cols]
                            prep2 = TabularPreprocessor(skew_threshold=3.0, clip_value=10.0)
                            prep2.fit(X_tr_raw)

                            X_tr = prep2.transform(X_tr_raw)
                            X_v = prep2.transform(X_val_df)
                            X_te = prep2.transform(X_test_df)

                            fin_ae = MLPAutoencoder(
                                input_dim=X_tr.shape[1],
                                hidden_dims=(64, 32, 16),
                                bottleneck_dim=16,
                                seed=seed,
                            )
                            tr_ae, _ = train_autoencoder(
                                fin_ae,
                                X_train=X_tr,
                                X_val_benign=X_v,
                                seed=seed,
                                max_epochs=30,
                                batch_size=256,
                                lr=0.001,
                                patience=3,
                                trim_frac=0.0,
                                warmup_epochs=0,
                                verbose=False,
                            )
                            t_scores = tr_ae.compute_reconstruction_error(X_te)
                        else:
                            X_tr_raw = df.loc[kept_pool_idx, feature_cols]
                            prep2 = TabularPreprocessor(skew_threshold=3.0, clip_value=10.0)
                            prep2.fit(X_tr_raw)

                            X_tr = prep2.transform(X_tr_raw)
                            X_te = prep2.transform(X_test_df)

                            fin_pca = PCAReconstructionBaseline(n_components=10, random_state=seed)
                            fin_pca.fit(X_tr)
                            t_scores = fin_pca.score(X_te)

                        m = compute_all_metrics(y_test, t_scores)

                        rec_4b = {
                            "family": fam,
                            "c": c_pct,
                            "seed": seed,
                            "model": m_name,
                            "q": q,
                            "auroc": round(float(m["auroc"]), 6),
                            "fpr_at_95_tpr": round(float(m["fpr_at_95_tpr"]), 6),
                            "frac_attack_removed": round(frac_att_rem, 6),
                            "frac_benign_removed": round(frac_ben_rem, 6),
                        }
                        save_single_result(rec_4b, csv_4b)
                        cached_4b = load_existing_results(csv_4b)

                        print(
                            f"  [4b DONE] q={q:.2f} | {fam:<9} | c={c_pct:02d}% | s={seed} | {m_name:<13} | "
                            f"AUROC: {m['auroc']:.4f} | FPR@95: {m['fpr_at_95_tpr']:.4f} | "
                            f"Att Rem: {frac_att_rem*100:.1f}% | Ben Rem: {frac_ben_rem*100:.1f}%"
                        )

    print("\nStage 4 finished successfully.")


if __name__ == "__main__":
    run_stage4()
