"""Protocol P4 Redo: Two Contamination Settings, Seeds 0, 1, 2.

DEV DATA, NOT FOR REPORTED RESULTS
Rules:
- max_threads = 4 everywhere
- Sequential execution: one job at a time
- Order: all conditions for seed 0, then seed 1, then seed 2
- Resumable: writes every run to disk immediately, skips existing runs
- c=0 AE trained once per seed and evaluated on all families/settings
- Setting A: BruteForce, PortScan, WebAttack (contamination from other families)
- Setting B: PortScan, DoS/DDoS (contamination from F itself, 50/50 contam/test split)
- c in {0, 2, 5, 10}%, N = 200,000 fixed
- Models: PCA, Isolation Forest, Plain AE, Trimmed AE (trim_frac=0.05, warmup=2)
- Report best_epoch and stopped_epoch for every AE run, note if best_epoch == 30
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

torch.set_num_threads(4)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from baselines.classical import IsolationForestBaseline, PCAReconstructionBaseline
from data.preprocess import TabularPreprocessor
from eval.metrics import compute_all_metrics
from models.autoencoders import (
    MLPAutoencoder,
    TrimmedAutoencoder,
    train_autoencoder,
    train_trimmed_autoencoder,
)
from splits.protocols import (
    generate_p4_contamination_splits,
    generate_p4_setting_b_splits,
)


def load_existing_results(csv_path: Path) -> Tuple[List[Dict[str, Any]], set]:
    if not csv_path.exists():
        return [], set()
    try:
        df = pd.read_csv(csv_path)
        records = df.to_dict(orient="records")
        existing_keys = {
            (r["setting"], r["family"], int(r["c"]), int(r["seed"]), r["model"])
            for r in records
        }
        return records, existing_keys
    except Exception as e:
        print(f"[Warning] Could not load existing CSV: {e}")
        return [], set()


def append_result(csv_path: Path, json_path: Path, record: Dict[str, Any], records: List[Dict[str, Any]]):
    records.append(record)
    df_row = pd.DataFrame([record])
    if not csv_path.exists():
        df_row.to_csv(csv_path, index=False)
    else:
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def main():
    status_tag = "DEV DATA, NOT FOR REPORTED RESULTS"
    max_threads = 4
    n_train_total = 200000
    seeds = [0, 1, 2]
    c_levels = [0, 2, 5, 10]
    c_rates = [c / 100.0 for c in c_levels]

    setting_a_families = ["BruteForce", "PortScan", "WebAttack"]
    setting_b_families = ["PortScan", "DoS/DDoS"]

    models_order = ["PCA", "Isolation Forest", "Plain AE", "Trimmed AE"]

    results_dir = REPO_ROOT / "results"
    figures_dir = REPO_ROOT / "results" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "p4_redo_seeds012.csv"
    json_path = results_dir / "p4_redo_seeds012.json"

    records, existing_keys = load_existing_results(csv_path)

    print("=" * 105)
    print(f"[{status_tag}]")
    print("PROTOCOL P4 REDO: Setting A & Setting B across Seeds 0, 1, 2")
    print(f"N = {n_train_total:,} | Max Threads: {max_threads} | Resumable: {len(existing_keys)} runs already cached")
    print("=" * 105)

    data_path = REPO_ROOT / "data" / "processed" / "cicids2017_cleaned_dev.parquet"
    print(f"Loading cleaned dataset from {data_path} ...")
    t0 = time.time()
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} rows in {time.time() - t0:.2f}s.\n")

    feature_cols = [
        c for c in df.columns
        if c not in ["Label", "Attack_Family", "Day", "Destination Port"]
    ]

    total_conditions = (
        len(seeds) * len(c_levels) * len(setting_a_families) +
        len(seeds) * len(c_levels) * len(setting_b_families)
    ) * len(models_order)
    print(f"Total Model Evaluation Runs Planned: {total_conditions} runs\n")

    run_counter = len(existing_keys)

    # Outer loop: seed 0, then seed 1, then seed 2
    for seed in seeds:
        print("\n" + "#" * 105)
        print(f"### EXECUTING SEED: {seed} ###")
        print("#" * 105)

        # -------------------------------------------------------------
        # STEP 1: Train c=0 Plain AE and Trimmed AE once for this seed
        # -------------------------------------------------------------
        print(f"\n--- [Seed {seed}] Preparing c=0 Shared Base Training Set (200,000 clean benign rows) ---")
        base_c0_split = generate_p4_contamination_splits(
            df,
            seed=seed,
            lofo_family="BruteForce",  # c=0 train indices are identical across all families
            contamination_rate=0.0,
            n_train_total=n_train_total,
            include_test_family=False,
            val_benign_ratio=0.2,
            test_benign_ratio=0.2,
        )

        base_train_df = df.loc[base_c0_split.train_indices, feature_cols]
        base_prep = TabularPreprocessor(skew_threshold=3.0, clip_value=10.0)
        base_prep.fit(base_train_df)
        X_base_train = base_prep.transform(base_train_df)
        X_base_val = base_prep.transform(df.loc[base_c0_split.val_indices, feature_cols])

        # Train shared c=0 Plain AE
        c0_plain_ae_path = results_dir / f"models_p4_c0_plain_ae_seed{seed}.pt"
        c0_plain_ae = MLPAutoencoder(
            input_dim=X_base_train.shape[1],
            hidden_dims=(64, 32, 16),
            bottleneck_dim=16,
            seed=seed,
        )
        print(f"  [Seed {seed}] Training Shared c=0 Plain AE (200k fit, 30 max epochs) ...")
        t_c0_p0 = time.time()
        c0_plain_ae, hist_c0_p = train_autoencoder(
            c0_plain_ae,
            X_train=X_base_train,
            X_val_benign=X_base_val,
            seed=seed,
            max_epochs=30,
            batch_size=256,
            lr=0.001,
            patience=3,
            trim_frac=0.0,
            warmup_epochs=0,
            verbose=False,
        )
        t_c0_plain = time.time() - t_c0_p0
        best_ep_c0_p = hist_c0_p[int(np.argmin([h["val_recon_loss"] for h in hist_c0_p]))]["epoch"]
        stop_ep_c0_p = hist_c0_p[-1]["epoch"]

        # Train shared c=0 Trimmed AE
        c0_trimmed_ae = TrimmedAutoencoder(
            input_dim=X_base_train.shape[1],
            hidden_dims=(64, 32, 16),
            bottleneck_dim=16,
            seed=seed,
        )
        print(f"  [Seed {seed}] Training Shared c=0 Trimmed AE (200k fit, trim_frac=0.05, warmup=2) ...")
        t_c0_t0 = time.time()
        c0_trimmed_ae, hist_c0_t = train_trimmed_autoencoder(
            c0_trimmed_ae,
            X_train=X_base_train,
            X_val_benign=X_base_val,
            seed=seed,
            max_epochs=30,
            batch_size=256,
            lr=0.001,
            patience=3,
            trim_frac=0.05,
            warmup_epochs=2,
            verbose=False,
        )
        t_c0_trim = time.time() - t_c0_t0
        best_ep_c0_t = hist_c0_t[int(np.argmin([h["val_recon_loss"] for h in hist_c0_t]))]["epoch"]
        stop_ep_c0_t = hist_c0_t[-1]["epoch"]

        # -------------------------------------------------------------
        # STEP 2: Iterate over Settings, Families, and Contamination Levels
        # -------------------------------------------------------------
        settings_configs = [
            ("Setting A", setting_a_families),
            ("Setting B", setting_b_families),
        ]

        for setting_name, families in settings_configs:
            print("\n" + "=" * 90)
            print(f"--- [Seed {seed}] {setting_name} ---")
            print("=" * 90)

            for family in families:
                for c_pct in c_levels:
                    c = c_pct / 100.0

                    # 1. Generate Split
                    if setting_name == "Setting A":
                        split = generate_p4_contamination_splits(
                            df,
                            seed=seed,
                            lofo_family=family,
                            contamination_rate=c,
                            n_train_total=n_train_total,
                            include_test_family=False,
                            val_benign_ratio=0.2,
                            test_benign_ratio=0.2,
                        )
                        # Test: all benign test + all held-out family test
                        benign_test_indices = df.index[
                            (df["Attack_Family"] == "Benign") & (df.index.isin(split.test_indices))
                        ].to_numpy()
                        attack_test_indices = df.index[
                            (df["Attack_Family"] == family) & (df.index.isin(split.test_indices))
                        ].to_numpy()
                    else:
                        split = generate_p4_setting_b_splits(
                            df,
                            seed=seed,
                            family=family,
                            contamination_rate=c,
                            n_train_total=n_train_total,
                            val_benign_ratio=0.2,
                            test_benign_ratio=0.2,
                        )
                        benign_test_indices = df.index[
                            (df["Attack_Family"] == "Benign") & (df.index.isin(split.test_indices))
                        ].to_numpy()
                        attack_test_indices = df.index[
                            (df["Attack_Family"] == family) & (df.index.isin(split.test_indices))
                        ].to_numpy()

                    eval_indices = np.sort(np.concatenate([benign_test_indices, attack_test_indices]))
                    X_test_df = df.loc[eval_indices, feature_cols]
                    y_test = (df.loc[eval_indices, "Attack_Family"] == family).astype(int).to_numpy()

                    n_benign_test = len(benign_test_indices)
                    n_attack_test = len(attack_test_indices)
                    attack_prev = n_attack_test / (n_benign_test + n_attack_test)

                    # 2. Fit Preprocessor (if c=0, reuse base_prep; else fit on contaminated train)
                    if c_pct == 0:
                        preprocessor = base_prep
                        X_train = X_base_train
                        X_val = X_base_val
                    else:
                        X_train_df = df.loc[split.train_indices, feature_cols]
                        preprocessor = TabularPreprocessor(skew_threshold=3.0, clip_value=10.0)
                        preprocessor.fit(X_train_df)
                        X_train = preprocessor.transform(X_train_df)
                        X_val = preprocessor.transform(df.loc[split.val_indices, feature_cols])

                    X_test = preprocessor.transform(X_test_df)

                    # ---------------------------------------------------------
                    # Model 1: PCA
                    # ---------------------------------------------------------
                    pca_key = (setting_name, family, c_pct, seed, "PCA")
                    if pca_key in existing_keys:
                        pass
                    else:
                        t_m0 = time.time()
                        pca_model = PCAReconstructionBaseline(
                            n_components=10,
                            random_state=seed,
                        )
                        pca_model.fit(X_train)
                        scores_pca = pca_model.score(X_test)
                        t_pca = time.time() - t_m0
                        m_pca = compute_all_metrics(y_test, scores_pca)

                        run_counter += 1
                        rec_pca = {
                            "setting": setting_name,
                            "family": family,
                            "c": c_pct,
                            "seed": seed,
                            "model": "PCA",
                            "auroc": round(m_pca["auroc"], 6),
                            "fpr_at_95_tpr": round(m_pca["fpr_at_95_tpr"], 6),
                            "tpr_at_1_fpr": round(m_pca["tpr_at_1_fpr"], 6),
                            "attack_prevalence": round(attack_prev, 8),
                            "best_epoch": None,
                            "stopped_epoch": None,
                            "runtime": round(t_pca, 2),
                        }
                        append_result(csv_path, json_path, rec_pca, records)
                        existing_keys.add(pca_key)
                        print(
                            f"  [{run_counter}/{total_conditions}] [DONE] {setting_name:<9} | Seed {seed} | "
                            f"Family: {family:<10} | c={c_pct:02d}% | Model: {'PCA':<16} | "
                            f"AUROC: {m_pca['auroc']:.4f} | FPR@95%: {m_pca['fpr_at_95_tpr']:.4f} | Time: {t_pca:.2f}s"
                        )

                    # ---------------------------------------------------------
                    # Model 2: Isolation Forest
                    # ---------------------------------------------------------
                    if_key = (setting_name, family, c_pct, seed, "Isolation Forest")
                    if if_key in existing_keys:
                        pass
                    else:
                        t_m0 = time.time()
                        if_model = IsolationForestBaseline(
                            n_estimators=100,
                            random_state=seed,
                            n_jobs=max_threads,
                        )
                        if_model.fit(X_train)
                        scores_if = if_model.score(X_test)
                        t_if = time.time() - t_m0
                        m_if = compute_all_metrics(y_test, scores_if)

                        run_counter += 1
                        rec_if = {
                            "setting": setting_name,
                            "family": family,
                            "c": c_pct,
                            "seed": seed,
                            "model": "Isolation Forest",
                            "auroc": round(m_if["auroc"], 6),
                            "fpr_at_95_tpr": round(m_if["fpr_at_95_tpr"], 6),
                            "tpr_at_1_fpr": round(m_if["tpr_at_1_fpr"], 6),
                            "attack_prevalence": round(attack_prev, 8),
                            "best_epoch": None,
                            "stopped_epoch": None,
                            "runtime": round(t_if, 2),
                        }
                        append_result(csv_path, json_path, rec_if, records)
                        existing_keys.add(if_key)
                        print(
                            f"  [{run_counter}/{total_conditions}] [DONE] {setting_name:<9} | Seed {seed} | "
                            f"Family: {family:<10} | c={c_pct:02d}% | Model: {'Isolation Forest':<16} | "
                            f"AUROC: {m_if['auroc']:.4f} | FPR@95%: {m_if['fpr_at_95_tpr']:.4f} | Time: {t_if:.2f}s"
                        )

                    # ---------------------------------------------------------
                    # Model 3: Plain AE
                    # ---------------------------------------------------------
                    ae_key = (setting_name, family, c_pct, seed, "Plain AE")
                    if ae_key in existing_keys:
                        pass
                    else:
                        t_m0 = time.time()
                        if c_pct == 0:
                            # Reuse shared c=0 Plain AE
                            model_plain = c0_plain_ae
                            best_ep = best_ep_c0_p
                            stop_ep = stop_ep_c0_p
                        else:
                            plain_ae = MLPAutoencoder(
                                input_dim=X_train.shape[1],
                                hidden_dims=(64, 32, 16),
                                bottleneck_dim=16,
                                seed=seed,
                            )
                            log_path = results_dir / f"loss_p4_{setting_name.replace(' ', '_')}_{family}_c{c_pct}_s{seed}_plain.csv"
                            model_plain, hist = train_autoencoder(
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
                                log_csv_path=log_path,
                                verbose=False,
                            )
                            best_ep = hist[int(np.argmin([h["val_recon_loss"] for h in hist]))]["epoch"]
                            stop_ep = hist[-1]["epoch"]

                        scores_plain = model_plain.compute_reconstruction_error(X_test)
                        t_plain = time.time() - t_m0
                        m_plain = compute_all_metrics(y_test, scores_plain)

                        cap_note = " [CAP=30 REACHED]" if best_ep == 30 else ""
                        run_counter += 1
                        rec_plain = {
                            "setting": setting_name,
                            "family": family,
                            "c": c_pct,
                            "seed": seed,
                            "model": "Plain AE",
                            "auroc": round(m_plain["auroc"], 6),
                            "fpr_at_95_tpr": round(m_plain["fpr_at_95_tpr"], 6),
                            "tpr_at_1_fpr": round(m_plain["tpr_at_1_fpr"], 6),
                            "attack_prevalence": round(attack_prev, 8),
                            "best_epoch": best_ep,
                            "stopped_epoch": stop_ep,
                            "runtime": round(t_plain, 2),
                        }
                        append_result(csv_path, json_path, rec_plain, records)
                        existing_keys.add(ae_key)
                        print(
                            f"  [{run_counter}/{total_conditions}] [DONE] {setting_name:<9} | Seed {seed} | "
                            f"Family: {family:<10} | c={c_pct:02d}% | Model: {'Plain AE':<16} | "
                            f"AUROC: {m_plain['auroc']:.4f} | FPR@95%: {m_plain['fpr_at_95_tpr']:.4f} | "
                            f"Epochs: best={best_ep}, stopped={stop_ep}{cap_note} | Time: {t_plain:.2f}s"
                        )

                    # ---------------------------------------------------------
                    # Model 4: Trimmed-Loss AE (trim_frac = 5%)
                    # ---------------------------------------------------------
                    trim_key = (setting_name, family, c_pct, seed, "Trimmed AE")
                    if trim_key in existing_keys:
                        pass
                    else:
                        t_m0 = time.time()
                        if c_pct == 0:
                            # Reuse shared c=0 Trimmed AE
                            model_trim = c0_trimmed_ae
                            best_ep_t = best_ep_c0_t
                            stop_ep_t = stop_ep_c0_t
                        else:
                            trimmed_ae = TrimmedAutoencoder(
                                input_dim=X_train.shape[1],
                                hidden_dims=(64, 32, 16),
                                bottleneck_dim=16,
                                seed=seed,
                            )
                            log_path = results_dir / f"loss_p4_{setting_name.replace(' ', '_')}_{family}_c{c_pct}_s{seed}_trimmed.csv"
                            model_trim, hist_t = train_trimmed_autoencoder(
                                trimmed_ae,
                                X_train=X_train,
                                X_val_benign=X_val,
                                seed=seed,
                                max_epochs=30,
                                batch_size=256,
                                lr=0.001,
                                patience=3,
                                trim_frac=0.05,
                                warmup_epochs=2,
                                log_csv_path=log_path,
                                verbose=False,
                            )
                            best_ep_t = hist_t[int(np.argmin([h["val_recon_loss"] for h in hist_t]))]["epoch"]
                            stop_ep_t = hist_t[-1]["epoch"]

                        scores_trim = model_trim.compute_reconstruction_error(X_test)
                        t_trim = time.time() - t_m0
                        m_trim = compute_all_metrics(y_test, scores_trim)

                        cap_note_t = " [CAP=30 REACHED]" if best_ep_t == 30 else ""
                        run_counter += 1
                        rec_trim = {
                            "setting": setting_name,
                            "family": family,
                            "c": c_pct,
                            "seed": seed,
                            "model": "Trimmed AE",
                            "auroc": round(m_trim["auroc"], 6),
                            "fpr_at_95_tpr": round(m_trim["fpr_at_95_tpr"], 6),
                            "tpr_at_1_fpr": round(m_trim["tpr_at_1_fpr"], 6),
                            "attack_prevalence": round(attack_prev, 8),
                            "best_epoch": best_ep_t,
                            "stopped_epoch": stop_ep_t,
                            "runtime": round(t_trim, 2),
                        }
                        append_result(csv_path, json_path, rec_trim, records)
                        existing_keys.add(trim_key)
                        print(
                            f"  [{run_counter}/{total_conditions}] [DONE] {setting_name:<9} | Seed {seed} | "
                            f"Family: {family:<10} | c={c_pct:02d}% | Model: {'Trimmed AE':<16} | "
                            f"AUROC: {m_trim['auroc']:.4f} | FPR@95%: {m_trim['fpr_at_95_tpr']:.4f} | "
                            f"Epochs: best={best_ep_t}, stopped={stop_ep_t}{cap_note_t} | Time: {t_trim:.2f}s"
                        )

    # -----------------------------------------------------------------
    # PART E: Aggregated Reporting & Figures
    # -----------------------------------------------------------------
    print("\n" + "=" * 105)
    print(f"[{status_tag}]")
    print("ALL RUNS COMPLETE. GENERATING AGGREGATED TABLES AND PLOTS...")
    print("=" * 105)

    all_df = pd.DataFrame(records)

    # Helper function for mean +- std tables
    def build_mean_std_table(metric_col: str) -> pd.DataFrame:
        rows = []
        for s_name in ["Setting A", "Setting B"]:
            fams = setting_a_families if s_name == "Setting A" else setting_b_families
            for fam in fams:
                for c_val in c_levels:
                    sub = all_df[
                        (all_df["setting"] == s_name) &
                        (all_df["family"] == fam) &
                        (all_df["c"] == c_val)
                    ]
                    if sub.empty:
                        continue
                    r_dict = {
                        "Setting": s_name,
                        "Family": fam,
                        "c (%)": f"{c_val}%",
                    }
                    for m in models_order:
                        m_sub = sub[sub["model"] == m][metric_col]
                        if len(m_sub) > 0:
                            mean_v = m_sub.mean()
                            std_v = m_sub.std(ddof=1) if len(m_sub) > 1 else 0.0
                            r_dict[m] = f"{mean_v:.4f} ± {std_v:.4f}"
                        else:
                            r_dict[m] = "N/A"
                    rows.append(r_dict)
        return pd.DataFrame(rows)

    print("\n" + "=" * 115)
    print(f"[{status_tag}]")
    print("TABLE 1: AUROC (Mean ± Std over 3 Seeds: 0, 1, 2) [Higher = Better]")
    print("=" * 115)
    df_auroc = build_mean_std_table("auroc")
    print(df_auroc.to_string(index=False))

    print("\n" + "=" * 115)
    print(f"[{status_tag}]")
    print("TABLE 2: FPR@95%TPR (Mean ± Std over 3 Seeds: 0, 1, 2) [Lower = Better]")
    print("=" * 115)
    df_fpr = build_mean_std_table("fpr_at_95_tpr")
    print(df_fpr.to_string(index=False))

    # Paired Comparison Table: Trimmed AE minus Plain AE
    print("\n" + "=" * 115)
    print(f"[{status_tag}]")
    print("TABLE 3: PAIRED COMPARISON (Trimmed AE minus Plain AE, AUROC Diff over 3 Seeds)")
    print("=" * 115)
    paired_rows = []
    for s_name in ["Setting A", "Setting B"]:
        fams = setting_a_families if s_name == "Setting A" else setting_b_families
        for fam in fams:
            for c_val in c_levels:
                diffs = []
                wins = 0
                for s in seeds:
                    p_val = all_df[
                        (all_df["setting"] == s_name) &
                        (all_df["family"] == fam) &
                        (all_df["c"] == c_val) &
                        (all_df["seed"] == s) &
                        (all_df["model"] == "Plain AE")
                    ]["auroc"].values
                    t_val = all_df[
                        (all_df["setting"] == s_name) &
                        (all_df["family"] == fam) &
                        (all_df["c"] == c_val) &
                        (all_df["seed"] == s) &
                        (all_df["model"] == "Trimmed AE")
                    ]["auroc"].values
                    if len(p_val) > 0 and len(t_val) > 0:
                        d = t_val[0] - p_val[0]
                        diffs.append(d)
                        if d > 0:
                            wins += 1

                if diffs:
                    mean_diff = np.mean(diffs)
                    std_diff = np.std(diffs, ddof=1) if len(diffs) > 1 else 0.0
                    verdict = "within noise" if abs(mean_diff) < std_diff else ("Trimmed Better" if mean_diff > 0 else "Plain Better")
                    paired_rows.append({
                        "Setting": s_name,
                        "Family": fam,
                        "c (%)": f"{c_val}%",
                        "Mean Diff (Trim - Plain)": f"{mean_diff:+.4f}",
                        "Seed Std": f"{std_diff:.4f}",
                        "Trimmed Wins": f"{wins}/{len(diffs)}",
                        "Status": verdict,
                    })

    df_paired = pd.DataFrame(paired_rows)
    print(df_paired.to_string(index=False))

    # Generate Figures: Line plots of AUROC vs c with error bars
    print("\n" + "=" * 105)
    print("SAVING AUROC vs CONTAMINATION LINE PLOTS (results/figures/)")
    print("=" * 105)

    color_map = {
        "PCA": "#2ca02c",
        "Isolation Forest": "#ff7f0e",
        "Plain AE": "#1f77b4",
        "Trimmed AE": "#d62728",
    }
    marker_map = {
        "PCA": "s",
        "Isolation Forest": "^",
        "Plain AE": "o",
        "Trimmed AE": "D",
    }

    for s_name in ["Setting A", "Setting B"]:
        fams = setting_a_families if s_name == "Setting A" else setting_b_families
        for fam in fams:
            fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
            for m in models_order:
                means = []
                stds = []
                for c_val in c_levels:
                    sub = all_df[
                        (all_df["setting"] == s_name) &
                        (all_df["family"] == fam) &
                        (all_df["c"] == c_val) &
                        (all_df["model"] == m)
                    ]["auroc"]
                    means.append(sub.mean() if len(sub) > 0 else np.nan)
                    stds.append(sub.std(ddof=1) if len(sub) > 1 else 0.0)

                ax.errorbar(
                    c_levels,
                    means,
                    yerr=stds,
                    label=m,
                    color=color_map[m],
                    marker=marker_map[m],
                    linewidth=2.0,
                    capsize=4,
                    capthick=1.5,
                )

            clean_s = s_name.lower().replace(" ", "_")
            ax.set_title(
                f"[{status_tag}]\nProtocol P4 {s_name}: AUROC vs Contamination Rate ({fam})\n"
                f"Mean ± 1 Std across Seeds 0, 1, 2 (N={n_train_total:,})",
                fontsize=11,
                fontweight="bold",
            )
            ax.set_xlabel("Contamination Rate c (%)", fontsize=11, fontweight="bold")
            ax.set_ylabel("AUROC", fontsize=11, fontweight="bold")
            ax.set_xticks(c_levels)
            ax.set_xticklabels([f"{c}%" for c in c_levels])
            ax.set_ylim([0.45, 1.02])
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(loc="lower right", framealpha=0.9)
            plt.tight_layout()

            fig_path = figures_dir / f"p4_{clean_s}_{fam.replace('/', '_')}_auroc.png"
            fig.savefig(fig_path)
            plt.close(fig)
            print(f"Saved figure: {fig_path}")

    # Total runtimes summary
    print("\n" + "=" * 105)
    print("TOTAL EXECUTION RUNTIMES PER MODEL (seconds across all finished runs)")
    print("=" * 105)
    runtime_summary = all_df.groupby("model")["runtime"].agg(["count", "mean", "sum"]).reset_index()
    runtime_summary.columns = ["Model", "Runs", "Mean Runtime (s)", "Total Runtime (s)"]
    print(runtime_summary.to_string(index=False))


if __name__ == "__main__":
    main()
