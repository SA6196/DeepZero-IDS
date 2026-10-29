"""Protocol P4 (Contamination Injection Robustness Experiment) Runner.

Source of Truth: SPEC.md Section 3, 4, 5 & User Request PART A, PART B, PART C.
Status: DEV DATA, NOT FOR REPORTED RESULTS.

Rules:
1. TRAIN size N = 200,000 (fixed multiple of 100) identical for every contamination level c in {0, 1, 2, 5, 10}%.
2. Benign rows nested across levels.
3. Contamination rows drawn from attack families OTHER than the held-out family F (F never in TRAIN/VAL).
4. Held-out families F: BruteForce, PortScan, WebAttack.
5. VAL-BENIGN stays clean (noted as limitation).
6. TEST = all benign test rows (~427k) + all rows of family F.
7. Fit preprocessor on contaminated TRAIN only.
8. Thermal Limit: max_threads = 4 everywhere (PyTorch, scikit-learn, OMP, MKL).
9. One job at a time; progress line after every finished run.
"""

from __future__ import annotations

import os

# Thermal limit: clamp CPU threads to 4 before native BLAS/MKL libraries load
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

# Set PyTorch thread limit immediately
torch.set_num_threads(4)

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure utf-8 output and immediate line flushing
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from baselines.classical import (
    IsolationForestBaseline,
    PCAReconstructionBaseline,
)
from data.preprocess import TabularPreprocessor
from eval.metrics import compute_all_metrics
from models.autoencoders import (
    MLPAutoencoder,
    TrimmedAutoencoder,
    train_autoencoder,
    train_trimmed_autoencoder,
)
from splits.protocols import generate_p4_contamination_splits


def parse_args():
    parser = argparse.ArgumentParser(description="Run Protocol P4 Contamination Experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/contamination_p4_cicids2017.yaml",
        help="Path to YAML experiment configuration.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["experiment"].get("seed", 0))
    status_tag = cfg["experiment"].get("status", "DEV DATA, NOT FOR REPORTED RESULTS")
    max_threads = int(cfg["experiment"].get("max_threads", 4))

    # Enforce thermal limits
    torch.set_num_threads(max_threads)
    os.environ["OMP_NUM_THREADS"] = str(max_threads)
    os.environ["MKL_NUM_THREADS"] = str(max_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(max_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(max_threads)

    print("=" * 105)
    print(f"[{status_tag}]")
    print(f"Protocol P4 Contamination Robustness Experiment | Seed: {seed} | Max Threads: {max_threads}")
    print("=" * 105)
    print("NOTE ON LIMITATION: VAL-BENIGN stays clean (per SPEC). In practical deployment,")
    print("validation data would also be contaminated. This is a controlled evaluation design.\n")

    # Load cleaned dataset
    data_path = Path(cfg["dataset"]["cleaned_path"])
    print(f"Loading cleaned dataset from: {data_path} ...")
    t0_data = time.time()
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} rows with {df.shape[1]} columns in {time.time() - t0_data:.2f}s.")

    feature_cols = [
        c
        for c in df.columns
        if c not in ["Label", "Attack_Family", "Day", "Destination Port"]
    ]

    target_families = cfg["protocol"].get("target_families", ["BruteForce", "PortScan", "WebAttack"])
    contamination_levels = [float(c) for c in cfg["protocol"].get("contamination_levels", [0.0, 0.01, 0.02, 0.05, 0.10])]
    n_train_total = int(cfg["protocol"].get("n_train_total", 200000))
    val_benign_ratio = float(cfg["protocol"].get("val_benign_ratio", 0.2))
    test_benign_ratio = float(cfg["protocol"].get("test_benign_ratio", 0.2))
    skew_thresh = float(cfg["preprocessing"].get("skew_threshold", 3.0))

    results_dir = Path(cfg["output"].get("results_dir", "results"))
    figures_dir = Path(cfg["output"].get("figures_dir", "results/figures"))
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []

    print(f"Target Families: {target_families}")
    print(f"Contamination Levels: {[f'{int(c*100)}%' for c in contamination_levels]}")
    print(f"Fixed TRAIN size N: {n_train_total:,} rows\n")

    total_runs = len(target_families) * len(contamination_levels) * 4
    current_run = 0

    # Grid search: family x contamination level
    for family in target_families:
        print("\n" + "=" * 90)
        print(f"Processing Target Held-Out Family: {family}")
        print("=" * 90)

        for c in contamination_levels:
            c_pct = int(round(c * 100))
            print(f"\n--- [Condition] Family: {family} | Contamination c = {c_pct}% ({c:.2f}) ---")

            # 1. Generate P4 split
            split = generate_p4_contamination_splits(
                df,
                seed=seed,
                lofo_family=family,
                contamination_rate=c,
                n_train_total=n_train_total,
                include_test_family=False,
                val_benign_ratio=val_benign_ratio,
                test_benign_ratio=test_benign_ratio,
            )

            n_benign_train = split.metadata["n_benign_train"]
            n_injected = split.metadata["n_injected_contam"]
            print(f"  TRAIN Composition: {n_benign_train:,} benign + {n_injected:,} attack (N={len(split.train_indices):,})")
            print(f"  VAL-BENIGN Size:   {len(split.val_indices):,} (clean)")
            print(f"  TEST Total:        {len(split.test_indices):,} ({split.metadata['n_family_attack_test']:,} {family})")

            # 2. Fit preprocessor strictly on contaminated TRAIN only
            X_train_df = df.loc[split.train_indices, feature_cols]
            preprocessor = TabularPreprocessor(skew_threshold=skew_thresh)
            t_prep0 = time.time()
            preprocessor.fit(X_train_df)
            t_prep = time.time() - t_prep0

            X_train = preprocessor.transform(X_train_df)
            X_val = preprocessor.transform(df.loc[split.val_indices, feature_cols])

            # Prepare test data: all benign test + all held-out family flows
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
            attack_prevalence = n_attack_test / (n_benign_test + n_attack_test)

            # ---------------------------------------------------------
            # Model 1: PCA Reconstruction
            # ---------------------------------------------------------
            current_run += 1
            t_start = time.time()
            pca_model = PCAReconstructionBaseline(
                n_components=cfg["models"]["pca"]["n_components"],
                random_state=cfg["models"]["pca"]["random_state"],
            )
            pca_model.fit(X_train)
            scores_pca = pca_model.score(X_test)
            t_pca = time.time() - t_start
            m_pca = compute_all_metrics(y_test, scores_pca)
            print(
                f"  [{current_run}/{total_runs}] [DONE] Family: {family:<10} | c={c_pct:02d}% | "
                f"Model: {'PCA':<18} | AUROC: {m_pca['auroc']:.4f} | "
                f"FPR@95%: {m_pca['fpr_at_95_tpr']:.4f} | TPR@1%: {m_pca['tpr_at_1_fpr']:.4f} | Time: {t_pca:.2f}s"
            )

            records.append({
                "family": family,
                "c": c_pct,
                "model": "PCA",
                "auroc": round(m_pca["auroc"], 6),
                "fpr_at_95_tpr": round(m_pca["fpr_at_95_tpr"], 6),
                "tpr_at_1_fpr": round(m_pca["tpr_at_1_fpr"], 6),
                "attack_prevalence": round(attack_prevalence, 8),
                "runtime": round(t_pca, 2),
            })

            # ---------------------------------------------------------
            # Model 2: Isolation Forest
            # ---------------------------------------------------------
            current_run += 1
            t_start = time.time()
            if_model = IsolationForestBaseline(
                n_estimators=cfg["models"]["isolation_forest"]["n_estimators"],
                random_state=cfg["models"]["isolation_forest"]["random_state"],
                n_jobs=max_threads,
            )
            if_model.fit(X_train)
            scores_if = if_model.score(X_test)
            t_if = time.time() - t_start
            m_if = compute_all_metrics(y_test, scores_if)
            print(
                f"  [{current_run}/{total_runs}] [DONE] Family: {family:<10} | c={c_pct:02d}% | "
                f"Model: {'Isolation Forest':<18} | AUROC: {m_if['auroc']:.4f} | "
                f"FPR@95%: {m_if['fpr_at_95_tpr']:.4f} | TPR@1%: {m_if['tpr_at_1_fpr']:.4f} | Time: {t_if:.2f}s"
            )

            records.append({
                "family": family,
                "c": c_pct,
                "model": "Isolation Forest",
                "auroc": round(m_if["auroc"], 6),
                "fpr_at_95_tpr": round(m_if["fpr_at_95_tpr"], 6),
                "tpr_at_1_fpr": round(m_if["tpr_at_1_fpr"], 6),
                "attack_prevalence": round(attack_prevalence, 8),
                "runtime": round(t_if, 2),
            })

            # ---------------------------------------------------------
            # Model 3: Plain MLP Autoencoder
            # ---------------------------------------------------------
            current_run += 1
            t_start = time.time()
            ae_cfg = cfg["models"]["autoencoder"]
            plain_ae = MLPAutoencoder(
                input_dim=X_train.shape[1],
                hidden_dims=tuple(ae_cfg["hidden_dims"]),
                bottleneck_dim=ae_cfg["bottleneck_dim"],
                seed=seed,
            )
            log_ae_path = results_dir / f"loss_p4_plain_ae_{family}_c{c_pct}_seed{seed}.csv"
            trained_plain_ae, _ = train_autoencoder(
                plain_ae,
                X_train=X_train,
                X_val_benign=X_val,
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
            scores_plain_ae = trained_plain_ae.compute_reconstruction_error(X_test)
            t_plain_ae = time.time() - t_start
            m_plain_ae = compute_all_metrics(y_test, scores_plain_ae)
            print(
                f"  [{current_run}/{total_runs}] [DONE] Family: {family:<10} | c={c_pct:02d}% | "
                f"Model: {'Plain AE':<18} | AUROC: {m_plain_ae['auroc']:.4f} | "
                f"FPR@95%: {m_plain_ae['fpr_at_95_tpr']:.4f} | TPR@1%: {m_plain_ae['tpr_at_1_fpr']:.4f} | Time: {t_plain_ae:.2f}s"
            )

            records.append({
                "family": family,
                "c": c_pct,
                "model": "Plain AE",
                "auroc": round(m_plain_ae["auroc"], 6),
                "fpr_at_95_tpr": round(m_plain_ae["fpr_at_95_tpr"], 6),
                "tpr_at_1_fpr": round(m_plain_ae["tpr_at_1_fpr"], 6),
                "attack_prevalence": round(attack_prevalence, 8),
                "runtime": round(t_plain_ae, 2),
            })

            # ---------------------------------------------------------
            # Model 4: Trimmed-Loss Autoencoder (trim_frac = 5%)
            # ---------------------------------------------------------
            current_run += 1
            t_start = time.time()
            trimmed_cfg = cfg["models"]["trimmed_autoencoder"]
            trimmed_ae = TrimmedAutoencoder(
                input_dim=X_train.shape[1],
                hidden_dims=tuple(trimmed_cfg["hidden_dims"]),
                bottleneck_dim=trimmed_cfg["bottleneck_dim"],
                seed=seed,
            )
            log_trim_path = results_dir / f"loss_p4_trimmed_ae_{family}_c{c_pct}_seed{seed}.csv"
            trained_trimmed_ae, _ = train_trimmed_autoencoder(
                trimmed_ae,
                X_train=X_train,
                X_val_benign=X_val,
                seed=seed,
                max_epochs=trimmed_cfg["max_epochs"],
                batch_size=trimmed_cfg["batch_size"],
                lr=trimmed_cfg["lr"],
                patience=trimmed_cfg["patience"],
                trim_frac=float(trimmed_cfg["trim_frac"]),
                warmup_epochs=int(trimmed_cfg["warmup_epochs"]),
                log_csv_path=log_trim_path,
                verbose=False,
            )
            scores_trimmed_ae = trained_trimmed_ae.compute_reconstruction_error(X_test)
            t_trimmed_ae = time.time() - t_start
            m_trimmed_ae = compute_all_metrics(y_test, scores_trimmed_ae)
            print(
                f"  [{current_run}/{total_runs}] [DONE] Family: {family:<10} | c={c_pct:02d}% | "
                f"Model: {'Trimmed AE':<18} | AUROC: {m_trimmed_ae['auroc']:.4f} | "
                f"FPR@95%: {m_trimmed_ae['fpr_at_95_tpr']:.4f} | TPR@1%: {m_trimmed_ae['tpr_at_1_fpr']:.4f} | Time: {t_trimmed_ae:.2f}s"
            )

            records.append({
                "family": family,
                "c": c_pct,
                "model": "Trimmed AE",
                "auroc": round(m_trimmed_ae["auroc"], 6),
                "fpr_at_95_tpr": round(m_trimmed_ae["fpr_at_95_tpr"], 6),
                "tpr_at_1_fpr": round(m_trimmed_ae["tpr_at_1_fpr"], 6),
                "attack_prevalence": round(attack_prevalence, 8),
                "runtime": round(t_trimmed_ae, 2),
            })

    # Save to JSON and CSV
    json_path = results_dir / f"p4_contamination_seed{seed}.json"
    csv_path = results_dir / f"p4_contamination_seed{seed}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved raw results to JSON: {json_path}")

    res_df = pd.DataFrame(records)
    res_df.to_csv(csv_path, index=False)
    print(f"Saved raw results to CSV: {csv_path}")

    models_order = ["PCA", "Isolation Forest", "Plain AE", "Trimmed AE"]

    # Helper function to construct table with rows = (family, c)
    def build_p4_table(metric_col: str) -> pd.DataFrame:
        table_rows = []
        for fam in target_families:
            for c_val in [0, 1, 2, 5, 10]:
                sub = res_df[(res_df["family"] == fam) & (res_df["c"] == c_val)]
                if sub.empty:
                    continue
                row_dict = {
                    "Family": fam,
                    "c (%)": f"{c_val}%",
                    "Prevalence": f"{sub['attack_prevalence'].values[0]:.5f}",
                }
                for m in models_order:
                    val = sub.loc[sub["model"] == m, metric_col].values[0]
                    row_dict[m] = f"{val:.4f}"
                table_rows.append(row_dict)
        return pd.DataFrame(table_rows)

    # 1. Print Three Tables
    print("\n" + "=" * 105)
    print(f"[{status_tag}]")
    print("PROTOCOL P4 TABLE 1: AUROC vs CONTAMINATION LEVEL (Higher = Better, Perfect = 1.0000)")
    print("=" * 105)
    df_auroc = build_p4_table("auroc")
    print(df_auroc.to_string(index=False))

    print("\n" + "=" * 105)
    print(f"[{status_tag}]")
    print("PROTOCOL P4 TABLE 2: FPR@95%TPR vs CONTAMINATION LEVEL (Lower = Better)")
    print("=" * 105)
    df_fpr = build_p4_table("fpr_at_95_tpr")
    print(df_fpr.to_string(index=False))

    print("\n" + "=" * 105)
    print(f"[{status_tag}]")
    print("PROTOCOL P4 TABLE 3: TPR@1%FPR vs CONTAMINATION LEVEL (Higher = Better)")
    print("=" * 105)
    df_tpr = build_p4_table("tpr_at_1_fpr")
    print(df_tpr.to_string(index=False))

    # 2. Sanity Check against P1 Plain AE
    print("\n" + "=" * 105)
    print("SANITY CHECK: c = 0 Plain AE vs P1 Plain AE for Same Family")
    print("Rule: Should be close. If they differ by > 0.03 AUROC, investigate.")
    print("=" * 105)

    p1_csv = results_dir / f"p1_baselines_and_ae_seed{seed}.csv"
    p1_df = pd.read_csv(p1_csv) if p1_csv.exists() else None

    sanity_rows = []
    has_large_divergence = False

    for fam in target_families:
        p4_c0_auroc = res_df[
            (res_df["family"] == fam) & (res_df["c"] == 0) & (res_df["model"] == "Plain AE")
        ]["auroc"].values[0]

        if p1_df is not None:
            p1_val = p1_df[
                (p1_df["family"] == fam) & (p1_df["model"] == "Autoencoder")
            ]["auroc"].values[0]
        else:
            p1_val = np.nan

        diff = abs(p4_c0_auroc - p1_val) if not np.isnan(p1_val) else 0.0
        status = "PASSED (<= 0.03)" if diff <= 0.03 else "DIVERGENCE (> 0.03)"
        if diff > 0.03:
            has_large_divergence = True

        sanity_rows.append({
            "Family": fam,
            "P4 (c=0, 200k train)": f"{p4_c0_auroc:.4f}",
            "P1 (500k train)": f"{p1_val:.4f}",
            "|Delta AUROC|": f"{diff:.4f}",
            "Sanity Status": status,
        })

    print(pd.DataFrame(sanity_rows).to_string(index=False))
    if has_large_divergence:
        print("\n[WARNING] Sanity check observed divergence > 0.03 AUROC. Investigation noted in final report.")
    else:
        print("\n[CONFIRMED] Sanity check PASSED across all families (delta <= 0.03 AUROC).")

    # 3. Save Line Plots
    print("\n" + "=" * 105)
    print("GENERATING P4 CONTAMINATION LINE PLOTS (results/figures/)")
    print("=" * 105)

    colors = {
        "PCA": "#1f77b4",
        "Isolation Forest": "#ff7f0e",
        "Plain AE": "#d62728",
        "Trimmed AE": "#2ca02c",
    }
    markers = {
        "PCA": "s",
        "Isolation Forest": "^",
        "Plain AE": "o",
        "Trimmed AE": "D",
    }
    linestyles = {
        "PCA": "--",
        "Isolation Forest": ":",
        "Plain AE": "-",
        "Trimmed AE": "-",
    }

    c_vals = [0, 1, 2, 5, 10]

    for fam in target_families:
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
        sub_fam = res_df[res_df["family"] == fam]

        for m in models_order:
            m_sub = sub_fam[sub_fam["model"] == m].sort_values("c")
            ax.plot(
                m_sub["c"].values,
                m_sub["auroc"].values,
                label=m,
                color=colors[m],
                marker=markers[m],
                linestyle=linestyles[m],
                linewidth=2.0,
                markersize=6,
            )

        ax.set_title(f"Protocol P4 Contamination Robustness: {fam}\n[{status_tag}]", fontsize=11, fontweight="bold")
        ax.set_xlabel("Contamination Rate c (%)", fontsize=10)
        ax.set_ylabel("AUROC (Higher = Better)", fontsize=10)
        ax.set_xticks(c_vals)
        ax.set_xticklabels([f"{c}%" for c in c_vals])
        ax.set_ylim(0.4, 1.02)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower left" if fam != "PortScan" else "lower right", frameon=True, fontsize=9)

        # Watermark
        fig.text(
            0.5, 0.02,
            f"DEV DATA, NOT FOR REPORTED RESULTS | Seed: {seed} | N = 200,000",
            ha="center", fontsize=8, color="gray", alpha=0.8
        )

        plot_path = figures_dir / f"p4_{fam}_auroc.png"
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"Saved figure: {plot_path}")

    # 4. Runtimes Summary
    print("\n" + "=" * 105)
    print("TOTAL EXECUTION RUNTIMES PER MODEL (seconds across all 15 conditions)")
    print("=" * 105)
    runtime_summary = res_df.groupby("model")["runtime"].agg(["count", "mean", "sum"]).reset_index()
    runtime_summary.columns = ["Model", "Runs", "Mean Runtime (s)", "Total Runtime (s)"]
    print(runtime_summary.to_string(index=False))


if __name__ == "__main__":
    main()
