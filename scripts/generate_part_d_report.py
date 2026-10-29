"""Part D: Results Reporting and Analysis for Setting B.

DEV DATA, NOT FOR REPORTED RESULTS
Merges results from results/p4_redo_seeds012.csv (Plain AE, Trimmed AE, Plain PCA)
and results/p4_sanitization_results.csv (Sanitized AE, Sanitized PCA).

Generates:
1. AUROC table (mean +- std over 3 seeds) for (family, c):
   Columns: Plain AE, Trimmed AE, Sanitized AE, Plain PCA, Sanitized PCA.
2. FPR@95%TPR table (mean +- std over 3 seeds) for (family, c):
   Columns: Plain AE, Trimmed AE, Sanitized AE, Plain PCA, Sanitized PCA.
3. Removal table per (family, c), averaged over seeds:
   Fraction of injected attack rows removed, fraction of benign pool rows removed (including c=0 cost).
4. Seed-size ablation table at c=5%: n_seed in {1,000, 10,000}.
5. AUROC vs c comparison plots with error bars saved to results/figures/.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

STATUS_TAG = "DEV DATA, NOT FOR REPORTED RESULTS"


def format_mean_std(val_series: pd.Series) -> str:
    if val_series.empty:
        return "N/A"
    m = val_series.mean()
    s = val_series.std(ddof=1) if len(val_series) > 1 else 0.0
    if pd.isna(s):
        s = 0.0
    return f"{m:.4f} ± {s:.4f}"


def build_comparison_dataframe(
    redo_df: pd.DataFrame,
    sanit_df: pd.DataFrame,
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

    # 1. Baseline rows from redo_df
    # Filter to Setting B, families PortScan and DoS/DDoS, c in {0, 2, 5, 10}
    b_df = redo_df[
        (redo_df["setting"] == "Setting B")
        & (redo_df["family"].isin(["PortScan", "DoS/DDoS"]))
        & (redo_df["c"].isin([0, 2, 5, 10]))
    ].copy()

    for _, row in b_df.iterrows():
        m = row["model"]
        if m in ["Plain AE", "Trimmed AE", "PCA"]:
            model_key = "Plain PCA" if m == "PCA" else m
            records.append({
                "setting": "Setting B",
                "family": row["family"],
                "c": int(row["c"]),
                "seed": int(row["seed"]),
                "n_seed": 0,
                "model": model_key,
                "auroc": float(row["auroc"]),
                "fpr_at_95_tpr": float(row["fpr_at_95_tpr"]),
                "frac_attack_removed": 0.0,
                "frac_benign_removed": 0.0,
            })

    # 2. Sanitized rows from sanit_df
    if not sanit_df.empty:
        for _, row in sanit_df.iterrows():
            records.append({
                "setting": "Setting B",
                "family": row["family"],
                "c": int(row["c"]),
                "seed": int(row["seed"]),
                "n_seed": int(row.get("n_seed", 10000)),
                "model": row["model"],
                "auroc": float(row["auroc"]),
                "fpr_at_95_tpr": float(row["fpr_at_95_tpr"]),
                "frac_attack_removed": float(row.get("frac_attack_removed", 0.0)),
                "frac_benign_removed": float(row.get("frac_benign_removed", 0.0)),
            })

    return pd.DataFrame(records)


def generate_reports():
    results_dir = REPO_ROOT / "results"
    figures_dir = REPO_ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    redo_path = results_dir / "p4_redo_seeds012.csv"
    sanit_path = results_dir / "p4_sanitization_results.csv"

    if not redo_path.exists():
        print(f"Error: {redo_path} not found.")
        return
    if not sanit_path.exists():
        print(f"Error: {sanit_path} not found.")
        return

    redo_df = pd.read_csv(redo_path)
    sanit_df = pd.read_csv(sanit_path)

    merged_df = build_comparison_dataframe(redo_df, sanit_df)

    families = ["PortScan", "DoS/DDoS"]
    c_levels = [0, 2, 5, 10]
    methods = ["Plain AE", "Trimmed AE", "Sanitized AE", "Plain PCA", "Sanitized PCA"]

    # Table 1: AUROC Table (mean ± std over 3 seeds, n_seed=10000)
    print("\n" + "=" * 115)
    print(f"[{STATUS_TAG}]")
    print("TABLE 1: AUROC (mean ± std over 3 seeds) UNDER SETTING B (n_seed = 10,000)")
    print("=" * 115)

    header = f"{'Family':<12} | {'c (%)':<6} | " + " | ".join([f"{m:<17}" for m in methods])
    print(header)
    print("-" * len(header))

    auroc_rows = []
    for fam in families:
        for c in c_levels:
            row_str = f"{fam:<12} | {c:<6} | "
            cols = []
            row_dict = {"family": fam, "c": c}
            for m in methods:
                if "Sanitized" in m:
                    sub = merged_df[
                        (merged_df["family"] == fam)
                        & (merged_df["c"] == c)
                        & (merged_df["model"] == m)
                        & (merged_df["n_seed"] == 10000)
                    ]
                else:
                    sub = merged_df[
                        (merged_df["family"] == fam)
                        & (merged_df["c"] == c)
                        & (merged_df["model"] == m)
                    ]
                cell_val = format_mean_std(sub["auroc"]) if not sub.empty else "N/A"
                cols.append(f"{cell_val:<17}")
                row_dict[m] = cell_val
            row_str += " | ".join(cols)
            print(row_str)
            auroc_rows.append(row_dict)

    # Table 2: FPR@95%TPR Table
    print("\n" + "=" * 115)
    print(f"[{STATUS_TAG}]")
    print("TABLE 2: FPR@95%TPR (mean ± std over 3 seeds) UNDER SETTING B (n_seed = 10,000)")
    print("=" * 115)

    print(header)
    print("-" * len(header))

    fpr_rows = []
    for fam in families:
        for c in c_levels:
            row_str = f"{fam:<12} | {c:<6} | "
            cols = []
            row_dict = {"family": fam, "c": c}
            for m in methods:
                if "Sanitized" in m:
                    sub = merged_df[
                        (merged_df["family"] == fam)
                        & (merged_df["c"] == c)
                        & (merged_df["model"] == m)
                        & (merged_df["n_seed"] == 10000)
                    ]
                else:
                    sub = merged_df[
                        (merged_df["family"] == fam)
                        & (merged_df["c"] == c)
                        & (merged_df["model"] == m)
                    ]
                cell_val = format_mean_std(sub["fpr_at_95_tpr"]) if not sub.empty else "N/A"
                cols.append(f"{cell_val:<17}")
                row_dict[m] = cell_val
            row_str += " | ".join(cols)
            print(row_str)
            fpr_rows.append(row_dict)

    # Table 3: Removal Table per (family, c) averaged over seeds
    print("\n" + "=" * 115)
    print(f"[{STATUS_TAG}]")
    print("TABLE 3: POOL SANITIZATION REMOVAL RATES (Averaged over 3 seeds, n_seed = 10,000)")
    print("Note: c = 0% rows show the cost of sanitizing a clean pool (false positive shedding)")
    print("=" * 115)

    rem_header = f"{'Family':<12} | {'c (%)':<6} | {'Model':<15} | {'Injected Attack Removed':<25} | {'Benign Pool Removed':<22}"
    print(rem_header)
    print("-" * len(rem_header))

    for fam in families:
        for c in c_levels:
            for m in ["Sanitized AE", "Sanitized PCA"]:
                sub = merged_df[
                    (merged_df["family"] == fam)
                    & (merged_df["c"] == c)
                    & (merged_df["model"] == m)
                    & (merged_df["n_seed"] == 10000)
                ]
                if sub.empty:
                    continue
                att_rem = sub["frac_attack_removed"].mean()
                ben_rem = sub["frac_benign_removed"].mean()
                att_str = f"{att_rem * 100:.2f}%" if c > 0 else "N/A (clean pool)"
                ben_str = f"{ben_rem * 100:.2f}%"
                print(f"{fam:<12} | {c:<6} | {m:<15} | {att_str:<25} | {ben_str:<22}")

    # Table 4: Seed Size Ablation Table at c = 5%
    print("\n" + "=" * 115)
    print(f"[{STATUS_TAG}]")
    print("TABLE 4: SEED-SIZE ABLATION AT c = 5% (n_seed in {1,000, 10,000}, averaged over 3 seeds)")
    print("=" * 115)

    abl_header = f"{'Family':<12} | {'Model':<15} | {'n_seed':<8} | {'AUROC (mean ± std)':<22} | {'FPR@95% (mean ± std)':<22} | {'Att Removed':<13} | {'Ben Removed':<13}"
    print(abl_header)
    print("-" * len(abl_header))

    for fam in families:
        for m in ["Sanitized AE", "Sanitized PCA"]:
            for ns in [1000, 10000]:
                sub = merged_df[
                    (merged_df["family"] == fam)
                    & (merged_df["c"] == 5)
                    & (merged_df["model"] == m)
                    & (merged_df["n_seed"] == ns)
                ]
                if sub.empty:
                    continue
                auroc_str = format_mean_std(sub["auroc"])
                fpr_str = format_mean_std(sub["fpr_at_95_tpr"])
                att_rem = f"{sub['frac_attack_removed'].mean() * 100:.2f}%"
                ben_rem = f"{sub['frac_benign_removed'].mean() * 100:.2f}%"
                print(f"{fam:<12} | {m:<15} | {ns:<8} | {auroc_str:<22} | {fpr_str:<22} | {att_rem:<13} | {ben_rem:<13}")

    # Plots: AUROC vs c per family with error bars
    print("\nGenerating AUROC vs c comparison plots ...")
    color_map = {
        "Plain AE": "#d95f02",
        "Trimmed AE": "#7570b3",
        "Sanitized AE": "#1b9e77",
        "Plain PCA": "#e7298a",
        "Sanitized PCA": "#66a61e",
    }
    marker_map = {
        "Plain AE": "o",
        "Trimmed AE": "s",
        "Sanitized AE": "D",
        "Plain PCA": "^",
        "Sanitized PCA": "v",
    }

    for fam in families:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for m in methods:
            means = []
            stds = []
            for c in c_levels:
                if "Sanitized" in m:
                    sub = merged_df[
                        (merged_df["family"] == fam)
                        & (merged_df["c"] == c)
                        & (merged_df["model"] == m)
                        & (merged_df["n_seed"] == 10000)
                    ]
                else:
                    sub = merged_df[
                        (merged_df["family"] == fam)
                        & (merged_df["c"] == c)
                        & (merged_df["model"] == m)
                    ]
                if not sub.empty:
                    means.append(sub["auroc"].mean())
                    stds.append(sub["auroc"].std(ddof=1) if len(sub) > 1 else 0.0)
                else:
                    means.append(np.nan)
                    stds.append(0.0)

            ax.errorbar(
                c_levels,
                means,
                yerr=stds,
                label=m,
                color=color_map.get(m, "black"),
                marker=marker_map.get(m, "o"),
                linewidth=2.0,
                capsize=4,
                capthick=1.5,
                alpha=0.9,
            )

        ax.set_title(f"[{STATUS_TAG}]\nSetting B: AUROC vs Contamination Rate ({fam})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Contamination Rate c (%)", fontsize=11)
        ax.set_ylabel("AUROC", fontsize=11)
        ax.set_xticks(c_levels)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower left", framealpha=0.9)
        plt.tight_layout()

        plot_path = figures_dir / f"p4_sanitization_{fam.replace('/', '_')}.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    generate_reports()
