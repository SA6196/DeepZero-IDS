"""Part A: Diagnostic Investigation into Why Trimming Fails (Seed 0).

DEV DATA, NOT FOR REPORTED RESULTS
Setting B, Families: PortScan and DoS/DDoS, c in {2, 5, 10}%, Models: Plain AE and Trimmed AE.
At epochs {1, 2, 3, 5, 10, final}:
(i) Fraction of injected attack rows in the top 5% highest-error TRAIN samples.
(ii) Median reconstruction error of injected attack rows vs benign rows.
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
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(4)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from data.preprocess import TabularPreprocessor
from models.autoencoders import MLPAutoencoder, TrimmedAutoencoder, compute_trimmed_mse_loss
from splits.protocols import generate_p4_setting_b_splits


def compute_train_errors(model: torch.nn.Module, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    errors: List[np.ndarray] = []
    n = len(X)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            bx = torch.from_numpy(X[i : i + batch_size]).float()
            recon = model(bx)
            # Per-sample MSE across features
            err = torch.mean((recon - bx) ** 2, dim=1).cpu().numpy()
            errors.append(err)
    return np.concatenate(errors)


def run_diagnostic_training(
    model: torch.nn.Module,
    X_train: np.ndarray,
    X_val: np.ndarray,
    is_attack_train: np.ndarray,
    is_trimmed: bool,
    trim_frac: float = 0.05,
    warmup_epochs: int = 2,
    seed: int = 0,
    max_epochs: int = 30,
    patience: int = 3,
    batch_size: int = 256,
    lr: float = 0.001,
    target_epochs: Sequence[int] = (1, 2, 3, 5, 10),
) -> Tuple[torch.nn.Module, List[Dict[str, Any]]]:
    rng = torch.Generator().manual_seed(seed)
    train_ds = TensorDataset(torch.from_numpy(X_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=rng)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    best_epoch = 1

    records: List[Dict[str, Any]] = []
    n_attack = int(np.sum(is_attack_train))

    def evaluate_train_checkpoint(ep_label: str, ep_num: int):
        errs = compute_train_errors(model, X_train)
        # Top 5% threshold
        k_top5 = int(np.ceil(0.05 * len(errs)))
        top5_thresh = np.partition(errs, len(errs) - k_top5)[len(errs) - k_top5]
        in_top5 = errs >= top5_thresh

        frac_in_top5 = float(np.sum(in_top5 & is_attack_train) / n_attack)
        med_att = float(np.median(errs[is_attack_train]))
        med_ben = float(np.median(errs[~is_attack_train]))

        records.append({
            "epoch_label": ep_label,
            "epoch_num": ep_num,
            "frac_attack_in_top5": round(frac_in_top5, 6),
            "median_err_attack": round(med_att, 6),
            "median_err_benign": round(med_ben, 6),
            "median_ratio": round(med_att / max(1e-12, med_ben), 4),
        })

    for epoch in range(1, max_epochs + 1):
        model.train()
        for (bx,) in train_loader:
            optimizer.zero_grad()
            recon = model(bx)
            if is_trimmed and epoch > warmup_epochs:
                loss, _ = compute_trimmed_mse_loss(recon, bx, trim_frac=trim_frac)
            else:
                loss = F.mse_loss(recon, bx, reduction="mean")
            loss.backward()
            optimizer.step()

        # Validation loss
        model.eval()
        val_loss_total = 0.0
        n_val = 0
        with torch.no_grad():
            for (bx,) in val_loader:
                recon = model(bx)
                val_loss_total += F.mse_loss(recon, bx, reduction="mean").item() * len(bx)
                n_val += len(bx)
        val_loss = val_loss_total / max(1, n_val)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            best_epoch = epoch
        else:
            epochs_no_improve += 1

        if epoch in target_epochs:
            evaluate_train_checkpoint(f"epoch_{epoch}", epoch)

        if epochs_no_improve >= patience:
            break

    # Final checkpoint (with best restored weights)
    model.load_state_dict(best_weights)
    evaluate_train_checkpoint("final (best)", best_epoch)

    return model, records


def main():
    status_tag = "DEV DATA, NOT FOR REPORTED RESULTS"
    seed = 0
    max_threads = 4
    families = ["PortScan", "DoS/DDoS"]
    c_levels = [2, 5, 10]
    models = ["Plain AE", "Trimmed AE"]
    target_epochs = [1, 2, 3, 5, 10]

    results_dir = REPO_ROOT / "results"
    figures_dir = REPO_ROOT / "results" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "diag_trimming_Setting_B.csv"

    print("=" * 105)
    print(f"[{status_tag}]")
    print("PART A: DIAGNOSTIC INVESTIGATION INTO WHY TRIMMING FAILS (Seed 0)")
    print("Setting B | Families: PortScan, DoS/DDoS | c in {2, 5, 10}% | Max Threads: 4")
    print("=" * 105)

    data_path = REPO_ROOT / "data" / "processed" / "cicids2017_cleaned_dev.parquet"
    print(f"Loading cleaned dataset from {data_path} ...")
    df = pd.read_parquet(data_path)
    feature_cols = [c for c in df.columns if c not in ["Label", "Attack_Family", "Day", "Destination Port"]]

    all_records: List[Dict[str, Any]] = []

    total_runs = len(families) * len(c_levels) * len(models)
    run_idx = 0

    for fam in families:
        print("\n" + "=" * 90)
        print(f"Family: {fam} (Setting B)")
        print("=" * 90)

        for c_pct in c_levels:
            c = c_pct / 100.0
            split = generate_p4_setting_b_splits(
                df,
                seed=seed,
                family=fam,
                contamination_rate=c,
                n_train_total=200000,
                val_benign_ratio=0.2,
                test_benign_ratio=0.2,
            )

            X_train_df = df.loc[split.train_indices, feature_cols]
            preprocessor = TabularPreprocessor(skew_threshold=3.0, clip_value=10.0)
            preprocessor.fit(X_train_df)
            X_train = preprocessor.transform(X_train_df)
            X_val = preprocessor.transform(df.loc[split.val_indices, feature_cols])

            # Label mask for reporting only
            train_families = df.loc[split.train_indices, "Attack_Family"].to_numpy()
            is_attack_train = (train_families == fam)
            n_attack = int(np.sum(is_attack_train))

            print(f"\n--- Contamination c={c_pct}% ({n_attack:,} {fam} flows in TRAIN) ---")

            for m_name in models:
                run_idx += 1
                is_trimmed = (m_name == "Trimmed AE")
                t0 = time.time()

                if is_trimmed:
                    model = TrimmedAutoencoder(
                        input_dim=X_train.shape[1],
                        hidden_dims=(64, 32, 16),
                        bottleneck_dim=16,
                        seed=seed,
                    )
                else:
                    model = MLPAutoencoder(
                        input_dim=X_train.shape[1],
                        hidden_dims=(64, 32, 16),
                        bottleneck_dim=16,
                        seed=seed,
                    )

                model, diag_records = run_diagnostic_training(
                    model,
                    X_train=X_train,
                    X_val=X_val,
                    is_attack_train=is_attack_train,
                    is_trimmed=is_trimmed,
                    trim_frac=0.05,
                    warmup_epochs=2,
                    seed=seed,
                    max_epochs=30,
                    patience=3,
                    batch_size=256,
                    lr=0.001,
                    target_epochs=target_epochs,
                )
                t_run = time.time() - t0

                # Log all checkpoints
                for r in diag_records:
                    rec = {
                        "family": fam,
                        "c": c_pct,
                        "model": m_name,
                        "epoch_label": r["epoch_label"],
                        "epoch_num": r["epoch_num"],
                        "frac_attack_in_top5": r["frac_attack_in_top5"],
                        "median_err_attack": r["median_err_attack"],
                        "median_err_benign": r["median_err_benign"],
                        "median_ratio": r["median_ratio"],
                    }
                    all_records.append(rec)

                final_r = [r for r in diag_records if "final" in r["epoch_label"]][0]
                print(
                    f"  [{run_idx}/{total_runs}] [DONE] {fam:<9} | c={c_pct:02d}% | Model: {m_name:<11} | "
                    f"Final Top5% Attack Frac: {final_r['frac_attack_in_top5']:.4f} | "
                    f"Med Attack Err: {final_r['median_err_attack']:.4f} | "
                    f"Med Benign Err: {final_r['median_err_benign']:.4f} | Time: {t_run:.2f}s"
                )

    # Save to CSV
    res_df = pd.DataFrame(all_records)
    res_df.to_csv(csv_path, index=False)
    print(f"\nSaved Part A diagnostic results to {csv_path}")

    # Compact Table Printing
    print("\n" + "=" * 115)
    print(f"[{status_tag}]")
    print("PART A TABLE: PER-SAMPLE RECONSTRUCTION ERROR DIAGNOSTIC (Seed 0)")
    print("Fraction of Injected Attack in Top 5% Highest Error & Median Errors (Attack vs Benign)")
    print("=" * 115)

    compact_rows = []
    for fam in families:
        for c_pct in c_levels:
            for m_name in models:
                sub = res_df[(res_df["family"] == fam) & (res_df["c"] == c_pct) & (res_df["model"] == m_name)]
                for ep in ["epoch_1", "epoch_2", "epoch_3", "epoch_5", "epoch_10", "final (best)"]:
                    ep_sub = sub[sub["epoch_label"] == ep]
                    if ep_sub.empty:
                        continue
                    r_val = ep_sub.iloc[0]
                    compact_rows.append({
                        "Family": fam,
                        "c (%)": f"{c_pct}%",
                        "Model": m_name,
                        "Epoch": ep,
                        "Attack in Top 5%": f"{r_val['frac_attack_in_top5']*100:.2f}%",
                        "Med Attack Err": f"{r_val['median_err_attack']:.6f}",
                        "Med Benign Err": f"{r_val['median_err_benign']:.6f}",
                        "Ratio (Att/Ben)": f"{r_val['median_ratio']:.2f}x",
                    })

    print(pd.DataFrame(compact_rows).to_string(index=False))

    # Generate Diagnostic Line Plots: Fraction of Contamination in Top 5% vs Epoch
    print("\n" + "=" * 105)
    print("SAVING DIAGNOSTIC PLOTS (results/figures/)")
    print("=" * 105)

    color_map = {
        "Plain AE": "#1f77b4",
        "Trimmed AE": "#d62728",
    }
    line_styles = {
        2: ":",
        5: "--",
        10: "-",
    }

    for fam in families:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        fam_sub = res_df[res_df["family"] == fam]

        # Filter strictly numeric epochs for plotting
        numeric_sub = fam_sub[fam_sub["epoch_label"].str.startswith("epoch_")].copy()
        numeric_sub["epoch"] = numeric_sub["epoch_num"].astype(int)

        for c_pct in c_levels:
            for m_name in models:
                sub_curve = numeric_sub[(numeric_sub["c"] == c_pct) & (numeric_sub["model"] == m_name)]
                sub_curve = sub_curve.sort_values("epoch")
                ax.plot(
                    sub_curve["epoch"],
                    sub_curve["frac_attack_in_top5"] * 100.0,
                    label=f"{m_name} (c={c_pct}%)",
                    color=color_map[m_name],
                    linestyle=line_styles[c_pct],
                    marker="o" if m_name == "Plain AE" else "s",
                    linewidth=2.0,
                )

        ax.axhline(5.0, color="gray", linestyle="-.", alpha=0.7, label="Baseline (Random = 5%)")
        ax.set_title(
            f"[{status_tag}]\nWhy Trimming Fails: Fraction of Injected {fam} in Top 5% Highest Loss\n"
            f"(Setting B, Seed 0)",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Training Epoch", fontsize=11, fontweight="bold")
        ax.set_ylabel("Injected Attack in Top 5% (%)", fontsize=11, fontweight="bold")
        ax.set_xticks(target_epochs)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best", framealpha=0.9, fontsize=9)
        plt.tight_layout()

        fig_path = figures_dir / f"diag_trimming_{fam.replace('/', '_')}.png"
        fig.savefig(fig_path)
        plt.close(fig)
        print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
