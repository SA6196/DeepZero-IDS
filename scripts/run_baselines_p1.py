"""Protocol P1 (Leave-One-Family-Out) Classical Baselines and Autoencoders Runner.

Global Rules Applied:
1. Fit all preprocessors and models on benign TRAIN data only. Zero TEST leakage.
2. Seeded and nested subsampling for TRAIN caps (OCSVM 20k c LOF 50k c IF/PCA 200k c AE/VAE 500k).
3. Score ALL TEST benign rows (~427k) + ALL test attack rows (no test benign cap).
4. Early stopping on VAL-BENIGN reconstruction loss ONLY for AE and VAE.
5. Thermal Limit: max_threads = 4 everywhere (sklearn, torch, OMP, MKL, OpenBLAS).
6. All results logged to JSON/CSV in results/.
7. DEV DATA, NOT FOR REPORTED RESULTS.
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
    LocalOutlierFactorBaseline,
    OneClassSVMBaseline,
    PCAReconstructionBaseline,
)
from data.preprocess import TabularPreprocessor
from eval.metrics import compute_all_metrics
from models.autoencoders import (
    MLPAutoencoder,
    VariationalAutoencoder,
    train_autoencoder,
)
from splits.protocols import generate_p1_lofo_splits


def parse_args():
    parser = argparse.ArgumentParser(description="Run Protocol P1 Baselines and Autoencoders")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baselines_p1_cicids2017.yaml",
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

    # Enforce thermal limits across all engines
    torch.set_num_threads(max_threads)
    os.environ["OMP_NUM_THREADS"] = str(max_threads)
    os.environ["MKL_NUM_THREADS"] = str(max_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(max_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(max_threads)

    print("=" * 95)
    print(f"[{status_tag}]")
    print(f"Protocol P1 LOFO: Classical Baselines & Autoencoders | Seed: {seed} | Max Threads: {max_threads}")
    print("=" * 95)

    # Speed caps & settings
    speed_caps = cfg.get("speed_caps", {})
    train_caps = speed_caps.get("train_rows", {})
    cap_if = int(train_caps.get("isolation_forest", 200000))
    cap_pca = int(train_caps.get("pca", 200000))
    cap_lof = int(train_caps.get("lof", 50000))
    cap_ocsvm = int(train_caps.get("ocsvm", 20000))
    cap_ae = int(train_caps.get("autoencoder", 500000))
    cap_vae = int(train_caps.get("vae", 500000))

    print(f"TRAIN Speed Caps (Seeded & Nested): OCSVM={cap_ocsvm}, LOF={cap_lof}, IF={cap_if}, PCA={cap_pca}, AE/VAE={cap_ae}")
    print("TEST Benign Cap: NONE (Scoring ALL available test benign rows + all attack rows)")

    # Load cleaned dataset
    data_path = Path(cfg["dataset"]["cleaned_path"])
    print(f"\nLoading cleaned dataset from: {data_path} ...")
    t0_data = time.time()
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} rows with {df.shape[1]} columns in {time.time() - t0_data:.2f}s.")

    feature_cols = [
        c
        for c in df.columns
        if c not in ["Label", "Attack_Family", "Day", "Destination Port"]
    ]

    # Generate P1 LOFO splits
    val_benign_ratio = float(cfg["protocol"].get("val_benign_ratio", 0.2))
    test_benign_ratio = float(cfg["protocol"].get("test_benign_ratio", 0.2))
    target_families = cfg["protocol"].get("target_families", None)

    print(f"\nGenerating Protocol P1 LOFO splits (seed={seed}) ...")
    p1_splits = generate_p1_lofo_splits(
        df,
        seed=seed,
        val_benign_ratio=val_benign_ratio,
        test_benign_ratio=test_benign_ratio,
        target_families=target_families,
    )
    print(f"Generated {len(p1_splits)} family splits: {list(p1_splits.keys())}")

    results_dir = Path(cfg["output"].get("results_dir", "results"))
    models_dir = Path(cfg["output"].get("models_dir", "results/models"))
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    runtimes: Dict[str, Dict[str, float]] = {}
    empirical_score_tests: List[Dict[str, Any]] = []

    skew_thresh = float(cfg["preprocessing"].get("skew_threshold", 3.0))

    # Iterate over attack families
    for family, split in p1_splits.items():
        print(f"\n" + "-" * 80)
        print(f"Processing Held-Out Family: {family}")
        print(f"Benign Train Rows available: {len(split.train_indices):,}")
        print(f"Benign Val Rows available:   {len(split.val_indices):,}")
        print(f"Benign Test Rows available:  {split.metadata['n_benign_test']:,}")
        print(f"Attack Test Rows:            {split.metadata['n_family_attack_test']:,}")

        is_unreliable = split.metadata["n_family_attack_test"] < 100
        if is_unreliable:
            print(f"  [FLAG] {family} has < 100 attack flows ({split.metadata['n_family_attack_test']}). Flagged as UNRELIABLE.")

        # 1. Seeded and strictly nested subsampling for TRAIN
        rng_train = np.random.default_rng(seed)
        shuffled_train = rng_train.permutation(split.train_indices)

        # Nested subsets: OCSVM (20k) subset of LOF (50k) subset of IF/PCA (200k) subset of AE/VAE (500k)
        max_train = max(cap_if, cap_pca, cap_lof, cap_ocsvm, cap_ae, cap_vae)
        train_idx_500k = shuffled_train[: min(max_train, len(shuffled_train))]
        train_idx_200k = shuffled_train[: min(cap_if, len(shuffled_train))]
        train_idx_50k = shuffled_train[: min(cap_lof, len(shuffled_train))]
        train_idx_20k = shuffled_train[: min(cap_ocsvm, len(shuffled_train))]

        # Verify strict nesting invariant
        assert set(train_idx_20k).issubset(set(train_idx_50k))
        assert set(train_idx_50k).issubset(set(train_idx_200k))
        assert set(train_idx_200k).issubset(set(train_idx_500k))

        # 2. Preprocessing pipeline fit strictly on TRAIN (using 500k subsample)
        print(f"  Fitting Preprocessor on TRAIN (500k rows, skew_thresh={skew_thresh}) ...")
        t_prep_0 = time.time()
        X_train_df = df.loc[train_idx_500k, feature_cols]
        preprocessor = TabularPreprocessor(skew_threshold=skew_thresh)
        preprocessor.fit(X_train_df)
        t_prep = time.time() - t_prep_0

        prep_save_path = models_dir / f"preprocessor_p1_{family.replace('/', '_')}_seed{seed}.joblib"
        preprocessor.save(prep_save_path)
        print(f"  Preprocessor fit in {t_prep:.2f}s. Saved to {prep_save_path}")
        print(f"  Constant columns dropped ({len(preprocessor.constant_cols_)}): {preprocessor.constant_cols_}")
        print(f"  Log1p transformed columns: {len(preprocessor.log1p_cols_)}")

        # Transform nested TRAIN subsets
        X_train_500k = preprocessor.transform(X_train_df)
        X_train_200k = X_train_500k[: len(train_idx_200k)]
        X_train_50k = X_train_500k[: len(train_idx_50k)]
        X_train_20k = X_train_500k[: len(train_idx_20k)]

        # Transform VAL-BENIGN (capped at 100k for training efficiency)
        rng_val = np.random.default_rng(seed + 500)
        val_sample_size = min(100000, len(split.val_indices))
        val_sub_indices = rng_val.choice(split.val_indices, size=val_sample_size, replace=False)
        X_val_benign_df = df.loc[val_sub_indices, feature_cols]
        X_val_benign = preprocessor.transform(X_val_benign_df)

        # 3. TEST: Score ALL test benign rows (~427k) + ALL test attack rows
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
        attack_prevalence = n_attack_test / (n_benign_test + n_attack_test)

        print(f"  Evaluating on ALL TEST: {n_benign_test:,} Benign + {n_attack_test:,} {family}")
        print(f"  Total Test Rows: {len(eval_indices):,} | Attack Prevalence: {attack_prevalence:.6f} ({attack_prevalence * 100:.3f}%)")

        t_trans_0 = time.time()
        X_test = preprocessor.transform(X_test_df)
        print(f"  TEST transformed in {time.time() - t_trans_0:.2f}s.")

        runtimes[family] = {}

        # -------------------------------------------------------------
        # Model 1: Isolation Forest
        # -------------------------------------------------------------
        print("  [1/6] Running Isolation Forest (200k fit) ...")
        t_start = time.time()
        n_jobs_if = min(max_threads, int(cfg["models"]["isolation_forest"].get("n_jobs", max_threads)))
        if n_jobs_if <= 0:
            n_jobs_if = max_threads
        if_model = IsolationForestBaseline(
            n_estimators=cfg["models"]["isolation_forest"]["n_estimators"],
            random_state=cfg["models"]["isolation_forest"]["random_state"],
            n_jobs=n_jobs_if,
        )
        if_model.fit(X_train_200k)
        t_fit = time.time() - t_start
        scores_if = if_model.score(X_test)
        t_total_if = time.time() - t_start
        runtimes[family]["Isolation Forest"] = t_total_if
        m_if = compute_all_metrics(y_test, scores_if)
        print(f"    IF done in {t_total_if:.2f}s (fit: {t_fit:.2f}s) | AUROC: {m_if['auroc']:.4f}, AUPRC: {m_if['auprc']:.4f}, FPR@95%TPR: {m_if['fpr_at_95_tpr']:.4f}, TPR@1%FPR: {m_if['tpr_at_1_fpr']:.4f}")

        # -------------------------------------------------------------
        # Model 2: One-Class SVM
        # -------------------------------------------------------------
        print("  [2/6] Running One-Class SVM (20k fit) ...")
        t_start = time.time()
        ocsvm_model = OneClassSVMBaseline(
            kernel=cfg["models"]["ocsvm"]["kernel"],
            gamma=cfg["models"]["ocsvm"]["gamma"],
            nu=cfg["models"]["ocsvm"]["nu"],
        )
        ocsvm_model.fit(X_train_20k)
        t_fit = time.time() - t_start
        scores_ocsvm = ocsvm_model.score(X_test)
        t_total_ocsvm = time.time() - t_start
        runtimes[family]["One-Class SVM"] = t_total_ocsvm
        m_ocsvm = compute_all_metrics(y_test, scores_ocsvm)
        print(f"    OCSVM done in {t_total_ocsvm:.2f}s (fit: {t_fit:.2f}s) | AUROC: {m_ocsvm['auroc']:.4f}, AUPRC: {m_ocsvm['auprc']:.4f}, FPR@95%TPR: {m_ocsvm['fpr_at_95_tpr']:.4f}, TPR@1%FPR: {m_ocsvm['tpr_at_1_fpr']:.4f}")

        # -------------------------------------------------------------
        # Model 3: Local Outlier Factor (novelty=True)
        # -------------------------------------------------------------
        print("  [3/6] Running LOF novelty=True (50k fit) ...")
        t_start = time.time()
        n_jobs_lof = min(max_threads, int(cfg["models"]["lof"].get("n_jobs", max_threads)))
        if n_jobs_lof <= 0:
            n_jobs_lof = max_threads
        lof_model = LocalOutlierFactorBaseline(
            n_neighbors=cfg["models"]["lof"]["n_neighbors"],
            novelty=cfg["models"]["lof"]["novelty"],
            n_jobs=n_jobs_lof,
        )
        lof_model.fit(X_train_50k)
        t_fit = time.time() - t_start
        scores_lof = lof_model.score(X_test)
        t_total_lof = time.time() - t_start
        runtimes[family]["LOF"] = t_total_lof
        m_lof = compute_all_metrics(y_test, scores_lof)
        print(f"    LOF done in {t_total_lof:.2f}s (fit: {t_fit:.2f}s) | AUROC: {m_lof['auroc']:.4f}, AUPRC: {m_lof['auprc']:.4f}, FPR@95%TPR: {m_lof['fpr_at_95_tpr']:.4f}, TPR@1%FPR: {m_lof['tpr_at_1_fpr']:.4f}")

        # -------------------------------------------------------------
        # Model 4: PCA Reconstruction Error
        # -------------------------------------------------------------
        print("  [4/6] Running PCA Reconstruction (200k fit) ...")
        t_start = time.time()
        pca_model = PCAReconstructionBaseline(
            n_components=cfg["models"]["pca"]["n_components"],
            random_state=cfg["models"]["pca"]["random_state"],
        )
        pca_model.fit(X_train_200k)
        t_fit = time.time() - t_start
        scores_pca = pca_model.score(X_test)
        t_total_pca = time.time() - t_start
        runtimes[family]["PCA"] = t_total_pca
        m_pca = compute_all_metrics(y_test, scores_pca)
        print(f"    PCA done in {t_total_pca:.2f}s (fit: {t_fit:.2f}s) | AUROC: {m_pca['auroc']:.4f}, AUPRC: {m_pca['auprc']:.4f}, FPR@95%TPR: {m_pca['fpr_at_95_tpr']:.4f}, TPR@1%FPR: {m_pca['tpr_at_1_fpr']:.4f}")

        # -------------------------------------------------------------
        # Model 5: Plain MLP Autoencoder (500k fit)
        # -------------------------------------------------------------
        print("  [5/6] Training Plain MLP Autoencoder (500k fit, early stopping on VAL-BENIGN) ...")
        t_start = time.time()
        ae_cfg = cfg["models"]["autoencoder"]
        ae_model = MLPAutoencoder(
            input_dim=X_train_500k.shape[1],
            hidden_dims=tuple(ae_cfg["hidden_dims"]),
            bottleneck_dim=ae_cfg["bottleneck_dim"],
            seed=seed,
        )
        ae_log_csv = results_dir / f"loss_ae_p1_{family.replace('/', '_')}_seed{seed}.csv"
        trained_ae, ae_history = train_autoencoder(
            ae_model,
            X_train=X_train_500k,
            X_val_benign=X_val_benign,
            seed=seed,
            max_epochs=ae_cfg["max_epochs"],
            batch_size=ae_cfg["batch_size"],
            lr=ae_cfg["lr"],
            patience=ae_cfg["patience"],
            log_csv_path=ae_log_csv,
        )
        t_fit = time.time() - t_start
        scores_ae = trained_ae.compute_reconstruction_error(X_test)
        t_total_ae = time.time() - t_start
        runtimes[family]["Autoencoder"] = t_total_ae
        m_ae = compute_all_metrics(y_test, scores_ae)
        print(f"    AE done in {t_total_ae:.2f}s (fit: {t_fit:.2f}s, {len(ae_history)} epochs) | AUROC: {m_ae['auroc']:.4f}, AUPRC: {m_ae['auprc']:.4f}, FPR@95%TPR: {m_ae['fpr_at_95_tpr']:.4f}, TPR@1%FPR: {m_ae['tpr_at_1_fpr']:.4f}")

        # -------------------------------------------------------------
        # Model 6: Variational Autoencoder (VAE) (500k fit)
        # -------------------------------------------------------------
        print("  [6/6] Training VAE (500k fit, deterministic recon error on mean) ...")
        t_start = time.time()
        vae_cfg = cfg["models"]["vae"]
        vae_model = VariationalAutoencoder(
            input_dim=X_train_500k.shape[1],
            hidden_dims=tuple(vae_cfg["hidden_dims"]),
            latent_dim=vae_cfg["latent_dim"],
            seed=seed,
        )
        vae_log_csv = results_dir / f"loss_vae_p1_{family.replace('/', '_')}_seed{seed}.csv"
        trained_vae, vae_history = train_autoencoder(
            vae_model,
            X_train=X_train_500k,
            X_val_benign=X_val_benign,
            seed=seed,
            max_epochs=vae_cfg["max_epochs"],
            batch_size=vae_cfg["batch_size"],
            lr=vae_cfg["lr"],
            patience=vae_cfg["patience"],
            log_csv_path=vae_log_csv,
        )
        t_fit = time.time() - t_start
        scores_vae = trained_vae.compute_reconstruction_error(X_test)
        t_total_vae = time.time() - t_start
        runtimes[family]["VAE"] = t_total_vae
        m_vae = compute_all_metrics(y_test, scores_vae)
        print(f"    VAE done in {t_total_vae:.2f}s (fit: {t_fit:.2f}s, {len(vae_history)} epochs) | AUROC: {m_vae['auroc']:.4f}, AUPRC: {m_vae['auprc']:.4f}, FPR@95%TPR: {m_vae['fpr_at_95_tpr']:.4f}, TPR@1%FPR: {m_vae['tpr_at_1_fpr']:.4f}")

        # Empirical test for AUROC < 0.5 (Botnet score comparison)
        if family == "Botnet":
            empirical_score_tests.append({
                "model": "One-Class SVM",
                "benign_mean": float(scores_ocsvm[y_test == 0].mean()),
                "botnet_mean": float(scores_ocsvm[y_test == 1].mean()),
            })
            empirical_score_tests.append({
                "model": "LOF",
                "benign_mean": float(scores_lof[y_test == 0].mean()),
                "botnet_mean": float(scores_lof[y_test == 1].mean()),
            })

        # Collate records
        model_results = [
            ("Isolation Forest", m_if, cap_if, t_total_if),
            ("One-Class SVM", m_ocsvm, cap_ocsvm, t_total_ocsvm),
            ("LOF", m_lof, cap_lof, t_total_lof),
            ("PCA", m_pca, cap_pca, t_total_pca),
            ("Autoencoder", m_ae, cap_ae, t_total_ae),
            ("VAE", m_vae, cap_vae, t_total_vae),
        ]

        for model_name, metrics, n_train_used, r_time in model_results:
            rec = {
                "experiment": cfg["experiment"]["name"],
                "protocol": "P1",
                "seed": seed,
                "status": status_tag,
                "family": family,
                "is_unreliable": is_unreliable,
                "model": model_name,
                "n_train": n_train_used,
                "n_test_benign": n_benign_test,
                "n_test_attack": n_attack_test,
                "attack_prevalence": round(attack_prevalence, 8),
                "runtime_seconds": round(r_time, 2),
                "auroc": round(metrics["auroc"], 6),
                "auprc": round(metrics["auprc"], 6),
                "fpr_at_95_tpr": round(metrics["fpr_at_95_tpr"], 6),
                "tpr_at_1_fpr": round(metrics["tpr_at_1_fpr"], 6),
                "speed_caps": speed_caps,
            }
            records.append(rec)

    # Save to JSON and CSV
    json_path = results_dir / f"p1_baselines_and_ae_seed{seed}.json"
    csv_path = results_dir / f"p1_baselines_and_ae_seed{seed}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved raw results to JSON: {json_path}")

    res_df = pd.DataFrame(records)
    res_df.to_csv(csv_path, index=False)
    print(f"Saved raw results to CSV: {csv_path}")

    # Also update p1_baselines_seed{seed}.csv / json for compatibility
    res_df.to_csv(results_dir / f"p1_baselines_seed{seed}.csv", index=False)
    with open(results_dir / f"p1_baselines_seed{seed}.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    models_order = ["Isolation Forest", "One-Class SVM", "LOF", "PCA", "Autoencoder", "VAE"]
    families_order = list(p1_splits.keys())

    # Helper to construct tables
    def build_metric_table(metric_col: str, title: str, subtitle: str) -> pd.DataFrame:
        rows = []
        for fam in families_order:
            fam_label = f"{fam} [UNRELIABLE]" if fam == "Infiltration" else fam
            sub_fam = res_df[res_df["family"] == fam]
            prev = sub_fam["attack_prevalence"].values[0]
            row = {"Family": fam_label, "Prevalence": f"{prev:.5f} ({prev*100:.3f}%)"}
            for m in models_order:
                sub_m = sub_fam[sub_fam["model"] == m]
                val = sub_m[metric_col].values[0]
                row[m] = f"{val:.4f}"
            rows.append(row)

        # Macro Average (Reliable Families)
        macro_row = {"Family": "Macro Average (Reliable)", "Prevalence": "-"}
        for m in models_order:
            sub_rel = res_df[(res_df["family"] != "Infiltration") & (res_df["model"] == m)]
            macro_row[m] = f"{sub_rel[metric_col].mean():.4f}"
        rows.append(macro_row)
        return pd.DataFrame(rows)

    # Print Headline Tables
    print("\n" + "=" * 105)
    print(f"[{status_tag}]")
    print("HEADLINE TABLE 1: Protocol P1 LOFO - AUROC (Higher = Better, Perfect = 1.0000)")
    print("=" * 105)
    df_auroc = build_metric_table("auroc", "AUROC", "Higher = Better")
    print(df_auroc.to_string(index=False))

    print("\n" + "=" * 105)
    print(f"[{status_tag}]")
    print("HEADLINE TABLE 2: Protocol P1 LOFO - False Positive Rate at 95% TPR (FPR@95%TPR, Lower = Better)")
    print("=" * 105)
    df_fpr = build_metric_table("fpr_at_95_tpr", "FPR@95%TPR", "Lower = Better")
    print(df_fpr.to_string(index=False))

    print("\n" + "=" * 105)
    print(f"[{status_tag}]")
    print("HEADLINE TABLE 3: Protocol P1 LOFO - True Positive Rate at 1% FPR (TPR@1%FPR, Higher = Better)")
    print("=" * 105)
    df_tpr = build_metric_table("tpr_at_1_fpr", "TPR@1%FPR", "Higher = Better")
    print(df_tpr.to_string(index=False))

    print("\n" + "=" * 105)
    print(f"[{status_tag}]")
    print("SEPARATE TABLE 4: Protocol P1 LOFO - Area Under Precision-Recall Curve (AUPRC) with Prevalence")
    print("Note: AUPRC baseline equals test attack prevalence. Direct comparison across families requires prevalence context.")
    print("=" * 105)
    df_auprc = build_metric_table("auprc", "AUPRC", "Depends on prevalence")
    print(df_auprc.to_string(index=False))

    # Print Runtimes
    print("\n" + "=" * 105)
    print(f"[{status_tag}]")
    print("MODEL EXECUTION RUNTIMES (seconds)")
    print("=" * 105)
    runtime_table = []
    for fam in families_order:
        row = {"Family": fam}
        for m in models_order:
            row[m] = f"{runtimes[fam].get(m, 0.0):.2f}s"
        runtime_table.append(row)
    print(pd.DataFrame(runtime_table).to_string(index=False))

    # Print Empirical Score Test for AUROC < 0.5
    print("\n" + "=" * 105)
    print("EMPIRICAL TEST: Mean Anomaly Scores on Botnet vs Benign (Higher = More Anomalous)")
    print("=" * 105)
    for test in empirical_score_tests:
        print(
            f"Model: {test['model']:<15} | Benign Test Mean: {test['benign_mean']:.6f} | "
            f"Botnet Test Mean: {test['botnet_mean']:.6f} | "
            f"Botnet > Benign: {test['botnet_mean'] > test['benign_mean']}"
        )


if __name__ == "__main__":
    main()
