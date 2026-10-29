"""Stage 5: Inference Latency and Throughput Benchmarking (Single-Threaded).

DEV DATA, NOT FOR REPORTED RESULTS
Models: PCA, IsolationForest, LOF, OneClassSVM, MLP AE, VAE.
Configuration:
- Single thread: torch.set_num_threads(1), n_jobs=1.
- Batch sizes: 1 (per-flow inline) and 256 (micro-batch).
- Features: 76.
- 20 warm-up calls, 200 timed repeats via time.perf_counter().
- Measures: median and p95 microseconds per flow, flows/sec.
- Parameter counts for AE / VAE.
- Fit wall-clock extracted from Stage 1 results (seed 0).
- System environment: CPU model and OS.
Output: results/final_latency.csv
"""

from __future__ import annotations

import os

# Enforce strictly 1 thread for latency benchmark
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

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
from models.autoencoders import MLPAutoencoder, VariationalAutoencoder

STATUS_TAG = "DEV DATA, NOT FOR REPORTED RESULTS"


def get_cpu_info() -> str:
    cpu_name = platform.processor()
    try:
        out = subprocess.check_output("wmic cpu get name", shell=True, text=True).strip()
        lines = [l.strip() for l in out.splitlines() if l.strip() and "Name" not in l]
        if lines:
            cpu_name = lines[0]
    except Exception:
        pass
    return cpu_name or "Unknown CPU"


def get_os_info() -> str:
    return f"{platform.system()} {platform.release()} (Build {platform.version()})"


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_stage5():
    print("=" * 105)
    print(f"[{STATUS_TAG}]")
    print("STAGE 5: LATENCY AND THROUGHPUT BENCHMARK (SINGLE THREAD)")
    print("Batch Sizes: {1, 256} | 76 Features | 20 Warmups | 200 Timed Repeats")
    print("=" * 105)

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "final_latency.csv"
    p1_csv = results_dir / "final_p1_seeds012.csv"

    cpu_info = get_cpu_info()
    os_info = get_os_info()
    print(f"Host CPU : {cpu_info}")
    print(f"Host OS  : {os_info}")

    # Read Stage 1 fit times (seed 0)
    fit_times: Dict[str, float] = {}
    if p1_csv.exists():
        try:
            p1_df = pd.read_csv(p1_csv)
            s0_df = p1_df[p1_df["seed"] == 0]
            for m in s0_df["model"].unique():
                sub = s0_df[s0_df["model"] == m]
                fit_times[m] = float(sub["train_time_seconds"].iloc[0])
        except Exception:
            pass

    n_features = 76
    batch_sizes = [1, 256]
    n_warmup = 20
    n_repeats = 200

    models_list = ["PCA", "IsolationForest", "LOF", "OneClassSVM", "MLP AE", "VAE"]

    # Pre-fit dummy models with synthetic benign training data to enable scoring
    rng = np.random.default_rng(0)
    X_dummy_train = rng.standard_normal((1000, n_features)).astype(np.float32)

    instantiated_models = {}

    pca = PCAReconstructionBaseline(n_components=10, random_state=0)
    pca.fit(X_dummy_train)
    instantiated_models["PCA"] = (pca.score, None)

    if_clf = IsolationForestBaseline(n_estimators=100, random_state=0, n_jobs=1)
    if_clf.fit(X_dummy_train)
    instantiated_models["IsolationForest"] = (if_clf.score, None)

    lof = LocalOutlierFactorBaseline(n_neighbors=20, novelty=True, n_jobs=1)
    lof.fit(X_dummy_train)
    instantiated_models["LOF"] = (lof.score, None)

    ocsvm = OneClassSVMBaseline(kernel="rbf", gamma="scale", nu=0.05)
    ocsvm.fit(X_dummy_train)
    instantiated_models["OneClassSVM"] = (ocsvm.score, None)

    ae = MLPAutoencoder(input_dim=n_features, hidden_dims=(64, 32, 16), bottleneck_dim=16, seed=0)
    ae.eval()
    instantiated_models["MLP AE"] = (ae.compute_reconstruction_error, count_parameters(ae))

    vae = VariationalAutoencoder(input_dim=n_features, hidden_dims=(64, 32), latent_dim=16, seed=0)
    vae.eval()
    instantiated_models["VAE"] = (vae.compute_reconstruction_error, count_parameters(vae))

    records: List[Dict[str, Any]] = []

    for m_name in models_list:
        score_fn, param_count = instantiated_models[m_name]
        fit_time_s1 = fit_times.get(m_name, None)

        for b_size in batch_sizes:
            X_bench = rng.standard_normal((b_size, n_features)).astype(np.float32)

            # Warm-up calls
            for _ in range(n_warmup):
                _ = score_fn(X_bench)

            # Timed repeats
            durations: List[float] = []
            for _ in range(n_repeats):
                t_start = time.perf_counter()
                _ = score_fn(X_bench)
                t_end = time.perf_counter()
                durations.append(t_end - t_start)

            durations_arr = np.array(durations)
            # Microseconds per flow
            us_per_flow = (durations_arr * 1e6) / b_size
            median_us = float(np.median(us_per_flow))
            p95_us = float(np.percentile(us_per_flow, 95.0))
            median_duration_batch = float(np.median(durations_arr))
            flows_per_sec = float(b_size / max(1e-12, median_duration_batch))

            rec = {
                "model": m_name,
                "batch_size": b_size,
                "num_features": n_features,
                "median_us_per_flow": round(median_us, 2),
                "p95_us_per_flow": round(p95_us, 2),
                "flows_per_second": round(flows_per_sec, 1),
                "parameter_count": param_count,
                "fit_wall_clock_s1": fit_time_s1,
                "cpu_model": cpu_info,
                "operating_system": os_info,
            }
            records.append(rec)

            print(
                f"  {m_name:<16} | B={b_size:<3} | Median: {median_us:8.2f} us/flow | "
                f"p95: {p95_us:8.2f} us/flow | Throughput: {flows_per_sec:10.1f} flows/s"
            )

    df_lat = pd.DataFrame(records)
    df_lat.to_csv(csv_path, index=False)
    print(f"\nSaved Stage 5 latency benchmark to {csv_path}")


if __name__ == "__main__":
    run_stage5()
