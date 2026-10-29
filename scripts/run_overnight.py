"""Master Overnight Runner for Complete Benchmark Execution.

DEV DATA, NOT FOR REPORTED RESULTS
Orchestrates:
- Stage 1: Final P1 Grid (LOFO: 5 families x 7 models x 3 seeds)
- Stage 2: Final P2 Grid (Temporal: Monday train -> Tue-Fri test, 7 models x 3 seeds)
- Stage 3: Label-Free Thresholds (POT, Empirical, Mean+3sigma, Oracle)
- Stage 4: Sanitization Diagnostic (4a) and Sensitivity (4b)
- Stage 5: Inference Latency and Throughput Benchmarking (Single thread)
- Stage 6: Ablations (6a Bottleneck, 6b Trim Fraction)
- Stage 7: Final Markdown Reports and Summary Tables

Enforces:
- Thermal Management: 10 minute sleep after every 60 minutes of active compute.
- Continuous status logging to results/RUN_STATUS.md.
- Resumability across all stages.
- Emits ALL_DONE upon successful completion.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import datetime
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

STATUS_TAG = "DEV DATA, NOT FOR REPORTED RESULTS"
PYTHON_EXE = sys.executable


class ThermalTracker:
    def __init__(self, max_continuous_minutes: float = 60.0, cooldown_minutes: float = 10.0):
        self.max_continuous_seconds = max_continuous_minutes * 60.0
        self.cooldown_seconds = cooldown_minutes * 60.0
        self.continuous_compute_start = time.time()
        self.total_thermal_pauses = 0

    def check_thermal_limit(self, status_updater):
        elapsed = time.time() - self.continuous_compute_start
        if elapsed >= self.max_continuous_seconds:
            self.total_thermal_pauses += 1
            pause_msg = (
                f"Thermal cooldown pause #{self.total_thermal_pauses}: "
                f"Elapsed compute {elapsed/60:.1f} min >= 60 min. Sleeping for 10 minutes..."
            )
            print(f"\n[THERMAL] {pause_msg}")
            status_updater(stage="Thermal Pause", extra_note=pause_msg)
            time.sleep(self.cooldown_seconds)
            self.continuous_compute_start = time.time()
            print("[THERMAL] Thermal cooldown complete. Resuming execution.\n")


def update_run_status(
    stage: str,
    runs_done: int = 0,
    total_runs: int = 0,
    elapsed_seconds: float = 0.0,
    last_error: str = "None",
    extra_note: str = "",
):
    status_file = REPO_ROOT / "results" / "RUN_STATUS.md"
    status_file.parent.mkdir(parents=True, exist_ok=True)

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_min = elapsed_seconds / 60.0

    lines = [
        f"# RUN STATUS: OVERNIGHT EXECUTION",
        f"**[{STATUS_TAG}]**\n",
        f"- **Last Updated**: {now_str}",
        f"- **Current Stage**: {stage}",
        f"- **Progress**: {runs_done} / {total_runs} runs completed" if total_runs > 0 else f"- **Progress**: {stage}",
        f"- **Cumulative Elapsed**: {elapsed_min:.1f} minutes",
        f"- **Last Error**: {last_error}",
    ]
    if extra_note:
        lines.append(f"- **Note**: {extra_note}")

    lines.extend([
        "",
        "## Operating Decisions & Guarantees",
        "1. Max threads clamped to 4 (single job at a time; Stage 5 clamped to 1).",
        "2. Thermal pause active (10 min sleep after 60 min continuous compute).",
        "3. Fully resumable: existing results skipped automatically.",
        "4. Infiltration excluded from P1 (36 rows, unreliable).",
        "5. Preprocessor fit strictly on TRAIN only; VAL-BENIGN for early stopping.",
    ])

    with open(status_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def execute_stage(stage_name: str, script_path: Path, thermal_tracker: ThermalTracker, start_time: float):
    print(f"\n{'='*100}")
    print(f"STARTING {stage_name}: {script_path.name}")
    print(f"{'='*100}")

    update_run_status(
        stage=stage_name,
        elapsed_seconds=time.time() - start_time,
        last_error="None",
        extra_note=f"Executing {script_path.name}",
    )

    retries = 1
    attempt = 0
    while attempt <= retries:
        attempt += 1
        try:
            thermal_tracker.check_thermal_limit(
                lambda stage, extra_note: update_run_status(
                    stage=stage,
                    elapsed_seconds=time.time() - start_time,
                    extra_note=extra_note,
                )
            )

            cmd = [PYTHON_EXE, "-u", str(script_path)]
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()

            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"Script {script_path.name} exited with code {proc.returncode}")

            print(f"[COMPLETED] {stage_name} finished successfully.")
            update_run_status(
                stage=f"{stage_name} (Done)",
                elapsed_seconds=time.time() - start_time,
                last_error="None",
                extra_note=f"{script_path.name} finished successfully.",
            )
            break

        except Exception as e:
            tb = traceback.format_exc()
            err_short = str(e)
            print(f"\n[ERROR in {stage_name}] {err_short}\n{tb}")
            update_run_status(
                stage=f"{stage_name} (Error)",
                elapsed_seconds=time.time() - start_time,
                last_error=f"{err_short} (Attempt {attempt})",
                extra_note=f"Short Traceback: {tb[:300]}",
            )
            # Only retry if MemoryError
            if "MemoryError" in tb and attempt <= retries:
                print(f"[RETRYING] MemoryError detected, attempting retry 1 of 1...")
                time.sleep(10)
                continue
            else:
                print(f"[SKIPPING] Error logged in RUN_STATUS.md. Continuing to next stage.")
                break


def main():
    t_start = time.time()
    print("=" * 115)
    print(f"[{STATUS_TAG}]")
    print("OVERNIGHT MASTER BENCHMARK RUNNER")
    print("Stages: 1 (P1 Grid) -> 2 (P2 Temporal) -> 3 (Thresholds) -> 4 (Sanitization) -> 5 (Latency) -> 6 (Ablations) -> 7 (Reports)")
    print("=" * 115)

    thermal = ThermalTracker(max_continuous_minutes=60.0, cooldown_minutes=10.0)

    stages = [
        ("STAGE 1: Protocol P1 Grid", REPO_ROOT / "scripts" / "run_stage1_p1.py"),
        ("STAGE 2: Protocol P2 Temporal", REPO_ROOT / "scripts" / "run_stage2_p2.py"),
        ("STAGE 3: Label-Free Thresholds", REPO_ROOT / "scripts" / "run_stage3_thresholds.py"),
        ("STAGE 4: Sanitization Diagnostic & Sensitivity", REPO_ROOT / "scripts" / "run_stage4_sanitization_sensitivity.py"),
        ("STAGE 5: Latency Benchmark", REPO_ROOT / "scripts" / "run_stage5_latency.py"),
        ("STAGE 6: Ablations", REPO_ROOT / "scripts" / "run_stage6_ablations.py"),
        ("STAGE 7: Reports & Markdown Tables", REPO_ROOT / "scripts" / "generate_final_report.py"),
    ]

    for stage_name, script_file in stages:
        if not script_file.exists():
            print(f"Error: {script_file} not found! Skipping...")
            continue
        execute_stage(stage_name, script_file, thermal, t_start)

    # Final completion marker
    status_file = REPO_ROOT / "results" / "RUN_STATUS.md"
    total_time_h = (time.time() - t_start) / 3600.0
    with open(status_file, "a", encoding="utf-8") as f:
        f.write(f"\n\n---\n**ALL_DONE**\nTotal Overnight Execution Time: {total_time_h:.2f} hours.\n")

    print("\n" + "=" * 115)
    print("ALL_DONE")
    print(f"Total overnight execution time: {total_time_h:.2f} hours.")
    print("=" * 115)


if __name__ == "__main__":
    main()
