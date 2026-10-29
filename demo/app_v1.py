"""Streamlit interactive live demo for network flow anomaly detection.

DEV DATA, NOT FOR REPORTED RESULTS

Features:
- "Load sample flows" button (loads demo/demo_samples.csv) or CSV upload.
- Scores flows using the trained demo MLP Autoencoder.
- Table of flows with anomaly score and suspicious flag.
- Score distribution histogram with the calibrated POT alarm threshold line.
- Real-time summary counts: flagged vs total.
- Checkbox "Show evaluation (uses labels for display only)" showing detection rate
  per attack family and false-alarm rate on benign flows.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Setup repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = REPO_ROOT / "demo"
ARTIFACTS_DIR = DEMO_DIR / "artifacts"
DEFAULT_SAMPLES_PATH = DEMO_DIR / "demo_samples.csv"
DEFAULT_LABELS_PATH = DEMO_DIR / "demo_samples_labels.csv"

STATUS_TAG = "DEV DATA, NOT FOR REPORTED RESULTS"


def get_artifacts_status() -> dict:
    """Check existence of demo artifacts."""
    weights = ARTIFACTS_DIR / "model_weights.pt"
    prep = ARTIFACTS_DIR / "preprocessor.joblib"
    thresh = ARTIFACTS_DIR / "threshold.json"
    ready = weights.exists() and prep.exists() and thresh.exists()
    info = {}
    if thresh.exists():
        try:
            with open(thresh, "r", encoding="utf-8") as f:
                info = json.load(f)
        except Exception:
            pass
    return {"ready": ready, "info": info}


def load_calibration_scores(artifacts_dir: Path = ARTIFACTS_DIR) -> Optional[np.ndarray]:
    """Load pre-scored benign calibration flows used for POT and empirical thresholds."""
    calib_file = artifacts_dir / "calibration_scores.npy"
    if calib_file.exists():
        try:
            return np.load(calib_file)
        except Exception:
            pass
    return None


def compute_threshold_for_q(
    q: float,
    pot_threshold: float,
    calib_scores: Optional[np.ndarray] = None,
) -> float:
    """Compute alarm threshold for false-alarm budget q.

    Default q=0.01 retains the POT threshold. For other q values (0.02, 0.05, 0.10),
    the threshold is computed as the (1 - q) percentile of the benign calibration scores.
    """
    if np.isclose(q, 0.01):
        return float(pot_threshold)
    if calib_scores is not None and len(calib_scores) > 0:
        pct = (1.0 - float(q)) * 100.0
        return float(np.percentile(calib_scores, pct))
    return float(pot_threshold)


def render_page() -> None:
    st.set_page_config(
        page_title="Network Flow Anomaly Detection Demo",
        page_icon="🛡️",
        layout="wide",
    )

    st.caption(f"🔒 [{STATUS_TAG}]")
    st.title("🛡️ Live Network Flow Anomaly Detection")
    st.markdown(
        "Demonstration of a lightweight **MLP Autoencoder** trained strictly on benign network flows. "
        "The alarm threshold is calibrated using **Extreme Value Theory (POT)** on held-out benign traffic "
        "for a guaranteed nominal false alarm budget (q = 1%)."
    )

    status = get_artifacts_status()
    pot_threshold_val = float(status["info"].get("threshold", 0.0)) if status["ready"] else 0.0
    calib_scores = load_calibration_scores(ARTIFACTS_DIR)

    # Sidebar: Model and Artifact Information
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        if status["ready"]:
            st.success("✅ Model Artifacts Loaded")
            st.subheader("🎯 False-Alarm Budget")
            q_options = [0.01, 0.02, 0.05, 0.10]
            q_selected = st.select_slider(
                "False-alarm budget (q)",
                options=q_options,
                value=0.01,
                format_func=lambda x: f"{x:.2f}",
                help="Select false-alarm budget (q). Default q=0.01 uses POT calibration; other options compute the (1-q) empirical percentile of benign calibration scores without retraining.",
            )
            current_threshold = compute_threshold_for_q(
                q=q_selected,
                pot_threshold=pot_threshold_val,
                calib_scores=calib_scores,
            )
            thresh_title = "Alarm Threshold (POT q=0.01)" if np.isclose(q_selected, 0.01) else f"Alarm Threshold (p={(1-q_selected)*100:.1f}%)"
            st.metric(thresh_title, f"{current_threshold:.5f}")
            st.write(f"**False-Alarm Budget (q):** {q_selected:.2f}")
            st.write(f"**Calibration Rows:** {status['info'].get('calibration_rows', 'N/A'):,}")
            st.write(f"**Architecture:** MLP (64 → 32 → 16 → 32 → 64)")
        else:
            q_selected = 0.01
            current_threshold = 0.0
            st.warning("⚠️ Model Artifacts Missing")
            st.info(
                "Run `python demo/train_demo_model.py` to generate the demo model and threshold."
            )

        st.markdown("---")
        st.markdown(f"**Status:** `{STATUS_TAG}`")
        st.markdown("Labels are strictly forbidden during inference.")

    # Main input section: Sample loader and file uploader
    col1, col2 = st.columns([1, 2])
    with col1:
        load_sample = st.button("📁 Load sample flows (demo_samples.csv)", use_container_width=True)
    with col2:
        uploaded_file = st.file_uploader(
            "Or upload custom flow CSV (raw features, unlabelled)",
            type=["csv"],
            help="Upload a CSV containing network flow features.",
        )

    # State management for loaded flows
    if "current_df" not in st.session_state:
        st.session_state.current_df = None
    if "data_source" not in st.session_state:
        st.session_state.data_source = None

    if load_sample:
        if DEFAULT_SAMPLES_PATH.exists():
            st.session_state.current_df = pd.read_csv(DEFAULT_SAMPLES_PATH)
            st.session_state.data_source = "demo_samples"
            st.success(f"Loaded {len(st.session_state.current_df):,} held-out sample flows.")
        else:
            st.error(
                f"Sample flows file not found at `{DEFAULT_SAMPLES_PATH}`. "
                f"Please run `python demo/make_demo_samples.py` first."
            )

    elif uploaded_file is not None:
        try:
            st.session_state.current_df = pd.read_csv(uploaded_file)
            st.session_state.data_source = "uploaded"
            st.success(f"Uploaded {len(st.session_state.current_df):,} flows.")
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")

    df_flows = st.session_state.current_df

    if df_flows is None:
        st.info("👆 Click **'Load sample flows'** or upload a CSV file above to begin.")
        return

    if not status["ready"]:
        st.error(
            "Demo model artifacts not found. Please run `python demo/train_demo_model.py` "
            "before scoring flows."
        )
        return

    # Scoring step (Import score_flows dynamically)
    from demo.score_flows import score_flows

    try:
        scored_result = score_flows(df_flows, artifacts_dir=ARTIFACTS_DIR)
        scores = scored_result.scores
    except ValueError as ve:
        st.error(f"Input Validation Error: {ve}")
        return
    except Exception as ex:
        st.error(f"Scoring Error: {ex}")
        return

    # Recompute flags using current_threshold corresponding to q_selected (no retraining)
    flags = scores > current_threshold

    total_flows = len(df_flows)
    n_flagged = int(np.sum(flags))
    pct_flagged = (n_flagged / max(1, total_flows)) * 100.0

    # Display Top Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Flows", f"{total_flows:,}")
    m2.metric("Flagged Anomalies", f"{n_flagged:,}", f"{pct_flagged:.1f}%")
    m3.metric("Benign-Rated Flows", f"{total_flows - n_flagged:,}", f"{100.0 - pct_flagged:.1f}%")
    m4_title = "POT Threshold (q=0.01)" if np.isclose(q_selected, 0.01) else f"Alarm Threshold (q={q_selected:.2f})"
    m4.metric(m4_title, f"{current_threshold:.5f}")

    # Visualizations: Score Histogram on log scale
    st.markdown("### 📊 Reconstruction Error Distribution")
    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=100)
    pos = scores[scores > 0]
    s_min = float(pos.min()) if len(pos) > 0 else 1e-6
    s_max = float(max(scores.max(), current_threshold * 1.5))
    num_bins = min(50, max(20, total_flows // 15))
    log_bins = np.logspace(np.log10(s_min), np.log10(s_max), num=num_bins)

    ax.hist(
        scores,
        bins=log_bins,
        color="#2b5c8f",
        edgecolor="#1b3b5c",
        alpha=0.85,
        label="Flow Anomaly Scores",
    )
    thresh_leg = f"Alarm Threshold ({current_threshold:.4f}, q={q_selected:.2f})"
    ax.axvline(
        current_threshold,
        color="#e63946",
        linestyle="--",
        linewidth=2.2,
        label=thresh_leg,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Reconstruction MSE Anomaly Score (log scale)", fontsize=10)
    ax.set_ylabel("Flow Count", fontsize=10)
    ax.set_title(
        f"Flow Reconstruction Scores vs. Calibrated Alarm Threshold [{STATUS_TAG}]",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.5, which="both")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Scored Dataframe Table
    st.markdown("### 📋 Scored Network Flows")
    df_display = df_flows.copy()
    df_display.insert(0, "Suspicious Flag", flags)
    df_display.insert(1, "Anomaly Score", np.round(scores, 5))

    st.dataframe(
        df_display,
        use_container_width=True,
        height=320,
    )

    # Download scored flows
    csv_bytes = df_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Scored Flows CSV",
        data=csv_bytes,
        file_name="flows_scored.csv",
        mime="text/csv",
    )

    # Evaluation Section
    st.markdown("---")
    show_eval = st.checkbox("🔍 Show evaluation (uses labels for display only)")

    if show_eval:
        st.subheader("🎯 Ground-Truth Evaluation Breakdown")
        labels_df: Optional[pd.DataFrame] = None

        if st.session_state.data_source == "demo_samples":
            if DEFAULT_LABELS_PATH.exists():
                labels_df = pd.read_csv(DEFAULT_LABELS_PATH)
            else:
                st.warning(
                    f"Labels file `{DEFAULT_LABELS_PATH}` not found. "
                    f"Please run `python demo/make_demo_samples.py`."
                )
        else:
            labels_file = st.file_uploader(
                "Upload ground-truth labels CSV (with Attack_Family or Label column)",
                type=["csv"],
                key="labels_uploader",
            )
            if labels_file:
                labels_df = pd.read_csv(labels_file)

        if labels_df is not None:
            if len(labels_df) != total_flows:
                st.error(
                    f"Labels row count ({len(labels_df):,}) does not match "
                    f"flows count ({total_flows:,})."
                )
            else:
                family_col = (
                    "Attack_Family"
                    if "Attack_Family" in labels_df.columns
                    else "Label" if "Label" in labels_df.columns else None
                )

                if family_col is None:
                    st.error("Labels CSV must contain an 'Attack_Family' or 'Label' column.")
                else:
                    eval_df = pd.DataFrame(
                        {
                            "Family": labels_df[family_col].values,
                            "Flagged": flags,
                        }
                    )

                    # Group statistics
                    stats = []
                    for fam, grp in eval_df.groupby("Family"):
                        count = len(grp)
                        flagged_grp = grp["Flagged"].sum()
                        rate = (flagged_grp / count) * 100.0
                        role = (
                            f"Benign (FPR Target <= {q_selected * 100:.1f}%)"
                            if str(fam).lower() == "benign"
                            else "Attack Family"
                        )
                        metric_name = "False Alarm Rate" if str(fam).lower() == "benign" else "Detection Rate"
                        stats.append(
                            {
                                "Traffic Family": fam,
                                "Type": role,
                                "Total Flows": count,
                                "Flagged Suspicious": flagged_grp,
                                f"{metric_name} (%)": f"{rate:.1f}%",
                                "Raw Rate": rate,
                            }
                        )

                    summary_df = pd.DataFrame(stats)
                    # Sort so Benign appears first, then attacks alphabetically
                    summary_df["sort_key"] = summary_df["Traffic Family"].apply(
                        lambda x: 0 if str(x).lower() == "benign" else 1
                    )
                    summary_df = summary_df.sort_values(
                        by=["sort_key", "Traffic Family"]
                    ).drop(columns=["sort_key", "Raw Rate"])

                    st.table(summary_df)

                    # Highlight Key Insights
                    benign_grp = eval_df[eval_df["Family"].str.lower() == "benign"]
                    attack_grp = eval_df[eval_df["Family"].str.lower() != "benign"]

                    c1, c2 = st.columns(2)
                    if len(benign_grp) > 0:
                        benign_far = (benign_grp["Flagged"].sum() / len(benign_grp)) * 100.0
                        c1.metric(
                            "Benign False-Alarm Rate (FAR)",
                            f"{benign_far:.2f}%",
                            f"Target Budget (q): ~{q_selected * 100:.2f}%",
                        )
                    if len(attack_grp) > 0:
                        overall_tpr = (attack_grp["Flagged"].sum() / len(attack_grp)) * 100.0
                        c2.metric(
                            "Overall Attack Detection Rate (TPR)",
                            f"{overall_tpr:.1f}%",
                            f"{attack_grp['Flagged'].sum()} / {len(attack_grp)} attacks",
                        )


# Streamlit entry point
if __name__ == "__main__" or "STREAMLIT_RUN" in os.environ:
    render_page()
