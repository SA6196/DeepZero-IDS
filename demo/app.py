"""Streamlit interactive live demo for network flow anomaly detection.

DEV DATA, NOT FOR REPORTED RESULTS

Features:
- "Load sample flows" button (loads demo/demo_samples.csv) or CSV upload.
- Scores flows using the trained demo MLP Autoencoder.
- Table of flows with anomaly score and suspicious flag.
- Interactive Plotly score distribution histogram with calibrated alarm threshold.
- Real-time summary counts: flagged vs total.
- Checkbox "Show evaluation (uses labels for display only)" showing detection rate
  per attack family and false-alarm rate on benign flows.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Setup repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEMO_DIR = REPO_ROOT / "demo"
ARTIFACTS_DIR = DEMO_DIR / "artifacts"
DEFAULT_SAMPLES_PATH = DEMO_DIR / "demo_samples.csv"
DEFAULT_LABELS_PATH = DEMO_DIR / "demo_samples_labels.csv"

STATUS_TAG = "DEV DATA, NOT FOR REPORTED RESULTS"

CUSTOM_CSS = """
<style>
    /* Security Operations Dark Theme */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
        max-width: 95%;
    }
    
    /* Metric Cards */
    .sec-metric-card {
        background: #101a30;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        margin-bottom: 8px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .sec-metric-card:hover {
        border-color: rgba(0, 245, 212, 0.35);
        transform: translateY(-2px);
    }
    .sec-metric-card-alert {
        border-left: 4px solid #ef476f !important;
        background: linear-gradient(135deg, #101a30 0%, #1e1124 100%);
    }
    .sec-metric-card-benign {
        border-left: 4px solid #06d6a0 !important;
    }
    .sec-metric-card-thresh {
        border-left: 4px solid #ffbe0b !important;
    }
    .sec-metric-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #8da4c4;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .sec-metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 4px;
    }
    .sec-metric-sub {
        font-size: 0.78rem;
        color: #94a3b8;
    }
    
    /* Operational Verdict Banner */
    .verdict-banner {
        background: linear-gradient(90deg, rgba(6, 214, 160, 0.12) 0%, rgba(16, 26, 48, 0.5) 100%);
        border-left: 4px solid #06d6a0;
        border-radius: 6px;
        padding: 12px 18px;
        margin: 12px 0 18px 0;
        font-size: 0.95rem;
        color: #f1f5f9;
    }
</style>
"""


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

    # Inject custom Security Operations CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Top Header: Title + one-line subtitle + small DEV DATA badge
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.markdown(
            """
            <div style="display: flex; align-items: baseline; gap: 14px; margin-bottom: 2px;">
                <span style="font-size: 1.85rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.5px;">
                    🛡️ Live Network Flow Anomaly Detection
                </span>
                <span style="background: rgba(255, 190, 11, 0.15); border: 1px solid #ffbe0b; color: #ffbe0b; padding: 3px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.5px;">
                    DEV DATA, NOT FOR REPORTED RESULTS
                </span>
            </div>
            <p style="margin: 0 0 10px 0; font-size: 0.92rem; color: #94a3b8;">
                Real-time unsupervised autoencoder flow reconstruction with extreme-value calibrated threat alarming.
            </p>
            """,
            unsafe_allow_html=True,
        )

    # "How to read this" expander with 3 plain sentences
    with st.expander("ℹ️ How to read this dashboard"):
        st.markdown(
            "1. **Reconstruction anomaly score:** A higher reconstruction error indicates that a flow deviates further from learned normal benign network patterns.\n"
            "2. **False-alarm budget (q):** Governs alarm sensitivity; a lower budget keeps false alarms down by setting a higher threshold, while a higher budget increases threat detection at the cost of more alarms.\n"
            "3. **Zero-label scoring:** Traffic is scored and flagged strictly without accessing ground-truth labels; labels are only loaded if the evaluation checkbox is manually ticked."
        )

    status = get_artifacts_status()
    pot_threshold_val = float(status["info"].get("threshold", 0.0)) if status["ready"] else 0.0
    calib_scores = load_calibration_scores(ARTIFACTS_DIR)

    # Sidebar: Model and Artifact Information
    with st.sidebar:
        st.markdown("### ⚙️ SecOps Engine")
        if status["ready"]:
            st.success("✅ Model Artifacts Loaded")
            st.markdown("---")
            st.markdown("#### 🎯 False-Alarm Budget (q)")
            q_options = [0.01, 0.02, 0.05, 0.10]
            q_selected = st.select_slider(
                "False-alarm budget (q)",
                options=q_options,
                value=0.01,
                format_func=lambda x: f"{x:.2f}",
                label_visibility="collapsed",
            )
            st.caption(
                "⚖️ **Trade-off:** Lower *q* minimizes benign false alarms with a conservative threshold, but subtle stealth attacks may slip through. "
                "Higher *q* lowers the threshold to catch more stealthy intrusions at the cost of higher benign alarm rates."
            )

            current_threshold = compute_threshold_for_q(
                q=q_selected,
                pot_threshold=pot_threshold_val,
                calib_scores=calib_scores,
            )

            thresh_type = "POT (q=0.01)" if np.isclose(q_selected, 0.01) else f"Empirical p={(1-q_selected)*100:.1f}%"
            st.metric(
                f"Active Threshold ({thresh_type})",
                f"{current_threshold:.5f}",
            )
            st.write(f"**Target Budget (q):** `{q_selected:.2f}`")
            st.write(f"**Calibration Rows:** `{status['info'].get('calibration_rows', 'N/A'):,}`")
            st.write(f"**Architecture:** `MLP (64 → 32 → 16 → 32 → 64)`")
        else:
            q_selected = 0.01
            current_threshold = 0.0
            st.warning("⚠️ Model Artifacts Missing")
            st.info(
                "Run `python demo/train_demo_model.py` to generate the demo model and threshold."
            )

        st.markdown("---")
        st.caption(f"🔒 [{STATUS_TAG}]")
        st.caption("Zero-label operational boundary strictly enforced.")

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
        st.info("👆 Click **'Load sample flows'** or upload a CSV file above to begin scoring.")
        return

    if not status["ready"]:
        st.error(
            "Demo model artifacts not found. Please run `python demo/train_demo_model.py` "
            "before scoring flows."
        )
        return

    # Scoring step (Import score_flows dynamically, labels strictly forbidden)
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
    n_benign = total_flows - n_flagged
    pct_benign = (n_benign / max(1, total_flows)) * 100.0

    # Display 4 Big Metric Cards
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        st.markdown(
            f"""
            <div class="sec-metric-card">
                <div class="sec-metric-label">TOTAL FLOWS</div>
                <div class="sec-metric-value" style="color: #f8fafc;">{total_flows:,}</div>
                <div class="sec-metric-sub">Ingested traffic volume</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_m2:
        st.markdown(
            f"""
            <div class="sec-metric-card sec-metric-card-alert">
                <div class="sec-metric-label" style="color: #ff8da1;">FLAGGED ANOMALIES</div>
                <div class="sec-metric-value" style="color: #ef476f;">{n_flagged:,}</div>
                <div class="sec-metric-sub" style="color: #ff8da1;">{pct_flagged:.1f}% suspicious traffic</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_m3:
        st.markdown(
            f"""
            <div class="sec-metric-card sec-metric-card-benign">
                <div class="sec-metric-label" style="color: #5eead4;">BENIGN-RATED</div>
                <div class="sec-metric-value" style="color: #06d6a0;">{n_benign:,}</div>
                <div class="sec-metric-sub" style="color: #5eead4;">{pct_benign:.1f}% normal traffic</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_m4:
        thresh_sub = "POT calibrated (q=0.01)" if np.isclose(q_selected, 0.01) else f"Empirical (1-q) p={(1-q_selected)*100:.1f}%"
        st.markdown(
            f"""
            <div class="sec-metric-card sec-metric-card-thresh">
                <div class="sec-metric-label" style="color: #fde047;">CURRENT THRESHOLD</div>
                <div class="sec-metric-value" style="color: #ffbe0b;">{current_threshold:.5f}</div>
                <div class="sec-metric-sub" style="color: #fde047;">{thresh_sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Interactive Plotly Score Distribution Histogram (Log x-axis)
    st.markdown("### 📊 Reconstruction Error Distribution (Log Scale)")

    pos = scores[scores > 0]
    s_min = float(pos.min()) if len(pos) > 0 else 1e-5
    s_max = float(max(scores.max(), current_threshold * 1.5))
    num_bins = min(45, max(20, total_flows // 16))
    log_bins = np.logspace(np.log10(s_min), np.log10(s_max), num=num_bins)
    bin_centers = np.sqrt(log_bins[:-1] * log_bins[1:])

    h_benign, _ = np.histogram(scores[~flags], bins=log_bins)
    h_flagged, _ = np.histogram(scores[flags], bins=log_bins)

    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Bar(
            x=bin_centers,
            y=h_benign,
            name="Normal / Benign-Rated (< Threshold)",
            marker_color="#06d6a0",
            marker_line_color="#0b132b",
            marker_line_width=1,
            opacity=0.85,
            customdata=np.stack([log_bins[:-1], log_bins[1:]], axis=-1),
            hovertemplate="Score range: %{customdata[0]:.4f} - %{customdata[1]:.4f}<br>Flows: %{y}<extra></extra>",
        )
    )
    fig_hist.add_trace(
        go.Bar(
            x=bin_centers,
            y=h_flagged,
            name="Suspicious Flag (≥ Threshold)",
            marker_color="#ef476f",
            marker_line_color="#0b132b",
            marker_line_width=1,
            opacity=0.85,
            customdata=np.stack([log_bins[:-1], log_bins[1:]], axis=-1),
            hovertemplate="Score range: %{customdata[0]:.4f} - %{customdata[1]:.4f}<br>Flows: %{y}<extra></extra>",
        )
    )
    fig_hist.add_vline(
        x=current_threshold,
        line_dash="dash",
        line_color="#ffbe0b",
        line_width=2.5,
        annotation_text=f"Alarm Threshold ({current_threshold:.5f}, q={q_selected:.2f})",
        annotation_position="top right",
        annotation_font=dict(color="#ffbe0b", size=11, family="monospace"),
    )
    fig_hist.update_xaxes(
        type="log",
        title_text="Reconstruction Error MSE (Log Scale — Benign flows clearly resolved)",
        gridcolor="#1b2a4a",
        zeroline=False,
    )
    fig_hist.update_yaxes(
        title_text="Flow Count",
        gridcolor="#1b2a4a",
        zeroline=False,
    )
    fig_hist.update_layout(
        barmode="stack",
        template="plotly_dark",
        paper_bgcolor="#0b132b",
        plot_bgcolor="#101a30",
        margin=dict(l=40, r=40, t=35, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        height=350,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Scored Network Flows Table
    st.markdown("### 📋 Scored Network Flows")

    # Define the 8 most useful columns first
    useful_cols = [
        "Suspicious Flag",
        "Anomaly Score",
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Flow Bytes/s",
        "Flow Packets/s",
    ]
    ordered_cols = [c for c in useful_cols if c in ["Suspicious Flag", "Anomaly Score"] or c in df_flows.columns]
    remaining_cols = [c for c in df_flows.columns if c not in ordered_cols]
    all_ordered_cols = ordered_cols + remaining_cols

    df_display = df_flows.copy()
    df_display.insert(0, "Anomaly Score", np.round(scores, 5))
    df_display.insert(0, "Suspicious Flag", np.where(flags, "🚨 FLAGGED", "🟢 BENIGN"))

    # Sort by score descending by default
    df_sorted = df_display[all_ordered_cols].sort_values(by="Anomaly Score", ascending=False)

    def style_flag(val):
        if "FLAGGED" in str(val):
            return "background-color: rgba(239, 71, 111, 0.25); color: #ff6b8b; font-weight: bold;"
        return "background-color: rgba(6, 214, 160, 0.18); color: #06d6a0; font-weight: bold;"

    styled_table = (
        df_sorted.style
        .map(style_flag, subset=["Suspicious Flag"])
        .background_gradient(subset=["Anomaly Score"], cmap="YlOrRd", vmin=float(scores.min()), vmax=float(scores.max()))
        .format({"Anomaly Score": "{:.5f}"})
    )

    st.dataframe(
        styled_table,
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

    # Evaluation Section (Labels isolated strictly inside this checkbox)
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
                        flagged_grp = int(grp["Flagged"].sum())
                        rate = (flagged_grp / count) * 100.0
                        role = (
                            f"Benign (FPR Target ≤ {q_selected * 100:.1f}%)"
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
                    summary_df["sort_key"] = summary_df["Traffic Family"].apply(
                        lambda x: 0 if str(x).lower() == "benign" else 1
                    )
                    summary_df = summary_df.sort_values(
                        by=["sort_key", "Traffic Family"]
                    )

                    benign_grp = eval_df[eval_df["Family"].str.lower() == "benign"]
                    attack_grp = eval_df[eval_df["Family"].str.lower() != "benign"]
                    benign_far = (benign_grp["Flagged"].sum() / len(benign_grp)) * 100.0 if len(benign_grp) > 0 else 0.0
                    overall_tpr = (attack_grp["Flagged"].sum() / len(attack_grp)) * 100.0 if len(attack_grp) > 0 else 0.0

                    # 1. False-alarm card and Overall detection card
                    c_ev1, c_ev2 = st.columns(2)
                    with c_ev1:
                        far_status = "✅ Within budget" if benign_far <= q_selected * 100 * 1.5 else "⚠️ Above budget"
                        st.markdown(
                            f"""
                            <div class="sec-metric-card sec-metric-card-benign">
                                <div class="sec-metric-label">BENIGN FALSE-ALARM RATE (FAR)</div>
                                <div class="sec-metric-value" style="color: #06d6a0;">{benign_far:.2f}%</div>
                                <div class="sec-metric-sub">Target budget: ~{q_selected * 100:.2f}% &nbsp;|&nbsp; {benign_grp['Flagged'].sum()} / {len(benign_grp)} benign flows flagged ({far_status})</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with c_ev2:
                        st.markdown(
                            f"""
                            <div class="sec-metric-card sec-metric-card-alert">
                                <div class="sec-metric-label">OVERALL ATTACK DETECTION RATE (TPR)</div>
                                <div class="sec-metric-value" style="color: #ef476f;">{overall_tpr:.1f}%</div>
                                <div class="sec-metric-sub">{attack_grp['Flagged'].sum()} / {len(attack_grp)} attack flows flagged as anomalous</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # 2. One-line plain-English verdict
                    attack_stats = [s for s in stats if str(s["Traffic Family"]).lower() != "benign"]
                    caught_fams = [s["Traffic Family"] for s in attack_stats if s["Raw Rate"] >= 50.0]
                    missed_fams = [s["Traffic Family"] for s in attack_stats if s["Raw Rate"] < 50.0]

                    if len(caught_fams) > 0 and len(missed_fams) > 0:
                        caught_str = ", ".join(caught_fams)
                        missed_str = ", ".join(missed_fams)
                        verdict_text = f"At q = {q_selected*100:.0f}%, {overall_tpr:.0f}% of attacks are flagged; {caught_str} is caught, subtler families ({missed_str}) are not."
                    elif len(caught_fams) > 0 and len(missed_fams) == 0:
                        verdict_text = f"At q = {q_selected*100:.0f}%, {overall_tpr:.0f}% of attacks are flagged; all tested attack families are successfully detected."
                    else:
                        verdict_text = f"At q = {q_selected*100:.0f}%, {overall_tpr:.0f}% of attacks are flagged; high-threshold conservative alarming lets subtle threats pass."

                    st.markdown(
                        f"""
                        <div class="verdict-banner">
                            <strong>Operational Verdict:</strong> {verdict_text}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # 3. Horizontal bar chart of detection rate per attack family
                    chart_fams = summary_df["Traffic Family"].tolist()
                    chart_rates = summary_df["Raw Rate"].tolist()
                    chart_counts = summary_df["Total Flows"].tolist()
                    chart_flagged = summary_df["Flagged Suspicious"].tolist()
                    chart_colors = [
                        "#06d6a0" if str(f).lower() == "benign" else "#ef476f" for f in chart_fams
                    ]

                    fig_bar = go.Figure(
                        go.Bar(
                            x=chart_rates,
                            y=chart_fams,
                            orientation="h",
                            marker_color=chart_colors,
                            marker_line_color="#0b132b",
                            marker_line_width=1,
                            text=[f"{r:.1f}% ({flg}/{cnt})" for r, flg, cnt in zip(chart_rates, chart_flagged, chart_counts)],
                            textposition="auto",
                            textfont=dict(color="#f8fafc", size=11, family="sans-serif"),
                        )
                    )
                    fig_bar.add_vline(
                        x=q_selected * 100.0,
                        line_dash="dot",
                        line_color="#ffbe0b",
                        line_width=2,
                        annotation_text=f"FAR budget: {q_selected*100:.1f}%",
                        annotation_position="top right",
                        annotation_font=dict(color="#ffbe0b", size=10),
                    )
                    fig_bar.update_xaxes(
                        title_text="Alarm / Detection Rate (%)",
                        range=[0, 105],
                        gridcolor="#1b2a4a",
                        zeroline=False,
                    )
                    fig_bar.update_yaxes(
                        gridcolor="#1b2a4a",
                        autorange="reversed",
                    )
                    fig_bar.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#0b132b",
                        plot_bgcolor="#101a30",
                        margin=dict(l=40, r=40, t=25, b=40),
                        height=280,
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                    # Detailed Table
                    table_display = summary_df.drop(columns=["sort_key", "Raw Rate"])
                    st.table(table_display)


# Streamlit entry point
if __name__ == "__main__" or "STREAMLIT_RUN" in os.environ:
    render_page()
