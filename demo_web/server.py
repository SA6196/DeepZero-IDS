"""FastAPI server for Zero-Day Flow Anomaly Detection web dashboard.

DEV DATA - NOT FOR REPORTED RESULTS
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.score_flows import score_flows

DEMO_DIR = REPO_ROOT / "demo"
ARTIFACTS_DIR = DEMO_DIR / "artifacts"
STATIC_DIR = REPO_ROOT / "demo_web" / "static"
DEFAULT_SAMPLES_PATH = DEMO_DIR / "demo_samples.csv"
DEFAULT_LABELS_PATH = DEMO_DIR / "demo_samples_labels.csv"

STATUS_TAG = "DEV DATA - NOT FOR REPORTED RESULTS"

app = FastAPI(
    title="Zero-Day Flow Anomaly Detector",
    description="Autoencoder trained on normal traffic only",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cached artifacts and state
thresh_info: Dict[str, Any] = {}
calib_scores: Optional[np.ndarray] = None
pot_threshold: float = 0.08813415505481022

cached_sample_df: Optional[pd.DataFrame] = None
cached_sample_scores: Optional[np.ndarray] = None

last_uploaded_df: Optional[pd.DataFrame] = None
last_uploaded_scores: Optional[np.ndarray] = None


def load_artifacts() -> None:
    """Load model artifacts and pre-compute calibration metrics."""
    global thresh_info, calib_scores, pot_threshold, cached_sample_df, cached_sample_scores

    thresh_file = ARTIFACTS_DIR / "threshold.json"
    if thresh_file.exists():
        try:
            with open(thresh_file, "r", encoding="utf-8") as f:
                thresh_info = json.load(f)
            pot_threshold = float(thresh_info.get("threshold", 0.08813415505481022))
        except Exception as e:
            print(f"Warning loading threshold.json: {e}")

    calib_file = ARTIFACTS_DIR / "calibration_scores.npy"
    if calib_file.exists():
        try:
            calib_scores = np.load(calib_file)
        except Exception as e:
            print(f"Warning loading calibration_scores.npy: {e}")

    # Pre-score sample flows for sub-10ms response times
    if DEFAULT_SAMPLES_PATH.exists():
        try:
            cached_sample_df = pd.read_csv(DEFAULT_SAMPLES_PATH)
            res = score_flows(cached_sample_df, artifacts_dir=ARTIFACTS_DIR)
            cached_sample_scores = res.scores
        except Exception as e:
            print(f"Warning pre-scoring sample flows: {e}")


load_artifacts()


def compute_threshold_for_q(q: float) -> float:
    """Compute empirical threshold for false-alarm budget q.

    Default q=0.01 retains the POT threshold (0.08813).
    For other q values (e.g. 0.02, 0.05, 0.10), threshold is (1 - q) percentile
    of held-out benign calibration scores.
    """
    if np.isclose(q, 0.01):
        return float(pot_threshold)
    if calib_scores is not None and len(calib_scores) > 0:
        pct = (1.0 - float(q)) * 100.0
        return float(np.percentile(calib_scores, pct))
    return float(pot_threshold)


class ScoreRequest(BaseModel):
    q: float = 0.01
    source: str = "sample"


KEY_FEATURE_COLS = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Packet Length Mean",
    "SYN Flag Count",
]


def format_flows_for_response(
    df: pd.DataFrame, scores: np.ndarray, flags: np.ndarray, top_n: int = 50
) -> List[Dict[str, Any]]:
    """Format top flows sorted by score descending for table display."""
    indices = np.argsort(-scores)[:top_n]
    max_score = float(np.max(scores)) if len(scores) > 0 else 1.0

    flows_list = []
    for rank, idx in enumerate(indices, start=1):
        score_val = float(scores[idx])
        is_flagged = bool(flags[idx])
        pct_bar = min(100.0, max(2.0, (score_val / max(1e-6, max_score)) * 100.0))

        row_data: Dict[str, Any] = {
            "id": int(idx + 1),
            "rank": rank,
            "anomaly_score": round(score_val, 6),
            "flagged": is_flagged,
            "bar_pct": round(pct_bar, 1),
        }

        # Include available key feature columns
        for col in KEY_FEATURE_COLS:
            if col in df.columns:
                val = df.iloc[idx][col]
                if isinstance(val, (int, np.integer)):
                    row_data[col] = int(val)
                elif isinstance(val, (float, np.floating)):
                    row_data[col] = round(float(val), 2)
                else:
                    row_data[col] = str(val)

        flows_list.append(row_data)

    return flows_list


@app.get("/api/meta")
def get_metadata() -> Dict[str, Any]:
    """Return model architecture and calibration metadata."""
    calib_count = len(calib_scores) if calib_scores is not None else int(thresh_info.get("calibration_rows", 427451))
    return {
        "status_tag": STATUS_TAG,
        "title": "Zero-Day Flow Anomaly Detector",
        "subtitle": "Autoencoder trained on normal traffic only",
        "architecture": "76-64-32-16-32-64-76",
        "calibration_rows": calib_count,
        "default_q": 0.01,
        "pot_threshold": pot_threshold,
        "current_threshold": compute_threshold_for_q(0.01),
        "threshold_q05": compute_threshold_for_q(0.05),
        "has_sample_file": DEFAULT_SAMPLES_PATH.exists(),
        "has_labels_file": DEFAULT_LABELS_PATH.exists(),
    }


@app.post("/api/score")
async def score_endpoint(
    q: float = Form(0.01),
    source: str = Form("sample"),
    file: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    """Score flows using the trained demo MLP Autoencoder.

    Handles either sample flows or an uploaded CSV file.
    Does NOT require retraining.
    """
    global last_uploaded_df, last_uploaded_scores, cached_sample_df, cached_sample_scores

    # Clamp q between 0.01 and 0.20
    q_val = float(max(0.001, min(0.50, q)))

    df_to_score: Optional[pd.DataFrame] = None
    scores: Optional[np.ndarray] = None
    has_evaluation: bool = False
    source_type: str = source

    if file is not None and file.filename:
        # Custom uploaded CSV
        try:
            content = await file.read()
            csv_str = content.decode("utf-8", errors="replace")
            uploaded_df = pd.read_csv(io.StringIO(csv_str))

            if len(uploaded_df) == 0:
                raise HTTPException(status_code=400, detail="Uploaded CSV file contains no rows.")

            # Validate column constraints (strictly forbidden to supply labels)
            for col in ["Label", "Attack_Family", "Attack"]:
                if col in uploaded_df.columns:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Ground-truth label column '{col}' detected. "
                            "Upload must contain unlabelled network traffic only."
                        ),
                    )

            # Score using demo model
            res = score_flows(uploaded_df, artifacts_dir=ARTIFACTS_DIR)
            df_to_score = uploaded_df
            scores = res.scores

            last_uploaded_df = uploaded_df
            last_uploaded_scores = scores
            source_type = "upload"
            has_evaluation = False
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process and score uploaded CSV: {str(e)}",
            )
    elif source == "upload" and last_uploaded_df is not None and last_uploaded_scores is not None:
        # Re-using last uploaded CSV with changed q
        df_to_score = last_uploaded_df
        scores = last_uploaded_scores
        source_type = "upload"
        has_evaluation = False
    else:
        # Bundled sample flows
        if cached_sample_df is None or cached_sample_scores is None:
            if not DEFAULT_SAMPLES_PATH.exists():
                raise HTTPException(status_code=404, detail="Sample flows file not found on server.")
            cached_sample_df = pd.read_csv(DEFAULT_SAMPLES_PATH)
            res = score_flows(cached_sample_df, artifacts_dir=ARTIFACTS_DIR)
            cached_sample_scores = res.scores

        df_to_score = cached_sample_df
        scores = cached_sample_scores
        source_type = "sample"
        has_evaluation = True

    threshold = compute_threshold_for_q(q_val)
    flags = scores > threshold

    total = int(len(scores))
    n_flagged = int(np.sum(flags))
    n_benign = int(total - n_flagged)

    pct_flagged = round((n_flagged / max(1, total)) * 100.0, 2)
    pct_benign = round((n_benign / max(1, total)) * 100.0, 2)

    top_flows = format_flows_for_response(df_to_score, scores, flags, top_n=50)

    # Convert scores to native float list for Plotly histogram
    scores_list = [round(float(s), 6) for s in scores]

    return {
        "source": source_type,
        "has_evaluation": has_evaluation,
        "q": round(q_val, 4),
        "threshold": round(threshold, 6),
        "total": total,
        "flagged": n_flagged,
        "benign": n_benign,
        "pct_flagged": pct_flagged,
        "pct_benign": pct_benign,
        "scores": scores_list,
        "flows": top_flows,
        "key_feature_columns": KEY_FEATURE_COLS,
    }


@app.get("/api/evaluation")
def get_evaluation(q: float = Query(0.01)) -> Dict[str, Any]:
    """Compute ground-truth evaluation metrics for the bundled sample flows.

    Labels are stored separately in server-side demo_samples_labels.csv and are
    NEVER used by the model or scoring logic.
    """
    if not DEFAULT_LABELS_PATH.exists() or cached_sample_scores is None:
        raise HTTPException(
            status_code=400,
            detail="Ground-truth labels are only available for the bundled sample dataset.",
        )

    try:
        labels_df = pd.read_csv(DEFAULT_LABELS_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read evaluation labels: {e}")

    if len(labels_df) != len(cached_sample_scores):
        raise HTTPException(
            status_code=500,
            detail=f"Labels count ({len(labels_df)}) does not match sample flows count ({len(cached_sample_scores)}).",
        )

    family_col = "Attack_Family" if "Attack_Family" in labels_df.columns else "Label"
    threshold = compute_threshold_for_q(q)
    flags = cached_sample_scores > threshold

    eval_df = pd.DataFrame(
        {
            "Family": labels_df[family_col].astype(str),
            "Flagged": flags,
        }
    )

    benign_mask = eval_df["Family"].str.lower() == "benign"
    benign_grp = eval_df[benign_mask]
    attack_grp = eval_df[~benign_mask]

    benign_total = int(len(benign_grp))
    benign_flagged = int(benign_grp["Flagged"].sum())
    benign_far = round((benign_flagged / max(1, benign_total)) * 100.0, 2)

    attack_total = int(len(attack_grp))
    attack_flagged = int(attack_grp["Flagged"].sum())
    overall_tpr = round((attack_flagged / max(1, attack_total)) * 100.0, 2)

    # Per-family breakdown
    families_list = []
    for fam, grp in eval_df.groupby("Family"):
        fam_str = str(fam)
        is_b = fam_str.lower() == "benign"
        count = int(len(grp))
        flg = int(grp["Flagged"].sum())
        rate = round((flg / max(1, count)) * 100.0, 1)
        families_list.append(
            {
                "family": fam_str,
                "is_benign": is_b,
                "total": count,
                "flagged": flg,
                "detection_rate": rate,
            }
        )

    # Sort: Benign first, then attack families by name
    families_list.sort(key=lambda x: (1 if not x["is_benign"] else 0, x["family"]))

    # Operational verdict in plain English
    attack_stats = [f for f in families_list if not f["is_benign"]]
    caught = [f["family"] for f in attack_stats if f["detection_rate"] >= 50.0]
    missed = [f["family"] for f in attack_stats if f["detection_rate"] < 50.0]

    q_pct = round(q * 100.0, 1)
    if caught and missed:
        c_str = ", ".join(caught)
        m_str = ", ".join(missed)
        verdict = f"At q = {q_pct:.0f}%, {overall_tpr:.0f}% of attacks are flagged; {c_str} is caught, subtler families ({m_str}) are not."
    elif caught and not missed:
        verdict = f"At q = {q_pct:.0f}%, {overall_tpr:.0f}% of attacks are flagged; all tested attack families are successfully detected."
    else:
        verdict = f"At q = {q_pct:.0f}%, {overall_tpr:.0f}% of attacks are flagged; conservative threshold lets subtle threats pass."

    return {
        "q": round(q, 4),
        "threshold": round(threshold, 6),
        "benign_far": benign_far,
        "benign_flagged": benign_flagged,
        "benign_total": benign_total,
        "overall_tpr": overall_tpr,
        "attack_flagged": attack_flagged,
        "attack_total": attack_total,
        "families": families_list,
        "verdict": verdict,
        "note": "True labels are used only to grade, never by the model.",
    }


@app.get("/api/download_flagged")
def download_flagged_csv(
    q: float = Query(0.01), source: str = Query("sample")
) -> Response:
    """Download CSV of all flagged flows for the selected budget q."""
    if source == "upload" and last_uploaded_df is not None and last_uploaded_scores is not None:
        df = last_uploaded_df
        scores = last_uploaded_scores
    else:
        df = cached_sample_df
        scores = cached_sample_scores

    if df is None or scores is None:
        raise HTTPException(status_code=400, detail="No flows currently loaded.")

    threshold = compute_threshold_for_q(q)
    flags = scores > threshold

    flagged_df = df[flags].copy()
    flagged_df.insert(0, "anomaly_score", scores[flags])

    buf = io.StringIO()
    flagged_df.to_csv(buf, index=False)
    csv_bytes = buf.getvalue().encode("utf-8")

    filename = f"flagged_flows_q{int(q*100):02d}.csv"
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# Serve Static Assets
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index() -> FileResponse:
    """Serve the single-page dashboard HTML."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard index.html not found.")
    return FileResponse(index_file)
