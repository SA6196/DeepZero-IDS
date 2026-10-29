"""Acceptance tests for demo_web FastAPI server and endpoints.

DEV DATA - NOT FOR REPORTED RESULTS
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from demo_web.server import app, compute_threshold_for_q

client = TestClient(app)
REPO_ROOT = Path(__file__).resolve().parent.parent


def test_offline_plotly_bundle_exists() -> None:
    """Verify local plotly.min.js bundle exists for offline use."""
    plotly_file = REPO_ROOT / "demo_web" / "static" / "plotly.min.js"
    assert plotly_file.exists(), "demo_web/static/plotly.min.js must exist for offline operation"
    assert plotly_file.stat().st_size > 1_000_000, "plotly.min.js file size is unexpectedly small"


def test_meta_endpoint() -> None:
    """Verify GET /api/meta returns architecture and calibration metadata."""
    res = client.get("/api/meta")
    assert res.status_code == 200
    data = res.json()
    assert data["architecture"] == "76-64-32-16-32-64-76"
    assert data["calibration_rows"] == 427451
    assert data["default_q"] == 0.01
    assert np.isclose(data["pot_threshold"], 0.088134, atol=1e-4)


def test_q_05_matches_streamlit_benchmark() -> None:
    """Acceptance Check 2: At q=0.05 the numbers match the Streamlit app.

    Expected: threshold ~0.03172, 112 of 700 flagged.
    """
    res = client.post("/api/score", data={"q": 0.05, "source": "sample"})
    assert res.status_code == 200
    data = res.json()

    assert data["total"] == 700
    assert data["flagged"] == 112, f"Expected 112 flagged at q=0.05, got {data['flagged']}"
    assert data["benign"] == 588
    assert np.isclose(data["threshold"], 0.03172, atol=1e-4), f"Unexpected threshold: {data['threshold']}"
    assert data["has_evaluation"] is True
    assert len(data["flows"]) <= 50
    assert len(data["scores"]) == 700


def test_evaluation_endpoint_q_05() -> None:
    """Acceptance Check 4: Evaluation panel per-family and false-alarm rates."""
    res = client.get("/api/evaluation?q=0.05")
    assert res.status_code == 200
    data = res.json()

    assert np.isclose(data["benign_far"], 3.00, atol=1e-2)
    assert data["benign_flagged"] == 9
    assert data["benign_total"] == 300

    assert np.isclose(data["overall_tpr"], 25.75, atol=1e-2)
    assert data["attack_flagged"] == 103
    assert data["attack_total"] == 400

    family_map = {f["family"]: f for f in data["families"]}
    assert np.isclose(family_map["DoS/DDoS"]["detection_rate"], 95.0, atol=1e-1)
    assert np.isclose(family_map["WebAttack"]["detection_rate"], 7.0, atol=1e-1)
    assert np.isclose(family_map["BruteForce"]["detection_rate"], 1.0, atol=1e-1)
    assert np.isclose(family_map["PortScan"]["detection_rate"], 0.0, atol=1e-1)
    assert np.isclose(family_map["Benign"]["detection_rate"], 3.0, atol=1e-1)
    assert "True labels are used only to grade, never by the model." in data["note"]


def test_upload_without_labels_works_and_disables_evaluation() -> None:
    """Acceptance Check 5: Uploading CSV without label column works and never shows evaluation."""
    sample_path = REPO_ROOT / "demo" / "demo_samples.csv"
    sample_df = pd.read_csv(sample_path).head(20)  # Take 20 rows
    assert "Attack_Family" not in sample_df.columns
    assert "Label" not in sample_df.columns

    csv_buf = io.StringIO()
    sample_df.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    files = {"file": ("unlabelled_flows.csv", csv_bytes, "text/csv")}
    res = client.post("/api/score", data={"q": 0.01, "source": "upload"}, files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["source"] == "upload"
    assert data["has_evaluation"] is False
    assert data["total"] == 20


def test_upload_with_labels_is_rejected() -> None:
    """Ensure ground-truth isolation: uploading CSV with label column is rejected."""
    df_with_labels = pd.DataFrame({
        "Flow Duration": [1000, 2000],
        "Label": ["BENIGN", "DoS"],
    })
    csv_buf = io.StringIO()
    df_with_labels.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    files = {"file": ("labelled_flows.csv", csv_bytes, "text/csv")}
    res = client.post("/api/score", data={"q": 0.01, "source": "upload"}, files=files)
    assert res.status_code == 400
    assert "Ground-truth label column" in res.json()["detail"]


def test_download_flagged_csv() -> None:
    """Verify download_flagged returns a valid CSV."""
    res = client.get("/api/download_flagged?q=0.05&source=sample")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/csv")
    csv_df = pd.read_csv(io.StringIO(res.text))
    assert len(csv_df) == 112
    assert "anomaly_score" in csv_df.columns
