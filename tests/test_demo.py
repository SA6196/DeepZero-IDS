"""Unit tests for network flow demo module using synthetic data only.

DEV DATA, NOT FOR REPORTED RESULTS

Invariants tested:
1. `score_flows` strictly rejects any DataFrame containing label columns.
2. `demo.app` imports cleanly without error.
3. A tiny synthetic model saved and loaded gives identical reconstruction scores twice.
4. Any tests requiring real artifacts are skipped with a clear reason when missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from data.preprocess import TabularPreprocessor
from demo.score_flows import DEFAULT_ARTIFACTS_DIR, score_flows
from models.autoencoders import MLPAutoencoder

STATUS_TAG = "DEV DATA, NOT FOR REPORTED RESULTS"


@pytest.fixture
def synthetic_unlabelled_flows() -> pd.DataFrame:
    """Generate small synthetic flow features without any labels."""
    rng = np.random.default_rng(123)
    n = 60
    data = {
        "Flow Duration": rng.uniform(10.0, 5000.0, size=n),
        "Total Fwd Packets": rng.integers(1, 50, size=n).astype(float),
        "Total Backward Packets": rng.integers(0, 50, size=n).astype(float),
        "Flow Bytes/s": rng.uniform(100.0, 50000.0, size=n),
        "Flow Packets/s": rng.uniform(1.0, 500.0, size=n),
    }
    return pd.DataFrame(data)


def test_score_flows_rejects_labels(synthetic_unlabelled_flows: pd.DataFrame):
    """Invariant: score_flows must NEVER accept or use labels."""
    # Test column name 'Label'
    df_with_label = synthetic_unlabelled_flows.copy()
    df_with_label["Label"] = "BENIGN"
    with pytest.raises(ValueError, match="Labels are never accepted or used"):
        score_flows(df_with_label)

    # Test column name 'Attack_Family'
    df_with_family = synthetic_unlabelled_flows.copy()
    df_with_family["Attack_Family"] = "DoS/DDoS"
    with pytest.raises(ValueError, match="Labels are never accepted or used"):
        score_flows(df_with_family)

    # Test case-insensitive label names ('target', 'attack', 'is_attack')
    for bad_col in ["target", "ATTACK", "is_attack", "Ground_Truth"]:
        df_bad = synthetic_unlabelled_flows.copy()
        df_bad[bad_col] = 1
        with pytest.raises(ValueError, match="Labels are never accepted or used"):
            score_flows(df_bad)


def test_app_module_imports_without_error():
    """Verify that demo.app module imports cleanly without side-effects or errors."""
    import demo.app as demo_app

    assert hasattr(demo_app, "render_page"), "demo.app must provide render_page"
    assert hasattr(demo_app, "get_artifacts_status"), "demo.app must provide get_artifacts_status"


def test_tiny_synthetic_model_scores_identical_twice(
    tmp_path: Path,
    synthetic_unlabelled_flows: pd.DataFrame,
):
    """Verify reproducibility: tiny synthetic model saved and loaded gives identical scores twice."""
    df = synthetic_unlabelled_flows

    # 1. Fit preprocessor on synthetic data
    preprocessor = TabularPreprocessor(skew_threshold=3.0, clip_value=10.0)
    preprocessor.fit(df)
    prep_path = tmp_path / "preprocessor.joblib"
    preprocessor.save(prep_path)

    X_scaled = preprocessor.transform(df)
    input_dim = X_scaled.shape[1]

    # 2. Build and save a tiny synthetic MLP Autoencoder
    model = MLPAutoencoder(
        input_dim=input_dim,
        hidden_dims=(8, 4),
        bottleneck_dim=2,
        seed=42,
    )
    weights_path = tmp_path / "model_weights.pt"
    torch.save(model.state_dict(), weights_path)

    config_path = tmp_path / "model_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({"hidden_dims": [8, 4], "bottleneck_dim": 2}, f)

    # 3. Save synthetic threshold
    thresh_val = 0.25
    thresh_meta = {
        "status": STATUS_TAG,
        "threshold": thresh_val,
        "q": 0.01,
        "calibration_rows": len(df),
    }
    thresh_path = tmp_path / "threshold.json"
    with open(thresh_path, "w", encoding="utf-8") as f:
        json.dump(thresh_meta, f)

    # 4. Score twice with the synthetic artifacts
    res1 = score_flows(df, artifacts_dir=tmp_path)
    res2 = score_flows(df, artifacts_dir=tmp_path)

    # Invariant: identical numerical reconstruction scores
    np.testing.assert_allclose(
        res1.scores,
        res2.scores,
        rtol=1e-6,
        atol=1e-6,
        err_msg="Synthetic model reconstruction scores must be bit-exact across calls.",
    )
    # Invariant: identical boolean flags
    np.testing.assert_array_equal(
        res1.flags,
        res2.flags,
        err_msg="Synthetic model anomaly flags must be identical across calls.",
    )

    # Verify score shapes and flag counts
    assert len(res1.scores) == len(df)
    assert len(res1.flags) == len(df)
    assert res1.flags.dtype == bool
    assert np.all(res1.flags == (res1.scores > thresh_val))


def test_real_artifacts_inference_skipped_if_missing():
    """Verify that tests requiring real trained artifacts skip cleanly with a clear message."""
    weights_path = DEFAULT_ARTIFACTS_DIR / "model_weights.pt"
    prep_path = DEFAULT_ARTIFACTS_DIR / "preprocessor.joblib"
    thresh_path = DEFAULT_ARTIFACTS_DIR / "threshold.json"

    artifacts_exist = weights_path.exists() and prep_path.exists() and thresh_path.exists()

    if not artifacts_exist:
        pytest.skip(
            "Demo artifacts missing in demo/artifacts/. "
            "Run 'C:\\venv\\Scripts\\python.exe demo\\train_demo_model.py' to generate them."
        )

    # If artifacts exist, run a fast verification on dummy input
    dummy_input = pd.DataFrame(
        np.zeros((5, len(TabularPreprocessor.load(prep_path).retained_cols_))),
        columns=TabularPreprocessor.load(prep_path).retained_cols_,
    )
    res = score_flows(dummy_input)
    assert len(res.scores) == 5
    assert len(res.flags) == 5


def test_cli_scoring_with_synthetic_artifacts(
    tmp_path: Path,
    synthetic_unlabelled_flows: pd.DataFrame,
    monkeypatch,
    capsys,
):
    """Verify CLI interface: python demo/score_flows.py flows.csv writes scored csv."""
    from demo.score_flows import main

    df = synthetic_unlabelled_flows
    input_csv = tmp_path / "test_input_flows.csv"
    output_csv = tmp_path / "test_output_scored.csv"
    df.to_csv(input_csv, index=False)

    # Setup synthetic artifacts
    prep = TabularPreprocessor(skew_threshold=3.0, clip_value=10.0)
    prep.fit(df)
    prep.save(tmp_path / "preprocessor.joblib")

    model = MLPAutoencoder(
        input_dim=len(prep.retained_cols_),
        hidden_dims=(8, 4),
        bottleneck_dim=2,
        seed=42,
    )
    torch.save(model.state_dict(), tmp_path / "model_weights.pt")

    with open(tmp_path / "model_config.json", "w", encoding="utf-8") as f:
        json.dump({"hidden_dims": [8, 4], "bottleneck_dim": 2}, f)

    with open(tmp_path / "threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": 0.25, "q": 0.01, "calibration_rows": len(df)}, f)

    # Invoke CLI via monkeypatched sys.argv
    test_args = [
        "score_flows.py",
        str(input_csv),
        "--output",
        str(output_csv),
        "--artifacts-dir",
        str(tmp_path),
    ]
    monkeypatch.setattr("sys.argv", test_args)

    main()
    captured = capsys.readouterr()
    assert "SCORING SUMMARY" in captured.out
    assert "Suspicious flows flagged" in captured.out
    assert output_csv.exists()

    df_out = pd.read_csv(output_csv)
    assert "flagged_anomaly" in df_out.columns
    assert "anomaly_score" in df_out.columns
    assert len(df_out) == len(df)


def test_false_alarm_budget_slider_thresholds_and_detection_rates():
    """Verify that false-alarm budget slider correctly adjusts thresholds without retraining."""
    import demo.app as demo_app
    from demo.score_flows import score_flows

    status = demo_app.get_artifacts_status()
    if not status["ready"]:
        pytest.skip("Demo artifacts missing.")

    pot_thresh = float(status["info"].get("threshold", 0.0))
    calib_scores = demo_app.load_calibration_scores(demo_app.ARTIFACTS_DIR)
    assert calib_scores is not None and len(calib_scores) > 0

    # 1. Check threshold values across q options
    t_01 = demo_app.compute_threshold_for_q(0.01, pot_thresh, calib_scores)
    t_02 = demo_app.compute_threshold_for_q(0.02, pot_thresh, calib_scores)
    t_05 = demo_app.compute_threshold_for_q(0.05, pot_thresh, calib_scores)
    t_10 = demo_app.compute_threshold_for_q(0.10, pot_thresh, calib_scores)

    # Invariant: q=0.01 must equal POT threshold
    assert np.isclose(t_01, pot_thresh)

    # Invariant: as budget q increases, alarm threshold strictly decreases (more sensitive)
    assert t_01 > t_02 > t_05 > t_10, f"Expected strictly decreasing thresholds, got {t_01}, {t_02}, {t_05}, {t_10}"

    # 2. Check scoring on demo samples
    df_flows = pd.read_csv(demo_app.DEFAULT_SAMPLES_PATH)
    labels_df = pd.read_csv(demo_app.DEFAULT_LABELS_PATH)
    scored = score_flows(df_flows)

    counts_flagged = []
    for q in [0.01, 0.02, 0.05, 0.10]:
        t = demo_app.compute_threshold_for_q(q, pot_thresh, calib_scores)
        flg = scored.scores > t
        counts_flagged.append(int(flg.sum()))

    # Invariant: number of flagged anomalies is non-decreasing with budget q
    assert counts_flagged[0] <= counts_flagged[1] <= counts_flagged[2] <= counts_flagged[3]

