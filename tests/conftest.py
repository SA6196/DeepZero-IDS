"""Shared pytest fixtures for the IDS benchmark test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_ids_dataframe() -> pd.DataFrame:
    """Generate reproducible synthetic flow dataframe mimicking CIC-IDS2017."""
    rng = np.random.default_rng(42)
    n_rows = 5000

    days = ["Monday"] * 2000 + ["Tuesday"] * 750 + ["Wednesday"] * 750 + ["Thursday"] * 750 + ["Friday"] * 750

    labels = []
    families = []
    for day in days:
        if day == "Monday":
            labels.append("BENIGN")
            families.append("Benign")
        elif day == "Tuesday":
            if rng.random() < 0.6:
                labels.append("BENIGN")
                families.append("Benign")
            else:
                lbl = rng.choice(["FTP-Patator", "SSH-Patator"])
                labels.append(lbl)
                families.append("BruteForce")
        elif day == "Wednesday":
            if rng.random() < 0.6:
                labels.append("BENIGN")
                families.append("Benign")
            else:
                lbl = rng.choice(["DoS slowloris", "DoS Slowhttptest", "DoS Hulk", "DoS GoldenEye"])
                labels.append(lbl)
                families.append("DoS/DDoS")
        elif day == "Thursday":
            if rng.random() < 0.6:
                labels.append("BENIGN")
                families.append("Benign")
            else:
                lbl = rng.choice(["Web Attack – Brute Force", "Web Attack – XSS"])
                labels.append(lbl)
                families.append("WebAttack")
        elif day == "Friday":
            if rng.random() < 0.6:
                labels.append("BENIGN")
                families.append("Benign")
            else:
                lbl = rng.choice(["PortScan", "Bot"])
                family = "PortScan" if lbl == "PortScan" else "Botnet"
                labels.append(lbl)
                families.append(family)

    features = {
        "Flow Duration": rng.uniform(10.0, 100000.0, size=n_rows),
        "Total Fwd Packets": rng.integers(1, 100, size=n_rows).astype(float),
        "Total Backward Packets": rng.integers(0, 100, size=n_rows).astype(float),
        "Flow Bytes/s": rng.uniform(100.0, 500000.0, size=n_rows),
        "Flow Packets/s": rng.uniform(1.0, 5000.0, size=n_rows),
        "Flow IAT Mean": rng.uniform(1.0, 20000.0, size=n_rows),
        "Init_Win_bytes_forward": rng.integers(0, 65535, size=n_rows).astype(float),
    }

    df = pd.DataFrame(features)
    df["Day"] = days
    df["Label"] = labels
    df["Attack_Family"] = families

    attack_mask = df["Attack_Family"] != "Benign"
    df.loc[attack_mask, "Flow Duration"] += 50000.0
    df.loc[attack_mask, "Flow Bytes/s"] += 200000.0

    return df
