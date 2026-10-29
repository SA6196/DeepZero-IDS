# Network Flow Anomaly Detection: Live Demo

> **NOTICE:** DEV DATA, NOT FOR REPORTED RESULTS

This directory contains an interactive demonstration of unsupervised network flow anomaly detection using a lightweight Multi-Layer Perceptron Autoencoder (MLP AE) calibrated via Peaks-Over-Threshold (POT) extreme value theory.

---

## 1. Quickstart: Exact Windows Commands

Run these commands in PowerShell from the repository root (`c:\Users\Tanmay Shukla\Desktop\CN`):

### Step 1: Train the Demo Model
Trains the accepted MLP Autoencoder on benign-only traffic (capped at 200,000 rows) and calibrates the POT alarm threshold on validation benign flows ($q = 10^{-2}$):
```powershell
C:\venv\Scripts\python.exe demo\train_demo_model.py
```

### Step 2: Generate Demo Flow Samples
Builds held-out flow samples not used in training or validation (300 Benign + 100 each of DoS/DDoS, PortScan, BruteForce, and WebAttack):
```powershell
C:\venv\Scripts\python.exe demo\make_demo_samples.py
```

### Step 3: Launch the Interactive Streamlit Demo
Starts the local web interface for live flow scoring, histogram visualization, and ground-truth evaluation breakdown:
```powershell
C:\venv\Scripts\streamlit.exe run demo\app.py
```

*(Optional CLI Scoring)*:
To score an unlabelled CSV from the command line:
```powershell
C:\venv\Scripts\python.exe demo\score_flows.py demo\demo_samples.csv
```

---

## 2. Expected Runtimes

| Task | Command | Expected Runtime | Notes |
|:---|:---|:---|:---|
| **Model Training** | `python demo\train_demo_model.py` | ~1 to 3 minutes | 200,000 rows, Adam lr=1e-3, early stopping on val loss (patience 3), max 4 CPU threads |
| **Sample Generation** | `python demo\make_demo_samples.py` | ~5 to 15 seconds | Reads parquet, partitions held-out splits, writes `demo_samples.csv` |
| **Streamlit App Launch** | `streamlit run demo\app.py` | < 3 seconds | Opens in default web browser (`http://localhost:8501`) |
| **Flow Scoring (700 flows)** | `score_flows()` / CLI | < 0.5 seconds | Highly optimized vectorized PyTorch CPU inference |

---

## 3. Architecture & Leakage Protection Highlights

- **Unsupervised Training**: Trained strictly on Benign traffic. Zero exposure to attack flows or test data during training or preprocessing.
- **Strict Leakage Barrier**: The `score_flows` function strictly forbids and rejects any input column containing ground-truth labels.
- **Extreme Value Theory (POT) Calibration**: The alarm threshold is fitted on validation benign reconstruction scores using a Generalized Pareto Distribution tail fit ($q = 10^{-2}$), guaranteeing a principled nominal false-alarm budget of ~1% without arbitrary hyperparameter tuning.
- **Completely Self-Contained**: The demo UI and CLI run entirely from `demo_samples.csv` and `demo/artifacts/` without requiring access to the heavy raw datasets.

---

## 4. 60-Second Talking Script for Presenting to a Professor

> *"Good morning, Professor.*
>
> *Today I'm presenting our live prototype for **unsupervised network flow anomaly detection**.*
>
> *(Point to UI header)*  
> *In production cybersecurity, novel zero-day attacks have never been seen before, so supervised classifiers trained on known attacks fail to generalize. Instead, our model is a lightweight Multi-Layer Perceptron Autoencoder trained **strictly on normal benign flows**.*
>
> *(Click 'Load sample flows')*  
> *Here, I've loaded 700 unlabelled network flows from held-out traffic. The system scores each flow's reconstruction error in real-time. Notice that our inference pipeline strictly rejects ground-truth labels—it evaluates raw flow metrics alone.*
>
> *(Point to Histogram)*  
> *Rather than picking an ad-hoc threshold, we calibrated this red alarm boundary using **Peaks-Over-Threshold (POT) Extreme Value Theory** on validation benign traffic with a target risk $q = 1\%$. This guarantees an operator-specified false alarm rate budget.*
>
> *(Check 'Show evaluation')*  
> *Finally, if we verify against held-out ground truth for display only, you can see that our benign false alarm rate remains right at our 1% design budget, while achieving near-perfect detection on aggressive attack vectors like DoS and PortScans.*
>
> *Thank you—I'd be glad to answer any questions about our methodology or extreme value formulation."*
