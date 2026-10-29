# Zero-Day Flow Anomaly Detector — Web Dashboard

> **DEV DATA - NOT FOR REPORTED RESULTS**

Polished, offline-ready web dashboard for real-time network flow anomaly detection using the trained MLP Autoencoder.

---

## 🚀 3 Commands to Run

### Command 1: (Optional) Install dependencies (if not already installed)
```powershell
C:\venv\Scripts\pip.exe install fastapi uvicorn python-multipart httpx
```

### Command 2: Start the Web Dashboard Server
```powershell
C:\venv\Scripts\python.exe -m uvicorn demo_web.server:app --port 8600
```

### Command 3: Open in Browser
Open your browser and navigate to:
```text
http://localhost:8600
```

*(Tip: In the dashboard, click **"⚡ Demo Mode [D]"** or press the **D** key to automatically load sample flows and display the evaluation breakdown in one click).*

---

## 🧪 Run Acceptance & Validation Tests
```powershell
C:\venv\Scripts\python.exe -m pytest demo_web/test_web.py -v
```

---

## 🌟 Key Features & Architecture
- **Offline Operation**: Plotly.js (`plotly.min.js`, 4.8MB) is locally bundled under `demo_web/static/` — zero internet access required at the demo venue.
- **Single Dark Theme**: High-contrast, WCAG AA compliant palette (`#0b1020` background, `#121a33` cards, `#e6ebff` text, cyan `#38bdf8`, alert red `#f43f5e`, safe green `#22c55e`, amber `#f59e0b`).
- **Real-Time Responsiveness**: Budget slider ($q \in [0.01, 0.10]$) updates active threshold, 4 KPI cards, log histogram, and ranked flows table in sub-100ms without retraining.
- **Label Isolation**: Ground-truth labels are stored strictly on the server (`demo/demo_samples_labels.csv`) and are exclusively consulted when evaluating the bundled sample. Custom CSV uploads are unlabelled and strictly guarded.
- **Export**: Instant "Download flagged CSV" button exports out-of-distribution flows for forensic analysis.
