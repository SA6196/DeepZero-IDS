# Contamination-Robust Autoencoder with Label-Free Thresholding for Unseen Network Attack Families

Research codebase accompanying the paper:
**Contamination-Robust Autoencoder Anomaly Detection with Label-Free Extreme-Value Thresholding for Unseen Network Attack Families**

---

## 1. Dataset Acquisition and Directory Setup

Per **SPEC.md Section 2**, we evaluate on **corrected versions** of CIC-IDS2017 rather than the uncorrected original CSVs:
- **Baseline Correction Target**: Liu et al. (IEEE CNS 2022) / Engelen et al. (IEEE SPW 2021).
  - *Citation*: Liu, Engelen, Lynar, Essam, Joosen. *"Error Prevalence in NIDS datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018."* IEEE Conference on Communications and Network Security (CNS), 2022.

### Files and Destination Paths

TODO: user to supply source and file list

Once downloaded, place the daily flow capture files into `data/raw/cicids2017/` (or specify via `--data_dir` in configuration):

| Day | File Name | Attack Families Present |
|---|---|---|
| **Monday** | `Monday-WorkingHours.pcap_ISCX.csv` | Benign baseline traffic only |
| **Tuesday** | `Tuesday-WorkingHours.pcap_ISCX.csv` | Benign, FTP-Patator, SSH-Patator (`BruteForce`) |
| **Wednesday** | `Wednesday-workingHours.pcap_ISCX.csv` | Benign, DoS slowloris, DoS Slowhttptest, DoS Hulk, DoS GoldenEye, Heartbleed (`DoS/DDoS`) |
| **Thursday Morning** | `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | Benign, Web Attack – Brute Force, Web Attack – XSS, Web Attack – Sql Injection (`WebAttack`) |
| **Thursday Afternoon** | `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | Benign, Infiltration (`Infiltration`) |
| **Friday Morning** | `Friday-WorkingHours-Morning.pcap_ISCX.csv` | Benign, Bot (`Botnet`) |
| **Friday Afternoon (PortScan)** | `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | Benign, PortScan (`PortScan`) |
| **Friday Afternoon (DDoS)** | `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | Benign, DDoS (`DoS/DDoS`) |

If using the consolidated parquet/CSV export from Liu et al. (2022), place `cicids2017_corrected.parquet` directly into `data/raw/cicids2017/`.

---

## 2. Preprocessing & Data Cleaning Pipeline (`data/`)

The cleaning pipeline ([data/cicids2017.py](file:///c:/Users/Tanmay%20Shukla/Desktop/CN/data/cicids2017.py)) implements the exact preprocessing constraints of **SPEC.md Section 2**:
1. **Deduplication**: Removes all duplicate flow records BEFORE split generation.
2. **Schema & Label Validation**:
   - Compares columns against [configs/schema_cicids2017.yaml](file:///c:/Users/Tanmay%20Shukla/Desktop/CN/configs/schema_cicids2017.yaml) and raises clear errors on missing/unexpected columns.
   - Audits all distinct raw labels and maps to attack families via [configs/attack_families_cicids2017.yaml](file:///c:/Users/Tanmay%20Shukla/Desktop/CN/configs/attack_families_cicids2017.yaml), flagging any unmapped labels.
3. **Negative-Value Analysis & Sentinel Preservation**:
   - Analyzes negative values per column and per label.
   - Legitimate protocol sentinels (e.g. `Init_Win_bytes_forward / backward = -1`) are preserved.
   - Rows are not dropped for negatives by default; when enabled, only drops rows with negatives in physically impossible columns.
4. **Identifier Stripping**: Drops network identity shortcuts (`Flow ID`, `Source IP`, `Source Port`, `Destination IP`, `Destination Port`, `Timestamp`).
5. **Infinite / NaN Filtering**: Purges rows with infinite or missing values resulting from zero-duration flow divisions (`Flow Bytes/s`, `Flow Packets/s`).
6. **Cleaning Report**: Generates an audit JSON log capturing raw count, rows dropped per filter, retained features, and label distributions before and after.

---

## 3. Evaluation Protocols (`splits/`)

- **Protocol P1 (Leave-One-Family-Out / LOFO)**:
  - For each attack family $F \in \mathcal{F}$:
    - `TRAIN`: Benign flows only.
    - `VAL-BENIGN`: Held-out benign flows only.
    - `TEST`: Benign test split + all flows of family $F$ only.
- **Protocol P2 (Temporal Split)**:
  - Strict chronological partition without shuffling across days:
    - `TRAIN`: Monday benign traffic.
    - `VAL-BENIGN`: Held-out Monday benign traffic.
    - `TEST`: All traffic from subsequent days (Tuesday to Friday, both benign and attacks).
- **Protocol P4 (Contamination Injection & Confound Elimination)**:
  - Injects $c \in \{0, 1, 2, 5, 10\}\%$ attack flows into the training set under Protocol P1.
  - **Fixed Total TRAIN Size $N$**: $N$ is identical across all $c$, eliminating training size confounds.
  - **Nested Benign Subset**: Benign rows for larger $c$ are a strict nested subset of benign rows for smaller $c$.
  - **Headline Variant (Unseen Isolation)**: Tested family $F$ is strictly excluded from contamination.

---

## 4. Running Tests and Summary Script will be generated

Using the dedicated virtual environment at `C:\venv`:
```bash
# Run unit tests
& C:\venv\Scripts\pytest.exe tests/ -v

# Run split summary script
& C:\venv\Scripts\python.exe scripts/summarize_splits.py
```
