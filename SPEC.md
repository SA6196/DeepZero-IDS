# Project Spec: Contamination-Robust Autoencoder with Label-Free Thresholding for Unseen Attack Families

Working title: **Contamination-Robust Autoencoder Anomaly Detection with Label-Free Extreme-Value Thresholding for Unseen Network Attack Families**

(Use "zero-day" only in the abstract/intro, and only because the evaluation is leave-one-family-out. Not in the title unless you keep that protocol.)

---

## 0. Read this first (honest positioning)

- Plain "deep autoencoder trained on benign traffic" is saturated. Comparative AE-for-NIDS studies already exist (e.g. the PeerJ Computer Science comparison on NSL-KDD).
- Contamination robustness is NOT unexplored. Nkashama et al. (arXiv 2407.08838, 2024) already evaluate six unsupervised DL detectors under training contamination and propose an AE with a constrained latent representation. Related: NCAE (Yu et al., NeurIPS 2021), robust per-feature loss (Triepels, EANN 2021). You MUST cite and compare against these.
- So your claim cannot be "first to handle contamination". Defensible claim:
  1. A contamination-robust AE (trimmed + robust per-feature loss) **combined with** a label-free EVT threshold (SPOT/POT, Siffer et al. KDD 2017),
  2. evaluated under strict unseen-family protocols on **corrected** datasets,
  3. reporting whether the label-free threshold actually delivers its target FPR (calibration), which most AE-IDS papers never check.
- If results show no gain over Nkashama's constrained-latent AE, the paper becomes an honest **benchmark/negative-results paper** on rigorous evaluation. That is still publishable at lower-tier venues. Decide after Phase 3.

---

## 1. Research questions

- RQ1: How much does an unsupervised AE degrade on **unseen attack families** vs. seen ones, per family, on corrected datasets?
- RQ2: How does degradation scale with training contamination (0, 1, 2, 5, 10%)?
- RQ3: Does trimmed/robust training reduce that degradation vs. standard AE, VAE, and Nkashama's constrained-latent AE?
- RQ4: Does a label-free EVT threshold hit its target FPR on held-out benign data and on later days (drift)? How far off is it?
- RQ5: Is the detector robust to simple evasion (mimicry/perturbation), and what is the CPU latency cost?

---

## 2. Datasets

| Dataset | Use | Notes |
|---|---|---|
| CIC-IDS2017, corrected | Main | Use a corrected version, not the original CSVs. Options: Engelen et al. (IEEE SPW 2021) improved pipeline; Liu et al. (IEEE CNS 2022) recreated dataset with corrected labels; LYCOS-IDS2017 (SCITEPRESS 2022); NFS-2023 (arXiv 2401.16843). Pick ONE, state it, cite it. |
| CSE-CIC-IDS2018, corrected | Cross-dataset partner for 2017 | Same CICFlowMeter toolchain, so features align. Use Liu et al. corrected version. |
| UNSW-NB15 | Second in-dataset LOFO benchmark | Different feature extractor; do NOT cross-test against CIC sets without a shared feature set. (Not re-verified in this research pass; check the official source and cite Moustafa & Slay 2015.) |
| CICIoT2023 | IoT benchmark | Neto et al., Sensors 2023: 33 attacks, 7 classes, 105 IoT devices. Classes: DDoS, DoS, Recon, Web, Brute Force, Spoofing, Mirai. |

**Cross-dataset caveat:** CIC-IDS2017 -> UNSW-NB15 directly is invalid because feature definitions differ. Do cross-dataset as **CIC-IDS2017 <-> CSE-CIC-IDS2018** (shared features), and optionally a reduced common-feature set (duration, packet/byte counts, rates, flag counts) recomputed with one extractor for the others.

Preprocessing rules:
- Drop duplicates, inf/NaN, negative-value artifacts; log what was dropped and how many rows.
- Remove identifier-like features (IPs, ports as identifiers, timestamps, flow IDs). Keep a documented list.
- Fit every scaler/encoder/PCA on TRAIN ONLY.

---

## 3. Evaluation protocols (all benign-only training unless contamination is injected)

**P1 Leave-one-family-out (LOFO).** For each attack family F: train on benign only; test on benign test split + family F only. Report per-family metrics, then macro-average. Families defined by dataset label groups (e.g. DoS/DDoS, PortScan, BruteForce, WebAttack, Botnet, Infiltration).

**P2 Temporal split.** CIC-IDS2017 is organized by day. Train on Monday benign (+ optionally early-week benign), validate on held-out benign, test on later days with attacks. No shuffling across days.

**P3 Cross-dataset.** Train on CIC-IDS2017 benign -> test on CSE-CIC-IDS2018, and reverse. Shared features only.

**P4 Contamination injection.** Inject c in {0, 1, 2, 5, 10}% attack flows (mixed families, unlabeled) into the training set. Evaluate under P1. Note: when family F is tested, do NOT include F in the injected contamination for the headline run; run a second variant where it is included.

**Data roles (leakage control):**
- TRAIN: benign (+ injected contamination for P4). Used for fitting model + transforms.
- VAL-BENIGN: held-out benign only. Used for early stopping and EVT threshold fitting. No attack labels touched.
- TEST: benign + attacks. Touched once per config for final numbers.
- Any hyperparameter tuning that uses attack labels (small labeled validation set) must be declared explicitly as a separate "semi-supervised tuning" setting. Default is label-free tuning.

---

## 4. Method

**Base model:** MLP autoencoder, bottleneck d_z in {8, 16, 32}, 3-layer encoder/decoder, BatchNorm or LayerNorm, ReLU. Reconstruct scaled features.

**Component A: robust training**
- Trimmed loss: each epoch, per batch, drop the top-k% highest-reconstruction-error samples from the loss. k = estimated contamination rate (try fixed grid {1,2,5,10}) and a variant with k annealed from 0 upward.
- Robust per-feature loss: Huber / per-feature adaptive loss (after Triepels, EANN 2021) vs plain MSE.

**Component B: scoring**
- s_rec: mean (optionally per-feature weighted) reconstruction error.
- s_lat: Mahalanobis distance in the latent space using mean/covariance from TRAIN latent codes (robust covariance, e.g. MCD, for contamination).
- Final score: rank-normalized combination of s_rec and s_lat on VAL-BENIGN; ablate each alone.

**Component C: label-free thresholding**
- Fit POT: pick initial high quantile on VAL-BENIGN scores, fit Generalized Pareto to the exceedances, compute threshold for target risk q (map q to target FPR in {1e-2, 1e-3}).
- Streaming variant (SPOT/DSPOT, Siffer et al. 2017) for the temporal protocol to handle drift.
- Baseline thresholds: fixed percentile of VAL-BENIGN (e.g. 99th, 99.9th), mean + 3 sigma, and oracle (best F1 on test, reported only as an upper bound and labeled as leaking).

**Headline claim to test:** A + B + C beats standard AE and Nkashama-style constrained-latent AE at c >= 2% contamination on macro-AUPRC and FPR@95%TPR, and the label-free threshold realizes FPR within a stated tolerance of its target.

---

## 5. Baselines (identical preprocessing, splits, tuning budget)

Classical: Isolation Forest, One-Class SVM, LOF, PCA reconstruction, (optionally) Kitsune-style ensemble AE.
Deep: standard AE, denoising AE, VAE, LSTM-AE or 1D-CNN-AE (only if using windowed flows), Deep SVDD, DAGMM.
Contamination-aware: constrained-latent AE (Nkashama et al. 2024), NCAE (Yu et al. 2021; code listed in their paper).
Recent: one transformer- or foundation-model-based unsupervised detector if code is available (e.g. NetVAD, arXiv 2606.01452) — reproduce or cite as reported with a clear note if not reproduced.
Supervised reference: Random Forest or XGBoost trained on seen families, tested via LOFO, to show the supervised gap on unseen families.

Equal tuning budget for every baseline (e.g. 30 random-search trials each on VAL-BENIGN using the label-free proxy). State the budget.

---

## 6. Metrics

Threshold-free: AUROC, AUPRC, FPR@95%TPR, TPR@1%FPR.
At the label-free operating point: precision, recall, F1, realized FPR vs target FPR, alerts per million flows.
Never headline accuracy.
All results: 5 seeds (0-4), mean ± std; Wilcoxon signed-rank (or paired bootstrap) across families/seeds for main comparisons; report effect sizes, not just p-values.

---

## 7. Robustness and deployment

**Evasion (white-box on the detector, realistic feature constraints):**
- Random feature perturbation within feasible ranges.
- Mimicry: optimize attack-flow features to minimize reconstruction/latent score subject to protocol-valid constraints (e.g. keep flags/protocol consistent, only modify features an attacker can control: packet size, inter-arrival timing, flow duration).
- Report detection rate vs. perturbation budget. Report failure cases.

**Deployment:** CPU-only inference latency per flow and per 1k-flow batch, throughput (flows/s), model size (MB), threshold update cost for SPOT. Report hardware.

**Explainability (optional, if time):** per-feature reconstruction error attribution per family; check that top features match known attack behavior (e.g. packet rate for DoS, destination-port diversity for scans).

---

## 7b. Ablation table (required)

Rows: plain AE; + trimmed loss; + robust loss; + latent Mahalanobis; + EVT threshold; full model. Columns: macro-AUPRC, FPR@95%TPR at c in {0, 2, 10}%.

---

## 8. Antigravity build plan

Give Antigravity this file plus the rules below, then build module by module. Do not ask for the whole project in one prompt.

**Global rules (paste at top of every prompt):**
1. Fit ALL scalers/encoders/PCA/covariances on TRAIN only. Never touch TEST statistics.
2. Thresholds and hyperparameters use VAL-BENIGN only unless a run is explicitly labeled semi-supervised.
3. Every experiment is driven by a YAML config with an explicit seed. Log every metric to a JSON/CSV file. Tables and figures are generated from logs only. No hand-typed numbers anywhere.
4. Write unit tests for leakage: (a) scaler `mean_` equals mean of TRAIN only; (b) no TEST index appears in TRAIN/VAL; (c) injected contamination count equals c% of TRAIN; (d) LOFO test family never appears in TRAIN unless contamination variant is on.
5. Do not invent citations or numbers. If unsure, leave a TODO.

**Modules, in order (each one done + tested before the next):**
1. `data/` loaders + cleaning for the chosen corrected CIC-IDS2017/2018, UNSW-NB15, CICIoT2023; cleaning report (rows dropped and why).
2. `splits/` P1-P4 split generators + leakage tests.
3. `baselines/` sklearn baselines (IF, OCSVM, LOF, PCA) with the shared preprocessing.
4. `models/` AE, DAE, VAE, Deep SVDD, then the contamination-aware ones.
5. `train/` training loop, trimmed loss, robust loss, early stopping on VAL-BENIGN.
6. `scoring/` s_rec, s_lat (robust covariance), combined score.
7. `thresholds/` percentile, mean+3sigma, POT (GPD fit), SPOT/DSPOT streaming.
8. `eval/` metrics, per-family tables, significance tests.
9. `robustness/` evasion attacks with feature constraints.
10. `bench/` latency/throughput/model size.
11. `report/` generate all tables/figures (ROC, PR, per-family heatmap, contamination curves, threshold calibration plot, latent t-SNE/UMAP) from logs.

**Compute:** tabular AEs train on CPU, but 4 datasets x ~12 methods x 5 seeds x LOFO x contamination levels is large. Use Kaggle/Colab GPU, subsample benign training flows (e.g. cap at 1-2M) and state it. Run the full grid only for the final config; use a reduced grid (1 seed, 2 contamination levels) for debugging.

---

## 9. Timeline (6 weeks)

- Week 1: data cleaning + splits + leakage tests; reproduce plain AE and 4 sklearn baselines on CIC-IDS2017 (corrected) under P1.
- Week 2: remaining baselines incl. constrained-latent AE and NCAE; P2/P3 protocols.
- Week 3: trimmed/robust loss + latent scoring + POT/SPOT; contamination sweep.
- Week 4: full grid, 5 seeds, ablations; second and third datasets.
- Week 5: evasion + latency; explainability if time; decide positioning (method paper vs. benchmark paper) from results.
- Week 6: write, internal review, release code + exact split files.

---

## 10. Paper outline

1. **Abstract** — problem, gap, method, protocols, 2-3 concrete numbers FROM LOGS.
2. **Introduction** — why zero-day detection is evaluated wrongly in much of the literature; 3 contributions max.
3. **Related work** — (a) AE-based NIDS and comparative studies, (b) dataset-quality critiques (Engelen 2021; Liu 2022; LYCOS; NFS-2023), (c) contamination-robust anomaly detection (Nkashama 2024; NCAE; Triepels), (d) EVT thresholding (Siffer 2017). End with an explicit gap statement.
4. **Threat model and problem setup** — what "unseen family" means, data roles.
5. **Method** — components A, B, C.
6. **Experimental protocol** — datasets (with corrections), P1-P4, baselines, metrics, seeds.
7. **Results** — LOFO per-family table; contamination curves; ablation; threshold calibration; temporal/cross-dataset; significance.
8. **Robustness and deployment** — evasion, latency.
9. **Limitations** — flow-level features only, benchmark traffic is synthetic/lab-generated, label noise residual even after correction, no encrypted-payload analysis, results may not transfer to production.
10. **Conclusion + reproducibility statement** — code, configs, split files, seeds.

**Figures/tables checklist:** per-family AUPRC table, contamination degradation curves, ablation table, FPR calibration (target vs realized), ROC/PR, latent UMAP benign vs attack, evasion curve, latency table.

---

## 11. Reference list (verified in this research pass; confirm details against the source before final citation)

- Engelen, Rimmer, Joosen. "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study." IEEE Security and Privacy Workshops (SPW), 2021.
- Liu, Engelen, Lynar, Essam, Joosen. "Error Prevalence in NIDS datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018." IEEE CNS, 2022.
- LYCOS-IDS2017 / LycoSTand: "Network Intrusion Detection: A Comprehensive Analysis of CIC-IDS2017." SCITEPRESS, 2022.
- "Evaluating ML-Based Anomaly Detection Across Datasets of Varied Integrity: A Case Study" (NFS-2023-nTE / NFS-2023-TE). arXiv 2401.16843.
- Nkashama et al. "Deep Learning for Network Anomaly Detection under Data Contamination: Evaluating Robustness and Mitigating Performance Degradation." arXiv 2407.08838, 2024.
- Yu, Oh, Kim, Kim. "Normality-Calibrated Autoencoder for Unsupervised Anomaly Detection on Data Contamination." NeurIPS 2021 / arXiv 2110.14825.
- Triepels. "Anomaly Detection by Robust Feature Reconstruction." EANN 2021 (Springer).
- Siffer, Fouque, Termier, Largouet. "Anomaly Detection in Streams with Extreme Value Theory." KDD 2017.
- Neto, Dadkhah, Ferreira, Zohourian, Lu, Ghorbani. "CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment." Sensors 23(13):5941, 2023. doi:10.3390/s23135941.
- "Application of deep autoencoder as an one-class classifier for unsupervised network intrusion detection: a comparative evaluation." PeerJ Computer Science (cs-327).
- LOAO evaluation example: "Hybrid Quantum-Classical Autoencoders for Unsupervised Network Intrusion Detection." arXiv 2512.05069.
- NetVAD: "Foundation-Model Representation Learning for Identifier-Free Unsupervised Intrusion Detection." arXiv 2606.01452.

Not verified here (check before citing): UNSW-NB15 (Moustafa & Slay, 2015), Deep SVDD (Ruff et al., 2018), DAGMM (Zong et al., 2018), Kitsune (Mirsky et al., 2018).

---

## 12. Venue guidance

- With full protocol + method + ablation + 2-3 datasets: aim for journals such as Computers & Security, Journal of Network and Computer Applications, IEEE Access, or IEEE TNSM-tier if results are strong.
- Benchmark/negative-results framing: workshops at RAID/ACSAC/CCS-family venues, or mid-tier conferences.
- Single dataset + plain AE: low-tier conference only. Check each venue's scope and current deadlines before committing.
