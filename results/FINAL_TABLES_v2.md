# Comprehensive Benchmark Evaluation & Results Report
**[DEV DATA, NOT FOR REPORTED RESULTS]**

Generated automatically from raw log files across seeds 0, 1, 2. No numbers typed by hand.

> [!IMPORTANT]
> **LOF Audit Finding & Deduplication Notice (DEV DATA, NOT FOR REPORTED RESULTS):**
> - `results/final_p1_seeds012.csv` contained 106 total rows (1 duplicate row detected).
> - Exact duplicate: Rows 10 and 11 are identical records for `(family='DoS/DDoS', seed=0, model='LOF')` with AUROC=0.939179, FPR@95=0.139951, AUPRC=0.860921.
> - Cause of discrepancy: Seed 0 LOF DoS/DDoS was cached and appended twice. When averaging DoS/DDoS across 4 rows [0.9392, 0.9392, 0.9191, 0.9336], the AUROC was 0.9328 instead of 0.9306.
> - Log verification: `results/console_run_stage1_p1.py.log` records seed 0 = 0.9392 (cached), seed 1 = 0.9191, seed 2 = 0.9336 (mean = 0.9306). All values in CSV match logs exactly; no underlying metrics differ.
> - Resolution: Created de-duplicated copy `results/final_p1_seeds012_dedup.csv` (105 rows, keeping first occurrence). This report reads from `results/final_p1_seeds012_dedup.csv`.

## 1. Protocol P1: Leave-One-Family-Out Evaluation (Seeds 0, 1, 2)
*Note: Infiltration (36 flows) excluded due to high statistical variance with small sample size.*
*Speed caps: OCSVM=20k, LOF=50k, IF=200k, PCA=200k, AE/VAE/Trimmed AE=500k. Test benign scored in full (~427k).*

### Table 1.1: P1 AUROC (mean ± std over 3 seeds)
| Family | IsolationForest | OneClassSVM | LOF | PCA | MLP AE | VAE | Trimmed AE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DoS/DDoS** | 0.9249 ± 0.0084 | 0.9284 ± 0.0078 | 0.9306 ± 0.0104 | 0.9736 ± 0.0002 | 0.9891 ± 0.0015 | 0.9826 ± 0.0042 | 0.9777 ± 0.0008 |
| **PortScan** | 0.6757 ± 0.0500 | 0.7293 ± 0.0121 | 0.8369 ± 0.2110 | 0.8119 ± 0.0016 | 0.9030 ± 0.0295 | 0.9097 ± 0.0076 | 0.9152 ± 0.0200 |
| **BruteForce** | 0.8261 ± 0.0199 | 0.5579 ± 0.0601 | 0.9114 ± 0.0580 | 0.8122 ± 0.0012 | 0.8767 ± 0.0279 | 0.8638 ± 0.0462 | 0.8812 ± 0.0092 |
| **WebAttack** | 0.8266 ± 0.0185 | 0.6306 ± 0.0574 | 0.9412 ± 0.0053 | 0.7595 ± 0.0039 | 0.8605 ± 0.0154 | 0.5958 ± 0.0405 | 0.8227 ± 0.0218 |
| **Botnet** | 0.6185 ± 0.0103 | 0.3805 ± 0.0096 | 0.4961 ± 0.0284 | 0.6589 ± 0.0034 | 0.6461 ± 0.0157 | 0.6370 ± 0.0340 | 0.6150 ± 0.0285 |
| **Macro Average** | 0.7743 ± 0.0077 | 0.6453 ± 0.0253 | 0.8233 ± 0.0430 | 0.8032 ± 0.0018 | 0.8550 ± 0.0086 | 0.7978 ± 0.0182 | 0.8424 ± 0.0064 |

### Table 1.2: P1 FPR@95%TPR (mean ± std over 3 seeds)
| Family | IsolationForest | OneClassSVM | LOF | PCA | MLP AE | VAE | Trimmed AE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DoS/DDoS** | 0.2521 ± 0.0126 | 0.5109 ± 0.0438 | 0.2048 ± 0.0990 | 0.1784 ± 0.0004 | 0.0547 ± 0.0118 | 0.0773 ± 0.0282 | 0.0710 ± 0.0104 |
| **PortScan** | 0.4004 ± 0.0405 | 0.5535 ± 0.0175 | 0.3260 ± 0.4570 | 0.2048 ± 0.0013 | 0.1586 ± 0.0227 | 0.1883 ± 0.0082 | 0.1444 ± 0.0372 |
| **BruteForce** | 0.3045 ± 0.0239 | 0.5698 ± 0.0805 | 0.2569 ± 0.1188 | 0.3225 ± 0.0018 | 0.1928 ± 0.0556 | 0.2102 ± 0.0545 | 0.1628 ± 0.0127 |
| **WebAttack** | 0.5604 ± 0.0137 | 0.4085 ± 0.0542 | 0.4474 ± 0.0719 | 0.6276 ± 0.0021 | 0.7398 ± 0.1017 | 0.8926 ± 0.0337 | 0.7585 ± 0.0139 |
| **Botnet** | 0.5969 ± 0.0295 | 0.9962 ± 0.0004 | 0.9472 ± 0.0181 | 0.5635 ± 0.0078 | 0.8641 ± 0.0579 | 0.9666 ± 0.0138 | 0.9234 ± 0.0898 |
| **Macro Average** | 0.4228 ± 0.0217 | 0.6078 ± 0.0304 | 0.4364 ± 0.0953 | 0.3794 ± 0.0020 | 0.4020 ± 0.0297 | 0.4670 ± 0.0129 | 0.4120 ± 0.0197 |

### Table 1.3: P1 AUPRC and Attack Prevalence (mean ± std)
| Family | Attack Prev | IsolationForest | OneClassSVM | LOF | PCA | MLP AE | VAE | Trimmed AE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DoS/DDoS** | 42.95% | 0.8413 ± 0.0138 | 0.9338 ± 0.0059 | 0.8485 ± 0.0110 | 0.9653 ± 0.0003 | 0.9867 ± 0.0022 | 0.9781 ± 0.0052 | 0.9620 ± 0.0010 |
| **PortScan** | 17.50% | 0.2315 ± 0.0231 | 0.3021 ± 0.0128 | 0.5645 ± 0.2277 | 0.3282 ± 0.0017 | 0.5094 ± 0.1132 | 0.5938 ± 0.0559 | 0.5315 ± 0.0444 |
| **BruteForce** | 2.10% | 0.0606 ± 0.0093 | 0.0218 ± 0.0033 | 0.2517 ± 0.0231 | 0.0529 ± 0.0003 | 0.0891 ± 0.0168 | 0.1432 ± 0.1373 | 0.0851 ± 0.0079 |
| **WebAttack** | 0.50% | 0.0159 ± 0.0023 | 0.0070 ± 0.0010 | 0.1222 ± 0.0131 | 0.0105 ± 0.0002 | 0.0219 ± 0.0044 | 0.0061 ± 0.0006 | 0.0156 ± 0.0028 |
| **Botnet** | 0.45% | 0.0070 ± 0.0005 | 0.0035 ± 0.0001 | 0.0048 ± 0.0003 | 0.0062 ± 0.0001 | 0.0096 ± 0.0013 | 0.0102 ± 0.0030 | 0.0084 ± 0.0020 |

### Table 1.4: Protocol P1 Per-Seed AUROC and FPR@95%TPR (MLP AE vs PCA)
| Family | Model | Seed 0 AUROC | Seed 1 AUROC | Seed 2 AUROC | Seed 0 FPR@95% | Seed 1 FPR@95% | Seed 2 FPR@95% |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DoS/DDoS** | MLP AE | 0.9880 | 0.9884 | 0.9908 | 0.0647 | 0.0575 | 0.0417 |
| **DoS/DDoS** | PCA | 0.9736 | 0.9733 | 0.9738 | 0.1787 | 0.1784 | 0.1780 |
| **PortScan** | MLP AE | 0.9363 | 0.8923 | 0.8803 | 0.1332 | 0.1771 | 0.1655 |
| **PortScan** | PCA | 0.8105 | 0.8137 | 0.8116 | 0.2059 | 0.2033 | 0.2051 |
| **BruteForce** | MLP AE | 0.8917 | 0.8445 | 0.8938 | 0.1611 | 0.2570 | 0.1603 |
| **BruteForce** | PCA | 0.8128 | 0.8130 | 0.8109 | 0.3227 | 0.3206 | 0.3241 |
| **WebAttack** | MLP AE | 0.8427 | 0.8688 | 0.8699 | 0.6861 | 0.6762 | 0.8572 |
| **WebAttack** | PCA | 0.7562 | 0.7638 | 0.7584 | 0.6259 | 0.6270 | 0.6299 |
| **Botnet** | MLP AE | 0.6631 | 0.6431 | 0.6321 | 0.7975 | 0.8930 | 0.9019 |
| **Botnet** | PCA | 0.6554 | 0.6621 | 0.6594 | 0.5719 | 0.5565 | 0.5621 |
| **Macro Average** | **MLP AE** | 0.8644 | 0.8474 | 0.8534 | 0.3685 | 0.4122 | 0.4253 |
| **Macro Average** | **PCA** | 0.8017 | 0.8052 | 0.8028 | 0.3810 | 0.3772 | 0.3799 |

## 2. Protocol P2: Temporal Split Evaluation (Monday Train -> Tue-Fri Test)

### Table 2.1: Overall Binary Metrics (Seeds 0, 1, 2)
| Model | Overall AUROC | Overall FPR@95% | Overall TPR@1% | Overall AUPRC | Prevalence |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **IsolationForest** | 0.8702 ± 0.0037 | 0.4073 ± 0.0347 | 0.0518 ± 0.0425 | 0.5831 ± 0.0163 | 20.66% |
| **OneClassSVM** | 0.8557 ± 0.0010 | 0.6141 ± 0.0038 | 0.4601 ± 0.0046 | 0.7442 ± 0.0020 | 20.66% |
| **LOF** | 0.8793 ± 0.0032 | 0.2953 ± 0.0042 | 0.0082 ± 0.0006 | 0.4912 ± 0.0057 | 20.66% |
| **PCA** | 0.9331 ± 0.0006 | 0.2211 ± 0.0111 | 0.4560 ± 0.0039 | 0.8007 ± 0.0017 | 20.66% |
| **MLP AE** | 0.9595 ± 0.0012 | 0.1567 ± 0.0144 | 0.3578 ± 0.1148 | 0.8508 ± 0.0137 | 20.66% |
| **VAE** | 0.9565 ± 0.0021 | 0.1448 ± 0.0132 | 0.2788 ± 0.1050 | 0.8390 ± 0.0237 | 20.66% |
| **Trimmed AE** | 0.9470 ± 0.0016 | 0.1655 ± 0.0087 | 0.1247 ± 0.0058 | 0.7699 ± 0.0149 | 20.66% |

### Table 2.2: Temporal Drift Check (Monday VAL-BENIGN vs Tue-Fri TEST-BENIGN)
| Model | VAL Benign Median | TEST Benign Median | Drift Ratio (Test/Val) |
| :--- | :---: | :---: | :---: |
| **IsolationForest** | 0.450919 | 0.447934 | 0.99x |
| **OneClassSVM** | -16.106747 | -15.698455 | N/A |
| **LOF** | -0.433892 | -0.392232 | N/A |
| **PCA** | 0.057243 | 0.060866 | 1.06x |
| **MLP AE** | 0.001302 | 0.001596 | 1.23x |
| **VAE** | 0.017175 | 0.022108 | 1.29x |
| **Trimmed AE** | 0.001029 | 0.001340 | 1.31x |

### Table 2.3: P2 Per-Family AUROC (Benign Test vs Held-In Family)
| Model | Botnet | BruteForce | DoS/DDoS | Infiltration | PortScan | WebAttack |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **IsolationForest** | 0.6222 ± 0.0138 | 0.7850 ± 0.0286 | 0.9236 ± 0.0062 | 0.9438 ± 0.0050 | 0.6945 ± 0.0383 | 0.8607 ± 0.0219 |
| **OneClassSVM** | 0.3637 ± 0.0020 | 0.6583 ± 0.0142 | 0.9167 ± 0.0039 | 0.8957 ± 0.0020 | 0.6729 ± 0.0079 | 0.7261 ± 0.0406 |
| **LOF** | 0.5329 ± 0.0745 | 0.8549 ± 0.0886 | 0.8750 ± 0.0029 | 0.7353 ± 0.0097 | 0.9040 ± 0.0041 | 0.9094 ± 0.0034 |
| **PCA** | 0.6949 ± 0.0006 | 0.7908 ± 0.0019 | 0.9656 ± 0.0006 | 0.9347 ± 0.0004 | 0.8404 ± 0.0006 | 0.8148 ± 0.0028 |
| **MLP AE** | 0.5839 ± 0.0109 | 0.8444 ± 0.0184 | 0.9653 ± 0.0015 | 0.9355 ± 0.0009 | 0.9641 ± 0.0033 | 0.7231 ± 0.0403 |
| **VAE** | 0.5614 ± 0.0162 | 0.8559 ± 0.0277 | 0.9641 ± 0.0024 | 0.9418 ± 0.0053 | 0.9566 ± 0.0068 | 0.6004 ± 0.1139 |
| **Trimmed AE** | 0.5766 ± 0.0066 | 0.8374 ± 0.0122 | 0.9551 ± 0.0021 | 0.9423 ± 0.0048 | 0.9425 ± 0.0031 | 0.7263 ± 0.0220 |

## 3. Label-Free Threshold Calibration & Operational Metrics

### Table 3.1: Protocol P1 Threshold Calibration (Averaged over all 7 Models, Families & Seeds)
| Threshold Method | Target FPR | Realized FPR on Test | Relative Error | Realized TPR | Precision | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **POT (q=0.01)** | 0.0100 | 0.0096 ± 0.0009 | 5.99% | 0.1195 ± 0.2602 | 0.1910 ± 0.3554 | 0.1361 ± 0.2960 |
| **POT (q=0.001)** | 0.0010 | 0.0012 ± 0.0005 | 29.46% | 0.0573 ± 0.1534 | 0.1650 ± 0.3552 | 0.0791 ± 0.2041 |
| **Empirical 99%** | 0.0100 | 0.0100 ± 0.0002 | 1.42% | 0.1264 ± 0.2618 | 0.2006 ± 0.3562 | 0.1438 ± 0.2976 |
| **Empirical 99.9%** | 0.0010 | 0.0010 ± 0.0001 | 7.10% | 0.0579 ± 0.1555 | 0.1641 ± 0.3557 | 0.0796 ± 0.2056 |
| **Mean + 3 sigma** | 0.0014 | 0.0083 ± 0.0056 | 538.88% | 0.1159 ± 0.2587 | 0.1707 ± 0.3529 | 0.1316 ± 0.2966 |
| **ORACLE, USES TEST LABELS, REFERENCE ONLY** | Oracle | 0.1523 ± 0.1240 | N/A | 0.7370 ± 0.2621 | 0.3351 ± 0.3515 | 0.3792 ± 0.3545 |

### Table 3.1.1: Protocol P1 Threshold Calibration — Per-Model Breakdown
#### Protocol P1 Calibration: MLP AE
| Threshold Method | Target FPR | Realized FPR on Test | Relative Error | Realized TPR | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **POT (q=0.01)** | 0.0100 | 0.0099 ± 0.0001 | 1.24% | 0.1745 ± 0.3432 | 0.1867 ± 0.3718 |
| **POT (q=0.001)** | 0.0010 | 0.0009 ± 0.0000 | 7.28% | 0.1318 ± 0.2720 | 0.1593 ± 0.3276 |
| **Empirical 99%** | 0.0100 | 0.0099 ± 0.0001 | 1.26% | 0.1744 ± 0.3429 | 0.1867 ± 0.3716 |
| **Empirical 99.9%** | 0.0010 | 0.0010 ± 0.0001 | 6.13% | 0.1338 ± 0.2757 | 0.1608 ± 0.3303 |
| **Mean + 3 sigma** | 0.0014 | 0.0057 ± 0.0004 | 325.09% | 0.1574 ± 0.3200 | 0.1768 ± 0.3583 |

#### Protocol P1 Calibration: PCA
| Threshold Method | Target FPR | Realized FPR on Test | Relative Error | Realized TPR | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **POT (q=0.01)** | 0.0100 | 0.0096 ± 0.0002 | 3.65% | 0.1534 ± 0.3085 | 0.1727 ± 0.3508 |
| **POT (q=0.001)** | 0.0010 | 0.0011 ± 0.0001 | 6.05% | 0.0739 ± 0.1530 | 0.1078 ± 0.2232 |
| **Empirical 99%** | 0.0100 | 0.0099 ± 0.0002 | 1.67% | 0.1539 ± 0.3086 | 0.1730 ± 0.3508 |
| **Empirical 99.9%** | 0.0010 | 0.0010 ± 0.0001 | 4.17% | 0.0733 ± 0.1518 | 0.1072 ± 0.2219 |
| **Mean + 3 sigma** | 0.0014 | 0.0156 ± 0.0004 | 1056.32% | 0.1629 ± 0.3105 | 0.1763 ± 0.3513 |

#### Protocol P1 Calibration: Isolation Forest
| Threshold Method | Target FPR | Realized FPR on Test | Relative Error | Realized TPR | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **POT (q=0.01)** | 0.0100 | 0.0099 ± 0.0009 | 8.14% | 0.0038 ± 0.0063 | 0.0069 ± 0.0124 |
| **POT (q=0.001)** | 0.0010 | 0.0010 ± 0.0004 | 35.99% | 0.0006 ± 0.0023 | 0.0012 ± 0.0045 |
| **Empirical 99%** | 0.0100 | 0.0099 ± 0.0001 | 1.27% | 0.0038 ± 0.0066 | 0.0070 ± 0.0128 |
| **Empirical 99.9%** | 0.0010 | 0.0011 ± 0.0000 | 8.24% | 0.0006 ± 0.0022 | 0.0012 ± 0.0044 |
| **Mean + 3 sigma** | 0.0014 | 0.0018 ± 0.0010 | 54.64% | 0.0006 ± 0.0024 | 0.0013 ± 0.0047 |

#### Protocol P1 Calibration: VAE
| Threshold Method | Target FPR | Realized FPR on Test | Relative Error | Realized TPR | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **POT (q=0.01)** | 0.0100 | 0.0103 ± 0.0000 | 3.29% | 0.2131 ± 0.3184 | 0.2442 ± 0.3650 |
| **POT (q=0.001)** | 0.0010 | 0.0009 ± 0.0001 | 7.68% | 0.0772 ± 0.1625 | 0.1106 ± 0.2311 |
| **Empirical 99%** | 0.0100 | 0.0099 ± 0.0002 | 1.54% | 0.2107 ± 0.3164 | 0.2429 ± 0.3635 |
| **Empirical 99.9%** | 0.0010 | 0.0011 ± 0.0000 | 5.82% | 0.0872 ± 0.1849 | 0.1204 ± 0.2520 |
| **Mean + 3 sigma** | 0.0014 | 0.0100 ± 0.0003 | 639.27% | 0.2116 ± 0.3173 | 0.2434 ± 0.3643 |

### Table 3.2: Protocol P2 Threshold Calibration (Averaged over all 7 Models & Seeds)
| Threshold Method | Target FPR | Realized FPR on Test | Relative Error | Realized TPR | Precision | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **POT (q=0.01)** | 0.0100 | 0.0343 ± 0.0129 | 242.57% | 0.4711 ± 0.2756 | 0.6903 ± 0.2569 | 0.5388 ± 0.2733 |
| **POT (q=0.001)** | 0.0010 | 0.0104 ± 0.0072 | 940.40% | 0.2865 ± 0.2222 | 0.6512 ± 0.3784 | 0.3824 ± 0.2834 |
| **Empirical 99%** | 0.0100 | 0.0345 ± 0.0127 | 244.85% | 0.4702 ± 0.2754 | 0.6902 ± 0.2609 | 0.5385 ± 0.2743 |
| **Empirical 99.9%** | 0.0010 | 0.0106 ± 0.0086 | 956.78% | 0.2827 ± 0.2227 | 0.6439 ± 0.3827 | 0.3777 ± 0.2795 |
| **Mean + 3 sigma** | 0.0014 | 0.0328 ± 0.0208 | 2344.57% | 0.4763 ± 0.3156 | 0.6283 ± 0.3037 | 0.5187 ± 0.3286 |
| **ORACLE, USES TEST LABELS, REFERENCE ONLY** | Oracle | 0.1064 ± 0.0654 | N/A | 0.8219 ± 0.1186 | 0.6969 ± 0.1225 | 0.7400 ± 0.0717 |

### Table 3.2.1: Protocol P2 Threshold Calibration — Per-Model Breakdown
#### Protocol P2 Calibration: MLP AE
| Threshold Method | Target FPR | Realized FPR on Test | Relative Error | Realized TPR | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **POT (q=0.01)** | 0.0100 | 0.0616 ± 0.0009 | 516.44% | 0.8286 ± 0.0323 | 0.8022 ± 0.0200 |
| **POT (q=0.001)** | 0.0010 | 0.0207 ± 0.0019 | 1965.74% | 0.5151 ± 0.0286 | 0.6458 ± 0.0222 |
| **Empirical 99%** | 0.0100 | 0.0610 ± 0.0005 | 510.49% | 0.8275 ± 0.0327 | 0.8024 ± 0.0195 |
| **Empirical 99.9%** | 0.0010 | 0.0250 ± 0.0049 | 2404.76% | 0.5619 ± 0.0500 | 0.6769 ± 0.0324 |
| **Mean + 3 sigma** | 0.0014 | 0.0594 ± 0.0099 | 4298.90% | 0.8062 ± 0.0401 | 0.7924 ± 0.0144 |

#### Protocol P2 Calibration: PCA
| Threshold Method | Target FPR | Realized FPR on Test | Relative Error | Realized TPR | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **POT (q=0.01)** | 0.0100 | 0.0307 ± 0.0005 | 207.32% | 0.5771 ± 0.0002 | 0.6809 ± 0.0007 |
| **POT (q=0.001)** | 0.0010 | 0.0091 ± 0.0004 | 811.22% | 0.4353 ± 0.0078 | 0.5921 ± 0.0069 |
| **Empirical 99%** | 0.0100 | 0.0285 ± 0.0007 | 185.37% | 0.5751 ± 0.0000 | 0.6827 ± 0.0012 |
| **Empirical 99.9%** | 0.0010 | 0.0075 ± 0.0002 | 649.78% | 0.3966 ± 0.0126 | 0.5564 ± 0.0126 |
| **Mean + 3 sigma** | 0.0014 | 0.0377 ± 0.0006 | 2693.66% | 0.5955 ± 0.0053 | 0.6843 ± 0.0032 |

#### Protocol P2 Calibration: Isolation Forest
| Threshold Method | Target FPR | Realized FPR on Test | Relative Error | Realized TPR | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **POT (q=0.01)** | 0.0100 | 0.0183 ± 0.0010 | 83.14% | 0.1797 ± 0.0518 | 0.2855 ± 0.0730 |
| **POT (q=0.001)** | 0.0010 | 0.0013 ± 0.0002 | 29.08% | 0.0023 ± 0.0038 | 0.0045 ± 0.0075 |
| **Empirical 99%** | 0.0100 | 0.0175 ± 0.0015 | 74.92% | 0.1846 ± 0.0880 | 0.2889 ± 0.1221 |
| **Empirical 99.9%** | 0.0010 | 0.0022 ± 0.0011 | 124.48% | 0.0024 ± 0.0039 | 0.0048 ± 0.0077 |
| **Mean + 3 sigma** | 0.0014 | 0.0064 ± 0.0022 | 374.52% | 0.0190 ± 0.0135 | 0.0362 ± 0.0255 |

#### Protocol P2 Calibration: VAE
| Threshold Method | Target FPR | Realized FPR on Test | Relative Error | Realized TPR | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **POT (q=0.01)** | 0.0100 | 0.0375 ± 0.0012 | 275.28% | 0.6895 ± 0.0316 | 0.7519 ± 0.0230 |
| **POT (q=0.001)** | 0.0010 | 0.0199 ± 0.0003 | 1888.42% | 0.4720 ± 0.0554 | 0.6085 ± 0.0488 |
| **Empirical 99%** | 0.0100 | 0.0375 ± 0.0013 | 275.43% | 0.6895 ± 0.0315 | 0.7518 ± 0.0230 |
| **Empirical 99.9%** | 0.0010 | 0.0203 ± 0.0004 | 1933.89% | 0.4825 ± 0.0490 | 0.6174 ± 0.0429 |
| **Mean + 3 sigma** | 0.0014 | 0.0496 ± 0.0015 | 3570.94% | 0.7596 ± 0.0387 | 0.7788 ± 0.0249 |

## 4. Seed-Guided Sanitization: Diagnostic & Sensitivity Analysis

### Table 4.1: Stage 4a Diagnostic on S_cal Percentiles (Setting B, c = 5%)
| Family | Model | Pool Sep AUROC | Att > p50 | Att > p90 | Att > p95 | Att > p99 | Ratio (Att/Scal) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PortScan** | Stage-1 AE | 0.9182 ± 0.0308 | 99.99% | 66.42% | 17.11% | 0.14% | 7.43x |
| **PortScan** | Stage-1 PCA | 0.8214 ± 0.0102 | 99.99% | 0.22% | 0.13% | 0.05% | 2.74x |
| **DoS/DDoS** | Stage-1 AE | 0.9761 ± 0.0014 | 99.59% | 93.39% | 88.06% | 72.68% | 45.35x |
| **DoS/DDoS** | Stage-1 PCA | 0.9718 ± 0.0019 | 99.70% | 90.27% | 88.40% | 63.62% | 20.87x |

### Table 4.2: Stage 4b POT Parameter Sensitivity (q in {0.01, 0.03, 0.05})
| Family | c (%) | Model | q | AUROC (mean ± std) | FPR@95% (mean ± std) | Attack Removed | Benign Removed |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **PortScan** | 0 | Sanitized AE | 0.01 | 0.9099 ± 0.0093 | 0.1428 ± 0.0202 | N/A (clean) | 0.92% |
| **PortScan** | 0 | Sanitized AE | 0.03 | 0.8741 ± 0.0384 | 0.2316 ± 0.0904 | N/A (clean) | 3.10% |
| **PortScan** | 0 | Sanitized AE | 0.05 | 0.9130 ± 0.0173 | 0.1456 ± 0.0458 | N/A (clean) | 5.24% |
| **PortScan** | 0 | Sanitized PCA | 0.01 | 0.8306 ± 0.0145 | 0.1838 ± 0.0181 | N/A (clean) | 1.22% |
| **PortScan** | 0 | Sanitized PCA | 0.03 | 0.8432 ± 0.0047 | 0.1720 ± 0.0051 | N/A (clean) | 3.11% |
| **PortScan** | 0 | Sanitized PCA | 0.05 | 0.8417 ± 0.0075 | 0.1732 ± 0.0072 | N/A (clean) | 5.05% |
| **PortScan** | 5 | Sanitized AE | 0.01 | 0.1706 ± 0.0415 | 0.9347 ± 0.0403 | 0.12% | 0.92% |
| **PortScan** | 5 | Sanitized AE | 0.03 | 0.2270 ± 0.1406 | 0.9737 ± 0.0400 | 15.31% | 3.10% |
| **PortScan** | 5 | Sanitized AE | 0.05 | 0.3399 ± 0.2284 | 0.8674 ± 0.0458 | 17.00% | 5.25% |
| **PortScan** | 5 | Sanitized PCA | 0.01 | 0.5002 ± 0.0049 | 0.6534 ± 0.0053 | 0.05% | 1.23% |
| **PortScan** | 5 | Sanitized PCA | 0.03 | 0.5101 ± 0.0088 | 0.6422 ± 0.0179 | 0.10% | 3.12% |
| **PortScan** | 5 | Sanitized PCA | 0.05 | 0.5068 ± 0.0183 | 0.6394 ± 0.0157 | 0.12% | 5.06% |
| **DoS/DDoS** | 0 | Sanitized AE | 0.01 | 0.9876 ± 0.0008 | 0.0586 ± 0.0041 | N/A (clean) | 0.92% |
| **DoS/DDoS** | 0 | Sanitized AE | 0.03 | 0.9849 ± 0.0018 | 0.0719 ± 0.0061 | N/A (clean) | 3.10% |
| **DoS/DDoS** | 0 | Sanitized AE | 0.05 | 0.9831 ± 0.0010 | 0.0817 ± 0.0110 | N/A (clean) | 5.24% |
| **DoS/DDoS** | 0 | Sanitized PCA | 0.01 | 0.9716 ± 0.0033 | 0.1898 ± 0.0273 | N/A (clean) | 1.22% |
| **DoS/DDoS** | 0 | Sanitized PCA | 0.03 | 0.9690 ± 0.0005 | 0.1996 ± 0.0140 | N/A (clean) | 3.11% |
| **DoS/DDoS** | 0 | Sanitized PCA | 0.05 | 0.9668 ± 0.0008 | 0.2128 ± 0.0157 | N/A (clean) | 5.05% |
| **DoS/DDoS** | 5 | Sanitized AE | 0.01 | 0.9466 ± 0.0027 | 0.2980 ± 0.0134 | 72.06% | 0.92% |
| **DoS/DDoS** | 5 | Sanitized AE | 0.03 | 0.9638 ± 0.0005 | 0.2185 ± 0.0316 | 83.39% | 3.10% |
| **DoS/DDoS** | 5 | Sanitized AE | 0.05 | 0.9705 ± 0.0026 | 0.2043 ± 0.0393 | 88.07% | 5.25% |
| **DoS/DDoS** | 5 | Sanitized PCA | 0.01 | 0.9538 ± 0.0128 | 0.2618 ± 0.0449 | 70.22% | 1.23% |
| **DoS/DDoS** | 5 | Sanitized PCA | 0.03 | 0.9511 ± 0.0086 | 0.2881 ± 0.0045 | 81.13% | 3.12% |
| **DoS/DDoS** | 5 | Sanitized PCA | 0.05 | 0.9616 ± 0.0018 | 0.2680 ± 0.0086 | 88.09% | 5.06% |

## 5. Inference Latency & System Benchmark (Single Thread)

- **CPU**: Intel64 Family 6 Model 189 Stepping 1, GenuineIntel
- **OS**: Windows 10 (Build 10.0.26200)
- **Threads**: 1 (Strictly single-threaded inference)

### Table 5.1: Inference Latency and Throughput
| Model | Parameters | Fit Wall-Clock (s) | Batch Size | Median (µs/flow) | p95 (µs/flow) | Throughput (flows/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PCA** | N/A | 1.48s | 1 |   107.40 |   253.90 |     9311.0 |
| **PCA** | N/A | 1.48s | 256 |     0.54 |     1.13 |  1859789.3 |
| **IsolationForest** | N/A | 5.64s | 1 |  3435.20 |  5589.67 |      291.1 |
| **IsolationForest** | N/A | 5.64s | 256 |    18.56 |    26.75 |    53890.8 |
| **LOF** | N/A | 111.82s | 1 |   261.95 |   570.21 |     3817.5 |
| **LOF** | N/A | 111.82s | 256 |     8.76 |    15.26 |   114216.9 |
| **OneClassSVM** | N/A | 77.97s | 1 |   116.40 |   319.74 |     8591.1 |
| **OneClassSVM** | N/A | 77.97s | 256 |    12.70 |    17.54 |    78754.7 |
| **MLP AE** | 15,132 | 385.66s | 1 |   155.10 |   390.90 |     6447.5 |
| **MLP AE** | 15,132 | 385.66s | 256 |     2.59 |     5.77 |   386094.5 |
| **VAE** | 15,660 | 415.89s | 1 |   155.20 |   413.31 |     6443.3 |
| **VAE** | 15,660 | 415.89s | 256 |     2.47 |     5.04 |   404263.8 |

## 6. Ablation Studies

### Table 6.1: Autoencoder Bottleneck Dimension Ablation (P1 LOFO)
| Family | Bottleneck b | AUROC (mean ± std) | FPR@95% (mean ± std) | TPR@1% (mean ± std) | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DoS/DDoS** | b=8 | 0.9765 ± 0.0013 | 0.1192 ± 0.0173 | 0.6902 ± 0.0198 | 264.64s |
| **DoS/DDoS** | b=16 | 0.9891 ± 0.0015 | 0.0547 ± 0.0118 | 0.8375 ± 0.0096 | 398.04s |
| **DoS/DDoS** | b=32 | 0.9803 ± 0.0019 | 0.1218 ± 0.0218 | 0.7484 ± 0.0223 | 205.23s |
| **PortScan** | b=8 | 0.9840 ± 0.0138 | 0.0222 ± 0.0109 | 0.4814 ± 0.4475 | 264.64s |
| **PortScan** | b=16 | 0.9030 ± 0.0295 | 0.1586 ± 0.0227 | 0.0046 ± 0.0002 | 398.04s |
| **PortScan** | b=32 | 0.9845 ± 0.0064 | 0.0209 ± 0.0094 | 0.3211 ± 0.5395 | 205.23s |
| **WebAttack** | b=8 | 0.7273 ± 0.0411 | 0.6398 ± 0.0861 | 0.0002 ± 0.0003 | 264.64s |
| **WebAttack** | b=16 | 0.8605 ± 0.0154 | 0.7398 ± 0.1017 | 0.0165 ± 0.0148 | 398.04s |
| **WebAttack** | b=32 | 0.8150 ± 0.0187 | 0.7447 ± 0.0476 | 0.0000 ± 0.0000 | 205.23s |

### Table 6.2: Trimmed AE Fraction Ablation (Setting B, c = 5%)
| Family | Trim Fraction | AUROC (mean ± std) | FPR@95% (mean ± std) | TPR@1% (mean ± std) | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DoS/DDoS** | 5% | 0.8027 ± 0.0015 | 0.3791 ± 0.0277 | 0.0342 ± 0.0223 | 178.44s |
| **DoS/DDoS** | 10% | 0.8007 ± 0.0063 | 0.3925 ± 0.0074 | 0.0181 ± 0.0025 | 76.62s |
| **DoS/DDoS** | 20% | 0.7676 ± 0.0121 | 0.4680 ± 0.0168 | 0.0564 ± 0.0267 | 24.24s |
| **PortScan** | 5% | 0.1225 ± 0.0488 | 0.9583 ± 0.0679 | 0.0002 ± 0.0003 | 231.61s |
| **PortScan** | 10% | 0.1416 ± 0.0710 | 0.9812 ± 0.0316 | 0.0001 ± 0.0000 | 60.87s |
| **PortScan** | 20% | 0.2078 ± 0.0391 | 0.8736 ± 0.0629 | 0.0031 ± 0.0005 | 22.67s |

## 7. Regenerated Setting B vs Defense Comparison

### Table 7.1: AUROC Comparison across Defenses (Setting B, n_seed=10,000)
| Family | c (%) | Plain AE | Trimmed AE (5%) | Sanitized AE | Plain PCA | Sanitized PCA |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PortScan** | 0% | 0.9109 ± 0.0215 | 0.9337 ± 0.0113 | 0.9099 ± 0.0093 | 0.8121 ± 0.0007 | 0.8306 ± 0.0145 |
| **PortScan** | 2% | 0.4291 ± 0.0910 | 0.4229 ± 0.0666 | 0.3840 ± 0.0607 | 0.7023 ± 0.0025 | 0.7319 ± 0.0153 |
| **PortScan** | 5% | 0.1797 ± 0.0800 | 0.1225 ± 0.0488 | 0.1706 ± 0.0415 | 0.4776 ± 0.0016 | 0.5002 ± 0.0049 |
| **PortScan** | 10% | 0.0411 ± 0.0006 | 0.0448 ± 0.0043 | 0.0677 ± 0.0443 | 0.2268 ± 0.0016 | 0.2303 ± 0.0107 |
| **DoS/DDoS** | 0% | 0.9867 ± 0.0012 | 0.9799 ± 0.0026 | 0.9876 ± 0.0008 | 0.9735 ± 0.0002 | 0.9716 ± 0.0033 |
| **DoS/DDoS** | 2% | 0.8629 ± 0.0137 | 0.8750 ± 0.0207 | 0.9651 ± 0.0047 | 0.9298 ± 0.0007 | 0.9663 ± 0.0057 |
| **DoS/DDoS** | 5% | 0.7996 ± 0.0127 | 0.8027 ± 0.0015 | 0.9466 ± 0.0027 | 0.8942 ± 0.0006 | 0.9538 ± 0.0128 |
| **DoS/DDoS** | 10% | 0.7797 ± 0.0065 | 0.7811 ± 0.0062 | 0.9286 ± 0.0012 | 0.7545 ± 0.0009 | 0.9272 ± 0.0136 |
