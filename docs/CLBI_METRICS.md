# CLBI Benchmark Evaluation Metrics Documentation

**Project:** Candidate Line Breach Intelligence (CLBI)  
**Document Version:** v1.0  
**Evaluation Protocol:** 5-fold Leave-One-Fire-Out (LOFO) Cross-Validation  

---

## Metric Definitions & Benchmark Results

### 1. Precision-Recall AUC (PR-AUC)
- **Definition:** Area under the Precision-Recall curve. Primary evaluation metric due to class imbalance (~29.9% breach prevalence).
- **Why It Matters:** Evaluates how effectively the model balances precision and recall without being inflated by true negatives.
- **Evaluation Protocol:** 5-fold Leave-One-Fire-Out (each fire held out).
- **Reported Result:** **0.9508 ± 0.0076**

---

### 2. Brier Score
- **Definition:** Mean squared error between predicted probability $P(\text{Breach})$ and actual binary outcome $Y \in \{0, 1\}$.
- **Why It Matters:** Evaluates overall probability accuracy and calibration penalty.
- **Evaluation Protocol:** 5-fold LOFO across 4,600 held-out test predictions.
- **Reported Result:** **0.1199**

---

### 3. Expected Calibration Error (ECE)
- **Definition:** Sample-weighted absolute difference between mean predicted probability and observed breach rate across equal-width probability bins ($\text{ECE} = \sum \frac{n_b}{N} |\text{acc}_b - \text{conf}_b|$).
- **Why It Matters:** Quantifies overall probability calibration.
- **Evaluation Protocol:** 10 equal-width bins and 20 equal-width bins over 4,600 held-out predictions.
- **Reported Result:** **0.0109 (10 Bins)** | **0.0151 (20 Bins)**

---

### 4. Top-10% Inspection Precision
- **Definition:** Percentage of actual breaches among the top 10% highest-risk predicted segments.
- **Why It Matters:** Measures prioritization accuracy when resources can inspect only 10% of candidate line.
- **Evaluation Protocol:** Evaluated per held-out fire and averaged across 5 folds.
- **Reported Result:** **89.6% Mean** (Range: 84.0% to 95.8%)

---

### 5. Top-20% Breach Recall
- **Definition:** Percentage of ALL actual line breaches captured within the top 20% highest-risk predicted segments.
- **Why It Matters:** Measures vulnerability capture coverage under resource constraints.
- **Evaluation Protocol:** Evaluated per held-out fire and averaged across 5 folds.
- **Reported Result:** **53.8% Mean** (Range: 48.8% to 57.2%)

---

### 6. Offline Breach-Yield Enrichment Ratio
- **Definition:** Ratio of breaches captured per 10 segments using model top-10% selection vs random selection ($8.74 / 2.99$).
- **Why It Matters:** Measures retrospective offline prioritization yield improvement over unprioritized inspection.
- **Evaluation Protocol:** 100-segment standardized offline simulation.
- **Reported Result:** **2.92× versus Random Selection**
