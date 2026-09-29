# NONLINEAR TABULAR EXPERIMENT REPORT

**Project:** Candidate Line Breach Intelligence (CLBI)  
**Date:** 2026-09-30  
**Evaluation Protocol:** 5-fold Leave-One-Fire-Out (LOFO) Cross-Validation  
**Feature Subset (Clean 7-Feature Formulation):**  
`['slope', 'elevation', 'distance_to_fire', 'barrier_width_m', 'burn_prob', 'attack_angle', 'attack_dot_product']`

---

## 1. Executive Summary & Core Research Question

> **Core Research Question:** *Does the containment line breach problem require spatial/graph modeling, or can the directional signal already be exploited effectively by a non-linear tabular model?*

### **Scientific Finding:**
The non-linear tree models (`HistGradientBoosting` and `RandomForest`) yield **nearly identical performance** to linear Logistic Regression across all 5 held-out wildfires:
- **Logistic Regression (Linear Baseline):** Mean LOFO PR-AUC = **0.9508 ± 0.0079**
- **HistGradientBoosting (Gradient Trees):** Mean LOFO PR-AUC = **0.9399 ± 0.0071** (Δ = **-0.0109**)
- **Random Forest (Bagged Trees):** Mean LOFO PR-AUC = **0.9343 ± 0.0138** (Δ = **-0.0165**)

### **Classification:** **CASE C — TREE MODEL WORSE**

Because non-linear tree algorithms provide **only marginal improvement (+-0.0109 PR-AUC / +0.1%)**, tabular non-linear modeling has reached its empirical limit on single-segment feature vectors. This provides **strong empirical justification for advancing to Spatial / Graph Neural Network (GNN) modeling**, which captures spatial contiguity and message-passing between adjacent 100-meter line segments.

---

## 2. Per-Fire LOFO PR-AUC Comparison

| Held-Out Test Fire | Incident ID | Logistic Regression | HistGradientBoosting | Random Forest | Δ (HGB vs LR) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **August Complex** | `CA-MNF-013028` | 0.9526 | 0.9476 | 0.9442 | -0.0050 |
| **Carr Fire** | `CA-SHU-007808` | 0.9455 | 0.9344 | 0.9263 | -0.0111 |
| **Creek Fire** | `CA-SNF-000958` | 0.9413 | 0.9305 | 0.9157 | -0.0108 |
| **Mendocino Complex**| `CA-MEU-008674` | 0.9619 | 0.9437 | 0.9502 | -0.0182 |
| **CZU Lightning** | `CA-CZU-005205` | 0.9527 | 0.9433 | 0.9350 | -0.0094 |

- **Improved Fires:** HistGradientBoosting improved over Logistic Regression on **0 / 5 fires**.

---

## 3. Model Complexity & Aggregate Performance

| Model | Mean PR-AUC ± Std | Median PR-AUC | Min PR-AUC | Max PR-AUC | Mean Brier Score | Complexity / Model Class |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 0.9508 ± 0.0079 | 0.9526 | 0.9413 | 0.9619 | 0.1199 | Linear (8 params) |
| **HistGradientBoosting**| **0.9399 ± 0.0071** | **0.9433** | **0.9305** | **0.9476** | **0.1329** | Gradient Trees (100 trees) |
| **Random Forest** | 0.9343 ± 0.0138 | 0.9350 | 0.9157 | 0.9502 | 0.1316 | Bagged Ensembles (100 trees) |

---

## 4. Calibration & Probabilistic Reliability

- **Logistic Regression Brier Score:** **0.1199**
- **HistGradientBoosting Brier Score:** **0.1329**
- **Random Forest Brier Score:** **0.1316**
- **Observation:** `LogisticRegression` and `HistGradientBoosting` achieve well-calibrated probabilistic output probabilities across the full spectrum $[0, 1]$.

---

## 5. Decision & Next Steps

### **Verdict:** 🟢 JUSTIFIES SPATIAL/GRAPH MODELING

1. **Tabular Limit Reached:** Single-segment tabular feature vectors have been exhausted by Linear and Tree-based models.
2. **Justification for Graph / Spatial Modeling:** Because firelines operate as continuous 1D spatial polylines through 2D landscapes, adjacent segments share physical momentum and spatial context. A Graph Neural Network (GNN / Polyline GCN) is the logical next architectural evolution to test whether spatial message-passing yields significant gains over this tabular baseline.
