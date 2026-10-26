# NONLINEAR TABULAR EXPERIMENT REPORT

> **SYNTHETIC DATA. NOT EMPIRICAL EVIDENCE.** `fireline_segments_v1` is simulated by `src/data/build_dataset.py` (random terrain/geometry and a hand-written logistic label rule). Fire names and incident IDs are scenario labels only; no data from those incidents is used. These results test the CLBI software pipeline, not real fireline behaviour, and are outside the Phase 1 NDWS evaluation (see `docs/CLBI_SYNTHETIC_DATASET_CARD.md`).

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
- **Random Forest (Bagged Trees):** Mean LOFO PR-AUC = **0.9336 ± 0.0141** (Δ = **-0.0172**)

### **Classification:** **CASE C — TREE MODEL WORSE**

Tree models are worse than logistic regression (-0.0109 PR-AUC). This is expected: the synthetic generator (`src/data/build_dataset.py`) assigns labels with a linear logistic rule, so a linear model is the correctly specified model. A result like this says nothing about whether graph/spatial models would help.

---

## 2. Per-Fire LOFO PR-AUC Comparison

| Held-Out Test Fire | Incident ID | Logistic Regression | HistGradientBoosting | Random Forest | Δ (HGB vs LR) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **August Complex** | `CA-MNF-013028` | 0.9526 | 0.9476 | 0.9444 | -0.0050 |
| **Carr Fire** | `CA-SHU-007808` | 0.9455 | 0.9344 | 0.9225 | -0.0111 |
| **Creek Fire** | `CA-SNF-000958` | 0.9413 | 0.9305 | 0.9169 | -0.0108 |
| **Mendocino Complex**| `CA-MEU-008674` | 0.9619 | 0.9437 | 0.9502 | -0.0182 |
| **CZU Lightning** | `CA-CZU-005205` | 0.9527 | 0.9433 | 0.9341 | -0.0094 |

- **Improved Fires:** HistGradientBoosting improved over Logistic Regression on **0 / 5 fires**.

---

## 3. Model Complexity & Aggregate Performance

| Model | Mean PR-AUC ± Std | Median PR-AUC | Min PR-AUC | Max PR-AUC | Mean Brier Score | Complexity / Model Class |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 0.9508 ± 0.0079 | 0.9526 | 0.9413 | 0.9619 | 0.1199 | Linear (8 params) |
| **HistGradientBoosting**| **0.9399 ± 0.0071** | **0.9433** | **0.9305** | **0.9476** | **0.1329** | Gradient Trees (100 trees) |
| **Random Forest** | 0.9336 ± 0.0141 | 0.9341 | 0.9169 | 0.9502 | 0.1320 | Bagged Ensembles (100 trees) |

---

## 4. Calibration & Probabilistic Reliability

- **Logistic Regression Brier Score:** **0.1199**
- **HistGradientBoosting Brier Score:** **0.1329**
- **Random Forest Brier Score:** **0.1320**
- **Observation:** `LogisticRegression` and `HistGradientBoosting` achieve well-calibrated probabilistic output probabilities across the full spectrum $[0, 1]$.

---

## 5. Decision & Next Steps

### **Verdict:** 🔴 NO NONLINEAR GAIN -> THE LINEAR MODEL IS SUFFICIENT FOR THESE FEATURES

1. Tree models are worse than logistic regression (-0.0109 PR-AUC). This is expected: the synthetic generator (`src/data/build_dataset.py`) assigns labels with a linear logistic rule, so a linear model is the correctly specified model. A result like this says nothing about whether graph/spatial models would help.
2. Whether spatial / graph models help must be decided by a separate ablation (see `SPATIAL_ABLATION_REPORT.md`), not by this linear-vs-tree comparison.
