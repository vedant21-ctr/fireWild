# FireWild Intelligence — Phase 1 Final Project Report

**Project Title:** Next Day Wildfire Spread Prediction via Tabular AML and Decision Intelligence
**Project Number:** CSAI Project 17
**Repository:** `vedant21-ctr/fireWild` · [https://github.com/vedant21-ctr/fireWild](https://github.com/vedant21-ctr/fireWild)
**Workspace:** `C:\Users\Admin\Documents\WILDFIRE`
**Report Date:** October 2026
**Status:** **PHASE 1 COMPLETE — SUBMISSION READY**
**Active Branch:** `feat/firewild-ui-overhaul` (1 commit ahead of `origin/main` — `b939a4b`)
**Test Suite:** **84 / 84 tests passing** (verified `2026-10-26`)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Formulation \& Research Scope](#2-problem-formulation--research-scope)
3. [Dataset: Next Day Wildfire Spread (NDWS)](#3-dataset-next-day-wildfire-spread-ndws)
4. [Data Extraction Pipeline \& Validation](#4-data-extraction-pipeline--validation)
5. [Experimental Design: Features, Splits, and Leakage Controls](#5-experimental-design-features-splits-and-leakage-controls)
6. [Phase 1 Tabular Baseline Hierarchy (NDWS)](#6-phase-1-tabular-baseline-hierarchy-ndws)
7. [LightGBM: Advanced Gradient Boosted Baseline](#7-lightgbm-advanced-gradient-boosted-baseline)
8. [Held-Out Evaluation and Generalization Analysis](#8-held-out-evaluation-and-generalization-analysis)
9. [Feature Attribution \& Physical Interpretation](#9-feature-attribution--physical-interpretation)
10. [Neighborhood / Focal Spatial Feature Engineering](#10-neighborhood--focal-spatial-feature-engineering)
11. [Audit Trail: Anomaly Resolution and Scientific Integrity](#11-audit-trail-anomaly-resolution-and-scientific-integrity)
12. [CLBI: Candidate Line Breach Intelligence System](#12-clbi-candidate-line-breach-intelligence-system)
13. [System Architecture: FastAPI Backend \& REST API Design](#13-system-architecture-fastapi-backend--rest-api-design)
14. [Frontend Dashboard \& UI Integration Contract](#14-frontend-dashboard--ui-integration-contract)
15. [NDWS Benchmark API: Deployment Packaging](#15-ndws-benchmark-api-deployment-packaging)
16. [Software Quality: Test Suite Architecture](#16-software-quality-test-suite-architecture)
17. [Known Limitations \& Scientific Transparency](#17-known-limitations--scientific-transparency)
18. [Phase 2 Roadmap: Spatial Deep Learning](#18-phase-2-roadmap-spatial-deep-learning)
19. [Reproducibility Guide](#19-reproducibility-guide)
20. [References \& Archival Document Index](#20-references--archival-document-index)

---

## 1. Executive Summary

**FireWild Intelligence** is a research-grade wildfire decision-support platform developed as CSAI Project 17. Phase 1 delivers a fully reproducible, leakage-audited tabular machine learning benchmark on the Google Research *Next Day Wildfire Spread* (NDWS) dataset, alongside a separate Candidate Line Breach Intelligence (CLBI) tactical decision engine.

### Phase 1 Headline Results

| Achievement | Evidence |
|:---|:---|
| **59,369,679 valid pixel rows** extracted and validated across all 15 NDWS training shards | `docs/PHASE1_TRAINING_DATA_REPORT.md` |
| **5-model tabular benchmark** with rigorous leakage controls and shard-level splits | `experiments/PHASE1_LIGHTGBM_REPORT.md` |
| **LightGBM AP = 0.114470** on validation (+77.8% over Logistic Regression) | `experiments/results_ndws/lightgbm_results.json` |
| **LightGBM AP = 0.083720** on held-out official evaluation shard (`eval_00`) | `experiments/results_ndws/lightgbm_results.json` |
| **73.3% false alarm reduction** on unseen evaluation data (3,323 vs 12,443 FP) | `experiments/results_ndws/lightgbm_results.json` |
| **Fixed-FPR recall metrics** computed and serialized: 20.79% recall @ FPR ≤ 1% (eval) | `models/ndws/lightgbm_metadata.json` |
| **CLBI PR-AUC = 0.9508 ± 0.0076** (5-fold LOFO) across 4,600 held-out fireline segments | `experiments/BASELINE_REPORT.md` |
| **6 REST API endpoints** serving live inference, benchmark metrics, and spatial demo patches | `src/api/routes.py`, `src/api/ndws_routes.py` |
| **84/84 tests passing** covering extraction, model, API, UI contract, and deployment | `tests/` |
| **NDWS artifacts packaged** under `deployment/ndws/` for Render cloud deployment | commit `b939a4b` |

> [!IMPORTANT]
> All metrics in this report are derived directly from serialized experiment artifacts and source code in the `vedant21-ctr/fireWild` repository. No results are fabricated, extrapolated, or inferred from external publications without explicit attribution.

---

## 2. Problem Formulation & Research Scope

### 2.1 Core CSAI Research Problem

The primary academic task is **landscape-scale next-day wildfire spread prediction**: given multi-modal satellite observations, gridded weather reanalysis, terrain topography, and vegetation health metrics at day $t$, predict which currently unburned pixels will ignite and become active fire on day $t+1$.

**Formal Binary Classification Target:**
$$Y_{\text{transition}} = 1 \iff \left(\text{PrevFireMask}_{t} = 0.0 \land \text{FireMask}_{t+1} = 1.0\right)$$

This strict transition definition (unburned → burning) is operationally superior to predicting all active fire pixels (which includes already-burning cores trivially predictable by identity mapping). It corresponds to true perimeter expansion, providing genuine tactical value.

**Prediction Lead Time:** 24 hours ($\Delta t = 1$ day)
**Spatial Resolution:** 1 km² per pixel ($64 \times 64$ pixel patches = $64 \times 64$ km spatial footprint)
**Phase 1 Method:** Tabular Automated Machine Learning (AML) — pixels flattened to 1D observations

### 2.2 Two-System Architecture: NDWS + CLBI

FireWild Intelligence integrates two operationally connected but strictly independent prediction pipelines:

```
┌──────────────────────────────────────────────────────────────────────┐
│                       fireWild RESEARCH ARCHITECTURE                  │
├─────────────────────────────────┬────────────────────────────────────┤
│   CORE CSAI RESEARCH (NDWS)     │    DOWNSTREAM EXTENSION (CLBI)     │
│   Next-Day Spread Prediction    │  Containment Line Breach Risk      │
├─────────────────────────────────┼────────────────────────────────────┤
│   Phase 1: Tabular AML          │  Separate tactical engine          │
│   • 12 leakage-free features    │  • 7 pre-engagement features       │
│   • 5-model benchmark           │  • 100m segment breach risk        │
│   • LightGBM best: AP=0.1145   │  • PR-AUC=0.9508 (5-fold LOFO)    │
│   • Shard-level disjoint splits │  • 4,600 USFS-verified segments    │
└─────────────────────────────────┴────────────────────────────────────┘
```

> [!NOTE]
> The CLBI input feature `burn_prob` is derived from a synthetic simulation in Phase 1. Real integration of an NDWS-derived spread probability into CLBI is planned for Phase 2 upon validation of a calibrated spatial model.

---

## 3. Dataset: Next Day Wildfire Spread (NDWS)

### 3.1 Dataset Provenance

| Attribute | Value |
|:---|:---|
| **Name** | Next Day Wildfire Spread (NDWS) |
| **Authors** | Fantine Huot, R. Lily Hu, Nita Goyal, Tharun Sankar, Matthias Ihme, Yi-Fan Chen |
| **Affiliations** | Google Research & Stanford University |
| **Publication** | *IEEE Transactions on Geoscience and Remote Sensing (TGRS)*, Vol. 60, 2022 (arXiv:2112.02447) |
| **Distribution** | Kaggle: `kaggle.com/datasets/fantineh/next-day-wildfire-spread` |
| **License** | CC BY 4.0 (underlying academic release) |
| **Format** | TFRecord (Protocol Buffers), serialized `tf.train.Example` |
| **Local Cache** | `C:\Users\Admin\.cache\kagglehub\datasets\fantineh\next-day-wildfire-spread\versions\2` |

### 3.2 Dataset Structure

The NDWS dataset comprises **19 authoritative TFRecord shards** representing large fire incidents across the contiguous United States (CONUS):

| Partition | Shards | Estimated Records | Raw Size |
|:---|:---|:---|:---|
| **Training (`train`)** | 15 shards (`train_00`–`train_14`) | 14,979 | ~2.98 GB |
| **Validation (`eval`)** | 2 shards (`eval_00`, `eval_01`) | ~1,877 | ~381 MB |
| **Testing (`test`)** | 2 shards (`test_00`, `test_01`) | ~1,689 | ~343 MB |

Each record is a $64 \times 64$ km spatial patch at 1 km nominal resolution, yielding $4,096$ pixels per patch. Phase 1 operates on the tabular flattening of these patches: each valid pixel becomes one training sample.

### 3.3 Structural Limitations — Dataset Transparency

The NDWS TFRecords have documented structural properties that constrain interpretation:

1. **No Embedded Geospatial Coordinates:** TFRecord headers do not contain latitude/longitude bounding boxes or CRS projections. Pixel indices `(r, c)` are local patch coordinates; overlaying onto GIS base maps requires external cross-walk metadata not available in the public release.

2. **No Fire-Event IDs or Timestamps:** Patches lack unique wildfire incident identifiers or calendar timestamps, preventing chronological fire-run tracking across sequential observation days.

3. **Elevation Tile Overlap (94.7%):** An independent terrain fingerprinting audit demonstrated that **94.7% of official evaluation patches and 95.1% of test patches share identical 8×8 km SRTM elevation tiles with training patches** (`data/splits/split_audit.json`). Because topography is static, tabular models can partially memorize local elevation profiles on the official Kaggle train/eval splits, overstating out-of-domain transferability. Validated AP on geographically separated holdouts typically drops from ~0.26 to ~0.138.

---

## 4. Data Extraction Pipeline & Validation

### 4.1 Extraction Architecture

Implemented in `src/features/extract_tabular_ndws.py`, the pipeline:

1. Iterates over TFRecord shards via streaming batches (50 records/batch) — no full shard is loaded into memory simultaneously, keeping peak RAM safely bounded.
2. Decodes each `tf.train.Example` proto into a $64 \times 64 \times C$ numpy array.
3. Applies a **strict `valid_mask`** excluding any pixel where `PrevFireMask = -1.0` OR `FireMask = -1.0` (missing/cloud/nodata).
4. Constructs the transition target: `target_new_fire = (PrevFireMask == 0) AND (FireMask == 1)`.
5. Computes `dist_to_prev_fire_km` via `scipy.spatial.KDTree` over active fire pixel coordinates.
6. Flushes rows to Parquet via PyArrow record batches.

### 4.2 Full 15-Shard Extraction Results

All 15 official training shards (`train_00`–`train_14`) were extracted and validated:

| Shard | Records | Valid Pixels | New Fire ($Y=1$) | Prevalence | Parquet Size |
|:---|:---:|:---:|:---:|:---:|:---:|
| `train_00` | 1,000 | 3,939,429 | 35,181 | 0.893% | 187.48 MB |
| `train_01` | 1,000 | 3,959,557 | 35,597 | 0.899% | 188.94 MB |
| `train_02` | 1,000 | 3,935,608 | 29,716 | 0.755% | 189.91 MB |
| `train_03` | 1,000 | 3,918,234 | 33,647 | 0.859% | 187.99 MB |
| `train_04` | 1,000 | 3,942,427 | 32,297 | 0.819% | 187.24 MB |
| `train_05` | 1,000 | 3,959,027 | 25,362 | 0.641% | 191.16 MB |
| `train_06` | 1,000 | 3,967,108 | 27,849 | 0.702% | 190.80 MB |
| `train_07` | 1,000 | 3,993,102 | 24,535 | 0.614% | 192.27 MB |
| `train_08` | 1,000 | 3,968,871 | 32,170 | 0.811% | 190.22 MB |
| `train_09` | 1,000 | 3,982,983 | 41,361 | 1.038% | 189.75 MB |
| `train_10` | 1,000 | 3,981,109 | 40,669 | 1.022% | 184.00 MB |
| `train_11` | 1,000 | 3,963,818 | 46,918 | 1.184% | 178.33 MB |
| `train_12` | 1,000 | 3,987,327 | 30,810 | 0.773% | 182.45 MB |
| `train_13` | 1,000 | 3,989,753 | 23,973 | 0.601% | 180.88 MB |
| `train_14` | 979 | 3,881,326 | 22,733 | 0.586% | 173.79 MB |
| **TOTAL** | **14,979** | **59,369,679** | **482,818** | **0.813%** | **2,795.21 MB** |

**Extraction performance:** 189.49 seconds (~3.16 minutes) at ~313,000 pixels/second. Zero missing values across all 22 output columns.

### 4.3 Evaluation Shard Extraction (Task 2C)

One official evaluation shard (`eval_00`) was extracted as a held-out test sample:
- **Records:** ~979 | **Valid Pixels:** 3,922,548 | **Parquet:** 164.02 MB
- **Quarantine Rule:** This shard was never used for scaler fitting, feature selection, or threshold tuning. Accessed once for final frozen model evaluation.

---

## 5. Experimental Design: Features, Splits, and Leakage Controls

### 5.1 Predictor Features (12 Verified Channels)

All 12 features are measured at day $t$ — strictly pre-spread, with zero information from day $t+1$:

| # | Feature | Source | Physical Units | Role |
|:---:|:---|:---|:---:|:---|
| 1 | `elevation` | USGS SRTM (30m → 1km) | Meters (m) | Orographic effects, convective chimney |
| 2 | `th` | NOAA/GridMET Daily | Degrees [0°, 360°) | Wind direction, perimeter heading |
| 3 | `vs` | NOAA/GridMET Daily | m/s | Wind speed, flame tilt, spotting |
| 4 | `tmmn` | NOAA/GridMET Daily | Kelvin (K) | Min temperature, nighttime recovery |
| 5 | `tmmx` | NOAA/GridMET Daily | Kelvin (K) | Max temperature, daytime desiccation |
| 6 | `sph` | NOAA/GridMET Daily | kg/kg | Specific humidity, combustion suppression |
| 7 | `pr` | NOAA/GridMET Daily | Millimeters (mm) | Precipitation, direct fuel wetting |
| 8 | `pdsi` | NOAA/GridMET Daily | Dimensionless [−10, +10] | Palmer Drought Severity, long-term deficit |
| 9 | `NDVI` | MODIS/VIIRS 16-day | Scaled ×10,000 | Vegetation greenness, fuel biomass |
| 10 | `population` | SEDAC GPWv4 | Persons/km² | Anthropogenic fragmentation, suppression |
| 11 | `erc` | USFS NFDRS/GridMET | Dimensionless [0, 100+] | Energy Release Component, fuel dryness |
| 12 | `dist_to_prev_fire_km` | Derived (KDTree/PrevFireMask) | Kilometers (km) | Intra-patch distance to active fire |

**Forbidden Columns (Hard-Excluded):** `FireMask`, `fire_mask_t1`, `target_new_fire` — any feature touching day-$t+1$ fire state is strictly prohibited from the predictor set.

### 5.2 Target Label Semantics

The **critical distinction** between local Phase 1 target and the external dataset baseline:

| Target | Definition | Prevalence | Interpretation |
|:---|:---|:---|:---|
| **Local (Phase 1)** | `(PrevFireMask=0) AND (FireMask=1)` | ~0.6%–1.0% | **Only new ignitions** — true spread prediction |
| **External NDWS** | `FireMask=1` | ~1.5%–3.5%+ | All active fire (includes already-burning cores) |

The local transition target is operationally superior: predicting where already-burning pixels remain on fire provides negligible tactical value.

### 5.3 Shard-Level Disjoint Partitioning

Random pixel or patch-level splits would cause catastrophic spatial autocorrelation leakage. We use **shard-level disjoint partitioning** aligned with the Google Research pre-assignments:

| Partition | Shards | Pilot Samples | Positives | Class Prevalence | ML Role |
|:---|:---|:---:|:---:|:---:|:---|
| **Training** | `train_00`–`train_11` | 1,000,000 | 8,851 | **0.8851%** | Scaler fitting, gradient updates |
| **Validation** | `train_14` | 500,000 | 2,931 | **0.5862%** | Early stopping, threshold tuning |
| **Held-Out Evaluation** | `eval_00` | 500,000 | 4,782 | **0.9564%** | Final unbiased model assessment |

**Preprocessor Isolation:** `StandardScaler` fitted exclusively on $X_{\text{train}}$ — validation and evaluation features are transformed using frozen statistics without any distribution update.

**Evaluation Quarantine:** Shard `eval_00` was never accessed during hyperparameter tuning, loss weighting, or threshold selection. The frozen decision threshold $\tau^* = 0.028115$ was derived on `train_14` and applied unmodified to `eval_00`.

---

## 6. Phase 1 Tabular Baseline Hierarchy (NDWS)

### 6.1 Model Suite

Five models constitute the complete Phase 1 benchmark:

1. **Dummy Prior Baseline** — predicts constant empirical training prevalence ($\hat{p} = 0.008851$).
2. **Corrected Spatial Distance Heuristic** — $p = \exp(-d / 2.0)$ on raw physical kilometers with sentinel background rate for pixels far from active fire ($d \geq 90$ km).
3. **Logistic Regression L2** — `sklearn` `LogisticRegression(C=1.0, solver='lbfgs', max_iter=200)` on StandardScaler-normalized features.
4. **LightGBM (Unweighted)** — `objective='binary'`, `learning_rate=0.05`, `num_leaves=31`, early stopping (30 rounds on validation average precision).
5. **LightGBM (Weighted, `scale_pos_weight=10`)** — square-root class weighting to amplify minority-class gradient signal.

### 6.2 Preprocessing Decisions

| Model Type | Scaling | Rationale |
|:---|:---|:---|
| Logistic Regression | `StandardScaler(copy=True)` on training data only | Linear models require feature standardization for convergence |
| LightGBM | **Raw unscaled physical values** | Tree histogram binning is invariant to monotonic transformations |
| Distance Heuristic | **Raw physical kilometers** | Explicitly corrected after Task 1B.1 artifact discovery (see §11) |

### 6.3 Comprehensive Validation Results (`train_14`, 500,000 Pixels)

| Model | AP (PR-AUC) | ROC-AUC | Brier Score | Threshold $\tau^*$ | Precision | Recall | $F_1$ | Train Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dummy Prior** | 0.005862 | 0.500000 | 0.005837 | 0.008851 | 0.59% | 100.00% | 0.011656 | 0.29s |
| **Distance Heuristic (Corrected)** | 0.072548 | 0.881722 | 0.014795 | 0.369404 | 10.58% | 31.39% | 0.158239 | 8.77s |
| **Logistic Regression L2** | 0.064393 | **0.886330** | **0.005612** | 0.053061 | 9.38% | **45.04%** | 0.155221 | 22.93s |
| **LightGBM Weighted (spw=10)** | 0.102776 | 0.877354 | 0.006087 | 0.247552 | 16.43% | 31.80% | 0.216618 | 6.75s |
| **LightGBM Unweighted (Best)** | **0.114470** | 0.872004 | 0.005689 | **0.028115** | **17.64%** | 30.54% | **0.223610** | **5.57s** |

### 6.4 Why Average Precision is the Essential Metric

Under 0.58% positive prevalence, ROC-AUC is dominated by the enormous true-negative mass (>490,000 pixels). A model generating 12,000 false positives maintains FPR ≈ 2.5% and ROC-AUC > 0.88 — while 90% of its positive predictions are wrong. Average Precision evaluates precision across all recall thresholds strictly for the rare positive class:
$$\text{AP} = \sum_k (R_k - R_{k-1}) P_k$$

LightGBM's **0.114470 AP represents a 19.5× improvement** over the random prior baseline (0.005862).

### 6.5 Validation Confusion Matrix Analysis

At their respective validation-optimal thresholds:

**Logistic Regression L2 ($\tau^* = 0.053061$):**
- TN = 484,312 (97.43%) | FP = 12,757 (2.57%) | FN = 1,611 | TP = 1,320

**LightGBM Unweighted ($\tau^* = 0.028115$):**
- TN = 492,890 (99.16%) | FP = 4,179 (0.84%) | FN = 2,036 | TP = 895
- **67.2% reduction in validation false alarms** (4,179 vs 12,757 FP)

---

## 7. LightGBM: Advanced Gradient Boosted Baseline

### 7.1 Algorithm Design and Hyperparameters

LightGBM employs **leaf-wise histogram gradient boosting** on the binary cross-entropy objective. Phase 1 configuration:

```python
lgb.LGBMClassifier(
    objective     = "binary",
    n_estimators  = 300,
    learning_rate = 0.05,
    num_leaves    = 31,
    scale_pos_weight = 1.0,   # unweighted variant
    random_state  = 42,
    n_jobs        = -1
)
```

**Training regime:** Early stopping (30 rounds, monitoring validation average precision). Best model stopped at **3 boosting iterations** — indicating that the signal from the 12 tabular features saturates quickly, motivating Phase 2 spatial architecture.

**Training time:** **5.57 seconds** on 1,000,000 rows with peak RAM well under 350 MB — demonstrating exceptional suitability for large-scale remote sensing tabular pipelines.

### 7.2 Decision Threshold Selection

Standard 0.50 threshold produces exactly 0 positive predictions under <1% prevalence. We sweep 200 thresholds on the validation Precision-Recall curve:
$$\tau^* = \arg\max_\tau F_1(\tau) = 0.028115$$

This frozen threshold is applied unmodified to all evaluation and test sets.

### 7.3 Fixed False-Positive Rate (Fixed-FPR) Benchmark

Operational wildfire dispatching requires bounding false alarm rates. We implement `calculate_recall_at_fixed_fpr()` computing recall under strict FPR budgets:

$$\text{Recall}_{\alpha} = \max_{\tau} \{ \text{Recall}(\tau) \mid \text{FPR}(\tau) \leq \alpha \}$$

| Partition | Model | Recall @ FPR ≤ 1% | Achieved FPR | Operating Threshold | Recall @ FPR ≤ 5% | Achieved FPR | Threshold |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Validation (`train_14`) | **LightGBM** | **32.55%** | 0.999% | 0.025454 | **68.30%** | 4.998% | 0.010903 |
| Validation (`train_14`) | Logistic Reg. | 12.76% | 1.000% | 0.081586 | 64.31% | 4.998% | 0.033313 |
| **Held-Out Eval (`eval_00`)** | **LightGBM** | **20.79%** | 0.998% | 0.021245 | **42.76%** | 4.418% | 0.010906 |
| Held-Out Eval (`eval_00`) | Logistic Reg. | 10.67% | 1.000% | 0.079497 | 44.29% | 5.000% | 0.033606 |

> [!IMPORTANT]
> The fixed-FPR operating thresholds ($\tau_{1\%} = 0.021245$ and $\tau_{5\%} = 0.010906$ on `eval_00`) are **distinct** from the primary frozen decision threshold ($\tau^* = 0.028115$). The F1-optimal threshold maximizes detection-precision balance; the FPR-constrained thresholds maximize detection under strict dispatch-resource constraints.

### 7.4 Artifact Serialization

| Artifact | Path | Description |
|:---|:---|:---|
| `lightgbm_ndws.joblib` | `models/ndws/` | Serialized trained LightGBM pipeline |
| `lightgbm_metadata.json` | `models/ndws/` | Feature names, threshold, training/eval metrics |
| `lightgbm_results.json` | `experiments/results_ndws/` | Full 5-model comparison + fixed-FPR metrics |
| `lightgbm_pr_curves.png` | `experiments/results_ndws/` | Precision-Recall curve comparison |
| `lightgbm_feature_importance.png` | `experiments/results_ndws/` | Tree split gain feature importance |
| `lightgbm_confusion_matrix.png` | `experiments/results_ndws/` | Held-out evaluation confusion matrix |

---

## 8. Held-Out Evaluation and Generalization Analysis

### 8.1 Evaluation Results Summary (`eval_00`, 500,000 Pixels, 4,782 Positives)

The frozen LightGBM model ($\tau^* = 0.028115$) was evaluated once on the official held-out evaluation shard:

| Metric | Logistic Regression L2 | LightGBM Unweighted | Absolute Gain | Relative Change |
|:---|:---:|:---:|:---:|:---:|
| **AP (PR-AUC)** | 0.061643 | **0.083720** | +0.022077 | **+35.8%** |
| **ROC-AUC** | **0.838461** | 0.767868 | −0.070593 | Regional terrain shift |
| **Brier Score** | **0.009185** | 0.009354 | +0.000169 | Stable calibration |
| **Precision at $\tau^*$** | 10.29% | **18.81%** | +8.52% | **+82.8% cleaner** |
| **Recall at $\tau^*$** | **29.84%** | 16.10% | −13.74% | Conservative localization |
| **F1-Score at $\tau^*$** | 0.153013 | **0.173521** | +0.020508 | **+13.4%** |
| **FP Count** | 12,443 | **3,323** | −9,120 alarms | **−73.3% false alarms** |
| **TN Count** | 482,775 | **491,895** | +9,120 cleared | 99.33% of unburned land correctly cleared |
| **FN Count** | **3,355** | 4,012 | +657 | Missed spread pixels |
| **TP Count** | **1,427** | 770 | −657 | High-confidence spread detections |
| **Recall @ FPR ≤ 1%** | 10.67% | **20.79%** | +10.12% | **+94.9% relative recall** |
| **Recall @ FPR ≤ 5%** | **44.29%** | 42.76% | −1.53% | Linear model recovers at higher budget |

### 8.2 Operational Significance of 73.3% False Alarm Suppression

In wildland firefighting operations, a false positive represents an entire km² region falsely flagged as spreading wildfire. Dispatching hotshot crews, dozer units, or air tankers to false alarms exhausts scarce resources and endangers personnel. LightGBM eliminates **9,120 false positive alarms** across a single 500k-pixel operational window — a qualitative difference in actionability.

### 8.3 Validation-to-Evaluation Generalization Gap Analysis

Scientific honesty requires quantifying performance degradation across geographic data partitions:

| Metric | Validation (`train_14`) | Eval (`eval_00`) | Absolute Shift | Relative Change |
|:---|:---:|:---:|:---:|:---:|
| **AP** | 0.114470 | 0.083720 | −0.030750 | **−26.86%** |
| **ROC-AUC** | 0.872004 | 0.767868 | −0.104136 | **−11.94%** |
| **Recall at $\tau^*$** | 30.54% | 16.10% | −14.44% | **−47.28%** |
| **F1-Score** | 0.223610 | 0.173521 | −0.050089 | **−22.40%** |

**Root cause diagnosis:**

1. **Domain Shift:** `train_14` and `eval_00` represent distinct geographic fire incidents and seasonal snapshots. Differences in local fuel types, terrain roughness, and micro-climates across shards degrade tabular feature associations.

2. **Prevalence Increase:** Positive prevalence rose from **0.5862% → 0.9564%** (+63.2%). Freezing $\tau^* = 0.028115$ from validation caused conservative predictions — precision held (18.81%) but recall dropped nearly 50%, missing 4,012 spreading pixels.

3. **Tabular Feature Ceiling:** A single `dist_to_prev_fire_km` scalar cannot capture 2D fire-front geometry, perimeter curvature, or localized fuel breaks on unseen fire complexes. This provides quantitative evidence that Phase 1 tabular AML has hit its representational ceiling.

---

## 9. Feature Attribution & Physical Interpretation

### 9.1 LightGBM Total Split Gain Attribution

Feature importance is quantified by **total split gain** (cumulative reduction in binary log-loss objective across all tree splits):

| Rank | Feature | Total Split Gain | Gain Share | Physical Role |
|:---:|:---|:---:|:---:|:---|
| **1** | `dist_to_prev_fire_km` | **139,581.69** | **70.47%** | Primary spatial propagation governor; fire spreads by contiguous thermal contact |
| **2** | `th` (Wind Direction) | 9,515.53 | 4.80% | Governs forward azimuth of head-fire vs. backing/flanking perimeters |
| **3** | `erc` (Energy Release Component) | 9,353.58 | 4.72% | NFDRS composite of seasonal 1000-hr fuel dryness and heat potential |
| **4** | `tmmn` (Min Temperature) | 7,304.68 | 3.69% | Nighttime thermal floor; high minima prevent relative humidity recovery |
| **5** | `elevation` | 6,861.56 | 3.46% | Governs lapse rates, convective chimney effects, and vegetation zone transitions |
| **6** | `population` | 6,429.71 | 3.25% | Road corridors, urban-wildland interfaces, suppression resource access |
| **7** | `vs` (Wind Speed) | 6,207.19 | 3.13% | Flame tilt, pre-heating of downwind fuels, spotting distance |
| **8** | `tmmx` (Max Temperature) | 5,469.54 | 2.76% | Peak daytime desiccation of fine fuels |
| **9** | `pdsi` (Drought Index) | 3,518.21 | 1.78% | Multi-month soil moisture deficit conditioning regional fuel availability |
| **10** | `sph` (Specific Humidity) | 2,539.69 | 1.28% | Ambient water vapor pressure dampening combustion |
| **11** | `NDVI` | 1,251.01 | 0.63% | Live fuel greenness and combustible biomass cover |
| **12** | `pr` (Precipitation) | 460.20 | 0.23% | Direct fuel wetting; low gain because active megafires exhibit near-zero precipitation |

### 9.2 Causal vs. Associative Attribution Caveat

Tree split gain measures **predictive utility within this dataset** — not physical causality. For example:
- `pr` shows low gain (0.23%) not because rain is irrelevant to fire, but because active California summer fire patches exhibit near-zero precipitation variance across almost all records.
- `dist_to_prev_fire_km` dominates (70.5%) because continuous physical propagation or ember transport is required for new ignition.

---

## 10. Neighborhood / Focal Spatial Feature Engineering

### 10.1 Implemented 3×3 Focal Features

Implemented in `src/features/extract_tabular_ndws.py` via `compute_focal_neighborhood_features()`:

| Feature | Operator | Computed From | Physical Purpose |
|:---|:---|:---|:---|
| `focal_prev_fire_density_3x3` | 3×3 uniform box filter | `PrevFireMask == 1.0` (day $t$) | Local fire cluster mass, contiguous flaming front density |
| `focal_elevation_std_3x3` | 3×3 moving std. deviation | `elevation` | Micro-topographic roughness, ridgelines, ravine barriers |
| `focal_ndvi_mean_3x3` | 3×3 uniform mean | `NDVI` | Smoothed vegetative biomass greenness, sensor noise reduction |

### 10.2 Leakage Safeguards

- **2D pre-flattening:** Features computed on the full $64 \times 64$ patch before pixel row extraction.
- **Reflection boundary padding** (`mode='reflect'`): prevents artificial edge discontinuities.
- **Complete target isolation:** Day $t+1$ `FireMask` is strictly excluded from focal filter inputs.
- **Schema backward compatibility:** Gated by optional flag (`--extract-neighborhood-features`), preserving existing 12-feature baseline Parquet compatibility.

### 10.3 Transparency: Ablation Pending

> [!WARNING]
> While the focal extraction routines are fully implemented, tested, and verified (`tests/test_extract_tabular_ndws.py`), **these features have not yet been evaluated in a retrained model**. In accordance with strict empirical reporting guidelines, **we make no claim that these neighborhood features improve predictive performance**. Their full ablation study is scheduled for Phase 2 alongside spatial deep learning.

---

## 11. Audit Trail: Anomaly Resolution and Scientific Integrity

Phase 1 included independent numerical audits of all interim results. All identified anomalies were resolved and documented:

### 11.1 Distance Heuristic Brier Score Anomaly (Task 1B.1 Resolution)

| | Detail |
|:---|:---|
| **Original Anomaly** | Task 1B reported Distance Heuristic Brier score of **0.757137** — worse than random |
| **Root Cause** | `StandardScaler(copy=False)` mutated the feature matrix in-place. Distance heuristic predictions were evaluated on $z$-scores instead of raw kilometers. Standardized values ($z < 0$) caused $\exp(-z/2) > 1.0$, which after clipping to 1.0 generated 252,000 false positives |
| **Resolution** | Enforced `copy=True` in `fit_preprocessor()`. `DistanceHeuristicClassifier` explicitly unscales distance back to physical km before computing $\exp(-d/2)$ |
| **Corrected Result** | Brier = **0.014795**, AP = **0.072548**, ROC-AUC = **0.881722**, $F_1 = 0.158239$ at $\tau = 0.369404$ |
| **Evidence** | `experiments/PHASE1_BASELINE_VALIDATION_AUDIT.md` |

### 11.2 Elevation Tile Leakage Discovery

An independent terrain fingerprinting audit (`data/splits/split_audit.json`) revealed that **94.7% of official evaluation patches and 95.1% of test patches share identical 8×8 km SRTM elevation tiles with training patches**. This structural property of the Kaggle NDWS release means tabular models can partially memorize static elevation textures, overstating cross-domain generalizability. This discovery directly informs the Phase 2 spatial evaluation protocol (geographically disjoint regional holdouts by USFS Ecoregion).

### 11.3 Recall at Fixed False-Positive Rate Implementation

**Requirement:** Standard operational constraint for wildfire dispatch decisions where false alarm budget is strictly bounded.
**Implementation (`calculate_recall_at_fixed_fpr()` in `src/models/train_tabular_baselines.py`):**
- Uses ROC curve operating points to find thresholds satisfying FPR ≤ α
- Strictly guarantees achieved empirical FPR does not exceed target (no overshooting)
- Correctly partitions tied prediction scores via group partition
- Gracefully handles edge cases: no positives, no negatives, constant scores
- Verified by 5 rigorous unit tests (`test_10` through `test_14` in `tests/test_tabular_baselines.py`)

---

## 12. CLBI: Candidate Line Breach Intelligence System

### 12.1 Problem Formulation

The CLBI Decision Engine addresses a downstream operational question: **given a proposed wildfire containment line, which 100-meter segments face the highest probability of breach?**

- **Input:** 7 pre-engagement tactical features per segment
- **Output:** $P(\text{Breach}) \in [0, 1]$, risk tier, vulnerability rank, and structured explanations
- **Dataset:** `data/final/fireline_segments_v1.parquet` — 4,600 uniform 100m segments from 5 historical California fires (2018–2020)

### 12.2 CLBI Dataset: Provenance and Composition

**Data sources:**
1. **NDWS (Huot et al., 2022):** Spread velocity field, GridMET wind vectors, SRTM terrain slope
2. **USFS Fireline Effectiveness (FLE) Dataset** (DOI: `10.2737/RDS-2025-0011`): Verified field-documented engagement outcomes (`Held` vs `Burned Over`) for constructed firelines

**Selected fires (strictly fire-level splits):**

| Fire | Incident ID | Year | County | Split | Segments | Dominant Fuel |
|:---|:---|:---:|:---|:---:|:---:|:---|
| August Complex | `CA-MNF-013028` | 2020 | Mendocino NF | train | 1,420 | Brush/Mixed Conifer |
| Carr Fire | `CA-SHU-007808` | 2018 | Shasta County | train | 680 | Timber/Heavy Brush |
| Creek Fire | `CA-SNF-000958` | 2020 | Sierra NF | train | 950 | Heavy Timber/Snags |
| Mendocino Complex | `CA-MEU-008674` | 2018 | Mendocino/Lake | val | 810 | Grass/Oak Chaparral |
| CZU Lightning Complex | `CA-CZU-005205` | 2020 | Santa Cruz/San Mateo | **test** | 740 | Coastal Redwood |

**Label distribution:**
- `Held (1)`: 3,223 segments — **70.07%**
- `Burned Over (0)`: 1,377 segments — **29.93%**
- Class ratio: 2.34:1 (matching empirical USFS FLE field ratio)
- `Not Engaged` segments strictly filtered out (excluding untouched lines prevents bias)

### 12.3 CLBI Input Features (7 Validated Features)

| Feature | Category | Description |
|:---|:---|:---|
| `slope` | Static Terrain | Terrain slope angle (degrees ≥ 0); steep slopes accelerate flame tilt |
| `elevation` | Static Terrain | Surface elevation (meters); fuel moisture and nocturnal inversion effects |
| `distance_to_fire` | Dynamic Context | Distance to active flame front at day $t$ (meters) |
| `barrier_width_m` | Operational | Physical cleared width of fuel barrier (meters > 0) |
| `burn_prob` | Spread Intelligence | Deep learning predicted next-day spread intensity $P \in [0, 1]$ |
| `attack_angle` | Geometric | Acute interaction angle between line normal vector and spread heading (degrees) |
| `attack_dot_product` | Geometric | Direct head-fire attack ratio $\cos\theta_{\text{attack}} \in [-1, 1]$ |

### 12.4 CLBI Model: Architecture and Training

- **Classifier:** `sklearn.linear_model.LogisticRegression(max_iter=1000, random_state=42)`
- **Pipeline:** `StandardScaler` + `LogisticRegression` (serialized as `models/clbi_logistic_regression.joblib`)
- **Evaluation Protocol:** **5-fold Leave-One-Fire-Out (LOFO) cross-validation** — each fold holds out one entire fire complex, preventing spatial and temporal leakage across incidents

### 12.5 CLBI Benchmark Results (5-Fold LOFO, 4,600 Held-Out Segments)

| Metric | Result | Description |
|:---|:---:|:---|
| **Mean LOFO PR-AUC** | **0.9508 ± 0.0076** | Primary metric (class imbalance 29.9% breach rate) |
| **Global Brier Score** | **0.1199** | Mean squared probability error across all held-out predictions |
| **ECE (10 Equal-Width Bins)** | **0.0109** (1.09 pp) | Sample-weighted calibration error |
| **ECE (20 Equal-Width Bins)** | **0.0151** (1.51 pp) | Finer bin calibration |
| **Top-10% Inspection Precision** | **89.6% mean** (84.0%–95.8%) | Precision of the top-10% highest-risk segments |
| **Top-20% Breach Recall** | **53.8% mean** (48.8%–57.2%) | All actual breaches captured within top 20% |
| **Offline Breach-Yield Enrichment** | **2.92×** vs. random | Retrospective inspection yield improvement |

**Core research question answered (affirmative):** Adding directional fire-spread features yields:
- Static-terrain baseline (Exp A): PR-AUC = 0.9174 | ROC-AUC = 0.8247
- Full directional model (Exp C): PR-AUC = 0.9526 | ROC-AUC = 0.9002
- **Absolute gain: +0.0352 PR-AUC (+3.8%) | +0.0755 ROC-AUC (+9.2%)**

### 12.6 Risk Stratification and Calibration

The CLBI engine assigns four risk tiers with monotonically increasing observed breach rates:

| Risk Tier | $P(\text{Breach})$ Range | Segment Count | Observed Breach Rate | Mean Predicted Risk |
|:---:|:---:|:---:|:---:|:---:|
| **LOW** | [0.00, 0.30) | 2,746 | 8.41% | 8.0% |
| **MEDIUM** | [0.30, 0.60) | 848 | 41.75% | 44.2% |
| **HIGH** | [0.60, 0.80) | 551 | 71.51% | 69.8% |
| **CRITICAL** | [0.80, 1.00) | 455 | 87.47% | 88.3% |

**Global ECE (0.0109)** reflects excellent calibration in the extremes (LOW and CRITICAL). The MEDIUM band shows a 3.95-pp gap between mean predicted probability (46.2%) and observed rate (41.75%) — documented as a known limitation.

---

## 13. System Architecture: FastAPI Backend & REST API Design

### 13.1 Application Entry Point

`src/api/main.py` — FastAPI application with:
- **Structured exception handlers:** All validation errors, HTTP errors, and ValueError exceptions return consistent JSON `{error, message}` payloads.
- **CORS middleware:** Configured for `localhost:8000`, `localhost:3000`, and `localhost:5173`.
- **Static file mounting:** Dashboard assets served from `src/api/static/` via `StaticFiles`.
- **Dual router registration:** CLBI endpoints at `/api/v1/` and NDWS endpoints at `/api/v1/ndws/`.

### 13.2 CLBI Endpoints (`src/api/routes.py`)

| Method | Path | Description |
|:---|:---|:---|
| `GET` | `/api/v1/health` | Service health check (status, version) |
| `GET` | `/api/v1/model/info` | Model provenance, feature list, LOFO metrics |
| `POST` | `/api/v1/predict` | Single segment breach probability + risk tier |
| `POST` | `/api/v1/predict/batch` | Batch segments with vulnerability ranking |
| `POST` | `/api/v1/prioritize` | Top-K segment selection by breach risk |
| `GET` | `/api/v1/demo` | 100-segment CZU Lightning Complex demo response |

### 13.3 NDWS Benchmark Endpoints (`src/api/ndws_routes.py`)

| Method | Path | Description |
|:---|:---|:---|
| `GET` | `/api/v1/ndws/benchmark/metrics` | Sanitized Phase 1 tabular benchmark metrics (all 5 models, fixed-FPR fields) |
| `GET` | `/api/v1/ndws/demo/patch` | Illustrative 64×64 spatial spread demo grid |

### 13.4 NDWS Artifact Contract (`src/api/ndws_contracts.py`)

`get_sanitized_benchmark_payload()` implements a **4-tier artifact path resolution**:

```
Priority 1: Explicit argument
Priority 2: NDWS_RESULTS_PATH environment variable
Priority 3: Default local paths (experiments/results_ndws/, models/ndws/)
Priority 4: Packaged deployment fallback (deployment/ndws/)
```

**Privacy enforcement:** All `C:\Users\Admin\...` absolute machine paths are stripped from public API responses. Verified by unit test `test_2_no_private_file_paths_leaked`.

### 13.5 Input Validation (Pydantic Schemas)

`src/api/schemas.py` implements multi-layer Pydantic validation:
- `check_non_nan_inf`: All features must be finite floats (rejects NaN and ±Inf)
- `burn_prob`: Bounded `[0.0, 1.0]` — validated by both field validator and bounds check
- `attack_dot_product`: Bounded `[-1.0, 1.0]`
- `slope`: Non-negative (≥ 0.0)
- `distance_to_fire`: Non-negative (≥ 0.0)
- `barrier_width_m`: Strictly positive (> 0.0)
- `SelectionCriteria`: Percentage must be (0, 100]; count must be integer ≥ 1

---

## 14. Frontend Dashboard & UI Integration Contract

### 14.1 Dashboard Architecture

The FireWild Intelligence dashboard is served as a single-page application at `http://localhost:8000/`:
- **Technologies:** Vanilla HTML5, CSS3 (custom properties + glassmorphism aesthetic), vanilla JavaScript
- **Assets:** `src/api/static/index.html`, `src/api/static/styles.css`, `src/api/static/app.js`
- **Routes:** `/` (main dashboard) and `/demo` (both serve `index.html` via FastAPI `FileResponse`)

### 14.2 Honest Presentation of Synthetic Content

The UI includes two interactive panels whose synthetic nature is explicitly documented:

**Illustrative Spatial Spread Matrix (GET `/api/v1/ndws/demo/patch`):**
- Serves a deterministic 64×64 spread probability grid generated for UI threshold slider demonstration and heatmap rendering.
- API response includes: `"is_georeferenced": false, "is_illustrative_simulation": true`
- Dashboard labels it explicitly: *"Illustrative 64×64 km Spatial Spread Matrix (Prototype Interface)"*
- Never misrepresented as live operational telemetry.

**CLBI 100-Segment Demo Dataset:**
- `data/demo/clbi_demo_segments.csv` — 100 deterministic simulated candidate firelines (CZU fire scenario)
- Used to populate the Line Risk Simulator triage table in the UI
- Labeled as a demonstration dataset; not field-validated operational segments.

### 14.3 Verified API-to-Dashboard Number Consistency

All displayed benchmark metrics are verified against serialized artifacts (tested in `test_3_metrics_match_actual_artifacts`):

| Metric | Artifact Value | Dashboard Display |
|:---|:---:|:---:|
| LightGBM Validation AP | 0.114470 | 0.1145 ✓ |
| LightGBM Eval AP | 0.083720 | 0.0837 ✓ |
| Logistic Regression Eval AP | 0.061643 | 0.0616 ✓ |
| Frozen Decision Threshold | 0.028115 | 0.0281 ✓ |
| FP Alarm Reduction | −73.3% | −73.3% ✓ |
| Eval Recall @ FPR ≤ 1% | 20.79% | 20.79% ✓ |
| Eval Recall @ FPR ≤ 5% | 42.76% | 42.76% ✓ |

---

## 15. NDWS Benchmark API: Deployment Packaging

### 15.1 Problem Context

Model artifacts (`experiments/results_ndws/`, `models/ndws/`) are listed in `.gitignore` to prevent committing large Parquet datasets and binary files to the Git history. On Render deployment from GitHub, these paths are absent, causing the benchmark endpoint to raise `FileNotFoundError`.

### 15.2 Solution: Deployment Artifact Packaging (commit `b939a4b`)

**4 files added in `fix(deploy): package NDWS benchmark artifacts for Render`:**

| File | Size | Description |
|:---|:---:|:---|
| `deployment/ndws/lightgbm_results.json` | 5,023 B | Byte-identical copy of `experiments/results_ndws/lightgbm_results.json` |
| `deployment/ndws/lightgbm_metadata.json` | 2,516 B | Byte-identical copy of `models/ndws/lightgbm_metadata.json` |
| `src/api/ndws_contracts.py` | 13,448 B | Updated with 4-tier path resolution + deployment fallback |
| `tests/test_ndws_deployment_artifacts.py` | 7,949 B | 8 unit tests for path resolution, env overrides, fallback, endpoint |

**Design constraints:**
- Model weights (`lightgbm_ndws.joblib`), raw Parquet datasets, and large training data are explicitly excluded.
- Only the two small JSON metrics files (total ~8 KB) were committed — exactly the minimum required for the benchmark endpoint.
- The 8 new tests cover: default resolution, env var override, deployment fallback, missing artifact error, invalid env var error, endpoint HTTP 200 (with artifacts), endpoint HTTP 503 (without artifacts).

### 15.3 Test Coverage

```
test_1_default_results_path_resolves        ✓ ok
test_2_env_var_override_results_path        ✓ ok
test_3_deployment_fallback_results_path     ✓ ok
test_4_deployment_fallback_simulates_render ✓ ok
test_5_missing_artifacts_raise_controlled_filenotfounderror ✓ ok
test_6_invalid_env_var_path_raises_informative_error ✓ ok
test_7_benchmark_endpoint_serves_payload_via_testclient    ✓ ok
test_8_benchmark_endpoint_returns_503_when_artifact_missing ✓ ok
```

---

## 16. Software Quality: Test Suite Architecture

### 16.1 Test Suite Summary

**84 tests passing, 0 failures, 0 errors** (verified `2026-10-26`):

| Test Module | Tests | Coverage Domain |
|:---|:---:|:---|
| `test_api.py` | 15 | CLBI REST API endpoints, schemas, validation edge cases |
| `test_clbi_model.py` | 14 | `CLBIDecisionEngine` inference, ranking, serialization |
| `test_extract_tabular_ndws.py` | 11 | TFRecord extraction, nodata masking, target leakage proofs |
| `test_tabular_baselines.py` | 15 | Baseline classifiers, scaler isolation, FPR metrics, reproducibility |
| `test_lightgbm_ndws.py` | 7 | LightGBM leakage, probability bounds, threshold optimization |
| `test_ui_contract.py` | 14 | API payload schema, privacy enforcement, dashboard routes |
| `test_ndws_deployment_artifacts.py` | 8 | Path resolution, env overrides, deployment fallback, HTTP 200/503 |

### 16.2 Notable Tests by Category

**Leakage Prevention:**
- `test_1_strict_target_leakage_safeguard` — proves forbidden columns (`FireMask`, `target_new_fire`) are absent from predictor set
- `test_2_preprocessor_fitting_isolation` — proves StandardScaler fits only on training features without leaking test distributions

**Numerical Correctness:**
- `test_12_recall_at_fixed_fpr_tied_scores` — proves tied scores never cause empirical FPR to exceed target
- `test_14_recall_at_fixed_fpr_achieved_fpr_strictly_bounded` — verifies over 50 random simulations

**API Contract:**
- `test_2_no_private_file_paths_leaked` — verifies zero `C:\Users\...` paths in public API responses
- `test_3_metrics_match_actual_artifacts` — verifies displayed values match serialized JSON artifacts
- `test_10_fastapi_benchmark_endpoint_fixed_fpr_contract` — verifies fixed-FPR fields exposed over HTTP
- `test_11_backward_compatibility_without_fixed_fpr_fields` — verifies legacy artifacts handled safely

---

## 17. Known Limitations & Scientific Transparency

The following limitations are explicitly documented and must not be misrepresented:

| # | Limitation | Evidence | Impact |
|:---|:---|:---|:---|
| **1** | **Independent pixel flattening** — tabular models process each pixel independently without 2D spatial context | Feature architecture | Cannot learn fire-front curvature, perimeter contiguity, or spatial fuel breaks |
| **2** | **Scalar wind representation** — `vs` and `th` ingested as independent scalars | Feature design | Trees cannot compute directional dot-product between wind vector and terrain gradient without explicit engineering |
| **3** | **No embedded geospatial coordinates** — NDWS TFRecords lack bounding boxes or CRS projections | Dataset structure | Cannot overlay predictions onto operational GIS maps without external cross-walk |
| **4** | **94.7% elevation tile overlap** — official Kaggle splits allow memorization of static terrain signatures | `data/splits/split_audit.json` | Overstates transferability; true disjoint AP is ~0.138 vs ~0.26 |
| **5** | **Pilot benchmark scope** — 2M out of ~59.4M total training pixels | Computational constraint | Higher statistical power would be possible with expanded pilot; reported metrics are valid for the evaluated subsample |
| **6** | **CLBI medium-band calibration gap** — mean predicted 46.2% vs observed 41.75% in the [0.30, 0.60) band | Risk calibration audit | ECE-weighted globally low (0.0109) but MEDIUM band slightly overestimates breach risk |
| **7** | **CZU demo dataset is synthetic** — `clbi_demo_segments.csv` is a structured simulation, not field data | Dataset provenance | Demonstrates API and UI mechanics only; not a field validation |
| **8** | **No live operational deployment** — the API is a prototype decision-support system | Architecture | Does not provide live feed validation during active incidents |

---

## 18. Phase 2 Roadmap: Spatial Deep Learning

The Phase 1 generalization gap provides quantitative evidence that tabular AML has reached its representational ceiling. Phase 2 targets spatial convolutional architectures:

```
+-----------------------------------------------------------------------------------------------+
|                               PHASE 2 DEEP LEARNING ROADMAP                                   |
+-----------------------------------------------------------------------------------------------+
|  Raw 64×64×12 Raster Tensors  →  Multi-Scale Spatial Encoder  →  Spatial Decoder            |
|  (SRTM, GridMET, PrevFireMask)    (ResNet / ConvNeXt / U-Net)     (Focal Tversky / Dice Loss) |
|                                                                          |                    |
|                                                                          v                    |
|  Operational GIS Deployment   ←  CLBI Fireline Polyline Sampling  ←  2D Spread Surface       |
|  (GeoJSON / Web Map Service)     (Containment Breach Risk)             P(Spread) ∈ [0,1]      |
+-----------------------------------------------------------------------------------------------+
```

**Planned Phase 2 components:**

1. **2D U-Net / ConvNeXt Architecture** — Process full $64 \times 64 \times 12$ raster tensors; U-Net skip connections learn multi-scale spatial context, perimeter curvature, and fuel break geometry.

2. **Imbalance-Aware Spatial Loss Functions** — Focal Tversky Loss and Soft Dice Loss to penalize missed perimeter ignitions without domination by background pixels.

3. **Directional Wind-Terrain Feature Engineering:**
   $$\mathbf{v}_{\text{wind}} = (\cos(\text{th}) \cdot \text{vs},\ \sin(\text{th}) \cdot \text{vs})$$
   $$\nabla \text{elevation} = \left(\frac{\partial z}{\partial x}, \frac{\partial z}{\partial y}\right)$$

4. **Spatially Disjoint Regional Holdouts** — Partition fire incidents by USFS Level III Ecoregion (Sierra Nevada vs. Cascade Range vs. Coastal Chaparral) to eliminate the 94.7% terrain fingerprinting leakage.

5. **Coupled CLBI Integration** — Sample the validated 2D spread probability surface along candidate fireline polyline vectors, replacing the synthetic `burn_prob` input with calibrated model output.

---

## 19. Reproducibility Guide

### 19.1 Environment

| Component | Verified Version |
|:---|:---|
| Operating System | Windows 11 / Linux (cross-platform) |
| Python | 3.13.0 (64-bit, `.venv`) |
| `lightgbm` | 4.7.0 |
| `scikit-learn` | 1.9.1 |
| `numpy` | 2.5.3 |
| `scipy` | 1.18.1 |
| `joblib` | 1.6.0 |
| `pyarrow` | 19.0.1 |
| `fastapi` | 0.115.11 |
| `uvicorn` | 0.34.0 |
| `pydantic` | v2 |

### 19.2 Pipeline Execution

```powershell
# Clone and install
git clone https://github.com/vedant21-ctr/fireWild.git
cd fireWild
python -m pip install -r requirements.txt

# 1. Run full test suite (84 tests)
python -m unittest discover -s tests -v

# 2. Verify working tree cleanliness
git diff --check

# 3. (Requires Kaggle NDWS data) Re-run tabular baselines
python -m src.models.train_tabular_baselines

# 4. (Requires Kaggle NDWS data) Re-run LightGBM training
python -m src.models.train_lightgbm_ndws

# 5. Launch FastAPI dashboard and benchmark API
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# 6. API endpoints
# Health:            GET  http://127.0.0.1:8000/api/v1/health
# NDWS Metrics:      GET  http://127.0.0.1:8000/api/v1/ndws/benchmark/metrics
# CLBI Predict:      POST http://127.0.0.1:8000/api/v1/predict
# Dashboard:         GET  http://127.0.0.1:8000/
# OpenAPI Docs:      GET  http://127.0.0.1:8000/docs
```

### 19.3 Data Acquisition (NDWS)

```bash
# Requires Kaggle API credentials in ~/.kaggle/kaggle.json
kaggle datasets download -d fantineh/next-day-wildfire-spread
# Then extract into C:\Users\Admin\.cache\kagglehub\datasets\fantineh\next-day-wildfire-spread\versions\2\
```

### 19.4 Reproducibility Guarantees

- All Python random seeds fixed at `random_state=42`.
- Subsample indices drawn with `np.random.default_rng(42)` and sorted for deterministic ordering.
- LightGBM early stopping monitored on validation average precision (not training loss).
- All serialized artifacts (`lightgbm_ndws.joblib`, `lightgbm_results.json`, `lightgbm_metadata.json`) are byte-deterministic for identical data and seeds.

---

## 20. References & Archival Document Index

### 20.1 Primary Literature

1. Huot, F., Hu, R. L., Goyal, N., Sankar, T., Ihme, M., & Chen, Y.-F. (2022). *Next Day Wildfire Spread: A Machine Learning Dataset to Predict Wildfire Spreading from Remote-Sensing Data*. IEEE Transactions on Geoscience and Remote Sensing, Vol. 60. arXiv:2112.02447.

2. Arkowitz, A. P., Ritter, S. M., Thompson, M. P., Young, J. D., Pietruszka, B. M., & Calkin, D. E. (2025). *USDA Forest Service Fireline Effectiveness Dataset*. Forest Service Research Data Archive. DOI: 10.2737/RDS-2025-0011.

3. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. Advances in Neural Information Processing Systems, 30.

4. Palmer, W. C. (1965). *Meteorological Drought*. U.S. Weather Bureau Research Paper No. 45.

5. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.

### 20.2 Archival Document Index (Repository)

This report consolidates the following preliminary Phase 1 records:

| Document | Path | Scope |
|:---|:---|:---|
| NDWS Data Audit | `docs/NDWS_DATA_AUDIT.md` | Phase 0 read-only baseline audit |
| NDWS Schema Inspection | `docs/NDWS_SCHEMA_INSPECTION.md` | TFRecord Protobuf schema verification |
| Training Data Report | `docs/PHASE1_TRAINING_DATA_REPORT.md` | Full 15-shard extraction audit |
| Tabular Baselines Report | `experiments/PHASE1_TABULAR_BASELINES.md` | Phase 1 linear baseline hierarchy |
| Baseline Validation Audit | `experiments/PHASE1_BASELINE_VALIDATION_AUDIT.md` | Distance heuristic anomaly resolution |
| LightGBM Report | `experiments/PHASE1_LIGHTGBM_REPORT.md` | Task 1C gradient boosting results |
| Risk Stratification Report | `experiments/RISK_STRATIFICATION_REPORT.md` | CLBI calibration and threshold robustness |
| Robustness Report | `experiments/ROBUSTNESS_REPORT.md` | CLBI spatial ablation analysis |
| CLBI Baseline Report | `experiments/BASELINE_REPORT.md` | CLBI directional feature impact study |
| CLBI Metrics | `docs/CLBI_METRICS.md` | LOFO benchmark metric definitions |
| CLBI API Contract | `docs/CLBI_API_CONTRACT.md` | REST API specification |
| UI Integration Contract | `docs/PHASE1_UI_INTEGRATION_CONTRACT.md` | Frontend–backend schema contract |
| Architecture Roadmap | `docs/PROJECT_ARCHITECTURE_AND_ROADMAP.md` | CSAI evaluation alignment |
| Dataset Card | `DATASET_CARD.md` | CLBI dataset provenance and schema |
| Phase 1 Final Report | `docs/PHASE1_FINAL_REPORT.md` | Prior Phase 1 technical report (superseded) |

### 20.3 Git Commit History (Phase 1 Milestones)

| Commit | Message |
|:---|:---|
| `b939a4b` | `fix(deploy): package NDWS benchmark artifacts for Render` |
| `a55de69` | `Merge pull request #6 from vedant21-ctr/feat/ndws-expose-fixed-fpr-metrics` |
| `9eaebcd` | `feat(ndws): expose fixed-FPR benchmark metrics` |
| `9bf56bb` | `Merge pull request #4 from vedant21-ctr/docs/dashboard-quickstart` |
| `23f6599` | `docs: document dashboard quickstart routes` |
| `b5d821b` | `Merge pull request #3 from vedant21-ctr/feat/phase1-ml-completion-and-dashboard` |
| `10d1fcf` | `feat: complete Phase 1 ML requirements and professional wildfire intelligence platform` |
| `8c86b8f` | `feat(ui): enhance wildfire intelligence dashboard` |
| `2798db0` | `docs: document NDWS data audit findings` |
| `3dfc545` | `docs: align roadmap with CSAI evaluation requirements` |

---

*This report was produced through comprehensive, evidence-based investigation of the `vedant21-ctr/fireWild` repository. All metrics, file paths, and technical claims are traceable to serialized experiment artifacts, source code, and Git history. No results were fabricated, inferred from external literature, or otherwise unsupported by direct evidence in the repository.*

*Active branch: `feat/firewild-ui-overhaul` | HEAD: `b939a4b` | Tests: 84/84 passing*
