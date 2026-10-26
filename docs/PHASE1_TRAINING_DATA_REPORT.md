# Phase 1 — Task 1A: NDWS Training Data Extraction and Validation Report

**Project:** Next Day Wildfire Spread Prediction via Tabular AML and Spatial Deep Learning (Project 17)  
**Date:** October 2026  
**Repository:** `C:\Users\Admin\Documents\WILDFIRE`  
**Execution Context:** Phase 1 (Tabular AML Baselines), Task 1A  

---

## 1. Executive Summary

In this task, the NDWS tabular feature extraction pipeline was validated, tested, and executed across all 15 official training shards (`train_00` to `train_14`) of the Google Research Next Day Wildfire Spread dataset. 

Key results:
- **15 official training shards** processed through streaming, memory-efficient batches.
- **14,979 total records** inspected; **59,369,679 valid pixel rows** extracted.
- **482,818 newly active fire pixels** (`target_new_fire == 1`), establishing an empirical training prevalence of **0.813%** (extreme class imbalance).
- **Zero missing values (nulls or NaNs)** across all 22 output columns in the extracted dataset.
- Total training Parquet footprint: **2.73 GB** (2,795.21 MB) with Snappy compression, safely residing under the ignored directory `data/processed/ndws/`.
- **Zero data leakage**: Target values (`FireMask`) and target derivatives were proven via unit tests to be completely excluded from predictor channels and spatial context calculations.
- Full test suite: **40 tests passing** (15 CLBI API + 14 CLBI Model + 11 NDWS Extraction).

---

## 2. Official Dataset Partition Layout

The NDWS dataset provided by Google Research (Huot et al., 2022) is structured into 19 authoritative TFRecord shards. Their partition assignment is explicitly embedded in the shard filenames and preserved in the extracted tabular records:

| Partition | Shards | Record Count | Raw TFRecord Size | Extracted Status |
|---|---|---|---|---|
| **Training (`train`)** | 15 shards (`train_00` – `train_14`) | 14,979 records | 3,047.18 MB (~2.98 GB) | **Extracted (59,369,679 rows, 2.73 GB Parquet)** |
| **Validation (`eval`)** | 2 shards (`eval_00`, `eval_01`) | ~1,877 records | 381.84 MB | `eval_00` extracted as test sample (3,922,548 rows, 164.02 MB); `eval_01` unextracted |
| **Testing (`test`)** | 2 shards (`test_00`, `test_01`) | ~1,689 records | 343.59 MB | Isolated, unextracted, held out for final evaluation |
| **Total** | **19 shards** | **~18,545 records** | **~3,772.61 MB (~3.68 GB)** | — |

> [!IMPORTANT]
> The evaluation sample (`eval_00_tabular.parquet`) extracted during Task 2C has been strictly quarantined. It was NOT used to fit scalers, select features, or compute training distributions.

---

## 3. Extraction Commands and Environment

### Command Executed for Training Partition
```bash
python -m src.features.extract_tabular_ndws `
  --input-dir "C:\Users\Admin\.cache\kagglehub\datasets\fantineh\next-day-wildfire-spread\versions\2" `
  --pattern "*train*.tfrecord" `
  --output-dir "data/processed/ndws" `
  --batch-size 50
```

### Extraction Performance
- **Elapsed Time:** 189.49 seconds (~3.16 minutes).
- **Processing Rate:** ~79 records / second (~313,000 pixels / second).
- **Peak RAM Usage:** Kept strictly bounded by flushing PyArrow record batches every 50 records without loading full shards into memory.

---

## 4. Per-Shard Extraction Statistics

| Shard Name | Records Inspected | Valid Pixels Extracted | Newly Active Fire ($Y=1$) | Class Prevalence ($Y=1$) | Output Parquet Size | Extraction Time |
|---|---|---|---|---|---|---|
| `next_day_wildfire_spread_train_00` | 1,000 | 3,939,429 | 35,181 | 0.893% | 187.48 MB | 14.45s |
| `next_day_wildfire_spread_train_01` | 1,000 | 3,959,557 | 35,597 | 0.899% | 188.94 MB | 13.36s |
| `next_day_wildfire_spread_train_02` | 1,000 | 3,935,608 | 29,716 | 0.755% | 189.91 MB | 14.86s |
| `next_day_wildfire_spread_train_03` | 1,000 | 3,918,234 | 33,647 | 0.859% | 187.99 MB | 12.30s |
| `next_day_wildfire_spread_train_04` | 1,000 | 3,942,427 | 32,297 | 0.819% | 187.24 MB | 11.30s |
| `next_day_wildfire_spread_train_05` | 1,000 | 3,959,027 | 25,362 | 0.641% | 191.16 MB | 9.84s |
| `next_day_wildfire_spread_train_06` | 1,000 | 3,967,108 | 27,849 | 0.702% | 190.80 MB | 10.55s |
| `next_day_wildfire_spread_train_07` | 1,000 | 3,993,102 | 24,535 | 0.614% | 192.27 MB | 11.45s |
| `next_day_wildfire_spread_train_08` | 1,000 | 3,968,871 | 32,170 | 0.811% | 190.22 MB | 25.55s |
| `next_day_wildfire_spread_train_09` | 1,000 | 3,982,983 | 41,361 | 1.038% | 189.75 MB | 17.13s |
| `next_day_wildfire_spread_train_10` | 1,000 | 3,981,109 | 40,669 | 1.022% | 184.00 MB | 9.84s |
| `next_day_wildfire_spread_train_11` | 1,000 | 3,963,818 | 46,918 | 1.184% | 178.33 MB | 8.97s |
| `next_day_wildfire_spread_train_12` | 1,000 | 3,987,327 | 30,810 | 0.773% | 182.45 MB | 9.33s |
| `next_day_wildfire_spread_train_13` | 1,000 | 3,989,753 | 23,973 | 0.601% | 180.88 MB | 9.63s |
| `next_day_wildfire_spread_train_14` | 979 | 3,881,326 | 22,733 | 0.586% | 173.79 MB | 10.75s |
| **Total / Overall** | **14,979** | **59,369,679** | **482,818** | **0.813%** | **2,795.21 MB (2.73 GB)** | **189.49s (3.16 min)** |

---

## 5. Output Parquet Schema and Column Definitions

Each row represents exactly one valid pixel from a $64 \times 64$ patch. The dataset contains 22 strictly typed columns:

```
Schema:
├── sample_id: string               # Provenance: e.g. 'train_00_rec_00042'
├── record_idx: int32               # Index of patch within shard (0..999)
├── row: int32                      # Grid pixel vertical index (0..63)
├── col: int32                      # Grid pixel horizontal index (0..63)
├── shard: string                   # Source shard name
├── partition: string               # Dataset partition ('train')
├── elevation: float32              # Digital Elevation Model (m)
├── th: float32                     # Wind direction (deg clockwise from N)
├── vs: float32                     # Wind speed (m/s)
├── tmmn: float32                   # Min temperature (Kelvin)
├── tmmx: float32                   # Max temperature (Kelvin)
├── sph: float32                    # Specific humidity (kg/kg)
├── pr: float32                     # Precipitation (mm)
├── pdsi: float32                   # Palmer Drought Severity Index
├── NDVI: float32                   # Normalized Difference Veg Index (*10,000)
├── population: float32             # Population density (people/km²)
├── erc: float32                    # Energy Release Component
├── dist_to_prev_fire_km: float32   # Intra-patch Euclidean distance to day-t fire (km)
├── has_prev_fire_in_patch: int32   # Binary indicator (1 if patch has day-t fire, 0 otherwise)
├── prev_fire_mask: int32           # Fire state at day t (0=unburned, 1=active fire)
├── fire_mask_t1: int32             # Fire state at day t+1 (0=unburned, 1=active fire)
└── target_new_fire: int32          # SUPERVISION TARGET: 1 iff (prev==0 and fire==1)
```

---

## 6. Scientific and Methodological Analysis

### A. Target Definition: Newly Active Fire Spread
The supervision target is explicitly defined as:
$$\text{target\_new\_fire} = 1 \iff (\text{PrevFireMask} == 0 \land \text{FireMask} == 1)$$

- **Physical interpretation:** Measures perimeter expansion, new ignitions, or spotting from day $t$ to day $t+1$.
- **Distinction from persistence:** Cells burning on day $t$ that remain burning on day $t+1$ ($\text{PrevFireMask}=1, \text{FireMask}=1$) receive label $0$. This separates forward fire propagation from interior fuel consumption/smoldering.
- **Validity filter:** Pixels with value $-1.0$ in either mask (due to heavy cloud cover, water bodies, or instrument nodata) are discarded before entering the dataset.

### B. Spatial Distance Feature: Physics and Limitations
The feature `dist_to_prev_fire_km` is computed via Euclidean Distance Transform on $\text{PrevFireMask} == 1.0$:

1. **Physical Resolution:** The NDWS grid cells are nominally $1\text{ km} \times 1\text{ km}$. The SciPy Euclidean Distance Transform measures grid steps $\sqrt{\Delta r^2 + \Delta c^2}$. At 1 km/cell, 1 grid step equals approximately 1.0 km.
2. **Sentinel Assignment:** If a patch contains no active fire on day $t$ (`has_prev_fire_in_patch == 0`), distance is assigned a sentinel value of $100.0\text{ km}$. The maximum possible diagonal distance inside a $64 \times 64$ patch is $\sqrt{63^2 + 63^2} \approx 89.1\text{ km}$, guaranteeing the sentinel is outside the patch boundary.
3. **Leakage Safeguard:** The distance transform takes exclusively `PrevFireMask`. In `test_8_spatial_distance_independent_of_target_firemask`, varying `FireMask` from empty to completely active was proven to produce identically 0 change in `dist_to_prev_fire_km`.
4. **Boundary Truncation Limitation:** The distance is strictly **intra-patch**. If a major wildfire front was burning immediately adjacent to the patch boundary (e.g., column -1), the intra-patch transform cannot observe it and will report a distance to the nearest intra-patch fire (or 100 km).

### C. Partition Integrity and Spatial Autocorrelation
- **Official Splits:** Google Research partitioned the shards into train (00–14), eval (00–01), and test (00–01).
- **Absence of Event IDs:** The raw TFRecord records do not contain incident IDs, geographic coordinates (lat/lon), or timestamps. Therefore, we **cannot claim guaranteed spatial or temporal disjointness across records within a partition**. 
- **Rule against pixel-level random splitting:** Pixels from the same patch share identical regional weather, fuel moisture, and terrain. Splitting pixels randomly into train and test sets would cause severe spatial data leakage. The official shard-level separation must be maintained.

---

## 7. Recommended Protocol for Phase 1 Baseline Modeling

Before running machine learning models, the following protocol must be observed:

### 1. Training-Only Fitted Preprocessing
- Preprocessing pipelines (e.g. `StandardScaler` or `RobustScaler`) must be fitted **exclusively on the training partition** (`train_*_tabular.parquet`).
- The fitted transformers must be serialized and applied to `eval` and `test` shards at inference time without re-fitting.
- Features with known scale constants (e.g. NDVI scaled by 10,000, temperatures in Kelvin) should be brought to natural units or standardized.

### 2. Handling Extreme Class Imbalance (0.813%)
With fewer than 1 positive pixel per 120 negative pixels:
- **Never evaluate on Raw Accuracy or ROC-AUC alone.** ROC-AUC is misleadingly optimistic under severe class imbalance because the false positive rate denominator is dominated by the huge negative majority.
- **Primary Metrics:** Precision-Recall AUC (PR-AUC / Average Precision) and $F_1$-score / $F_\beta$-score.
- **Model Training Strategies:**
  - Class-weighted loss functions (`class_weight='balanced'` in Logistic Regression).
  - Subsampling majority negatives during initial training exploration (e.g. 10:1 negative-to-positive ratio for initial linear models), while validating on full, unadulterated evaluation distributions.

### 3. Model Progression
1. **Dummy Baseline:**
   - Uniform random prediction based on class prior ($P=0.00813$).
   - Constant majority class predictor ($Y=0$).
   - Distance-only heuristic (predict fire if `dist_to_prev_fire_km <= 1.5 km`).
2. **Regularized Logistic Regression:**
   - Ridge ($L_2$) and Lasso ($L_1$) logistic regression on standardized 11 environmental features + spatial distance feature.
   - Evaluated using Average Precision (AP) and Brier Score for probability calibration.
3. **Decision Threshold Tuning:**
   - The default threshold of 0.5 is inappropriate for a 0.81% prior.
   - Optimal decision thresholds must be selected on training/validation folds to maximize $F_1$ or recall at fixed precision before applying to evaluation shards.

---

## 8. Unit Test Suite Verification

Test command executed:
```bash
python -m unittest discover -s tests -v
```

**Results:**
- **Ran 40 tests in 5.65s — All PASSED (OK).**
- 15 tests: CLBI Backend API contract and error handling.
- 14 tests: CLBI Machine Learning engine and prioritization logic.
- 11 tests: NDWS Tabular Feature Extractor:
  - `test_1_validate_schema_passes_on_valid_record`: Validates expected 13 keys and 4096-float vectors.
  - `test_2_validate_schema_raises_on_missing_key`: Asserts failure on missing channels.
  - `test_3_validate_schema_raises_on_invalid_array_length`: Asserts failure on truncated arrays.
  - `test_4_target_new_fire_logic`: Verifies $Y=1$ for unburned $\to$ burned; $Y=0$ for burned $\to$ burned core.
  - `test_5_exclusion_of_nodata_pixels`: Verifies $-1.0$ pixels are completely filtered out.
  - `test_6_target_derived_values_not_in_environmental_features`: Confirms no target leakage in features.
  - `test_7_stable_provenance_identifiers`: Verifies `sample_id`, `record_idx`, `row`, `col`.
  - `test_8_spatial_distance_independent_of_target_firemask`: Proves distance feature is independent of target.
  - `test_9_spatial_distance_sentinel_when_no_active_fire`: Confirms 100.0 km sentinel for empty patches.
  - `test_10_infer_partition_from_name`: Validates filename-based partition routing (`train`, `eval`, `test`).
  - `test_11_schema_dtypes_and_column_completeness`: Asserts PyArrow schema matches record dictionary fields.
