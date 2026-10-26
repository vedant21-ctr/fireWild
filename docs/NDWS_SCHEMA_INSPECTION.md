# Phase 0, Task 2B: NDWS Schema Inspection & Record Framing Report

**Project:** Wildfire Intelligence (`fireWild`)  
**Repository:** `vedant21-ctr/fireWild`  
**Inspected Sample File:** `data/raw/ndws/sample/next_day_wildfire_spread_eval_00.tfrecord`  
**Expected File Size:** `213,313,000` bytes  
**Expected SHA-256:** `85522369187ADC7E0F76F7CB9E45AD55C47CC8351304CE726BEE27DE20C32D7B`  
**Inspection Date:** 24 October 2026  
**Status:** **SAMPLE VERIFIED & PARSED — ZERO COMPONENT REGRESSIONS**

---

## 1. Sample Identity & Integrity Verification

| Verification Property | Expected Value | Measured Value | Match Status |
| :--- | :--- | :--- | :---: |
| **File Path** | `data/raw/ndws/sample/next_day_wildfire_spread_eval_00.tfrecord` | `data/raw/ndws/sample/next_day_wildfire_spread_eval_00.tfrecord` | **MATCH** |
| **File Size (Bytes)** | `213,313,000` | `213,313,000` | **MATCH** |
| **SHA-256 Checksum** | `85522369187ADC7E0F76F7CB9E45AD55C47CC8351304CE726BEE27DE20C32D7B` | `85522369187ADC7E0F76F7CB9E45AD55C47CC8351304CE726BEE27DE20C32D7B` | **MATCH** |
| **Git Exclusion Status** | Excluded by `.gitignore` | `.gitignore:29:data/raw/` | **EXCLUDED** |

The sample was verified bit-for-bit before parsing.

---

## 2. Reader Selection & Compatibility

* **Active Python Environment:** Python 3.13.0 64-bit on Windows.
* **Lightweight Reader Employed:** `tfrecord` version **1.14.6** + `crc32c` version **2.9.post0** (PyPI pure-Python / wheel).
* **Dependencies Utilized:** `google.protobuf` (7.36.2), `numpy` (2.5.3), `crc32c`.
* **Zero Heavy Frameworks:** **TensorFlow was NOT installed.** `requirements.txt` remains clean and unchanged.
* **Framing Validation:** Standard TensorFlow TFRecord framing passed:
  - Header: 8-byte uint64 record length + 4-byte masked CRC32C.
  - Data: serialized byte payload.
  - Footer: 4-byte masked CRC32C of data payload.
  - Shard record count: exactly **1,000 records** in `next_day_wildfire_spread_eval_00.tfrecord`.
  - Record payload size: exactly **213,297 bytes** per serialized `tf.train.Example`.

---

## 3. Verified Record Schema & Data Types

Every inspected record in the shard strictly parses into a dictionary of **exactly 13 feature keys**. All 13 features are stored as 1D flattened arrays of length **4,096** ($64 \times 64$ pixels) with `dtype=float32`:

| Key Name | Modality | Shape | Data Type | Measured Value Range across Inspected Records | Physical Interpretation & Scaling |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **`elevation`** | Topography | `(4096,)` | `float32` | $[55.0, 409.0]$ m | Surface elevation above sea level (SRTM 30m aggregated to 1km). |
| **`th`** | Meteorology | `(4096,)` | `float32` | $[70.0^\circ, 232.9^\circ]$ | Wind direction azimuth from North (GridMET daily). |
| **`vs`** | Meteorology | `(4096,)` | `float32` | $[2.32, 4.25]$ m/s | Wind velocity at 10m height (GridMET daily). |
| **`tmmn`** | Meteorology | `(4096,)` | `float32` | $[277.5, 283.9]$ K | Daily minimum surface temperature in Kelvin ($\approx 4^\circ\text{C}$ to $11^\circ\text{C}$). |
| **`tmmx`** | Meteorology | `(4096,)` | `float32` | $[291.8, 300.4]$ K | Daily maximum surface temperature in Kelvin ($\approx 18^\circ\text{C}$ to $27^\circ\text{C}$). |
| **`sph`** | Meteorology | `(4096,)` | `float32` | $[0.0058, 0.0091]$ kg/kg | Surface specific humidity (GridMET daily). |
| **`pr`** | Meteorology | `(4096,)` | `float32` | $[0.00, 7.13]$ mm | Daily accumulated precipitation (GridMET daily). |
| **`pdsi`** | Climatology | `(4096,)` | `float32` | $[-2.55, +3.88]$ | Palmer Drought Severity Index (negative = drought; positive = wet). |
| **`NDVI`** | Biosphere | `(4096,)` | `float32` | $[-4315.0, +8935.0]$ | Normalized Difference Vegetation Index scaled by $10,000$ (actual NDVI $[-0.43, +0.89]$). |
| **`population`** | Anthropogenic | `(4096,)` | `float32` | $[0.0, 2684.7]$ | Population density per $\text{km}^2$ (SEDAC GPWv4). |
| **`erc`** | Fire Danger | `(4096,)` | `float32` | $[13.5, 35.5]$ | NFDRS Energy Release Component (fuel dryness indicator). |
| **`PrevFireMask`** | Fire State ($t$) | `(4096,)` | `float32` | $\{-1.0, 0.0, 1.0\}$ | Binary active fire mask at day $t$ (plus nodata/cloud mask). |
| **`FireMask`** | Target ($t+1$) | `(4096,)` | `float32` | $\{-1.0, 0.0, 1.0\}$ | **Prediction Target:** Active fire mask at day $t+1$. |

---

## 4. Label Semantics & Nodata Encodings

Inspection of `FireMask` and `PrevFireMask` across multiple records confirms three distinct numerical states matching MODIS active fire observation standards:

* **`-1.0` (Missing / Cloud / Nodata / Water):**
  - Indicates grid cells where satellite observation was obstructed by cloud cover, smoke blindness, water bodies, or falls outside the satellite sensor swath.
  - In Record 0, 368 cells ($9.0\%$) are $-1.0$.
  - In Record 9, all 4,096 cells ($100\%$) are $-1.0$ (complete cloud obscuration on day $t+1$).
  - **Critical Rule:** All cells where `FireMask == -1.0` or `PrevFireMask == -1.0` **must be masked out** via an explicit `valid_mask` during training and evaluation.
* **`0.0` (Unburned / Extinguished):**
  - Indicates unburned vegetation, soil, or extinguished fire areas.
  - Represents the overwhelming majority class ($>98\%$ to $99.5\%$ of valid cells).
* **`1.0` (Actively Burning / Fire Front):**
  - Indicates thermal anomaly detection (active flaming or smoldering front).
  - In Record 1: `PrevFireMask` has 18 burning cells ($0.44\%$); `FireMask` has 39 burning cells ($0.95\%$).

---

## 5. Metadata & Bounding-Box Findings

### Explicit Absence of String Metadata Keys:
* **Incident ID / Fire Name:** **ABSENT.** The TFRecord samples do not contain string fields such as `fire_id`, `fire_name`, or `nifc_id`.
* **Timestamps / Dates:** **ABSENT.** No explicit calendar date strings (e.g., `YYYY-MM-DD`) or POSIX timestamps are embedded inside the `tf.train.Example` records.
* **Geographic Coordinates:** **ABSENT.** Latitude/longitude coordinates and bounding box coordinates are not stored as explicit scalar or vector keys. The spatial coordinates are implicit 2D grid cell indices $(r, c) \in [0, 63] \times [0, 63]$.

---

## 6. Scientific Implications for Phase 1 & Phase 2

1. **Target Formulation (True Spread vs. Core Persistence):**
   * Evaluating raw `FireMask == 1` conflates persistent burning with actual forward spread.
   * **Verified Target Definition for Phase 1:** A cell $(r, c)$ represents true forward spread if:
     $$Y = 1 \iff \left(\text{PrevFireMask} = 0.0 \land \text{FireMask} = 1.0\right)$$
     Valid sample filter:
     $$\text{Valid} \iff \left(\text{PrevFireMask} \ge 0.0 \land \text{FireMask} \ge 0.0\right)$$
2. **Defensible Split Strategy without Synthetic Incident IDs:**
   * Because raw TFRecords do not contain incident name tags, inventing synthetic fire IDs would be unscientific and untestable.
   * **Author-Validated Shard Partitions:** The authoritative dataset published by Google Research / Fantine Huot pre-shards the data into non-overlapping geographic and seasonal partitions:
     - `train_00.tfrecord` to `train_14.tfrecord`: ~14,979 training samples.
     - `eval_00.tfrecord` and `eval_01.tfrecord`: 1,877 evaluation samples.
     - `test_00.tfrecord` and `test_01.tfrecord`: 1,689 test samples.
   * Adopting this partition guarantees zero patch overlap between train, eval, and test sets and ensures fair, reproducible comparison with published literature.
3. **Tabular AML Extraction Design (Phase 1):**
   * Each $64 \times 64$ patch yields up to 4,096 tabular rows.
   * Feature vector per valid cell:
     - 11 environmental values (`elevation`, `pdsi`, `NDVI`, `pr`, `sph`, `th`, `vs`, `tmmn`, `tmmx`, `erc`, `population`).
     - Spatial neighborhood context: distance to nearest `PrevFireMask == 1` cell, wind-alignment dot product ($\vec{v}_{\text{wind}} \cdot \vec{d}_{\text{fire}}$), and focal count of surrounding burning cells within 3 km and 5 km radii.

---

## 7. Recommended Next Step: Phase 0, Task 2C

**Phase 0, Task 2C: Implement Preprocessing & Tabular Extractor**
* Create `src/features/extract_tabular_ndws.py` to ingest TFRecord shards, apply the nodata validity mask ($\ne -1.0$), compute pre-engagement directional and proximity features, and output structured Parquet datasets for Phase 1 ML baseline experiments.
