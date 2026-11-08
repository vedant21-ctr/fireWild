# Methodology

This document explains what was done and why. Measured results are in `experiments/ndws/FINAL_RESULTS.md`.

## 1. Problem

For every 1 km pixel of a 64 x 64 km patch holding an active fire on day *t*, estimate the probability that the pixel
is burning on day *t*+1, using only information available at the end of day *t*. About 1% of pixels are positive.
This is a statistical association task, so predictions should not be read as proof of how a fire physically propagated.

## 2. Preventing leakage

| Leakage type | Control | Verified by |
|---|---|---|
| Future target in inputs | Features use day-*t* inputs only | `test_features_ignore_target`, `test_inputs_ignore_target` |
| Same terrain in train and test | `spatial` and `strict` protocols | `verify_manifest()` and `tests/test_aml.py` |
| Same weeks in train and test | `official` and `strict` protocols | `verify_manifest()` |
| Patch position predicts the label | Crop training, leak probes, position-controlled scoring | Sections 3 and 7 |
| Preprocessing fitted on held-out data | Scaling, imputation and normalisation use training data only | `src/dl/train.py`, `src/aml/train.py` |
| Resampling held-out data | Negative subsampling is applied to training data only; validation and test keep every known pixel | `src/aml/dataset.py` |
| Tuning on test data | Trees are tuned on the protocol's validation split; networks on the `dev` split; thresholds on validation | Code and `configs/` |

## 3. The patch-centre position leak

NDWS samples are centred on fires, so a pixel's position within the patch predicts its label. A network
given full patches can work out its distance from the padded border and learn that "the middle burns tomorrow"
(Kayhan & van Gemert, 2020). That signal would not exist in a real forecast region. Controls:

- Networks train on random 32x32 crops, so the original centre can fall anywhere in the input.
- The **patch-centre prior** (position only) and a **full-patch U-Net** are kept as diagnostics that measure the leak.
- `src/aml/leak_test.py` tests the tabular models in three ways: adding position columns, predicting position from the
  features, and retraining and scoring on windows whose features are computed from the window alone.
- Every model is also scored in a **position-controlled regime**: one fixed random 32x32 window per test patch.

## 4. Feature engineering

Every feature is computed from day-*t* inputs of the same patch, so all are available at prediction time. Fire windows use
a box filter over the day-*t* fire mask; *d* is the Euclidean distance in km to the nearest burning pixel.

| Feature | Computation | Environmental meaning | Why included |
|---|---|---|---|
| `elevation` | SRTM value | Terrain height | Fuel and climate vary with altitude |
| `wind_speed` | GRIDMET `vs` | Wind strength | Wind drives spread |
| `tmmn`, `tmmx` | GRIDMET, K | Daily min and max temperature | Heat dries fuel |
| `sph` | GRIDMET | Specific humidity | Moist air slows drying |
| `pr` | GRIDMET, clipped at 0 | Precipitation | Rain wets fuel |
| `pdsi` | GRIDMET drought | Palmer Drought Severity Index | Long-term dryness |
| `ndvi` | VIIRS / 10^4 | Vegetation greenness | Fuel amount and condition |
| `log_population` | log(1 + density) | Human presence | Proxy for ignitions and suppression |
| `erc` | NFDRS | Energy release component | Composite fuel dryness |
| `prev_fire` | mask = 1 | Burning today | Fire tends to persist |
| `prev_unknown` | mask = -1 | Today's mask is unknown | Marks unreliable fire context |
| `fire_frac_{3,5,9,17}` | Box-filter mean of fire mask | Burn density at four scales | Neighbourhood burn activity |
| `dist_to_fire_km` | Distance transform | Distance to the front | New fire is mostly near existing fire |
| `log_patch_fire_count` | log(1 + burning pixels) | Fire size | Large fires behave differently |
| `wind_u`, `wind_v` | -speed x (sin th, cos th) | Wind vector | Directional wind information |
| `upwind_fire_3km` | Fire 1-3 km upwind | Fire is blowing toward the pixel | Directional exposure |
| `downwind_fire_3km` | Fire 1-3 km downwind | Fire is blowing away | Directional contrast |
| `fire_wind_alignment` | Cosine of wind and fire-to-pixel vectors | Pixel is downwind of the nearest fire | Directional spread |
| `wind_push` | alignment x speed / (1 + *d*) | Downwind pressure, decaying with distance | Encodes the wind-distance interaction |
| `slope_deg` | arctan of elevation gradient | Terrain steepness | Slope alters spread |
| `aspect_sin`, `aspect_cos` | Downslope direction | Slope orientation | Sun exposure and fuel drying |
| `upslope_from_fire` | (z - z_fire) / *d*, clipped to [-1, 1] | Pixel lies above the fire | Fire runs uphill |
| `temp_range` | tmmx - tmmn | Diurnal range | Dry-air indicator |
| `vpd_kpa` | e_s(tmmx) - e_a | Vapour-pressure deficit | Atmospheric dryness |
| `ndvi_mean_5`, `ndvi_std_5` | 5x5 mean and std of NDVI | Local fuel level and patchiness | Fuel continuity |

Formulas: e_s = 0.6108 exp(17.27 T / (T + 237.3)) kPa with T in Celsius; e_a = q p / (0.622 + 0.378 q);
p = 101.3 ((293 - 0.0065 z) / 293)^5.26 kPa.

## 5. Models and class imbalance

**Tabular.** Dummy (training rate), persistence, neighbourhood persistence, logistic regression, random forest, XGBoost
and LightGBM. Training keeps every positive pixel and 2% of negatives. Models are fitted either unweighted with a
closed-form prior correction (Saerens et al., 2002) or importance-weighted, whichever validates better.

**Spatial.** A compact dilated CNN and a U-Net (`docs/ARCHITECTURE.md`). Settings are chosen greedily on the `dev`
protocol: loss (BCE, weighted BCE, Dice, BCE + Dice), dropout, weight decay and width. Training uses AdamW, a
plateau learning-rate scheduler, early stopping on validation PR-AUC, and flips/transposes that also transform the
wind and aspect vectors. Final runs repeat three seeds, except the full-patch diagnostic.

## 6. Hybrid approach

The tabular ablations show that hand-built context (distance, wind alignment) carries most of the signal, while
networks can in principle learn shape that a pixel-wise model cannot see. Two ways of combining them are tested:

- **Option B: feature-assisted network.** Adds the engineered channels to the U-Net input, so it starts from the physics
  that trees exploit. The feature set is selected on `dev`.
- **Option A: late fusion (stacking; Wolpert, 1992).** logit(p) = a + b1 logit(p_tree) + b2 logit(p_network). The
  weights are fitted on half the validation patches. The other half chooses the hybrid and its threshold, and test data
  is used only for the final score.

Option C, spatial context for tabular models, is the existing feature set and is measured by the tabular ablations.

## 7. Evaluation

- Primary metric: **PR-AUC**. Also ROC-AUC, recall at 1% and 5% false-positive rates, F1, precision, recall, Brier.
- Thresholds come from validation data.
- 95% intervals come from a **patch-level bootstrap**, because pixels in a patch are correlated. Differences use a paired
  bootstrap on identical resamples.
- Positive rates differ between protocols, so protocols are compared using lift (PR-AUC / positive rate) and ROC-AUC.
- Reported for three protocols, two targets (active and transition) and two regimes (full patch and fixed window).

## 8. Limitations

NDWS has no coordinates, dates, event labels, fuel moisture or suppression data. MODIS 1 km daily detections are noisy. The
`strict` test set is small. Terrain matching only detects exact 8 x 8 km overlaps, so adjacent fires may remain in
different splits. Weather spans all of day *t*, so a forecast can only be issued after day *t* ends.
