# fireWild: Next-Day Wildfire Spread Prediction

**Course project 17 (CSAI AML/DL):** *Next Day Wildfire Spread Prediction via Tabular AML and Spatial Deep Learning.*
**Phase 1 (this submission): machine learning.** For every 1 km pixel of a 64 × 64 km region with an active fire
on day *t*, predict the probability that it is burning on day *t*+1. The predictions come from engineered
environmental, fire, wind and terrain features, using logistic regression, random forest, XGBoost and LightGBM.
Phase 2 adds a CNN / U-Net on the same splits.

| | |
|---|---|
| Data | [Next Day Wildfire Spread (Kaggle)](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread), Huot et al., IEEE TGRS 2022. 18,545 patches, US 2012–2020 |
| Primary metric | PR-AUC (about 1.1% positive pixels, so accuracy is meaningless) |
| Also reported | ROC-AUC, F1 / precision / recall at a validation-chosen threshold, recall at 1% and 5% FPR, Brier score, 95% patch-bootstrap CIs, paired significance tests |
| Splits | **official** (random weeks, temporally separated) and **spatial** (location-grouped, no shared terrain with train) |

---

## Key findings

<!-- RESULTS:BEGIN (filled from experiments/ndws/RESULTS.md) -->
Test PR-AUC with 95% patch-bootstrap CI (thresholds and hyperparameters chosen on validation):

| Model | Official split (prevalence 1.25%) | Spatial split (prevalence 0.48%) | Spatial ROC-AUC |
|---|---|---|---|
| Persistence | 0.107 [0.097, 0.118] | 0.037 [0.032, 0.041] | 0.572 |
| Neighbourhood persistence | 0.173 [0.157, 0.191] | 0.064 [0.055, 0.071] | 0.641 |
| Logistic regression | 0.230 [0.213, 0.249] | 0.117 [0.106, 0.127] | 0.867 |
| Random forest | 0.265 [0.248, 0.282] | 0.135 [0.123, 0.149] | 0.885 |
| **XGBoost** | **0.265** [0.248, 0.283] | **0.141** [0.128, 0.154] | **0.899** |
| LightGBM | 0.263 [0.246, 0.281] | 0.138 [0.127, 0.150] | 0.896 |
| *Huot et al. 2022 (official): persistence / LR / RF / CNN* | *0.115 / 0.198 / 0.225 / 0.284* | | |

1. **Beats the published tabular baselines.** On the official split, engineered features lift logistic regression
   from 0.198 to 0.230 and the random forest from 0.225 to 0.265. Boosting reaches 0.265, close to the paper's CNN
   (0.284). Our persistence (0.107) matches theirs (0.115), so the numbers are comparable.
2. **Boosting beats logistic regression significantly** (paired bootstrap ΔPR-AUC +0.021 spatial, +0.032 official;
   CIs exclude 0). It is **not** significantly better than the random forest, and XGBoost ≈ LightGBM.
3. **Spatial leakage audit.** About 95% of official test patches share terrain with training patches. On a
   location-disjoint split, raw PR-AUC halves, but prevalence also drops (1.25% → 0.48%). Lift over chance
   (20.9× → 28.6×), ROC-AUC (0.905 → 0.896) and the model ranking barely change. The overlap is real, but it does
   not measurably inflate pixel-level tabular results.
4. **What drives predictions.** Lagged burn features dominate (raw environmental channels alone: test PR-AUC 0.010;
   fire features alone: 0.111). Wind adds the next-largest gain. Terrain, weather and vegetation effects are within
   bootstrap noise. SHAP ranks distance to fire, the 17/9/5 km burn fractions, then downwind "wind push" and
   relative elevation.
5. **Physically plausible.** Predicted risk tracks observed ignition across distance bins, rises downwind, and is
   higher on sloped ground. The physics also fixed the unknown image orientation (downwind/upwind ratio 1.12, 95% CI
   1.09–1.15). Probability maps form halos around the active front and miss distant new ignitions. That is the gap
   Phase 2's CNN / U-Net targets.
<!-- RESULTS:END -->

Full tables: [experiments/ndws/RESULTS.md](experiments/ndws/RESULTS.md). Walk-through:
[notebooks/02_ndws_results.ipynb](notebooks/02_ndws_results.ipynb).

---

## Phase 1 deliverables

| Deliverable | Location |
|---|---|
| IEEE-format report (LaTeX source + compiled PDF) | [report/main.pdf](report/main.pdf), [report/main.tex](report/main.tex) |
| Presentation (17 slides, speaker notes name each presenter) | [presentation/fireWild_phase1.pptx](presentation/fireWild_phase1.pptx), [presentation/fireWild_phase1.pdf](presentation/fireWild_phase1.pdf) |
| Results tables | [experiments/ndws/RESULTS.md](experiments/ndws/RESULTS.md) |
| EDA and results notebooks (executed) | [notebooks/](notebooks/) |
| Literature review | [docs/LITERATURE_REVIEW.md](docs/LITERATURE_REVIEW.md) |
| Theory and viva preparation | [docs/THEORY_AND_VIVA_PREP.md](docs/THEORY_AND_VIVA_PREP.md) |
| Dataset card | [DATASET_CARD.md](DATASET_CARD.md) |
| Team contributions | [CONTRIBUTIONS.md](CONTRIBUTIONS.md) |

Rebuild the report with `cd report && tectonic main.tex` (or upload `report/` to Overleaf). Rebuild the slides with
`python presentation/build_deck.py` (needs Google Chrome).

---

## Reproducing Phase 1

```bash
# 0. environment (Python 3.12). macOS: brew install libomp
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. download NDWS from Kaggle (kagglehub) and cache it as NumPy (~4.3 GB in data/ndws/, gitignored)
python -m src.aml.ndws

# 2. terrain-fingerprint leakage audit and spatial split -> data/splits/ (committed)
python -m src.aml.splits

# 3. EDA and data-quality audit -> experiments/ndws/eda/
python -m src.aml.eda

# 4. train and select models (about 10 min each on an 8-core laptop)
python -m src.aml.train --protocol spatial --ablation
python -m src.aml.train --protocol official

# 5. test metrics, bootstrap CIs, significance, figures -> experiments/ndws/results/, RESULTS.md
python -m src.aml.evaluate

# tests (CLBI + Phase 1 pipeline)
python -m unittest discover -s tests -p "test_*.py"
```

If you already have the TFRecords, set `NDWS_RAW_DIR=/path/to/tfrecords` before step 1.

---

## Method summary

1. **Data quality.** Physically impossible values become NaN: temperatures outside 200–330 K (missing values are
   stored as 0 K), non-positive humidity, negative wind speed, and corrupted wind directions (down to −496,522°).
   Negative precipitation is clipped, NDVI rescaled and population log-transformed.
   Unknown labels (−1, about 2.4%) are excluded.
2. **Orientation check.** The Kaggle release has no georeferencing. Of the four row/column conventions, only
   rows south→north, columns west→east makes new fire significantly more frequent downwind than upwind:
   1.12× (95% CI 1.09–1.15) over all training patches, vs 1.01× (0.98–1.04) for rows north→south. It is adopted
   ([EDA](experiments/ndws/eda/wind_orientation.png)).
3. **Leakage audit.** Elevation is static, so patches sharing an identical non-flat 8×8 km elevation block cover the same
   ground. Union-find over these matches gives 845 location groups (mirrored-patch control: 0 / 1,855 matches).
   **About 95% of official test patches share terrain with training.** The spatial protocol assigns whole groups to
   train / val / test (14,822 / 1,855 / 1,868 patches), with zero shared terrain.
4. **Features (32).** Raw channels; lagged burn activity (day-t fire, burning fraction in 3/5/9/17 km windows,
   distance to fire, patch fire count); wind (u/v, fire upwind/downwind within 3 km, wind–fire alignment, wind push);
   terrain (slope, aspect, rise-over-run from the nearest fire); dryness (temperature range, vapour-pressure deficit);
   vegetation (local NDVI mean/std). Every feature uses day-t inputs only (unit-tested).
5. **Imbalance.** All positives plus 2% of negatives from the training split only. Models are trained unweighted
   with a closed-form prior correction, or importance-weighted, whichever is better on validation. Val and test are
   never resampled.
6. **Models.** Persistence and neighbourhood-persistence baselines; logistic regression; random forest; XGBoost and
   LightGBM with small grids and early stopping on validation PR-AUC.
7. **Evaluation.** Thresholds and hyperparameters come from validation; the test split is scored once. Patch-level
   bootstrap CIs, paired bootstrap ΔPR-AUC, feature-group ablations, SHAP, calibration, probability maps and
   distance / wind / slope plausibility checks.

---

## Repository layout

```
src/aml/                     Phase 1 pipeline
  ndws.py                    TFRecord download + NumPy cache
  splits.py                  terrain-fingerprint leakage audit, spatial split
  features.py                per-pixel feature engineering (32 features)
  dataset.py                 pixel sampling, protocols
  train.py                   model fitting, validation-based selection, ablations
  metrics.py                 PR-AUC, recall@FPR, fast patch bootstrap
  evaluate.py                test metrics, significance, figures, RESULTS.md
  eda.py                     EDA and data-quality figures
notebooks/                   01 EDA, 02 results (executed)
experiments/ndws/            EDA figures, results per protocol, logs, RESULTS.md
data/splits/                 split manifest + audit (committed); data/ndws/ is the gitignored cache
models/ndws/                 saved LightGBM / XGBoost / LR models
report/                      IEEE-format LaTeX report (main.tex, references.bib, main.pdf)
presentation/                slide sources, build script, fireWild_phase1.pptx / .pdf
docs/                        literature review, theory and viva prep, CLBI dataset card and API docs
tests/test_aml.py            pipeline unit tests (features, metrics, splits)

src/model, src/api, src/evaluation, data/final, experiments/*.md
                             CLBI prototype on SYNTHETIC data (see below)
```

---

## CLBI prototype (synthetic data, not part of Phase 1)

The repository also contains **Candidate Line Breach Intelligence (CLBI)**, a FastAPI decision-support prototype that
ranks 100 m containment-line segments by breach risk. **Its dataset `fireline_segments_v1` is simulated** by
`src/data/build_dataset.py`. Labels come from a hand-written logistic rule, and the fire names are scenario labels,
not data sources (see [docs/CLBI_SYNTHETIC_DATASET_CARD.md](docs/CLBI_SYNTHETIC_DATASET_CARD.md)). Its metrics
(e.g. PR-AUC about 0.95 for the majority "Held" class) exercise the software and are **not** empirical evidence. A
future version could use real NDWS spread predictions and USFS Fireline Effectiveness outcomes.

```bash
python -m src.model.train_clbi_model          # train the CLBI logistic model
python -m src.model.clbi_demo                 # 100-segment demo
python -m uvicorn src.api.main:app --reload   # API at http://localhost:8000/docs
```

API details: [docs/CLBI_API_CONTRACT.md](docs/CLBI_API_CONTRACT.md).

---

## Limitations

1. The Kaggle NDWS release has no coordinates or dates, so spatial independence is audited by terrain fingerprint
   (exact block matches). Partial overlaps with fewer than 64 km² shared, or resampled terrain, would be missed.
2. Labels are 1 km daily MODIS detections: clouds, detection timing and smoke make them noisy, and 74% of next-day
   fire is new rather than persistent.
3. Inputs lack fuel moisture, fuel type, suppression activity and sub-daily wind.
4. Pixel-level tabular models see fire-front geometry only through hand-made summaries. That is the gap Phase 2's
   spatial models address.
