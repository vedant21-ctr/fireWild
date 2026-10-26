# Literature Review: Next-Day Wildfire Spread Prediction (Phase 1, ML)

This review covers four strands of work that the Phase 1 pipeline builds on: physical and simulation
models of fire spread, machine-learning spread prediction (especially on the NDWS benchmark), the
tabular learners used here, and evaluation methodology for imbalanced, spatially autocorrelated data.
Full references are in `report/references.bib`.

## 1. Physical and simulation models

Operational fire-spread tools descend from **Rothermel's (1972)** semi-empirical surface spread model,
which gives rate of spread as a function of fuel properties, fuel moisture, wind and slope. Simulators such
as **FARSITE (Finney, 1998)** propagate a fire perimeter over gridded fuel, terrain and weather layers using
Huygens' wavelet principle. These models encode the physics we use as plausibility checks: spread is
faster **downwind** and **upslope**, and it slows in moist or sparse fuel. Their weakness is the input
data they need (detailed fuel maps, fine-scale wind) and the computational cost of ensembles. This
motivates data-driven models that learn spread directly from remote-sensing observations.

## 2. Machine learning for fire spread

**Jain et al. (2020)** survey about 300 ML papers in wildfire science. They find that fire-behaviour
prediction is dominated by random forests, gradient boosting, SVMs and, increasingly, CNNs, and they point
out the lack of shared benchmarks. Early deep-learning work such as **FireCast (Radke et al., 2019)** and
**Hodges & Lattimer (2019)** showed that CNNs can learn spread from rasterised inputs, but each used its
own region-specific data. **Sayad et al. (2019)** built a remote-sensing dataset (NDVI, land-surface
temperature, thermal anomalies) and showed that standard classifiers can predict fire occurrence from it.

**Huot et al. (2022)** released **Next Day Wildfire Spread (NDWS)**, the dataset used in this project. It
contains 18,545 samples of 64 km × 64 km at 1 km resolution across the contiguous US (2012–2020). Each
sample has 11 environmental inputs (SRTM elevation; GRIDMET wind, temperature, humidity, precipitation;
drought index; VIIRS NDVI; population; ERC) plus the day-t fire mask, and the target is the day t+1 MODIS
fire mask. Their split assigns random weeks to train/eval/test in an 8:1:1 ratio with a one-day buffer.
Their pixel-level baselines use each pixel plus its 8 neighbours: logistic regression reaches
**PR-AUC 19.8%**, a random forest **22.5%**, persistence **11.5%**, and a convolutional autoencoder
**28.4%**. Their feature analysis finds the previous fire mask to be the most important input. Weather and
vegetation add information mainly when combined with it.

**Gerard et al. (2023), WildfireSpreadTS,** extend the idea to multi-day time series (607 fire events,
2018–2021) with event identifiers and dates. That supports year-based splits and temporal models
(ConvLSTM, temporal U-Nets), which matters for Phase 2.

## 3. Learners used in Phase 1

- **Logistic regression** is the linear, well-calibrated reference model, and Huot et al.'s baseline.
- **Random forests (Breiman, 2001)** average decorrelated deep trees grown on bootstrap samples, which
  reduces variance.
- **Gradient boosting (Friedman, 2001)** fits trees stage-wise to the negative gradient of the loss.
  **XGBoost (Chen & Guestrin, 2016)** adds second-order (Newton) leaf weights, L1/L2 regularisation and
  histogram split finding. **LightGBM (Ke et al., 2017)** grows trees leaf-wise and adds
  gradient-based one-side sampling and exclusive feature bundling for speed. The project brief names both
  as the required AML baseline.
- **SHAP (Lundberg & Lee, 2017)** attributes each prediction additively to features using Shapley values.
  TreeSHAP computes these exactly and quickly for tree ensembles.

## 4. Evaluation under imbalance and spatial dependence

- **Metrics:** With about 1% positive pixels, accuracy is uninformative. **Saito & Rehmsmeier (2015)** show
  that precision-recall analysis is more informative than ROC under heavy imbalance, which is why PR-AUC is
  the primary metric here (as in Huot et al.). Recall at a fixed false-positive rate reflects an
  operational alarm budget.
- **Calibration:** **Niculescu-Mizil & Caruana (2005)** show that boosted trees and random forests are
  often poorly calibrated. Training on class-rebalanced data also shifts probabilities;
  **Saerens et al. (2002)** give the closed-form prior correction we apply after negative subsampling.
- **Spatial leakage:** **Roberts et al. (2017)** show that random cross-validation on spatially
  autocorrelated data inflates performance estimates, and they recommend blocked or grouped splits.
  NDWS's week-based split keeps time periods apart but not locations. Overlapping or recurring fires can put
  the same terrain in train and test. We quantify this with a terrain fingerprint and add a spatially
  disjoint protocol.

## 5. Gap addressed in Phase 1

1. Published NDWS tabular baselines use only raw pixels plus their 3×3 neighbourhood. They do not
   **engineer physically motivated features**: multi-scale lagged burn density, distance to the front,
   wind–fire alignment, upslope-from-fire, vapour-pressure deficit. They also do not test **gradient
   boosting**, which the brief asks for.
2. No published NDWS evaluation that we know of checks the **spatial independence** of the official
   split. We show that about 95% of official test patches share terrain with training patches, and we report
   results on a location-grouped split as well. Prevalence-normalised results and model rankings turn out to be
   robust to the overlap.
3. We report **uncertainty** (patch-level bootstrap CIs and paired tests) instead of single point
   estimates, and we check **physical plausibility** (distance, wind and slope response) of the learned
   model.

Phase 2 will test whether convolutional models (U-Net; ConvLSTM only if multi-day sequences are added),
which see fire-front geometry directly, close the gap these tabular features leave.
