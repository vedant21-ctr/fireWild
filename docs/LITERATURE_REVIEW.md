# Literature Review: Next-Day Wildfire Spread Prediction

This review covers the work the project builds on: physical and simulation models of fire spread,
machine-learning spread prediction (especially on the NDWS benchmark), the tabular learners, spatial deep
learning, hybrid models, and evaluation methodology for imbalanced, spatially autocorrelated data. Full
references, each checked against CrossRef, arXiv or the publisher's page, are in `report/references.bib`.

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

## 3. Tabular learners

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

## 5. Spatial deep learning

- **Fully convolutional networks (Long et al., 2015)** replace the classifier head with convolutions, so a
  network outputs a value per pixel for any input size. Both of our networks are fully convolutional. That is
  why they can train on 32x32 crops and predict on 64x64 patches.
- **U-Net (Ronneberger et al., 2015)** adds a contracting encoder, an expanding decoder and skip connections.
  The encoder gathers context, and the skips return fine detail lost to pooling. Fire fronts need both: a
  regional view of the fire and a sharp boundary.
- **Dilated convolutions (Yu & Koltun, 2016)** widen the receptive field without pooling, which keeps resolution.
  Our compact CNN baseline uses them, so comparing it with the U-Net isolates the value of the encoder-decoder.
- **Losses.** Binary cross-entropy treats pixels independently. Class weighting raises the cost of missed fire.
  The **Dice loss (Milletari et al., 2016)** optimises region overlap directly and was proposed for strongly
  imbalanced segmentation. We compare these on validation data.
- **Regularisation and optimisation.** Batch normalisation (Ioffe & Szegedy, 2015), dropout (Srivastava et al.,
  2014) and AdamW's decoupled weight decay (Loshchilov & Hutter, 2019) are each tested by ablation.
- **Convolutions and absolute position.** CNNs are often assumed to be translation-equivariant, but
  **Kayhan & van Gemert (2020)** show that boundary effects let convolutional layers exploit absolute position, and
  **Liu et al. (2018)** study how CNNs handle coordinates. This explains the NDWS patch-centre leak: a network given
  full patches can locate the centre from the padded border. Huot et al. (2022) already train on random crops.
- **Sequence models.** **ConvLSTM (Shi et al., 2015)** needs a multi-day input sequence. Each NDWS sample has only
  two snapshots, so, as the project brief instructs, we do not use it.

## 6. Hybrid models and the tabular-versus-deep question

**Grinsztajn et al. (2022)** show that tree ensembles still beat deep networks on typical tabular data, partly
because trees handle irregular targets and uninformative features well. That makes tuned trees a serious baseline
here, not a formality. **Stacked generalisation (Wolpert, 1992)** learns how to combine base models using data they
were not fitted on. Our late fusion is a small stacker of this kind, fitted on held-out validation patches.

## 7. Gaps this project addresses

1. Published NDWS tabular baselines use raw pixels plus their 3x3 neighbourhood. They do not engineer physically
   motivated features (multi-scale burn density, distance to the front, wind-fire alignment, upslope from fire,
   vapour-pressure deficit) or test gradient boosting, which the brief asks for.
2. We know of no published NDWS evaluation that checks whether the official split is spatially independent. Our
   terrain fingerprint shows that about 95% of official test patches share terrain with training patches, and the
   project reports a location-disjoint split and a split that is disjoint in both location and week.
3. Patch centring is a target leak. We measure how far it reaches both the networks and the tabular models.
4. We report **uncertainty** (patch-level bootstrap intervals, paired tests, repeated seeds) instead of single point
   estimates, and check **physical plausibility**.
5. We test a **hybrid** against both of its parents on identical pixels, and report the outcome either way.
