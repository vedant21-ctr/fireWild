# Theory and Viva Preparation (Phase 1, ML)

Every team member should be able to explain everything on this page without notes. Numbers quoted
come from `experiments/ndws/RESULTS.md`, `experiments/ndws/eda/eda_summary.json` and
`data/splits/split_audit.json`.

---

## 1. Problem formulation

- **Input:** at day *t*, a 64×64 km patch (1 km pixels) with 11 environmental rasters plus the day-*t*
  fire mask.
- **Output:** for every pixel, P(pixel burns on day *t*+1).
- **Unit of prediction:** the pixel (1 km²). This is binary classification with extreme imbalance: about
  1.1% of known-label pixels are positive.
- **Unknown labels:** `-1` (cloud / no data) appears in about 2.4% of target pixels. These pixels are
  dropped from training and evaluation, as in Huot et al. (2022).
- **Why tabular?** Phase 1 is the ML phase. Each pixel becomes one row of engineered features, so
  spatial context has to be encoded by hand (neighbourhood aggregates, distances, directional features).
  The brief's "gap" is exactly that hand-made summaries may miss fire-front geometry, which Phase 2's CNN /
  U-Net can learn directly.

## 2. Data quality findings (EDA)

| Issue | Handling |
|---|---|
| `tmmn`, `tmmx` outside 200–330 K (missing stored as 0 K; maxima up to 344 K ≈ 71 °C), about 0.35% of pixels | NaN; trees route NaN natively, LR/RF use median imputation fitted on train |
| `sph` ≤ 0, `vs` < 0 | NaN |
| `th` (wind direction) corrupted: values from −496,522 to 37,620 in about 0.2% of pixels | outside [0, 361] → NaN; (360, 361] wrapped |
| elevation below 0 m (minimum −5 m) | kept: sub-sea-level terrain is real |
| `pr` slightly negative (interpolation artefact) | clipped at 0 |
| `NDVI` stored ×10⁴ | rescaled to [-1, 1] |
| `population` heavy-tailed | log1p |
| 74% of next-day fire pixels were **not** burning at *t* | persistence alone is weak; this is the hard part |
| Kaggle release has no coordinates or orientation metadata | orientation inferred empirically (section 4) |

## 3. Feature engineering (32 features, `src/aml/features.py`)

| Group | Features | Physical rationale |
|---|---|---|
| raw (10) | elevation, wind speed, tmin, tmax, humidity, precip, PDSI, NDVI, log population, ERC | fuel, weather, drought, ignition sources |
| fire (8) | prev_fire, prev_unknown, burning fraction in 3/5/9/17 km windows, distance to nearest fire, log patch fire count | fire spreads from existing fire; multi-scale "lagged burn activity" |
| wind (6) | u, v components, fire 1–3 km upwind / downwind, cosine alignment of wind with the fire→pixel vector, wind push = alignment × speed / (1 + distance) | spread is faster downwind |
| terrain (4) | slope, aspect (sin/cos), rise-over-run from nearest fire pixel | fire runs faster upslope (preheating of fuel above the flame); empirically, sloped ground near fire burns about 2× more than level ground |
| weather (2) | temperature range, vapour-pressure deficit | VPD = e_s(T_max) − e_a measures atmospheric dryness, which drives fuel moisture |
| vegetation (2) | 5×5 NDVI mean and std | fuel amount and continuity |

Formulas worth knowing:
- Saturation vapour pressure (Tetens): e_s = 0.6108·exp(17.27·T / (T + 237.3)) kPa, with T in °C.
- Actual vapour pressure from specific humidity q: e_a = q·p / (0.622 + 0.378·q).
- Pressure from elevation (FAO-56): p = 101.3·((293 − 0.0065·z)/293)^5.26 kPa.
- Wind *toward* vector from meteorological direction θ (wind blows FROM θ): (u, v) = −speed·(sin θ, cos θ).

**Leakage rule:** every feature uses day-*t* inputs only. `tests/test_aml.py::test_features_ignore_target`
checks that changing the t+1 mask leaves the features unchanged.

## 4. Orientation check (a physical plausibility test)

The Kaggle TFRecords carry no georeferencing. We tested all four row/column conventions on all 14,979
training patches. Only rows south→north, columns west→east gives significantly more new fire downwind than
upwind: ratio 1.12, 95% patch-bootstrap CI [1.09, 1.15]. Rows north→south gives 1.01 [0.98, 1.04] (no wind
effect), and the fully reversed convention gives 0.89 (physically backwards). We adopt south→north. The
effect is modest because GRIDMET wind is a daily ~4 km average. Terrain: within 3 km of fire, upslope (17.0%) and downslope (16.6%)
pixels burn about twice as often as level ground (7.9%) (EDA `terrain_effect.png`). Slope magnitude matters;
the upslope asymmetry is marginal at 1 km daily resolution. Be honest about this in the viva. Viva point: *we let physics choose between two
conventions instead of assuming one.*

## 5. Splits and leakage

- **Official split:** random **weeks** (8:1:1) with a one-day buffer, so it is temporally separated.
- **Problem found:** the same terrain appears in different splits (recurring or overlapping fires).
- **Terrain fingerprint:** elevation is static, so if two patches share an identical, non-flat 8×8 km
  elevation block, they cover the same ground. Algorithm: hash every 8×8 block on a stride-8 grid; slide
  an 8×8 window over every patch and look up hashes (an exact match is O(1) per window); join matches with
  **union-find**, giving 845 location groups.
- **Control:** horizontally mirrored patches should never match. They match 0 / 1,855 times, so the hits
  are not chance.
- **Finding:** about 95% of official test patches share terrain with training patches.
- **Spatial protocol:** whole location groups are assigned to train / val / test (80 / 10 / 10 by patch
  count), so held-out patches share no terrain with training. Consecutive days of the same fire overlap in
  terrain and therefore fall in the same group. This makes the split **event-disjoint** as well.
- **Result:** raw PR-AUC halves on the spatial test, but its prevalence also drops (1.25% → 0.48%) because the
  held-out groups are smaller, isolated fires (the two giant recurring-fire groups must stay in train). Lift over
  chance (20.9× → 28.6×), ROC-AUC (0.905 → 0.896) and the model ranking barely change. Conclusion: the overlap is
  real but does not measurably inflate pixel-level tabular results. Only a hint of location memorisation shows
  (elevation and population carry more SHAP weight on the official split). See `experiments/ndws/results/protocol_gap.png`.

## 6. Models

### Logistic regression
p = σ(wᵀx + b), trained by minimising log-loss L = −Σ[y log p + (1−y) log(1−p)] + λ‖w‖². The loss is
convex, so there is a single optimum (found with L-BFGS). Features are standardised so the L2 penalty treats
them equally. It is linear in log-odds, so interactions such as "wind matters only near fire" must be
hand-made (`wind_push`).

### Random forest
Many deep trees, each trained on a bootstrap sample with a random feature subset at each split. The
variance of the average of B trees with pairwise correlation ρ is ρσ² + (1−ρ)σ²/B. Feature subsampling
lowers ρ, and averaging shrinks the second term. Trained on a 400k-row subsample for memory.

### Gradient boosting (XGBoost, LightGBM)
An additive model F_M(x) = Σ_m η·f_m(x). Each tree f_m is fitted to the gradient of the loss at the
current prediction. XGBoost uses a second-order Taylor expansion: with gradient sums G and hessian sums H
per leaf, the optimal leaf weight is **w\* = −G / (H + λ)** and the split gain is
½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − (G_L+G_R)²/(H_L+H_R+λ)] − γ. For log-loss, g = p − y and h = p(1−p).
- **XGBoost** grows trees depth-wise (we use `max_depth`) and finds splits with histograms.
- **LightGBM** grows trees **leaf-wise** (it splits the leaf with the largest gain; we use `num_leaves`). It
  adds GOSS (keeps large-gradient rows and samples small-gradient rows) and EFB (bundles mutually exclusive
  sparse features).
- **Regularisation we needed:** pixels in one patch share almost constant weather values, so small leaves
  can memorise individual patches. Large minimum leaf sizes (2,000–5,000 rows), column subsampling (0.5),
  L2 penalty and **early stopping** on validation PR-AUC control this. Viva point: the first configuration
  (min leaf 200, importance weights) stopped after 5 trees. Diagnosing and fixing that is part of the work.

## 7. Imbalance handling and calibration

- All positive pixels are kept, plus a random 2% of negatives (training split only). Val and test are
  never subsampled.
- Two options were compared on validation for each family:
  1. **Importance weights:** weight 1/0.02 = 50 on negatives, which restores the true prior during training.
  2. **Unweighted + prior correction:** keeping each negative with probability r multiplies the odds by
     1/r, so odds_true = r·odds_sample, i.e. **p_true = r·p / (r·p + 1 − p)** (Saerens et al., 2002). The
     map is monotone, so ranking metrics (PR-AUC, ROC-AUC) are unchanged and only calibration changes.
- Boosting worked better unweighted. LR and RF pick whichever option is better on validation (stored in
  `results.json → selection`).
- **Brier score** = mean (p − y)². It rewards calibrated probabilities. The reliability plot compares
  mean predicted probability with observed frequency per quantile bin.

## 8. Metrics

- **PR-AUC (average precision)** = Σ_k (R_k − R_{k−1})·P_k. Chance level equals the positive rate (about
  0.01), so 0.2 is about 20× chance. With 1% positives a model can have ROC-AUC 0.9 and still produce
  mostly false alarms. PR-AUC exposes this because precision depends on the negative count.
- **ROC-AUC** = P(score of a random positive > score of a random negative). It is insensitive to class
  ratio, so it is reported but is not the headline metric.
- **F1 / precision / recall** use a threshold chosen **on validation** (max F1) and are then applied to test
  unchanged. Choosing the threshold on test would leak.
- **Recall at 1% / 5% FPR:** the fraction of new fire caught if at most 1% / 5% of non-fire pixels may be
  flagged, i.e. an alarm budget.
- **Accuracy is not reported:** predicting "no fire" everywhere scores about 99%.

## 9. Uncertainty and significance

Pixels in one patch are correlated, so resampling pixels would understate uncertainty. We resample
**patches** with replacement (500 bootstrap replicates) and recompute PR-AUC on all pixels of the resampled
patches. That gives a 95% percentile CI. For model comparisons we use a **paired** bootstrap (same
resampled patches for both models) on ΔPR-AUC. If the 95% CI of Δ excludes 0, the difference is robust.
Implementation trick: sort scores once, then each replicate is just a weighted cumulative sum
(`src/aml/metrics.py::BootstrapAP`).

## 10. Likely viva questions

1. *Why PR-AUC and not accuracy or ROC-AUC?* See section 8.
2. *How do you know there is no leakage?* Features use day-*t* inputs only (unit-tested); thresholds and
   hyperparameters are chosen on validation; test is scored once; the spatial protocol removes shared
   terrain (audited, with a mirrored-patch control).
3. *Why is the spatial-split PR-AUC so much lower?* Mostly prevalence: PR-AUC's chance level is the positive
   rate, and the spatial test has 0.48% positives vs 1.25% (smaller, isolated fires). Lift and ROC-AUC are
   nearly unchanged, so the overlap does not substantially inflate the official score. Do not claim it does.
4. *Why gradient boosting over random forest?* Boosting reduces bias stage by stage, while RF reduces
   variance by averaging. Here boosting is **not** significantly better than RF (paired ΔPR-AUC CI includes 0
   on both protocols). Both clearly beat logistic regression, so non-linearity matters but the ensemble type does not.
5. *What does SHAP show?* Distance to fire, then the 17/9/5 km burn fractions, then wind push (downwind of
   fire raises risk), upslope_from_fire, elevation and NDVI. High VPD raises risk.
6. *Why is performance far from perfect?* MODIS 1 km daily detections are noisy (clouds, detection
   timing); 74% of t+1 fire is new; there is no fuel-moisture, suppression or sub-daily wind information.
   Huot et al.'s CNN gets 28.4% PR-AUC on the official split.
7. *What would you do in Phase 2?* A U-Net on the multichannel patch, with the same splits and metrics.
   Add ConvLSTM only with multi-day sequences (e.g. WildfireSpreadTS). Possible hybrid: feed LightGBM
   probabilities or engineered channels into the U-Net, or stack CNN embeddings into boosting. The DL model
   must beat the tuned LightGBM on the **spatial** test with a paired bootstrap.
8. *Why 2% negative sampling?* It keeps memory manageable (1.9M training rows) while keeping all
   positives. 5% gave the same validation PR-AUC (0.226 vs 0.225).
9. *How was the image orientation determined?* See section 4.
10. *What is the CLBI part of the repository?* A software prototype trained on **synthetic** data (see
    `docs/CLBI_SYNTHETIC_DATASET_CARD.md`). It is not part of the Phase 1 evaluation.
