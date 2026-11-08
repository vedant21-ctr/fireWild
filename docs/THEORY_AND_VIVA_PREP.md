# Viva Guide

Project 17: Next Day Wildfire Spread Prediction via Tabular AML and Spatial Deep Learning.

The viva is individual, so every member should be able to answer every question here. Each answer has four parts:

- **Simple:** how you would explain it to a friend.
- **Technical:** the precise statement.
- **Maths:** the idea behind it.
- **Code:** where to open the project and show it.

Learn the reasoning, not the wording. Results to quote are in `experiments/ndws/FINAL_RESULTS.md`.

## The project in five sentences

1. For each 1 km pixel of a 64 x 64 km area around a fire, we predict the probability that it is burning the next day.
2. Only about 1% of pixels burn, so we judge models by PR-AUC rather than accuracy.
3. We compare tabular models, which see one pixel at a time plus hand-made features, with CNN and U-Net models, which see the whole image.
4. We test on three splits, because a random split can flatter a model by reusing the same places and weeks.
5. We also check whether models are cheating by learning that the middle of a patch usually burns.

---

## Basic questions

### B1. What exactly are you predicting?

- **Simple:** Given today's fire map and today's weather, terrain and vegetation, which pixels will be on fire tomorrow?
- **Technical:** Binary per-pixel classification. The input is 11 day-*t* environmental rasters plus the day-*t* fire mask. The label is the MODIS fire mask at *t*+1. Pixels with label -1 (cloud or no data) are ignored.
- **Maths:** Each model estimates P(y = 1 | x), where x is the information available at the end of day *t*.
- **Code:** `src/aml/features.py::patch_target`, `src/dl/model.py::masked_bce`.

### B2. What is the difference between the `active` and `transition` targets?

- **Simple:** `active` asks "will this pixel be burning tomorrow?", which includes fires that just keep burning. `transition` looks only at pixels not burning today, so it asks "will fire arrive here?".
- **Technical:** `active` uses every known pixel. `transition` keeps pixels known not to be burning at *t*. All models train on `active`. Their scores are then evaluated on both targets.
- **Maths:** Transition is P(y(t+1) = 1 | y(t) = 0), a conditional form of the same probability, so the same scores can be used.
- **Code:** `src/evaluation/compare.py::target_masks`.

### B3. Why not use accuracy?

- **Simple:** If 99% of pixels never burn, a model that always says "no fire" is 99% accurate and completely useless.
- **Technical:** Accuracy is dominated by the majority class. PR-AUC measures how well the rare positives are ranked at the top.
- **Maths:** A random ranking has PR-AUC equal to the positive rate, so about 0.01 here. Lift is PR-AUC divided by the positive rate.
- **Code:** `src/aml/metrics.py::compute_metrics`.

### B4. What are precision, recall and F1?

- **Simple:** Precision: when we say fire, how often are we right? Recall: of the real fires, how many did we catch? F1 balances the two.
- **Technical:** They are computed at one threshold, chosen on validation to maximise F1, and then applied unchanged to test.
- **Maths:** Precision = TP/(TP+FP). Recall = TP/(TP+FN). F1 = 2PR/(P+R), the harmonic mean, which is low if either value is low.
- **Code:** `src/aml/metrics.py::best_f1_threshold`.

### B5. What are ROC-AUC and recall at a fixed false-positive rate?

- **Simple:** ROC-AUC is the chance that a random burning pixel scores higher than a random non-burning one. Recall at 1% FPR says how many fires we catch if we accept false alarms on 1% of safe pixels.
- **Technical:** ROC-AUC can look high under imbalance because the huge number of negatives keeps the FPR small. Fixed-FPR recall gives an operational reading.
- **Maths:** ROC plots TPR against FPR, where FPR = FP/(FP+TN). With a very large TN, FPR barely moves even when false alarms are numerous.
- **Code:** `src/aml/metrics.py::recall_at_fpr`.

### B6. What are your baselines and why do you need them?

- **Simple:** A baseline tells you whether a clever model is actually better than a trivial rule.
- **Technical:** Dummy predicts the training positive rate everywhere. Persistence predicts that today's fire stays. Neighbourhood persistence uses the share of fire around each pixel.
- **Maths:** Dummy has a constant score, so its PR-AUC equals the prevalence and its ROC-AUC is 0.5.
- **Code:** `src/aml/train.py::heuristic_scores`.

### B7. What is a train, validation and test split?

- **Simple:** Train is for learning, validation is for choosing settings, and test is the final exam, opened once.
- **Technical:** Hyperparameters, early stopping, thresholds and hybrid selection use validation only. Test is scored once with the chosen settings.
- **Maths:** If you pick the best of many options using test, the score you report is an optimistically biased maximum.
- **Code:** `src/aml/dataset.py`, `src/dl/train.py::train`.

### B8. What is a U-Net?

- **Simple:** It shrinks the image to understand the big picture, then grows it back to give an answer for each pixel. Shortcuts bring the fine detail back.
- **Technical:** Three downsampling stages, a bottleneck and three upsampling stages. Skip connections concatenate encoder features onto decoder features of the same resolution.
- **Maths:** Each pooling step doubles the area each later convolution covers. Skip connections restore the high-resolution information that pooling removed.
- **Code:** `src/dl/model.py::UNet`.

---

## Intermediate questions

### I1. Why three splits?

- **Simple:** Each split asks a different question. `official` is the published benchmark, `spatial` tests new places, and `strict` tests new places and new weeks together.
- **Technical:** `official` is NDWS's random-week split. `spatial` holds out location groups inferred from identical terrain. `strict` keeps patches that are in the held-out set under both definitions, so test patches share neither weeks nor terrain groups with training.
- **Maths:** Nearby or repeated samples are not independent. Random splitting therefore underestimates the error on genuinely new data.
- **Code:** `src/aml/splits.py::derive_strict_columns`, `src/aml/splits.py::verify_manifest`.

### I2. How can you hold out locations when the data has no coordinates?

- **Simple:** Elevation barely changes between days, so two patches with the same terrain are the same place. We use that terrain as a fingerprint.
- **Technical:** Patches are linked through matching elevation fingerprints and connected into location groups. Whole groups go to one side of the split.
- **Maths:** This is finding connected components in a graph whose edges are terrain matches.
- **Limit to admit:** Only exact 8 x 8 km overlaps are detected, so nearby fires that do not overlap can still land on both sides. Two groups also hold most of the patches.
- **Code:** `src/aml/splits.py`.

### I3. What is the patch-centre leak?

- **Simple:** NDWS patches are centred on fires. A model could learn "the middle burns" without learning anything about fire.
- **Technical:** A full-patch CNN can infer its position from zero padding at the borders. That signal would not exist in a real forecast over a fixed map.
- **Maths:** P(y | position) differs strongly from P(y), so position is a predictive feature that comes from how samples were chosen.
- **Code:** `src/dl/data.py::random_crop`, `src/evaluation/compare.py::centre_prior`.

### I4. How did you test whether the tabular models use that leak?

- **Simple:** Three checks: add position as a feature, try to recover position from our features, and re-score on random windows where the centre is no longer special.
- **Technical:** Adding explicit position raises PR-AUC sharply, so the position signal exists and our features do not already contain all of it. A classifier can still recover "near the centre" from the features to some degree. On position-controlled windows, models trained on full patches perform about as well as models trained on windows.
- **Maths:** If a model depended heavily on position, then removing the relationship between position and label would reduce its lift.
- **Limit to admit:** The features still partly encode closeness to the centre, most likely through where the fire is.
- **Code:** `src/aml/leak_test.py::add_position_test`, `recover_position_test`, `crop_control_test`.

### I5. Why subsample negatives, and how do you fix the probabilities?

- **Simple:** There are too many non-fire pixels to train on, so we keep all fires and 2% of the rest. This makes the model overestimate fire, so we scale the probability back down.
- **Technical:** Trees are trained on all positives and 2% of negatives. Their scores are mapped back to the true prior. Validation and test are never subsampled.
- **Maths:** Keeping negatives at rate r multiplies the odds by 1/r, so true odds = r x sample odds, and p = r p_s / (r p_s + 1 - p_s).
- **Code:** `src/aml/train.py::prior_correct`.

### I6. Which features did you engineer and why?

- **Simple:** One pixel alone does not know what surrounds it, so we add summaries: how much fire is nearby, how far the nearest fire is, whether wind blows from fire toward this pixel, and whether it lies uphill of the fire.
- **Technical:** 32 features in six groups: raw, fire, wind, terrain, weather and vegetation. Each is computed from day-*t* inputs in the same patch only.
- **Maths:** Wind push is a dot product between the wind vector and the direction from the fire. Vapour pressure deficit measures how dry the air is.
- **Code:** `src/aml/features.py::patch_features`, `FEATURE_GROUPS`.

### I7. How does a gradient-boosted tree model work?

- **Simple:** Trees are added one at a time, and each new tree corrects the mistakes of the trees so far.
- **Technical:** LightGBM and XGBoost fit each tree to the gradients of the log loss. Size, learning rate and early stopping control overfitting. Missing values go down a learned branch.
- **Maths:** F_m(x) = F_{m-1}(x) + eta h_m(x), where h_m approximates the negative gradient of the loss.
- **Code:** `src/aml/train.py::fit_lightgbm`, `fit_xgboost`, `configs/aml.yaml`.

### I8. How did you choose U-Net settings without touching the test set?

- **Simple:** We tried one setting at a time and kept the best on a separate tuning split, then froze the settings for the final runs.
- **Technical:** A greedy one-factor search ran on the `dev` protocol, whose validation patches come from the official training set but held-out spatial groups. No reported test patch is used.
- **Maths:** A greedy search cannot capture interactions between settings, but it needs far fewer runs than a full grid.
- **Code:** `src/dl/run_experiments.py::run_sweep`, `configs/dl.yaml`.

### I9. What loss did you use, and why not Dice?

- **Simple:** Weighted cross-entropy treats each pixel separately and makes missing a fire ten times as costly. Dice scores overlap and did much worse here.
- **Technical:** Weighted BCE with pos_weight 10 had the best tuning score. BCE keeps per-pixel probabilities usable for ranking, which PR-AUC rewards.
- **Maths:** BCE = -[w y log p + (1 - y) log(1 - p)]. Soft Dice = 1 - 2 sum(p y)/(sum p + sum y). Dice rewards overlap of a thresholded shape more than the ranking of scores.
- **Limit to admit:** pos_weight changes calibration, so raw U-Net probabilities are not true probabilities. PR-AUC depends only on the ranking of scores, so it does not need calibrated probabilities.
- **Code:** `src/dl/model.py::masked_bce`, `masked_dice`, `make_loss`.

### I10. What do augmentation and random crops do?

- **Simple:** Flips and rotations show the model more variations. Crops also hide where the original centre was.
- **Technical:** One of the eight dihedral transforms is applied per batch, and wind and aspect vectors are transformed with the image. Training uses a random 32 x 32 crop of each patch.
- **Maths:** If wind blows east and you mirror east-west, wind must become west. Otherwise you create physically inconsistent examples.
- **Honest result:** Removing augmentation scored slightly higher on the tuning split, within the variation we saw between settings. We do not claim that augmentation improved the model.
- **Code:** `src/dl/data.py::apply_dihedral`, `augment`, `random_crop`.

---

## Advanced questions

### A1. Why might trees match or beat a U-Net on this data?

- **Simple:** The most useful information is close to the fire, and our hand-made features already summarise it. Patches are small and there are not many of them for a deep network.
- **Technical:** Tree models receive explicit multi-scale fire context. Coarse weather rasters offer little fine spatial detail for convolutions to exploit. Grinsztajn et al. (2022) also find that trees perform strongly on medium-sized tabular data.
- **Maths:** A CNN's advantage is translation-equivariant local pattern learning. If features already approximate those patterns, the extra hypothesis space brings more variance than reduced bias.
- **Code:** `src/aml/features.py`, `experiments/ndws/FINAL_RESULTS.md`.

### A2. What does the receptive field mean here?

- **Simple:** It is how far a network can see around a pixel when making a prediction.
- **Technical:** The dilated CNN has a 29-pixel receptive field. The U-Net's pooling gives it a much larger one.
- **Maths:** Each layer adds (k - 1) x dilation x stride product to the receptive field.
- **Code:** `src/dl/model.py::CompactCNN`, `tests/test_dl.py`.

### A3. Why late fusion by logistic regression on logits?

- **Simple:** It learns how much to trust each model, using data neither model trained on.
- **Technical:** Logistic regression over the base models' logits is fitted on half of the validation patches. The other half selects among hybrid candidates and sets the threshold. U-Net with engineered channels also competes as hybrid option B.
- **Maths:** logit(p) = w0 + w1 logit(p_tree) + w2 logit(p_unet), which is a weighted combination of log-odds. Under conditional independence, log-odds add.
- **Why not fit on train:** Base models' scores on their own training data are overconfident, so a stacker would over-trust them (Wolpert, 1992).
- **Code:** `src/hybrid/fusion.py::fit_fusion`, `split_val_patches`.

### A4. How do you know a difference between models is real?

- **Simple:** We resample the test patches many times and see whether the winner stays the same.
- **Technical:** A patch-level bootstrap with 500 resamples gives PR-AUC intervals. The paired version uses the same resample for both models. Pixels in the same patch are correlated, so we resample patches rather than pixels.
- **Maths:** For each resample compute delta = PR-AUC(A) - PR-AUC(B). The 2.5th and 97.5th percentiles give a 95% interval. P(delta <= 0) is the share of resamples where A does not beat B.
- **Limit to admit:** The bootstrap describes only the uncertainty from the choice of test patches. Training seeds are checked separately.
- **Code:** `src/aml/metrics.py::bootstrap_pr_auc`, `paired_bootstrap_delta`, `src/evaluation/compare.py::seed_stability`.

### A5. Why is the strict test set small, and how does that affect conclusions?

- **Simple:** Very few patches are outside the training weeks and also outside the training places.
- **Technical:** Strict test has 405 patches, so its intervals are wide and overlap. Claims of small differences on that split are not justified.
- **Maths:** Standard error shrinks as 1/sqrt(n), so fewer independent patches means more uncertainty.
- **Code:** `data/splits/split_audit.json`.

### A6. Are NDWS weeks chronological? Could you forecast the future?

- **Simple:** No. Weeks are randomly assigned, so training can include weeks after the test weeks. We test generalisation to other weeks, not forecasting into the future.
- **Technical:** The Kaggle release contains no dates, so a chronological split cannot be built. A random-week split with a buffer reduces temporal overlap but is not a forecast test.
- **Maths:** A chronological split would detect drift between training and future distributions. Random weeks mix those distributions.
- **Code:** `docs/DATASET_CARD.md`.

### A7. What did removing the day-*t* fire channels show?

- **Simple:** Without today's fire map the model nearly collapses. Tomorrow's fire is mostly about where fire is today.
- **Technical:** On the tuning split, removing `prev_fire` and `prev_unknown` reduced U-Net validation PR-AUC from 0.239 to 0.030, a much larger fall than removing any other group.
- **Maths:** Next-day fire is strongly autocorrelated in space and time, so the previous state carries most of the mutual information.
- **Limit to admit:** Ablations are single runs on the tuning split, so small differences should not be interpreted.
- **Code:** `configs/dl.yaml`, `experiments/ndws/dl/dev/ablations.csv`.

### A8. What can this model not claim?

- **Simple:** It cannot say why a fire spread, cannot replace fire officials, and has not been tested across future seasons.
- **Technical:** Results come from one satellite-derived dataset, patches centred on known fires and retrospective weather. MODIS labels are noisy, and a day's weather is averaged over the day of the observation. A patch is only selected because a fire exists, so the problem is easier than monitoring all land.
- **Maths:** The models estimate P(y | x) under the sampling distribution of NDWS. Changing the sampling changes P(y | x).
- **Code:** `docs/DATASET_CARD.md`, `docs/METHODOLOGY.md`.

### A9. What changed between Phase 1 and this final submission, and why?

- **Simple:** We kept the Phase 1 results but added a stricter split, a test for the position leak, the deep models and a hybrid.
- **Technical:** The graded Phase 1 materials are frozen in `phase1_submission/`. The official and spatial tabular results reproduce them to three decimals after a full re-run. New work: the strict protocol, the leak tests, the Dummy baseline, the CNN, the tuned U-Net and late fusion.
- **Maths:** The strict split removes both sources of dependence at once, so it costs training data and test size in exchange for a harder test.
- **Code:** `src/evaluation/report_assets.py::phase1_change_table`, `report/tables/phase1_changes.tex`.

---

## Where each member should go deeper

| Member | Phase 1 area | Be ready to go deep on |
|---|---|---|
| Vedant Satbhai | Data and spatial audit | I1, I2, A5, A6 |
| Pratik Agade | EDA and data quality | B1, B2, I6, A8 |
| Lakshay Saharan | Feature engineering | I4, I6, A1, A7 |
| Raghvendra Singh | Models and tuning | B6, I5, I7, I8, I9 |
| Aaryan Krishna | Evaluation and integration | B3, B5, A3, A4, A9 |

Everyone should be able to run `pytest -q`, open `configs/dl.yaml`, and explain one prediction map from the report.
