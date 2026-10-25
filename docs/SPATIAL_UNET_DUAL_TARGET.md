# Spatial U-Net, dual-target evaluation and the NDWS patch-centre leak

This document covers the first spatial deep-learning model in the repository (`src/dl/`) and the
evaluation that compares it with the Phase 1 tabular models on both prediction targets. Full tables:
[experiments/ndws/dl/DUAL_TARGET_RESULTS.md](../experiments/ndws/dl/DUAL_TARGET_RESULTS.md).

## 1. Two targets, kept separate

`docs/NDWS_DATA_AUDIT.md` (section 8) distinguishes two labels. Both are reported, never mixed:

| Target | Definition | Pixels scored |
|---|---|---|
| `active` | next-day active fire: `FireMask(t+1) == 1` | every pixel with a known label |
| `transition` | new fire: `PrevFireMask(t) == 0` and `FireMask(t+1) == 1` | only pixels known **not** burning at t (burning or unknown day-t pixels excluded) |

All models are trained on the `active` label. For a pixel that is not burning at day t, the predicted
P(burning at t+1) is exactly the 0 → 1 transition probability, so the same scores are evaluated on both
targets. The transition subset is chosen from the day-t mask only, never from test labels. Persistence
scores at chance on `transition` by construction (PR-AUC = prevalence), which is the point of that target.

## 2. Model

- **Inputs (13 channels, day t only):** elevation; wind as a (u, v) vector the wind blows toward;
  min / max temperature; humidity; precipitation; PDSI; NDVI; log population; ERC; day-t fire;
  day-t unknown flag. Cleaning is shared with the tabular pipeline (`clean_channels`).
- **Normalisation:** per-channel mean / std from the training split of the protocol only. Wind
  components are scaled but not centred, so they can be negated during augmentation.
- **Architecture:** compact U-Net (Ronneberger et al., 2015). Three down / up stages, base 24
  channels, dropout 0.2 at the bottleneck, 1.09 M parameters.
- **Loss:** binary cross-entropy averaged over known-label pixels (`-1` masked).
- **Augmentation:** random flips / transposes (8 symmetries) with the wind vector transformed to
  match. A unit test checks that a point downwind of the fire stays downwind under all 8 transforms.
- **Training:** AdamW (lr 1e-3, wd 1e-4), batch 64 (32 for the full-patch diagnostic), up to 25 epochs, early stopping on validation
  PR-AUC (patience 5), F1 threshold from validation, test scored once. Splits are the same `official`
  and `spatial` protocols as Phase 1 (`data/splits/ndws_split_manifest.csv`).

## 3. Finding: NDWS patches leak the target through position

NDWS patches are centred on fire events, and the centre tracks the **next-day** fire more than the
day-t fire. Measured on 6,000 training patches:

| Distance from patch centre | Day-t fire rate | Day t+1 fire rate | New-fire rate, pixels > 8 km from any day-t fire |
|---|---|---|---|
| 0–3 px | 11.7% | 28.2% | 16.7% |
| 24–46 px (edges) | 0.3% | 0.3% | 0.08% |

A convolutional network trained on full 64×64 patches can infer its distance from the zero-padded
border, and therefore learn "the centre burns tomorrow" without any physical cause. That is target
leakage: it would not be available for a real forecast region.

Two diagnostics quantify it on the test splits:

- **Patch-centre prior:** a score that is just the training-set fire rate at each distance from the
  patch centre, with no inputs at all. On the spatial split it reaches PR-AUC 0.121 (active) and
  0.111 (transition), about the level of logistic regression on real features.
- **U-Net trained on full patches:** PR-AUC 0.241 (spatial, active) vs 0.124 for the same network
  trained on random 32×32 crops, where the patch centre can fall anywhere in the input. The paired
  difference is +0.117 [+0.107, +0.126]. Probability maps show the full-patch model predicting fire
  at the patch centre where no day-t fire exists (`experiments/ndws/dl/*/probability_maps_unet.png`).

**Fix used here:** train on random 32×32 crops (as Huot et al., 2022 did), infer on full patches. Both
diagnostics are reported, clearly labelled, and must not be read as valid forecasters. The Phase 1
tabular features use only positions relative to the nearest fire, never absolute position, so the
tabular results are not affected. Any future spatial model (including Phase 2 work) must also prevent
position from reaching the network.

## 4. Results (test PR-AUC, 95% patch-bootstrap CI)

| Model | Spatial, active | Spatial, transition | Official, active | Official, transition |
|---|---|---|---|---|
| Patch-centre prior (leak probe) | 0.121 [0.115, 0.129] | 0.111 [0.105, 0.119] | 0.135 [0.129, 0.142] | 0.119 [0.112, 0.126] |
| Logistic regression | 0.117 [0.106, 0.127] | 0.078 [0.070, 0.087] | 0.230 [0.213, 0.249] | 0.123 [0.115, 0.130] |
| Random forest | 0.135 [0.123, 0.149] | 0.087 [0.078, 0.096] | 0.265 [0.248, 0.282] | 0.135 [0.126, 0.143] |
| XGBoost | **0.141** [0.128, 0.154] | **0.090** [0.081, 0.100] | **0.265** [0.248, 0.283] | **0.141** [0.131, 0.149] |
| LightGBM | 0.138 [0.127, 0.150] | 0.089 [0.081, 0.097] | 0.263 [0.246, 0.281] | 0.138 [0.128, 0.147] |
| U-Net, 32×32 crops (leak-free) | 0.124 [0.113, 0.136] | 0.070 [0.062, 0.077] | 0.246 [0.231, 0.261] | 0.136 [0.127, 0.145] |
| U-Net, full patches (leaky) | 0.241 [0.227, 0.255] | 0.196 [0.183, 0.210] | 0.357 [0.341, 0.372] | 0.263 [0.250, 0.277] |

Prevalence: spatial 0.48% (active) / 0.42% (transition); official 1.25% / 0.92%.

Paired ΔPR-AUC, leak-free U-Net minus XGBoost: spatial active −0.016 [−0.021, −0.012], spatial
transition −0.021 [−0.024, −0.017], official active −0.019 [−0.024, −0.014], official transition
−0.004 [−0.007, −0.002]. Against logistic regression the U-Net is better on three of four cells
and worse on spatial transition.

**Reading:** once position leakage is removed, this first U-Net does not beat the tuned tree
ensembles on engineered features. It is a working, leakage-safe spatial baseline, not yet an
improvement. The large gains of the full-patch model come from the leak.

## 5. Reproducing

```bash
python -m src.dl.train --protocol spatial --batch 64           # leak-free U-Net (32x32 crops), ~5 min on Apple M-series GPU
python -m src.dl.train --protocol official --batch 64
python -m src.dl.train --protocol spatial --crop 0 --batch 32  # diagnostic full-patch variant, ~12 min
python -m src.dl.train --protocol official --crop 0 --batch 32
python -m src.dl.evaluate                            # needs the tabular predictions from src/aml/train.py
python -m unittest tests.test_dl
```

The first run builds normalised tensors under `data/ndws/dl/` (gitignored, about 4 GB). Trained weights:
`models/ndws_dl/<protocol>_unet.pt`. The diagnostic full-patch weights are not committed.

## 6. Limitations and next steps

1. **One configuration, one seed.** No architecture or loss tuning was done. Seed variance is not
   measured, and early stopping on noisy validation PR-AUC stopped after 13–16 epochs.
2. **Crop size trades leak removal for context.** 32×32 crops hide the patch centre but give the
   network less surrounding context than the tabular 17 km windows. Alternatives worth testing:
   larger crops combined with a position-free architecture (no zero padding, e.g. reflect/replicate
   padding and global-context blocks), or adding engineered channels (distance to fire, wind push)
   as a hybrid model.
3. **The leak affects labels, not just models.** Evaluating on fixed, centred test patches still
   rewards any position-aware model. A centre-masked or randomly re-cropped test protocol would
   make spatial-model comparisons fully position-neutral.
4. **No temporal sequences.** NDWS has single (t, t+1) pairs, so ConvLSTM is not applicable without
   a multi-day dataset (e.g. WildfireSpreadTS).
