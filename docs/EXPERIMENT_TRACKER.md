# Experiment Tracker

All experiments run through `scripts/run_all.sh`. Settings are in `configs/`. Results are in the listed files. Measured values are in
`experiments/ndws/FINAL_RESULTS.md`.

| ID | Experiment | Protocol | Settings | Selection data | Results |
|---|---|---|---|---|---|
| T1 | Dummy, persistence, neighbourhood persistence | official, spatial, strict | none | none | `experiments/ndws/results/<protocol>/results.json` |
| T2 | Logistic regression, random forest | official, spatial, strict | `configs/aml.yaml`; weighted or prior-corrected | protocol validation | as T1 |
| T3 | XGBoost, LightGBM | official, spatial, strict | Validation grid and early stopping | protocol validation | as T1 |
| T4 | LightGBM feature-group ablations | spatial, strict | Grouped feature subsets | protocol validation | `results.json`, key `ablation` |
| L1 | Add position columns to LightGBM | official, spatial, strict | Best LightGBM settings | none (test) | `experiments/ndws/results/<protocol>/leak_test.json` |
| L2 | Predict centre proximity from features | official, spatial, strict | LightGBM | training | as L1 |
| L3 | Window-trained LightGBM against full-patch model | official, spatial, strict | Fixed 32x32 windows, seed 2024 | training | as L1 |
| D1 | Architecture sweep: U-Net, CNN | dev | One-factor stage | dev validation | `experiments/ndws/dl/dev/sweep.csv` |
| D2 | Loss sweep: BCE, weighted BCE 3 and 10, Dice, BCE + Dice | dev | One-factor stage | dev validation | as D1 |
| D3 | Dropout 0, 0.2, 0.4 | dev | One-factor stage | dev validation | as D1 |
| D4 | Weight decay 0, 1e-4, 1e-3 | dev | One-factor stage | dev validation | as D1 |
| D5 | Width 16, 24, 32 | dev | One-factor stage | dev validation | as D1 |
| D6 | Feature channels: base, fire, fire and wind, all | dev | Recorded, chooses the hybrid input | dev validation | as D1, and `selected.json` |
| D7 | Nine U-Net ablations | dev | `configs/dl.yaml`, key `ablations` | dev validation | `experiments/ndws/dl/dev/ablations.csv` |
| F1 | Compact CNN, three seeds | official, spatial, strict | `selected.json` | protocol validation | `experiments/ndws/dl/<protocol>/cnn/` |
| F2 | U-Net, three seeds | official, spatial, strict | `selected.json` | protocol validation | `experiments/ndws/dl/<protocol>/unet/` |
| F3 | Feature-assisted U-Net (31 channels), three seeds | official, spatial, strict | `selected.json` | protocol validation | `experiments/ndws/dl/<protocol>/unet_features/` |
| F4 | Full-patch U-Net, one seed (leak diagnostic) | official, spatial, strict | Crop off | protocol validation | `experiments/ndws/dl/<protocol>/unet_fullpatch/` |
| H1 | Late fusion candidates and hybrid selection | official, spatial, strict | `configs/hybrid.yaml` | half of validation fits, half selects | `experiments/ndws/hybrid/<protocol>/fusion.json` |
| C1 | Unified comparison, bootstrap, paired tests, maps, errors | official, spatial, strict | 500 patch resamples | none | `experiments/ndws/final/`, `experiments/ndws/FINAL_RESULTS.md` |

## Notes for reading the results

- Reported networks and hybrid inputs use the first seed (42); `seed_stability.csv` reports all three.
- D1 to D7 are single runs on the tuning split, so small differences should not be interpreted.
- The `spatial` and `strict` tabular runs include ablations. The `official` run does not, because the feature-group questions concern held-out locations.
- F4, the patch-centre prior and the position prior are diagnostics and not valid forecasters.
- F3's first attempt froze an 8 GB machine while loading 31 channels. The loader now normalises in chunks into a temporary file. On three sample cases the arrays matched the earlier in-memory path exactly, and the six runs already completed were kept.
