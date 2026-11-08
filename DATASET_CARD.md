# Dataset Card: Next Day Wildfire Spread (NDWS)

**Source:** Kaggle `fantineh/next-day-wildfire-spread` (version 2), released with Huot, Hu, Goyal, Sankar,
Ihme & Chen, "Next Day Wildfire Spread: A Machine Learning Dataset to Predict Wildfire Spreading From
Remote-Sensing Data", *IEEE TGRS* 60, 2022 (arXiv:2112.02447). Built with Google Earth Engine.
**Download:** `python -m src.aml.ndws` (kagglehub). Raw TFRecords are about 2 GB; the NumPy cache in
`data/ndws/` (gitignored) is about 4.3 GB.
**Task:** per-pixel binary classification. Is a 1 km² pixel burning at day *t*+1, given day-*t* inputs?

## Composition

| Split (official) | Patches | Pixels | Positive rate (known labels) | Unknown labels |
|---|---:|---:|---:|---:|
| train | 14,979 | 61.4 M | 1.10% | 2.35% |
| eval | 1,877 | 7.7 M | 1.37% | 2.44% |
| test | 1,689 | 6.9 M | 1.25% | 2.75% |

Each patch is 64 × 64 pixels at 1 km, covering the contiguous US from 2012 to 2020. A sample exists only if
the day-*t* mask contains fire.

| Channel | Meaning | Source | Units |
|---|---|---|---|
| `elevation` | terrain height | SRTM | m |
| `th` | wind direction (blowing **from**) | GRIDMET | degrees from north |
| `vs` | wind speed | GRIDMET | m/s |
| `tmmn`, `tmmx` | daily min / max temperature | GRIDMET | K |
| `sph` | specific humidity | GRIDMET | kg/kg |
| `pr` | precipitation | GRIDMET | mm |
| `pdsi` | Palmer Drought Severity Index | GRIDMET | index |
| `NDVI` | vegetation index | VIIRS | ×10⁴ |
| `population` | population density | GPWv4 | people/km² |
| `erc` | energy release component | NFDRS | index |
| `PrevFireMask` | fire at day *t* (input) | MODIS MOD14A1 | 1 / 0 / −1 unknown |
| `FireMask` | fire at day *t*+1 (**target**) | MODIS MOD14A1 | 1 / 0 / −1 unknown |

## What is predicted, and when

| Question | Answer |
|---|---|
| Prediction time | End of day *t*. Weather is the average over day *t* (Huot et al., Section II-C), so a forecast can only be issued once day *t* is complete |
| Inputs | Day-*t* rasters only. Slowly changing layers (vegetation, drought, population) use the latest value available at *t* |
| Target | MODIS MOD14A1 fire mask on day *t*+1, per 1 km pixel |
| Observation window | One day. Each sample holds only *t* and *t*+1, so there are no multi-day sequences |
| Spatial prediction | Yes, every pixel has a label, so segmentation models apply |
| Two evaluated targets | **active** (`FireMask` = 1, includes fire that persists) and **transition** (new fire: scored only on pixels known not burning at *t*) |

## Evaluation protocols

Huot et al. assign **random weeks** to train / eval / test (8:1:1) with a one-day buffer. That separates periods but
not places. Our terrain-fingerprint audit (`src/aml/splits.py`) finds that **about 95% of official test patches share an
identical, non-flat 8 x 8 km elevation block with a training patch**; a mirrored-patch control gives 0 / 1,855 matches.

| Protocol | Weeks kept apart? | Terrain kept apart? | Train / val / test patches |
|---|---|---|---|
| `official` | Yes | No | 14,979 / 1,877 / 1,689 |
| `spatial` | No | Yes | 14,822 / 1,855 / 1,868 |
| `strict` | Yes (held-out patches come only from official eval/test weeks) | Yes | 12,050 / 389 / 405 |
| `dev` (tuning only) | Not reported | Yes | 12,050 / 1,466 / none |

`dev` selects deep-learning hyperparameters. Its patches are all from official training weeks and never from a
reported test split. Manifest: `data/splits/ndws_split_manifest.csv`. Audit: `data/splits/split_audit.json`. Both are
checked by `verify_manifest()` every time the manifest is built, and again by the unit tests.

## What the dataset cannot support

| Requirement in the project brief | Status | What we do instead |
|---|---|---|
| Coordinates and CRS metadata | The Kaggle files contain none | Maps use patch-grid coordinates; orientation is inferred from wind physics |
| Evaluation per named fire event | No event identifiers | Terrain-derived location groups act as a stand-in for events, and are described as such |
| Strict chronological order | No dates; weeks were randomly assigned | Weeks are kept apart. We do not claim to predict the future from the past |
| Lagged burn history longer than one day | Only the day-*t* mask exists | Day-*t* burn features at several spatial scales |
| ConvLSTM | No multi-day sequences | Not attempted |

## Data-quality issues found (EDA) and handling

| Issue | Handling |
|---|---|
| `tmmn`, `tmmx` outside 200–330 K (missing stored as 0 K; maxima up to 344 K ≈ 71 °C), about 0.35% of pixels | NaN |
| `sph` ≤ 0, `vs` < 0 | NaN |
| `th` (wind direction) corrupted: values from −496,522 to 37,620 in about 0.2% of pixels | outside [0, 361] → NaN; (360, 361] wrapped |
| elevation below 0 m (minimum −5 m) | kept: sub-sea-level terrain is real |
| `pr` slightly negative | clip at 0 |
| `NDVI` ×10⁴ scale; heavy-tailed `population` | rescale; log1p |
| label −1 (cloud / missing), about 2.4% | excluded from training and evaluation |
| no georeferencing; orientation unknown | rows run south→north, columns west→east: the only convention with significantly more spread downwind than upwind (1.12×, 95% CI 1.09–1.15; EDA) |

## Known limitations

- MODIS daily 1 km active-fire detections are noisy (clouds, overpass timing, smoke). About 74% of
  day-*t*+1 fire pixels were not burning on day *t*, so persistence recall is only about 26%.
- Weather is daily-aggregated GRIDMET (about 4 km) interpolated to 1 km, so sub-daily wind shifts are
  lost.
- There is no fuel type, fuel moisture or suppression information.
- Patches are centred on fires, so the pixel distribution is not the distribution of the US landscape, and pixel
  position inside a patch predicts the label. See the patch-centre leak in `docs/METHODOLOGY.md`.
- The two largest location groups hold 71% of patches and always stay in training, so location-disjoint test sets
  contain smaller, more isolated fires than the training data.

## License and citation

Use under the terms on the Kaggle dataset page. Cite Huot et al. (2022) when using the data.
