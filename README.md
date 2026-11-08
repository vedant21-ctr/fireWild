# fireWild: Next-Day Wildfire Spread Prediction

**CSAI Project 17:** *Next Day Wildfire Spread Prediction via Tabular AML and Spatial Deep Learning.*

For every 1 km pixel of a 64 x 64 km region with an active fire on day *t*, predict the probability that it is burning on
day *t*+1. The project compares engineered tabular models, spatial deep learning and a hybrid of the two, on the
[Next Day Wildfire Spread](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread) dataset.

## Results

<!-- RESULTS:START -->
| Model | Official | Spatial | Strict |
|---|---|---|---|
| Dummy (training rate) | 0.013 [0.012, 0.014] | 0.005 [0.005, 0.005] | 0.006 [0.005, 0.007] |
| Persistence | 0.107 [0.097, 0.118] | 0.037 [0.032, 0.041] | 0.027 [0.021, 0.035] |
| Logistic regression | 0.230 [0.213, 0.249] | 0.117 [0.106, 0.127] | 0.095 [0.079, 0.113] |
| Random forest | 0.265 [0.248, 0.282] | 0.135 [0.123, 0.149] | 0.120 [0.097, 0.145] |
| XGBoost | 0.265 [0.248, 0.283] | 0.141 [0.128, 0.154] | 0.115 [0.094, 0.141] |
| LightGBM | 0.263 [0.246, 0.281] | 0.138 [0.127, 0.150] | 0.124 [0.103, 0.148] |
| Compact CNN | 0.256 [0.240, 0.272] | 0.129 [0.117, 0.141] | 0.093 [0.071, 0.119] |
| U-Net | 0.257 [0.242, 0.272] | 0.130 [0.118, 0.142] | 0.101 [0.079, 0.124] |
| U-Net + engineered channels | 0.254 [0.239, 0.270] | 0.129 [0.117, 0.141] | 0.116 [0.091, 0.141] |
| Hybrid (selected on validation) | 0.259 [0.244, 0.276] | 0.140 [0.127, 0.152] | 0.115 [0.093, 0.140] |

Test PR-AUC with 95% patch-bootstrap intervals, active target, full patches. Network rows use the first of three seeds.

- No network or hybrid exceeds the best tree on any split.
- Hybrid minus XGBoost: official -0.006 [-0.010, -0.002], spatial -0.001 [-0.004, +0.002], strict -0.001 [-0.002, +0.001].
- Training the U-Net on full patches adds +0.084 (official), +0.103 (spatial), +0.089 (strict) to PR-AUC, which is the patch-centre position leak. Those runs are diagnostics only.
<!-- RESULTS:END -->

Full tables, including position-controlled scoring, the new-fire target and paired tests, are in
[experiments/ndws/FINAL_RESULTS.md](experiments/ndws/FINAL_RESULTS.md).

**The splits.** `official` is the published random-week split. `spatial` holds out locations. `strict` holds out both
locations and weeks. The strict test set is small, so its intervals are wide. See the
[dataset card](docs/DATASET_CARD.md).

**The limits.** The data has no dates or coordinates, weeks are randomly assigned, and every sample contains fire. These
results measure generalisation to held-out weeks and locations, and are not a test of forecasting future seasons.

## Deliverables

| Item | File |
|---|---|
| IEEE report | [report/main.pdf](report/main.pdf) |
| Slides | [presentation/fireWild_final.pdf](presentation/fireWild_final.pdf), [.pptx](presentation/fireWild_final.pptx) |
| Viva guide | [docs/THEORY_AND_VIVA_PREP.md](docs/THEORY_AND_VIVA_PREP.md) |
| Graded Phase 1 report and slides (frozen) | [phase1_submission/](phase1_submission/) |
| Team contributions | [CONTRIBUTIONS.md](CONTRIBUTIONS.md) |

## Documentation

| Topic | File |
|---|---|
| Requirements and where each is met | [docs/REQUIREMENTS_CHECKLIST.md](docs/REQUIREMENTS_CHECKLIST.md) |
| Dataset, splits and limits | [docs/DATASET_CARD.md](docs/DATASET_CARD.md) |
| Methodology | [docs/METHODOLOGY.md](docs/METHODOLOGY.md) |
| Architecture | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Literature review | [docs/LITERATURE_REVIEW.md](docs/LITERATURE_REVIEW.md) |
| Experiments | [docs/EXPERIMENT_TRACKER.md](docs/EXPERIMENT_TRACKER.md) |
| Development history | [docs/DEVELOPMENT_LOG.md](docs/DEVELOPMENT_LOG.md) |
| Project plan | [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) |
| Status | [PROGRESS.md](PROGRESS.md) |

## Reproducing

Requires Python 3.12. On macOS, run `brew install libomp` first. The dataset downloads through `kagglehub` without credentials.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHON=python bash scripts/run_all.sh
python -m pytest
python presentation/build_deck.py
```

`run_all.sh` skips a stage when its output exists, so an interrupted run can be resumed. The full run took about 7 hours on
an 8 GB Apple M3 laptop, and needs about 15 GB of free disk space for data and predictions. To rebuild the report, run `tectonic report/main.tex`.

## Repository layout

| Path | Contents |
|---|---|
| `configs/` | YAML settings for data, tabular models, networks and hybrids |
| `src/aml/` | Data loading, splits, features, EDA, tabular training, metrics, leak tests |
| `src/dl/` | Network data, models, training, sweep and ablations |
| `src/hybrid/` | Late fusion |
| `src/evaluation/` | Unified comparison, and generation of report tables and numbers |
| `scripts/run_all.sh` | Runs every experiment in order |
| `data/splits/` | Split manifest and leakage audit (patch data is downloaded) |
| `experiments/ndws/` | Results, figures and final tables |
| `notebooks/` | EDA and final results |
| `report/`, `presentation/` | IEEE report and slides |
| `tests/` | Unit tests for leakage controls, features, models and metrics |

Numbers in the results block above, the report and the slides are generated from the result files by
`python -m src.evaluation.report_assets`.
