# Project Plan

**Project 17: Next Day Wildfire Spread Prediction via Tabular AML and Spatial Deep Learning.**
Phase 1 (ML) was graded and is frozen; its submission is kept in `phase1_submission/`. This plan covers the final submission, which includes Phase 2 (DL).

## Research question

Can spatial deep learning capture local wildfire patterns that engineered tabular models miss, when leakage is
controlled in space, in time and through patch position?

## Milestones

| # | Milestone | Main output |
|---|---|---|
| 0 | Consolidate the repository | CLBI archived, one branch, configs, this plan |
| 1-3 | Data, EDA, validated splits | `src/aml/ndws.py`, `eda.py`, `splits.py`, `docs/DATASET_CARD.md` |
| 4-5 | Features and tabular models | `features.py`, `train.py`, `evaluate.py` |
| 6 | Phase 1 fixes for the final report | `strict` split and `leak_test.py` |
| 7-10 | Spatial models and tuning | `src/dl/`, `experiments/ndws/dl/dev/` |
| 11 | Hybrid models | `src/hybrid/fusion.py`, feature-assisted U-Net |
| 12 | Ablations | Tabular, regularisation, channel and hybrid ablations |
| 13-14 | Unified comparison and maps | `src/evaluation/compare.py`, `experiments/ndws/final/` |
| 15-18 | Literature, report, slides, viva guide | `docs/`, `report/`, `presentation/` |
| 19 | Clean-up and review | Pull request into `main`, reviewed by the repository owner |

Progress is tracked in `PROGRESS.md`. Each requirement is mapped to evidence in `docs/REQUIREMENTS_CHECKLIST.md`.

## Main risks

| Risk | Response |
|---|---|
| Spatial and temporal leakage | Three protocols with programmatic checks |
| Position leak through patch centring | Crop training, leak probes, position-controlled scoring |
| Small `strict` test set | Patch-bootstrap intervals are reported and interpreted |
| Tuning leaking into test results | Hyperparameters chosen only on `dev` or a protocol's validation split |
| 8 GB of RAM | Streaming tabular sampling, float16 caches, sequential experiments |
