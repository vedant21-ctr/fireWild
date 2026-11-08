# Requirements Checklist

Maps each CSAI requirement for Project 17 to where it is met, and its status.

**Status:** Done = implemented and checked. Partial = done with a stated limit. Team = needs a team member.

## Phase 1: Machine Learning

| Requirement | Evidence | Status |
|---|---|---|
| Literature review | `docs/LITERATURE_REVIEW.md`, report Section IV, `report/references.bib` | Done |
| Dataset quality and EDA | `src/aml/eda.py`, `notebooks/01_ndws_eda.ipynb`, `experiments/ndws/eda/`, `docs/DATASET_CARD.md` | Done |
| Feature engineering | `src/aml/features.py`, `docs/METHODOLOGY.md` Section 4, report Section VIII | Done |
| Theoretical rigor | Report Sections IX and XIII, `docs/THEORY_AND_VIVA_PREP.md` | Done |
| Model application | `src/aml/train.py`, `configs/aml.yaml`, `experiments/ndws/results/` | Done |
| GitHub repository and code quality | Modular `src/`, YAML configs, `tests/` passing, Ruff passing at line length 150 | Done |
| Project presentation | `presentation/fireWild_phase1.pdf` and `.pptx` on branch `phase1-final-submission` (frozen) | Done |
| Viva voce | `docs/THEORY_AND_VIVA_PREP.md` | Team |
| Report in LaTeX/IEEE format | `report/main.pdf` on branch `phase1-final-submission` (frozen), and the final `report/main.pdf` | Done |

## Phase 2: Deep Learning

| Requirement | Evidence | Status |
|---|---|---|
| Architecture logic | `src/dl/model.py`, `docs/ARCHITECTURE.md`, report Section X | Done |
| Hybrid innovation | `src/hybrid/fusion.py`, `configs/hybrid.yaml`, report Section XI | Done; the hybrid did not beat the best tree, and this is reported |
| Ablation studies | `experiments/ndws/dl/dev/ablations.csv`, tabular ablations in `results.json`, report Section XV | Done |
| Deep learning literature review | `docs/LITERATURE_REVIEW.md` Sections 5 and 6 | Done |
| DL dataset and regularisation | `src/dl/data.py`, sweep and ablations for dropout, weight decay and augmentation | Done |
| Technical validation | Three protocols, bootstrap intervals, paired tests, three seeds, position-leak tests, `tests/` | Done |
| Theoretical rigor | Report Sections IX to XIII, viva guide | Done |
| GitHub repository and code quality | As above; `scripts/run_all.sh` reruns all experiments | Done |
| Project presentation | `presentation/fireWild_final.pdf` and `.pptx` | Done |
| Viva voce | `docs/THEORY_AND_VIVA_PREP.md` | Team |
| Report in LaTeX/IEEE format | `report/main.tex`, `report/main.pdf` | Done |

## Project brief and development rules

| Requirement | Evidence | Status |
|---|---|---|
| Tabular baselines including XGBoost and LightGBM | Dummy, persistence, neighbourhood persistence, logistic regression, random forest, XGBoost, LightGBM | Done |
| CNN and U-Net | `CompactCNN` and `UNet` in `src/dl/model.py` | Done |
| PR-AUC, ROC-AUC, precision, recall, F1, recall at fixed FPR, confusion matrices | `src/aml/metrics.py`, `experiments/ndws/final/metrics.csv`, `confusion_matrices.png` | Done |
| Final results table across all models | `experiments/ndws/FINAL_RESULTS.md`, report Tables on the three protocols, `README.md` | Done |
| Prediction maps: actual, AML, CNN, U-Net, hybrid, false positives, false negatives | `experiments/ndws/final/<protocol>/prediction_maps.png` | Done |
| Leakage prevention | `docs/METHODOLOGY.md` Section 2, `src/aml/splits.py::verify_manifest`, `tests/` | Done |
| Spatial and temporal separation of events | `spatial` and `strict` protocols | Partial: no event IDs or dates exist, so terrain groups and random weeks stand in for them |
| Geospatial maps with coordinates | Patch-grid maps only | Partial: NDWS on Kaggle has no coordinates |
| ConvLSTM or temporal models | Not attempted | Not applicable: each NDWS sample has only one input day, and the project decided to stay on NDWS |
| Fixed seeds and configuration files | `configs/`, seeds in YAML, three final seeds | Done |
| Tests and linting | `pytest` and `ruff check .` pass | Done |
| Development and experiment records | `docs/DEVELOPMENT_LOG.md`, `docs/EXPERIMENT_TRACKER.md`, `PROGRESS.md` | Done |
| Individual contributions documented | `CONTRIBUTIONS.md` from repository history | Team: each member adds non-git work |
| What changed since Phase 1 | Report Section XII-D and Table III, slide "What changed, and why" | Done |
