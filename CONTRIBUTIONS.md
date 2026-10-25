# Team Contributions

Project 17: Next Day Wildfire Spread Prediction. Phase 1 (Machine Learning).

The viva is individual, so every member must be able to explain the whole pipeline (see
`docs/THEORY_AND_VIVA_PREP.md`). Each member also owns one module end to end: they commit its code,
present its slides, and write its report section.

| Member | Enrollment | Owned module | Code and artefacts | Report section | Slides |
|---|---|---|---|---|---|
| Vedant Satbhai | 2401010500 | Data access and spatial leakage audit | `src/aml/ndws.py`, `src/aml/splits.py`, `data/splits/`, `DATASET_CARD.md` | III-A Dataset; IV Spatial leakage audit and splits | Dataset; leakage audit |
| Pratik Agade | 2401010346 | EDA and data quality | `src/aml/eda.py`, `notebooks/01_ndws_eda.ipynb`, `experiments/ndws/eda/` | III-B to III-D Class balance, data quality, orientation test | EDA findings |
| Lakshay Saharan | 2401020034 | Feature engineering | `src/aml/features.py`, `tests/test_aml.py` | V Feature engineering | Features and physics |
| Raghvendra Singh | 2401010367 | Models, imbalance handling and tuning | `src/aml/dataset.py`, `src/aml/train.py`, `models/ndws/` | VI Models and training (theory, prior correction, model selection) | Models and training |
| Aaryan Krishna | 2401010010 | Evaluation, interpretation and report integration | `src/aml/metrics.py`, `src/aml/evaluate.py`, `notebooks/02_ndws_results.ipynb`, `report/` | VII Evaluation protocol; VIII Results; IX Discussion | Results, conclusions, Phase 2 plan |

## Shared work

- **Literature review** (`docs/LITERATURE_REVIEW.md`): each member summarised the papers for their module:
  - Vedant: Huot et al. 2022, Roberts et al. 2017
  - Pratik: Jain et al. 2020, Sayad et al. 2019
  - Lakshay: Rothermel 1972, FARSITE, GRIDMET
  - Raghvendra: Breiman 2001, Friedman 2001, XGBoost, LightGBM, Saerens et al. 2002
  - Aaryan: Saito & Rehmsmeier 2015, SHAP, Niculescu-Mizil & Caruana 2005
- **IEEE report** (`report/main.tex`): Aaryan integrated it; each member wrote their section.
- **Presentation**: each member presents the slides for their module.

## Separate from Phase 1

Vedant Satbhai built the CLBI prototype (`src/model`, `src/api`, `src/evaluation`, synthetic data in
`data/final`) before the Phase 1 pipeline. It is documented as a synthetic-data software prototype and is not
part of the Phase 1 evaluation.
