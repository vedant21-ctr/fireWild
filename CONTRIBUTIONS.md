# Team Contributions

Project 17: Next Day Wildfire Spread Prediction via Tabular AML and Spatial Deep Learning.

This file records what the repository history shows. Git cannot see discussion, reading, testing on other machines or
report writing done outside the repo, so each member should add that work to their row in their own commit.

## Members

| Member | Enrollment | Commits before the final-submission branch (all branches) |
|---|---|---|
| Vedant Satbhai | 2401010500 | 26, under two git identities, including pull-request merges |
| Aaryan Krishna | 2401010010 | 26 |
| Lakshay Saharan | 2401020034 | 2 |
| Raghvendra Singh | 2401010367 | 1 |
| Pratik Agade | 2401010346 | 0 |

Counts come from `git log --all`. They measure commits, not effort or understanding.

## What the history shows

**Vedant Satbhai.** Repository owner. Wrote the project architecture, contributor roadmap, execution plan and NDWS data
audit documents; built an earlier NDWS tabular pipeline with its Markdown reports and tests (since replaced by
`src/aml/`), the CLBI prototype, the dashboard and the deployment (now on `archive/clbi-prototype`); reviewed and merged
the team's pull requests.

**Aaryan Krishna.** Committed the Phase 1 NDWS pipeline, spatial split audit, report and presentation
(`feat/phase1-ndws-submission`, `phase1-ndws-pipeline`, `phase1-final-submission`), the first leakage-safe U-Net, and
the October repair work on dependencies, tests and API code. Committed all of `phase2/final-submission`.

**Lakshay Saharan.** Documented the dashboard quickstart and clarified the synthetic-benchmark evidence (`docs/`).

**Raghvendra Singh.** Extended the README with project details.

**Pratik Agade.** No commits in the repository.

## Phase 1 module allocation

The graded Phase 1 materials listed one module per member. That table was the team's division of responsibility. Git
shows `src/aml/`, the LaTeX report and the slides committed by Aaryan, so it should not be read as a record of who
committed each file.

| Member | Phase 1 module listed |
|---|---|
| Vedant Satbhai | Data access and spatial leakage audit |
| Pratik Agade | EDA and data quality |
| Lakshay Saharan | Feature engineering |
| Raghvendra Singh | Models, imbalance handling and tuning |
| Aaryan Krishna | Evaluation, interpretation and report integration |

## How the final-submission branch was produced

Aaryan directed the work on `phase2/final-submission` and made its commits. This covers the strict split, the position-leak test, the CNN,
the tuning sweep, the ablations, late fusion, the re-run of every experiment, and the final report, slides and viva
guide. Every reported number comes from a results file produced by running that code. Each member should still be able
to explain the work themselves at the viva.

## Suggested remaining tasks

No commits have been made under anyone else's name. These tasks are open for each member to take and commit themselves.

| Member | Suggested task | Files |
|---|---|---|
| Vedant Satbhai | Review and merge the pull request into `main`; follow the README from a fresh clone and fix anything that fails | `README.md`, `scripts/run_all.sh` |
| Pratik Agade | Re-run the EDA notebook, check each statistic in the dataset card against `eda_summary.json`, and correct any mismatch | `notebooks/01_ndws_eda.ipynb`, `docs/DATASET_CARD.md` |
| Lakshay Saharan | Read the feature code against the report's feature section and add a unit test for one feature family not yet tested | `src/aml/features.py`, `tests/test_aml.py` |
| Raghvendra Singh | Re-run one final U-Net seed on the strict split, and record whether the result matches the committed one | `src/dl/train.py`, `docs/EXPERIMENT_TRACKER.md` |
| Aaryan Krishna | Proofread the report and rehearse the results and limitations sections | `report/main.tex` |
| Everyone | Add non-git contributions to this file; prepare for the viva with `docs/THEORY_AND_VIVA_PREP.md`; agree who presents which slides | `CONTRIBUTIONS.md` |
