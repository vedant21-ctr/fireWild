# Wildfire Intelligence — Architecture, Status & Contributor Roadmap

**Status snapshot:** 24 October 2026  
**Repository:** vedant21-ctr/fireWild  
**Default branch:** main  
**Audience:** project members, collaborators, reviewers, and new contributors.

> **Critical scope distinction:** This repository currently contains a research prototype for Candidate Line Breach Intelligence (CLBI). The wider academic project is to build and fairly compare tabular AML and spatial deep-learning models for next-day wildfire spread prediction using the Kaggle Next Day Wildfire Spread (NDWS) dataset. These are connected, but they are not the same prediction task. Never present CLBI breach probabilities as next-day wildfire spread probabilities.

## 1. Vision and research goals

The overall goal is to investigate whether environmental, weather, terrain, and remotely sensed wildfire information can support next-day spread prediction, and whether predicted spread information can subsequently help rank candidate containment-line segments for retrospective inspection.

The planned modelling work has three connected but distinct parts:

1. **AML — tabular next-day spread prediction.** Engineer leakage-aware tabular features from NDWS and train a strong baseline such as XGBoost or LightGBM.
2. **DL — spatial next-day spread prediction.** Train a compact CNN or U-Net on aligned multichannel spatial patches. Use ConvLSTM only if the selected data actually provides suitable multi-day sequences.
3. **CLBI — candidate line breach intelligence.** The existing downstream prototype estimates historical containment-line breach risk using seven tabular features, including `burn_prob`, intended to represent a spread model's predicted burn probability.

The intended research outcome is an event-aware AML-versus-DL comparison, with clear limitations and visual checks, plus a CLBI demonstration whose dependency on the spread model is explicit.

### Research safeguards

- Define the prediction timestamp and target before feature generation.
- Split by fire/event and time where applicable; do not randomly mix nearby pixels or segments across train and test.
- Fit preprocessing and class weighting on training data only.
- Exclude post-outcome perimeter/label information from model inputs.
- Report PR-AUC, recall at fixed false-positive rates, F1 and ROC-AUC; accuracy alone is misleading for rare burns.
- Preserve coordinate reference systems, raster transforms, resolution, spatial alignment, and split metadata.
- Treat retrospective results as research evidence, not live fire-ground validation or guaranteed operational guidance.

## 2. Current status

This snapshot combines repository documentation with the last reported local audit. Recheck the local checkout before relying on the status of uncommitted work.

### Implemented / available

- CLBI experiment reports covering baseline, robustness, nonlinear baseline, spatial ablation, and risk stratification.
- Standardized seven-feature Logistic Regression pipeline using StandardScaler, with serialized artifact at `models/clbi_logistic_regression.joblib` and metadata at `models/clbi_model_metadata.json`.
- CLBI inference engine, deterministic 100-segment demo payload, and candidate-line prioritization logic.
- FastAPI service with request validation, structured errors, CORS configuration, and OpenAPI documentation.
- API contract at `docs/CLBI_API_CONTRACT.md` and metric definitions at `docs/CLBI_METRICS.md`.
- The last reported local test run passed **29 tests**: 14 model tests and 15 API tests.
- Documented endpoints:
  - `GET /api/v1/health`
  - `GET /api/v1/model/info`
  - `GET /api/v1/demo`
  - `POST /api/v1/predict`
  - `POST /api/v1/predict/batch`
  - `POST /api/v1/prioritize`

### Not yet implemented or not verified as complete

- A React/Vite frontend was not present in the last local audit.
- A reproducible end-to-end AML pipeline for next-day spread prediction on NDWS has not been confirmed as implemented.
- A spatial DL training/evaluation pipeline (CNN/U-Net) and probability-map outputs have not been confirmed as implemented.
- A fair AML-versus-DL evaluation using the same event/time splits has not been confirmed as complete.
- A validated interface connecting real AML/DL outputs into CLBI has not been confirmed as complete.
- Full-stack containerization, CI automation for the whole stack, and production deployment have not been confirmed.
- Dataset lineage and provenance of every CLBI feature need a reproducibility audit before making broader generalization claims.

**Metric caution:** CLBI reports retrospective benchmark metrics; they are not live performance. Five independent fire events are a small sample for generalization claims. Some reports describe generated or simulated feature construction, so inspect the data builder, source data and experiment artifacts before extending claims or treating those results as evidence for the separate NDWS spread-prediction task.

## 3. Architecture

### Existing CLBI path

```text
Historical fireline-segment dataset
  data/final/fireline_segments_v1.parquet
                  |
                  v
     Feature / dataset preparation
        src/data/build_dataset.py
                  |
                  v
 Research evaluation and training
    src/evaluation/ + experiments/
                  |
                  v
 StandardScaler + Logistic Regression
    src/model/train_clbi_model.py
                  |
                  v
 Serialized model + provenance metadata
 models/clbi_logistic_regression.joblib
 models/clbi_model_metadata.json
                  |
                  v
 CLBIDecisionEngine / inference
       src/model/clbi_model.py
                  |
                  v
 FastAPI routes + Pydantic schemas
 src/api/main.py, routes.py, schemas.py
                  |
                  v
 REST API / OpenAPI docs / demo payload
                  |
                  v
 Planned frontend dashboard
 (not present in the last local audit)
```

CLBI uses seven inputs: `slope`, `elevation`, `distance_to_fire`, `barrier_width_m`, `burn_prob`, `attack_angle`, and `attack_dot_product`. Outputs include breach probability, hold probability, risk tier and—when a batch/population is supplied—vulnerability ranking and prioritization. A single-segment prediction cannot produce a meaningful population rank, so null ranking fields are expected.

### Target academic spread-prediction path

```text
Kaggle Next Day Wildfire Spread dataset
                  |
                  v
 Dataset inventory + provenance + CRS audit
                  |
                  v
 Fire/event and temporal split manifest
                  |
          +-------+--------+
          |                |
          v                v
   AML tabular path   Spatial DL path
   raster-to-table    aligned image patches
   feature pipeline   multichannel tensors
   XGBoost/LightGBM   CNN or U-Net
          |                |
          v                v
   Probability output   Probability map output
          |                |
          +-------+--------+
                  |
                  v
 Shared event-held-out evaluation
 PR-AUC, ROC-AUC, F1, recall at fixed FPR,
 calibration where appropriate, visual QA
                  |
                  v
 Comparative report + error analysis
                  |
                  v
 Optional documented integration into CLBI
 via the validated burn-probability input
```

AML and DL need compatible target definitions and comparable held-out events. Spatial patches must not cross split boundaries. Document the input channels, label horizon, selected region/time range and split manifest before training.

## 4. Suggested repository boundaries

This is a target organization, not a claim that every directory already exists. Adapt to the actual tree; do not move files just to match this illustration.

```text
docs/
  PROJECT_ARCHITECTURE_AND_ROADMAP.md
  CLBI_API_CONTRACT.md
  CLBI_METRICS.md
data/
  raw/          # downloaded source data; normally gitignored
  interim/      # derived intermediate data
  processed/    # reproducible documented subsets
  final/        # current CLBI benchmark data
src/
  data/         # dataset inventory, validation, preparation
  features/     # feature engineering and QA
  aml/          # planned tabular spread modelling
  dl/           # planned spatial modelling and data loaders
  evaluation/   # shared metrics, splits and reports
  model/        # existing CLBI engine and training code
  api/          # existing FastAPI service
frontend/       # planned React/Vite dashboard
tests/
experiments/
  results/
models/
```

Do not commit large raw datasets, secrets, credentials, private incident data or unapproved derived datasets. Follow the existing .gitignore, and document source URLs, licenses, checksums, download steps and expected directory layout.

## 5. Roadmap and acceptance criteria

### Milestone 0 — Shared ground truth and reproducibility
**Priority: immediate**

- [ ] Confirm current local branch, working tree, actual files, dependencies and test results.
- [ ] Inventory the NDWS dataset: source, license, regions/dates, raster channels, labels, resolution, CRS and size.
- [ ] Audit CLBI data construction and provenance; label measured, generated, simulated and inferred quantities.
- [ ] Decide the canonical target, prediction timestamp, region/time subset and event-level split manifest.
- [ ] Ensure a clean-clone setup guide is reproducible.

**Done when:** a new contributor can set up the environment, locate the data, run CLBI tests and distinguish real, simulated and planned outputs.

### Milestone 1 — Frontend shell and CLBI dashboard
**Next product milestone**

- [ ] Build a responsive React/Vite frontend in `frontend/`.
- [ ] Add Overview, AML Spread Prediction, DL Spatial Prediction and CLBI Line Breach Intelligence sections.
- [ ] Connect the CLBI view to the existing API and implement loading, error and empty states.
- [ ] Show demo risk counts, prioritization, model provenance and explanations from API responses.
- [ ] Keep a persistent retrospective-decision-support disclaimer.
- [ ] Mark AML/DL screens as research-in-development until real outputs exist.
- [ ] Do not claim a segment strip/list is a geographic map; no accurate map without georeferenced coordinates.
- [ ] Verify production build and API integration.

**Done when:** frontend and backend run locally together, CLBI displays real API data, and no AML/DL results are fabricated.

### Milestone 2 — NDWS data and AML baseline

- [ ] Implement data inventory, validation and reproducible subset selection.
- [ ] Create an event-level split manifest and document it.
- [ ] Define next-day target and timestamp/label alignment.
- [ ] Engineer supported lagged burn activity, vegetation, weather, drought, terrain and neighborhood features.
- [ ] Train a baseline and XGBoost or LightGBM.
- [ ] Save configuration, seed, data version, split IDs, preprocessing and model artifacts.
- [ ] Generate PR curves, recall at fixed FPR, calibration checks where appropriate and spatial error maps.
- [ ] Add tests for leakage-sensitive features and split isolation.

**Done when:** training and evaluation are reproducible from documented commands and metrics point to the exact split manifest.

### Milestone 3 — Spatial DL baseline

- [ ] Build aligned multichannel patches and target masks/labels.
- [ ] Verify CRS, transform, resolution, channel order, nodata and normalization.
- [ ] Ensure patches cannot cross train/validation/test event boundaries.
- [ ] Train a compact CNN or U-Net; use ConvLSTM only if genuine multi-day sequences justify it.
- [ ] Apply class weighting/sampling to training data only and report class prevalence.
- [ ] Produce georeferenced probability maps and physically plausible visual QA.
- [ ] Save reproducible configurations, checkpoints, metrics and inference scripts.

**Done when:** model training/inference is reproducible, spatial metadata is preserved, and evaluation uses the same event-level protocol as AML.

### Milestone 4 — Fair comparison

- [ ] Evaluate AML and DL on the same held-out fires and compatible labels/horizons.
- [ ] Report PR-AUC, ROC-AUC, F1, recall at predeclared false-positive rates and calibration where appropriate.
- [ ] Include per-event results, class prevalence, uncertainty, limitations and error analysis.
- [ ] Compare compute cost, inference time, failure cases and map plausibility.
- [ ] Do not choose a winner from accuracy alone or compare different test sets.
- [ ] Publish a report and the artifacts needed to reproduce it.

**Done when:** each result is traceable to data version, code revision, configuration, seed and split manifest.

### Milestone 5 — Integrate spread output into CLBI

- [ ] Specify a versioned schema for spread probability, timestamp, spatial reference and model provenance.
- [ ] Define how probability is sampled or aggregated onto candidate line segments.
- [ ] Document missing data, CRS alignment and timestamp handling.
- [ ] Validate the semantics of CLBI's `burn_prob` input before passing model outputs.
- [ ] Run event-held-out ablations with and without the spread-derived feature.
- [ ] Keep spread probability and downstream breach probability separately named.

**Done when:** the end-to-end path is reproducible and the source model for each probability is explicit. Never connect placeholder probabilities.

### Milestone 6 — Packaging and handover

- [ ] Add CI for Python tests and frontend build.
- [ ] Add Docker/Compose after local instructions are stable.
- [ ] Add a clean-clone smoke test and documented configuration examples.
- [ ] Document deployment limitations, model/data versioning and rollback.
- [ ] Prepare demo scenarios and final handover.

**Done when:** a new member can run tests, launch the stack, reproduce a prediction and explain its limitations.

## 6. Contributor workstreams

Assign named owners in the team's issue tracker. These are suggested responsibilities, not assignments.

| Workstream | Responsibilities | Deliverables |
|---|---|---|
| Coordination / reproducibility | Milestones, issues, decisions and experiment registry | Issues, split manifest, decision log |
| Data engineering / GIS | NDWS inventory, raster alignment, CRS, patches, leakage checks | Validated pipeline, data dictionary, QA report |
| AML | Tabular features, baselines, XGBoost/LightGBM, threshold metrics | Training scripts, model, PR curves |
| Spatial DL | Patch loader, CNN/U-Net, maps and spatial QA | Reproducible training, checkpoint, maps |
| Backend / CLBI | Preserve API contract; validate feature semantics and later integration | Tested API, versioned schema, integration tests |
| Frontend / UX | Dashboard shell, CLBI integration, future AML/DL result views | Responsive UI, API states, screenshots |
| Validation / documentation | Independent reproduction, experiment review and limitations | Reproduction checklist, review notes, final report |

A small team can combine workstreams. Each issue should have a clear scope, acceptance criteria and owner before implementation starts.

## 7. Contributor workflow

1. Read this roadmap, root README, relevant API contract and issue before editing.
2. Sync with current `main` and inspect `git status`. Never overwrite another contributor's uncommitted work.
3. Create a focused branch, e.g. `docs/ndws-data-audit`, `feat/aml-baseline`, `feat/dl-patch-loader` or `feat/clbi-dashboard`.
4. Keep PRs focused; avoid unrelated formatting changes or unexplained model-artifact changes.
5. Add tests for behavior changes and update docs/schemas when interfaces change.
6. Record dataset version, split manifest, seed, configuration and exact commands for experiments.
7. Run relevant checks before review. Existing Python test command:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

8. Open a PR against `main` with motivation, approach, tests, metrics/screenshots, limitations and follow-ups.
9. Require review for changes to methodology, evaluation splits, model artifacts and API contracts.
10. Never commit credentials, API keys, local absolute paths, large raw datasets or generated caches.

### Pull request checklist

- [ ] Scope and acceptance criteria are clear.
- [ ] No unrelated files or local secrets are included.
- [ ] Tests/reproduction commands are recorded.
- [ ] Data split and leakage implications are explained for model changes.
- [ ] Metrics name the correct task, dataset and held-out events.
- [ ] Docs and API contracts are updated where needed.
- [ ] Retrospective results are not described as live validation.
- [ ] Limitations and follow-up work are stated.

## 8. Local development

### Existing CLBI backend

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation. In another terminal:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### Planned frontend

After the frontend scaffold and package scripts exist:

```bash
cd frontend
npm install
npm run dev
```

These frontend commands are planned and will not work until the frontend exists. The API currently allows local development origins including port 5173.

## 9. Decisions to record before model implementation

1. Exact NDWS version, license, source URL and chosen subset.
2. Definition of “next day”: input timestamp, target timestamp, time zone and lead time.
3. Burn/spread target and pixel/patch resolution.
4. Event-level/temporal split strategy and immutable split manifest.
5. AML feature definitions, missing-data treatment, class imbalance strategy and baseline.
6. DL channels, normalization, patch size/stride, architecture and output geospatial format.
7. Shared evaluation metrics and thresholds selected before final test evaluation.
8. Meaning and provenance of CLBI `burn_prob` and which model/version may supply it.
9. Which results come from real source data versus generated/simulated data.
10. Team owners, review requirements and demo acceptance criteria.

## 10. Key risks and guardrails

- **Task confusion:** CLBI predicts breach/holding risk for line segments; AML/DL predicts next-day spread. Keep labels, metrics, API fields and UI wording distinct.
- **Dataset provenance:** audit builders and source files. A filename or README statement alone does not prove a field came from real NDWS observations.
- **Limited event count:** five fire complexes are few independent events; report per-fire results and avoid broad claims.
- **Spatial leakage:** random splits of neighboring patches/segments can inflate results. Split by event/time and keep patch generation split-safe.
- **Probability misuse:** probability is not certainty or causality. Validate calibration and communicate uncertainty.
- **No live validation:** current CLBI is retrospective and not validated for active incident operations.
- **Feature contract drift:** changes to the seven CLBI inputs require schema, tests, metadata and docs updates.

## 11. Immediate next actions

1. **Coordinator:** confirm this snapshot against the local checkout and assign owners for data audit, AML, DL, frontend and CLBI.
2. **Data/GIS owner:** publish an NDWS inventory and provenance/leakage audit before modelling.
3. **Frontend owner:** implement the CLBI dashboard against the existing API; keep AML/DL views marked unavailable until real outputs exist.
4. **AML owner:** implement the first reproducible tabular baseline using a documented event/time split.
5. **DL owner:** inspect spatial channels and prepare a split-safe patch loader before selecting a larger architecture.
6. **Validation owner:** define shared evaluation metrics and a baseline reporting template.
7. **All contributors:** use focused issues, branches and PRs; update this roadmap when milestone status materially changes.

---

**Status policy:** This is a living plan. Update checkboxes only when relevant code, tests, artifacts or reports exist and have been reviewed. Keep dated experiment results separate from future plans, and do not mark a milestone complete based only on a proposed design.
