
# Candidate Line Breach Intelligence (CLBI)

**Predictive Wildfire Containment Line Holding & Breach Intelligence**

CLBI is a machine-learning-based decision-support system designed to help wildfire managers assess candidate containment line segments and prioritize inspection resources based on historical patterns associated with containment breach risk.

The system combines terrain characteristics, fire proximity, barrier dimensions, and predicted next-day wildfire spread to estimate breach probabilities for individual 100-meter candidate fireline segments.

> **Disclaimer:** CLBI is a research prototype evaluated on retrospective historical wildfire data. Its predictions are not a substitute for professional wildfire assessment, real-time field intelligence, or incident command decisions.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Research Question](#research-question)
- [System Architecture](#system-architecture)
- [Input Features](#input-features)
- [Output Schema](#output-schema)
- [Benchmark Results](#benchmark-results)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [REST API](#rest-api)
- [Testing](#testing)
- [Research Methodology](#research-methodology)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

Wildfire containment planning requires evaluating potential firelines under changing terrain, weather, and fire-spread conditions. Inspecting every candidate segment with equal priority can make resource allocation difficult.

CLBI addresses this problem by estimating breach risk at the individual segment level and ranking candidate segments for further inspection.

The system provides:

- Segment-level breach probability estimates.
- Corresponding containment-holding probability estimates.
- Four operational risk categories.
- Ranked inspection priorities.
- Structured risk and protective factor summaries.
- REST API endpoints for model inference and candidate-line prioritization.
- Reproducible research evaluation and automated tests.

CLBI is designed as a decision-support tool rather than an autonomous fireline selection or deployment system.

## Key Features

- **Predictive risk scoring:** Estimates breach probability using seven input features.
- **Risk stratification:** Categorizes segments into LOW, MEDIUM, HIGH, and CRITICAL risk tiers.
- **Inspection prioritization:** Ranks candidate segments to support targeted inspection.
- **Model explainability:** Provides structured feature-attribution summaries.
- **Fast inference:** Loads a pre-trained Logistic Regression artifact for production predictions.
- **REST API:** Exposes model functionality through FastAPI endpoints.
- **Research reproducibility:** Separates held-out wildfire evaluation from production model training.
- **Automated testing:** Includes unit tests for the inference engine and API.

## Research Question

**Can predicted next-day wildfire spread provide useful directional information for identifying vulnerable segments of a proposed containment line?**

The project investigates whether predicted burn probability, combined with terrain, barrier geometry, and fireline interaction features, can help distinguish segments with historically observed breach outcomes.

The research focuses on predictive performance, probability calibration, generalization across wildfire incidents, and inspection prioritization.

## System Architecture

```text
Historical Wildfire Data
          |
          v
Dataset Preparation and Feature Validation
          |
          v
Feature Engineering and Quality Checks
          |
          v
Leave-One-Fire-Out Evaluation
          |
          v
Production Model Training
          |
          v
Serialized Logistic Regression Model
          |
          v
CLBI Decision Engine
          |
          v
Breach Probability and Risk Classification
          |
          v
Candidate-Line Inspection Prioritization
          |
          v
FastAPI REST Interface
```

### Research and Production Separation

**Research Evaluation**
- Uses five-fold Leave-One-Fire-Out (LOFO) cross-validation.
- Holds out wildfire incidents during evaluation to assess generalization across fires.
- Generates benchmark metrics and evaluation reports.

**Production Inference**
- Trains a final model using the designated historical training data.
- Serializes the trained model to `models/clbi_logistic_regression.joblib`.
- Loads the saved artifact through `CLBIDecisionEngine`.
- Performs predictions without retraining during ordinary inference requests.

## Input Features

The model uses seven input features.

| Feature | Description | Unit |
|---|---|---|
| `slope` | Terrain slope angle | Degrees |
| `elevation` | Surface elevation | Meters |
| `distance_to_fire` | Distance to the active flame front at time *t* | Meters |
| `barrier_width_m` | Physical cleared width of the proposed barrier | Meters |
| `burn_prob` | Predicted next-day burn probability | 0–1 |
| `attack_angle` | Acute interaction angle between line normal and spread heading | Degrees |
| `attack_dot_product` | Head-fire attack alignment ratio, represented by cosine of the attack angle | 0–1 under the stated acute-angle convention |

Prediction quality depends on the validity, consistency, and availability of these input features.

## Output Schema

For each candidate fireline segment, CLBI is designed to provide the following outputs.

| Output | Description |
|---|---|
| `breach_probability` | Estimated probability of a breach, between 0 and 1 |
| `hold_probability` | Complement of breach probability |
| `risk_tier` | Operational risk category |
| `vulnerability_rank` | Ordinal rank, where 1 represents the highest predicted risk |
| `priority_percentile` | Relative risk percentile among evaluated candidate segments |
| `priority_recommended` | Indicates whether a segment belongs to the top 10% inspection-priority set |
| `top_risk_factors` | Structured summary of features associated with increased predicted risk |
| `protective_factors` | Structured summary of features associated with decreased predicted risk |
| `explanation` | Non-causal summary of the model's prediction |

### Risk Classification

| Risk Tier | Breach Probability |
|---|---|
| LOW | Below 0.30 |
| MEDIUM | 0.30 to below 0.60 |
| HIGH | 0.60 to below 0.80 |
| CRITICAL | 0.80 or above |

These categories are configurable decision-support thresholds, not universal physical definitions of wildfire containment failure.

## Benchmark Results

The reported benchmark uses five-fold Leave-One-Fire-Out evaluation across 4,600 held-out segments from five historical California wildfire incidents.

| Metric | Reported Result |
|---|---:|
| Mean LOFO PR-AUC | 0.9508 ± 0.0076 |
| Global Brier Score | 0.1199 |
| ECE — 10 equal-width bins | 0.0109 |
| ECE — 20 equal-width bins | 0.0151 |
| Mean top-10% inspection precision | 89.6% |
| Top-10% precision across fires | 84.0%–95.8% |
| Mean top-20% breach recall | 53.8% |
| Top-20% recall across fires | 48.8%–57.2% |
| Offline breach-yield enrichment | 2.92× versus random selection |

*These are reported retrospective benchmark results, not independently verified live operational performance.*

### Metric Definitions

- **PR-AUC:** Measures precision-recall performance across classification thresholds.
- **Brier Score:** Measures the mean squared difference between predicted probabilities and binary outcomes; lower is better.
- **Expected Calibration Error (ECE):** Summarizes the difference between predicted confidence and observed event frequency across probability bins.
- **Inspection precision:** Measures the proportion of selected inspection segments associated with the target breach outcome under the evaluation definition.
- **Breach recall:** Measures the proportion of all observed breaches captured by the selected inspection set.
- **Breach-yield enrichment:** Compares the breach yield of a selected group with that expected under random selection.

For additional definitions and methodology, see [`docs/CLBI_METRICS.md`](docs/CLBI_METRICS.md).

## Project Structure

```text
WILDFIRE/
├── data/
│   ├── demo/
│   │   └── clbi_demo_segments.csv
│   └── final/
│       ├── fireline_segments_v1.parquet
│       ├── fireline_segments_v1.csv
│       └── fireline_segments_v1.json
├── docs/
│   ├── CLBI_API_CONTRACT.md
│   └── CLBI_METRICS.md
├── models/
│   ├── clbi_logistic_regression.joblib
│   └── clbi_model_metadata.json
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── data/
│   │   └── build_dataset.py
│   ├── features/
│   │   └── generate_data_quality_report.py
│   ├── evaluation/
│   └── model/
│       ├── train_clbi_model.py
│       ├── clbi_model.py
│       └── clbi_demo.py
├── tests/
│   ├── test_clbi_model.py
│   └── test_api.py
├── experiments/
│   ├── BASELINE_REPORT.md
│   ├── ROBUSTNESS_REPORT.md
│   ├── NONLINEAR_BASELINE_REPORT.md
│   ├── SPATIAL_ABLATION_REPORT.md
│   ├── RISK_STRATIFICATION_REPORT.md
│   └── results/
├── requirements.txt
└── README.md
```

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| pandas | Tabular data processing |
| NumPy | Numerical operations |
| scikit-learn | Model training and evaluation |
| Joblib | Model artifact serialization |
| FastAPI | REST API framework |
| Uvicorn | ASGI server |
| Pydantic | Request and response validation |
| unittest | Automated testing |
| Parquet, CSV, JSON | Dataset storage and exchange |

The exact installed dependencies are defined in `requirements.txt`.

## Installation

### Prerequisites

- Python 3.10 or a compatible version supported by project dependencies.
- Git.
- pip.

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd WILDFIRE
```

Replace the repository URL and directory name with the actual values for your project.

### 2. Create a Virtual Environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Usage

### Train the Production Model

```bash
python -m src.model.train_clbi_model
```

This command trains the production model using the designated training data and saves the model artifact according to the training script's configuration.

### Run the Command-Line Demonstration

```bash
python -m src.model.clbi_demo
```

The demo demonstrates the inference workflow using the project's demonstration data.

### Run the Automated Test Suite

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The repository documents 29 unit tests covering the model and API. Confirm the actual test count and results by running the suite.

## REST API

CLBI provides a FastAPI interface for health checks, model information, individual segment predictions, candidate-line prioritization, and demonstration predictions.

### Start the Server

```bash
python -m uvicorn src.api.main:app --reload
```

Assuming the corresponding routes are implemented, the local interface is available at:

| Resource | URL |
|---|---|
| Dashboard | http://localhost:8000/ |
| Demo page | http://localhost:8000/demo |
| Swagger UI | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |
| OpenAPI schema | http://localhost:8000/openapi.json |

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

### Retrieve Model Information

```bash
curl http://localhost:8000/api/v1/model/info
```

### Predict Breach Risk for One Segment

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "segment_id": "demo_001",
    "slope": 35.0,
    "elevation": 800.0,
    "distance_to_fire": 100.0,
    "barrier_width_m": 2.0,
    "burn_prob": 0.85,
    "attack_angle": 20.0,
    "attack_dot_product": 0.94
  }'
```

The response schema is defined by the API implementation and is intended to include the predicted breach probability and associated risk information.

### Prioritize Candidate Segments

```bash
curl -X POST http://localhost:8000/api/v1/prioritize \
  -H "Content-Type: application/json" \
  -d '{
    "segments": [
      {
        "segment_id": "seg_001",
        "slope": 35.0,
        "elevation": 800.0,
        "distance_to_fire": 100.0,
        "barrier_width_m": 2.0,
        "burn_prob": 0.85,
        "attack_angle": 20.0,
        "attack_dot_product": 0.94
      }
    ],
    "selection": {
      "type": "percentage",
      "value": 10
    }
  }'
```

This example illustrates the intended request format for selecting a percentage of candidate segments. Meaningful top-10% ranking requires a sufficiently sized candidate set.

### Retrieve Demonstration Predictions

```bash
curl http://localhost:8000/api/v1/demo
```

For authoritative request fields, response definitions, and validation requirements, consult [`docs/CLBI_API_CONTRACT.md`](docs/CLBI_API_CONTRACT.md).

## Testing

Run the full test suite with:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The tests are intended to cover model inference behavior, API responses, request validation, and related functionality implemented in the repository.

## Research Methodology

### 1. Dataset Preparation

The project uses standardized historical wildfire segment data with a defined feature schema and target outcome.

Dataset preparation and quality checks are handled through scripts in `src/data/` and `src/features/`.

### 2. Leave-One-Fire-Out Evaluation

LOFO evaluation excludes one wildfire incident from training in each fold and evaluates the model on that held-out incident.

This approach is intended to assess generalization across distinct fires rather than relying exclusively on randomly partitioned segments from the same incidents.

### 3. Production Model Training

Following research evaluation, a separate production model is trained on the designated historical training data and serialized for inference.

The production model is not retrained during ordinary API prediction requests.

### 4. Risk Prioritization

Predicted probabilities are mapped to risk tiers and used to rank candidate segments for inspection.

The prioritization process is intended to help allocate inspection resources, not to autonomously determine where fire crews should operate.

### 5. Interpretation

Feature attributions are interpreted as statistical associations within the fitted model. They do not establish that an individual feature physically causes containment failure.

## Limitations

1. **Limited incident diversity:** The benchmark covers five historical California wildfire incidents. Broader validation is needed before generalizing to different regions and fire regimes.
2. **Retrospective evaluation:** The reported results come from historical data and do not establish performance during live incidents.
3. **Daily spread resolution:** Daily next-day burn-probability layers do not support claims about sub-hourly fire behavior.
4. **No field operational trial:** Inspection precision, recall, and enrichment are offline evaluation results, not evidence of improved field outcomes.
5. **Non-causal interpretation:** Model coefficients and feature attributions represent statistical associations, not verified physical causal mechanisms.
6. **Data dependence:** Model reliability depends on feature quality, target-label accuracy, and consistency between training and deployment data.
7. **Probability calibration limits:** Calibration measured on historical evaluation data may not persist when environmental conditions or incident characteristics change.
8. **Human oversight required:** Outputs must be interpreted alongside current weather, fuel conditions, topography, field observations, and qualified incident-command judgment.

## Future Improvements

Potential directions for further development include:

- Validation across additional geographic regions and wildfire incidents.
- Temporal and spatial robustness analysis.
- Monitoring for changes in feature distributions and calibration.
- Integration of updated weather and fuel-condition information.
- Interactive maps for visualizing candidate fireline segments and risk tiers.
- Improved reporting of model explanations and uncertainty.
- Deployment monitoring, versioned model artifacts, and reproducible evaluation pipelines.
- Prospective field evaluation under appropriate operational and safety oversight.

These items represent potential extensions and should not be interpreted as already implemented features.

## License

A license has not been specified in this README. Add a `LICENSE` file before distributing the project under an open-source license.

---

**Project:** Candidate Line Breach Intelligence (CLBI)  
**Benchmark:** `fireline_segments_v1`  
**Primary objective:** Predictive risk stratification and inspection prioritization for candidate wildfire containment line segments.
