# Candidate Line Breach Intelligence (CLBI)

**Research Focus:** Predictive Wildfire Containment Line Holding & Breach Intelligence  
**Core Question:** *Can predicted next-day wildfire spread provide useful directional information for identifying vulnerable segments of a proposed containment line?*  
**Benchmark Version:** `fireline_segments_v1`  

---

## Candidate Line Breach Intelligence (CLBI) Decision Engine

### **Problem Statement**
Fire managers evaluating candidate containment lines need to identify which 100-meter segments historically exhibit conditions associated with higher breach risk. The CLBI Decision Engine processes candidate firelines to stratify risk and prioritize inspection resources.

### **Input Features (7 Validated Features)**
1. `slope`: Terrain slope angle (degrees)
2. `elevation`: Surface elevation (meters)
3. `distance_to_fire`: Distance to active flame front at day $t$ (meters)
4. `barrier_width_m`: Physical cleared width of barrier (meters)
5. `burn_prob`: Deep learning predicted next-day burn probability $[0, 1]$
6. `attack_angle`: Acute interaction angle between line normal and spread heading (degrees)
7. `attack_dot_product`: Direct head-fire attack alignment ratio ($\cos\theta_{\text{attack}}$)

### **Output Schema**
- `breach_probability`: Predicted probability of breach $P(\text{Breach}) \in [0, 1]$
- `hold_probability`: Predicted probability of holding $1 - P(\text{Breach})$
- `risk_tier`: Operational risk category (`LOW` $<0.30$, `MEDIUM` $0.30\text{--}0.60$, `HIGH` $0.60\text{--}0.80$, `CRITICAL` $\ge 0.80$)
- `vulnerability_rank`: Ordinal ranking (1 = highest breach risk)
- `priority_percentile`: Percentile rank across candidate fireline
- `priority_recommended`: Boolean indicator for top 10% inspection set
- `top_risk_factors` & `protective_factors`: Structured feature attributions
- `explanation`: Non-causal association summary

---

## Validated Benchmark Evidence

Evaluated via 5-fold Leave-One-Fire-Out (LOFO) cross-validation across 4,600 held-out segments:

| Metric | Measured Benchmark Result |
| :--- | :--- |
| **Mean LOFO PR-AUC** | **0.9508 ± 0.0076** |
| **Global Brier Score** | **0.1199** |
| **ECE (10 Equal-Width Bins)** | **0.0109** (1.09 percentage points) |
| **ECE (20 Equal-Width Bins)** | **0.0151** (1.51 percentage points) |
| **Top-10% Inspection Precision** | **89.6% mean** (84.0% to 95.8% across fires) |
| **Top-20% Breach Recall** | **53.8% mean** (48.8% to 57.2% across fires) |
| **Offline Breach-Yield Enrichment**| **2.92× versus random selection** |

---

## CLBI Production Architecture

```
Research Evaluation (LOFO 5-Fold)
            ↓
    LOFO Validation (Mean PR-AUC = 0.9508)
            ↓
Production Model Training (`train_clbi_model.py`)
            ↓
Saved Model Artifact (`models/clbi_logistic_regression.joblib`)
            ↓
CLBI Inference Engine (`CLBIDecisionEngine`)
            ↓
   Risk Stratification (`LOW` / `MEDIUM` / `HIGH` / `CRITICAL`)
            ↓
Candidate-Line Prioritization (Top-K / Resource Selection)
            ↓
Future REST API (`docs/CLBI_API_CONTRACT.md`) & Frontend
```

### **Methodological Separation: Research vs Production**
- **LOFO Research Evaluation:** Evaluates model generalization across 5 held-out wildfire incidents where each fire is excluded from training before evaluation. Used solely for unbiased metric reporting.
- **Production Model:** Fitted on all historical training data (`split == 'train'`) and serialized to `models/clbi_logistic_regression.joblib`. The production `CLBIDecisionEngine` strictly loads this artifact for deterministic, fast inference without retraining.

---

## Project Structure

```
WILDFIRE/
├── data/
│   ├── demo/
│   │   └── clbi_demo_segments.csv         # 100 deterministic demo segments (CZU fire)
│   └── final/
│       ├── fireline_segments_v1.parquet   # Production tabular dataset (187 KB)
│       ├── fireline_segments_v1.csv       # Inspection CSV dataset (749 KB)
│       └── fireline_segments_v1.json      # JSON schema representation
│
├── docs/
│   ├── CLBI_API_CONTRACT.md               # REST API specification & contract
│   └── CLBI_METRICS.md                    # Benchmark metric definitions & terms
│
├── models/
│   ├── clbi_logistic_regression.joblib    # Serialized production Logistic Regression model
│   └── clbi_model_metadata.json          # Production model metadata & provenance
│
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                        # FastAPI application entry point & CORS
│   │   ├── routes.py                      # REST API endpoint route handlers
│   │   └── schemas.py                     # Pydantic request/response validation schemas
│   ├── data/
│   │   └── build_dataset.py               # Standardized dataset builder pipeline
│   ├── features/
│   │   └── generate_data_quality_report.py# Audit & visualization generator
│   ├── evaluation/                        # Reproducible benchmark experiments
│   └── model/
│       ├── train_clbi_model.py            # Reproducible model training script
│       ├── clbi_model.py                  # Core CLBIDecisionEngine inference class
│       └── clbi_demo.py                   # CLI demonstration script
│
├── tests/
│   ├── test_clbi_model.py                 # Unit tests for CLBIDecisionEngine (14 tests)
│   └── test_api.py                        # Unit tests for CLBI REST API (15 tests)
│
├── experiments/                           # Research reports and results
│   ├── BASELINE_REPORT.md
│   ├── ROBUSTNESS_REPORT.md
│   ├── NONLINEAR_BASELINE_REPORT.md
│   ├── SPATIAL_ABLATION_REPORT.md
│   ├── RISK_STRATIFICATION_REPORT.md
│   └── results/
│       ├── clbi_segment_predictions.csv   # Model inference output for 4,600 segments
│       ├── clbi_demo_predictions.csv      # 100-segment demo predictions
│       ├── clbi_demo_response.json        # Product-ready JSON demo payload
│       ├── calibration_audit_results.json # Calibration audit & risk tier statistics
│       └── clbi_model_metadata.json       # Research metadata and benchmark metrics
│
├── requirements.txt                       # Minimal workspace dependencies
└── README.md                              # Project overview
```

---

## Quickstart & Usage

### 1. Train Production Model Artifact
```bash
python -m src.model.train_clbi_model
```

### 2. Run Unit Tests (29 Tests)
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 3. Run CLBI Demo & Generate Product JSON
```bash
python -m src.model.clbi_demo
```

---

## Running the CLBI REST API

### 1. Install API Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Start API Server
```bash
python -m uvicorn src.api.main:app --reload
```

### 3. Interactive API Documentation
- **Swagger / OpenAPI UI:** `http://localhost:8000/docs`
- **ReDoc UI:** `http://localhost:8000/redoc`
- **OpenAPI Schema:** `http://localhost:8000/openapi.json`

### 4. Example Curl Requests

- **Health Check:**
```bash
curl http://localhost:8000/api/v1/health
```

- **Model Info & Provenance:**
```bash
curl http://localhost:8000/api/v1/model/info
```

- **Single Segment Risk Prediction:**
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

- **Candidate Line Prioritization (Top 10% Cutoff):**
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

- **100-Segment Benchmark Demo:**
```bash
curl http://localhost:8000/api/v1/demo
```

> **Disclaimer:** The CLBI REST API is a prototype decision-support interface based on retrospective historical wildfire data. It does NOT provide live field validation or guaranteed fire behavior containment.

---

## Explicit Project Limitations

1. **Five Wildfire Complexes:** Benchmark evaluated on 5 historical California incidents ($N=5$).
2. **Offline Retrospective Evaluation:** Demonstrates retrospective predictive accuracy on historical data; not live field validation during active incidents.
3. **Daily NDWS Resolution:** Input spread predictions use daily NDWS layers; cannot support sub-hourly tactical fire behavior claims.
4. **Decision Support Simulation:** Resource prioritization metrics represent offline mathematical simulations, not field operational trials.
5. **Non-Causal Associations:** Model coefficients represent multivariate statistical associations, not physical causal mechanisms.
