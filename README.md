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
│   └── test_clbi_model.py                 # Unit tests for CLBIDecisionEngine (14 tests)
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

### 2. Run Unit Tests (14 Tests)
```bash
python -m pytest -q
```

### 3. Run CLBI Demo & Generate Product JSON
```bash
python -m src.model.clbi_demo
```

### 4. Python API Usage
```python
import pandas as pd
from src.model.clbi_model import CLBIDecisionEngine

# Load production decision engine (loads saved model artifact)
engine = CLBIDecisionEngine(model_path="models/clbi_logistic_regression.joblib")

# Load candidate line segments
df = pd.read_csv("data/demo/clbi_demo_segments.csv")

# Run inference & risk stratification
predictions_df = engine.predict_df(df, top_k_recommend_pct=10.0)

# Prioritize top candidate segments
prioritized = engine.prioritize_candidate_line(predictions_df, resource_pct=10.0)
print(prioritized["selected_segments"][["segment_id", "breach_probability", "risk_tier", "priority_recommended"]])
```

---

## Explicit Project Limitations

1. **Five Wildfire Complexes:** Benchmark evaluated on 5 historical California incidents ($N=5$).
2. **Offline Retrospective Evaluation:** Demonstrates retrospective predictive accuracy on historical data; not live field validation during active incidents.
3. **Daily NDWS Resolution:** Input spread predictions use daily NDWS layers; cannot support sub-hourly tactical fire behavior claims.
4. **Decision Support Simulation:** Resource prioritization metrics represent offline mathematical simulations, not field operational trials.
5. **Non-Causal Associations:** Model coefficients represent multivariate statistical associations, not physical causal mechanisms.
