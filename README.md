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

## Project Structure

```
WILDFIRE/
├── data/
│   └── final/
│       ├── fireline_segments_v1.parquet   # Production tabular dataset (187 KB)
│       ├── fireline_segments_v1.csv       # Inspection CSV dataset (749 KB)
│       └── fireline_segments_v1.json      # JSON schema representation
│
├── src/
│   ├── data/
│   │   └── build_dataset.py               # Standardized dataset builder pipeline
│   ├── features/
│   │   └── generate_data_quality_report.py# Audit & visualization generator
│   ├── evaluation/                        # Reproducible benchmark experiments
│   └── model/
│       ├── clbi_model.py                  # Core CLBIDecisionEngine inference class
│       └── clbi_demo.py                   # CLI demonstration script
│
├── tests/
│   └── test_clbi_model.py                 # Unit tests for CLBIDecisionEngine
│
├── experiments/                           # Research reports and results
│   ├── BASELINE_REPORT.md
│   ├── ROBUSTNESS_REPORT.md
│   ├── NONLINEAR_BASELINE_REPORT.md
│   ├── SPATIAL_ABLATION_REPORT.md
│   ├── RISK_STRATIFICATION_REPORT.md
│   └── results/
│       ├── clbi_segment_predictions.csv   # Model inference output for 4,600 segments
│       └── clbi_model_metadata.json       # Model metadata and benchmark metrics
│
├── requirements.txt                       # Minimal workspace dependencies
└── README.md                              # Project overview
```

---

## Quickstart & Usage

### 1. Run Unit Tests
```bash
python -m unittest discover -s tests
```

### 2. Run CLBI CLI Demonstration
```bash
python -m src.model.clbi_demo
```

### 3. Python API Example
```python
import pandas as pd
from src.model.clbi_model import CLBIDecisionEngine

# Load candidate line segments
df = pd.read_parquet("data/final/fireline_segments_v1.parquet")

# Initialize and fit decision engine
engine = CLBIDecisionEngine()
engine.fit(df[df['split'] == 'train'])

# Run inference
predictions_df = engine.predict_df(df, top_k_recommend_pct=10.0)

# Prioritize top 10% candidate segments
prioritized = engine.prioritize_candidate_line(predictions_df, resource_pct=10.0)
print(prioritized["selected_segments"][["segment_id", "breach_probability", "risk_tier"]])
```

---

## Explicit Project Limitations

1. **Five Wildfire Complexes:** Benchmark evaluated on 5 historical California incidents ($N=5$).
2. **Offline Retrospective Evaluation:** Demonstrates retrospective predictive accuracy on historical data; not live field validation during active incidents.
3. **Daily NDWS Resolution:** Input spread predictions use daily NDWS layers; cannot support sub-hourly tactical fire behavior claims.
4. **Decision Support Simulation:** Resource prioritization metrics represent offline mathematical simulations, not field operational trials.
5. **Non-Causal Associations:** Model coefficients represent multivariate statistical associations, not physical causal mechanisms.
