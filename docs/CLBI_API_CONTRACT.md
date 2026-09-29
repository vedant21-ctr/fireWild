# Candidate Line Breach Intelligence (CLBI) — REST API Contract

**Version:** v1.0.0  
**Service:** CLBI Decision Engine API  
**Framework:** FastAPI / Uvicorn  
**Data Basis:** NDWS-derived historical benchmark dataset  

> **Disclaimer:** CLBI is a retrospective decision-support prototype built on historical wildfire benchmark data. It does NOT provide guaranteed fire behavior predictions, exact arrival times, guaranteed containment, or live field operational validation.

---

## Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | System health and service status check |
| `GET` | `/api/v1/model/info` | Production model provenance, features, and evaluation metrics |
| `POST` | `/api/v1/predict` | Single-segment breach risk inference (unranked) |
| `POST` | `/api/v1/predict/batch` | Batch breach risk inference with population vulnerability ranking |
| `POST` | `/api/v1/prioritize` | Candidate line prioritization by percentage or count cutoff |
| `GET` | `/api/v1/demo` | Retrieves 100-segment demo response matching canonical JSON format |

---

## Endpoint Details

### 1. `GET /api/v1/health`
- **Purpose:** System health check.
- **Request:** None
- **Response (200 OK):**
```json
{
  "status": "ok",
  "service": "CLBI API",
  "version": "1.0.0"
}
```

---

### 2. `GET /api/v1/model/info`
- **Purpose:** Model provenance, required feature schema, risk thresholds, and evaluation metrics.
- **Request:** None
- **Response (200 OK):**
```json
{
  "model_type": "Standardized 7-Feature Logistic Regression Pipeline (StandardScaler + LogisticRegression)",
  "feature_list": [
    "slope",
    "elevation",
    "distance_to_fire",
    "barrier_width_m",
    "burn_prob",
    "attack_angle",
    "attack_dot_product"
  ],
  "risk_thresholds": {
    "low": 0.30,
    "medium": 0.60,
    "high": 0.80
  },
  "training_rows": 3050,
  "training_fires": [
    "CA-MNF-013028",
    "CA-SHU-007808",
    "CA-SNF-000958"
  ],
  "evaluation": {
    "method": "5-fold Leave-One-Fire-Out",
    "mean_pr_auc": 0.9508,
    "brier": 0.1199,
    "ece_10": 0.0109,
    "ece_20": 0.0151
  }
}
```

---

### 3. `POST /api/v1/predict`
- **Purpose:** Single candidate segment breach risk stratification and explainability.
- **Request Body:**
```json
{
  "segment_id": "demo_001",
  "fire_id": "CA-CZU-005205",
  "fire_name": "CZU Lightning Complex",
  "date": "2020-08-20",
  "slope": 35.0,
  "elevation": 800.0,
  "distance_to_fire": 100.0,
  "barrier_width_m": 2.0,
  "burn_prob": 0.85,
  "attack_angle": 20.0,
  "attack_dot_product": 0.94
}
```
- **Response (200 OK):**
```json
{
  "segment_id": "demo_001",
  "fire_id": "CA-CZU-005205",
  "fire_name": "CZU Lightning Complex",
  "date": "2020-08-20",
  "breach_probability": 0.9421,
  "hold_probability": 0.0579,
  "risk_tier": "CRITICAL",
  "vulnerability_rank": null,
  "priority_percentile": null,
  "priority_recommended": null,
  "top_risk_factors": [
    "Direct Head-Fire Attack Alignment (cos theta) (attack_dot_product=0.94)",
    "Predicted Next-Day Spread Probability Intensity (burn_prob=0.85)"
  ],
  "protective_factors": [],
  "explanation": "CRITICAL breach vulnerability (94.2% predicted risk). Elevated predicted risk is associated with: Direct Head-Fire Attack Alignment (cos theta) (attack_dot_product=0.94), Predicted Next-Day Spread Probability Intensity (burn_prob=0.85)."
}
```

---

### 4. `POST /api/v1/predict/batch`
- **Purpose:** Batch inference with population vulnerability ranking across multiple candidate segments.
- **Request Body:**
```json
{
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
    },
    {
      "segment_id": "seg_002",
      "slope": 10.0,
      "elevation": 300.0,
      "distance_to_fire": 500.0,
      "barrier_width_m": 12.0,
      "burn_prob": 0.15,
      "attack_angle": 45.0,
      "attack_dot_product": -0.20
    }
  ]
}
```
- **Response (200 OK):**
```json
{
  "total_segments": 2,
  "segments": [
    {
      "segment_id": "seg_001",
      "breach_probability": 0.9421,
      "hold_probability": 0.0579,
      "risk_tier": "CRITICAL",
      "vulnerability_rank": 1,
      "priority_percentile": 100.0,
      "priority_recommended": true,
      "top_risk_factors": [...],
      "protective_factors": [...],
      "explanation": "..."
    },
    {
      "segment_id": "seg_002",
      "breach_probability": 0.0812,
      "hold_probability": 0.9188,
      "risk_tier": "LOW",
      "vulnerability_rank": 2,
      "priority_percentile": 50.0,
      "priority_recommended": false,
      "top_risk_factors": [...],
      "protective_factors": [...],
      "explanation": "..."
    }
  ]
}
```

---

### 5. `POST /api/v1/prioritize`
- **Purpose:** Rank candidate segments and select top-vulnerable subset based on percentage or count.
- **Request Body (Percentage Selection):**
```json
{
  "segments": [...],
  "selection": {
    "type": "percentage",
    "value": 10
  }
}
```
- **Request Body (Count Selection):**
```json
{
  "segments": [...],
  "selection": {
    "type": "count",
    "value": 5
  }
}
```
- **Response (200 OK):**
```json
{
  "total_segments": 100,
  "selected_count": 10,
  "selection_type": "percentage",
  "selection_value": 10.0,
  "segments": [...]
}
```

---

### 6. `GET /api/v1/demo`
- **Purpose:** Retrieve pre-computed 100-segment demonstration predictions response.
- **Response (200 OK):** Returns product JSON matching `experiments/results/clbi_demo_response.json`.

---

## Validation & Error Handling

Invalid inputs return HTTP status codes `400` or `422` with clean, structured error responses:

```json
{
  "error": "validation_error",
  "message": "burn_prob must be between 0 and 1"
}
```

Common Validation Rules:
- `burn_prob` must be in $[0.0, 1.0]$.
- `attack_dot_product` must be in $[-1.0, 1.0]$.
- `barrier_width_m` must be $> 0.0$.
- `slope` and `distance_to_fire` must be $\ge 0.0$.
- Selection percentage must be in $(0.0, 100.0]$. Selection count must be $\ge 1$.
- Input segment lists cannot be empty.
