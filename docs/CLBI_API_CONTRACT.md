# Candidate Line Breach Intelligence (CLBI) — REST API Contract

**Version:** v1.0.0  
**Service:** CLBI Decision Engine  
**Data Basis:** NDWS-derived historical benchmark dataset  

> **Disclaimer:** CLBI is a retrospective decision-support prototype built on historical wildfire benchmark data. It does NOT provide guaranteed fire behavior predictions, exact arrival times, guaranteed containment, or live field operational validation.

---

## Endpoints Summary

1. `GET /api/v1/health` — System health and status check.
2. `GET /api/v1/model/info` — Metadata and feature specification of the production model.
3. `POST /api/v1/predict` — Runs breach risk inference over candidate line segments.
4. `POST /api/v1/prioritize` — Ranks and returns top vulnerable segments based on resource budget.
5. `GET /api/v1/demo` — Returns the pre-computed 100-segment demo response.

---

## Endpoint Details

### 1. `GET /api/v1/health`
- **Purpose:** Health check.
- **Request:** None
- **Response (200 OK):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "v1.0.0"
}
```

---

### 2. `GET /api/v1/model/info`
- **Purpose:** Model metadata, feature requirements, and risk thresholds.
- **Request:** None
- **Response (200 OK):**
```json
{
  "model_name": "CLBI Production Model",
  "model_type": "Standardized 7-Feature Logistic Regression Pipeline",
  "required_features": [
    "slope",
    "elevation",
    "distance_to_fire",
    "barrier_width_m",
    "burn_prob",
    "attack_angle",
    "attack_dot_product"
  ],
  "risk_thresholds": {
    "LOW": "< 0.30",
    "MEDIUM": "0.30 to < 0.60",
    "HIGH": "0.60 to < 0.80",
    "CRITICAL": ">= 0.80"
  },
  "benchmark_metrics": {
    "lofo_mean_pr_auc": "0.9508 ± 0.0076",
    "global_brier_score": 0.1199,
    "offline_breach_yield_enrichment_ratio": "2.92x"
  }
}
```

---

### 3. `POST /api/v1/predict`
- **Purpose:** Run CLBI risk stratification inference on candidate segments.
- **Request Body:**
```json
{
  "segments": [
    {
      "segment_id": "CA-CZU-005205_seg_00001",
      "fire_id": "CA-CZU-005205",
      "slope": 24.5,
      "elevation": 380.0,
      "distance_to_fire": 1250.0,
      "barrier_width_m": 6.5,
      "burn_prob": 0.65,
      "attack_angle": 15.2,
      "attack_dot_product": 0.965
    }
  ]
}
```
- **Validation Errors (400 Bad Request):** Returned if missing features, NaN/Inf, or out-of-bound feature values occur.
```json
{
  "error": "Validation Error",
  "detail": "Feature 'burn_prob' must be bounded in [0.0, 1.0]"
}
```
- **Response (200 OK):**
```json
{
  "summary": {
    "total_segments": 1,
    "low": 0,
    "medium": 0,
    "high": 1,
    "critical": 0
  },
  "segments": [
    {
      "segment_id": "CA-CZU-005205_seg_00001",
      "fire_id": "CA-CZU-005205",
      "breach_probability": 0.7425,
      "hold_probability": 0.2575,
      "risk_tier": "HIGH",
      "vulnerability_rank": 1,
      "priority_percentile": 100.0,
      "priority_recommended": true,
      "top_risk_factors": ["Direct Head-Fire Attack Alignment (cos theta) (attack_dot_product=0.965)"],
      "protective_factors": ["Cleared Barrier Width (barrier_width_m=6.5)"],
      "explanation": "HIGH breach risk (74.3% predicted risk). Elevated predicted risk is associated with: Direct Head-Fire Attack Alignment (cos theta) (attack_dot_product=0.965)."
    }
  ]
}
```

---

### 4. `POST /api/v1/prioritize`
- **Purpose:** Rank candidate segments and return top-k priority inspection set.
- **Request Body:**
```json
{
  "segments": [...],
  "resource_pct": 10.0
}
```
- **Response (200 OK):**
```json
{
  "prioritization_type": "risk-based candidate-line prioritization",
  "total_candidate_segments": 100,
  "total_selected_segments": 10,
  "pct_line_selected": 10.0,
  "selected_mean_breach_probability": 0.8910,
  "selected_segments": [...]
}
```

---

### 5. `GET /api/v1/demo`
- **Purpose:** Retrieve 100-segment demo predictions response.
- **Response (200 OK):** Returns content of `experiments/results/clbi_demo_response.json`.
