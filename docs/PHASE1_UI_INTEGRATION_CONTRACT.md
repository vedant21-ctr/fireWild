# Phase 1: Tabular AML UI Integration Contract & Benchmark Specifications

**Project:** Next Day Wildfire Spread Prediction via Tabular AML and Spatial Deep Learning (Project 17)
**Status:** **PHASE 1 FROZEN & VERIFIED**
**Repository:** `C:\Users\Admin\Documents\WILDFIRE`
**Formatter Module:** [`src/api/ndws_contracts.py`](file:///C:/Users/Admin/Documents/WILDFIRE/src/api/ndws_contracts.py)
**Test Suite:** [`tests/test_ui_contract.py`](file:///C:/Users/Admin/Documents/WILDFIRE/tests/test_ui_contract.py)

---

## 1. Overview & Objectives

This document establishes the official frontend integration contract for presenting Phase 1 Machine Learning benchmarks in the project demonstration interface.

### Core Principles
1. **Zero Hallucination:** Every displayed metric (AP, ROC-AUC, Brier score, Precision, Recall, $F_1$, and Confusion Matrix counts) is sourced directly from verified test artifacts (`experiments/results_ndws/lightgbm_results.json` and `models/ndws/lightgbm_metadata.json`).
2. **Privacy & Path Sanitization:** Private local filesystem paths (e.g., `C:\Users\...`, `.joblib`, `.parquet`) are strictly scrubbed before transmission.
3. **Spatial Demonstration Transparency:** Spatial patch visualizers are explicitly flagged as **illustrative prototype simulations** because raw TFRecord benchmark records omit public geographic coordinate tags.
4. **Independent Modules:** Preserves the existing Candidate Line Breach Intelligence (CLBI) endpoints (`/api/v1/predict`, `/api/v1/prioritize`) intact while defining the Phase 1 Wildfire Spread Tabular Benchmark UI contract.

---

## 2. Benchmark Artifact Source of Truth

The frontend consumes verified benchmark data produced by Task 1C:

| Artifact Type | File Location (Internal) | Public Exposure |
|---|---|---|
| **Full Benchmark Results** | `experiments/results_ndws/lightgbm_results.json` | Consumed via `get_sanitized_benchmark_payload()` |
| **Model Metadata & Gains** | `models/ndws/lightgbm_metadata.json` | Consumed via `get_sanitized_benchmark_payload()` |
| **Precision-Recall Plot** | `experiments/results_ndws/lightgbm_pr_curves.png` | Static visual asset |
| **Feature Importance Plot** | `experiments/results_ndws/lightgbm_feature_importance.png` | Static visual asset |
| **Confusion Matrix Plot** | `experiments/results_ndws/lightgbm_confusion_matrix.png` | Static visual asset |

---

## 3. Public API Contract: Model Performance Summary

### Endpoint: `GET /api/v1/ndws/benchmark/metrics`
*(Or loaded directly by frontend via `src/api/ndws_contracts.py:get_sanitized_benchmark_payload`)*

### Response Schema (JSON)
```json
{
  "project": "Next Day Wildfire Spread Prediction via Tabular AML and Spatial Deep Learning",
  "phase": "Phase 1 - Tabular Machine Learning Baselines",
  "benchmark_summary": {
    "evaluation_metric": "Average Precision (PR-AUC)",
    "primary_objective": "Predict newly active wildfire spread (unburned -> burning transition)",
    "class_imbalance_prevalence": "0.58% - 0.96% positive",
    "pilot_sample_size": {
      "training_rows": 1000000,
      "validation_rows": 500000,
      "evaluation_rows": 500000
    }
  },
  "models_comparison": [
    {
      "model_id": "LightGBM_Unweighted",
      "display_name": "LightGBM (Unweighted, Best)",
      "is_best_model": true,
      "average_precision": 0.11447,
      "roc_auc": 0.872,
      "brier_score": 0.005689,
      "decision_threshold": 0.028115,
      "precision_at_threshold": 0.176389,
      "recall_at_threshold": 0.305357,
      "f1_at_threshold": 0.22361,
      "confusion_matrix": {
        "tn": 492890,
        "fp": 4179,
        "fn": 2036,
        "tp": 895
      },
      "training_time_seconds": 5.57
    },
    {
      "model_id": "LightGBM_Weighted_spw10",
      "display_name": "LightGBM (Weighted, spw=10)",
      "is_best_model": false,
      "average_precision": 0.102776,
      "roc_auc": 0.877354,
      "brier_score": 0.006087,
      "decision_threshold": 0.247552,
      "precision_at_threshold": 0.164258,
      "recall_at_threshold": 0.31798,
      "f1_at_threshold": 0.216618,
      "confusion_matrix": {
        "tn": 492327,
        "fp": 4742,
        "fn": 1999,
        "tp": 932
      },
      "training_time_seconds": 6.75
    },
    {
      "model_id": "Distance_Heuristic_Corrected",
      "display_name": "Corrected Distance Heuristic (Physical km)",
      "is_best_model": false,
      "average_precision": 0.072548,
      "roc_auc": 0.881722,
      "brier_score": 0.014795,
      "decision_threshold": 0.369404,
      "precision_at_threshold": 0.105784,
      "recall_at_threshold": 0.313886,
      "f1_at_threshold": 0.158239,
      "confusion_matrix": {
        "tn": 489292,
        "fp": 7777,
        "fn": 2011,
        "tp": 920
      },
      "training_time_seconds": 8.77
    },
    {
      "model_id": "Logistic_Regression_L2",
      "display_name": "Regularized Logistic Regression (L2)",
      "is_best_model": false,
      "average_precision": 0.064393,
      "roc_auc": 0.88633,
      "brier_score": 0.005612,
      "decision_threshold": 0.053061,
      "precision_at_threshold": 0.09377,
      "recall_at_threshold": 0.450358,
      "f1_at_threshold": 0.155221,
      "confusion_matrix": {
        "tn": 484312,
        "fp": 12757,
        "fn": 1611,
        "tp": 1320
      },
      "training_time_seconds": 22.93
    },
    {
      "model_id": "Dummy_Prior",
      "display_name": "Empirical Prior Baseline",
      "is_best_model": false,
      "average_precision": 0.005862,
      "roc_auc": 0.5,
      "brier_score": 0.005837,
      "decision_threshold": 0.008851,
      "precision_at_threshold": 0.005862,
      "recall_at_threshold": 1.0,
      "f1_at_threshold": 0.011656,
      "confusion_matrix": {
        "tn": 0,
        "fp": 497069,
        "fn": 0,
        "tp": 2931
      },
      "training_time_seconds": 0.29
    }
  ],
  "best_model_details": {
    "model_id": "LightGBM_Unweighted",
    "model_architecture": "LightGBM Gradient Boosted Decision Trees",
    "best_iteration": 3,
    "optimal_decision_threshold": 0.028115,
    "held_out_evaluation_performance": {
      "partition_source": "Official held-out evaluation shard (eval_00)",
      "average_precision": 0.08372,
      "roc_auc": 0.767868,
      "brier_score": 0.009354,
      "precision": 0.188126,
      "recall": 0.16102,
      "f1_score": 0.173521,
      "confusion_matrix": {
        "tn": 491895,
        "fp": 3323,
        "fn": 4012,
        "tp": 770
      },
      "false_positive_reduction_vs_logistic": "73.3% reduction in false alarms (3,323 vs 12,443)"
    },
    "feature_importance_ranking": [
      {"feature_name": "dist_to_prev_fire_km", "gain": 139581.69, "percentage_share": 70.47},
      {"feature_name": "th", "gain": 9515.53, "percentage_share": 4.80},
      {"feature_name": "erc", "gain": 9353.58, "percentage_share": 4.72},
      {"feature_name": "tmmn", "gain": 7304.68, "percentage_share": 3.69},
      {"feature_name": "elevation", "gain": 6861.56, "percentage_share": 3.46},
      {"feature_name": "population", "gain": 6429.71, "percentage_share": 3.25},
      {"feature_name": "vs", "gain": 6207.19, "percentage_share": 3.13},
      {"feature_name": "tmmx", "gain": 5469.54, "percentage_share": 2.76},
      {"feature_name": "pdsi", "gain": 3518.21, "percentage_share": 1.78},
      {"feature_name": "sph", "gain": 2539.69, "percentage_share": 1.28},
      {"feature_name": "NDVI", "gain": 1251.01, "percentage_share": 0.63},
      {"feature_name": "pr", "gain": 460.20, "percentage_share": 0.23}
    ]
  },
  "ui_display_disclaimers": [
    "Metrics reflect retrospective academic benchmarks on the Next Day Wildfire Spread (NDWS) dataset.",
    "All metrics are sourced directly from verified test artifacts.",
    "Decision thresholds were selected strictly using training validation partitions."
  ]
}
```

---

## 4. Spatial Map / Heatmap Integration Contract

### Endpoint: `GET /api/v1/ndws/demo/patch`
*(Generated via `src/api/ndws_contracts.py:generate_illustrative_spatial_demo_patch`)*

### Response Schema (JSON)
```json
{
  "patch_id": "illustrative_demo_patch_64x64",
  "is_georeferenced": false,
  "is_illustrative_simulation": true,
  "grid_dimensions": {
    "rows": 64,
    "cols": 64,
    "pixel_resolution_nominal_km": 1.0
  },
  "disclaimer": "Illustrative 64x64 spatial grid generated for user interface layout and heatmap visualization. Raw TFRecord benchmark patches lack embedded public geographic coordinates.",
  "risk_classification_bands": {
    "low_risk": "< 0.02 (Background)",
    "moderate_risk": "0.02 - 0.05",
    "elevated_risk": "0.05 - 0.10",
    "high_spread_risk": ">= 0.10"
  },
  "probabilities_2d": [
    [0.0010, 0.0012, "... 64 values per row ..."]
  ]
}
```

### Frontend UI Component Guidelines
1. **Heatmap Display:** Render `probabilities_2d` using a standard colormap (e.g., Viridis, Inferno, or a custom Green-Yellow-Red spread risk gradient).
2. **Disclaimer Badge:** When rendering spatial heatmaps, display the prominent badge:
   `[PROTOTYPE VISUALIZATION: Illustrative 64x64 Patch - Not Georeferenced]`.
3. **Threshold Marker:** Mark the tuned decision threshold ($\tau^* = 0.0281$) on the legend as the operational spread boundary.
4. **False Alarm Comparison Card:** Feature the **73.3% false alarm reduction** metric (3,323 vs 12,443) to highlight the tactical benefit of LightGBM over linear baselines.
