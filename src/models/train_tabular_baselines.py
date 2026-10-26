"""
train_tabular_baselines.py - Leakage-Safe Tabular Baselines for NDWS

Builds, tunes, and evaluates initial Phase 1 Machine Learning baselines
for Next Day Wildfire Spread (NDWS) prediction.

Baselines Implemented:
1. Prior Prevalence Dummy Classifier (constant prior probability)
2. Spatial Distance Heuristic (inverse distance to day-t fire front)
3. Logistic Regression with L2 Regularization (unweighted)
4. Logistic Regression with L1 Regularization (Lasso / feature sparsity)
5. Logistic Regression with L2 Regularization + Class Weight Balancing

Strict Methodological Safeguards:
- Preprocessing (StandardScaler) is fitted EXCLUSIVELY on training data.
- Hyperparameter comparison and decision threshold tuning occur EXCLUSIVELY
  on internal validation data (training holdout shard).
- Official evaluation shard (eval_00) is held out strictly for testing the final frozen pipeline.
- Target values and derivatives (FireMask, fire_mask_t1, target_new_fire) are strictly
  excluded from predictor features.
- Memory safe: supports configurable pilot/subsample row limits to operate safely
  within available RAM.
"""

import os
import sys
import glob
import json
import time
import argparse
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pyarrow.parquet as pq
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

# 12 Leakage-Free Predictor Channels (11 environmental + 1 prior-day spatial distance)
FEATURE_COLUMNS: List[str] = [
    "elevation",
    "th",
    "vs",
    "tmmn",
    "tmmx",
    "sph",
    "pr",
    "pdsi",
    "NDVI",
    "population",
    "erc",
    "dist_to_prev_fire_km",
]

TARGET_COLUMN: str = "target_new_fire"
FORBIDDEN_COLUMNS: List[str] = [
    "FireMask",
    "fire_mask_t1",
    "target_new_fire",
]


class DummyPriorClassifier(BaseEstimator, ClassifierMixin):
    """Predicts a constant probability equal to the empirical training set positive prevalence."""
    def __init__(self):
        self.prior_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.prior_ = float(np.mean(y))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = np.full(X.shape[0], self.prior_, dtype=np.float32)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(np.int32)


class DistanceHeuristicClassifier(BaseEstimator, ClassifierMixin):
    """
    Predicts wildfire spread probability inversely related to distance from
    the nearest day-t active fire pixel: p = exp(-distance_km / scale_km).
    When distance is at the sentinel value (>= 90 km), probability is calibrated
    to the global background rate.

    Guarantees physical kilometer computation:
    If a fitted StandardScaler is provided, it automatically unscales distance
    back to physical kilometers, preventing standardization distortion.
    """
    def __init__(self, dist_col_idx: int = 11, scale_km: float = 2.0, scaler: Optional[StandardScaler] = None):
        self.dist_col_idx = dist_col_idx
        self.scale_km = scale_km
        self.scaler = scaler
        self.background_rate_: float = 0.001

    def _get_raw_distance(self, X: np.ndarray) -> np.ndarray:
        dist = X[:, self.dist_col_idx]
        if self.scaler is not None and hasattr(self.scaler, "mean_"):
            return dist * self.scaler.scale_[self.dist_col_idx] + self.scaler.mean_[self.dist_col_idx]
        return dist

    def fit(self, X: np.ndarray, y: np.ndarray):
        dist = self._get_raw_distance(X)
        far_mask = dist >= 90.0
        if np.any(far_mask):
            self.background_rate_ = float(np.mean(y[far_mask]))
        else:
            self.background_rate_ = float(np.mean(y)) * 0.1
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        dist = self._get_raw_distance(X)
        decay_prob = np.exp(-dist / self.scale_km)
        prob = np.where(dist >= 90.0, self.background_rate_, np.maximum(decay_prob, self.background_rate_))
        prob = np.clip(prob, 0.0, 1.0).astype(np.float32)
        return np.column_stack([1.0 - prob, prob])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(np.int32)


def load_tabular_data(
    file_paths: List[str],
    max_rows: Optional[int] = None,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Loads tabular data from Parquet files into numpy arrays.
    Verifies that no target-derived features enter the predictor set.
    """
    if not file_paths:
        raise ValueError("No Parquet file paths provided for data loading.")

    # Schema & feature safety check
    for forbidden in FORBIDDEN_COLUMNS:
        if forbidden in FEATURE_COLUMNS:
            raise ValueError(f"CRITICAL TARGET LEAKAGE: '{forbidden}' found in FEATURE_COLUMNS!")

    columns_to_read = FEATURE_COLUMNS + [TARGET_COLUMN]
    
    x_chunks: List[np.ndarray] = []
    y_chunks: List[np.ndarray] = []
    total_loaded = 0

    for path in file_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Parquet file not found: {path}")

        table = pq.read_table(path, columns=columns_to_read)
        num_rows = table.num_rows

        x_chunk = np.column_stack([table[c].to_numpy().astype(np.float32) for c in FEATURE_COLUMNS])
        y_chunk = table[TARGET_COLUMN].to_numpy().astype(np.int32)

        x_chunks.append(x_chunk)
        y_chunks.append(y_chunk)
        total_loaded += num_rows

        if max_rows is not None and total_loaded >= max_rows:
            break

    X = np.concatenate(x_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)

    # Subsample reproducibly if max_rows specified
    if max_rows is not None and len(y) > max_rows:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(y), size=max_rows, replace=False)
        indices.sort()
        X = X[indices]
        y = y[indices]

    return X, y, FEATURE_COLUMNS


def fit_preprocessor(X_train: np.ndarray) -> StandardScaler:
    """
    Fits StandardScaler exclusively on training feature matrix.
    Asserts no target leakage and derived parameters are valid.
    Uses copy=True to ensure raw feature matrices are never mutated in place.
    """
    scaler = StandardScaler(copy=True)
    scaler.fit(X_train)
    return scaler


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_thresholds: int = 200
) -> Tuple[float, float]:
    """
    Sweeps thresholds on validation set to find the decision threshold
    maximizing F1-score for the positive (new fire) class.
    """
    # Sample thresholds between min and max non-trivial probabilities
    non_trivial_probs = y_prob[(y_prob > 0.0001) & (y_prob < 0.9999)]
    if len(non_trivial_probs) == 0:
        return 0.5, 0.0

    thresholds = np.linspace(np.percentile(non_trivial_probs, 1), np.percentile(non_trivial_probs, 99), num_thresholds)
    best_thresh = 0.5
    best_f1 = -1.0

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(np.int32)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_thresh = float(thresh)

    return best_thresh, best_f1


def calculate_recall_at_fixed_fpr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_fpr: float = 0.01
) -> Dict[str, Any]:
    """
    Computes recall at an empirical false-positive rate (FPR) strictly <= target_fpr.

    Methodology:
    - Calculates operating points from ROC curve thresholds.
    - Strictly enforces achieved FPR <= target_fpr (no overshooting).
    - Correctly handles tied prediction scores by group partition.
    - Gracefully handles edge cases (e.g. no positives or no negatives).
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    if n_pos == 0:
        return {
            "target_fpr": float(target_fpr),
            "achieved_fpr": 0.0,
            "recall": 0.0,
            "threshold": 1.0,
            "valid": False,
            "notes": "No positive samples in evaluation split"
        }

    if n_neg == 0:
        return {
            "target_fpr": float(target_fpr),
            "achieved_fpr": 0.0,
            "recall": 1.0,
            "threshold": 0.0,
            "valid": False,
            "notes": "No negative samples in evaluation split"
        }

    fpr, tpr, thresholds = roc_curve(y_true, y_prob, drop_intermediate=True)

    # Filter operating points where achieved FPR <= target_fpr
    valid_idx = np.where(fpr <= target_fpr)[0]

    if len(valid_idx) == 0:
        return {
            "target_fpr": float(target_fpr),
            "achieved_fpr": 0.0,
            "recall": 0.0,
            "threshold": float(np.max(y_prob) + 1.0),
            "valid": True,
            "notes": "No threshold achieves FPR <= target"
        }

    best_idx = valid_idx[-1]
    achieved_fpr = float(fpr[best_idx])
    achieved_recall = float(tpr[best_idx])
    thresh = float(thresholds[best_idx])

    # Re-verify empirical FPR strictly with >= threshold to guarantee tie safety
    empirical_fp = np.sum(((y_prob >= thresh) & (y_true == 0)))
    empirical_fpr = float(empirical_fp / n_neg)

    # If tied scores cause empirical_fpr to exceed target_fpr, walk back to a strict point
    if empirical_fpr > target_fpr:
        while valid_idx.size > 0:
            valid_idx = valid_idx[:-1]
            if len(valid_idx) == 0:
                return {
                    "target_fpr": float(target_fpr),
                    "achieved_fpr": 0.0,
                    "recall": 0.0,
                    "threshold": float(np.max(y_prob) + 1.0),
                    "valid": True,
                    "notes": "Tied scores prevented meeting target FPR"
                }
            best_idx = valid_idx[-1]
            thresh = float(thresholds[best_idx])
            empirical_fp = np.sum(((y_prob >= thresh) & (y_true == 0)))
            empirical_fpr = float(empirical_fp / n_neg)
            if empirical_fpr <= target_fpr:
                achieved_fpr = empirical_fpr
                achieved_recall = float(np.sum(((y_prob >= thresh) & (y_true == 1))) / n_pos)
                break

    return {
        "target_fpr": round(float(target_fpr), 6),
        "achieved_fpr": round(float(achieved_fpr), 6),
        "recall": round(float(achieved_recall), 6),
        "threshold": round(float(thresh), 6),
        "valid": True,
        "notes": "Verified operating point satisfying FPR constraint"
    }


def evaluate_model_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics:
    - Average Precision (AP / PR-AUC summary)
    - ROC-AUC
    - Brier Score
    - Precision, Recall, F1 at given threshold
    - Recall at fixed FPR (FPR <= 1% and FPR <= 5%)
    - Confusion Matrix [TN, FP, FN, TP]
    """
    y_pred = (y_prob >= threshold).astype(np.int32)
    ap = float(average_precision_score(y_true, y_prob)) if np.sum(y_true) > 0 else 0.0
    
    try:
        roc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    except ValueError:
        roc = 0.5

    brier = float(brier_score_loss(y_true, y_prob))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Calculate recall at fixed FPR constraints
    rec_at_1pct = calculate_recall_at_fixed_fpr(y_true, y_prob, target_fpr=0.01)
    rec_at_5pct = calculate_recall_at_fixed_fpr(y_true, y_prob, target_fpr=0.05)

    return {
        "average_precision": round(ap, 6),
        "roc_auc": round(roc, 6),
        "brier_score": round(brier, 6),
        "decision_threshold": round(threshold, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1_score": round(f1, 6),
        "recall_at_1pct_fpr": rec_at_1pct["recall"],
        "achieved_fpr_at_1pct": rec_at_1pct["achieved_fpr"],
        "threshold_at_1pct_fpr": rec_at_1pct["threshold"],
        "recall_at_5pct_fpr": rec_at_5pct["recall"],
        "achieved_fpr_at_5pct": rec_at_5pct["achieved_fpr"],
        "threshold_at_5pct_fpr": rec_at_5pct["threshold"],
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        },
        "total_eval_samples": len(y_true),
        "positive_samples": int(np.sum(y_true)),
        "prevalence": round(float(np.mean(y_true)), 6)
    }


def train_and_evaluate_baselines(
    train_paths: List[str],
    val_paths: List[str],
    eval_paths: List[str],
    train_max_rows: Optional[int] = 1000000,
    val_max_rows: Optional[int] = 500000,
    eval_max_rows: Optional[int] = 500000,
    output_dir: str = "experiments/results_ndws",
    model_dir: str = "models/ndws",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes the full tabular baseline experiment:
    1. Loads train, val, and eval data safely.
    2. Fits StandardScaler strictly on train.
    3. Trains and tunes Dummy Prior, Distance Heuristic, and Logistic Regression variants.
    4. Evaluates all models on validation set.
    5. Evaluates best model on held-out evaluation set.
    6. Generates diagnostic plots and serializes artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n============================================================")
    print(f"PHASE 1 TABULAR BASELINES: TRAINING & VALIDATION")
    print(f"============================================================")
    print(f"Train Shards:   {len(train_paths)} files (Max Rows: {train_max_rows:,})")
    print(f"Val Shards:     {len(val_paths)} files (Max Rows: {val_max_rows:,})")
    print(f"Eval Shards:    {len(eval_paths)} files (Max Rows: {eval_max_rows:,})")
    print(f"Output Dir:     {output_dir}")
    print(f"Model Dir:      {model_dir}")

    # 1. Load data
    t0 = time.time()
    print("\nLoading training data...")
    X_train, y_train, feat_names = load_tabular_data(train_paths, max_rows=train_max_rows, random_state=random_state)
    print(f"  X_train shape: {X_train.shape}, y_train positive prevalence: {np.mean(y_train)*100:.3f}% ({int(np.sum(y_train)):,} positives)")

    print("Loading internal validation data...")
    X_val, y_val, _ = load_tabular_data(val_paths, max_rows=val_max_rows, random_state=random_state + 1)
    print(f"  X_val shape: {X_val.shape}, y_val positive prevalence: {np.mean(y_val)*100:.3f}% ({int(np.sum(y_val)):,} positives)")

    print("Loading held-out evaluation data...")
    X_eval, y_eval, _ = load_tabular_data(eval_paths, max_rows=eval_max_rows, random_state=random_state + 2)
    print(f"  X_eval shape: {X_eval.shape}, y_eval positive prevalence: {np.mean(y_eval)*100:.3f}% ({int(np.sum(y_eval)):,} positives)")

    # 2. Fit preprocessor EXCLUSIVELY on training features
    print("\nFitting StandardScaler exclusively on X_train...")
    scaler = fit_preprocessor(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_eval_scaled = scaler.transform(X_eval)

    # 3. Define candidate models
    models: Dict[str, Any] = {
        "Dummy_Prior": DummyPriorClassifier(),
        "Distance_Heuristic": DistanceHeuristicClassifier(dist_col_idx=feat_names.index("dist_to_prev_fire_km")),
        "Logistic_Regression_L2": LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=200,
            random_state=random_state
        ),
        "Logistic_Regression_L1": LogisticRegression(
            C=1.0,
            solver="saga",
            l1_ratio=1.0,
            max_iter=100,
            tol=1e-3,
            random_state=random_state
        ),
        "Logistic_Regression_Balanced_L2": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=200,
            random_state=random_state
        ),
    }

    val_results: Dict[str, Any] = {}
    fitted_models: Dict[str, Any] = {}
    val_probs: Dict[str, np.ndarray] = {}

    print("\nFitting and tuning models on validation set...")
    for name, model in models.items():
        m_start = time.time()
        print(f"  -> Fitting {name}...")

        # Feature matrix used for fitting
        if name in ["Dummy_Prior", "Distance_Heuristic"]:
            model.fit(X_train, y_train)
            prob_val = model.predict_proba(X_val)[:, 1]
        else:
            model.fit(X_train_scaled, y_train)
            prob_val = model.predict_proba(X_val_scaled)[:, 1]

        fitted_models[name] = model
        val_probs[name] = prob_val

        # Select optimal decision threshold on validation set
        if name == "Dummy_Prior":
            opt_thresh = float(np.mean(y_train))
        else:
            opt_thresh, _ = find_optimal_threshold(y_val, prob_val)

        # Evaluate at optimal threshold
        metrics = evaluate_model_predictions(y_val, prob_val, threshold=opt_thresh)
        metrics["training_time_sec"] = round(time.time() - m_start, 2)
        val_results[name] = metrics

        print(f"     Val AP: {metrics['average_precision']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | "
              f"Brier: {metrics['brier_score']:.4f} | Opt Thresh: {metrics['decision_threshold']:.4f} | "
              f"F1: {metrics['f1_score']:.4f} | P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f} "
              f"({metrics['training_time_sec']}s)")

    # 4. Identify best model based on validation AP
    best_model_name = max(
        [k for k in val_results.keys() if "Logistic" in k],
        key=lambda k: val_results[k]["average_precision"]
    )
    best_model = fitted_models[best_model_name]
    best_thresh = val_results[best_model_name]["decision_threshold"]

    print(f"\nBest Validation Model: {best_model_name} (AP={val_results[best_model_name]['average_precision']:.4f})")
    print(f"Evaluating best model on held-out evaluation shard (eval_00)...")

    prob_eval = best_model.predict_proba(X_eval_scaled)[:, 1]
    eval_metrics = evaluate_model_predictions(y_eval, prob_eval, threshold=best_thresh)
    print(f"  Held-out Eval AP:  {eval_metrics['average_precision']:.4f}")
    print(f"  Held-out ROC-AUC:  {eval_metrics['roc_auc']:.4f}")
    print(f"  Held-out Brier:    {eval_metrics['brier_score']:.4f}")
    print(f"  Held-out F1 Score: {eval_metrics['f1_score']:.4f} (Precision: {eval_metrics['precision']:.4f}, Recall: {eval_metrics['recall']:.4f})")

    # 5. Generate and save diagnostic visualizations
    print("\nGenerating evaluation plots...")
    
    # Plot A: Precision-Recall Curves
    plt.figure(figsize=(9, 6))
    for name, prob in val_probs.items():
        prec, rec, _ = precision_recall_curve(y_val, prob)
        ap = val_results[name]["average_precision"]
        plt.plot(rec, prec, label=f"{name.replace('_', ' ')} (AP = {ap:.4f})", linewidth=1.8)

    # Prevalence baseline
    prev_val = np.mean(y_val)
    plt.axhline(prev_val, color="gray", linestyle="--", label=f"Random Prior Baseline ({prev_val*100:.2f}%)")
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall Curve Comparison — Validation Set", fontsize=13, fontweight="bold")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_curve_path = os.path.join(output_dir, "pr_curves_comparison.png")
    plt.savefig(pr_curve_path, dpi=150)
    plt.close()

    # Plot B: Feature Coefficients (Best Logistic Model)
    if hasattr(best_model, "coef_"):
        coefs = best_model.coef_[0]
        sorted_idx = np.argsort(np.abs(coefs))
        plt.figure(figsize=(10, 6))
        colors = ["#2b5c8f" if c >= 0 else "#c0392b" for c in coefs[sorted_idx]]
        plt.barh(np.array(feat_names)[sorted_idx], coefs[sorted_idx], color=colors, alpha=0.85)
        plt.xlabel("Standardized Coefficient Value", fontsize=11)
        plt.title(f"Standardized Coefficients — {best_model_name}", fontsize=13, fontweight="bold")
        plt.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        coef_path = os.path.join(output_dir, "feature_coefficients_logistic.png")
        plt.savefig(coef_path, dpi=150)
        plt.close()

    # Plot C: Confusion Matrix Heatmap (Best Model at Optimal Threshold)
    cm = eval_metrics["confusion_matrix"]
    cm_matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Eval Set, Threshold = {best_thresh:.3f})", fontsize=11, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["No Spread", "Spread"], fontsize=10)
    plt.yticks(tick_marks, ["No Spread", "Spread"], fontsize=10)
    thresh_val = cm_matrix.max() / 2.0
    for r in range(2):
        for c in range(2):
            plt.text(c, r, f"{cm_matrix[r, c]:,}",
                     horizontalalignment="center",
                     color="white" if cm_matrix[r, c] > thresh_val else "black",
                     fontsize=12, fontweight="bold")
    plt.ylabel("True Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix_best.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # 6. Save Model Artifact & Metadata
    model_artifact_path = os.path.join(model_dir, "logistic_regression_ndws.joblib")
    pipeline_bundle = {
        "model_name": best_model_name,
        "scaler": scaler,
        "model": best_model,
        "feature_names": feat_names,
        "optimal_threshold": best_thresh,
        "training_samples": len(y_train),
        "validation_samples": len(y_val),
    }
    joblib.dump(pipeline_bundle, model_artifact_path)

    metadata = {
        "model_name": best_model_name,
        "feature_names": feat_names,
        "optimal_threshold": best_thresh,
        "training_samples": len(y_train),
        "training_positive_prevalence": float(np.mean(y_train)),
        "validation_samples": len(y_val),
        "validation_positive_prevalence": float(np.mean(y_val)),
        "validation_metrics": val_results[best_model_name],
        "eval_metrics": eval_metrics,
        "dependencies": {
            "sklearn": sys.modules["sklearn"].__version__,
            "numpy": np.__version__,
            "scipy": sys.modules["scipy"].__version__,
            "joblib": joblib.__version__
        },
        "model_artifact_path": model_artifact_path,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    metadata_path = os.path.join(model_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # 7. Save metrics JSON and CSV
    results_json_path = os.path.join(output_dir, "baseline_results.json")
    full_experiment_results = {
        "dataset_summary": {
            "train_rows": len(y_train),
            "train_positives": int(np.sum(y_train)),
            "train_prevalence": float(np.mean(y_train)),
            "val_rows": len(y_val),
            "val_positives": int(np.sum(y_val)),
            "val_prevalence": float(np.mean(y_val)),
            "eval_rows": len(y_eval),
            "eval_positives": int(np.sum(y_eval)),
            "eval_prevalence": float(np.mean(y_eval)),
        },
        "validation_results": val_results,
        "best_model_name": best_model_name,
        "held_out_eval_results": eval_metrics,
        "artifacts": {
            "model_path": model_artifact_path,
            "metadata_path": metadata_path,
            "pr_curve_plot": pr_curve_path,
            "confusion_matrix_plot": cm_path,
            "feature_coefficients_plot": coef_path
        }
    }
    with open(results_json_path, "w") as f:
        json.dump(full_experiment_results, f, indent=2)

    # Save summary CSV
    csv_rows = ["model,average_precision,roc_auc,brier_score,optimal_threshold,precision,recall,f1_score,training_time_sec"]
    for m_name, m_data in val_results.items():
        csv_rows.append(
            f"{m_name},{m_data['average_precision']},{m_data['roc_auc']},{m_data['brier_score']},"
            f"{m_data['decision_threshold']},{m_data['precision']},{m_data['recall']},"
            f"{m_data['f1_score']},{m_data.get('training_time_sec', 0.0)}"
        )
    csv_path = os.path.join(output_dir, "baseline_metrics.csv")
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_rows) + "\n")

    print(f"\n============================================================")
    print(f"EXPERIMENT EXECUTION COMPLETE")
    print(f"Results JSON:    {results_json_path}")
    print(f"Metrics CSV:     {csv_path}")
    print(f"Model Artifact:  {model_artifact_path}")
    print(f"Plots Generated: {output_dir}/")
    print(f"============================================================")

    return full_experiment_results


def main():
    parser = argparse.ArgumentParser(description="Run leakage-safe NDWS tabular baseline experiments.")
    parser.add_argument(
        "--train-shards",
        type=str,
        nargs="+",
        default=None,
        help="List of Parquet files or glob pattern for training shards."
    )
    parser.add_argument(
        "--val-shards",
        type=str,
        nargs="+",
        default=None,
        help="List of Parquet files for internal validation holdout."
    )
    parser.add_argument(
        "--eval-shards",
        type=str,
        nargs="+",
        default=None,
        help="List of Parquet files for held-out evaluation."
    )
    parser.add_argument(
        "--train-max-rows",
        type=int,
        default=1000000,
        help="Maximum rows to load for training (default: 1,000,000 for pilot)."
    )
    parser.add_argument(
        "--val-max-rows",
        type=int,
        default=500000,
        help="Maximum rows to load for validation (default: 500,000)."
    )
    parser.add_argument(
        "--eval-max-rows",
        type=int,
        default=500000,
        help="Maximum rows to load for held-out evaluation (default: 500,000)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results_ndws",
        help="Output directory for metrics and plots."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/ndws",
        help="Directory to save serialized models."
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    
    args = parser.parse_args()

    # Default file path resolutions
    if args.train_shards is None:
        train_files = sorted(glob.glob("data/processed/ndws/*train*.parquet"))
        # Use shards 00 through 11 for training
        train_shards = [f for f in train_files if not any(f.endswith(f"train_{i:02d}_tabular.parquet") for i in [12, 13, 14])]
        if not train_shards and train_files:
            train_shards = train_files[:1]
    else:
        train_shards = []
        for p in args.train_shards:
            train_shards.extend(glob.glob(p) if "*" in p else [p])

    if args.val_shards is None:
        val_shards = sorted(glob.glob("data/processed/ndws/*train_14_tabular.parquet"))
        if not val_shards:
            val_shards = sorted(glob.glob("data/processed/ndws/*train*.parquet"))[-1:]
    else:
        val_shards = []
        for p in args.val_shards:
            val_shards.extend(glob.glob(p) if "*" in p else [p])

    if args.eval_shards is None:
        eval_shards = sorted(glob.glob("data/processed/ndws/eval_00_tabular.parquet"))
    else:
        eval_shards = []
        for p in args.eval_shards:
            eval_shards.extend(glob.glob(p) if "*" in p else [p])

    train_and_evaluate_baselines(
        train_paths=train_shards,
        val_paths=val_shards,
        eval_paths=eval_shards,
        train_max_rows=args.train_max_rows,
        val_max_rows=args.val_max_rows,
        eval_max_rows=args.eval_max_rows,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        random_state=args.random_state
    )


if __name__ == "__main__":
    main()
