"""
train_lightgbm_ndws.py - LightGBM Wildfire Spread Classifier for NDWS

Phase 1 Task 1C: Gradient-Boosted Decision Tree Baseline.

Evaluates LightGBM against:
1. Dummy Prior Baseline
2. Corrected Spatial Distance Heuristic (raw physical km)
3. Logistic Regression L2 Baseline

Key Principles:
- 12 verified predictor channels (11 environmental + 1 prior-day spatial distance).
- Explicit target: target_new_fire (1 iff PrevFireMask == 0 and FireMask == 1).
- Zero target leakage: FireMask, fire_mask_t1, target_new_fire excluded.
- Corrected distance heuristic: always computes on physical kilometers.
- Validation-only decision threshold selection: sweeps F1 on validation holdout only.
- Official evaluation shard (eval_00) evaluated only once with the frozen model.
- Memory safe: operates on pilot row allocations (1,000,000 train, 500,000 val, 500,000 eval).
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
import lightgbm as lgb
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

from src.models.train_tabular_baselines import (
    FEATURE_COLUMNS,
    FORBIDDEN_COLUMNS,
    TARGET_COLUMN,
    DummyPriorClassifier,
    DistanceHeuristicClassifier,
    load_tabular_data,
    fit_preprocessor,
    find_optimal_threshold,
    evaluate_model_predictions,
    calculate_recall_at_fixed_fpr
)


def train_lightgbm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    scale_pos_weight: Optional[float] = None,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    random_state: int = 42
) -> Tuple[lgb.LGBMClassifier, float]:
    """
    Trains a LightGBM binary classifier with early stopping on the validation set.
    Uses logloss and average precision tracking.
    """
    clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        scale_pos_weight=scale_pos_weight if scale_pos_weight is not None else 1.0,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1
    )

    t0 = time.time()
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["binary_logloss", "average_precision"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )
    fit_time = time.time() - t0
    return clf, fit_time


def run_lightgbm_experiment(
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
    Executes the complete LightGBM benchmark:
    1. Loads train, val, eval datasets.
    2. Evaluates baselines (Dummy Prior, Corrected Distance Heuristic, Logistic Regression L2).
    3. Trains LightGBM Unweighted and LightGBM Weighted (scale_pos_weight).
    4. Selects optimal decision threshold on validation set.
    5. Evaluates best model on held-out evaluation set (eval_00).
    6. Produces plots and serializes model artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n============================================================")
    print(f"PHASE 1 TASK 1C: LIGHTGBM WILDFIRE SPREAD BENCHMARK")
    print(f"============================================================")
    print(f"Train Shards:   {len(train_paths)} files (Max Rows: {train_max_rows:,})")
    print(f"Val Shards:     {len(val_paths)} files (Max Rows: {val_max_rows:,})")
    print(f"Eval Shards:    {len(eval_paths)} files (Max Rows: {eval_max_rows:,})")
    print(f"Output Dir:     {output_dir}")
    print(f"Model Dir:      {model_dir}")

    # 1. Load data
    print("\nLoading datasets...")
    X_train, y_train, feat_names = load_tabular_data(train_paths, max_rows=train_max_rows, random_state=random_state)
    print(f"  X_train: {X_train.shape}, positive prevalence: {np.mean(y_train)*100:.3f}% ({int(np.sum(y_train)):,} positives)")

    X_val, y_val, _ = load_tabular_data(val_paths, max_rows=val_max_rows, random_state=random_state + 1)
    print(f"  X_val:   {X_val.shape}, positive prevalence: {np.mean(y_val)*100:.3f}% ({int(np.sum(y_val)):,} positives)")

    X_eval, y_eval, _ = load_tabular_data(eval_paths, max_rows=eval_max_rows, random_state=random_state + 2)
    print(f"  X_eval:  {X_eval.shape}, positive prevalence: {np.mean(y_eval)*100:.3f}% ({int(np.sum(y_eval)):,} positives)")

    # 2. Scaler fitted strictly on training data (used for Logistic Regression comparison)
    scaler = fit_preprocessor(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_eval_scaled = scaler.transform(X_eval)

    # 3. Model dictionary
    val_results: Dict[str, Any] = {}
    fitted_models: Dict[str, Any] = {}
    val_probs: Dict[str, np.ndarray] = {}

    # Model 1: Dummy Prior
    print("\nEvaluating Baseline 1: Dummy Prior...")
    t0 = time.time()
    dummy = DummyPriorClassifier().fit(X_train, y_train)
    p_dummy = dummy.predict_proba(X_val)[:, 1]
    m_dummy = evaluate_model_predictions(y_val, p_dummy, threshold=float(np.mean(y_train)))
    m_dummy["training_time_sec"] = round(time.time() - t0, 2)
    val_results["Dummy_Prior"] = m_dummy
    val_probs["Dummy_Prior"] = p_dummy
    print(f"  Dummy Prior AP: {m_dummy['average_precision']:.6f} | ROC-AUC: {m_dummy['roc_auc']:.4f} | Brier: {m_dummy['brier_score']:.6f}")

    # Model 2: Corrected Distance Heuristic (strictly raw kilometers)
    print("\nEvaluating Baseline 2: Corrected Distance Heuristic (Raw km)...")
    t0 = time.time()
    dist_idx = feat_names.index("dist_to_prev_fire_km")
    dist_clf = DistanceHeuristicClassifier(dist_col_idx=dist_idx, scale_km=2.0)
    dist_clf.fit(X_train, y_train)
    p_dist = dist_clf.predict_proba(X_val)[:, 1]
    opt_thresh_dist, _ = find_optimal_threshold(y_val, p_dist)
    m_dist = evaluate_model_predictions(y_val, p_dist, threshold=opt_thresh_dist)
    m_dist["training_time_sec"] = round(time.time() - t0, 2)
    val_results["Distance_Heuristic_Corrected"] = m_dist
    val_probs["Distance_Heuristic_Corrected"] = p_dist
    fitted_models["Distance_Heuristic_Corrected"] = dist_clf
    print(f"  Distance Heuristic AP: {m_dist['average_precision']:.6f} | ROC-AUC: {m_dist['roc_auc']:.4f} | "
          f"Brier: {m_dist['brier_score']:.6f} | F1: {m_dist['f1_score']:.4f} at tau={opt_thresh_dist:.4f}")

    # Model 3: Logistic Regression L2 (from baseline)
    print("\nEvaluating Baseline 3: Logistic Regression L2...")
    t0 = time.time()
    from sklearn.linear_model import LogisticRegression
    lr_l2 = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200, random_state=random_state)
    lr_l2.fit(X_train_scaled, y_train)
    p_lr = lr_l2.predict_proba(X_val_scaled)[:, 1]
    opt_thresh_lr, _ = find_optimal_threshold(y_val, p_lr)
    m_lr = evaluate_model_predictions(y_val, p_lr, threshold=opt_thresh_lr)
    m_lr["training_time_sec"] = round(time.time() - t0, 2)
    val_results["Logistic_Regression_L2"] = m_lr
    val_probs["Logistic_Regression_L2"] = p_lr
    fitted_models["Logistic_Regression_L2"] = lr_l2
    print(f"  Logistic Regression L2 AP: {m_lr['average_precision']:.6f} | ROC-AUC: {m_lr['roc_auc']:.4f} | "
          f"Brier: {m_lr['brier_score']:.6f} | F1: {m_lr['f1_score']:.4f} at tau={opt_thresh_lr:.4f}")

    # Model 4: LightGBM Unweighted (Native histogram gradient boosting)
    print("\nTraining Model 4: LightGBM (Unweighted)...")
    lgb_unweighted, fit_time_unw = train_lightgbm_model(
        X_train, y_train, X_val, y_val,
        feature_names=feat_names,
        scale_pos_weight=1.0,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state
    )
    p_lgb_unw = lgb_unweighted.predict_proba(X_val)[:, 1]
    opt_thresh_lgb_unw, _ = find_optimal_threshold(y_val, p_lgb_unw)
    m_lgb_unw = evaluate_model_predictions(y_val, p_lgb_unw, threshold=opt_thresh_lgb_unw)
    m_lgb_unw["training_time_sec"] = round(fit_time_unw, 2)
    m_lgb_unw["best_iteration"] = int(lgb_unweighted.best_iteration_ or 300)
    val_results["LightGBM_Unweighted"] = m_lgb_unw
    val_probs["LightGBM_Unweighted"] = p_lgb_unw
    fitted_models["LightGBM_Unweighted"] = lgb_unweighted
    print(f"  LightGBM Unweighted AP: {m_lgb_unw['average_precision']:.6f} | ROC-AUC: {m_lgb_unw['roc_auc']:.4f} | "
          f"Brier: {m_lgb_unw['brier_score']:.6f} | F1: {m_lgb_unw['f1_score']:.4f} at tau={opt_thresh_lgb_unw:.4f} "
          f"({fit_time_unw:.1f}s, iter={m_lgb_unw['best_iteration']})")

    # Model 5: LightGBM Weighted (scale_pos_weight = 10.0 for class imbalance)
    # Using scale_pos_weight=10 (balanced square-root weighting for smooth probabilities)
    print("\nTraining Model 5: LightGBM (Weighted, scale_pos_weight=10.0)...")
    lgb_weighted, fit_time_w = train_lightgbm_model(
        X_train, y_train, X_val, y_val,
        feature_names=feat_names,
        scale_pos_weight=10.0,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state
    )
    p_lgb_w = lgb_weighted.predict_proba(X_val)[:, 1]
    opt_thresh_lgb_w, _ = find_optimal_threshold(y_val, p_lgb_w)
    m_lgb_w = evaluate_model_predictions(y_val, p_lgb_w, threshold=opt_thresh_lgb_w)
    m_lgb_w["training_time_sec"] = round(fit_time_w, 2)
    m_lgb_w["best_iteration"] = int(lgb_weighted.best_iteration_ or 300)
    val_results["LightGBM_Weighted_spw10"] = m_lgb_w
    val_probs["LightGBM_Weighted_spw10"] = p_lgb_w
    fitted_models["LightGBM_Weighted_spw10"] = lgb_weighted
    print(f"  LightGBM Weighted AP: {m_lgb_w['average_precision']:.6f} | ROC-AUC: {m_lgb_w['roc_auc']:.4f} | "
          f"Brier: {m_lgb_w['brier_score']:.6f} | F1: {m_lgb_w['f1_score']:.4f} at tau={opt_thresh_lgb_w:.4f} "
          f"({fit_time_w:.1f}s, iter={m_lgb_w['best_iteration']})")

    # 4. Select best model on validation AP
    best_model_name = "LightGBM_Unweighted" if val_results["LightGBM_Unweighted"]["average_precision"] >= val_results["LightGBM_Weighted_spw10"]["average_precision"] else "LightGBM_Weighted_spw10"
    best_model = fitted_models[best_model_name]
    best_thresh = val_results[best_model_name]["decision_threshold"]

    print(f"\n============================================================")
    print(f"BEST LIGHTGBM MODEL ON VALIDATION: {best_model_name}")
    print(f"Validation AP: {val_results[best_model_name]['average_precision']:.6f}")
    print(f"Frozen Optimal Threshold: {best_thresh:.6f}")
    print(f"============================================================")

    # 5. Evaluate best model ONCE on held-out evaluation shard (eval_00)
    print("\nEvaluating frozen best LightGBM model on held-out evaluation shard (eval_00)...")
    prob_eval = best_model.predict_proba(X_eval)[:, 1]
    eval_metrics = evaluate_model_predictions(y_eval, prob_eval, threshold=best_thresh)
    print(f"  Held-out Eval AP:  {eval_metrics['average_precision']:.6f}")
    print(f"  Held-out ROC-AUC:  {eval_metrics['roc_auc']:.6f}")
    print(f"  Held-out Brier:    {eval_metrics['brier_score']:.6f}")
    print(f"  Held-out F1 Score: {eval_metrics['f1_score']:.6f} (Precision: {eval_metrics['precision']:.4f}, Recall: {eval_metrics['recall']:.4f})")
    print(f"  Held-out CM:       TN={eval_metrics['confusion_matrix']['tn']:,}, FP={eval_metrics['confusion_matrix']['fp']:,}, "
          f"FN={eval_metrics['confusion_matrix']['fn']:,}, TP={eval_metrics['confusion_matrix']['tp']:,}")

    # 6. Generate Diagnostic Plots
    print("\nGenerating diagnostic plots...")

    # Plot A: PR Curves Comparison
    plt.figure(figsize=(9, 6))
    for name, prob in val_probs.items():
        prec, rec, _ = precision_recall_curve(y_val, prob)
        ap = val_results[name]["average_precision"]
        plt.plot(rec, prec, label=f"{name.replace('_', ' ')} (AP = {ap:.4f})", linewidth=2.0)

    prev_val = float(np.mean(y_val))
    plt.axhline(prev_val, color="gray", linestyle="--", label=f"Random Prior ({prev_val*100:.2f}%)")
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall Comparison — LightGBM vs Baselines (Validation)", fontsize=13, fontweight="bold")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_plot_path = os.path.join(output_dir, "lightgbm_pr_curves.png")
    plt.savefig(pr_plot_path, dpi=150)
    plt.close()

    # Plot B: Feature Importances (Gain)
    importance_gain = best_model.booster_.feature_importance(importance_type="gain")
    sorted_idx = np.argsort(importance_gain)
    plt.figure(figsize=(10, 6))
    plt.barh(np.array(feat_names)[sorted_idx], importance_gain[sorted_idx], color="#27ae60", alpha=0.85)
    plt.xlabel("Total Gain", fontsize=11)
    plt.title(f"LightGBM Feature Importance (Gain) — {best_model_name}", fontsize=13, fontweight="bold")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    feat_plot_path = os.path.join(output_dir, "lightgbm_feature_importance.png")
    plt.savefig(feat_plot_path, dpi=150)
    plt.close()

    # Plot C: Confusion Matrix (Eval Set at Frozen Threshold)
    cm = eval_metrics["confusion_matrix"]
    cm_matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_matrix, interpolation="nearest", cmap=plt.cm.Greens)
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
    cm_plot_path = os.path.join(output_dir, "lightgbm_confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()

    # 7. Serialize Best Model and Metadata
    model_artifact_path = os.path.join(model_dir, "lightgbm_ndws.joblib")
    bundle = {
        "model_name": best_model_name,
        "model": best_model,
        "feature_names": feat_names,
        "optimal_threshold": best_thresh,
        "best_iteration": int(best_model.best_iteration_ or 300),
        "training_samples": len(y_train),
        "validation_samples": len(y_val),
    }
    joblib.dump(bundle, model_artifact_path)

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
        "feature_importance_gain": {
            feat_names[i]: float(importance_gain[i]) for i in range(len(feat_names))
        },
        "dependencies": {
            "lightgbm": lgb.__version__,
            "sklearn": sys.modules["sklearn"].__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__
        },
        "model_artifact_path": model_artifact_path,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    metadata_path = os.path.join(model_dir, "lightgbm_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Full results JSON
    results_json = {
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
            "pr_curve_plot": pr_plot_path,
            "feature_importance_plot": feat_plot_path,
            "confusion_matrix_plot": cm_plot_path
        }
    }
    results_json_path = os.path.join(output_dir, "lightgbm_results.json")
    with open(results_json_path, "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"\n============================================================")
    print(f"LIGHTGBM EXPERIMENT COMPLETE")
    print(f"Results JSON:    {results_json_path}")
    print(f"Model Artifact:  {model_artifact_path}")
    print(f"Plots Generated: {output_dir}/")
    print(f"============================================================")

    return results_json


def main():
    parser = argparse.ArgumentParser(description="Run LightGBM wildfire spread baseline experiments.")
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
        help="Maximum rows to load for training (default: 1,000,000)."
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

    run_lightgbm_experiment(
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
