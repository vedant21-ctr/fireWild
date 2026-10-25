"""
train.py - Train and select the Phase 1 tabular models for NDWS next-day fire prediction.

For one protocol (official or spatial):
  1. Build the training sample (all positive pixels + 2% of negatives). Models are fit on the
     subsampled data without weights; their probabilities are then mapped back to the true class
     prior with the closed-form correction odds_true = odds_sample * NEG_RATE (see prior_correct).
     Importance-weighting instead (weight 1/NEG_RATE on negatives) made boosting early-stop after
     a handful of trees and lowered validation PR-AUC (0.214 vs 0.229 for LightGBM). For LR and RF
     both options are fitted and the better one on validation is kept.
  2. Fit baselines and models:
       persistence           p = day-t fire mask (no learning)
       neighbourhood_persist p = fraction of burning pixels in the 3x3 window at day t
       logistic_regression   median imputation + standardisation + L2 logistic regression
       random_forest         bagged trees on a 400k-row subsample
       xgboost               histogram gradient boosting, small grid, early stopping
       lightgbm              leaf-wise gradient boosting, small grid, early stopping
  3. Select each family's configuration and its F1 threshold on the validation split.
  4. Score every known-label test pixel once and save the predictions for evaluate.py.
  5. (--ablation) Retrain LightGBM on feature-group subsets.

Usage:
    python -m src.aml.train --protocol spatial --ablation
    python -m src.aml.train --protocol official
"""

import os
import json
import time
import argparse
import numpy as np
import joblib
import lightgbm as lgb
import xgboost as xgb
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from .ndws import BASE_DIR, CACHE_DIR
from .features import FEATURES, FEATURE_GROUPS, FX
from .dataset import training_sample, eval_pixels, feature_columns, NEG_RATE
from .metrics import best_f1_threshold, compute_metrics

SEED = 42
RF_ROWS = 400_000
RESULTS_DIR = os.path.join(BASE_DIR, "experiments", "ndws", "results")
MODEL_DIR = os.path.join(BASE_DIR, "models", "ndws")
PRED_DIR = os.path.join(CACHE_DIR, "preds")

# Pixels of one patch share near-constant weather values, so small leaves let trees memorise
# individual patches. Large minimum leaf sizes and column subsampling regularise against that.
LGB_COMMON = dict(objective="binary", learning_rate=0.05, n_estimators=3000, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.5, reg_lambda=1.0, metric="average_precision", n_jobs=8,
                  random_state=SEED, verbose=-1)
LGB_GRID = [
    dict(num_leaves=31, min_child_samples=2000),
    dict(num_leaves=63, min_child_samples=2000),
    dict(num_leaves=127, min_child_samples=5000),
]
XGB_COMMON = dict(learning_rate=0.05, n_estimators=3000, subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0,
                  tree_method="hist", eval_metric="aucpr", early_stopping_rounds=100, n_jobs=8, random_state=SEED)
XGB_GRID = [
    dict(max_depth=6, min_child_weight=100),
    dict(max_depth=8, min_child_weight=1000),
]
ABLATIONS = {
    "raw_only": ["raw"],
    "fire_only": ["fire"],
    "raw+fire": ["raw", "fire"],
    "raw+fire+wind": ["raw", "fire", "wind"],
    "raw+fire+wind+terrain": ["raw", "fire", "wind", "terrain"],
    "all_features": list(FEATURE_GROUPS),
}
ABLATIONS.update({f"all_minus_{g}": [h for h in FEATURE_GROUPS if h != g] for g in FEATURE_GROUPS})


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fit_lightgbm(X, y, Xv, yv, wv, params):
    """Early stopping uses a validation pixel sample weighted back to the true class prior."""
    model = lgb.LGBMClassifier(**LGB_COMMON, **params)
    model.fit(X, y, eval_set=[(Xv, yv)], eval_sample_weight=[wv], callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


def fit_xgboost(X, y, Xv, yv, wv, params):
    model = xgb.XGBClassifier(**XGB_COMMON, **params)
    model.fit(X, y, eval_set=[(Xv, yv)], sample_weight_eval_set=[wv], verbose=False)
    return model


def fit_logreg(X, y, w=None):
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=1.0, max_iter=2000))
    model.fit(X, y, logisticregression__sample_weight=w)
    return model


def fit_random_forest(X, y, w=None):
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y), size=min(RF_ROWS, len(y)), replace=False)
    model = make_pipeline(SimpleImputer(strategy="median"),
                          RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=50,
                                                 max_features="sqrt", n_jobs=-1, random_state=SEED))
    model.fit(X[idx], y[idx], randomforestclassifier__sample_weight=None if w is None else w[idx])
    return model


def prior_correct(p, neg_rate=NEG_RATE):
    """Map probabilities learned on negative-subsampled data back to the true prior.

    Keeping each negative with probability r multiplies the odds by 1/r, so
    odds_true = r * odds_sample  =>  p_true = r p / (r p + 1 - p). The map is monotone,
    so PR-AUC / ROC-AUC are unchanged; only calibration (Brier, thresholds) is affected.
    """
    return (neg_rate * p / (neg_rate * p + 1.0 - p)).astype(np.float32)


def predict(model, X, weighted=False):
    """Probabilities on the true prior. Importance-weighted models already are; others are prior-corrected."""
    p = model.predict_proba(X)[:, 1]
    return p.astype(np.float32) if weighted else prior_correct(p)


def heuristic_scores(X):
    return {
        "persistence": X[:, FX["prev_fire"]].astype(np.float32),
        "neighbourhood_persist": X[:, FX["fire_frac_3"]].astype(np.float32),
    }


def run_protocol(protocol, ablation=False):
    out_dir = os.path.join(RESULTS_DIR, protocol)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)
    timings, selection = {}, {}

    log(f"[{protocol}] building training sample (neg_rate={NEG_RATE})")
    X, y, w = training_sample(protocol, "train")
    Xes, yes, wes = training_sample(protocol, "val", seed=SEED + 1)  # early-stopping sample
    log(f"[{protocol}] train rows={len(y):,} positives={int(y.sum()):,}; early-stop rows={len(yes):,}")
    log(f"[{protocol}] scoring features for the full validation split")
    Xv, yv, _, _ = eval_pixels(protocol, "val")
    log(f"[{protocol}] val pixels={len(yv):,} positive rate={yv.mean():.4f}")

    models, val_scores = {}, heuristic_scores(Xv)

    # LR and RF: choose importance-weighted vs unweighted (+ prior correction) training on validation
    weighted = {}
    for family, fit in (("logistic_regression", fit_logreg), ("random_forest", fit_random_forest)):
        best = None
        selection[family] = []
        for use_w in (False, True):
            t = time.time()
            m = fit(X, y, w if use_w else None)
            p = predict(m, Xv, weighted=use_w)
            ap = average_precision_score(yv, p)
            selection[family].append({"importance_weighted": use_w, "val_pr_auc": float(ap), "fit_seconds": time.time() - t})
            log(f"[{protocol}] {family} weighted={use_w} val PR-AUC={ap:.4f}")
            if best is None or ap > best[0]:
                best = (ap, m, p, use_w)
        models[family], val_scores[family], weighted[family] = best[1], best[2], best[3]
        selection[family + "_best"] = {"importance_weighted": best[3]}

    for family, grid, fit in (("xgboost", XGB_GRID, fit_xgboost), ("lightgbm", LGB_GRID, fit_lightgbm)):
        best = None
        selection[family] = []
        for params in grid:
            t = time.time()
            m = fit(X, y, Xes, yes, wes, params)
            p = predict(m, Xv)
            ap = average_precision_score(yv, p)
            n_trees = int(m.best_iteration_ if family == "lightgbm" else m.best_iteration + 1)
            selection[family].append({"params": params, "val_pr_auc": float(ap), "n_trees": n_trees,
                                      "fit_seconds": time.time() - t})
            log(f"[{protocol}] {family} {params} trees={n_trees} val PR-AUC={ap:.4f}")
            if best is None or ap > best[0]:
                best = (ap, m, p, params, n_trees)
        models[family], val_scores[family] = best[1], best[2]
        selection[family + "_best"] = {"params": best[3], "n_trees": best[4]}

    thresholds = {name: best_f1_threshold(yv, p) for name, p in val_scores.items()}
    val_metrics = {name: compute_metrics(yv, p, thresholds[name]) for name, p in val_scores.items()}
    del Xv, val_scores

    log(f"[{protocol}] scoring the test split")
    Xt, yt, patch, pix = eval_pixels(protocol, "test")
    test_scores = heuristic_scores(Xt)
    for name, m in models.items():
        test_scores[name] = predict(m, Xt, weighted=weighted.get(name, False))
    test_metrics = {name: compute_metrics(yt, p, thresholds[name]) for name, p in test_scores.items()}
    for name, mt in test_metrics.items():
        log(f"[{protocol}] TEST {name:<22} PR-AUC={mt['pr_auc']:.4f} ROC-AUC={mt['roc_auc']:.4f} F1={mt['f1']:.4f}")

    np.savez_compressed(os.path.join(PRED_DIR, f"{protocol}_test.npz"), y=yt, patch=patch, pix=pix,
                        **{f"p_{k}": v.astype(np.float16) for k, v in test_scores.items()})
    # 20k-pixel test sample for SHAP / partial-dependence plots
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(yt), 20000, replace=False)
    np.savez_compressed(os.path.join(PRED_DIR, f"{protocol}_test_sample.npz"), X=Xt[idx], y=yt[idx])

    models["lightgbm"].booster_.save_model(os.path.join(MODEL_DIR, f"{protocol}_lightgbm.txt"))
    models["xgboost"].save_model(os.path.join(MODEL_DIR, f"{protocol}_xgboost.json"))
    joblib.dump(models["logistic_regression"], os.path.join(MODEL_DIR, f"{protocol}_logistic_regression.joblib"))
    joblib.dump(models["random_forest"], os.path.join(CACHE_DIR, f"{protocol}_random_forest.joblib"), compress=3)

    results = {
        "protocol": protocol,
        "features": FEATURES,
        "neg_rate": NEG_RATE,
        "train_rows": int(len(y)),
        "train_positive_pixels": int(y.sum()),
        "selection": selection,
        "thresholds_from_val": thresholds,
        "val": val_metrics,
        "test": test_metrics,
        "fit_seconds": timings,
    }

    if ablation:
        del Xt
        results["ablation"] = run_ablation(protocol, X, y, Xes, yes, wes, selection["lightgbm_best"]["params"])

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log(f"[{protocol}] wrote {out_dir}/results.json")
    return results


def run_ablation(protocol, X, y, Xes, yes, wes, params):
    """LightGBM (selected config) on feature-group subsets; PR-AUC on full val and test."""
    Xv, yv, _, _ = eval_pixels(protocol, "val")
    Xt, yt, _, _ = eval_pixels(protocol, "test")
    out = {}
    for name, groups in ABLATIONS.items():
        cols = feature_columns(groups)
        t = time.time()
        m = fit_lightgbm(X[:, cols], y, Xes[:, cols], yes, wes, params)
        out[name] = {
            "groups": groups,
            "n_features": len(cols),
            "val_pr_auc": float(average_precision_score(yv, predict(m, Xv[:, cols]))),
            "test_pr_auc": float(average_precision_score(yt, predict(m, Xt[:, cols]))),
            "n_trees": int(m.best_iteration_),
            "fit_seconds": time.time() - t,
        }
        log(f"[{protocol}] ablation {name:<24} val={out[name]['val_pr_auc']:.4f} test={out[name]['test_pr_auc']:.4f}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["official", "spatial"], required=True)
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()
    run_protocol(args.protocol, ablation=args.ablation)


if __name__ == "__main__":
    main()
