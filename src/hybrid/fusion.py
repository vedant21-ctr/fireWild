"""
fusion.py - Late fusion of tabular and spatial next-day fire probabilities.

The tree models are strongest on engineered context (distance to fire, wind-fire alignment); the spatial networks can
learn front shape directly. Late fusion tests whether their errors differ enough for a combination to help:

    logit(p_hybrid) = a + b1 * logit(p_tree) + b2 * logit(p_spatial)

Protocol: the weights are learned on one half of the validation patches. The other half selects the hybrid among the
candidates in configs/hybrid.yaml and sets its F1 threshold. The test split is used only for the final score.
Single-input stackers (tree only, U-Net only) are the "hybrid without DL input" and "hybrid without AML input" ablations.

Each reported protocol is processed in two scoring regimes: full patches, and the position-controlled windows from
src/aml/leak_test.py (where only LightGBM trained on crops is available as the tabular input).

Writes data/ndws/preds/<protocol>_<split>_hybrid[_crop].npz and experiments/ndws/hybrid/<protocol>/fusion.json.

Usage:
    python -m src.hybrid.fusion
"""

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from src.aml.dataset import REPORTED_PROTOCOLS
from src.aml.leak_test import CROP
from src.aml.metrics import best_f1_threshold, compute_metrics
from src.aml.ndws import BASE_DIR, CACHE_DIR
from src.config import load_config

CFG = load_config("hybrid")
SEED = CFG["seed"]
PRED_DIR = os.path.join(CACHE_DIR, "preds")
OUT_DIR = os.path.join(BASE_DIR, "experiments", "ndws", "hybrid")
SPATIAL_MODELS = ("cnn", "unet", "unet_features")
MAX_FIT_ROWS = 2_000_000
EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def gather(maps, patch, pix):
    """Per-pixel scores from (N, H, W) maps in the tabular row order."""
    return maps.reshape(len(maps), -1)[patch, pix]


def split_val_patches(patch, fit_fraction=None, seed=SEED):
    """Boolean row masks (fit, select). Whole patches go to one side."""
    frac = CFG["fit_fraction"] if fit_fraction is None else fit_fraction
    n = int(patch.max()) + 1
    in_fit = np.random.default_rng(seed).random(n) < frac
    fit = in_fit[patch]
    return fit, ~fit


def design(scores, inputs):
    return np.column_stack([logit(scores[name]) for name in inputs])


def fit_fusion(scores, y, fit_mask, inputs, seed=SEED):
    """Logistic regression on input logits, using at most MAX_FIT_ROWS uniformly sampled fit-half pixels."""
    rows = np.flatnonzero(fit_mask)
    if len(rows) > MAX_FIT_ROWS:
        rows = np.random.default_rng(seed).choice(rows, MAX_FIT_ROWS, replace=False)
    model = LogisticRegression(C=CFG["logistic_C"], max_iter=500)
    model.fit(design({k: v[rows] for k, v in scores.items() if k in inputs}, inputs), y[rows])
    return model


def fusion_predict(model, scores, inputs):
    return model.predict_proba(design(scores, inputs))[:, 1].astype(np.float32)


def load_regime(protocol, split, regime, tree):
    """y, patch, pix and input scores for a split in the "full" or "crop" regime."""
    suffix = "_crop" if regime == "crop" else ""
    d = np.load(os.path.join(PRED_DIR, f"{protocol}_{split}{suffix}.npz"))
    y, patch = d["y"].astype(np.int8), d["patch"]
    pix = d["pix"].astype(np.int64)
    tree_key = "p_lightgbm_crop" if regime == "crop" else f"p_{tree}"
    scores = {"tree": d[tree_key].astype(np.float32)}
    for name in SPATIAL_MODELS:
        maps = np.load(os.path.join(PRED_DIR, f"{protocol}_{name}_{split}{suffix}.npy"))
        scores[name] = gather(maps, patch, pix).astype(np.float32)
    return y, patch, pix, scores


def best_tree(protocol):
    """The tabular model with the best validation PR-AUC among the boosted trees and the random forest."""
    with open(os.path.join(BASE_DIR, "experiments", "ndws", "results", protocol, "results.json")) as f:
        val = json.load(f)["val"]
    return max(("xgboost", "lightgbm", "random_forest"), key=lambda m: val[m]["pr_auc"])


def run_regime(protocol, regime, tree):
    y, patch, _, val_scores = load_regime(protocol, "val", regime, tree)
    yt, patch_t, pix_t, test_scores = load_regime(protocol, "test", regime, tree)
    fit_mask, select_mask = split_val_patches(patch)

    candidates = {**CFG["candidates"], "fusion_tree_only": ["tree"], "fusion_unet_only": ["unet"]}
    report, val_out, test_out = {}, {}, {}
    for name, inputs in candidates.items():
        model = fit_fusion(val_scores, y, fit_mask, inputs)
        pv, pt = fusion_predict(model, val_scores, inputs), fusion_predict(model, test_scores, inputs)
        threshold = best_f1_threshold(y[select_mask], pv[select_mask])
        report[name] = {"inputs": inputs, "intercept": float(model.intercept_[0]),
                        "coefficients": {k: float(v) for k, v in zip(inputs, model.coef_[0], strict=True)},
                        "select_half_pr_auc": float(average_precision_score(y[select_mask], pv[select_mask])),
                        "threshold_from_val": threshold, "test": compute_metrics(yt, pt, threshold)}
        val_out[name], test_out[name] = pv, pt

    # unet_features is hybrid option B: it has no fusion weights but competes in selection
    threshold = best_f1_threshold(y[select_mask], val_scores["unet_features"][select_mask])
    report["unet_features"] = {"inputs": ["unet_features"],
                               "select_half_pr_auc": float(average_precision_score(y[select_mask], val_scores["unet_features"][select_mask])),
                               "threshold_from_val": threshold,
                               "test": compute_metrics(yt, test_scores["unet_features"], threshold)}
    eligible = [*CFG["candidates"], "unet_features"]
    selected = max(eligible, key=lambda name: report[name]["select_half_pr_auc"])
    suffix = "_crop" if regime == "crop" else ""
    np.savez_compressed(os.path.join(PRED_DIR, f"{protocol}_test_hybrid{suffix}.npz"), y=yt, patch=patch_t, pix=pix_t,
                        **{f"p_{k}": v for k, v in test_out.items()})
    return {"tree_input": "lightgbm_crop" if regime == "crop" else tree, "selected_hybrid": selected,
            "crop_size": CROP if regime == "crop" else None, "candidates": report}


def main():
    for protocol in REPORTED_PROTOCOLS:
        tree = best_tree(protocol)
        output = {"protocol": protocol, "best_tree": tree, "full": run_regime(protocol, "full", tree),
                  "crop": run_regime(protocol, "crop", tree)}
        out = os.path.join(OUT_DIR, protocol)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "fusion.json"), "w") as f:
            json.dump(output, f, indent=2)
        print(protocol, "selected:", output["full"]["selected_hybrid"],
              {k: round(v["test"]["pr_auc"], 4) for k, v in output["full"]["candidates"].items()}, flush=True)


if __name__ == "__main__":
    main()
