"""
leak_test.py - Does the patch-centre position leak reach the tabular models?

NDWS patches are centred on fires, so a pixel's position inside the patch predicts the label.
The U-Net can see position through the patch border and is therefore trained on random crops.
The tabular features never use row or column directly, but that alone does not prove they are
safe: windowed features are attenuated near the border and every feature is computed from the
full patch. Three measurements, per protocol, using the protocol's selected LightGBM setting:

  1. add_position   Retrain with row, column and distance-from-centre added. A large gain means
                    position still carries label signal the features did not capture.
  2. recover        Train a LightGBM on the 32 features to predict "pixel is within 8 km of the patch
                    centre". The AUC measures how much position the features encode.
  3. crop_control   Pick one fixed random 32x32 window per held-out patch. Compute features from that
                    window only (so the original centre is at an unknown place), train a LightGBM on
                    random training crops, and score the window pixels. Also score the same pixels
                    with the full-patch LightGBM, and a position-only prior in crop coordinates,
                    which should fall to chance if the crop removes position.

Writes experiments/ndws/results/<protocol>/leak_test.json and predictions for the crop regime to
data/ndws/preds/<protocol>_<split>_crop.npz.

Usage:
    python -m src.aml.leak_test --protocol strict
"""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import load_config

from .dataset import NEG_RATE, REPORTED_PROTOCOLS, _flatten, eval_pixels, iter_patch_batches, patch_refs
from .features import FEATURES, batch_features
from .ndws import CH
from .train import MODEL_DIR, PRED_DIR, RESULTS_DIR, SEED, fit_lightgbm, log, predict, prior_correct

_CTRL = load_config("data")["position_control"]
CROP = _CTRL["crop_size"]
CROP_SEED = _CTRL["seed"]
PATCH = 64
INNER_KM = 8.0


def position_columns(pix, size=PATCH):
    """Row, column and distance from the window centre for flat pixel indices."""
    r, c = np.divmod(np.asarray(pix), size)
    centre = (size - 1) / 2
    return np.column_stack([r, c, np.hypot(r - centre, c - centre)]).astype(np.float32)


def crop_windows(n, size=CROP, seed=CROP_SEED, patch=PATCH):
    """Fixed (row, col) window origins, one per held-out patch."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, patch - size + 1, n), rng.integers(0, patch - size + 1, n)


def crop_raw(raw, rows, cols, size=CROP):
    """Crop raw patches (B, 13, 64, 64) to (B, 13, size, size) with one window per patch."""
    return np.stack([p[:, r:r + size, c:c + size] for p, r, c in zip(raw, rows, cols, strict=True)])


def crop_pixels(protocol, split, rows, cols):
    """Features computed only from each patch's window. Returns X, y, patch index, crop-flat pixel index."""
    Xs, ys, patches, pixels = [], [], [], []
    offset = 0
    for raw in iter_patch_batches(patch_refs(protocol, split)):
        b = len(raw)
        cropped = crop_raw(raw, rows[offset:offset + b], cols[offset:offset + b])
        X, y, pos, pix = _flatten(batch_features(cropped), cropped[:, CH["FireMask"]])
        Xs.append(X)
        ys.append(y)
        patches.append((pos + offset).astype(np.int32))
        pixels.append(pix.astype(np.int16))
        offset += b
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(patches), np.concatenate(pixels)


def full_pixels_in_window(protocol, split, rows, cols):
    """Full-patch features for the window pixels only, aligned to the crop_pixels row order."""
    X, y, patch, pix = eval_pixels(protocol, split)
    r, c = np.divmod(pix.astype(np.int64), PATCH)
    keep = (r >= rows[patch]) & (r < rows[patch] + CROP) & (c >= cols[patch]) & (c < cols[patch] + CROP)
    crop_pix = (r[keep] - rows[patch[keep]]) * CROP + (c[keep] - cols[patch[keep]])
    return X[keep], y[keep], patch[keep], crop_pix.astype(np.int16)


def crop_training_sample(protocol, split, neg_rate=NEG_RATE, seed=SEED):
    """Positives plus sampled negatives from one random window per patch, features from the window only."""
    rng = np.random.default_rng(seed)
    Xs, ys, pixels = [], [], []
    for raw in iter_patch_batches(patch_refs(protocol, split)):
        rows, cols = rng.integers(0, PATCH - CROP + 1, len(raw)), rng.integers(0, PATCH - CROP + 1, len(raw))
        cropped = crop_raw(raw, rows, cols)
        X, y, _, pix = _flatten(batch_features(cropped), cropped[:, CH["FireMask"]])
        keep = (y == 1) | (rng.random(len(y)) < neg_rate)
        Xs.append(X[keep])
        ys.append(y[keep])
        pixels.append(pix[keep])
    X, y, pix = np.concatenate(Xs), np.concatenate(ys), np.concatenate(pixels)
    return X, y, np.where(y == 1, 1.0, 1.0 / neg_rate).astype(np.float32), pix


def crop_position_prior(y_train, pix_train, weights, pix_test):
    """Weighted training fire rate at each rounded distance from the crop centre."""
    def dist(p):
        return np.rint(position_columns(p, CROP)[:, 2]).astype(np.int64)

    size = dist(np.arange(CROP * CROP)).max() + 1
    d = dist(pix_train)
    pos = np.bincount(d, weights=weights * (y_train == 1), minlength=size)
    cnt = np.bincount(d, weights=weights, minlength=size)
    return (pos / np.maximum(cnt, 1e-9))[dist(pix_test)].astype(np.float32)


def add_position_test(protocol, params):
    """Retrain with position columns; PR-AUC with and without them on the same pixels."""
    X, y, _ = position_sample(protocol, "train", SEED)
    Xes, yes, wes = position_sample(protocol, "val", SEED + 1)
    out = {}
    n = len(FEATURES)
    base = fit_lightgbm(X[:, :n], y, Xes[:, :n], yes, wes, params)
    plus = fit_lightgbm(X, y, Xes, yes, wes, params)
    for split in ("val", "test"):
        Xe, ye, _, pix = eval_pixels(protocol, split)
        Xp = np.hstack([Xe, position_columns(pix)])
        a = average_precision_score(ye, predict(base, Xe))
        b = average_precision_score(ye, predict(plus, Xp))
        out[split] = {"pr_auc_features": float(a), "pr_auc_features_plus_position": float(b), "delta": float(b - a)}
    return out


def position_sample(protocol, split, seed):
    """Training-sample rows with position columns appended."""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for raw in iter_patch_batches(patch_refs(protocol, split)):
        X, y, _, pix = _flatten(batch_features(raw), raw[:, CH["FireMask"]])
        keep = (y == 1) | (rng.random(len(y)) < NEG_RATE)
        Xs.append(np.hstack([X[keep], position_columns(pix[keep])]))
        ys.append(y[keep])
    X, y = np.concatenate(Xs), np.concatenate(ys)
    return X, y, np.where(y == 1, 1.0, 1.0 / NEG_RATE).astype(np.float32)


def position_probe_rows(protocol, split, keep_fraction, rng):
    """Uniform pixel sample: features plus "within INNER_KM of the patch centre". Streams to stay within RAM."""
    Xs, inner = [], []
    for raw in iter_patch_batches(patch_refs(protocol, split)):
        X, _, _, pix = _flatten(batch_features(raw), raw[:, CH["FireMask"]])
        keep = rng.random(len(X)) < keep_fraction
        Xs.append(X[keep])
        inner.append((position_columns(pix[keep])[:, 2] <= INNER_KM).astype(np.int8))
    return np.concatenate(Xs), np.concatenate(inner)


def recover_position_test(protocol, params):
    """How well do the 32 features predict being near the patch centre?"""
    import lightgbm as lgb  # lazy: see the OpenMP note in train.fit_xgboost
    rng = np.random.default_rng(SEED)
    Xtr, ytr = position_probe_rows(protocol, "train", 0.04, rng)
    Xva, yva = position_probe_rows(protocol, "val", 0.08, rng)
    Xte, yte = position_probe_rows(protocol, "test", 0.08, rng)
    model = lgb.LGBMClassifier(objective="binary", learning_rate=0.1, n_estimators=2000, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.5, n_jobs=8, random_state=SEED, verbose=-1, **params)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return {"target": f"pixel within {INNER_KM:g} km of patch centre",
            "test_roc_auc": float(roc_auc_score(yte, model.predict_proba(Xte)[:, 1])),
            "test_positive_rate": float(yte.mean()), "n_trees": int(model.best_iteration_)}


def crop_control_test(protocol, params):
    """Train on random crops; score fixed crops with crop-trained and full-patch-trained models."""
    import lightgbm as lgb  # lazy: see the OpenMP note in train.fit_xgboost
    X, y, w, pix_tr = crop_training_sample(protocol, "train", seed=SEED)
    Xes, yes, wes, _ = crop_training_sample(protocol, "val", seed=SEED + 1)
    crop_model = fit_lightgbm(X, y, Xes, yes, wes, params)
    full_model = lgb.Booster(model_file=os.path.join(MODEL_DIR, f"{protocol}_lightgbm.txt"))
    out = {}
    for split in ("val", "test"):
        n = len(patch_refs(protocol, split))
        rows, cols = crop_windows(n)
        Xc, yc, patch, pix = crop_pixels(protocol, split, rows, cols)
        Xf, yf, patch_f, pix_f = full_pixels_in_window(protocol, split, rows, cols)
        if not (np.array_equal(yc, yf) and np.array_equal(patch, patch_f) and np.array_equal(pix, pix_f)):
            raise ValueError("crop and full-patch pixel sets are misaligned")
        p_crop = predict(crop_model, Xc)
        p_full = prior_correct(full_model.predict(Xf))
        p_prior = crop_position_prior(y, pix_tr, w, pix)
        np.savez_compressed(os.path.join(PRED_DIR, f"{protocol}_{split}_crop.npz"), y=yc, patch=patch, pix=pix,
                            p_lightgbm_crop=p_crop, p_lightgbm_full=p_full, p_position_prior=p_prior)
        out[split] = {
            "n_pixels": len(yc), "positive_rate": float(yc.mean()),
            "pr_auc_crop_trained": float(average_precision_score(yc, p_crop)),
            "pr_auc_full_patch_model": float(average_precision_score(yc, p_full)),
            "pr_auc_position_prior": float(average_precision_score(yc, p_prior)),
        }
    return out


def run(protocol):
    results_path = os.path.join(RESULTS_DIR, protocol, "results.json")
    with open(results_path) as f:
        params = json.load(f)["selection"]["lightgbm_best"]["params"]
    log(f"[{protocol}] add-position test")
    report = {"protocol": protocol, "lightgbm_params": params, "add_position": add_position_test(protocol, params)}
    log(f"[{protocol}] position-recovery test")
    report["recover_position"] = recover_position_test(protocol, params)
    log(f"[{protocol}] crop-control test")
    report["crop_control"] = crop_control_test(protocol, params)
    with open(os.path.join(RESULTS_DIR, protocol, "leak_test.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(f"[{protocol}] leak test: {json.dumps(report['add_position']['test'])}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=REPORTED_PROTOCOLS, required=True)
    run(parser.parse_args().protocol)


if __name__ == "__main__":
    main()
