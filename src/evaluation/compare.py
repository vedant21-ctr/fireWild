"""
compare.py - Unified test-set comparison of the tabular, spatial and hybrid models.

Two prediction targets are reported, never mixed:
    active      FireMask(t+1) on every pixel with a known label (includes fire persistence)
    transition  FireMask(t+1) on pixels known not burning at day t (persistence excluded)

All models are trained on the active target. For a pixel not burning at t, P(burning at t+1) is the probability of a
0 -> 1 transition, so the same scores are evaluated on both targets without using test labels to choose pixels.

Two scoring regimes are reported:
    full      every known pixel of the 64x64 test patches
    crop      one fixed random 32x32 window per test patch, every feature computed from the window only

The patch-centre prior (position only) and the full-patch U-Net are leak diagnostics, not valid forecasters.

Writes, under experiments/ndws/final/: metrics.csv, significance.json, seed_stability.csv, model_comparison.png,
<protocol>/ figures and error analysis; also experiments/ndws/FINAL_RESULTS.md.

Usage:
    python -m src.evaluation.compare
"""

import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve

from src.aml.dataset import REPORTED_PROTOCOLS, eval_pixels, iter_patch_batches, patch_refs
from src.aml.features import FX
from src.aml.metrics import (
    FPR_LEVELS,
    bootstrap_pr_auc,
    compute_metrics,
    paired_bootstrap_delta,
    recall_at_fpr,
)
from src.aml.ndws import BASE_DIR, CACHE_DIR, CH, load_split
from src.aml.splits import load_manifest
from src.config import load_config
from src.dl.data import LABELS_PATH, global_indices

N_BOOT = load_config("aml")["evaluation"]["n_boot"]
PRED_DIR = os.path.join(CACHE_DIR, "preds")
EXP_DIR = os.path.join(BASE_DIR, "experiments", "ndws")
OUT_DIR = os.path.join(EXP_DIR, "final")
DL_DIR = os.path.join(EXP_DIR, "dl")
TABULAR = ["dummy", "persistence", "neighbourhood_persist", "logistic_regression", "random_forest", "xgboost", "lightgbm"]
SPATIAL = ["cnn", "unet", "unet_features"]
FUSION = ["fusion_tree_unet", "fusion_tree_unet_features", "fusion_tree_cnn", "fusion_tree_only", "fusion_unet_only"]
DIAGNOSTIC = ["centre_prior", "unet_fullpatch"]
FULL_MODELS = [*TABULAR, *SPATIAL, *FUSION, "hybrid", *DIAGNOSTIC]
CROP_MODELS = ["position_prior", "lightgbm_full", "lightgbm_crop", *SPATIAL, *FUSION, "hybrid"]
LABELS = {
    "dummy": "Dummy (training rate)", "persistence": "Persistence", "neighbourhood_persist": "Neighbourhood persistence",
    "logistic_regression": "Logistic regression", "random_forest": "Random forest", "xgboost": "XGBoost", "lightgbm": "LightGBM",
    "cnn": "Compact CNN", "unet": "U-Net", "unet_features": "U-Net + engineered channels",
    "fusion_tree_unet": "Fusion: tree + U-Net", "fusion_tree_unet_features": "Fusion: tree + feature U-Net",
    "fusion_tree_cnn": "Fusion: tree + CNN", "fusion_tree_only": "Fusion input: tree only", "fusion_unet_only": "Fusion input: U-Net only",
    "hybrid": "Hybrid (selected on validation)", "centre_prior": "Patch-centre prior (leak probe)",
    "unet_fullpatch": "U-Net, full patches (leak probe)", "position_prior": "Position prior in window (leak probe)",
    "lightgbm_full": "LightGBM (full-patch model)", "lightgbm_crop": "LightGBM (window-trained)",
}
COLORS = {"dummy": "#dee2e6", "persistence": "#adb5bd", "neighbourhood_persist": "#6c757d", "logistic_regression": "#00798c",
          "random_forest": "#edae49", "xgboost": "#30638e", "lightgbm": "#d1495b", "cnn": "#8ac926", "unet": "#6a4c93",
          "unet_features": "#b5179e", "hybrid": "#111111", "centre_prior": "#bc6c25", "unet_fullpatch": "#b8a1d9"}
CURVE_MODELS = ["persistence", "logistic_regression", "random_forest", "xgboost", "lightgbm", "cnn", "unet", "unet_features", "hybrid"]
TARGETS = {"active": "next-day active fire (all known pixels)", "transition": "new fire (pixels not burning at day t)"}
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})


def target_masks(prev):
    """Row masks for the two targets. Transition keeps only pixels known not burning at t."""
    return {"active": np.ones_like(prev, dtype=bool), "transition": prev == 0}


def gather_maps(maps, patch, pix):
    """Per-pixel scores from (N, H, W) maps in the tabular row order (patch index, flat pixel index)."""
    return maps.reshape(len(maps), -1)[patch, pix]


def centre_distance(pix, size=64):
    r, c = np.divmod(np.asarray(pix), size)
    return np.rint(np.hypot(r - (size - 1) / 2, c - (size - 1) / 2)).astype(np.int64)


def centre_prior(y_train, pix):
    """Leak probe: fire rate in TRAINING labels at each (rounded) distance from the patch centre."""
    d = centre_distance(np.arange(64 * 64))
    flat = np.asarray(y_train).reshape(len(y_train), -1)
    known = flat >= 0
    grid = np.broadcast_to(d, flat.shape)
    pos = np.bincount(grid[known], weights=(flat[known] == 1), minlength=d.max() + 1)
    cnt = np.bincount(grid[known], minlength=d.max() + 1)
    return (pos / np.maximum(cnt, 1))[centre_distance(pix)].astype(np.float32)


def read_json(*parts):
    with open(os.path.join(*parts)) as f:
        return json.load(f)


def prev_fire_pixels(protocol, patch, pix, crop=False):
    """Day-t fire mask value (1 / 0 / -1) at each (patch, pixel) of the test split."""
    from src.aml.leak_test import CROP, crop_raw, crop_windows

    refs = patch_refs(protocol, "test")
    rows, cols = crop_windows(len(refs))
    out, offset = [], 0
    for raw in iter_patch_batches(refs):
        b = len(raw)
        if crop:
            raw = crop_raw(raw, rows[offset:offset + b], cols[offset:offset + b], CROP)
        out.append(raw[:, CH["PrevFireMask"]].reshape(b, -1))
        offset += b
    return np.concatenate(out)[patch, pix].astype(np.int8)


def selected_score(protocol, regime, scores, fusion):
    name = fusion[regime]["selected_hybrid"]
    return scores[name], name


def load_full(protocol, fusion):
    """y, patch, pix, scores and val-derived thresholds for the full-patch regime."""
    d = np.load(os.path.join(PRED_DIR, f"{protocol}_test.npz"))
    y, patch, pix = d["y"].astype(np.int8), d["patch"], d["pix"].astype(np.int64)
    scores = {m: d[f"p_{m}"].astype(np.float32) for m in TABULAR}
    thresholds = dict(read_json(EXP_DIR, "results", protocol, "results.json")["thresholds_from_val"])
    for name in [*SPATIAL, "unet_fullpatch"]:
        scores[name] = gather_maps(np.load(os.path.join(PRED_DIR, f"{protocol}_{name}_test.npy")), patch, pix).astype(np.float32)
        thresholds[name] = read_json(DL_DIR, protocol, name, "results.json")["threshold_from_val"]
    h = np.load(os.path.join(PRED_DIR, f"{protocol}_test_hybrid.npz"))
    for name in FUSION:
        scores[name] = h[f"p_{name}"].astype(np.float32)
        thresholds[name] = fusion["full"]["candidates"][name]["threshold_from_val"]
    scores["hybrid"], chosen = selected_score(protocol, "full", scores, fusion)
    thresholds["hybrid"] = fusion["full"]["candidates"][chosen]["threshold_from_val"]
    labels = np.load(LABELS_PATH, mmap_mode="r")[global_indices(protocol, "train")]
    scores["centre_prior"] = centre_prior(labels, pix)
    return y, patch, pix, scores, thresholds


def load_crop(protocol, fusion):
    """Same as load_full for the fixed-window regime."""
    d = np.load(os.path.join(PRED_DIR, f"{protocol}_test_crop.npz"))
    y, patch, pix = d["y"].astype(np.int8), d["patch"], d["pix"].astype(np.int64)
    scores = {"position_prior": d["p_position_prior"].astype(np.float32), "lightgbm_full": d["p_lightgbm_full"].astype(np.float32),
              "lightgbm_crop": d["p_lightgbm_crop"].astype(np.float32)}
    for name in SPATIAL:
        scores[name] = gather_maps(np.load(os.path.join(PRED_DIR, f"{protocol}_{name}_test_crop.npy")), patch, pix).astype(np.float32)
    h = np.load(os.path.join(PRED_DIR, f"{protocol}_test_hybrid_crop.npz"))
    for name in FUSION:
        scores[name] = h[f"p_{name}"].astype(np.float32)
    scores["hybrid"], _ = selected_score(protocol, "crop", scores, fusion)
    return y, patch, pix, scores


def threshold_free(y, p):
    out = {"pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
           "brier": float(brier_score_loss(y, p)), "positive_rate": float(y.mean()), "n_pixels": len(y)}
    out["pr_auc_lift"] = out["pr_auc"] / out["positive_rate"]
    for lvl in FPR_LEVELS:
        out[f"recall_at_{int(lvl * 100)}pct_fpr"] = recall_at_fpr(y, p, lvl)
    return out


def metric_rows(protocol, regime, y, patch, prev, scores, models, thresholds=None):
    rows = []
    for target, mask in target_masks(prev).items():
        yt, pt = y[mask], patch[mask]
        for m in models:
            met = threshold_free(yt, scores[m][mask])
            b = bootstrap_pr_auc(yt, scores[m][mask], pt, n_boot=N_BOOT)
            row = {"protocol": protocol, "regime": regime, "target": target, "model": m, **met,
                   "pr_auc_ci_low": b["ci_low"], "pr_auc_ci_high": b["ci_high"]}
            if target == "active" and thresholds and m in thresholds:
                full = compute_metrics(yt, scores[m][mask], thresholds[m])
                row.update({k: full[k] for k in ("f1", "precision", "recall", "threshold", "tn", "fp", "fn", "tp")})
            rows.append(row)
    return rows


def significance(y, patch, prev, scores, pairs):
    out = {}
    for target, mask in target_masks(prev).items():
        out[target] = {f"{a} - {b}": paired_bootstrap_delta(y[mask], scores[a][mask], scores[b][mask], patch[mask], n_boot=N_BOOT)
                       for a, b in pairs}
    return out


def curves_figure(protocol, y, scores):
    out = os.path.join(OUT_DIR, protocol)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for m in CURVE_MODELS:
        p, r, _ = precision_recall_curve(y, scores[m])
        axes[0].plot(r, p, color=COLORS[m], lw=2.0 if m in ("hybrid", "unet") else 1.1, label=LABELS[m])
        fpr, tpr, _ = roc_curve(y, scores[m])
        axes[1].plot(fpr, tpr, color=COLORS[m], lw=2.0 if m in ("hybrid", "unet") else 1.1, label=LABELS[m])
    axes[0].axhline(y.mean(), color="k", ls=":", lw=1, label=f"chance ({y.mean():.4f})")
    axes[0].set(xlabel="Recall", ylabel="Precision", title=f"Precision-recall, {protocol} test")
    axes[1].set(xscale="log", xlim=(1e-4, 1), xlabel="False positive rate (log)", ylabel="Recall", title=f"ROC, {protocol} test")
    for lvl in FPR_LEVELS:
        axes[1].axvline(lvl, color="k", ls=":", lw=0.8)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pr_roc_curves.png"), bbox_inches="tight")
    plt.close(fig)


def confusion_figure(protocol, table):
    sub = table[(table.protocol == protocol) & (table.regime == "full") & (table.target == "active")].set_index("model")
    models = ["xgboost", "cnn", "unet", "hybrid"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, m in zip(axes, models, strict=True):
        cm = np.array([[sub.loc[m, "tn"], sub.loc[m, "fp"]], [sub.loc[m, "fn"], sub.loc[m, "tp"]]], dtype=np.int64)
        ax.imshow(np.log10(cm + 1), cmap="Blues")
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, f"{v:,}", ha="center", va="center", fontsize=9)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(LABELS[m], fontsize=10)
    fig.suptitle(f"Confusion matrices at validation-chosen thresholds, {protocol} test (colour = log10 count)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, protocol, "confusion_matrices.png"), bbox_inches="tight")
    plt.close(fig)


def threshold_figure(protocol, y, scores, thresholds):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, m in zip(axes, ("xgboost", "unet", "hybrid"), strict=True):
        p, r, t = precision_recall_curve(y, scores[m])
        t = np.r_[t, 1.0]
        idx = np.linspace(0, len(t) - 1, 4000).astype(int)
        f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
        ax.plot(t[idx], p[idx], label="precision")
        ax.plot(t[idx], r[idx], label="recall")
        ax.plot(t[idx], f1[idx], label="F1")
        ax.axvline(thresholds[m], color="k", ls=":", label="val threshold")
        ax.set(xscale="log", xlim=(1e-3, 1), xlabel="decision threshold (log)", title=LABELS[m])
        ax.legend(fontsize=8)
    fig.suptitle(f"Threshold analysis, {protocol} test")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, protocol, "threshold_analysis.png"), bbox_inches="tight")
    plt.close(fig)


def training_curves(protocol):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for name in SPATIAL:
        hist = read_json(DL_DIR, protocol, name, "results.json")["history"]
        epochs = [h["epoch"] for h in hist]
        axes[0].plot(epochs, [h["train_loss"] for h in hist], label=LABELS[name], color=COLORS[name])
        axes[1].plot(epochs, [h["val_pr_auc"] for h in hist], label=LABELS[name], color=COLORS[name])
    axes[0].set(xlabel="epoch", ylabel="training loss", title="Training loss")
    axes[1].set(xlabel="epoch", ylabel="validation PR-AUC", title="Validation PR-AUC")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Training history, {protocol}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, protocol, "training_curves.png"), bbox_inches="tight")
    plt.close(fig)


def pixels_to_map(pix, values, fill=np.nan, size=64):
    img = np.full(size * size, fill, dtype=np.float32)
    img[pix] = values
    return img.reshape(size, size)


def map_figure(protocol, y, patch, pix, scores, thresholds, n=4):
    """Side-by-side maps for high-activity test patches, plus false-positive and false-negative maps for the hybrid."""
    refs = patch_refs(protocol, "test")
    pos = np.bincount(patch, weights=(y == 1), minlength=len(refs))
    order = np.argsort(-pos)
    picks = order[np.linspace(2, max(3, int(np.sum(pos > 20)) - 1), n).astype(int)]
    mask_cmap = matplotlib.colors.ListedColormap(["black", "#e9ecef", "#d1495b"])
    err_cmap = matplotlib.colors.ListedColormap(["#ffffff", "#e63946"])
    panels = ["Fire at t", "Truth t+1", "XGBoost", "CNN", "U-Net", "Hybrid", "Hybrid FP", "Hybrid FN"]
    fig, axes = plt.subplots(n, len(panels), figsize=(18, 2.4 * n))
    for r, i in enumerate(picks):
        split, idx = refs[i]
        raw = np.asarray(load_split(split)[idx])
        sel = patch == i

        pred = scores["hybrid"][sel] >= thresholds["hybrid"]
        truth = y[sel] == 1
        prob_maps = [pixels_to_map(pix[sel], scores[m][sel]) for m in ("xgboost", "cnn", "unet", "hybrid")]
        images = [raw[CH["PrevFireMask"]], raw[CH["FireMask"]], *prob_maps,
                  pixels_to_map(pix[sel], pred & ~truth, 0.0), pixels_to_map(pix[sel], ~pred & truth, 0.0)]
        for k, (img, title) in enumerate(zip(images, panels, strict=True)):
            ax = axes[r, k]
            if k < 2:
                ax.imshow(img, cmap=mask_cmap, vmin=-1, vmax=1, origin="lower", interpolation="nearest")
            elif k < 6:
                im = ax.imshow(img, cmap="inferno", vmin=0, vmax=1, origin="lower", interpolation="nearest")
            else:
                ax.imshow(img, cmap=err_cmap, vmin=0, vmax=1, origin="lower", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=10)
        axes[r, 0].set_ylabel(f"{split} #{idx}", fontsize=8)
    fig.colorbar(im, ax=axes[:, 2:6], shrink=0.6, label="predicted probability")
    fig.suptitle(f"Test probability and error maps, {protocol} (patch grid coordinates; NDWS has no georeferencing)")
    fig.savefig(os.path.join(OUT_DIR, protocol, "prediction_maps.png"), bbox_inches="tight")
    plt.close(fig)


def by_bin(y, mask_bins, p, threshold):
    """Precision and recall inside each boolean bin."""
    out = {}
    for label, mask in mask_bins.items():
        yt, yh = y[mask], p[mask] >= threshold
        tp = int(np.sum(yt & yh))
        out[label] = {"n_pixels": int(mask.sum()), "positive_rate": float(yt.mean()) if len(yt) else 0.0,
                      "recall": tp / max(int(yt.sum()), 1), "precision": tp / max(int(yh.sum()), 1)}
    return out


def error_analysis(protocol, y, patch, scores, thresholds):
    """Where the models err: by distance to the day-t front, by patch fire size and by location group."""
    X, y2, patch2, _ = eval_pixels(protocol, "test")
    if not (np.array_equal(y, y2) and np.array_equal(patch, patch2)):
        raise ValueError("prediction rows do not align with the test features")
    d, fire_count = X[:, FX["dist_to_fire_km"]], np.expm1(X[:, FX["log_patch_fire_count"]])
    wind, not_burning = X[:, FX["fire_wind_alignment"]], X[:, FX["prev_fire"]] == 0
    del X
    edges = [0.5, 1.5, 3.5, 7, 15, 100]
    dist_bins = {f"{a:g}-{b:g} km": not_burning & (d > a) & (d <= b) for a, b in zip([0, *edges[:-1]], edges, strict=True)}
    size_bins = {"small fire (<= 10 px)": fire_count <= 10, "medium (11-50 px)": (fire_count > 10) & (fire_count <= 50),
                 "large (> 50 px)": fire_count > 50}
    models = ["xgboost", "unet", "hybrid"]
    out = {"by_distance": {m: by_bin(y, dist_bins, scores[m], thresholds[m]) for m in models},
           "by_fire_size": {m: by_bin(y, size_bins, scores[m], thresholds[m]) for m in models}}

    near = not_burning & (d > 0) & (d <= 3)
    out["plausibility_near_fire"] = {}
    for label, cond in (("downwind (alignment > 0.5)", wind > 0.5), ("upwind (alignment < -0.5)", wind < -0.5)):
        sel = near & cond
        out["plausibility_near_fire"][label] = {"observed": float(y[sel].mean()), **{m: float(scores[m][sel].mean()) for m in models}}

    groups = np.asarray(load_manifest_groups(protocol))[patch]
    per_group = {}
    for m in models:
        ap = []
        for g in np.unique(groups):
            sel = groups == g
            if y[sel].sum() >= 20:
                ap.append(average_precision_score(y[sel], scores[m][sel]))
        per_group[m] = {"n_groups": len(ap), "mean_group_pr_auc": float(np.mean(ap)), "median_group_pr_auc": float(np.median(ap))}
    out["by_location_group"] = per_group

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    x = np.arange(len(dist_bins))
    for k, m in enumerate(models):
        axes[0].bar(x + (k - 1) * 0.27, [out["by_distance"][m][b]["recall"] for b in dist_bins], 0.27, label=LABELS[m], color=COLORS[m])
        axes[1].bar(x + (k - 1) * 0.27, [out["by_distance"][m][b]["precision"] for b in dist_bins], 0.27, label=LABELS[m], color=COLORS[m])
    for ax, name in zip(axes, ("recall", "precision"), strict=True):
        ax.set_xticks(x, list(dist_bins), fontsize=8)
        ax.set(xlabel="distance to day-t fire", ylabel=name)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"New-fire errors by distance from the front, {protocol} test")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, protocol, "error_by_distance.png"), bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(OUT_DIR, protocol, "error_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def load_manifest_groups(protocol):
    m = load_manifest()
    lookup = {(a, int(b)): g for a, b, g in zip(m.official_split, m.official_index, m.location_group, strict=True)}
    return [lookup[(a, int(b))] for a, b in patch_refs(protocol, "test")]


def seed_stability():
    rows = []
    for protocol in REPORTED_PROTOCOLS:
        for name in SPATIAL:
            base = os.path.join(DL_DIR, protocol, name)
            runs = {"42": read_json(base, "results.json")}
            for f in sorted(os.listdir(base)):
                if f.startswith("results_seed"):
                    runs[f[len("results_seed"):-len(".json")]] = read_json(base, f)
            values = [r["test"]["pr_auc"] for r in runs.values()]
            rows.append({"protocol": protocol, "model": name, "n_seeds": len(values), "mean_test_pr_auc": float(np.mean(values)),
                         "std_test_pr_auc": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
                         "min_test_pr_auc": float(np.min(values)), "max_test_pr_auc": float(np.max(values))})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "seed_stability.csv"), index=False)
    return df


def summary_figure(table):
    models = ["xgboost", "lightgbm", "cnn", "unet", "unet_features", "hybrid"]
    sub = table[(table.regime == "full") & (table.target == "active")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    for ax, protocol in zip(axes, REPORTED_PROTOCOLS, strict=True):
        s = sub[sub.protocol == protocol].set_index("model").loc[models]
        err = np.vstack([s.pr_auc - s.pr_auc_ci_low, s.pr_auc_ci_high - s.pr_auc])
        ax.bar(np.arange(len(models)), s.pr_auc, yerr=err, capsize=3, color=[COLORS[m] for m in models])
        ax.set_xticks(np.arange(len(models)), [LABELS[m] for m in models], rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{protocol} (prevalence {s.positive_rate.iloc[0]:.2%})")
        ax.set_ylabel("Test PR-AUC (95% CI)")
    fig.suptitle("Trees, networks and the hybrid across the three test protocols (active target, full patches)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "model_comparison.png"), bbox_inches="tight")
    plt.close(fig)


def fmt_row(r):
    extra = " | ".join(f"{r[k]:.3f}" if k in r and pd.notna(r[k]) else "–" for k in ("precision", "recall", "f1"))
    return (f"| {LABELS[r['model']]} | {r['pr_auc']:.3f} [{r['pr_auc_ci_low']:.3f}, {r['pr_auc_ci_high']:.3f}] | {r['pr_auc_lift']:.1f}x | "
            f"{r['roc_auc']:.3f} | {extra} | {r['recall_at_1pct_fpr']:.3f} | {r['recall_at_5pct_fpr']:.3f} | {r['brier']:.4f} |")


def write_markdown(table, sigs, seeds, fusions):
    lines = ["# Final model comparison", "", "Generated by `python -m src.evaluation.compare`. Do not edit by hand.", "",
             f"PR-AUC 95% CIs: patch bootstrap ({N_BOOT} resamples). Lift = PR-AUC / positive rate. Thresholded metrics use the "
             "validation-chosen threshold. Leak probes are diagnostics, not valid forecasters.", ""]
    for protocol in REPORTED_PROTOCOLS:
        lines += [f"## `{protocol}` protocol", "", f"Selected hybrid: `{fusions[protocol]['full']['selected_hybrid']}`; best tree: "
                  f"`{fusions[protocol]['best_tree']}`.", ""]
        for regime in ("full", "crop"):
            for target in TARGETS:
                sub = table[(table.protocol == protocol) & (table.regime == regime) & (table.target == target)]
                lines += [f"### {regime} regime, target `{target}`", "", f"Test pixels: {int(sub.n_pixels.iloc[0]):,}; positive rate "
                          f"{sub.positive_rate.iloc[0]:.4f}.", "",
                          "| Model | PR-AUC [95% CI] | Lift | ROC-AUC | Precision | Recall | F1 | Recall@1%FPR | Recall@5%FPR | Brier |",
                          "|---|---|---|---|---|---|---|---|---|---|"]
                lines += [fmt_row(r) for r in sub.to_dict("records")]
                lines += ["", "| Paired comparison | ΔPR-AUC | 95% CI | P(Δ ≤ 0) |", "|---|---|---|---|"]
                for k, s in sigs[protocol][regime][target].items():
                    lines.append(f"| {k} | {s['delta']:+.4f} | [{s['ci_low']:+.4f}, {s['ci_high']:+.4f}] | {s['p_le_zero']:.3f} |")
                lines.append("")
    lines += ["## Seed stability (test PR-AUC, active target, full patches)", "",
              "| Protocol | Model | Seeds | Mean | SD | Min | Max |", "|---|---|---|---|---|---|---|"]
    for r in seeds.to_dict("records"):
        lines.append(f"| {r['protocol']} | {LABELS[r['model']]} | {r['n_seeds']} | {r['mean_test_pr_auc']:.4f} | {r['std_test_pr_auc']:.4f} | "
                     f"{r['min_test_pr_auc']:.4f} | {r['max_test_pr_auc']:.4f} |")
    with open(os.path.join(EXP_DIR, "FINAL_RESULTS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows, sigs, fusions = [], {}, {}
    for protocol in REPORTED_PROTOCOLS:
        os.makedirs(os.path.join(OUT_DIR, protocol), exist_ok=True)
        fusion = read_json(EXP_DIR, "hybrid", protocol, "fusion.json")
        fusions[protocol] = fusion
        tree = fusion["best_tree"]
        y, patch, pix, scores, thresholds = load_full(protocol, fusion)
        prev = prev_fire_pixels(protocol, patch, pix)
        rows += metric_rows(protocol, "full", y, patch, prev, scores, FULL_MODELS, thresholds)
        pairs = [("unet", tree), ("cnn", tree), ("unet_features", tree), ("hybrid", tree), ("hybrid", "unet"), ("unet", "cnn"),
                 ("unet_features", "unet"), ("unet_fullpatch", "unet"), ("lightgbm", "logistic_regression"), ("xgboost", "lightgbm")]
        sigs[protocol] = {"full": significance(y, patch, prev, scores, list(dict.fromkeys(p for p in pairs if p[0] != p[1])))}
        curves_figure(protocol, y, scores)
        threshold_figure(protocol, y, scores, thresholds)
        training_curves(protocol)
        map_figure(protocol, y, patch, pix, scores, thresholds)
        error_analysis(protocol, y, patch, scores, thresholds)
        del scores

        yc, patch_c, pix_c, scores_c = load_crop(protocol, fusion)
        prev_c = prev_fire_pixels(protocol, patch_c, pix_c, crop=True)
        rows += metric_rows(protocol, "crop", yc, patch_c, prev_c, scores_c, CROP_MODELS)
        sigs[protocol]["crop"] = significance(yc, patch_c, prev_c, scores_c,
                                              [("lightgbm_crop", "lightgbm_full"), ("unet", "lightgbm_crop"),
                                               ("unet_features", "lightgbm_crop"), ("hybrid", "lightgbm_crop"), ("hybrid", "unet")])
        del scores_c
        table = pd.DataFrame(rows)
        table.to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False)
        print(f"[{protocol}] done", flush=True)
    table = pd.DataFrame(rows)
    for protocol in REPORTED_PROTOCOLS:
        confusion_figure(protocol, table)
    summary_figure(table)
    with open(os.path.join(OUT_DIR, "significance.json"), "w") as f:
        json.dump(sigs, f, indent=2)
    seeds = seed_stability()
    write_markdown(table, sigs, seeds, fusions)


if __name__ == "__main__":
    main()
