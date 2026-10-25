"""
evaluate.py - Dual-target test evaluation of the U-Net against the tabular models.

The project distinguishes two prediction targets (docs/NDWS_DATA_AUDIT.md, section 8):

  active      next-day ACTIVE fire: FireMask(t+1) on every pixel with a known label.
              Includes pixels already burning at day t (fire persistence).
  transition  NEW fire: FireMask(t+1) on pixels that were known NOT burning at day t
              (PrevFireMask == 0). Pixels burning at t, or with an unknown day-t label,
              are excluded, so persistence cannot score.

All models are trained on the active target. For a pixel not burning at t,
P(burning at t+1 | inputs) is exactly the probability of a 0 -> 1 transition, so the
same scores are evaluated on both targets without retraining and without using any
test label to choose pixels.

Two diagnostics are included to expose the patch-centre position leak (NDWS patches are
centred on next-day fire):
  centre_prior     score = training-set fire rate at the pixel's distance from the patch
                   centre. Uses position only (no inputs at all), fitted on training labels.
  unet_fullpatch   U-Net trained on full 64x64 patches, which can learn the patch centre.
The primary U-Net ("unet") is trained on random 32x32 crops and cannot.

Only threshold-free metrics are reported here (PR-AUC with patch-bootstrap CI, lift,
ROC-AUC, recall at 1% / 5% FPR, Brier). Thresholded metrics for the active target are
in experiments/ndws/RESULTS.md and the U-Net results.json.

Inputs (produced earlier):
    data/ndws/preds/<protocol>_test.npz       tabular test predictions (src/aml/train.py)
    data/ndws/preds/<protocol>_unet_test.npy  U-Net test maps (src/dl/train.py; also _unet_fullpatch)
    data/ndws/dl/<protocol>_train_y.npy       training labels (centre prior only)

Outputs:
    experiments/ndws/dl/<protocol>/dual_target_metrics.csv, significance.json,
        pr_curves_dual_target.png, probability_maps_unet.png
    experiments/ndws/dl/DUAL_TARGET_RESULTS.md

Usage:
    python -m src.dl.evaluate
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_recall_curve

from src.aml.ndws import CH, load_split
from src.aml.dataset import patch_refs, iter_patch_batches
from src.aml.metrics import recall_at_fpr, bootstrap_pr_auc, paired_bootstrap_delta
from src.aml.evaluate import LABELS as TABULAR_LABELS, COLORS as TABULAR_COLORS
from .data import DL_CACHE_DIR
from .train import PRED_DIR, RESULTS_DIR

N_BOOT = 500
MODELS = ["centre_prior", "persistence", "neighbourhood_persist", "logistic_regression", "random_forest", "xgboost",
          "lightgbm", "unet_fullpatch", "unet"]
UNET_VARIANTS = ["unet", "unet_fullpatch"]
LABELS = {**TABULAR_LABELS, "centre_prior": "Patch-centre prior (leak probe)",
          "unet_fullpatch": "U-Net, full patches (position leak)", "unet": "U-Net, 32x32 crops"}
COLORS = {**TABULAR_COLORS, "centre_prior": "#bc6c25", "unet_fullpatch": "#b8a1d9", "unet": "#6a4c93"}
TARGETS = {
    "active": "next-day active fire (all known pixels)",
    "transition": "new fire (pixels not burning at day t)",
}
COMPARISONS = [("unet", "xgboost"), ("unet", "lightgbm"), ("unet", "random_forest"), ("unet", "logistic_regression"),
               ("unet_fullpatch", "unet")]
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})


def prev_fire_pixels(protocol, patch, pix):
    """Day-t fire mask value (1 / 0 / -1) for each (patch, pixel) of the test split."""
    prev = np.concatenate([raw[:, CH["PrevFireMask"]].reshape(len(raw), -1)
                           for raw in iter_patch_batches(patch_refs(protocol, "test"))])
    return prev[patch, pix].astype(np.int8)


def target_masks(prev):
    """Row masks for the two targets. Transition keeps only pixels known not burning at t."""
    return {"active": np.ones_like(prev, dtype=bool), "transition": prev == 0}


def load_scores(protocol):
    d = np.load(os.path.join(PRED_DIR, f"{protocol}_test.npz"))
    y, patch, pix = d["y"].astype(np.int8), d["patch"], d["pix"].astype(np.int64)
    scores = {m: d[f"p_{m}"].astype(np.float32) for m in MODELS if f"p_{m}" in d}
    maps = {}
    for v in UNET_VARIANTS:
        maps[v] = np.load(os.path.join(PRED_DIR, f"{protocol}_{v}_test.npy")).astype(np.float32)
        scores[v] = gather_maps(maps[v], patch, pix)
    scores["centre_prior"] = centre_prior(np.load(os.path.join(DL_CACHE_DIR, f"{protocol}_train_y.npy"), mmap_mode="r"), pix)
    return y, patch, pix, scores, maps


def centre_distance(pix, size=64):
    r, c = np.divmod(np.asarray(pix), size)
    return np.rint(np.hypot(r - (size - 1) / 2, c - (size - 1) / 2)).astype(np.int64)


def centre_prior(y_train, pix):
    """Leak probe: fire rate in TRAINING labels at each (rounded) distance from the patch centre."""
    d = centre_distance(np.arange(64 * 64))
    flat = np.asarray(y_train).reshape(len(y_train), -1)
    known = flat >= 0
    pos = np.bincount(np.broadcast_to(d, flat.shape)[known], weights=(flat[known] == 1), minlength=d.max() + 1)
    cnt = np.bincount(np.broadcast_to(d, flat.shape)[known], minlength=d.max() + 1)
    rate = pos / np.maximum(cnt, 1)
    return rate[centre_distance(pix)].astype(np.float32)


def gather_maps(maps, patch, pix):
    """Per-pixel scores from (N, 64, 64) maps in the tabular row order (patch index, flat pixel index)."""
    return maps.reshape(len(maps), -1)[patch, pix]


def threshold_free_metrics(y, p):
    out = {"pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
           "brier": float(brier_score_loss(y, p)), "positive_rate": float(y.mean()), "n_pixels": int(len(y))}
    out["pr_auc_lift"] = out["pr_auc"] / out["positive_rate"]
    for lvl in (0.01, 0.05):
        out[f"recall_at_{int(lvl * 100)}pct_fpr"] = recall_at_fpr(y, p, lvl)
    return out


def evaluate_protocol(protocol):
    y, patch, pix, scores, maps = load_scores(protocol)
    prev = prev_fire_pixels(protocol, patch, pix)
    rows, sig = [], {}
    for target, mask in target_masks(prev).items():
        yt, pt = y[mask], patch[mask]
        sig[target] = {}
        for m in MODELS:
            met = threshold_free_metrics(yt, scores[m][mask])
            b = bootstrap_pr_auc(yt, scores[m][mask], pt, n_boot=N_BOOT)
            rows.append({"protocol": protocol, "target": target, "model": m, **met,
                         "pr_auc_ci_low": b["ci_low"], "pr_auc_ci_high": b["ci_high"]})
        for a, b_ in COMPARISONS:
            sig[target][f"{a} - {b_}"] = paired_bootstrap_delta(yt, scores[a][mask], scores[b_][mask], pt, n_boot=N_BOOT)
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, protocol)
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "dual_target_metrics.csv"), index=False)
    with open(os.path.join(out, "significance.json"), "w") as f:
        json.dump(sig, f, indent=2)
    pr_figure(protocol, y, prev, scores)
    map_figure(protocol, y, patch, pix, scores, maps)
    return df, sig


def pr_figure(protocol, y, prev, scores):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (target, mask) in zip(axes, target_masks(prev).items()):
        for m in MODELS:
            if target == "transition" and m == "persistence":
                continue  # constant 0 on this target
            p, r, _ = precision_recall_curve(y[mask], scores[m][mask])
            ax.plot(r, p, color=COLORS[m], lw=2.0 if m == "unet" else 1.1, label=LABELS[m])
        ax.axhline(y[mask].mean(), color="k", ls=":", lw=1, label=f"chance ({y[mask].mean():.4f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{target}: {TARGETS[target]}", fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle(f"Test precision-recall, {protocol} split")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, protocol, "pr_curves_dual_target.png"), bbox_inches="tight")
    plt.close(fig)


def map_figure(protocol, y, patch, pix, scores, maps, n=4):
    refs = patch_refs(protocol, "test")
    pos = np.bincount(patch, weights=(y == 1), minlength=len(refs))
    order = np.argsort(-pos)
    picks = order[np.linspace(5, max(6, int(np.sum(pos > 20)) - 1), n).astype(int)]
    cmap = matplotlib.colors.ListedColormap(["black", "#e9ecef", "#d1495b"])
    fig, axes = plt.subplots(n, 5, figsize=(13.5, 2.8 * n))
    for r, i in enumerate(picks):
        split, idx = refs[i]
        raw = np.asarray(load_split(split)[idx])
        lgb = np.full(64 * 64, np.nan, dtype=np.float32)
        sel = patch == i
        lgb[pix[sel]] = scores["lightgbm"][sel]
        panels = [(raw[CH["PrevFireMask"]], "Fire at t", True), (raw[CH["FireMask"]], "Fire at t+1 (truth)", True),
                  (lgb.reshape(64, 64), "LightGBM P(fire)", False),
                  (maps["unet_fullpatch"][i], "U-Net full patch (leaky)", False), (maps["unet"][i], "U-Net 32x32 crops", False)]
        for k, (img, title, is_mask) in enumerate(panels):
            ax = axes[r, k]
            if is_mask:
                ax.imshow(img, cmap=cmap, vmin=-1, vmax=1, origin="lower", interpolation="nearest")
            else:
                im = ax.imshow(img, cmap="inferno", vmin=0, vmax=1, origin="lower", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=10)
        axes[r, 0].set_ylabel(f"{split} #{idx}", fontsize=8)
    fig.colorbar(im, ax=axes[:, 2:], shrink=0.6, label="predicted probability")
    fig.suptitle(f"U-Net vs LightGBM on test patches ({protocol} split; north up)")
    fig.savefig(os.path.join(RESULTS_DIR, protocol, "probability_maps_unet.png"), bbox_inches="tight")
    plt.close(fig)


def write_markdown(tables, sigs):
    lines = ["# Dual-target test results: spatial U-Net vs tabular models",
             "", "Generated by `python -m src.dl.evaluate`. Do not edit by hand.", "",
             "- **active**: next-day active fire, FireMask(t+1) on every known pixel (includes fire persistence).",
             "- **transition**: new fire, FireMask(t+1) on pixels known not burning at day t (persistence excluded).",
             "- **Patch-centre prior** uses pixel position only; **U-Net, full patches** can learn position. "
             "Both measure the NDWS patch-centring leak and are not valid forecasters. The primary spatial model is "
             "**U-Net, 32x32 crops**.",
             "", f"PR-AUC 95% CIs: patch-level bootstrap ({N_BOOT} resamples). Lift = PR-AUC / positive rate. "
             "All models were trained on the active target; thresholds are not used here.", ""]
    for proto, df in tables.items():
        for target in TARGETS:
            sub = df[df.target == target].set_index("model").loc[MODELS]
            lines += [f"## `{proto}` split, target `{target}`", "",
                      f"Test pixels: {int(sub.n_pixels.iloc[0]):,}; positive rate {sub.positive_rate.iloc[0]:.4f}.", "",
                      "| Model | PR-AUC [95% CI] | Lift | ROC-AUC | Recall@1%FPR | Recall@5%FPR | Brier |",
                      "|---|---|---|---|---|---|---|"]
            for m, r in sub.iterrows():
                lines.append(f"| {LABELS[m]} | {r.pr_auc:.3f} [{r.pr_auc_ci_low:.3f}, {r.pr_auc_ci_high:.3f}] | "
                             f"{r.pr_auc_lift:.1f}x | {r.roc_auc:.3f} | {r.recall_at_1pct_fpr:.3f} | "
                             f"{r.recall_at_5pct_fpr:.3f} | {r.brier:.4f} |")
            lines += ["", "| Paired comparison | ΔPR-AUC | 95% CI | P(Δ ≤ 0) |", "|---|---|---|---|"]
            for k, s in sigs[proto][target].items():
                lines.append(f"| {k} | {s['delta']:+.4f} | [{s['ci_low']:+.4f}, {s['ci_high']:+.4f}] | {s['p_le_zero']:.3f} |")
            lines.append("")
    with open(os.path.join(RESULTS_DIR, "DUAL_TARGET_RESULTS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    tables, sigs = {}, {}
    for protocol in ("official", "spatial"):
        needed = [os.path.join(PRED_DIR, f"{protocol}_test.npz")] + \
                 [os.path.join(PRED_DIR, f"{protocol}_{v}_test.npy") for v in UNET_VARIANTS]
        missing = [p for p in needed if not os.path.exists(p)]
        if missing:
            print(f"skip {protocol}: missing {', '.join(os.path.basename(p) for p in missing)}")
            continue
        print(f"[{protocol}] dual-target evaluation")
        tables[protocol], sigs[protocol] = evaluate_protocol(protocol)
        print(tables[protocol][["target", "model", "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high", "roc_auc"]].to_string(index=False))
    if tables:
        write_markdown(tables, sigs)


if __name__ == "__main__":
    main()
