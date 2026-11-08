"""
evaluate.py - Test-set evaluation, uncertainty and figures for the tabular models.

Reads the test predictions saved by train.py and writes, per protocol, to
experiments/ndws/results/<protocol>/:
    metrics_table.csv          all metrics, with 95% patch-bootstrap CI for PR-AUC
    significance.json          paired bootstrap PR-AUC differences between models
    pr_curves.png, roc_curves.png, calibration.png
    probability_maps.png       day-t fire, day t+1 truth and model probability maps
    feature_importance.png     LightGBM gain importance
    shap_summary.png           SHAP values of LightGBM on a 20k-pixel test sample
    plausibility.png           predicted vs observed spread by distance, wind and slope
    ablation.png               LightGBM PR-AUC by feature-group subset (where run)
and, across protocols:
    experiments/ndws/results/protocol_gap.png
    experiments/ndws/RESULTS.md (generated tables)

Usage:
    python -m src.aml.evaluate
"""

import json
import os
from itertools import pairwise

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from src.config import load_config

from .dataset import REPORTED_PROTOCOLS, eval_pixels, patch_refs
from .features import FEATURE_GROUPS, FEATURES, FX
from .metrics import bootstrap_pr_auc, compute_metrics, paired_bootstrap_delta
from .ndws import BASE_DIR, CH, load_split
from .train import MODEL_DIR, PRED_DIR, RESULTS_DIR

N_BOOT = load_config("aml")["evaluation"]["n_boot"]
MODELS = ["dummy", "persistence", "neighbourhood_persist", "logistic_regression", "random_forest", "xgboost", "lightgbm"]
LABELS = {
    "dummy": "Dummy (training rate)",
    "persistence": "Persistence",
    "neighbourhood_persist": "Neighbourhood persistence",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}
COLORS = {
    "dummy": "#dee2e6", "persistence": "#adb5bd", "neighbourhood_persist": "#6c757d", "logistic_regression": "#00798c",
    "random_forest": "#edae49", "xgboost": "#30638e", "lightgbm": "#d1495b",
}
COMPARISONS = [("lightgbm", "logistic_regression"), ("lightgbm", "random_forest"), ("lightgbm", "xgboost"),
               ("lightgbm", "persistence"), ("xgboost", "logistic_regression"), ("logistic_regression", "persistence")]
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def load_predictions(protocol):
    d = np.load(os.path.join(PRED_DIR, f"{protocol}_test.npz"))
    scores = {m: d[f"p_{m}"].astype(np.float32) for m in MODELS}
    return d["y"].astype(np.int8), d["patch"], d["pix"], scores


def metrics_table(protocol, results, y, patch, scores):
    rows, boots = [], {}
    for m in MODELS:
        met = compute_metrics(y, scores[m], results["thresholds_from_val"][m])
        b = bootstrap_pr_auc(y, scores[m], patch, n_boot=N_BOOT)
        boots[m] = b
        rows.append({"model": m, **met, "pr_auc_ci_low": b["ci_low"], "pr_auc_ci_high": b["ci_high"],
                     "val_pr_auc": results["val"][m]["pr_auc"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, protocol, "metrics_table.csv"), index=False)
    sig = {f"{a} - {b}": paired_bootstrap_delta(y, scores[a], scores[b], patch, n_boot=N_BOOT) for a, b in COMPARISONS}
    with open(os.path.join(RESULTS_DIR, protocol, "significance.json"), "w") as f:
        json.dump(sig, f, indent=2)
    return df, sig


def curves(protocol, y, scores):
    out = os.path.join(RESULTS_DIR, protocol)
    fig, ax = plt.subplots(figsize=(6, 5))
    for m in MODELS:
        p, r, _ = precision_recall_curve(y, scores[m])
        ax.plot(r, p, label=LABELS[m], color=COLORS[m], lw=1.6 if m in ("lightgbm", "xgboost") else 1.1)
    ax.axhline(y.mean(), color="k", ls=":", lw=1, label=f"chance ({y.mean():.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-recall, test ({protocol} split)")
    ax.legend(fontsize=8)
    save(fig, os.path.join(out, "pr_curves.png"))

    fig, ax = plt.subplots(figsize=(6, 5))
    for m in MODELS:
        fpr, tpr, _ = roc_curve(y, scores[m])
        ax.plot(fpr, tpr, label=LABELS[m], color=COLORS[m])
    for lvl in (0.01, 0.05):
        ax.axvline(lvl, color="k", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_xlabel("False positive rate (log scale)")
    ax.set_ylabel("Recall (true positive rate)")
    ax.set_title(f"ROC, test ({protocol}); dotted = 1% and 5% FPR")
    ax.legend(fontsize=8)
    save(fig, os.path.join(out, "roc_curves.png"))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for m in ("logistic_regression", "random_forest", "xgboost", "lightgbm"):
        frac, mean_p = calibration_curve(y, scores[m], n_bins=15, strategy="quantile")
        ax.plot(mean_p, frac, "o-", ms=3, label=LABELS[m], color=COLORS[m])
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fire frequency")
    ax.set_title(f"Reliability (quantile bins), test ({protocol})")
    ax.legend(fontsize=8)
    save(fig, os.path.join(out, "calibration.png"))


def probability_maps(protocol, y, patch, pix, scores, n=4):
    refs = patch_refs(protocol, "test")
    pos_per_patch = np.bincount(patch, weights=(y == 1), minlength=len(refs))
    order = np.argsort(-pos_per_patch)
    picks = order[np.linspace(5, max(6, int(np.sum(pos_per_patch > 20)) - 1), n).astype(int)]
    cmap_mask = matplotlib.colors.ListedColormap(["black", "#e9ecef", "#d1495b"])
    fig, axes = plt.subplots(n, 5, figsize=(13, 2.7 * n))
    for r, i in enumerate(picks):
        split, idx = refs[i]
        raw = np.asarray(load_split(split)[idx])
        sel = patch == i
        cols = [(raw[CH["PrevFireMask"]], "Fire at t", "mask"), (raw[CH["FireMask"]], "Fire at t+1 (truth)", "mask")]
        for m in ("logistic_regression", "lightgbm", "xgboost"):
            img = np.full(64 * 64, np.nan, dtype=np.float32)
            img[pix[sel]] = scores[m][sel]
            cols.append((img.reshape(64, 64), f"{LABELS[m]} P(fire)", "prob"))
        for k, (img, title, kind) in enumerate(cols):
            ax = axes[r, k]
            if kind == "mask":
                ax.imshow(img, cmap=cmap_mask, vmin=-1, vmax=1, origin="lower", interpolation="nearest")
            else:
                im = ax.imshow(img, cmap="inferno", vmin=0, vmax=1, origin="lower", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=10)
        axes[r, 0].set_ylabel(f"{split} #{idx}", fontsize=8)
    fig.colorbar(im, ax=axes[:, 2:], shrink=0.6, label="predicted probability")
    fig.suptitle(f"Test probability maps ({protocol} split; north up, black = unknown label)")
    fig.savefig(os.path.join(RESULTS_DIR, protocol, "probability_maps.png"), bbox_inches="tight")
    plt.close(fig)


def importance_and_shap(protocol):
    import lightgbm as lgb  # lazy: keeps this module importable alongside PyTorch (see train.fit_xgboost)
    booster = lgb.Booster(model_file=os.path.join(MODEL_DIR, f"{protocol}_lightgbm.txt"))
    gain = booster.feature_importance(importance_type="gain")
    gain = gain / gain.sum()
    order = np.argsort(gain)
    group_of = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
    palette = dict(zip(FEATURE_GROUPS, ["#8d99ae", "#d1495b", "#30638e", "#6a994e", "#edae49", "#00798c"], strict=False))
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(np.array(FEATURES)[order], gain[order], color=[palette[group_of[FEATURES[i]]] for i in order])
    ax.set_xlabel("share of total split gain")
    ax.set_title(f"LightGBM feature importance ({protocol})")
    handles = [matplotlib.patches.Patch(color=c, label=g) for g, c in palette.items()]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    save(fig, os.path.join(RESULTS_DIR, protocol, "feature_importance.png"))

    import shap
    sample = np.load(os.path.join(PRED_DIR, f"{protocol}_test_sample.npz"))
    X = sample["X"][:5000]
    contrib = booster.predict(X, pred_contrib=True)[:, :-1]
    shap.summary_plot(contrib, X, feature_names=FEATURES, show=False, max_display=18)
    fig = plt.gcf()
    fig.suptitle(f"SHAP values (log-odds), LightGBM, 5k test pixels ({protocol})")
    save(fig, os.path.join(RESULTS_DIR, protocol, "shap_summary.png"))
    mean_abs = np.abs(contrib).mean(axis=0)
    return {f: float(v) for f, v in sorted(zip(FEATURES, mean_abs, strict=False), key=lambda t: -t[1])}


def plausibility(protocol, scores):
    """Compare predicted and observed new-fire rates across physically meaningful bins."""
    X, y, _, _ = eval_pixels(protocol, "test")
    p = scores["lightgbm"]
    nf = X[:, FX["prev_fire"]] == 0
    d = X[:, FX["dist_to_fire_km"]]
    out = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    bins = [0, 1.5, 2.5, 3.5, 5, 7, 10, 15, 100]
    labels = [f"{a:g}-{b:g}" for a, b in pairwise(bins)]
    obs = [y[nf & (d > a) & (d <= b)].mean() for a, b in pairwise(bins)]
    pred = [p[nf & (d > a) & (d <= b)].mean() for a, b in pairwise(bins)]
    out["distance_km"] = {lab: {"observed": float(o), "predicted": float(q)} for lab, o, q in zip(labels, obs, pred, strict=True)}
    x = np.arange(len(labels))
    axes[0].bar(x - 0.2, np.array(obs) * 100, 0.4, label="observed", color="#8d99ae")
    axes[0].bar(x + 0.2, np.array(pred) * 100, 0.4, label="LightGBM mean P", color="#d1495b")
    axes[0].set_xticks(x, labels, rotation=45, fontsize=8)
    axes[0].set_xlabel("distance to day-t fire (km)")
    axes[0].set_ylabel("% new fire at t+1")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Distance to front")

    near = nf & (d > 0) & (d <= 3)
    for ax, feat, edges, title in (
        (axes[1], "fire_wind_alignment", [-1.01, -0.5, 0, 0.5, 1.01], "Wind alignment (+1 = downwind of fire)"),
        (axes[2], "upslope_from_fire", [-1, -0.02, 0.02, 1], "Relative elevation (+ = upslope of fire)"),
    ):
        v = X[:, FX[feat]]
        labs = [f"{a:.2g}..{b:.2g}" for a, b in pairwise(edges)]
        o = [y[near & (v > a) & (v <= b)].mean() for a, b in pairwise(edges)]
        q = [p[near & (v > a) & (v <= b)].mean() for a, b in pairwise(edges)]
        out[feat] = {lab: {"observed": float(a), "predicted": float(b)} for lab, a, b in zip(labs, o, q, strict=True)}
        x = np.arange(len(labs))
        ax.bar(x - 0.2, np.array(o) * 100, 0.4, label="observed", color="#8d99ae")
        ax.bar(x + 0.2, np.array(q) * 100, 0.4, label="LightGBM mean P", color="#d1495b")
        ax.set_xticks(x, labs, fontsize=8)
        ax.set_title(title + "\n(pixels within 3 km of fire)", fontsize=10)
    fig.suptitle(f"Physical plausibility, test ({protocol})")
    save(fig, os.path.join(RESULTS_DIR, protocol, "plausibility.png"))
    return out


def ablation_plot(results):
    ab = results.get("ablation")
    if not ab:
        return
    names = list(ab)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(names))
    ax.bar(x - 0.2, [ab[n]["val_pr_auc"] for n in names], 0.4, label="val", color="#8d99ae")
    ax.bar(x + 0.2, [ab[n]["test_pr_auc"] for n in names], 0.4, label="test", color="#d1495b")
    ax.set_xticks(x, names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("PR-AUC")
    ax.set_title(f"LightGBM feature-group ablation ({results['protocol']} split)")
    ax.legend()
    save(fig, os.path.join(RESULTS_DIR, results["protocol"], "ablation.png"))


def protocol_gap(tables):
    """Raw PR-AUC and prevalence-normalised lift per model across the test protocols."""
    if len(tables) < 2:
        return
    palette = {"official": "#adb5bd", "spatial": "#d1495b", "strict": "#30638e"}
    width = 0.8 / len(tables)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(len(MODELS))
    for k, (proto, df) in enumerate(tables.items()):
        df = df.set_index("model").loc[MODELS]
        err = np.vstack([df.pr_auc - df.pr_auc_ci_low, df.pr_auc_ci_high - df.pr_auc])
        prev = df.positive_rate.iloc[0]
        offset = x + (k - (len(tables) - 1) / 2) * width
        axes[0].bar(offset, df.pr_auc, width, yerr=err, capsize=2, color=palette[proto],
                    label=f"{proto} test (prevalence {prev:.2%})")
        axes[1].bar(offset, df.pr_auc_lift, width, yerr=err / prev, capsize=2, color=palette[proto], label=f"{proto} test")
    axes[0].set_ylabel("Test PR-AUC (95% CI)")
    axes[0].set_title("Raw PR-AUC (chance level = prevalence)")
    axes[1].set_ylabel("PR-AUC / prevalence")
    axes[1].set_title("Lift over chance")
    for ax in axes:
        ax.set_xticks(x, [LABELS[m] for m in MODELS], rotation=20, fontsize=8)
        ax.legend(fontsize=8)
    fig.suptitle("Tabular models across the three test protocols")
    save(fig, os.path.join(RESULTS_DIR, "protocol_gap.png"))


def fmt(v, d=3):
    return f"{v:.{d}f}"


def write_markdown(tables, sigs, extras):
    lines = ["# NDWS tabular results (generated by `python -m src.aml.evaluate`)", "",
             "All numbers are on the held-out **test** split; thresholds for F1 / precision / recall were chosen on "
             "the validation split. PR-AUC 95% CIs come from a patch-level bootstrap "
             f"({N_BOOT} resamples). Lift = PR-AUC / test positive rate (chance level = 1x). Positive rates differ "
             "across protocols, so compare protocols with Lift and ROC-AUC, not raw PR-AUC. Do not edit by hand; "
             "rerun the script.", ""]
    for proto, df in tables.items():
        lines += [f"## Protocol: `{proto}`", "",
                  "| Model | PR-AUC [95% CI] | Lift | ROC-AUC | F1 | Precision | Recall | Recall@1%FPR | Recall@5%FPR | Brier |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in df.iterrows():
            lines.append(f"| {LABELS[r.model]} | {fmt(r.pr_auc)} [{fmt(r.pr_auc_ci_low)}, {fmt(r.pr_auc_ci_high)}] | "
                         f"{r.pr_auc_lift:.1f}x | {fmt(r.roc_auc)} | {fmt(r.f1)} | {fmt(r.precision)} | {fmt(r.recall)} | "
                         f"{fmt(r.recall_at_1pct_fpr)} | {fmt(r.recall_at_5pct_fpr)} | {fmt(r.brier, 4)} |")
        lines += ["", f"Test pixels: {int(df.n_pixels.iloc[0]):,}; positive rate {df.positive_rate.iloc[0]:.4f}.", "",
                  "Paired bootstrap PR-AUC differences:", "", "| Comparison | Delta | 95% CI | P(delta <= 0) |", "|---|---|---|---|"]
        for k, s in sigs[proto].items():
            lines.append(f"| {k} | {s['delta']:+.4f} | [{s['ci_low']:+.4f}, {s['ci_high']:+.4f}] | {s['p_le_zero']:.3f} |")
        ab = extras[proto].get("ablation")
        if ab:
            lines += ["", "LightGBM feature-group ablation:", "", "| Feature set | # features | Val PR-AUC | Test PR-AUC |", "|---|---|---|---|"]
            for name, a in ab.items():
                lines.append(f"| {name} | {a['n_features']} | {fmt(a['val_pr_auc'])} | {fmt(a['test_pr_auc'])} |")
        lines += ["", "Top SHAP features (mean |SHAP|, log-odds): " +
                  ", ".join(f"`{f}` {v:.3f}" for f, v in list(extras[proto]["shap"].items())[:8]), ""]
    with open(os.path.join(BASE_DIR, "experiments", "ndws", "RESULTS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    tables, sigs, extras = {}, {}, {}
    for protocol in REPORTED_PROTOCOLS:
        path = os.path.join(RESULTS_DIR, protocol, "results.json")
        if not os.path.exists(path):
            print(f"skip {protocol}: run `python -m src.aml.train --protocol {protocol}` first")
            continue
        with open(path) as f:
            results = json.load(f)
        print(f"[{protocol}] metrics + bootstrap")
        y, patch, pix, scores = load_predictions(protocol)
        tables[protocol], sigs[protocol] = metrics_table(protocol, results, y, patch, scores)
        print(f"[{protocol}] figures")
        curves(protocol, y, scores)
        probability_maps(protocol, y, patch, pix, scores)
        shap_rank = importance_and_shap(protocol)
        plaus = plausibility(protocol, scores)
        ablation_plot(results)
        extras[protocol] = {"shap": shap_rank, "plausibility": plaus, "ablation": results.get("ablation")}
        with open(os.path.join(RESULTS_DIR, protocol, "interpretation.json"), "w") as f:
            json.dump({"shap_mean_abs": shap_rank, "plausibility": plaus}, f, indent=2)
        print(tables[protocol][["model", "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high", "roc_auc", "f1"]].to_string(index=False))
    protocol_gap(tables)
    write_markdown(tables, sigs, extras)


if __name__ == "__main__":
    main()
