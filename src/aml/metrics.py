"""
metrics.py - Imbalance-aware metrics for pixel-level next-day fire prediction.

Primary metric: PR-AUC (average precision). Its chance level equals the positive rate, so
PR-AUC is only comparable across test sets with equal prevalence; pr_auc_lift = PR-AUC / prevalence
and ROC-AUC (prevalence-independent) are used when comparing protocols. Also ROC-AUC, F1 / precision / recall at a
threshold chosen on validation data, recall at fixed false-positive rates, Brier score.
Accuracy is deliberately not reported: with ~1% positives, predicting "no fire" everywhere
is ~99% accurate.

Confidence intervals use a patch-level bootstrap (pixels in a patch are correlated, so
patches, not pixels, are resampled). Average precision under bootstrap weights is computed
from a single pre-sort, which makes 1000 replicates over millions of pixels feasible.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score, roc_curve, precision_recall_curve,
    brier_score_loss, f1_score, precision_score, recall_score,
)

FPR_LEVELS = (0.01, 0.05)


def best_f1_threshold(y, p):
    """Threshold maximising F1 (used on validation data only)."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i])


def recall_at_fpr(y, p, fpr_level):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(fpr_level, fpr, tpr))


def compute_metrics(y, p, threshold):
    yhat = (p >= threshold).astype(np.int8)
    out = {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
        "positive_rate": float(y.mean()),
        "pr_auc_lift": float(average_precision_score(y, p) / y.mean()),  # PR-AUC relative to chance
        "n_pixels": int(len(y)),
    }
    for lvl in FPR_LEVELS:
        out[f"recall_at_{int(lvl * 100)}pct_fpr"] = recall_at_fpr(y, p, lvl)
    return out


class BootstrapAP:
    """Average precision under patch-level bootstrap weights, from one sort."""

    def __init__(self, y, p, patch):
        order = np.argsort(-p, kind="mergesort")
        self.y = y[order].astype(np.float64)
        self.patch = patch[order]
        ps = p[order]
        # last index of each run of tied scores: AP is evaluated only at distinct thresholds
        self.ends = np.flatnonzero(np.r_[ps[1:] != ps[:-1], True])
        self.n_patches = int(patch.max()) + 1

    def ap(self, patch_weights):
        w = patch_weights[self.patch]
        tp = np.cumsum(w * self.y)[self.ends]
        fp = np.cumsum(w * (1 - self.y))[self.ends]
        if tp[-1] == 0:
            return np.nan
        precision = tp / np.maximum(tp + fp, 1e-12)
        recall = tp / tp[-1]
        return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def bootstrap_pr_auc(y, p, patch, n_boot=1000, seed=0):
    """Point estimate plus 95% percentile CI of PR-AUC with patches resampled."""
    b = BootstrapAP(y, p, patch)
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, b.n_patches, b.n_patches), minlength=b.n_patches)
        reps.append(b.ap(counts.astype(np.float64)))
    reps = np.array(reps)
    return {"pr_auc": b.ap(np.ones(b.n_patches)), "ci_low": float(np.nanpercentile(reps, 2.5)),
            "ci_high": float(np.nanpercentile(reps, 97.5)), "replicates": reps}


def paired_bootstrap_delta(y, p_a, p_b, patch, n_boot=1000, seed=0):
    """95% CI for PR-AUC(a) - PR-AUC(b) with the same patch resamples for both models."""
    a, b = BootstrapAP(y, p_a, patch), BootstrapAP(y, p_b, patch)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, a.n_patches, a.n_patches), minlength=a.n_patches).astype(np.float64)
        deltas.append(a.ap(w) - b.ap(w))
    deltas = np.array(deltas)
    ones = np.ones(a.n_patches)
    return {"delta": a.ap(ones) - b.ap(ones), "ci_low": float(np.percentile(deltas, 2.5)),
            "ci_high": float(np.percentile(deltas, 97.5)), "p_le_zero": float((deltas <= 0).mean())}
