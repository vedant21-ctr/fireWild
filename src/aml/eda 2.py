"""
eda.py - Exploratory data analysis and data-quality audit for NDWS.

Writes figures and eda_summary.json to experiments/ndws/eda/:
    class_balance.png          positive rate and unknown-label rate per split
    channel_distributions.png  input channels for next-day fire vs no-fire pixels
    example_patches.png        inputs, day-t fire and day t+1 fire for sample patches
    fire_dynamics.png          fire growth/shrinkage and persistence of burning pixels
    spread_vs_distance.png     P(new fire) vs distance to day-t fire
    wind_orientation.png       downwind/upwind spread ratio under each orientation convention (all train patches)
    terrain_effect.png         P(new fire) for up-, flat- and down-slope pixels near fire
    feature_correlation.png    Spearman correlation of engineered features
    split_overlap.png          terrain-overlap audit of official vs spatial splits

Usage:
    python -m src.aml.eda
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .ndws import BASE_DIR, CH, INPUT_CHANNELS, load_split
from .features import clean_channels, batch_features, FEATURES, FX, _directional_fire
from .splits import load_manifest, SPLIT_DIR

OUT_DIR = os.path.join(BASE_DIR, "experiments", "ndws", "eda")
SAMPLE_PATCHES = 3000
SEED = 42
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), bbox_inches="tight")
    plt.close(fig)


def class_balance():
    rows = []
    for split in ("train", "eval", "test"):
        a = load_split(split)
        y = np.asarray(a[:, CH["FireMask"]])
        prev = np.asarray(a[:, CH["PrevFireMask"]])
        known = y >= 0
        size_t = (prev == 1).sum(axis=(1, 2))
        size_t1 = (y == 1).sum(axis=(1, 2))
        rows.append({
            "split": split,
            "patches": int(a.shape[0]),
            "pixels": int(y.size),
            "positive_rate": float(y[known].mean()),
            "unknown_label_rate": float((~known).mean()),
            "prev_fire_rate": float((prev == 1).mean()),
            "prev_unknown_rate": float((prev < 0).mean()),
            "patches_without_next_day_fire": int((size_t1 == 0).sum()),
            "persistence_recall": float((prev[y == 1] == 1).mean()),
            "frac_next_day_fire_new": float((prev[y == 1] != 1).mean()),
            "frac_patches_growing": float((size_t1 > size_t).mean()),
            "frac_patches_shrinking": float((size_t1 < size_t).mean()),
        })
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].bar(df.split, df.positive_rate * 100, color="#d1495b")
    ax[0].set_ylabel("% pixels on fire at t+1")
    ax[0].set_title("Class balance (known labels)")
    ax[1].bar(df.split, df.unknown_label_rate * 100, color="#8d99ae")
    ax[1].set_ylabel("% pixels with unknown label (-1)")
    ax[1].set_title("Unknown labels (cloud / no data)")
    save(fig, "class_balance.png")
    return df


def quality_audit():
    """Fraction of physically invalid / sentinel values per raw channel (all splits)."""
    checks = {
        "elevation": lambda x: x < -100,  # lowest US point is -86 m
        "th": lambda x: (x < 0) | (x > 361),
        "vs": lambda x: x < 0,
        "tmmn": lambda x: (x < 200) | (x > 330),
        "tmmx": lambda x: (x < 200) | (x > 330),
        "sph": lambda x: x <= 0,
        "pr": lambda x: x < 0,
        "NDVI": lambda x: (x < -10000) | (x > 10000),
        "population": lambda x: x < 0,
        "erc": lambda x: x < 0,
    }
    out = {}
    for name, bad in checks.items():
        n_bad = n = 0
        stats = []
        for split in ("train", "eval", "test"):
            x = np.asarray(load_split(split)[:, CH[name]])
            n_bad += int(bad(x).sum())
            n += x.size
            stats.append(x[::7, ::4, ::4].ravel())
        s = np.concatenate(stats)
        out[name] = {"invalid_fraction": n_bad / n, "min": float(s.min()), "p1": float(np.percentile(s, 1)),
                     "median": float(np.median(s)), "p99": float(np.percentile(s, 99)), "max": float(s.max())}
    return out


def sample_patches(n=SAMPLE_PATCHES):
    a = load_split("train")
    idx = np.sort(np.random.default_rng(SEED).choice(a.shape[0], n, replace=False))
    return np.asarray(a[idx])


def channel_distributions(raw):
    y = raw[:, CH["FireMask"]]
    names = [c for c in INPUT_CHANNELS if c != "PrevFireMask"]
    cleaned = [clean_channels(p) for p in raw]
    fig, axes = plt.subplots(3, 4, figsize=(13, 8))
    for ax, name in zip(axes.ravel(), names):
        vals = np.stack([c[name] for c in cleaned])
        fire, nofire = vals[y == 1], vals[y == 0]
        fire, nofire = fire[np.isfinite(fire)], nofire[np.isfinite(nofire)]
        nofire = np.random.default_rng(SEED).choice(nofire, min(200000, nofire.size), replace=False)
        lo, hi = np.percentile(np.r_[fire, nofire], [0.5, 99.5])
        bins = np.linspace(lo, hi, 50)
        ax.hist(nofire, bins=bins, density=True, alpha=0.55, label="no fire at t+1", color="#8d99ae")
        ax.hist(fire, bins=bins, density=True, alpha=0.55, label="fire at t+1", color="#d1495b")
        ax.set_title(name)
        ax.set_yticks([])
    axes.ravel()[-1].axis("off")
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle("Input channels: next-day fire vs no-fire pixels (3,000 training patches, cleaned)")
    save(fig, "channel_distributions.png")


def example_patches(raw):
    order = np.argsort(-(raw[:, CH["FireMask"]] == 1).sum(axis=(1, 2)))
    picks = order[[5, 60, 200, 600]]
    cols = ["elevation", "NDVI", "vs", "erc", "PrevFireMask", "FireMask"]
    titles = ["Elevation (m)", "NDVI", "Wind speed (m/s)", "ERC", "Fire at t", "Fire at t+1"]
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(13, 9))
    for r, i in enumerate(picks):
        c = clean_channels(raw[i])
        for k, name in enumerate(cols):
            ax = axes[r, k]
            img = c[name]
            if "Mask" in name:
                ax.imshow(img, cmap=matplotlib.colors.ListedColormap(["black", "#d9d9d9", "#d1495b"]),
                          vmin=-1, vmax=1, origin="lower", interpolation="nearest")
            else:
                ax.imshow(img, cmap="viridis", origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[k], fontsize=10)
    fig.suptitle("Example training patches (64 x 64 km, north up; black = unknown label)")
    save(fig, "example_patches.png")


def fire_dynamics(raw):
    prev = raw[:, CH["PrevFireMask"]] == 1
    nxt = raw[:, CH["FireMask"]] == 1
    s0, s1 = prev.sum(axis=(1, 2)), nxt.sum(axis=(1, 2))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].scatter(s0, s1, s=4, alpha=0.3, color="#2e4057")
    m = max(s0.max(), s1.max())
    ax[0].plot([1, m], [1, m], "k--", lw=1)
    ax[0].set_xscale("symlog")
    ax[0].set_yscale("symlog")
    ax[0].set_xlabel("burning pixels at t")
    ax[0].set_ylabel("burning pixels at t+1")
    ax[0].set_title("Fire size per patch")
    stayed = (prev & nxt).sum() / prev.sum()
    new = (nxt & ~prev).sum() / nxt.sum()
    ax[1].bar(["day-t fire still\nburning at t+1", "t+1 fire that is\nnew (not burning at t)"], [stayed, new],
              color=["#edae49", "#d1495b"])
    ax[1].set_ylim(0, 1)
    ax[1].set_title("Persistence vs new ignition")
    save(fig, "fire_dynamics.png")
    return {"stayed_burning": float(stayed), "new_fraction_of_t1_fire": float(new)}


def spread_vs_distance(X, y):
    nf = (X[:, FX["prev_fire"]] == 0) & (y >= 0)
    d = X[:, FX["dist_to_fire_km"]]
    bins = [0, 1.5, 2.5, 3.5, 5, 7, 10, 15, 25, 100]
    rates = [y[nf & (d > lo) & (d <= hi)].mean() for lo, hi in zip(bins[:-1], bins[1:])]
    labels = [f"{lo:g}-{hi:g}" for lo, hi in zip(bins[:-1], bins[1:])]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(labels, np.array(rates) * 100, color="#d1495b")
    ax.set_xlabel("distance to nearest day-t fire pixel (km)")
    ax.set_ylabel("% newly burning at t+1")
    ax.set_title("New fire concentrates within a few km of the day-t front")
    save(fig, "spread_vs_distance.png")
    return dict(zip(labels, map(float, rates)))


ORIENTATIONS = {
    "rows N->S, cols W->E": (-1, 1),
    "rows S->N, cols W->E (adopted)": (1, 1),
    "rows N->S, cols E->W": (-1, -1),
    "rows S->N, cols E->W": (1, -1),
}


def wind_orientation(n_boot=1000):
    """Downwind/upwind spread ratio under each image-orientation convention, all training patches.

    For non-burning pixels with day-t fire 1-3 km on exactly one side along the patch-mean wind axis,
    compare P(fire at t+1) when that fire is upwind (pixel downwind) vs downwind (pixel upwind).
    95% CIs from a patch bootstrap.
    """
    a = load_split("train")
    counts = {k: np.zeros((a.shape[0], 4)) for k in ORIENTATIONS}  # down_pos, down_n, up_pos, up_n
    for i in range(a.shape[0]):
        c = clean_channels(a[i])
        fire = (c["PrevFireMask"] == 1).astype(np.float32)
        y = c["FireMask"]
        th = np.deg2rad(c["th"])
        if not np.isfinite(th).any():
            continue
        m = np.arctan2(np.nanmean(np.sin(th)), np.nanmean(np.cos(th)))
        ok = (fire == 0) & (y >= 0)
        for k, (rs, cs) in ORIENTATIONS.items():
            r, cc = rs * np.cos(m), cs * np.sin(m)
            fire_upwind = _directional_fire(fire, r, cc)
            fire_downwind = _directional_fire(fire, -r, -cc)
            down = ok & (fire_upwind == 1) & (fire_downwind == 0)
            up = ok & (fire_downwind == 1) & (fire_upwind == 0)
            counts[k][i] = [y[down].sum(), down.sum(), y[up].sum(), up.sum()]
    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, a.shape[0], a.shape[0]) for _ in range(n_boot)]
    out = {}
    for k, c in counts.items():
        s = c.sum(0)
        reps = [(cb[0] / cb[1]) / (cb[2] / cb[3]) for cb in (c[b].sum(0) for b in boots)]
        out[k] = {"p_new_fire_downwind_of_fire": float(s[0] / s[1]), "p_new_fire_upwind_of_fire": float(s[2] / s[3]),
                  "ratio": float((s[0] / s[1]) / (s[2] / s[3])),
                  "ratio_ci95": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]}
    fig, ax = plt.subplots(figsize=(8, 3.8))
    labels = list(out)
    ratios = [out[k]["ratio"] for k in labels]
    err = np.array([[out[k]["ratio"] - out[k]["ratio_ci95"][0], out[k]["ratio_ci95"][1] - out[k]["ratio"]] for k in labels]).T
    colors = ["#d1495b" if "adopted" in k else "#8d99ae" for k in labels]
    ax.bar(range(len(labels)), ratios, yerr=err, capsize=4, color=colors)
    ax.axhline(1.0, color="k", lw=1, ls="--")
    ax.set_xticks(range(len(labels)), [k.replace(" (adopted)", "\n(adopted)") for k in labels], fontsize=8)
    ax.set_ylabel("P(new fire | downwind of fire) /\nP(new fire | upwind of fire)")
    ax.set_ylim(0.8, 1.2)
    ax.set_title("Wind check on all 14,979 training patches (95% patch-bootstrap CI)")
    save(fig, "wind_orientation.png")
    return out


def terrain_effect(X, y):
    near = (X[:, FX["prev_fire"]] == 0) & (X[:, FX["dist_to_fire_km"]] <= 3) & (y >= 0)
    up = X[:, FX["upslope_from_fire"]]
    groups = {"downslope of fire": up < -0.02, "level": (up >= -0.02) & (up <= 0.02), "upslope of fire": up > 0.02}
    rates = {k: float(y[near & m].mean()) for k, m in groups.items()}
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(list(rates), np.array(list(rates.values())) * 100, color=["#8d99ae", "#edae49", "#d1495b"])
    ax.set_ylabel("% newly burning at t+1")
    ax.set_title("Pixels within 3 km of day-t fire, by relative elevation")
    save(fig, "terrain_effect.png")
    return rates


def feature_correlation(X):
    idx = np.random.default_rng(SEED).choice(len(X), 200000, replace=False)
    df = pd.DataFrame(X[idx], columns=FEATURES)
    corr = df.rank().corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=90, fontsize=7)
    ax.set_yticks(range(len(FEATURES)), FEATURES, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman rho")
    ax.set_title("Engineered feature correlation (200k training pixels)")
    save(fig, "feature_correlation.png")
    pairs = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack()
    return {f"{a} ~ {b}": float(v) for (a, b), v in pairs[pairs.abs() > 0.85].sort_values().items()}


def split_overlap():
    with open(os.path.join(SPLIT_DIR, "split_audit.json")) as f:
        audit = json.load(f)
    m = load_manifest()
    o = audit["overlap"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    labels = ["official eval", "official test", "spatial val", "spatial test"]
    keys = ["eval", "test", "spatial_val", "spatial_test"]
    frac = [o[k]["share_terrain_with_train"] / o[k]["n_patches"] * 100 for k in keys]
    ax[0].bar(labels, frac, color=["#d1495b", "#d1495b", "#00798c", "#00798c"])
    for i, v in enumerate(frac):
        ax[0].text(i, v + 1, f"{v:.0f}%", ha="center")
    ax[0].set_ylabel("% patches sharing terrain with train")
    ax[0].set_ylim(0, 110)
    ax[0].set_title("Spatial leakage audit (terrain fingerprint)")
    sizes = m.groupby("location_group").size().sort_values(ascending=False).values
    ax[1].loglog(np.arange(1, len(sizes) + 1), sizes, ".", color="#2e4057")
    ax[1].set_xlabel("location group rank")
    ax[1].set_ylabel("patches in group")
    ax[1].set_title(f"{len(sizes)} location groups from {len(m):,} patches")
    save(fig, "split_overlap.png")
    return audit


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = {}
    print("class balance ...")
    summary["class_balance"] = class_balance().to_dict(orient="records")
    print("quality audit ...")
    summary["quality_audit"] = quality_audit()
    raw = sample_patches()
    X = batch_features(raw).transpose(0, 2, 3, 1).reshape(-1, len(FEATURES))
    y = raw[:, CH["FireMask"]].reshape(-1)
    print("distributions / examples ...")
    channel_distributions(raw)
    example_patches(raw)
    summary["fire_dynamics_sample"] = fire_dynamics(raw)
    summary["spread_vs_distance_km"] = spread_vs_distance(X, y)
    print("physical plausibility checks ...")
    summary["wind_orientation"] = wind_orientation()
    summary["terrain_effect"] = terrain_effect(X, y)
    summary["highly_correlated_feature_pairs"] = feature_correlation(X[y >= 0])
    summary["split_audit"] = split_overlap()
    with open(os.path.join(OUT_DIR, "eda_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("wind_orientation", "terrain_effect", "fire_dynamics_sample")}, indent=2))


if __name__ == "__main__":
    main()
