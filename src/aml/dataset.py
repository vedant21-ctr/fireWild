"""
dataset.py - Pixel-level tabular datasets built from NDWS patches.

Two evaluation protocols share one interface:
    official - NDWS train / eval / test as released (comparable with Huot et al., 2022)
    spatial  - location-group split from src/aml/splits.py (no terrain shared with train)

Split names are normalised to train / val / test for both protocols.

Training rows: every positive pixel plus a uniform random fraction NEG_RATE of negative
pixels, drawn from the training split only. `w` (1 / NEG_RATE for negatives) restores the
true class prior; train.py uses it for the early-stopping sample and prior-corrects predictions.
Validation and test are never subsampled or reweighted: every pixel with a known label
is scored.
"""

import numpy as np

from .ndws import load_split
from .splits import load_manifest
from .features import batch_features, patch_target, FEATURES

PROTOCOLS = ("official", "spatial")
NEG_RATE = 0.02
BATCH = 256

_OFFICIAL_NAME = {"train": "train", "val": "eval", "test": "test"}


def patch_refs(protocol, split):
    """List of (official_split, index) pairs belonging to `split` under `protocol`."""
    if protocol == "official":
        name = _OFFICIAL_NAME[split]
        return [(name, i) for i in range(load_split(name).shape[0])]
    if protocol == "spatial":
        m = load_manifest()
        sub = m[m.spatial_split == split]
        return list(zip(sub.official_split, sub.official_index))
    raise ValueError(f"unknown protocol {protocol!r}")


def iter_patch_batches(refs, batch=BATCH):
    """Yield raw patch arrays (B, 13, 64, 64) for the given refs, in order."""
    arrays = {}
    for start in range(0, len(refs), batch):
        chunk = refs[start:start + batch]
        out = []
        for split, idx in chunk:
            if split not in arrays:
                arrays[split] = load_split(split)
            out.append(np.asarray(arrays[split][idx]))
        yield np.stack(out)


def _flatten(feats, target):
    """(B, F, 64, 64), (B, 64, 64) -> X (n, F), y (n,), patch position (n,), pixel index (n,)
    for known-label pixels."""
    b = feats.shape[0]
    n_pix = target.shape[1] * target.shape[2]
    X = feats.transpose(0, 2, 3, 1).reshape(-1, feats.shape[1])
    y = target.reshape(-1)
    pos = np.repeat(np.arange(b), n_pix)
    pix = np.tile(np.arange(n_pix), b)
    keep = y >= 0
    return X[keep], y[keep].astype(np.int8), pos[keep], pix[keep]


def training_sample(protocol, split="train", neg_rate=NEG_RATE, seed=42):
    """Positive pixels + subsampled negatives with importance weights. Returns X, y, w."""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for raw in iter_patch_batches(patch_refs(protocol, split)):
        X, y, _, _ = _flatten(batch_features(raw), np.stack([patch_target(p) for p in raw]))
        keep = (y == 1) | (rng.random(len(y)) < neg_rate)
        Xs.append(X[keep])
        ys.append(y[keep])
    X, y = np.concatenate(Xs), np.concatenate(ys)
    w = np.where(y == 1, 1.0, 1.0 / neg_rate).astype(np.float32)
    return X, y, w


def eval_pixels(protocol, split):
    """Every known-label pixel of a split: X (n, F), y (n,), patch index in split (n,), pixel index (n,)."""
    Xs, ys, patches, pixels = [], [], [], []
    offset = 0
    for raw in iter_patch_batches(patch_refs(protocol, split)):
        X, y, pos, pix = _flatten(batch_features(raw), np.stack([patch_target(p) for p in raw]))
        Xs.append(X)
        ys.append(y)
        patches.append((pos + offset).astype(np.int32))
        pixels.append(pix.astype(np.int16))
        offset += raw.shape[0]
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(patches), np.concatenate(pixels)


def feature_columns(groups=None):
    """Indices into FEATURES for a list of feature-group names (None = all)."""
    from .features import FEATURE_GROUPS, FX
    if groups is None:
        return list(range(len(FEATURES)))
    return [FX[f] for g in groups for f in FEATURE_GROUPS[g]]
