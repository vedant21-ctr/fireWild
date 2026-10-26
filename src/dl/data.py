"""
data.py - Patch tensors for the spatial U-Net.

Inputs are the day-t rasters only, cleaned exactly as in the tabular pipeline
(src/aml/features.py::clean_channels), so both model families see the same data:

    elevation, wind_u, wind_v, tmmn, tmmx, sph, pr, pdsi, ndvi, log_population, erc,
    prev_fire, prev_unknown

Wind is given as the (east, north) vector the wind blows TOWARD instead of a direction
angle, so geometric augmentation can transform it consistently (see augment()).

Normalisation statistics (per-channel mean / std) are computed from the TRAINING split
of the chosen protocol only and saved next to the model. Missing values become 0 after
standardisation (the training mean). Labels keep -1 for unknown pixels; they are masked
out of the loss and of every metric.

Tensors are cached as float16 / int8 .npy files under data/ndws/dl/ (gitignored).
"""

import os
import json
import numpy as np
import torch

from src.aml.ndws import CACHE_DIR, CH
from src.aml.features import clean_channels
from src.aml.dataset import patch_refs, iter_patch_batches

DL_CHANNELS = [
    "elevation", "wind_u", "wind_v", "tmmn", "tmmx", "sph", "pr", "pdsi", "ndvi",
    "log_population", "erc", "prev_fire", "prev_unknown",
]
# fire-mask channels are already 0/1 and are not standardised
BINARY_CHANNELS = {"prev_fire", "prev_unknown"}
# signed vector components: divided by their RMS but not mean-centred
VECTOR_CHANNELS = {"wind_u", "wind_v"}
DL_CACHE_DIR = os.path.join(CACHE_DIR, "dl")


def patch_inputs(raw):
    """(13, 64, 64) raw NDWS patch -> (len(DL_CHANNELS), 64, 64) float32 inputs (unnormalised, may hold NaN)."""
    c = clean_channels(raw)
    th = np.deg2rad(c["th"])
    prev = c["PrevFireMask"]
    out = {
        "elevation": c["elevation"],
        "wind_u": -c["vs"] * np.sin(th),
        "wind_v": -c["vs"] * np.cos(th),
        "tmmn": c["tmmn"],
        "tmmx": c["tmmx"],
        "sph": c["sph"],
        "pr": c["pr"],
        "pdsi": c["pdsi"],
        "ndvi": c["NDVI"],
        "log_population": np.log1p(c["population"]),
        "erc": c["erc"],
        "prev_fire": (prev == 1).astype(np.float32),
        "prev_unknown": (prev < 0).astype(np.float32),
    }
    return np.stack([out[k].astype(np.float32) for k in DL_CHANNELS])


def build_arrays(protocol, split):
    """Raw inputs X (N, C, 64, 64) float32, labels y (N, 64, 64) int8 for one split."""
    Xs, ys = [], []
    for raw in iter_patch_batches(patch_refs(protocol, split)):
        Xs.append(np.stack([patch_inputs(p) for p in raw]))
        ys.append(raw[:, CH["FireMask"]].astype(np.int8))
    return np.concatenate(Xs), np.concatenate(ys)


def compute_norm(X):
    """Per-channel mean / std over finite values (call on TRAINING inputs only)."""
    mean, std = [], []
    for i, name in enumerate(DL_CHANNELS):
        if name in BINARY_CHANNELS:
            mean.append(0.0)
            std.append(1.0)
            continue
        v = X[:, i][np.isfinite(X[:, i])]
        if name in VECTOR_CHANNELS:
            # scaled but not centred, so augment() can negate them (see augment)
            mean.append(0.0)
            std.append(float(max(np.sqrt(np.mean(v ** 2)), 1e-6)))
            continue
        mean.append(float(v.mean()))
        std.append(float(max(v.std(), 1e-6)))
    return {"channels": DL_CHANNELS, "mean": mean, "std": std}


def normalise(X, norm):
    m = np.asarray(norm["mean"], dtype=np.float32)[None, :, None, None]
    s = np.asarray(norm["std"], dtype=np.float32)[None, :, None, None]
    Z = (X - m) / s
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)


def load_split_tensors(protocol, split, norm=None):
    """Normalised inputs (float16) and labels for a split, cached on disk.

    The training split computes and returns the normalisation; other splits require it.
    Returns X, y, norm.
    """
    os.makedirs(DL_CACHE_DIR, exist_ok=True)
    x_path = os.path.join(DL_CACHE_DIR, f"{protocol}_{split}_X.npy")
    y_path = os.path.join(DL_CACHE_DIR, f"{protocol}_{split}_y.npy")
    n_path = os.path.join(DL_CACHE_DIR, f"{protocol}_norm.json")
    if split == "train" and norm is None and os.path.exists(n_path) and os.path.exists(x_path):
        with open(n_path) as f:
            norm = json.load(f)
    if not (os.path.exists(x_path) and os.path.exists(y_path) and norm is not None):
        X, y = build_arrays(protocol, split)
        if split == "train" and norm is None:
            norm = compute_norm(X)
            with open(n_path, "w") as f:
                json.dump(norm, f, indent=2)
        if norm is None:
            raise ValueError("normalisation statistics come from the training split; load it first")
        np.save(x_path, normalise(X, norm))
        np.save(y_path, y)
    return np.load(x_path, mmap_mode="r"), np.load(y_path, mmap_mode="r"), norm


U, V = DL_CHANNELS.index("wind_u"), DL_CHANNELS.index("wind_v")


def apply_dihedral(x, y, flip_cols, flip_rows, transpose):
    """Apply one dihedral transform to inputs and labels, transforming the wind vector with the image.

    Image axes: rows run south -> north, columns west -> east (src/aml/features.py).
    Transposing swaps the axes (so u and v swap), flipping columns reverses east (u -> -u),
    flipping rows reverses north (v -> -v). Wind channels are not mean-centred (compute_norm),
    so negating the normalised value negates the physical vector.
    x: (B, C, H, W) tensor, y: (B, H, W) tensor.
    """
    x = x.clone()
    if transpose:
        x, y = x.transpose(-1, -2), y.transpose(-1, -2)
        x[:, [U, V]] = x[:, [V, U]]
    if flip_cols:
        x, y = x.flip(-1), y.flip(-1)
        x[:, U] = -x[:, U]
    if flip_rows:
        x, y = x.flip(-2), y.flip(-2)
        x[:, V] = -x[:, V]
    return x.contiguous(), y.contiguous()


def augment(x, y, generator=None):
    """Random element of the 8-element dihedral group (flips and transposes), one per batch."""
    flip_cols, flip_rows, transpose = torch.randint(0, 2, (3,), generator=generator).tolist()
    return apply_dihedral(x, y, flip_cols, flip_rows, transpose)


def random_crop(x, y, size, generator=None):
    """Independent random size x size crop of every sample in the batch.

    NDWS patches are centred on fire events, and the centre is more strongly tied to the
    day t+1 fire than to the day t fire (see docs/SPATIAL_UNET_DUAL_TARGET.md). A network
    trained on full patches can learn "the patch centre burns" from its distance to the
    zero-padded border, which leaks the target. With size = 32, the patch centre can fall
    anywhere in the crop, so position inside the input carries no information.
    """
    b, _, h, w = x.shape
    rows = torch.randint(0, h - size + 1, (b,), generator=generator).tolist()
    cols = torch.randint(0, w - size + 1, (b,), generator=generator).tolist()
    xs = torch.stack([x[i, :, r:r + size, c:c + size] for i, (r, c) in enumerate(zip(rows, cols))])
    ys = torch.stack([y[i, r:r + size, c:c + size] for i, (r, c) in enumerate(zip(rows, cols))])
    return xs, ys
