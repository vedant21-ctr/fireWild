"""
data.py - Patch tensors for the spatial models.

Inputs are day-t rasters only, cleaned exactly as in the tabular pipeline
(src/aml/features.py::clean_channels), so both model families see the same data. A single cache holds
every patch in manifest order (data/ndws/dl/, gitignored); each protocol selects its patches from it and
computes its own normalisation from its training split.

Channel sets:
    base       the 13 raster inputs below
    fire       base + engineered burn-context channels
    fire_wind  fire + engineered wind-fire channels
    all        every engineered channel (hybrid option B)

Wind is given as the (east, north) vector the wind blows TOWARD instead of a direction angle, and aspect as
(sin, cos) east/north components, so geometric augmentation can transform them consistently (see apply_dihedral).
Labels keep -1 for unknown pixels; they are masked out of the loss and of every metric.
"""

import os

import numpy as np
import torch

from src.aml.dataset import patch_refs
from src.aml.features import FX, batch_features, clean_channels
from src.aml.ndws import CACHE_DIR, CH, load_split
from src.aml.splits import load_manifest

DL_CHANNELS = [
    "elevation", "wind_u", "wind_v", "tmmn", "tmmx", "sph", "pr", "pdsi", "ndvi",
    "log_population", "erc", "prev_fire", "prev_unknown",
]
FIRE_CHANNELS = ["fire_frac_3", "fire_frac_5", "fire_frac_9", "fire_frac_17", "dist_to_fire_km", "log_patch_fire_count"]
WIND_CHANNELS = ["upwind_fire_3km", "downwind_fire_3km", "fire_wind_alignment", "wind_push"]
OTHER_CHANNELS = ["slope_deg", "aspect_sin", "aspect_cos", "upslope_from_fire", "temp_range", "vpd_kpa", "ndvi_mean_5", "ndvi_std_5"]
ENGINEERED_CHANNELS = FIRE_CHANNELS + WIND_CHANNELS + OTHER_CHANNELS
ALL_CHANNELS = DL_CHANNELS + ENGINEERED_CHANNELS
CHANNEL_SETS = {
    "base": DL_CHANNELS,
    "fire": DL_CHANNELS + FIRE_CHANNELS,
    "fire_wind": DL_CHANNELS + FIRE_CHANNELS + WIND_CHANNELS,
    "all": ALL_CHANNELS,
}
# already 0/1 and left unstandardised
BINARY_CHANNELS = {"prev_fire", "prev_unknown", "upwind_fire_3km", "downwind_fire_3km"}
# signed (east, north) vector components: scaled by their RMS but not mean-centred, so augmentation can negate them
VECTOR_PAIRS = (("wind_u", "wind_v"), ("aspect_sin", "aspect_cos"))
VECTOR_CHANNELS = {c for pair in VECTOR_PAIRS for c in pair}
DL_CACHE_DIR = os.path.join(CACHE_DIR, "dl")
INPUTS_PATH = os.path.join(DL_CACHE_DIR, "inputs.npy")
LABELS_PATH = os.path.join(DL_CACHE_DIR, "labels.npy")
PATCH = 64
ALL_SPLIT_ORDER = ("train", "eval", "test")


def patch_inputs(raw):
    """(13, H, W) raw NDWS patch -> (len(DL_CHANNELS), H, W) float32 inputs (unnormalised, may hold NaN)."""
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


def batch_all_channels(raw):
    """(B, 13, H, W) raw patches -> (B, len(ALL_CHANNELS), H, W) float32 inputs."""
    base = np.stack([patch_inputs(p) for p in raw])
    engineered = batch_features(raw)[:, [FX[c] for c in ENGINEERED_CHANNELS]]
    return np.concatenate([base, engineered], axis=1)


def build_cache(batch=256):
    """Write every patch's channels and labels once, in manifest order."""
    os.makedirs(DL_CACHE_DIR, exist_ok=True)
    n = sum(load_split(s).shape[0] for s in ALL_SPLIT_ORDER)
    X = np.lib.format.open_memmap(INPUTS_PATH, mode="w+", dtype=np.float16, shape=(n, len(ALL_CHANNELS), PATCH, PATCH))
    y = np.lib.format.open_memmap(LABELS_PATH, mode="w+", dtype=np.int8, shape=(n, PATCH, PATCH))
    pos = 0
    for split in ALL_SPLIT_ORDER:
        arr = load_split(split)
        for i in range(0, len(arr), batch):
            raw = np.asarray(arr[i:i + batch])
            X[pos:pos + len(raw)] = batch_all_channels(raw).astype(np.float16)
            y[pos:pos + len(raw)] = raw[:, CH["FireMask"]].astype(np.int8)
            pos += len(raw)
    X.flush()
    y.flush()


def global_indices(protocol, split):
    """Positions in the shared cache for a protocol's split, in the same order as patch_refs."""
    m = load_manifest().reset_index(drop=True)
    lookup = {(a, int(b)): i for i, (a, b) in enumerate(zip(m.official_split, m.official_index, strict=True))}
    return np.array([lookup[(a, int(b))] for a, b in patch_refs(protocol, split)], dtype=np.int64)


def channel_indices(channels):
    return [ALL_CHANNELS.index(c) for c in channels]


def load_channels(protocol, split, channels):
    """Unnormalised float32 inputs and int8 labels for a split, read into RAM."""
    if not os.path.exists(INPUTS_PATH):
        build_cache()
    idx = global_indices(protocol, split)
    X_mm = np.load(INPUTS_PATH, mmap_mode="r")
    y_mm = np.load(LABELS_PATH, mmap_mode="r")
    cols = channel_indices(channels)
    return np.asarray(X_mm[idx][:, cols], dtype=np.float32), np.asarray(y_mm[idx])


def compute_norm(X, channels=DL_CHANNELS):
    """Per-channel mean / std over finite values (call on TRAINING inputs only)."""
    mean, std = [], []
    for i, name in enumerate(channels):
        if name in BINARY_CHANNELS:
            mean.append(0.0)
            std.append(1.0)
            continue
        v = X[:, i][np.isfinite(X[:, i])]
        if name in VECTOR_CHANNELS:
            mean.append(0.0)
            std.append(float(max(np.sqrt(np.mean(v ** 2)), 1e-6)))
            continue
        mean.append(float(v.mean()))
        std.append(float(max(v.std(), 1e-6)))
    return {"channels": list(channels), "mean": mean, "std": std}


def normalise(X, norm):
    """Standardise inputs; missing values become 0 (the training mean)."""
    m = np.asarray(norm["mean"], dtype=np.float32)[None, :, None, None]
    s = np.asarray(norm["std"], dtype=np.float32)[None, :, None, None]
    Z = (X - m) / s
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)


def vector_index_pairs(channels):
    """Index pairs of (east, north) vector channels present in `channels`."""
    return tuple((channels.index(a), channels.index(b)) for a, b in VECTOR_PAIRS if a in channels and b in channels)


U, V = DL_CHANNELS.index("wind_u"), DL_CHANNELS.index("wind_v")


def apply_dihedral(x, y, flip_cols, flip_rows, transpose, vector_pairs=((U, V),)):
    """Apply one dihedral transform to inputs and labels, transforming every vector channel with the image.

    Image axes: rows run south -> north, columns west -> east (src/aml/features.py). Transposing swaps the axes
    (so east and north swap), flipping columns reverses east, flipping rows reverses north. Vector channels are
    not mean-centred (compute_norm), so negating the normalised value negates the physical vector.
    x: (B, C, H, W) tensor, y: (B, H, W) tensor.
    """
    x = x.clone()
    if transpose:
        x, y = x.transpose(-1, -2), y.transpose(-1, -2)
        for east, north in vector_pairs:
            x[:, [east, north]] = x[:, [north, east]]
    if flip_cols:
        x, y = x.flip(-1), y.flip(-1)
        for east, _ in vector_pairs:
            x[:, east] = -x[:, east]
    if flip_rows:
        x, y = x.flip(-2), y.flip(-2)
        for _, north in vector_pairs:
            x[:, north] = -x[:, north]
    return x.contiguous(), y.contiguous()


def augment(x, y, generator=None, vector_pairs=((U, V),)):
    """Random element of the 8-element dihedral group (flips and transposes), one per batch."""
    flip_cols, flip_rows, transpose = torch.randint(0, 2, (3,), generator=generator).tolist()
    return apply_dihedral(x, y, flip_cols, flip_rows, transpose, vector_pairs)


def random_crop(x, y, size, generator=None):
    """Independent random size x size crop of every sample in the batch.

    NDWS patches are centred on fire events, and the centre is more strongly tied to the day t+1 fire than to the
    day t fire (see docs/METHODOLOGY.md). A network trained on full patches can learn "the patch centre burns" from
    its distance to the zero-padded border, which leaks the target. With size = 32 the patch centre can fall anywhere
    in the crop, so position inside the input carries no information.
    """
    b, _, h, w = x.shape
    rows = torch.randint(0, h - size + 1, (b,), generator=generator).tolist()
    cols = torch.randint(0, w - size + 1, (b,), generator=generator).tolist()
    xs = torch.stack([x[i, :, r:r + size, c:c + size] for i, (r, c) in enumerate(zip(rows, cols, strict=True))])
    ys = torch.stack([y[i, r:r + size, c:c + size] for i, (r, c) in enumerate(zip(rows, cols, strict=True))])
    return xs, ys
