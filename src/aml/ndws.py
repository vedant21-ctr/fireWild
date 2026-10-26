"""
ndws.py - Next Day Wildfire Spread (NDWS) data access.

Reads the official Kaggle TFRecords (Huot et al., 2022) and caches each split as a
float32 NumPy memmap of shape (n_patches, 13, 64, 64) for fast repeated access.

Channel order follows CHANNELS below. PrevFireMask is the fire mask at day t,
FireMask is the target at day t+1. Mask values: 1 = fire, 0 = no fire, -1 = unknown.

Usage:
    python -m src.aml.ndws            # download (if needed) and cache all splits
"""

import os
import glob
import json
import numpy as np

PATCH = 64
SPLITS = ("train", "eval", "test")

# 11 environmental inputs + previous fire mask + target
INPUT_CHANNELS = [
    "elevation",   # m (SRTM)
    "th",          # wind direction, degrees clockwise from north, direction wind blows FROM (GRIDMET)
    "vs",          # wind speed, m/s (GRIDMET)
    "tmmn",        # min temperature, K (GRIDMET)
    "tmmx",        # max temperature, K (GRIDMET)
    "sph",         # specific humidity, kg/kg (GRIDMET)
    "pr",          # precipitation, mm (GRIDMET)
    "pdsi",        # Palmer Drought Severity Index (GRIDMET)
    "NDVI",        # vegetation index x 1e4 (VIIRS)
    "population",  # people / km^2 (GPWv4)
    "erc",         # energy release component (NFDRS)
    "PrevFireMask",
]
TARGET_CHANNEL = "FireMask"
CHANNELS = INPUT_CHANNELS + [TARGET_CHANNEL]
CH = {name: i for i, name in enumerate(CHANNELS)}

KAGGLE_DATASET = "fantineh/next-day-wildfire-spread"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(BASE_DIR, "data", "ndws")


def raw_dir():
    """Locate the directory with NDWS TFRecords (NDWS_RAW_DIR env var, else kagglehub cache)."""
    env = os.environ.get("NDWS_RAW_DIR")
    if env:
        return env
    import kagglehub
    return kagglehub.dataset_download(KAGGLE_DATASET)


def tfrecord_files(split, directory):
    files = sorted(glob.glob(os.path.join(directory, f"*_{split}_*.tfrecord")))
    if not files:
        raise FileNotFoundError(f"No NDWS '{split}' TFRecords found in {directory}")
    return files


def cache_path(split):
    return os.path.join(CACHE_DIR, f"{split}.npy")


def build_cache(split, directory):
    """Convert one split's TFRecords into a (N, 13, 64, 64) float32 .npy file."""
    from tfrecord.reader import tfrecord_loader

    patches = []
    for f in tfrecord_files(split, directory):
        for rec in tfrecord_loader(f, None):
            patches.append(np.stack([rec[c].reshape(PATCH, PATCH) for c in CHANNELS]).astype(np.float32))
    arr = np.stack(patches)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(cache_path(split), arr)
    return arr.shape


def load_split(split, mmap=True):
    """Return the cached split as an array of shape (N, 13, 64, 64)."""
    path = cache_path(split)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing. Run `python -m src.aml.ndws` first.")
    return np.load(path, mmap_mode="r" if mmap else None)


def main():
    directory = raw_dir()
    print(f"NDWS TFRecords: {directory}")
    manifest = {"source": f"kaggle:{KAGGLE_DATASET}", "channels": CHANNELS, "splits": {}}
    for split in SPLITS:
        if os.path.exists(cache_path(split)):
            shape = load_split(split).shape
            print(f"  {split:<5}: cached {shape}")
        else:
            shape = build_cache(split, directory)
            print(f"  {split:<5}: built  {shape}")
        manifest["splits"][split] = {"n_patches": int(shape[0]), "files": [os.path.basename(f) for f in tfrecord_files(split, directory)]}
    with open(os.path.join(CACHE_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
