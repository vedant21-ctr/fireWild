"""
splits.py - Official and spatially disjoint NDWS splits.

The Kaggle NDWS release has no coordinates or dates, so spatial overlap between the
official train/eval/test splits cannot be read from metadata. Elevation is a static
field, however: two patches that contain an identical, non-flat 8x8 km elevation
block cover the same ground. We use that as a terrain fingerprint:

1. Hash every non-flat 8x8 block on the stride-8 grid of every patch.
2. Slide an 8x8 window over every patch; a hash hit links the two patches.
3. Union-find over the links gives connected "location groups".
4. A control run with mirrored patches must produce ~0 hits (it does: 0 / 1855).

The spatial split assigns whole location groups to train / val / test, so no test
patch shares any terrain with a training patch.

Output: data/splits/ndws_split_manifest.csv with one row per patch
    patch_uid, official_split, official_index, location_group, spatial_split

Usage:
    python -m src.aml.splits
"""

import os
import json
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .ndws import BASE_DIR, SPLITS, CH, load_split

BLOCK = 8
MIN_BLOCK_STD_M = 5.0
SPATIAL_FRACTIONS = {"test": 0.10, "val": 0.10}
SEED = 42

SPLIT_DIR = os.path.join(BASE_DIR, "data", "splits")
MANIFEST_PATH = os.path.join(SPLIT_DIR, "ndws_split_manifest.csv")


def _grid_blocks(elev):
    b = elev.reshape(8, BLOCK, 8, BLOCK).transpose(0, 2, 1, 3).reshape(64, BLOCK * BLOCK)
    return b[b.std(1) > MIN_BLOCK_STD_M]


def _window_blocks(elev):
    w = sliding_window_view(elev, (BLOCK, BLOCK)).reshape(-1, BLOCK * BLOCK)
    return w[w.std(1) > MIN_BLOCK_STD_M]


def terrain_groups(elevations):
    """Connected components of the terrain-overlap graph. Returns (groups, control_hits, control_n)."""
    n = len(elevations)
    index = {}
    for i in range(n):
        for b in _grid_blocks(elevations[i]):
            index.setdefault(b.tobytes(), []).append(i)

    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for w in _window_blocks(elevations[i]):
            for j in index.get(w.tobytes(), ()):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    control_idx = range(0, n, 10)
    control_hits = sum(
        any(w.tobytes() in index for w in _window_blocks(elevations[i][:, ::-1])) for i in control_idx
    )
    groups = np.array([find(i) for i in range(n)])
    return groups, control_hits, len(control_idx)


def assign_spatial_split(groups, seed=SEED):
    """Assign whole groups to test/val (shuffled, greedy until each fraction is reached), rest to train."""
    n = len(groups)
    ids, counts = np.unique(groups, return_counts=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    # groups larger than half a held-out fraction always stay in train so held-out sizes stay close to target
    cap = min(SPATIAL_FRACTIONS.values()) * n / 2
    assignment = {}
    filled = {k: 0 for k in SPATIAL_FRACTIONS}
    for k in order:
        gid, cnt = ids[k], counts[k]
        target = next((s for s, frac in SPATIAL_FRACTIONS.items() if filled[s] < frac * n), None)
        if target is None or cnt > cap:
            assignment[gid] = "train"
        else:
            assignment[gid] = target
            filled[target] += cnt
    return np.array([assignment[g] for g in groups])


def build_manifest():
    rows, elevations = [], []
    for split in SPLITS:
        arr = load_split(split)
        elev = np.asarray(arr[:, CH["elevation"]]).astype(np.int32)
        elevations.append(elev)
        for i in range(len(arr)):
            rows.append({"patch_uid": f"{split}_{i:05d}", "official_split": split, "official_index": i})
    elevations = np.concatenate(elevations)
    groups, hits, n_ctrl = terrain_groups(elevations)
    df = pd.DataFrame(rows)
    _, df["location_group"] = np.unique(groups, return_inverse=True)
    df["spatial_split"] = assign_spatial_split(groups)
    return df, hits, n_ctrl


def overlap_audit(df):
    """How many official eval/test patches share a location group with the official train split."""
    train_groups = set(df.loc[df.official_split == "train", "location_group"])
    out = {}
    for split in ("eval", "test"):
        sub = df[df.official_split == split]
        out[split] = {"n_patches": int(len(sub)), "share_terrain_with_train": int(sub.location_group.isin(train_groups).sum())}
    spatial_train = set(df.loc[df.spatial_split == "train", "location_group"])
    for split in ("val", "test"):
        sub = df[df.spatial_split == split]
        out[f"spatial_{split}"] = {"n_patches": int(len(sub)), "share_terrain_with_train": int(sub.location_group.isin(spatial_train).sum())}
    return out


def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(f"{MANIFEST_PATH} missing. Run `python -m src.aml.splits` first.")
    return pd.read_csv(MANIFEST_PATH)


def main():
    df, hits, n_ctrl = build_manifest()
    os.makedirs(SPLIT_DIR, exist_ok=True)
    df.to_csv(MANIFEST_PATH, index=False)
    sizes = df.groupby("location_group").size()
    audit = {
        "n_patches": int(len(df)),
        "n_location_groups": int(sizes.size),
        "largest_groups": sizes.sort_values(ascending=False).head(5).astype(int).tolist(),
        "singleton_groups": int((sizes == 1).sum()),
        "mirrored_control_hits": f"{hits} / {n_ctrl}",
        "spatial_split_sizes": df.spatial_split.value_counts().astype(int).to_dict(),
        "overlap": overlap_audit(df),
    }
    with open(os.path.join(SPLIT_DIR, "split_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
