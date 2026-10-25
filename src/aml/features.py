"""
features.py - Per-pixel feature engineering for NDWS next-day fire prediction.

Every 64x64 patch is turned into a (n_features, 64, 64) stack. Each feature for a
pixel is computed only from day-t inputs of the same patch (no day t+1 information),
so a pixel row can be scored as soon as the day-t rasters are available.

Feature groups (used for ablations):
    raw         - cleaned environmental channels
    fire        - lagged burn activity and neighbourhood aggregates of PrevFireMask
    wind        - wind vectors and wind-fire alignment
    terrain     - slope, aspect, upslope relation to the nearest fire
    weather     - derived dryness indicators (temperature range, vapour pressure deficit)
    vegetation  - local NDVI statistics

Image orientation: the Kaggle release carries no georeferencing, so the orientation
was determined empirically (src/aml/eda.py::wind_orientation, all training patches).
Only with rows running SOUTH -> NORTH and columns west -> east is new fire significantly
more likely downwind than upwind of day-t fire: ratio 1.12 (95% CI 1.09-1.15). Rows
north -> south gives 1.01 (0.98-1.04), i.e. no wind effect. `th` is the meteorological wind
direction (wind blows FROM th).
"""

import numpy as np
from scipy import ndimage

from .ndws import CH

PIXEL_M = 1000.0
NO_FIRE_DIST_KM = 64.0

FEATURE_GROUPS = {
    "raw": [
        "elevation", "wind_speed", "tmmn", "tmmx", "sph", "pr", "pdsi",
        "ndvi", "log_population", "erc",
    ],
    "fire": [
        "prev_fire", "prev_unknown", "fire_frac_3", "fire_frac_5", "fire_frac_9",
        "fire_frac_17", "dist_to_fire_km", "log_patch_fire_count",
    ],
    "wind": [
        "wind_u", "wind_v", "upwind_fire_3km", "downwind_fire_3km",
        "fire_wind_alignment", "wind_push",
    ],
    "terrain": ["slope_deg", "aspect_sin", "aspect_cos", "upslope_from_fire"],
    "weather": ["temp_range", "vpd_kpa"],
    "vegetation": ["ndvi_mean_5", "ndvi_std_5"],
}
FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]
FX = {name: i for i, name in enumerate(FEATURES)}


def clean_channels(patch):
    """Return a dict of cleaned (64, 64) float32 arrays from a raw (13, 64, 64) patch.

    Values outside physically possible ranges become NaN (see the EDA quality audit):
    temperatures outside 200-330 K (NDWS stores missing as 0 K; maxima reach 344 K),
    humidity <= 0, negative wind speed, and wind directions outside [0, 361] degrees
    (corrupted values down to -5e5 occur). Directions in (360, 361] are wrapped.
    Negative precipitation (interpolation artefacts) is clipped to 0. Elevations below
    0 m are kept: sub-sea-level terrain is real (e.g. Death Valley, Salton Sea).
    """
    c = {name: np.array(patch[i], dtype=np.float32) for name, i in CH.items()}
    for k in ("tmmn", "tmmx"):
        c[k][(c[k] < VALID_TEMP_K[0]) | (c[k] > VALID_TEMP_K[1])] = np.nan
    c["sph"][c["sph"] <= 0] = np.nan
    c["vs"][c["vs"] < 0] = np.nan
    th = c["th"]
    th[(th < 0) | (th > 361)] = np.nan
    c["th"] = np.mod(th, 360.0)
    c["pr"] = np.clip(c["pr"], 0.0, None)
    c["NDVI"] = np.clip(c["NDVI"] / 1e4, -1.0, 1.0)
    c["population"] = np.clip(c["population"], 0.0, None)
    return c


ROW_NORTH = 1.0  # +1 row = 1 km north (see module docstring)
VALID_TEMP_K = (200.0, 330.0)


def _shift(arr, drow, dcol):
    """Integer shift with zero fill: out[r, c] = arr[r + drow, c + dcol]."""
    out = np.zeros_like(arr)
    h, w = arr.shape
    r0, r1 = max(0, -drow), min(h, h - drow)
    c0, c1 = max(0, -dcol), min(w, w - dcol)
    if r0 < r1 and c0 < c1:
        out[r0:r1, c0:c1] = arr[r0 + drow:r1 + drow, c0 + dcol:c1 + dcol]
    return out


def _directional_fire(fire, row_step, col_step, max_km=3):
    """1 if any fire pixel lies 1..max_km pixels along (row_step, col_step) from the pixel."""
    out = np.zeros_like(fire)
    for d in range(1, max_km + 1):
        out = np.maximum(out, _shift(fire, int(round(d * row_step)), int(round(d * col_step))))
    return out


def patch_features(patch):
    """Compute the (len(FEATURES), 64, 64) feature stack for one raw patch."""
    c = clean_channels(patch)
    prev = c["PrevFireMask"]
    fire = (prev == 1).astype(np.float32)
    f = {}

    # raw
    f["elevation"] = c["elevation"]
    f["wind_speed"] = c["vs"]
    for k in ("tmmn", "tmmx", "sph", "pr", "pdsi", "erc"):
        f[k] = c[k]
    f["ndvi"] = c["NDVI"]
    f["log_population"] = np.log1p(c["population"])

    # fire: lagged burn activity and neighbourhood aggregates
    f["prev_fire"] = fire
    f["prev_unknown"] = (prev < 0).astype(np.float32)
    for k in (3, 5, 9, 17):
        f[f"fire_frac_{k}"] = ndimage.uniform_filter(fire, size=k, mode="constant")
    n_fire = fire.sum()
    if n_fire > 0:
        dist, (ir, ic) = ndimage.distance_transform_edt(1 - fire, return_indices=True)
        dist_km = dist.astype(np.float32) * PIXEL_M / 1000.0
    else:
        dist_km = np.full_like(fire, NO_FIRE_DIST_KM)
        ir, ic = np.indices(fire.shape)
    f["dist_to_fire_km"] = dist_km
    f["log_patch_fire_count"] = np.full_like(fire, np.log1p(n_fire))

    # wind: the field is ~4 km GRIDMET, so a patch-mean direction is used for the shifts
    th_rad = np.deg2rad(c["th"])
    to_east, to_north = -np.sin(th_rad), -np.cos(th_rad)  # unit vector the wind blows TOWARD
    f["wind_u"] = c["vs"] * to_east
    f["wind_v"] = c["vs"] * to_north
    if np.isfinite(th_rad).any():
        mean_th = np.arctan2(np.nanmean(np.sin(th_rad)), np.nanmean(np.cos(th_rad)))
        # upwind point = pixel + d * (sin th, cos th) in (east, north)
        up_row, up_col = ROW_NORTH * np.cos(mean_th), np.sin(mean_th)
        f["upwind_fire_3km"] = _directional_fire(fire, up_row, up_col)
        f["downwind_fire_3km"] = _directional_fire(fire, -up_row, -up_col)
    else:
        f["upwind_fire_3km"] = f["downwind_fire_3km"] = np.full_like(fire, np.nan)
    rows, cols = np.indices(fire.shape)
    vec_east = (cols - ic).astype(np.float32)   # vector from nearest fire pixel to this pixel
    vec_north = ROW_NORTH * (rows - ir).astype(np.float32)
    norm = np.hypot(vec_east, vec_north)
    with np.errstate(invalid="ignore", divide="ignore"):
        align = (vec_east * to_east + vec_north * to_north) / norm
    align[(norm == 0) | (n_fire == 0)] = 0.0
    f["fire_wind_alignment"] = np.nan_to_num(align).astype(np.float32)  # unknown wind -> no alignment
    f["wind_push"] = f["fire_wind_alignment"] * c["vs"] / (1.0 + dist_km)

    # terrain
    elev = np.nan_to_num(c["elevation"], nan=np.nanmean(c["elevation"]) if np.isfinite(c["elevation"]).any() else 0.0)
    dz_drow, dz_dcol = np.gradient(elev, PIXEL_M)
    dz_dnorth, dz_deast = ROW_NORTH * dz_drow, dz_dcol
    f["slope_deg"] = np.degrees(np.arctan(np.hypot(dz_deast, dz_dnorth))).astype(np.float32)
    aspect = np.arctan2(-dz_deast, -dz_dnorth)  # downslope-facing direction, clockwise from north
    f["aspect_sin"] = np.sin(aspect).astype(np.float32)
    f["aspect_cos"] = np.cos(aspect).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        upslope = (elev - elev[ir, ic]) / (dist_km * 1000.0)
    upslope[(dist_km == 0) | (n_fire == 0)] = 0.0
    f["upslope_from_fire"] = np.clip(np.nan_to_num(upslope), -1.0, 1.0).astype(np.float32)

    # weather: dryness indicators
    f["temp_range"] = c["tmmx"] - c["tmmn"]
    t_c = c["tmmx"] - 273.15
    es = 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))                       # kPa, at Tmax
    p_kpa = 101.3 * ((293.0 - 0.0065 * elev) / 293.0) ** 5.26                # FAO-56 pressure
    ea = c["sph"] * p_kpa / (0.622 + 0.378 * c["sph"])                       # actual vapour pressure
    f["vpd_kpa"] = np.clip(es - ea, 0.0, None)

    # vegetation
    ndvi = np.nan_to_num(c["NDVI"])
    mean5 = ndimage.uniform_filter(ndvi, size=5, mode="nearest")
    sq5 = ndimage.uniform_filter(ndvi * ndvi, size=5, mode="nearest")
    f["ndvi_mean_5"] = mean5
    f["ndvi_std_5"] = np.sqrt(np.clip(sq5 - mean5 ** 2, 0.0, None))

    return np.stack([np.asarray(f[name], dtype=np.float32) for name in FEATURES])


def patch_target(patch):
    """Return the (64, 64) day t+1 target with -1 for unknown pixels."""
    return np.asarray(patch[CH["FireMask"]], dtype=np.float32)


def batch_features(patches):
    """Features for a batch of patches: (N, F, 64, 64)."""
    return np.stack([patch_features(p) for p in patches])
