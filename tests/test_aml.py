"""
test_aml.py - Unit tests for the Phase 1 NDWS pipeline (src/aml).

Feature tests use hand-built patches with known answers, so they run without the dataset.
The split test runs against the committed manifest in data/splits/.
"""

import os
import unittest
import numpy as np
from sklearn.metrics import average_precision_score

from src.aml.ndws import CH, CHANNELS
from src.aml.features import patch_features, clean_channels, FEATURES, FX
from src.aml.metrics import BootstrapAP, best_f1_threshold, compute_metrics
from src.aml.train import prior_correct
from src.aml.splits import MANIFEST_PATH


def make_patch(fire_cells=((32, 32),), wind_from_deg=0.0, wind_speed=5.0, elev_north_gradient=0.0):
    """A clean synthetic (13, 64, 64) patch. Row index increases northward (see features.py)."""
    p = np.zeros((len(CHANNELS), 64, 64), dtype=np.float32)
    rows = np.arange(64, dtype=np.float32)[:, None] * np.ones((1, 64), dtype=np.float32)
    p[CH["elevation"]] = 500.0 + elev_north_gradient * rows
    p[CH["th"]] = wind_from_deg
    p[CH["vs"]] = wind_speed
    p[CH["tmmn"]] = 285.0
    p[CH["tmmx"]] = 300.0
    p[CH["sph"]] = 0.005
    p[CH["NDVI"]] = 5000.0
    p[CH["erc"]] = 60.0
    for r, c in fire_cells:
        p[CH["PrevFireMask"], r, c] = 1.0
    return p


class TestFeatures(unittest.TestCase):
    def test_shape_and_finite(self):
        f = patch_features(make_patch())
        self.assertEqual(f.shape, (len(FEATURES), 64, 64))
        self.assertTrue(np.isfinite(f).all())

    def test_missing_sentinels_become_nan(self):
        p = make_patch()
        p[CH["tmmx"], 0, 0] = 0.0
        p[CH["tmmx"], 0, 1] = 344.0
        p[CH["th"], 1, 1] = -496522.0
        p[CH["th"], 1, 2] = 360.5
        p[CH["elevation"], 2, 2] = -50.0
        c = clean_channels(p)
        self.assertTrue(np.isnan(c["tmmx"][0, 0]))
        self.assertTrue(np.isnan(c["tmmx"][0, 1]))
        self.assertTrue(np.isnan(c["th"][1, 1]))
        self.assertAlmostEqual(float(c["th"][1, 2]), 0.5, places=4)
        self.assertEqual(float(c["elevation"][2, 2]), -50.0)  # below sea level is valid
        self.assertTrue(np.isfinite(patch_features(p)).sum() > 0)

    def test_distance_to_fire(self):
        f = patch_features(make_patch())
        self.assertEqual(f[FX["dist_to_fire_km"], 32, 32], 0.0)
        self.assertAlmostEqual(float(f[FX["dist_to_fire_km"], 32, 35]), 3.0, places=5)
        self.assertAlmostEqual(float(f[FX["dist_to_fire_km"], 35, 36]), 5.0, places=5)

    def test_neighbourhood_fraction(self):
        f = patch_features(make_patch())
        self.assertAlmostEqual(float(f[FX["fire_frac_3"], 33, 33]), 1 / 9, places=5)
        self.assertEqual(float(f[FX["fire_frac_3"], 40, 40]), 0.0)

    def test_wind_from_north_puts_downwind_to_the_south(self):
        # Wind FROM north (0 deg) blows toward the south (lower row index).
        f = patch_features(make_patch(wind_from_deg=0.0))
        south, north = (30, 32), (34, 32)
        self.assertEqual(f[FX["upwind_fire_3km"]][south], 1.0)    # fire lies upwind of a southern pixel
        self.assertEqual(f[FX["downwind_fire_3km"]][south], 0.0)
        self.assertEqual(f[FX["downwind_fire_3km"]][north], 1.0)
        self.assertGreater(f[FX["fire_wind_alignment"]][south], 0.99)
        self.assertLess(f[FX["fire_wind_alignment"]][north], -0.99)
        self.assertLess(f[FX["wind_v"], 0, 0], 0.0)                # toward-south component is negative

    def test_wind_from_west_puts_downwind_to_the_east(self):
        f = patch_features(make_patch(wind_from_deg=270.0))
        self.assertEqual(f[FX["upwind_fire_3km"], 32, 34], 1.0)
        self.assertGreater(f[FX["fire_wind_alignment"], 32, 34], 0.99)
        self.assertGreater(f[FX["wind_u"], 0, 0], 0.0)

    def test_upslope_from_fire(self):
        # elevation rises 50 m per km northward
        f = patch_features(make_patch(elev_north_gradient=50.0))
        self.assertGreater(f[FX["upslope_from_fire"], 36, 32], 0.04)
        self.assertLess(f[FX["upslope_from_fire"], 28, 32], -0.04)
        self.assertAlmostEqual(float(f[FX["slope_deg"], 20, 20]), float(np.degrees(np.arctan(0.05))), places=3)
        self.assertAlmostEqual(float(f[FX["aspect_cos"], 20, 20]), -1.0, places=5)  # faces south (downslope)

    def test_no_fire_patch(self):
        f = patch_features(make_patch(fire_cells=()))
        self.assertTrue(np.isfinite(f).all())
        self.assertEqual(f[FX["fire_wind_alignment"]].max(), 0.0)

    def test_features_ignore_target(self):
        p = make_patch()
        a = patch_features(p)
        p[CH["FireMask"]] = 1.0
        np.testing.assert_array_equal(a, patch_features(p))


class TestMetrics(unittest.TestCase):
    def test_bootstrap_ap_matches_sklearn_with_ties(self):
        rng = np.random.default_rng(0)
        y = (rng.random(50000) < 0.03).astype(int)
        p = np.round(rng.random(50000) * 0.5 + 0.3 * y, 2)
        patch = rng.integers(0, 200, 50000)
        self.assertAlmostEqual(BootstrapAP(y, p, patch).ap(np.ones(200)), average_precision_score(y, p), places=10)

    def test_threshold_and_metrics(self):
        y = np.array([0, 0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        t = best_f1_threshold(y, p)
        m = compute_metrics(y, p, t)
        self.assertEqual(m["f1"], 1.0)
        self.assertEqual(m["pr_auc"], 1.0)

    def test_prior_correction(self):
        p = np.array([0.0, 0.5, 0.9, 1.0])
        np.testing.assert_allclose(prior_correct(p, 1.0), p, atol=1e-7)
        q = prior_correct(p, 0.02)
        self.assertAlmostEqual(float(q[1]), 0.02 / 1.02, places=6)
        self.assertTrue(np.all(np.diff(q) > 0))


@unittest.skipUnless(os.path.exists(MANIFEST_PATH), "split manifest not built")
class TestSplits(unittest.TestCase):
    def test_spatial_split_is_location_disjoint(self):
        import pandas as pd
        m = pd.read_csv(MANIFEST_PATH)
        groups = {s: set(m.loc[m.spatial_split == s, "location_group"]) for s in ("train", "val", "test")}
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["train"] & groups["val"])
        self.assertFalse(groups["val"] & groups["test"])
        self.assertEqual(len(m), 18545)


if __name__ == "__main__":
    unittest.main()
