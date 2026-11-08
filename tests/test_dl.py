"""
test_dl.py - Unit tests for the spatial models and hybrid (src/dl, src/hybrid).

All tests use synthetic tensors and run without the NDWS data or a GPU.
"""

import itertools
import unittest

import numpy as np
import torch  # must load before LightGBM / XGBoost on macOS (see tests/conftest.py)

from src.aml.ndws import CH, CHANNELS
from src.dl.data import (
    ALL_CHANNELS,
    CHANNEL_SETS,
    DL_CHANNELS,
    U,
    V,
    apply_dihedral,
    compute_norm,
    normalise,
    patch_inputs,
    random_crop,
    vector_index_pairs,
)
from src.dl.model import CompactCNN, UNet, build_model, make_loss, masked_bce, masked_dice
from src.evaluation.compare import centre_prior, gather_maps, target_masks
from src.hybrid.fusion import fit_fusion, fusion_predict, logit, split_val_patches

ELEV = DL_CHANNELS.index("elevation")
FIRE = DL_CHANNELS.index("prev_fire")


def raw_patch(wind_from_deg=0.0):
    p = np.zeros((len(CHANNELS), 64, 64), dtype=np.float32)
    p[CH["th"]] = wind_from_deg
    p[CH["vs"]] = 5.0
    p[CH["tmmn"]], p[CH["tmmx"]], p[CH["sph"]] = 285.0, 300.0, 0.005
    p[CH["PrevFireMask"], 32, 32] = 1.0
    return p


class TestInputs(unittest.TestCase):
    def test_channels_and_wind_vector(self):
        x = patch_inputs(raw_patch(wind_from_deg=0.0))  # wind from north blows toward south
        self.assertEqual(x.shape, (len(DL_CHANNELS), 64, 64))
        self.assertAlmostEqual(float(x[U, 0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(x[V, 0, 0]), -5.0, places=5)
        self.assertEqual(float(x[FIRE, 32, 32]), 1.0)

    def test_inputs_ignore_target(self):
        p = raw_patch()
        a = patch_inputs(p)
        p[CH["FireMask"]] = 1.0
        np.testing.assert_array_equal(a, patch_inputs(p))

    def test_normalisation(self):
        X = np.stack([patch_inputs(raw_patch(d)) for d in (0.0, 90.0, 200.0)])
        X[0, ELEV, 0, 0] = np.nan
        norm = compute_norm(X)
        self.assertEqual(norm["mean"][U], 0.0)  # wind is scaled, never centred
        self.assertEqual(norm["mean"][FIRE], 0.0)
        self.assertEqual(norm["std"][FIRE], 1.0)
        Z = normalise(X, norm)
        self.assertTrue(np.isfinite(Z.astype(np.float32)).all())
        np.testing.assert_array_equal(Z[:, FIRE], X[:, FIRE])


class TestAugmentation(unittest.TestCase):
    def test_wind_stays_consistent_with_geometry(self):
        """A marker placed one pixel downwind of the fire must stay downwind under all 8 transforms."""
        x = torch.zeros(1, len(DL_CHANNELS), 64, 64)
        x[0, FIRE, 32, 32] = 1.0
        x[0, U], x[0, V] = 1.0, 1.0           # blowing toward north-east
        x[0, ELEV, 33, 33] = 1.0              # +1 row = north, +1 col = east
        y = x[:, ELEV].clone()
        for fc, fr, tr in itertools.product((0, 1), repeat=3):
            xa, ya = apply_dihedral(x, y, fc, fr, tr)
            r0, c0 = map(int, torch.nonzero(xa[0, FIRE])[0])
            u, v = float(xa[0, U, r0, c0]), float(xa[0, V, r0, c0])
            expected = (r0 + int(np.sign(v)), c0 + int(np.sign(u)))
            self.assertEqual(tuple(map(int, torch.nonzero(xa[0, ELEV])[0])), expected, (fc, fr, tr))
            torch.testing.assert_close(ya, xa[:, ELEV])  # labels move with the inputs

    def test_random_crop_keeps_inputs_and_labels_aligned(self):
        x = torch.randn(4, len(DL_CHANNELS), 64, 64)
        y = x[:, 0].clone()
        xs, ys = random_crop(x, y, 32, torch.Generator().manual_seed(0))
        self.assertEqual(tuple(xs.shape), (4, len(DL_CHANNELS), 32, 32))
        torch.testing.assert_close(ys, xs[:, 0])

    def test_identity(self):
        x, y = torch.randn(2, len(DL_CHANNELS), 64, 64), torch.randint(-1, 2, (2, 64, 64))
        xa, ya = apply_dihedral(x, y, 0, 0, 0)
        torch.testing.assert_close(xa, x)
        torch.testing.assert_close(ya, y)


class TestModel(unittest.TestCase):
    def test_output_shape(self):
        model = UNet(len(DL_CHANNELS), base=8)
        self.assertEqual(model(torch.randn(2, len(DL_CHANNELS), 64, 64)).shape, (2, 64, 64))

    def test_masked_loss_ignores_unknown_labels(self):
        logits = torch.randn(1, 4, 4)
        y = torch.tensor([[[1, 0, -1, 0]] * 4])
        known = y >= 0
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[known], y[known].float())
        torch.testing.assert_close(masked_bce(logits, y), expected)
        logits2 = logits.clone()
        logits2[~known] = 100.0  # predictions on unknown pixels must not matter
        torch.testing.assert_close(masked_bce(logits2, y), expected)


class TestSpatialModels(unittest.TestCase):
    def test_cnn_keeps_resolution(self):
        model = CompactCNN(len(DL_CHANNELS), base=8)
        self.assertEqual(model(torch.randn(2, len(DL_CHANNELS), 32, 32)).shape, (2, 32, 32))

    def test_cnn_receptive_field(self):
        torch.manual_seed(0)
        model = CompactCNN(1, base=4, dropout=0.0).eval()
        x = torch.randn(1, 1, 63, 63, requires_grad=True)
        model(x)[0, 31, 31].backward()
        rows = torch.nonzero(x.grad[0, 0].abs().sum(1))
        self.assertLessEqual(int(rows.max() - rows.min()) + 1, 29)

    def test_unet_without_skip_connections(self):
        model = UNet(len(DL_CHANNELS), base=8, skip_connections=False)
        self.assertEqual(model(torch.randn(2, len(DL_CHANNELS), 64, 64)).shape, (2, 64, 64))

    def test_build_model(self):
        cfg = {"architecture": "cnn", "base": 8, "dropout": 0.1}
        self.assertIsInstance(build_model(cfg, 13), CompactCNN)
        with self.assertRaises(ValueError):
            build_model({**cfg, "architecture": "x"}, 13)

    def test_dice_is_low_for_a_correct_mask_and_ignores_unknown_labels(self):
        y = torch.zeros(1, 8, 8, dtype=torch.long)
        y[0, 2:4, 2:4] = 1
        good = (y.float() * 40) - 20
        bad = -good
        self.assertLess(float(masked_dice(good, y)), 0.1)
        self.assertGreater(float(masked_dice(bad, y)), 0.9)
        y2 = y.clone()
        y2[0, 7, 7] = -1
        logits = good.clone()
        logits[0, 7, 7] = 100.0
        self.assertAlmostEqual(float(masked_dice(good, y)), float(masked_dice(logits, y2)), places=5)

    def test_loss_registry(self):
        logits, y = torch.randn(1, 4, 4), torch.randint(0, 2, (1, 4, 4))
        for loss in ("bce", "weighted_bce", "dice", "bce_dice"):
            self.assertTrue(torch.isfinite(make_loss({"loss": loss, "pos_weight": 3.0}, "cpu")(logits, y)))

    def test_channel_sets_nest_and_are_listed(self):
        self.assertEqual(CHANNEL_SETS["all"], ALL_CHANNELS)
        self.assertTrue(set(CHANNEL_SETS["base"]) < set(CHANNEL_SETS["fire"]) < set(CHANNEL_SETS["fire_wind"]) < set(CHANNEL_SETS["all"]))
        self.assertEqual(len(set(ALL_CHANNELS)), len(ALL_CHANNELS))

    def test_aspect_transforms_like_a_vector(self):
        ch = ["aspect_sin", "aspect_cos"]
        pairs = vector_index_pairs(ch)
        x = torch.zeros(1, 2, 4, 4)
        x[0, 0], x[0, 1] = 1.0, 0.0  # facing east
        xa, _ = apply_dihedral(x, torch.zeros(1, 4, 4), 1, 0, 0, pairs)
        self.assertEqual(float(xa[0, 0, 0, 0]), -1.0)
        xt, _ = apply_dihedral(x, torch.zeros(1, 4, 4), 0, 0, 1, pairs)
        self.assertEqual(float(xt[0, 1, 0, 0]), 1.0)


class TestFusion(unittest.TestCase):
    def test_logit_is_finite_at_the_boundaries(self):
        self.assertTrue(np.isfinite(logit(np.array([0.0, 0.5, 1.0]))).all())

    def test_validation_halves_keep_patches_whole(self):
        patch = np.repeat(np.arange(100), 10)
        fit, select = split_val_patches(patch, 0.5, seed=1)
        self.assertFalse((fit & select).any())
        self.assertTrue((fit | select).all())
        for i in range(100):
            self.assertEqual(len(set(fit[patch == i])), 1)

    def test_fusion_uses_the_more_informative_input(self):
        rng = np.random.default_rng(0)
        y = (rng.random(20000) < 0.2).astype(np.int8)
        good = np.clip(0.2 + 0.5 * y + rng.normal(0, 0.1, len(y)), 0.01, 0.99)
        noise = rng.random(len(y)) * 0.98 + 0.01
        scores = {"tree": good.astype(np.float32), "unet": noise.astype(np.float32)}
        model = fit_fusion(scores, y, np.ones(len(y), dtype=bool), ["tree", "unet"])
        self.assertGreater(model.coef_[0][0], 5 * abs(model.coef_[0][1]))
        p = fusion_predict(model, scores, ["tree", "unet"])
        self.assertTrue(((p >= 0) & (p <= 1)).all())


class TestEvaluation(unittest.TestCase):
    def test_transition_mask(self):
        masks = target_masks(np.array([1, 0, -1, 0], dtype=np.int8))
        np.testing.assert_array_equal(masks["active"], [True, True, True, True])
        np.testing.assert_array_equal(masks["transition"], [False, True, False, True])

    def test_centre_prior_uses_position_only(self):
        y_train = np.zeros((10, 64, 64), dtype=np.int8)
        y_train[:, 31:33, 31:33] = 1          # training fire only at the patch centre
        y_train[0, 0, 0] = -1                 # unknown labels are ignored
        p = centre_prior(y_train, np.array([31 * 64 + 31, 0, 64 * 64 - 1]))
        self.assertGreater(p[0], 0.5)
        self.assertEqual(p[1], 0.0)
        self.assertEqual(p[2], 0.0)

    def test_gather_maps_alignment(self):
        maps = np.arange(2 * 64 * 64, dtype=np.float32).reshape(2, 64, 64)
        patch, pix = np.array([0, 1, 1]), np.array([0, 65, 4095])
        np.testing.assert_array_equal(gather_maps(maps, patch, pix), [maps[0, 0, 0], maps[1, 1, 1], maps[1, 63, 63]])


if __name__ == "__main__":
    unittest.main()
