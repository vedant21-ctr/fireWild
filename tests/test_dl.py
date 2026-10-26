"""
test_dl.py - Unit tests for the spatial U-Net pipeline (src/dl).

All tests use synthetic tensors and run without the NDWS data or a GPU.
"""

import itertools
import unittest
import numpy as np
import torch

from src.aml.ndws import CH, CHANNELS
from src.dl.data import DL_CHANNELS, patch_inputs, compute_norm, normalise, apply_dihedral, random_crop, U, V
from src.dl.model import UNet, masked_bce
from src.dl.evaluate import target_masks, gather_maps, centre_prior

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
