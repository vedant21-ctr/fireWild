"""
test_lightgbm_ndws.py - Unit tests for LightGBM NDWS baseline pipeline
"""

import os
import tempfile
import unittest
import numpy as np
import joblib
import lightgbm as lgb
from src.models.train_tabular_baselines import (
    FEATURE_COLUMNS,
    FORBIDDEN_COLUMNS,
    TARGET_COLUMN,
    DistanceHeuristicClassifier,
    find_optimal_threshold,
    evaluate_model_predictions
)
from src.models.train_lightgbm_ndws import train_lightgbm_model


class TestLightGBMNDWS(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(101)
        self.n_train = 2000
        self.n_val = 1000
        self.n_features = len(FEATURE_COLUMNS)

        self.X_train = rng.normal(10.0, 5.0, (self.n_train, self.n_features)).astype(np.float32)
        self.X_train[:, 11] = rng.uniform(0.0, 50.0, self.n_train).astype(np.float32)

        self.X_val = rng.normal(10.0, 5.0, (self.n_val, self.n_features)).astype(np.float32)
        self.X_val[:, 11] = rng.uniform(0.0, 50.0, self.n_val).astype(np.float32)

        # 1% positive class prevalence
        self.y_train = np.zeros(self.n_train, dtype=np.int32)
        self.y_train[rng.choice(self.n_train, size=20, replace=False)] = 1

        self.y_val = np.zeros(self.n_val, dtype=np.int32)
        self.y_val[rng.choice(self.n_val, size=10, replace=False)] = 1

    def test_1_target_leakage_safeguard(self):
        """Verifies forbidden columns are absent from LightGBM predictor channels."""
        for forbidden in FORBIDDEN_COLUMNS:
            self.assertNotIn(forbidden, FEATURE_COLUMNS, f"Forbidden column '{forbidden}' found in FEATURE_COLUMNS")

    def test_2_valid_predicted_probabilities(self):
        """Verifies that LightGBM produces probabilities strictly within [0.0, 1.0]."""
        clf, _ = train_lightgbm_model(
            self.X_train, self.y_train, self.X_val, self.y_val,
            feature_names=FEATURE_COLUMNS,
            n_estimators=10,
            learning_rate=0.1,
            num_leaves=15,
            random_state=42
        )
        probs = clf.predict_proba(self.X_val)[:, 1]
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))
        self.assertEqual(len(probs), self.n_val)

    def test_3_reproducibility(self):
        """Verifies identical seed produces identical model predictions."""
        clf1, _ = train_lightgbm_model(
            self.X_train, self.y_train, self.X_val, self.y_val,
            feature_names=FEATURE_COLUMNS,
            n_estimators=15,
            learning_rate=0.1,
            random_state=77
        )
        clf2, _ = train_lightgbm_model(
            self.X_train, self.y_train, self.X_val, self.y_val,
            feature_names=FEATURE_COLUMNS,
            n_estimators=15,
            learning_rate=0.1,
            random_state=77
        )
        p1 = clf1.predict_proba(self.X_val)
        p2 = clf2.predict_proba(self.X_val)
        np.testing.assert_array_equal(p1, p2)

    def test_4_correct_metric_calculations(self):
        """Verifies evaluation metrics match manual arithmetic."""
        y_true = np.array([1, 1, 0, 0], dtype=np.int32)
        y_prob = np.array([0.9, 0.4, 0.1, 0.8], dtype=np.float32)
        m = evaluate_model_predictions(y_true, y_prob, threshold=0.5)
        self.assertEqual(m["confusion_matrix"]["tp"], 1)
        self.assertEqual(m["confusion_matrix"]["fn"], 1)
        self.assertEqual(m["confusion_matrix"]["tn"], 1)
        self.assertEqual(m["confusion_matrix"]["fp"], 1)
        self.assertEqual(m["precision"], 0.5)
        self.assertEqual(m["recall"], 0.5)
        self.assertEqual(m["f1_score"], 0.5)

    def test_5_validation_only_threshold_selection(self):
        """Verifies threshold selection operates on provided validation arrays and stays in bounds."""
        y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
        y_prob = np.array([0.05, 0.08, 0.12, 0.7, 0.8, 0.9], dtype=np.float32)
        thresh, f1 = find_optimal_threshold(y_true, y_prob)
        self.assertGreater(thresh, 0.1)
        self.assertLess(thresh, 0.7)
        self.assertEqual(f1, 1.0)

    def test_6_model_serialization_and_reload(self):
        """Verifies LightGBM model bundle serialization and reloading via joblib."""
        clf, _ = train_lightgbm_model(
            self.X_train, self.y_train, self.X_val, self.y_val,
            feature_names=FEATURE_COLUMNS,
            n_estimators=10,
            random_state=42
        )
        probs_orig = clf.predict_proba(self.X_val)

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = os.path.join(tmp_dir, "test_lgb.joblib")
            bundle = {
                "model": clf,
                "feature_names": FEATURE_COLUMNS,
                "optimal_threshold": 0.05
            }
            joblib.dump(bundle, bundle_path)

            loaded = joblib.load(bundle_path)
            loaded_clf = loaded["model"]
            probs_reloaded = loaded_clf.predict_proba(self.X_val)

            np.testing.assert_array_equal(probs_orig, probs_reloaded)
            self.assertEqual(loaded["feature_names"], FEATURE_COLUMNS)

    def test_7_raw_distance_heuristic_integrity(self):
        """Verifies that the distance heuristic on raw km produces positive decaying probabilities."""
        dist_clf = DistanceHeuristicClassifier(dist_col_idx=11, scale_km=2.0)
        dist_clf.fit(self.X_train, self.y_train)
        probs = dist_clf.predict_proba(self.X_val)[:, 1]

        # Valid range
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))
        # Ensure non-trivial distribution
        self.assertGreater(np.max(probs), np.min(probs))


if __name__ == "__main__":
    unittest.main()
