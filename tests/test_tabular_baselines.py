"""
test_tabular_baselines.py - Unit tests for NDWS Tabular Baseline Module
"""

import os
import tempfile
import unittest
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from src.models.train_tabular_baselines import (
    FEATURE_COLUMNS,
    FORBIDDEN_COLUMNS,
    DummyPriorClassifier,
    DistanceHeuristicClassifier,
    fit_preprocessor,
    find_optimal_threshold,
    evaluate_model_predictions,
    load_tabular_data,
    calculate_recall_at_fixed_fpr
)


class TestNDWSTabularBaselines(unittest.TestCase):

    def setUp(self):
        # Create synthetic feature matrix X and target y for 1,000 samples
        rng = np.random.default_rng(42)
        self.n_samples = 1000
        self.n_features = len(FEATURE_COLUMNS)
        self.X = rng.normal(loc=10.0, scale=5.0, size=(self.n_samples, self.n_features)).astype(np.float32)
        # Distance feature (col 11): between 0.0 and 50.0 km
        self.X[:, 11] = rng.uniform(0.0, 50.0, size=self.n_samples).astype(np.float32)
        
        # 1% positive class prevalence
        self.y = np.zeros(self.n_samples, dtype=np.int32)
        pos_idx = rng.choice(self.n_samples, size=10, replace=False)
        self.y[pos_idx] = 1

    def test_1_strict_target_leakage_safeguard(self):
        """Verifies forbidden columns are strictly absent from predictor feature set."""
        for forbidden in FORBIDDEN_COLUMNS:
            self.assertNotIn(forbidden, FEATURE_COLUMNS, f"Forbidden column '{forbidden}' found in FEATURE_COLUMNS")

    def test_2_preprocessor_fitting_isolation(self):
        """Proves StandardScaler fits exclusively on training features and does not leak test distributions."""
        X_train = self.X[:700]
        X_test = self.X[700:] + 100.0  # shifted distribution
        
        scaler = fit_preprocessor(X_train)
        mean_before = scaler.mean_.copy()
        scale_before = scaler.scale_.copy()
        
        # Transform test data
        _ = scaler.transform(X_test)
        
        # Scaler parameters must remain 100% unchanged
        np.testing.assert_array_equal(scaler.mean_, mean_before, err_msg="Scaler mean changed during transform")
        np.testing.assert_array_equal(scaler.scale_, scale_before, err_msg="Scaler scale changed during transform")

    def test_3_dummy_prior_classifier_predicts_exact_prevalence(self):
        """Verifies DummyPriorClassifier outputs probabilities exactly equal to training positive prevalence."""
        dummy = DummyPriorClassifier()
        dummy.fit(self.X, self.y)
        expected_prior = np.mean(self.y)
        self.assertAlmostEqual(dummy.prior_, expected_prior, places=5)
        
        probs = dummy.predict_proba(self.X)
        self.assertEqual(probs.shape, (self.n_samples, 2))
        np.testing.assert_allclose(probs[:, 1], expected_prior)
        np.testing.assert_allclose(probs[:, 0], 1.0 - expected_prior)

    def test_4_distance_heuristic_monotonic_decay(self):
        """Verifies DistanceHeuristicClassifier probability decreases as distance increases."""
        clf = DistanceHeuristicClassifier(dist_col_idx=11, scale_km=2.0)
        clf.fit(self.X, self.y)
        
        # Test 3 distances: 0 km, 5 km, 20 km
        X_test = np.zeros((3, self.n_features), dtype=np.float32)
        X_test[0, 11] = 0.0
        X_test[1, 11] = 5.0
        X_test[2, 11] = 20.0
        
        probs = clf.predict_proba(X_test)[:, 1]
        self.assertGreater(probs[0], probs[1], "Distance 0 km must have higher probability than 5 km")
        self.assertGreater(probs[1], probs[2], "Distance 5 km must have higher probability than 20 km")

    def test_5_distance_heuristic_sentinel_handling(self):
        """Verifies sentinel distance (>= 90 km) receives floor background rate."""
        clf = DistanceHeuristicClassifier(dist_col_idx=11, scale_km=2.0)
        clf.fit(self.X, self.y)
        
        X_sentinel = np.zeros((1, self.n_features), dtype=np.float32)
        X_sentinel[0, 11] = 100.0  # sentinel
        prob = clf.predict_proba(X_sentinel)[0, 1]
        self.assertAlmostEqual(prob, clf.background_rate_, places=5)

    def test_6_find_optimal_threshold(self):
        """Verifies threshold optimization yields valid probability threshold in (0, 1)."""
        rng = np.random.default_rng(123)
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 1], dtype=np.int32)
        y_prob = np.array([0.05, 0.1, 0.15, 0.2, 0.8, 0.85, 0.9, 0.3, 0.25, 0.75], dtype=np.float32)
        
        best_thresh, best_f1 = find_optimal_threshold(y_true, y_prob)
        self.assertGreater(best_thresh, 0.0)
        self.assertLess(best_thresh, 1.0)
        self.assertEqual(best_f1, 1.0, "With perfectly separable test probs, F1 must be 1.0")

    def test_7_evaluate_model_predictions_metrics(self):
        """Verifies metrics structure, confusion matrix math, and precision/recall arithmetic."""
        y_true = np.array([1, 1, 0, 0], dtype=np.int32)
        y_prob = np.array([0.9, 0.4, 0.1, 0.8], dtype=np.float32)
        # Threshold = 0.5:
        # y_pred = [1, 0, 0, 1]
        # TP = 1 (idx 0), FN = 1 (idx 1), TN = 1 (idx 2), FP = 1 (idx 3)
        metrics = evaluate_model_predictions(y_true, y_prob, threshold=0.5)
        
        cm = metrics["confusion_matrix"]
        self.assertEqual(cm["tp"], 1)
        self.assertEqual(cm["fn"], 1)
        self.assertEqual(cm["tn"], 1)
        self.assertEqual(cm["fp"], 1)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1_score"], 0.5)
        self.assertEqual(metrics["total_eval_samples"], 4)
        self.assertEqual(metrics["positive_samples"], 2)

    def test_8_pipeline_serialization_and_reload(self):
        """Verifies that scaler + model pipeline serializes and reloads deterministically."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        dummy = DummyPriorClassifier()
        dummy.fit(X_scaled, self.y)
        preds_original = dummy.predict_proba(X_scaled)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = os.path.join(tmp_dir, "test_pipeline.joblib")
            bundle = {
                "scaler": scaler,
                "model": dummy,
                "feature_names": FEATURE_COLUMNS,
                "threshold": 0.05
            }
            joblib.dump(bundle, bundle_path)
            
            # Reload
            loaded_bundle = joblib.load(bundle_path)
            loaded_scaler = loaded_bundle["scaler"]
            loaded_model = loaded_bundle["model"]
            
            X_reloaded_scaled = loaded_scaler.transform(self.X)
            preds_reloaded = loaded_model.predict_proba(X_reloaded_scaled)
            
            np.testing.assert_array_equal(preds_original, preds_reloaded)
            self.assertEqual(loaded_bundle["feature_names"], FEATURE_COLUMNS)
            self.assertEqual(loaded_bundle["threshold"], 0.05)

    def test_9_distance_heuristic_independent_of_standardization(self):
        """
        Regression test proving that DistanceHeuristicClassifier yields identical
        predictions regardless of whether features are raw or standardized.
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        # 1. Unscaled model directly on raw features
        clf_raw = DistanceHeuristicClassifier(dist_col_idx=11, scale_km=2.0)
        clf_raw.fit(self.X, self.y)
        probs_raw = clf_raw.predict_proba(self.X)
        
        # 2. Scaled model with unscaling scaler provided
        clf_scaled = DistanceHeuristicClassifier(dist_col_idx=11, scale_km=2.0, scaler=scaler)
        clf_scaled.fit(X_scaled, self.y)
        probs_scaled = clf_scaled.predict_proba(X_scaled)
        
        # Proves exact independence from standardization
        np.testing.assert_allclose(
            probs_raw,
            probs_scaled,
            rtol=1e-4,
            atol=1e-5,
            err_msg="Distance heuristic predictions must be identical on raw vs standardized matrices"
        )

    def test_10_recall_at_fixed_fpr_perfect_predictions(self):
        """Verifies that perfect predictions achieve 100% recall at FPR <= 1% and 5%."""
        y_true = np.array([0] * 900 + [1] * 100, dtype=np.int32)
        y_prob = np.array([0.05] * 900 + [0.95] * 100, dtype=np.float32)

        res_1 = calculate_recall_at_fixed_fpr(y_true, y_prob, target_fpr=0.01)
        res_5 = calculate_recall_at_fixed_fpr(y_true, y_prob, target_fpr=0.05)

        self.assertEqual(res_1["recall"], 1.0)
        self.assertLessEqual(res_1["achieved_fpr"], 0.01)
        self.assertEqual(res_5["recall"], 1.0)
        self.assertLessEqual(res_5["achieved_fpr"], 0.05)

    def test_11_recall_at_fixed_fpr_imbalanced_labels(self):
        """Verifies recall computation on severely imbalanced test data (e.g. 1% prevalence)."""
        rng = np.random.default_rng(42)
        y_true = np.zeros(2000, dtype=np.int32)
        y_true[rng.choice(2000, size=20, replace=False)] = 1  # 1% positive

        # Positives have higher mean score
        y_prob = np.zeros(2000, dtype=np.float32)
        y_prob[y_true == 0] = rng.uniform(0.0, 0.4, size=1980)
        y_prob[y_true == 1] = rng.uniform(0.2, 0.9, size=20)

        res = calculate_recall_at_fixed_fpr(y_true, y_prob, target_fpr=0.01)
        self.assertLessEqual(res["achieved_fpr"], 0.01)
        self.assertGreaterEqual(res["recall"], 0.0)
        self.assertLessEqual(res["recall"], 1.0)
        self.assertTrue(res["valid"])

    def test_12_recall_at_fixed_fpr_tied_scores(self):
        """Verifies that tied prediction scores never cause empirical FPR to exceed target."""
        # 100 negatives all with score 0.8; 10 positives with score 0.8
        # If threshold is 0.8, FPR is 1.0 (100%), which exceeds 0.01.
        # Function must refuse to select 0.8 and respect achieved_fpr <= target_fpr.
        y_true = np.array([0] * 100 + [1] * 10, dtype=np.int32)
        y_prob = np.full(110, 0.8, dtype=np.float32)

        res = calculate_recall_at_fixed_fpr(y_true, y_prob, target_fpr=0.01)
        self.assertLessEqual(res["achieved_fpr"], 0.01)
        self.assertEqual(res["achieved_fpr"], 0.0)

    def test_13_recall_at_fixed_fpr_edge_cases(self):
        """Verifies handling of no-positive and no-negative edge cases without crashing."""
        y_prob = np.array([0.2, 0.5, 0.8], dtype=np.float32)

        # No positives
        y_no_pos = np.zeros(3, dtype=np.int32)
        res_no_pos = calculate_recall_at_fixed_fpr(y_no_pos, y_prob, target_fpr=0.05)
        self.assertEqual(res_no_pos["recall"], 0.0)
        self.assertFalse(res_no_pos["valid"])

        # No negatives
        y_no_neg = np.ones(3, dtype=np.int32)
        res_no_neg = calculate_recall_at_fixed_fpr(y_no_neg, y_prob, target_fpr=0.05)
        self.assertEqual(res_no_neg["recall"], 1.0)
        self.assertFalse(res_no_neg["valid"])

    def test_14_recall_at_fixed_fpr_achieved_fpr_strictly_bounded(self):
        """Verifies that achieved FPR is strictly <= target across 50 random simulations."""
        rng = np.random.default_rng(999)
        for _ in range(50):
            n = 500
            y_t = (rng.uniform(0, 1, n) > 0.9).astype(np.int32)
            y_p = rng.uniform(0, 1, n).astype(np.float32)
            if np.sum(y_t == 1) == 0 or np.sum(y_t == 0) == 0:
                continue

            for target in [0.01, 0.05, 0.10]:
                out = calculate_recall_at_fixed_fpr(y_t, y_p, target_fpr=target)
                self.assertLessEqual(out["achieved_fpr"], target + 1e-6)


if __name__ == "__main__":
    unittest.main()
