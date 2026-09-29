"""
test_clbi_model.py - Unit tests for CLBIDecisionEngine
"""

import unittest
import pandas as pd
import numpy as np
from src.model.clbi_model import CLBIDecisionEngine, CLEAN_FEATURES

class TestCLBIDecisionEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create synthetic training data
        np.random.seed(42)
        n = 100
        cls.train_df = pd.DataFrame({
            "fire_id": ["CA-TEST-000001"] * n,
            "segment_id": [f"seg_{i:03d}" for i in range(n)],
            "slope": np.random.uniform(5, 45, n),
            "elevation": np.random.uniform(100, 2000, n),
            "distance_to_fire": np.random.uniform(100, 3000, n),
            "barrier_width_m": np.random.choice([2.0, 6.5, 9.0], n),
            "burn_prob": np.random.uniform(0.1, 0.9, n),
            "attack_angle": np.random.uniform(0, 90, n),
            "attack_dot_product": np.random.uniform(0.1, 0.95, n),
            "label": np.random.choice([0, 1], n)
        })
        
        cls.engine = CLBIDecisionEngine(random_state=42)
        cls.engine.fit(cls.train_df)
        cls.preds_df = cls.engine.predict_df(cls.train_df, top_k_recommend_pct=10.0)

    def test_1_probability_bounds(self):
        p_breach = self.preds_df["breach_probability"].values
        self.assertTrue(np.all(p_breach >= 0.0) and np.all(p_breach <= 1.0))

    def test_2_probability_sum(self):
        p_breach = self.preds_df["breach_probability"].values
        p_hold = self.preds_df["hold_probability"].values
        np.testing.assert_allclose(p_breach + p_hold, 1.0, atol=1e-3)

    def test_3_risk_thresholds(self):
        for _, row in self.preds_df.iterrows():
            p = row["breach_probability"]
            tier = row["risk_tier"]
            if p < 0.30:
                self.assertEqual(tier, "LOW")
            elif p < 0.60:
                self.assertEqual(tier, "MEDIUM")
            elif p < 0.80:
                self.assertEqual(tier, "HIGH")
            else:
                self.assertEqual(tier, "CRITICAL")

    def test_4_ranking_order(self):
        sorted_p = self.preds_df.sort_values(by="vulnerability_rank")["breach_probability"].values
        self.assertTrue(np.all(sorted_p[:-1] >= sorted_p[1:]))

    def test_5_ranking_deterministic(self):
        preds_2 = self.engine.predict_df(self.train_df, top_k_recommend_pct=10.0)
        np.testing.assert_array_equal(
            self.preds_df["vulnerability_rank"].values,
            preds_2["vulnerability_rank"].values
        )

    def test_6_missing_feature_raises_error(self):
        incomplete_df = self.train_df.drop(columns=["slope"])
        with self.assertRaises(ValueError):
            self.engine.predict_df(incomplete_df)

    def test_7_invalid_values_handled(self):
        invalid_df = self.train_df.copy()
        invalid_df.loc[0, "slope"] = 999.0 # Extreme value
        pred_invalid = self.engine.predict_df(invalid_df)
        self.assertFalse(np.isnan(pred_invalid.loc[0, "breach_probability"]))

    def test_8_output_schema_correct(self):
        required_schema = [
            "segment_id", "fire_id", "breach_probability", "hold_probability",
            "risk_tier", "vulnerability_rank", "priority_percentile",
            "priority_recommended", "top_risk_factors", "protective_factors", "explanation"
        ]
        for col in required_schema:
            self.assertIn(col, self.preds_df.columns)

    def test_9_prioritization_selects_highest_risk(self):
        prio = self.engine.prioritize_candidate_line(self.preds_df, resource_count=10)
        selected_df = prio["selected_segments"]
        self.assertEqual(len(selected_df), 10)
        self.assertEqual(selected_df["vulnerability_rank"].min(), 1)
        self.assertEqual(selected_df["vulnerability_rank"].max(), 10)

    def test_10_explanations_generated(self):
        for exp in self.preds_df["explanation"]:
            self.assertIsInstance(exp, str)
            self.assertTrue(len(exp) > 10)

if __name__ == "__main__":
    unittest.main()
