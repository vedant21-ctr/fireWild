"""
test_clbi_model.py - Suite of unit tests for CLBIDecisionEngine and production pipeline
"""

import os
import json
import unittest
import numpy as np
import pandas as pd
from src.model.clbi_model import CLBIDecisionEngine, CLEAN_FEATURES
from src.model.clbi_demo import create_demo_dataset, run_demo

class TestCLBIProductionEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.model_path = os.path.join(base_dir, "models", "clbi_logistic_regression.joblib")
        
        # Ensure production model artifact exists
        if not os.path.exists(cls.model_path):
            from src.model.train_clbi_model import train_production_model
            train_production_model()
            
        cls.engine = CLBIDecisionEngine(model_path=cls.model_path)
        cls.demo_df = create_demo_dataset()
        cls.preds_df = cls.engine.predict_df(cls.demo_df, top_k_recommend_pct=10.0)

    def test_1_saved_model_loads(self):
        self.assertIsNotNone(self.engine.pipeline)
        self.assertTrue(self.engine.is_fitted if hasattr(self.engine, 'is_fitted') else True)

    def test_2_missing_model_artifact_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            CLBIDecisionEngine(model_path="models/non_existent_model.joblib")

    def test_3_deterministic_predictions(self):
        preds_1 = self.engine.predict_df(self.demo_df)
        preds_2 = self.engine.predict_df(self.demo_df)
        np.testing.assert_array_equal(preds_1["breach_probability"].values, preds_2["breach_probability"].values)

    def test_4_same_input_gives_same_output(self):
        row_0 = self.demo_df.iloc[:1].copy()
        res_1 = self.engine.predict_df(row_0)
        res_2 = self.engine.predict_df(row_0)
        self.assertEqual(res_1.iloc[0]["breach_probability"], res_2.iloc[0]["breach_probability"])

    def test_5_probabilities_in_bounds(self):
        p_breach = self.preds_df["breach_probability"].values
        self.assertTrue(np.all(p_breach >= 0.0) and np.all(p_breach <= 1.0))

    def test_6_risk_boundaries(self):
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

    def test_7_ranking_uniqueness_and_order(self):
        ranks = self.preds_df["vulnerability_rank"].values
        self.assertEqual(len(set(ranks)), len(ranks))
        sorted_p = self.preds_df.sort_values(by="vulnerability_rank")["breach_probability"].values
        self.assertTrue(np.all(sorted_p[:-1] >= sorted_p[1:]))

    def test_8_missing_feature_raises_error(self):
        incomplete_df = self.demo_df.drop(columns=["slope"])
        with self.assertRaises(ValueError):
            self.engine.predict_df(incomplete_df)

    def test_9_nan_validation_raises_error(self):
        nan_df = self.demo_df.copy()
        nan_df.loc[0, "slope"] = np.nan
        with self.assertRaises(ValueError):
            self.engine.predict_df(nan_df)

    def test_10_infinite_validation_raises_error(self):
        inf_df = self.demo_df.copy()
        inf_df.loc[0, "burn_prob"] = np.inf
        with self.assertRaises(ValueError):
            self.engine.predict_df(inf_df)

    def test_11_demo_contains_exactly_100_segments(self):
        self.assertEqual(len(self.demo_df), 100)

    def test_12_demo_json_contains_100_segment_objects(self):
        run_demo()
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        json_path = os.path.join(base_dir, "experiments", "results", "clbi_demo_response.json")
        self.assertTrue(os.path.exists(json_path))
        with open(json_path, "r") as f:
            data = json.load(f)
        self.assertEqual(len(data["segments"]), 100)

    def test_13_demo_csv_and_json_agree_on_count(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, "experiments", "results", "clbi_demo_predictions.csv")
        json_path = os.path.join(base_dir, "experiments", "results", "clbi_demo_response.json")
        csv_df = pd.read_csv(csv_path)
        with open(json_path, "r") as f:
            j_data = json.load(f)
        self.assertEqual(len(csv_df), 100)
        self.assertEqual(len(j_data["segments"]), 100)

    def test_14_prioritization_selects_highest_risk(self):
        prio = self.engine.prioritize_candidate_line(self.preds_df, resource_count=10)
        selected_df = prio["selected_segments"]
        self.assertEqual(len(selected_df), 10)
        self.assertEqual(selected_df["vulnerability_rank"].min(), 1)
        self.assertEqual(selected_df["vulnerability_rank"].max(), 10)

if __name__ == "__main__":
    unittest.main()
