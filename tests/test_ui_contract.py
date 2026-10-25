"""
test_ui_contract.py - Unit tests for NDWS UI Integration Contract & Payload Formatter
"""

import unittest
import numpy as np
from src.api.ndws_contracts import (
    get_sanitized_benchmark_payload,
    generate_illustrative_spatial_demo_patch
)


class TestNDWSUIContract(unittest.TestCase):

    def test_1_payload_generation_and_schema(self):
        """Verifies that public payload generates with all required top-level sections."""
        payload = get_sanitized_benchmark_payload()
        self.assertIn("benchmark_summary", payload)
        self.assertIn("models_comparison", payload)
        self.assertIn("best_model_details", payload)
        self.assertIn("ui_display_disclaimers", payload)

        models = payload["models_comparison"]
        self.assertEqual(len(models), 5)
        # Verify sorted by Average Precision descending
        aps = [m["average_precision"] for m in models]
        self.assertEqual(aps, sorted(aps, reverse=True))

    def test_2_no_private_file_paths_leaked(self):
        """Verifies that no private local machine file paths exist anywhere in the public payload."""
        payload = get_sanitized_benchmark_payload()
        payload_str = str(payload)

        # Forbidden local paths
        forbidden_patterns = [
            "C:\\Users",
            "C:/Users",
            "Admin",
            ".joblib",
            ".parquet",
            ".tfrecord"
        ]
        for pattern in forbidden_patterns:
            self.assertNotIn(pattern, payload_str, f"Private pattern '{pattern}' leaked into UI payload!")

    def test_3_metrics_match_actual_artifacts(self):
        """Verifies that payload numbers match actual verified benchmark results."""
        payload = get_sanitized_benchmark_payload()
        best_details = payload["best_model_details"]

        # Best model is LightGBM_Unweighted
        self.assertEqual(best_details["model_id"], "LightGBM_Unweighted")
        val_perf = best_details["validation_performance"]
        self.assertAlmostEqual(val_perf["average_precision"], 0.114470, places=5)
        self.assertAlmostEqual(val_perf["f1_score"], 0.223610, places=5)
        self.assertAlmostEqual(best_details["optimal_decision_threshold"], 0.028115, places=5)

        eval_perf = best_details["held_out_evaluation_performance"]
        self.assertAlmostEqual(eval_perf["average_precision"], 0.083720, places=5)
        self.assertAlmostEqual(eval_perf["f1_score"], 0.173521, places=5)
        self.assertEqual(eval_perf["confusion_matrix"]["fp"], 3323)

    def test_4_illustrative_spatial_demo_patch(self):
        """Verifies that spatial demo data is explicitly flagged as illustrative and valid 64x64."""
        patch = generate_illustrative_spatial_demo_patch()
        self.assertFalse(patch["is_georeferenced"])
        self.assertTrue(patch["is_illustrative_simulation"])
        self.assertIn("disclaimer", patch)

        grid = np.array(patch["probabilities_2d"])
        self.assertEqual(grid.shape, (64, 64))
        self.assertTrue(np.all(grid >= 0.0))
        self.assertTrue(np.all(grid <= 1.0))

    def test_5_fastapi_ndws_benchmark_endpoint(self):
        """Verifies that GET /api/v1/ndws/benchmark/metrics returns HTTP 200 with valid schema."""
        from fastapi.testclient import TestClient
        from src.api.main import app

        client = TestClient(app)
        response = client.get("/api/v1/ndws/benchmark/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("models_comparison", data)
        self.assertIn("best_model_details", data)
        self.assertEqual(data["best_model_details"]["model_id"], "LightGBM_Unweighted")
        self.assertNotIn("C:\\Users", response.text)

    def test_6_fastapi_ndws_demo_patch_endpoint(self):
        """Verifies that GET /api/v1/ndws/demo/patch returns HTTP 200 with 64x64 grid."""
        from fastapi.testclient import TestClient
        from src.api.main import app

        client = TestClient(app)
        response = client.get("/api/v1/ndws/demo/patch")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["is_illustrative_simulation"])
        self.assertEqual(data["grid_dimensions"]["rows"], 64)
        self.assertEqual(data["grid_dimensions"]["cols"], 64)
        self.assertEqual(len(data["probabilities_2d"]), 64)
        self.assertEqual(len(data["probabilities_2d"][0]), 64)

    def test_7_frontend_dashboard_routes(self):
        """Verifies that GET / and GET /demo serve the HTML dashboard with HTTP 200."""
        from fastapi.testclient import TestClient
        from src.api.main import app

        client = TestClient(app)
        for path in ("/", "/demo"):
            response = client.get(path)
            self.assertEqual(response.status_code, 200, f"Failed on path {path}")
            self.assertIn("text/html", response.headers["content-type"])
            self.assertIn("FireWild Intelligence", response.text)
            self.assertIn("spatial-canvas", response.text)
            self.assertIn("clbi-segments-tbody", response.text)

    def test_8_static_assets_serving(self):
        """Verifies that static assets (styles.css and app.js) are accessible."""
        from fastapi.testclient import TestClient
        from src.api.main import app

        client = TestClient(app)
        res_css = client.get("/static/styles.css")
        self.assertEqual(res_css.status_code, 200)
        self.assertIn("var(--accent-orange)", res_css.text)

        res_js = client.get("/static/app.js")
        self.assertEqual(res_js.status_code, 200)
        self.assertIn("loadNDWSBenchmarkMetrics", res_js.text)


if __name__ == "__main__":
    unittest.main()
