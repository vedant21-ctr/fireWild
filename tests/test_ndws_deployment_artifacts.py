"""
test_ndws_deployment_artifacts.py - Unit tests for NDWS Deployment Artifact Packaging & Configurable Paths
"""

import os
import json
import hashlib
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.ndws_contracts import (
    DEFAULT_RESULTS_PATH,
    DEPLOYMENT_RESULTS_PATH,
    DEFAULT_METADATA_PATH,
    DEPLOYMENT_METADATA_PATH,
    resolve_results_path,
    resolve_metadata_path,
    load_raw_benchmark_results,
    load_raw_model_metadata,
    get_sanitized_benchmark_payload,
)


class TestNDWSDeploymentArtifacts(unittest.TestCase):

    def setUp(self):
        # Save and restore environment variables for isolation
        self._orig_env = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._orig_env)

    def test_1_deployment_artifacts_exist_and_match_source_identically(self):
        """Verifies that packaged deployment artifacts exist and are byte-for-byte identical to source."""
        self.assertTrue(os.path.exists(DEPLOYMENT_RESULTS_PATH), f"Packaged {DEPLOYMENT_RESULTS_PATH} missing")
        self.assertTrue(os.path.exists(DEPLOYMENT_METADATA_PATH), f"Packaged {DEPLOYMENT_METADATA_PATH} missing")

        # Source artifacts are gitignored and only exist after running the NDWS pipeline locally
        if not (os.path.exists(DEFAULT_RESULTS_PATH) and os.path.exists(DEFAULT_METADATA_PATH)):
            self.skipTest("Source NDWS artifacts not generated locally; byte-identity check requires them")

        # Results JSON
        self.assertTrue(os.path.exists(DEFAULT_RESULTS_PATH), f"Source {DEFAULT_RESULTS_PATH} missing")
        self.assertTrue(os.path.exists(DEPLOYMENT_RESULTS_PATH), f"Packaged {DEPLOYMENT_RESULTS_PATH} missing")

        with open(DEFAULT_RESULTS_PATH, "rb") as f1, open(DEPLOYMENT_RESULTS_PATH, "rb") as f2:
            bytes1 = f1.read()
            bytes2 = f2.read()
            self.assertEqual(bytes1, bytes2, "deployment/ndws/lightgbm_results.json is not byte-identical to source!")
            self.assertEqual(hashlib.sha256(bytes1).hexdigest(), hashlib.sha256(bytes2).hexdigest())

        # Metadata JSON
        self.assertTrue(os.path.exists(DEFAULT_METADATA_PATH), f"Source {DEFAULT_METADATA_PATH} missing")
        self.assertTrue(os.path.exists(DEPLOYMENT_METADATA_PATH), f"Packaged {DEPLOYMENT_METADATA_PATH} missing")

        with open(DEFAULT_METADATA_PATH, "rb") as f1, open(DEPLOYMENT_METADATA_PATH, "rb") as f2:
            bytes1 = f1.read()
            bytes2 = f2.read()
            self.assertEqual(bytes1, bytes2, "deployment/ndws/lightgbm_metadata.json is not byte-identical to source!")
            self.assertEqual(hashlib.sha256(bytes1).hexdigest(), hashlib.sha256(bytes2).hexdigest())

    def test_2_default_configuration_resolves_primary_or_deployment(self):
        """Verifies that default resolution resolves without env vars."""
        os.environ.pop("NDWS_RESULTS_PATH", None)
        os.environ.pop("NDWS_METADATA_PATH", None)

        res_path = resolve_results_path()
        meta_path = resolve_metadata_path()

        self.assertTrue(os.path.exists(res_path))
        self.assertTrue(os.path.exists(meta_path))

        # Local source artifacts take priority; a clean clone falls back to the packaged deployment copies
        expected_res = DEFAULT_RESULTS_PATH if os.path.exists(DEFAULT_RESULTS_PATH) else DEPLOYMENT_RESULTS_PATH
        expected_meta = DEFAULT_METADATA_PATH if os.path.exists(DEFAULT_METADATA_PATH) else DEPLOYMENT_METADATA_PATH
        self.assertTrue(
            res_path == expected_res or res_path.endswith(expected_res.replace("/", os.sep))
        )
        self.assertTrue(
            meta_path == expected_meta or meta_path.endswith(expected_meta.replace("/", os.sep))
        )

    def test_3_env_var_override_loads_deployment_directory(self):
        """Verifies that setting NDWS_RESULTS_PATH and NDWS_METADATA_PATH directs loading to deployment dir."""
        os.environ["NDWS_RESULTS_PATH"] = DEPLOYMENT_RESULTS_PATH
        os.environ["NDWS_METADATA_PATH"] = DEPLOYMENT_METADATA_PATH

        res_path = resolve_results_path()
        meta_path = resolve_metadata_path()

        self.assertEqual(res_path, DEPLOYMENT_RESULTS_PATH)
        self.assertEqual(meta_path, DEPLOYMENT_METADATA_PATH)

        payload = get_sanitized_benchmark_payload()
        self.assertEqual(payload["best_model_details"]["model_id"], "LightGBM_Unweighted")
        self.assertAlmostEqual(
            payload["best_model_details"]["validation_performance"]["average_precision"], 0.114470, places=5
        )
        self.assertAlmostEqual(
            payload["best_model_details"]["held_out_evaluation_performance"]["average_precision"], 0.083720, places=5
        )

    def test_4_fallback_to_deployment_when_local_experiment_dir_missing(self):
        """Simulates Render environment where experiments/ and models/ndws/ are gitignored and absent."""
        os.environ.pop("NDWS_RESULTS_PATH", None)
        os.environ.pop("NDWS_METADATA_PATH", None)

        orig_exists = os.path.exists

        def mock_exists(path):
            norm = path.replace("\\", "/")
            # Simulate local experiment/model paths missing on Render
            if "experiments/results_ndws" in norm or "models/ndws" in norm:
                return False
            # Deployment paths exist
            if "deployment/ndws" in norm:
                return True
            return orig_exists(path)

        with patch("os.path.exists", side_effect=mock_exists):
            resolved_res = resolve_results_path()
            resolved_meta = resolve_metadata_path()

            self.assertTrue("deployment/ndws" in resolved_res.replace("\\", "/"))
            self.assertTrue("deployment/ndws" in resolved_meta.replace("\\", "/"))

    def test_5_missing_artifacts_raise_controlled_filenotfounderror(self):
        """Verifies that if neither local nor deployment nor env var artifacts exist, a descriptive error is raised."""
        os.environ.pop("NDWS_RESULTS_PATH", None)
        os.environ.pop("NDWS_METADATA_PATH", None)

        with patch("os.path.exists", return_value=False):
            with self.assertRaises(FileNotFoundError) as ctx_res:
                resolve_results_path()
            self.assertIn("Benchmark results artifact not found", str(ctx_res.exception))
            self.assertIn("NDWS_RESULTS_PATH", str(ctx_res.exception))

            with self.assertRaises(FileNotFoundError) as ctx_meta:
                resolve_metadata_path()
            self.assertIn("Model metadata artifact not found", str(ctx_meta.exception))
            self.assertIn("NDWS_METADATA_PATH", str(ctx_meta.exception))

    def test_6_invalid_env_var_path_raises_informative_error(self):
        """Verifies that an explicitly set nonexistent env var path raises an informative error."""
        os.environ["NDWS_RESULTS_PATH"] = "nonexistent/custom/path.json"
        with self.assertRaises(FileNotFoundError) as ctx:
            resolve_results_path()
        self.assertIn("NDWS_RESULTS_PATH", str(ctx.exception))
        self.assertIn("nonexistent/custom/path.json", str(ctx.exception))

    def test_7_benchmark_endpoint_serves_payload_via_testclient(self):
        """Verifies GET /api/v1/ndws/benchmark/metrics returns HTTP 200 with deployment files configured."""
        os.environ["NDWS_RESULTS_PATH"] = DEPLOYMENT_RESULTS_PATH
        os.environ["NDWS_METADATA_PATH"] = DEPLOYMENT_METADATA_PATH

        client = TestClient(app)
        response = client.get("/api/v1/ndws/benchmark/metrics")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("models_comparison", data)
        self.assertEqual(data["best_model_details"]["model_id"], "LightGBM_Unweighted")
        self.assertEqual(data["best_model_details"]["optimal_decision_threshold"], 0.028115)
        self.assertEqual(len(data["models_comparison"]), 5)

    def test_8_benchmark_endpoint_returns_503_when_artifact_missing(self):
        """Verifies GET /api/v1/ndws/benchmark/metrics returns HTTP 503 when artifacts cannot be found."""
        client = TestClient(app)
        with patch("src.api.ndws_routes.get_sanitized_benchmark_payload", side_effect=FileNotFoundError("Artifact absent")):
            response = client.get("/api/v1/ndws/benchmark/metrics")
            self.assertEqual(response.status_code, 503)
            data = response.json()
            self.assertEqual(data["error"], "artifact_missing")
            self.assertIn("Artifact absent", data["message"])


if __name__ == "__main__":
    unittest.main()
