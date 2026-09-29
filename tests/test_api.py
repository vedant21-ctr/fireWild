"""
test_api.py - Unit Tests for CLBI REST API Endpoints
"""

import unittest
from fastapi.testclient import TestClient
from src.api.main import app

class TestCLBIAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.valid_segment = {
            "segment_id": "test_seg_001",
            "fire_id": "CA-CZU-005205",
            "fire_name": "CZU Lightning Complex",
            "date": "2020-08-20",
            "slope": 25.0,
            "elevation": 500.0,
            "distance_to_fire": 150.0,
            "barrier_width_m": 5.0,
            "burn_prob": 0.75,
            "attack_angle": 15.0,
            "attack_dot_product": 0.85
        }

    # 1. Health endpoint
    def test_health_endpoint(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "CLBI API")
        self.assertEqual(data["version"], "1.0.0")

    # 2. Model info endpoint
    def test_model_info_endpoint(self):
        response = self.client.get("/api/v1/model/info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("model_type", data)
        self.assertIn("feature_list", data)
        self.assertEqual(len(data["feature_list"]), 7)
        self.assertIn("risk_thresholds", data)
        self.assertIn("evaluation", data)
        self.assertEqual(data["evaluation"]["method"], "5-fold Leave-One-Fire-Out")

    # 3. Valid single prediction
    def test_valid_single_prediction(self):
        response = self.client.post("/api/v1/predict", json=self.valid_segment)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["segment_id"], "test_seg_001")
        self.assertGreaterEqual(data["breach_probability"], 0.0)
        self.assertLessEqual(data["breach_probability"], 1.0)
        self.assertIn(data["risk_tier"], ["LOW", "MEDIUM", "HIGH", "CRITICAL"])
        self.assertIsNone(data["vulnerability_rank"])
        self.assertIsNone(data["priority_percentile"])
        self.assertIsNone(data["priority_recommended"])

    # 4. Invalid burn_prob
    def test_invalid_burn_prob(self):
        invalid_payload = self.valid_segment.copy()
        invalid_payload["burn_prob"] = 1.5
        response = self.client.post("/api/v1/predict", json=invalid_payload)
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertEqual(data["error"], "validation_error")

    # 5. Invalid attack_dot_product
    def test_invalid_attack_dot_product(self):
        invalid_payload = self.valid_segment.copy()
        invalid_payload["attack_dot_product"] = -2.0
        response = self.client.post("/api/v1/predict", json=invalid_payload)
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertEqual(data["error"], "validation_error")

    # 6. Missing feature
    def test_missing_feature(self):
        invalid_payload = self.valid_segment.copy()
        del invalid_payload["slope"]
        response = self.client.post("/api/v1/predict", json=invalid_payload)
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertEqual(data["error"], "validation_error")

    # 7. Batch prediction
    def test_batch_prediction(self):
        seg2 = self.valid_segment.copy()
        seg2["segment_id"] = "test_seg_002"
        seg2["burn_prob"] = 0.10
        payload = {"segments": [self.valid_segment, seg2]}

        response = self.client.post("/api/v1/predict/batch", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_segments"], 2)
        self.assertEqual(len(data["segments"]), 2)
        self.assertEqual(data["segments"][0]["segment_id"], "test_seg_001")
        self.assertEqual(data["segments"][1]["segment_id"], "test_seg_002")
        self.assertIsNotNone(data["segments"][0]["vulnerability_rank"])
        self.assertIsNotNone(data["segments"][1]["vulnerability_rank"])

    # 8. Empty batch
    def test_empty_batch(self):
        payload = {"segments": []}
        response = self.client.post("/api/v1/predict/batch", json=payload)
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertEqual(data["error"], "validation_error")

    # 9. Prioritization by percentage
    def test_prioritization_by_percentage(self):
        segments = []
        for i in range(10):
            seg = self.valid_segment.copy()
            seg["segment_id"] = f"seg_{i:03d}"
            seg["burn_prob"] = 0.1 * (i + 1)
            segments.append(seg)

        payload = {
            "segments": segments,
            "selection": {"type": "percentage", "value": 20.0}
        }
        response = self.client.post("/api/v1/prioritize", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_segments"], 10)
        self.assertEqual(data["selected_count"], 2)
        self.assertEqual(data["selection_type"], "percentage")
        self.assertEqual(len(data["segments"]), 2)
        # Highest risk first
        self.assertGreaterEqual(
            data["segments"][0]["breach_probability"],
            data["segments"][1]["breach_probability"]
        )

    # 10. Prioritization by count
    def test_prioritization_by_count(self):
        segments = []
        for i in range(5):
            seg = self.valid_segment.copy()
            seg["segment_id"] = f"seg_{i:03d}"
            seg["burn_prob"] = 0.2 * (i + 1)
            segments.append(seg)

        payload = {
            "segments": segments,
            "selection": {"type": "count", "value": 3.0}
        }
        response = self.client.post("/api/v1/prioritize", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["selected_count"], 3)
        self.assertEqual(data["selection_type"], "count")
        self.assertEqual(len(data["segments"]), 3)

    # 11. Invalid percentage
    def test_invalid_percentage(self):
        payload = {
            "segments": [self.valid_segment],
            "selection": {"type": "percentage", "value": 150.0}
        }
        response = self.client.post("/api/v1/prioritize", json=payload)
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertEqual(data["error"], "validation_error")

    # 12. Invalid count
    def test_invalid_count(self):
        payload = {
            "segments": [self.valid_segment],
            "selection": {"type": "count", "value": 0}
        }
        response = self.client.post("/api/v1/prioritize", json=payload)
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertEqual(data["error"], "validation_error")

    # 13. Demo endpoint
    def test_demo_endpoint(self):
        response = self.client.get("/api/v1/demo")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("metadata", data)
        self.assertIn("summary", data)
        self.assertIn("prioritization", data)
        self.assertIn("segments", data)
        self.assertEqual(data["summary"]["total_segments"], 100)
        self.assertEqual(len(data["segments"]), 100)

    # 14. Response schema
    def test_response_schema(self):
        response = self.client.post("/api/v1/predict", json=self.valid_segment)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        required_keys = [
            "segment_id", "breach_probability", "hold_probability",
            "risk_tier", "vulnerability_rank", "priority_percentile",
            "priority_recommended", "top_risk_factors", "protective_factors", "explanation"
        ]
        for key in required_keys:
            self.assertIn(key, data)

    # 15. Deterministic results
    def test_deterministic_results(self):
        resp1 = self.client.post("/api/v1/predict", json=self.valid_segment).json()
        resp2 = self.client.post("/api/v1/predict", json=self.valid_segment).json()
        self.assertEqual(resp1["breach_probability"], resp2["breach_probability"])
        self.assertEqual(resp1["risk_tier"], resp2["risk_tier"])
        self.assertEqual(resp1["explanation"], resp2["explanation"])


if __name__ == "__main__":
    unittest.main()
