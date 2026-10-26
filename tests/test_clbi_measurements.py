"""
test_clbi_measurements.py - Regression tests for missing / invalid fireline measurements
in the CLBI Risk Simulator segment audit table.

Covers the API helper that maps invalid values to null, the /api/v1/demo payload, and the
browser formatting helpers in src/api/static/format.js (run under Node when it is installed).
"""

import json
import math
import os
import shutil
import subprocess
import unittest

import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes import optional_measurement, DEMO_MEASUREMENT_BOUNDS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORMAT_JS = os.path.join(BASE_DIR, "src", "api", "static", "format.js")
TABLE_MEASUREMENTS = ("barrier_width_m", "distance_to_fire", "slope")


class TestOptionalMeasurement(unittest.TestCase):
    def test_missing_and_invalid_values_become_none(self):
        for value in (None, float("nan"), float("inf"), float("-inf"), "", "abc", True, False, pd.NA):
            with self.subTest(value=value):
                self.assertIsNone(optional_measurement(value))

    def test_valid_numbers_are_preserved(self):
        self.assertEqual(optional_measurement(6.5), 6.5)
        self.assertEqual(optional_measurement("158.2"), 158.2)
        self.assertEqual(optional_measurement(0, 0.0, True), 0.0)  # level ground is a real slope, not missing

    def test_out_of_range_values_become_none(self):
        self.assertIsNone(optional_measurement(-3.0, 0.0, True))      # negative slope / distance
        self.assertIsNone(optional_measurement(0.0, 0.0, False))      # zero barrier width
        self.assertEqual(optional_measurement(2.0, 0.0, False), 2.0)
        self.assertEqual(optional_measurement(-5.0), -5.0)            # no bound: below-sea-level elevation is valid


class TestDemoPayloadMeasurements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        response = TestClient(app).get("/api/v1/demo")
        cls.status = response.status_code
        cls.segments = response.json()["segments"]
        cls.source = pd.read_csv(os.path.join(BASE_DIR, "data", "demo", "clbi_demo_segments.csv")).set_index("segment_id")

    def test_measurements_present_and_finite(self):
        self.assertEqual(self.status, 200)
        self.assertEqual(len(self.segments), 100)
        for seg in self.segments:
            for name in DEMO_MEASUREMENT_BOUNDS:
                self.assertIn(name, seg)
                self.assertIsInstance(seg[name], float, f"{seg['segment_id']} {name}")
                self.assertTrue(math.isfinite(seg[name]))

    def test_measurements_match_source_rows(self):
        for seg in self.segments:
            row = self.source.loc[seg["segment_id"]]
            for name in TABLE_MEASUREMENTS:
                self.assertAlmostEqual(seg[name], float(row[name]), places=6)

    def test_probabilities_still_valid(self):
        for seg in self.segments:
            self.assertTrue(0.0 <= seg["breach_probability"] <= 1.0)
            self.assertAlmostEqual(seg["breach_probability"] + seg["hold_probability"], 1.0, places=3)

    def test_existing_fields_unchanged(self):
        expected = {"segment_id", "fire_id", "fire_name", "date", "breach_probability", "hold_probability",
                    "risk_tier", "vulnerability_rank", "priority_percentile", "priority_recommended",
                    "top_risk_factors", "protective_factors", "explanation"}
        self.assertTrue(expected.issubset(self.segments[0].keys()))


class TestStaticAssets(unittest.TestCase):
    def test_format_helpers_loaded_before_app(self):
        client = TestClient(app)
        html = client.get("/").text
        self.assertLess(html.index("/static/format.js"), html.index("/static/app.js"))
        self.assertEqual(client.get("/static/format.js").status_code, 200)

    def test_table_no_longer_coerces_raw_values(self):
        with open(os.path.join(BASE_DIR, "src", "api", "static", "app.js"), encoding="utf-8") as f:
            js = f.read()
        for name in TABLE_MEASUREMENTS:
            self.assertNotIn(f"Number(seg.{name})", js)
            self.assertIn(f"formatMeasurement(seg.{name}", js)


@unittest.skipUnless(shutil.which("node"), "Node.js not installed")
class TestFormatJs(unittest.TestCase):
    def run_js(self, body):
        script = f"const f = require({json.dumps(FORMAT_JS)});\n{body}"
        out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
        return json.loads(out.stdout)

    def test_measurement_formatting(self):
        result = self.run_js("""
            const cases = {undef: undefined, nul: null, nan: NaN, inf: Infinity, ninf: -Infinity,
                           empty: '', text: 'abc', bool: true, zero: 0, num: 6.5, str: '158.2'};
            const out = {};
            for (const [k, v] of Object.entries(cases)) out[k] = f.formatMeasurement(v, 1, ' m');
            console.log(JSON.stringify(out));
        """)
        for key in ("undef", "nul", "nan", "inf", "ninf", "empty", "text", "bool"):
            self.assertIn("Unavailable", result[key], key)
            self.assertIn('class="value-unavailable"', result[key], key)
            self.assertNotIn("NaN", result[key])
        self.assertEqual(result["zero"], "0.0 m")
        self.assertEqual(result["num"], "6.5 m")
        self.assertEqual(result["str"], "158.2 m")

    def test_probability_formatting(self):
        result = self.run_js("""
            console.log(JSON.stringify([f.formatProbability(0.6599, 1), f.formatProbability(null, 1),
                                        f.formatProbability(NaN, 1), f.formatProbability(1.5, 1)]));
        """)
        self.assertEqual(result[0], "66.0%")
        for text in result[1:]:
            self.assertIn("Unavailable", text)

    def test_missing_values_sort_last(self):
        result = self.run_js("""
            const vals = [3, null, 1, NaN, 2];
            const asc = [...vals].sort((a, b) => f.compareValues(a, b, 1));
            const desc = [...vals].sort((a, b) => f.compareValues(a, b, -1));
            console.log(JSON.stringify({asc: asc.map(String), desc: desc.map(String)}));
        """)
        self.assertEqual(result["asc"][:3], ["1", "2", "3"])
        self.assertEqual(result["desc"][:3], ["3", "2", "1"])
        self.assertEqual(sorted(result["asc"][3:]), ["NaN", "null"])
        self.assertEqual(sorted(result["desc"][3:]), ["NaN", "null"])


if __name__ == "__main__":
    unittest.main()
