"""
test_extract_tabular_ndws.py - Unit Tests for NDWS Tabular Feature Extractor
"""

import unittest
import numpy as np
from src.features.extract_tabular_ndws import (
    validate_record_schema,
    compute_distance_to_prev_fire,
    process_single_record,
    EXPECTED_CHANNELS,
    ENVIRONMENTAL_FEATURES,
    FOCAL_FEATURES,
    compute_focal_neighborhood_features,
    GRID_DIM,
    EXPECTED_LEN,
    SENTINEL_NO_FIRE_DIST_KM
)


def create_synthetic_record(prev_fire_coords=None, fire_coords=None, nodata_coords=None):
    """
    Creates a synthetic 13-channel record dictionary for isolated unit testing.
    All channels default to valid float32 values.
    """
    record = {}
    for key in EXPECTED_CHANNELS:
        record[key] = np.zeros(EXPECTED_LEN, dtype=np.float32)
        
    # Baseline defaults
    record["elevation"] += 500.0
    record["NDVI"] += 4500.0  # 0.45 NDVI
    record["tmmn"] += 280.0
    record["tmmx"] += 295.0
    record["vs"] += 3.5
    record["erc"] += 25.0
    
    # PrevFireMask setup (0 = unburned by default)
    if prev_fire_coords:
        for r, c in prev_fire_coords:
            record["PrevFireMask"][r * GRID_DIM + c] = 1.0
            
    # FireMask setup (0 = unburned by default)
    if fire_coords:
        for r, c in fire_coords:
            record["FireMask"][r * GRID_DIM + c] = 1.0
            
    # Nodata setup (-1.0)
    if nodata_coords:
        for r, c in nodata_coords:
            record["FireMask"][r * GRID_DIM + c] = -1.0
            
    return record


class TestNDWSTabularExtractor(unittest.TestCase):

    def test_1_validate_schema_passes_on_valid_record(self):
        rec = create_synthetic_record()
        # Should execute without raising
        validate_record_schema(rec, record_idx=0, shard_name="test_shard")

    def test_2_validate_schema_raises_on_missing_key(self):
        rec = create_synthetic_record()
        del rec["elevation"]
        with self.assertRaises(ValueError) as ctx:
            validate_record_schema(rec, record_idx=1, shard_name="test_shard")
        self.assertIn("missing expected feature key 'elevation'", str(ctx.exception))

    def test_3_validate_schema_raises_on_invalid_array_length(self):
        rec = create_synthetic_record()
        rec["pr"] = np.zeros(100, dtype=np.float32)  # Wrong length
        with self.assertRaises(ValueError) as ctx:
            validate_record_schema(rec, record_idx=2, shard_name="test_shard")
        self.assertIn("invalid length 100", str(ctx.exception))

    def test_4_target_new_fire_logic(self):
        # Pixel (10, 10): Prev=0, Fire=1 -> Newly ignited spread (Target = 1)
        # Pixel (10, 11): Prev=1, Fire=1 -> Persistent fire core (Target = 0)
        # Pixel (10, 12): Prev=0, Fire=0 -> Unburned background (Target = 0)
        # Pixel (10, 13): Prev=1, Fire=0 -> Extinguished fire (Target = 0)
        prev_coords = [(10, 11), (10, 13)]
        fire_coords = [(10, 10), (10, 11)]
        rec = create_synthetic_record(prev_fire_coords=prev_coords, fire_coords=fire_coords)
        
        cols = process_single_record(rec, record_idx=0, shard_name="test_shard", partition="eval")
        self.assertIsNotNone(cols)
        
        # Check specific pixel coordinates
        row_arr = cols["row"]
        col_arr = cols["col"]
        target_arr = cols["target_new_fire"]
        
        idx_10_10 = np.where((row_arr == 10) & (col_arr == 10))[0][0]
        idx_10_11 = np.where((row_arr == 10) & (col_arr == 11))[0][0]
        idx_10_12 = np.where((row_arr == 10) & (col_arr == 12))[0][0]
        idx_10_13 = np.where((row_arr == 10) & (col_arr == 13))[0][0]
        
        self.assertEqual(target_arr[idx_10_10], 1, "Unburned -> Burned must have target_new_fire = 1")
        self.assertEqual(target_arr[idx_10_11], 0, "Burned -> Burned core must have target_new_fire = 0")
        self.assertEqual(target_arr[idx_10_12], 0, "Unburned -> Unburned background must have target_new_fire = 0")
        self.assertEqual(target_arr[idx_10_13], 0, "Burned -> Extinguished must have target_new_fire = 0")

    def test_5_exclusion_of_nodata_pixels(self):
        # Set pixel (5, 5) to nodata (-1.0) in FireMask
        # Set pixel (5, 6) to nodata (-1.0) in PrevFireMask
        rec = create_synthetic_record()
        rec["FireMask"][5 * GRID_DIM + 5] = -1.0
        rec["PrevFireMask"][5 * GRID_DIM + 6] = -1.0
        
        cols = process_single_record(rec, record_idx=0, shard_name="test_shard", partition="eval")
        row_arr = cols["row"]
        col_arr = cols["col"]
        
        # Total pixels should be 4096 - 2 = 4094
        self.assertEqual(len(row_arr), EXPECTED_LEN - 2)
        
        # Verify (5, 5) and (5, 6) are excluded
        has_5_5 = np.any((row_arr == 5) & (col_arr == 5))
        has_5_6 = np.any((row_arr == 5) & (col_arr == 6))
        self.assertFalse(has_5_5, "Pixel with FireMask == -1.0 must be excluded")
        self.assertFalse(has_5_6, "Pixel with PrevFireMask == -1.0 must be excluded")

    def test_6_target_derived_values_not_in_environmental_features(self):
        # Environmental feature list must NOT contain FireMask or any target derivation
        self.assertNotIn("FireMask", ENVIRONMENTAL_FEATURES)
        self.assertNotIn("target_new_fire", ENVIRONMENTAL_FEATURES)
        self.assertNotIn("fire_mask_t1", ENVIRONMENTAL_FEATURES)

    def test_7_stable_provenance_identifiers(self):
        rec = create_synthetic_record()
        cols = process_single_record(rec, record_idx=42, shard_name="shard_eval_00", partition="eval")
        
        self.assertEqual(cols["record_idx"][0], 42)
        self.assertEqual(cols["shard"][0], "shard_eval_00")
        self.assertEqual(cols["partition"][0], "eval")
        self.assertEqual(cols["sample_id"][0], "shard_eval_00_rec_00042")
        
        # All sample_ids in this record should match
        self.assertTrue(np.all(cols["sample_id"] == "shard_eval_00_rec_00042"))

    def test_8_spatial_distance_independent_of_target_firemask(self):
        # Active fire on day t at (20, 20)
        prev_coords = [(20, 20)]
        
        # Record A: Target day t+1 has fire at (20, 21)
        rec_a = create_synthetic_record(prev_fire_coords=prev_coords, fire_coords=[(20, 21)])
        # Record B: Target day t+1 has NO fire anywhere
        rec_b = create_synthetic_record(prev_fire_coords=prev_coords, fire_coords=[])
        # Record C: Target day t+1 has massive fire everywhere
        rec_c = create_synthetic_record(prev_fire_coords=prev_coords, fire_coords=[(r, c) for r in range(10) for c in range(10)])
        
        cols_a = process_single_record(rec_a, record_idx=0, shard_name="s", partition="eval")
        cols_b = process_single_record(rec_b, record_idx=0, shard_name="s", partition="eval")
        cols_c = process_single_record(rec_c, record_idx=0, shard_name="s", partition="eval")
        
        # Distance feature must be 100% IDENTICAL across all three records
        np.testing.assert_array_equal(
            cols_a["dist_to_prev_fire_km"],
            cols_b["dist_to_prev_fire_km"],
            err_msg="Distance to prev fire must not depend on FireMask"
        )
        np.testing.assert_array_equal(
            cols_a["dist_to_prev_fire_km"],
            cols_c["dist_to_prev_fire_km"],
            err_msg="Distance to prev fire must not depend on FireMask"
        )
        
        # Pixel at (20, 20) should have distance 0.0
        row_arr = cols_a["row"]
        col_arr = cols_a["col"]
        idx_20_20 = np.where((row_arr == 20) & (col_arr == 20))[0][0]
        self.assertEqual(cols_a["dist_to_prev_fire_km"][idx_20_20], 0.0)
        
        # Pixel at (20, 23) should have distance 3.0 km
        idx_20_23 = np.where((row_arr == 20) & (col_arr == 23))[0][0]
        self.assertAlmostEqual(cols_a["dist_to_prev_fire_km"][idx_20_23], 3.0, places=4)

    def test_9_spatial_distance_sentinel_when_no_active_fire(self):
        # Patch with zero active fire on day t
        rec = create_synthetic_record(prev_fire_coords=[], fire_coords=[(10, 10)])
        cols = process_single_record(rec, record_idx=0, shard_name="s", partition="eval")
        
        # When has_prev_fire_in_patch == 0, distance is sentinel 100.0 km
        self.assertEqual(cols["has_prev_fire_in_patch"][0], 0)
        self.assertTrue(np.all(cols["dist_to_prev_fire_km"] == SENTINEL_NO_FIRE_DIST_KM))

    def test_10_infer_partition_from_name(self):
        from src.features.extract_tabular_ndws import infer_partition_from_name
        self.assertEqual(infer_partition_from_name("next_day_wildfire_spread_train_00.tfrecord"), "train")
        self.assertEqual(infer_partition_from_name("next_day_wildfire_spread_eval_01.tfrecord"), "eval")
        self.assertEqual(infer_partition_from_name("next_day_wildfire_spread_test_00.tfrecord"), "test")
        self.assertEqual(infer_partition_from_name("custom_patch_05.tfrecord"), "unknown")

    def test_11_schema_dtypes_and_column_completeness(self):
        from src.features.extract_tabular_ndws import get_pyarrow_schema
        schema = get_pyarrow_schema()
        rec = create_synthetic_record()
        cols = process_single_record(rec, record_idx=1, shard_name="train_00", partition="train")
        
        self.assertIsNotNone(cols)
        schema_field_names = set(schema.names)
        record_col_names = set(cols.keys())
        self.assertEqual(
            schema_field_names,
            record_col_names,
            "Every column produced by process_single_record must exist in PyArrow schema"
        )

    def test_12_focal_neighborhood_feature_computation(self):
        """Verifies accurate calculation of 3x3 focal fire density, NDVI mean, and elevation std."""
        rec = create_synthetic_record(prev_fire_coords=[(10, 10)])
        cols = process_single_record(rec, record_idx=1, shard_name="train_00", partition="train", include_neighborhood=True)
        self.assertIsNotNone(cols)

        # Check that all focal features are present
        for f in FOCAL_FEATURES:
            self.assertIn(f, cols, f"Focal feature {f} missing from extracted columns")

        row_arr = cols["row"]
        col_arr = cols["col"]
        idx_10_10 = np.where((row_arr == 10) & (col_arr == 10))[0][0]

        # 1 active fire pixel in a 3x3 window (9 pixels) => density = 1/9 ~ 0.1111
        fire_density = cols["focal_prev_fire_density_3x3"][idx_10_10]
        self.assertAlmostEqual(fire_density, 1.0 / 9.0, places=4)

        # In a uniform elevation field, elevation std should be 0.0
        elev_std = cols["focal_elevation_std_3x3"][idx_10_10]
        self.assertAlmostEqual(elev_std, 0.0, places=4)

    def test_13_focal_neighborhood_no_target_leakage(self):
        """Proves focal neighborhood features are completely independent of day t+1 FireMask."""
        prev_coords = [(15, 15)]
        rec_empty = create_synthetic_record(prev_fire_coords=prev_coords, fire_coords=[])
        rec_heavy = create_synthetic_record(
            prev_fire_coords=prev_coords,
            fire_coords=[(r, c) for r in range(12, 18) for c in range(12, 18)]
        )

        cols_empty = process_single_record(rec_empty, record_idx=0, shard_name="s", partition="eval", include_neighborhood=True)
        cols_heavy = process_single_record(rec_heavy, record_idx=0, shard_name="s", partition="eval", include_neighborhood=True)

        for f in FOCAL_FEATURES:
            np.testing.assert_array_equal(
                cols_empty[f],
                cols_heavy[f],
                err_msg=f"Focal feature {f} leaked day t+1 FireMask!"
            )

    def test_14_focal_neighborhood_schema_completeness(self):
        """Verifies schema and columns match perfectly when include_neighborhood=True."""
        from src.features.extract_tabular_ndws import get_pyarrow_schema
        schema = get_pyarrow_schema(include_neighborhood=True)
        rec = create_synthetic_record()
        cols = process_single_record(rec, record_idx=2, shard_name="eval_00", partition="eval", include_neighborhood=True)

        self.assertEqual(set(schema.names), set(cols.keys()))


if __name__ == "__main__":
    unittest.main()
