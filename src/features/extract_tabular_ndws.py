"""
extract_tabular_ndws.py - NDWS Spatial Raster to Tabular Feature Extractor

Extracts structured, tabular feature vectors from raw 64x64 multi-channel
TFRecord patches for Phase 1 Machine Learning experiments.

Key Principles:
1. Strict pre-engagement features: 11 non-mask channels + spatial context derived
   SOLELY from PrevFireMask (day t).
2. Explicit target: target_new_fire = 1 when (PrevFireMask == 0) and (FireMask == 1).
3. Explicit validity mask: valid = (PrevFireMask != -1) and (FireMask != -1).
   Pixels with cloud obscuration, water bodies, or nodata (-1.0) are excluded.
4. Provenance tracking: Every row carries sample_id, record_idx, row, col, shard, partition.
5. Scalable streaming: Uses ParquetWriter to stream record batches with minimal memory footprint.
"""

import os
import argparse
import time
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.ndimage as ndi
import tfrecord

EXPECTED_CHANNELS: List[str] = [
    "FireMask",
    "PrevFireMask",
    "elevation",
    "th",
    "vs",
    "tmmn",
    "tmmx",
    "sph",
    "pr",
    "pdsi",
    "NDVI",
    "population",
    "erc",
]

ENVIRONMENTAL_FEATURES: List[str] = [
    "elevation",
    "th",
    "vs",
    "tmmn",
    "tmmx",
    "sph",
    "pr",
    "pdsi",
    "NDVI",
    "population",
    "erc",
]

GRID_DIM: int = 64
EXPECTED_LEN: int = GRID_DIM * GRID_DIM  # 4096
SENTINEL_NO_FIRE_DIST_KM: float = 100.0  # Max diagonal in 64x64 grid is ~89.1 km

FOCAL_FEATURES: List[str] = [
    "focal_prev_fire_density_3x3",
    "focal_elevation_std_3x3",
    "focal_ndvi_mean_3x3",
]


def infer_partition_from_name(filename: str) -> str:
    """Infers official dataset partition from shard filename."""
    lower_name = os.path.basename(filename).lower()
    if "train" in lower_name:
        return "train"
    elif "eval" in lower_name or "val" in lower_name:
        return "eval"
    elif "test" in lower_name:
        return "test"
    return "unknown"


def validate_record_schema(record: Dict[str, Any], record_idx: int = 0, shard_name: str = "unknown") -> None:
    """
    Validates that a record contains all 13 expected feature keys and that
    all arrays contain exactly 4096 elements.
    Raises ValueError with detailed diagnostics if validation fails.
    """
    for key in EXPECTED_CHANNELS:
        if key not in record:
            raise ValueError(
                f"Record {record_idx} in shard '{shard_name}' is malformed: "
                f"missing expected feature key '{key}'."
            )
        val = record[key]
        length = getattr(val, "shape", (len(val),))[0] if hasattr(val, "shape") else len(val)
        if length != EXPECTED_LEN:
            raise ValueError(
                f"Record {record_idx} in shard '{shard_name}' key '{key}' "
                f"has invalid length {length}; expected {EXPECTED_LEN} (64x64)."
            )


def compute_distance_to_prev_fire(prev_mask_2d: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Calculates the Euclidean distance (in kilometers / grid pixel units) to the nearest
    actively burning pixel (PrevFireMask == 1.0) on day t.
    
    CRITICAL LEAKAGE SAFEGUARD:
    This function takes ONLY PrevFireMask. It has zero knowledge of FireMask (day t+1).
    
    If no active fire exists in the 64x64 patch, distance is assigned a sentinel value of 100.0 km
    (greater than the maximum possible intra-patch diagonal distance of ~89.1 km), and
    has_prev_fire is False.
    """
    prev_fire_binary = (prev_mask_2d == 1.0)
    has_prev_fire = bool(np.any(prev_fire_binary))
    
    if has_prev_fire:
        # Distance transform computes distance from non-zero elements to the nearest zero element.
        # Passing ~prev_fire_binary computes distance to the nearest True element in prev_fire_binary.
        dist_grid = ndi.distance_transform_edt(~prev_fire_binary).astype(np.float32)
    else:
        dist_grid = np.full((GRID_DIM, GRID_DIM), SENTINEL_NO_FIRE_DIST_KM, dtype=np.float32)
        
    return dist_grid, has_prev_fire


def compute_focal_neighborhood_features(
    prev_mask_2d: np.ndarray,
    elevation_2d: np.ndarray,
    ndvi_2d: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Computes 2D focal / neighborhood aggregates on the 64x64 spatial patch
    strictly prior to flattening into tabular rows.

    Scientific rationale & leakage safeguards:
    1. Active fire density: 3x3 uniform filter on (PrevFireMask == 1.0).
       Measures local fire clustering at day t. Day t+1 FireMask is strictly excluded.
    2. Terrain roughness: 3x3 standard deviation of elevation.
       Measures local topographic steepness/variability.
    3. Vegetation density: 3x3 uniform filter of NDVI.
       Smooths local fuel density.
    4. Boundary handling: mode='reflect' preserves spatial continuity at patch edges
       without introducing zero-padding artifacts.
    5. Isolation: Computed per-patch with zero leakage across shard boundaries.
    """
    # 1. Day-t active fire density (3x3 window)
    prev_fire_binary = (prev_mask_2d == 1.0).astype(np.float32)
    focal_fire_density = ndi.uniform_filter(prev_fire_binary, size=3, mode="reflect")

    # 2. Local elevation roughness (std over 3x3 window)
    mean_elev = ndi.uniform_filter(elevation_2d.astype(np.float32), size=3, mode="reflect")
    sq_mean_elev = ndi.uniform_filter((elevation_2d.astype(np.float32)) ** 2, size=3, mode="reflect")
    elev_var = np.maximum(0.0, sq_mean_elev - (mean_elev ** 2))
    focal_elevation_std = np.sqrt(elev_var)

    # 3. Local NDVI mean (3x3 window)
    focal_ndvi_mean = ndi.uniform_filter(ndvi_2d.astype(np.float32), size=3, mode="reflect")

    return {
        "focal_prev_fire_density_3x3": focal_fire_density.astype(np.float32),
        "focal_elevation_std_3x3": focal_elevation_std.astype(np.float32),
        "focal_ndvi_mean_3x3": focal_ndvi_mean.astype(np.float32),
    }


def process_single_record(
    record: Dict[str, Any],
    record_idx: int,
    shard_name: str,
    partition: str,
    include_neighborhood: bool = False
) -> Optional[Dict[str, np.ndarray]]:
    """
    Processes a single record:
    1. Validates the schema.
    2. Reshapes arrays to 64x64.
    3. Evaluates the validity mask (filtering cloud/nodata -1.0).
    4. Evaluates the target (target_new_fire = 1 iff PrevFireMask == 0 and FireMask == 1).
    5. Computes spatial distance strictly from PrevFireMask.
    6. Filters all arrays by the validity mask.
    Returns a dictionary of column arrays, or None if no valid pixels exist.
    """
    validate_record_schema(record, record_idx, shard_name)
    
    # 2D Reshaping
    prev_mask_2d = record["PrevFireMask"].reshape((GRID_DIM, GRID_DIM))
    fire_mask_2d = record["FireMask"].reshape((GRID_DIM, GRID_DIM))
    
    # Step 3: Explicit validity mask
    # -1.0 denotes missing, cloudy, water, or sensor nodata
    valid_mask_2d = (prev_mask_2d != -1.0) & (fire_mask_2d != -1.0)
    valid_count = int(np.sum(valid_mask_2d))
    
    if valid_count == 0:
        return None
        
    # Step 3: Explicit target definition
    # target_new_fire = 1 iff the cell was unburned at day t (PrevFireMask == 0) and burning at day t+1 (FireMask == 1)
    target_new_fire_2d = ((prev_mask_2d == 0.0) & (fire_mask_2d == 1.0)).astype(np.int32)
    
    # Optional target reference for point active fire presence at t+1:
    fire_mask_t1_2d = (fire_mask_2d == 1.0).astype(np.int32)
    prev_fire_mask_t0_2d = (prev_mask_2d == 1.0).astype(np.int32)
    
    # Step 5: Spatial context feature strictly from PrevFireMask
    dist_to_prev_fire_2d, has_prev_fire = compute_distance_to_prev_fire(prev_mask_2d)
    
    # Coordinate grids
    row_indices_2d, col_indices_2d = np.indices((GRID_DIM, GRID_DIM))
    
    # Apply validity filter to extract 1D flat vectors
    valid_rows = row_indices_2d[valid_mask_2d]
    valid_cols = col_indices_2d[valid_mask_2d]
    
    columns_data: Dict[str, np.ndarray] = {
        "sample_id": np.array([f"{shard_name}_rec_{record_idx:05d}"] * valid_count, dtype=object),
        "record_idx": np.full(valid_count, record_idx, dtype=np.int32),
        "row": valid_rows.astype(np.int32),
        "col": valid_cols.astype(np.int32),
        "shard": np.array([shard_name] * valid_count, dtype=object),
        "partition": np.array([partition] * valid_count, dtype=object),
    }
    
    # Add 11 verified environmental features
    for feat_name in ENVIRONMENTAL_FEATURES:
        feat_2d = record[feat_name].reshape((GRID_DIM, GRID_DIM))
        columns_data[feat_name] = feat_2d[valid_mask_2d].astype(np.float32)
        
    # Add spatial context features
    columns_data["dist_to_prev_fire_km"] = dist_to_prev_fire_2d[valid_mask_2d].astype(np.float32)
    columns_data["has_prev_fire_in_patch"] = np.full(valid_count, int(has_prev_fire), dtype=np.int32)

    # Optional 2D focal / neighborhood features
    if include_neighborhood:
        elev_2d = record["elevation"].reshape((GRID_DIM, GRID_DIM))
        ndvi_2d = record["NDVI"].reshape((GRID_DIM, GRID_DIM))
        focal_map = compute_focal_neighborhood_features(prev_mask_2d, elev_2d, ndvi_2d)
        for focal_name in FOCAL_FEATURES:
            columns_data[focal_name] = focal_map[focal_name][valid_mask_2d].astype(np.float32)
    
    # Add state and targets
    columns_data["prev_fire_mask"] = prev_fire_mask_t0_2d[valid_mask_2d].astype(np.int32)
    columns_data["fire_mask_t1"] = fire_mask_t1_2d[valid_mask_2d].astype(np.int32)
    columns_data["target_new_fire"] = target_new_fire_2d[valid_mask_2d].astype(np.int32)
    
    return columns_data


def get_pyarrow_schema(include_neighborhood: bool = False) -> pa.Schema:
    """Constructs the explicit PyArrow schema for typed Parquet streaming."""
    fields = [
        pa.field("sample_id", pa.string()),
        pa.field("record_idx", pa.int32()),
        pa.field("row", pa.int32()),
        pa.field("col", pa.int32()),
        pa.field("shard", pa.string()),
        pa.field("partition", pa.string()),
    ]
    for feat in ENVIRONMENTAL_FEATURES:
        fields.append(pa.field(feat, pa.float32()))
    fields.extend([
        pa.field("dist_to_prev_fire_km", pa.float32()),
        pa.field("has_prev_fire_in_patch", pa.int32()),
    ])
    if include_neighborhood:
        for focal_feat in FOCAL_FEATURES:
            fields.append(pa.field(focal_feat, pa.float32()))
    fields.extend([
        pa.field("prev_fire_mask", pa.int32()),
        pa.field("fire_mask_t1", pa.int32()),
        pa.field("target_new_fire", pa.int32()),
    ])
    return pa.schema(fields)


def extract_shard_to_parquet(
    input_tfrecord: str,
    output_parquet: str,
    max_records: Optional[int] = None,
    batch_size: int = 50,
    partition: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extracts tabular records from an NDWS TFRecord shard and writes them incrementally
    to a Parquet file.
    """
    if not os.path.exists(input_tfrecord):
        raise FileNotFoundError(f"Input TFRecord not found: {input_tfrecord}")
        
    shard_name = os.path.splitext(os.path.basename(input_tfrecord))[0]
    assigned_partition = partition or infer_partition_from_name(shard_name)
    
    # Ensure output parent directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_parquet)), exist_ok=True)
    
    schema = get_pyarrow_schema()
    loader = tfrecord.tfrecord_loader(input_tfrecord, None)
    
    start_time = time.time()
    total_records = 0
    records_with_valid_pixels = 0
    total_valid_pixels = 0
    total_new_fire_pixels = 0
    
    batch_dict: Dict[str, List[np.ndarray]] = {field.name: [] for field in schema}
    records_in_batch = 0
    
    writer = None
    try:
        writer = pq.ParquetWriter(output_parquet, schema, compression="snappy")
        
        for idx, record in enumerate(loader):
            if max_records is not None and idx >= max_records:
                break
                
            total_records += 1
            rec_cols = process_single_record(record, idx, shard_name, assigned_partition)
            
            if rec_cols is not None:
                records_with_valid_pixels += 1
                n_pixels = len(rec_cols["row"])
                total_valid_pixels += n_pixels
                total_new_fire_pixels += int(np.sum(rec_cols["target_new_fire"]))
                
                for k in rec_cols:
                    batch_dict[k].append(rec_cols[k])
                records_in_batch += 1
                
            if records_in_batch >= batch_size:
                # Flush batch to Parquet
                concatenated_cols = {
                    k: np.concatenate(batch_dict[k]) if len(batch_dict[k]) > 0 else np.array([], dtype=object)
                    for k in batch_dict
                }
                table = pa.Table.from_pydict(concatenated_cols, schema=schema)
                writer.write_table(table)
                
                # Reset batch
                batch_dict = {field.name: [] for field in schema}
                records_in_batch = 0
                
        # Final flush
        if records_in_batch > 0:
            concatenated_cols = {
                k: np.concatenate(batch_dict[k]) if len(batch_dict[k]) > 0 else np.array([], dtype=object)
                for k in batch_dict
            }
            table = pa.Table.from_pydict(concatenated_cols, schema=schema)
            writer.write_table(table)
            
    finally:
        if writer is not None:
            writer.close()
            
    elapsed = time.time() - start_time
    file_size_bytes = os.path.getsize(output_parquet) if os.path.exists(output_parquet) else 0
    
    summary = {
        "shard_name": shard_name,
        "partition": assigned_partition,
        "input_tfrecord": input_tfrecord,
        "output_parquet": output_parquet,
        "output_size_bytes": file_size_bytes,
        "total_records_read": total_records,
        "records_with_valid_pixels": records_with_valid_pixels,
        "total_valid_pixels": total_valid_pixels,
        "total_new_fire_pixels": total_new_fire_pixels,
        "new_fire_prevalence_pct": (total_new_fire_pixels / total_valid_pixels * 100.0) if total_valid_pixels > 0 else 0.0,
        "elapsed_seconds": round(elapsed, 2),
    }
    
    return summary


def main():
    import glob
    parser = argparse.ArgumentParser(description="Extract tabular features from NDWS TFRecord shard(s).")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input TFRecord shard or wildcard pattern (e.g. 'path/*train*.tfrecord')."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing TFRecord shards to extract."
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*train*.tfrecord",
        help="Filename pattern to match when using --input-dir (default: '*train*.tfrecord')."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output Parquet destination (for single shard extraction)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/ndws",
        help="Directory to write output Parquet files (default: 'data/processed/ndws')."
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Maximum records to process per shard (default: all)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Record batch buffer size before flushing to Parquet (default: 50)."
    )
    parser.add_argument(
        "--partition",
        type=str,
        default=None,
        choices=["train", "eval", "test", "unknown"],
        help="Override partition label (default: inferred from filename)."
    )
    
    args = parser.parse_args()
    
    # Determine target shard list
    shard_paths: List[str] = []
    if args.input_dir:
        shard_paths = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
        if not shard_paths:
            print(f"Error: No shards matching '{args.pattern}' found in '{args.input_dir}'.")
            return
    elif args.input:
        if "*" in args.input or "?" in args.input:
            shard_paths = sorted(glob.glob(args.input))
            if not shard_paths:
                print(f"Error: No shards matching pattern '{args.input}'.")
                return
        else:
            shard_paths = [args.input]
    else:
        # Default single sample fallback
        shard_paths = ["data/raw/ndws/sample/next_day_wildfire_spread_eval_00.tfrecord"]
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"============================================================")
    print(f"NDWS Tabular Feature Extractor")
    print(f"Total Shards to Process: {len(shard_paths)}")
    print(f"Output Directory:        {args.output_dir}")
    print(f"Max Records per Shard:   {args.max_records or 'All'}")
    print(f"Streaming Batch Size:    {args.batch_size}")
    print(f"============================================================")
    
    all_summaries: List[Dict[str, Any]] = []
    total_start = time.time()
    
    for i, shard_path in enumerate(shard_paths, 1):
        shard_base = os.path.splitext(os.path.basename(shard_path))[0]
        if len(shard_paths) == 1 and args.output:
            out_path = args.output
        else:
            out_path = os.path.join(args.output_dir, f"{shard_base}_tabular.parquet")
            
        print(f"\n[{i}/{len(shard_paths)}] Processing shard: {os.path.basename(shard_path)}")
        summary = extract_shard_to_parquet(
            input_tfrecord=shard_path,
            output_parquet=out_path,
            max_records=args.max_records,
            batch_size=args.batch_size,
            partition=args.partition
        )
        all_summaries.append(summary)
        print(f"    Records: {summary['total_records_read']} | Valid Pixels: {summary['total_valid_pixels']:,} | "
              f"New Fire (Y=1): {summary['total_new_fire_pixels']:,} ({summary['new_fire_prevalence_pct']:.3f}%) | "
              f"Size: {summary['output_size_bytes'] / (1024*1024):.2f} MB | Time: {summary['elapsed_seconds']}s")
              
    total_elapsed = time.time() - total_start
    total_records = sum(s["total_records_read"] for s in all_summaries)
    total_valid = sum(s["total_valid_pixels"] for s in all_summaries)
    total_new_fire = sum(s["total_new_fire_pixels"] for s in all_summaries)
    total_bytes = sum(s["output_size_bytes"] for s in all_summaries)
    overall_prev = (total_new_fire / total_valid * 100.0) if total_valid > 0 else 0.0
    
    print(f"\n============================================================")
    print(f"BATCH EXTRACTION COMPLETE")
    print(f"============================================================")
    print(f"Total Shards Extracted:      {len(all_summaries)}")
    print(f"Total Records Inspected:     {total_records:,}")
    print(f"Total Valid Pixels:          {total_valid:,}")
    print(f"Total New Fire Pixels (Y=1): {total_new_fire:,}")
    print(f"Overall Positive Prevalence: {overall_prev:.3f}%")
    print(f"Total Parquet Output Size:   {total_bytes / (1024*1024):.2f} MB ({total_bytes / (1024**3):.2f} GB)")
    print(f"Total Elapsed Time:          {total_elapsed:.2f}s ({total_elapsed / 60:.2f} min)")
    print(f"============================================================")


if __name__ == "__main__":
    main()
