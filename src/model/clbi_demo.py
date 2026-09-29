"""
clbi_demo.py - CLI Demonstration & Production Demo Generator

1. Generates data/demo/clbi_demo_segments.csv containing 100 contiguous candidate segments from the CZU Lightning Complex fire.
2. Loads the serialized production model artifact (models/clbi_logistic_regression.joblib).
3. Runs production CLBI inference over the 100 demo segments.
4. Generates experiments/results/clbi_demo_predictions.csv.
5. Generates experiments/results/clbi_demo_response.json.
6. Prints a formatted CLI vulnerability and prioritization summary.
"""

import os
import json
import pandas as pd
from src.model.clbi_model import CLBIDecisionEngine

def create_demo_dataset():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parquet_path = os.path.join(base_dir, "data", "final", "fireline_segments_v1.parquet")
    demo_dir = os.path.join(base_dir, "data", "demo")
    os.makedirs(demo_dir, exist_ok=True)
    
    df = pd.read_parquet(parquet_path)
    
    # Deterministically select 100 contiguous segments from the CZU Lightning Complex test fire
    czu_df = df[df['fire_id'] == 'CA-CZU-005205'].sort_values(by="segment_id").reset_index(drop=True)
    demo_df = czu_df.head(100).copy()
    
    demo_csv_path = os.path.join(demo_dir, "clbi_demo_segments.csv")
    demo_df.to_csv(demo_csv_path, index=False)
    print(f"Created 100-segment demo dataset: {demo_csv_path}")
    return demo_df

def run_demo():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(base_dir, "experiments", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 80)
    print("CANDIDATE LINE BREACH INTELLIGENCE (CLBI) — DEMO EXECUTION")
    print("=" * 80)
    
    demo_df = create_demo_dataset()
    
    # Initialize engine (loads pre-trained production model artifact)
    engine = CLBIDecisionEngine()
    metadata = engine.get_model_metadata()
    print(f"Loaded production model: {metadata.get('model_name', 'CLBI Model')}")
    
    print("Running production inference across 100 candidate fireline segments...")
    predicted_df = engine.predict_df(demo_df, top_k_recommend_pct=10.0)
    
    summary = engine.summarize_risk(predicted_df)
    
    # Export Demo Predictions CSV
    out_cols = [
        "fire_id", "fire_name", "segment_id", "date",
        "breach_probability", "hold_probability", "risk_tier",
        "vulnerability_rank", "priority_percentile", "priority_recommended",
        "top_risk_factors", "protective_factors", "explanation"
    ]
    
    export_df = predicted_df[out_cols].copy()
    export_df["top_risk_factors"] = export_df["top_risk_factors"].apply(lambda x: "; ".join(x) if isinstance(x, list) else str(x))
    export_df["protective_factors"] = export_df["protective_factors"].apply(lambda x: "; ".join(x) if isinstance(x, list) else str(x))
    
    demo_csv_out = os.path.join(results_dir, "clbi_demo_predictions.csv")
    export_df.to_csv(demo_csv_out, index=False)
    print(f"Saved demo predictions CSV to: {demo_csv_out}")
    
    # Export Product-Ready JSON Response
    segment_json_list = []
    for idx, row in predicted_df.iterrows():
        segment_json_list.append({
            "segment_id": row["segment_id"],
            "fire_id": row["fire_id"],
            "fire_name": row["fire_name"],
            "date": str(row["date"]),
            "breach_probability": float(row["breach_probability"]),
            "hold_probability": float(row["hold_probability"]),
            "risk_tier": row["risk_tier"],
            "vulnerability_rank": int(row["vulnerability_rank"]),
            "priority_percentile": float(row["priority_percentile"]),
            "priority_recommended": bool(row["priority_recommended"]),
            "top_risk_factors": row["top_risk_factors"],
            "protective_factors": row["protective_factors"],
            "explanation": row["explanation"]
        })
        
    demo_json_response = {
        "metadata": {
            "product": "Candidate Line Breach Intelligence",
            "version": "v1.0.0",
            "model": "Standardized 7-Feature Logistic Regression Pipeline",
            "data_source": "NDWS-derived historical benchmark (CZU Lightning Complex)",
            "retrospective": True
        },
        "summary": {
            "total_segments": summary["total_segments"],
            "low": summary["risk_tier_counts"]["LOW"],
            "medium": summary["risk_tier_counts"]["MEDIUM"],
            "high": summary["risk_tier_counts"]["HIGH"],
            "critical": summary["risk_tier_counts"]["CRITICAL"],
            "high_or_critical": summary["high_or_critical_count"]
        },
        "prioritization": {
            "top_10_percent_count": summary["top_10_pct_cutoff_count"],
            "top_20_percent_count": summary["top_20_pct_cutoff_count"]
        },
        "segments": segment_json_list
    }
    
    demo_json_out = os.path.join(results_dir, "clbi_demo_response.json")
    with open(demo_json_out, "w") as f:
        json.dump(demo_json_response, f, indent=2)
    print(f"Saved product-ready demo JSON response to: {demo_json_out}")
    
    # Print formatted CLI summary
    print("\n" + "=" * 80)
    print("CLBI 100-SEGMENT DEMO SUMMARY")
    print("=" * 80)
    print(f"Total Candidate Segments:     {summary['total_segments']}")
    print(f"LOW Risk (<0.30):             {summary['risk_tier_counts']['LOW']}")
    print(f"MEDIUM Risk (0.30-0.60):       {summary['risk_tier_counts']['MEDIUM']}")
    print(f"HIGH Risk (0.60-0.80):         {summary['risk_tier_counts']['HIGH']}")
    print(f"CRITICAL Risk (>=0.80):       {summary['risk_tier_counts']['CRITICAL']}")
    print(f"High / Critical Total:        {summary['high_or_critical_count']} ({summary['pct_line_high_or_critical']}%)")
    print(f"Mean Breach Probability:      {summary['mean_breach_probability']:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    run_demo()
