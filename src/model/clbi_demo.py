"""
clbi_demo.py - CLI Demonstration for Candidate Line Breach Intelligence (CLBI)

Loads the validated benchmark dataset, fits the CLBI decision engine, executes inference across all 4,600
candidate fireline segments, and displays formatted risk stratification and prioritization reports.
"""

import os
import pandas as pd
from src.model.clbi_model import CLBIDecisionEngine

def run_demo():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "final", "fireline_segments_v1.parquet")
    
    print("=" * 80)
    print("CANDIDATE LINE BREACH INTELLIGENCE (CLBI) — CLI DEMONSTRATION")
    print("=" * 80)
    print(f"Loading benchmark dataset: {data_path}")
    
    df = pd.read_parquet(data_path)
    train_df = df[df['split'] == 'train'].copy()
    
    # Initialize and fit decision engine on training split
    engine = CLBIDecisionEngine()
    engine.fit(train_df)
    
    print("Executing CLBI inference across all 4,600 candidate fireline segments...")
    predicted_df = engine.predict_df(df, top_k_recommend_pct=10.0)
    
    summary = engine.summarize_risk(predicted_df)
    
    print("\n" + "=" * 80)
    print("CLBI CANDIDATE-LINE VULNERABILITY SUMMARY")
    print("=" * 80)
    print(f"Total Candidate Segments:     {summary['total_segments']}")
    print(f"LOW Risk (<0.30):             {summary['risk_tier_counts']['LOW']}")
    print(f"MEDIUM Risk (0.30-0.60):       {summary['risk_tier_counts']['MEDIUM']}")
    print(f"HIGH Risk (0.60-0.80):         {summary['risk_tier_counts']['HIGH']}")
    print(f"CRITICAL Risk (>=0.80):       {summary['risk_tier_counts']['CRITICAL']}")
    print(f"High / Critical Total:        {summary['high_or_critical_count']} ({summary['pct_line_high_or_critical']}%)")
    print(f"Mean Breach Probability:      {summary['mean_breach_probability']:.4f}")
    print(f"Max Breach Probability:       {summary['max_breach_probability']:.4f}")
    print(f"Min Breach Probability:       {summary['min_breach_probability']:.4f}")
    print("=" * 80)
    
    # Top 10 Most Vulnerable Segments
    print("\n--- TOP 10 MOST VULNERABLE CANDIDATE SEGMENTS ---")
    top_10 = predicted_df.sort_values(by="vulnerability_rank").head(10)
    print(f"{'Rank':<6} {'Segment ID':<25} {'Incident':<18} {'P(Breach)':<12} {'Risk Tier':<10} {'Main Risk Factor'}")
    print("-" * 90)
    for idx, row in top_10.iterrows():
        rank = row["vulnerability_rank"]
        seg_id = row["segment_id"]
        fname = row["fire_name"]
        p_b = row["breach_probability"]
        tier = row["risk_tier"]
        main_factor = row["top_risk_factors"][0] if row["top_risk_factors"] else "N/A"
        print(f"{rank:<6} {seg_id:<25} {fname:<18} {p_b:<12.4f} {tier:<10} {main_factor}")
        
    # Export predictions to CSV artifact
    results_dir = os.path.join(base_dir, "experiments", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "clbi_segment_predictions.csv")
    
    # Ensure correct column ordering for export
    out_cols = [
        "fire_id", "fire_name", "segment_id", "date", "split",
        "slope", "elevation", "distance_to_fire", "barrier_width_m",
        "burn_prob", "attack_angle", "attack_dot_product",
        "breach_probability", "hold_probability", "risk_tier",
        "vulnerability_rank", "priority_percentile", "priority_recommended",
        "top_risk_factors", "protective_factors", "explanation"
    ]
    
    # Format list columns as JSON strings for clean CSV export
    export_df = predicted_df[out_cols].copy()
    export_df["top_risk_factors"] = export_df["top_risk_factors"].apply(lambda x: "; ".join(x) if isinstance(x, list) else str(x))
    export_df["protective_factors"] = export_df["protective_factors"].apply(lambda x: "; ".join(x) if isinstance(x, list) else str(x))
    
    export_df.to_csv(out_csv, index=False)
    print(f"\nSaved CLBI segment predictions (4,600 rows) to: {out_csv}")
    
    # Prioritization Analysis
    prio_result = engine.prioritize_candidate_line(predicted_df, resource_pct=10.0)
    print("\n" + "=" * 80)
    print("RISK-BASED CANDIDATE-LINE PRIORITIZATION (TOP 10% INSPECTION SET)")
    print("=" * 80)
    print(f"Total Line Selected:           {prio_result['total_selected_segments']} / {prio_result['total_candidate_segments']} segments ({prio_result['pct_line_selected']}%)")
    print(f"Selected Mean Breach Risk:     {prio_result['selected_mean_breach_probability']:.4f}")
    print(f"Selected Risk Tier Breakdown:  {prio_result['selected_risk_tier_counts']}")
    print("=" * 80)

if __name__ == "__main__":
    run_demo()

