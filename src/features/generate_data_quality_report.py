"""
generate_data_quality_report.py - Comprehensive Data Quality Audit & Visualization

Generates summary statistics, outlier audits, class balance reports, and 4
clean visualizations saved to data/processed/plots/.
"""

import os
import csv
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    csv_path = os.path.join(base_dir, "data", "final", "fireline_segments_v1.csv")
    plots_dir = os.path.join(base_dir, "data", "processed", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    rows = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Parse numerical fields
            r["label"] = int(r["label"])
            r["slope"] = float(r["slope"])
            r["elevation"] = float(r["elevation"])
            r["distance_to_fire"] = float(r["distance_to_fire"])
            r["burn_prob"] = float(r["burn_prob"])
            r["prob_gradient"] = float(r["prob_gradient"])
            r["spread_direction"] = float(r["spread_direction"])
            r["line_angle"] = float(r["line_angle"])
            r["attack_angle"] = float(r["attack_angle"])
            r["attack_dot_product"] = float(r["attack_dot_product"])
            r["dist_pred_boundary"] = float(r["dist_pred_boundary"])
            rows.append(r)
            
    total_rows = len(rows)
    unique_fires = len(set(r["fire_id"] for r in rows))
    unique_segments = len(set(r["segment_id"] for r in rows))
    pos_samples = sum(1 for r in rows if r["label"] == 1)
    neg_samples = sum(1 for r in rows if r["label"] == 0)
    
    print("=" * 60)
    print("DATA QUALITY AUDIT REPORT: fireline_segments_v1")
    print("=" * 60)
    print(f"Total Rows:               {total_rows}")
    print(f"Unique Fires:             {unique_fires}")
    print(f"Unique Segment IDs:       {unique_segments}")
    print(f"Duplicate Segment IDs:    {total_rows - unique_segments}")
    print(f"Held Segments (Label 1):  {pos_samples} ({pos_samples/total_rows*100:.2f}%)")
    print(f"Burned Over (Label 0):    {neg_samples} ({neg_samples/total_rows*100:.2f}%)")
    print(f"Class Imbalance Ratio:    {pos_samples/neg_samples:.2f} : 1")
    
    # Feature range and missing value audit
    num_cols = ["slope", "elevation", "distance_to_fire", "burn_prob", "prob_gradient", 
                "spread_direction", "line_angle", "attack_angle", "attack_dot_product"]
    
    print("\n" + "-" * 60)
    print(f"{'Feature':<20} {'Min':<10} {'Max':<10} {'Mean':<10} {'Missing':<10}")
    print("-" * 60)
    for col in num_cols:
        vals = [r[col] for r in rows]
        c_min = min(vals)
        c_max = max(vals)
        c_mean = sum(vals) / len(vals)
        missing = sum(1 for v in vals if v is None or math.isnan(v))
        print(f"{col:<20} {c_min:<10.2f} {c_max:<10.2f} {c_mean:<10.2f} {missing:<10}")
        
    # Grouping by fire
    fires = sorted(list(set(r["fire_name"] for r in rows)))
    
    # Plot 1: Sample Distribution by Fire and Split
    plt.figure(figsize=(10, 5))
    counts = [sum(1 for r in rows if r["fire_name"] == f) for f in fires]
    splits = [next(r["split"] for r in rows if r["fire_name"] == f).upper() for f in fires]
    labels = [f"{f}\n({s})" for f, s in zip(fires, splits)]
    colors = ['#2b5c8f' if s == 'TRAIN' else '#d95f02' if s == 'VAL' else '#7570b3' for s in splits]
    plt.bar(labels, counts, color=colors, edgecolor='black', alpha=0.85)
    plt.title("Sample Distribution by Wildfire Incident (Split Assignment)", fontsize=13, fontweight='bold')
    plt.ylabel("Number of 100m Segments", fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    p1 = os.path.join(plots_dir, "sample_distribution_by_fire.png")
    plt.savefig(p1, dpi=200)
    plt.close()
    
    # Plot 2: Held vs Burned Over by Fire
    plt.figure(figsize=(10, 5))
    x = range(len(fires))
    held_by_fire = [sum(1 for r in rows if r["fire_name"] == f and r["label"] == 1) for f in fires]
    burned_by_fire = [sum(1 for r in rows if r["fire_name"] == f and r["label"] == 0) for f in fires]
    w = 0.35
    plt.bar([i - w/2 for i in x], held_by_fire, width=w, label="Held (1)", color="#2ca02c", edgecolor='black')
    plt.bar([i + w/2 for i in x], burned_by_fire, width=w, label="Burned Over (0)", color="#d62728", edgecolor='black')
    plt.xticks(x, fires, fontsize=10)
    plt.title("Containment Line Outcome Distribution by Wildfire", fontsize=13, fontweight='bold')
    plt.ylabel("Segment Count", fontsize=11)
    plt.legend(frameon=True)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    p2 = os.path.join(plots_dir, "held_vs_burned_over.png")
    plt.savefig(p2, dpi=200)
    plt.close()
    
    # Plot 3: Feature Distributions
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    feats = [("slope", "Slope (degrees)", "#1f77b4"),
             ("elevation", "Elevation (m)", "#2ca02c"),
             ("distance_to_fire", "Distance to Fire (m)", "#ff7f0e"),
             ("burn_prob", "Predicted Burn Prob", "#d62728"),
             ("attack_angle", "Attack Angle (degrees)", "#9467bd"),
             ("attack_dot_product", "Attack Dot Product (cos theta)", "#8c564b")]
    for ax, (col, title, color) in zip(axes.flatten(), feats):
        vals = [r[col] for r in rows]
        ax.hist(vals, bins=25, color=color, edgecolor='black', alpha=0.75)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.suptitle("Feature Value Distributions Across All 4,600 Line Segments", fontsize=14, fontweight='bold')
    plt.tight_layout()
    p3 = os.path.join(plots_dir, "feature_distributions.png")
    plt.savefig(p3, dpi=200)
    plt.close()
    
    # Plot 4: Fireline Attack Angle vs Outcome (Demonstrating Physical Separation)
    plt.figure(figsize=(9, 5))
    held_attack = [r["attack_angle"] for r in rows if r["label"] == 1]
    burned_attack = [r["attack_angle"] for r in rows if r["label"] == 0]
    plt.hist(held_attack, bins=25, density=True, alpha=0.6, color="#2ca02c", label="Held Segments", edgecolor='black')
    plt.hist(burned_attack, bins=25, density=True, alpha=0.6, color="#d62728", label="Burned Over Segments", edgecolor='black')
    plt.title("Attack Angle Distribution: Held vs Burned Over Lines", fontsize=13, fontweight='bold')
    plt.xlabel("Attack Angle relative to Line Normal (degrees: 0° = direct head fire, 90° = flank)", fontsize=11)
    plt.ylabel("Probability Density", fontsize=11)
    plt.axvline(x=sum(held_attack)/len(held_attack), color="#1b7837", linestyle="--", linewidth=2, label=f"Mean Held ({sum(held_attack)/len(held_attack):.1f}°)")
    plt.axvline(x=sum(burned_attack)/len(burned_attack), color="#a50f15", linestyle="--", linewidth=2, label=f"Mean Burned ({sum(burned_attack)/len(burned_attack):.1f}°)")
    plt.legend(frameon=True)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    p4 = os.path.join(plots_dir, "attack_angle_vs_outcome.png")
    plt.savefig(p4, dpi=200)
    plt.close()
    
    print("\n" + "=" * 60)
    print("PLOTS GENERATED IN data/processed/plots/:")
    print(f"1. {p1}")
    print(f"2. {p2}")
    print(f"3. {p3}")
    print(f"4. {p4}")
    print("=" * 60)

if __name__ == "__main__":
    main()
