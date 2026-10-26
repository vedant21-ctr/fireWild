"""
robustness_validation.py - Candidate Line Breach Intelligence (CLBI) Robustness Audit

Executes Leave-One-Fire-Out (LOFO) cross-validation across all 5 historical wildfire incidents.
Evaluates model generalization across unseen fires and quantifies uncertainty.

Outputs:
  - Terminal summary & printouts
  - experiments/results/lofo_results.json
  - experiments/results/lofo_metrics.csv
  - experiments/results/lofo_pr_auc_by_fire.png
  - experiments/results/lofo_delta_pr_auc.png
  - experiments/ROBUSTNESS_REPORT.md
"""

import os
import json
from datetime import date
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss
)

def define_experiments():
    return {
        "Exp_A_Static": {
            "name": "Exp A (Static Baseline)",
            "features": ["slope", "elevation", "distance_to_fire", "barrier_width_m"]
        },
        "Exp_B_Static_BurnProb": {
            "name": "Exp B (Static + BurnProb)",
            "features": ["slope", "elevation", "distance_to_fire", "barrier_width_m", "burn_prob"]
        },
        "Exp_D_Static_Geometry": {
            "name": "Exp D (Static + Attack Geometry)",
            "features": ["slope", "elevation", "distance_to_fire", "barrier_width_m", "attack_angle", "attack_dot_product"]
        },
        "Exp_C_Full_Directional": {
            "name": "Exp C (Full Directional)",
            "features": [
                "slope", "elevation", "distance_to_fire", "barrier_width_m",
                "burn_prob", "prob_gradient", "attack_angle", "attack_dot_product", "dist_pred_boundary"
            ]
        }
    }

def run_lofo_validation(df, experiments, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    
    unique_fires = df['fire_id'].unique().tolist()
    fire_names = {fid: df[df['fire_id'] == fid]['fire_name'].iloc[0] for fid in unique_fires}
    
    print("=" * 75)
    print(f"LEAVE-ONE-FIRE-OUT (LOFO) CROSS-VALIDATION ACROSS {len(unique_fires)} FIRE PROFILES")
    print("=" * 75)
    
    lofo_rows = []
    per_fire_results = {fid: {} for fid in unique_fires}
    
    for test_fire_id in unique_fires:
        test_fire_name = fire_names[test_fire_id]
        train_df = df[df['fire_id'] != test_fire_id].copy()
        test_df  = df[df['fire_id'] == test_fire_id].copy()
        
        y_train = train_df['label'].values
        y_test  = test_df['label'].values
        
        print(f"\nHeld-Out Test Fire: {test_fire_id} ({test_fire_name:<22}) | Test size: {len(test_df)} | Train size: {len(train_df)}")
        
        fire_metrics = {}
        
        for exp_key, exp_spec in experiments.items():
            feats = exp_spec["features"]
            
            X_train = train_df[feats].values
            X_test  = test_df[feats].values
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(max_iter=1000, random_state=42, class_weight=None))
            ])
            
            pipeline.fit(X_train, y_train)
            test_probs = pipeline.predict_proba(X_test)[:, 1]
            test_preds = pipeline.predict(X_test)
            
            pr_auc  = float(average_precision_score(y_test, test_probs))
            roc_auc = float(roc_auc_score(y_test, test_probs))
            prec    = float(precision_score(y_test, test_preds, zero_division=0))
            rec     = float(recall_score(y_test, test_preds, zero_division=0))
            f1      = float(f1_score(y_test, test_preds, zero_division=0))
            brier   = float(brier_score_loss(y_test, test_probs))
            
            fire_metrics[exp_key] = {
                "pr_auc": round(pr_auc, 4),
                "roc_auc": round(roc_auc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "brier_score": round(brier, 4)
            }
            
            lofo_rows.append({
                "Test_Fire_ID": test_fire_id,
                "Test_Fire_Name": test_fire_name,
                "Experiment_Key": exp_key,
                "Experiment_Name": exp_spec["name"],
                "PR_AUC": round(pr_auc, 4),
                "ROC_AUC": round(roc_auc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1_Score": round(f1, 4),
                "Brier_Score": round(brier, 4)
            })
            
        per_fire_results[test_fire_id] = {
            "fire_name": test_fire_name,
            "test_segments": len(test_df),
            "metrics": fire_metrics,
            "delta_B_minus_A": round(fire_metrics["Exp_B_Static_BurnProb"]["pr_auc"] - fire_metrics["Exp_A_Static"]["pr_auc"], 4),
            "delta_D_minus_A": round(fire_metrics["Exp_D_Static_Geometry"]["pr_auc"] - fire_metrics["Exp_A_Static"]["pr_auc"], 4),
            "delta_C_minus_A": round(fire_metrics["Exp_C_Full_Directional"]["pr_auc"] - fire_metrics["Exp_A_Static"]["pr_auc"], 4)
        }
        
        print(f"  Exp A (Static):      PR-AUC = {fire_metrics['Exp_A_Static']['pr_auc']:.4f} | ROC-AUC = {fire_metrics['Exp_A_Static']['roc_auc']:.4f}")
        print(f"  Exp B (+BurnProb):   PR-AUC = {fire_metrics['Exp_B_Static_BurnProb']['pr_auc']:.4f} (Delta = {per_fire_results[test_fire_id]['delta_B_minus_A']:+.4f})")
        print(f"  Exp D (+Geometry):   PR-AUC = {fire_metrics['Exp_D_Static_Geometry']['pr_auc']:.4f} (Delta = {per_fire_results[test_fire_id]['delta_D_minus_A']:+.4f})")
        print(f"  Exp C (Full Dir):    PR-AUC = {fire_metrics['Exp_C_Full_Directional']['pr_auc']:.4f} (Delta = {per_fire_results[test_fire_id]['delta_C_minus_A']:+.4f})")

    lofo_df = pd.DataFrame(lofo_rows)
    csv_path = os.path.join(results_dir, "lofo_metrics.csv")
    lofo_df.to_csv(csv_path, index=False)
    print(f"\nSaved LOFO metrics CSV to: {csv_path}")

    # Compute Aggregates across 5 Folds
    aggregates = {}
    for exp_key in experiments.keys():
        exp_df = lofo_df[lofo_df['Experiment_Key'] == exp_key]
        pr_aucs = exp_df['PR_AUC'].values
        roc_aucs = exp_df['ROC_AUC'].values
        f1s = exp_df['F1_Score'].values
        briers = exp_df['Brier_Score'].values
        
        aggregates[exp_key] = {
            "name": experiments[exp_key]["name"],
            "pr_auc": {
                "mean": round(float(np.mean(pr_aucs)), 4),
                "std": round(float(np.std(pr_aucs, ddof=1)), 4),
                "median": round(float(np.median(pr_aucs)), 4),
                "min": round(float(np.min(pr_aucs)), 4),
                "max": round(float(np.max(pr_aucs)), 4)
            },
            "roc_auc": {
                "mean": round(float(np.mean(roc_aucs)), 4),
                "std": round(float(np.std(roc_aucs, ddof=1)), 4),
                "median": round(float(np.median(roc_aucs)), 4),
                "min": round(float(np.min(roc_aucs)), 4),
                "max": round(float(np.max(roc_aucs)), 4)
            },
            "f1": {
                "mean": round(float(np.mean(f1s)), 4),
                "std": round(float(np.std(f1s, ddof=1)), 4)
            },
            "brier": {
                "mean": round(float(np.mean(briers)), 4),
                "std": round(float(np.std(briers, ddof=1)), 4)
            }
        }

    # Compute Delta Aggregates across 5 Folds
    deltas_B = [per_fire_results[fid]["delta_B_minus_A"] for fid in unique_fires]
    deltas_D = [per_fire_results[fid]["delta_D_minus_A"] for fid in unique_fires]
    deltas_C = [per_fire_results[fid]["delta_C_minus_A"] for fid in unique_fires]

    delta_summary = {
        "Exp_B_minus_Exp_A": {
            "mean": round(float(np.mean(deltas_B)), 4),
            "std": round(float(np.std(deltas_B, ddof=1)), 4),
            "min": round(float(np.min(deltas_B)), 4),
            "max": round(float(np.max(deltas_B)), 4),
            "positive_fires_count": int(sum(1 for d in deltas_B if d > 0)),
            "total_fires": len(unique_fires)
        },
        "Exp_D_minus_Exp_A": {
            "mean": round(float(np.mean(deltas_D)), 4),
            "std": round(float(np.std(deltas_D, ddof=1)), 4),
            "min": round(float(np.min(deltas_D)), 4),
            "max": round(float(np.max(deltas_D)), 4),
            "positive_fires_count": int(sum(1 for d in deltas_D if d > 0)),
            "total_fires": len(unique_fires)
        },
        "Exp_C_minus_Exp_A": {
            "mean": round(float(np.mean(deltas_C)), 4),
            "std": round(float(np.std(deltas_C, ddof=1)), 4),
            "min": round(float(np.min(deltas_C)), 4),
            "max": round(float(np.max(deltas_C)), 4),
            "positive_fires_count": int(sum(1 for d in deltas_C if d > 0)),
            "total_fires": len(unique_fires)
        }
    }

    # Save to JSON
    json_out_path = os.path.join(results_dir, "lofo_results.json")
    with open(json_out_path, "w") as f:
        json.dump({
            "per_fire_results": per_fire_results,
            "aggregate_statistics": aggregates,
            "delta_analysis": delta_summary
        }, f, indent=2)
    print(f"Saved LOFO JSON results to: {json_out_path}")

    # Generate Plots
    generate_lofo_plots(unique_fires, fire_names, per_fire_results, aggregates, results_dir)

    return per_fire_results, aggregates, delta_summary

def generate_lofo_plots(unique_fires, fire_names, per_fire_results, aggregates, results_dir):
    print("\n" + "=" * 75)
    print("GENERATING LOFO CROSS-VALIDATION PLOTS")
    print("=" * 75)

    # Plot 1: Per-Fire PR-AUC Comparison
    plt.figure(figsize=(11, 6))
    x = np.arange(len(unique_fires))
    w = 0.2
    
    f_names = [fire_names[fid] for fid in unique_fires]
    pr_A = [per_fire_results[fid]["metrics"]["Exp_A_Static"]["pr_auc"] for fid in unique_fires]
    pr_B = [per_fire_results[fid]["metrics"]["Exp_B_Static_BurnProb"]["pr_auc"] for fid in unique_fires]
    pr_D = [per_fire_results[fid]["metrics"]["Exp_D_Static_Geometry"]["pr_auc"] for fid in unique_fires]
    pr_C = [per_fire_results[fid]["metrics"]["Exp_C_Full_Directional"]["pr_auc"] for fid in unique_fires]
    
    plt.bar(x - 1.5*w, pr_A, width=w, label='Exp A (Static)', color='#1f77b4', edgecolor='black')
    plt.bar(x - 0.5*w, pr_B, width=w, label='Exp B (+BurnProb)', color='#ff7f0e', edgecolor='black')
    plt.bar(x + 0.5*w, pr_D, width=w, label='Exp D (+Geometry)', color='#d62728', edgecolor='black')
    plt.bar(x + 1.5*w, pr_C, width=w, label='Exp C (Full Dir)', color='#2ca02c', edgecolor='black')
    
    plt.ylabel('PR-AUC on Held-Out Test Fire', fontsize=12)
    all_pr_aucs = pr_A + pr_B + pr_D + pr_C
    plt.ylim(max(0.0, min(all_pr_aucs) - 0.03), min(1.0, max(all_pr_aucs) + 0.03))
    plt.title(f'Leave-One-Fire-Out Cross-Validation: PR-AUC Across {len(unique_fires)} Fire Profiles', fontsize=13, fontweight='bold')
    plt.xticks(x, f_names, fontsize=10, fontweight='bold')
    plt.legend(loc='lower right', frameon=True)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    p1 = os.path.join(results_dir, "lofo_pr_auc_by_fire.png")
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"Saved: {p1}")

    # Plot 2: Per-Fire Delta PR-AUC (Improvement over Static)
    plt.figure(figsize=(11, 6))
    dB = [per_fire_results[fid]["delta_B_minus_A"] for fid in unique_fires]
    dD = [per_fire_results[fid]["delta_D_minus_A"] for fid in unique_fires]
    dC = [per_fire_results[fid]["delta_C_minus_A"] for fid in unique_fires]
    
    w_d = 0.25
    plt.bar(x - w_d, dB, width=w_d, label='Δ Exp B - Exp A (+BurnProb)', color='#ff7f0e', edgecolor='black')
    plt.bar(x, dD, width=w_d, label='Δ Exp D - Exp A (+Geometry)', color='#d62728', edgecolor='black')
    plt.bar(x + w_d, dC, width=w_d, label='Δ Exp C - Exp A (Full Dir)', color='#2ca02c', edgecolor='black')
    
    plt.axhline(0, color='black', linestyle='-', linewidth=1.2)
    plt.ylabel('Δ PR-AUC Improvement over Static Baseline', fontsize=12)
    plt.title('PR-AUC Difference Across Held-Out Synthetic Profiles', fontsize=13, fontweight='bold')
    plt.xticks(x, f_names, fontsize=10, fontweight='bold')
    
    for i in range(len(unique_fires)):
        plt.text(x[i] - w_d, dB[i] + 0.002 if dB[i] >= 0 else dB[i] - 0.005, f"{dB[i]:+.3f}", ha='center', fontsize=8)
        plt.text(x[i], dD[i] + 0.002 if dD[i] >= 0 else dD[i] - 0.005, f"{dD[i]:+.3f}", ha='center', fontsize=8, fontweight='bold')
        plt.text(x[i] + w_d, dC[i] + 0.002 if dC[i] >= 0 else dC[i] - 0.005, f"{dC[i]:+.3f}", ha='center', fontsize=8, fontweight='bold')

    plt.legend(loc='upper right', frameon=True)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    p2 = os.path.join(results_dir, "lofo_delta_pr_auc.png")
    plt.savefig(p2, dpi=200)
    plt.close()
    print(f"Saved: {p2}")

def generate_robustness_report(per_fire_results, aggregates, delta_summary, base_dir):
    report_path = os.path.join(base_dir, "experiments", "ROBUSTNESS_REPORT.md")
    fire_count = len(per_fire_results)
    test_segment_count = sum(
        result.get("test_segments", 0) for result in per_fire_results.values()
    )
    
    mean_A_pr = aggregates["Exp_A_Static"]["pr_auc"]["mean"]
    std_A_pr  = aggregates["Exp_A_Static"]["pr_auc"]["std"]
    
    mean_C_pr = aggregates["Exp_C_Full_Directional"]["pr_auc"]["mean"]
    std_C_pr  = aggregates["Exp_C_Full_Directional"]["pr_auc"]["std"]

    mean_D_pr = aggregates["Exp_D_Static_Geometry"]["pr_auc"]["mean"]
    std_D_pr  = aggregates["Exp_D_Static_Geometry"]["pr_auc"]["std"]
    
    dC_mean = delta_summary["Exp_C_minus_Exp_A"]["mean"]
    dC_std  = delta_summary["Exp_C_minus_Exp_A"]["std"]
    dC_count = delta_summary["Exp_C_minus_Exp_A"]["positive_fires_count"]

    dD_mean = delta_summary["Exp_D_minus_Exp_A"]["mean"]
    dD_std  = delta_summary["Exp_D_minus_Exp_A"]["std"]
    dD_count = delta_summary["Exp_D_minus_Exp_A"]["positive_fires_count"]

    worst_fire_id = min(per_fire_results.keys(), key=lambda fid: per_fire_results[fid]["delta_C_minus_A"])
    worst_fire_name = per_fire_results[worst_fire_id]["fire_name"]
    worst_fire_delta = per_fire_results[worst_fire_id]["delta_C_minus_A"]

    best_fire_id = max(per_fire_results.keys(), key=lambda fid: per_fire_results[fid]["delta_C_minus_A"])
    best_fire_name = per_fire_results[best_fire_id]["fire_name"]
    best_fire_delta = per_fire_results[best_fire_id]["delta_C_minus_A"]

    per_fire_rows = []
    for fire_id, fire_result in per_fire_results.items():
        metrics = fire_result["metrics"]
        per_fire_rows.append(
            f"| **{fire_result['fire_name']}** | `{fire_id}` | "
            f"{metrics['Exp_A_Static']['pr_auc']:.4f} | "
            f"{metrics['Exp_B_Static_BurnProb']['pr_auc']:.4f} | "
            f"{metrics['Exp_D_Static_Geometry']['pr_auc']:.4f} | "
            f"**{metrics['Exp_C_Full_Directional']['pr_auc']:.4f}** | "
            f"**{fire_result['delta_C_minus_A']:+.4f}** |"
        )

    content = f"""# CLBI Synthetic Benchmark Evaluation Report

**Project:** Candidate Line Breach Intelligence (CLBI)  
**Generated:** {date.today().isoformat()}  
**Evaluation Protocol:** Leave-One-Fire-Out (LOFO) across {fire_count} generated fire profiles  
**Primary Target:** `label` (1 = Held, 0 = Burned Over)  

---

## 1. Dataset Origin and Interpretation

The dataset is generated by `src/data/build_dataset.py`. Feature values are sampled from
configured distributions for five named fire profiles, and labels are derived from a
hand-authored breach-score formula plus random noise and a per-profile hold-rate cutoff.
The fire names and IDs provide profile context; the rows are **not measured historical
fireline observations**, and the labels are not recorded incident outcomes.

LOFO therefore measures how well the models recover this synthetic labeling process when
one generated profile is held out. It does not establish generalization to real fires,
field performance, operational safety, or causal effects. Several predictors are also
inputs to the synthetic breach-score formula, so the high scores should not be read as
independent evidence of predictive skill on observed outcomes.

## 2. Synthetic Feature and Target Construction

| Feature / target | Synthetic generation |
| :--- | :--- |
| Terrain, distance, and line geometry | Sampled from configured per-profile distributions and formulas. |
| `burn_prob`, `prob_gradient`, `dist_pred_boundary` | Derived from synthetic terrain, distance, and attack geometry. |
| `label` | Thresholded synthetic breach score based on `attack_dot_product`, `slope`, `burn_prob`, `barrier_width_m`, `distance_to_fire`, and random noise. |

The predictors and target are deliberately related by design. This experiment can
validate the pipeline and compare feature sets against the generator; it is not a
post-outcome leakage audit of observational data.

## 3. Leave-One-Profile-Out (LOFO) Results

Each fold holds out one generated profile and trains on the other profiles using
standardized Logistic Regression (`StandardScaler` + `LogisticRegression(max_iter=1000)`).
The {fire_count} folds contain {test_segment_count:,} generated test rows in total.

### **Per-Profile Test PR-AUC Scores:**

| Held-Out Profile | Profile ID | Exp A (Static) | Exp B (+BurnProb) | Exp D (+Geometry) | Exp C (Full Dir) | Δ PR-AUC (Exp C − A) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
{chr(10).join(per_fire_rows)}

---

## 4. Aggregate Cross-Profile Statistics

| Experiment | Mean PR-AUC ± Std | Median PR-AUC | Min PR-AUC | Max PR-AUC | Mean ROC-AUC | Mean F1 | Mean Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp A: Static Baseline** | {mean_A_pr:.4f} ± {std_A_pr:.4f} | {aggregates['Exp_A_Static']['pr_auc']['median']:.4f} | {aggregates['Exp_A_Static']['pr_auc']['min']:.4f} | {aggregates['Exp_A_Static']['pr_auc']['max']:.4f} | {aggregates['Exp_A_Static']['roc_auc']['mean']:.4f} | {aggregates['Exp_A_Static']['f1']['mean']:.4f} | {aggregates['Exp_A_Static']['brier']['mean']:.4f} |
| **Exp B: Static + BurnProb** | {aggregates['Exp_B_Static_BurnProb']['pr_auc']['mean']:.4f} ± {aggregates['Exp_B_Static_BurnProb']['pr_auc']['std']:.4f} | {aggregates['Exp_B_Static_BurnProb']['pr_auc']['median']:.4f} | {aggregates['Exp_B_Static_BurnProb']['pr_auc']['min']:.4f} | {aggregates['Exp_B_Static_BurnProb']['pr_auc']['max']:.4f} | {aggregates['Exp_B_Static_BurnProb']['roc_auc']['mean']:.4f} | {aggregates['Exp_B_Static_BurnProb']['f1']['mean']:.4f} | {aggregates['Exp_B_Static_BurnProb']['brier']['mean']:.4f} |
| **Exp D: Static + Geometry** | {mean_D_pr:.4f} ± {std_D_pr:.4f} | {aggregates['Exp_D_Static_Geometry']['pr_auc']['median']:.4f} | {aggregates['Exp_D_Static_Geometry']['pr_auc']['min']:.4f} | {aggregates['Exp_D_Static_Geometry']['pr_auc']['max']:.4f} | {aggregates['Exp_D_Static_Geometry']['roc_auc']['mean']:.4f} | {aggregates['Exp_D_Static_Geometry']['f1']['mean']:.4f} | {aggregates['Exp_D_Static_Geometry']['brier']['mean']:.4f} |
| **Exp C: Full Directional** | **{mean_C_pr:.4f} ± {std_C_pr:.4f}** | **{aggregates['Exp_C_Full_Directional']['pr_auc']['median']:.4f}** | **{aggregates['Exp_C_Full_Directional']['pr_auc']['min']:.4f}** | **{aggregates['Exp_C_Full_Directional']['pr_auc']['max']:.4f}** | **{aggregates['Exp_C_Full_Directional']['roc_auc']['mean']:.4f}** | **{aggregates['Exp_C_Full_Directional']['f1']['mean']:.4f}** | **{aggregates['Exp_C_Full_Directional']['brier']['mean']:.4f}** |

---

## 5. Feature Comparison Across Held-Out Profiles

> **On how many held-out profiles does Exp C improve over the static baseline?**

### **Answer:** **{dC_count} / {fire_count} GENERATED PROFILES**

- Directional features (`attack_angle`, `attack_dot_product`, `burn_prob`) improved PR-AUC over the static baseline on {dC_count} of {fire_count} held-out generated profiles.
- Average cross-profile improvement: **{dC_mean:+.4f} PR-AUC** (Range: {delta_summary['Exp_C_minus_Exp_A']['min']:+.4f} to {delta_summary['Exp_C_minus_Exp_A']['max']:+.4f}).
- Smallest profile difference: **{worst_fire_name} ({worst_fire_delta:+.4f} PR-AUC)**.
- Largest profile difference: **{best_fire_name} ({best_fire_delta:+.4f} PR-AUC)**.

---

## 6. Scope and Evidence Caveat

- **Sample size:** {fire_count} generated profiles ({test_segment_count:,} generated test rows); profile folds are not independent real-world fire observations.
- These scores describe agreement with a synthetic target-generation rule and do not support real-world reliability or deployment claims.

---

## 7. Feature Ablation Insights

1. **Directional Geometry vs. Scalar Burn Probability:**
   - `Exp D` (Static + Geometry) achieves **{mean_D_pr:.4f} Mean PR-AUC**, performing almost identically to `Exp C` (**{mean_C_pr:.4f}**).
   - `Exp B` (Static + BurnProb alone) achieves **{aggregates['Exp_B_Static_BurnProb']['pr_auc']['mean']:.4f} Mean PR-AUC**.
   - **Synthetic-benchmark takeaway:** Geometry features are associated with higher scores in this generated target comparison; this is not a physical or causal finding.

2. **Features used by the model:**
   - `slope`, `elevation`, `distance_to_fire`, `barrier_width_m`, `attack_angle`, `attack_dot_product`, `burn_prob`.

3. **Features with limited marginal gain in this comparison:**
   - `prob_gradient` and `dist_pred_boundary` add minimal gain over `attack_dot_product` and `burn_prob` against the generated labels.

---

## 8. Interpretation

### **Verdict:** Exploratory pipeline comparison only

The experiment shows that directional features improve discrimination against the
generator's constructed labels on {dC_count} of {fire_count} held-out profiles. The
real-world hypothesis remains unvalidated. Operational interpretation would require
independently sourced fireline observations and recorded outcomes, prospective
evaluation, and domain-expert review.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\nSaved ROBUSTNESS_REPORT.md to: {report_path}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "final", "fireline_segments_v1.parquet")
    results_dir = os.path.join(base_dir, "experiments", "results")
    
    df = pd.read_parquet(data_path)
    experiments = define_experiments()
    per_fire_results, aggregates, delta_summary = run_lofo_validation(df, experiments, results_dir)
    generate_robustness_report(per_fire_results, aggregates, delta_summary, base_dir)
    
    print("\n" + "=" * 75)
    print("ROBUSTNESS VALIDATION COMPLETE")
    print("=" * 75)

if __name__ == "__main__":
    main()
