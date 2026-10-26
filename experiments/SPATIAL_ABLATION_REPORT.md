# SPATIAL NEIGHBORHOOD ABLATION REPORT

> **Evidence boundary:** This experiment uses generated features and labels from
> `src/data/build_dataset.py`. Sequential segment IDs are not spatial coordinates or
> verified line topology. The results compare aggregation across generated row order
> only; they do not test real spatial neighborhood effects.

**Project:** Candidate Line Breach Intelligence (CLBI)  
**Date:** 2026-09-30  
**Evaluation Protocol:** 5-fold leave-one-generated-profile-out (LOFO)  
**Hypothesis:** *A fireline segment's outcome depends partly on the conditions of neighboring segments, and adding local spatial-neighborhood information improves prediction beyond the isolated 7-feature segment model.*

---

## 1. Generated Row Order (Not Spatial Adjacency)

- **Segment Sequence:** `segment_id` increments as rows are generated for each profile.
- **Geometry:** The generator does not assign coordinates or construct continuous line segments.
- **Aggregation:** Neighbor features use adjacent generated rows ($i-1$ and $i+1$), not verified spatial neighbors.
- **Boundary Handling:** Endpoint rows have one adjacent row; no zero-padding is used.

---

## 2. Generated Neighborhood Features

All 8 neighborhood features were calculated from adjacent generated-row attributes. These are not pre-engagement observations:

| Engineered Neighborhood Feature | Base Attribute | Aggregation | Value source |
| :--- | :--- | :--- | :--- |
| `nbr_burn_prob_mean` | `burn_prob` | Mean of $i-1, i+1$ | Generated values |
| `nbr_burn_prob_max` | `burn_prob` | Max of $i-1, i+1$ | Generated values |
| `nbr_attack_dot_mean` | `attack_dot_product` | Mean of $i-1, i+1$ | Generated values |
| `nbr_attack_dot_max` | `attack_dot_product` | Max of $i-1, i+1$ | Generated values |
| `nbr_attack_angle_mean` | `attack_angle` | Mean of $i-1, i+1$ | Generated values |
| `nbr_slope_mean` | `slope` | Mean of $i-1, i+1$ | Generated values |
| `nbr_slope_max` | `slope` | Max of $i-1, i+1$ | Generated values |
| `nbr_barrier_width_min` | `barrier_width_m` | Min of $i-1, i+1$ | Generated values |

Neighbor target labels (`label`, `outcome_str`) are not inputs to the neighborhood
aggregations. However, the targets are synthetic and constructed from predictor
features; this is not validation on observed outcomes.

---

## 3. Leave-One-Profile-Out (LOFO) Results

| Held-Out Profile | Profile ID | Model A (7-Feat Base LR) | Model B (Spatial-Enriched LR) | Model C (Threat-Context LR) | Δ PR-AUC (Model B vs A) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **August Complex** | `CA-MNF-013028` | 0.9526 | **0.9522** | 0.9526 | **-0.0004** |
| **Carr Fire** | `CA-SHU-007808` | 0.9455 | **0.9453** | 0.9453 | **-0.0002** |
| **Creek Fire** | `CA-SNF-000958` | 0.9413 | **0.9411** | 0.9413 | **-0.0002** |
| **Mendocino Complex**| `CA-MEU-008674` | 0.9619 | **0.9613** | 0.9617 | **-0.0006** |
| **CZU Lightning** | `CA-CZU-005205` | 0.9527 | **0.9524** | 0.9527 | **-0.0003** |

---

## 4. Aggregate Cross-Profile Statistics

| Model | Mean PR-AUC ± Std | Median PR-AUC | Min PR-AUC | Max PR-AUC | Mean ROC-AUC | Mean Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A: Current Base LR** | 0.9508 ± 0.0079 | 0.9526 | 0.9413 | 0.9619 | 0.8963 | 0.1205 |
| **Model B: Spatial-Enriched** | **0.9505 ± 0.0077** | **0.9522** | **0.9411** | **0.9613** | **0.8953** | **0.1208** |
| **Model C: Threat-Context** | 0.9507 ± 0.0078 | 0.9526 | 0.9413 | 0.9617 | 0.8961 | 0.1208 |

---

## 5. Synthetic Row-Order Ablation Results

1. **Number of Profiles Improved:** **0 / 5 generated profiles**.
2. **Average Performance Difference:** **-0.0003 PR-AUC** (ROC-AUC difference: **-0.0010**).
3. **Signal Strength Classification:** **NO EVIDENCE**

---

## 6. GNN Evidence Boundary

### **Action Status:** No spatial conclusion

> **Is there sufficient empirical evidence to justify implementing a Graph Neural Network?**

### **Answer:** **Not from this synthetic row-order experiment**

### **Justification:**
1. The average PR-AUC difference is negative, and no generated profile improved.
2. The row order is not verified spatial topology. This ablation provides no evidence
   for or against real spatial neighborhood effects. Real geometry and outcomes are
   required before making a spatial-modeling decision.
