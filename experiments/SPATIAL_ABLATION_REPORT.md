# SPATIAL NEIGHBORHOOD ABLATION REPORT

> **SYNTHETIC DATA. NOT EMPIRICAL EVIDENCE.** `fireline_segments_v1` is simulated by `src/data/build_dataset.py` (random terrain/geometry and a hand-written logistic label rule). Fire names and incident IDs are scenario labels only; no data from those incidents is used. These results test the CLBI software pipeline, not real fireline behaviour, and are outside the Phase 1 NDWS evaluation (see `docs/CLBI_SYNTHETIC_DATASET_CARD.md`).

**Project:** Candidate Line Breach Intelligence (CLBI)  
**Date:** 2026-09-30  
**Evaluation Protocol:** 5-fold Leave-One-Fire-Out (LOFO) Cross-Validation  
**Hypothesis:** *A fireline segment's outcome depends partly on the conditions of neighboring segments, and adding local spatial-neighborhood information improves prediction beyond the isolated 7-feature segment model.*

---

## 1. Geometry & Spatial Adjacency Verification

- **Segment Sequence:** Inspected `src/data/build_dataset.py`. `segment_id` is generated sequentially as `[fire_id]_seg_00001`, `00002`, ..., `n` for each fire.
- **Line Length:** Uniform 100-meter straight line segments.
- **Topology:** Continuous 1D polyline chains. For segment $i$ within a fire incident, immediate spatial neighbors are segment $i-1$ (left) and segment $i+1$ (right).
- **Boundary Endpoint Handling:** Endpoint segments ($i=1$ and $i=n$) have only 1 neighbor. Neighborhood statistics are computed over the single available adjacent neighbor (one-sided aggregation). **No artificial zero-padding was used.**

---

## 2. Feature Provenance & Critical Leakage Audit

Strict pre-engagement aggregation was enforced. All 8 neighborhood features were calculated **exclusively from pre-engagement segment attributes**:

| Engineered Neighborhood Feature | Base Attribute | Aggregation | Leakage Audit |
| :--- | :--- | :--- | :--- |
| `nbr_burn_prob_mean` | `burn_prob` | Mean of $i-1, i+1$ | **SAFE (Pre-engagement ML)** |
| `nbr_burn_prob_max` | `burn_prob` | Max of $i-1, i+1$ | **SAFE (Pre-engagement ML)** |
| `nbr_attack_dot_mean` | `attack_dot_product` | Mean of $i-1, i+1$ | **SAFE (Pre-engagement Geometry)** |
| `nbr_attack_dot_max` | `attack_dot_product` | Max of $i-1, i+1$ | **SAFE (Pre-engagement Geometry)** |
| `nbr_attack_angle_mean` | `attack_angle` | Mean of $i-1, i+1$ | **SAFE (Pre-engagement Geometry)** |
| `nbr_slope_mean` | `slope` | Mean of $i-1, i+1$ | **SAFE (Static Topography)** |
| `nbr_slope_max` | `slope` | Max of $i-1, i+1$ | **SAFE (Static Topography)** |
| `nbr_barrier_width_min` | `barrier_width_m` | Min of $i-1, i+1$ | **SAFE (Operational Spec)** |

**Zero Outcome Leakage:** Ground-truth breach labels (`label`, `outcome_str`) of neighbors were **strictly excluded**.

---

## 3. Leave-One-Fire-Out (LOFO) Per-Fire Results

| Held-Out Test Fire | Incident ID | Model A (7-Feat Base LR) | Model B (Spatial-Enriched LR) | Model C (Threat-Context LR) | Δ PR-AUC (Model B vs A) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **August Complex** | `CA-MNF-013028` | 0.9526 | **0.9522** | 0.9526 | **-0.0004** |
| **Carr Fire** | `CA-SHU-007808` | 0.9455 | **0.9453** | 0.9453 | **-0.0002** |
| **Creek Fire** | `CA-SNF-000958` | 0.9413 | **0.9411** | 0.9413 | **-0.0002** |
| **Mendocino Complex**| `CA-MEU-008674` | 0.9619 | **0.9613** | 0.9617 | **-0.0006** |
| **CZU Lightning** | `CA-CZU-005205` | 0.9527 | **0.9524** | 0.9527 | **-0.0003** |

---

## 4. Aggregate Cross-Fire Statistics

| Model | Mean PR-AUC ± Std | Median PR-AUC | Min PR-AUC | Max PR-AUC | Mean ROC-AUC | Mean Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A: Current Base LR** | 0.9508 ± 0.0079 | 0.9526 | 0.9413 | 0.9619 | 0.8963 | 0.1205 |
| **Model B: Spatial-Enriched** | **0.9505 ± 0.0077** | **0.9522** | **0.9411** | **0.9613** | **0.8953** | **0.1208** |
| **Model C: Threat-Context** | 0.9507 ± 0.0078 | 0.9526 | 0.9413 | 0.9617 | 0.8961 | 0.1208 |

---

## 5. Empirical Findings & Spatial Signal Test

1. **Number of Fires Improved:** **0 / 5 FIRES**
   - Spatial neighborhood enrichment improved PR-AUC on **0 of 5 held-out fires**.
2. **Average Performance Change:** **-0.0003 PR-AUC** (ROC-AUC change: **-0.0010**).
3. **Signal Strength Classification:** **NO EVIDENCE**

---

## 6. GNN Justification Verdict

### **Action Status:** 🔴 STOP

> **Is there sufficient empirical evidence to justify implementing a Graph Neural Network?**

### **Answer:** **NO**

### **Justification:**
1. Neighbour aggregates improved PR-AUC on only 0/5 held-out fires (mean change -0.0003).
2. Fixed neighbourhood pooling adds no measurable signal, so there is no empirical case for a GNN on this data.
3. Expected here: in the synthetic generator (`src/data/build_dataset.py`) each segment's label depends only on its own features.
