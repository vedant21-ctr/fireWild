# DATASET CARD: fireline_segments_v1

**Dataset Name:** Candidate Line Breach Intelligence Benchmark (v1)  
**Task Type:** Binary Classification / Spatial Line Survivability Prediction  
**Primary Target:** `label` (1 = Held, 0 = Burned Over)  
**File Formats Available:**  
- `data/final/fireline_segments_v1.parquet` (187 KB)  
- `data/final/fireline_segments_v1.csv` (749 KB)  
- `data/final/fireline_segments_v1.json` (2.78 MB)  

---

## 1. Provenance and Sources

This dataset connects the **Next Day Wildfire Spread (NDWS)** benchmark with the **USDA Forest Service Fireline Effectiveness (FLE)** empirical database:

1. **Next Day Wildfire Spread (NDWS):**
   - *Authors:* Fantine Huot, R. Lily Hu, Nita Goyal, Tharun Sankar, Matthias Ihme, Yi-Fan Chen (Google Research / Stanford).
   - *Publication:* *IEEE Transactions on Geoscience and Remote Sensing (TGRS)*, Vol. 60, 2022 (arXiv:2112.02447).
   - *Role in Pipeline:* Source of the deep learning fire spread velocity field, 10m synoptic wind vectors (GridMET), and regional topographic slope.
2. **USFS Fireline Effectiveness Dataset (FLE):**
   - *Authors:* Alexander P. Arkowitz, Scott M. Ritter, Matthew P. Thompson, Jesse D. Young, Bradley M. Pietruszka, David E. Calkin.
   - *Repository:* USDA Forest Service Research Data Archive (DOI: [`10.2737/RDS-2025-0011`](https://doi.org/10.2737/RDS-2025-0011)).
   - *Role in Pipeline:* Source of polyline geometries for constructed firelines and verified post-incident ground-truth engagement outcomes (`Held` vs. `Burned Over`).

---

## 2. Selected Fires & Geographic Scope

Five major historical California wildfire incidents (2018–2020) were selected based on high fireline engagement density, distinct fuel regimes, and documented extreme spread events:

| Fire Incident | Incident ID | Year | County / Forest | Split Role | Segment Count | Dominant Fuel |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- |
| **August Complex** | `CA-MNF-013028` | 2020 | Mendocino NF, Glenn/Tehama | `train` | 1,420 | Brush / Mixed Conifer |
| **Carr Fire** | `CA-SHU-007808` | 2018 | Shasta County, Whiskeytown | `train` | 680 | Timber / Heavy Brush |
| **Creek Fire** | `CA-SNF-000958` | 2020 | Sierra NF, Fresno/Madera | `train` | 950 | Heavy Timber / Snags |
| **Mendocino Complex**| `CA-MEU-008674`| 2018 | Mendocino/Lake Counties | `val` | 810 | Grass / Oak Chaparral |
| **CZU Lightning Complex**| `CA-CZU-005205`| 2020 | Santa Cruz/San Mateo | `test` | 740 | Coastal Redwood/Timber |

---

## 3. Discretization & Spatial Geometry

- **Segment Length:** Every containment line is discretized into uniform **100-meter straight-line segments**.
- **Coordinate Reference System (Standardized):** `EPSG:5070` (USA Contiguous Albers Equal Area Conic).
  - *Rationale:* Preserves true surface distance in meters and area without distortion across CONUS, enabling exact angular dot-product calculations.
- **Normal Vector Calculation:** For each 100m segment running between $(x_1, y_1)$ and $(x_2, y_2)$ with azimuth $\theta_{\text{line}} \in [0, 180)^\circ$, the unit normal vector is:
  $$\hat{n} = (-\sin(\theta_{\text{line}}), \cos(\theta_{\text{line}}))$$

---

## 4. Target Label Construction

- **Positive Class (`label = 1`):** `Held` — The fireline successfully stopped the flame front; the fire did not cross the constructed barrier.
- **Negative Class (`label = 0`):** `Burned Over` — The fire breached, spotted over, or overrun the barrier segment.
- **Excluded Class:** `Not Engaged` — Firelines constructed miles from the active perimeter that fire never reached were **strictly filtered out**. Including unengaged lines would severely bias the model by treating untouched lines as successful barriers.
- **Class Balance:**
  - `Held (1)`: 3,223 samples (**70.07%**)
  - `Burned Over (0)`: 1,377 samples (**29.93%**)
  - *Imbalance Ratio:* 2.34 : 1 (matches empirical USFS FLE field ratio).

---

## 5. Feature Schema & Definitions

| Column Name | Category | Type | Units / Range | Mathematical Definition | Physical Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `fire_id` | Metadata | String | Categorical | Official NIFC / IRWIN Incident Identifier | Tracking and group-splitting. |
| `segment_id` | Metadata | String | Unique Key | Unique identifier per 100m segment | Primary key. |
| `date` | Temporal | Date | YYYY-MM-DD | Date of observation / operational period | Temporal alignment. |
| `split` | Protocol | String | `{train, val, test}` | Strict fire-level allocation | Prevents spatial leakage. |
| `label` | Target | Integer | `{0, 1}` | Binary outcome: 1 = Held, 0 = Burned Over | Ground truth prediction target. |
| `slope` | Static | Float | Degrees ($[1^\circ, 55^\circ]$) | $\arctan(\|\nabla z\|) \times \frac{180}{\pi}$ | Steep slopes accelerate flame tilt. |
| `elevation` | Static | Float | Meters ($[50, 3200]$) | Elevation $z$ from USGS 3DEP | Fuel moisture and nocturnal inversion. |
| `distance_to_fire`| Static | Float | Meters ($[50, 5000]$) | Euclidean distance to active front at $t$ | Proximity of flame attack. |
| `burn_prob` | Spread ML | Float | Probability ($[0, 1]$) | $\sigma(\mathbf{W} \cdot \mathbf{x})$ from 2D spatial backbone | Predicted intensity of oncoming front. |
| `prob_gradient`| Spread ML | Float | Magnitude ($[0, 0.5]$) | $\|\nabla P_{\text{burn}}\|$ | Sharpness / velocity of the fire wave. |
| `spread_direction`| Spread ML | Float | Degrees ($[0, 360)^\circ$) | Azimuth of vector $\vec{J}_{\text{fire}}$ | Heading of the oncoming flame front. |
| `line_angle` | Geometry | Float | Degrees ($[0, 180)^\circ$) | Segment bearing: $\arctan2(\Delta x, \Delta y)$ | Spatial alignment of the physical line. |
| `attack_angle` | Geometry | Float | Degrees ($[0^\circ, 90^\circ]$) | Acute angle between $\vec{J}_{\text{fire}}$ and $\hat{n}_{\text{line}}$ | Directness of head-fire attack. |
| `attack_dot_product`| Geometry | Float | Ratio ($[0, 1]$) | $\cos(\theta_{\text{attack}}) = \frac{\vec{J}_{\text{fire}} \cdot \hat{n}}{\|\vec{J}_{\text{fire}}\|}$ | 1.0 = direct head fire; 0.0 = flank fire. |
| `dist_pred_boundary`| Spread ML | Float | Meters | Distance to predicted $P=0.5$ contour | Estimated buffer clearance before arrival. |
| `line_type` | Operational | String | Categorical | `Dozer Line`, `Road as Line`, `Hand Line` | Equipment construction type. |
| `barrier_width_m` | Operational | Float | Meters ($[2.0, 9.0]$) | Physical cleared width of scrape | Wider barriers resist spotting. |
| `fuel_model_group` | Environmental | String | Categorical | `Brush`, `Timber`, `Grass/Brush` | Dominant fuel carrying the fire. |

---

## 6. Critical Leakage Audit

To prevent subtle temporal or spatial leakage:

1. **No Post-Event Perimeter Features:** Fire perimeter features are sampled strictly from day $t$ (before engagement). Next-day burn perimeters ($t+1$) are never used as inputs.
2. **No Post-Incident Suppression Data:** Attributes that record post-incident mop-up or suppression repairs are excluded.
3. **Strict Fire-Level Grouping:** No individual segment from the August Complex or Creek Fire is allowed in the test set. The model is trained on 3 fires, validated on 1 fire, and tested on a completely unseen 5th fire (CZU Complex).

---

## 7. Known Limitations

1. **Tactical Resource Absence:** The dataset records where lines held or breached, but does not record whether air tankers dropped retardant on that specific segment.
2. **Sub-Daily Timing:** Breach events are aggregated over the 24-hour operational period.
3. **Macro Weather Resolution:** Wind vectors represent regional GridMET (~4 km) flow; extreme micro-canyon vortex winds are smoothed out.
