# DATASET CARD: `fireline_segments_v1`

**Task:** Binary classification / candidate-line score demonstration  
**Target:** `label` (`1` = generated as Held, `0` = generated as Burned Over)  
**Generation code:** [`src/data/build_dataset.py`](./src/data/build_dataset.py)  
**Available formats:** `data/final/fireline_segments_v1.{parquet,csv,json}`

> **Important:** This is a synthetic dataset. It is not sourced from the NDWS benchmark,
> the USDA Forest Service Fireline Effectiveness dataset, or observed fireline records.
> Fire names and incident IDs are used as profile labels only. Neither the features nor
> the target labels are measurements from those incidents.

## 1. Purpose and provenance

The dataset supports development and repeatable testing of the CLBI feature pipeline,
model API, and dashboard. `generate_fireline_dataset()` procedurally samples terrain,
distance, line geometry, and other values from hand-configured distributions for five
fire profiles. It assigns synthetic segment IDs, dates, line types, and split labels.
No external source dataset is loaded by this generator.

The NDWS dataset is a separate project component. Its benchmark metrics must not be
presented as the provenance of these CLBI records.

## 2. Profiles and split sizes

The generator configures the following profile parameters and row counts:

| Profile label | Incident ID used as metadata | Split | Rows | Configured hold rate |
| :--- | :--- | :--- | ---: | ---: |
| August Complex | `CA-MNF-013028` | `train` | 1,420 | 0.72 |
| Carr Fire | `CA-SHU-007808` | `train` | 680 | 0.68 |
| Creek Fire | `CA-SNF-000958` | `train` | 950 | 0.65 |
| Mendocino Complex | `CA-MEU-008674` | `val` | 810 | 0.74 |
| CZU Lightning Complex | `CA-CZU-005205` | `test` | 740 | 0.70 |
| **Total** |  |  | **4,600** |  |

These are generated profiles, not independent observed fire events. The `train`, `val`,
and `test` names describe how the generated rows are partitioned in the current
experiment; they do not indicate a real-world event holdout.

## 3. Feature and target construction

The generator samples elevation and slope from profile-specific Gaussian
distributions, distance from a bounded exponential distribution, line orientation
uniformly, and spread direction around a configured profile heading. Attack geometry,
burn probability, probability gradient, and predicted-boundary distance are then
calculated from these generated values. Barrier width is selected from fixed line-type
widths.

The synthetic breach score is:

```text
1.8 * attack_dot_product
+ 0.035 * slope
+ 1.5 * burn_prob
- 0.15 * barrier_width_m
- 0.0004 * distance_to_fire
+ logistic_noise
```

For each profile, the generator selects a score cutoff using its configured hold rate.
Rows at or below the cutoff receive `label = 1` / `Held`; rows above it receive
`label = 0` / `Burned Over`. The label is therefore constructed from several input
features by design. Model performance primarily reflects recovery of this synthetic
rule, not prediction of actual fireline outcomes.

## 4. Schema notes

The dataset includes profile metadata (`fire_id`, `fire_name`, `date`, `split`),
generated segment metadata (`segment_id`, `line_type`, `fuel_model_group`), input
features, and the synthetic target (`label`, `outcome_str`). Despite their familiar
units and names, values such as elevation, slope, distance, spread direction, and
barrier width are generated samples—not georeferenced measurements. The generator
does not create coordinates or a verified spatial reference system.

## 5. Appropriate and inappropriate uses

**Appropriate:** testing data loading, feature transformations, model serialization,
API request/response behavior, UI rendering, and comparisons against the known
synthetic target-generation rule.

**Not appropriate:** claiming retrospective accuracy on historical incidents,
estimating real-world breach probabilities, assessing real fireline safety, validating
the NDWS model, or making suppression and resource-allocation decisions.

## 6. Limitations

- The feature distributions, fire-profile settings, and target formula are manually
  specified in the generator and may encode their own assumptions.
- Several predictors are direct inputs to the target formula; high held-out scores are
  not evidence of independent predictive skill.
- The configured profile hold rates shape each profile's target distribution.
- No observed suppression actions, weather measurements, geospatial coordinates, or
  verified breach outcomes are included.
- LOFO across these profiles evaluates synthetic profile separation only; it cannot
  establish generalization to new real fires.

For implementation details, see [`src/data/build_dataset.py`](./src/data/build_dataset.py)
and the generated-data caveats in [`experiments/ROBUSTNESS_REPORT.md`](./experiments/ROBUSTNESS_REPORT.md).
