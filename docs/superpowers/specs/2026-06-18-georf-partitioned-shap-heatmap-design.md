# GeoRF Partitioned SHAP Group Heatmap Design

## Purpose

Create a standalone paper-facing SHAP heatmap for the current no-leak GeoRF partitioned model results. The figure addresses the interpretability gap left by the fixed-partition feature-exclude ablation workbook, which is better framed as grouped retraining sensitivity than direct feature importance.

The deliverable is a `7 x 3` heatmap saved under `final_artifacts_in_paper_updated/01_main_results/`. Rows are feature groups aligned with the feature-exclude definitions, excluding the broad secondary group. Columns are forecasting horizons: 4, 8, and 12 months.

## Figure Contract

Rows:

- Weather
- Agri
- Conflict
- Econ
- Food Prices
- Geographic
- Lag

Columns:

- 4-month horizon (`fs1`)
- 8-month horizon (`fs2`)
- 12-month horizon (`fs3`)

Each heatmap cell reports the monthly mean group share of mean absolute SHAP values for the GeoRF partitioned model, annotated as `mean% +/- sd`. The mean and standard deviation are computed across the 12 evaluated target months in 2021-2024: February, June, and October for each year.

Group share is computed within the seven reported groups only. For each month and horizon, sum mean absolute SHAP values within each group, then divide by the total summed mean absolute SHAP across all seven groups. This makes attribution shares comparable across horizons while keeping the interpretation focused on the paper-facing feature taxonomy.

## Scope

In scope:

- GeoRF partitioned model only.
- Standard no-leak Stage 3 evaluation window, 2021-2024.
- Forecasting scopes `fs1`, `fs2`, and `fs3`.
- Target months February, June, and October.
- Current refined GeoRF partition maps already used by `result_partition_k40_compare_GF_fs{1,2,3}/`.
- Standalone script and focused tests.
- Paper artifact outputs in `final_artifacts_in_paper_updated/01_main_results/`.

Out of scope:

- GeoDT, GeoXGB, and pooled GeoRF SHAP summaries.
- Changing the main Stage 3 comparison script or its result contract.
- Rebuilding the feature-exclude ablation workbook.
- Adding the secondary feature group to the heatmap.

## Data Flow

The standalone script rebuilds the Stage 3 GeoRF partitioned evaluation context without modifying existing result folders.

1. Read the current source panel, defaulting to `FEWSNET_forecast_unadjusted_bm.csv`.
2. Prepare features for each forecasting scope using the same Stage 3 feature path and lag schedule as the existing GeoRF comparison workflow.
3. Load the refined GeoRF partition maps from `result_partition_k40_compare_GF_fs{scope}/refined/`, including month-specific maps where the current Stage 3 run uses them.
4. For each scope and each 2021-2024 target month in February, June, and October, train partition-specific RF models with the 36-month rolling training window.
5. Compute SHAP values for test samples using the local partition RF assigned to each sample.
6. Exclude pooled-fallback samples from the main heatmap aggregation, but record fallback counts and shares in the manifest.
7. Map feature names into the seven feature groups, aggregate mean absolute SHAP to group shares, and summarize each group-horizon pair across months.

## Outputs

Write all deliverables to `final_artifacts_in_paper_updated/01_main_results/`:

- `georf_partitioned_shap_group_heatmap.png`
- `georf_partitioned_shap_group_heatmap.pdf`
- `georf_partitioned_shap_group_summary.csv`
- `georf_partitioned_shap_monthly.csv`
- `georf_partitioned_shap_manifest.json`
- `georf_partitioned_shap_note.md`

The monthly CSV should contain one row per scope, target month, and feature group, including raw summed mean absolute SHAP, normalized group share, matched feature count, and total denominator used for normalization.

The summary CSV should contain one row per scope and feature group, including mean share, standard deviation, number of months, and display labels used by the heatmap.

The manifest should record the source CSV, scope-to-lag mapping, evaluated months, partition map paths, RF parameters, SHAP sample cap if used, feature group column matches and misses, fallback sample counts, output paths, and timestamp.

The note should explain that values are relative SHAP attribution shares for the partitioned GeoRF model, not causal effects and not retraining ablation deltas.

## Error Handling

The script should fail when:

- A required source dataset is missing.
- A required refined partition map is missing.
- A required feature group has no matched columns after feature preparation.
- A horizon has fewer than 12 summarized target months.
- SHAP feature dimensions do not match resolved feature names.

The script may warn and continue when:

- Some expected columns within a group are absent after feature preparation, as long as the group still has matched features.
- A small number of samples require pooled fallback; these samples are excluded from the main aggregation and counted in the manifest.

## Testing

Unit tests should cover pure functions:

- Mapping feature names to the seven groups.
- Reporting matched and missing columns by group.
- Aggregating feature-level mean absolute SHAP to group-level raw sums.
- Normalizing group sums to shares that sum to 1.0 per scope-month.
- Computing mean and standard deviation across months.
- Building the heatmap matrix in the expected row and column order.

Full model execution is expected to be validated by a dry-run or bounded run because it retrains partitioned RF models and computes SHAP values across 36 scope-month combinations.

## Implementation Boundary

Prefer a new script named `scripts/build_georf_partitioned_shap_heatmap.py`. Do not patch `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py` for this deliverable unless implementation discovers a shared helper extraction is necessary and can be kept behavior-preserving.

The script should reuse existing project helpers and constants where practical, but it must keep all new generated artifacts in `final_artifacts_in_paper_updated/01_main_results/` and leave existing Stage 3 result folders untouched.
