# Paper Artifact Horizon Labels Design

## Purpose

Update paper-facing artifacts so the forecast intervals are described as
4-month, 8-month, and 12-month forecasting horizons rather than 4-month,
8-month, and 12-month lags. This is a terminology correction for final paper
presentation artifacts and their generators. It must not change model training,
feature construction, result metrics, or source data.

## Current Context

The active final artifact bundle is `final_artifacts_in_paper_updated/`.
Inventory and text scans found user-visible lag wording across:

- Final CSV/Markdown/JSON artifacts in sections 01, 02, 04, 05, 07, 09, 10,
  11, and 12.
- Workbook display cells in `01_main_results/main_month_ind_cont3.xlsx` and
  `01_main_results/ablation_feature_exclude.xlsx`.
- Generator scripts that format paper-facing horizon labels and embedded figure
  titles, including monthly performance, seasonal/region performance, error
  grids, probability uncertainty, humanitarian metrics, false-negative error
  modes, threshold-free metrics, thresholded GeoRF artifacts, partition
  stability, global cluster maps, GeoDT branch-location diagnostics, and table
  builders under `scripts/` and `other_outputs/`.

The same scan also found implementation terms that should remain unchanged,
including `lag_months`, `ACTIVE_LAGS`, `forecasting_scope_to_lag`, lagged
feature column names, and the `Lag Exclude` ablation condition.

## Scope

In scope:

- Replace paper-facing labels such as `4-month lag`, `8-month lag`,
  `12-month lag`, `Forecasting horizon / lag`, and
  `Forecasting horizon (month lag)` with `4-month horizon`,
  `8-month horizon`, `12-month horizon`, and `Forecasting horizon`.
- Update generator scripts so future artifacts use the corrected labels.
- Directly patch existing CSV, Markdown, JSON, and workbook display labels when
  the change is label-only.
- Regenerate PNG figures only when label text is embedded in the image.
- Overwrite final paper PNGs only after their generator scripts are corrected.

Out of scope:

- No Stage 1, Stage 2, or Stage 3 model reruns.
- No recomputation of metric values.
- No changes to raw result directories unless needed only to make a final
  artifact generator emit corrected display labels.
- No renaming of `fs1`, `fs2`, or `fs3` files or directories.
- No blanket rename of technical lag terminology.
- No rename of the `Lag Exclude` ablation row, because it describes a feature
  ablation condition rather than a forecasting horizon label.

## Terminology Rules

Use `horizon` for forecast interval labels shown to paper readers:

- `fs1` -> `4-month horizon`
- `fs2` -> `8-month horizon`
- `fs3` -> `12-month horizon`

Keep `lag` where it describes actual lagged covariates, source feature names,
implementation variables, data-window mechanics, or the ablation feature group:

- `lag_months`
- `ACTIVE_LAGS`
- `forecasting_scope_to_lag`
- feature suffixes such as `_lag4m`, `_lag8m`, `_lag12m`
- phrases such as `lagged outcomes` and `lagged non-crisis states`
- `Lag Exclude`

## Implementation Approach

Use a deterministic presentation relabel plus targeted figure regeneration.

1. Patch generator scripts that create paper-facing display labels.
2. Patch existing text/table artifacts where the only changed values are
   display strings.
3. Patch workbook display cells with `openpyxl` while preserving data values and
   the `Lag Exclude` row.
4. Regenerate figures with embedded horizon/lag labels after script updates.
5. Leave figures untouched when they have no visible horizon/lag label.

This approach preserves current metrics while preventing future regeneration
from reintroducing the old wording.

## Candidate Artifact Updates

Direct display-label patches are expected for:

- `final_artifacts_in_paper_updated/README.md`
- `01_main_results/monthly_performance_manifest.json`
- `01_main_results/table1_season_performance.csv`
- `01_main_results/table2_region_performance.csv`
- `01_main_results/table2_region_performance_partitioned_pooled_fewsnet.csv`
- `02_methods_and_temporal_scope/temporal_data_splits_table.csv`
- `02_methods_and_temporal_scope/temporal_data_splits_schematic_note.md`
- `04_error_analysis/error_rate_seasonal*.csv`
- `05_partition_diagnostics/georf_stage1_partition_stability_note.md`
- `07_probability_uncertainty/*.csv` and compact Markdown table
- `09_humanitarian_metrics/*.csv` and compact Markdown table
- `10_false_negative_error_modes/*.csv` and note text where the phrase is a
  dominant horizon label
- `11_threshold_free_metrics/*.csv` and compact Markdown table
- `12_thresholded_georf_results/*.csv` and compact Markdown table
- Workbook header cells in `main_month_ind_cont3.xlsx` and
  `ablation_feature_exclude.xlsx`

Candidate figures for regeneration or inspection include:

- `01_main_results/georf_monthly_performance.png`
- `01_main_results/global_cluster_map_2x2_georf_refined.png`
- `01_main_results/global_cluster_map_2x2_geodt_refined.png`
- `04_error_analysis/error_rate_seasonal_3x3*.png`
- `05_partition_diagnostics/georf_stage1_partition_stability.png`
- `07_probability_uncertainty/georf_probability_reliability.png`
- `09_humanitarian_metrics/georf_humanitarian_population_bars.png`
- `08_geodt_diagnostics/geodt_branch_1_vs_001_locations_2024-10_fs1_global.png`

## Validation

Run these checks after implementation:

- Targeted scan over `final_artifacts_in_paper_updated/` for forbidden
  presentation phrases: `4-month lag`, `8-month lag`, `12-month lag`,
  `month lag`, `horizon / lag`, and `forecasting horizon / lag`.
- Manual review of remaining `lag` hits to verify they are true lagged-feature
  or implementation references.
- Workbook inspection with `openpyxl` to verify `Forecasting horizon` appears in
  the relevant header cells and `Lag Exclude` remains unchanged.
- CSV row-count and numeric-column comparison before and after direct patches,
  confirming only display strings changed.
- Figure existence, modified time, and dimension checks for regenerated PNGs.
- Visual inspection of regenerated figures with visible horizon labels.

## Acceptance Criteria

- All paper-facing forecast-interval labels in
  `final_artifacts_in_paper_updated/` use `4-month horizon`,
  `8-month horizon`, or `12-month horizon`.
- Generator scripts no longer emit `4-month lag`, `8-month lag`, or
  `12-month lag` as paper-facing display labels.
- Metrics, row counts, and workbook numeric values are unchanged.
- Remaining `lag` occurrences are limited to lagged covariates, feature names,
  implementation variables, data-window mechanics, or the `Lag Exclude`
  ablation condition.
- Regenerated figures overwrite only the intended final artifact PNGs.
