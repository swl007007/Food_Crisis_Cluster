# Final Paper Artifacts

This is the active final artifact folder for the current no-temporal-leak
workflow. The legacy `final_artifacts_in_paper/` folder is not present in the
current workspace.

## Core Tables

- `main_month_ind_cont3.xlsx`: main GeoRF/GeoDT/FEWSNET comparison table for
  the current fixed-partition, month-indicator, contiguity-refined setup.
- `ablation_feature_exclude.xlsx`: fixed-partition GeoRF feature-exclude
  ablation workbook generated from
  `main_ablation_exclude_updated_stage3_fixed_partitions/`.
- `table1_season_performance.csv`: season-level paper table.
- `table2_region_performance.csv`: region-level paper table.
- `table2_region_performance_partitioned_pooled_fewsnet.csv`: region-level
  partitioned/pooled/FEWSNET comparison table.

## Core Figures

- `georf_monthly_performance.png`: monthly GeoRF performance comparison.
- `global_cluster_map_2x2_georf_refined.png`: refined GeoRF global cluster map.
- `global_cluster_map_2x2_geodt_refined.png`: refined GeoDT global cluster map.
- `predictions_2024_feb_jun_oct.png`: 2024 actual vs predicted GeoRF map.
- `error_rate_seasonal_3x3.png`: seasonal error-rate panel.
- `error_rate_seasonal_3x3_crisis.png`: crisis-only seasonal error-rate panel.
- `error_rate_seasonal_3x3_noncrisis.png`: non-crisis seasonal error-rate panel.
- `fewsnet_crisis_stack_2018.png`: FEWSNET crisis stack figure.
- `geodt_branch_1_vs_001_locations_2024-10_fs1_global.png`: GeoDT branch
  location diagnostic.
- `geodt_branch_tree_compare_2024-10_fs1_001_vs_1.png`: GeoDT branch tree
  diagnostic.
- `feature_engineering.png` and `walkthrough.png`: static explanatory figures.

## Diagnostic Data

- `error_rate_seasonal.csv`
- `error_rate_seasonal_crisis.csv`
- `error_rate_seasonal_noncrisis.csv`
- `monthly_performance_manifest.json`

## Reproduction Checks

Run from the repository root:

```bat
python scripts\verify_current_results_reproducibility.py
```

For the full folder-level reproduction map, see:

```text
CURRENT_RESULTS_REPRODUCTION.md
```
