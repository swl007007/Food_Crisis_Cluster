# Final Paper Artifacts

This is the active final artifact folder for the current no-temporal-leak
workflow. Files are grouped by paper use case so the root stays navigable.

## Folder Index

- `01_main_results/`: core result tables, main performance figures, cluster
  maps, prediction maps, and the feature-ablation workbook.
- `02_methods_and_temporal_scope/`: workflow notes, temporal split table,
  technical-detail notes, and method schematic figures.
- `03_class_prevalence/`: region-level class prevalence figure/table and
  FEWSNET crisis-stack context figure.
- `04_error_analysis/`: seasonal error-rate figures and source tables.
- `05_partition_diagnostics/`: GeoRF partition-stability and adjacency
  refinement appendix diagnostics.
- `06_cluster_profiles/`: GeoRF m2 local-domain profile tables, similarity
  figure, and note.
- `07_probability_uncertainty/`: GeoRF probability export, reliability, Brier,
  and bootstrap uncertainty diagnostics.
- `08_geodt_diagnostics/`: auxiliary GeoDT branch-location and branch-tree
  diagnostic figures.
- `09_humanitarian_metrics/`: GeoRF population-weighted humanitarian metrics
  using raw FEWSNET population.
- `10_false_negative_error_modes/`: GeoRF partitioned false-negative crisis
  error-mode diagnostics for selected hotspots.
- `11_threshold_free_metrics/`: GeoRF PR-AUC and fixed precision/recall
  operating-point diagnostics from existing probability outputs.

## 01 Main Results

- `01_main_results/main_month_ind_cont3.xlsx`: main GeoRF/GeoDT/FEWSNET
  comparison table for the current fixed-partition, month-indicator,
  contiguity-refined setup.
- `01_main_results/ablation_feature_exclude.xlsx`: fixed-partition GeoRF
  feature-exclude ablation workbook generated from
  `main_ablation_exclude_updated_stage3_fixed_partitions/`.
- `01_main_results/table1_season_performance.csv`: season-level paper table.
- `01_main_results/table2_region_performance.csv`: region-level paper table.
- `01_main_results/table2_region_performance_partitioned_pooled_fewsnet.csv`:
  region-level partitioned/pooled/FEWSNET comparison table.
- `01_main_results/georf_monthly_performance.png`: monthly GeoRF performance
  comparison.
- `01_main_results/global_cluster_map_2x2_georf_refined.png`: refined GeoRF
  global cluster map.
- `01_main_results/global_cluster_map_2x2_geodt_refined.png`: refined GeoDT
  global cluster map.
- `01_main_results/predictions_2024_feb_jun_oct.png`: 2024 actual vs
  predicted GeoRF map.
- `01_main_results/monthly_performance_manifest.json`: manifest for the
  monthly performance figure generation.

## 02 Methods And Temporal Scope

- `02_methods_and_temporal_scope/technical_details_review_note.md`: Chinese
  reviewer-facing technical detail note plus English appendix text for
  implemented methods.
- `02_methods_and_temporal_scope/technical_details_unimplemented_patch_note.md`:
  items that are not implemented and therefore should not be described in the
  paper appendix.
- `02_methods_and_temporal_scope/temporal_data_splits_table.csv`: compact
  target-month and forecasting-horizon data-split table.
- `02_methods_and_temporal_scope/temporal_data_splits_schematic_note.md`:
  temporal split schematic note.
- `02_methods_and_temporal_scope/feature_engineering.png`: feature engineering
  explanatory figure.
- `02_methods_and_temporal_scope/walkthrough.png`: workflow walkthrough figure.

## 03 Class Prevalence

- `03_class_prevalence/region_class_prevalence_2021_2024.csv`: test-period
  class prevalence by FEWSNET region and evaluated target month. The Middle
  East rows for 2021-10 through 2023-02 are flagged as `data not validated`.
- `03_class_prevalence/region_class_prevalence_2021_2024.png`: region-level
  stacked class prevalence over the 2021-2024 evaluated test months.
- `03_class_prevalence/fewsnet_crisis_stack_2018.png`: FEWSNET crisis-stack
  context figure.

## 04 Error Analysis

- `04_error_analysis/error_rate_seasonal.csv`: source table for the seasonal
  error-rate panel.
- `04_error_analysis/error_rate_seasonal_crisis.csv`: crisis-only seasonal
  error-rate source table.
- `04_error_analysis/error_rate_seasonal_noncrisis.csv`: non-crisis seasonal
  error-rate source table.
- `04_error_analysis/error_rate_seasonal_3x3.png`: seasonal error-rate panel.
- `04_error_analysis/error_rate_seasonal_3x3_crisis.png`: crisis-only
  seasonal error-rate panel.
- `04_error_analysis/error_rate_seasonal_3x3_noncrisis.png`: non-crisis
  seasonal error-rate panel.

## 05 Partition Diagnostics

- `05_partition_diagnostics/georf_stage1_partition_stability.png`: GeoRF-only
  Stage 1 partition stability overview with ARI/NMI boxplots and cluster-size
  distributions.
- `05_partition_diagnostics/georf_stage1_partition_stability_pairwise.csv`:
  pairwise ARI/NMI for GeoRF Stage 1 partition plans across years, months, and
  forecasting horizons / lags.
- `05_partition_diagnostics/georf_stage1_partition_stability_summary.csv`:
  comparison-axis summary of pairwise stability metrics.
- `05_partition_diagnostics/georf_stage1_partition_stability_appendix_table.csv`:
  compact appendix-ready version of the GeoRF Stage 1 partition-stability
  summary.
- `05_partition_diagnostics/georf_stage1_partition_stability_appendix_table.md`:
  Markdown rendering of the compact appendix-ready stability table.
- `05_partition_diagnostics/georf_stage1_partition_cluster_sizes.csv`:
  per-plan cluster-size summary after excluding out-of-scope `s-1`
  assignments.
- `05_partition_diagnostics/georf_stage1_partition_cluster_size_distribution.csv`:
  per-cluster size records used for the distribution panel.
- `05_partition_diagnostics/georf_stage1_partition_stability_note.md`: Chinese
  reviewer-facing note and English appendix text for the partition-stability
  diagnostic.
- `05_partition_diagnostics/georf_m2_adjacency_refinement_1x3.png`: GeoRF m2
  pre/post adjacency refinement example with reassigned polygons highlighted.
- `05_partition_diagnostics/georf_m2_adjacency_refinement_summary.csv`: GeoRF
  m2 adjacency-refinement reassignment counts and proportions.
- `05_partition_diagnostics/georf_m2_adjacency_refinement_note.md`: Chinese
  reviewer-facing note and English appendix text for the adjacency-refinement
  example.

## 06 Cluster Profiles

- `06_cluster_profiles/georf_m2_cluster_profile_similarity.png`:
  market-access, conflict-exposure, and error-mode inter-cluster similarity
  heatmaps with within-cluster cohesion bars for the representative GeoRF m2
  local-model partition.
- `06_cluster_profiles/georf_m2_cluster_profile_table.csv`: descriptive
  cluster-level profiles for the representative GeoRF m2 refined local-model
  partition.
- `06_cluster_profiles/georf_m2_cluster_profile_similarity_matrices.csv`:
  long-format inter-cluster similarity matrices for market access, conflict
  exposure, and error-mode composition.
- `06_cluster_profiles/georf_m2_cluster_profile_cohesion.csv`: within-cluster
  cohesion values paired with the similarity heatmaps.
- `06_cluster_profiles/georf_m2_cluster_profile_note.md`: Chinese
  reviewer-facing note and English appendix text for the cluster profile
  diagnostics.

## 07 Probability Uncertainty

- `07_probability_uncertainty/georf_probability_reliability.png`: GeoRF pooled
  vs partitioned probability reliability curves by forecasting horizon / lag.
- `07_probability_uncertainty/georf_probability_bootstrap_ci.csv`: GeoRF
  pooled vs partitioned paired country-clustered bootstrap confidence intervals
  for precision, recall, F1, and Brier score by forecasting horizon / lag.
- `07_probability_uncertainty/georf_probability_bootstrap_compact_table.csv`:
  compact appendix-ready paired-bootstrap delta table with one row per
  forecasting horizon / lag.
- `07_probability_uncertainty/georf_probability_bootstrap_compact_table.md`:
  Markdown rendering of the compact appendix-ready paired-bootstrap delta
  table.
- `07_probability_uncertainty/georf_probability_bootstrap_region_ci.csv`:
  region-specific version of the paired country-clustered bootstrap confidence
  intervals.
- `07_probability_uncertainty/georf_probability_brier_reliability.csv`: Brier
  scores and probability reliability-bin summaries for GeoRF pooled and
  partitioned models.
- `07_probability_uncertainty/georf_probability_uncertainty_summary.csv`:
  descriptive probability uncertainty summaries by model and forecasting
  horizon / lag.
- `07_probability_uncertainty/georf_probability_uncertainty_note.md`: Chinese
  reviewer-facing note and English appendix text for probability export,
  threshold framing, calibration diagnostics, Brier score, and uncertainty
  intervals.

## 08 GeoDT Diagnostics

- `08_geodt_diagnostics/geodt_branch_1_vs_001_locations_2024-10_fs1_global.png`:
  GeoDT 4-month-lag branch location diagnostic.
- `08_geodt_diagnostics/geodt_branch_tree_compare_2024-10_fs1_001_vs_1.png`:
  GeoDT 4-month-lag branch tree diagnostic.
- `08_geodt_diagnostics/geodt_branch_tree_compare_2024-10_fs1_001_vs_1_2.png`:
  alternate GeoDT branch tree diagnostic render.

## 09 Humanitarian Metrics

- `09_humanitarian_metrics/georf_humanitarian_population_compact_table.csv`:
  compact appendix-ready GeoRF population-weighted humanitarian metrics by
  forecasting horizon / lag.
- `09_humanitarian_metrics/georf_humanitarian_population_compact_table.md`:
  Markdown rendering of the compact humanitarian population metrics table.
- `09_humanitarian_metrics/georf_humanitarian_population_summary.csv`:
  long-format horizon-level population-month metrics for GeoRF pooled and
  partitioned models.
- `09_humanitarian_metrics/georf_humanitarian_population_by_month.csv`:
  month-level diagnostic population-month metrics by horizon and model.
- `09_humanitarian_metrics/georf_humanitarian_population_bars.png`: compact
  missed-crisis and false-alert population-month bar figure.
- `09_humanitarian_metrics/georf_humanitarian_population_note.md`: Chinese
  reviewer-facing note and English appendix text for population-weighted
  humanitarian metrics.

## 10 False Negative Error Modes

- `10_false_negative_error_modes/georf_partitioned_false_negative_hotspot_summary.csv`:
  hotspot-by-horizon GeoRF partitioned false-negative counts, population-months,
  probability, seasonal crisis-error context, and missed-crisis shares.
- `10_false_negative_error_modes/georf_partitioned_false_negative_error_modes.csv`:
  hotspot-level descriptive proxy evidence for conflict, prices, lagged
  outcomes, covariate missingness, near-threshold predictions, and boundary
  context.
- `10_false_negative_error_modes/georf_partitioned_false_negative_hotspot_compact_table.md`:
  compact appendix-ready Markdown table for selected hotspot error modes.
- `10_false_negative_error_modes/georf_partitioned_false_negative_note.md`:
  Chinese reviewer-facing note and English appendix text for GeoRF partitioned
  crisis false-negative error modes.

## 11 Threshold-Free Metrics

- `11_threshold_free_metrics/georf_threshold_free_metrics.csv`: long-format
  GeoRF pooled and partitioned PR-AUC and fixed operating-point metrics by
  forecasting horizon / lag.
- `11_threshold_free_metrics/georf_threshold_free_metrics_compact_table.csv`:
  compact appendix-ready comparison table with partitioned-minus-pooled deltas.
- `11_threshold_free_metrics/georf_threshold_free_metrics_compact_table.md`:
  Markdown rendering of the compact threshold-free metrics table.
- `11_threshold_free_metrics/georf_threshold_free_metrics_note.md`: Chinese
  reviewer-facing note and English appendix text for PR-AUC and fixed
  precision/recall diagnostics.

## Reproduction Checks

Run from the repository root:

```bat
python scripts\verify_current_results_reproducibility.py
```

For the full folder-level reproduction map, see:

```text
CURRENT_RESULTS_REPRODUCTION.md
```
