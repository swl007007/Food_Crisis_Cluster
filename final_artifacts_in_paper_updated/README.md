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
- `region_class_prevalence_2021_2024.csv`: test-period class prevalence by
  FEWSNET region and evaluated target month. The Middle East rows for
  2021-10 through 2023-02 are flagged as `data not validated`.

## Core Figures

- `georf_monthly_performance.png`: monthly GeoRF performance comparison.
- `global_cluster_map_2x2_georf_refined.png`: refined GeoRF global cluster map.
- `global_cluster_map_2x2_geodt_refined.png`: refined GeoDT global cluster map.
- `predictions_2024_feb_jun_oct.png`: 2024 actual vs predicted GeoRF map.
- `error_rate_seasonal_3x3.png`: seasonal error-rate panel.
- `error_rate_seasonal_3x3_crisis.png`: crisis-only seasonal error-rate panel.
- `error_rate_seasonal_3x3_noncrisis.png`: non-crisis seasonal error-rate panel.
- `fewsnet_crisis_stack_2018.png`: FEWSNET crisis stack figure.
- `region_class_prevalence_2021_2024.png`: region-level stacked class
  prevalence over the 2021-2024 evaluated test months, with the Middle East
  2021-10 through 2023-02 interval shaded as data not validated.
- `georf_stage1_partition_stability.png`: GeoRF-only Stage 1 partition
  stability overview with ARI/NMI boxplots and cluster-size distributions.
- `georf_m2_adjacency_refinement_1x3.png`: GeoRF m2 pre/post adjacency
  refinement example with reassigned polygons highlighted.
- `georf_m2_cluster_profile_similarity.png`: market-access,
  conflict-exposure, and error-mode inter-cluster similarity heatmaps with
  within-cluster cohesion bars for the representative GeoRF m2 local-model
  partition.
- `georf_probability_reliability.png`: GeoRF pooled vs partitioned probability
  reliability curves by forecasting horizon / lag.
- GeoDT 4-month-lag branch location diagnostic.
- GeoDT 4-month-lag branch tree diagnostic.
- `feature_engineering.png` and `walkthrough.png`: static explanatory figures.

## Diagnostic Data

- `error_rate_seasonal.csv`
- `error_rate_seasonal_crisis.csv`
- `error_rate_seasonal_noncrisis.csv`
- `monthly_performance_manifest.json`
- `georf_stage1_partition_stability_pairwise.csv`: pairwise ARI/NMI for
  GeoRF Stage 1 partition plans across years, months, and forecasting
  horizons / lags.
- `georf_stage1_partition_stability_summary.csv`: comparison-axis summary of
  the pairwise stability metrics.
- `georf_stage1_partition_stability_appendix_table.csv`: compact appendix-ready
  version of the GeoRF Stage 1 partition stability summary.
- `georf_stage1_partition_stability_appendix_table.md`: Markdown rendering of
  the compact appendix-ready stability table.
- `georf_stage1_partition_cluster_sizes.csv`: per-plan cluster-size summary
  after excluding out-of-scope `s-1` assignments.
- `georf_stage1_partition_cluster_size_distribution.csv`: per-cluster size
  records used for the distribution panel in the stability figure.
- `georf_stage1_partition_stability_note.md`: Chinese reviewer-facing note and
  English appendix text for the GeoRF Stage 1 partition-stability diagnostic.
- `georf_m2_adjacency_refinement_summary.csv`: GeoRF m2 adjacency-refinement
  reassignment counts and proportions for the pre/post figure.
- `georf_m2_adjacency_refinement_note.md`: Chinese reviewer-facing note and
  English appendix text for the GeoRF m2 adjacency-refinement example.
- `georf_m2_cluster_profile_table.csv`: descriptive cluster-level profiles for
  the representative GeoRF m2 refined local-model partition.
- `georf_m2_cluster_profile_similarity_matrices.csv`: long-format
  inter-cluster similarity matrices for market access, conflict exposure, and
  error-mode composition.
- `georf_m2_cluster_profile_cohesion.csv`: within-cluster cohesion values
  paired with the similarity heatmaps.
- `georf_m2_cluster_profile_note.md`: Chinese reviewer-facing note and English
  appendix text for the cluster profile diagnostics.
- `georf_probability_bootstrap_ci.csv`: GeoRF pooled vs partitioned paired
  country-clustered bootstrap confidence intervals for precision, recall, F1,
  and Brier score by forecasting horizon / lag.
- `georf_probability_bootstrap_compact_table.csv`: compact appendix-ready
  paired-bootstrap delta table with one row per forecasting horizon / lag.
- `georf_probability_bootstrap_compact_table.md`: Markdown rendering of the
  compact appendix-ready paired-bootstrap delta table.
- `georf_probability_bootstrap_region_ci.csv`: region-specific version of the
  paired country-clustered bootstrap confidence intervals.
- `georf_probability_brier_reliability.csv`: Brier scores and probability
  reliability-bin summaries for GeoRF pooled and partitioned models.
- `georf_probability_uncertainty_summary.csv`: descriptive probability
  uncertainty summaries by model and forecasting horizon / lag.
- `georf_probability_uncertainty_note.md`: Chinese reviewer-facing note and
  English appendix text for probability export, threshold framing, calibration
  diagnostics, Brier score, and uncertainty intervals.

## Reproduction Checks

Run from the repository root:

```bat
python scripts\verify_current_results_reproducibility.py
```

For the full folder-level reproduction map, see:

```text
CURRENT_RESULTS_REPRODUCTION.md
```
