# GeoRF Paper No-Leak Spatial Clustering Pipeline Workflow

## Overview

This document describes the release-facing no-leak workflow for the GeoRF paper
results. Stage 1 learns GeoRF partition candidates on 2018-2020, Stage 2 learns
fixed consensus partitions from those outputs, and Stage 3 evaluates fixed
partitions on 2021-2024. GeoDT outputs are retained as appendix and
interpretability provenance. GeoXGB, fs0 launch guidance, and 2026-2027
forward/scenario prediction have been archived as non-paper workflows.

## Pipeline Architecture

```text
Stage 1: GeoRF partition candidate learning
  run_batches_2018_2020_partition_learning_visual_monthly.bat georf
  -> archived/release_20260624_reproducibility_inputs/GeoRFExperiment/GeoRFResults/
  -> result_GeoRF_YYYY_fsN_YYYY-MM_visual/

Stage 2: GeoRF consensus clustering
  spatial_weighted_consensus_clustering.bat georf
  -> archived/release_20260624_reproducibility_inputs/GeoRFExperiment/knn_sparsification_results/
  -> cluster_mapping_manifest.json

Stage 3: GeoRF fixed-partition evaluation
  run_partition_k40_comparison_unified.bat georf --visual --month-ind
  -> archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_fs{1,2,3}/
  -> final_artifacts_in_paper_updated/01_main_results/main_month_ind_cont3.xlsx
  -> paper_reproducibility_package/paper_artifacts/final_artifacts_in_paper_updated/01_main_results/main_month_ind_cont3.xlsx
```

### Pipeline Flow Summary

The release workflow uses three active GeoRF commands:

| Stage | Active command | Purpose | Duration |
|-------|----------------|---------|----------|
| 1 | `run_batches_2018_2020_partition_learning_visual_monthly.bat georf` | Learn GeoRF partition candidates on 2018-2020 | ~4-6 hrs |
| 2 | `spatial_weighted_consensus_clustering.bat georf` | Aggregate GeoRF partitions into fixed spatial clusters | ~30-60 min |
| 3 | `run_partition_k40_comparison_unified.bat georf --visual --month-ind` | Evaluate fixed GeoRF partitions on 2021-2024 | ~4-6 hrs |

Key handoff outputs:

- Stage 1 to Stage 2: combined `results_df_*_fsN_YYYY_YYYY.csv` /
  `y_pred_test_*_fsN_YYYY_YYYY.csv` plus archived
  `result_GeoRF_YYYY_fsN_YYYY-MM_visual/` folders.
- Stage 2 to Stage 3: `cluster_mapping_k40_nc*_general.csv`,
  `cluster_mapping_k40_nc*_m2.csv`, `cluster_mapping_k40_nc*_m6.csv`,
  `cluster_mapping_k40_nc*_m10.csv`, and `cluster_mapping_manifest.json`.
- Stage 3 final deliverables: archived
  `archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_fsN/`
  result folders, `final_artifacts_in_paper_updated/01_main_results/main_month_ind_cont3.xlsx`,
  and the package copy at
  `paper_reproducibility_package/paper_artifacts/final_artifacts_in_paper_updated/01_main_results/main_month_ind_cont3.xlsx`.

GeoDT result directories and figures remain in the reproducibility-input archive
as appendix and interpretability provenance. They are not the release quickstart
path.

## Detailed Stage Breakdown

### Stage 1: GeoRF Model Training and Monthly Partition Generation

**Purpose**: Train GeoRF models for each month in 2018-2020, generating
partition candidates before the 2021-2024 evaluation window.

**Script**:

```batch
run_batches_2018_2020_partition_learning_visual_monthly.bat georf
```

**Configuration**:

```batch
YEARS_START=2018
YEARS_END=2020
FORECASTING_SCOPES=1,2,3
VISUAL=1
```

`FORECASTING_SCOPES=1,2,3` maps to the paper horizons in `ACTIVE_LAGS = (4, 8,
12)`.

**Inputs**:

- FEWS NET panel data from `FEWSNET_forecast_unadjusted_bm.csv`
- Satellite, ground-indicator, price, conflict, and encoded geography features
- FEWS NET admin geometry and correspondence metadata

**Outputs**:

```text
result_GeoRF_YYYY_fsX_YYYY-MM_visual/
├── correspondence_table_YYYY-MM.csv
├── vis/
├── space_partitions/
│   ├── s_branch.pkl
│   ├── X_branch_id.npy
│   └── partition files
└── log_print.txt

archived/release_20260624_reproducibility_inputs/GeoRFExperiment/GeoRFResults/
├── results_df_gp_fsX_YYYY_YYYY.csv
└── y_pred_test_gp_fsX_YYYY_YYYY.csv
```

**Duration**: ~4-6 hours for the full 2018-2020 x fs1/fs2/fs3 run.

### Stage 2: GeoRF Consensus Clustering

**Purpose**: Aggregate monthly GeoRF partitions into stable general and
month-specific cluster assignments.

**Script**:

```batch
spatial_weighted_consensus_clustering.bat georf
```

The batch script runs three parts:

- **Part 1 (shared)**: step1 -> step3 merge and link partition tables.
- **Part 2 (general)**: step4 -> step5 -> step6 for an all-month general
  partition.
- **Part 3 (monthly)**: step4 -> step5 -> step6 for February, June, and
  October partitions.

The batch uses refactored root-level scripts for step1, step4, and step5, plus
`scripts/step3_create_linked_tables.py` and
`scripts/step6_complete_clustering_pipeline.py`. `step2_load_correspondence.py`
is retained as a legacy helper, but the unified Stage 2 batch skips it because
its `correspondence_tables_loaded.pkl` output is not consumed downstream.

**Workspace**:

```text
archived/release_20260624_reproducibility_inputs/GeoRFExperiment/
├── GeoRFResults/
│   ├── results_df_*_fsX_YYYY_YYYY.csv
│   └── y_pred_test_*_fsX_YYYY_YYYY.csv
├── linked_tables/
├── similarity_matrices/
├── similarity_matrices_m02/
├── similarity_matrices_m06/
├── similarity_matrices_m10/
└── knn_sparsification_results/
    ├── cluster_mapping_k40_nc*_general.csv
    ├── cluster_mapping_k40_nc*_m2.csv
    ├── cluster_mapping_k40_nc*_m6.csv
    ├── cluster_mapping_k40_nc*_m10.csv
    └── cluster_mapping_manifest.json
```

`k40` is the KNN graph-neighbor count, not the selected number of clusters. The
selected cluster count is recorded in the `nc*` token.

### Stage 3: GeoRF Fixed-Partition Model Comparison

**Purpose**: Re-run GeoRF using fixed consensus partitions to evaluate
partitioned vs pooled performance on 2021-2024.

**Script**:

```batch
run_partition_k40_comparison_unified.bat georf --visual --month-ind
```

**Options used by the release command**:

- `--visual`: generate visualization maps and performance plots.
- `--month-ind`: enable month-specific partitions for February, June, and
  October.

**Default configuration**:

```batch
START_MONTH=2021-01
END_MONTH=2024-12
TRAIN_WINDOW=36
MONTH_IND=1
CONTIGUITY=1
REFINE_ITERS=3
```

**Partition discovery**: Stage 3 reads
`archived/release_20260624_reproducibility_inputs/GeoRFExperiment/knn_sparsification_results/cluster_mapping_manifest.json`
to locate general and month-specific partition files. If the manifest is
missing, it falls back to `cluster_mapping_k40_nc*_general.csv` pattern
matching.

**Comparisons**:

1. **Pooled**: One GeoRF model trained across all available admin units.
2. **Partitioned General**: Separate GeoRF models per general cluster.
3. **Partitioned Month-Specific**: Separate GeoRF models per February, June, or
   October cluster for release-month evaluation.

**Outputs**:

```text
archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_fs1/
archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_fs2/
archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_fs3/
```

Each output folder must include:

- `metrics_monthly.csv`
- `predictions_monthly.csv`
- `metrics_polygon_overall.csv`
- `run_manifest.json`
- refined partition maps under `refined/`

Paper-facing aggregate tables are stored in the final artifact folder and
package copy:

```text
final_artifacts_in_paper_updated/01_main_results/main_month_ind_cont3.xlsx
paper_reproducibility_package/paper_artifacts/final_artifacts_in_paper_updated/01_main_results/main_month_ind_cont3.xlsx
```

## Paper Verification

For fast package validation:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_paper_reproducibility_package.py
```

For read-only validation of the current repository result bundle:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_current_results_reproducibility.py
```

The verifier checks archived Stage 3 result contracts, archived Stage 1/2
handoff artifacts, final paper artifacts in
`final_artifacts_in_paper_updated/`, archived fixed-partition ablation outputs,
and the no-leak root CSV archive.

## Non-Paper Workflow Archive

The former GeoXGB, fs0-only, 2026-2027 forward/scenario prediction, legacy
notebook, and regional exploratory entry points are preserved under:

```text
archived/release_20260624_nonpaper_pipelines/
```

These archived files are historical provenance and are not maintained as
release quickstart workflows.

## Troubleshooting

- If Stage 2 cannot find Stage 1 outputs, confirm the Stage 1 GeoRF run
  produced yearly combined CSVs under
  `archived/release_20260624_reproducibility_inputs/GeoRFExperiment/GeoRFResults/`
  for the current release archive, or under `GeoRFExperiment/GeoRFResults/`
  immediately after a fresh full rerun.
- If Stage 3 cannot find partitions, inspect
  `archived/release_20260624_reproducibility_inputs/GeoRFExperiment/knn_sparsification_results/cluster_mapping_manifest.json`.
- If a Windows batch file fails near an `echo` statement inside an
  `if (...) else (...)` block, check that literal parentheses in the echoed
  text are escaped as `^(` and `^)`.
