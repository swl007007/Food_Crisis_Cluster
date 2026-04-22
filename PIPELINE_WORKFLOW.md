# GeoRF/GeoXGB/GeoDT Spatial Clustering Pipeline Workflow

## Overview

This document describes the complete workflow for generating spatial partitions using the GeoRF/GeoXGB/GeoDT framework with consensus clustering. Each model type (Random Forest, XGBoost, Decision Tree) follows the same three-stage pipeline independently, producing its own partitions and comparison results. All three stages use unified batch scripts that accept the model type as an argument.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         STAGE 1: MODEL TRAINING                         │
│                      Generate Monthly Partition Results                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                              ┌──────────────────────────────────┐
                              │   run_batches_2021_2024_         │
                              │   visual_monthly.bat             │
                              │   <model_type>                   │
                              │   (georf | geoxgb | geodt)      │
                              └──────────────────────────────────┘
                                              │
                              result_GeoRF_* / result_GeoXGB_* / result_GeoDT_*
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: CONSENSUS CLUSTERING                        │
│            Generate General & Month-Specific Partitions                 │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                              ┌──────────────────────────────────┐
                              │   spatial_weighted_consensus_    │
                              │   clustering.bat                 │
                              │   <model_type>                   │
                              │   (georf | geoxgb | geodt)      │
                              └──────────────────────────────────┘
                                              │
                              Part 1 (shared): step1->step3
                              Part 2 (general): step4->step5->step6
                              Part 3 (monthly): step4->step5->step6
                                     x3 (Feb, Jun, Oct)
                                              │
                              Generates partition files:
                              - cluster_mapping_k40_nc*_general.csv
                              - cluster_mapping_k40_nc*_m2.csv (Feb)
                              - cluster_mapping_k40_nc*_m6.csv (Jun)
                              - cluster_mapping_k40_nc*_m10.csv (Oct)
                              - cluster_mapping_manifest.json
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                STAGE 3: PARTITIONED MODEL COMPARISON                    │
│          Re-run Models with Generated Spatial Partitions                │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                              ┌──────────────────────────────────┐
                              │   run_partition_k40_comparison_  │
                              │   unified.bat                    │
                              │   <model_type> [options]         │
                              │   (georf | geoxgb | geodt)      │
                              └──────────────────────────────────┘
                                     │
                                     ▼
                     Final comparison results:
                     - Pooled vs Partitioned
                     - General vs Month-Specific
                     - Performance metrics & plots
```

### Pipeline Flow Summary

The complete workflow uses **three unified batch scripts**, each accepting a model type argument:

| Stage | Script | Purpose | Duration |
|-------|--------|---------|----------|
| 1 | `run_batches_2021_2024_visual_monthly.bat <model>` | Train models, generate monthly partitions | ~6-8 hrs |
| 2 | `spatial_weighted_consensus_clustering.bat <model>` | Aggregate partitions into spatial clusters | ~30-60 min |
| 3 | `run_partition_k40_comparison_unified.bat <model>` | Compare partitioned vs pooled models | ~4-6 hrs |

Key handoff outputs:
- Stage 1 -> Stage 2: combined `results_df_*_fsN_YYYY_YYYY.csv` / `y_pred_test_*_fsN_YYYY_YYYY.csv` plus archived `result_Geo{Model}_YYYY_fsN_YYYY-MM_visual/`
- Stage 2 -> Stage 3: `cluster_mapping_k40_nc*_general.csv`, optional `cluster_mapping_k40_nc*_m2/_m6/_m10.csv`, and `cluster_mapping_manifest.json`
- Stage 3 final deliverables: `result_partition_k40_compare_{GF,XGB,DT}_fsN/`, `other_outputs/Table_Format.xlsx`, and `other_outputs/Model_Comparison_Table.xlsx`

**Example (full pipeline for GeoRF)**:
```batch
run_batches_2021_2024_visual_monthly.bat georf
spatial_weighted_consensus_clustering.bat georf
run_partition_k40_comparison_unified.bat georf --visual --month-ind
```

---

## FS0-Only Mode (Stand-Alone Lag-1 Pipeline)

`fs0` is a separate forecasting scope with **lag = 1 month** that runs alongside — not in place of — the standard fs1+fs2+fs3 pipeline. It was added without modifying `ACTIVE_LAGS = (4, 8, 12)` or any fs1/fs2/fs3 code path. Running or not running fs0 has zero impact on fs1/fs2/fs3 behavior and artifacts.

### When to use fs0

- You need a very short-horizon forecast (1 month out) alongside the longer-horizon fs1/fs2/fs3 forecasts.
- You want to compare concurrent-feature-plus-lag1 performance against concurrent-feature-plus-lag4/8/12 performance under otherwise identical model/partition logic.
- You want a stand-alone deliverable (`Table_Format_fs0.xlsx`) that does not co-mingle with the main fs1/2/3 comparison table.

### Activation

Pass `--fs0-only` to **every** stage in the same run. Do not mix modes across stages.

```batch
run_batches_2021_2024_visual_monthly.bat <model> --fs0-only
spatial_weighted_consensus_clustering.bat <model> --fs0-only
run_partition_k40_comparison_unified.bat <model> --fs0-only
```

`<model>` is one of `georf`, `geoxgb`, `geodt`. The standard 3-stage ordering is unchanged.

### Stage-by-stage differences

| Concern | Default pipeline | `--fs0-only` pipeline |
|---|---|---|
| Stage 1 scope loop | `SCOPE_LIST="1 2 3"` | `SCOPE_LIST="0"` |
| Stage 1 batch count | 144 (4 y × 3 s × 12 m) | 48 (4 y × 1 s × 12 m) |
| Stage 1 combined CSVs | `results_df_*_fs{1,2,3}_YYYY_YYYY.csv` | `results_df_*_fs0_YYYY_YYYY.csv` |
| Stage 1 visual archives | `result_Geo{Model}_*_fs{1,2,3}_*_visual/` | `result_Geo{Model}_*_fs0_*_visual/` |
| Stage 2 input glob | `results_df_*_fs*.csv` | `results_df_*_fs0_*.csv` |
| Stage 2 Part 3 (m2/m6/m10) | Runs | **Skipped** (fs0 alone produces too few candidate partitions for month-specific consensus) |
| Stage 2 partition outputs | general + m2 + m6 + m10 | **general only** |
| Stage 3 scope list | `SCOPES="1 2 3"` (or `--scope N` for N ∈ 1..3) | Forced to `SCOPES="0"` |
| Stage 3 `--month-ind` | Honored | **Forcibly cleared** regardless of what is passed |
| Stage 3 contiguity refinement | General + m2 + m6 + m10 | General only |
| Stage 3 output dirs | `result_partition_k40_compare_{GF,XGB,DT}_fs{1,2,3}` | `result_partition_k40_compare_{GF,XGB,DT}_fs0` |
| Stage 3 aggregated table | `other_outputs/Table_Format.xlsx` | `other_outputs/Table_Format_fs0.xlsx` |
| FEWSNET baseline rows | Included (fs1, fs2; fs3 extrapolated) | Omitted (no fs0 ground truth in FEWSNET) |

### Implementation notes

- `fs0` is defined in `src/utils/lag_schedules.py` as module constants `FS0_SCOPE = 0` and `FS0_LAG_MONTHS = 1`. `forecasting_scope_to_lag(0, lags)` returns `1` via an early-return branch **before** the normal range check, so scopes 1/2/3 take the identical code path as before.
- The three `main_model_*.py` entry points expose `--forecasting_scope {0,1,2,3}` with default `1`. Default runs are unaffected.
- Aggregation scripts `other_outputs/aggregate_results.py` and `other_outputs/generate_table.py` both accept `--fs0-only` and emit distinct xlsx filenames (`*_fs0.xlsx`) to avoid clobbering fs1/2/3 tables.
- Feature engineering is unchanged: `prepare_features` already includes both the concurrent original columns and `{variant}_lag{N}m` columns. With fs0 it simply produces `{variant}_lag1m`.

### What fs0 does NOT change

- `ACTIVE_LAGS` stays `(4, 8, 12)`. Extending it to `(1, 4, 8, 12)` would silently reindex every `len(ACTIVE_LAGS)`-based loop and break fs1/fs2/fs3 artifacts. The config validator rejects any such change.
- The `nowcasting` parameter in `main_model_*.py` (a 2-layer-model toggle, hardcoded `False`) is untouched.
- Running the default pipeline without the flag is byte-for-byte equivalent to pre-fs0 behavior.

---

## Detailed Stage Breakdown

### Stage 1: Model Training & Monthly Partition Generation

**Purpose**: Train GeoRF/XGBoost/GeoDT models for each month, generating baseline partitions.

**Script**: `run_batches_2021_2024_visual_monthly.bat <model_type>`

Unified batch script that accepts a model type argument:
```batch
run_batches_2021_2024_visual_monthly.bat georf    # Random Forest
run_batches_2021_2024_visual_monthly.bat geoxgb   # XGBoost
run_batches_2021_2024_visual_monthly.bat geodt    # Decision Tree
run_batches_2021_2024_visual_monthly.bat geodt --no-dt-rules  # DT without rule export
run_batches_2021_2024_visual_monthly.bat georf --fs0-only     # Stand-alone fs0 (lag-1) pipeline
```

**Configuration**:
```batch
# GeoRF/GeoXGB
YEARS_START=2021
YEARS_END=2024
FORECASTING_SCOPES=1,2,3  # fs1/fs2/fs3, default pipeline
                          # --fs0-only narrows this to SCOPE_LIST="0" (fs0, lag=1)
VISUAL=1  # Enable visualizations

# GeoDT (additional env vars)
SAVE_DT_RULES=1         # Export decision tree rules per partition
SAVE_DT_NODE_DUMP=0     # Optional detailed node dump
```

**Inputs**:
- Raw FEWSNET data
- Satellite imagery features
- Ground-based indicators

**Outputs** (per month):
```
result_GeoRF_YYYY_fsX_YYYY-MM_*/
├── correspondence_table_YYYY-MM.csv
├── space_partitions/
│   ├── s_branch.pkl
│   ├── X_branch_id.npy
│   └── partition files
└── results_df_gp_fsX_YYYY_YYYY.csv

result_GeoXGB_YYYY_fsX_YYYY-MM_*/
└── (same structure)

result_GeoDT_YYYY_fsX_YYYY-MM_visual/
├── correspondence_table_YYYY-MM.csv
├── vis/
│   ├── comprehensive_partition_metrics.csv
│   ├── final_f1_performance_map.png
│   ├── final_partition_map.png
│   └── score_details_*.csv
├── dt_rules/                          # DT-specific: exported tree rules
│   ├── dt_rules_manifest.csv
│   └── dt_rules_*.csv
└── log_print.txt

# GeoDT combined results (produced after all 12 months per year+scope):
results_df_dt_gp_fsX_YYYY_YYYY.csv    # Yearly combined
y_pred_test_dt_gp_fsX_YYYY_YYYY.csv   # Yearly combined predictions
```

**GeoDT-specific behavior**:
- Processes **one month at a time** to manage memory (144 total batches: 4 years x 3 scopes x 12 months)
- Archives visual files into `result_GeoDT_YYYY_fsX_YYYY-MM_visual/` folders before cleanup
- Combines 12 monthly CSVs into yearly result files after each year+scope completes
- Accumulates DT rules into a global `dt_rules/` directory with merged manifest

**Duration**: ~6-8 hours per model (full 4-year × 3-scope run)

---

### Stage 2: Consensus Clustering (Per-Model Experiment Directories)

**Purpose**: Aggregate monthly partitions into stable general and month-specific cluster assignments.

**Script**: `spatial_weighted_consensus_clustering.bat <model_type>`

Automated batch that orchestrates the shared/linking and clustering steps used by the current Stage 2 workflow, producing both general and month-specific (Feb, Jun, Oct) spatial partitions.

```batch
spatial_weighted_consensus_clustering.bat georf                # GeoRF partitions
spatial_weighted_consensus_clustering.bat geoxgb               # GeoXGB partitions
spatial_weighted_consensus_clustering.bat geodt                # GeoDT partitions
spatial_weighted_consensus_clustering.bat georf --fs0-only     # Stand-alone fs0 (Part 3 skipped)
```

The batch script runs three parts:
- **Part 1 (shared)**: step1 -> step3 (merge, link)
- **Part 2 (general)**: step4 -> step5 -> step6 (general partition, all months aggregated)
- **Part 3 (monthly)**: step4 -> step5 -> step6, repeated for months 2, 6, 10

> **fs0-only mode**: Part 3 is skipped entirely because fs0 alone does not produce enough candidate partitions for month-specific consensus clustering. Only the general partition is generated.

The batch uses refactored root-level scripts for step1, step4, and step5, plus `scripts\step3_create_linked_tables.py` and `scripts\step6_complete_clustering_pipeline.py`. `step2_load_correspondence.py` remains in the repo as a legacy helper, but the unified Stage 2 batch skips it because its `correspondence_tables_loaded.pkl` output is not consumed downstream.

> **Note**: Each model type has its own experiment directory. The root-level `step1*.ipynb` through `step6*.py` are **deprecated** legacy files retained for reference only.

> **Manual alternative**: You can still run steps individually from within the experiment directory using the Jupyter notebooks and Python scripts. See the Step 1-6 descriptions below.

| Model | Experiment Directory | Result Prefix | Results Subdirectory |
|-------|---------------------|---------------|----------------------|
| GeoRF | `GeoRFExperiment/` | `GeoRF_` | `GeoRFResults/` |
| GeoXGB | `GeoXGBExperiment/` | `GeoXGB_` | `GeoXgboostResults/` |
| GeoDT | `GeoDTExperiment/` | `GeoDT_` | `GeoDTResults/` |

Each experiment directory has the same internal structure:
```
Geo{Model}Experiment/
├── step1_merge_results_with_correspondence.ipynb
├── step2_load_correspondence_tables.ipynb
├── step3_create_linked_tables.py
├── step4_compute_similarity_matrix.ipynb
├── step5_sparsification_connectivity_check.ipynb
├── step6_complete_clustering_pipeline.py
├── FEWSNET_admin_code_lat_lon.csv
├── Geo{Model}Results/                    # Stage 1 result CSVs (copied here)
│   ├── results_df_*_fsX_YYYY_YYYY.csv
│   └── y_pred_test_*_fsX_YYYY_YYYY.csv
├── merged_correspondence_tables.pkl      # Step 1 output
├── merged_correspondence_tables/         # Step 1 output (per-plan CSVs)
├── linked_tables/                        # Step 3 output
│   ├── main_index.csv
│   ├── table_links.csv
│   ├── summary_report.txt
│   └── partitions/
├── similarity_matrices/                  # Step 4 output (general)
├── similarity_matrices_m02/              # Step 4 output (Feb, if generated)
├── similarity_matrices_m06/              # Step 4 output (Jun, if generated)
├── similarity_matrices_m10/              # Step 4 output (Oct, if generated)
└── knn_sparsification_results/           # Step 5-6 output
    ├── knn_graph_k40.npz
    ├── connected_components_k40.npz
    ├── knn_analysis_report_k40.json
    ├── cluster_mapping_k40_nc{N}_general.csv
    └── final_cluster_labels_k40_nc{N}.npz
```

All steps below are run **from within** the experiment directory (e.g., `cd GeoRFExperiment/`). The description applies identically to all three model types; only the naming prefix and input result files differ.

#### Step 1: Merge Results with Correspondence Tables

**File**: `step1_merge_results_with_correspondence.ipynb`

**Purpose**: Combine model results with correspondence tables that map admin codes to partition IDs.

**Inputs**:
- `result_Geo{Model}_*/correspondence_table_*.csv` (from Stage 1, referenced or copied into experiment dir)
- `Geo{Model}Results/results_df_*.csv` (performance metrics)

**Outputs**:
- `merged_correspondence_tables.pkl`
- `merged_correspondence_tables/Geo{Model}_YYYY_MM_fsX_merged.csv` (one per plan)

**Key Operations**:
- Extract metadata (year, month, forecasting_scope) from directory/file names
- Merge performance metrics with partition assignments
- Create unified format across all months

#### Step 2: Load Correspondence Tables (Legacy / Manual Only)

**File**: `step2_load_correspondence_tables.ipynb`

**Purpose**: Validate and prepare correspondence tables for clustering in the older notebook-driven flow.

**Outputs**:
- `correspondence_tables_loaded.pkl`

> The unified `spatial_weighted_consensus_clustering.bat` workflow does **not** run this step because `correspondence_tables_loaded.pkl` is not consumed by downstream scripts. Keep it only for manual inspection or notebook-based troubleshooting.

#### Step 3: Create Linked Tables

**File**: `step3_create_linked_tables.py`

**Purpose**: Create indexed tables linking partitions to performance metrics.

**Inputs**:
- `merged_correspondence_tables.pkl`

**Outputs**:
```
linked_tables/
├── main_index.csv (36 rows per single-model experiment)
├── table_links.csv
├── summary_report.txt
└── partitions/
    ├── Geo{Model}_2021_02_fs1_partition.csv
    ├── Geo{Model}_2021_02_fs2_partition.csv
    └── ... (~36 partition tables)
```

**Main Index Columns**:
- `name`: Unique partition identifier (e.g., `GeoRF_2021_02_fs1`)
- `year`, `month`, `forecasting_scope`
- `f1(1)`, `f1_base(1)`: Performance metrics for weighting
- `partition_file`: Link to partition CSV

#### Step 4: Compute Similarity Matrix

**File**: `step4_compute_similarity_matrix.ipynb`

**Purpose**: Compute weighted similarity between admin units based on co-grouping frequency across partitions.

**Key Options**:
```python
# General partition (all months)
--month None

# Month-specific partition (e.g., February)
--month 2
--month 6
--month 10
```

**Inputs**:
- `linked_tables/main_index.csv`
- `linked_tables/partitions/*.csv`
- `FEWSNET_admin_code_lat_lon.csv`

**Algorithm**:
1. **Plan Weighting**: `w = logit(f1) - logit(f1_base)` (performance improvement)
2. **Co-grouping Matrix**: Weighted frequency of admin units grouped together
3. **Spatial Kernel**: Gaussian kernel based on haversine distance
4. **Normalization**: Row-normalize similarity matrix

**Outputs**:
- `similarity_matrices/similarity_matrices.npz` (general)
- `similarity_matrices_m02/similarity_matrices_m02.npz` (February, if generated)
- `similarity_matrices_m06/similarity_matrices_m06.npz` (June, if generated)
- `similarity_matrices_m10/similarity_matrices_m10.npz` (October, if generated)

**Matrix Properties**:
- Shape: (5718, 5718) for FEWSNET admin units
- Sparsity: ~98% (most pairs have zero similarity)
- Range: [0, 5] (normalized, non-negative)

#### Step 5: KNN Sparsification & Eigengap Analysis

**File**: `step5_sparsification_connectivity_check.ipynb`

**Purpose**: Build sparse KNN graph and determine optimal cluster count via eigengap.

**Inputs**:
- `similarity_matrices/similarity_matrices.npz`

**Algorithm**:
1. **KNN Graph**: Keep only k=40 nearest neighbors per admin unit
2. **Connectivity**: Analyze connected components
3. **Laplacian Eigenvalues**: Compute spectrum of graph Laplacian
4. **Eigengap**: Find largest gap between eigenvalues -> optimal cluster count

**Outputs**:
- `knn_sparsification_results/knn_graph_k40.npz`
- `knn_sparsification_results/connected_components_k40.npz`
- `knn_sparsification_results/knn_analysis_report_k40.json`
- Recommended cluster count (printed to console)

**Typical Results**:
- Cluster count: 2-10 (determined by eigengap; varies by model type)
- Main component: ~67% of admin units
- Minor components: Isolated regions

#### Step 6: Spectral Clustering

**File**: `step6_complete_clustering_pipeline.py`

**Purpose**: Apply spectral clustering to generate final partition assignments.

**Inputs**:
- `knn_sparsification_results/knn_graph_k40.npz`
- `knn_sparsification_results/connected_components_k40.npz`
- `similarity_matrices/similarity_matrices.npz`
- `--n-clusters` (from Step 5 eigengap analysis)

**Algorithm**:
1. **Spectral Clustering**: On main connected component
2. **Outlier Assignment**: Assign isolated nodes to nearest cluster
3. **Label Creation**: Map cluster IDs to admin codes

**Outputs**:
```
knn_sparsification_results/
├── cluster_mapping_k40_nc{N}_general.csv (general partition)
├── cluster_mapping_k40_nc{N}_m2.csv     (February-specific, if generated)
├── cluster_mapping_k40_nc{N}_m6.csv     (June-specific, if generated)
├── cluster_mapping_k40_nc{N}_m10.csv    (October-specific, if generated)
├── final_cluster_labels_k40_nc{N}.npz
└── knn_analysis_report_k40.json (updated with clustering results)
```

**Cluster Mapping Format**:
```csv
FEWSNET_admin_code,cluster_id,latitude,longitude,is_outlier
0,2,8.123,38.456,False
1,0,7.890,37.123,False
...
```

> **Note**: The cluster count `nc{N}` differs across models because eigengap analysis operates on model-specific similarity matrices. Observed values:
> - **GeoRF**: nc4 (general) — not yet available for month-specific
> - **GeoXGB**: nc4 (general), nc3 (Feb), nc1 (Jun), nc10 (Oct)
> - **GeoDT**: nc6 (general), nc9 (Feb), nc7 (Jun), nc8 (Oct)

---

### Stage 3: Partitioned Model Comparison

**Purpose**: Re-run models using generated partitions to evaluate partitioned vs pooled performance.

**Script**: `run_partition_k40_comparison_unified.bat <model_type> [options]`

Unified comparison script that auto-discovers partition files from `cluster_mapping_manifest.json` (written by Stage 2).

```batch
run_partition_k40_comparison_unified.bat georf                # Default settings
run_partition_k40_comparison_unified.bat geoxgb --visual      # With maps
run_partition_k40_comparison_unified.bat geodt 1 3 --visual   # Contiguity + maps
run_partition_k40_comparison_unified.bat georf --fs0-only     # Stand-alone fs0 run
```

**Options**:
- First numeric arg = `CONTIGUITY` (0 or 1, default 1)
- Second numeric arg = `REFINE_ITERS` (default 3)
- `--visual` / `-v` = Enable visualization maps
- `--month-ind` = Enable month-specific partitions (enabled by default when `MONTH_IND=1`)
- `--scope N` = Run only scope N, where N ∈ {0, 1, 2, 3}. Default: all of 1..3.
- `--fs0-only` = Stand-alone fs0 mode. Forces `SCOPES=0` and clears `--month-ind` regardless of other flags. Aggregation writes to `Table_Format_fs0.xlsx` instead of `Table_Format.xlsx`.

**Auto-discovery**: The script looks for `{ExperimentDir}/knn_sparsification_results/cluster_mapping_manifest.json` to locate partition files. Falls back to pattern matching (`cluster_mapping_k40_nc*_general.csv`) if manifest is missing.

**Configuration** (defaults inside batch, overridable via args):
```batch
START_MONTH=2021-01
END_MONTH=2024-12
TRAIN_WINDOW=36
FORECASTING_SCOPE=2
MONTH_IND=1          # Month-specific partitions enabled
CONTIGUITY=1         # Contiguity refinement enabled
REFINE_ITERS=3       # Refinement iterations
```

**Comparisons**:
1. **Pooled**: Single model trained on all data
2. **Partitioned**: Separate model per cluster
3. **General**: Year-round partition (k40_nc*.csv)
4. **Month-Specific**: Season-specific partitions (m02, m06, m10)

**Outputs** (per model type):
- `metrics_monthly.csv` - Monthly performance comparison
- `predictions_monthly.csv` - Monthly predictions
- `metrics_admin0_overall.csv` - Country-level metrics
- `run_manifest.json` - Run configuration record
- F1 improvement plots (with `--visual` flag)
- Cluster-wise metrics

**Typical Results**:
- Partitioned models: +5-15% F1 improvement over pooled
- Month-specific: +2-5% additional improvement over general
- Spatial heterogeneity captured by clustering

---

## File Naming Conventions

### Result Directories
```
result_GeoRF_{YEAR}_fs{SCOPE}_{YEAR}-{MONTH}_{TIMESTAMP}/
result_GeoXGB_{YEAR}_fs{SCOPE}_{YEAR}-{MONTH}_{TIMESTAMP}/
result_GeoDT_{YEAR}_fs{SCOPE}_{YEAR}-{MONTH}_visual/
```

Examples:
- `result_GeoRF_2024_fs1_2024-02_20260203_103045/`
- `result_GeoDT_2024_fs1_2024-02_visual/` (GeoDT uses `_visual` suffix instead of timestamp)

### Partition Files
```
# General partition (all months)
cluster_mapping_k{K}_nc{NC}.csv

# Month-specific partitions
cluster_mapping_k{K}_nc{NC}_m{MONTH}.csv
```

Example:
- `cluster_mapping_k40_nc4.csv` (4 clusters, k=40 neighbors)
- `cluster_mapping_k40_nc4_m02.csv` (February-specific, 4 clusters)

### Correspondence Tables
```
correspondence_table_{YEAR}-{MONTH}.csv
```

Example: `correspondence_table_2024-02.csv`

---

## Key Parameters

### Similarity Matrix Computation
- `k = 40`: Number of nearest neighbors for KNN graph
- `sigma = 5.0`: Spatial kernel bandwidth (degrees)
- `n_eigenvalues = 20`: Number of eigenvalues to compute

### Cluster Count Determination
- **Automatic (Eigengap)**: Algorithm detects optimal count
- **Manual Override**: `--n-clusters` parameter in Step 6
- **Typical Range**: 2-6 clusters for FEWSNET data

### Weighting Scheme
```python
plan_weight = logit(f1_partitioned) - logit(f1_baseline)
```
- Positive weight: Partition improves over baseline
- Negative weight: Partition worse than baseline (truncated to 0)
- Logit transformation: Unbounded scale for performance differences

---

## Best Practices

### Data Preparation
1. **Ensure Complete Runs**: Stage 1 must complete all months before Stage 2
2. **Check File Counts**: Verify ~36 partition files in `Geo{Model}Experiment/linked_tables/partitions/` per model
3. **Validate Correspondence**: Check that admin codes match across files

### Partition Generation
1. **General Partition First**: Always generate general partition before month-specific
2. **Eigengap Inspection**: Manually inspect eigenvalue plots to validate cluster count
3. **Connectivity Check**: Verify main component has >50% of admin units

### Quality Checks
1. **Similarity Matrix**: Check sparsity is ~98%, range is [0, ~5]
2. **Cluster Balance**: Ensure clusters have >100 admin units each (avoid tiny clusters)
3. **Geographic Coherence**: Visualize clusters on map to check spatial contiguity

### Common Issues
1. **Missing Files**: If Step 1 fails, check that Stage 1 results exist
2. **Encoding Errors**: All Python scripts now use ASCII-safe characters for Windows GBK
3. **Memory Issues**: Step 4 (distance matrix) requires ~8GB RAM for 5718 admin units

---

## Performance Expectations

### Runtime (per model type)
- **Stage 1**: 6-8 hours (4 years x 3 scopes x 12 months)
- **Stage 2 (Steps 1-6)**: 30-60 minutes (automated via batch)
- **Stage 3**: 4-6 hours (re-training with partitions)
- **Full pipeline (all 3 models)**: ~30-50 hours total

### Resource Requirements
- **CPU**: 32 cores recommended (N_JOBS=32)
- **RAM**: 16GB minimum, 32GB recommended
- **Disk**: ~50GB per model for full 4-year results (~150GB total for all 3 models)

### Expected Improvements
- **Partitioned vs Pooled**: +5-15% F1 score
- **Month-Specific vs General**: +2-5% F1 score
- **Spatial Clustering**: Captures regional heterogeneity

---

## References

- **Consensus Clustering**: Monti et al. (2003) - Consensus clustering
- **Spectral Clustering**: Ng et al. (2002) - Spectral clustering
- **Spatial Weighting**: Tobler's First Law of Geography
- **GeoRF Framework**: Xie et al. (2024) - Spatial partitioning for crisis prediction
