# GeoRF/GeoXGB/GeoDT Food Crisis Prediction with Spatial Consensus Clustering

Spatial transformation framework for food security crisis prediction using geo-aware machine learning (GeoRF/GeoXGB/GeoDT) with consensus-based spatial partitioning.

## Overview

This project applies the GeoRF framework to **FEWSNET food crisis prediction** in Sub-Saharan Africa, using:
- **GeoRF**: Spatially-partitioned Random Forest
- **GeoXGB**: Spatially-partitioned XGBoost
- **GeoDT**: Spatially-partitioned Decision Tree
- **Consensus Clustering**: Aggregate monthly partitions into stable spatial clusters
- **Month-Specific Partitions**: Season-aware clustering for improved temporal adaptation

## Quick Start

### Complete Workflow (3 Stages)

See **[PIPELINE_WORKFLOW.md](PIPELINE_WORKFLOW.md)** for detailed documentation. All stages use unified batch scripts that accept a model type argument (`georf`, `geoxgb`, or `geodt`).

**Stage 1**: Generate monthly partition results (~6-8 hours)
```batch
run_batches_2021_2024_visual_monthly.bat georf
run_batches_2021_2024_visual_monthly.bat geoxgb
run_batches_2021_2024_visual_monthly.bat geodt
```
Output: yearly combined `results_df_*_fsN_YYYY_YYYY.csv` / `y_pred_test_*_fsN_YYYY_YYYY.csv` plus archived `result_Geo{Model}_YYYY_fsN_YYYY-MM_visual/` folders used by Stage 2.

**Stage 2**: Generate spatial partitions (automated, ~30-60 min)
```batch
spatial_weighted_consensus_clustering.bat georf
spatial_weighted_consensus_clustering.bat geoxgb
spatial_weighted_consensus_clustering.bat geodt
```
Output: `cluster_mapping_k40_nc*_general.csv`, `_m2.csv`, `_m6.csv`, `_m10.csv` + `cluster_mapping_manifest.json`

**Stage 3**: Evaluate partitioned models (~4-6 hours)
```batch
run_partition_k40_comparison_unified.bat georf --visual --month-ind
run_partition_k40_comparison_unified.bat geoxgb --visual --month-ind
run_partition_k40_comparison_unified.bat geodt --visual --month-ind
```
Output: `result_partition_k40_compare_{GF,XGB,DT}_fsN/` plus aggregated tables in `other_outputs/Table_Format.xlsx` and `other_outputs/Model_Comparison_Table.xlsx`

### Stand-Alone fs0 (Lag-1) Pipeline

fs0 is a separate forecasting scope with **lag = 1 month**. It is orthogonal to fs1/fs2/fs3 and is opt-in via `--fs0-only` on every stage. The default fs1+fs2+fs3 pipeline is unaffected whether or not you run fs0.

Run all three stages with the same `--fs0-only` flag:
```batch
run_batches_2021_2024_visual_monthly.bat <model> --fs0-only
spatial_weighted_consensus_clustering.bat <model> --fs0-only
run_partition_k40_comparison_unified.bat <model> --fs0-only
```
`<model>` is `georf`, `geoxgb`, or `geodt`. Keep the mode consistent across all three stages.

What changes in fs0-only mode:
- **Stage 1**: Runs 48 batches (4 years × 1 scope × 12 months) instead of 144. Produces `results_df_*_fs0_*.csv` and `result_Geo{Model}_*_fs0_*_visual/` archives only.
- **Stage 2**: Copies only fs0 artifacts; generates the **general** consensus partition only. Month-specific partitions (m2/m6/m10) are skipped because fs0 alone does not yield enough candidate partitions.
- **Stage 3**: Forces `SCOPES=0` and disables `--month-ind`; writes results to `result_partition_k40_compare_{GF,XGB,DT}_fs0/` and aggregates to `other_outputs/Table_Format_fs0.xlsx` (separate from the fs1/2/3 `Table_Format.xlsx`).

## Key Features

### Spatial Partitioning
- **Hierarchical clustering** of admin units based on model behavior similarity
- **Consensus-based** aggregation across 70+ monthly partition plans
- **Spatial weighting** using haversine distance kernel
- **Spectral clustering** with eigengap-based cluster count selection

### Model Types
- **GeoRF**: Random Forest with spatial partitioning
- **GeoXGB**: XGBoost with spatial partitioning
- **GeoDT**: Decision Tree with spatial partitioning
- All three share identical partitioning logic for fair comparison

### Partition Types
- **General Partition**: Year-round clustering (all months aggregated)
- **Month-Specific**: Season-aware partitions (February, June, October)
- **Adaptive**: Captures temporal variations in spatial patterns

## Performance

### Expected Improvements
- **Partitioned vs Pooled**: +5-15% F1 score
- **Month-Specific vs General**: +2-5% F1 score
- **Crisis Prediction F1**: 0.70-0.80 (partitioned XGBoost)

### Resource Requirements
- **CPU**: 32 cores recommended
- **RAM**: 16GB minimum, 32GB recommended
- **Runtime**: ~10-15 hours for full workflow
- **Storage**: ~50GB for 4-year results

## Directory Structure

```
Food_Crisis_Cluster/
├── app/
│   ├── main_model_GF.py          # GeoRF main script
│   ├── main_model_XGB.py         # GeoXGB main script
│   └── main_model_DT.py          # GeoDT main script
├── src/
│   ├── model/                    # GeoRF/GeoXGB/GeoDT implementations
│   ├── partition/                # Spatial partitioning algorithms
│   ├── preprocess/               # Data loading & cleaning
│   └── vis/                      # Visualization
├── scripts/
│   ├── step1_merge_results.py    # Clustering step 1 (refactored)
│   ├── step3_create_linked_tables.py # Clustering step 3
│   ├── step4_similarity_matrix.py    # Clustering step 4 (refactored)
│   ├── step5_sparsification.py       # Clustering step 5 (refactored)
│   ├── step6_complete_clustering_pipeline.py  # Clustering step 6
│   └── compare_partitioned_vs_pooled_*.py  # Stage 3 comparison scripts
├── GeoRFExperiment/              # GeoRF clustering workspace
├── GeoXGBExperiment/             # GeoXGB clustering workspace
├── GeoDTExperiment/              # GeoDT clustering workspace
├── run_batches_2021_2024_visual_monthly.bat        # Stage 1: model training
├── spatial_weighted_consensus_clustering.bat        # Stage 2: clustering
└── run_partition_k40_comparison_unified.bat         # Stage 3: comparison
```

## Documentation

- **[PIPELINE_WORKFLOW.md](PIPELINE_WORKFLOW.md)**: Complete 3-stage workflow
- **[CLAUDE.md](CLAUDE.md)**: AI assistant guide with troubleshooting
- **[CRITICAL_PIPELINE_FIXES.md](CRITICAL_PIPELINE_FIXES.md)**: Known issues & fixes

## Configuration

Key parameters in `config.py`:

```python
# Spatial
STEP_SIZE = 0.1                    # Grid cell size (degrees)
CONTIGUITY = True                  # Spatial refinement
USE_ADJACENCY_MATRIX = True        # True polygon boundaries

# Partitioning
MIN_DEPTH = 1                      # Minimum partition depth
MAX_DEPTH = 4                      # Maximum partition depth
MIN_BRANCH_SAMPLE_SIZE = 5         # Minimum samples per partition

# Consensus Clustering
K = 40                             # KNN graph neighbors
SIGMA = 5.0                        # Spatial kernel bandwidth
```

## Key Outputs

### Monthly Results (Stage 1)
```
results_df_*_fsN_YYYY_YYYY.csv
y_pred_test_*_fsN_YYYY_YYYY.csv
result_Geo{RF,XGB,DT}_YYYY_fsN_YYYY-MM_visual/
├── correspondence_table_YYYY-MM.csv
├── vis/
└── space_partitions/
```

### Spatial Partitions (Stage 2)
```
{ExperimentDir}/knn_sparsification_results/
├── cluster_mapping_k40_nc*_general.csv   # General partition (all months)
├── cluster_mapping_k40_nc*_m2.csv        # February partition
├── cluster_mapping_k40_nc*_m6.csv        # June partition
├── cluster_mapping_k40_nc*_m10.csv       # October partition
└── cluster_mapping_manifest.json          # Paths to all partition files
```

### Comparison Results (Stage 3)
- Partitioned vs pooled F1 comparisons
- Cluster-wise performance metrics
- Spatial visualization of clusters
- Aggregated workbooks: `other_outputs/Table_Format.xlsx` / `Model_Comparison_Table.xlsx`
- fs0-only workbooks: `other_outputs/Table_Format_fs0.xlsx` / `Model_Comparison_Table_fs0.xlsx`

## Known Issues

**Legacy Files**:
- The end-to-end pipeline (`run_full_ablation.bat`) has been deprecated
- Root-level `step*.ipynb` / `step*.py` files are deprecated; use the unified batch scripts instead
- Model-specific batch files (`run_georf_batches_*`, `run_xgboost_batches_*`, etc.) have been superseded by the unified 3-stage scripts

**Unicode Encoding**:
- Fixed for Windows Chinese locale (GBK encoding)
- All print statements use ASCII-safe characters

## Citation

If you use this framework, please cite:

**GeoRF Framework**:
```
Xie, Y., Nhu, A., Song, X.-P., Jia, X., Skakun, S., Li, H., & Wang, Z. (2024).
Accounting for Spatial Variability with Geo-aware Random Forest:
A Case Study for US Major Crop Mapping.
Remote Sensing of Environment, 2024.
```

**Consensus Clustering**:
```
Monti, S., Tamayo, P., Mesirov, J., & Golub, T. (2003).
Consensus clustering: a resampling-based method for class discovery
and visualization of gene expression microarray data.
Machine Learning, 52(1-2), 91-118.
```

## Contact

For questions or issues, please refer to:
- **[CLAUDE.md](CLAUDE.md)** for AI assistant guidance
- **[PIPELINE_WORKFLOW.md](PIPELINE_WORKFLOW.md)** for workflow details
- `.ai/issues/` directory for known issues

## License

This project extends the GeoRF framework for food security applications.
Original GeoRF code: [https://github.com/yiqun-geo/STAR](https://github.com/yiqun-geo/STAR)
