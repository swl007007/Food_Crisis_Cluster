# GeoRF Food Crisis Prediction with Spatial Consensus Clustering

Spatial transformation framework for food security crisis prediction using geo-aware machine learning (GeoRF/XGBoost) with consensus-based spatial partitioning.

## Overview

This project applies the GeoRF framework to **FEWSNET food crisis prediction** in Sub-Saharan Africa, using:
- **GeoRF**: Spatially-partitioned Random Forest
- **GeoXGB**: Spatially-partitioned XGBoost
- **Consensus Clustering**: Aggregate monthly partitions into stable spatial clusters
- **Month-Specific Partitions**: Season-aware clustering for improved temporal adaptation

## Quick Start

### Complete Workflow (3 Stages)

See **[PIPELINE_WORKFLOW.md](PIPELINE_WORKFLOW.md)** for detailed documentation. All stages use unified batch scripts that accept a model type argument (`georf`, `geoxgb`, or `geodt`).

**Stage 1**: Generate monthly partition results (~6-8 hours)
```batch
run_batches_2021_2024_visual_monthly.bat georf
run_batches_2021_2024_visual_monthly.bat geoxgb
```

**Stage 2**: Generate spatial partitions (automated, ~30-60 min)
```batch
spatial_weighted_consensus_clustering.bat georf
spatial_weighted_consensus_clustering.bat geoxgb
```
Output: `cluster_mapping_k40_nc*_general.csv`, `_m2.csv`, `_m6.csv`, `_m10.csv` + `cluster_mapping_manifest.json`

**Stage 3**: Evaluate partitioned models (~4-6 hours)
```batch
run_partition_k40_comparison_unified.bat georf --visual --month-ind
run_partition_k40_comparison_unified.bat geoxgb --visual --month-ind
```

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
│   └── main_model_XGB.py         # XGBoost main script
├── src/
│   ├── model/                    # GeoRF/XGBoost implementations
│   ├── partition/                # Spatial partitioning algorithms
│   ├── preprocess/               # Data loading & cleaning
│   └── vis/                      # Visualization
├── scripts/
│   ├── step1_merge_results.py    # Clustering step 1 (refactored)
│   ├── step2_load_correspondence.py  # Clustering step 2 (refactored)
│   ├── step4_similarity_matrix.py    # Clustering step 4 (refactored)
│   ├── step5_sparsification.py       # Clustering step 5 (refactored)
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
result_GeoRF_YYYY_fsX_YYYY-MM_*/
├── correspondence_table_YYYY-MM.csv
├── results_df_gp_fsX_YYYY_YYYY.csv
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

## Known Issues

**Legacy Files**:
- The end-to-end pipeline (`run_full_ablation.bat`) has been deprecated
- Root-level `step*.ipynb` / `step*.py` files are deprecated; use the unified batch scripts instead
- Model-specific batch files (`run_georf_batches_*`, `run_xgboost_batches_*`, etc.) have been superseded by unified versions

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
