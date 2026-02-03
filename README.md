# GeoRF Food Crisis Prediction with Spatial Consensus Clustering

Spatial transformation framework for food security crisis prediction using geo-aware machine learning (GeoRF/XGBoost) with consensus-based spatial partitioning.

## Overview

This project applies the GeoRF framework to **FEWSNET food crisis prediction** in Sub-Saharan Africa, using:
- **GeoRF**: Spatially-partitioned Random Forest
- **GeoXGB**: Spatially-partitioned XGBoost
- **Consensus Clustering**: Aggregate monthly partitions into stable spatial clusters
- **Month-Specific Partitions**: Season-aware clustering for improved temporal adaptation

## Quick Start

### Complete Workflow (4 Stages)

See **[PIPELINE_WORKFLOW.md](PIPELINE_WORKFLOW.md)** for detailed documentation.

**Stage 1**: Generate monthly partition results (~6-8 hours)
```batch
run_georf_batches_2021_2024_visual_monthly.bat
run_xgboost_batches_2021_2024_visual_monthly.bat
```

**Stage 2-3**: Generate spatial partitions (manual, ~30-60 min)
- Run Jupyter notebooks: `step1*.ipynb` → `step2*.ipynb` → `step4*.ipynb` → `step5*.ipynb`
- Run Python scripts: `step3_create_linked_tables.py` → `step6_complete_clustering_pipeline.py`
- Output: `cluster_mapping_k40_nc4.csv` (general partition)

**Stage 4**: Evaluate partitioned models (~4-6 hours)
```batch
run_partition_k40_clustered_comparison_GF.bat
run_partition_k40_clustered_comparison_XGB.bat
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
- Both share identical partitioning logic for fair comparison

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
│   └── statlearn/                # Consensus clustering scripts (deprecated)
├── step*.py                      # Clustering pipeline (Python)
├── step*.ipynb                   # Clustering pipeline (Jupyter)
├── run_georf_batches*.bat        # Batch processing scripts
└── run_partition*.bat            # Partition comparison scripts
```

## Documentation

- **[PIPELINE_WORKFLOW.md](PIPELINE_WORKFLOW.md)**: Complete 4-stage workflow
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

### Spatial Partitions (Stage 2-3)
```
cluster_mapping_k40_nc4.csv         # General partition (all months)
cluster_mapping_k40_nc4_m02.csv     # February partition
cluster_mapping_k40_nc4_m06.csv     # June partition
cluster_mapping_k40_nc4_m10.csv     # October partition
```

### Comparison Results (Stage 4)
- Partitioned vs pooled F1 comparisons
- Cluster-wise performance metrics
- Spatial visualization of clusters

## Known Issues

**Automated Pipeline (Deprecated)**:
- The end-to-end pipeline (`run_full_ablation.bat`) has been deprecated
- Use semi-automated workflow with manual Jupyter notebook steps instead

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
