# GeoRF/XGBoost Spatial Clustering Pipeline Workflow

## Overview

This document describes the complete workflow for generating spatial partitions using the GeoRF/XGBoost framework with consensus clustering.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         STAGE 1: MODEL TRAINING                         │
│                      Generate Monthly Partition Results                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌─────────────────────────────────────────────────┐
        │  run_georf_batches_2021_2024_visual_monthly.bat │
        │  run_xgboost_batches_2021_2024_visual_monthly.bat│
        └─────────────────────────────────────────────────┘
                                    │
                    Generates monthly results:
                    - result_GeoRF_YYYY_fsX_YYYY-MM_*/
                    - result_GeoXGB_YYYY_fsX_YYYY-MM_*/
                    - correspondence_table_YYYY-MM.csv
                    - partition files per month
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE 2-3: CONSENSUS CLUSTERING                      │
│         Generate General & Month-Specific Partitions (Manual)          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                        ┌───────────┴───────────┐
                        ▼                       ▼
            ┌───────────────────┐   ┌──────────────────────┐
            │  Python Scripts   │   │ Jupyter Notebooks    │
            │  (Automated)      │   │ (Interactive)        │
            └───────────────────┘   └──────────────────────┘
                        │                       │
         ┌──────────────┼───────────────────────┤
         │              │                       │
         ▼              ▼                       ▼
    step1*.py      step3*.py              step1*.ipynb
    step2*.py      step6*.py              step2*.ipynb
                                          step4*.ipynb
                                          step5*.ipynb
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
   Step 1:         Step 2:         Step 3:
   Merge          Load Corr.       Create
   Results        Tables          Linked Tables
        │               │               │
        └───────────────┴───────────────┘
                        │
                        ▼
                   Step 4:
            Compute Similarity Matrix
            (General or Month-Specific)
                        │
            ┌───────────┴────────────┐
            │                        │
            ▼                        ▼
       General                 Month-Specific
       (All months)           (Filter by month)
            │                        │
            └───────────┬────────────┘
                        ▼
                   Step 5:
          KNN Sparsification & Eigengap
                        │
                        ▼
                   Step 6:
             Spectral Clustering
                        │
          Generates partition files:
          - cluster_mapping_k40_nc4.csv (general)
          - cluster_mapping_k40_nc4_m02.csv (Feb)
          - cluster_mapping_k40_nc4_m06.csv (Jun)
          - cluster_mapping_k40_nc4_m10.csv (Oct)
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                STAGE 4: PARTITIONED MODEL COMPARISON                    │
│          Re-run Models with Generated Spatial Partitions                │
└─────────────────────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┴────────────────┐
        ▼                                ▼
┌─────────────────────┐      ┌─────────────────────────┐
│ run_partition_k40_  │      │ run_partition_k40_      │
│ clustered_          │      │ clustered_              │
│ comparison_GF.bat   │      │ comparison_XGB.bat      │
└─────────────────────┘      └─────────────────────────┘
        │                                │
        └────────────┬───────────────────┘
                     ▼
         Final comparison results:
         - Pooled vs Partitioned
         - General vs Month-Specific
         - Performance metrics & plots
```

## Detailed Stage Breakdown

### Stage 1: Model Training & Monthly Partition Generation

**Purpose**: Train GeoRF/XGBoost models for each month, generating baseline partitions.

**Scripts**:
- `run_georf_batches_2021_2024_visual_monthly.bat`
- `run_xgboost_batches_2021_2024_visual_monthly.bat`

**Configuration**:
```batch
YEARS_START=2021
YEARS_END=2024
FORECASTING_SCOPES=1,2,3
VISUAL=1  # Enable visualizations
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
```

**Duration**: ~6-8 hours (full 4-year × 3-scope run)

---

### Stage 2-3: Consensus Clustering (Manual Workflow)

**Purpose**: Aggregate monthly partitions into stable general and month-specific cluster assignments.

**Tools**: Jupyter notebooks (.ipynb) or Python scripts (.py)

#### Step 1: Merge Results with Correspondence Tables

**Files**:
- `step1_merge_results_with_correspondence.ipynb`
- `step1_merge_results_with_correspondence.py` (if automated)

**Purpose**: Combine model results with correspondence tables that map admin codes to partition IDs.

**Inputs**:
- `result_GeoRF_*/correspondence_table_*.csv`
- `result_GeoXGB_*/correspondence_table_*.csv`
- `results_df_*.csv` (performance metrics)

**Outputs**:
- `merged_correspondence_tables.pkl`
- `merged_correspondence_tables/*.csv` (one per month)

**Key Operations**:
- Extract metadata (year, month, forecasting_scope) from directory names
- Merge performance metrics with partition assignments
- Create unified format across all months

#### Step 2: Load Correspondence Tables

**Files**:
- `step2_load_correspondence_tables.ipynb`
- `step2_load_correspondence_tables.py`

**Purpose**: Validate and prepare correspondence tables for clustering.

**Outputs**:
- `correspondence_tables_loaded.pkl`

#### Step 3: Create Linked Tables

**Files**:
- `step3_create_linked_tables.py` (script only)

**Purpose**: Create indexed tables linking partitions to performance metrics.

**Inputs**:
- `merged_correspondence_tables.pkl`

**Outputs**:
```
linked_tables/
├── main_index.csv (70-72 rows: GeoRF + XGBoost × months × scopes)
├── table_links.csv
└── partitions/
    ├── GeoRF_2021_02_fs1_partition.csv
    ├── GeoRF_2021_02_fs2_partition.csv
    └── ... (70-72 files)
```

**Main Index Columns**:
- `name`: Unique partition identifier
- `year`, `month`, `forecasting_scope`
- `f1(1)`, `f1_base(1)`: Performance metrics for weighting
- `partition_file`: Link to partition CSV

#### Step 4: Compute Similarity Matrix

**Files**:
- `step4_compute_similarity_matrix.ipynb`

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
- `similarity_matrices_m02/similarity_matrices_m02.npz` (February)
- `similarity_matrices_m06/similarity_matrices_m06.npz` (June)
- `similarity_matrices_m10/similarity_matrices_m10.npz` (October)

**Matrix Properties**:
- Shape: (5718, 5718) for FEWSNET admin units
- Sparsity: ~98% (most pairs have zero similarity)
- Range: [0, 5] (normalized, non-negative)

#### Step 5: KNN Sparsification & Eigengap Analysis

**Files**:
- `step5_sparsification_connectivity_check.ipynb`

**Purpose**: Build sparse KNN graph and determine optimal cluster count via eigengap.

**Inputs**:
- `similarity_matrices/similarity_matrices.npz`

**Algorithm**:
1. **KNN Graph**: Keep only k=40 nearest neighbors per admin unit
2. **Connectivity**: Analyze connected components
3. **Laplacian Eigenvalues**: Compute spectrum of graph Laplacian
4. **Eigengap**: Find largest gap between eigenvalues → optimal cluster count

**Outputs**:
- `knn_sparsification_results/knn_graph_k40.npz`
- `knn_sparsification_results/connected_components_k40.npz`
- `knn_sparsification_results/knn_analysis_report_k40.json`
- Recommended cluster count (printed to console)

**Typical Results**:
- Cluster count: 2-6 (determined by eigengap)
- Main component: ~67% of admin units
- Minor components: Isolated regions

#### Step 6: Spectral Clustering

**Files**:
- `step6_complete_clustering_pipeline.py` (script only)

**Purpose**: Apply spectral clustering to generate final partition assignments.

**Inputs**:
- `knn_graph_k40.npz`
- `connected_components_k40.npz`
- `similarity_matrices.npz`
- `--n-clusters` (from Step 5 eigengap analysis)

**Algorithm**:
1. **Spectral Clustering**: On main connected component
2. **Outlier Assignment**: Assign isolated nodes to nearest cluster
3. **Label Creation**: Map cluster IDs to admin codes

**Outputs**:
```
knn_sparsification_results/
├── cluster_mapping_k40_nc4.csv (general partition)
├── cluster_labels_k40_nc4.npy
└── knn_analysis_report_k40.json (updated with clustering results)

similarity_matrices_m02/
└── cluster_mapping_k40_nc4_m02.csv (February-specific)

similarity_matrices_m06/
└── cluster_mapping_k40_nc4_m06.csv (June-specific)

similarity_matrices_m10/
└── cluster_mapping_k40_nc4_m10.csv (October-specific)
```

**Cluster Mapping Format**:
```csv
admin_code,cluster_id
2901,0
2902,1
2903,0
...
```

---

### Stage 4: Partitioned Model Comparison

**Purpose**: Re-run models using generated partitions to evaluate partitioned vs pooled performance.

**Scripts**:
- `run_partition_k40_clustered_comparison_GF.bat`
- `run_partition_k40_clustered_comparison_XGB.bat`

**Configuration**:
```batch
# Partition to use
PARTITION_FILE=cluster_mapping_k40_nc4.csv (general)
# or
PARTITION_FILE=cluster_mapping_k40_nc4_m02.csv (February-specific)

# Comparison mode
COMPARISON=partitioned_vs_pooled
```

**Comparisons**:
1. **Pooled**: Single model trained on all data
2. **Partitioned**: Separate model per cluster
3. **General**: Year-round partition (k40_nc4.csv)
4. **Month-Specific**: Season-specific partitions (m02, m06, m10)

**Outputs**:
- Performance comparison CSVs
- F1 improvement plots
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
```

Example: `result_GeoRF_2024_fs1_2024-02_20260203_103045/`

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
2. **Check File Counts**: Verify 70-72 partition files in `linked_tables/partitions/`
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

### Runtime
- **Stage 1**: 6-8 hours (4 years × 3 scopes × 12 months)
- **Stage 2-3 (Steps 1-6)**: 30-60 minutes (manual, interactive)
- **Stage 4**: 4-6 hours (re-training with partitions)

### Resource Requirements
- **CPU**: 32 cores recommended (N_JOBS=32)
- **RAM**: 16GB minimum, 32GB recommended
- **Disk**: ~50GB for full 4-year results

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
