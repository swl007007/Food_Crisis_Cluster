# GeoRF Food Crisis Forecasting Paper Reproducibility

This repository contains the manuscript-facing FEWS NET food-crisis forecasting
workflow built around GeoRF, a geo-aware Random Forest that learns spatially
heterogeneous predictive regimes. The current paper results use a no-leak
three-stage workflow: Stage 1 learns candidate partitions on 2018-2020, Stage 2
builds fixed consensus maps, and Stage 3 evaluates fixed partitions on
2021-2024 FEWS NET release months.

The paper scope covers 22 FEWS NET monitored countries across Africa, the
Middle East, Asia, and Latin America. It is not limited to one region. The main
model is GeoRF; GeoDT is retained as an auxiliary appendix and interpretability
comparison. GeoXGB, fs0 lag-1, and 2026-2027 forward/scenario prediction entry
points are archived under `archived/release_20260624_nonpaper_pipelines/` and
are not part of the paper reproducibility package.

## Quick Start for Paper Reproduction

For fast review, start with the lightweight package:

```text
paper_reproducibility_package/README.md
paper_reproducibility_package/MANIFEST.csv
paper_reproducibility_package/SHA256SUMS.txt
```

Validate it from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_paper_reproducibility_package.py
```

Then verify the current repository result bundle:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_current_results_reproducibility.py
```

See **[PIPELINE_WORKFLOW.md](PIPELINE_WORKFLOW.md)** for the full pipeline
walkthrough.

## Complete No-Leak Regeneration Path

The full regeneration path is slower than the package audit path because Stage 1
partition learning is expensive. Use it when you need to regenerate the complete
workflow rather than audit the packaged paper artifacts.

### Stage 1: Learn partition candidates on 2018-2020

```batch
run_batches_2018_2020_partition_learning_visual_monthly.bat georf
```

Output: yearly combined `results_df_*_fsN_YYYY_YYYY.csv` /
`y_pred_test_*_fsN_YYYY_YYYY.csv` plus archived
`result_GeoRF_YYYY_fsN_YYYY-MM_visual/` folders used by Stage 2.

### Stage 2: Generate fixed consensus partitions

```batch
spatial_weighted_consensus_clustering.bat georf
```

Output: `cluster_mapping_k40_nc*_general.csv`, `_m2.csv`, `_m6.csv`, `_m10.csv`
and `cluster_mapping_manifest.json`. `k40` is the KNN graph-neighbor parameter,
not 40 clusters. The selected cluster count is the `nc*` token.

### Stage 3: Evaluate 2021-2024 fixed partitions

```batch
run_partition_k40_comparison_unified.bat georf --visual --month-ind
```

GeoDT result directories and figures are retained in the reproducibility-input
archive as appendix and interpretability provenance, but GeoDT is not part of
the release quickstart.
Non-paper workflows are archived under
`archived/release_20260624_nonpaper_pipelines/`.

Output: archived Stage 3 result folders under
`archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_fsN/`.
Paper-facing aggregate tables are in
`final_artifacts_in_paper_updated/01_main_results/` and the
`paper_reproducibility_package/` copy.

## Non-Paper Workflow Archive

GeoXGB, fs0 lag-1 launch guidance, 2026-2027 forward/scenario prediction,
legacy notebooks, regional exploratory scripts, and legacy baseline/demo entry
points are preserved in:

```text
archived/release_20260624_nonpaper_pipelines/
```

Those files are historical provenance for the development repository. They are
not part of the release quickstart and are not maintained as runnable workflows
from their archived paths.

## Key Features

### Spatial Partitioning
- **Hierarchical clustering** of admin units based on model behavior similarity
- **Consensus-based** aggregation across 70+ monthly partition plans
- **Spatial weighting** using haversine distance kernel
- **Spectral clustering** with eigengap-based cluster count selection

### Main Model Types
- **GeoRF**: Random Forest with spatial partitioning
- **GeoDT**: Decision Tree appendix and interpretability provenance

### Partition Types
- **General Partition**: Year-round clustering (all months aggregated)
- **Month-Specific**: Season-aware partitions (February, June, October)
- **Adaptive**: Captures temporal variations in spatial patterns

## Performance

### Expected Improvements
- **Partitioned vs Pooled**: +5-15% F1 score
- **Month-Specific vs General**: +2-5% F1 score
- **GeoRF paper summary**: In the current manuscript table, partitioned GeoRF
  improves crisis-class F1 over pooled RF at 4-, 8-, and 12-month horizons.
  Use `final_artifacts_in_paper_updated/01_main_results/main_month_ind_cont3.xlsx`
  and `paper_reproducibility_package/PAPER_ARTIFACT_MAP.md` as the source of
  paper-facing numbers.

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
│   └── main_model_DT.py          # GeoDT main script
├── src/
│   ├── model/                    # GeoRF/GeoDT model adapters
│   ├── partition/                # Spatial partitioning algorithms
│   ├── preprocess/               # Data loading & cleaning
│   └── vis/                      # Visualization
├── scripts/
│   ├── step1_merge_results.py    # Clustering step 1 (refactored)
│   ├── step3_create_linked_tables.py # Clustering step 3
│   ├── step4_similarity_matrix.py    # Clustering step 4 (refactored)
│   ├── step5_sparsification.py       # Clustering step 5 (refactored)
│   ├── step6_complete_clustering_pipeline.py  # Clustering step 6
│   ├── paper_artifacts/          # Paper figure/table builders
│   └── compare_partitioned_vs_pooled_*.py  # Stage 3 comparison scripts
├── final_artifacts_in_paper_updated/ # Paper-facing figures/tables/PDF
├── paper_reproducibility_package/ # Fast paper artifact audit package
├── archived/
│   ├── release_20260624_reproducibility_inputs/ # Heavy local verifier/package inputs
│   ├── release_20260624_legacy_workspace/       # Historical helper scripts and outputs
│   └── release_20260624_nonpaper_pipelines/     # Non-paper workflow provenance
├── run_batches_2018_2020_partition_learning_visual_monthly.bat # Stage 1
├── spatial_weighted_consensus_clustering.bat        # Stage 2: clustering
└── run_partition_k40_comparison_unified.bat         # Stage 3: comparison
```

## Documentation

- **[PIPELINE_WORKFLOW.md](PIPELINE_WORKFLOW.md)**: Complete 3-stage workflow
- **[paper_reproducibility_package/README.md](paper_reproducibility_package/README.md)**: Fast paper package validation
- **[CLAUDE.md](CLAUDE.md)**: AI assistant guide with troubleshooting
- **[CRITICAL_PIPELINE_FIXES.md](CRITICAL_PIPELINE_FIXES.md)**: Known issues & fixes

## Configuration

Key parameters in `config.py`:

```python
ACTIVE_LAGS = (4, 8, 12)
TRAIN_WINDOW_MONTHS = 36
DATA_MODE = "unadjusted"
DATA_PATH = "FEWSNET_forecast_unadjusted_bm.csv"
CONTIGUITY = True
USE_ADJACENCY_MATRIX = True
K = 40
SIGMA = 5.0
RF_STAGE3 = {"n_estimators": 100, "max_depth": None, "random_state": 5, "n_jobs": 1}
```

`K=40` is the graph-neighbor count used during consensus clustering, not the
number of clusters. Stage 3 paper artifacts use contiguity-refined maps with
three refinement iterations.

## Key Outputs

### Paper Reproducibility Package
```
paper_reproducibility_package/
├── README.md
├── MANIFEST.csv
├── SHA256SUMS.txt
├── SOURCE_DATA.md
├── PAPER_ARTIFACT_MAP.md
├── CONSISTENCY_AUDIT.md
├── stage2_cluster_maps/
├── stage3_results/
├── paper_artifacts/
└── ablation/
```

### Monthly Results (Stage 1)
```
results_df_*_fsN_YYYY_YYYY.csv
y_pred_test_*_fsN_YYYY_YYYY.csv
result_GeoRF_YYYY_fsN_YYYY-MM_visual/
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
`k40` means 40 nearest neighbors in graph construction; the selected cluster count is recorded by `nc*`, not by `k40`.

### Comparison Results (Stage 3)
- Partitioned vs pooled F1 comparisons
- Cluster-wise performance metrics
- Spatial visualization of clusters
- Archived local result folders:
  `archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_fsN/`
- Paper-facing workbooks:
  `final_artifacts_in_paper_updated/01_main_results/main_month_ind_cont3.xlsx`
  and the package copy under `paper_reproducibility_package/`

## Known Issues

**Legacy Files**:
- The end-to-end pipeline (`run_full_ablation.bat`) has been deprecated
- Root-level `step*.ipynb` / `step*.py` files are deprecated; use the unified batch scripts instead
- Model-specific batch files (`run_georf_batches_*`, `run_xgboost_batches_*`, etc.) have been superseded by the unified GeoRF release scripts or preserved under `archived/release_20260624_nonpaper_pipelines/`

**Unicode Encoding**:
- Windows CMD-facing batch `echo` and status output must be ASCII-safe for Chinese locale (GBK encoding)
- Avoid emoji, box-drawing characters, smart quotes, special arrows, em dashes, and other non-ASCII punctuation unless encoding is explicitly set and documented

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
