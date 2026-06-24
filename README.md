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
comparison. GeoXGB, fs0 lag-1, and 2026-2027 forward/scenario prediction
workflows are experimental extensions and are not part of the paper
reproducibility package.

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

Then verify the live repository result bundle:

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
run_batches_2018_2020_partition_learning_visual_monthly.bat geodt
```

Output: yearly combined `results_df_*_fsN_YYYY_YYYY.csv` /
`y_pred_test_*_fsN_YYYY_YYYY.csv` plus archived
`result_Geo{Model}_YYYY_fsN_YYYY-MM_visual/` folders used by Stage 2.

### Stage 2: Generate fixed consensus partitions

```batch
spatial_weighted_consensus_clustering.bat georf
spatial_weighted_consensus_clustering.bat geodt
```

Output: `cluster_mapping_k40_nc*_general.csv`, `_m2.csv`, `_m6.csv`, `_m10.csv`
and `cluster_mapping_manifest.json`. `k40` is the KNN graph-neighbor parameter,
not 40 clusters. The selected cluster count is the `nc*` token.

### Stage 3: Evaluate 2021-2024 fixed partitions

```batch
run_partition_k40_comparison_unified.bat georf --visual --month-ind
run_partition_k40_comparison_unified.bat geodt --visual --month-ind
```

Output: `result_partition_k40_compare_{GF,DT}_fsN/` plus aggregated tables in
`other_outputs/Table_Format.xlsx` and `other_outputs/Model_Comparison_Table.xlsx`.
GeoRF is the paper-facing model. GeoDT is kept for appendix comparison and
branch-level interpretability. `run_partition_k40_comparison_unified.bat all`
is still available for local convenience, but paper reproduction should treat
GeoRF as the main model family.

## Experimental and Extension Workflows

These workflows remain in the repository for development continuity but are not
part of the current paper reproducibility package. They should not be interpreted
as manuscript main results. Paths and scripts are not moved in this task.

### GeoXGB

`app/main_model_XGB.py`, `GeoXGBExperiment/`, and XGBoost comparison scripts are
legacy/experimental. They are not included in the no-leak paper workflow or the
lightweight package.

### fs0 Lag-1 Extension

fs0 is a stand-alone lag-1 extension. It is orthogonal to the paper's fs1/fs2/fs3
4-, 8-, and 12-month workflow and is not included in the paper reproducibility
package.

```batch
run_batches_2018_2020_partition_learning_visual_monthly.bat <model> --fs0-only
spatial_weighted_consensus_clustering.bat <model> --fs0-only
run_partition_k40_comparison_unified.bat <model> --fs0-only
```

Keep `<model>` as `georf` or `geodt`. Running fs0 requires the flag on all three
stages and writes separate fs0 workbooks, so it must not be mixed with the paper
fs1/fs2/fs3 outputs. In fs0-only mode, Stage 2 generates only the general
consensus partition because fs0 alone does not yield enough candidate partitions
for month-specific maps.

### 2026-2027 Forward and Scenario Prediction

The `prediction_pipeline/` launchers support GeoRF-only forward prediction for
June 2026 and February 2027 plus a separate synthetic scenario overlay. These
outputs are operational extensions, not manuscript backtest results.

```batch
prediction_pipeline\spatial_weighted_consensus_clustering_predict.bat georf
prediction_pipeline\run_predict_2026_2027.bat georf
prediction_pipeline\run_partition_predict_unified.bat georf
prediction_pipeline\run_scenario_predict_jun2026_feb2027.bat
```

Standard forward outputs go to `deliverables\predict_2026_2027\`. Synthetic
scenario outputs go to `deliverables\predict_scenario_jun2026_feb2027\` and
should not be described as standard manuscript forecasts.

## Key Features

### Spatial Partitioning
- **Hierarchical clustering** of admin units based on model behavior similarity
- **Consensus-based** aggregation across 70+ monthly partition plans
- **Spatial weighting** using haversine distance kernel
- **Spectral clustering** with eigengap-based cluster count selection

### Main Model Types
- **GeoRF**: Random Forest with spatial partitioning
- **GeoDT**: Decision Tree with spatial partitioning
- Both share identical partitioning logic for fair comparison

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
│   ├── main_model_XGB.py         # Legacy/experimental GeoXGB script
│   └── main_model_DT.py          # GeoDT main script
├── src/
│   ├── model/                    # GeoRF/GeoDT plus legacy GeoXGB adapters
│   ├── partition/                # Spatial partitioning algorithms
│   ├── preprocess/               # Data loading & cleaning
│   └── vis/                      # Visualization
├── scripts/
│   ├── step1_merge_results.py    # Clustering step 1 (refactored)
│   ├── step3_create_linked_tables.py # Clustering step 3
│   ├── step4_similarity_matrix.py    # Clustering step 4 (refactored)
│   ├── step5_sparsification.py       # Clustering step 5 (refactored)
│   ├── step6_complete_clustering_pipeline.py  # Clustering step 6
│   ├── predict_partitioned_2026_2027.py       # Standalone GeoRF prediction
│   ├── predict_scenario_2026_2027.py          # Synthetic scenario overlay
│   └── compare_partitioned_vs_pooled_*.py  # Stage 3 comparison scripts
├── paper_reproducibility_package/ # Fast paper artifact audit package
├── prediction_pipeline/          # Experimental 2026-2027 prediction launchers
├── GeoRFExperiment/              # GeoRF clustering workspace
├── GeoXGBExperiment/             # Legacy/experimental GeoXGB workspace
├── GeoDTExperiment/              # GeoDT clustering workspace
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
result_Geo{RF,DT}_YYYY_fsN_YYYY-MM_visual/
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
- Aggregated workbooks: `other_outputs/Table_Format.xlsx` / `Model_Comparison_Table.xlsx`
- Experimental fs0-only workbooks: `other_outputs/Table_Format_fs0.xlsx` / `Model_Comparison_Table_fs0.xlsx`

## Known Issues

**Legacy Files**:
- The end-to-end pipeline (`run_full_ablation.bat`) has been deprecated
- Root-level `step*.ipynb` / `step*.py` files are deprecated; use the unified batch scripts instead
- Model-specific batch files (`run_georf_batches_*`, `run_xgboost_batches_*`, etc.) have been superseded by the unified 3-stage scripts

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
