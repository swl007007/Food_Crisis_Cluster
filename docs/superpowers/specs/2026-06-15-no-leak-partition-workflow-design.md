# No-Leak Partition Workflow Design

Date: 2026-06-15
Status: Draft for user review

## Problem

The current main workflow has temporal leakage. Stage 1 generates partition candidates on 2021-2024 monthly tests. Stage 2 learns optimized consensus partitions from those Stage 1 outputs. Stage 3 then evaluates partitioned vs pooled models on the same 2021-2024 period. Even though Stage 3 retrains lower models, the partition selection already saw the final evaluation horizon.

The main workflow must instead learn partitions only from pre-evaluation years, then apply those fixed partitions to later held-out years.

## User Decisions

- Main partition-learning period: 2018-2020.
- Main final-evaluation period: 2021-2024.
- Main results use global `FEWSNET_forecast_unadjusted_bm.csv` and the global FEWSNET admin-boundary shapefile.
- `NGA_only` was a prior small experiment and is not part of main results.
- GeoXGB is too premature and is excluded from main results.
- Main workflow model set is GeoRF and GeoDT only.
- Existing output filename/schema patterns should be preserved where practical.
- Legacy main workflow names such as `run_batches_2021_2024_visual_monthly.bat` should not remain as active project workflow paths.

## Proposed Workflow

### Stage 1: Partition Candidate Learning

Replace the active Stage 1 entrypoint with a no-leak partition-learning script, for example:

```text
run_batches_2018_2020_partition_learning_visual_monthly.bat <model> [--fs0-only] [--no-dt-rules]
```

Allowed main models:

```text
georf
geodt
```

The Stage 1 script keeps the existing monthly-by-month pattern but loops 2018, 2019, and 2020 instead of 2021-2024. It still sets `DESIRED_TERMS=YYYY-MM`, still calls the model entrypoint with `--start_year`, `--end_year`, and `--forecasting_scope`, and still writes the same style of result and visual archive artifacts:

```text
results_df_gp_fsN_YYYY_YYYY.csv
results_df_dt_gp_fsN_YYYY_YYYY.csv
y_pred_test_gp_fsN_YYYY_YYYY.csv
y_pred_test_dt_gp_fsN_YYYY_YYYY.csv
result_GeoRF_YYYY_fsN_YYYY-MM_visual/
result_GeoDT_YYYY_fsN_YYYY-MM_visual/
```

The naming pattern stays compatible with Stage 2, but the years represented by those files are now 2018-2020 for partition learning.

### Stage 2: Partition Optimization

`spatial_weighted_consensus_clustering.bat` continues to produce:

```text
knn_sparsification_results/cluster_mapping_manifest.json
cluster_mapping_k40_nc*_general.csv
cluster_mapping_k40_nc*_m2.csv
cluster_mapping_k40_nc*_m6.csv
cluster_mapping_k40_nc*_m10.csv
```

For main results, Stage 2 must consume only 2018-2020 Stage 1 outputs. It must not consume 2021-2024 Stage 1 outputs when building partitions for final evaluation.

The script should advertise and validate only `georf` and `geodt` for the main workflow. Existing GeoXGB implementation files may remain as non-main experimental code, but GeoXGB should not be part of main workflow usage text, model loops, aggregation, or paper artifact generation.

### Stage 3: Final Evaluation

`run_partition_k40_comparison_unified.bat` remains the final evaluation entrypoint but must use partitions learned from the 2018-2020 Stage 2 run. Its evaluation window remains:

```text
START_MONTH=2021-01
END_MONTH=2024-12
```

It should continue writing the same Stage 3 output shapes:

```text
result_partition_k40_compare_GF_fsN/metrics_monthly.csv
result_partition_k40_compare_GF_fsN/predictions_monthly.csv
result_partition_k40_compare_GF_fsN/metrics_polygon_overall.csv
result_partition_k40_compare_DT_fsN/metrics_monthly.csv
result_partition_k40_compare_DT_fsN/predictions_monthly.csv
result_partition_k40_compare_DT_fsN/metrics_polygon_overall.csv
```

The `all` mode should loop only:

```text
georf geodt
```

GeoXGB/XGB output folders may be discovered by exploratory scripts as excluded historical artifacts, but they should not be generated or aggregated as main results.

## Provenance And Guardrails

Add explicit no-leak provenance to manifests or run logs:

- Partition learning years: 2018-2020.
- Evaluation months: 2021-01 through 2024-12.
- Main model allowlist: GeoRF, GeoDT.
- Excluded model family: GeoXGB/XGBoost.
- Global data path: `FEWSNET_forecast_unadjusted_bm.csv`.
- Global polygon path: `FEWS_Admin_LZ_v3.shp`.

Stage 3 should fail or warn loudly if it is about to reuse existing `metrics_monthly.csv` from a previous run when the no-leak workflow is expected. The current skip-if-existing behavior is useful for long reruns, but no-leak regeneration requires an explicit cleanup or output overwrite decision.

## Legacy Cleanup Requirement

Current scan found these main-workflow legacy items:

- Root file: `run_batches_2021_2024_visual_monthly.bat`.
- README and workflow docs that advertise `run_batches_2021_2024_visual_monthly.bat`.
- Main batch files and docs that advertise `geoxgb` / GeoXGB as a main workflow model.
- Aggregation and comparison scripts that include GeoXGB in main model lists.

Implementation should remove or rename active main workflow references so a project-wide scan no longer exposes the old Stage 1 path as a main entrypoint:

```bash
find . -path './.git' -prune -o -path './.venv-geodt-diagnostic' -prune -o -type f -iname '*2021*2024*' -printf '%p\n'
rg -n "run_batches_2021_2024_visual_monthly|geoxgb|GeoXGB|result_partition_k40_compare_XGB" README.md PIPELINE_WORKFLOW.md *.bat other_outputs scripts app specs
```

Expected interpretation:

- `run_batches_2021_2024_visual_monthly.bat` must not remain as an active root workflow file.
- Main README / workflow docs / batch usage should describe the no-leak 2018-2020 partition-learning workflow and 2021-2024 evaluation workflow.
- GeoXGB may remain in implementation modules or historical specs only if clearly non-main or excluded. It must not appear in main workflow model loops, paper artifact generation, or main aggregation outputs.

## Verification Plan

Before implementation completion:

1. Verify Stage 1 script has the 2018-2020 loop and no active 2021-2024 partition-learning language.
2. Verify Stage 2 consumes only 2018-2020 Stage 1 result patterns for main runs.
3. Verify Stage 3 keeps `START_MONTH=2021-01` and `END_MONTH=2024-12`.
4. Verify `all` mode runs only `georf` and `geodt`.
5. Verify aggregation excludes GeoXGB from main workbooks.
6. Verify manifests or logs record partition-learning and evaluation windows.
7. Run targeted syntax checks on edited Python files and inspect batch file control flow.
8. Run repository scans for legacy active path names and main GeoXGB references.

## Open Implementation Notes

- Keep output CSV and PNG schemas stable so paper artifact generation scripts can be reused.
- Do not delete historical result directories unless explicitly asked; generated artifacts can remain as historical data, but main scripts and docs must not treat them as current main outputs.
- Preserve fs0-only behavior as a separate mode if it remains useful, but no-leak main results are fs1/fs2/fs3.
- Preserve GeoDT-specific `--no-dt-rules` behavior.
