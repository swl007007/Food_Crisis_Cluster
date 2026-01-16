# GeoRF vs GeoXGB Pipeline Parity Report

## Overview
- Both pipelines share a single implementation path parameterized by `MODEL_ADAPTER`.
- Dry-run instrumentation records call order and configuration snapshots without requiring heavy dependencies.

## Call Graph Comparison
- Generated traces:
  - `result_GeoRF/call_graph_trace_gf.txt`
  - `result_GeoRF/call_graph_trace_xgb.txt`
- Diff status: **identical** (`parse_args → resolve_configuration → load_data → setup_spatial_groups → prepare_features → run_temporal_evaluation → save_results → summary`).

## Configuration Snapshot
Parsed from `result_GeoRF/logs/parity_check.log`:

| Field | GeoRF | GeoXGB |
| --- | --- | --- |
| model | gf | xgb |
| data_mode | nogis | nogis |
| assignment | polygons | polygons |
| nowcasting | False | False |
| max_depth | 6 | 6 |
| min_depth | 1 | 1 |
| forecasting_scope | 1 | 1 |
| n_jobs | 4 | 4 |
| track_partition_metrics | False | False |
| enable_metrics_maps | False | False |
| start_year | 2024 | 2024 |
| end_year | 2024 | 2024 |
| vis_debug_mode | False | False |

Only the `model` identifier differs, as required.

## Source Alignment
- `app/main_model_GF.py` and `app/main_model_XGB.py` differ solely in adapter selection (`GFAdapter` vs `XGBAdapter`) and dry-run labels.
- Shared utilities reside in the same module; both scripts import identical dependencies conditionally.

## Artifacts Produced
- `result_GeoRF/call_graph_trace_gf.txt`
- `result_GeoRF/call_graph_trace_xgb.txt`
- `result_GeoRF/logs/parity_check.log`

These artifacts confirm structural parity and matching non-model configuration across the two pipelines.
