# Contract: Dashboard Generation

## Purpose

Defines the expected input, output, validation, and UI behavior for generating the exploratory actual-vs-predicted crisis dashboard.

## Generator Inputs

| Input | Required | Contract |
|-------|----------|----------|
| Source result directory | Yes | Defaults to `main_ablation_results/march2026_main_backup_month_ind_cont3` unless explicitly overridden. |
| Prediction files | Yes | Include only `result_partition_k40_compare_GF_fs{1,2,3}/predictions_monthly.csv` and `result_partition_k40_compare_DT_fs{1,2,3}/predictions_monthly.csv` when present. |
| Excluded files | Yes | Discover and document `result_partition_k40_compare_XGB_fs*`, but never include them in dashboard model options or map data. |
| Shapefile | Yes | Use global `FEWS_Admin_LZ_v3.shp`; reject Nigeria-only shapefile defaults for this feature. |
| Output directory | Yes | Defaults to `main_ablation_results/march2026_main_backup_month_ind_cont3/actual_predicted_dashboard/`. |

## Required Prediction Columns

Each included prediction file must provide:

| Column | Meaning |
|--------|---------|
| `FEWSNET_admin_code` | Polygon join key. |
| `month_start` | Test month/date; normalized to `YYYY-MM-DD` and sorted chronologically. |
| `y_true` | Actual binary crisis label. |
| `y_pred_partitioned` | Partitioned predicted binary crisis label. |

Optional provenance columns such as `partition_id` may be preserved in manifest summaries or diagnostics.

## Label Contract

- `0` means non-crisis.
- `1` means class-1 crisis.
- Existing labels/classes are plotted as-is.
- No probability-to-label threshold may be applied silently.
- If probability conversion becomes necessary, implementation must stop or require an explicit threshold name, numeric value, and source before continuing.

## Spatial Join Contract

- Canonical join key: `FEWSNET_admin_code`.
- Shapefile aliases may be normalized only when they represent the same FEWSNET admin code.
- The smoke-test selection must produce a non-empty join.
- Polygons without selected prediction rows render as no data.

## Dashboard UI Contract

- Model selector contains exactly `GeoRF` and `GeoDT` when both are available; it never contains GeoXGB/XGBoost.
- Scope selector contains available `fs1`, `fs2`, and `fs3` combinations only.
- Date selector contains normalized available `month_start` dates sorted chronologically.
- For the active selection, the left map panel shows actual crisis distribution from `y_true`.
- For the active selection, the right map panel shows predicted crisis distribution from `y_pred_partitioned`.
- Both panels use the same geographic canvas and clearly visible labels.
- Missing or incomplete selections display an explicit no-data/incomplete-data message and must not retain stale maps.
- Prediction overlay transparency defaults to actual-only on the left and predicted-visible on the right; an alpha control is included when practical.

## Output Contract

| Output | Required | Contract |
|--------|----------|----------|
| Dashboard HTML | Yes | Prefer one self-contained static HTML file. Supporting assets are allowed only if needed for size or performance. |
| Manifest | Yes | JSON or short note documenting provenance, availability, exclusions, join key, shapefile source, value meanings, smoke-test result, and output path. |
| Assets | Optional | If emitted, must stay in the same exploratory output directory and be listed in the manifest. |

## Forbidden Operations

- No model training or retraining.
- No full multi-hour batch execution.
- No notebook execution.
- No changes to `ACTIVE_LAGS`.
- No changes to partition-map semantics or spatial clustering logic.
- No overwriting standard forecast, standalone prediction, or synthetic scenario deliverables.
