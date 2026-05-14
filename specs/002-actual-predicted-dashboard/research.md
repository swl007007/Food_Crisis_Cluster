# Research: Actual Predicted Dashboard

## Decision: Use existing `predictions_monthly.csv` files as the dashboard data source

**Rationale**: The inspected March 2026 month-ind result directory contains the exact actual and partitioned predicted labels required by the spec. Included folders are `result_partition_k40_compare_GF_fs1`, `GF_fs2`, `GF_fs3`, `DT_fs1`, `DT_fs2`, and `DT_fs3`, each with 62,189 rows across 12 test months and columns `FEWSNET_admin_code`, `month_start`, `partition_id`, `y_true`, `y_pred_pooled`, and `y_pred_partitioned`.

**Alternatives considered**:
- Recompute predictions from trained models: rejected because the feature is exploratory diagnostics and must not run training or batch workflows.
- Use `metrics_monthly.csv`: rejected as the primary map source because metrics are aggregate records and do not contain per-polygon labels.
- Use `y_pred_pooled`: rejected because the feature asks for predicted crisis spatial distribution from partitioned model outputs and the reference script uses `y_pred_partitioned`.

## Decision: Include GeoRF and GeoDT by folder token, exclude XGB by folder token

**Rationale**: Existing source folders identify GeoRF with `GF`, GeoDT with `DT`, and GeoXGB/XGBoost with `XGB`. Including only `GF` and `DT` directly satisfies the selector contract and prevents accidental use of XGBoost data present in the same source directory.

**Alternatives considered**:
- Infer model labels from run manifests only: rejected because folder tokens are sufficient, stable for these outputs, and simpler for availability discovery.
- Include XGB as hidden data for future toggles: rejected because the spec explicitly excludes GeoXGB/XGBoost entirely.

## Decision: Use `month_start` as the canonical test-month key

**Rationale**: Every inspected prediction file includes `month_start`, and the existing reference plotting script normalizes this field as a date string before filtering and rendering. The available inspected dates are February, June, and October for years 2021 through 2024.

**Alternatives considered**:
- Infer dates from file names: rejected because the files are scope-level aggregate CSVs rather than per-month files.
- Use metrics dates: rejected because the dashboard maps prediction records, and the map rows already carry `month_start`.

## Decision: Join to the global FEWSNET shapefile through `FEWSNET_admin_code`

**Rationale**: The reference plotting script joins shapefile geometries to predictions using `FEWSNET_admin_code` and supports aliases that are normalized to that name. The user explicitly requires the global FEWSNET shapefile and rejects Nigeria-specific development defaults.

**Alternatives considered**:
- Use `Nigeria.shp`: rejected by the feature requirements and constitution geographic-scope rule.
- Use centroid or country-level joins: rejected because the data already has polygon-level FEWSNET admin codes.

## Decision: Plot existing binary labels without thresholding

**Rationale**: Inspected `y_true` and `y_pred_partitioned` values are binary `0` and `1` in all included and excluded prediction files. The dashboard can therefore plot existing labels/classes and does not need a probability-to-binary conversion threshold.

**Alternatives considered**:
- Apply `PREDICTION_THRESHOLD = 0.5`: rejected because no probabilities are needed for this feature and adding a threshold would create unnecessary ambiguity.
- Plot probabilities or IPC phases: rejected because the source fields and user goal are binary actual/predicted crisis distributions.

## Decision: Generate static browser controls from compact embedded data

**Rationale**: The dashboard needs interactive selection without a backend server or full GIS frontend. A generator can convert the shapefile and prediction rows into compact browser-ready data, embed it in a self-contained HTML file by default, and use simple controls to color the same polygon canvas for actual and predicted panels.

**Alternatives considered**:
- Pre-render one image per selection: simpler but less flexible for alpha controls and can make a self-contained HTML unnecessarily large.
- Build a Dash app or server: rejected because the feature asks for static HTML and no backend.
- Use Leaflet-style dynamic maps: rejected unless it proves clearly simpler, which is not necessary for this fixed diagnostic.

## Decision: Write dashboard output under the source result directory in an exploratory subfolder

**Rationale**: `main_ablation_results/march2026_main_backup_month_ind_cont3/actual_predicted_dashboard/` keeps the dashboard close to its source data while avoiding standard forecast, standalone prediction, and scenario deliverables.

**Alternatives considered**:
- Write under `deliverables/`: rejected because the dashboard is exploratory unless explicitly promoted.
- Write at repository root: rejected because it separates generated artifacts from provenance and increases clutter.
