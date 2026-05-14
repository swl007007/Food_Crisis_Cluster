# Data Model: Actual Predicted Dashboard

## Entity: Dashboard Selection

**Purpose**: Identifies one view state in the dashboard.

**Fields**:
- `model`: Display model family, one of `GeoRF` or `GeoDT`.
- `model_token`: Source folder token, one of `GF` or `DT`.
- `scope`: Forecasting scope, one of `fs1`, `fs2`, or `fs3`.
- `month_start`: Test month in normalized `YYYY-MM-DD` form.

**Validation Rules**:
- The combination must exist in the discovered availability index.
- `model_token = XGB` is never valid for dashboard selection.
- Dates sort chronologically, not lexicographically by display label.

**Relationships**:
- Selects one Prediction Dataset.
- Controls the actual and predicted layer values displayed for each Global FEWSNET Polygon.

## Entity: Prediction Dataset

**Purpose**: Represents one included `predictions_monthly.csv` source file for a model/scope.

**Fields**:
- `source_path`: Path to the source CSV.
- `model`: `GeoRF` or `GeoDT`.
- `model_token`: `GF` or `DT`.
- `scope`: `fs1`, `fs2`, or `fs3`.
- `available_months`: Sorted list of normalized `month_start` values.
- `row_count`: Number of rows read from the file.
- `required_columns_present`: Boolean result for `FEWSNET_admin_code`, `month_start`, `y_true`, and `y_pred_partitioned`.
- `excluded`: Always false for this entity; XGB discoveries are represented in the manifest, not as datasets.

**Validation Rules**:
- Required columns must be present before a dataset is included.
- `y_true` values must be binary 0/1 for included rows.
- `y_pred_partitioned` values must be binary 0/1 for included rows.
- `month_start` must be parseable and normalizable to dates.
- Rows must not be modified in the source CSV.

**Relationships**:
- Provides Prediction Records.
- Is summarized in Dashboard Manifest.

## Entity: Prediction Record

**Purpose**: Represents one polygon-month actual/predicted label record.

**Fields**:
- `FEWSNET_admin_code`: Spatial join key.
- `month_start`: Normalized test month.
- `y_true`: Actual binary crisis label, where `1` is crisis and `0` is non-crisis.
- `y_pred_partitioned`: Partitioned predicted binary crisis label, where `1` is crisis and `0` is non-crisis.
- `partition_id`: Existing source partition identifier, retained for provenance when present.

**Validation Rules**:
- `FEWSNET_admin_code` and `month_start` must be present for map display.
- `y_true` and `y_pred_partitioned` must not be silently coerced from probabilities or IPC phases.
- Records with missing actual or predicted labels for a selected combination trigger visible incomplete-data handling.

**Relationships**:
- Joins to one Global FEWSNET Polygon by `FEWSNET_admin_code`.
- Supplies actual and predicted panel coloring for one Dashboard Selection.

## Entity: Global FEWSNET Polygon

**Purpose**: Represents one map polygon from the global FEWSNET shapefile.

**Fields**:
- `FEWSNET_admin_code`: Canonical spatial key after alias normalization.
- `geometry`: Polygon or multipolygon geometry from `FEWS_Admin_LZ_v3.shp`.
- `display_path`: Browser-ready rendered path or equivalent geometry representation.
- `metadata`: Optional shapefile attributes retained only when useful for display or debugging.

**Validation Rules**:
- Shapefile source must be the global FEWSNET shapefile, not `Nigeria.shp`.
- Alias normalization is allowed only when it resolves to `FEWSNET_admin_code`.
- Invalid geometries must be repaired or reported before rendering.
- A selected model-scope-month must have at least one non-empty join to polygons for a valid smoke test.

**Relationships**:
- May have zero or one Prediction Record for a Dashboard Selection.
- Polygons without matching records render as no data.

## Entity: Dashboard Manifest

**Purpose**: Captures provenance and validation details for the generated dashboard.

**Fields**:
- `workflow_mode`: `exploratory diagnostics/tooling`.
- `production_status`: `exploratory`.
- `source_directory`: Source result directory.
- `included_sources`: List of included Prediction Datasets.
- `excluded_sources`: List or patterns for excluded XGB folders.
- `shapefile_path`: Global FEWSNET shapefile path used.
- `join_key`: `FEWSNET_admin_code`.
- `available_models`: `GeoRF`, `GeoDT`.
- `available_scopes`: Available subset of `fs1`, `fs2`, `fs3`.
- `available_dates`: Sorted normalized test months.
- `label_contract`: Actual and predicted binary label meanings.
- `threshold_contract`: `none; existing binary labels only`.
- `output_html`: Dashboard path.
- `optional_assets`: Supporting assets if generated.
- `smoke_test`: Selected model/scope/date and join result.
- `omissions_or_warnings`: Missing folders, unmatched rows, unexpected values, or other validation notes.

**Validation Rules**:
- Manifest must identify 100% of included source files.
- Manifest must explicitly document XGB exclusion when XGB source folders exist.
- Manifest must record the output path and shapefile source.
- Manifest must not claim production forecast status.
