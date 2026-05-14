# Data Model: Monthly Performance Plots

## Entity: ModelPerformanceFigure

Represents one generated figure for one model family.

**Fields**:
- `model_family`: One of `GeoDT`, `GeoRF`.
- `source_token`: One of `DT`, `GF`.
- `filename`: One of `geodt_monthly_performance.png`, `georf_monthly_performance.png`.
- `layout_rows`: Fixed ordered list `[fs1, fs2, fs3]`.
- `layout_columns`: Fixed ordered list `[precision, recall, f1]`.
- `series_per_subplot`: Fixed ordered list `[partitioned, pooled, FEWSNET baseline]`.

**Validation rules**:
- Exactly two figures are generated for a full run.
- GeoXGB/XGB is never a valid `model_family` or `source_token`.
- Each figure has exactly nine subplots.

## Entity: MonthlyMetricsRecord

Represents one row from a model `metrics_monthly.csv` file.

**Fields**:
- `test_month`: Month string in `YYYY-MM` format.
- `scope`: One of `fs1`, `fs2`, `fs3`, derived from the source directory suffix.
- `model_family`: One of `GeoDT`, `GeoRF`, derived from the source directory token.
- `series_kind`: One of `pooled`, `partitioned`, from the `model` column.
- `precision`: Numeric metric value or missing.
- `recall`: Numeric metric value or missing.
- `f1`: Numeric metric value or missing.
- `source_file`: Source CSV path.

**Validation rules**:
- `test_month`, `model`, `precision`, `recall`, and `f1` columns must exist.
- Only `pooled` and `partitioned` model rows are plotted.
- Missing plotted values become visible gaps and manifest missing-point records.
- Rows are sorted chronologically by parsed `test_month`.

## Entity: FEWSNETBaselineRecord

Represents one row from a FEWSNET baseline comparison file.

**Fields**:
- `baseline_scope`: One of `fs1`, `fs2`, from the source filename.
- `year`: Year from the `year` column.
- `quarter`: Quarter from the `quarter` column.
- `aligned_test_month`: Derived month key using Q1 -> February, Q2 -> June, Q4 -> October.
- `precision`: Numeric value from `precision(1)`.
- `recall`: Numeric value from `recall(1)`.
- `f1`: Numeric value from `f1(1)`.
- `source_file`: Source CSV path.

**Validation rules**:
- `year`, `quarter`, `precision(1)`, `recall(1)`, and `f1(1)` columns must exist.
- Quarter 3 is ignored unless a future plan explicitly defines a plotted test-month mapping.
- fs3 FEWSNET series must be built by reusing fs2 baseline records and labeled as `FEWSNET baseline (fs2 reused for fs3)`.

## Entity: PlotSeries

Represents one line in one subplot.

**Fields**:
- `model_family`: `GeoDT` or `GeoRF` for model series; `FEWSNET` for baseline series.
- `scope`: `fs1`, `fs2`, or `fs3`.
- `metric`: `precision`, `recall`, or `f1`.
- `series_label`: `partitioned`, `pooled`, `FEWSNET baseline`, or `FEWSNET baseline (fs2 reused for fs3)`.
- `line_style`: Solid for partitioned, dashed for pooled, distinct comparison style/color for FEWSNET.
- `x_values`: Chronologically ordered test months.
- `y_values`: Metric values with gaps for missing points.

**Validation rules**:
- Each applicable subplot contains exactly three series.
- x-values are ordered chronologically.
- Missing values remain missing; no interpolation or imputation.

## Entity: MissingPointRecord

Represents a missing plotted point that must be reported in the manifest.

**Fields**:
- `source_kind`: `model` or `fewsnet`.
- `model_family`: `GeoDT`, `GeoRF`, or null for baseline-only missing points.
- `scope`: `fs1`, `fs2`, or `fs3`.
- `series_label`: A plotted line label.
- `metric`: `precision`, `recall`, or `f1`.
- `test_month`: Expected aligned test month.
- `reason`: Missing file, missing row, missing metric value, or missing aligned baseline.

**Validation rules**:
- Every missing plotted point is listed in the manifest.
- Missing points do not block figure generation unless a required source file is entirely unavailable and cannot be represented as plotted gaps.

## Entity: MonthlyPerformanceManifest

Represents the provenance and validation note written with the generated figures.

**Fields**:
- `workflow_mode`: `baseline comparison`.
- `status`: `exploratory diagnostics`.
- `source_paths`: Exact model and FEWSNET source files read.
- `excluded_paths`: GeoXGB and other result files intentionally ignored.
- `column_contract`: Required model and FEWSNET columns.
- `scope_contract`: fs1/fs2/fs3 handling.
- `series_contract`: pooled, partitioned, FEWSNET labeling.
- `fewsnet_fs3_assumption`: Explicit statement that fs2 FEWSNET is reused for fs3 and is not native fs3.
- `generated_artifacts`: Figure filenames and manifest path.
- `missing_points`: List of MissingPointRecord values.
- `validation_summary`: Line counts, chronological ordering result, and smoke/full mode.

**Validation rules**:
- Manifest must identify GeoXGB exclusion.
- Manifest must identify the FEWSNET fs3 proxy assumption.
- Manifest must be concise and stored in the output directory with the figures.
