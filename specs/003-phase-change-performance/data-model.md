# Data Model: Phase-Change Monthly Performance

## Entity: Source Prediction File

Represents one row-level `predictions_monthly.csv` input for a model family and forecasting scope.

### Fields

- `path`: project-relative source file path.
- `model_key`: canonical model key, `georf` or `geodt`.
- `model_label`: display label, `GeoRF` or `GeoDT`.
- `source_token`: folder token, `GF` or `DT`.
- `scope`: one of `fs1`, `fs2`, `fs3`.
- `row_count_before`: total source rows read.
- `row_count_after`: rows retained after phase-change filtering.
- `first_observation_count`: rows excluded because no previous true value exists.

### Validation Rules

- Must point to a GeoRF/GF or GeoDT/DT `predictions_monthly.csv` file.
- Must not point to XGB/GeoXGB files.
- Must contain required columns: `FEWSNET_admin_code`, `month_start`, `y_true`, `y_pred_pooled`, `y_pred_partitioned`.
- `scope` must remain separate from other scopes.

## Entity: Prediction Row

Represents one spatial unit and test month prediction record.

### Fields

- `FEWSNET_admin_code`: spatial key used for phase-change grouping.
- `month_start`: test-month date used for chronological ordering.
- `partition_id`: optional audit context retained from the source file when present.
- `y_true`: binary true class-1 crisis indicator.
- `y_pred_pooled`: binary pooled-model prediction.
- `y_pred_partitioned`: binary partitioned-model prediction.
- `model_key`: `georf` or `geodt` inherited from the source file.
- `scope`: `fs1`, `fs2`, or `fs3` inherited from the source file.

### Validation Rules

- `month_start` must be parseable as a date.
- `y_true`, `y_pred_pooled`, and `y_pred_partitioned` must represent binary class labels.
- Duplicate `(model_key, scope, FEWSNET_admin_code, month_start)` rows must be flagged in provenance before filtering proceeds or before results are interpreted.

## Entity: Spatial Unit Series

Represents the ordered rows for one `FEWSNET_admin_code` within one model family and forecasting scope.

### Fields

- `model_key`
- `scope`
- `FEWSNET_admin_code`
- `ordered_rows`: rows sorted by `month_start`.

### State Transitions

1. Source rows loaded.
2. Rows sorted by `month_start`.
3. First row marked as ineligible due to no previous true value.
4. Remaining rows marked as phase-change or non-phase-change based on `y_true` compared with the previous available row.

## Entity: Phase-Change Row

Represents a retained prediction row where the true crisis state changed from the previous available test month.

### Fields

- All Prediction Row fields.
- `previous_month_start`: previous available test month for the same model/scope/spatial unit.
- `previous_y_true`: true class label from the previous available test month.
- `phase_change`: true for retained rows.

### Validation Rules

- Must never be the first observed row in its Spatial Unit Series.
- Must satisfy `y_true != previous_y_true`.
- Must be computed separately by model family and forecasting scope.

## Entity: Monthly Metric Result

Represents recomputed monthly performance for one result series.

### Fields

- `model_key`
- `model_label`
- `scope`
- `test_month`
- `series`: `pooled` or `partitioned`.
- `precision`
- `recall`
- `f1`
- `phase_change_rows`
- `true_positive_count`
- `false_positive_count`
- `false_negative_count`
- `undefined_metrics`: list or flags for metrics recorded as blank/NA due to zero denominators.

### Validation Rules

- Must be computed from phase-change rows only.
- Must not use already aggregated monthly metrics as input.
- Undefined precision, recall, or F1 must be represented as blank/NA and documented in provenance.

## Entity: Summary Table Row

Represents overall phase-change performance for one model, scope, and result series.

### Fields

- `model_label`: `GeoRF(phase change)` or `GeoDT(phase change)`.
- `lag_months`: 4 for fs1, 8 for fs2, 12 for fs3.
- `scope`: `fs1`, `fs2`, or `fs3`.
- `series`: `pooled` or `partitioned`.
- `precision`
- `recall`
- `f1`
- `phase_change_rows`

### Validation Rules

- Metrics must be recomputed from all filtered phase-change rows pooled across months for the model/scope/series.
- Must not include XGB, FEWSNET, original population-level, or unrelated baseline rows.

## Entity: Provenance Manifest

Represents audit information for the analysis run.

### Fields

- `workflow_mode`: exploratory diagnostics / baseline-comparison analysis.
- `status`: exploratory.
- `source_root`
- `output_dir`
- `included_source_files`
- `excluded_source_files`
- `column_contract`
- `filter_definition`
- `row_counts`
- `zero_denominator_metrics`
- `duplicate_month_flags`
- `generated_artifacts`
- `smoke_or_full_mode`

### Validation Rules

- Must document all included GeoRF and GeoDT files.
- Must document excluded XGB files.
- Must document row counts before and after filtering by model and scope.
- Must document blank/NA metrics caused by zero denominators.
