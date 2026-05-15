# Feature Specification: Phase-Change Monthly Performance

**Feature Branch**: `003-phase-change-performance`  
**Created**: 2026-05-15  
**Status**: Draft  
**Input**: User description: "Create a new Spec Kit feature for phase-change-only monthly performance analysis using row-level predictions_monthly.csv outputs for GeoRF and GeoDT only."

## Clarifications

### Session 2026-05-15

- Q: How should undefined precision, recall, or F1 values caused by zero denominators be represented? → A: Record as blank/NA and document the zero-denominator reason in the manifest.
- Q: How should summary-table metrics aggregate phase-change performance across months? → A: Recompute metrics from all filtered phase-change rows pooled across months for each model, scope, and series.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Identify phase-change prediction rows (Priority: P1)

As a food-crisis model analyst, I need the existing row-level monthly prediction outputs filtered to cases where the true crisis state changed from the same area's previous available test month, so that performance is evaluated on transition cases rather than the full population of stable and changing areas.

**Why this priority**: The phase-change filter is the core value of the feature; plots and tables are only valid if they are derived from the correct row-level subset.

**Independent Test**: Can be tested by loading one GeoRF or GeoDT `predictions_monthly.csv`, selecting one forecasting scope and a small month range, and verifying that rows are retained only when the current `y_true` differs from the previous available `y_true` for the same spatial unit.

**Acceptance Scenarios**:

1. **Given** a GeoRF or GeoDT prediction file with multiple months for the same `FEWSNET_admin_code`, **When** the phase-change filter is applied within one model family and forecasting scope, **Then** the first observed month for each `FEWSNET_admin_code` is excluded and later rows are retained only when `y_true` changes from the immediately previous available test month.
2. **Given** a spatial unit whose observed test months skip calendar months, **When** the previous value is determined, **Then** the comparison uses the previous available test month in the file, not a strict previous calendar month.
3. **Given** GeoXGB/XGBoost prediction files under the same source root, **When** source files are selected, **Then** those files are excluded from the analysis and documented as excluded.

---

### User Story 2 - Recompute phase-change monthly metrics (Priority: P2)

As a food-crisis model analyst, I need monthly precision, recall, and F1 recomputed from the filtered phase-change rows for GeoRF and GeoDT, preserving forecasting scopes and pooled versus partitioned result series, so that transition-case performance can be compared to the existing monthly diagnostics.

**Why this priority**: Recomputing from filtered rows avoids misleading results that would occur if already aggregated monthly metrics were filtered or reused.

**Independent Test**: Can be tested by manually calculating precision, recall, and F1 from a filtered month/scope/model subset and confirming the reported metric values match that subset for both pooled and partitioned predictions.

**Acceptance Scenarios**:

1. **Given** filtered phase-change rows for one model family, scope, month, and result series, **When** metrics are calculated, **Then** precision, recall, and F1 are based only on the retained rows for that exact grouping.
2. **Given** all GeoRF and GeoDT scopes fs1, fs2, and fs3, **When** phase-change metrics are produced, **Then** each scope remains a separate result and is not pooled with other scopes.
3. **Given** existing monthly metrics files, **When** phase-change metrics are produced, **Then** those aggregate metrics are not used as inputs for the phase-change calculation.

---

### User Story 3 - Produce labeled exploratory outputs (Priority: P3)

As a food-crisis model analyst, I need clearly labeled phase-change-only plots, a summary table in the existing ablation-report style, and a manifest of sources and row counts, so that the outputs can be reviewed without confusing them with standard forecast deliverables or original population-level metrics.

**Why this priority**: The analysis is exploratory and must be traceable, visually comparable, and safely separated from established deliverables.

**Independent Test**: Can be tested by generating the reduced outputs and confirming that all artifact names, labels, and manifest entries state phase-change-only exploratory diagnostics and that no existing standard output directory or file is overwritten.

**Acceptance Scenarios**:

1. **Given** the source ablation root, **When** phase-change outputs are written, **Then** they appear in a new clearly labeled location under `main_ablation_results/march2026_main_backup_month_ind_cont3` and do not overwrite `monthly_performance_plots` or standard forecast deliverables.
2. **Given** the generated monthly plots, **When** a reviewer opens them, **Then** each plot clearly identifies the analysis as phase-change-only and follows the existing monthly performance layout where practical.
3. **Given** the generated summary table, **When** a reviewer inspects its rows, **Then** it includes only GeoRF(phase change) and GeoDT(phase change) for fs1, fs2, and fs3, and excludes XGB, FEWSNET, original population-level metrics, and unrelated baseline rows.
4. **Given** the generated manifest, **When** a reviewer checks provenance, **Then** it lists source files, excluded XGB files, the filter definition, output files, and row counts before and after filtering.

---

### Edge Cases

- If a spatial unit has only one observed test month in a model/scope group, it contributes no phase-change rows because no previous true value exists.
- If a spatial unit has repeated rows for the same test month within a model/scope group, the analysis must flag the ambiguity in the manifest rather than silently treating duplicates as a normal chronology.
- If a required prediction column is missing from a selected GeoRF or GeoDT file, the analysis must stop for that file and report the missing contract field.
- If a filtered month has no retained phase-change rows, the output must mark the month as having no phase-change sample instead of reusing population-level metrics.
- If precision, recall, or F1 is undefined because the filtered subset has no positive predictions or no positive true labels, the output must record the metric as blank/NA and document the zero-denominator condition in the manifest.
- If the source root contains additional model folders, only folders matching GeoRF/GF or GeoDT/DT for fs1, fs2, and fs3 are in scope.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The analysis MUST read row-level `predictions_monthly.csv` files from `main_ablation_results/march2026_main_backup_month_ind_cont3` for GeoRF and GeoDT only.
- **FR-002**: The analysis MUST include exactly the GeoRF/GF and GeoDT/DT forecasting-scope combinations fs1, fs2, and fs3 when their prediction files are present.
- **FR-003**: The analysis MUST exclude GeoXGB, XGBoost, and XGB-tokened files entirely, and MUST record those exclusions in the manifest or equivalent provenance output.
- **FR-004**: The analysis MUST treat the inspected source contract as follows: spatial key `FEWSNET_admin_code`; test-month column `month_start`; true crisis indicator `y_true`; predicted crisis indicators `y_pred_pooled` and `y_pred_partitioned`; forecasting scope encoded by the `fs1`, `fs2`, or `fs3` folder suffix; model family encoded by `GF` for GeoRF and `DT` for GeoDT; pooled versus partitioned results encoded by the two prediction columns.
- **FR-005**: The analysis MUST compute phase-change rows separately within each model family, forecasting scope, prediction file, and spatial key, sorted by `month_start`.
- **FR-006**: The analysis MUST exclude the first observed test month for each spatial key within each model-family and forecasting-scope group before identifying phase changes.
- **FR-007**: The analysis MUST define a phase-change row as a row whose current `y_true` differs from the `y_true` for the same spatial key at the immediately previous available test month in that group.
- **FR-008**: The analysis MUST use previous available test month, not strict previous calendar month, for the phase-change comparison.
- **FR-009**: The analysis MUST filter row-level predictions first, then recompute monthly class-1 precision, recall, and F1 from the filtered rows.
- **FR-010**: The analysis MUST NOT filter or reuse already aggregated monthly metrics as the source for phase-change metrics.
- **FR-011**: The analysis MUST preserve fs1, fs2, and fs3 as separate forecasting scopes throughout filtering, metric calculation, plotting, summary tables, and provenance outputs.
- **FR-012**: The analysis MUST preserve pooled and partitioned prediction series as separate result series when recomputing metrics and plotting outputs.
- **FR-013**: The analysis MUST regenerate monthly performance plots using only phase-change rows, following the existing monthly performance plot organization where practical and labeling every plot as phase-change-only.
- **FR-014**: The analysis MUST create a summary table in the style of `main_ablation_results/Complete_ablation_test_1.xlsx` that includes only GeoRF(phase change) and GeoDT(phase change) for fs1, fs2, and fs3, with metrics recomputed from all filtered phase-change rows pooled across months for each model, scope, and result series.
- **FR-015**: The analysis MUST NOT include XGB, original population-level metrics, FEWSNET rows, or unrelated baseline rows in the phase-change summary table.
- **FR-016**: The analysis MUST write outputs to a new clearly labeled location under `main_ablation_results/march2026_main_backup_month_ind_cont3` without overwriting existing `monthly_performance_plots` or standard forecast deliverables.
- **FR-017**: The analysis MUST produce a short manifest or README documenting source files, excluded files, the filter definition, output files, row counts before and after phase-change filtering for each model family and scope, and any blank/NA metrics caused by zero denominators.
- **FR-018**: The analysis SHOULD produce filtered row-level audit files when useful for verification, provided they are clearly labeled as phase-change-only exploratory diagnostics.
- **FR-019**: The analysis MUST support a smoke-test or dry-run path that loads one GeoRF or GeoDT prediction file, selects one forecasting scope and a small set of months, verifies first-row exclusion, reports row counts before and after filtering, recomputes precision/recall/F1, and produces one reduced plot plus one draft summary row.
- **FR-020**: The feature MUST NOT change ACTIVE_LAGS, temporal lag logic, partition-map semantics, model predictions, model-training workflows, standard three-stage evaluation workflows, fs0-only workflows, standalone prediction workflows, synthetic scenario workflows, or notebook outputs.

### Key Entities *(include if feature involves data)*

- **Prediction Row**: A row-level observation for one spatial unit and test month, containing the true crisis indicator and pooled/partitioned predicted crisis indicators.
- **Spatial Unit Series**: The ordered sequence of prediction rows for one `FEWSNET_admin_code` within a single model family and forecasting scope.
- **Phase-Change Row**: A prediction row retained for analysis because its true crisis indicator differs from the same spatial unit's previous available test month.
- **Monthly Metric Result**: A recomputed precision, recall, and F1 result for one model family, forecasting scope, test month, and result series using only phase-change rows.
- **Summary Table Row**: A report row for one phase-change model family and forecasting scope in the same broad presentation style as the reference ablation workbook.
- **Provenance Manifest**: A short audit record describing inputs, exclusions, filter definition, generated artifacts, and row counts before and after filtering.

### Pipeline & Data Assumptions *(mandatory for model/pipeline changes)*

- **Workflow**: Exploratory diagnostics / baseline-comparison analysis. Outputs are exploratory diagnostics unless explicitly promoted later; they are not standard forecasts, fs0-only artifacts, standalone prediction outputs, synthetic scenario overlays, or model-training results.
- **Temporal Scope**: Existing labeled monthly prediction outputs cover fs1 = 4 months, fs2 = 8 months, and fs3 = 12 months. The inspected files contain 12 existing labeled test/target months from 2021-02-01 through 2024-10-01. This feature analyzes only materialized `month_start` test rows, performs no feature-month-to-target-month remapping, does not alter ACTIVE_LAGS or lag assumptions, and introduces no missing FEWSNET publication handling or synthetic target rows.
- **Crisis Label**: The binary class-1 crisis label is represented by `y_true`, with values 0 and 1 in the inspected files. Phase-change status is based only on changes in this true label, not on changes in predicted labels.
- **Spatial Inputs**: The inspected spatial key is `FEWSNET_admin_code`, not `area_id`. Existing `partition_id` values may be retained for audit context but do not define the phase-change grouping. Partition-map identity, k-neighbor graph settings, cluster counts, shapefiles, polygon IDs, and unmapped-polygon fallback behavior are out of scope because predictions already exist.
- **Geographic Scope**: No map rendering or shapefile joining is in scope. The analysis inherits the geographic coverage of the existing prediction files and must not imply a new shapefile scope.
- **Threshold Contract**: No new prediction threshold is introduced. The feature analyzes already materialized binary predictions in `y_pred_pooled` and `y_pred_partitioned`; scenario-only thresholds are out of scope.
- **Prediction Partition Maps**: Partition-map semantics are not changed. The partitioned result series is identified solely from the existing `y_pred_partitioned` column in each source file.
- **Artifacts**: Expected artifacts are phase-change-only monthly performance plots, a phase-change-only summary table, an optional filtered row-level audit output, and a manifest or README with workflow mode, scope, model family, source files, excluded XGB files, output files, and row-count provenance.
- **Environment**: This is an incremental analysis on existing repository outputs. It must not require model retraining, full multi-hour batch execution, notebook execution, or a new parallel launcher unless a later implementation plan proves it necessary.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All six in-scope source files are represented in the analysis when present: GeoRF/GF and GeoDT/DT for fs1, fs2, and fs3; all three XGB files under the source root are excluded.
- **SC-002**: For every included model-family and scope combination, the manifest reports source row counts before filtering, row counts after phase-change filtering, and the count of first observations excluded because no previous true value exists.
- **SC-003**: For every generated monthly metric value, the calculation can be traced to phase-change-filtered row-level predictions for one model family, one forecasting scope, one test month, and one prediction series.
- **SC-004**: The monthly plots clearly state phase-change-only analysis and include separate fs1, fs2, and fs3 views for GeoRF and GeoDT where phase-change rows exist.
- **SC-005**: The summary table contains only GeoRF(phase change) and GeoDT(phase change) rows for fs1, fs2, and fs3, reports metrics recomputed from all filtered phase-change rows pooled across months for each result series, and has no XGB, FEWSNET, original population-level, or unrelated baseline rows.
- **SC-006**: No files in existing `monthly_performance_plots` or standard forecast deliverable locations are overwritten; all generated outputs are placed in a new clearly labeled phase-change location.
- **SC-007**: The smoke-test path demonstrates, on one GeoRF or GeoDT configuration and a reduced month set, first-row exclusion, before/after row counts, recomputed precision/recall/F1, one reduced monthly plot, and one draft summary row.
- **SC-008**: A reviewer can determine the source column contract, filter definition, output location, included models, excluded models, and exploratory status from the generated manifest or README alone.

## Assumptions

- The reference `predictions_monthly.csv` files are the authoritative row-level inputs for this exploratory analysis.
- `FEWSNET_admin_code` is the effective spatial key because the inspected files do not use an `area_id` column.
- `month_start` is the effective test-month column and can be ordered chronologically for previous-available-month comparison.
- The observed prediction files for each included model/scope have the same column contract and comparable month coverage.
- The broad style of the reference ablation workbook is sufficient; the phase-change table does not need to reproduce unrelated baseline sections that are explicitly out of scope.
- Optional filtered row-level audit outputs are useful if they make verification easier, but they must remain clearly labeled exploratory diagnostics.
