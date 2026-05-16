# Feature Specification: Crisis Onset Analysis

**Feature Branch**: `004-crisis-onset-analysis`  
**Created**: 2026-05-16  
**Status**: Draft  
**Input**: User description: "Create a lightweight follow-up Spec Kit feature that minimally extends the previous phase-change-only monthly performance analysis. Reference the previous spec, plan, tasks, data-contract discovery, plotting logic, summary-table logic, and smoke-test approach. Do not redesign or duplicate logic. Add one new filter mode: previous broader mode `any_phase_change`, where current true crisis indicator differs from previous true crisis indicator; new mode `crisis_onset`, where previous true crisis indicator = 0 and current true crisis indicator = 1. Save all outputs to a separate folder named `crisis_onset_analysis`. Do not overwrite standard monthly plots or broader phase-change outputs. Use only GeoRF and GeoDT. Exclude XGB. Recompute metrics from filtered row-level predictions, not from precomputed metrics. This is exploratory diagnostics / baseline-comparison analysis, not model training or pipeline modification."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Isolate crisis-onset rows (Priority: P1)

As a food-crisis model analyst, I need a crisis-onset view of the existing row-level monthly prediction outputs, limited to cases where a spatial unit moves from non-crisis in the previous available test month to crisis in the current test month, so that model performance can be evaluated on onset events rather than all phase changes.

**Why this priority**: The crisis-onset filter is the core value of the follow-up. Plots and summary results are only valid if they are based on the correct subset of row-level predictions.

**Independent Test**: Can be tested by taking one existing GeoRF or GeoDT prediction source and verifying that retained rows have `previous_y_true = 0` and `y_true = 1`, while first observations and crisis recoveries are excluded.

**Acceptance Scenarios**:

1. **Given** a spatial unit with multiple observed test months, **When** crisis-onset filtering is applied, **Then** only rows with a previous available true crisis indicator of 0 and a current true crisis indicator of 1 are retained.
2. **Given** a row where the previous available true crisis indicator is 1 and the current true crisis indicator is 0, **When** crisis-onset filtering is applied, **Then** the row is excluded even though it would qualify for the broader `any_phase_change` mode.
3. **Given** the first observed test month for a spatial unit, **When** crisis-onset filtering is applied, **Then** the row is excluded because no previous true value exists.

---

### User Story 2 - Recompute onset-only metrics (Priority: P2)

As a food-crisis model analyst, I need monthly precision, recall, and F1 recomputed from crisis-onset rows for GeoRF and GeoDT only, preserving forecasting scopes and pooled versus partitioned result series, so that onset-event performance can be compared with the broader phase-change diagnostics.

**Why this priority**: Recomputing metrics from filtered row-level predictions prevents misleading comparisons that would result from reusing precomputed monthly metrics or broader phase-change aggregates.

**Independent Test**: Can be tested by manually calculating class-1 precision, recall, and F1 from a retained crisis-onset subset for one model, scope, month, and prediction series, then confirming the reported values match.

**Acceptance Scenarios**:

1. **Given** crisis-onset rows for one model family, forecasting scope, month, and result series, **When** metrics are calculated, **Then** precision, recall, and F1 are based only on those retained rows.
2. **Given** the existing broader `any_phase_change` outputs, **When** crisis-onset metrics are generated, **Then** those broader metrics are not reused as inputs.
3. **Given** XGB or GeoXGB prediction sources exist in the same source area, **When** the crisis-onset analysis runs, **Then** those sources are excluded from all metrics and documented as excluded.

---

### User Story 3 - Produce separated onset diagnostics (Priority: P3)

As a food-crisis model analyst, I need crisis-onset-only plots, summary tables, audit outputs, and provenance written to a separate `crisis_onset_analysis` location, so that the onset diagnostics can be reviewed without overwriting standard monthly plots or the broader phase-change outputs.

**Why this priority**: The feature is exploratory and must remain clearly separated from standard diagnostics and from the existing broader phase-change analysis.

**Independent Test**: Can be tested by generating the crisis-onset outputs and confirming that all artifacts are labeled crisis-onset-only, are located under `crisis_onset_analysis`, and leave existing standard and broader phase-change output locations unchanged.

**Acceptance Scenarios**:

1. **Given** existing standard monthly plots and broader phase-change outputs, **When** crisis-onset outputs are generated, **Then** no file in those existing output locations is overwritten.
2. **Given** the generated plots and summary table, **When** a reviewer opens them, **Then** the labels and row names identify the diagnostics as crisis-onset-only and include only GeoRF and GeoDT.
3. **Given** the generated provenance record, **When** a reviewer inspects it, **Then** it states the filter mode, source files, excluded XGB files, output files, row counts before and after filtering, and exploratory status.

---

### Edge Cases

- If a spatial unit has only one observed test month for a model and scope, it contributes no crisis-onset rows because no previous true value exists.
- If the current true crisis indicator differs from the previous true crisis indicator because crisis ended (`1 -> 0`), the row is excluded from `crisis_onset` even though it remains part of the broader `any_phase_change` definition.
- If a filtered month has no crisis-onset rows, the output marks the month as having no onset sample rather than reusing population-level or broader phase-change metrics.
- If precision, recall, or F1 is undefined because the crisis-onset subset has a zero denominator, the output records the metric as blank/NA and documents the reason in provenance.
- If a selected GeoRF or GeoDT source lacks the required row-level prediction contract, the analysis reports the missing field instead of producing partial or silently incorrect metrics.
- If repeated rows exist for the same spatial unit and test month within a model/scope group, the ambiguity is reported in provenance before results are interpreted.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The analysis MUST be a lightweight follow-up to the previous phase-change monthly performance feature and MUST reuse the established source contract, plotting conventions, summary-table conventions, provenance expectations, and smoke-test style unless a difference is explicitly required here.
- **FR-002**: The analysis MUST preserve the broader `any_phase_change` definition as rows where the current true crisis indicator differs from the previous available true crisis indicator for the same spatial unit, model family, and forecasting scope.
- **FR-003**: The analysis MUST add a `crisis_onset` filter mode defined as rows where the previous available true crisis indicator is 0 and the current true crisis indicator is 1.
- **FR-004**: The crisis-onset comparison MUST use the previous available test month within the same spatial unit, model family, and forecasting scope, not a strict previous calendar month.
- **FR-005**: The analysis MUST exclude first observations for each spatial unit, model family, and forecasting scope before evaluating crisis-onset eligibility.
- **FR-006**: The analysis MUST use only GeoRF and GeoDT row-level prediction sources for fs1, fs2, and fs3 when present.
- **FR-007**: The analysis MUST exclude XGB, GeoXGB, and XGBoost-tokened sources from all crisis-onset outputs and MUST document those exclusions.
- **FR-008**: The analysis MUST use the same inspected row-level prediction contract as the previous phase-change feature: spatial key `FEWSNET_admin_code`, test-month field `month_start`, true crisis indicator `y_true`, and prediction series represented by `y_pred_pooled` and `y_pred_partitioned`.
- **FR-009**: The analysis MUST filter row-level predictions before calculating metrics.
- **FR-010**: The analysis MUST recompute class-1 precision, recall, and F1 from crisis-onset-filtered rows and MUST NOT use precomputed monthly metrics, standard monthly metrics, or broader phase-change metrics as metric inputs.
- **FR-011**: The analysis MUST preserve fs1, fs2, and fs3 as separate forecasting scopes throughout filtering, metric calculation, plotting, summary tables, and provenance.
- **FR-012**: The analysis MUST preserve pooled and partitioned prediction series as separate result series throughout metric calculation, plotting, summary tables, and provenance.
- **FR-013**: The analysis MUST write all crisis-onset outputs to a separate output folder named `crisis_onset_analysis`.
- **FR-014**: The analysis MUST NOT overwrite standard monthly plot outputs, broader phase-change outputs, standard forecast deliverables, fs0-only artifacts, standalone prediction outputs, scenario outputs, or notebook outputs.
- **FR-015**: The analysis MUST produce crisis-onset-only monthly performance plots following the previous phase-change plot organization where practical and labeling every plot as crisis-onset-only.
- **FR-016**: The analysis MUST produce a crisis-onset-only summary table in the same broad style as the previous phase-change summary, including only GeoRF and GeoDT for fs1, fs2, and fs3, with metrics recomputed from all retained crisis-onset rows pooled across months for each model, scope, and result series.
- **FR-017**: The analysis MUST NOT include XGB, FEWSNET baseline rows, original population-level metrics, broader phase-change metrics, or unrelated baseline rows in the crisis-onset summary table.
- **FR-018**: The analysis MUST produce provenance documenting workflow mode, exploratory status, filter mode, filter definition, source files, excluded files, row counts before filtering, first-observation exclusions, rows retained after crisis-onset filtering, zero-denominator metric reasons, duplicate month flags, and generated artifacts.
- **FR-019**: The analysis SHOULD produce filtered row-level audit outputs when useful for verification, provided they are clearly labeled as crisis-onset-only exploratory diagnostics.
- **FR-020**: The smoke-test path MUST mirror the previous phase-change smoke-test approach while demonstrating the crisis-onset-specific conditions: first-observation exclusion, `0 -> 1` retention, `1 -> 0` exclusion, before/after row counts, recomputed metrics, one reduced plot or equivalent reduced visual output, and one draft summary row.
- **FR-021**: The feature MUST NOT change model training, prediction generation, ACTIVE_LAGS, lag mapping, partition-map semantics, threshold semantics, source predictions, standard three-stage evaluation workflows, fs0-only workflows, standalone prediction workflows, synthetic scenario workflows, map rendering, or shapefile scope.

### Key Entities *(include if feature involves data)*

- **Prediction Row**: A row-level observation for one spatial unit and test month, containing the true crisis indicator and pooled/partitioned predicted crisis indicators.
- **Spatial Unit Series**: The ordered sequence of prediction rows for one `FEWSNET_admin_code` within a single model family and forecasting scope.
- **Filter Mode**: The named row-selection rule applied to a Spatial Unit Series. `any_phase_change` retains any true-label transition; `crisis_onset` retains only non-crisis to crisis transitions.
- **Crisis-Onset Row**: A retained prediction row where `previous_y_true = 0` and current `y_true = 1` for the same spatial unit, model family, and forecasting scope.
- **Monthly Metric Result**: Recomputed precision, recall, and F1 for one model family, forecasting scope, test month, and result series using only crisis-onset rows.
- **Summary Table Row**: Overall crisis-onset performance for one model family, forecasting scope, and result series, recomputed from retained row-level predictions pooled across months.
- **Provenance Manifest**: Audit record describing inputs, exclusions, filter mode, filter definition, generated artifacts, and row counts before and after crisis-onset filtering.

### Pipeline & Data Assumptions *(mandatory for model/pipeline changes)*

- **Workflow**: Exploratory diagnostics / baseline-comparison analysis. Outputs are exploratory and must not be promoted as standard forecasts, model-training results, fs0-only artifacts, standalone prediction deliverables, or synthetic scenario outputs.
- **Temporal Scope**: The analysis uses existing labeled monthly prediction rows for fs1 = 4 months, fs2 = 8 months, and fs3 = 12 months. It analyzes only materialized `month_start` test rows, performs no feature-month-to-target-month remapping, and introduces no missing FEWSNET publication handling or synthetic target rows.
- **Crisis Label**: The binary class-1 crisis label is represented by `y_true`, with crisis onset defined only by a true-label transition from 0 to 1. Prediction-label changes do not define onset eligibility.
- **Spatial Inputs**: The effective spatial key is `FEWSNET_admin_code`. Existing partition identifiers may be retained for audit context but do not define the onset grouping. Partition maps, graph-neighbor settings, cluster counts, shapefiles, polygon IDs, and unmapped-polygon fallback behavior are out of scope because predictions already exist.
- **Geographic Scope**: No map rendering or shapefile joining is in scope. The analysis inherits the geographic coverage of the existing prediction files and must not imply a new shapefile scope.
- **Threshold Contract**: No new prediction threshold is introduced. The feature analyzes already materialized binary predictions in `y_pred_pooled` and `y_pred_partitioned`; scenario-only thresholds are out of scope.
- **Prediction Partition Maps**: Partition-map semantics are unchanged. The partitioned result series is identified solely from the existing partitioned prediction series in each source file.
- **Artifacts**: Expected artifacts are crisis-onset-only monthly performance plots, a crisis-onset-only summary table, optional filtered row-level audit output, and provenance under `crisis_onset_analysis` with workflow mode, scope, model family, source files, excluded files, filter definition, output files, and row-count evidence.
- **Environment**: This is an incremental analysis on existing repository outputs. It must not require model retraining, full multi-hour batch execution, notebook execution, or a new batch launcher unless a later plan proves one is necessary.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For every included model-family and scope combination, 100% of retained crisis-onset audit rows satisfy `previous_y_true = 0` and current `y_true = 1`.
- **SC-002**: For every included model-family and scope combination, provenance reports source row counts before filtering, first-observation exclusions, rows excluded as non-onset, and rows retained after crisis-onset filtering.
- **SC-003**: All generated monthly metric values can be traced to crisis-onset-filtered row-level predictions for exactly one model family, one forecasting scope, one test month, and one prediction series.
- **SC-004**: The crisis-onset summary table contains only GeoRF and GeoDT rows for fs1, fs2, and fs3, reports pooled and partitioned result series separately, and contains no XGB, FEWSNET, population-level, or broader phase-change rows.
- **SC-005**: No existing standard monthly plot file or broader phase-change output file is modified; all new artifacts are written under `crisis_onset_analysis`.
- **SC-006**: The smoke-test path demonstrates at least one retained `0 -> 1` row, exclusion of first observations, exclusion of `1 -> 0` rows from crisis-onset mode when present in the sampled data, recomputed precision/recall/F1, one reduced plot or equivalent reduced visual output, and one draft summary row.
- **SC-007**: A reviewer can determine the included models, excluded models, source row contract, filter mode, filter definition, output location, and exploratory status from provenance alone.
- **SC-008**: The feature can be planned without redefining the previous phase-change data contract, plotting layout, summary-table style, or smoke-test strategy.

## Assumptions

- The previous phase-change feature remains the authoritative reference for source discovery, row-level data contract, metric naming, undefined-metric handling, plotting organization, summary-table style, and smoke-test expectations.
- The existing row-level prediction files are the authoritative inputs for this exploratory follow-up.
- The new `crisis_onset` mode is narrower than `any_phase_change`; it does not include crisis recovery (`1 -> 0`) rows.
- `FEWSNET_admin_code` and `month_start` remain valid for ordering previous available observations within each model/scope/spatial-unit series.
- Optional filtered row-level audit outputs are valuable for verification but must remain clearly labeled exploratory diagnostics.
