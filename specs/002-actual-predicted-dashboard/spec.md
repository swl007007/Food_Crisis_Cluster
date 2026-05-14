# Feature Specification: Actual Predicted Dashboard

**Feature Branch**: `002-actual-predicted-dashboard`
**Created**: 2026-05-14
**Status**: Draft
**Input**: User description: "Create a new Spec Kit feature for a static HTML dashboard that compares actual crisis spatial distribution against predicted crisis spatial distribution for GeoRF and GeoDT across fs1, fs2, and fs3, using existing March 2026 month-ind results and the global FEWSNET shapefile."

## Clarifications

### Session 2026-05-14

- Q: Should the dashboard be self-contained or may it use supporting assets? → A: Self-contained HTML preferred; use assets only if needed for size/performance.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Compare actual and predicted crisis maps (Priority: P1)

A food-crisis model analyst opens one generated dashboard, selects a test month, model family, and forecasting scope, and compares the actual crisis spatial distribution with the partitioned predicted crisis distribution in two side-by-side panels.

**Why this priority**: This is the primary diagnostic value: a fast visual comparison of where the model agrees or disagrees with observed crisis labels without rerunning model training or batch evaluation.

**Independent Test**: Can be fully tested by opening the generated dashboard, selecting one known model-scope-month combination with complete data, and confirming that the left panel shows actual crisis labels while the right panel shows predicted crisis labels for the same geography.

**Acceptance Scenarios**:

1. **Given** the dashboard has loaded available GeoRF and GeoDT result data, **When** the user selects GeoRF, fs2, and 2024-06-01, **Then** the left panel displays actual crisis distribution and the right panel displays predicted crisis distribution for that exact selection.
2. **Given** a selected month, model, and scope have matched prediction and actual rows, **When** the dashboard renders both panels, **Then** both panels use the same geographic canvas and are clearly labeled as actual reference and predicted comparison.
3. **Given** the source directory also contains GeoXGB/XGBoost results, **When** the user opens the model selector, **Then** only GeoRF and GeoDT choices are available.

---

### User Story 2 - Inspect available dates, scopes, and models safely (Priority: P2)

A model analyst uses dashboard controls to switch among available dates, GeoRF/GeoDT models, and fs1/fs2/fs3 scopes and sees a visible message if a requested combination is unavailable rather than seeing stale or misleading maps.

**Why this priority**: The result directory may contain partial or uneven artifacts over time, so the diagnostic must make availability explicit and avoid silently mixing data across months, models, or scopes.

**Independent Test**: Can be tested by checking selector contents against the source result folders and by selecting or simulating a missing combination to verify that the dashboard reports the omission visibly.

**Acceptance Scenarios**:

1. **Given** source files exist for GeoRF and GeoDT fs1, fs2, and fs3, **When** the dashboard loads, **Then** the selectors list only available model-scope-date combinations derived from those files.
2. **Given** one model-scope-date combination is missing or has no matched polygons, **When** that combination would otherwise be selected, **Then** the dashboard shows a clear no-data message and does not display stale maps from a previous selection.

---

### User Story 3 - Preserve provenance and output boundaries (Priority: P3)

A project maintainer reviews the generated artifacts and can identify the source result files, selected global shapefile, join key, available models/scopes/dates, value meaning, and output location without confusing this exploratory dashboard with standard forecast deliverables.

**Why this priority**: The GeoRF constitution requires workflow classification, artifact hygiene, geographic scope, and threshold provenance so exploratory diagnostics are not mistaken for production forecasts.

**Independent Test**: Can be tested by reading the generated note or manifest and verifying it names the source directory, included/excluded model families, shapefile, join key, label fields, and output path.

**Acceptance Scenarios**:

1. **Given** the dashboard and manifest are generated, **When** a maintainer reviews the manifest, **Then** it documents the workflow as exploratory diagnostics/tooling and not a model-training, standard 3-stage evaluation, fs0-only, standalone prediction, or synthetic scenario feature.
2. **Given** the dashboard is generated from existing predicted labels, **When** a maintainer reviews the manifest, **Then** it states that no new probability-to-label threshold was introduced.

### Edge Cases

- If a model-scope folder is absent, empty, or lacks `predictions_monthly.csv`, that combination is omitted from selectors and documented in the manifest.
- If a selected date has actual labels but no predicted labels, or predicted labels but no actual labels, the dashboard shows a visible incomplete-data message for that selection.
- If prediction rows do not join to any global FEWSNET polygons through `FEWSNET_admin_code`, generation fails or records a validation failure rather than producing an empty dashboard as if it were valid.
- If some polygons are present in the shapefile but absent from the selected prediction rows, those polygons are visually treated as no data and counted in the manifest or dashboard summary.
- If source values outside the expected binary classes 0 and 1 appear in `y_true` or `y_pred_partitioned`, generation records the unexpected values and does not silently reinterpret them as crisis labels.
- If the source directory contains GeoXGB/XGBoost result folders, they are ignored for dashboard data and explicitly listed as excluded in the manifest.
- If a Nigeria-specific shapefile path is present in older plotting defaults, it is not accepted as the geographic source for this feature.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The feature MUST generate one static dashboard artifact that supports selecting test month/date, model family, and forecasting scope from available data, with self-contained HTML preferred and supporting assets used only when needed for size or performance.
- **FR-002**: The model selector MUST include GeoRF and GeoDT only, identified from `result_partition_k40_compare_GF_fs*` and `result_partition_k40_compare_DT_fs*` folders respectively.
- **FR-003**: The feature MUST exclude GeoXGB/XGBoost entirely, including all `result_partition_k40_compare_XGB_fs*` source folders and any XGBoost option in the dashboard controls.
- **FR-004**: The scope selector MUST include fs1, fs2, and fs3 where available, identified from result folder suffixes `_fs1`, `_fs2`, and `_fs3`.
- **FR-005**: The date selector MUST use the `month_start` field from included prediction files, normalize dates consistently, and sort test months chronologically.
- **FR-006**: For each selected date, model, and scope, the dashboard MUST show two synchronized side-by-side map panels over the same geographic canvas.
- **FR-007**: The left panel MUST present actual crisis spatial distribution as the comparison reference, using the `y_true` field.
- **FR-008**: The right panel MUST present partitioned predicted crisis spatial distribution, using the `y_pred_partitioned` field.
- **FR-009**: The dashboard MUST make panel meanings visually explicit, including labels that distinguish actual/reference from predicted/comparison.
- **FR-010**: The dashboard SHOULD support prediction overlay transparency, with actual-only viewing equivalent to prediction alpha 0 on the left panel and predicted viewing equivalent to prediction alpha 1 on the right panel.
- **FR-011**: The feature MUST use the global FEWSNET shapefile path `C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp` for the spatial join and rendered geography.
- **FR-012**: The feature MUST NOT use Nigeria-specific development shapefile defaults for this dashboard.
- **FR-013**: The feature MUST join prediction and actual rows to the global shapefile through `FEWSNET_admin_code`, accepting the same known shapefile aliases used by existing plotting logic only when they resolve to `FEWSNET_admin_code`.
- **FR-014**: The feature MUST use existing prediction labels/classes from source files and MUST NOT introduce a new prediction threshold unless probability-to-label conversion becomes unavoidable and is explicitly documented before implementation.
- **FR-015**: The feature MUST document that inspected source values are binary labels, with `0` representing non-crisis and `1` representing crisis, for both `y_true` and `y_pred_partitioned` in the March 2026 result files.
- **FR-016**: Missing dates, scopes, models, unmatched joins, or empty selections MUST be handled gracefully with a visible dashboard message or documented omission.
- **FR-017**: The feature MUST generate a short note or manifest documenting source files, selected shapefile, join key, included and excluded model families, available dates, available scopes, value meanings, and output path.
- **FR-018**: Generated artifacts MUST be saved in an appropriate exploratory output location without overwriting standard forecast deliverables, prediction deliverables, synthetic scenario deliverables, or the existing source result CSVs.
- **FR-019**: The feature MUST be generated without model retraining, full multi-hour batch execution, notebook execution, changes to `ACTIVE_LAGS`, changes to partition-map semantics, or changes to spatial clustering logic.
- **FR-020**: The feature MUST include a smoke-test or dry-run validation path using at least one date, one scope, and one model with both actual and predicted data.

### Key Entities *(include if feature involves data)*

- **Dashboard Selection**: A user-selected combination of test month/date, model family, and forecasting scope. It determines which rows are shown in both map panels.
- **Model Family**: The included model type represented by result folders. GeoRF maps to `GF`; GeoDT maps to `DT`; GeoXGB/XGBoost maps to `XGB` and is excluded.
- **Forecasting Scope**: The horizon identifier fs1, fs2, or fs3 available in result folder suffixes. This feature reads existing evaluated outputs and does not alter scope semantics.
- **Prediction Record**: A row from an included `predictions_monthly.csv` containing `FEWSNET_admin_code`, `month_start`, `y_true`, and `y_pred_partitioned` for one polygon-month.
- **Global FEWSNET Polygon**: A spatial unit from `FEWS_Admin_LZ_v3.shp` joined to prediction records through `FEWSNET_admin_code` and rendered in the dashboard.
- **Dashboard Manifest**: A short provenance artifact that records source assumptions, availability, exclusions, join behavior, and output paths.

### Pipeline & Data Assumptions *(mandatory for model/pipeline changes)*

- **Workflow**: Exploratory diagnostics/tooling. This is not model training, not the standard 3-stage evaluation workflow, not fs0-only, not standalone 2026-2027 prediction, and not a synthetic scenario overlay. Outputs are exploratory unless explicitly promoted later as reviewed deliverables.
- **Temporal Scope**: The dashboard uses already-evaluated labeled test months from `month_start` in the March 2026 result files. Inspected available months are February, June, and October for each year 2021 through 2024. Forecasting scopes are fs1, fs2, and fs3 as represented by existing result folders; no feature-month to target-month mapping is recomputed by this feature.
- **Crisis Label**: Actual crisis is read from `y_true`; predicted crisis is read from `y_pred_partitioned`. Inspected values are binary `0` and `1`; `1` is the class-1 crisis label and `0` is non-crisis. The dashboard does not reinterpret IPC phases or probabilities.
- **Spatial Inputs**: Source prediction/result files are `predictions_monthly.csv` under `main_ablation_results/march2026_main_backup_month_ind_cont3/result_partition_k40_compare_GF_fs1`, `...GF_fs2`, `...GF_fs3`, `...DT_fs1`, `...DT_fs2`, and `...DT_fs3`. Each inspected file contains 62,189 rows and columns `FEWSNET_admin_code`, `month_start`, `partition_id`, `y_true`, `y_pred_pooled`, and `y_pred_partitioned`. GeoXGB/XGBoost folders with `XGB` are present but excluded. Existing partition identities are inherited from these source outputs and not modified.
- **Geographic Scope**: The dashboard uses the global FEWSNET shapefile `C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp`. Nigeria-only shapefiles are out of scope for this feature.
- **Threshold Contract**: No new threshold is used because the dashboard plots existing binary labels/classes from `y_true` and `y_pred_partitioned`. If a later plan discovers only probabilities are available for a required case, that plan must name the threshold, numeric value, and source before conversion.
- **Prediction Partition Maps**: The feature consumes post-evaluation `predictions_monthly.csv` outputs that already encode partitioned predictions. It does not read, generate, or change partition maps.
- **Artifacts**: Expected artifacts are one generated static dashboard, preferably as self-contained HTML, optional supporting static assets only if needed for size or performance, and one short manifest or note. Outputs should live under an exploratory diagnostics/output location such as `main_ablation_results/march2026_main_backup_month_ind_cont3/actual_predicted_dashboard/` unless planning identifies a safer equivalent location. Standard forecast deliverables must not be overwritten.
- **Environment**: This is an existing-repository tooling feature. Generation may be run from WSL or Windows with the project’s existing Python/geospatial environment, but it must not require notebook execution, full batch workflows, new launcher proliferation, or non-ASCII Windows CMD-facing output.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Opening the generated dashboard allows a user to select among all included GeoRF and GeoDT model-scope-date combinations found in the source directory in under 30 seconds on a typical workstation.
- **SC-002**: The model selector contains exactly two model choices, GeoRF and GeoDT, and contains zero GeoXGB/XGBoost choices.
- **SC-003**: For a smoke-tested combination with complete data, the dashboard renders two side-by-side panels where the left panel uses actual labels and the right panel uses predicted labels for the same date, model, scope, and geographic extent.
- **SC-004**: The generated manifest identifies 100% of included prediction source files, the excluded XGBoost source pattern, the global shapefile path, the join key, available scopes, available dates, and output artifact path.
- **SC-005**: A dry-run or smoke-test validates at least one GeoRF or GeoDT fs1/fs2/fs3 month with a successful non-empty shapefile join and no model training, full batch execution, or notebook execution.
- **SC-006**: Missing or unavailable model-scope-date combinations produce a visible no-data message or documented omission instead of stale maps or silent failure.
- **SC-007**: The output process overwrites zero standard forecast deliverables, zero standalone prediction deliverables, and zero synthetic scenario deliverables.

## Assumptions

- Users are analysts or maintainers who can open a local static dashboard artifact from the project workspace.
- The existing March 2026 month-ind result directory remains the authoritative source for this dashboard unless a later command explicitly changes the source directory.
- The global FEWSNET shapefile is available at the documented project workstation path or an equivalent path resolved during implementation and recorded in the manifest.
- Existing `predictions_monthly.csv` files contain the complete data needed for actual-vs-predicted comparison; `metrics_monthly.csv`, `metrics_polygon_overall.csv`, and `run_manifest.json` may support provenance but are not the primary map data source.
- The feature is exploratory diagnostics/tooling and should not be promoted as a production forecast or reviewed deliverable without a separate explicit decision.
- The alpha-control behavior is preferred when practical, but the core requirement is that the dashboard clearly provides actual-only reference and predicted comparison views side by side.
