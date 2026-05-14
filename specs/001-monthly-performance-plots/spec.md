# Feature Specification: Monthly Performance Plots

**Feature Branch**: `001-monthly-performance-plots`  
**Created**: 2026-05-14  
**Status**: Draft  
**Input**: User description: "Create a new Spec Kit feature for plotting monthly performance metrics from existing ablation results for GeoDT and GeoRF, excluding GeoXGB, with FEWSNET baseline comparison and fs2 reused as the fs3 baseline proxy."

## Clarifications

### Session 2026-05-14

- Q: Where should the generated figures and manifest be saved? → A: Save under `main_ablation_results/march2026_main_backup_month_ind_cont3/monthly_performance_plots/`.
- Q: How should missing required model or FEWSNET plotted values be handled? → A: Show gaps for any missing model or FEWSNET values and report them in the manifest.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Compare monthly model performance (Priority: P1)

As a food-crisis modeling analyst, I want one GeoDT figure and one GeoRF figure showing monthly precision, recall, and F1 across fs1, fs2, and fs3, so that I can visually compare partitioned, pooled, and FEWSNET baseline performance over time without rerunning model training.

**Why this priority**: This is the primary feature goal and directly supports baseline-comparison review from existing ablation outputs.

**Independent Test**: Can be fully tested by generating the two figures from the existing result directories and verifying that each figure contains a 3x3 grid with the required line series and chronological x-axis ordering.

**Acceptance Scenarios**:

1. **Given** the existing GeoDT and GeoRF monthly ablation result files and FEWSNET baseline files, **When** the plotting feature is run, **Then** exactly two model figures are generated: one for GeoDT and one for GeoRF.
2. **Given** either generated model figure, **When** a reviewer inspects the layout, **Then** the rows correspond to fs1, fs2, and fs3, the columns correspond to precision, recall, and F1, and each applicable subplot contains partitioned, pooled, and FEWSNET comparison lines.
3. **Given** the x-axis labels in any subplot, **When** a reviewer reads them from left to right, **Then** test months are ordered chronologically.

---

### User Story 2 - Understand baseline comparison provenance (Priority: P2)

As a reviewer, I want a short note or manifest describing source paths, file selection logic, metric columns, and the FEWSNET fs3 proxy assumption, so that I can interpret the figures correctly and avoid treating the fs3 FEWSNET line as a native fs3 baseline.

**Why this priority**: The FEWSNET fs3 proxy rule is a required governance constraint and mislabeling it would make the comparison misleading.

**Independent Test**: Can be tested by reading the generated note or manifest and confirming that it identifies source files, model and scope identification rules, metric columns, missing-data handling, and the exact fs3 FEWSNET labeling assumption.

**Acceptance Scenarios**:

1. **Given** the generated manifest, **When** a reviewer checks source provenance, **Then** it lists the ablation source directory and FEWSNET baseline source directory used for the figures.
2. **Given** the fs3 row in either figure or its supporting manifest, **When** a reviewer checks the FEWSNET line label, **Then** it clearly states that FEWSNET fs2 values are reused for fs3 and does not imply a true fs3 FEWSNET baseline exists.

---

### User Story 3 - Validate with a lightweight smoke test (Priority: P3)

As a maintainer, I want a dry-run or reduced plotting validation path before relying on the full figures, so that I can verify data loading, line counts, legends, and chronological ordering without launching long batch workflows.

**Why this priority**: This supports reproducibility and protects the existing pipeline from unnecessary long-running work.

**Independent Test**: Can be tested by running the documented smoke-test path on a minimal subset or a single-model figure and checking the expected validation assertions.

**Acceptance Scenarios**:

1. **Given** a minimal subset of the monthly result files, **When** the smoke-test path is run, **Then** it verifies line count, legend labels, chronological test-month order, and FEWSNET fs3 proxy labeling.
2. **Given** the smoke-test path, **When** it is executed, **Then** it does not retrain models, does not run multi-hour batch workflows, and does not overwrite standard forecast deliverables.

---

### Edge Cases

- If a monthly result file is missing for GeoDT or GeoRF for any required scope, the figure generation must fail clearly or mark the affected model/scope as incomplete in the manifest rather than silently substituting another model family.
- If a required metric row is missing for either pooled or partitioned results in a month, the affected line must show a gap and the manifest must identify the missing model point; missing values must not be fabricated.
- If FEWSNET lacks an aligned baseline value for a plotted test month, the FEWSNET line must show a gap and the manifest must identify the missing comparison point.
- If GeoXGB files are present in the ablation directory, they must be ignored and must not affect the generated figures or manifest.
- If FEWSNET fs3 data is absent, fs3 must reuse FEWSNET fs2 values only as a clearly labeled comparison proxy.
- If output files with the intended names already exist, the feature must avoid overwriting standard forecast deliverables and must make overwritten diagnostic artifacts explicit in the manifest or output note.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST generate exactly two model figures for the full feature run: one GeoDT monthly performance figure and one GeoRF monthly performance figure.
- **FR-002**: System MUST exclude GeoXGB from file selection, plotting, labels, manifests, and success reporting.
- **FR-003**: Each model figure MUST contain a 3x3 subplot layout with rows representing fs1, fs2, and fs3, and columns representing precision, recall, and F1.
- **FR-004**: Each applicable subplot MUST plot three comparison series: partitioned model performance, pooled model performance, and FEWSNET baseline comparison.
- **FR-005**: Partitioned performance MUST be visually represented with a solid line, pooled performance MUST be represented with a dashed line in the same model color family, and FEWSNET baseline MUST use a distinct comparison color.
- **FR-006**: System MUST use the same styling convention across GeoDT and GeoRF figures.
- **FR-007**: System MUST display test month on the x-axis in chronological order and metric value on the y-axis.
- **FR-008**: System MUST include clear subplot titles, row or column labeling sufficient to identify each scope and metric, and a readable legend.
- **FR-009**: For fs3 FEWSNET comparison, System MUST reuse FEWSNET fs2 values as a proxy and clearly label this as “FEWSNET baseline (fs2 reused for fs3)” in the figure, caption, subplot annotation, or manifest.
- **FR-010**: System MUST NOT imply that FEWSNET has a native fs3 baseline.
- **FR-011**: System MUST save generated figures and the manifest under `main_ablation_results/march2026_main_backup_month_ind_cont3/monthly_performance_plots/`, separate from standard forecast deliverables.
- **FR-012**: System MUST produce a short note or manifest describing source paths, file selection logic, model-family identification, pooled-versus-partitioned identification, forecasting-scope identification, metric columns, missing-data handling, and the FEWSNET fs3 proxy assumption.
- **FR-016**: System MUST show gaps for missing plotted model or FEWSNET values and report each missing point in the manifest rather than fabricating or imputing missing metric values.
- **FR-013**: System MUST provide a smoke-test or dry-run validation path that loads a minimal subset or one single-model figure first and verifies line count, legend labels, chronological ordering, and FEWSNET fs3 labeling.
- **FR-014**: System MUST NOT change ACTIVE_LAGS, retrain models, run full multi-hour batch workflows, create a new parallel launcher unless necessary, or overwrite standard forecast deliverables.
- **FR-015**: System MUST identify the affected workflow as baseline comparison with exploratory diagnostic output unless the generated plots are explicitly promoted to deliverables in a later decision.

### Key Entities *(include if feature involves data)*

- **Model Performance Figure**: A generated visual artifact for a single model family, containing nine subplots that compare monthly precision, recall, and F1 across fs1, fs2, and fs3.
- **Monthly Metrics Record**: A result row for one test month and one model mode. Key attributes are test month, mode (`pooled` or `partitioned`), precision, recall, F1, and supporting counts.
- **FEWSNET Baseline Record**: A baseline comparison row keyed by year and quarter, with crisis-class precision, recall, and F1 values used for comparison against aligned test months.
- **Figure Manifest**: A short provenance artifact that records source paths, file selection rules, column mapping, missing-data handling, generated artifact names, and the FEWSNET fs3 comparison assumption.

### Pipeline & Data Assumptions *(mandatory for model/pipeline changes)*

- **Workflow**: Primary workflow mode is baseline comparison. The generated figures are exploratory diagnostics unless explicitly promoted to deliverables later.
- **Source Files Used**: The feature uses `metrics_monthly.csv` files under `main_ablation_results/march2026_main_backup_month_ind_cont3/result_partition_k40_compare_DT_fs1`, `..._DT_fs2`, `..._DT_fs3`, `..._GF_fs1`, `..._GF_fs2`, and `..._GF_fs3`. FEWSNET comparison values come from `fewsnet_baseline_results_backup/fewsnet_baseline_results_fs1.csv` and `fewsnet_baseline_results_backup/fewsnet_baseline_results_fs2.csv`.
- **Files Excluded**: `result_partition_k40_compare_XGB_fs*` files, seasonal performance files, prediction-level files, and polygon-overall files are out of scope for the required figures.
- **GeoDT vs GeoRF Identification**: GeoDT is identified by the `DT` model-family token in `result_partition_k40_compare_DT_fsN`; GeoRF is identified by the `GF` model-family token in `result_partition_k40_compare_GF_fsN`.
- **Pooled vs Partitioned Identification**: Pooled and partitioned lines are identified by the `model` column values `pooled` and `partitioned` in each `metrics_monthly.csv` file.
- **Forecasting Scope Identification**: fs1, fs2, and fs3 are identified by the `fs1`, `fs2`, and `fs3` suffixes in the result directory names and must map to the three figure rows in that order.
- **Metric Columns**: Ablation monthly metrics use `test_month`, `precision`, `recall`, and `f1`. FEWSNET baseline metrics use `year`, `quarter`, `precision(1)`, `recall(1)`, and `f1(1)`.
- **FEWSNET Time Alignment**: FEWSNET year/quarter rows are aligned to plotted test months by matching quarter to the existing test-month cadence: quarter 1 to `YYYY-02`, quarter 2 to `YYYY-06`, and quarter 4 to `YYYY-10`; quarter 3 is ignored when no corresponding test month is plotted.
- **Temporal Scope**: Existing ablation files inspected for this feature contain test months from 2021-02 through 2024-10 at the observed February, June, and October cadence. No new label construction, lag remapping, or synthetic target rows are part of this feature.
- **Crisis Label**: Metrics are crisis-class comparison metrics already present in the existing result files and FEWSNET baseline files; the feature does not redefine the crisis label.
- **Spatial Inputs**: The feature reads already aggregated monthly metrics and does not read partition maps, shapefiles, polygon IDs, or spatial adjacency artifacts.
- **Geographic Scope**: The feature compares existing aggregate evaluation outputs and does not render maps or perform shapefile joins.
- **Threshold Contract**: No prediction or scenario threshold is introduced or modified; the feature visualizes existing metrics only.
- **Prediction Partition Maps**: Not applicable to this diagnostic plotting feature because it uses existing monthly metrics, not prediction pipeline partition maps.
- **Artifacts**: Expected artifacts are one GeoDT figure, one GeoRF figure, and one short note or manifest written under `main_ablation_results/march2026_main_backup_month_ind_cont3/monthly_performance_plots/`.
- **Environment**: The feature must support a lightweight local validation path and must not depend on Windows batch execution of the full training or evaluation pipeline.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A full successful run produces exactly two model figures: one GeoDT figure and one GeoRF figure.
- **SC-002**: Each generated figure contains exactly nine subplots arranged as three forecasting-scope rows by three metric columns.
- **SC-003**: Each applicable subplot contains exactly three plotted comparison series: partitioned, pooled, and FEWSNET baseline.
- **SC-004**: In all subplots, x-axis values are ordered chronologically by test month.
- **SC-005**: GeoXGB contributes zero plotted lines and zero generated model figures.
- **SC-006**: Every fs3 FEWSNET comparison is labeled as reused fs2 baseline data and is never presented as a native fs3 FEWSNET baseline.
- **SC-007**: The generated note or manifest identifies all source paths, file selection rules, required columns, and missing-data behavior in under one page of text or equivalent concise structured content.
- **SC-008**: The smoke-test path verifies line count, legend labels, chronological ordering, and FEWSNET fs3 proxy labeling without launching retraining or long batch workflows.
- **SC-009**: No standard forecast deliverables are overwritten during feature validation or full figure generation.

## Assumptions

- The intended users are analysts and maintainers reviewing baseline-comparison diagnostics from existing ablation outputs.
- The existing ablation and FEWSNET baseline result files are authoritative for this feature; the feature does not recompute model metrics from raw predictions.
- The February, June, and October cadence observed in the monthly ablation files is the comparison cadence for the figures.
- Missing monthly data should be made visible as plotted gaps and reported in the manifest rather than imputed.
- Generated figures and manifests are diagnostic artifacts, not standard forecast deliverables, and are saved alongside the source ablation bundle in `monthly_performance_plots/`.
