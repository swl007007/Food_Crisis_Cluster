# Implementation Plan: Phase-Change Monthly Performance

**Branch**: `003-phase-change-performance` | **Date**: 2026-05-15 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/003-phase-change-performance/spec.md`

**Note**: This plan covers Phase 0 and Phase 1 design only. Implementation tasks are generated separately by `/speckit-tasks`.

## Summary

Create an exploratory phase-change-only monthly performance analysis over existing row-level `predictions_monthly.csv` outputs. The implementation will add a dedicated analysis entry point under `scripts/` that reads GeoRF/GF and GeoDT/DT fs1-fs3 prediction files, excludes XGB, filters rows where `y_true` changes from the previous available test month for each `FEWSNET_admin_code`, recomputes class-1 precision/recall/F1 for pooled and partitioned series, and writes clearly labeled plots, tables, audit files, and provenance under a new phase-change output directory.

## Technical Context

**Language/Version**: Python 3.12+  
**Primary Dependencies**: pandas, numpy, matplotlib, openpyxl or the repository's existing Excel writer dependency  
**Storage**: Local CSV, XLSX, PNG, and JSON files under existing repository output directories  
**Testing**: Smoke/dry-run path over one GeoRF or GeoDT prediction file plus direct validation of row counts, first-row exclusion, metric recomputation, and artifact manifest  
**Target Platform**: WSL/Linux command execution for development; output paths must also be understandable from the existing Windows project layout  
**Project Type**: Repository-local exploratory analysis script  
**Performance Goals**: Process the six in-scope prediction files, each approximately 62k rows, without model training or multi-hour batch execution; smoke mode should run on a reduced file/month subset suitable for quick validation  
**Constraints**: No model retraining; no notebook execution; no ACTIVE_LAGS, lag-logic, partition-map, or prediction changes; no XGB/GeoXGB inclusion; no FEWSNET or original population-level rows in the summary table; no overwriting `monthly_performance_plots` or standard deliverables; use ASCII-safe CLI/status text if output may be copied into Windows CMD workflows  
**Scale/Scope**: Six included source files: GeoRF/GF and GeoDT/DT for fs1, fs2, and fs3. Inspected files contain 12 test months from 2021-02-01 through 2024-10-01 and 5,713 spatial units per file, with 62,189 rows per file.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Pipeline Contract**: PASS. Affected workflow is exploratory diagnostics / baseline-comparison analysis. Canonical entry point will be a repository-local script under `scripts/`. Inputs are existing `predictions_monthly.csv` files under `main_ablation_results/march2026_main_backup_month_ind_cont3/result_partition_k40_compare_{GF,DT}_fs{1,2,3}/`. Outputs are phase-change-only plots, recomputed metric tables, optional filtered audit CSVs, and a manifest/README under a new phase-change output directory.
- **Temporal Integrity**: PASS. The feature uses existing labeled prediction rows only: fs1 = 4 months, fs2 = 8 months, and fs3 = 12 months. The target/test months are the materialized `month_start` rows already present in the prediction files, inspected as 2021-02-01 through 2024-10-01. No feature-month-to-target-month remapping is performed; `ACTIVE_LAGS` and all lag mapping logic remain unchanged. Phase-change comparison uses previous available test month within each spatial unit/model/scope series. Missing FEWSNET publication months and synthetic target rows are not introduced or touched.
- **Spatial Partitioning**: PASS. No partition maps are created or changed. `partition_id` may be retained for audit context only; phase-change grouping uses `FEWSNET_admin_code`. Existing pooled vs partitioned semantics are read from `y_pred_pooled` and `y_pred_partitioned`.
- **Threshold & Prediction Contract**: PASS. No prediction thresholds are introduced or changed. The feature analyzes already materialized binary prediction columns and excludes scenario thresholds.
- **Geographic Scope**: PASS. No map rendering or shapefile joining is in scope. The analysis inherits the geographic coverage of the existing prediction files and must not imply any new shapefile scope.
- **Crisis-Class Validation**: PASS. Class-1 crisis evidence is recomputed as precision, recall, and F1 using `y_true` as the binary positive crisis label. Smoke validation is required before full artifact generation.
- **Operational Hygiene**: PASS. Work is performed from WSL/Linux-style commands against existing files; no Windows batch launcher is required. Generated outputs stay under a clearly labeled phase-change directory and should remain out of source control unless explicitly promoted. No notebooks or large scratch artifacts are created.

## Project Structure

### Documentation (this feature)

```text
specs/003-phase-change-performance/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── phase-change-analysis-cli.md
└── tasks.md
```

### Source Code (repository root)

```text
scripts/
├── plot_monthly_performance_metrics.py          # Existing reference style and manifest conventions
└── plot_phase_change_monthly_performance.py     # New phase-change-only analysis entry point

main_ablation_results/march2026_main_backup_month_ind_cont3/
├── result_partition_k40_compare_GF_fs1/predictions_monthly.csv
├── result_partition_k40_compare_GF_fs2/predictions_monthly.csv
├── result_partition_k40_compare_GF_fs3/predictions_monthly.csv
├── result_partition_k40_compare_DT_fs1/predictions_monthly.csv
├── result_partition_k40_compare_DT_fs2/predictions_monthly.csv
├── result_partition_k40_compare_DT_fs3/predictions_monthly.csv
├── result_partition_k40_compare_XGB_fs*/predictions_monthly.csv  # excluded and documented
├── monthly_performance_plots/                                  # existing outputs; do not overwrite
└── phase_change_monthly_performance/                           # new generated outputs
    ├── georf_phase_change_monthly_performance.png
    ├── geodt_phase_change_monthly_performance.png
    ├── metrics_monthly_phase_change.csv
    ├── summary_phase_change.xlsx
    ├── filtered_predictions_phase_change.csv
    └── phase_change_manifest.json
```

**Structure Decision**: Use a dedicated script in the existing `scripts/` analysis area and a dedicated generated-output directory under the existing ablation root. This keeps exploratory phase-change artifacts separate from standard `monthly_performance_plots` and avoids adding a new batch launcher.

## Complexity Tracking

No constitution violations are introduced; no complexity exemptions are required.

## Phase 0: Research Summary

See [research.md](research.md). Key decisions:

- Use a dedicated phase-change analysis script rather than extending the existing standard monthly plotter with mode flags.
- Select only GF/GeoRF and DT/GeoDT prediction folders for fs1, fs2, and fs3; record XGB exclusions.
- Compare each spatial unit to the previous available test month, not strict previous calendar month.
- Represent undefined metrics as blank/NA and document zero-denominator reasons in the manifest.
- Aggregate summary-table metrics by recomputing from all filtered phase-change rows pooled across months for each model, scope, and series.

## Phase 1: Design Summary

See [data-model.md](data-model.md), [contracts/phase-change-analysis-cli.md](contracts/phase-change-analysis-cli.md), and [quickstart.md](quickstart.md).

Implementation design:

1. Resolve the ablation root path and output path without overwriting existing standard outputs.
2. Discover or explicitly enumerate the six in-scope prediction files.
3. Validate required source columns: `FEWSNET_admin_code`, `month_start`, `y_true`, `y_pred_pooled`, `y_pred_partitioned`, with optional retention of `partition_id` for audit context.
4. Sort each model/scope/spatial-unit series by `month_start`, exclude first observations, flag duplicate month ambiguity, and retain only rows where current `y_true` differs from previous available `y_true`.
5. Recompute monthly metrics for pooled and partitioned predictions using only retained rows, with blank/NA for undefined metrics and manifest entries for zero denominators.
6. Recompute summary-table metrics from all filtered phase-change rows pooled across months for each model/scope/series.
7. Render phase-change-only monthly plots using the existing GeoRF/GeoDT colors and fs1-fs3 by metric layout where practical, without FEWSNET series.
8. Write audit CSVs, summary workbook, plots, and manifest to `phase_change_monthly_performance/`.
9. Provide smoke/dry-run behavior that validates a reduced source subset before full artifact generation.

## Post-Design Constitution Check

- **Pipeline Contract**: PASS. The planned script, inputs, outputs, and manifest provenance are explicit.
- **Temporal Integrity**: PASS. The design reads existing labeled prediction rows only, preserves fs1 = 4 months, fs2 = 8 months, and fs3 = 12 months as separate scopes, treats `month_start` as the existing target/test month, performs no feature-month-to-target-month remapping, and changes no lag logic.
- **Spatial Partitioning**: PASS. No partitioning semantics are changed; pooled and partitioned series are read from existing prediction columns.
- **Threshold & Prediction Contract**: PASS. No threshold is introduced; existing binary predictions are analyzed as-is.
- **Geographic Scope**: PASS. No map rendering or shapefile joining is planned.
- **Crisis-Class Validation**: PASS. The design recomputes class-1 precision, recall, and F1 from filtered row-level labels and predictions.
- **Operational Hygiene**: PASS. Outputs are isolated under a new generated directory, no notebook execution is planned, and no standard deliverables are overwritten.
