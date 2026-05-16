# Implementation Plan: Crisis Onset Analysis

**Branch**: `004-crisis-onset-analysis` | **Date**: 2026-05-16 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/004-crisis-onset-analysis/spec.md`

**Note**: This plan covers Phase 0 and Phase 1 design only. Implementation tasks are generated separately by `/speckit-tasks`.

## Summary

Create a lightweight crisis-onset follow-up to the existing phase-change monthly performance analysis. The implementation will minimally extend `scripts/plot_phase_change_monthly_performance.py` so the established source discovery, contract validation, previous-month fields, metric recomputation, plotting layout, summary-table logic, manifest structure, and smoke-test path can run in a new `crisis_onset` filter mode. Crisis-onset outputs will be written under `crisis_onset_analysis/` and will remain separate from standard monthly plots and the broader `any_phase_change` outputs.

## Technical Context

**Language/Version**: Python 3.12+  
**Primary Dependencies**: pandas, numpy, matplotlib, openpyxl or the repository's existing Excel writer dependency  
**Storage**: Local CSV, XLSX, PNG, and JSON files under existing repository output directories  
**Testing**: Dry-run and smoke path over one GeoRF or GeoDT prediction file plus direct validation of retained `0 -> 1` rows, row counts, metric recomputation, and artifact separation  
**Target Platform**: WSL/Linux command execution for development; output paths must remain understandable from the existing Windows project layout  
**Project Type**: Repository-local exploratory analysis script  
**Performance Goals**: Process the six in-scope prediction files without model training or multi-hour batch execution; smoke mode should run on one model/scope subset suitable for quick validation  
**Constraints**: No model retraining; no notebook execution; no ACTIVE_LAGS, lag-logic, partition-map, threshold, or prediction changes; no XGB/GeoXGB inclusion; no FEWSNET or original population-level rows in the summary table; no overwriting `monthly_performance_plots/` or `phase_change_monthly_performance/`; use ASCII-safe CLI/status text if output may be copied into Windows CMD workflows  
**Scale/Scope**: Six included source files: GeoRF/GF and GeoDT/DT for fs1, fs2, and fs3. The prior phase-change design inspected approximately 62k rows per file across labeled monthly test rows.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Pipeline Contract**: PASS. Affected workflow is exploratory diagnostics / baseline-comparison analysis. Canonical entry point remains `scripts/plot_phase_change_monthly_performance.py`, extended with a filter mode for `crisis_onset`. Inputs are existing `predictions_monthly.csv` files under `main_ablation_results/march2026_main_backup_month_ind_cont3/result_partition_k40_compare_{GF,DT}_fs{1,2,3}/`. Expected outputs are crisis-onset-only plots, metrics, summary workbook, optional filtered audit CSV, and manifest under `main_ablation_results/march2026_main_backup_month_ind_cont3/crisis_onset_analysis/`.
- **Temporal Integrity**: PASS. The feature reads existing labeled monthly prediction rows only. fs1 = 4 months, fs2 = 8 months, and fs3 = 12 months remain separate. No feature-month-to-target-month remapping is performed; `ACTIVE_LAGS` and all lag mapping logic remain unchanged. Crisis-onset comparison uses previous available test month within each spatial unit/model/scope series.
- **Spatial Partitioning**: PASS. No partition maps are created or changed. `partition_id` may be retained for audit context only; onset grouping uses `FEWSNET_admin_code`. Existing pooled vs partitioned semantics are read from `y_pred_pooled` and `y_pred_partitioned`.
- **Threshold & Prediction Contract**: PASS. No prediction threshold is introduced or changed. The feature analyzes already materialized binary prediction columns and excludes scenario thresholds.
- **Geographic Scope**: PASS. No map rendering or shapefile joining is in scope. The analysis inherits the geographic coverage of the existing prediction files and must not imply any new shapefile scope.
- **Crisis-Class Validation**: PASS. Class-1 crisis evidence is recomputed as precision, recall, and F1 using `y_true` as the binary positive crisis label. Crisis-onset eligibility is true-label `0 -> 1`, not prediction-label change. Dry-run and smoke validation are required before full artifact generation.
- **Operational Hygiene**: PASS. Work is performed from WSL/Linux-style commands against existing files; no Windows batch launcher is required. Generated outputs stay under a clearly labeled crisis-onset directory and should remain out of source control unless explicitly promoted. No notebooks or large scratch artifacts are created.

## Project Structure

### Documentation (this feature)

```text
specs/004-crisis-onset-analysis/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── crisis-onset-analysis-cli.md
└── tasks.md
```

### Source Code (repository root)

```text
scripts/
├── plot_monthly_performance_metrics.py          # Existing standard monthly plotter; reference only
└── plot_phase_change_monthly_performance.py     # Existing phase-change entry point to extend with filter modes

main_ablation_results/march2026_main_backup_month_ind_cont3/
├── result_partition_k40_compare_GF_fs1/predictions_monthly.csv
├── result_partition_k40_compare_GF_fs2/predictions_monthly.csv
├── result_partition_k40_compare_GF_fs3/predictions_monthly.csv
├── result_partition_k40_compare_DT_fs1/predictions_monthly.csv
├── result_partition_k40_compare_DT_fs2/predictions_monthly.csv
├── result_partition_k40_compare_DT_fs3/predictions_monthly.csv
├── result_partition_k40_compare_XGB_fs*/predictions_monthly.csv      # excluded and documented
├── monthly_performance_plots/                                        # existing standard outputs; do not overwrite
├── phase_change_monthly_performance/                                 # existing broader any_phase_change outputs; do not overwrite
└── crisis_onset_analysis/                                            # new generated outputs
    ├── georf_crisis_onset_monthly_performance.png
    ├── geodt_crisis_onset_monthly_performance.png
    ├── metrics_monthly_crisis_onset.csv
    ├── summary_crisis_onset.xlsx
    ├── filtered_predictions_crisis_onset.csv
    └── crisis_onset_manifest.json
```

**Structure Decision**: Reuse the existing phase-change analysis script rather than creating a parallel script. Add a filter-mode/output-profile seam so `any_phase_change` remains available while `crisis_onset` writes to its own generated-output directory and file names.

## Complexity Tracking

No constitution violations are introduced; no complexity exemptions are required.

## Phase 0: Research Summary

See [research.md](research.md). Key decisions:

- Extend the existing phase-change script with a filter-mode/output-profile seam rather than duplicating analysis logic.
- Preserve `any_phase_change` as the broader mode and add `crisis_onset` as the narrower `previous_y_true = 0` and `y_true = 1` mode.
- Default crisis-onset outputs to `crisis_onset_analysis/` with crisis-onset-specific filenames and labels.
- Keep GeoRF/GF and GeoDT/DT fs1-fs3 only; continue documenting XGB exclusions.
- Recompute monthly and summary metrics from filtered row-level predictions, using blank/NA for undefined metrics and manifest entries for zero denominators.

## Phase 1: Design Summary

See [data-model.md](data-model.md), [contracts/crisis-onset-analysis-cli.md](contracts/crisis-onset-analysis-cli.md), and [quickstart.md](quickstart.md).

Implementation design:

1. Introduce a filter-mode concept with two canonical modes: `any_phase_change` and `crisis_onset`.
2. Keep existing source discovery, XGB exclusion discovery, required-column validation, binary-label validation, and duplicate detection unchanged.
3. Keep existing previous available test-month fields: `previous_month_start`, `previous_y_true`, and `is_first_observation`.
4. Derive filter-mode-specific retained rows: `any_phase_change` retains `y_true != previous_y_true`; `crisis_onset` retains `previous_y_true = 0` and `y_true = 1`.
5. Generalize row-count labels so provenance records before-filter rows, first-observation exclusions, mode-specific exclusions, and retained rows by model/scope.
6. Generalize metric and summary code so labels, row-count column names, sheet names, plot titles, and manifest fields reflect the active mode while reusing the same calculation functions.
7. Default `crisis_onset` full-mode output to `main_ablation_results/march2026_main_backup_month_ind_cont3/crisis_onset_analysis/`; keep `any_phase_change` default at `phase_change_monthly_performance/`.
8. Render crisis-onset-only plots using the same GeoRF/GeoDT colors and fs1-fs3 by precision/recall/F1 layout where practical, without FEWSNET or XGB series.
9. Write audit CSVs, summary workbook, plots, and manifest to `crisis_onset_analysis/` for crisis-onset mode.
10. Provide dry-run and smoke behavior that validates the crisis-onset subset before full artifact generation.

## Post-Design Constitution Check

- **Pipeline Contract**: PASS. The planned script extension, inputs, outputs, and manifest provenance are explicit; standard and broader phase-change outputs remain separate.
- **Temporal Integrity**: PASS. The design reads existing labeled prediction rows only, preserves fs1 = 4 months, fs2 = 8 months, and fs3 = 12 months as separate scopes, treats `month_start` as the existing target/test month, performs no feature-month-to-target-month remapping, and changes no lag logic.
- **Spatial Partitioning**: PASS. No partitioning semantics are changed; pooled and partitioned series are read from existing prediction columns.
- **Threshold & Prediction Contract**: PASS. No threshold is introduced; existing binary predictions are analyzed as-is.
- **Geographic Scope**: PASS. No map rendering or shapefile joining is planned.
- **Crisis-Class Validation**: PASS. The design recomputes class-1 precision, recall, and F1 from crisis-onset-filtered row-level labels and predictions.
- **Operational Hygiene**: PASS. Outputs are isolated under a new generated directory, no notebook execution is planned, and no standard or broader phase-change deliverables are overwritten.
