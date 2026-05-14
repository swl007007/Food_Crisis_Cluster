# Implementation Plan: Actual Predicted Dashboard

**Branch**: `002-actual-predicted-dashboard` | **Date**: 2026-05-14 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-actual-predicted-dashboard/spec.md`

**Note**: This template is filled in by the Spec Kit Plan command or skill. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Generate an exploratory static HTML dashboard that compares actual versus partitioned predicted crisis distributions for existing GeoRF and GeoDT month-ind evaluation outputs. The implementation will add a lightweight generator under the existing `scripts/` tooling area, reuse the current CSV/shapefile data contract, embed compact browser-ready map/data payloads into one self-contained HTML file by default, and write a manifest beside the dashboard without rerunning training or standard batch workflows.

## Technical Context

**Language/Version**: Python 3.12+ for generation; static HTML/CSS/browser JavaScript for local viewing
**Primary Dependencies**: Existing project geospatial/data stack: pandas, GeoPandas, Shapely, NumPy; standard-library JSON/path handling; no server framework or Leaflet-style GIS frontend
**Storage**: Local filesystem inputs and generated artifacts only; source CSVs and shapefile are read-only
**Testing**: Generator smoke/dry-run validation, data-contract checks, manifest inspection, and manual browser validation of selector/panel behavior
**Target Platform**: Local project workspace, generated from WSL or Windows with the project geospatial environment and opened in a desktop browser
**Project Type**: Existing repository diagnostic/tooling script plus generated static dashboard artifact
**Performance Goals**: Dashboard opens and selectors populate in under 30 seconds for included GeoRF/GeoDT fs1-fs3 month-ind data on a typical workstation
**Constraints**: No model retraining, no full batch execution, no notebook execution, no `ACTIVE_LAGS` changes, no partition-map or clustering changes, no standard forecast/scenario deliverable overwrites, self-contained HTML preferred with supporting assets only if needed for size/performance
**Scale/Scope**: Six included prediction files (GeoRF/GeoDT x fs1/fs2/fs3), each inspected at 62,189 polygon-month rows across 12 test months; XGB folders are discovered only to document exclusion; global FEWSNET shapefile is the spatial source

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Initial Gate

- **Pipeline Contract**: PASS. Affected workflow is exploratory diagnostics/tooling. Canonical entry point will be a generator script under `scripts/`; required inputs are existing `predictions_monthly.csv` files and the global FEWSNET shapefile; expected outputs are one dashboard and one manifest in an exploratory output directory.
- **Temporal Integrity**: PASS. The feature reads `month_start` from existing labeled outputs, includes fs1/fs2/fs3 as folder suffixes, does not recompute feature-month to target-month mappings, and leaves `ACTIVE_LAGS` unchanged.
- **Spatial Partitioning**: PASS. The feature consumes post-evaluation `y_pred_partitioned` labels only and does not read, generate, modify, or reinterpret partition maps. `k40` remains inherited from source folder names and is not described as 40 clusters.
- **Threshold & Prediction Contract**: PASS. No prediction threshold is introduced because the dashboard plots existing binary `y_true` and `y_pred_partitioned` labels. Probability-to-label conversion remains out of scope unless explicitly replanned.
- **Geographic Scope**: PASS. Map joining/rendering must use the global FEWSNET shapefile `FEWS_Admin_LZ_v3.shp`; Nigeria-only defaults are explicitly rejected.
- **Crisis-Class Validation**: PASS. Class 1 is the repository positive crisis label from existing `y_true`/`y_pred_partitioned` values. This is a labeled diagnostic, so smoke validation focuses on non-empty joins and actual-vs-predicted display rather than rerunning model metrics.
- **Operational Hygiene**: PASS. Outputs remain in an exploratory result subdirectory, source CSVs are read-only, no notebook execution is required, and any Windows-facing launcher/status text must remain ASCII-safe if added later.

### Post-Design Gate

- **Pipeline Contract**: PASS. Research, data model, contracts, and quickstart define the generator inputs, dashboard/manifest outputs, and validation path.
- **Temporal Integrity**: PASS. Design uses only discovered `month_start` values and does not alter forecasting scope behavior.
- **Spatial Partitioning**: PASS. Design preserves source partitioned predictions as data fields and makes no partition-map changes.
- **Threshold & Prediction Contract**: PASS. Contracts require binary labels and reject silent thresholding.
- **Geographic Scope**: PASS. Contracts and quickstart require global FEWSNET shapefile provenance.
- **Crisis-Class Validation**: PASS. Data-model validation requires binary 0/1 values and smoke-test comparison for one model-scope-month.
- **Operational Hygiene**: PASS. Quickstart avoids full batch/training/notebooks and writes to `main_ablation_results/march2026_main_backup_month_ind_cont3/actual_predicted_dashboard/`.

## Project Structure

### Documentation (this feature)

```text
specs/002-actual-predicted-dashboard/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── dashboard-generation-contract.md
│   └── manifest.schema.json
└── tasks.md
```

### Source Code (repository root)

```text
scripts/
├── plot_predictions_2024.py                 # Existing reference plotting/data-join logic
└── generate_actual_predicted_dashboard.py    # Planned dashboard generator

main_ablation_results/march2026_main_backup_month_ind_cont3/
├── result_partition_k40_compare_GF_fs1/predictions_monthly.csv
├── result_partition_k40_compare_GF_fs2/predictions_monthly.csv
├── result_partition_k40_compare_GF_fs3/predictions_monthly.csv
├── result_partition_k40_compare_DT_fs1/predictions_monthly.csv
├── result_partition_k40_compare_DT_fs2/predictions_monthly.csv
├── result_partition_k40_compare_DT_fs3/predictions_monthly.csv
├── result_partition_k40_compare_XGB_fs*/     # Input discovery documents exclusion only
└── actual_predicted_dashboard/               # Planned exploratory output directory
    ├── actual_predicted_dashboard.html
    ├── manifest.json
    └── assets/                               # Optional only if HTML size/performance requires it
```

**Structure Decision**: Keep implementation as one existing-repository diagnostic script under `scripts/` and generated artifacts under the source result directory’s exploratory output subfolder. Do not add a backend server, notebook, new batch launcher, or independent application structure.

## Complexity Tracking

No constitution violations or complexity exceptions are required.
