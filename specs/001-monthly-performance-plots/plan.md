# Implementation Plan: Monthly Performance Plots

**Branch**: `001-monthly-performance-plots` | **Date**: 2026-05-14 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-monthly-performance-plots/spec.md`

**Note**: This plan covers an incremental baseline-comparison plotting feature. It does not propose a new stack, architecture, model family, lag schedule, partitioning method, launcher, notebook workflow, or training/evaluation batch run.

## Summary

Add a lightweight plotting script at `scripts/plot_monthly_performance_metrics.py` that reads existing monthly ablation metrics for GeoDT and GeoRF plus FEWSNET fs1/fs2 baseline comparison files, then writes exactly two diagnostic figures and one manifest under `main_ablation_results/march2026_main_backup_month_ind_cont3/monthly_performance_plots/`. The script will create a 3x3 subplot figure per model family, ignore GeoXGB entirely, plot partitioned/pooled/FEWSNET series for precision, recall, and F1 across fs1/fs2/fs3, and label fs3 FEWSNET as a fs2-reused comparison proxy because no native FEWSNET fs3 series exists.

## Technical Context

**Language/Version**: Python 3.12+ per repository requirements  
**Primary Dependencies**: Existing repository dependencies: pandas for CSV loading, matplotlib for plotting, standard library `json`/`pathlib`/`argparse` for manifest and CLI behavior  
**Storage**: Existing CSV result files as inputs; generated PNG figures plus JSON manifest as outputs  
**Testing**: Lightweight script smoke test/dry-run; no full batch jobs, model training, or notebooks  
**Target Platform**: Local repository execution from WSL or Windows-compatible Python environment; input paths are repository-relative by default but must tolerate the equivalent Windows source paths documented in the spec  
**Project Type**: Existing GeoRF research pipeline with standalone utility scripts  
**Performance Goals**: Load 8 small CSV files and generate two figures in seconds on a local development machine  
**Constraints**: Do not modify `ACTIVE_LAGS`; do not retrain models; do not run full batch workflows; do not alter target-month mapping; do not alter partition maps or spatial clustering semantics; do not use notebooks; do not commit generated figures, caches, pickles, or unrelated deliverables  
**Scale/Scope**: Six model metrics files for GeoDT/GeoRF fs1-fs3 plus two FEWSNET baseline CSVs; observed plotted cadence is February, June, and October for 2021-2024

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Pipeline Contract**: PASS. Workflow mode is baseline comparison with exploratory diagnostic output, not a promoted deliverable. Canonical entry point will be `scripts/plot_monthly_performance_metrics.py`. Required inputs are the six GeoDT/GeoRF `metrics_monthly.csv` files and the two FEWSNET baseline CSVs listed below. Expected outputs are generated diagnostic artifacts: `geodt_monthly_performance.png`, `georf_monthly_performance.png`, and `monthly_performance_manifest.json` under `main_ablation_results/march2026_main_backup_month_ind_cont3/monthly_performance_plots/`.
- **Temporal Integrity**: PASS. `ACTIVE_LAGS`, model training, temporal lag logic, feature-month mapping, target-month mapping, label construction, missing FEWSNET publication-month handling, and synthetic target-row handling are unchanged. fs1/fs2/fs3 are read only as labels from existing result directory suffixes. FEWSNET fs3 has no native input series; fs2 FEWSNET values are reused for the fs3 row only as a clearly labeled comparison proxy.
- **Spatial Partitioning**: PASS. Partition maps, spatial clustering, k40/KNN semantics, selected cluster counts, partition-map fallback behavior, and polygon mapping are not changed or read by this feature. The feature reads already aggregated monthly metrics.
- **Threshold & Prediction Contract**: PASS. Thresholds are not applicable because the feature plots existing precision, recall, and F1 metrics; it performs no prediction, scenario overlay, thresholding, or probability conversion.
- **Geographic Scope**: PASS. Shapefiles, map rendering, shapefile joins, and geographic-scope inference are not applicable because the feature renders metric line plots only.
- **Crisis-Class Validation**: PASS. The plotted metrics are existing crisis-class precision/recall/F1 values from repository result files. The feature does not redefine class 1 or recompute labels. Smoke validation will check line counts, legend labels, chronological month ordering, GeoXGB exclusion, and fs3 FEWSNET proxy labeling before full figure generation is accepted.
- **Operational Hygiene**: PASS. The plan uses an existing `scripts/` location and no new parallel launcher. It supports WSL-style relative paths and should not rely on user-specific absolute Windows Python paths. Generated PNG/JSON artifacts are output data and should not be committed unless explicitly promoted; no large outputs, caches, notebooks, pickles, shapefile caches, or unrelated deliverables should be committed.

### Required Constitution Check Answers

1. **Workflow mode**: Baseline comparison; exploratory diagnostics unless promoted by a later explicit decision.
2. **Production vs exploratory status**: Exploratory diagnostic output, not production forecast deliverables.
3. **Temporal/model changes**: No changes to `ACTIVE_LAGS`, model training, temporal lag logic, feature-month mapping, target-month mapping, labels, or synthetic rows.
4. **Partition changes**: No changes to partition maps, spatial clustering semantics, k40/KNN semantics, selected cluster counts, or polygon fallback behavior.
5. **Threshold relevance**: Not applicable; the feature plots existing metrics and does not generate predictions or apply thresholds.
6. **Geographic/shapefile relevance**: Not applicable; no shapefile joins, map rendering, or geographic-scope inference.
7. **Expected artifacts**: Generated diagnostic outputs: two PNG figures and one JSON manifest under `monthly_performance_plots/`; not source deliverables by default.
8. **Smoke-test path**: Run the plotting script in smoke mode for GeoDT or a reduced model subset to validate input loading, line count, legend labels, chronological ordering, GeoXGB exclusion, and fs3 FEWSNET proxy labeling without model jobs.
9. **Windows/WSL path considerations**: Default to repository-relative paths in the script; document that the Windows paths in the spec correspond to the same repository directories under WSL `/mnt/c/...`; preserve spaces in paths by using `pathlib` and quoted shell examples.
10. **Artifact hygiene**: Commit only source/spec/plan/docs changes. Do not commit generated PNGs, JSON manifests unless explicitly requested, caches, notebooks, pickles, shapefile caches, or unrelated deliverables.

## Project Structure

### Documentation (this feature)

```text
specs/001-monthly-performance-plots/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── plot-monthly-performance-cli.md
└── tasks.md              # Created later by Spec Kit Tasks, not this command
```

### Source Code (repository root)

```text
scripts/
├── plot_monthly_performance_metrics.py     # New lightweight plotting script
├── plot_seasonal_performance.py            # Existing related plotting script for reference only
└── baseline_comparison*.ipynb              # Existing notebooks; not used for this feature

main_ablation_results/march2026_main_backup_month_ind_cont3/
├── result_partition_k40_compare_DT_fs1/metrics_monthly.csv
├── result_partition_k40_compare_DT_fs2/metrics_monthly.csv
├── result_partition_k40_compare_DT_fs3/metrics_monthly.csv
├── result_partition_k40_compare_GF_fs1/metrics_monthly.csv
├── result_partition_k40_compare_GF_fs2/metrics_monthly.csv
├── result_partition_k40_compare_GF_fs3/metrics_monthly.csv
└── monthly_performance_plots/              # Generated diagnostic outputs; not committed by default
    ├── geodt_monthly_performance.png
    ├── georf_monthly_performance.png
    └── monthly_performance_manifest.json

fewsnet_baseline_results_backup/
├── fewsnet_baseline_results_fs1.csv
└── fewsnet_baseline_results_fs2.csv
```

**Structure Decision**: Implement as a single script in `scripts/` because the repository already keeps plotting utilities there and this feature is an incremental diagnostic utility. Do not add a new batch launcher, package, notebook, or architecture layer.

## Complexity Tracking

No constitution violations or architectural complexity exceptions are planned.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | N/A |

## Phase 0: Research

Research output is captured in [research.md](./research.md). Key decisions resolved:

- Use `scripts/plot_monthly_performance_metrics.py` as the entry point.
- Read only `metrics_monthly.csv` for DT/GF fs1-fs3 and FEWSNET fs1/fs2 baseline CSVs.
- Align FEWSNET quarters to the observed monthly cadence: Q1 -> February, Q2 -> June, Q4 -> October; ignore Q3 when no plotted test month exists.
- Treat FEWSNET fs3 as `fs2 reused for fs3` and label it explicitly.
- Show gaps for missing plotted values and list missing points in the manifest.

## Phase 1: Design and Contracts

Design outputs:

- [data-model.md](./data-model.md): Defines model figure, monthly metrics record, FEWSNET baseline record, aligned plot series, missing-point record, and manifest entities.
- [contracts/plot-monthly-performance-cli.md](./contracts/plot-monthly-performance-cli.md): Documents the script command contract, defaults, inputs, outputs, smoke mode, and validation behavior.
- [quickstart.md](./quickstart.md): Gives dry-run/smoke and full-generation commands without notebooks or long batch jobs.

## Post-Design Constitution Check

- **Pipeline Contract**: PASS. The plan, contract, and quickstart name the script entry point, required inputs, output directory, exact figure filenames, and manifest.
- **Temporal Integrity**: PASS. Design uses existing `fs1`/`fs2`/`fs3` suffixes only and explicitly states no lag or target-month changes.
- **Spatial Partitioning**: PASS. Design reads aggregate metrics only; no partition maps or shapefiles are used.
- **Threshold & Prediction Contract**: PASS. Thresholds remain not applicable.
- **Geographic Scope**: PASS. No map rendering or shapefile joins are planned.
- **Crisis-Class Validation**: PASS. Existing crisis-class metrics are plotted; smoke validation is defined.
- **Operational Hygiene**: PASS. Generated outputs stay in the clarified diagnostic directory and should remain uncommitted unless promoted.
