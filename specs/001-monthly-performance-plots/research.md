# Research: Monthly Performance Plots

## Decision: Implement as an existing-style plotting script under `scripts/`

**Rationale**: The repository already keeps plotting utilities such as `scripts/plot_seasonal_performance.py`, `scripts/plot_f1_improvement_comparison.py`, and map plotting scripts under `scripts/`. This feature is an incremental diagnostic plotting utility and does not need a new architecture, batch launcher, notebook, or package.

**Alternatives considered**:
- New Windows batch launcher: rejected because the user requested no new launcher unless necessary and this feature does not require CMD orchestration.
- Notebook: rejected because the plan must not run notebooks and notebook outputs create artifact-hygiene risk.
- New package/module tree: rejected because the existing `scripts/` location is sufficient and aligned with repository practice.

## Decision: Use the six GeoDT/GeoRF monthly metrics files as model inputs

**Rationale**: The required figure rows and model families map directly to existing Stage 3 comparison output directories. The selected files are:

- `main_ablation_results/march2026_main_backup_month_ind_cont3/result_partition_k40_compare_DT_fs1/metrics_monthly.csv`
- `main_ablation_results/march2026_main_backup_month_ind_cont3/result_partition_k40_compare_DT_fs2/metrics_monthly.csv`
- `main_ablation_results/march2026_main_backup_month_ind_cont3/result_partition_k40_compare_DT_fs3/metrics_monthly.csv`
- `main_ablation_results/march2026_main_backup_month_ind_cont3/result_partition_k40_compare_GF_fs1/metrics_monthly.csv`
- `main_ablation_results/march2026_main_backup_month_ind_cont3/result_partition_k40_compare_GF_fs2/metrics_monthly.csv`
- `main_ablation_results/march2026_main_backup_month_ind_cont3/result_partition_k40_compare_GF_fs3/metrics_monthly.csv`

Each file uses columns `test_month`, `model`, `precision`, `recall`, and `f1`; `model` values identify `pooled` and `partitioned` series. The scope is identified by the containing `fs1`, `fs2`, or `fs3` directory suffix. GeoDT is identified by `DT`; GeoRF is identified by `GF`.

**Alternatives considered**:
- Use `metrics_polygon_overall.csv`: rejected because the feature requires monthly x-axis series.
- Use `predictions_monthly.csv`: rejected because metrics are already computed and the feature must not recompute model performance.
- Use `main_month_ind_cont3.xlsx`: rejected because the inspected CSV files provide a clearer per-scope contract.

## Decision: Exclude GeoXGB by explicit allowlist

**Rationale**: The source directory contains `result_partition_k40_compare_XGB_fs*` files, but the feature goal explicitly excludes GeoXGB. An allowlist of model tokens `DT` and `GF` prevents accidental inclusion if globbing over result directories.

**Alternatives considered**:
- Glob all `result_partition_k40_compare_*` directories and filter later: rejected because it increases the chance of accidental XGB inclusion in manifest or validation output.

## Decision: Use FEWSNET fs1 and fs2 baseline CSVs only

**Rationale**: FEWSNET baseline source files are:

- `fewsnet_baseline_results_backup/fewsnet_baseline_results_fs1.csv`
- `fewsnet_baseline_results_backup/fewsnet_baseline_results_fs2.csv`

They contain `year`, `quarter`, `precision(1)`, `recall(1)`, and `f1(1)`. The feature requires FEWSNET comparison lines for fs1, fs2, and fs3, but FEWSNET has no native fs3 series. Therefore fs1 uses the FEWSNET fs1 file, fs2 uses the FEWSNET fs2 file, and fs3 reuses FEWSNET fs2 values as a clearly labeled comparison proxy.

**Alternatives considered**:
- Omit FEWSNET for fs3: rejected because the spec requires a FEWSNET comparison line in each applicable subplot.
- Invent or interpolate fs3 FEWSNET values: rejected because it would imply unsupported baseline data.
- Treat fs2 reused as native fs3: rejected because it violates the baseline-comparison rule.

## Decision: Align FEWSNET quarter rows to observed test-month cadence

**Rationale**: The inspected monthly metrics use February, June, and October test months. FEWSNET rows are keyed by year and quarter. The alignment contract is:

- Quarter 1 -> `YYYY-02`
- Quarter 2 -> `YYYY-06`
- Quarter 4 -> `YYYY-10`
- Quarter 3 -> ignored unless future plotted model data introduces a matching month contract

This preserves chronological month plotting while using available baseline keys.

**Alternatives considered**:
- Plot FEWSNET by quarter labels instead of month labels: rejected because the required x-axis is test month.
- Map Q3 to an arbitrary missing month: rejected because no inspected model metric uses that cadence.

## Decision: Show missing plotted values as gaps and record them in the manifest

**Rationale**: The clarified spec requires gaps for missing model or FEWSNET plotted values. This avoids fabricating metrics while keeping available series visible.

**Alternatives considered**:
- Fail on any missing values: rejected by clarification.
- Impute missing values: rejected because it would fabricate evaluation results.

## Decision: Write diagnostic outputs beside the source ablation bundle

**Rationale**: The clarified output directory is `main_ablation_results/march2026_main_backup_month_ind_cont3/monthly_performance_plots/`. This keeps figures close to their source result bundle while separate from standard forecast deliverables.

**Alternatives considered**:
- `other_outputs/monthly_performance_plots/`: rejected because it separates the figures from the specific ablation bundle provenance.
- `deliverables/monthly_performance_plots/`: rejected because the feature is exploratory diagnostics, not promoted deliverables.
