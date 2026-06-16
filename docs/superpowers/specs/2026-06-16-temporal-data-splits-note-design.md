# Temporal Data Splits Note Design

## Purpose

Add a reviewer-response documentation artifact under
`final_artifacts_in_paper_updated/` that clarifies the temporal data used by the
current no-leak GeoRF/GeoDT workflow. The note must give readers a schematic and
compact tables showing, for each target-month/horizon rule, the training period,
split-acceptance validation data, Stage 2 consensus data, threshold-selection
data, hyperparameter-selection data, and final test data.

The appendix should not enumerate all candidate target-month x horizon rows.
Instead, it should define the implemented formula and use compact tables that
distinguish the configured Stage 3 candidate window from the currently
evaluated result rows.

## Scope

Create final-artifact documentation only. No model code, batch scripts,
pipeline outputs, or manuscript files outside `final_artifacts_in_paper_updated/`
are modified.

Planned final artifacts:

1. `final_artifacts_in_paper_updated/temporal_data_splits_schematic_note.md`
   - Chinese audit section first.
   - English appendix-ready section last.
   - Includes a compact ASCII schematic.
   - Includes compact tables, not an all-candidate appendix table.
2. `final_artifacts_in_paper_updated/temporal_data_splits_table.csv`
   - Audit/reproducibility companion table with one row per actually evaluated
     final-test target month and forecasting scope.
   - Current evaluated result rows cover February, June, and October for
     2021-2024 across `fs1`, `fs2`, and `fs3`, for 36 rows total.
   - May be referenced as an artifact-folder companion, but the appendix text
     should remain compact and formula-based.

## Evidence Base

The note will be grounded in current implementation and existing artifacts:

- Workflow split:
  `PIPELINE_WORKFLOW.md`,
  `run_batches_2018_2020_partition_learning_visual_monthly.bat`,
  `spatial_weighted_consensus_clustering.bat`,
  `run_partition_k40_comparison_unified.bat`.
- Monthly rolling window:
  `src/customize/customize.py::train_test_split_rolling_window`.
- Forecasting horizons:
  `config.py` and `src/utils/lag_schedules.py`.
- Internal train/validation split for recursive partition acceptance:
  `src/model/GeoRF.py`, `src/model/GeoRF_DT.py`, `src/utils/split.py`,
  `src/initialization/initialization.py`.
- GeoDT max-depth selection:
  `app/main_model_DT.py` and `src/model/model_DT.py::select_dt_max_depth`.
- Stage 3 fixed-parameter comparison and hard predictions:
  `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py`.
- Current Stage 2 consensus inputs and maps:
  `GeoRFExperiment/linked_tables/main_index.csv`,
  `GeoDTExperiment/linked_tables/main_index.csv`,
  `GeoRFExperiment/knn_sparsification_results/cluster_mapping_manifest.json`,
  `GeoDTExperiment/knn_sparsification_results/cluster_mapping_manifest.json`,
  and `Geo*Experiment/similarity_matrices*/summary_statistics*.json`.

## Appendix Structure

The Markdown note should include these sections.

### Chinese Audit

Summarize the implemented temporal separation:

- Stage 1 learns partition candidates on 2018-2020 target months.
- Stage 2 builds consensus maps only from Stage 1 linked partition plans.
- Stage 3 is configured with a candidate target-month loop from 2021-01 through
  2024-12 (`n_test_months=48` in run manifests), but current evaluated result
  rows exist only for February, June, and October in each year
  (`n_test_months_evaluated=12` per scope in run manifests).
- The appendix will not claim independent threshold tuning or extra sensitivity
  analyses that are not implemented.

### Schematic

Use a text schematic showing:

```text
Stage 1: 2018-2020 partition-learning runs
    rolling temporal train window -> internal validation subset -> split acceptance
        |
        v
Stage 2: consensus maps from 2018-2020 linked plans only
    general map + month-specific maps for Feb/Jun/Oct
        |
        v
Stage 3: configured 2021-01..2024-12 candidate loop
    current evaluated rows: February, June, October only
    rolling temporal train window -> fixed model params -> final test month T
```

### Compact Tables

Table 1: stage-level data-use table.

- Rows: Stage 1 partition learning, Stage 2 consensus clustering, Stage 3 final
  evaluation, threshold selection, hyperparameter selection.
- Columns: data window, data role, implementation detail, leakage guard.

Table 2: horizon rule table.

- Rows: `fs1`, `fs2`, `fs3`.
- Columns: horizon months, Stage 3 training period formula, split-acceptance
  validation data, final test period formula.
- Formula must follow current code:
  - For target month `T` and horizon `h`, `train_end = T - h months`.
  - `train_start = train_end - 35 months`.
  - Training mask is `[train_start, train_end)`.
  - Final test mask is `[T, T + 1 month)`.

Table 3: partition-map selection by target calendar month.

- Rows: 12 calendar months.
- Columns: Stage 3 partition map used and Stage 2 consensus input filter.
- February uses `m2`, June uses `m6`, October uses `m10`; other months use
  `general` if they are configured candidates with result data in a future run.
- The table must not imply that non-2/6/10 months have final test rows in the
  current results. It should explicitly state that current evaluated result rows
  are only for February, June, and October.

The 36-row companion CSV should include exact computed date ranges for the
actually evaluated result rows, while the appendix text should cite compact
rules and examples.

## Paper-Facing Claims Allowed

The English appendix may state:

- Stage 1/2 use 2018-2020 partition-learning data. Stage 3 is configured with
  2021-01 through 2024-12 candidate target months, while the current evaluated
  final-test result rows cover February, June, and October for 2021-2024.
- For each Stage 3 target month `T` and horizon `h`, the rolling train/test
  dates are determined by the code formula above.
- Stage 1 split acceptance uses an internal validation subset from the rolling
  training window, created by group-aware validation splitting when enabled.
- Current `GROUP_SPLIT` uses `val_ratio=0.20`, `min_val_per_group=1`,
  `skip_singleton_groups=True`, and `random_state=42`.
- Stage 2 general consensus uses the current linked plans in
  `main_index.csv`; current artifacts contain 24 GeoRF plans and 27 GeoDT plans,
  with month-specific consensus maps based on February, June, and October plan
  subsets.
- Threshold-selection data are not separate in the standard Stage 3 comparison;
  the current comparison path uses classifier hard predictions.
- GeoRF does not use a separate hyperparameter-selection dataset in this
  workflow. GeoDT Stage 1 selects `max_depth` from configured candidates using
  validation class-1 F1 inside the rolling training window. Stage 3 comparison
  uses fixed RF/DT parameters.

## Claims To Exclude

The appendix must not claim:

- The appendix or companion CSV covers all configured 2021-2024 candidate
  months as evaluated final-test rows.
- Non-2/6/10 target months have current final-test result rows.
- Stage 3 threshold tuning, probability calibration, AUC/log-loss optimization,
  or an independent threshold-selection validation set.
- A separate final-test-period hyperparameter tuning step.
- Stage 2 uses 2021-2024 outcomes or any final-test data.
- k sensitivity, spatial-kernel sensitivity, or other experiments not run for
  this artifact.

## Verification

Implementation verification should check:

- `temporal_data_splits_schematic_note.md` exists and ends with the English
  appendix-ready section.
- The appendix uses compact tables and formula-based coverage, not an
  all-candidate printed table.
- `temporal_data_splits_table.csv` exists and has 36 rows for 4 years x 3
  evaluated calendar months x 3 scopes, if the companion CSV is created.
- Computed CSV dates match `train_test_split_rolling_window`:
  `[T - h - 35 months, T - h)` for training and `[T, T + 1 month)` for test.
- CSV targets are limited to 2021-2024 February, June, and October, with 12
  unique target months per scope.
- Stage 2 plan counts in the note match current summary artifacts.
- `git diff --check` passes.

## Out Of Scope

- No model reruns.
- No manuscript editing outside `final_artifacts_in_paper_updated/`.
- No new temporal split logic.
- No threshold tuning, hyperparameter tuning changes, or sensitivity analysis.
