# Secondary And Lag Ablation Completion Design

## Goal

Complete the updated `ablation_feature_exclude.xlsx` by adding the two rows
that existed in the old paper artifact:

- `Secondary Exclude`
- `Lag Exclude`

The output remains:

`final_artifacts_in_paper_updated/ablation_feature_exclude.xlsx`

## Scope

This is a completion of the existing feature-exclude ablation artifact, not a
new `secondary only` or `lag only` experiment. The workbook should return to
the old paper-facing layout:

- 6 feature-group exclude blocks
- `Secondary Exclude`
- `Lag Exclude`
- `Main`
- `FEWSNET (baseline)`

That is 30 data rows plus two header rows, or `32 x 12`.

## Experimental Conditions

Both new blocks use the same Stage 3-only setup as the existing updated
ablation runs:

- base model: GeoRF / GF
- fixed current GeoRF partition maps
- scopes: fs1, fs2, fs3
- lags: 4, 8, 12 months
- evaluation window: 2021-01 through 2024-12
- training window: 36 months
- month indicator: enabled
- contiguity refinement: `cont3`
- source lineage: current `FEWSNET_forecast_unadjusted_bm.csv`

## Secondary Exclude

`Secondary Exclude` should be generated from the current unadjusted BM source,
not copied from old `PastExogeneous.csv` or the old workbook.

The fresh `secondary_exclude.csv` should drop the same secondary-like column set
implied by the current `PastExogeneous.csv` / `Exogeneous_only.csv` comparison
against `FEWSNET_forecast_unadjusted_bm.csv`: conflict, macro/economic,
food-price, market/nightlight/population, and FEWSNET projection columns that
are not present in those secondary-exclude historical inputs.

## Lag Exclude

`Lag Exclude` should use the full current unadjusted BM source but run Stage 3
with:

`ENABLE_LAG_FEATURES=false`

This disables preprocessing-created target lag features:

- `fews_ipc_crisis_lag_4`
- `fews_ipc_crisis_lag_8`
- `fews_ipc_crisis_lag_12`
- `fews_ipc_lag_4`
- `fews_ipc_lag_8`
- `fews_ipc_lag_12`

It does not disable the forecasting scope itself, and it does not disable
`prepare_features()` dynamic time-variant feature lags. This preserves the old
`Lag Exclude` meaning without changing the Stage 3 forecast horizon mechanism.

## Output Layout

New Stage 3 outputs should stay under the existing isolated run root:

`main_ablation_exclude_updated_stage3_fixed_partitions/`

Suggested subfolders:

- `secondary_exclude/result_partition_k40_compare_GF_fs1..fs3`
- `lag_exclude/result_partition_k40_compare_GF_fs1..fs3`

The workbook builder should append these two blocks between
`Geographic Exclude` and `Main`.

## Validation

Before promoting the workbook:

- verify `secondary_exclude` and `lag_exclude` each have fs1, fs2, fs3 outputs
- verify every new `metrics_monthly.csv` is 24 rows with pooled and partitioned rows
- verify each new run manifest records the intended data source and, for
  `Lag Exclude`, `ENABLE_LAG_FEATURES=false`
- verify final workbook is `32 x 12`
- verify no values for the new blocks are copied from the old artifact
