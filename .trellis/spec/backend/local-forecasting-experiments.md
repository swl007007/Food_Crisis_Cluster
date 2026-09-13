# Local Forecasting Experiments

## 1. Scope / Trigger

Use this contract for isolated country-level Stage 1-3 runs that add alternate
data, lag, seed, threshold, or output-path CLI options without changing
production defaults.

## 2. Signatures

- Runner: `run_local_partition_experiment.py --aligned-dir PATH --working-panel PATH --season-lookup PATH --fewsnet PATH --output-root PATH --run-id ID`
- Aligned Stage 1: `stage1_aligned_georf.py --data SCOPE_SNAPSHOT --forecasting_scope N --desired_terms YYYY-MM --random-seed INT`
- Aligned Stage 3: `run_stage3_aligned.py --data SCOPE_SNAPSHOT --partition-map PATH --forecasting-scope N --start-month YYYY-MM --end-month YYYY-MM --enable-symmetric-validation-threshold`
- FEWS residual comparison: `run_fewsnet_residual_xgb.py --v5-run PATH --binary-run PATH --fewsnet PATH --output-root PATH --run-id ID`

## 3. Contracts

- Freeze the cohort by canonical admin-month key and source hashes before fitting.
- A scope changes only its horizon (fs0/fs1/fs2/fs3 = 1/4/8/12 months). Aligned
  snapshots contain the exact forecast-origin values and must not be lagged again.
- The aligned Ethiopia model contract is the ordered 85 canonical predictors plus
  `fews_ipc_release_lag1/2/3`. Do not call production feature engineering, dummy
  generation, automatic dynamic detection, or feature selection.
- Fit median imputation from each training fold only; map training-all-null columns
  to zero, transform validation/test with the frozen medians, then run SMOTE.
- Stage 1 uses only months with observed targets. Never fill unlabeled months to
  reach a planned fit count.
- Repository-local output paths must remain under
  `EthiopiaForecastingExperiment/outputs/local_partition_experiment/`; scratch
  paths outside the repository are allowed for tests.
- Existing run IDs are immutable. Create a new run directory for every rerun.
- For the ETH multiclass XGBoost comparison, pass predictor `NaN` values to
  XGBoost natively and never synthesize a missing IPC class. A validation-fit
  subset may legitimately omit a phase even when the complete 36-month refit
  contains all four; record the absent phase with zero support and null weight.
- For the ETH FEWS residual comparison, attach expert anchors by explicit
  calendar key: fs0/fs1 use `T-4 fews_proj_near`, fs2 uses `T-8 fews_proj_med`,
  and fs3 uses `T-12 fews_proj_med`. Never derive these anchors by row shifting.
- Define Layer 2 targets only as `y - expert - residual_1_oof`, where Layer 1
  OOF predictions hold out whole target months and use labels strictly before
  pseudo-origin `m-H`. The final additive value is a score, not a probability.
- Exit audit-only suppressed folds before model fitting or hyperparameter search.

## 4. Validation & Error Matrix

| Condition | Required result |
|---|---|
| Null/duplicate cohort key or hash drift | Abort |
| Unlabeled Stage 1 month | Exclude; do not impute target |
| Repository path outside the approved experiment root | `ValueError` before directory creation |
| Existing run directory | `FileExistsError` |
| Scope, horizon, origin, cohort, key, or 88-column order mismatch | Abort before fitting |
| Seasonal lookup value mismatch or season end not before origin | Abort before fitting |
| Infinite model input after fold-local imputation | Abort before SMOTE |
| FEWS NET projection coverage below 90% | Keep row, null metrics, `suppressed_low_coverage` |
| Target coverage below 90% | Keep all model audit rows, null metrics, `suppressed_low_target_coverage`; omit plot point |
| Residual expert/common support below 90% | Keep four audit rows with one suppression flag; do not fit or fall back to plain XGBoost |
| Multiclass validation-fit subset omits a phase | Fit observed classes only; retain all four labels when scoring macro-F1; record zero support; do not synthesize rows |
| Partition/admin-code mismatch | Abort |

## 5. Good / Base / Bad Cases

- Good: 2018-2020 February/June/October labels yield exactly 36 plans for four scopes.
- Base: a temporary directory outside the repository is accepted by unit tests;
  run-local snapshots keep all 88 predictors even when some values are missing.
- Bad: feeding an already aligned snapshot through `prepare_features`, applying a
  second horizon lag, filling target labels, or writing under `result_GeoRF*`.

## 6. Tests Required

- Assert the labeled-month plan count and lag schedule.
- Assert exact `target_month - forecast_origin_month` horizons and ordered 88-column snapshots.
- Assert release histories use the latest three qualifying publications strictly before origin.
- Assert train-only imputation keeps all columns, produces finite values, and precedes SMOTE.
- Assert pooled and partitioned thresholds use validation data only.
- Assert `suppressed_low_target_coverage` creates no plotted point for any model.
- Assert protected repository paths fail before any directory is created.
- Independently recompute exported metrics and verify source/code hashes.
- Assert suppressed folds create no fit and multiclass missing-class weights are
  auditable without synthetic observations.
- Assert all four residual expert calendar mappings, month-grouped horizon-safe
  OOF targets, and score thresholds outside `[0, 1]` when validation selects them.

## 7. Wrong vs Correct

```python
# Wrong: re-run production feature generation on an aligned snapshot.
X = prepare_features(aligned_snapshot, forecasting_scope=scope)

# Correct: consume the frozen ordered predictors, then fit fold-local medians.
X = aligned_snapshot.loc[:, MODEL_PREDICTORS].to_numpy(dtype=float)
medians = fit_fold_medians(X[train_rows])

# Correct for the separate multiclass XGBoost comparison: native NaN handling,
# with weights defined only for classes observed in this fit subset.
model.fit(X_fit, y_fit, sample_weight=observed_class_weights)

# Wrong: train Layer 2 on a Layer-1 in-sample fit.
layer2_target = y - expert - layer1.predict(X_fit)

# Correct: use target-month temporal OOF predictions only.
layer2_target = y[oof_rows] - expert[oof_rows] - layer1_oof[oof_rows]
```
