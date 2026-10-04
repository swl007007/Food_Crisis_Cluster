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

## 6a. Multiclass GeoRF packages (FEWSNETFourClassBaseline pattern)

- Score every stage with fixed-K macro F1 on aggregated counts; a zero F1
  denominator scores 0 and the class is still averaged. Never reuse the release
  `get_prf` mean fill or any class-1 helper.
- Each RF owns its imputer: fit on its real fitting rows, save forest + imputer +
  fit record as ONE checkpoint bundle, so `load(parent); save(child)` inheritance and
  pooled fallback can never pair a forest with another estimator's fill values.
  Append class-recovery rows only after imputation.
- Build features once per (area, target, horizon) key from the complete monthly
  scaffold at calendar offsets from O; feed the aligned snapshot directly. Never pass
  it through `prepare_features`, `comp_impute`, row shifts or FEATURE_DROP.
- Keep bulky Stage 1 working trees (checkpoints) outside Dropbox-synced paths; Dropbox
  locks fresh directories (WinError 32). Copy retained evidence back afterwards.
- Keep inherited gate definitions exactly (e.g. Stage 3 unmapped coverage is the share
  of all labelled panel rows); disclose additional shares instead of gating on them.
- A missing Stage 1 candidate or non-finite score stops Stage 2; only a complete ledger
  of all-zero weights may take the null-consensus route.

## 6b. GeoXGB interruption availability (2026-10-02 task)

### Scope / Trigger

Applies only to the approved `exogenous-transition-forecast-design` scenario path in
`FEWSNETGeoXGBExperiment`. Its PRD/design/evaluation-contract override the older
Ethiopia and fixed-macro conventions above. Synthetic implementation checks do not
certify source availability; `data-readiness.md` must pass before real fitting.

### Signatures

- `continue_booster(parent, X, y, config, sample_weight=None)`; `XGBmodel.train(..., sample_weight=None)`.
- `covariate_features(scaffold, schema, areas, targets, origins, alignment=None)`.
- `ReleaseLedger(frame, real=False)` and `Availability(..., alignment=None, truth=None)`.
- `ScenarioPanel(availability, k, strategy, prediction_areas=None, gate_k=None)` uses the existing Stage3 engine; actual internal replay may supply country-specific `gate_k`.
- `plan.scenario_stage1_schedule()` freezes 648 candidate identities; `run_stage1.py --split-mode scen` consumes prepared scenario inputs.
- `run_experiment.py --run-dir RUN scen-develop|scen-select|scen-freeze|scen-historical|scen-actual|scen-report|scen-evaluate` uses the separate interruption schedule.
- `scen-actual` requires `--actual-availability CSV --actual-scaffold MANIFEST`; reporting accepts `--expert-table CSV`; `scen-evaluate` requires `--truth-release DIR`.
- `keyed_expert(keys, experts, real=True)` returns matched expert classes or explicit unavailability reasons; experts remain comparators only.

### Contracts

Ledger columns are `cycle_id, product, country, reference_month, release_date,
evidence, source`. Reference month is not release date. Monthly covariates use the
declared exact lag; missing or unreleased values remain NaN. Annual covariates use
the latest eligible reference year. `alignment=None` preserves the legacy feature
builder only; real scenario views require explicit alignment.

Original area/target keys determine F/S/C roles and support. B repeats fitting keys
at weights 1/3 for each of k=0/1/2; S/C are single designated-scenario observations.
Weights reach root and local fits. Local continuation preserves the shared root.
Gate dates are the latest six lawful dates U<O, independent of the 59-month fitting
window. Scenario Stage3 uses exact crisis-F1 gain >0.01; legacy Panel retains its
frozen metric. Prediction cohorts do not require target labels; truth is evaluator-only.

Stage2 weights use matched E3 crisis F1 against each candidate's own root. Undefined
scores are ineligible with a reason; zero weight, no prior candidates and no scorable
evidence remain distinct. A/B selection consumes exactly the 72 scheduled development
folds. Screen normal matched persistence gain at >=-0.02, then rank the equal-weight
mean of k1/k2 F1; ties choose A, no qualifier stops that horizon's final release.

Actual country-specific masks operate within the shared global/local population;
they do not create separate country models. Extension covariates need a hashed source
manifest and agreement with the pinned overlap; final truth is never a predictor.
Expert rows carry `area, issue_month, product, horizon, validity_start, validity_end,
class_code, release_date, evidence, source`. Match exact origin, declared horizon,
target validity and release by cutoff. Do not reinterpret an old projection as a
different horizon. Missing experts leave coverage rows, not zero predictions.

Truth release uses `release.json` with `approved, approved_by, crosswalk, truth_file,
truth_sha256, frozen_actual`; the last field binds the frozen actual-prediction
record. Truth rows are unique `area,target_month,class_code` keys on the 0..3 axis.
Study2 uses genuine exact-origin truth, including lawful historical observations or
separately released evaluator labels, never the latest earlier label. Reports retain
same-input pooled comparisons, all-country coverage and the keyed evaluator joins.

### Validation & Error Matrix

| Condition | Required result |
|---|---|
| Synthetic ledger/alignment in a real run, missing release coverage or missing real alignment | Refuse |
| Fewer documented due cycles than requested k | Unsupported scenario; never silently reduce k |
| Hidden IPC or release after cutoff | Exclude from fitting labels and every derived input |
| Empty/unsupported local fitting or undefined gate F1 | Preserve evaluation keys; global fallback |
| Gate gain exactly 0.01 | Local disabled |
| Changed strategy, masks, fitting keys, features, labels or weights | Cannot reuse an incompatible global model |
| Duplicate/missing development fold identity | Refuse selection even if the total remains 72 |
| Expert validity misses target or release follows origin | Comparator unavailable with a reason |
| Final truth release does not bind frozen actual predictions | Refuse evaluation |
| Country/target without genuine truth | Preserve forecast/coverage; accuracy remains NA |
| Undefined bootstrap draw | Count it without redrawing; numerical CI requires all 2,000 draws defined |

### Good / Base / Bad Cases

Good: three B variants conserve one original row's weight and support. Base: A uses
unit weights. Bad: counting variants as three independent observations or filling
an unlabeled target with persistence to make it evaluable.

### Tests Required

Run `tests/test_baseline.py` with `ReleaseAwareViews`, `WeightedContinuation`,
`ScenarioStage3`, `Stage3Engine`, `ScenarioStage1` and `Stage1Variants` on the pinned
Windows Python. Check hidden-input invariance, grouped roles, original-key support,
root-prefix invariance, cache isolation and forecast-only rows. Full regression is
required before the final task check; retain explicit unresolved data blockers.
Also run `ScenarioStage2`, `ScenarioDevelopment`, `ScenarioReporting`,
`ScenarioFinalPath` and `ScenarioDriverSmoke`. Exercise actual driver I/O on synthetic
inputs, both qualifying/no-qualifier paths, country-specific replay, same-horizon
expert admission, frozen-prediction truth release and missing-label coverage.

### Wrong vs Correct

Wrong: choose gate dates from the 59-month fitting pool or use one memo key for all
scenarios. Correct: select lawful gate dates independently and bind caches to the
actual fitting inputs, labels, weights and scenario identity.
Wrong: treat an existing fold file as sufficient for reuse or accept its current
predictions without checking the frozen record. Correct: verify recorded input and
upstream identities plus output inventories before consuming saved results.
Country intensity maps contain integers; exclusion maps contain sets of cycles.
Canonicalise their cache keys separately. Applying a set-valued mask serializer to
`{"AAA": 1, "BBB": 2}` fails on the actual-country path even when simulated k tests pass.

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
