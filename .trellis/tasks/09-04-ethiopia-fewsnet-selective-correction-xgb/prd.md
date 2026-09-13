# ETH FEWS NET selective correction XGBoost

## Goal

Test whether a conservative binary XGBoost correction gate can make a small,
validated improvement over the forecast-origin-available FEWS NET expert
forecast while leaving expert decisions unchanged by default.

## Confirmed evidence

- The completed additive-residual v2 is a valid result for its original
  contract, but underperforms the expert in every scope: mean F1 is
  `0.7310/0.6955/0.6066/0.5765` versus expert
  `0.8091/0.8091/0.7699/0.6689` for fs0/fs1/fs2/fs3.
- Its candidate set has no expert-only (zero-correction) option; even
  `layer_count=1` means expert plus one residual learner.
- The old objective predicts a continuous residual and then tunes a free score
  threshold. It does not directly minimize damage to correct expert calls.
- The v2 run, its six artifacts, its PRD hash, the frozen 88 predictors, prior
  GeoRF v5 and binary-XGBoost outputs, and production Stage 1-3 are immutable.

## Requirements

- Reuse the existing ETH cohort, expert-to-scope timing map, frozen predictors,
  temporal folds, common-support rules, metrics, comparisons, and seed 5.
- Replace additive residual regression with one `XGBClassifier` that estimates
  whether the binary expert call is wrong. A triggered correction flips the
  expert class; otherwise the expert call passes through unchanged.
- Include expert-only as the zero-correction candidate and prefer it on ties.
- Choose the correction threshold and whether correction is enabled using
  validation data only; never use the test month for selection.
- For each scope and outer fold, enable a correction candidate only when its
  validation F1 on identical support is strictly higher than expert-only.
  Select expert-only on an exact tie or any non-improvement.
- For each flip direction separately, require validation correction precision
  of at least 75% and at least 20 flips spanning at least two target months.
  Disable only a direction that fails; if neither passes, select expert-only.
- Train the `expert_wrong` gate at the observed error frequency without class
  weights. Do not reuse the old true-crisis class weights for this new target.
- Allow the single gate to flip expert calls in either binary direction. Do not
  create separate models for non-crisis-to-crisis and crisis-to-non-crisis.
- Treat the untouched test period as evaluation only. If a validation-enabled
  correction underperforms expert-only on test, report the negative result; do
  not use test outcomes to retune, disable, or retrospectively replace it.
- Reuse the prior eight-candidate XGBoost grid (`max_depth` in `{3, 6}`,
  `min_child_weight` in `{1, 5}`, and `n_estimators` in `{200, 400}`) so the
  correction target and selection contract are the controlled method changes.
- Define high confidence by validation correction precision, not by treating the
  raw XGBoost score as a calibrated probability. Do not add calibration or a
  raw-score floor of `0.5`.
- Record corrections of expert errors, damage to correct expert calls, net
  corrections, correction precision, and the existing classification metrics.
- Write once to the fixed new run directory
  `eth_fewsnet_selective_correction_xgb_20260904_seed5_v1`; refuse a collision
  and do not overwrite v1 or v2 predecessor runs.
- Write only `predictions.csv`, `tuning_results.csv`, `metrics_monthly.csv`,
  `metrics_summary.csv`, and `run_metadata.json`.

## Acceptance criteria

- [ ] Every emitted prediction equals the expert call unless the fitted gate
  exceeds its validation-selected correction threshold.
- [ ] Expert-only is an actual candidate and remains selected when the agreed
  activation rule is not met.
- [ ] Selection and fitting retain the existing month-grouped, horizon-safe
  no-leak contract; test outcomes do not affect either decision.
- [ ] The comparison uses identical support for selective correction, FEWS NET,
  GeoRF v5, and plain binary XGBoost across fs0-fs3.
- [ ] Outputs make beneficial and harmful expert flips directly auditable and
  preserve all protected-input hashes.
- [ ] An independently recomputed equal-month summary agrees with exported
  metrics; a test-period loss to expert-only remains visible and unchanged.

## Out of scope

- Multiple residual layers, separate direction models, new predictors,
  production-pipeline changes, or reinterpretation of the completed v2 run.
