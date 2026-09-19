# ETH FEWS NET residual XGBoost

## Goal

Test whether an Ethiopia binary multi-layer residual XGBoost can improve on a
forecast-origin-available FEWS NET expert prediction using the already frozen
88-predictor aligned inputs.

## Confirmed starting point

- The experiment is binary crisis classification (`fews_ipc >= 3`).
- Layer 1 is an expert FEWS NET prediction available at the forecast origin.
- Later layer(s) use XGBoost and the frozen aligned predictors to model the
  remaining error.
- Use additive continuous residuals: collapse the expert phase to binary
  `e`, train the first residual learner on `y-e`, and form the final score as
  `e + sum(predicted residuals)`. Treat this as a score, not a probability;
  choose its classification threshold using training-period validation only.
- Compare one versus two residual XGBoost layers and select residual depth using
  training-period validation only. For the two-residual-layer candidate, create
  the second-layer target from temporally out-of-fold first-layer predictions;
  never use first-layer in-sample fitted predictions as residual targets.
- For fs0 nowcasting, use the latest expert prediction available one period
  earlier. For fs3, use the eight-month expert prediction available at the
  twelve-month forecast origin.
- FEWS NET is observed in February/June/October release rows. For fs0 targets,
  the latest release available by `T-1` is the `T-4` near-term projection and it
  is target-aligned to `T`. There is no literal `T-1` FEWS row.
- For fs3, the medium-term projection read at `origin=T-12` predicts
  `origin+8=T-4`, not target `T`; it is therefore an available expert anchor
  that requires a learned four-month bridge rather than a target-aligned expert
  forecast.
- Run all four scopes with the frozen expert mapping: fs0 and fs1 use the
  `T-4` near-term projection, fs2 uses the `T-8` medium-term projection, and
  fs3 uses the medium-term projection issued at `T-12`.
- Build first-residual predictions for the second residual layer by
  target-month forward cross-fitting. Keep every admin row from a target month
  together, use the same 36-month rolling window, and require training labels
  to precede the pseudo-origin `m-H`. Exclude early months without enough
  history instead of substituting in-sample predictions; exclude `2021-06`.
- Use `XGBRegressor` residual learners with the existing eight-candidate grid:
  `max_depth` in `{3, 6}`, `min_child_weight` in `{1, 5}`, and
  `n_estimators` in `{200, 400}`; retain the existing fixed learning rate,
  row subsampling, feature subsampling, and seed. Both residual layers share
  one candidate tuple. Select the residual-layer count and tuple together on
  validation data, breaking exact ties toward one layer, shallower trees, and
  fewer trees.
- Weight every residual and OOF fit by the existing square-root inverse
  frequency weights computed from the true binary class. Do not add a second
  multiplier for FEWS NET errors.
- Select the final score threshold on outer-fold validation data only. Search
  the unique validation scores rounded to two decimals plus `0.50`, without
  clipping candidates to `[0, 1]`; maximize crisis-class F1 and break ties
  toward the higher threshold. Fall back to `0.50` when validation has no
  positive cases. Never use the test month for threshold selection.
- Do not impute a missing expert anchor or fall back to plain XGBoost. Drop
  such rows from residual training and compare test predictions only on common
  FEWS/residual/GeoRF/truth support. Suppress a target month's metrics when
  expert coverage is below 90%, while retaining the coverage fraction and a
  single suppression flag in the audit output.
- Compare four methods on identical support: the FEWS expert anchor, the
  immutable existing plain binary XGBoost, the immutable GeoRF v5, and the
  validation-selected one- or two-layer residual model. Record all validation
  candidates and the selected layer count per fold, but do not create a test
  leaderboard for unselected residual depths. Exclude multiclass XGBoost
  because it has a different target.
- Report monthly crisis precision, recall, F1, balanced accuracy, confusion
  counts, and common-support coverage. Summarize each scope by equal-weighted
  means across eligible target months, with F1 primary and balanced accuracy
  as the imbalance check. Do not report probability metrics because neither
  the binary FEWS anchor nor the additive residual score is a probability.
- Write only `predictions.csv`, `tuning_results.csv`, `metrics_monthly.csv`,
  `metrics_summary.csv`, `comparison_by_scope.png`, and `run_metadata.json` in
  a new run directory. Keep selected parameters, confusion counts, coverage,
  and the suppression flag inside those files rather than splitting more
  artifacts. Print the compact scope summary to the TUI.
- Give every residual learner the frozen 88 predictors plus the binary expert
  anchor available at the forecast origin. Do not add preceding residual
  predictions as features; use their temporally OOF values only to define the
  next residual target.
- Use `reg:squarederror` and the frozen seed 5. Pass predictor `NaN` values to
  XGBoost's native missing-value branches without imputation or extra missing
  indicators; reject infinite values. Expert-anchor missingness follows the
  strict row exclusion and coverage rules above.
- Prior GeoRF v5, multiclass XGBoost, binary XGBoost, production Stage 1-3, and
  their outputs remain immutable.

## Candidate minimal boundary

- One isolated ETH experiment and a new output directory.
- Explicit calendar joins only; expert availability must be defined from the
  forecast origin and never by row shifting.
- Reuse the frozen 88-predictor snapshots and established no-leak folds.
- Compare only on common admin-month support.

## Open decisions

None. The grill is converged pending the execution gate.

## Out of scope unless explicitly added

- Changes to production Stage 1-3, existing run directories, FEWS NET source,
  or the 88-predictor contract.
