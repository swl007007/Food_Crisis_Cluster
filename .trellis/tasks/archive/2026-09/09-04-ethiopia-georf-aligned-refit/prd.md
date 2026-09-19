# Ethiopia GeoRF refit from aligned features

## Goal

Rerun the isolated Ethiopia GeoRF Stage 1-3 experiment using the revised ETH
data contract, without the production feature-engineering chain or a second
horizon shift.

## Confirmed Data Facts

- For every fs0-fs3 modeling row in the 2018-2024 February/June/October window,
  three observed FEWS IPC releases exist strictly before forecast origin.
- The latest/second/third histories are respectively 3-8, 7-12, and 11-16
  calendar months old across the four scopes; they remain release-sequence
  histories rather than fixed calendar lags.

## Requirements

1. Keep the frozen Ethiopia cohort: 1,040 `FEWSNET_admin_code` values on the
   complete 2010-01 through 2024-12 monthly panel. Abort on source, key, or
   cohort drift.
2. Keep fs0/fs1/fs2/fs3 horizons at 1/4/8/12 months. For target month `M`, all
   time-varying predictors must come from the exact forecast origin `F=M-H`.
3. Use the revised ETH base variables as stored. Do not run the production
   feature-engineering helpers, year/month dummy generation, automatic dynamic
   detection, or another horizon lag inside Stage 1 or Stage 3. Treat all 85
   predictors in the current aligned contract as the frozen baseline feature
   set, including the previously approved moving-average, WB price, conflict,
   and seven seasonal predictors.
4. Before model fitting, join the seven approved previous-growing-season values
   by `FEWSNET_admin_code + F`. The selected season must end strictly before
   `F`; season identity and coverage audit fields do not enter the model.
5. The only newly derived model predictors are the latest three observed FEWS
   IPC releases of raw `fews_ipc` phase, ordered backward from forecast origin
   and kept separate as publication-sequence histories. The prediction target
   remains binary `fews_ipc_crisis`. Every selected
   release month must be strictly earlier than forecast origin and have observed
   outcome coverage for at least 90% of the frozen national cohort. Do not
   interpret them as calendar-month lags and do not fill missing outcomes. Keep
   each history as one ordinal numeric predictor with values 1-4; do not one-hot
   encode the phases.
6. Rerun the complete Ethiopia Stage 1 partition learning, shared Stage 2
   ensemble/stabilization, and Stage 3 validation with the previously frozen
   seeds and evaluation months. Store the complete run under one new dedicated
   `outputs/local_partition_experiment/<run_id>/` subdirectory, including input
   snapshots, Stage 1-3 outputs, metrics, plots, and manifest. Do not alter
   production defaults or existing runs.
7. Preserve the existing FEWS NET fs1/fs2 comparison contract and fs0/fs3
   unavailability unless a later approved decision changes it.
8. Execute the four 2021-06 Stage 3 folds for run completeness, but suppress
   pooled, partitioned, and FEWS NET metrics because target support is only
   1/1,040. Retain explicit `suppressed_low_target_coverage` rows and omit the
   month from plotted lines.
9. Create Ethiopia-specific Stage 1 and Stage 3 implementations under
   `EthiopiaForecastingExperiment/` that directly consume the aligned contract.
   Do not add a prealigned mode to the shared production entrypoints. These
   experiment-specific implementations may intentionally evolve separately.
10. Reuse the existing Stage 2 consensus and stabilization scripts unchanged,
    but execute them only against the dedicated Ethiopia run workspace and its
    Stage 1 artifacts.
11. Keep the existing 85-predictor aligned artifacts unchanged. Materialize four
    run-local input snapshots containing those 85 predictors plus the three
    predictors `fews_ipc_release_lag1`, `fews_ipc_release_lag2`, and
    `fews_ipc_release_lag3`, and record their hashes in the run manifest.
12. Preserve all 88 predictor columns through fitting. Within each temporal
    training fold, fit median imputation from training rows only and apply that
    frozen transform to validation and test rows. A column that is entirely null
    in the training fold receives constant zero. Run SMOTE only after this
    transformation; do not derive missingness indicators.
13. Freeze the comparison design: Stage 1 uses February/June/October 2018-2020;
    Stage 3 uses the same months in 2021-2024; GeoRF and SMOTE use seed 5,
    spectral clustering uses seed 42, the rolling training window is 36 months,
    and pooled/partitioned thresholds are selected symmetrically from the latest
    six eligible training months.
14. Deliver per-fold predictions, `metrics_monthly.csv`,
    `thresholds_by_fold.csv`, the existing 4x3 monthly performance plot, input
    and feature manifests, and a run manifest. Plot pooled and partitioned for
    every scope and FEWS NET only for fs1/fs2. Do not add an old-versus-new run
    comparison artifact.

## Acceptance Criteria

- [x] Stage 1 and Stage 3 consume the same explicit per-scope aligned feature
      contract; no predictor is shifted twice and no contemporaneous
      time-varying predictor survives.
- [x] Every seasonal value matches the existing lookup at forecast origin and
      its recorded season ends before that origin.
- [x] Each autoregressive value is reproducible from observed outcomes that
      were available no later than forecast origin; missing histories remain
      null and their coverage is reported before fitting.
- [x] Fold-local imputation uses no validation or test statistics, keeps all 88
      predictors, produces finite model inputs, and precedes SMOTE in both
      Ethiopia-specific Stage 1 and Stage 3.
- [x] The complete 36-plan Stage 1, shared Stage 2, and 48-fold Stage 3 workflow
      either completes or fails before fitting with a precise contract error.
- [x] A new run manifest records input/code hashes, feature names, seeds,
      partitions, thresholds, row accounting, and the no-leak checks; existing
      run directories and production artifacts remain unchanged.
- [x] Focused regressions demonstrate unchanged production defaults and
      independently verify aligned dates, feature columns, and exported metrics.
- [x] The dedicated run directory contains the confirmed predictions, metrics,
      thresholds, 4x3 plot, and manifests; 2021-06 is present only as suppressed
      audit rows and creates no plotted point.

## Out of Scope

- Changing the global GeoRF/GeoDT feature defaults, horizon schedule, growing-
  season calendar, labels, seeds, threshold policy, or production artifacts.
- Normalization, missingness indicators, feature selection, hyperparameter
  tuning, multi-seed analysis, or a generic alternate-pipeline framework.
