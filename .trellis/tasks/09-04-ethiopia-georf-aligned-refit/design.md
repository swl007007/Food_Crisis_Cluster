# Design

## Chosen boundary

Keep production Stage 1/3 unchanged. Add Ethiopia-specific aligned-data Stage 1
and Stage 3 entrypoints under `EthiopiaForecastingExperiment/`; reuse the
existing Stage 2 scripts unchanged. The existing Ethiopia runner remains the
single orchestrator and writes one immutable run directory.

Rejected alternatives:

- Default-off prealigned hooks in production Stage 1/3: rejected in favor of an
  independently evolving Ethiopia pipeline.
- Feeding aligned data through production preprocessing: rejected because it
  would recreate features and apply the horizon twice.

## Data flow

```text
current fs0-fs3 aligned panels (85 predictors)
  + working-panel raw fews_ipc publication history
  -> keep release months with >=90% national outcome coverage
  -> for each forecast origin F, attach latest three per-admin phases from
     qualifying release months strictly before F
  -> run-local fs0-fs3 snapshots (88 predictors)
  -> ETH Stage 1 (36 plans)
  -> unchanged shared Stage 2 consensus/stabilization
  -> ETH Stage 3 (48 folds)
  -> predictions, thresholds, monthly metrics, 4x3 plot, manifests
```

The seven seasonal predictors already present in each aligned panel remain
joined at `FEWSNET_admin_code + forecast_origin_month`; their recorded season
must end strictly before that month.

## Model-input contract

- Key: `scope + FEWSNET_admin_code + target_month`; origin is exactly `M-H`.
- Predictors: the ordered 85-column aligned list followed by
  `fews_ipc_release_lag1/2/3` as ordinal values 1-4.
- Target: binary `fews_ipc_crisis`.
- No production preprocessing, dummy creation, automatic dynamic detection,
  additional lagging, normalization, or feature selection.
- Within each temporal fold, fit median imputation on training rows only; map a
  training-all-null column to zero; apply the frozen transform to validation and
  test; then run SMOTE.

## Execution and outputs

The runner selects one scope-specific snapshot for every Stage 1/3 command.
Stage 2 consumes only copied Stage 1 correspondence artifacts. Existing run IDs
remain immutable, and all new artifacts stay under
`outputs/local_partition_experiment/<run_id>/`.

Suppress all 2021-06 model and FEWS NET metrics as
`suppressed_low_target_coverage` because only one target is observed. Keep its
four fold records for audit, but omit the point from plots. Other evaluation,
seed, threshold, FEWS NET, and plotting behavior remains matched to the prior
experiment.

## Failure rules

Abort before fitting on cohort/key/hash drift, feature order or count other than
88, a non-exact origin, seasonal lookup mismatch, fewer than three qualifying
histories in the modeling window, imputation fitted outside training rows,
non-finite model input after transformation, partition/admin mismatch, or an
existing run directory.

