# Ethiopia local partition ensemble experiment

## Goal

Run an isolated Ethiopia-only GeoRF Stage 1-3 experiment for fs0/fs1/fs2/fs3,
then deliver matched monthly metrics and a 4x3 performance figure with
the corrected Ethiopia FEWS NET baseline where a native comparison exists.

## Background

- The authoritative panel cohort is exact `ISO3 == "ETH"`: 187,200 rows, 1,040
  `FEWSNET_admin_code` values, 180 months. Reference `area_id` is forbidden.
- `FEWSNET.csv` exact `country == "Ethiopia"` filtering yields 55,120 rows and
  the identical 1,040-code set.
- Current production fs0 launchers change Stage 2/3 behavior; they cannot satisfy
  the requirement that scope lag be the only experimental difference.
- Current feature preparation retains unlagged time-varying columns; strict
  forecast-time-safe mode must remove them.
- The unadjusted Ethiopia panel has 180 calendar months but non-null target labels
  only in February, June, and October. Stage 1 must not invent or fill targets for
  unlabeled months.

## Requirements

1. Freeze one Ethiopia cohort and source hash before Stage 1; reuse it everywhere.
2. Learn partitions in February, June, and October of 2018-2020 and evaluate the
   same months in 2021-2024.
3. Run matched fs0/fs1/fs2/fs3 configurations with lags 1/4/8/12 months. Static
   features are shared; time-varying features must come only from `t-lag`.
4. Build one shared Stage 2 consensus from all four scopes and reuse its
   general/m2/m6/m10 mappings for every Stage 3 run.
5. Freeze GeoRF/SMOTE seed 5 and spectral-clustering seed 42. Do not run a
   multi-seed experiment.
6. Select pooled and partitioned thresholds symmetrically within each fold using
   the latest six eligible training months and class-1 F1; test labels are
   forbidden. Keep fixed-0.5 results as diagnostics only.
7. Corrected Ethiopia FEWS NET baselines are:
   - fs1: `fews_proj_near(T-4)` versus `fews_ipc(T)`;
   - fs2: `fews_proj_med(T-8)` versus `fews_ipc(T)`.
   Use explicit calendar-month joins, not row-position shifts. fs0/fs3 are
   unavailable. Missing values stay missing.
8. For fs1/fs2 comparisons, compute all plotted series on identical valid
   admin-month keys. Suppress a FEWS NET point below 90% projection coverage.
9. Keep source data, existing result roots, archives, and paper artifacts
   unchanged. Minimal shared hooks are allowed only when default-off and covered
   by production-default regression checks.
10. Store all generated files under an experiment-specific run directory in
    `EthiopiaForecastingExperiment/outputs/local_partition_experiment/`.

## Acceptance Criteria

- [x] Cohort manifest records both source hashes, exact key equality, row/admin/
      month counts, and rejects null or duplicate admin-month keys.
- [x] Stage 1 produces 36 fitted Ethiopia partition-learning plans: 4 scopes x
      3 years x 3 labeled months, with no target filling.
- [x] Stage 2 produces one shared manifest plus general/m2/m6/m10 mappings.
- [x] Stage 3 completes all four scopes on the same cohort, folds, partitions,
      seeds, feature policy, and threshold-selection policy.
- [x] `metrics_monthly.csv`, prediction files, thresholds, and a run manifest
      have unique documented keys; FEWS NET coverage/status is recorded in the
      monthly metrics rows.
- [x] The 4x3 figure shows thresholded pooled/partitioned everywhere, FEWS NET
      only in fs1/fs2, and explicit unavailable labels in fs0/fs3.
- [x] Fixed-0.5 diagnostics, common-row recomputation, hashes, and focused tests
      independently reproduce exported values.
- [x] Production defaults pass regression checks and no protected artifact is
      overwritten.

## Out of Scope

- CDS acquisition or CDS-enhanced forecasting.
- Production horizon/config changes, paper-artifact promotion, multi-seed runs,
  or committing large generated experiment outputs.

## Open Questions

None.
