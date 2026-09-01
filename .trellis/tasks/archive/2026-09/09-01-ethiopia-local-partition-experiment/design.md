# Design

## Boundaries

- Reuse the existing GeoRF Stage 1, Stage 2 Python steps, and Stage 3 comparison
  code. Do not fork/copy the production pipelines.
- Add only default-off shared hooks required for an alternate data path and
  strict lag-only feature selection. Existing CLI defaults and result naming
  remain unchanged.
- Use the fewest experiment-local files needed to run the sequence, evaluate the
  FEWS NET baseline, aggregate monthly metrics, and draw the figure. No component
  split is mandatory.

## Data Flow

```text
authoritative panel
  -> exact ETH filter + cohort manifest
  -> Ethiopia panel CSV
  -> Stage 1: 2018-2020 x fs0-fs3 x February/June/October
  -> Stage 2: joint weighted consensus + K40 stabilization
  -> shared general/m2/m6/m10 mappings
  -> Stage 3: four matched 2021-2024 validations
  -> common-row metrics + thresholds

FEWSNET.csv
  -> country == Ethiopia
  -> calendar joins near(T-4), medium(T-8)
  -> missing/coverage audit
  -> fs1/fs2 common-row baseline metrics

metrics -> 4x3 figure + run manifest
```

## Shared Hooks

1. `prepare_features(..., strict_lag_only=False)`:
   when enabled, retain static columns and the active lagged copies, and remove
   original time-varying columns. Default behavior stays unchanged.
2. Stage 1 CLI accepts an optional data path and strict-lag flag. Outputs remain
   isolated by running it with the experiment Stage 1 directory as cwd.
3. Stage 3 exposes the same strict-lag flag and applies validation thresholding
   symmetrically to pooled and partitioned models. Default threshold behavior
   stays unchanged unless the experiment flag is passed.

Before editing each shared symbol, run GitNexus upstream impact analysis. Stop
and warn on HIGH or CRITICAL risk.

## Experiment-Local Logic

- Create the filtered input, invoke the three stages, validate expected files,
  calculate the corrected FEWS NET baseline, aggregate monthly metrics, draw the
  figure, and write the run manifest. Keep this inline unless extraction is
  required by existing shared code.

## Output Contract

```text
outputs/local_partition_experiment/<run_id>/
  input/ethiopia_panel.csv
  manifests/cohort.json
  stage1/
  stage2/
  stage3/fs0|fs1|fs2|fs3/
  metrics_monthly.csv
  thresholds_by_fold.csv
  ethiopia_monthly_performance.png
  run_manifest.json
```

`metrics_monthly.csv` key: `(scope, test_month, model)` where model is
`pooled`, `partitioned`, or `fewsnet`. Unavailable FEWS NET rows remain
present with status, coverage counts, and null metrics.

## Failure Rules

- Abort on cohort/hash/key drift, missing Stage outputs, inconsistent partitions,
  test-label threshold access, or a protected output path.
- Treat months without non-null target labels as ineligible; do not fill targets
  merely to increase the Stage 1 plan count.
- Do not overwrite an existing run directory; create a new run ID.
- Generated artifacts remain local/uncommitted unless separately promoted.
