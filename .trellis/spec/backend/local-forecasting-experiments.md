# Local Forecasting Experiments

## 1. Scope / Trigger

Use this contract for isolated country-level Stage 1-3 runs that add alternate
data, lag, seed, threshold, or output-path CLI options without changing
production defaults.

## 2. Signatures

- Runner: `run_local_partition_experiment.py --panel PATH --fewsnet PATH --output-root PATH --run-id ID`
- Stage 1 experiment flags: `--data PATH --strict-lag-only --random-seed INT`
- Stage 3 experiment flags: `--strict-lag-only --enable-symmetric-validation-threshold`

## 3. Contracts

- Freeze the cohort by canonical admin-month key and source hashes before fitting.
- A scope changes only its lag; static features stay shared and time-varying
  features come only from the active lag.
- Stage 1 uses only months with observed targets. Never fill unlabeled months to
  reach a planned fit count.
- Repository-local output paths must remain under
  `EthiopiaForecastingExperiment/outputs/local_partition_experiment/`; scratch
  paths outside the repository are allowed for tests.
- Existing run IDs are immutable. Create a new run directory for every rerun.

## 4. Validation & Error Matrix

| Condition | Required result |
|---|---|
| Null/duplicate cohort key or hash drift | Abort |
| Unlabeled Stage 1 month | Exclude; do not impute target |
| Repository path outside the approved experiment root | `ValueError` before directory creation |
| Existing run directory | `FileExistsError` |
| FEWS NET projection coverage below 90% | Keep row, null metrics, `suppressed_low_coverage` |
| Partition/admin-code mismatch | Abort |

## 5. Good / Base / Bad Cases

- Good: 2018-2020 February/June/October labels yield exactly 36 plans for four scopes.
- Base: a temporary directory outside the repository is accepted by unit tests.
- Bad: filling the other calendar months or writing under `result_GeoRF*`.

## 6. Tests Required

- Assert the labeled-month plan count and lag schedule.
- Assert strict-lag selection removes contemporaneous time-varying features.
- Assert pooled and partitioned thresholds use validation data only.
- Assert protected repository paths fail before any directory is created.
- Independently recompute exported metrics and verify source/code hashes.

## 7. Wrong vs Correct

```python
# Wrong: manufacture target support to satisfy an expected loop count.
panel["target"] = panel["target"].fillna(0)

# Correct: derive eligible Stage 1 months from observed target support.
eligible = panel.loc[panel["target"].notna(), "date"].dt.month.unique()
```
