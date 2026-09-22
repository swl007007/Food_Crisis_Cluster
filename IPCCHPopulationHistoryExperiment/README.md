# IPCCH population-history and pooled XGBoost objective comparison

Does rich continuous phase-distribution history improve IPCCH crisis
forecasting, and does learning *persistence errors* or *continuous P3+ shares*
beat learning the binary outcome directly?

Spec: `.trellis/tasks/09-21-ipcch-population-history-xgb/` — `prd.md` for
requirements and acceptance, `technical-contract.md` for the exact formulas,
`design.md` and `implement.md` for build and verification order. The two frozen
inventories live here in `config/` and are copied verbatim into each run.

This experiment does not fit GeoRF, learn partitions, touch geometry, or modify
the earlier IPCCH experiment. It reuses that experiment's target QC, its
93-column feature builder and its country lookup so the two cannot drift apart
on what a valid outcome is.

## The arms

| arm | estimator / target | X | fitting support |
|---|---|---|---|
| `binary_history_xgb` | XGBClassifier, `y` | 93 | matched |
| `rich_direct_xgb` | XGBClassifier, `y` | 561 | matched |
| `correction_xgb` | XGBClassifier, `e = 1[y != b]` | 561 | matched |
| `share_xgb` | XGBRegressor (squared error), `q3` | 561 | matched |
| `rich_rf` | RandomForestClassifier, `y` | 561 | matched |
| `fullpool_xgb` | XGBClassifier, `y` | 561 | full pool |
| `persistence` | `b`, no fitting | — | history available |

`b` is the latest valid binary label at or before the row's own origin. The
five matched arms fit on **byte-identical ordered keys**, so any difference
between them is a difference of features or objective and nothing else.
`fullpool_xgb` deliberately trains on the superset that also contains rows
without `b`, and is the only arm that can score rows with no persistence at all.

## Features

`rich561` = the frozen `original93` block, unchanged and in its original order,
plus 468 appended history columns built from eight series — `q2 q3 q4 q5`,
`severity_index`, `entropy`, `concentration`, `severe_fraction` — across six
observation slots, five windows (`m06 m12 m24 m36 all`) and six statistics.
`config/feature-schema.json` is the authority for every name and its position.

Three rules the code is built around:

* **Own origin.** Every value for a row is computed at that row's own
  `o = T - h`. A later refit never re-dates a training row's history.
* **Full ledger.** History comes from all valid outcomes, not from the fitting
  or evaluation subset. A window may reach back before the fitting window; it
  can never reach past `o`.
* **Missing stays missing.** Mean needs one value, std two, slope three. Short
  of that the cell is `NaN`, never 0, and no row is ever dropped for it.

Five names are deliberately *not* materialised because an `original93` column
already holds that exact quantity (`hist_age_obs1`, `hist_crisis_obs1`,
`hist_crisis_age`, `hist_no_crisis`, `hist_support_common_all_age`).
`prepare_data.check_aliases` recomputes each one and compares it exactly, so the
deduplication is verified per run rather than assumed.

## Running it

Use the pinned Windows interpreter; `prepare_data.verify_runtime` compares the
whole stack (Python 3.12.10, numpy 2.2.6, pandas 2.2.3, sklearn 1.6.1,
XGBoost 3.0.0, scipy 1.15.2) and stops on any mismatch.

```
python -B IPCCHPopulationHistoryExperiment/test_contracts.py
python -B IPCCHPopulationHistoryExperiment/prepare_data.py --source-root SRC --run-dir RUN
python -B IPCCHPopulationHistoryExperiment/run_pipeline.py --run-dir RUN --stage pilot
python -B IPCCHPopulationHistoryExperiment/run_pipeline.py --run-dir RUN --stage development --workers 12
python -B IPCCHPopulationHistoryExperiment/run_pipeline.py --run-dir RUN --stage select
python -B IPCCHPopulationHistoryExperiment/run_pipeline.py --run-dir RUN --stage main --workers 12
python -B IPCCHPopulationHistoryExperiment/run_pipeline.py --run-dir RUN --stage verify
python -B IPCCHPopulationHistoryExperiment/run_pipeline.py --run-dir RUN --stage persist --workers 12
python -B IPCCHPopulationHistoryExperiment/run_pipeline.py --run-dir RUN --stage replay-models
python -B IPCCHPopulationHistoryExperiment/report_results.py --run-dir RUN --out-dir RUN/reports \
    --replay-into RUN/validation/reports_replay
```

`--stage select` reads only stored 2020-2022 predictions and writes
`freeze.json`; the main schedule refuses to run without it.

### Stages are immutable once complete

A freeze is written once. Re-running `select` into a run that already has one is
**refused**, because a second freeze with a fresh timestamp would silently
relabel main-schedule predictions computed under the first. To re-derive it for
comparison, use `--stage select --replay-into DIR`; `--refreeze` exists only for
deliberately discarding a freeze.

Every fold record carries an `identity`: the source/schema/candidate hashes, the
key and calendar digests, the frozen selection digest and the code hashes.
Resuming reuses only folds whose identity still matches, so an interrupted run
resumes safely.

A completed fold whose identity does **not** match is neither reused nor
re-fitted — the run is rejected and you are pointed at a fresh run directory.
There is deliberately no override: overwriting it in place would destroy the
evidence of what that fold produced while keeping the same run id, freeze and
reports. A second guard checks that nothing already recorded as complete can
enter the fitting queue at all, whatever the reuse logic decided.

`--stage main` also refuses to start when the frozen inputs no longer describe
the run. The comparison is field by field, and a field the freeze never carried
is treated as an *old schema*, not as drift: it is named and refused
(`--allow-legacy-freeze`) rather than passed over or filled in with a hash
invented after the fact. Code that moved since the freeze needs
`--allow-code-drift`, and a reason in the run record.

### Retained models

`--stage persist` refits each selected model at each main origin under the
frozen choices and **keeps** the fitted object, recording its digest, its full
`get_params()` readback, its booster configuration and round count, and the
digest of the ordered fitting keys it was built from. It then requires the
regenerated predictions to be identical to the stored ones, fold by fold,
across the whole schedule — and writes to a scratch directory so `main/folds`
is never rewritten.

`--stage replay-models` is the check a refit cannot make: it loads each saved
estimator off disk, verifies its digest and predicts, requiring the stored
scores back. A model silently replaced or re-fitted under different data fails
here even though a refit would pass.

The 533 MB of estimators stay local. Their identity records are committed in
full, along with the binaries for the first main fold at each horizon so a
reviewer can exercise prediction-only replay directly.

### Recorded deviation: fold-level parallelism

`technical-contract.md` §7 asks for sequential folds and candidates with
`n_jobs=1`, and also says a resource problem returns for a bounded adjustment.
Measured: the schedule is ~21 h sequential. It was run with `--workers 12`.

Every estimator keeps the frozen `n_jobs=1`, so each fit sees the same data,
the same seed and the same thread count regardless of what runs beside it. That
is asserted nowhere and verified instead: `--stage verify` replays eight main
folds **sequentially, one at a time**, and requires the predictions to be
identical to the parallel run's. `validation/replay.json` holds the comparison.

## Selection and the freeze

For each arm, horizon and candidate, development predictions are pooled over
all 2020-2022 targets on `E_history`, and a threshold pair `(t0, t1)` — one per
`b` state — is chosen to maximise the **pooled** class-1 F1. Decisions are
always `score > t_b`; equality is negative. `t0 = +inf` and `t1 = -inf` are the
no-flip options, which reproduce persistence exactly.

Ties break in the declared order: higher F1, then fewer changes from `b`, then
the lexicographically smallest `(t0, t1)` in extended-real order. Across
candidates: higher F1, then earlier JSON order. The primary family is chosen
among `rich_direct_xgb`, `correction_xgb` and `share_xgb` by the equal mean of
four horizon development deltas against persistence.

Everything — candidate, both thresholds, the primary family, schema and code
hashes — is written to `freeze.json` before a single main-schedule prediction
is computed. Rolling refits later in the calendar may use newly available past
labels; they never revisit a choice.

## Evaluation cohorts

* `E_history` — test keys that have `b`. All seven methods share these keys
  exactly, and every headline number and claim uses them.
* `E_no_history` — test keys with no `b` at all. Persistence is *undefined*
  here, not zero. Every combined stream routes these rows to `fullpool_xgb` at
  a fixed 0.5 cutoff, never tuned.
* `E_all` — the disjoint union, reported separately and labelled as a
  method-plus-shared-fallback stream rather than as the method.

`report_results._cohort_audit` proves all three before any metric is computed:
identical keys across arms, identical truth and persistence on those keys, and
an exact partition.

## Claims and what counts as a gain

Three claims are reported separately, and all of them are reported whether they
succeed or not:

1. **prediction gain** — primary family vs `rich_rf` *and* vs `persistence`;
2. **formulation advantage** — only if the primary is `correction_xgb` or
   `share_xgb`: additionally vs `rich_direct_xgb` *and* `fullpool_xgb`;
3. **information gain** — `rich_direct_xgb` vs `binary_history_xgb`.

A gain is *stable* only when four conditions hold together: mean delta F1 > 0,
the 95% lower bound > 0, no horizon with a negative point delta, and a positive
mean delta after omitting each of 2023, 2024 and 2025 in turn. Required
comparisons are conjunctive — the most favourable baseline cannot be selected
after the fact. Missing evidence is reported as `incomplete`, which is not a
pass. A complete negative or mixed result is a valid outcome of this study.

The interval comes from 2,000 valid country-clustered draws (seed 42, at most
20,000 attempts). One multiplicity vector per draw is reused by **every** method
and horizon, so a method and its baseline are always compared on the same
resampled cohort; a draw that leaves any required cell undefined is rejected in
full rather than kept for some contrasts.

## Limitations carried deliberately

* Retrospective. The 2023-2025 outcomes were inspected before this design
  existed; this is a controlled comparison, not an untouched holdout.
* Source-month alignment does not establish publication-time availability.
* Claim 3 contrasts a *tuned feature pipeline*, not a fixed-hyperparameter
  ablation, and the rich schema adds continuous shares and longer binary
  history together — it does not isolate the contribution of shares alone.
* The interval is conditional on the trained predictions. It is not a refit
  bootstrap and says nothing about future years or countries.
* The RF imputer keeps the pinned `max_plus` rule (training max x 100; max 0
  gives 100; an all-missing column gives 0). For a negative column the fill
  lands *inside* the column's range. That is retained for baseline continuity
  and disclosed here rather than silently changed.
* A good F1 decision threshold is not a claim of calibrated probability, and
  the mean predicted share is not a crisis probability.

## Run layout

```
runs/<id>/
  manifest.json  run.log  freeze.json  stage_*.json
  inputs/        frozen inventories, baseline identity
  baseline/      verified pristine imputer provider
  data/          ledger, rich561 matrix, schema, per-row history slots, audits
  folds/         calendar with per-fold support counts
  development/   per-fold scores, selection_ledger.csv
  main/          per-fold scores under the frozen choices
  validation/    sequential replay, reporter replay
  reports/       metrics, deltas, strata, bootstrap draws, verdicts
```
