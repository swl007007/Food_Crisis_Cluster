# Persistence-correction 2-layer architecture

Created 2026-09-18. Scope fixed by a full design grill on the same date; every requirement below
is a decision the user made explicitly, not an inference.

## Context

The corrected, calendar-aligned FEWS NET expert benchmark (fs1 0.807 / fs2 0.763 crisis-class F1)
lies **outside** GeoRF's PR curve at every operating point, so "GeoRF beats the FEWS NET expert"
is refuted and abandoned. Separately, GeoRF (fs1 0.682 partitioned) scores **below** a one-line
persistence rule (fs1 0.776), despite holding a correctly-dated persistence feature. Leadership
redirected the work on 2026-09-18 to: land slightly above persistence and close to the expert.

Evidence base: `docs/notes/2026-09-18_benchmark_and_direction_review.md`. A measured, test-tuned
ceiling for persistence-base + model-override is fs1 0.786 / fs2 0.741.

## Goal

Run an **internal feasibility gate** on a 2-layer forecaster: persistence as the base layer, a
single calibrated-probability threshold override as the second layer. Decide, on pre-registered
criteria, whether the mechanism produces a real gain over persistence on FEWS NET fs1/fs2 — or
whether to stop and switch to IPCCH.

This task does **not** commit to paper scope. A null result is a valid, complete deliverable.

## Requirements

- **R1 — Positioning.** Internal feasibility gate. Do not pre-commit paper scope or reviewer-facing
  framing. Do not pursue any "beat the FEWS NET expert" angle.
- **R2 — Dataset.** FEWS NET (`1.Source Data/FEWSNET_forecast_unadjusted_bm.csv`, outcome from
  `1.Source Data/Outcome/FEWSNET_IPC/FEWSNET.csv`). IPCCH is the pre-committed fallback, out of
  scope for this task's implementation.
- **R3 — Sequencing.** Fix the model before building layer 2. Layer-2 work must not start until R5
  is met or the conditional branch in R4 is exhausted.
- **R4 — Model fix method.** Conditional probability calibration first. Retraining changes (mtry
  dilution at sqrt(166)~13, the 36-month rolling window, `comp_impute` `max_plus x100` interacting
  with the binary lag column) are a **conditional branch**, attempted only if calibration alone
  fails R5.
- **R5 — Model fix exit criterion.** Within each persistence group, the mean predicted probability
  must be within 0.05 of the actual crisis rate in that group. Baseline to beat: fs1 `persist=1`
  currently sits at mean prob 0.623 against an actual rate of 0.816.
- **R6 — Availability convention.** `fews_ipc(D)` is treated as available to a forecaster at month
  D. This is an explicit assumption, justified by CS/ML1/ML2 arriving in one publication
  (`1.Source Data/Outcome/FEWSNET_IPC/scrape_fewsnet.py:4,22`): accepting the expert baseline as a
  forecast made at D requires accepting the same row's CS at D. The repo contradicts itself here
  (`src/preprocess/preprocess.py:267-273` assumes available; `EthiopiaForecastingExperiment/aligned_refit.py:122-127`
  and `.trellis/spec/backend/local-forecasting-experiments.md:73` assume not). Record the
  contradiction; do **not** modify the Ethiopia spec or its tests.
- **R7 — Horizons.** fs1 and fs2 only. fs3 and fs0 are out of scope.
- **R8 — Persistence contract.** Persistence for target T is the binarised observed phase
  (`fews_ipc >= 3`) at calendar month `T-H`, joined by `(admin_code, T-H)`. Assert join coverage
  equals 1.0 and halt otherwise. Never impute a missing observation.
- **R9 — Layer-2 form.** A probability threshold override with exactly **one** free parameter:
  when `persist = 0` and the calibrated probability exceeds the threshold, flip to 1.
- **R10 — Layer-2 scope.** One global threshold per scope. Not per-partition.
- **R11 — Calibration granularity.** Calibrators fit per `(calendar month, partition_id)` — 13 (Feb)
  + 11 (Jun) + 16 (Oct) = 40 groups, because Stage 3 ran with `--month-ind` and partitions differ by
  month. Any group with fewer than 50 cross-fit rows falls back to that month's global calibrator,
  reusing the existing abstention convention.
- **R12 — Calibration data.** Fit calibrators on out-of-fold predictions cross-fitted **inside the
  36-month training window**. Validation label months must not be consumed by calibration.
- **R13 — Threshold selection.** Select one threshold per scope on 2018-2020, freeze it, and apply
  it unchanged to all twelve 2021-2024 folds.
- **R14 — Known bias in R13.** The Stage 1 partition maps were learned on 2018-2020, so partition
  structure is in-sample there. Accept this, state it, and run a pre-registered consistency check:
  also compute the threshold that the first six 2021-2024 folds would have selected and report
  both. Divergence is a declared red flag, not a licence to switch.
- **R15 — Direction.** Only `0 -> 1` flips are reachable; enforce structurally so a `1 -> 0` flip is
  a contract error. Justification is asymmetric cost (silencing an issued crisis warning), which is
  independent of any test result. The down-flip variant's numbers may be computed as a
  **pre-registered diagnostic** with no candidate status.
- **R16 — Probability source.** Layer 2 consumes `y_prob_partitioned`, not `y_prob_pooled`.
- **R17 — Metric.** Crisis-class (class 1) F1, for both threshold selection and reporting.
- **R18 — Adjudication comparator.** persistence only (fs1 0.776 / fs2 0.709). Uncalibrated GeoRF,
  calibrated GeoRF standalone, and the expert are reported without adjudication power.
- **R19 — Statistical criteria.** All three must hold: the threshold was selected without touching
  test data; the fold-level bootstrap 95% CI excludes zero; and no single fold flips the conclusion
  under leave-one-fold-out.
- **R20 — Effect size.** Minimum detectable effect `+0.02` crisis-class F1 over persistence.
- **R21 — Gate outcome.** fs2 must clear R19 and R20. fs1 must be same-direction: point estimate
  positive and its CI not significantly negative.
- **R22 — Null reporting.** If R21 is not met, report the null result and stop. Do not add post-hoc
  variants — the prior Step 3 experiment produced two nulls and its second variant was specified
  after the first variant's test results were known.
- **R23 — Switch trigger.** Failing R21 pre-commits the project to stopping FEWS NET work on this
  mechanism and starting IPCCH. Sunk cost is explicitly excluded as a reason to continue.
- **R24 — Test-set discipline.** The 2021-2024 test set is evaluated once, after the threshold is
  frozen.
- **R25 — Code location.** New parallel package `PersistenceCorrectionExperiment/`, reusing Step 3
  utilities by import. Do not modify `Step3ExpertCorrectionExperiment/`; its 78 tests and Variant A
  byte-for-byte regression must keep passing untouched.
- **R26 — Artifact protection.** New output tree; reuse the Step 3 protected-hash gate before and
  after every run. Never write to `paper_reproducibility_package/` or `archived/`.
- **R27 — 2018-2020 probability regeneration.** 2018-2020 GeoRF probabilities do not exist and
  cannot be recovered (`app/main_model_GF.py:233` hard-codes a hard-label schema; no fitted model
  was saved). Regenerate with `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py` using
  `--month-ind` and all three `m2/m6/m10` maps at **explicit** paths under
  `paper_reproducibility_package/stage3_results/georf_fs{1,2}/refined/` — the repo-root
  `result_partition_k40_compare_GF_fs1/refined/` is empty. Set
  `NO_LEAK_PARTITION_LEARNING_YEARS` / `NO_LEAK_EVALUATION_YEARS` so the emitted manifest does not
  falsely claim `evaluation_years: 2021-2024`. Do not edit that script.
- **R28 — Testing.** Contract tests only (see AC list). Exploratory analysis code is exempt.
- **R29 — Deliverable.** `RESULTS.md` with a one-page gate verdict first: each of R19/R20/R21
  pass/fail, a one-sentence conclusion, and the R23 next step. Full detail follows.

## Acceptance Criteria

- [ ] **AC1 (R8, R6).** Persistence series is rebuilt by calendar join at `T-H` with asserted
      coverage 1.0, per-row provenance of the source month, and a halt on any violation. The R6
      assumption is stated in the run manifest.
- [ ] **AC2 (R5, R4).** Per-persistence-group reliability is reported before and after calibration,
      and the post-calibration gap is under 0.05 in every group — or the conditional retraining
      branch is documented as attempted and its outcome recorded.
- [ ] **AC3 (R11, R12).** Calibrators are shown to be fit per `(month, partition_id)` with the
      under-50-row fallback exercised and counted, and it is independently verified that no
      validation label month row entered calibration fitting.
- [ ] **AC4 (R13, R14, R24).** The frozen threshold is recomputable from the 2018-2020 artifacts
      alone; the 2021-2024 first-six-fold consistency threshold is reported alongside it; and no
      test-month row influenced selection.
- [ ] **AC5 (R15).** Assert structurally that zero `1 -> 0` flips exist in any output row, with an
      independent re-check that a forced-off defeat would be caught.
- [ ] **AC6 (R17-R21).** The gate verdict is independently recomputed from saved per-row
      predictions: per-scope F1, delta versus persistence, fold bootstrap CI, and leave-one-fold-out
      — not read back from the run's own summary.
- [ ] **AC7 (R26, R25).** Protected-input hashes are unchanged before and after every run, and the
      Step 3 test suite still passes 78/78 untouched.
- [ ] **AC8 (R27).** The 2018-2020 regeneration used `--month-ind` with all three month maps, proven
      by distinct `partition_id` counts of 13/11/16 by calendar month in its own output, and its
      manifest reports the true evaluation years.
- [ ] **AC9 (R29, R22).** `RESULTS.md` leads with the verdict page, and if the gate fails it states
      the null plainly and names IPCCH as the next step without proposing a variant.
