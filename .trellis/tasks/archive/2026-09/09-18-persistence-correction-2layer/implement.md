# Implementation plan — Persistence-correction 2-layer architecture

Ordered. Each phase ends at a review gate; do not cross a gate that failed. `R*` / `AC*` refer to
`prd.md`.

Execution mode: `trellis-implement` does the build; the **adjudication numbers in Phase 5 are
recomputed in the main session from saved artifacts**, not accepted from the sub-agent's report.

## Phase 0 — Scaffold and guard rails

1. Create `PersistenceCorrectionExperiment/` with `.gitignore` for `outputs/`, mirroring
   `Step3ExpertCorrectionExperiment/.gitignore`.
2. `protected.py`: re-export the Step 3 hash gate, extending the path list with the frozen
   `refined/` maps and `predictions_monthly.csv` for fs1/fs2 (R26).
3. Record the baseline hash set.

**Validate:** `python3 -c "from persistencecorrection.protected import hash_protected; hash_protected()"`
succeeds and reports the expected file count.
**Gate:** Step 3 suite still 78/78 (`pytest Step3ExpertCorrectionExperiment/tests/ -q`) — proves
nothing was touched (AC7).

## Phase 1 — Persistence series (R8, R6)

1. `persistence.py`: calendar join of binarised `fews_ipc >= 3` at `(admin_code, T-H)`, reusing the
   `_calendar_align` pattern from `Step3ExpertCorrectionExperiment/step3correction/expert.py:142`
   (own path — do **not** route through `ExpertTable.for_scope`, which has a firewall at `:93`).
2. Assert coverage == 1.0, halt otherwise. Carry `persistence_source_month` per row.
3. Write the R6 availability assumption into the run manifest verbatim.

**Contract tests:** coverage assertion halts on an induced gap; source month is exactly `T-H` on
every row; binarisation happens before the join; no imputation path exists.
**Validate:** reproduce fs1 0.7761 / fs2 0.7085 crisis-class F1 on the 2021-2024 support
(n=62,189 per scope) to 1e-12 — these are the numbers in
`docs/notes/2026-09-18_benchmark_and_direction_review.md`.
**Gate:** if those two numbers do not reproduce, stop; the join is wrong.

## Phase 2 — Regenerate 2018-2020 probabilities (R27)

1. Run `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py` **unmodified**, per scope:
   `--start-month 2018-01 --end-month 2020-12 --forecasting-scope {1,2} --train-window 36
    --month-ind --partition-map <...nc17_general...> --partition-map-m2 <...nc13_m2...>
    --partition-map-m6 <...nc11_m6...> --partition-map-m10 <...nc16_m10...>
    --out-dir PersistenceCorrectionExperiment/outputs/probs_2018_2020_fs{n}`
   with all four map paths taken explicitly from
   `paper_reproducibility_package/stage3_results/georf_fs{n}/refined/` (the repo-root `refined/`
   directory is empty).
2. Export `NO_LEAK_PARTITION_LEARNING_YEARS` / `NO_LEAK_EVALUATION_YEARS` before each run so the
   manifest does not falsely claim `evaluation_years: 2021-2024`.

**Validate (AC8):** the new `predictions_monthly.csv` has distinct `partition_id` counts of 14/12/17
for Feb/Jun/Oct (i.e. each month-map's `nc + 1`, matching the frozen 2021-2024 pattern), contains
`y_prob_partitioned`, and covers 9 labeled months per scope.
**Cost check:** roughly 6 minutes per scope; if a run exceeds ~20 minutes, stop and re-examine the
configuration rather than waiting.
**Gate:** partition-count pattern must match, or the probabilities are not comparable to Stage 3 and
everything downstream is void.

## Phase 3 — Calibration (R11, R12, R5, R4)

1. `calibration.py`: cross-fit inside the 36-month training window; fit isotonic (fall back to Platt
   on degenerate groups) per `(calendar month, partition_id)`; pooled per-month fallback below 50
   cross-fit rows. Count and report fallbacks.
2. Reliability tables before and after, per persistence group, reusing the binning in
   `scripts/paper_artifacts/analyze_georf_probability_uncertainty.py:115-148`.

**Contract tests (AC3):** no validation-label-month row appears in any calibration fit; the
under-50 fallback fires and is counted; `(month, partition)` grouping is used, never `partition`
alone.
**Validate (AC2):** post-calibration, every persistence group has
`|mean predicted prob − actual crisis rate| < 0.05`. Current fs1 `persist=1` baseline is 0.623 vs
0.816.
**Gate — hard stop (R5):** if calibration alone fails, run the R4 conditional branch (mtry, window,
`comp_impute` interaction) as a **time-boxed** investigation. If that also fails, stop the task and
report per `design.md` "What would falsify the whole design". Do **not** proceed to Phase 4 with a
model that fails R5.

## Phase 4 — Threshold selection and freeze (R13, R14, R9, R10, R15, R16)

1. `selection.py`: sweep `tau` on the 2018-2020 calibrated probabilities, argmax crisis-class F1 of
   the override, one scalar per scope. Freeze to a checked-in JSON with a hash.
2. Compute and record the R14 consistency threshold from the first six 2021-2024 folds. Report both;
   never substitute.
3. `override.py`: apply the frozen `tau`. Up-only is structural — there is no code path that can
   emit `1 -> 0`.

**Contract tests (AC5, AC4):** zero `1 -> 0` flips in any output, plus an independent re-check that
would catch a forced-off defeat; the frozen `tau` is recomputable from 2018-2020 artifacts alone;
no test-month row influenced selection.
**Gate:** `tau` is frozen and hashed **before** any 2021-2024 row is scored (R24).

## Phase 5 — Adjudication (R17-R22, R29)

1. `adjudicate.py`: apply the frozen `tau` to 2021-2024, once. Per-scope crisis-class F1, delta vs
   persistence, fold-level bootstrap 95% CI, leave-one-fold-out.
2. Report without adjudication power: uncalibrated GeoRF, calibrated GeoRF standalone, expert.
3. Compute the pre-registered down-flip diagnostic, labelled as diagnostic (R15).
4. `RESULTS.md`: verdict page first — R19/R20/R21 pass/fail, one-sentence conclusion, R23 next step.

**Validate (AC6) — main session, not the sub-agent:** recompute every adjudication number from the
saved per-row predictions independently of the run's own summary.
**Gate (R21, R22):** fs2 clears `+0.02` with a CI excluding zero and survives LOFO, and fs1 is
same-direction. If not, `RESULTS.md` states the null and names IPCCH as the next step. **No variant
may be added after seeing these numbers** — that is exactly how Step 3's Variant B became
post-hoc.

## Phase 6 — Close out

1. Re-run the protected-hash gate; confirm unchanged (AC7).
2. Re-run the Step 3 suite; confirm 78/78.
3. Update `docs/notes/2026-09-18_benchmark_and_direction_review.md` with the outcome and write the
   verdict into project memory.

## Rollback points

- After Phase 0: delete the package directory.
- After Phase 2: additionally delete `outputs/probs_2018_2020_fs*`. Nothing outside the new tree was
  written at any point, so rollback is always a directory deletion.

## Pre-registration freeze

`prd.md` R15, R19, R20, R21, R22, R23 are frozen at Phase 0 and must not be edited after Phase 2
begins. If a genuine specification defect is found, stop, record it in `RESULTS.md` as a
specification change with its date and reason, and treat every subsequent number as post-hoc.
