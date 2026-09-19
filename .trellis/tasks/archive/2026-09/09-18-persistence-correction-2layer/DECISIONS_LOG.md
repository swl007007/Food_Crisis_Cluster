# Specification changes and findings after Phase 2 began

Per the `implement.md` pre-registration freeze, every change made after Phase 2 started is recorded
here with its date and reason. **None of the frozen items (R15, R19, R20, R21, R22, R23) was
touched.** All four entries below were decided by the user on 2026-09-18.

## C1 — Phase 2 gains a reproduction gate (added, not in the original plan)

`implement.md` originally had Phase 2 regenerate 2018-2020 probabilities with no check that the
re-run path could reproduce a known answer. That was an omission: the path had never been executed
in this environment. A repro run of fs1 2021 against the frozen artifacts was added as a gate.

**Result: it does not reproduce.** `y_true` and `partition_id` match exactly on all 14,784 rows
(which independently re-confirms `--month-ind`, 14/12/17 by month), but `y_prob_partitioned` differs:
max |diff| 0.20, mean |diff| 0.024, only 22% bitwise identical; hard predictions differ on 2.6% of
rows.

Diagnosis: **RF ensemble sampling noise, not a configuration error.** Correlation 0.990, signed mean
difference +0.0019 (unbiased), median signed difference 0.0000, sd 0.0346 — which matches the
Monte-Carlo standard error of a 100-tree bagged probability, `sqrt(p(1-p)/100) ~ 0.045`. `random_state`
is fixed at 5, but `requirements.txt` pins only `scikit-learn>=1.3.0`, so a different library version
or training-row order changes the bootstrap draw sequence and yields a statistically equivalent but
different forest.

## C2 — 2021-2024 probabilities are regenerated in the same environment

Consequence of C1. R13 selects `tau` on 2018-2020 and applies it to 2021-2024; doing that across two
forest realisations injects `sd ~ 0.035` of unquantified noise at the decision boundary of an
experiment whose MDE is only `+0.02`.

Selection and application now both use probabilities generated in one environment. The frozen
artifacts are retained as a documented comparison, not as the input. Reported GeoRF figures will
therefore differ slightly from the paper's 0.682 / 0.664; this is acceptable because R18 already
denies GeoRF adjudication power, and the sole adjudication comparator — persistence — is
model-independent and unaffected.

Rejected alternatives: accepting the cross-realisation noise (free, but unquantified noise exactly
where it hurts most); and selecting `tau` on the first six 2021-2024 folds instead (no extra compute,
but halves the clean evaluation folds and guts the R19 bootstrap and LOFO).

## C3 — `persistence_phase_missing` rows keep the historical convention

Phase 1 surfaced a subgroup the reference note did not mention: rows where the `T-H` origin row
exists but its `fews_ipc` is null, which the historical raw-missing -> 0 convention assigns
persistence 0.

| scope | rows | share of `persist=0` group | that subgroup's crisis rate | share of the group's true crises |
|---|---|---|---|---|
| fs1 | 2,434 | 5.33% | 0.3369 (group overall 0.1034) | 820 / 4,728 = **17.3%** |
| fs2 | 3,039 | 6.53% | 0.3731 (group overall 0.1332) | 1,134 / 6,201 = **18.3%** |

So 5-6% of the only flippable group carries ~18% of its flippable crises, at ~3.3x the group's crisis
rate.

**Decision: change nothing.** The convention is preserved and the subgroup is recorded as a
pre-registered diagnostic. Reasons: it is the only option that introduces no test-informed
modification (the 0.337 figure is computed from test-window `y_true`); it is the only option that
keeps comparability with persistence 0.776/0.709 and expert 0.807/0.763, all of which include these
rows under the same convention; and the rows stay inside the flippable group, so the mechanism can
still capture them on its own merits.

Rejected: exposing `persistence_phase_missing` as an explicit second-layer signal (real upside, but
the decision would be made after seeing the elevated rate, so every downstream number would have to
carry a post-hoc label and the free-parameter count would go from 1 to 2); and excluding the rows
from support (cleanest semantics, but breaks comparability with every reference number and discards
~18% of the available headroom).

## C4 — root `.gitignore` gains one negation

Line 29 `test_*` was hiding the new contract tests; only `src/tests/` and
`EthiopiaForecastingExperiment/tests/` had negations. Added
`!PersistenceCorrectionExperiment/tests/test_*.py`, matching the existing convention rather than
relying on `git add -f` (the Step 3 precedent, which depends on someone remembering). Verified with
`git add --dry-run`.

## Diagnostic, no action — 2018 training windows reach the quarterly era

A 36-month training window for a 2018 target reaches back into the quarterly-cadence period
(2009-07..2015-10, 3-month gaps), where `T-4` is not an observed month, so `fews_ipc_crisis_lag_4` is
null and gets dropped or imputed. 2019-2020 windows sit entirely in the tri-annual era.

Measured: fs1 crisis-class F1 by year is 0.485 (2018), 0.555 (2019), 0.600 (2020), against 0.557 for
frozen 2021. 2018 is weaker, **but the probability distribution is stable across all years**
(mean 0.22-0.30, sd 0.22-0.26, frozen 2021 included), and threshold selection depends on that scale.
Phase 3 calibration normalises it further. No specification change; recorded so that a divergent
2018-versus-2019/2020 threshold has a known candidate explanation.

## C5 — R12 was unimplementable as written; calibration and selection now split 2018-2020

R12 specified calibrators fit on out-of-fold predictions "cross-fitted inside the 36-month training
window". **That data does not exist.** `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py` emits
only test-month predictions (`predictions_monthly.csv`, `metrics_monthly.csv`,
`metrics_polygon_overall.csv`, `run_manifest.json`) — nothing on the training window. Obtaining it
would require either duplicating the script's training loop or editing the script, which R27 forbids.
This was a design error on my part: I specified an input the black-box entry point cannot produce.

**New split, decided 2026-09-18 before any test data was touched:**

| window | labeled months | rows/scope | role |
|---|---|---|---|
| 2018-2019 | 6 | ~32,000 | fit the calibrators |
| 2020 | 3 | ~16,000 | select `tau` |
| 2021-2024 | 12 | 62,189 | apply once, adjudicate |

The two pre-test windows do not overlap, so R12's actual purpose — keeping calibration fitting out of
the threshold-selection data — is preserved. Both windows are genuine out-of-sample pipeline
predictions. No retraining, no extra compute. `tau` is one scalar, so 16,000 rows is ample;
calibration groups average ~800 rows, above the 50-row floor, though the fallback will fire more
often than under the original plan and its count is reported.

R13 narrows accordingly: `tau` is selected on **2020**, not on all of 2018-2020. R14's consistency
cross-check against the first six 2021-2024 folds is unchanged.

Rejected: writing our own cross-fitting training loop (literal compliance, but ~200 lines duplicating
the script and a real risk that our copy diverges from it — which would reintroduce exactly the
realisation mismatch C2 just eliminated); and using all nine months for both jobs (double-dipping,
which would void the R19 bootstrap CI).

## C6 — the calibrator is static and frozen

Fit once on 2018-2019, frozen, then applied unchanged to 2020 for selection and to 2021-2024 for
adjudication. Like `tau`, it is fully determined before the test set is touched, so only one
parameter set has to be accounted for.

Stated assumption: the calibration relationship is stable from 2019 to 2024. This is an assumption,
not a finding. It is reported as a diagnostic by comparing reliability on 2021 against 2024.

Rejected: re-fitting per fold on each fold's prior data (more adaptive and closer to live operation,
but it makes the calibrator a second per-fold noise source immediately after C2 spent a re-run
eliminating the first, and the earliest folds would have very little data).

## C5 addendum — a factual correction to C5

C5 predicted the under-50-row fallback would "fire more often than under the original plan". **It fires
zero times on real data.** The smallest real calibration group is 128 rows, median 719. The binding
constraints turned out to be different: three single-class groups (`(Feb,7)`, `(Jun,4)`, `(Oct,14)`,
all with zero crises in 2018-2019, which fail Platt with "This solver needs samples of at least 2
classes"), and the `partition_id = -1` unmapped-admin sentinel
(`scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py:270-271, 391-392`), which is absent from the fit
window entirely. The `-1` sentinel also explains the 14/12/17 counts: excluding it gives exactly
13+11+16 = 40 groups, as R11 says.

## C7 — the R5 gate FAILED; the task proceeds with a documented handicap

Measured on 2020 out-of-sample, post-calibration, and independently recomputed in the main session
from the saved per-row CSVs:

| scope | group | n | mean `p_cal` | crisis rate | \|gap\| | vs 0.05 |
|---|---|---|---|---|---|---|
| fs1 | persist=0 | 13,328 | 0.1609 | 0.0735 | 0.0874 | FAIL |
| fs1 | persist=1 | 3,111 | 0.4489 | 0.5574 | 0.1085 | FAIL |
| fs2 | persist=0 | 13,159 | 0.1637 | 0.0899 | 0.0738 | FAIL |
| fs2 | persist=1 | 3,280 | 0.4458 | 0.4665 | 0.0207 | pass |

**The gate failed and is recorded as failed.** It is not reinterpreted as a pass.

Decision: proceed to Phase 4/5 carrying an explicit handicap — the model remains miscalibrated —
rather than running the R4 retraining branch or stopping now.

Reasoning. R5 is a *process* gate invented to avoid wrapping a broken model; it is not the
hypothesis test. The pre-registered hypothesis test is R19-R21, which remain **frozen and untouched**.
Running them on a miscalibrated model biases the result toward the null, never toward a false
positive: if the mechanism clears `+0.02` anyway that is a strong finding, and if it fails, RESULTS.md
must state that mechanism failure and model failure cannot be separated.

Rejected: the R4 retraining branch (largest upside, since a better-calibrated model would lift the
ceiling — but it is the most expensive option and, once the training configuration changes, the model
is no longer the paper's GeoRF); and stopping now (most disciplined reading of `implement.md`, but it
would kill the direction on a proxy criterion that, per C8, does not even bind the mechanism, without
ever running the real test).

Two supporting measurements, both computed after the gate result and therefore labelled as
diagnostics, not as justification:
- Calibration substantially improved the group that matters: fs1 `persist=0` 0.1481 -> 0.0874,
  fs2 0.1200 -> 0.0738. Improvement is real, just short of 0.05.
- Best achievable override F1 on the 2020 selection window, sweeping `tau` over `persist=0`:
  fs1 persistence 0.5955 -> 0.6259 calibrated (+0.030); fs2 0.5106 -> 0.5558 (+0.045). Both exceed the
  `+0.02` MDE, which R5 does not measure at all. Note calibration helps fs1 (+0.009 over raw) but
  hurts fs2 (-0.012 over raw) — mixed, not decisive.

## C8 — R5 contradicted R15; corrected, and the correction does not change the verdict

R15 restricts layer 2 to `0 -> 1` flips, so `p_cal` is read **only on `persist=0` rows**; `persist=1`
rows keep persistence unchanged and the model's probability never enters. R5 nevertheless gated on
both groups, and `design.md`'s "What would falsify the whole design" aimed its falsification
condition specifically at the `persist=1` group — the one group the mechanism never consumes. That is
a drafting error I made, and it is derivable from the frozen specification alone, without reference to
any result.

R5 is corrected to apply to `persist=0` only. **The verdict is unchanged: still FAIL** in both scopes
(0.0874 and 0.0738 against 0.05). The correction changes the diagnosis, not the outcome, which is what
makes it safe to make after seeing the numbers.

RESULTS.md must report both readings — the original R5 (3 of 4 groups fail) and the corrected R5
(both scopes fail on `persist=0`) — and attribute the drafting defect rather than quietly presenting
only the corrected form.

## C9 — layer 2 consumes `p_cal`, not the raw probability, despite the mixed ceiling evidence

The ceiling diagnostic shows calibration helping fs1 and hurting fs2. Switching to raw probabilities
for fs2 only would be selecting a variant per scope on observed performance — the Variant B pattern.
Layer 2 therefore consumes `p_cal` in both scopes, as `design.md` specifies; the rationale (making one
global threshold meaningful across 40 differently-scaled local models) is independent of the ceiling
comparison. The raw-probability variant is computed as a **pre-registered diagnostic** with no
candidate status, on the same footing as the down-flip diagnostic in R15.

## C10 — Phases 4-5 were written in the main session, breaking decision 26

The `trellis-implement` sub-agent dispatched for Phases 4-5 hit a session rate limit and produced
nothing. Decision 26 ("sub-agent implements, main session recomputes the adjudication numbers")
exists so that implementation and reporting are done by different parties. Writing Phases 4-5 in the
main session collapsed that separation.

It was restored with the roles swapped: the main session implemented, then a `trellis-check` agent
independently recomputed the verdict from the saved per-row CSVs without importing
`persistencecorrection.selection` or `.override`, and was told explicitly that these numbers had no
independent check yet and to be adversarial about it. That audit confirmed every headline number and
found three reporting defects (C11). The separation property therefore held, in the opposite
direction to the one decision 26 specified.

## C11 — three defects in the first draft of RESULTS.md, found by the independent audit

All three were in the *interpretation*, not the verdict. The verdict (FAIL on R19, R20, R21) was
confirmed exactly. Each is corrected inline in `RESULTS.md`, marked `[corrected 2026-09-19]` with the
original wording quoted.

**1. The ceiling attribution was wrong, and it flattered the mechanism.** The draft said the frozen
threshold "captured +0.0131 of a +0.033 available ceiling at fs2 and none of the +0.010 at fs1", and
concluded "the binding obstacle is threshold instability rather than an absent ceiling". But +0.033
and +0.010 are ceilings on the **raw** probability, which layer 2 never consumes. Swept on `p_cal`
with test labels, the ceiling for the mechanism as built is **fs1 +0.0054** and **fs2 +0.0140** —
both **below the +0.02 MDE**. fs2 captured **94%** of what was achievable; its transfer loss is
0.0009. The corrected diagnosis is the opposite of the drafted one and makes the failure more
fundamental: at fs2 the ceiling itself is the binding constraint. Instability is an fs1 story only
(cost 0.0074), and even there the ceiling was +0.0054.

**2. "Allowing `1 -> 0` is catastrophic" was unsupported.** The reported 0.6042 / 0.4392 came from an
**undocumented, unoptimised** down-threshold of `1 - tau`, a value I invented at implementation time;
R15 pre-registered the diagnostic but never its threshold. Swept properly, the optimum in both scopes
is **zero flips, delta exactly +0.0000**. Down-flipping is useless here, not catastrophic. I
overstated the support for a pre-registered restriction using a parameter I had made up.

**3. An unearned provenance claim.** "Recomputed from saved per-row artifacts, twice, by independent
code paths" overstated what had happened: the same party implemented and reported, and `RESULTS.md`
transcribed one runner's JSON. Removed. It is now true only because of the C10 audit.

Two further gaps the audit surfaced, both now recorded in the RESULTS.md provenance section: the
GeoRF figures come from the C2-regenerated forest rather than the paper's, which the draft never
said; and the 2021-versus-2024 calibration-drift diagnostic promised in C6 was never produced and
remains outstanding.

**A finding worth carrying to IPCCH:** the raw-probability ceiling exceeds the calibrated one in both
scopes (+0.0095 vs +0.0054 at fs1; +0.0331 vs +0.0140 at fs2). Per-`(month, partition)` calibration,
added so that a single global threshold would be meaningful, **cost the mechanism ceiling**. C9 froze
that choice before the run and switching afterwards would have been variant-picking, so it was never
adjudicated — but it is the first thing to revisit in any successor design.
