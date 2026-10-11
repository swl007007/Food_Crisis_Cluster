# c8 probe result (release codex-D30-c8-probe-20261011) — 2026-10-11

## Preconditions

All passed at 06:00Z, recorded in `run-20261011/precheck.json`:
- **Full invocation 1:** exactly one runner `exit 0`; stopped on "invocation wall budget 21600.0 s reached"; checker exit
  0 with 0 violations.
- **Budgets compared explicitly:** calls 3,811 / 59,576; invocations 1 / 20; projection within cap.
- **No overlap:** no active EE runner; the full run's `invocation.lock` was held for the whole probe.
- **Hashes:** the 9 sources plus released manifest, body and pieces all match.
- **Fresh requests:** none of the 24 probe geometry+candidate signatures was among the 3,807 attempted full requests.

## Run

- One invocation, 06:00:53Z → 06:03:05Z: runner exit 0, invocation 131.5 s, calls 29 / 50, projection check 2.8e-9 m.
- 23 of 24 pieces were saved and pass every invariant:
  - 129 finite sums;
  - owned area by rank = valid area, within ≤ 2.1e-16;
  - valid cells ≤ lattice box;
  - every piece < 90 s.
- **1 piece did not complete: `4525|w007_001`** (1 candidate, 1.6e7 cells). It stays pending; there was no second probe
  attempt.

**Request errors** (exact text "Too many concurrent aggregations." on all four attempts):

| n | utc | piece | attempt | s | outcome |
|---|---|---|---|---|---|
| 10 | 06:00:56Z | 2637\|w003_005 | 1 | 0.35 | retry ok |
| 17 | 06:01:40Z | 4525\|w007_001 | 1 | 0.17 | retried |
| 18 | 06:01:40Z | 4525\|w007_001 | 2 | 0.64 | retry exhausted → ledger `failed` |
| 19 | 06:01:41Z | 4673\|w005_000 | 1 | 0.59 | retry ok |

**Guard and stop behaviour:**
- The text matches neither FATAL nor NONRETRYABLE, so it took the bounded retry: one wrapper retry per call, counted
  against the call cap.
- The exhausted piece counted as 1 failure. The consecutive-failure count was reset by the next success, and the failure
  fraction was 1/24 = 0.042, below the probe's 0.25.
- The stop control never fired (`stopped` null). The run finished with `remaining` 1 and `complete` false, and the
  checker reports 0 violations, 23 ok, 1 failed.
- The rejections are fast (0.17–0.64 s): the service refuses new aggregations at submission time.

## Timing (client-operation evidence, not server execution instrumentation)

`Budget.call` durations include response serialization, and the log records completion timestamps at 1 s precision.

- **Overlap** (corrected wording per the supervisor check `run-20261011/ipcch-alphaearth-c8-supervisor-check-20261011.json`):
  - The guaranteed client overlap is 8 (`ThreadPoolExecutor(8)`; a retry runs on the same thread).
  - Reconstructing start = completion − duration from 1 s timestamps gives a loose interval upper bound of 10–11. That is
    an interval bound, not an observed overlap above 8.
  - None of this is server concurrency.
  - Rejections occurred with an estimated 6–8 client operations outstanding.
- **Throughput.** Request span 130.2 s, Σ ok seconds 870 → effective client parallelism 6.7 (supervisor's span definition: 6.63). 3.218e8 cells completed, i.e.
  2.45e6 cells per invocation-second.
- **Unmatched c4 comparators.**
  - Full invocation 1: 3.389e10 cells / 21,617 s = 1.57e6 cells/s; parallelism about 3.7; 2 recovered errors in 3,811
    calls.
  - S2: 1.72e6 cells/s.
- **Per-request time rose at c8.** The 1.6e7-cell pieces took 32–58 s, against 23–32 s (1–2 candidates) and 46 s (16
  candidates) at c4.

The cells/s ratio of about 1.4–1.6 is not an unbiased speedup:
- the probe pieces are heavier, which amortizes the per-request overhead;
- they come from one draw at one time of day;
- shared asset or tile caches cannot be ruled out.

## Interpretation

At c8 the service rejected 4 of 27 piece attempts (15%) with "Too many concurrent aggregations.", and 1 of 24 pieces
(4%) failed after its one retry. At c4, invocation 1 had 0 failures in 3,807 pieces, with 2 transient errors, both
recovered.

Under the released production guards (failure fraction ≤ 2% after 200 attempts) a 4% piece-failure rate would trip the
stop rule. More rejections also mean more re-request calls against the cumulative cap.

On this evidence c8 is not a supported production setting, and c4 remains the validated default. This is a feasibility
observation from one bounded probe, not a measured capacity limit of the project.

Independent check (supervisor): 23 saved outputs pass all hash, binding, 129-finite, rank-area and count checks; only
4525|w007_001 is missing; 29 calls, 4 errors, 3 retries; incomplete and FAIL for promotion.

Follow-up: "too many concurrent aggregations" is now a shared fatal stop in `scripts/ee_metadata.py` FATAL
(evidence/p1/c-full-20261010/fatal-repair/). The c8 run itself is unchanged and is not repeated.
