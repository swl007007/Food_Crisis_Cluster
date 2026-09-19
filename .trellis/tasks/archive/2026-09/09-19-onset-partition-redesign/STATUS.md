# Closure state — 2026-09-19

Closed and archived **while still in planning**, not because the work finished. No code was written
for this task; its entire output is the decision record in `prd.md`.

## Why it stopped

Midway through the design grill, the cause of the missing 2025-02 / 2025-06 labels was identified —
USAID suspended FEWS NET — and the team decided to treat that suspension as a research subject.
Whether it becomes the main line is a **leadership decision that had not been taken** when this task
was closed. That decision changes the test window, the horizon definition, and the success criterion,
so continuing to plan would have been guesswork.

Briefings prepared for that discussion:
`docs/notes/2026-09-19_suspension-window-briefing.md` and its Chinese counterpart
`docs/notes/2026-09-19_停摆窗口与方向提案_中文.md`.

## What survives and is ready to act on

`prd.md` D1-D12 are independent of the direction decision, because they fix defects in the existing
pipeline rather than serving any particular framing. The highest-value one is small and self-contained:

> **Re-run Stage 3 with SMOTE disabled in the partitioned arm** (~8 minutes per scope using the
> re-run path validated in the previous task). `train_partitioned_model` applies SMOTE while
> `train_pooled_model` does not, so "partitioned > pooled" — the project's one surviving claim —
> currently measures spatial partitioning and oversampling together. This settles whether that claim
> is real, and it does not depend on any pending decision.

## What must be redone before resuming

`prd.md` D13-D18, marked *direction-dependent*: adjudication comparator, effect-size targets, scope
structure, time windows, and the uncertainty estimator (fold-level bootstrap is unusable at three
test folds; an admin-level cluster bootstrap is the replacement).

Any successor needs a **fresh pre-registration**. The prior task's R19-R23 were consumed, and
reusing them on the same data would void their meaning.

## Next work

The team is moving to IPCCH GeoRF modelling. Constraints already measured for that dataset are in
`docs/notes/2026-09-18_benchmark_and_direction_review.md` section 5 and in project memory.
