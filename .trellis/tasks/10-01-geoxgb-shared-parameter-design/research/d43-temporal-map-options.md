# D43 temporal map-learning options: design note (2026-10-02)

**Status:** D43 is NOT approved and has NOT been run. This note is design discussion only: no spec, code, fits or production edits.

## Considered and deferred: refitting legacy D26/D27 maps

The proposal was to take the D27 time-block (tb3) map and the D26 random-split (r80) map at the same H/T and refit both onto the same D26 r80 root and fitting pool with the D42 common-refit routine.

- **Feasible:** six matched pairs (H4/H8/H12 × 2018-02/2020-10) exist with `s_branch.pkl` and memberships.
- **What it would need:**
  - spatial IDs derived with the D32 `assignment_evidence()` function, since there is no `assignment_evidence.csv`;
  - a D26-root gate;
  - a small runner adaptation;
  - E3-only evaluation, because there is no C split;
  - new global+20 fits if that control is kept.

**Deferred by the supervisor, because the result would be hard to interpret:**
1. **Tie degeneration.** Both maps come from the hard-F1 E1 scan, whose boundary placement is largely set by tie order (D34: 77–83% of zero-mass areas cross the cut). Map differences may mostly be that noise.
2. **Asymmetric label reuse.** About 80% of D27's search rows (the latest 3 label months) lie inside the D26 refit pool, while D26's search rows (the random 20%) do not. D27 regions would get locals fitted on the labels that chose their boundaries.
3. **Dropping the tb3 search rows from both pools is rejected.** It would bring back the stale forecasting-pool disadvantage that common refitting is meant to remove.

## Candidate next question (discussion only; not spec authorization)

Generate a fresh Brier-E1 time-block map and compare it with the existing D34 Brier random-split map. Refit both onto the same fresh/current forecast root and the same current refit pool, and evaluate on E3 only.

- **Search-root age:** this is a stated property of the map-learning pipeline, not a forecasting-root difference.
- **Selection labels:** known at O, so they can legitimately be reused in the final refit. The overlap is not E3 leakage. But it means the effect cannot be attributed to validation chronology alone.
- **Before any run:** decide whether the information gain warrants the cost of new searches plus regional refits.
- **Out of scope:** no more disjoint-random, C-split, window or alpha grids.

**Unchanged:** Stage 1 overfitting is unresolved; no Stage 2/3, full 648, final evaluation or close.
