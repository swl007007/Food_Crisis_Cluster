# D50: zero-fit exact-origin-phase ranking diagnostic (2026-10-02)

**Scope:** contract `d50-origin-phase-ranking-plan.md`, planning commit `7cd246d`. The primary crisis-F1 endpoint and the final criterion are unchanged.
- **Inputs:** the 21 saved D38 `rows_E3.csv.gz` files, hash-bound to D39, D40 and D49; exact-origin-known keys only (112,795 rows; 713 excluded).
- **Arms:** the original root, then the D38 anchored root, on the D49 score s = (p2 + p3)/Σp (float64).
- **Cells:** each root split by exact `persistence_code` 0 / 1 / 2 / 3 (origin phase 1 / 2 / 3 / 4-or-5). All cells reported, including one-class cells, with no outcome-dependent removal or sample cutoff.
- **Per cell:** n, P, N, within-cell crisis AUC (ties half credit; null with reason when P·N = 0) and fixed argmax confusions summed back to D49.
- **No fits,** thresholds, calibration, policies, pooling or adoption.

## Producer and run (executor factual record, `7c398c3`)

- **Producer** `386b25a`: task-research script `research/d50_origin_phase_ranking.py` (218 lines), git blob `5611ced859686ed0515837f83b641d1fe7dcc5d1`, sha256 `3eb010db6787d2d01c19113c1deaf32397860367ca7f34555642117fc5f8c3a2`. AUC via sklearn only; no package edits, no fit or model APIs.
- **Native implement** (about 1.2 min) and **native check** (about 1.1 min, no result-affecting finding). The check's two evidence-only edits were accepted by the supervisor: pre-output roots-per-H and one-target-month gates, and a `confusion()` selftest.
- **Run:** `C:\Users\swl00\geoxgb_runs\geoxgb-d50-origin-phase-ranking-20261002`, frozen Windows Python with assertions on (`sys.flags.optimize = 0` recorded by a launch of the same interpreter immediately before the script), exit 0. The selftest passed first; every gate passed; all cell confusions add back to D49.

## Verification (supervisor, distinct from the executor record)

`research/d50_supervisor_verify.py` → `research/d50_supervisor_verification.json`: **PASS**.
- 1,016 checks; 35,537,568 direct positive–negative comparisons; 84 root-phase cells and 168 arm cells; 113,508 source / 112,795 matched / 713 excluded rows.
- Exact integer pair numerators (wins 1, ties ½) saved; no sklearn or producer imports; no fits.
- Supports, null reasons, every cell confusion, the D49 add-backs, the by-date copies, the valid-only per-H means and the producer, input and output hashes all checked.

These are numerical checks, not statistical tests.

## Results (mean-fold within-cell crisis AUC over valid folds; original / anchored)

| H | Phase 1 | Phase 2 | Phase 3 | Phase 4-or-5 |
|---|---|---|---|---|
| 4 | .773484 / .779195 (7/7) | .655702 / .658357 (7/7) | .736468 / .739761 (7/7) | .803153 / .825947 (4/7) |
| 8 | .824208 / .826501 (7/7) | .598012 / .605486 (7/7) | .745844 / .762594 (7/7) | .938756 / .910750 (4/7) |
| 12 | .818068 / .818261 (7/7) | .605804 / .573847 (7/7) | .728402 / .730962 (7/7) | .854736 / .831863 (4/7) |

Parentheses give valid folds out of 7.

**Folds with AUC > .5 (original / anchored):**

| H | Phase 1 | Phase 2 | Phase 3 | Phase 4-or-5 |
|---|---|---|---|---|
| 4 | 6 / 6 | 7 / 7 | 7 / 7 | 4 / 4 of 4 valid |
| 8 | 7 / 7 | 4 / 4 | 7 / 7 | 4 / 4 of 4 valid |
| 12 | 7 / 7 | 7 / 6 | 7 / 7 | 4 / 4 of 4 valid |

**Supports and caveats:**
- **Phase 2 at H8** is below .5 in both arms on 2018-06, 2018-10 and 2019-10. Original per-date AUC ranges .414721–.837987 (P 235–586, N 911–1,506).
- **Phase 1:** valid on every date, but onset positives are few (P 4–159; as few as 4 in one H4 cell).
- **Phase 4-or-5:** defined on only 4/7 dates per H; null (`no_negative`) on the other 3 in both arms. Negative supports on valid dates are 1–38 (H4 6, 8, 1, 8; H8 31, 5, 6, 8; H12 38, 26, 2, 6). These AUCs are not robust.
- **Phase 3** (relief vs sustained crisis) is above .5 on 7/7 dates at every H (P 312–798, N 176–688).

## Supervisor synthesis

- **No policy or model is adopted; Stage 1 remains unresolved.**
- The descriptive cells refute "no within-state ordering anywhere": within exact phase 2 (onset) the original mean-fold AUC is .655702 / .598012 / .605804 with > .5 on 7/7, 4/7 and 7/7 dates, and within exact phase 3 (relief) .736468 / .745844 / .728402 with > .5 on 7/7 at every H.
- They do **not** identify causal feature contributions, prove transfer, justify routing by date, or solve overfitting. AUC above .5 is descriptive ranking beyond constant exact-phase information and can reflect other history, country or era features.
- **H8 onset ranking is unstable** across dates. A strictly increasing scalar calibration of the same fixed scores cannot repair that, because it preserves ranks; this is a statement about rank preservation, not an F1 impossibility.
- **Anchored phase 2 at H12** drops from .605804 to .573847 although the anchored arm had better aggregate crisis F1 earlier. That illustrates an aggregation/operating-point trade-off, not a uniform signal improvement.
- Phase 1 and phase 4-or-5 results are reported in full; small-support cells are not claimed robust.
- No D51 is specified or run. Conceptual next direction only: understand which within-state signal is stable and which is unstable before any further capacity or partition sweep.

## Evidence

**Task research holds exact byte copies** (line endings preserved):

| File | Bytes | sha256 |
|---|---|---|
| `d50_summary.json` | 123,304 | `6e296f83390429fac1692cd0e9756fc2e1c2534e8e64935ba1560e6320f6376e` |
| `d50_identity.json` | 2,818 | `3bf01888b050409491f42916bce9aaad0e049816ff512543ad001325c43fb9ac` |
| `d50_supervisor_verify.py` | 7,334 | `a8f1a800f7ebaf5103ce8e51e3ebd1e207beadd70928f084e802485faf79b45d` |
| `d50_supervisor_verification.json` | 46,294 | `535e82880628c331d0a2ce382484a88235c585a24960978013cd686974f7b01e` |
| `d50-run.log` | 153 | `debf22680afa18b4aebdca55beaa88e1a444ebbb47200bceef50cf7fc5bb246a` |

The run directory holds only `summary.json` and `identity.json`; nothing bulky remains external.
