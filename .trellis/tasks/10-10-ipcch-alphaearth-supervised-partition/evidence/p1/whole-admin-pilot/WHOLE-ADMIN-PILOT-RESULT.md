# Whole-admin pilot result (release codex-D30-whole-admin-pilot-20261011) — 2026-10-11

**Run:**
- One run, 12:57:58Z → 13:02:08Z, in a fresh directory, using `/opt/earthengine/bin/python` with `timeout 600`.
- Pinned commit b6e6074; released manifest copied unchanged (file c05ee967…, body 65efd14b…).
- Calls 5 / 5; projection check 2.8e-9 m.
- `stopped` = "'WA-LARGE_q75_fresh-1050' failed (no automatic retry): … Read timed out. (read timeout=120.0)". Exit 0 is
  not success.

| admin | role | cells (Σ windows) | request seconds | s/cell | result |
|---|---|---|---|---|---|
| 658 | REF (completed) | 1.024e7 | 30.4 | 2.97e-6 | ok, finite, coverage 0.999917 (usable) |
| 4711 | MED | 3.152e7 | 98.0 | 3.11e-6 | ok, finite, coverage 0.999995 (usable) |
| 1050 | LARGE | 7.652e7 | 120.11 → read timeout | — | not completed; no retry; no request left unattempted after it |

Each request was one aggregation (response key `sums` only).

**REF against its 2 retained windows (raw, no renormalization):**
- valid area: whole 1,023,674,028.235294 m² against windows 1,023,674,028.235294 (relative 2.3e-16);
- S max relative difference 4.4e-14; Q max relative difference 1.7e-14;
- descriptors: |Δmean| ≤ 3.6e-15, |ΔSD| ≤ 2.5e-14; coverage identical;
- timing: windows 23.5 + 15.0 = 38.6 request-seconds (3 aggregations each, run at c4) against 30.4 s whole (1 aggregation).
  These are unmatched.

**Interpretation, measured on these admins only:**
- Whole-admin and window outputs agree to floating-point precision.
- The one-aggregation whole-admin graph ran at about 3e-6 s/cell. Fewer requests did not lower the per-pixel cost here.
- At that rate a single 120 s request fits about 4e7 cells: 3,542 admins (56.9%) but only 11.3% of cells (see
  `run-20261011/analysis.txt`; descriptive only, since the rate is unmeasured elsewhere).
- Exhaustive 10 m computation, about 4.67e11 cells, dominates the cost under either request shape.

HOLD: no bulk, retry or fit; the next step is a user discussion.
