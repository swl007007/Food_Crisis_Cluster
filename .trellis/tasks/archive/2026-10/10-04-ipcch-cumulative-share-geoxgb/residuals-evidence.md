# Residual list at 91552d0 — corrections (final bounded pass before P6)

Source: `/tmp/ipcch-geoxgb-review-residuals-91552d0.md`. Boundary: the user chose
"fix list, then P6" (`user-review-boundary.md`); later pure replay/evidence
issues are repaired on saved artifacts without refitting, no further whole-task
review loop. No finding changes a number from the frozen recipe; configs fixed.

1. **Cohort metadata** (replay): per scored fold, country_key, period,
   horizon, target_ord/target_month and every persistence field used in scoring
   (available, phase, source month, age, q3 with NaN equality) must equal the
   prepared keys/calendar before any report or bootstrap reconstruction.
2. **Report evidence** (replay): each scored cohort arm must carry the full
   schema (binary metrics + counts + NA reasons; four-class accuracy, macro-F1,
   per-class support/predicted/tp/fp/fn/f1, confusion matrix, NA reasons; n;
   q3 R² projected/raw with NA reasons) and every item is compared with the
   keyed recomputation; the delta dict must contain exactly the nine metrics;
   bootstrap saved country counts, defined/undefined counts, NA reason,
   per-draw f1 of both arms, draw ids, multiplicities, Δ, point, eligibility
   and interval are compared with the independent reconstruction; the four
   mandatory diagnostic files per H are required and recomputed row by row
   (keys, persistence keys/coverage, both F1 pairs, deltas, NA-reason presence;
   all scheduled months present).
3. **Pairs and gates** (replay): all pair raw/star/truth/provider columns are
   required; saved stars equal the independent projection; pair truth equals
   prepared truth and internal origin equals U − H; the region gate replay now
   returns the full decision record, and keys, areas, target months,
   crisis/noncrisis keys, successful local dates, global/local confusion
   counts, both F1s, reason, areas-in-map, test-keys and current-fit support
   are each compared with the saved decision.
4. **Fit identity** (replay): every model record's identity must hash to its
   recorded digest and directory (otherwise the entry is unusable and the
   check fails); Stage3 requests also rebuild ordered fit_keys; a new Stage1
   request replay reconstructs every root and child fit (rejected children
   included) from its saved prepared-row references — root rows equal the F
   rows, child rows equal the F rows of its members, member digest, same-H/G
   root link and fit_keys/X/Y/n_rows/y digests.
5. **Frozen binding** (production): `load_frozen` also requires the record's
   contract version and schema to equal the current ones, and the Stage1
   summary's base identity schema/prepared digest to match — a consistent but
   stale record+summary pair is rejected.
6. **Interface/context** (production): route coverage adds global-routed,
   fallback, no-split and unmapped AREA counts and REGION counts with
   denominators; diagnostic tables keep fixed headers when empty; Stage3 local
   prediction calls carry region/gate-month/provider notes on failure.

Evidence:
- `evidence/residuals-pytest.log` — full suite on the pinned runtime: **291 passed**.
- `tests/test_review_residuals.py` (28 tests): country/period rewrites with a
  regenerated report, persistence q3 rewrite, removed deltas, zeroed confusion,
  bootstrap country counts / defined counts / NA reason / per-draw arm scores,
  missing or wrong diagnostics, pair stars removed or .99, pair truth = 1,
  eight gate-record field tampers, stage1-global and stage3-global fit_keys
  tampers, Stage1 child row-reference tamper, clean Stage1 reconstruction,
  route area/region coverage, empty diagnostic headers, local prediction
  failure notes. Clean symmetric and adopted-local runs replay with 0 failures
  (2,876 and 3,084 checks).
- Red check (`evidence/residuals-red-91552d0-replay.log`): with only replay.py
  reverted to 91552d0, 25 of the 28 fail and the 3 production-side interface
  tests pass. The three cohort-rewrite cases fail there through a lineage
  check (the test helper re-serializes floats) rather than a metadata check;
  the supervisor's own probe established that 91552d0 accepted those rewrites.
