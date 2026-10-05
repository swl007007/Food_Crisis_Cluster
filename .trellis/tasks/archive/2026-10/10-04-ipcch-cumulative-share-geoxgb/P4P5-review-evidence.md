# P4/P5 supervisor review corrections (review pinned at 13e74b5; fixes after 3597834)

Source: `/tmp/ipcch-geoxgb-p4p5-supervisor-review-20261004.md`. Classification:
every item is an identity, evidence, reporting or verification-strength gap; no
item changes a number produced by the frozen recipe; no recipe, scope or
config change. This is the first review round on P4/P5.

## P4

1. **Frozen-map binding** — freeze records now carry `H`, the prepared-manifest
   digest, contract/schema versions and the selection-ledger digest;
   `predict.load_frozen(stage1, H, prepared_sha)` rejects a record for another
   H, other prepared data, a record that differs from the completed Stage1
   summary or its selection winner/digest, a candidate id not equal to G+L, map
   bytes not equal to the record, and duplicate areas (also refused at freeze).
2. **Keyed historical quartets** — gate pairs save global and local-routed
   q_raw/q_star for all four targets plus `local_routed_provider` (on a
   support-fallback date the routed provider is that date's global) and the
   target month; replay derives phases and gate counts from them.
3. **All-empty H** — fixed prediction column contract; an H whose scheduled
   folds are all empty writes its ledger (`no_valid_target`, n = 0), a
   header-only predictions file and an empty model-request ledger (the store now
   creates its ledger at start, so zero requests are durable evidence).

## P5 report

4. Every H × period has both cohorts; empty cohorts are explicit
   (`status: empty_cohort`, `n: 0`, NA reason); panels/deltas unchanged for
   scored cohorts. Country and month diagnostics carry geo/pool and
   geo/persistence crisis F1 points, paired deltas, NA reasons and persistence
   coverage; monthly tables include every scheduled month, with
   `no_valid_target` rows at n = 0. No diagnostic intervals or new metrics.
5. `routes` coverage with explicit denominators: learned-map size, accepted
   split, terminal regions; cohort areas in-map / unmapped / ever local-routed;
   cohort rows local / global_fallback (by reason) / global_only_no_accepted_split
   / unmapped_area_global; gate region×fold decisions, enabled, adopted, regions
   ever adopted.

## Independent replay (rewritten)

Authority: frozen contract, `verify_prepared` (exact inventory + digests) and
the actual boosters; never a stage's own summary/report. Scientific
recomputation remains independent (own PAVA, counts, four-class, R² rules,
support, gates, selection, bootstrap); artifact loading reuses only the store's
byte/fit-record integrity primitives. Saved CSVs are read with round-trip float
parsing so lineage is exact.

6. Models/input lineage: every referenced quartet is loaded with byte and
   fit-record validation; Stage1 provider predictions, current pooled
   predictions, adopted local predictions and both sides of every historical
   pair are re-predicted from the referenced boosters on prepared X and must be
   bit-identical; Stage3 identities must name the current prepared X/keys
   digests; fitting rows, member sets and per-target y digests are rebuilt.
7. Metrics/bootstrap: every R8 metric, NA policy, delta, count and coverage is
   recomputed from the keyed cohorts; bootstrap countries, counts, 2000/42
   multiplicities, per-arm F1, point delta, eligibility, interval and saved
   draws are recomputed from predictions, not from the report.
8. Inventory/schedule: stage1/stage3/report presence, contract horizons, the
   8 candidates per H, the full fold ledger equal to the prepared calendar,
   empty-fold semantics, scored-fold key sets, and the report's periods,
   cohorts and bootstrap entries are required; missing evidence fails.
9. Gate/provider completeness: exactly one decision per frozen region per scored
   fold; full historical pair keys per gate date and region; recomputed local
   support per date and current support; local providers must exist with the
   right scope, origin, region and same-fold global; local rows only where the
   gate adopted them.

## Evidence

- `evidence/P4P5-review-pytest.log`: **261 passed**.
- Fresh-run tamper classes, each failing at its specific check: missing boosters,
  changed fitting X, wrong full metrics (accuracy, R², delta), fabricated
  bootstrap counts/draws/CI, missing report, omitted scored fold (predictions,
  gates, pairs, ledger, refreshed digest), removed gates with a fake local
  provider; plus the earlier dropped row, q2/q5 swap, persistence after O, stale
  gate, re-parented prefix, report count and per-target digest swap.
- Genuine adopted-local E2E (`west_areas=56`): current local quartets adopted,
  every adopted row's raw quartet differs from pooled, gate records
  enabled/local, report route counts match, replay zero failures; nudging one
  adopted prediction is caught by `local_lineage`.
- Frozen-map tests: another-H record, other prepared data, record ≠ summary,
  duplicate area — all rejected. All-empty-H run: ledger kept, zero requests,
  empty predictions, report and replay pass.
- Supervisor's 13e74b5 tamper directories (`evidence/P4P5-review-supervisor-dirs.log`):
  all eight, including their baseline, now fail — but at the first, stricter
  `prepared.inventory_and_digests` check, because their fixture predates the
  inventory requirement. They therefore do not exercise the deeper checks;
  the fresh-run tamper tests above are the class-specific evidence.
