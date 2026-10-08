# A01/A02 repair evidence

| Finding | Change | Regression | Against b7572b5 |
|---|---|---|---|
| A01 (major) late child conflict mutates parent | `reconcile_family` validates parent + all children before any write; explicit `reconcile_target` resume; `import` refuses a reconciling parent | `test_late_child_conflict_refused_before_any_write` (store snapshot identical, original plan import = 13 noop, verify ok); `test_interrupted_reconcile_resumes_same_plan_only` (different plan refused, same plan resumes 2 remaining children, run IDs kept, one log per child) | late-conflict test FAILS (store changed); resume test fails (no interruption point) |
| A02 (minor) `--family` misses inventory.reference_parents | `planning_order()` dependency closure; writes/verify limited to selection | `test_selected_family_plans_reference_parent_without_writing_it` (dependent listed first; plans parent, writes only fy; verify --family ok; unknown family refused) | ERROR `fy: reference parent fx not planned` (same as audit) |

Logs: tests-25.log (25/25 OK), new-tests-against-b7572b5.log, window-only-plan.log +
window-only-plan-summary.json (real config, scratch store, rehash 7,036 files, records 17,
read dependency p6_geoxgb, fingerprints equal the accepted plan), live-store-readonly-check.json
(126 runs complete, 20,528 metrics, parent fingerprints equal; no writes).

Not done (out of scope): no real-store reconcile, no import into the live store, no source
reruns. Importer version and plan fingerprints unchanged, so the live records need no change.
