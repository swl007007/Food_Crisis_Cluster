# Existing audit gate — 2026-10-08

Controller job938a82278edb625c94c67468, audited/completion2f6c304, verdict findings, severity major, gate_open1. Native prior task completed; no active run. Source: trellis-audit show/status read-only.

A01 correctness major: import_runs.py:653–665 writes parent manifests/fingerprint before child validation:670–687. A late child metric conflict leaves parent reconciling and ordinary old-plan retries fail. Repair validates all prospective changes before any write and verifies refusal leaves original usable; retained reconciliation recovery should remain bounded and explicit.

A02 minor: cmd_plan:237–243 omits inventory.reference_parents from selected-family dependency expansion. Actual window-only plan fails because P6 unplanned. Include read dependencies without adding unrequested writes.

Audit independently validated126 records,20,384 child scores and144 parent metrics against archived report source paths; no current data corruption found. Do not rerun scientific experiments or repair live data speculatively. Need dedicated repair child using supported --remediation-for, then controller accepted gate clearance before ordinary main feature start. This file is planning evidence, not an override or pass.

## Status update — 2026-10-08

Resolved by user waiver, not by audit pass. Round 1 repair c22112d (close-audit 14cfd4ce: findings, major — content preflight by size, parent log overwrite). Final round 659f72d under the user's no-audit override. Both gates waived (gate_open 0, resolved_by null); run 497d0907 closed manually without audit. See archive/2026-10/10-08-mlflow-reconcile-content-preflight/evidence/non-audit-closure.md.
