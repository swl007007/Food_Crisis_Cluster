# Existing audit gate — 2026-10-08

Controller job938a82278edb625c94c67468, audited/completion2f6c304, verdict findings, severity major, gate_open1. Native prior task completed; no active run. Source: trellis-audit show/status read-only.

A01 correctness major: import_runs.py:653–665 writes parent manifests/fingerprint before child validation:670–687. A late child metric conflict leaves parent reconciling and ordinary old-plan retries fail. Repair validates all prospective changes before any write and verifies refusal leaves original usable; retained reconciliation recovery should remain bounded and explicit.

A02 minor: cmd_plan:237–243 omits inventory.reference_parents from selected-family dependency expansion. Actual window-only plan fails because P6 unplanned. Include read dependencies without adding unrequested writes.

Audit independently validated126 records,20,384 child scores and144 parent metrics against archived report source paths; no current data corruption found. Do not rerun scientific experiments or repair live data speculatively. Need dedicated repair child using supported --remediation-for, then controller accepted gate clearance before ordinary main feature start. This file is planning evidence, not an override or pass.
