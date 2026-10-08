# Repair MLflow reconciliation refusal and family dependencies

## Goal / authority

Resolve controller findings A01/A02 from accepted job938a82278edb625c94c67468 so the approved parent task can proceed. User approved this dedicated repair prerequisite in the final parent plan on2026-10-08 (“批准”). No new scope decision is delegated to this child.

## Bounded requirements

- A01 (major correctness): validate proposed parent AND every child change before any record mutation. A late-child metrics/params/tag conflict must leave all existing records/manifests/fingerprints unchanged and the original plan usable. If reconciliation is interrupted after validation, preserve previous evidence and resume same plan or stop with explicit recoverable state; do not introduce transaction/journal frameworks or manual DB editing.
- A02 (minor): include declared inventory.reference_parents and existing shared-input dependencies in selected-family planning closure. Dependency planning must not expand requested writes/verification targets. Actual window-only configuration depends on P6 even without shared_inputs.
- Retain existing contracts and source scientific semantics; no production import/reconcile, source artifacts edits, fit, model load, full download or live DB mutation required for these code repairs.

## Acceptance

- Isolated regression reproduces late-child conflict after valid earlier records; rejection leaves parent/children/artifacts unchanged and original-plan retry succeeds.
- Small interrupted-reconciliation test covers any newly changed recovery branch; existing successful additive reconciliation retains run IDs/superseded history.
- Reference-parent-only fixture plans required P6 but selects writes only for window; actual configured window-only no-write planning succeeds with scratch plan/cache.
- Existing22 tests plus focused regressions pass, changed code scope verified; findings/evidence/commit recorded.
- Dedicated controller remediation run retains actual Claude identity and original gate linkage. Close through supported wrapper; parent feature remains blocked until accepted controller repair clears the gate.

## Dependency

Parent `.trellis/tasks/10-08-mlflow-focused-evaluation-registry` owns Summary/Models/Inputs implementation. This child owns only importer refusal/recovery and planning-dependency code, focused tests, and directly relevant docs/spec notes. Parent may start only after this child is completed and the controller gate is resolved. Do not reset controller state or infer pass from queued audit.
