# Repair four-class audit findings A01/A02

Remediation for close-audit job `fa38f19ad8fafc36567f90ac` of task
`fewsnet-four-class-perturbation` (archived at
`.trellis/tasks/archive/2026-09/09-28-fewsnet-four-class-perturbation/`, completion SHA
ecd2b70). The approved scientific specification of that task (prd.md R1-R15, A1-A9,
design.md, feature-contract.md, feature-schema.json) is unchanged and binding. No
feature, gate, model, cohort or reporting choice may change.

## Findings being repaired

- **A01 (major, reproducibility).** Stage 3 discards every fitted pooled/local RF; the
  replay verifier refits instead of loading original models. design.md "Reporting and
  reproducibility" requires run-local fitted models/transforms and replay of
  representative saved models.
- **A02 (major, data integrity).** `run_stage1.py` skips a fold when `candidate.json`
  exists, without verifying input/code/runtime identity or the complete output
  inventory (including requested retained checkpoints); `candidate.json` is written
  before other outputs. design.md requires exact-input continuation to verify complete
  identity and outputs.

## Requirements

- R1 (A01). Every Stage 3 estimator actually used for predictions (pooled and each
  local RF, per fold and horizon) is serialized as one bundle: forest, its fitted
  imputer, ordered feature names, and fit identity (fold, role, cluster, training-key
  digest, class counts, params). Bundles are hash-recorded in the fold record. The
  null-consensus route stores only the pooled bundle.
- R2 (A01). Replay loads the saved bundles (no refit) for the first and last fitted
  fold per horizon and reproduces labels and all four probabilities exactly; Stage 1
  replay (retained checkpoints) continues for first/last fold per scope.
- R3 (A01). Fitted models are auditor-accessible: committed (Git LFS if configured,
  otherwise plain git objects) or published at a hash-bound location referenced from
  the run. Size is disclosed; if all Stage 3 bundles are too large to commit, commit
  at least the replayed folds' bundles and hash every bundle in the committed records.
- R4 (A02). A fold may be skipped only when its completion record binds identities
  (snapshot, schema, geometry, package code, runtime) equal to the current ones and
  every required output exists and matches recorded hashes, including retained
  checkpoints when requested. The completion record is written last, atomically.
  Otherwise the run stops (no silent refit/mix). Same rule for Stage 3 folds if any
  continuation exists; simplest acceptable alternative: refuse any existing fold.
- R5. Focused regression tests: empty/forged completion marker, changed input identity,
  missing output, missing retained checkpoint all refuse; Stage 3 bundle round-trip
  reproduces probabilities.
- R6. Produce a fresh authoritative run; verify it reproduces the reported numbers of
  `fourclass-v2-20260928` exactly (same scientific result) and passes the extended
  verifier. Update the original task's RESULTS.md pointer and evidence index.

## Acceptance

- A1: findings A01 and A02 each have code, tests and run evidence closing the whole
  class (all estimators persisted; all continuation paths verified).
- A2: fresh run metrics/contrasts identical to v2; verifier passes including
  saved-model Stage 3 replay.
- A3: lifecycle via `trellis-audit start --remediation-for fa38f19ad8fafc36567f90ac`,
  commit, wrapper close; accepted re-audit required before claiming the gate cleared.
