# P6 release to the learn-map checkpoint

Supervisor release under the user's explicit bounded-review decision in
`user-review-boundary.md`. This is permission to execute the frozen experiment,
not scientific acceptance or a controller audit pass.

## Identity and evidence checked live

- HEAD `44951cc` contains only readiness updates after implementation commit
  `6798df21ea87d4916c7f36fa6e0753c32bb3ef98`.
- All 44 tracked package paths match the complete saved code inventory and
  implementation-commit Git blobs; package working tree matches that commit.
  The only untracked root item is unrelated `.birdview/`; leave it alone.
- Contract/schema/inputs/runtime-lock are byte-identical to accepted P0 freeze
  `f01ab3e`; source provenance is unchanged from reviewed `13e74b5`.
- Live runtime probe matches the lock: Windows Python3.12.10, XGBoost3.0.0,
  NumPy2.2.6, pandas2.2.3 and the pinned geospatial dependencies.
- Committed `residuals-evidence.md` describes the fixed six-item inventory;
  `evidence/residuals-pytest.log` records 291 passed, 1 deprecation warning.
  The 28 new focused tests passed; replay-only red check at91552d0 has25fails.
  No new full review round was initiated, per the user's decision.
- Foreground Claude remains PID1827, pane wN:p2,
  terminal term_65d044bdb69412, session d148c921-36bd-4b42-9ff9-a4f16979e6b5.
  Herdr, repository registration and active run agree; controller running.
- Run7ced754ea36c48c0a6d24ba2a17addec remains active, task in_progress, base
  6c98f73c34272101ddc124cf20ed5ef338563646 unchanged.

## Execute now

Use the pinned Windows runtime and frozen package/configuration. Run preflight,
fresh formal prepare, then learn-map for all four H and all eight recipes per H.
Use a new formal run ID, preserving the old dev prepare as superseded evidence.
No retuning, skipped candidate/fold, alternate environment or source edit during
this fitting sequence. Preserve logs, artifact hashes and code/run identity.

After learn-map, STOP at the agreed checkpoint and report:

- per-H winner/G/L, frozen map digests, learned areas, terminal-region count,
  accepted splits, connectivity component sizes/isolates and selection evidence;
- exact Stage3 scientific request enumeration and distinct fit-identity budget,
  separating global/local, H, current/historical and cache reuse; distinguish
  an upper bound for current locals that depend on gates from exact known fits;
- fit counts/failed requests, prepared identity, unchanged configuration and
  numerical environment, and any scientific/runtime limitation encountered.

Supervisor checks that checkpoint before predict -> report -> replay continues.
Any actual R41 data/fitting/recipe failure stops with durable partial evidence;
no silent fallback or changed-recipe retry. Later pure replay-verifier issues
are repaired against saved artifacts under the approved review boundary, not
used to restart an open-ended pre-P6 review loop. Do not close the task or claim
final acceptance until the full results/evidence and required lifecycle review.

Copy this release record into the active task evidence at the next suitable
executor commit; keep implementation code identity6798df2 explicit even if
evidence-only commits advance HEAD.
