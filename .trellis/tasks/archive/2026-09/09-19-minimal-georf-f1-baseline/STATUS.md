# Status and handoff — 2026-09-19

Implementation and local release verification are complete. No Git commit, push,
remote publication, real scientific rerun or IPCCH model fitting was performed.
User explicitly requested closure on 2026-09-19. Task archived as completed;
source package remains uncommitted. No commit or push is implied by closure.

## Deliverable
- Source: `GeoRFBaseline/`
- Local release: `GeoRFBaseline/releases/georf-baseline-v0.1.0.zip`
- SHA-256: `39a26138e3fafb0be2bbd22e9760095d6cdefa7d79b98b4f798cb3aa79b500a0`
- Archive bytes: 209831; 45 hashed payload files plus manifest.
- Parent commit: `2dfa121a9398de9a1918ba9c0af34b31ecbb117a`.
- All 37 original source hashes remain unchanged.

## Approved scientific changes
- Full class-1 F1 chain: regional additive counts including FP, q proposal statistics,
  parent/child checkpoint selection and strict >.01 gain gate at every depth.
- Gate evaluates identical complete validation rows of the current parent branch;
  F1 is not averaged across children and no significance claim is made.
- SMOTE disabled in Stage1 and Stage3; original pooled remains comparison arm.
- User explicitly retains Stage1's per-class zero-feature pseudo rows (two for binary target).
  These participate in training and can matter for small/single-class branches;
  effect size has not been measured. Stage3 adds no such rows.

## Required package integration fixes
- Monthly Stage1 runner supplies Stage2's existing artifact naming/layout.
- Coordinates read from experiment directory; boundaries selectable via GEORF_POLYGONS.
- NaN filtering also filters metadata; unsorted input is explicitly rejected.
- Unsplit root exports as root; leading-zero branch labels remain strings.
- Empty child proposals keep parent; missing standalone import corrected.
- No original source, historical outputs or previous release changed.

## Evidence
- `validation/release-verification.json` pins the exact verified archive hash.
- `validation/focused-tests.log`: 11 passed including non-root gate, exact boundary,
  real checkpoint persistence, pseudo-row count and noSMOTE in Stage3.
- Eight CLI/import smokes passed outside the repository.
- Full Stage2 synthetic chain passed with 60 areas and 3 output clusters.
- All package Python sources parsed; payload hashes verified after extraction.
- Two independent read-only reviews: mathematics had no blocker; package review
  found row-order precondition, now asserted and tested. No clean dependency install
  or real-data Stage1/3 end-to-end run claimed.

## Next work: IPCCH spec and task
Use this archive as the corrected source baseline. Do not substitute the old
published result bundle or the ETH selective-correction expert definition.
Read the previous archived handoff
`.trellis/tasks/archive/2026-09/09-19-onset-partition-redesign/STATUS.md`
and its PRD, plus `docs/notes/2026-09-19_suspension-window-briefing.md` and
`docs/notes/2026-09-18_benchmark_and_direction_review.md` for scientific conclusions.

Resolve IPCCH area keys and sparse calendar, target definition, forecast origin,
availability-aware history, temporal validation and class-support rules in the
next design discussion. Preserve the user's requirement to stop and grill on
material ambiguity. This release alone does not authorize IPCCH experiments.

## Spec sync and Git boundary
Contracts are captured in design.md and package README/tests. Root specs remain
unchanged because this task authorized edits only to the package and this task.
Working tree additions are limited to these two directories. No commit approval
was requested or exercised; source release is ready for review.
