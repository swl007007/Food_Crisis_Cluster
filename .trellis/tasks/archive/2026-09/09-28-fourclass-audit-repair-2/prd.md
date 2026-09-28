# Repair round 2 — four-class audit repair findings

Remediates close-audit `026f8908081e8266ea640366` of task `fourclass-audit-repair`
(archived `.trellis/tasks/archive/2026-09/09-28-fourclass-audit-repair/`), and, once
available, any findings of spot re-audit `175302c2b7958c4f943d74fb` of the original
task. The original task's approved specification stays binding: no feature, gate, model,
cohort or reporting choice changes. The round-1 repair PRD's R1-R6 remain the target.

## Findings and the class each one names

- A01 (major). Run bound to different code than the audited commit (verifier edited
  after the run). Class: a run must be produced by the exact committed code.
- A02 (major). Local model bundles carry the pooled training-key digest. Class: every
  recorded fit identity must describe that estimator's own fitting rows.
- A03 (major). `FEWSNETFourClassBaseline/feature-schema.json` not committed (root
  `*.json` ignore). Class: every file the package reads must be in a clean checkout.
- A04 (major). Stage 2's fold check accepts an identity-matching record with an empty
  inventory. Class: every completion record at every stage must prove its REQUIRED
  outputs (not only the ones it lists) and its own identity.
- A05 (minor). Replay tolerance instead of exact probabilities. Class: predictions must
  be deterministic so replay can be exact.
- Evidence gap: task records (`task.json`) absent from git.

## Requirements

- R1. Required-output inventories per stage (preparation, Stage 1 fold, Stage 2,
  Stage 3 fold and horizon), checked on acceptance together with the record's own
  identity fields (fold name, horizon, month); empty or incomplete inventories refuse.
- R2. Every Stage 3 bundle records the SHA-256 of its own ordered fitting keys; the
  verifier recomputes local digests from training keys and the cluster map.
- R3. Prediction probabilities are computed single-threaded (fixed tree summation
  order) in Stage 1 and Stage 3; replay requires bit-identical probabilities.
- R4. Schema and task records are committed; a clean clone runs the tests and the
  identity check.
- R5. Code (including the verifier) is committed before the authoritative run; the run
  records that code identity; the verifier checks run identity equals current code and
  that the working tree equals HEAD for package code.
- R6. Fresh run; reported tables identical to earlier runs unless deterministic
  prediction changes a label (disclose any such change); verification all pass.

## Acceptance

Regression tests for each class; clean-clone check; fresh run bound to the committed
code; re-audit accepted before claiming any gate cleared.
