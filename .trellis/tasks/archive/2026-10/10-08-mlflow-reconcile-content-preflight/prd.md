# Reconcile content preflight and parent log preservation

## Goal / authority

Resolve the two findings of repair close-audit 14cfd4ce3da73a32c59993bc (which audited
c22112d and keeps gate 938a82278edb625c94c67468 open). The user chose "one more class fix"
on 2026-10-08 under their round-2 rule: if round 3 reports the same class again, stop
for good and hand back. No other scope.

## Bounded requirements

- R1 (repair-A01, major): the read-only phase of `reconcile` must compare the content
  identity of every archived item — each `source/…` and extras file and each
  `models.tar` member — between the plan and the record's own retained evidence (its
  `manifests/include.json`, `manifests/extras.json`, `manifests/models.tar.members.json`
  and bundle tags), and refuse with the whole family unchanged on any difference, before
  any tag, manifest or log is written. Fix the class: every archived-content comparison in
  phase 1 is by SHA256, not size. No multi-GB downloads.
- R2 (repair-A02, minor): a same-plan resume must keep the parent's reconciliation log
  entries (old/new hashes, kept_as) and add only newly completed entries, idempotently.
- Keep all other contracts; no live-store reconcile/import, no source reruns, no change
  to extraction or plan fingerprints.

## Acceptance

- Regression: same-size content change in an archived source file (and in a bundle member)
  is refused; all record tags/metrics/params/artifact bytes unchanged; original plan
  imports as a no-op and verifies.
- Regression: interrupted then resumed reconcile leaves the parent log with the original
  supersession entries (content asserted), plus the existing child-log assertions.
- Existing 25 tests pass; changed scope limited to import_runs.py, tests, spec/README note.
- Start with `trellis-audit start mlflow-reconcile-content-preflight --remediation-for
  14cfd4ce3da73a32c59993bc` from the bound session; close via controller. The parent
  feature stays blocked until the chain actually clears.
