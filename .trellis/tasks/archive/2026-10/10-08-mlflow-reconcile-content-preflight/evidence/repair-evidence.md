# Round-2 repair evidence (final, not audited — see user-no-audit-authorization.md)

| Finding (job 14cfd4ce) | Change | Regression | Against c22112d |
|---|---|---|---|
| repair-A01 (major): size-only preflight lets a same-size content change publish parent state before the late deep check refuses | Phase 1 compares SHA256 of every `source/…`/extras file and every `models.tar` member (plan) with the record's retained `manifests/include.json`, `extras.json`, `models.tar.members.json` (superseded copy on resume); refuses before any write. Small JSON downloads only. | `test_same_size_archived_content_change_refused_before_any_write`: same-length edit to an archived source file and to a bundle member; both refused with "content identity differs"; tags/metrics/params/artifact sizes identical; stored bytes unchanged; original plan import = 13 noop; verify ok | FAIL (refused only after download: "checksum mismatch after download"; store changed) |
| repair-A02 (minor): resume overwrites the parent reconciliation log with an empty change list | On resume the existing parent log is read and kept; only newly completed entries are appended; rewritten only if content differs | `test_interrupted_reconcile_resumes_same_plan_only` now asserts the parent log bytes after resume equal the first run's (with the plan-summary entry) | FAIL (log became `"changes": []`) |

Tests: 26/26 (tests-26.log). New tests against c22112d: new-tests-against-c22112d.log.
No live-store reconcile/import, no source reruns; importer version and plan fingerprints
unchanged, so the live 126 records need no change.
