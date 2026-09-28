# Round-3 repair log

Findings: close-audit 140f13de (A01 major dynamic inventories, A02 minor verifier binding);
spot re-audits e16d2fc1 (A01 omitted fitted Stage 3 fold, A02 omitted retained terminal
checkpoint rf_00) and b414ad5d (A01 partial horizon inventory with all forests missing).
All five instances are one class: membership taken from the record under check.

- `src/utils/inventories.py`: membership derived from independent evidence, compared in
  both directions — Stage 3 fold records vs prepared schedule (months and statuses),
  predicted months vs fitted folds, prediction keys vs baseline truth keys; per fold,
  estimators and saved models vs local routes the saved predictions used, local_support
  clusters vs mapped clusters in the target month; retained Stage 1 checkpoints for every
  branch of the retained s_branch; Stage 1 fold population and Stage 2 ledger vs schedule.
- Enforced where artifacts are produced (Stage 3 refuses to publish a horizon that does not
  reconcile), where they are consumed (report reconciles again, requires fold records ==
  fold directories and a non-partial manifest; Stage 2 checks population and ledger;
  verify_fold checks every retained branch checkpoint).
- Verifier: `verifier_identity_at(rev)`; the verifier must equal its blob at the run's
  git_head and at HEAD; preparation refuses to start if the verifier differs from HEAD.
- Tests (43, clean clone): genuine inventories accept; omitted fitted fold (dropped or
  relabelled), omitted routed local model / model file / support row, omitted branch
  checkpoint, missing or extra Stage 1 fold, ledger mismatch, verifier mismatch refuse.
- Auditor fixtures: b414ad5d's /tmp/b414-audit/incomplete-stage3 (one skipped fold per
  horizon, all forests absent) is refused by the round-3 code ("fold records: missing
  2021-06 ..."); e16d2fc1's retained-terminal omission is refused on v7 (dropping any
  non-root branch checkpoint from fs2_2020-10 or fs3_2020-10).
- Evidence gap (task/start evidence for fourclass-audit-repair): exported read-only from the
  controller's durable state (runs table) as `controller_run_records.json`: task, base_sha at
  start, remediation_for, executor, close data for all four runs. Not reconstructed.
- Fresh run v7 from committed fe40de3 (code_equals_git_head true): tables identical to v2;
  verification 36/36 incl. committed-verifier check and bit-identical saved-model replay.
  Committed evidence moves from v6 to v7 (v6 stays in history at 55706e3).
