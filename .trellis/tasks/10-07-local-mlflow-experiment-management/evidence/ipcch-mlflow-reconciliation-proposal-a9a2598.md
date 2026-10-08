# MLflow one-time reconciliation proposal (fixes at a9a2598)

Not applied. Read-only diff of live records vs re-plan: task evidence/reconciliation-diff-before.json.
Old plans kept: ~/.local/share/ipcch-mlflow/plans-8c88f48/ and evidence/plan-8c88f48/plan-summary.json.

## Current state
| Record | Run ID | Status | Old fp -> new fp | Difference vs fixed plan |
|---|---|---|---|---|
| P6 parent | 8ad47dc9cde2438d98b7bd50903dfe6a | complete | d5da27a7 -> def85c10 | missing manifests/original-inventory-check.json; plan-summary.json content (fp/config) |
| P6 children x12 | - | complete | unchanged | none (same fingerprint) |
| MLP parent | 5ba961810a214cbfa0138eb1dd1cf3c9 | complete | 4cb760b7 -> d7950970 | same as P6 parent |
| MLP children x24 (local+pool, 3 seeds, 4 H) | - | complete | changed | +6 tags each: cohort_keys.{main,supplementary}.local_eligible.{adopted,gain_rejected,historical_support_rejected}; view/evaluation_view.json tags block. 0 metric/param/existing-tag changes |
| MLP children x28 | - | complete | unchanged | none |
| Yearly parent shell | c19cd67502184f49a8d7fb800365f56b | in_progress, empty (0 metrics/params/artifacts) | 1d5474eb -> 8b9ca0b0 | everything (never populated) |

No metric value, param, source artifact or models.tar changes for any imported record.

## Proposed one-time action (new `reconcile` command, import lock held, logged)
Only additive or superseding writes; nothing deleted; no record recreated.
1. P6 and MLP parents: set import_status=reconciling; copy current manifests/plan-summary.json to
   manifests/superseded/plan-summary.<oldfp16>.json; upload new plan-summary.json and
   original-inventory-check.json; upload manifests/superseded/reconciliation-<oldfp16>.json
   (old/new fp, actions, superseded file sha256); tags import_fingerprint.previous=<old>,
   import_fingerprint=<new>, reconciliation="supervisor review of 8c88f48; fixes a9a2598".
2. 24 MLP children: same pattern, copy view/evaluation_view.json to view/superseded/evaluation_view.<oldfp16>.json,
   upload new view JSON, add the 6 tags, record previous fingerprint.
3. Each reconciled record: deep readback (download + hash every artifact, all metrics/params/tags)
   before import_status=complete. Refuse (stop) if any metric/param/existing-tag value would change.
4. Yearly shell: tag import_fingerprint.previous=<old>, set the new fingerprint, then the normal
   import resumes it (same run ID).
5. verify accepts extra artifacts only under manifests/superseded/ or view/superseded/, and only
   when import_fingerprint.previous is set.
Then continue the approved import (yearly, climate, window, split), full deep verify, verified
no-op rerun, Windows download, backup + scratch-server restore. No push/merge/close.
