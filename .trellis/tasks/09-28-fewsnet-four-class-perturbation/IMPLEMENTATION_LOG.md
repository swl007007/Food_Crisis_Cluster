# Implementation log

Chronological record of execution findings and deviations. The approved spec
(prd.md, design.md, feature-contract.md) is not changed by this log.

## 2026-09-28

- Base SHA 8a56272; audit run 321911af42cc46809237d7a003700ed9 started by the
  bound executor (pane wF:p2). Package `FEWSNETFourClassBaseline/` extracted from
  the hash-verified release ZIP (45/45 payload files matched before any edit).
- Source preflight passed on the four pinned hashes. Findings recorded in
  `prepared/manifests/preflight.json`: complete 5718 x 180 scaffold; 259,440 valid
  labels, all integral 1..5, binary label equals phase>=3 on every row; panel and
  FEWSNET.csv agree on every shared truth/near/medium key (FEWSNET.csv adds only
  2009-07/2009-10 truth, before the panel); 30,780 +/-inf cells each in
  Tair_zscore and Rainf_zscore, converted to NaN before imputation exactly as the
  release's comp_impute does; no static-invariance conflict.
- The panel's `date` column is `YYYY-MM`; parsed with that exact format.
- Schedule (from the pinned label calendar): Stage 1 has 27 supported folds
  (Feb/Jun/Oct 2018-2020 x 3 scopes) and 81 empty target months, recorded not
  fitted. Stage 3 has 30 supported folds; first supported targets 2021-06,
  2021-10, 2022-02 as the PRD anticipated.
- Stage 1 fold scratch (bulky checkpoints) runs in the OS temp directory rather
  than inside the Dropbox-synced tree, after Dropbox held a lock on a fresh
  checkpoint directory (WinError 32). Retained evidence is copied into the run.
- Run `fourclass-v1-20260928` aborted at Stage 3 h8 on the unmapped-coverage gate.
  Cause: the first implementation added a second gate on the evaluated-target
  share (2.07% at h8) that the release does not have. The release gate
  (`create_partition_group_array`) is the unmapped share of all labelled panel
  rows. Restored that definition; the evaluated-target share is disclosed in the
  manifest, not gated. No model, feature, split or consensus setting changed.
  The aborted run is superseded; a fresh run directory is used.
- Independent read-only review (trellis-check agent): no blocker or major finding.
  Disclosed an inherited release behaviour: scan candidates come from validation
  groups only, so after an accepted split an area without validation rows keeps
  the parent branch and routes to the parent checkpoint. Routing and
  correspondence stay consistent (both derive from s_branch); each Stage 1
  `candidate.json` now counts these areas/rows under `partition.parent_routed_areas`.
  `hist_*_absent = 1` for areas with no history follows the contract text literally;
  `hist_no_history` separates the two cases.
- Authoritative run `fourclass-v2-20260928` (fresh directory, code as committed)
  completed all stages; verification 34/34 including exact replays. Results and the
  acceptance index are in RESULTS.md. No feature, gate, model or cohort choice was
  changed after any Stage 3 or report number was seen; the only post-v1 edit was the
  coverage-gate restoration above plus disclosure counters.
- Evidence committed per the package `.gitignore`: manifests, ledgers, Stage 1
  candidate records/correspondence/predictions/membership, Stage 2 ledger, weights,
  consensus map and reports, Stage 3 fold records/keys/imputer statistics/predictions,
  report tables and bootstrap draws, verification and replay outputs. Kept local only
  (reproducible or bulky): snapshots (hashes recorded), retained replay checkpoints,
  similarity matrices, per-fold GeoRF print logs, v1 and dev runs.
