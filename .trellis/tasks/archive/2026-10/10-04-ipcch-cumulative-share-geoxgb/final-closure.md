# Final closure: ipcch-cumulative-share-geoxgb

## Status

The task is **closed by user waiver of the close and spot audits**. This is
not an audit pass: the controller recorded a waiver, not an independent audit
result.

The user's instruction, as relayed by the supervisor (Codex, wN:p1) on
2026-10-04:

> 跳过close audit和spot audit。supervisor-executor模式一般不需要close audit。

("Skip the close audit and the spot audit. Supervisor-executor mode generally
does not need a close audit.")

Supervisor action: reviewer wQ:p1 was interrupted. The task is durably waived in
the trellis-audit controller, with the pinned evidence preserved and the
waiver marked. The executor did not restart or retry an audit and did not
change controller state.

Durable waiver confirmation (supervisor, 2026-10-04):
- Job 8c464aace2bb937f88d42f45 has status `waived`, gate_open=0, result=null.
- Run 7ced754e remains closed, and the pinned SHAs are unchanged.
- The controller backup/incident record is
  `~/.local/state/trellis-audit-controller/19ea69ef13d88fde/user-waivers/20261004-ipcch-geoxgb-013039/waiver.json`
  (sha256 6ad599ca17be3b462902aa413da1fa885c80fdc5067803167f055cc05428f12c).
- The record's scope is "this task and audits sampled by its completion; no
  unrelated jobs changed".
- Its reason text quotes the instruction above and states that this is "a user
  waiver, not an audit pass. Do not retry or sample this task."

## Audit lifecycle record

| Item | Value |
|---|---|
| Audit run | 7ced754ea36c48c0a6d24ba2a17addec |
| base_sha | 6c98f73c34272101ddc124cf20ed5ef338563646 |
| Executor | Claude, session d148c921-36bd-4b42-9ff9-a4f16979e6b5, terminal term_65d044bdb69412, pane wN:p2 |
| Close job | 8c464aace2bb937f88d42f45 (close-audit, attempt 1), queued from the bound session; the controller started running it |
| Pinned audited/completion sha | d30617f4efc3eb253b7a7d81a466200569c7e20d |
| Snapshot sha256 | 305c5295893c9b2826f99982acb0eb0fc62362675218a7f96ac55cc187ff478a |
| Outcome | Waived by user instruction; no audit pass or finding is claimed |

## What remains accepted

- **Supervisor acceptance of P6 execution and results** (`p6-supervisor-acceptance.md`):
  - All scientific and numerical checks passed. That covers 2912 metric
    comparisons, the cohort and persistence reconstruction, eight bootstrap
    contrasts × 2000 draws, all 4900 UBJ hashes over 1225 model records, and
    the request/fit reconciliation.
  - The only open item was the final-inventory bookkeeping; it was completed in
    1f2ebff/d30617f.
- **Implementation identity:** 6798df21ea87d4916c7f36fa6e0753c32bb3ef98, package
  tree 3ed52ea85555acabc118ef4f66a6710eb445a041, unchanged through P6.
- **Scientific run:** `p6-formal-20261004b`.
  - report sha256 142d717dedaf8afb573b544743cc6856c0d33d18e1b79a82543bb2dc514691f0;
  - replay sha256 1195716881b8509c53efd47b1d22920b6cb5c6d3952185f9ce316953682e30e0
    (91 880 checks, 0 failures);
  - final post-replay inventory CSV sha256
    7c96007f7311604200523cbf9e03e65576c5ebb92c5afb98f1e955257aa2b6d0.
- **Failed run `p6-formal-20261004`:** preserved as an R41 file-lock stop
  (`p6-restart-evidence.md`).

## Scientific reading (from P6-results.md / supervisor acceptance)

- **Versus pooled.** The spatial layer shows no demonstrated benefit over the
  matched pooled model. Main-period GeoXGB minus pooled crisis F1 is
  −.000322/−.000298/+.000090/−.000270 for H1/3/6/12, and every CI includes 0.
  Local routing covers only 3.39/3.43/3.10/0.79 % of main rows.
- **Versus persistence.** Paired GeoXGB minus persistence crisis F1 is
  +.005026/+.007434/+.002674/+.007521, and every CI includes 0. Recall, F2 and
  continuous q3 R² improve; precision, binary accuracy and four-class macro F1
  fall. Do not describe this as general superiority.
- **2026.** The 2026 supplement is a separate set of point estimates.
  No retuning was performed or is authorized.

## Archive

The controller's native close moved the task directory from
`.trellis/tasks/10-04-ipcch-cumulative-share-geoxgb/` to
`.trellis/tasks/archive/2026-10/10-04-ipcch-cumulative-share-geoxgb/`. All 113
committed files are byte-identical after the move except `task.json`, where the
controller set `status: completed` and `completedAt: 2026-10-04`. The
archive commit records exactly that move plus this note. It force-adds the 32
`*.csv`/`*.json` evidence files that `.gitignore` would otherwise drop. There
is no fit, no scientific change and no code change.
