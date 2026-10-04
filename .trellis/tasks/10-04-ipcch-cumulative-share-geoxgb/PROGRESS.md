# IPCCH GeoXGB execution progress

Authority: `prd.md`, `design.md`, `implement.md` v1.0. This is operational state,
not approval. Updated 2026-10-04.

Branch: `ipcch-cumulative-share-geoxgb`; frozen baseline `6c98f73c34272101ddc124cf20ed5ef338563646`.
Scope: `IPCCHGeoXGBExperiment/` and this task's documentation/evidence.

| Phase | Status | Evidence / next action |
|---|---|---|
| Science/spec/grill | component-complete | R1–R51; source-grounded clarifications in implement.md |
| Baseline/context/manifests | complete | Committed `6c98f73c` |
| Audit binding/start | complete | Run `7ced754ea36c48c0a6d24ba2a17addec` active, base_sha `6c98f73c…`, executor session `afc97c82…`/`term_65d044bdb69412`; first attempt refused by in_progress 10-03, user-authorized native archive (housekeeping commit `8160f68`); see audit-start-evidence.md |
| P0 infrastructure | in progress | Claude only; no fit; commit P0 evidence |
| Foundation freeze | pending | Codex reviews actual P0 artifacts/checks and releases same Claude |
| P1 data/features | pending | Depends on foundation freeze |
| P2 quartet/metrics | pending | Depends on foundation freeze and data contracts |
| P3 Stage1 | pending | Depends on P1/P2 |
| P4 Stage3 | pending | Depends on P1/P2/P3 |
| P5 report/replay | pending | Depends on saved P3/P4 evidence |
| P6 formal run/delivery | pending | Depends on implementation tests and frozen code |
| Controller close audit | pending | Only after complete committed delivery |

No new package, model fitting, prediction or bootstrap has run at baseline
preparation. The planning byte-hash observations are not geometry structural
validation or performance evidence. No audit pass is claimed.

Resume: reconcile this ledger with Git, the three authority documents, the live
Claude identity and audit status. Preserve the unrelated pooled-onset task.
Next: commit baseline, hand off P0 to the verified Claude, verify real audit start.
