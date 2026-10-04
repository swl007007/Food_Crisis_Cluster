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
| P0 infrastructure | **awaiting_supervisor_foundation_freeze** | Committed by Claude (commit subject "P0 foundation: independent IPCCHGeoXGBExperiment package"); 64 tests pass, preflight passed; see P0-evidence.md and evidence/ |
| Foundation freeze | pending | Codex reviews actual P0 artifacts/checks and releases same Claude |
| P1 data/features | pending | Depends on foundation freeze |
| P2 quartet/metrics | pending | Depends on foundation freeze and data contracts |
| P3 Stage1 | pending | Depends on P1/P2 |
| P4 Stage3 | pending | Depends on P1/P2/P3 |
| P5 report/replay | pending | Depends on saved P3/P4 evidence |
| P6 formal run/delivery | pending | Depends on implementation tests and frozen code |
| Controller close audit | pending | Only after complete committed delivery |

P0 built the independent package foundation (configs, runtime probe, read-only
preflight, explicit not-implemented scientific commands). No model fitting,
label/feature construction, prediction or bootstrap has run. Input structural
validation passed (P0-evidence.md); this is not performance evidence. No audit
pass is claimed.

Resume: reconcile this ledger with Git, the three authority documents, the live
Claude identity and audit run `7ced754ea36c48c0a6d24ba2a17addec`. Do not start
P1 until Codex records the accepted foundation SHA.
Next: Codex reviews P0 artifacts/checks and freezes the foundation commit.
