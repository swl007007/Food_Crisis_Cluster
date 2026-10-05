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
| P0 infrastructure | component-complete | Accepted `e1f0e6568398d5a6c2164909279abcff89ab6ab5`; Codex full rerun 70 passed; current preflight evidence identity verified |
| Foundation freeze | component-complete | foundation-freeze.md; same Claude released to implement.md P1–P6 with phase supervision |
| P1 data/features | implemented, checked | 98 tests pass; dev prepare `p1-dev-20261004` passed (42,695 valid; crisis 17,807; 122 folds, 110 non-empty; 2026-01..04); see P1-evidence.md |
| P2 quartet/metrics | implemented, checked | 131 tests pass (synthetic only, no project-data fit); see P2-evidence.md |
| P3 Stage1 | implemented, checked | 175 tests pass (synthetic + end-to-end with real quartets/adjacency, test-only small contract); see P3-evidence.md |
| P4 Stage3 | implemented, checked | synthetic tests; see P4P5-evidence.md |
| P5 report/replay | implemented, checked | 218 tests pass incl. end-to-end replay and tamper detection; see P4P5-evidence.md |
| P6 formal run/delivery | complete on run b; awaiting supervisor acceptance | run a `p6-formal-20261004` R41 file-lock stop (preserved); run b `p6-formal-20261004b` (code 6798df2): learn-map 429 fits; predict 796 fits/0 failed, reconciled with enumeration; report sha 142d717d…; replay passed 91 880 checks/0 failures; GeoXGB≈pooled (|ΔF1|≤.0003), vs persistence ΔF1 +.003..+.008 with CIs incl. 0; see P6-results.md |
| Controller close audit | pending | Only after complete committed delivery |

P0 built the independent package foundation. P1 added the QC ledger, phase
truth, rich561 matrices, F/S split and calendars (development run only). No
model fitting, prediction or bootstrap has run. Input structural
validation passed (P0-evidence.md); this is not performance evidence. No audit
pass is claimed.

Resume: reconcile this ledger with Git, the three authority documents, the live
Claude identity and audit run `7ced754ea36c48c0a6d24ba2a17addec`.
Next: Claude reports implementation-ready identities; formal P6 waits for supervisor release, reporting phase commits/evidence
to Codex (P1-P3 reported with their commits). Before P6 formal fits,
report the implementation-ready identity for supervisor verification.
