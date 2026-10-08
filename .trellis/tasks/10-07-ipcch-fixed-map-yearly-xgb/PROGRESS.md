# PROGRESS — ipcch-fixed-map-yearly-xgb (operational log, not approval authority)

## 2026-10-07 lifecycle

- Identity: Claude Code session `174ea213-be60-4f2a-9bcb-e5e3523ad738`, Herdr pane wN:p2, terminal `term_65d3f9b51d7fa2`, PID 2367, model `claude-opus-5-5` (1M context). Supervisor: Codex wN:p1.
- HEAD verified `7bd681d` on `ipcch-fixed-map-yearly-xgb` (planning commit, 10 files).
- `trellis-audit --repo '/mnt/c/users/swl00/ifpri dropbox/weilun shi/google fund/analysis/2.source_code/step5_geo_rf_trial/food_crisis_cluster' start ipcch-fixed-map-yearly-xgb` → run `252b7ae51dcb4e53905e16f0de483088`, phase active, base_sha `7bd681deb3e6935b59c0af12ed9f792a961d9188`, executor claude session above; native status `in_progress`. Reported to wN:p1.

## P0 implementation

New sibling package `IPCCHYearlyGeoXGBExperiment/` (namespace `ipcch_yearly_xgb`). Original P6/MLP/climate packages and runs untouched.

- Attributed copies (6798df2): `errors.py`, `projection.py` (unchanged logic), `metrics.py` (targets.four_class inlined), `artifacts.py` (explicit run root).
- Weighted adaptations: `quartet.py` (decay weights into DMatrix; records protocol float64 and effective float32 weight digests, sum, ESS; root bytes/base score/rounds/prefix-margin checks unchanged); `modelstore.py` (identities must bind weights; records must match them; read-only mode refuses to fit).
- New: `contract.py`, `runtime.py`, `sources.py` (36 pinned inputs; frozen-map lineage re-validated against the original P6 contract/schema/manifest/summary/selection), `schedule.py` (fit_origin rule, annual blocks anchored on the first scheduled month, gate dates, decay weights), `engine.py` (annual historical replay, frozen gates, ungated current L, atomic G routing), `planning.py` (no-fit inventory), `run.py`, `report.py`, `replay.py` (zero-fit re-execution + independent calendar/weights/support/gate/projection/metric/bootstrap checks), `timing.py`, `cli.py`.
- Model identities bind `fit_source_sha256` over quartet/modelstore/sources/schedule/engine/contract and the three config files; every run manifest records the complete 22-file source inventory.
- GitNexus: no existing symbol was edited (copies and new modules only). `detect_changes` attempted before commit; result recorded in the commit message.

## P0 checks (synthetic and no-fit only)

| Check | Result |
| --- | --- |
| `validate-config` | passed (contract `ipcch-yearly-xgb-contract-v1.0`) |
| Runtime probe | Windows Python 3.12.10, numpy 2.2.6, pandas 2.2.3, xgboost 3.0.0 and all 8 locked packages match |
| Tests | 24 passed (13 unit, 11 synthetic end-to-end/tamper) |
| Preflight `p0-preflight-20261007` | passed: 36/36 inputs rehashed and matched; lineage valid for H1/3/6/12; planner equals `research/fit-enumeration.json` field by field; 15 current blocks; 21 global + 160 local quartets = 724 scalar fits (main 584); full keys 17,322/16,919/16,413/14,087 + 4,092; diagnostic keys 9,943/9,866/9,522/7,904 + 1,829; 138 folds, 126 scored; lone unsupported H12/2021-01/r0011 |
| Timing `p0-timing-20261007` (synthetic) | estimated fit time ≈ 610 s for all 724 fits (H1 119 s, H3 125 s, H6 216 s, H12 150 s); probe wall 183 s |

Synthetic end-to-end covers: partial first block anchored on an empty first month, row origin varying under a fixed fit origin, same-year historical model reuse (one fit), adopted/gain-rejected/support-rejected/unmapped routes, ungated L under a rejected gate with G = P bit-for-bit, inventory equal to the planner, gates unchanged when later outcomes change, zero-fit replay passing, and replay catching a within-block route change, an omitted validation key, a changed fit origin, a wrong local parent and corrupted booster bytes.

Resource estimate for P1: ≈ 10 min of fitting plus prediction, report and replay (expected well under 1 h); model storage ≈ 0.3 GB (quartets 0.8–2.1 MB). Output plan: fresh run under `C:\Users\swl00\AppData\Local\Temp\ipcch-yearly-xgb-runs\<run-id>` (outside Dropbox; 302 GB free on C:).

Interpreter `C:\Users\swl00\AppData\Local\Microsoft\WindowsApps\python3.12.exe`; commands as in the package README (`run_experiment.py --run-dir … preflight|predict|report|replay`).

STOP: no project-data fit or pilot has been run. Awaiting Codex release for P1.

## P0 supervisor review and repair (2026-10-07)

- Supervisor review of `eec001d`: HOLD P1 with a fixed repair list (copied to `evidence/p0-review-eec001d.md`). No science/timing/weight/root error was found; inputs (36), source inventory (22) and the 724 inventory were independently verified.
- Supervisor reran the synthetic suite with the locked Windows Python from `IPCCHYearlyGeoXGBExperiment/` with `PYTHONPATH=. WSLENV=PYTHONPATH/p`: `python -m pytest -q -p no:cacheprovider` → 24 passed in 37.27 s. An initial invocation from the repository root without the package PYTHONPATH failed at collection only; the documented invocation passed.

Repairs (no scientific contract, recipe, map or inventory change):

1. Fit provenance: each model entry stores `fit_rows.npy` (ordered row references) and `fit_keys.npy` (canonical admin/target keys); their array digests must equal the identity's `fit_rows`/`fit_keys` on write and every load. Identities also bind `X_fit_sha256`, the digest (dtype/shape header included) of the actual selected float64 X rows. Replay recomputes rows/keys/X/y from the reconstructed inputs and compares with sidecar and identity.
2. Complete source freeze: new `freeze.py`. predict/report/replay compare the preflight's complete source inventory with the current one before any other work; a difference stops unless a pre-written `source-reconciliation.json` authorizes it file by file (old/new hash, reason, authorizer). Fit-defining sources can never be reconciled. `report` now runs the same context verification (runtime, staged inputs, freeze) without fitting.
3. Input staging: preflight copies the 36 pinned inputs once into `<run>/inputs/{run,source}/<relative path>` (outside Dropbox), verifies the copies and writes `staging-manifest.json`; lineage, training data, maps, calendar and P6 comparators are read only from that snapshot. The report has no Dropbox fallback.
4. Report: new `local_persistence_matched` cohort per H × period (L-eligible ∩ persistence-available keys) with local/pool/geo/persistence panels, deltas and coverage; the full-cohort G−P stays primary.
5. Independent metrics in replay: a separately written panel (four-class confusion/accuracy/macro-F1, binary accuracy/precision/recall/F1/F2, projected/raw q3 R², NA semantics) checks every reported panel and delta of E_all, E_persist, the local diagnostic (all and each gate group) and the new matched cohort; counts and NA status exact, floats within 1e-12 relative.

Checks:

| Check | Result |
| --- | --- |
| Tests | 35 passed (24 existing + 11 focused: provenance-less entry refused, sidecar tamper, selected-X change, fitted-key tamper, evaluator/projection/report source mismatch, reconciliation rules incl. fit sources, CLI stops before any work, staging copy/verify/tamper/no re-stage, local-persistence cohort, 5 non-F1 metric/cohort mutations caught) |
| Preflight `p0b-preflight-20261007` | passed: 36 inputs staged (748 MB) and verified; lineage valid; inventory unchanged: 21 + 160 quartets = 724 scalar fits; main full keys 17,322/16,919/16,413/14,087; diagnostic 9,943/9,866/9,522/7,904 |
| fit_source_sha256 | `d7a693959b5ed6fd7c1907b4d1401d713d7ea3a4d44c419c2424cd5a72a3ed36` (23-file inventory in the preflight) |
| Timing | not rerun: the fitting path (quartet weights/XGBoost calls) is unchanged; only provenance files and checks were added |

STOP: no project-data fit. Awaiting supervisor delta review and P1 release.
