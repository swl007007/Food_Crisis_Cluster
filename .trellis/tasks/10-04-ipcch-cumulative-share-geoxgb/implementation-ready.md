# Implementation-ready identities for supervisor review (2026-10-04, refreshed after P2/P3 and P4/P5 reviews)

State: **awaiting supervisor release of formal P6**. No formal fit, Stage1 map,
Stage3 prediction or report has been produced on project data. P1 has one
development preparation (`p1-dev-20261004`, superseded by the formal prepare).

## Code

- Implementation commit: `91552d04826b322073e0a40c17d49313e173cb1b` on `ipcch-cumulative-share-geoxgb`
  (chain from foundation `e1f0e65`: cf26c4e P1, 567fb3e P1 fix, 8a83580 P2,
  0087683 P3, 791b4fd P2 review fixes, 13e74b5 P4/P5, 3597834 P2/P3 review
  fixes, 91552d0 P4/P5 review fixes). Package blob ids in
  `evidence/implementation-ready-code-identity.txt`.
- Tests on the pinned runtime: **261 passed** (`evidence/P4P5-review-pytest.log`);
  review evidence in `P2P3-review-evidence.md` and `P4P5-review-evidence.md`.
- Provenance: `config/source-provenance.json` (sha256 `3a77066b…6fabc`):
  20 copied/adapted components, 5 reference-only sources, 7 new components,
  0 pending.

## Frozen configuration (scientific files unchanged since the P0 freeze)

| File | SHA256 |
|---|---|
| `config/experiment-contract.json` | `49a367ce6fbabb7efda3d2f86a78bac8c3f3f1e31790958d290505cd0b6ffd72` |
| `config/feature-schema.json` | `a6093121691e5dce5958f7713872561d89f9d3e19646dcce9c62ce16d62c643d` |
| `config/inputs.json` | `49826724a8812b66f1941c69cbbd15e816f9a9a43e03f01633a0a27be473d550` |
| `config/runtime-lock.json` | `b700a32be856d150ab2f0d49d4b25d45a63d5ce4387ac5a3501119cda3d89b84` |

No scientific constant changed during P1–P5. The 1e-9° coordinate tolerance
(accepted) and all R1–R51 choices are as frozen.

## Data and environment

- Inputs: the 13 pinned files of `config/inputs.json` (raw CSV `ae696087…`,
  five repaired-geometry components, adjacency cache `e6b9562a…`, lookups,
  audits); the formal run re-verifies them (preflight; prepare re-hashes raw
  and lookup; learn-map re-hashes the adjacency cache; every stage re-hashes
  all prepared artifacts against the prepared manifest).
- Runtime: `runtime-probe` on 2026-10-04: Python 3.12.10, XGBoost 3.0.0,
  numpy 2.2.6, pandas 2.2.3, geopandas 1.0.1, pyogrio 0.11.0, shapely 2.1.0,
  pyproj 3.7.1; `matches_lock: true`. Model runtime CPU hist, nthread 4, seed 42.
- Executor: PID 1827, pane `wN:p2`, terminal `term_65d044bdb69412`, session
  `d148c921-36bd-4b42-9ff9-a4f16979e6b5` (see `executor-identity-repair.md`);
  audit run `7ced754e…`, base `6c98f73c…` unchanged.

## Proposed formal P6 run (to start only after release)

From `IPCCHGeoXGBExperiment/` on the pinned runtime, with the code commit
above checked out and a clean package tree:

```
python -m ipcch_geoxgb preflight --run-id p6-preflight-<date>
python -m ipcch_geoxgb prepare   --run-id p6-formal-<date>
python -m ipcch_geoxgb learn-map --run-id p6-formal-<date>   # 4 H x 8 recipes, freezes maps
python -m ipcch_geoxgb predict   --run-id p6-formal-<date>   # 110 main + 16 supplementary scored folds, 12 ledger-only
python -m ipcch_geoxgb report    --run-id p6-formal-<date>
python -m ipcch_geoxgb replay    --run-id p6-formal-<date>   # must report zero failures
```

Stage1 maps are frozen by `learn-map` (map SHA256 + freeze record bound to H,
prepared data, Stage1 summary and selection winner) before `predict` may read
them. Per the supervisor's instruction, after `learn-map` I stop at a checkpoint
and report frozen-map diagnostics plus the exact Stage3 request enumeration and
unique-fit budget (R48) before running `predict`. Any R41 technical failure stops the stage (exit 4) and is reported
with partial evidence; no retuning, retries with changed settings or skipped
folds.

## Compute envelope (measured on synthetic data of real shape; not a promise)

Global quartet 7 s (G1) to 24 s (G4) at 8.5k rows, up to 28 s at 20k rows;
L2 local quartet about 2 s. Stage1 per H: about 1 min of roots plus at most
about 8 min of locals. Stage3 reuse: a historical global for (H, V) equals the
current global of the fold whose origin is V, so unique globals are roughly the
distinct observed origins per H (on the order of 50 per H); locals similarly
per region. Expected order of magnitude is a few hours for the whole run; the
R48 ceilings (3,904 Stage1 + 80,920 Stage3 scalar fits before reuse) remain
the declared upper bound.

## Known limitations carried into P6

- GitNexus impact/detect_changes unavailable (LadybugDB shadow-page error);
  bounded Git/source checks used throughout.
- Upstream geometry matching/vintage uncertainty (R39) and observation-month
  availability convention (R22) remain disclosed limitations.
- Synthetic tests do not certify the scientific run; only the formal run,
  its zero-failure replay and the supervisor/controller audit can.
