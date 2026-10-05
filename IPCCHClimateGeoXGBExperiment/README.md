# IPCCHGeoXGBExperiment

Independent package for the IPCCH cumulative-share GeoXGB experiment
(Trellis task `10-04-ipcch-cumulative-share-geoxgb`; authority: that task's
`prd.md` R1–R51, `design.md` v1.0, `implement.md` v1.0).

Four scalar XGBoost regressors (q2..q5 cumulative population shares) share one
spatial partition map per horizon H ∈ {1, 3, 6, 12}; predictions are projected
to `1 >= q2 >= q3 >= q4 >= q5 >= 0` and decoded with phase `>= 0.20`.

## Status: implementation and formal run complete

| Command | Status |
|---|---|
| `validate-config` | implemented — validates `config/*.json` against the accepted contract |
| `runtime-probe` | implemented — compares the interpreter with `config/runtime-lock.json` |
| `preflight --run-id ID` | implemented — read-only input identity, geometry, cache and raw-key checks |
| `prepare --run-id ID` | QC targets, rich561 features, Stage1 F/S split and evaluation calendars |
| `learn-map --run-id ID` | Stage1 candidate search and frozen winning maps; no consensus Stage2 |
| `predict --run-id ID` | rolling Stage3, historical local-model gates, matched pooled and persistence |
| `report --run-id ID` | saved-prediction metrics, coverage and paired country bootstrap |
| `replay --run-id ID` | independently recomputes saved evidence and model predictions |

Formal run `p6-formal-20261004b` completed with implementation frozen at
`6798df21ea87d4916c7f36fa6e0753c32bb3ef98`: 291 tests passed and saved-artifact
replay passed 91,880 checks with zero failures. This README was updated after
the run; executable code and scientific configuration remain frozen.

The spatial layer showed no demonstrated gain over matched pooled predictions.
Main-period crisis F1 gains over persistence were small, with all four country
bootstrap intervals including zero. Recall/F2 and paired q3 R² improved while
precision, binary accuracy and four-class macro F1 fell. The 2026 supplement is
reported separately and reverses the crisis-F1 difference at H12.

- [Chinese group-meeting report](../docs/notes/2026-10-04_组会讨论稿_IPCCH人口份额与GeoXGB.md)
- [Complete metric and country tables](../docs/notes/2026-10-04_IPCCH_GeoXGB_指标附表.md)
- [Archived task and scientific contract](../.trellis/tasks/archive/2026-10/10-04-ipcch-cumulative-share-geoxgb/)
- [Results and evidence](../.trellis/tasks/archive/2026-10/10-04-ipcch-cumulative-share-geoxgb/P6-results.md)
- [Final closure](../.trellis/tasks/archive/2026-10/10-04-ipcch-cumulative-share-geoxgb/final-closure.md):
  supervisor verification retained; close and spot audits waived by the user,
  not an independent audit pass.

## Running

Numerical work uses only the pinned Windows runtime
(`C:\Users\swl00\AppData\Local\Microsoft\WindowsApps\python3.12.exe`,
Python 3.12.10, XGBoost 3.0.0); see `config/runtime-lock.json`. Put the package
directory on the import path explicitly:

```bash
# from the repository root (WSL shown; WSLENV forwards PYTHONPATH to Windows Python)
PY=/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe
PYTHONPATH=IPCCHGeoXGBExperiment WSLENV=PYTHONPATH/p "$PY" -m ipcch_climate_geoxgb validate-config
PYTHONPATH=IPCCHGeoXGBExperiment WSLENV=PYTHONPATH/p "$PY" -m ipcch_climate_geoxgb preflight --run-id <new-id>

# The preflight ID and prepared scientific-run ID are distinct.
# For a separately authorized new experiment, use fresh IDs; do not overwrite
# the completed p6-formal-20261004b run.
PYTHONPATH=IPCCHGeoXGBExperiment WSLENV=PYTHONPATH/p "$PY" -m ipcch_climate_geoxgb prepare --run-id <new-scientific-id>
PYTHONPATH=IPCCHGeoXGBExperiment WSLENV=PYTHONPATH/p "$PY" -m ipcch_climate_geoxgb learn-map --run-id <new-scientific-id>
# Inspect the frozen-map and fit-budget checkpoint before predict.
PYTHONPATH=IPCCHGeoXGBExperiment WSLENV=PYTHONPATH/p "$PY" -m ipcch_climate_geoxgb predict --run-id <new-scientific-id>
PYTHONPATH=IPCCHGeoXGBExperiment WSLENV=PYTHONPATH/p "$PY" -m ipcch_climate_geoxgb report --run-id <new-scientific-id>
PYTHONPATH=IPCCHGeoXGBExperiment WSLENV=PYTHONPATH/p "$PY" -m ipcch_climate_geoxgb replay --run-id <new-scientific-id>

# tests (from IPCCHGeoXGBExperiment/)
PYTHONPATH=. WSLENV=PYTHONPATH/p "$PY" -m pytest -q
```

Exit codes: 0 success, 1 contract failure, 2 runtime/lock mismatch or bad
arguments, 3 phase not implemented.

## Layout

- `ipcch_climate_geoxgb/` — package code; imports are `ipcch_climate_geoxgb.*`, stdlib and
  pinned libraries only. No sibling experiment package, old GeoRF ZIP backend,
  bare repository `config`/`src`, or `sys.path` manipulation.
- `config/experiment-contract.json` — frozen scientific constants (horizons,
  calendar, G1–G4/L1–L2 recipes, gates, support floors, budgets).
- `config/feature-schema.json` — ordered rich561 names, semantics
  `ipcch-geoxgb-rich561-ge020-v1` (values are rebuilt by prepare under `>= 0.20`).
- `config/inputs.json` — read-only external inputs with byte/SHA256 identities.
- `config/runtime-lock.json` — interpreter and package pins.
- `config/source-provenance.json` — copied/adapted spans, source inventory and
  removed legacy dependencies.
- `runs/<run-id>/` — local run outputs (git-ignored; existing run IDs are immutable).
  Raw inputs, prepared arrays and fitted models are not shipped in Git. The
  archived task contains compact reports, logs and the final path/size/SHA256
  inventory; reproducing the full fit requires the pinned external inputs.

## Inputs and limits

Raw `IPCCH_2026_completed.csv` and the R39 frozen geography under
`IPCCHGeoRFExperiment/runs/ipcch-v1-20260920d/geography/` are read-only; any
identity or structural mismatch stops the run. Known, preserved limitation:
topology repair does not establish administrative identity; upstream
nearest-neighbour matching and boundary vintage are unverified.
