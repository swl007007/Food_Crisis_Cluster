# IPCCHGeoXGBExperiment

Independent package for the IPCCH cumulative-share GeoXGB experiment
(Trellis task `10-04-ipcch-cumulative-share-geoxgb`; authority: that task's
`prd.md` R1–R51, `design.md` v1.0, `implement.md` v1.0).

Four scalar XGBoost regressors (q2..q5 cumulative population shares) share one
spatial partition map per horizon H ∈ {1, 3, 6, 12}; predictions are projected
to `1 >= q2 >= q3 >= q4 >= q5 >= 0` and decoded with phase `>= 0.20`.

## Status: P0 foundation only

| Command | Status |
|---|---|
| `validate-config` | implemented — validates `config/*.json` against the accepted contract |
| `runtime-probe` | implemented — compares the interpreter with `config/runtime-lock.json` |
| `preflight --run-id ID` | implemented — read-only input identity, geometry, cache and raw-key checks |
| `prepare`, `learn-map`, `predict`, `report` | **not implemented** (P1–P5); exit code 3, nothing written |

No model has been fitted and no scientific result exists in this package yet.

## Running

Numerical work uses only the pinned Windows runtime
(`C:\Users\swl00\AppData\Local\Microsoft\WindowsApps\python3.12.exe`,
Python 3.12.10, XGBoost 3.0.0); see `config/runtime-lock.json`. Put the package
directory on the import path explicitly:

```bash
# from the repository root (WSL shown; WSLENV forwards PYTHONPATH to Windows Python)
PY=/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe
PYTHONPATH=IPCCHGeoXGBExperiment WSLENV=PYTHONPATH/p "$PY" -m ipcch_geoxgb validate-config
PYTHONPATH=IPCCHGeoXGBExperiment WSLENV=PYTHONPATH/p "$PY" -m ipcch_geoxgb preflight --run-id <new-id>

# tests (from IPCCHGeoXGBExperiment/)
PYTHONPATH=. WSLENV=PYTHONPATH/p "$PY" -m pytest -q
```

Exit codes: 0 success, 1 contract failure, 2 runtime/lock mismatch or bad
arguments, 3 phase not implemented.

## Layout

- `ipcch_geoxgb/` — package code; imports are `ipcch_geoxgb.*`, stdlib and
  pinned libraries only. No sibling experiment package, old GeoRF ZIP backend,
  bare repository `config`/`src`, or `sys.path` manipulation.
- `config/experiment-contract.json` — frozen scientific constants (horizons,
  calendar, G1–G4/L1–L2 recipes, gates, support floors, budgets).
- `config/feature-schema.json` — ordered rich561 names, semantics
  `ipcch-geoxgb-rich561-ge020-v1` (values are rebuilt in P1 under `>= 0.20`).
- `config/inputs.json` — read-only external inputs with byte/SHA256 identities.
- `config/runtime-lock.json` — interpreter and package pins.
- `config/source-provenance.json` — copied/adapted spans, pending sources and
  removed legacy dependencies.
- `runs/<run-id>/` — run outputs (git-ignored; existing run IDs are immutable).

## Inputs and limits

Raw `IPCCH_2026_completed.csv` and the R39 frozen geography under
`IPCCHGeoRFExperiment/runs/ipcch-v1-20260920d/geography/` are read-only; any
identity or structural mismatch stops the run. Known, preserved limitation:
topology repair does not establish administrative identity; upstream
nearest-neighbour matching and boundary vintage are unverified.
