# IPCCHClimateGeoXGBExperiment

Climate-feature perturbation of the IPCCH cumulative-share GeoXGB experiment
(Trellis task `10-05-ipcch-climate-perturbation`, spec `prd.md` D1–D9).
Sibling copy of `IPCCHGeoXGBExperiment` frozen at `6798df2`; namespace
`ipcch_climate_geoxgb`.

What differs from the original package (everything else is unchanged —
contract constants, G1–G4/L1–L2 recipes, gates, support floors, seeds,
calendar, projection/decoding, report and replay logic, pinned runtime):

- **rich601 instead of rich561.** `EVI_mean`, `GPP_mean`,
  `Rainf_f_tavg_mean`, `Tair_f_tavg_mean` and the 12 `EVI_mean` lags are
  removed. 14 `*_month_ensmean` columns from
  `IPCCH_shared_folder/climate_monthly_2015_2026_MODELING_READY.csv` are taken
  at origin O, with the 12 lags on `evi_anom_month_ensmean`. 30 growing-season
  columns from `IPCCH_shared_folder/climate_2015_2026_MODELING_READY.csv`: for
  each season type s1/s2, the latest season whose last day falls in a month
  ≤ O (14 `*_gs_ensmean` values + age in months). History468 is unchanged.
- `config/inputs.json` pins both climate files (bytes + SHA256).
- `IPCCH_CLIMATE_RUNS_DIR` optionally moves the run root outside Dropbox.
- `scripts/compare_with_p6.py` compares a run with `p6-formal-20261004b` on
  identical keys (metric panels, new−old deltas, paired country bootstrap).
- The synthetic end-to-end fixture duplicates its signal column, because at
  width 601 the seed-42 column subsampling otherwise prevents any split.

Formal run `climate-20261005`; results and limits are in the task evidence
(`.trellis/tasks/10-05-ipcch-climate-perturbation/`).

## Running

Numerical work uses only the pinned Windows runtime
(`C:\Users\swl00\AppData\Local\Microsoft\WindowsApps\python3.12.exe`,
Python 3.12.10, XGBoost 3.0.0); see `config/runtime-lock.json`. Put the package
directory on the import path explicitly:

```bash
# from the repository root (WSL shown; WSLENV forwards PYTHONPATH to Windows Python)
PY=/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe
PYTHONPATH=IPCCHClimateGeoXGBExperiment WSLENV=PYTHONPATH/p:IPCCH_CLIMATE_RUNS_DIR/p "$PY" -m ipcch_climate_geoxgb validate-config
PYTHONPATH=IPCCHClimateGeoXGBExperiment WSLENV=PYTHONPATH/p:IPCCH_CLIMATE_RUNS_DIR/p "$PY" -m ipcch_climate_geoxgb preflight --run-id <new-id>

# The preflight ID and prepared scientific-run ID are distinct.
# For a separately authorized new experiment, use fresh IDs; do not overwrite
# the completed p6-formal-20261004b run.
PYTHONPATH=IPCCHClimateGeoXGBExperiment WSLENV=PYTHONPATH/p:IPCCH_CLIMATE_RUNS_DIR/p "$PY" -m ipcch_climate_geoxgb prepare --run-id <new-scientific-id>
PYTHONPATH=IPCCHClimateGeoXGBExperiment WSLENV=PYTHONPATH/p:IPCCH_CLIMATE_RUNS_DIR/p "$PY" -m ipcch_climate_geoxgb learn-map --run-id <new-scientific-id>
# Inspect the frozen-map and fit-budget checkpoint before predict.
PYTHONPATH=IPCCHClimateGeoXGBExperiment WSLENV=PYTHONPATH/p:IPCCH_CLIMATE_RUNS_DIR/p "$PY" -m ipcch_climate_geoxgb predict --run-id <new-scientific-id>
PYTHONPATH=IPCCHClimateGeoXGBExperiment WSLENV=PYTHONPATH/p:IPCCH_CLIMATE_RUNS_DIR/p "$PY" -m ipcch_climate_geoxgb report --run-id <new-scientific-id>
PYTHONPATH=IPCCHClimateGeoXGBExperiment WSLENV=PYTHONPATH/p:IPCCH_CLIMATE_RUNS_DIR/p "$PY" -m ipcch_climate_geoxgb replay --run-id <new-scientific-id>

# tests (from IPCCHClimateGeoXGBExperiment/)
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
- `config/feature-schema.json` — ordered rich601 names, semantics
  `ipcch-geoxgb-rich601-ge020-v1` (values are rebuilt by prepare under `>= 0.20`).
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
