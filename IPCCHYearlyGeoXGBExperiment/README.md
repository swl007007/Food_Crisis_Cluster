# IPCCHYearlyGeoXGBExperiment

Fixed-map yearly GeoXGB for IPCCH cumulative shares (Trellis task
`10-07-ipcch-fixed-map-yearly-xgb`; authority: that task's `prd.md` R1–R15,
`design.md` v1.0, `implement.md`).

Annual protocol on the original P6 frozen maps (`p6-formal-20261004b`): one
weighted global quartet P per H and current block (period × target year) on
all valid rows with target month ≤ O, decay weights `0.5**((O−t)/24)`;
fresh L2 continuation L of P's own root per supported mapped region (ungated
diagnostic); one gate per region × block at O from the latest six observed
months before O, each scored by its own historical annual pair; G uses L only
where the gate and current support pass, otherwise P. P6 rich561 features,
share truth, bounded isotonic projection, unrounded `>=0.20` decoding and the
frozen per-H recipes (G1L2/G3L2/G4L2/G2L2) are unchanged.

## Running

Only the original P6 Windows runtime (`config/runtime-lock.json`: Python
3.12.10, xgboost 3.0.0, numpy 2.2.6, pandas 2.2.3). Runs live outside Dropbox
under `C:\Users\swl00\AppData\Local\Temp\ipcch-yearly-xgb-runs`.

```bash
PY=/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe
cd IPCCHYearlyGeoXGBExperiment
PYTHONPATH=. WSLENV=PYTHONPATH/p "$PY" -m pytest -q
PYTHONPATH=. WSLENV=PYTHONPATH/p "$PY" run_experiment.py validate-config
PYTHONPATH=. WSLENV=PYTHONPATH/p "$PY" run_experiment.py --run-dir 'C:\...\RUN' preflight   # no fitting
PYTHONPATH=. WSLENV=PYTHONPATH/p "$PY" run_experiment.py --run-dir 'C:\...\RUN' predict     # 724 scalar fits
PYTHONPATH=. WSLENV=PYTHONPATH/p "$PY" run_experiment.py --run-dir 'C:\...\RUN' report
PYTHONPATH=. WSLENV=PYTHONPATH/p "$PY" run_experiment.py --run-dir 'C:\...\RUN' replay      # zero fits
PYTHONPATH=. WSLENV=PYTHONPATH/p "$PY" run_experiment.py --run-dir 'C:\...\TIMING' timing   # synthetic only
```

`--config` may be given but must be this package's frozen `config/`.

## Layout

- `ipcch_yearly_xgb/` — attributed copies of P6 `errors`, `projection`,
  `metrics`, `artifacts`; weighted adaptations of `quartet` and `modelstore`;
  new `contract`, `runtime`, `sources` (input hashes, frozen-map lineage),
  `schedule` (annual calendar), `engine` (gates and routing), `planning`
  (no-fit inventory), `run`, `report`, `replay` (zero-fit re-execution plus
  independent checks), `timing`, `cli`.
- `config/` — `yearly-contract.json` (protocol, recipes, expected inventory),
  `inputs.json` (33 P6 run files + 3 original configs, pinned SHA256),
  `runtime-lock.json` (P6 runtime).
- `tests/` — unit checks and a synthetic annual world with tamper cases.
