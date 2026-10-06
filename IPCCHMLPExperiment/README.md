# IPCCHMLPExperiment

Fixed-map MLP residual adaptation for IPCCH cumulative population shares
(Trellis task `10-05-ipcch-mlp-spatial-adaptation`; authority: that task's
`prd.md` R1–R24, `design.md` v0.2, `implement.md`).

Four independent scalar MLPs (q2..q5) per pool; arms B (global), P (B + pooled
residual), L (B + regional residual on the original P6 XGB-selected maps) and
G (L where the historical MLP gate and current support pass, else P). Shares
are projected to `1 >= q2 >= q3 >= q4 >= q5 >= 0` and decoded with phase
`>= 0.20`, as in the original GeoXGB package.

Inputs are the verified `p6-formal-20261004b` prepared data, maps and
predictions (28 files pinned in `config/inputs.json`), read only.

## Running

Numerical work uses only the isolated venv in `config/runtime-lock.json`
(`C:\Users\swl00\.venvs\ipcch-mlp`, Python 3.12.10, torch 2.6.0+cu124).
Scratch runs live outside Dropbox under
`C:\Users\swl00\AppData\Local\Temp\ipcch-mlp-runs`.

```bash
VP=/mnt/c/Users/swl00/.venvs/ipcch-mlp/Scripts/python.exe
cd IPCCHMLPExperiment
PYTHONPATH=. WSLENV=PYTHONPATH/p "$VP" -m pytest -q
PYTHONPATH=. WSLENV=PYTHONPATH/p "$VP" -m ipcch_mlp validate-config
PYTHONPATH=. WSLENV=PYTHONPATH/p "$VP" -m ipcch_mlp timing --out 'C:\...\p0-timing-ID'      # synthetic only
PYTHONPATH=. WSLENV=PYTHONPATH/p "$VP" -m ipcch_mlp preflight --run-dir 'C:\...\RUN'         # no fitting
PYTHONPATH=. WSLENV=PYTHONPATH/p "$VP" -m ipcch_mlp develop  --run-dir 'C:\...\RUN'          # P1, 288 fits
PYTHONPATH=. WSLENV=PYTHONPATH/p "$VP" -m ipcch_mlp predict  --run-dir 'C:\...\RUN'          # P2, 12,972 fits
PYTHONPATH=. WSLENV=PYTHONPATH/p "$VP" -m ipcch_mlp report   --run-dir 'C:\...\RUN'
PYTHONPATH=. WSLENV=PYTHONPATH/p "$VP" -m ipcch_mlp replay   --run-dir 'C:\...\RUN'
```

`run_experiment.py <command> --run-dir DIR` is an equivalent wrapper.
Project-data stages refuse to run until `runtime-lock.json` has a frozen
device and the run has a passed preflight.

## Layout

- `ipcch_mlp/` — package; reused helpers are attributed copies from
  `ipcch_geoxgb` at 6798df2 (`config/source-provenance.json`).
- `config/experiment-mlp.json` — frozen scientific constants and the expected
  no-fit inventory (13,260 scalar fits).
- `tests/` — unit tests and a synthetic end-to-end world with accepted,
  support-fallback, gain-fallback, unmapped and empty-fold cases plus replay
  tamper tests. `IPCCH_MLP_TEST_DEVICE=cuda` runs the end-to-end test on CUDA.
