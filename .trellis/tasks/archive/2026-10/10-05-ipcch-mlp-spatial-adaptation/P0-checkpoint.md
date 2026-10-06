# P0 checkpoint — IPCCH fixed-map MLP (2026-10-05)

Executor: Claude Opus 5.5 1M (`claude-opus-5-5[1m]`), branch `ipcch-mlp-spatial-adaptation`.
No project-data fitting has been performed. Everything below is configuration
checks, a no-fit enumeration on the real inputs, and synthetic data only.
Project-data fitting (P1 develop onward) waits for the user's release.

## Package and code

- New sibling package `IPCCHMLPExperiment/` (namespace `ipcch_mlp`); original
  `IPCCHGeoXGBExperiment/` unchanged. Reused helpers are attributed copies from
  `ipcch_geoxgb` at 6798df2, listed with source hashes in
  `IPCCHMLPExperiment/config/source-provenance.json`.
- CLI: `validate-config`, `probe`, `timing`, `preflight`, `develop`, `predict`,
  `report`, `replay` (`python -m ipcch_mlp <command> --run-dir DIR`, or the
  `run_experiment.py` wrapper). Config paths are fixed inside the package
  rather than a `--config` argument.
- Replay re-executes develop/Stage3/report against a read-only model store
  (no refit possible) and requires byte-identical artifacts, then independently
  recomputes projection, phases, routing, gate decisions, selection and the
  fit inventory from the saved keyed files.

## Runtime

| Item | Value |
| --- | --- |
| Interpreter | `C:\Users\swl00\.venvs\ipcch-mlp\Scripts\python.exe` (Python 3.12.10, isolated venv) |
| Packages | numpy 2.2.6, pandas 2.2.3, torch 2.6.0+cu124, pytest 9.1.0 (full list in `evidence/pip-freeze.txt`) |
| torch RECORD sha256 | `99987a65675814aa0271318de829846eb464df31a6691295642748e1e8963d4f` |
| GPU | NVIDIA GeForce RTX 4070, CUDA build 12.4, cuDNN 90100 |
| Deterministic settings | float32, deterministic algorithms on, cuDNN benchmark off, TF32 off, no autocast, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, 4 CPU threads |
| Frozen device | **cuda** (rule below) |
| C: free space | 301 GB; RAM 31.8 GB total, 5.4 GB available at preflight |

The venv was first requested under `AppData\Local`, which the Windows Store
Python silently redirects into its package sandbox; it was recreated under
`C:\Users\swl00\.venvs\ipcch-mlp`. pip did not verify the wheel against the
planning SHA256 (`3313061c…`) because hash checking was not requested; the
installed distribution's RECORD hash is recorded instead.

## Tests

30 tests pass (20 unit, 10 synthetic end-to-end/tamper), on CPU and again with
the end-to-end world on CUDA. Covered: transform rules (all-missing → 0/scale 1,
zero variance → scale 1, flags, float32, infinity stops, unseen missingness),
residual output exactly zero in train and eval mode, zero target keeps zero
output, nonzero target trains output then hidden layers, repeatability and RNG
isolation across intervening fits, update counts with partial batches, decay on
weight matrices only, dropout off at prediction, seed derivation, store
hit/conflict/corruption/read-only, selection ties/undefined/unavailable, gate
support and exact 1/100 equality, inclusive 0.20 decoding, calendar cutoffs.
The synthetic world exercises an adopted regional route, support fallback, gain
fallback with existing ungated L predictions that leave G = P bit-for-bit,
unmapped keys and an empty fold; replay passes and catches a tampered
prediction, a corrupted model and a changed map identity.

## Preflight on the real inputs (no fitting)

`p0-preflight-20261005`: all 28 pinned P6 files rehashed and matched; maps have
3,264 areas and 9/7/6/9 regions. The no-fit enumerator reproduces every planned
count exactly:

| H | Unique global / pooled residual | Regional (hist / current-only) | Scalar fits per seed | Global pool min–max | Main keys | Support-eligible main keys |
| --- | --- | --- | ---: | --- | ---: | ---: |
| 1 | 43 / 43 | 183 / 12 | 1,124 | 10,465–19,012 | 17,322 | 3,211 |
| 3 | 43 / 43 | 143 / 18 | 988 | 10,465–19,052 | 16,919 | 4,997 |
| 6 | 43 / 43 | 129 / 25 | 960 | 10,339–19,052 | 16,413 | 3,423 |
| 12 | 43 / 43 | 157 / 70 | 1,252 | 9,577–16,394 | 14,087 | 845 |

Total: 288 development + 3 × 4,324 Stage3 = **13,260 scalar fits**.

## Synthetic timing (`p0-timing-20261005`)

Both devices eligible: repeated fits identical, save/reload predictions
identical, RNG isolation identical. Fit time is dominated by per-update
overhead (about 4 ms per optimizer step on both devices), so the GPU gives
little speed-up for these small networks.

| Case (n = 19,052) | CPU seconds | CUDA seconds |
| --- | --- | --- |
| Global G1, 100 epochs (7,500 updates) | 29.6–35.1 | 29.8–35.7 |
| Global G2 | 39.3–44.6 | 32.1–37.0 |
| Residual R1, 40 epochs (3,000 updates) | 5.3–5.8 | 8.2–10.4 |
| Residual R2 | 9.0–9.4 | 4.1–8.7 |

Extrapolated over the enumerated pool sizes (min–max of repeat timings, hours):

| Recipe | Stage3, three seeds, CPU | Stage3, three seeds, CUDA |
| --- | --- | --- |
| G1R1 | 15.6–17.5 | 17.0–19.6 |
| G1R2 | 17.0–18.9 | 16.5–19.5 |
| G2R1 | 19.0–22.1 | 17.1–20.6 |
| G2R2 | 20.4–23.5 | 16.6–20.5 |

Development (288 fits) ≈ 0.6–0.7 h. Replay is inference only (not timed; expected
well under the training time). Device rule: faster eligible device by the upper
estimate, ties CPU → **CUDA** (20.6 h vs 23.5 h worst case). Expected wall time
for the planned run is therefore roughly **18–22 hours** of serial fitting,
plus report and replay. Estimated model storage ≈ 3 GB.

## Decision for the user

Release P1 + P2 (develop, then Stage3 for three seeds) as specified, about a
day of serial fitting on CUDA. The design requires serial fits. Running the
three replicates as separate parallel processes could cut wall time to roughly
a third, and should not change fitted tensors because seeds and RNG resets are
per fit, but it departs from the written "execute fits serially" rule, needs
per-process ledgers (a small code change and re-test), and would need your
explicit approval. Without that approval the run stays serial.

## Addendum — replicate-parallel execution and P1+P2 release (2026-10-05)

User decision after P0: run replicates in parallel (PRD R25) and release P1+P2;
the implementation session stops before the formal development run, which the
user starts as a goal run.

- Implemented in `ipcch_mlp/parallel.py` (commit `cd7bcd5`): one spawned worker
  process per replicate for develop and Stage3, per-worker request ledgers,
  parent-side selection/summary after all workers succeed, no retry on failure.
- The parallel test exposed a real Windows race: two workers writing the same
  shared transform with `os.replace` raised `WinError 5`. Transforms now use
  exclusive create (rename only if absent, else verify), with a 4-process
  stress test.
- Synthetic tests: parallel and serial runs produce byte-identical develop and
  Stage3 artifacts and identical model tensors; a failed worker stops the run
  before selection. 33 tests pass on CPU; the end-to-end file also passes on CUDA.
- Three-process synthetic probe on a fixed case list (`evidence/p0-timing-parallel-*.json`),
  digests identical to a single process:

| Device | 1 process wall | 3 processes wall | Throughput |
| --- | ---: | ---: | ---: |
| CUDA | 25.2 s | 33.3 s | 2.27× |
| CPU | 28.8 s | 35.3 s | 2.45× |

  CUDA keeps the lower three-process wall time, so the frozen device stays CUDA.
  Expected wall time: Stage3 ≈ 16.5–20.6 h serial ÷ 2.27 ≈ **7–9 h**, development
  ≈ 0.3 h, plus report and serial replay (inference only).
- Formal run directory created with a passed preflight (no fitting):
  `C:\Users\swl00\AppData\Local\Temp\ipcch-mlp-runs\mlp-formal-20261005`
  (`evidence/formal-preflight.json`; 13,260 fits reproduced; 300 GB free;
  13.4 GB RAM available at that moment).

Goal-run commands (WSL, from `IPCCHMLPExperiment/`; each stage refuses to run if
a previous stage left `RUN_INCOMPLETE.json`):

```text
VP=/mnt/c/Users/swl00/.venvs/ipcch-mlp/Scripts/python.exe
RUN='C:\Users\swl00\AppData\Local\Temp\ipcch-mlp-runs\mlp-formal-20261005'
PYTHONPATH=. WSLENV=PYTHONPATH/p "$VP" -m ipcch_mlp develop --run-dir "$RUN"
PYTHONPATH=. WSLENV=PYTHONPATH/p "$VP" -m ipcch_mlp predict --run-dir "$RUN"
PYTHONPATH=. WSLENV=PYTHONPATH/p "$VP" -m ipcch_mlp report  --run-dir "$RUN"
PYTHONPATH=. WSLENV=PYTHONPATH/p "$VP" -m ipcch_mlp replay  --run-dir "$RUN"
```
