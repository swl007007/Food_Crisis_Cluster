# FEWS NET four-class GeoXGBoost with frozen shared trees (first bounded round)

Four ordered IPC classes — 1, 2, 3 and merged "4或5" — forecast at 4/8/12 months with a
three-stage partition pipeline whose base learner is native XGBoost. Every regional model
is the fold's **global booster with its learned trees frozen** plus a small fixed number
of appended local rounds; a region uses its increment only if it beats the global on its
own historical, origin-legal gate dates.

Specification (authority): `.trellis/tasks/10-01-geoxgb-shared-parameter-design/`
(`prd.md` R1–R16/A1–A16 and decisions D1–D25, `design.md`, `evaluation-contract.md`,
`experiment-plan.md` v1.0 — the numeric contract, `implement.md`). Execution ledger:
`PROGRESS.md` in the same directory.

Results are recorded in the task directory and in the run's `report/` once the
frozen final evaluation has run; this README describes the method only.

## Lineage

Forked at commit `aa7ac82` from the 55 tracked files of `FEWSNETFourClassBaseline/` at
`14c89bc150194452361bb495c601de070cd94ce7` (no runs, no caches). The mother package, its
inputs and its v7 results are read-only; v7 (historical producer `fe40de37`, 35-month RF)
appears here only as committed reference evidence (candidate maps and keyed predictions),
never as a matched-window backend control.

## What is fixed (D6, D7, D9, D14)

- 162-column frozen schema and origin-aligned features from the pinned sources (schema
  SHA-256 `51b6f8b2…`), snapshots now keyed from 2010-01 so that every 59-month window of
  the schedule is covered; early keys keep the NaN the frozen formulas produce.
- Label window `[O−59, O)` for every global, parent, child and local fit at its own origin.
- Native NaN: dense float input, ±inf→NaN, `missing=NaN`; no imputer, no pseudo rows, no
  sample/class weights. Expert projections never enter features, correction or fusion.
- XGBoost 3.0.0 `xgb.train`, `multi:softprob`, `num_class=4`, `one_output_per_tree`,
  `hist`, CPU, seed 42, `nthread=4`; G1–G4 / L1–L2 as in `src/experiment/plan.py`.

## Pipeline

```
scripts/prepare_fourclass.py  pinned sources, preflight, ledgers, snapshots, schedules (162 roots,
                              648 candidates, 18 development folds, final 2021-2024), six gate dates
scripts/run_experiment.py gscreen   G1-G4 pooled on the development folds; one G per H
scripts/run_stage1.py         162 roots: one global root per (H, T, ratio, seed); four candidate
                              searches each (L1/L2 x E2 gain > 0 / > .01), children = continuations
scripts/run_experiment.py maps      general consensus per (L vector, strategy, origin) from candidates
                              scored before O, all H pooled; truncated v7 maps for the diagnostic
scripts/run_experiment.py develop   24 schemes x 18 folds, shared arm with the six-date gate
scripts/run_experiment.py select    lexicographic rule on development predictions only
scripts/run_experiment.py oldmap    selected G/L on truncated v7 maps: independent and shared arms
scripts/run_experiment.py freeze    final map from all 2018-2020 candidates, rules frozen (cutoff 2020-12)
scripts/run_experiment.py final     pooled / rfmap_independent / rfmap_shared / xgbmap_shared, run once
scripts/report_fourclass.py   keyed cohorts, D3 decision, shared country bootstrap (never fits)
scripts/verify_fourclass.py   independent recomputation and saved-booster replay (never modifies a run)
run_all.sh                    the whole sequence into one fresh run directory (outside Dropbox)
```

Stage 3 engine: `src/experiment/stage3.py`; native booster contract: `src/model/native_xgb.py`.

### Stage 1 (candidate generation, D4/D12/D19/D20/D23)

Within-area random validation (80/20 and 50/50, split seeds 42/43/44) inside `[O−59, O)`.
The root is fitted **once** (the inherited double root fit is removed). A child continues
its current parent with L only if its fitting pool has ≥500 rows/50 areas/6 dates/2
classes, its validation side ≥100 rows/20 areas/3 dates and the path stays ≤80 appended
rounds; otherwise that side only offers the parent route. E2 scores parent/child routes on
the parent's complete validation keys (exact rationals), strict gain >0 or >.01 by
candidate family, parent wins ties. Fitting-only areas (no validation rows) stay on the
parent consistently in row routing, child fitting pools and `s_branch`. E3 compares the
partition with the candidate's own root on the target month; E4 weights are inherited.

### Stage 3 (D13, D16, D18)

For an external origin O, the gate uses the six most recent globally observed label months
U < O; each internal global is fitted at V = U − H on `[V−59, V)`, and each region's local
increment on its own rows. Dates whose local fit fails keep their rows with the global
prediction. A region is enabled only with ≥100 rows/20 areas/3 dates/3 local-fit dates and
an exact macro-F1 gain > .01; a failing current fit falls back to the global; unmapped
areas use the global. No map / null consensus = the pooled global.

## Reproduce

```bash
python -B tests/test_baseline.py                   # 35 focused contract tests
WORKERS=6 ./run_all.sh /mnt/c/Users/<you>/geoxgb_runs/<fresh-id>
python -B scripts/verify_fourclass.py --run-dir <run>
```

Use the pinned Windows Python 3.12.10 (numpy 2.2.6, pandas 2.2.3, scikit-learn 1.6.1,
scipy 1.15.2, geopandas 1.0.1, shapely 2.1.0, polars 1.27.1, xgboost 3.0.0); preparation
refuses other versions and refuses code that differs from the committed HEAD. Every stage
writes its completion record last and refuses to overwrite completed output.

## Disclosed limits

- G and the 24-way scheme choice use the whole development period (D24): development scores
  are conditional on those choices and are not an independent forward test.
- The gate uses the map known at O for earlier dates (D18): a conditional internal screen.
- E1/E2 reuse the same random validation rows (D10/D20).
- 2021–2024 baselines had been inspected before this study: a retrospective evaluation (D16).
- Month alignment is not verified real-time publication availability.
