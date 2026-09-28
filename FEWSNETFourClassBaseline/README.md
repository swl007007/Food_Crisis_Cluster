# FEWS NET four-class baseline (GeoRF three-stage, v1)

Four ordered IPC classes — 1, 2, 3 and merged "4或5" — forecast at 4/8/12 months
with the three-stage GeoRF pipeline, compared with pooled RF, exact-origin
persistence and the calendar-aligned FEWS NET expert (4 and 8 months only).

Specification: `.trellis/tasks/09-28-fewsnet-four-class-perturbation/` (`prd.md`
R1-R15/A1-A9, `design.md`, `feature-contract.md`, `feature-schema.json`).
Execution record: `.trellis/tasks/archive/2026-09/09-28-fewsnet-four-class-perturbation/` (`IMPLEMENTATION_LOG.md`, `RESULTS.md`); audit repair in `.trellis/tasks/09-28-fourclass-audit-repair/`..
Authoritative run: `runs/fourclass-v6-20260928/`.

This is a four-class baseline with explicit preprocessing and feature changes. It is
**not** a single-variable perturbation of the old binary experiment; no gain or loss
can be attributed to the target change alone (D21).

## Result

Macro F1 (fixed four classes), identical keys within each cohort, 2021-2024 targets:

| horizon | cohort n | partitioned RF | pooled RF | FEWS NET expert | persistence |
|---|---|---|---|---|---|
| 4 months | 54,330 | 0.6532 | 0.6834 | **0.7651** | 0.7308 |
| 8 months | 49,340 | 0.5894 | 0.6102 | **0.6980** | 0.6662 |
| 12 months | 43,441 | 0.5337 | 0.5522 | — | **0.6406** |

Partitioned RF minus each baseline, 95% country-cluster bootstrap interval
(2,000 shared draws, seed 42; per-horizon, marginal):

| horizon | vs pooled | vs expert | vs persistence |
|---|---|---|---|
| 4 | −0.030 [−0.040, −0.021] | −0.112 [−0.188, −0.068] | −0.078 [−0.140, −0.039] |
| 8 | −0.021 [−0.030, −0.002] | −0.109 [−0.147, −0.041] | −0.077 [−0.098, −0.018] |
| 12 | −0.018 [−0.032, −0.010] | — | −0.107 [−0.132, −0.052] |

All eight contrasts are negative with intervals excluding zero. The learned 13-cluster
partition is worse than one pooled RF at every horizon, and both RF arms are below
persistence and the expert. This is a complete negative result, not missing evidence.

The supplementary fs1/fs2 cohorts (expert not required) are identical to the main
cohorts: on the pinned panel every key with an exact-origin observation also has the
expert projection published at that origin.

## Pipeline

```
scripts/prepare_fourclass.py   pinned sources, preflight, ledgers, origin-aligned snapshots, schedule, geometry
scripts/run_stage1.py          27 Stage 1 folds (app/main_model_GF.py per fold, isolated process)
scripts/run_stage2.py          complete-ledger check, D9 weights, general consensus (release steps 1/3/4/5/6)
scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py   Stage 3 per horizon
scripts/report_fourclass.py    keyed cohorts, metrics, shared country bootstrap (never fits)
scripts/verify_fourclass.py    independent recomputation and replay (never modifies the run)
run_all.sh                     the whole sequence into one fresh run directory
```

Reproduce with the pinned Windows Python 3.12.10 (numpy 2.2.6, pandas 2.2.3,
scikit-learn 1.6.1, scipy 1.15.2, geopandas 1.0.1, shapely 2.1.0, polars 1.27.1;
preparation refuses to run on any other versions):

```bash
python -B tests/test_baseline.py                  # 37 focused contract tests
./run_all.sh runs/<fresh-id>                        # ~15 minutes on 32 GB
python -B scripts/verify_fourclass.py --run-dir runs/<fresh-id>
```

Every stage refuses to overwrite existing output.

## What changed relative to GeoRFBaseline v0.1.0

The package was extracted from `GeoRFBaseline/releases/georf-baseline-v0.1.0.zip`
(SHA-256 `39a26138…b500a0`; 45/45 payload hashes verified). The per-file diff against
the release is recorded in `prepared/manifests/sources.json` (`package.modified`,
`package.added`). Substantive changes:

- **Target and metric (R1, R2, D6).** Classes 1/2/3/4或5 encoded 0..3; missing is never
  a class. Per-class F1 = 2TP/(2TP+FP+FN), 0 when undefined, always averaged over four
  classes after aggregating counts (`src/metrics/fourclass.py`). The release's
  `get_prf` mean-fill for absent classes is removed.
- **Scan and split gate (R9, R10, D7, D8).** Four parent-normalized scan columns
  Y = D/(4D_k), A = 2TP/(4D_k), zero columns when D_k = 0, no candidate for zero
  exposure or zero error. Parent and the three child/parent combinations are scored
  on identical parent validation rows in exact rational arithmetic; strict gain > .01
  at every depth, parent wins ties. The class-1 path is removed.
- **Features (R6-R8, D18-D23).** The frozen 162-column schema is built at each key's
  own origin from the complete monthly scaffold (`src/feature/fourclass_features.py`);
  no whole-panel imputation, record shifts, automatic dynamic detection, dummy
  discovery or admin-code predictor. Outcome history uses only same-area records at
  months ≤ O.
- **Imputation (D18).** Every RF — Stage 1 root, children, pooled comparator, Stage 3
  pooled and local — fits the release max_plus fill on its own real fitting rows
  only, ±inf first set to NaN as in the release. Checkpoints store forest, imputer and
  fit record as one bundle, so an inherited parent copy carries the parent transform.
  Stage 1's four class-recovery rows are appended after imputation; Stage 3 has none.
- **Stage 1 integrity.** The grid-only refinement no longer runs in polygon mode; a
  same-area terminal conflict raises; saved `X_branch_id` is checked against
  `s_branch` routing and against the exported correspondence.
- **Stage 2 (R11, D9, D15).** Explicit `macro_f1`/`macro_f1_base` fields through step
  1/3/4. A missing candidate or non-finite score stops Stage 2; only a complete ledger
  of all-zero weights takes the null-consensus route.
- **Stage 3 (R12, R13, D17).** Fixed-axis argmax, no threshold arms, no SMOTE; local
  fits need ≥ 50 rows and ≥ 2 classes, else the same fold's pooled RF. The unmapped
  coverage gate keeps the release definition (all labelled panel rows).
- **Pre-partition CV diagnostic** (binary-only, output discarded) is skipped through the
  release's own `ImportError` branch.

## Inherited behaviours kept and disclosed

- Within-area random validation (.20, seed 42, singletons train-only); repeated split
  search can overfit it, so Stage 1 gains are not forecast evidence (D14).
- Stage 1 training areas are restricted to areas present in the target month.
- Scan candidates come from validation groups only; after an accepted split an area
  without validation rows keeps the parent branch and checkpoint. Counted per fold in
  `candidate.json` → `partition.parent_routed_areas` (2 areas in 2 of 27 folds).
- Stage 2 connected components: 1,177 of 5,506 in-scope areas lie outside the main
  kNN component and are assigned by 1-NN on coordinates.
- max_plus with a negative column maximum can fill inside the observed range.
- Source-month alignment is not verified real-time availability; the 2021-2024
  evaluation years had been inspected before this study.

## Run integrity (audit repair A01/A02)

No stage continues into existing output. Each stage writes its completion record last
(`prepared/manifests/identity.json`, `stage1/folds/*/completion.json`,
`stage2/consensus.json`, `stage3/h*/folds/*/fold.json`, `stage3/h*/run_manifest.json`),
binding code, runtime, upstream identity and the SHA-256 of every output; downstream
stages accept inputs only through those records. Every Stage 3 pooled/local estimator is
saved as `stage3/h*/folds/*/models/*.pkl.xz` (forest + imputer + feature order + fit
identity) and replay loads them without refitting.
