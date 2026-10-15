# IPCCH AlphaEarth supervised partition experiment

Task: `.trellis/tasks/10-10-ipcch-alphaearth-supervised-partition/` (PRD, design, implement, decisions D1–D29 plus the D30 extraction recipe).
Question: does a supervised, frozen AlphaEarth (2017) environmental grouping add transferable predictive value to a
strong pooled IPCCH model, separately for H = 1/3/6/12, beyond the added descriptors and a shared calibration?

| Role | Model |
|---|---|
| B0 | compact296 pooled XGB quartet (q2–q5), inherited D21 recipes |
| B1 | same + AlphaEarth 64 means + 64 SDs (424 inputs) |
| B2 | exactly B1 + shared low-dimensional ridge calibration |
| G  | exactly B1 + jointly fitted shared calibration and shrunk regional deviations on a frozen supervised map |

Primary contrast G−B2 per H; every H is reported separately (no all-H gate), under a fixed four-H Bonferroni family.

## Stages and commands

Run from this directory with the numerical environment. `RUN_DIR` must be a new path on the WSL filesystem (never
under Dropbox); runs are immutable and refuse reuse after any configuration or input-request drift.

```bash
PY=/home/swl007007/.venvs/ipcch-geo/bin/python
$PY -m unittest discover -s tests                                   # full offline test suite (synthetic only)
$PY -m ipcch_alphaearth preflight --run "$RUN_DIR" [--kind formal] # runtime lock, contract, inputs, no fit/EE
$PY -m ipcch_alphaearth prepare --run "$RUN_DIR" --local-only       # P0: truth, cohorts, compact296 @ row origin, schedule
$PY -m ipcch_alphaearth benchmark --out DIR --prepared "$RUN_DIR/prepared"   # synthetic compute benchmark
# P1, D30 analysis-grid extraction. Plan + sha-bound manifest locally (release_id null until a supervisor release):
PYTHONPATH=.:scripts $PY scripts/build_c_production.py --geometry "$GEOMETRY_SHP" --list-images "$LIST_IMAGES_JSON" --out "$PLAN_DIR"
# cloud release only: separate EE interpreter, one invocation at a time, resumable in the same OUT_DIR
# (optional --concurrency-ceiling N <= manifest concurrency; the hashed manifest is never edited):
timeout "$OUTER_TIMEOUT_S" /opt/earthengine/bin/python scripts/ee_pilot_c.py --request "$RELEASED_MANIFEST" --out "$OUT_DIR"
/opt/earthengine/bin/python scripts/check_c_outputs.py --request "$RELEASED_MANIFEST" --out "$OUT_DIR" --json "$CHECK_JSON"
# after a COMPLETE, clean extraction (checker complete:true, pending 0, no violations), in a fresh run:
$PY -m ipcch_alphaearth ingest-extraction --run "$RUN_DIR" --plan "$PLAN_CSV" --manifest "$RELEASED_MANIFEST" --outputs "$OUT_DIR"
$PY -m ipcch_alphaearth prepare --run "$RUN_DIR"                    # attach descriptors (requires extraction COMPLETE)
# scripts/extract_alphaearth.py and scripts/ee_pilot.py are the superseded native-grid recipe: comparison only (D30).
# P2/P3/P4 (project-data fitting release only):
$PY -m ipcch_alphaearth develop --run "$RUN_DIR"      # 19 historical B1 + 8 dev B0; 9 recipes x 2 years; selection
$PY -m ipcch_alphaearth freeze-map --run "$RUN_DIR"   # final maps through 2022-12; MAP_FREEZE hash chain
$PY -m ipcch_alphaearth predict --run "$RUN_DIR"      # 4 bridge B1 + 15 main/supp B1 + 15 B0; annual B2/G
$PY -m ipcch_alphaearth report --run "$RUN_DIR"       # per-H metrics, guards, 20,000-draw country bootstrap
$PY -m ipcch_alphaearth replay --run "$RUN_DIR"       # independent zero-XGB-fit replay
```

Every stage checks its prerequisites and never launches another stage. Fitting commands refuse to start until the
P1 descriptor stage exists. The fresh backbone ceiling is 61 quartets / 244 scalar XGB fits.

## Formal result (2026-10-14)

Run `p1-formal-d30-20261014` (local, outside Dropbox): all 57,801 D30 extraction pieces (6,227 admins, all usable);
61 quartets / 244 scalar XGB fits. Development: 0 of 72 recipe-years passed the guards, so every H fell back to B2
and **no regional partition map was selected** (G = B2 bitwise; G−B2 = 0; no qualifying horizon). Replay: 488/488
checks, zero fits, after a reviewed analysis-only amendment of the replay prediction reader. Discussion note:
`docs/notes/2026-10-14_组会讨论稿_IPCCH_AlphaEarth监督分区.md`; evidence and progress are in the task folder.

## Modules

`targets/projection/metrics` (copied from P6 at 6798df2), `compact` (vendored compact296 recipe with an explicit H1
rebuild; sibling IPCCH at 9b41d7a), `prepare`, `sources`, `schedule` (annual blocks, unique OOT providers),
`backbone` (XGB quartets, exact-identity model store), `correction` (D19 transforms, constrained joint ridge),
`partition` (deterministic supervised search, frozen maps), `pipeline` (develop/freeze/predict), `evaluation` and
`report` (D22–D24), `replay` + `replay_report` (independent recount), `aemoments`/`extraction`/`descriptors` (D25/D26
moments, ownership, coverage), `benchmark`, `synthetic` (tests only). Attribution and source hashes are in each module
docstring.

## Environments

Numerical: `/home/swl007007/.venvs/ipcch-geo/bin/python` (Python 3.12.3, numpy 2.4.4, pandas 3.0.3, scikit-learn 1.8.0,
xgboost 3.2.0, scipy 1.17.1; see `config/runtime-lock.json`). Earth Engine: `/opt/earthengine/bin/python`
(earthengine-api 1.7.36), exchanging hashed files only. Neither environment is modified.

## Limitations carried by every result

Retrospective study over previously viewed 2023–2026 years; observation-month availability proxy (release vintages,
static snapshots and supplier climatology samples unverified); unknown original tuning history of the inherited XGB
recipes; AlphaEarth product/pretraining history not certified; EE boundary weights approximate (~1/256); administrative
matching/boundary vintage unverified; country-bootstrap inference conditional on the fitted predictions and calendar.
