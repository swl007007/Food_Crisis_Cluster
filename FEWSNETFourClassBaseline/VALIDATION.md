# Validation — 2026-09-28

Executed with Windows Python 3.12.10 and the pinned package versions recorded in
`runs/fourclass-v3-20260928/prepared/manifests/runtime.json`.

## Focused tests

`python -B tests/test_baseline.py`: 30 tests, all pass (log:
`.trellis/tasks/09-28-fewsnet-four-class-perturbation/research/focused-tests.log`).
Hand-computable fixtures cover: 4/5 merge and missing phases; fixed-four macro F1
with absent classes; FP+FN counting for wrong-class errors; counts aggregated before
F1; no mean fill; probability-axis alignment and argmax ties; D8 masses, zero columns
and finite scans; exact .01 rejection, strict acceptance and parent ties; mixed
parent/child combinations; root rejection and inherited checkpoints carrying the
parent imputer; zero-error parents; removal of the class-1 path; max_plus rules and
training-only fills; pseudo rows appended after imputation and bundle round-trip;
exact origin offsets, window edges, sparse gaps, events, runs, interactions, area
isolation and no future months; covariate sums/lags and scaffold edges; schema order
and exclusions; null-consensus exact reuse; local-model, single-class, <50-row and
unmapped fallbacks with their own imputers; empty fitting pool as an error; D9
weights and all-zero weights; explicit macro columns through step 1; missing
candidates are not a null consensus; synthetic Stage 2 integration (steps 1/3/4/5/6)
for both the learned-map and null routes; bootstrap recount; cohort keys; calendar
expert/persistence joins without backfill.

## Real run

`./run_all.sh runs/fourclass-v3-20260928` completed: 27 Stage 1 folds (81 empty
target months recorded, not fitted), learned 13-cluster general consensus from 3
positive-weight candidates, Stage 3 11/10/9 fitted folds at 4/8/12 months (33/30/27
empty), report with 2,000 of 2,000 bootstrap draws accepted.

`python -B scripts/verify_fourclass.py --run-dir runs/fourclass-v3-20260928`:
**34/34 checks pass** (`runs/fourclass-v3-20260928/verification/verification.json`).
These include independent re-derivation of 1,200 sampled keys x 17 feature columns
from the raw panel, sklearn recomputation of every Stage 1 score and every reported
arm metric, bootstrap draws recomputed by row replication from saved multiplicities,
exact replay (including probabilities) of the first and last fitted Stage 3 fold per
horizon, and reloading of retained Stage 1 checkpoints for the first and last fold
per scope reproducing the saved held-out predictions exactly.

## Limits

Replay covers the selected folds, not every fold. Checkpoints for replay are retained
locally (`stage1/retained/`, ~1 GB) and are not committed; the committed evidence
recomputes every reported number without them. Snapshots are reproducible from the
pinned sources and their hashes are in `prepared/manifests/outputs.json`.
