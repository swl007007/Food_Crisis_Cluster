# Execution plan — approved v0.9, executor handoff

User approved the complete spec/defaults on 2026-09-28 and handed execution to
the bound Claude instance. Task stays planning until the executor's audit start.
The experiment is a four-class baseline, not a strict causal ablation of the old
binary target.

## Audit setup receipt — 2026-09-28

- Registered repository (preserve this exact spelling):
  `/mnt/c/users/swl00/ifpri dropbox/weilun shi/google fund/analysis/2.source_code/step5_geo_rf_trial/food_crisis_cluster`.
- Verified live Claude pane `wF:p2`, session `18826b65-ca2f-4276-b70a-2888ebc93341`,
  terminal `term_65c8c701e80398`; register succeeded against this identity.
  Pane displays `Opus 5 (1M context)`; the requested `5.5` version is not verified.
- Controller boot succeeded in `wF:p3` and reported `running: true`.
  No active task run existed at setup; no base SHA has been established for this task.
- Next, the bound executor verifies current state, runs GitNexus detect_changes,
  commits these approved planning files, then runs the following from its own
  session before implementation (Codex has not committed or started this task):

  ```bash
  trellis-audit --repo '/mnt/c/users/swl00/ifpri dropbox/weilun shi/google fund/analysis/2.source_code/step5_geo_rf_trial/food_crisis_cluster' start fewsnet-four-class-perturbation
  ```

- Verify actual run, executor, base SHA and `in_progress` status. After completion,
  commit implementation/evidence and use wrapper `close` from the same bound
  session to queue a fresh independent Codex audit. Registration is not audit pass.

## 1. Finalize and establish lifecycle

- D1-D23 and exact feature defaults are approved. Preserve PRD requirements,
  acceptance criteria and chronological decision/evidence anchors.
- Read design.md, feature-contract.md, feature-schema.json and release research.
  Curate implement/check context entries; validate the task. Do not treat this
  checklist or a created task as execution authorization.
- At separately authorized execution, inspect git/audit status and actual executor;
  honor prior gates and exact registered repository spelling. Commit only approved
  planning files after GitNexus detect_changes; bound Claude executor uses the
  audit wrapper start. Verify run/base_sha/task state, never invent past approval.

## 2. Isolate and prepare

- Verify release hash/payload, create fresh experiment package and run directory.
  Keep imports local to the copied package. Read trellis-before-dev; package
  contract supersedes unrelated ETH SMOTE/median-imputation/expert-proxy rules.
- Pin data/runtime/country/geometry/schema hashes; validate unique keys, source
  phase mapping, expert alignment and geographic correspondence.
- Implement aligned complete-panel feature construction and separate truth,
  baseline-support and training masks. No automatic full-panel role discovery,
  row-count lags, cross-area rolling or repeated lagging of aligned snapshots.
- Before editing existing symbols, run GitNexus upstream impact and report scope.

## 3. Adapt the three stages

- Adapt target/feature loaders and per-estimator imputer persistence. Add four
  Stage1 synthetic rows only after real-data imputation. Test inherited checkpoint
  copies and pooled fallback preserve the matching transform.
- Replace binary metrics, scan statistics, support checks and child selection
  with D2/D6-D8; do not leave a hidden class1 path or absent-class mean fill.
- Export multiclass monthly test scores and matching correspondence; adapt the
  Stage2 consumers to explicit macroF1 fields and D9/D15 behavior.
- Adapt Stage3 class axes/argmax/local fallback; attach D3/D5 baselines by exact
  keys and score D13 cohorts. No threshold calibration or new model family.
- Implement deterministic keyed reporting, country bootstrap and coverage routes.

## 4. Validate, then run only if authorized

- A small focused test suite must use hand-computable four-class fixtures:
  4/5 merge, missing labels, absent classes, wrong-class FP+FN exposure, fixed
  denominator scan, zero masses, exact .01 rejection, parent ties, mixed parent/
  child checkpoints, sparse origin lags, event ages, area/window boundaries,
  train-only fills, pseudo-row isolation, single-class/global/local fallback,
  full-zero consensus vs missing candidates, and baseline cohort matching.
- Verify dependency versions with the pinned Windows environment, not Linux
  numerical substitutes. Run preflight before any costly fitting.
- Pilot one supported Stage1 fold and one Stage3 fold plus bounded synthetic
  Stage2 integration. Pilot is wiring/resource evidence, not final-score tuning;
  use isolated pilot output or explicitly accounted exact-identity reuse.
- Complete all scheduled stages with a frozen schema and choices. Record skipped
  empty folds, fit counts, timings and failures. Do not change features/gates after
  viewing final scores or silently expand the candidate schedule.
- Recompute all metrics/intervals and audit sample membership; independently
  replay selected first/last nonempty Stage3 folds and relevant Stage1 checkpoints.

## 5. Evidence and closure

- Produce an acceptance index with source/code/feature identities, all model and
  imputer routes, actual learned partitions, keyed predictions and replay commands.
  Distinguish scientific negative outcomes from missing evidence.
- Run trellis-check and GitNexus detect_changes. Commit only agreed implementation
  and reviewable evidence when authorized. Bound executor closes via trellis-audit;
  verify actual independent audit result. An archive or running review is not pass.
- Preserve outstanding findings and original artifacts; do not clear audit gates
  manually, rewrite history or claim an ignored local run is remotely reviewable.

No new framework, distributed runner, model registry, external data download,
SHAP analysis, map publication or hyperparameter/feature search is required.
