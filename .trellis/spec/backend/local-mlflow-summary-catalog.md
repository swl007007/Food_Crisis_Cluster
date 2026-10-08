# Local MLflow Summary, External Model Catalog and Evaluation Inputs

Contract for `IPCCHMLflow/summary_catalog.py` (task 10-08-mlflow-focused-evaluation-registry,
delivery accepted by the supervisor at e6bf94e; live store holds plan a4d69a00).
Builds on [local-mlflow-import.md](./local-mlflow-import.md); the original `IPCCH` records are
read-only input.

## 1. Scope / Trigger

- Trigger: new command, new tracking objects (experiment, LoggedModels, registered models,
  run Inputs) and a cross-experiment provenance contract.
- Projection only: scores are copied from the accepted evaluation views; prediction rows are
  read only to bind keys and truth. No fit, rescoring, model load or row upload.

## 2. Signatures

```bash
$PY IPCCHMLflow/summary_catalog.py plan                     # read-only (writes plan JSON under store/summary/)
$PY IPCCHMLflow/summary_catalog.py inventory --out INV.json
$PY IPCCHMLflow/summary_catalog.py apply --plan-fingerprint FP --inventory INV.json --backup BACKUP_DIR
$PY IPCCHMLflow/summary_catalog.py verify --plan-fingerprint FP [--inventory INV.json]
python.exe IPCCHMLflow/browser_probe.py --out R.json URL ...  # Windows Python with websockets; port 9714
```

MLflow 3.17 APIs used: `mlflow.create_external_model(name, source_run_id, tags, params, model_type,
experiment_id)`, `create_registered_model`, `create_model_version(name, source="models:/<id>",
model_id)`, `log_inputs(run_id, datasets=[DatasetInput(Dataset, tags)], models=[LoggedModelInput])`,
`log_batch(metrics=[Metric(..., model_id, dataset_name, dataset_digest)])`.

## 3. Contracts

- Experiment `IPCCH Summary`; one run per (original evaluation view x saved own panel) matching
  `^(main|supplementary)\.(E_all|E_persist|local_eligible|local_persist_matched)$` or
  `^selected_dates\.(all|mapped|common_local_support|new_local_support)$`. Metric names exactly
  the 9 KEYS; NA stays absent (`tags.na_metrics`, reasons in `summary/row.json`).
- Stable keys: run `projection_key = <original source_key>#<namespace>`; LoggedModel and model
  version `projection_key = <original source_key>`. All objects carry `summary_plan_fingerprint`.
- Names: LoggedModel `{family}-h{H}-{arm}-seed{seed}` (no `.`/`/`/`:`/`%`/quotes allowed);
  registered model `ipcch.{family}.h{H}.{arm}` (dots allowed); persistence has no model.
- Model provenance: `source_run_id` = original IPCCH child run (cross-experiment accepted; the
  original run's own rows are not modified). Tags give bundle URI/SHA, member manifest URI,
  code commit; `mlflow.note.content` says not loadable.
- Dataset: name `ipcch-eval.h{H}.{period}.{cohort}`, digest = first 32 hex of sha256(canonical
  descriptor: H, period, cohort, n, keys_sha256, truth_sha256 over `admin|target_ord|phase_truth|q3_truth`
  text, month range, truth definition, selected dates). Same (name, digest) must have the same
  full SHA. Input tags carry full SHA, keys/truth SHA, prediction artifact, original run.
- Experiment tag `catalog_status` = `incomplete` during apply, `complete` only after full verify.

## 4. Validation & Error Matrix

| Condition | Result |
|---|---|
| recomputed plan fingerprint != `--plan-fingerprint` | `SourceConflict`, no writes |
| counts differ from approved scope (492/4,424/4/68/100, >9 metric names) | `SourceConflict` at plan |
| backup DB row counts != current store | refuse ("fresh backup") |
| original experiment differs from the pre-apply inventory (before or after) | refuse / verify fails |
| prediction artifact SHA != parent `manifests/include.json` | `SourceConflict` |
| recomputed keys hash or n != accepted `cohort_keys.*` / panel n | `SourceConflict` |
| two objects share a projection key; object of another plan | `SourceConflict` |
| dataset (name, digest) with a different full descriptor SHA | `SourceConflict` (collision) |
| interrupted apply | `catalog_status=incomplete`; rerun same plan resumes, reuses IDs |

## 5. Good / Base / Bad Cases

- Good: repeat apply = all rows `noop`, no new models/versions/inputs, one history entry per metric.
- Base: browser first-row time unchanged (~1.5–1.9 s headless on this machine) although the
  100-row search payload fell from 3.82 MB to 0.71 MB — page startup dominates first paint.
- Bad: claiming a browser speedup from API bytes alone; treating READY external models as
  loadable; comparing rows with different dataset digests; deleting/reconciling original metrics
  to compact the old experiment.

## 6. Tests Required

`$PY -m unittest discover -s IPCCHMLflow/tests -v` (30 tests; `test_summary_catalog.py` 4):
counts/links/NA (NA absent + tagged, model source run = original child, persistence without
model, shared dataset only for equal keys+truth, original runs unchanged); repeat apply no-op
without metric history; interrupted apply visibly incomplete then resumed (5 noop + 27 created);
refusals (wrong frozen fingerprint, stale backup → nothing written).

## 7. Wrong vs Correct

```python
# Wrong: dataset identity from the cohort key hash alone (same keys, different truth text merge).
digest = cohort_keys_sha[:32]
# Correct: descriptor over keys AND truth content AND definitions; full SHA kept and checked.
digest = sha256(canon(descriptor(H, period, cohort, frame, truth_definition, dates)))[:32]
```

```text
Wrong: verify Summary rows from search_runs results — they omit run model inputs.
Correct: get_run(run_id) for each row before checking inputs.
```
