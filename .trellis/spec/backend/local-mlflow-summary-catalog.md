# Local MLflow Dashboard, External Model Catalog and Dataset Inputs

Contract for `IPCCHMLflow/summary_catalog.py` (task 10-08-mlflow-focused-evaluation-registry
introduced the compact Summary; task 10-09-mlflow-readable-naming replaced it with the wide
`IPCCH - dashboard`). Builds on [local-mlflow-import.md](./local-mlflow-import.md); the detailed
records are read-only input.

## 1. Scope / Trigger

- Trigger: tracking objects (experiment, LoggedModels, registered models, run Inputs) and a
  cross-experiment provenance contract.
- Projection only: scores are copied from the detailed records; prediction rows are read only
  to bind keys and truth. The only derived values are MLP seed means. No fit, rescoring,
  model load or row upload.

## 2. Signatures

```bash
$PY IPCCHMLflow/summary_catalog.py plan                     # read-only (plan JSON under store/dashboard/)
$PY IPCCHMLflow/summary_catalog.py inventory --out INV.json
$PY IPCCHMLflow/summary_catalog.py apply --plan-fingerprint FP --inventory INV.json --backup BACKUP_DIR
$PY IPCCHMLflow/summary_catalog.py verify --plan-fingerprint FP [--inventory INV.json]
python.exe IPCCHMLflow/browser_probe.py --out R.json URL ...  # Windows Python with websockets; port 9714
```

MLflow 3.17 APIs used: `mlflow.create_external_model`, `create_registered_model`,
`create_model_version(name, source="models:/<id>", model_id)`, `log_inputs(run_id,
datasets=[DatasetInput(Dataset, tags)], models=[LoggedModelInput, ...])`,
`log_batch(metrics=[Metric(..., model_id, dataset_name, dataset_digest)])`.

## 3. Contracts

- Experiment `IPCCH - dashboard`; one run per detailed child (family x lead x arm x seed) plus
  one `seed=mean` run per MLP arm x lead (136 rows). Run name = `naming.run_name`; tags = the
  child's readable tags + `record_kind=dashboard_row`, `aggregation` (`single_run`,
  `single_seed`, `mean_of_3_seeds`), description, `_prov.*` links.
- Metrics: `<role>.<cohort>.<leaf>` for role in primary|holdout|combined|selected_months,
  the seven main cohorts and 9 leaves (`binary.accuracy|precision|recall|f1|f2`,
  `four_class.accuracy|macro_f1`, `share_phase3plus_r2`, `n_rows`); contrasts
  `<role>.<cohort>.binary.f1.minus_persistence|minus_pooled[.ci_low|.ci_high]` for the row's own
  arm, from the saved bootstrap (with interval) else the delta block. Gate subsets, reference
  panels and year blocks stay in the detailed records. NA stays absent (`_prov.na_metrics`,
  reasons in `dashboard/row.json`).
- Seed mean: a metric is averaged only when all seeds have it on the same evaluation dataset;
  intervals are never averaged; otherwise NA with a reason. The mean row links all three
  seed models as inputs and logs metrics without a model id.
- Evaluation dataset per (lead, period, cohort): name `IPCCH eval | <lead> | <span> | <cohort>`
  (+ ` (<family>)` for cohorts that depend on the family's regional fits); digest = first 32 hex
  of sha256(canonical descriptor: keys hash, truth hash over `admin|target_ord|phase|q3`, n,
  month range, span, role, cohort, truth definition, selected months). One name = one
  digest. The named span must contain the cohort's months and equal the all_scored months.
  Each metric carries `dataset_name/digest`; each row lists every dataset it uses.
- Training dataset per family x lead (context `training`): `IPCCH training pool | <features> |
  <lead>`, descriptor of the prepared X and key table (paths + SHA256 from the family run's
  manifests, or from the reference run for families that read its files in place).
- Models: registered `IPCCH <family> | <arm> | <lead>` (+ window), versions = seeds; external
  LoggedModel `<family> | <arm> | <lead> | seed N` (no `.`/`/`/`:`/`%`/quotes), source run =
  the detailed child. Persistence has no model. Stable keys `_prov.projection_key` on rows,
  models and versions; all objects carry `_prov.dashboard_plan_fingerprint`.
- Experiment tag `catalog_status` = `incomplete` during apply, `complete` after full verify;
  the experiment description is the reading guide (vocabulary, comparability, filters).

## 4. Validation & Error Matrix

| Condition | Result |
|---|---|
| recomputed plan fingerprint != `--plan-fingerprint` | `SourceConflict`, no writes |
| rows / registered models / versions != EXPECTED (136/68/100) | `SourceConflict` at plan |
| backup DB row counts != current store | refuse ("fresh backup") |
| detailed experiment differs from the pre-apply inventory (before or after) | refuse / verify fails |
| prediction artifact SHA != family run manifest | `SourceConflict` |
| recomputed keys hash or n != `_prov.cohort_keys.*` / panel n_rows | `SourceConflict` |
| scored months outside / different from the named span | `SourceConflict` |
| one dataset name with two contents | `SourceConflict` |
| incomplete seed set for a mean | `SourceConflict` |
| two objects share a projection key; object of another plan | `SourceConflict` |
| interrupted apply | `catalog_status=incomplete`; rerun same plan resumes, reuses IDs |

## 5. Good / Base / Bad Cases

- Good: repeat apply = all rows `noop`, no new models/versions/inputs, one history entry per
  metric; `rebuild_check.py` explains every dashboard value.
- Base: the 100-row default search is larger than the old 9-metric Summary because rows are
  wide; measured and reported in the task evidence.
- Bad: comparing values on different dataset digests; averaging intervals across seeds;
  treating READY external models as loadable; reading a dataset name as a guarantee without
  its digest.

## 6. Tests Required

`$PY -m unittest discover -s IPCCHMLflow/tests -v` (`test_summary_catalog.py` 6): wide row names,
tags, NA, bootstrap vs delta contrasts, eval + training inputs, shared dataset for equal rows,
model names/provenance, registry description, guide; seed means (no interval, rows-differ NA,
incomplete seeds refused); repeat apply no-op; interrupted apply resumed (5 noop + 7 created);
refusals; period span check.

## 7. Wrong vs Correct

```python
# Wrong: dataset identity from the cohort key hash alone (same keys, different truth text merge).
digest = cohort_keys_sha[:32]
# Correct: descriptor over keys AND truth content AND definitions; full SHA kept and checked.
digest = sha256(canon(eval_descriptor(family, H, period, cohort, frame, truth, dates)))[:32]
```

```text
Wrong: one dataset name for regional_model_fitted in two families (rows depend on each
       family's own regional fits -> one name, two digests).
Correct: append the family to names of family-dependent cohorts; plan refuses a name with two
       contents.
```

```text
Wrong: verify dashboard rows from search_runs results — they omit run model inputs.
Correct: get_run(run_id) for each row before checking inputs.
```
