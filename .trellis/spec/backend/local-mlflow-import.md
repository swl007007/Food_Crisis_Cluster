# Local MLflow Import of Completed Runs

Contract for `IPCCHMLflow/` (task 10-07-local-mlflow-experiment-management, accepted at
efc5d11). Use it when adding a completed run family to the local MLflow store or when
changing extraction/inventory/verification code.

## 1. Scope / Trigger

- Trigger: infra integration (SQLite tracking store, proxied artifact store, localhost
  service) plus a cross-run provenance contract read by later comparisons.
- MLflow holds **historical imports**. It never retrains, loads models or recomputes
  scores; it reads saved reports/predictions/ledgers and copies retained artifacts.

## 2. Signatures

```bash
PY=/home/swl007007/.venvs/ipcch-mlflow/bin/python   # uv venv, MLflow 3.17.0, requirements.lock
IPCCHMLflow/manage.sh start|stop|status             # transient user unit, 127.0.0.1:5000
$PY IPCCHMLflow/import_runs.py plan [--rehash] [--family F] [--evidence DIR]
$PY IPCCHMLflow/import_runs.py import [--family F]
$PY IPCCHMLflow/import_runs.py verify [--shallow]
$PY IPCCHMLflow/import_runs.py reconcile             # one-time, additive only
$PY IPCCHMLflow/backup_restore.py backup --dest DIR
$PY IPCCHMLflow/backup_restore.py restore-check --backup DIR --dest SCRATCH [--port 5001]
```

Store: `/home/swl007007/.local/share/ipcch-mlflow/{mlflow.db,artifacts/,plans/,cache/,logs/}`.

## 3. Contracts

- One experiment `IPCCH`. Parent `record_kind=source_run` per source run (artifacts kept
  once); child `record_kind=evaluation_view` per family x H x arm x seed, linked by
  `mlflow.parentRunId`. Persistence views use `seed=none`.
- Identity tags: `source_key` (stable), `import_fingerprint` (sha256 over plan content +
  importer version), `import_status` in {`in_progress`, `reconciling`, `complete`},
  optional `import_fingerprint.previous` after a reconcile.
- Metric names `{period}.{cohort}[.{group}].{metric}`, MLflow-safe `[\w\-. /]`. Only finite
  source values are metrics; undefined values go to `view/na.json` with source path and
  reason. Every logged value has its source JSON path in `view/evaluation_view.json`.
- Cohort identity: `cohort_keys.{period}.{cohort}` = sha256 of sorted
  `admin_code|target_ord`; its key count must equal the reported n. Gate subsets
  (`local_eligible.{gate}`) are derived with the source report's own `gate_category` rule.
- Source inventory policy (`sources.json` -> `inventory`): every `*sha256` key in the run's
  own ledgers/manifests has an explicit decision — `required_digest_keys`, `name_keys`,
  `informational_keys`, or a scoped `exempt` (key + JSON-path regex [+ file regex] +
  reason) — plus `model_contract` (`xgb` or `mlp`).
- Provenance tags `execution` and `mlflow_timestamps` state that MLflow times are import
  times. `source.scientific_acceptance`, `source.lifecycle_status` and `import_status`
  are separate fields.

## 4. Validation & Error Matrix

| Condition | Result |
|---|---|
| reported cohort n != prediction key count (incl. gate subsets) | `SourceConflict` at plan |
| two different source paths map to one metric name | `SourceConflict` (only panel-n vs cohort-n is an equal-value cross-check) |
| `*sha256` key without an inventory decision | `SourceConflict` |
| required digest matches no planned file (own or reference parent) | `SourceConflict` |
| XGB identity lacks its own `q*.ubj` with the ledger digest, or `record.json` | `SourceConflict` |
| MLP identity lacks `record.json`, `state.pt` or its referenced transform | `SourceConflict` |
| existing record has a different fingerprint | `import` stops; only `reconcile` may proceed |
| reconcile would change a metric/param/existing tag/archived source or model, in the parent or ANY child | `SourceConflict` before any write (whole family validated first; store unchanged). Archived content is compared by SHA256 per `source/…`/extras file and per `models.tar` member against the record's retained `manifests/*.json` — never by size alone |
| reconcile interrupted after validation (parent `import_status=reconciling`) | `import` refuses; `reconcile` resumes only when `reconcile_target` equals the current plan fingerprint, else stops |
| `--family F` | plans F plus transitive read dependencies (`shared_inputs.parent`, `inventory.reference_parents`), dependencies first; writes/verify only F; unknown family refused |
| equal-size artifact corruption | caught by deep verify, including the repeat-import no-op |
| second importer while one runs | refused by `import.lock` |
| restore-check on port 5000 | refused; scratch checks use their own server |

## 5. Good / Base / Bad Cases

- Good: plan shows 0 unlisted files; every exemption names a reason; import marks
  `complete` only after deep readback; repeat import = all `noop`, 0 writes.
- Base: path-level-only coverage where the source stores no file digests (MLP states and
  transforms), documented in the policy `note`.
- Bad: counting surviving files as coverage; reporting ledger reference occurrences as
  unique models; claiming backup bytes (which include `superseded/` files) as
  deep-verify downloaded bytes.

## 6. Tests Required

`$PY -m unittest discover -s IPCCHMLflow/tests -v` (26 tests). Assertion points:
- NA not logged and present in `na.json`; subset cohort digests differ; count mismatch refused.
- Interrupted import resumes without duplicates (two interruption points); idempotent no-op.
- Missing/changed retained booster refused; own booster required even if another
  directory holds the same digest; MLP missing `state.pt` / referenced transform refused.
- Unclassified digest key refused; name collision refused; gate subsets bound to routes.
- Reconcile: additive change keeps run IDs + superseded evidence; value change refused;
  empty shell rebound then resumed; a conflict in the LAST child leaves every record and
  artifact unchanged and the original plan importable; an interrupted reconcile resumes
  the same plan only (one reconciliation log per child; the parent log keeps the first run's
  entries). A same-size content change in an archived file or bundle member is refused
  with the store unchanged.
- `--family` with a reference-parent-only dependency plans the parent but writes only the
  selected family.
- Restore check downloads through an independent scratch server and refuses port 5000.

## 7. Wrong vs Correct

```python
# Wrong: coverage = "the files that exist hash fine" (a deleted model disappears silently),
# and any file with the right digest satisfies a ledger entry.
assert ledger_digest in all_planned_hashes
# Correct: bind each ledger identity to its own directory and members.
r = by_path.get(f"{identity_dir}/{q}.ubj"); assert r and r["sha256"] == ledger_digest
```

```python
# Wrong: mutate while validating — a late child conflict strands the parent half-reconciled.
set_tag(parent, "import_fingerprint", new_fp)
for child in children: validate(child)          # raises after the parent already changed
# Correct: validate parent and every child read-only, then write; record the target first.
todo = [validate(child) for child in children]   # may raise; nothing written yet
set_tag(parent, "reconcile_target", new_fp); set_tag(parent, "import_status", "reconciling")
```

```text
Wrong: stop an importer between families with a log-watcher kill — the next family's
       create_run can land in the same second (it did: an empty yearly shell).
Correct: run one family per `import --family F` invocation when a pause point is needed;
       if a shell exists, rebind it with `reconcile` and resume under the same run ID.
```
