# Local MLflow Import of Completed Runs

Contract for `IPCCHMLflow/` (task 10-07-local-mlflow-experiment-management, accepted at
efc5d11; names, tags and descriptions replaced by the readable vocabulary of `naming.py` and
the store rebuilt in task 10-09-mlflow-readable-naming). Use it when adding a completed run
family to the local MLflow store or when changing extraction/naming/inventory/verification
code.

## 1. Scope / Trigger

- Trigger: infra integration (SQLite tracking store, proxied artifact store, localhost
  service) plus a cross-run provenance contract read by later comparisons.
- MLflow holds **historical imports**. It never retrains, loads models or recomputes
  scores; it reads saved reports/predictions/ledgers and copies retained artifacts.

## 2. Signatures

```bash
PY=/home/swl007007/.venvs/ipcch-mlflow/bin/python   # uv venv, MLflow 3.17.0, requirements.lock
IPCCHMLflow/manage.sh start|stop|status             # user unit or setsid fallback, 127.0.0.1:5000
$PY IPCCHMLflow/import_runs.py plan [--rehash] [--family F] [--evidence DIR]
$PY IPCCHMLflow/import_runs.py import [--family F]
$PY IPCCHMLflow/import_runs.py verify [--shallow]
$PY IPCCHMLflow/rebuild_check.py --old OLD/mlflow.db --new NEW/mlflow.db --out R.json
$PY IPCCHMLflow/backup_restore.py backup --dest DIR
$PY IPCCHMLflow/backup_restore.py restore-check --backup DIR --dest SCRATCH [--port 5001]
```

Store: `/home/swl007007/.local/share/ipcch-mlflow/{mlflow.db,artifacts/,plans/,cache/,logs/}`.
Pre-rebuild store: `~/ipcch-mlflow-backups/20261009-store-before-readable-naming/`.

## 3. Contracts

- Experiment `IPCCH - detailed runs`. Parent `record_kind=family` per source run (run name =
  long family title; artifacts kept once); child `record_kind=evaluation` per family x lead x
  arm x seed (run name `<family short title> | <arm> | <lead>[ | <window>][ | seed N]`),
  linked by `mlflow.parentRunId`. Persistence views use `seed=none`.
- Vocabulary lives only in `naming.py`: family slug/title, arm (`persistence`, `pooled`,
  `partitioned_gated`, `regional_ungated`, `global_base`; `reference_<arm>` for reference-run
  panels), `arm_role`, `window`, `lead_months` (`03`), period roles (`primary`, `holdout`,
  `combined`, `year_2025|2026`, `selected_months`, `target_month_YYYY-MM`) with actual scored
  spans, cohorts (`all_scored`, `persistence_available`, `regional_model_fitted`, ...). Visible
  tags are this vocabulary plus `status`, `features`, `maps`, `training_window`,
  `period.<role>` and the description (`mlflow.note.content`); everything else (hashes,
  fingerprints, original IDs and names, source tags) is under `zz_prov.`.
- Runs are created with their full tag set in key order (`import_runs.ordered`): the MLflow run
  page lists tags in insertion order, so readable tags come first and `zz_prov.*` last. The
  test reads the SQLite `tags` rowid order.
- Identity tags: `zz_prov.source_key` (stable extractor key), `zz_prov.import_fingerprint`
  (sha256 over plan content incl. readable names + importer version), `zz_prov.import_status`
  in {`in_progress`, `complete`}.
- `extract.py` keeps the source reports' own names; `naming.metric_name` translates them and
  refuses any name without a rule; `check_one_to_one` refuses two source names that land on
  one readable name in a record, and an NA name equal to a logged metric. Only finite source
  values are metrics; undefined values go to `view/na.json` (readable name, original name,
  source path, reason). `view/evaluation_view.json` maps every value to its original name and
  source JSON path.
- Cohort identity: `zz_prov.cohort_keys.<period_role>.<cohort>` = sha256 of sorted
  `admin_code|target_ord`; its key count must equal the reported n. Gate subsets are derived
  with the source report's own `gate_category` rule.
- Source inventory policy (`sources.json` -> `inventory`): every `*sha256` key in the run's
  own ledgers/manifests has an explicit decision — `required_digest_keys`, `name_keys`,
  `informational_keys`, or a scoped `exempt` (key + JSON-path regex [+ file regex] +
  reason) — plus `model_contract` (`xgb` or `mlp`).
- Provenance tags `zz_prov.execution` and `zz_prov.mlflow_timestamps` state that MLflow times are
  import times. `status` (short), `zz_prov.source.scientific_acceptance`,
  `zz_prov.source.lifecycle_status` and `zz_prov.import_status` are separate fields; the family
  description quotes the accepted conclusion and full status wording.

## 4. Validation & Error Matrix

| Condition | Result |
|---|---|
| reported cohort n != prediction key count (incl. gate subsets) | `SourceConflict` at plan |
| two different source paths map to one source metric name | `SourceConflict` (only panel-n vs cohort-n is an equal-value cross-check) |
| metric / NA entry / namespace / extractor tag without a naming rule | `SourceConflict` at plan |
| two source names map to one readable name in a record | `SourceConflict` at plan |
| `*sha256` key without an inventory decision | `SourceConflict` |
| required digest matches no planned file (own or reference parent) | `SourceConflict` |
| XGB identity lacks its own `q*.ubj` with the ledger digest, or `record.json` | `SourceConflict` |
| MLP identity lacks `record.json`, `state.pt` or its referenced transform | `SourceConflict` |
| existing record has a different fingerprint (source, vocabulary or importer changed) | `import` stops; rebuild into a fresh store (old store kept whole) |
| `--family F` | plans F plus transitive read dependencies (`shared_inputs.parent`, `inventory.reference_parents`), dependencies first; writes/verify only F; unknown family refused |
| equal-size artifact corruption | caught by deep verify, including the repeat-import no-op |
| second importer while one runs | refused by `import.lock` |
| restore-check on port 5000 | refused; scratch checks use their own server |

## 5. Good / Base / Bad Cases

- Good: plan shows 0 unlisted files; every exemption names a reason; import marks
  `complete` only after deep readback; repeat import = all `noop`, 0 writes; after a
  vocabulary change, `rebuild_check.py` finds every old value unchanged under its new name.
- Base: path-level-only coverage where the source stores no file digests (MLP states and
  transforms), documented in the policy `note`.
- Bad: counting surviving files as coverage; reporting ledger reference occurrences as
  unique models; renaming by hand in the UI (logged-model and dataset names cannot be
  renamed; the next import would conflict).

## 6. Tests Required

`$PY -m unittest discover -s IPCCHMLflow/tests -v` (41 tests). Assertion points:
- NA not logged and present in `na.json` under its readable name; subset cohort digests
  differ; count mismatch refused.
- Interrupted import resumes without duplicates (two interruption points); idempotent no-op.
- Missing/changed retained booster refused; own booster required even if another
  directory holds the same digest; MLP missing `state.pt` / referenced transform refused.
- Unclassified digest key refused; name collision refused; gate subsets bound to routes.
- Naming: every old metric pattern maps (panels, gates, reference panels, deltas, bootstrap,
  coverage/routes, seed summaries, window cells); unknown names refused; one-to-one per
  record; run/model names respect MLflow character rules; dataset names carry actual spans.
- `--family` with a reference-parent-only dependency plans the parent but writes only the
  selected family.
- Rebuild check: equal values pass; a changed value, an unexplained dashboard value or a
  dataset name with two digests is reported.
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
# Wrong: one tag/arm name with different meanings per family ("pool" = pooled XGB in one,
# MLP base + pooled residual in another), or codes in names ("E_persist", "H6", "p6geo").
# Correct: role-based names from naming.py, meaning in the description, code in zz_prov.
naming.arm("mlp_fixed_map", "pool")   # ('pooled', None, 'candidate') + MLP_ARM_MEANING text
```

```text
Wrong: "_prov." prefix and identity tags passed to create_run first -- the run page lists tags in
       insertion order, so provenance showed at the top ('_' also sorts before 'a').
Correct: create_run(tags=ordered(full_tags)) with prefix "zz_prov."; later batches are sorted too.
```

```text
Wrong: stop an importer between families with a log-watcher kill — the next family's
       create_run can land in the same second (it did: an empty yearly shell).
Correct: run one family per `import --family F` invocation when a pause point is needed.
```
