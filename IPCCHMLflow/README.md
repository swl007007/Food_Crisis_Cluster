# IPCCH local MLflow

A local MLflow 3.17.0 tracking server that gives one view over six completed IPCCH
runs. It imports saved reports, predictions and models; nothing is retrained and no
model is loaded. Records are historical imports — MLflow did not execute these runs,
and MLflow start/metric times are import times, not source execution times.

| Item | Value |
|---|---|
| URL (WSL and Windows browser) | http://localhost:5000 (bound to 127.0.0.1 only) |
| venv | `/home/swl007007/.venvs/ipcch-mlflow` (uv, Python 3.12, `requirements.lock`) |
| Store | `/home/swl007007/.local/share/ipcch-mlflow/` — `mlflow.db` (SQLite), `artifacts/`, `logs/`, `plans/`, `cache/` |
| Source list | `sources.json` (finite: six families) |

## Start / stop / status

```bash
IPCCHMLflow/manage.sh start    # transient user unit "ipcch-mlflow"; waits for /health
IPCCHMLflow/manage.sh status
IPCCHMLflow/manage.sh stop
```

The server is started on demand with `systemd-run --user`. It is not enabled at boot and
user linger is not changed, so **it stops when WSL shuts down** (`wsl --shutdown`, Windows
restart, or WSL idle shutdown after the last terminal closes). Run `manage.sh start` again
afterwards; the data in the store is persistent. Log: `logs/server.log`.

## Layout in the UI

One experiment, `IPCCH`, with 126 records:

- 6 parent records (`record_kind=source_run`), one per source run (P6 `8ad47dc9…`,
  MLP `5ba96181…`, yearly `c19cd675…`, climate `a3cb329b…`, window `13b03ec1…`,
  split2024 `116eadde…`). These hold the retained
  artifacts once: `source/…` (reports, configs, predictions, ledgers), `models.tar`
  (uncompressed, deterministic, per-member manifest in `manifests/models.tar.members.json`),
  `source_code/…`, `task_evidence/…`, and `manifests/` (included, excluded, shared and
  unlisted files with bytes and SHA256).
- 120 child records (`record_kind=evaluation_view`), one per family × horizon × arm × seed,
  nested under their parent. Persistence views have `seed=none`; MLP persistence is stored
  once per horizon after checking it is identical across seeds.

Filter with tags such as `tags.family = 'yearly_geoxgb'`, `tags.arm = 'geo'`,
`tags.arm_kind = 'persistence_baseline'`.

`arm_kind` distinguishes `fresh_trained`, `persistence_baseline`, `diagnostic_local`
(ungated local predictions) and `reused_comparator` (window-probe base arms).

### Metric names

`{period}.{cohort}[.{group}].{metric}`, for example `main.E_all.binary.f1` or
`supplementary.E_persist.four_class.macro_f1`.

- period: `main`, `supplementary`; split2024 adds `combined`, `y2025`, `y2026`; the window
  probe uses `selected_dates` (aggregate over its 11 dates) and `selected_date_YYYY-MM` —
  these are not full-period results.
- cohort: `E_all`, `E_persist` (persistence-available keys), `local_eligible[.{gate}]`,
  `local_persist_matched` (yearly), window cohorts `all|mapped|common_local_support|new_local_support`.
- group: `delta.{a_minus_b}`, `contrast.{name}` (country cluster bootstrap: `point_delta`,
  `ci_lower`, `ci_upper`, `K`, `defined_draws`; pointwise, conditional on saved predictions),
  `comparator.{p6geo|p6pool}` (P6 panels recomputed on the same keys in MLP/yearly),
  `matched_old.p6{geo|pool}` (climate/split2024 comparisons with the old P6 panels).

The same metric name in two records does **not** mean the same evaluation keys. Each
record tags `cohort_keys.{period}.{cohort}` with the SHA256 of its sorted
`admin_code|target_ord` key set; compare those digests before comparing numbers.

Undefined values (empty cohorts, absent classes, undefined deltas) are never logged as
metrics. Each child has `view/na.json` (metric, source path, reason) and
`view/evaluation_view.json` (every logged value with its source JSON path, plus the raw
source panels).

Source scientific acceptance, lifecycle/audit status and import completion are separate
tags: `source.scientific_acceptance`, `source.lifecycle_status`, `import_status`.

## Import and verify

```bash
PY=/home/swl007007/.venvs/ipcch-mlflow/bin/python
$PY IPCCHMLflow/import_runs.py plan   [--rehash] [--evidence DIR]   # read-only
$PY IPCCHMLflow/import_runs.py import [--family NAME ...]            # server must be running
$PY IPCCHMLflow/import_runs.py verify [--shallow]
$PY IPCCHMLflow/import_runs.py reconcile                           # one-time, see below
```

- `plan` hashes every source file (stat-keyed cache in `cache/hashes.json`; `--rehash`
  ignores it), reconciles the files against the run's own saved inventories
  (`inventory.py`, per-family `inventory` policy in `sources.json`), extracts the
  evaluation views, and writes full plans to `plans/`. Every `*sha256` key in a source's
  ledgers/manifests needs an explicit decision (required file digest, bundle name,
  informational, or scoped exemption with a reason); model stores must satisfy their
  per-identity contract (XGB: own `q*.ubj` with the ledger digest + `record.json`;
  MLP: `record.json` + `state.pt` + the referenced transform). The result is archived
  per parent as `manifests/original-inventory-check.json`.
- `import` creates or resumes records serially. A record is marked
  `import_status=complete` only after its metrics/params/tags and artifacts have been read
  back (artifacts downloaded and hashed; tar members checked).
- Rerunning with unchanged sources is a verified no-op (all artifacts downloaded and
  hashed again). `verify --shallow` is metadata-only (values, tags, names, sizes).
- If a source or the importer changed, the fingerprint differs and `import` stops.
  `reconcile` is the explicit one-time path: it refuses any change to a logged metric,
  param, existing tag or archived source/model artifact; it may add tags and replace
  manifests/view JSON, keeping the old copies under `manifests/superseded/` or
  `view/superseded/` with a reconciliation log, and records `import_fingerprint.previous`.
  Nothing is deleted and run IDs are kept. The parent and every child are validated
  read-only first; any refused change leaves the whole family untouched. If a reconcile is
  interrupted, the parent stays `import_status=reconciling` with `reconcile_target`;
  `import` then stops, and `reconcile` resumes only with that same plan. Used once on
  2026-10-07 (27 records; see the task evidence).
- `--family F` also plans F's read dependencies (`shared_inputs.parent`,
  `inventory.reference_parents`, e.g. P6 for the window probe) but writes/verifies only F.
- An interrupted import leaves `import_status=in_progress`; rerun `import` to resume.
- A file lock (`import.lock`) refuses concurrent imports.

Excluded on purpose (listed with path, bytes and SHA256 in `manifests/excluded.json`):
large prepared feature matrices (`X_rich*.npy`) and replay re-execution duplicates. MLP
inputs and yearly staged inputs are the P6 prepared files and are referenced to the P6
parent (`manifests/shared.json`). Retained artifacts stay readable if the Temp run
folders are deleted; retraining would still need the original inputs.

## Adding a future run

1. Add one family entry to `sources.json` (root, report, prediction pattern, arms with
   arm kinds, include/bundle/exclude globs, tags, and an `inventory` policy: a decision
   for every `*sha256` key its ledgers/manifests carry, plus `model_contract`).
2. If its report schema is new, add an extractor in `extract.py` and a fixture test.
3. `plan`, check the plan's `unlisted` and `excluded` lists, then `import --family NAME`
   and `verify --family NAME`.

## Backup and restore check

Stop imports first (the backup takes the import lock).

```bash
$PY IPCCHMLflow/backup_restore.py backup --dest ~/ipcch-mlflow-backups/$(date +%Y%m%d)
$PY IPCCHMLflow/backup_restore.py restore-check --backup ~/ipcch-mlflow-backups/YYYYMMDD --dest /tmp/ipcch-mlflow-restore
```

`backup` uses the SQLite online backup API plus a byte copy of `artifacts/` and writes
`backup-manifest.json` (DB hash and row counts, per-file artifact hashes).
`restore-check` copies a backup into a new scratch root, checks every hash and row count,
starts a temporary server on its own port (default 5001; 5000 is refused) over the scratch
DB and artifacts, downloads every parent's `manifests/plan-summary.json` and the smallest
`models.tar` through it, compares hashes with the backup manifest, and stops it. To actually restore: stop the server, replace
`mlflow.db` and `artifacts/` in the store with the backup copies, start the server.

## Tests

```bash
$PY -m unittest discover -s IPCCHMLflow/tests -v
```

## UI notes

The UI opens on the GenAI overview; use **Model training → Training runs** (or
`#/experiments/1/runs`). Parents are nested with their views; the oldest parent (P6) is on
the next page ("Load more"). The Duration column is import time, not source run time. The
assistant side panel is a built-in UI feature and is not configured or used here.
