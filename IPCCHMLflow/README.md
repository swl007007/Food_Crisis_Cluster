# IPCCH local MLflow

A local MLflow 3.17.0 tracking server that gives one view over six completed IPCCH
runs. It imports saved reports, predictions and models; nothing is retrained and no
model is loaded. Records are historical imports — MLflow did not execute these runs,
and MLflow start/metric times are import times, not source execution times.

| Item | Value |
|---|---|
| URL (WSL and Windows browser) | http://localhost:5000 (bound to 127.0.0.1 only) |
| venv | `/home/swl007007/.venvs/ipcch-mlflow` (uv, Python 3.12, `requirements.lock`) |
| Store | `/home/swl007007/.local/share/ipcch-mlflow/` — `mlflow.db` (SQLite), `artifacts/`, `logs/`, `plans/`, `cache/`, `dashboard/` |
| Source list | `sources.json` (finite: six families) |
| Vocabulary | `naming.py` (every user-facing name, tag and description) |

Since 2026-10-09 the store uses the readable vocabulary below (task
`10-09-mlflow-readable-naming`). The previous store (experiments `IPCCH` / `IPCCH Summary`)
is kept whole at `~/ipcch-mlflow-backups/20261009-store-before-readable-naming/`;
`rebuild_check.py` compares the two value by value.

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

If the WSL login is not registered with logind (`loginctl list-sessions` shows nothing, and
`systemctl --user` fails with `Failed to connect to bus`), `manage.sh` falls back to a `setsid`
background process tracked by `server.pid` in the store root; `status` then shows
`active(pid N)`. Same no-autostart / no-linger behaviour.

## Reading guide

Two experiments:

- **`IPCCH - dashboard`** — start here. One row per family x arm x lead time (x seed); the
  MLP also has a `mean of 3 seeds` row per arm and lead. Run names read
  `<family> | <arm> | <lead>`, e.g. `GeoXGB reference | partitioned_gated | 3-month`.
- **`IPCCH - detailed runs`** — one family run (long title, archived source files, manifests,
  `models.tar`, the family's accepted conclusion in its description) with one child run per
  arm x lead x seed holding every saved metric: deltas, bootstrap intervals, gate-decision
  subsets, coverage and route counts.

Every run, model and registered model has a description (run page "Description"): what it
is, what to compare it with, acceptance status, caveats and the original source.

### Vocabulary

| Tag | Values |
|---|---|
| `family` / `family_title` | `geoxgb_reference` GeoXGB reference (monthly refit, maps learned on 2014-2022) · `geoxgb_yearly_refit` GeoXGB yearly refit (full history, 24-month decay, reference maps) · `geoxgb_climate_swap` GeoXGB climate swap (rich601 climate inputs, maps relearned on 2014-2022) · `geoxgb_maps_2024` GeoXGB maps to 2024 (maps relearned on 2014-2024, evaluated 2025-2026) · `geoxgb_window_probe` GeoXGB window probe (36-month vs full-history training window, selected months) · `mlp_residual_fixed_maps` MLP residual on fixed maps (global MLP + residual correction on the reference maps, 3 seeds) |
| `arm` | `persistence` last observed phase, no model · `pooled` one model for all areas · `partitioned_gated` regional model where the historical gate accepted it, pooled elsewhere · `regional_ungated` the regional model's own prediction wherever one was fitted (diagnostic) · `global_base` MLP without residual model. Comparison panels from the reference run: `reference_<arm>` |
| `arm_role` | `baseline` not trained in this run · `candidate` trained in this run, main comparison · `diagnostic` trained for diagnosis only |
| `window` | window probe only: `36_month`, `full_history` |
| `lead_months` | `01`, `03`, `06`, `12` (forecast origin = target month minus lead); param `lead_months` is the number |
| `seed` | `42` (`43`, `44` for the MLP), `none` (persistence), `mean` (MLP mean rows) |
| `aggregation` | dashboard only: `single_run`, `single_seed` (one MLP seed), `mean_of_3_seeds` |
| `status` | `supervisor_accepted`, `user_accepted`, `exploratory` (full wording in the description) |
| `period.<role>` | actual scored target-month span, e.g. `period.primary = 2023-04..2025-10` |
| `features`, `maps`, `training_window`, `model_type` | design facts of the family |

Metric keys read `<period_role>.<cohort>.<metric>`:

- **period_role**: `primary` (main evaluation period of the family) · `holdout` (2026-01..2026-04,
  4 months, point estimates) · `combined` and `year_2025` / `year_2026` (GeoXGB maps to 2024
  only; the year blocks come from a recomputed report and match primary / holdout to
  floating-point precision) · `selected_months` and `target_month_YYYY-MM` (window probe;
  selected months only, not a full period). Scored primary spans: 2023-02/04/07 or 2024-01
  (1/3/6/12-month) to 2025-10; GeoXGB maps to 2024: 2025-02/04/07..2025-10, nothing at 12-month.
- **cohort**: `all_scored` every scored area x target month · `persistence_available` rows with
  an IPCCH observation at or before the forecast origin (all comparisons with persistence use
  this) · `regional_model_fitted` rows whose region got a regional model (>= 500 rows, 50 areas,
  6 months of training data), whatever the gate decided · `regional_model_fitted_and_persistence_available`
  · `in_partition_map` · `regional_model_fitted_both_windows` and
  `regional_model_fitted_full_history_only` (window probe) · detailed runs only, below
  `regional_model_fitted`: `gate_used_regional`, `gate_rejected_no_gain`,
  `gate_rejected_too_little_validation`.
- **metric**: `binary.*` crisis (phase 3+) vs not (`accuracy|precision|recall|f1|f2`, counts
  `binary.count.tp|fp|fn|tn` in detailed runs) · `four_class.accuracy|macro_f1` over phases
  1|2|3|4-5 · `share_phase3plus_r2` (and `_raw`) R2 of the population share in phase 3+ ·
  `n_rows`.
- **contrasts**: dashboard `*.binary.f1.minus_persistence` / `minus_pooled` (+ `.ci_low` /
  `.ci_high` where a 95% country-cluster bootstrap was saved) = crisis-F1 difference of the
  row's arm on the same rows. Detailed runs: `*.delta.<A>_minus_<B>.<metric>` and
  `*.bootstrap.<A>_minus_<B>.binary.f1.delta|ci_low|ci_high|countries|draws|...`;
  `*_minus_reference_*` compares with the matching GeoXGB reference arm on the same rows.

### Comparability

Each dashboard metric is bound to the evaluation dataset of its period and cohort (run page
Inputs, and per metric). Dataset names state lead, span and cohort, e.g.
`IPCCH eval | 3-month | 2023-04..2025-10 | persistence_available`; one name always means one
content digest (keys + truth). Cohorts that depend on a family's own regional fits carry the
family in the name. Compare values only on the same dataset. GeoXGB maps to 2024 has a
different primary period (2025 only); the window probe covers selected months only. No global
best-model ranking is implied. Training inputs are `IPCCH training pool | rich561|rich601 | <lead>`:
the prepared feature matrix + key table every refit draws from (the pool, not the exact rows
of one refit).

### Copy-paste filters

- One lead, one row per family x arm (MLP as seed means):
  `tags.lead_months = '03' AND tags.aggregation != 'single_seed'`.
- One family: `tags.family = 'geoxgb_reference'`.
- Candidates only: `tags.arm_role = 'candidate'`.
- Chart suggestions: `primary.persistence_available.binary.f1` grouped by `arm`;
  `primary.persistence_available.binary.f1.minus_persistence` with its `ci_low` / `ci_high`;
  `primary.all_scored.binary.f1.minus_pooled`.

Provenance (hashes, fingerprints, original IDs and the old report names) is kept in tags with
the prefix `zz_prov.`. The run page lists tags in the order they were written, so every run
is created with its tags in key order: readable tags first, `zz_prov.*` last (tag names allow
only letters, digits, `_ - . / space`, so `zz_` is the shortest prefix that sorts last). Each detailed child also has
`view/evaluation_view.json` (every value with its readable name, original name and source JSON
path) and `view/na.json` (undefined values with reasons; they are never logged as metrics).
Each dashboard row has `dashboard/row.json` (values, sources, datasets, models).

## Import and verify

```bash
PY=/home/swl007007/.venvs/ipcch-mlflow/bin/python
$PY IPCCHMLflow/import_runs.py plan   [--rehash] [--evidence DIR]   # read-only
$PY IPCCHMLflow/import_runs.py import [--family NAME ...]            # server must be running
$PY IPCCHMLflow/import_runs.py verify [--shallow]
```

- `plan` hashes every source file (stat-keyed cache in `cache/hashes.json`; `--rehash`
  ignores it), reconciles the files against the run's own saved inventories
  (`inventory.py`, per-family `inventory` policy in `sources.json`), extracts the
  evaluation views (`extract.py`, source names), translates every name with `naming.py`
  (refusing any name without a rule and any two names that would collide), and writes full
  plans to `plans/`. Every `*sha256` key in a source's ledgers/manifests needs an explicit
  decision; model stores must satisfy their per-identity contract (XGB: own `q*.ubj` with the
  ledger digest + `record.json`; MLP: `record.json` + `state.pt` + the referenced transform).
- `import` creates or resumes records serially. A record is marked
  `zz_prov.import_status=complete` only after its metrics/params/tags and artifacts have been
  read back (artifacts downloaded and hashed; tar members checked).
- Rerunning with unchanged sources is a verified no-op. `verify --shallow` is metadata-only.
- If a source, the vocabulary or the importer changed, the fingerprint differs and `import`
  stops. (The one-time `reconcile` command of the old store was removed with the rebuild.)
- `--family F` also plans F's read dependencies (`shared_inputs.parent`,
  `inventory.reference_parents`) but writes/verifies only F.
- A file lock (`import.lock`) refuses concurrent imports.

Excluded on purpose (listed with path, bytes and SHA256 in `manifests/excluded.json`):
large prepared feature matrices (`X_rich*.npy`) and replay re-execution duplicates. MLP
inputs and yearly staged inputs are the reference run's prepared files and are referenced to
it (`manifests/shared.json`).

## Dashboard, Models and Inputs

```bash
$PY IPCCHMLflow/summary_catalog.py plan                               # read-only; prints the plan fingerprint
$PY IPCCHMLflow/summary_catalog.py inventory --out INV.json             # detailed experiment snapshot
$PY IPCCHMLflow/backup_restore.py backup --dest BACKUP                  # fresh backup (required)
$PY IPCCHMLflow/summary_catalog.py apply --plan-fingerprint FP --inventory INV.json --backup BACKUP
$PY IPCCHMLflow/summary_catalog.py verify --plan-fingerprint FP --inventory INV.json
```

- 136 dashboard rows: 120 copied from the detailed child runs plus 16 MLP seed means (mean of
  the three saved values; a metric is averaged only when every seed has it on the same
  evaluation dataset; intervals are never averaged).
- Models: 68 registered models `IPCCH <family> | <arm> | <lead>` with 100 versions (one per
  seed; version numbers are MLflow allocation), each pointing to an external LoggedModel
  `<family> | <arm> | <lead> | seed N` whose source run is the detailed child run.
  External = catalog descriptor only: **not loadable, no inference, nothing retrained**.
  Weights stay in the family run's `models.tar`. Persistence has no model.
- `apply` refuses a changed plan, a stale backup or a changed detailed experiment; it is
  serial, journals new IDs under `dashboard/`, resumes the same plan, reads each row back
  before marking it complete, and sets `catalog_status=complete` last. Re-running is a no-op.

UI notes: the runs list's Models column shows logged-model outputs, so it reads "-" here; the
model appears on the run page under Logged models (Input) and per metric. The UI opens on the
GenAI overview; use **Model training → Training runs**.

## Rebuild check

```bash
$PY IPCCHMLflow/rebuild_check.py --old OLD/mlflow.db --new ~/.local/share/ipcch-mlflow/mlflow.db --out R.json
```

Read-only on both stores: every old detailed and Summary value is found with the identical
value under its new name, every other dashboard value is a seed mean or a contrast equal to
its detailed source, and every dataset name has one digest.

## Adding a future run

1. Add one family entry to `sources.json` (root, report, prediction pattern, arms with arm
   kinds, include/bundle/exclude globs, tags, `inventory` policy).
2. Add the family to `naming.py` (`FAMILIES`, `ARMS`, period spans, `TRAINING_POOL`, accepted
   conclusion and status text). New report fields need a naming rule; `plan` refuses unnamed
   metrics.
3. If its report schema is new, add an extractor in `extract.py` and a fixture test.
4. `plan`, check the plan's `unlisted` and `excluded` lists, then `import --family NAME` and
   `verify --family NAME`; then re-plan and apply the dashboard (update `EXPECTED`).

## Backup and restore check

Stop imports first (the backup takes the import lock).

```bash
$PY IPCCHMLflow/backup_restore.py backup --dest ~/ipcch-mlflow-backups/$(date +%Y%m%d)
$PY IPCCHMLflow/backup_restore.py restore-check --backup ~/ipcch-mlflow-backups/YYYYMMDD --dest /tmp/ipcch-mlflow-restore
```

`backup` uses the SQLite online backup API plus a byte copy of `artifacts/` and writes
`backup-manifest.json` (DB hash and row counts, per-file artifact hashes). `restore-check`
copies a backup into a new scratch root, checks every hash and row count, starts a temporary
server on its own port (default 5001; 5000 is refused), downloads every family run's
`manifests/plan-summary.json` and the smallest `models.tar` through it, and stops it. To
actually restore: stop the server, replace `mlflow.db` and `artifacts/` with the backup
copies, start the server.

## Tests

```bash
$PY -m unittest discover -s IPCCHMLflow/tests -v
```
