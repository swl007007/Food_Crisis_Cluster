# IPCCH local MLflow

A local MLflow 3.17.0 tracking server that gives one view over seven completed IPCCH
runs (six earlier families plus the AlphaEarth supervised partition, added 2026-10-14). It imports saved reports, predictions and models; nothing is retrained and no
model is loaded. Records are historical imports — MLflow did not execute these runs,
and MLflow start/metric times are import times, not source execution times.

| Item | Value |
|---|---|
| URL (WSL and Windows browser) | http://localhost:5000 (bound to 127.0.0.1 only) |
| venv | `/home/swl007007/.venvs/ipcch-mlflow` (uv, Python 3.12, `requirements.lock`) |
| Store | `/home/swl007007/.local/share/ipcch-mlflow/` — `mlflow.db` (SQLite), `artifacts/`, `logs/`, `plans/`, `cache/`, `dashboard/` |
| Source list | `sources.json` (finite: seven families) |
| Vocabulary | `naming.py` (every user-facing name, tag and description) |

Since 2026-10-09 the store uses the readable vocabulary below (task
`10-09-mlflow-readable-naming`). The previous store (experiments `IPCCH` / `IPCCH Summary`)
is kept whole at `~/ipcch-mlflow-backups/20261009-store-before-readable-naming/`;
`rebuild_check.py` compares the two value by value.

## Storage policy (research retention, 2026-10-14)

MLflow here is for research version and feature management and a temporary internal dashboard
(e.g. shared through a Cloudflare tunnel), not a production deployment or a permanent model archive.
What is kept: raw data, training/running scripts, and the small metadata and results needed for
version tracking, feature tracking and dashboard comparison.

- **No model packages by default.** `import` records each family's model files in
  `manifests/models.tar.members.json` (path, bytes, SHA256 in the source run folder) and tags
  `zz_prov.bundle.models.tar.retention` / `.local_source`; it neither builds nor uploads `models.tar`.
  A family may opt in with `"upload_model_bundles": true`.
- **Legacy model links are historical.** All earlier `models.tar` files were removed on purpose. The
  `zz_prov.model_bundle_uri`, `zz_prov.model_bundle_sha256` and `zz_prov.bundle.models.tar.*` tags and the
  "weights stay in models.tar" wording on the six older families are historical references from their
  import, **not a claim that the bundle can still be downloaded**. They are not retagged or rewritten.
- **Older families are historical metadata.** Their values, datasets, IDs and dashboard rows are unchanged
  and readable. `import`/`verify` of those families is not part of routine work: their bundles are gone and
  four source roots (MLP, climate, window probe, split2024) no longer exist, so such a run stops with a
  clear error instead of passing. Import new work with `--family NAME`.
- **Proportionate checks.** Ordinary registration needs no full-store backup or rehearsal: the importer's
  fingerprints, read-back and duplicate refusal, and the dashboard extension's inventory check, journal and
  resumable states guard the actual change. `backup_restore.py` stays available when a copy is wanted.
- **Imported extras are stable inputs only.** A family imports its code and required scientific inputs,
  not mutable task notes, progress files or receipts (they would change its fingerprint on every edit).

## Start / stop / status

```bash
IPCCHMLflow/manage.sh install     # once: write + enable the two user services, start them
IPCCHMLflow/manage.sh status      # state, health, tunnel, autostart(mlflow/tunnel)=enabled/enabled
IPCCHMLflow/manage.sh restart     # after code/patch changes (re-copies assistant_patch.py, ipcch-model)
IPCCHMLflow/manage.sh stop | start | tunnel-stop | tunnel-start
IPCCHMLflow/manage.sh uninstall   # disable + remove the services (store, auth, tunnel config kept)
```

`install` writes two enabled systemd **user** services, `~/.config/systemd/user/ipcch-mlflow.service` (MLflow on
127.0.0.1:5000) and `ipcch-cloudflared.service` (the tunnel, ordered after MLflow). They start automatically when a
WSL session starts (user manager; linger is off by choice) and restart 10 s after a crash; they stop when WSL shuts
down (`wsl --shutdown`, Windows restart, or WSL idle shutdown after the last terminal closes). Nothing starts WSL at
Windows logon. The unit files use only local paths (venv, `~/.local/share/ipcch-mlflow`, `~/.local/bin/cloudflared`),
never the Dropbox repo; `install`/`start`/`restart` first refresh the auth files, the assistant patch and the
`ipcch-model` command from the repo. `stop`/`tunnel-stop` stop the service for now only; to stop autostart use
`systemctl --user disable ipcch-cloudflared` (or `ipcch-mlflow`). Logs: `logs/server.log`, `logs/cloudflared.log`.
Without a systemd user bus, `start`/`tunnel-start` fall back to `setsid` processes with pid files.

### Login (basic-auth) and sharing through a Cloudflare tunnel

The server runs MLflow's basic-auth app (`--app-name basic-auth`). Every page and API call needs a
user/password except `/health`. Files in `auth/` (dir 700) are created at the first start:
`basic_auth.ini` (600; admin user `admin` and its initial random password), `basic_auth.db`
(users, permissions) and `server.env` (600; CSRF secret). Default permission is `READ`, so any
logged-in user can read every experiment; only the admin (or users granted EDIT/MANAGE) can write.

```bash
grep admin_password ~/.local/share/ipcch-mlflow/auth/basic_auth.ini   # initial admin password (own terminal)
```

- Change a password (as that user, or as admin for anyone):
  `curl -u admin:OLD -X PATCH http://127.0.0.1:5000/api/2.0/mlflow/users/update-password -H 'Content-Type: application/json' -d '{"username":"admin","password":"NEW"}'`.
  The ini value is used only when the user store is first created; after a change it is stale.
- Add a team member (admin only): open `http://127.0.0.1:5000/signup` (or `https://mlflow.weilunshi.com/signup`), or
  `curl -u admin:PW -X POST http://127.0.0.1:5000/api/2.0/mlflow/users/create -H 'Content-Type: application/json' -d '{"username":"alice","password":"..."}'`.
  Remove: `curl -u admin:PW -X DELETE http://127.0.0.1:5000/api/2.0/mlflow/users/delete -H 'Content-Type: application/json' -d '{"username":"alice"}'`.
- Python clients and the importer/catalog scripts need credentials now:
  `export MLFLOW_TRACKING_USERNAME=admin MLFLOW_TRACKING_PASSWORD=...` before `import_runs.py` / `summary_catalog.py`.
- Accepted Host headers: localhost, 127.0.0.1 and `mlflow.weilunshi.com` (`IPCCH_MLFLOW_PUBLIC_HOST` overrides); others get 403.

Tunnel (temporary internal sharing; `cloudflared` in `~/.local/bin`, config `~/.cloudflared/ipcch-mlflow.yml`,
named tunnel `ipcch-mlflow` -> `http://127.0.0.1:5000`, DNS `mlflow.weilunshi.com`):

```bash
IPCCHMLflow/manage.sh tunnel-start    # needs the MLflow server running; log logs/cloudflared.log
IPCCHMLflow/manage.sh tunnel-stop
```

AI assistant (MLflow 3.17 UI, top-right Assistant), usable locally and through the tunnel:

- Model: **Gemini 3.8 Flash on Vertex AI** (project `manuel-remote-sensing`, location `global`, billed there), via the
  MLflow AI Gateway endpoint `ipcch-assistant-gemini` (model definition `gemini-3.8-flash`, secret `ipcch-vertex-sa`).
  Credentials: service account `ipcch-mlflow-assistant@manuel-remote-sensing.iam.gserviceaccount.com` with
  `roles/aiplatform.user`; its JSON key exists only inside the Gateway secret, encrypted with
  `MLFLOW_CRYPTO_KEK_PASSPHRASE` from `auth/server.env` (never change that passphrase without re-entering the key).
- Assistant config `~/.mlflow/assistant/config.json`: provider `mlflow_gateway`, model `ipcch-assistant-gemini`, file
  tools off. Remote use is on (`MLFLOW_ENABLE_REMOTE_ASSISTANT=true`, sandbox off) in `auth/server.env`; changing
  assistant settings stays localhost-only. Any logged-in MLflow user can chat.
- It answers from MLflow (tags such as `features` / `maps` / `training_window`, descriptions, metrics) by running
  `mlflow` commands as the READ-only user `assistant` (granted USE on the endpoint only).
- `assistant_patch.py` (MLflow 3.17.0 only; installed into the venv by `manage.sh start`, active only in that server)
  works around four 3.17 gaps: (1) the assistant's self-calls carry no MLflow login (401 under basic-auth) and, from a
  tunnel page, would go to the public URL — they now go to http://127.0.0.1:5000 with the `assistant` login;
  (2) Gemini 3 tool calls lose their `thought_signature` in the assistant's history — the Gateway fills Google's
  placeholder `skip_thought_signature_validator`; (3) an approved prompt would run any shell command with full access
  on this host — tools are now a read-only allowlist (`mlflow experiments search|get`, `runs list|describe`,
  `artifacts list --run-id/--artifact-path`), denied calls go back to the model, no approval prompts;
  (4) `mlflow runs describe` crashes on runs with logged-model inputs — its JSON falls back to `str`. Tests:
  `tests/test_assistant_patch.py`. Re-check after any MLflow upgrade (the patch skips other versions).
- Turn remote use off: delete the two `MLFLOW_ENABLE_*ASSISTANT*` lines from `auth/server.env` and restart.
  Revoke the key: `gcloud iam service-accounts keys list --iam-account ipcch-mlflow-assistant@manuel-remote-sensing.iam.gserviceaccount.com`
  then `... keys delete KEY_ID`; rotate by creating a new key and updating the `ipcch-vertex-sa` Gateway secret
  (AI Gateway page, localhost).

Like the server, the tunnel stops when WSL shuts down. Stop it when sharing is not needed; the data is
Internal Use (unpublished research), so consider a Cloudflare Access policy in front of the hostname.

## Reading guide

Two experiments:

- **`IPCCH dashboard`** — start here. One row per family x arm x lead time (x seed); the
  MLP also has a `mean of 3 seeds` row per arm and lead. Run names read
  `<family> | <arm> | <lead>`, e.g. `GeoXGB reference | partitioned_gated | 3-month`.
- **`IPCCH detailed runs`** — one family run (long title, archived source files, manifests,
  model member manifest, the family's accepted conclusion in its description; see Storage policy for
  the historical `models.tar` references) with one child run per
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
  read back (artifacts downloaded and hashed; tar members checked when a family uploads bundles).
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

Experiment names (renamed 2026-10-14; IDs unchanged): `IPCCH detailed runs` (1), `IPCCH dashboard` (2),
`IPCCH forecasting detailed runs` (3), `IPCCH forecasting dashboard` (4). Models tab per experiment shows the logged
models of that experiment: IPCCH's 116 are in `IPCCH dashboard`; the Forecasting 126 were moved from experiment 3 to 4
on 2026-10-14 (SQL on `logged_models*.experiment_id` only; IDs, registered versions and links unchanged; DB copy
`mlflow.db.before-forecasting-model-move-20261014`). The Forecasting importer (`ipcch-forecasting-import-v1`) lives
outside this repo and still uses the old names and model placement: update it before re-importing Forecasting.

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
  Older families' descriptors name the family run's `models.tar` as a historical reference (removed; not
  downloadable); newer ones carry an empty bundle URI and point to the member manifest. Persistence has no model.
- `apply` refuses a changed plan, a stale backup or a changed detailed experiment; it is
  serial, journals new IDs under `dashboard/`, resumes the same plan, reads each row back
  before marking it complete, and sets `catalog_status=complete` last. Re-running is a no-op.

UI notes: the runs list's Models column shows logged-model outputs, so it reads "-" here; the
model appears on the run page under Logged models (Input) and per metric. The UI opens on the
GenAI overview; use **Model training → Training runs**.

### Bounded extension (AlphaEarth, task 10-10)

A completed dashboard is extended additively, never rebuilt. `sources.json` →
`dashboard_extensions.alphaearth_partition` freezes the predecessor fingerprint and plan file (SHA256), the exact
20 row / 16 model projection keys, the 20 dataset digests (12 family-qualified evaluation, 8 training pools),
the addition counts and the legacy counts (136 / 100 / 68).

```bash
# every command names BOTH the store and the tracking URI (live: ~/.local/share/ipcch-mlflow + http://127.0.0.1:5000)
$PY IPCCHMLflow/summary_catalog.py plan --extend alphaearth_partition --store STORE --tracking-uri URI
$PY IPCCHMLflow/summary_catalog.py inventory --store STORE --tracking-uri URI --out INV.json
$PY IPCCHMLflow/summary_catalog.py apply --extend alphaearth_partition --plan-fingerprint NEW_FP \
    --inventory INV.json [--backup BACKUP] --store STORE --tracking-uri URI    # backup optional (Storage policy)
$PY IPCCHMLflow/summary_catalog.py verify --extend alphaearth_partition --plan-fingerprint NEW_FP --inventory INV.json \
    --store STORE --tracking-uri URI
```

- `plan --extend` removes exactly the bound additions; the remaining legacy projection must reproduce the
  predecessor fingerprint byte-for-byte, or it refuses (no family filter can hide drift).
- `apply --extend` states: S0 predecessor complete → S1 (`zz_prov.dashboard_extension_intent`=new fp,
  `catalog_status=incomplete`) → S2 (new `zz_prov.dashboard_plan_fingerprint`, `zz_prov.dashboard_plan_lineage`,
  new description) → S3 (`complete`, intent removed). Every experiment-tag and journal write is its own crash
  boundary; a restart resumes only its own intent (`dashboard/journal-ext-<fp>.json`) and refuses a foreign intent,
  a foreign incomplete state or any object outside legacy ∪ additions. Legacy objects are deep-verified against
  the predecessor before any write and keep their IDs, tags (old fingerprint), metrics, history, inputs and bytes.
- A completed same-plan repeat deep-verifies and writes nothing.
- **Current result (2026-10-14, live store):** AlphaEarth family imported (detailed +21 runs, 1,080 metrics, ~79 MB
  uploaded: source files + code; model weights listed, not uploaded). The dashboard was extended additively:
  `c7abf4cd` → `cb3db908` (lineage tag). That added 20 rows, 16 registered models / versions / logged models,
  12 evaluation and 8 training datasets. The 136 legacy rows and the Forecasting experiments are unchanged.
- AlphaEarth evaluation datasets carry the family suffix: their saved q3 truth text differs in the last digits
  from the other families' datasets with the same keys and phases, so they are not the same dataset.

## Model cards and tags (both dashboards, since 2026-10-14)

Every logged model's description is an AI-readable model card and every logged/registered model carries the same
human tags; both are generated from `model_cards.json` (facts per family and arm, each citing its archived source;
"not recorded" when the source did not record it). `summary_catalog.py` builds the same cards for new IPCCH models;
forecasting models get them from `model_cards.py`. Format, tag list and `card.feature_list` syntax: see the
"MLflow principles" section of the repository AGENTS.md / CLAUDE.md.

```bash
export MLFLOW_TRACKING_USERNAME=admin MLFLOW_TRACKING_PASSWORD=...
$PY IPCCHMLflow/model_cards.py plan --out PLAN.json          # read-only: cards, lengths, differences, fingerprint
$PY IPCCHMLflow/model_cards.py apply --plan-fingerprint FP   # writes only descriptions and the card tags
$PY IPCCHMLflow/model_cards.py verify --plan-fingerprint FP
ipcch-model families | search --family F --lead 3 | get "<words from the name>" | features "<name>" | registered "<name>"
```

After editing `model_cards.json`, re-run plan/apply/verify; `summary_catalog.py verify` expects the current cards.
Check that each `card.feature_list` resolves to exactly `n_features` names (`ipcch-model features`).

## Rebuild check

```bash
$PY IPCCHMLflow/rebuild_check.py --old OLD/mlflow.db --new ~/.local/share/ipcch-mlflow/mlflow.db --out R.json
```

Read-only on both stores: every old detailed and Summary value is found with the identical
value under its new name, every other dashboard value is a seed mean or a contrast equal to
its detailed source, and every dataset name has one digest.

## Adding a future run

1. Add one family entry to `sources.json` (root, report, prediction pattern, arms with arm
   kinds, include/bundle/exclude globs, tags, `inventory` policy). `bundle` globs are listed by member
   manifest only (no upload); `extra` holds code and required inputs, not task notes or receipts.
2. Add the family to `naming.py` (`FAMILIES`, `ARMS`, period spans, `TRAINING_POOL`, accepted
   conclusion and status text). New report fields need a naming rule; `plan` refuses unnamed
   metrics.
3. If its report schema is new, add an extractor in `extract.py` and a fixture test.
4. `plan`, check the plan's `unlisted` and `excluded` lists, then `import --family NAME` and
   `verify --family NAME`; then re-plan and apply the dashboard (update `EXPECTED`).
5. To add it to a completed dashboard without rebuilding, freeze a `dashboard_extensions` entry and use
   `--extend` (see Bounded extension). A family whose source has no per-file ledgers may use the strict
   inventory mode (`strict_bindings`: every digest leaf, bundle record JSON included, matches exactly one scoped
   own / external / typed / historical rule); the AlphaEarth family also binds its native CSV fit ledgers one-to-one
   to the schedule and to its own bundle directories (`model_contract: alphaearth_xgb`).

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
`manifests/plan-summary.json` and the smallest `models.tar` present (if any) through it, and stops it.
Backups are optional under the Storage policy. To
actually restore: stop the server, replace `mlflow.db` and `artifacts/` with the backup
copies, start the server.

## Tests

```bash
$PY -m unittest discover -s IPCCHMLflow/tests -v
```
