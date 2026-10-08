# PROGRESS — local MLflow experiment management

Operational log only; approval authority is the supervisor (wN:p1) and the user.

## Lifecycle

- Yearly GeoXGB task closed via `trellis-audit close` (run 252b7ae5…; close-audit job
  2020a1286705955fcfe26e75 queued, not passed). Archive commit 4820a28.
- MLflow task started by the controller: run a75c2b1747d54966ba84671c3cb67fdc, base 4820a28,
  executor 174ea213 / term_65d3f9b51d7fa2.

## Steps

1. [x] Environment snapshot before install: `evidence/env-before.json` (Windows Store
   python 242 pkgs, Windows MLP venv 22, WSL system python 282; none has MLflow).
2. [x] uv venv `/home/swl007007/.venvs/ipcch-mlflow` (Python 3.12), MLflow 3.17.0;
   `IPCCHMLflow/requirements.lock` (91 lines).
3. [x] `sources.json` (six families), `manage.sh`, `extract.py`, `import_runs.py`,
   `backup_restore.py`, `README.md`, tests (7 pass: NA not logged, subset cohort +
   count check, conflicting identity refused, two interruption points resumed, idempotent
   no-op, concurrent lock refused).
4. [x] Server started (`manage.sh start`, transient user unit, 127.0.0.1:5000); /health OK
   and API reachable from WSL curl and Windows curl.exe (localhost:5000, UI HTML 200).
5. [x] Read-only plan (`evidence/plan/`): 126 records = 6 parents + 120 views; 20,285
   finite child metrics, 174 NA entries; every reported cohort n equals its prediction key
   count; MLP persistence identical across seeds 42/43/44; 0 unlisted files in any root.
   Fix during planning: one route-reason key contained `:`; leaf names are sanitised to
   `_` (source path kept in provenance). Yearly `inputs/source/**` (3 staged P6 config
   files, 42 KB) added to the archived set.
6. [ ] Import six families incrementally (each record verified before `complete`).
7. [ ] Full `verify`, idempotent rerun, Windows artifact download, UI check.
8. [ ] Backup + scratch restore; env snapshot after install.
9. [ ] Evidence to supervisor; hold close/push/merge.
