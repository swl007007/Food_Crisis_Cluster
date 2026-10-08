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
6. [~] Import. P6 (8ad47dc9…, 13 records) and MLP (5ba96181…, 53 records) imported and
   deep-verified at import. Paused after MLP per supervisor checkpoint review of 8c88f48;
   an empty yearly parent shell (c19cd675…, in_progress, 0 metrics/params/artifacts) was
   created in the 1 s before the kill.
6a. [x] Supervisor fixed list (evidence/supervisor-checkpoint-8c88f48.md), commit a9a2598:
   original-inventory reconciliation (inventory.py + per-family policy), scratch-server
   restore check, deep verified no-op, by_gate key binding (MLP 142 / yearly 42 subsets),
   split E_persist deltas (11 blocks), metric-name collision guard. Tests 15/15.
6b. [x] Reconcile command prepared and tested (e44128e, 18/18); supervisor isolated run
   18/18 (evidence/supervisor-e44128e-tests.log). Delta review item-1 gap closed in
   ec2686e: exact per-identity model contracts (XGB own boosters; MLP record+state+
   transform). Tests 22/22 (evidence/tests-item1-fix.log). Note SHA 75e9989.
7. [x] Refreshed diff (evidence/reconciliation-diff-ec2686e.json) within the release bound
   (0 metric/param/source/model changes). Applied: reconcile 22:53-22:59 (27 records:
   P6+MLP parents, 24 MLP children, yearly shell rebound), import 23:00-23:10 (yearly
   resumed, climate/window/split created), deep verify 23:10-23:19 (126 records, 20,384
   child metrics, 1,987 artifacts, 8.33 GB, 44,203 tar members), verified no-op import
   23:19-23:26 (126 noop, 0 writes). API: 126 runs, all complete/FINISHED, no duplicate
   source keys. Windows curl.exe download of window models.tar via localhost:5000:
   bytes and SHA (WSL + PowerShell) match. UI rendered in headless Edge (screenshots).
8. [x] Backup (SQLite backup API + artifacts, 2,039 files, integrity ok) and scratch
   restore check through an independent server on port 5001 (7 downloads matched).
   env-after.json: all three frozen environments identical to env-before.json.
   Cache-free rehash of 46,468 source files: all six plan fingerprints unchanged.
9. [ ] Final evidence sent to supervisor; hold close/push/merge for acceptance.
