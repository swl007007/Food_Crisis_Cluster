# Local IPCCH MLflow — design v1.0

Approved design: user accepted the final summary with “可以执行” on 2026-10-07. R6–R11 and the complete execution scope are released, subject to the recorded lifecycle transition. No installation, import or service startup has occurred at approval.

## Outcome and boundaries

Install a local MLflow UI/API for browsing the six recent IPCCH studies, comparing compatible scores and downloading retained artifacts after their original Temp directories cease to exist. Historical import only: no scientific reruns, autologging changes, model conversion, model serving or remote/cloud store. Original artifacts and frozen numerical environments remain unchanged. A completed MLflow import is not scientific/audit acceptance.

## Host and installation

- New isolated venv: `/home/swl007007/.venvs/ipcch-mlflow`, WSL Python3.12, no system site packages. Use installed `uv`; pin `mlflow==3.17.0` (PyPI stable metadata verified 2026-10-07, Python>=3.10) and retain the resolved dependency lock. If installation is incompatible, report/reconcile rather than modify model environments.
- Persistent root: `/home/swl007007/.local/share/ipcch-mlflow/`; SQLite `mlflow.db`, `artifacts/`, import logs/manifests. Outside Dropbox, Git and Temp. Existing nonempty roots/venvs must be inspected and never replaced blindly.
- Server `127.0.0.1:5000`, explicit absolute SQLite URI and artifact destination, artifact HTTP proxy enabled. Clients use `http://127.0.0.1:5000`; UI artifacts must be served through MLflow rather than links into `/mnt/c`.
- Provide start/stop/status commands via one shell helper and a named user service launched on demand. Do not enable login autostart, linger, firewall changes or LAN binding. Starting the server during acceptance is authorized by final execution approval; leave it available to the user afterward. State that WSL shutdown stops availability; persisted data survives.
- Verify `/health`, tracking API, UI HTML and a real artifact download from both WSL and Windows localhost. Check rendered UI through an available browser tool; if browser automation is unavailable, report that limitation separately from verified Windows HTTP access.

## Sources and comparison layout

One MLflow experiment `IPCCH`. Six parent records represent original scientific runs and own their archived artifacts. Child records are evaluation views, explicitly tagged `record_kind=evaluation_view`, not independent new training jobs. Parents use `record_kind=source_run`, `import_mode=historical`.

| Parent source | Child arms | Seeds | Expected children |
| --- | --- | --- | ---: |
| Original P6 GeoXGB | pool, geo, persistence | model42; persistence none | 12 |
| Fixed-map MLP | base, pool, local, geo; shared persistence | models42/43/44; persistence none | 52 |
| Yearly GeoXGB | pool, local, geo, persistence | model42; persistence none | 16 |
| Climate perturbation | pool, geo, persistence | model42; persistence none | 12 |
| History-window sensitivity | base_global, exp_global, base_local, exp_local | 42 | 16 |
| Split2024 sensitivity | pool, geo, persistence | model42; persistence none | 12 |

Each arm has H1/3/6/12. Proposed total: 6 parents +120 evaluation views. Count distinct MLP persistence views once per H only after verifying repeated seed reports agree. Preserve MLP descriptive seed mean/min/max as parent summary, not another fitted model or confidence interval.

P6 comparator panels embedded in MLP/yearly/climate and split2024 matched reports stay in `comparators`/`matched_old` namespaces and reference their original P6 parent; do not manufacture fresh P6 fitted runs. Window base arms are explicitly reused comparators; the local arms include pooled fallback outside local support. Only their common-local-support cohort describes a pure supported-local comparison. Preserve these semantics in tags and the original report artifact.

Periods/cohorts form metric namespaces on each child (for example `main.E_all.binary.f1`, `supplementary.E_persist.binary.f1`, `main.local_eligible.binary.f1`). Parent metadata binds exact target calendar, feature recipe, truth/decoding definitions, fit window/weights, maps and configurations. Key-set digests or explicit source-supported comparison-group tags identify compatible cohorts; labels like E_all alone never establish identical samples.

Keep the full requested existing panel: four-class accuracy/macroF1; binary accuracy/F1/precision/recall/F2; projected/raw q3R2. Preserve confusion/per-class counts, NA reasons and source metric paths in a JSON artifact. Log only finite numeric source values as MLflow metrics; absent/undefined scores are not zeros. Log sample counts even for empty cohorts. Do not backfill scores missing from the original reports by recomputing predictions. For example P6/climate do not supply pool/E_persist, and split H12 main is empty.

Store reported deltas and available bootstrap CI endpoints under named contrast/cohort namespaces with draws/seed/conditional interpretation. MLP seed ranges remain descriptive. Window uses `selected_dates` namespaces, not full main/supplementary performance; preserve its cross-H descriptive aggregates as parent report only. Split2024 `combined`/year summaries and same-key old-P6 comparisons remain separate from original P6 full-period scores. Do not create a global best-model ranking across incompatible cohorts.

Source roots and JSON extraction paths are in `research/report-layout.md`; freeze a finite source manifest during implementation, before import. The 126-record count is a schema expectation, not proof of source completeness. Any missing source or conflicting identity is an explicit gap, never silently skipped while claiming six-family completion.

## Artifact retention

Copy the retained models, preprocessing/transformation states, configs, source scripts/patches, maps/recipes, predictions, fit/gate/fold ledgers, reports, CI draws, verification evidence, original inventories and source identifiers into each parent once. MLP needs transforms as well as model weights. Preserve compact ordered row/key/target ledgers and prepared manifests as provenance. Raw data and large prepared feature matrices are excluded and recorded with original paths/hashes; this is not self-contained retraining.

Keep an explicit include/exclude manifest per source, with original relative path, archived relative path or shared-parent reference, byte count, SHA256 and exclusion reason. Reuse one archived P6 copy for inherited artifacts where identities match. Do not copy venvs, caches, preflight/dev/failed-run model stores or duplicate replay predictions/reports. Preserve replay results and original inventories, making clear that the archive is a documented subset of the original full-run inventory.

Use native MLflow artifact logging/proxy access. Preserve original formats; do not load or deserialize models just to import them. A quartet's four UBJ files and its fitting record remain a quartet; no misleading single-model registry object. If bundling many files is needed, a standard uncompressed tar with per-file manifest is acceptable, documented as the parent download artifact. No custom object store or deduplication framework.

Validate copied content against source checksums when available and calculate a new archive checksum inventory. Verify every retained file/entry, not just file counts. If a historical script lacks Git HEAD, use its actual saved source hashes and record HEAD unknown; do not invent start/end times from mtimes. Preserve known source timestamps as tags; MLflow timestamps describe import/evaluation-record creation, not training duration.

## Import identity and recovery

A single serial importer uses explicit native MlflowClient APIs. Stable source key = family + original run ID; child key also includes H/arm/seed. Bind source report/config/input identity and retained artifact inventory to an import fingerprint. Search existing tagged records before creating them. Same identity and completed fingerprint is a verified no-op; a different fingerprint under the same source key is a conflict requiring reconciliation, not overwrite.

Use a stdlib file lock to prevent two local imports. Persist per-parent progress and record IDs so a failed import can be resumed against the same inputs. Mark complete only after archived artifacts, metric readback and expected child counts pass. Incomplete imports remain visibly incomplete; they cannot be confused with failed scientific runs. Do not delete source runs or MLflow records automatically.

## Minimal implementation and operations

Use an isolated `IPCCHMLflow/` tool directory: a small explicit importer for the six known schemas, finite source config, start/stop helper, pinned requirements/lock, README and focused tests. No plugin framework, custom dashboard, database tables outside MLflow, watcher, training hooks or migration of other projects.

Document how to add a future run using the supported schema and an explicit manifest; automatic tracking inside future training code is out of scope. Back up SQLite with its backup API and archived artifacts/manifests while the importer is stopped. Exercise restore into a separate scratch location; don't replace live data. A local backup is not protection against loss of the whole machine.

## Acceptance and lifecycle

Validate source plans before writing tracking records, then test importer behavior on tiny representative fixtures (undefined metric, subset cohort, conflicting identity, interrupted import and idempotent rerun). Perform the full six-source import, API readback of all scalar metrics/params/tags, archive checksum validation, matched comparisons and UI/artifact-download checks. Re-run importer once to demonstrate no duplicates or extra artifacts. Final delivery reports exact imported counts, retained bytes, missing-source limits and commands/URL.

Final approval must cover this proposed layout and implementation. Before starting this new task, the bound executor must reconcile the already accepted yearly-XGB lifecycle through the controller's normal supported close path; preserve its acceptance and base, never reset/rebind an active run. An archive/queue operation is not an audit pass. If a real gate blocks new work, report its exact state. Then commit the approved MLflow plan and verify the new task's actual executor/start/base before implementation. No reuse of an old task-specific waiver. Proposed executor remains verified Claude Opus5.5 1M under Codex supervision.
