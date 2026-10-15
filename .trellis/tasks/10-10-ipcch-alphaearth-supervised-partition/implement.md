# IPCCH AlphaEarth supervised partitions — implementation plan

Status: final planning review, 2026-10-10. No phase below has begun. Accepted science is recorded in `decisions.md`; executable contracts are in `design.md`. This plan is not permission to implement, export or fit. Record each subsequent release and its scope in PROGRESS; do not infer a later release from completion of an earlier phase.

Execution status, 2026-10-14 (appended; the planning text below is preserved). P0–P3 are complete and supervisor-accepted; the formal result (all H fall back to B2, no partition map selected, replay 488/488) is accepted. P4 delivery is in progress: note, README and inventory are written; MLflow is an offline plan only (`mlflow-plan.md`). The task is not closed. Receipts and dates are in PROGRESS.md.

Delivery status, 2026-10-14 (appended). Under the user's research-storage policy (MLflow for research version/feature management and a temporary internal dashboard; no model packages, full-store backups or rehearsals), the AlphaEarth family is registered additively in the local store. Detailed: +21 runs. Dashboard: +20 rows; +16 registered models, versions and logged models; +12 evaluation and +8 training datasets. The dashboard fingerprint moved c7abf4cd -> cb3db908 with lineage, and the Forecasting experiments are untouched. The task lifecycle stays open (no commit/archive).

## Scope and sequence

| Phase | Work | Evidence before proceeding |
|---|---|---|
| P0 | Offline implementation, synthetic checks, local source preparation and no-fit planning | Source/runtime/feature identities, cohort and fit enumeration, synthetic replay, compute benchmark; no project-data model fitting or EE operations |
| P1 | EE metadata, bounded pilot, then full table extraction | Concrete existing account/project/destination and limits before pilot; pilot evidence and approved full envelope before bulk export |
| P2 | Freeze extracted inputs; historical backbones, forward development and final-map learning | Complete coverage/exports, exact model requests, frozen code/configs; save selected recipes/maps before main scoring |
| P3 | Bridge and annual main/supplement predictions | Frozen map identities; annual coefficients and unique historical providers; no selection on main results |
| P4 | Report, zero-fit replay, meeting note and completed-run MLflow registration | Same-key metrics, per-H decisions, artifact inventory, accepted result bundle, tracking plan/backup/verification |

P0 is the proposed first execution release. Cloud metadata/pilot, full extraction and project-data fitting remain explicit later checkpoints. P2/P3 may be released together after P1 once the concrete compute envelope is accepted; the map-freeze checkpoint must still be recorded before main results are inspected. No speculative repeated review cycle is required.

## Before implementation

- [x] Obtain final design/plan acceptance and a concrete execution scope; keep task `planning` until then. (2026-10-10: execution goal activated by the user, P0 offline released; see execution-handoff.md.)
- [x] Commit the approved task documents and nonempty context manifests. Verify repository scope and run GitNexus detect_changes before a commit; record its known database failure if still present and check the diff directly.
- [x] When a Claude executor is authorized, verify its real foreground session/pane/terminal and the repository's exact registered path; use the audit wrapper to bind/start this task from that executor. Verify run ID, actual base SHA and native `in_progress` status. Do not inherit a historical executor identity or start from the supervisor.
- [x] Read the scoped backend specs in implement.jsonl. For any existing symbol changed, inspect callers/impact first. Do not change sibling IPCCH or completed packages to accommodate this experiment.

## P0 — offline implementation and readiness

- [x] Create only `IPCCHAlphaEarthExperiment/` with the small `ipcch_alphaearth` package, config/schema/runtime/source-provenance files, README and tests. Group modules by data/preparation, extraction, backbone/provider schedule, correction/tree, and evaluation/replay responsibilities; avoid generic plugin or experiment frameworks.
- [x] Attribute adapted target/projection/metric, compact feature and record helpers to source paths/SHAs. Keep H1 adaptation, canonical-population cohort, month-end semantics and the new runtime explicit. No dynamic dependency on another run's mutable scripts.
- [x] Stage local consumed sources outside Dropbox; rehash all bytes and pin static29's concrete source, invariance checks, five geometry files and full source-code inventory. Do not treat manifest-reported large-file hashes as freshly checked content.
- [x] Rebuild canonical target QC and compact296 at each row's origin, including the 14 keys missing from old compact tables. Verify 42,695 valid rows / 6,224 admins, exact saved P6 key sets, no source-induced row deletion, ordered feature names and H1 values. Compare inherited H3/6/12 feature values on compatible overlapping rows; explain any mismatch before fitting. Keep legitimate early-climate/IDP/static missingness.
- [x] Enumerate historical/development/bridge/main/supplement blocks and unique providers without fitting: B1 38 + B0 23 = 61 quartets / 244 scalar ceiling. Show actual training counts, weights, keys and empty pools. Check 122 main + 16 supplement slots and main counts 17,322/16,919/16,413/14,087, supplement 4,092 per H. Source drift is a checkpoint finding, not permission to change counts silently.
- [x] Implement D19's normalized constrained ridge, fit-only transforms, D20 support and D25 routing with synthetic descriptors. Implement design section 9's deterministic search with cached sufficient statistics and complete candidate score/support ledger. Separate all ridge work from XGB request counts.
- [x] Implement independent report/replay paths early. Use a small synthetic end-to-end case with a genuine selected split and another with exact B2 fallback. Verify saved models, transforms and routes can regenerate every prediction.
- [x] Benchmark the sufficient-statistic search at representative admin/row/leaf dimensions with synthetic data, and synthetic XGB timing under the frozen numerical runtime. Estimate wall time, memory and output sizes; do not call the candidate ceiling a measured cost. Avoid nested thread oversubscription. An impractical benchmark returns to the checkpoint without shrinking the grid or changing science.
- [x] Prepare the extraction script offline, including native-grid moment calculations, deterministic ownership, inventory/completeness checks and local moment replay. Use synthetic boundary/mask fixtures; do not authenticate, query assets or export yet.
- [x] Save `P0-evidence.md`, source/runtime manifests, feature and schedule checks, tests and benchmark summaries in task evidence. Pin implementation SHA and all scientific configs. Stop at readiness before EE or any project-data fit. (2026-10-10: complete; evidence/p0/P0-evidence.md. The `export` subcommand writes to GCS/Drive; under the user's query/download-only cloud pre-authorization, P1 first specifies direct EE computation returning the same local table — see PROGRESS.)

### Required focused tests

| Contract | Minimum discriminating cases |
|---|---|
| Truth/cohort/projection | Exact .20 truth and prediction boundaries; invalid/missing population/share cases; q2–q5 crossing; projection-before-clipping; constant/absent-class NA semantics; retain old-compact-missing keys |
| Availability/schedule | Label U=fit O allowed, U>O refused; training features cut at own row origin; season boundary at O+1; H1 distinct from H0; partial first block and empty first month do not move anchors |
| OOT/provider | No fitted target becomes residual supervision; four bridge providers fill exactly 22 H/month slots; no duplicate providers; future-label perturbation cannot alter an earlier map/fit/prediction |
| Moments/routing | Merge A/S/Q instead of averaging SD; shared finite-band mask; below/exact .80 coverage; no-data vs failed export; overlap ownership; no-zero-imputation AE; unseen admin follows frozen thresholds |
| Correction/support | Joint objective and centering against direct constrained reference; normalized loss, unweighted regional penalty, unpenalized shared intercept; inactive inputs; low-coverage/sparse rows retain shared loss; fewer than two leaves exactly reuses B2 |
| Tree/selection | Reference search on small complete candidate set; threshold and tie order; both-child support; depth/leaf limits; later splits rescore other leaves; two forward years; all nine recipes and secondary guards; exact-Fraction ranking/fallback |
| Inference/report | Paired country multiplicities; constant-zero G=B2 deltas; fixed four-H correction; undefined draws not removed; one qualifying H remains a positive per-H result; matched persistence coverage |
| Identity/replay | Changed fit keys, source, provider, map/coefficients or saved prediction refused; metrics recomputed independently; no XGB fit in replay; complete outputs inventoried after replay |

Do not mirror every line with a test or enumerate unrelated tamper scenarios. These cases protect the scientific contrasts and reuse boundaries.

## P1 — concrete cloud release and extraction

First inspect the configured existing account/project and intended storage through read-only metadata **after that scope is released**. Do not create projects/buckets, change billing/IAM or upload IPCCH outcomes. Only necessary geometry/IDs go to EE. The checkpoint proposal must identify:

- actual 2017 assets, product metadata/version and native grids, access method and geometry payload;
- an existing destination/prefix and local run directory, expected bytes and retention;
- geometrically chosen pilot IDs, source pieces/pixel-area estimate, task/concurrency/runtime limits and available cost/quota information;
- exact commands and an explicit limit/stop policy if service usage or cost is not reliably estimated.

Do not invent a dollar allowance or assume an account tier is free. Missing budget/destination facts are operational inputs for the concrete proposal, not a reason to reopen the scientific choices.

- [x] Choose a small pilot before seeing outcomes: representative small/large polygons, coastal or mask-edge areas, and source-grid/UTM overlaps. List the IDs and maximum processed area in the proposal; a large stress polygon may need a separately bounded piece. Get the bounded pilot release before reduction/export jobs. (done 2026-10-10: pilots A/A2/B1; PROGRESS)
- [x] Confirm band/year/version metadata, full-polygon area convention, actual grid transforms, source ownership and common masks. Compare moment aggregation across work-piece batchings; inspect uncovered and overlap areas, variance cancellation and raw coverage around .80. Record any approximation beyond D26. Do not introduce a common coarse grid, bestEffort or masking fill to hide failure. (done 2026-10-10: metadata release and pilots; ownership/mask evidence in evidence/p1)
- [x] D30 amendment (2026-10-10): implement the EPSG:6933 first-valid whole-vector analysis-grid extraction (smallest connected change + focused offline tests); re-enumerate requests with ordered candidate asset IDs; prepare a bounded C pilot (overlap piece of admin 1722, single-zone control 3447, grid-aligned batching/resource checks) for supervisor pinning before any live EE computation. Native-grid pilot results remain comparison evidence only. (done 2026-10-10: D30 implemented, C pilot rev2 executed; PROGRESS)
- [x] Present pilot export IDs, completed-file hashes, row/piece completeness, resource measurements and a full 6,227-geometry request inventory with anticipated costs/time/space and caps. First-owner masked overlap loss must be visible; if it makes the proposed ownership rule unsuitable, resolve the rule before bulk extraction or model fitting. (done 2026-10-10: C-PRODUCTION-PROPOSAL.md; 57,801-piece plan)
- [x] After full-envelope release, export only sufficient-statistic tables and manifests. Keep a per-piece ledger and resume only known incomplete pieces under identical inputs; never duplicate successful pieces. Retain technical failures separately from genuine masked coverage. (done 2026-10-14: release codex-D30-full-20261010 plus the c2 continuation; 57,801/57,801 pieces; no duplicate successes)
- [x] Reaggregate descriptors offline, freeze all source-piece/geometry/runtime hashes, and show coverage by admin/country plus cohort routing counts. Retain full canonical cohorts. No target-based coverage threshold or pilot/geometry selection. (done 2026-10-14: ingest + attach in p1-formal-d30-20261014; 6,227 usable)
- [x] Freeze prepared296/424 arrays, descriptor table and manifests; reconcile the exact fit inventory and compute estimate for P2/P3. Stop for the project-data fit release if not already explicitly given. (done 2026-10-14: freeze_report; P2 release ae558aab)

## P2 — development and map freeze

- [x] Fit historical B1 through target2022 (19 quartets) and the development-only B0 (8 quartets), with saved keys/X/y/weights/parameters. Cache exact identities; don't refit B1 separately for each correction candidate. (done 2026-10-14: 27 quartets)
- [x] Generate honest historical residual providers. Reconstruct maps/transforms/corrections separately at each 2021/2022 validation block's earliest scheduled fit origin, using only residual labels U<=fit_origin. Evaluate all 9 recipes/H using the same B1 and matched B0/B2; save complete yearly metric and guard records. (done 2026-10-14: 72 recipe-years, 0 passed)
- [x] Select by the fixed rule or choose B2 per H. Relearn only selected-lambda final maps with labels through2022-12; preserve no-beneficial-split fallback. Save trees, eligible training refs, scales/support, split objectives, map memberships and selection hashes. (done 2026-10-14: B2 at every H; no map)
- [x] Record the map-freeze checkpoint before generating/scoring main G predictions. Report development scores and realized complexity without adjusting thresholds, support, grids or features. A scientific fallback is a valid result. (done 2026-10-14: MAP_FREEZE 7e5778ea; P2 accepted)

## P3 — fixed-map annual evaluation

- [x] Fit the four B1 bridge quartets and generate only the pre-main2023 gap predictions. For each H, process formal annual blocks in chronological order; save raw B1 outputs for later eligible correction updates. (done 2026-10-14: P3 predict)
- [x] Fit the 11 main + 4 supplement quartets for each of B0/B1. Refit B2/G scaling and coefficients annually using eligible saved OOT residuals, with unchanged maps/recipes and half-life24. Year-internal coefficients stay fixed while inputs change by row origin. Sparse leaves use shared G; fewer than two supported leaves uses exact B2. (done 2026-10-14: 34 quartets; 61/244 total)
- [x] Produce every scheduled row's B0/B1/B2/G raw/projected/decoded outputs plus persistence lineage. Preserve empty-fold reasons. The stage totals are 27 + 34 = 61 fresh quartets, 244 scalar fits at most; explain any authorized exact reuse. No extra local backbone or score-driven refits. (done 2026-10-14: G == B2 bitwise under fallback)
- [x] Save per-block support, provider and route ledgers and technical completion state. Do not widen the cohort or use supplementary2026 for recipe selection. (done 2026-10-14; one corrupted provider cell later reconstructed and archived, see PROGRESS)

## P4 — verification and delivery

- [x] Independently reconstruct all metrics and the G−B2 paired country bootstrap (20,000 draws, seed42, fixed-four-H 1.25th-percentile lower bounds). Show each H's effect, annual signs and all secondary guards. Report B0/B1/B2 and matched persistence, coverage and the separate2026 supplement. No all-four-win gate. (done 2026-10-14: report; replay recount)
- [x] Run zero-XGB-fit replay from frozen source/prepared/model artifacts: truth/keys/time/provider/fitting identities, raw re-predictions, projection/decoding, transforms/routes/selection, metric/NA/bootstrap results, and offline moments. Verify winning split objectives and the recorded candidate ordering; state that production replay does not refit every rejected candidate or fetch raw EE pixels. (done 2026-10-14: 488/488 after the reviewed analysis-only amendment)
- [x] Inventory outputs **after** report/replay. Keep ignored large artifacts outside Dropbox and commit compact evidence, explicitly including any referenced files hidden by global CSV/JSON ignore patterns. Evidence is not complete merely because aggregate counts match. (done 2026-10-14: ARTIFACT_INVENTORY.json; 33 evidence JSON files need force-add)
- [x] Write the Chinese meeting note and README reproduction commands. Distinguish partition, feature and calibration effects; include all H, negative results, computational envelope, approximate/country-conditional inference and release/vintage limitations. Do not infer causal mechanisms from regions. (done 2026-10-14: docs/notes/2026-10-14_组会讨论稿_IPCCH_AlphaEarth监督分区.md; README updated)
- [x] After result acceptance, add one completed MLflow family using existing import/summary contracts. (2026-10-14 done under the research-storage policy: no store backup or model bundle by user decision; import read-back, `verify --family`, and the extension's built-in legacy/addition verification passed.) Process used: AE `import_runs.py plan` → `import --family` → `verify --family`, then `summary_catalog.py plan|inventory|apply --extend alphaearth_partition` (legacy projection reproduced, additions read back by apply's built-in verify). Model weights listed by member manifest, not uploaded; no store backup or rehearsal (research-storage policy). Touched adapters tested with focused synthetic fixtures.
- [x] Record delivery acceptance and remaining limitations. (2026-10-14: supervisor accepted the research delivery. Checked: catalog_status complete; experiments 147/156/202/174; models/versions/logged models 196/242/242; no AE models.tar; AE parent artifacts 79,443,114 bytes; health 200. Limitations: legacy model bundles removed by policy (their URI/SHA tags are historical); four legacy Temp source roots are missing, so legacy import/verify is not supported. Lifecycle close/commit/archive is not done.) Any later closure/handoff/push/merge follows its separately authorized scope and actual lifecycle state; do not assume previous tasks' audit waivers transfer to this one.

## Planned commands and checkpoints

These are the intended small CLI, **not commands to run during planning**. P0 must implement and document them, or update the command spelling in this plan without changing stage behavior. Commands run from `IPCCHAlphaEarthExperiment/`; `RUN_DIR` is a new absolute WSL path, outside Dropbox.

```bash
/home/swl007007/.venvs/ipcch-geo/bin/python -m unittest discover -s tests
/home/swl007007/.venvs/ipcch-geo/bin/python -m ipcch_alphaearth preflight --run "$RUN_DIR"
/home/swl007007/.venvs/ipcch-geo/bin/python -m ipcch_alphaearth prepare --run "$RUN_DIR" --local-only
# D30 plan + sha-bound manifests (local; release_id null until a supervisor release):
PYTHONPATH=.:scripts /home/swl007007/.venvs/ipcch-geo/bin/python scripts/build_c_production.py --geometry "$GEOMETRY_SHP" --list-images "$LIST_IMAGES_JSON" --out "$PLAN_DIR"
# Cloud releases only (direct EE computation returned locally; one invocation at a time; resumable in the same OUT_DIR):
timeout "$OUTER_TIMEOUT_S" /opt/earthengine/bin/python scripts/ee_pilot_c.py --request "$RELEASED_MANIFEST" --out "$OUT_DIR"
/opt/earthengine/bin/python scripts/check_c_outputs.py --request "$RELEASED_MANIFEST" --out "$OUT_DIR" --json "$CHECK_JSON"
# After a complete, clean extraction (fresh run under the current config):
/home/swl007007/.venvs/ipcch-geo/bin/python -m ipcch_alphaearth ingest-extraction --run "$RUN_DIR" --plan "$PLAN_CSV" --manifest "$RELEASED_MANIFEST" --outputs "$OUT_DIR"
/home/swl007007/.venvs/ipcch-geo/bin/python -m ipcch_alphaearth prepare --run "$RUN_DIR"     # attach_descriptors
# (scripts/extract_alphaearth.py and scripts/ee_pilot.py are the superseded native-grid recipe: comparison only, per D30.)
# Project-data fitting release only:
/home/swl007007/.venvs/ipcch-geo/bin/python -m ipcch_alphaearth develop --run "$RUN_DIR"
/home/swl007007/.venvs/ipcch-geo/bin/python -m ipcch_alphaearth freeze-map --run "$RUN_DIR"
/home/swl007007/.venvs/ipcch-geo/bin/python -m ipcch_alphaearth predict --run "$RUN_DIR"
/home/swl007007/.venvs/ipcch-geo/bin/python -m ipcch_alphaearth report --run "$RUN_DIR"
/home/swl007007/.venvs/ipcch-geo/bin/python -m ipcch_alphaearth replay --run "$RUN_DIR"
```

`preflight` and local-only preparation must not implicitly fit or contact EE. Stage-specific prerequisites must fail clearly, not auto-launch another stage. MLflow commands come from its existing README; no new tracking runner is needed.

## Failure and recovery

Preserve original data, completed experiments, source inventories and partial-run evidence. Record a technical stop with context for missing/changed sources, incomplete exports, invalid model identities, temporal/key failures or numerical failures. An unchanged extraction plan can resume identified missing pieces; an unchanged fitting request can reuse only verified completed models. Never promote a partial model or silently refit a corrupt existing identity. Changes to scientific inputs/code create a newly identified run or a documented pre-fit freeze, not a hidden overwrite. Routine storage/network faults do not justify retuning. No-result/sparse/low-coverage cases follow the explicit scientific fallbacks and remain reported.
