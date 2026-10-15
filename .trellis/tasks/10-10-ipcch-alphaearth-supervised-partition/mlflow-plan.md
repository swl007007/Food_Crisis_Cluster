# Offline MLflow registration plan r2 — IPCCH AlphaEarth supervised partition (R15, design §14)

> **2026-10-14 user research-storage policy supersedes the retention, backup and rehearsal gates below.**
> MLflow is for research version/feature management and a temporary internal dashboard, not production. Keep raw
> data, training/running scripts and the small metadata/results; trust existing copies of earlier experiments.
> Changes to this plan:
> - **Models:** no `models.tar` is built or uploaded. The 305 model files are listed in
>   `manifests/models.tar.members.json` (path, bytes, SHA256 in the source run folder). AE model descriptors carry an
>   empty bundle URI. Older families' tar URI/SHA tags are historical references; their tars were removed on purpose.
> - **Extras / copies:** only the experiment code is imported as extras. The EE image list, the production pieces
>   and the extraction ledgers (`extraction/plan.csv.gz`, `pieces.csv.gz`) stay in place and are referenced by path +
>   SHA256 (external bindings / excluded). Task notes, progress, evidence and receipts stay in the task folder.
> - **No B0/scratch rehearsal or full-store backup gate.** The extension apply needs only the inventory; journal and
>   S0-S3 recovery remain. The earlier scratch rehearsal attempts and their stores/backups were stopped and removed
>   by the user/supervisor and are not recreated. `rebuild_check.py` (an old-names migration checker) is not used.
> - **Older families** stay as historical metadata (values, datasets, IDs unchanged). Their `import`/`verify` is not
>   routine and would stop: their bundles and four Temp source roots are gone.
> - The research-retention delta was reviewed (scoped pass) and live additive registration was authorized (§8).

Status: **revision 2, implemented** (answering review `codex-D30-delivery-and-mlflow-plan-review-r1-20261014`,
`271032d1…`, findings M1–M6 and D1; r1 is kept as `mlflow-plan-history/mlflow-plan.r1-c0aa7a02.md`). The adapter
code in §7 and the research-retention delta are in `IPCCHMLflow/`. The §2 baseline and §9 evidence are historical
planning records.

## 1. Target (unchanged)

- **Deliverable:** the existing store `/home/swl007007/.local/share/ipcch-mlflow/`, using its existing
  experiments `IPCCH - detailed runs` and `IPCCH - dashboard`. The family is added there, additively.
- **Not a deliverable:** an isolated dashboard.
- **Untouched:** all pre-existing objects in all four experiments (including `IPCCH Forecasting - *`) and all
  registered models, versions and logged models, with their IDs and values.

## 2. Baseline (supervisor read-only SQLite, 2026-10-14; re-freeze before writes)

| Object | Count |
|---|---:|
| exp 1 `IPCCH - detailed runs` | 126 |
| exp 2 `IPCCH - dashboard` | 136; plan fp `c7abf4cdc2f5abe8f85300ad41fcf08085ceb14f170b81121e5f1605365158dc`; `catalog_status=complete` |
| exp 3 / exp 4 (`IPCCH Forecasting - *`) | 202 / 174 |
| global registered models / versions / logged models | 180 / 226 / 226 |

Protection snapshot (before and after, for all four experiments):
- per-run ID, name, status, tags, metric keys/values/steps/timestamps and inputs;
- every registered model, version and logged model with its tags, sources and run links.

The only permitted differences are §4's additions and §6's named experiment-metadata transitions.

## 3. Family entry (sources.json + naming.py)

### 3.1 Files

The family is `alphaearth_partition`, format `alphaearth`, source run `p1-formal-d30-20261014`, root
`/home/swl007007/ipcch-alphaearth-runs/p1-formal-d30-20261014`.

| Scope | Exact paths |
|---|---|
| include | `run.json`; `preflight/preflight-20261014T124450Z.json`; `prepared/{COMPLETE.json, prepared-manifest.json, *.csv.gz, schedule_blocks.json}`; `extraction/{COMPLETE.json, manifest.json}`; `descriptors/**`; `develop/**`; `maps/**`; `predict/**`; `report/**`; `replay/**`; `archive/replay-attempt-01-20261014T141645Z/**` |
| bundle `models.tar` | `models/**`: 61 identity dirs × (`q2..q5.ubj` + `record.json`) = 305 files |
| excluded (recorded with path + SHA, not copied) | `extraction/{plan,pieces}.csv.gz`; `prepared/X296_h{01,03,06,12}.npy`; all 22 files under `sources/` listed individually: climate monthly + seasonal, 4 compact baselines, compact manifest, country lookup, 5 geometry files, IDP, interim, 4 p6 key tables, p6 valid target ledger, raw panel, source-manifest |
| extras | `IPCCHAlphaEarthExperiment/{ipcch_alphaearth,config,scripts,tests}/*` only (research policy: `list_images.json` and `production_pieces.jsonl.gz` stay external, SHA-pinned; task notes/evidence/receipts and the package README are not imported) |

Notes on these files:
- Excluded records are hashed and enter `by_sha`, so references to them need no exemption.
- `pieces.csv.gz` is the INGESTED piece table (`46f46ab5`). It is not the external `production_pieces.jsonl.gz`
  (`9f22cb26`, extraction run), which `extraction/manifest.json`'s `pieces_sha256` names (§3.2).
- The active `predict/providers_all_h03.csv.gz` is the reconstructed file (`06eefdc6`). The original corrupt bytes
  (`e454ce15`) and failed replay attempt 1 are included only under `archive/…`, with `ARCHIVE.json` and
  `ANALYSIS_AMENDMENT.json`.

### 3.2 Digest policy (complete classification of all 1,298 `*sha256` leaves; evidence §9)

**Required file digests.** Each must equal an own planned file; if it does not, the import is refused.
- `booster_sha256` → own `q*.ubj` via the §3.3 ledger adapter.
- `artifacts_sha256`: the 40 prepared files.
- `prepared_manifest_sha256`, `descriptor_table_sha256`, `extraction_complete_sha256`.
- `plan_sha256` → `extraction/plan.csv.gz`.
- `geometry_sha256` and `pinned_sha256` → the excluded `sources/*` records.
- `preflight .inputs.*.sha256` → excluded `sources/*`.
- `file_sha256` / `selection_sha256` / `map_freeze_sha256`, `report_sha256`, `replay_sha256`.
- `config_sha256` → `config/*.json` in the source-code extras.
- archive `moves[].sha256` / `after_sha256` / `original_sha256` → archived files.
- `install.sha256` / `reconstructed_sha256` → the active `providers_all_h03`.
- `regression_test_source_sha256` → `tests/test_replay_contracts.py` in extras.

**Required external evidence files.** Each is bound by exact path + SHA in the policy and must exist and match
at plan time; it is not retained in the run.

| Key and JSON path | Bound file |
|---|---|
| `extraction/manifest.json .pieces_sha256` | `p1-c-full-20261010/inputs/production_pieces.jsonl.gz` (`9f22cb26`) |
| `{extraction,descriptors}/COMPLETE.json` + `extraction/manifest.json` `.provenance.list_images_sha256` | `/home/swl007007/ipcch-alphaearth-runs/p1-metadata-20261010/ee/list_images.json` (`b233255b`; `build_c_production.py:78`) |
| `ARCHIVE.json .receipt_sha256` | the install release receipt (`991ce446`), copied into task evidence |
| `ANALYSIS_AMENDMENT.json .replay_diff_sha256` | `replay-repair/replay.py.diff` (`4c0fc964`) in task evidence |
| `ANALYSIS_AMENDMENT.json .supervisor_test_log_sha256` | the supervisor test log (`fc21c10b`) in task evidence |

**Typed non-file identities.** Each key is limited to exact file-class + JSON path; none is a generic exemption.
- `X_sha256`, `y_sha256.*` and `identity.weights.sha256` in `models/*/*/record.json`: `array_digest` of the
  float64 fitting slice (dtype|shape| + C bytes, `artifacts.py:40-46`).
- `features_sha256` (record identity): canonical-JSON digest of the ordered feature names.
- `compact296_list_sha256` (prepared manifest): newline-joined ordered names.
- `resolved_config_sha256` (`targets.q*`): digest of the resolved XGB config dict.
- `request_sha256` (`extraction/manifest.json`) and `manifest_sha256` (`extraction/COMPLETE.json`): the canonical
  request BODY digest `63d0c7ab`, which is not the manifest file SHA `b7acb5fa`.

**Historical upstream (external) references.** These are exact-path exceptions with reasons, found only inside
the excluded `sources/compact_manifest__…json`.
- `code_sha256.*`: sibling IPCCH source files.
- `feature_sha256` / `features_by_run_sha256.*`: ordered-name identities, newline convention (e.g. `e97388…`).
- `label_keys_sha256` / `eval_keys_sha256`: key-set identities.
- `contract.*.sha256`, `horizons.*.{history,idp,season,oracle}_ledger.sha256`,
  `horizons.*.parent_dataset.sha256`, `…compact_weather_oracle.dataset.sha256`: upstream run outputs and spec
  documents.

These stay authoritative as recorded in the accepted original source evidence. They are not re-acquired. Its
`compact_baseline.dataset.sha256` entries are required (they equal excluded `sources/*` files).

There is no generic `sources/**` exemption and no unclassified key. Every required path is checked for existence.

### 3.3 Native AlphaEarth model ledger adapter (inventory.py; M2)

The existing `model_contract` scans `model_requests*.jsonl`, which do not exist in this run, so it would pass
0/0 vacuously. A new contract kind, `alphaearth_xgb`, used only by this family, will:
- read `develop/request_ledger.csv.gz` (27 rows) and `predict/request_ledger.csv.gz` (34 rows), and require exactly
  61 unique identities;
- check that each identity's (arm, horizon, block_id, role, fit_origin) matches `prepared/schedule_requests.csv.gz`
  exactly, and that every row has action `fit`;
- for each identity, require exactly one bundle directory `models/<id[:2]>/<id>/` with its own `record.json`. The
  record's `identity_digest` must equal `<id>`, and its `identity.{arm, horizon, block_id, role, fit_origin}` must
  equal the ledger row. The record must name `q2..q5`, and each `booster_sha256` must equal
  that directory's own `q*.ubj` SHA. That makes 244 boosters, and a hash found elsewhere does not satisfy it.
- refuse:
  - a missing or extra identity, or a duplicate ledger row;
  - a missing own booster or `record.json`;
  - a tampered booster or record;
  - an empty or absent ledger.

The legacy `xgb`/`mlp` contracts are unchanged.

### 3.4 Arms, metrics and periods

- **Arms:** B0 `pooled`, B1 `pooled_plus_alphaearth`, B2 `shared_calibration`, G `regional_calibration`, and
  `persistence`. G is stated as bitwise B2 (no map selected) in the tags and description.
- **Metrics:** existing names only. `q3_r2_projected` → `share_phase3plus_r2`; `q3_r2_raw`, the annual blocks and
  the G−B2 bootstrap stay in the detailed record and its artifacts. Dashboard contrasts are `minus_persistence` and
  `minus_pooled` (pooled = B0), with no interval unless one is saved.
- **Periods:** `primary` = main, `holdout` = supplement. There is **no** supplement `persistence_available` cohort:
  no such panel is saved, so it stays absent or NA and is never rescored.

## 4. Frozen expected additions (r2)

| Object | Added | IPCCH-slice total after | Global after |
|---|---:|---:|---:|
| detailed runs (exp 1) | 1 parent + 20 children | 147 | — |
| dashboard rows (exp 2) | 20 | 156 | — |
| registered models / versions / logged models | 16 / 16 / 16 | 84 / 116 / — | 196 / 242 / 242 |
| evaluation datasets | **12, family-qualified** | 66 → 78 | — |
| training-pool datasets | 8 composite descriptors | 8 → 16 | — |
| exp 3 / exp 4 | 0 | unchanged | — |

**M1 dataset identity.** `dataset_descriptors_v2.json` reads the predictions exactly as the dashboard does
(`dtype=str`, `keep_default_na=False`), and `p6_truth_compare.json` compares against the archived P6 predictions.
- Keys are equal for 16/16 cohorts, and the phase is equal on every key.
- The saved q3 truth TEXT differs for 16/16 cohorts, on about 15% of rows, by at most 9.9e-17. For example,
  admin 2145 target 24278: `0.40404040404040403` vs `0.404040404040404`.
- So the AlphaEarth datasets are **new, family-qualified names**, e.g.
  `IPCCH eval | 1-month | 2023-02..2025-10 | all_scored (AlphaEarth supervised partition)`, with exact truth text
  and `rows_depend_on`.
- Only the 12 cohorts with saved scores are registered: main `all_scored`, main `persistence_available` and
  supplement `all_scored`, × 4 leads. Their full descriptors and digests are frozen in
  `dataset_descriptors_v2.json` (e.g. H1 main all_scored `cfbfabdf…`).
- The r1 probe (`mlflow_dataset_compat.*`) is kept as superseded evidence.

**M6 training pools** (`training_descriptors_v2.json`; the old 8 descriptors are unchanged):
- **B0:** `IPCCH training pool | compact296 | <lead>`. It binds the saved `prepared/X296_hXX.npy` + `keys_hXX.csv.gz`
  hashes (re-verified on disk against the prepared manifest) and the ordered-296 schema.
- **B1/B2/G:** `IPCCH training pool | compact296+alphaearth128 (composite) | <lead>`. It binds the same X296 + keys,
  `descriptors/descriptor_table.csv.gz` (`362c58b0`), `descriptors/COMPLETE.json`, the exact `admin_code` join rule,
  and the ordered-424 schema. Two digests are given: newline-joined and JSON `0163a677`; the table column order
  equals `ae_names`.
- `materialized_x424` is "none": no X424 file is invented or written.

## 5. Model descriptors

There are 16 external composite LoggedModels and versions (not loadable; no inference flavor). All point to the
family parent's member manifest (`manifests/models.tar.members.json`); the bundle URI is empty because no tar is
uploaded (research policy).
- **B0/B1:** list the lead's main + supplement quartet identities (4 per lead, 3 at 12-month). B1 also lists the
  historical-provider and bridge identities.
- **B2/G:** the B1 list, `predict/contexts_hXX.json` and `maps/MAP_FREEZE.json` (`map: null`).

## 6. Dashboard extension state machine (summary_catalog.py; M5)

**Pins in the extension plan:**
- `predecessor_fingerprint` `c7abf4cd…`;
- `predecessor_plan_sha256` (the on-disk `plan-c7abf4cd….json`);
- the bound addition set (projection keys of the 20 rows, 16 models, 16 versions, 20 datasets);
- `new_fingerprint`.

The **legacy projection** (all families except exactly this addition set) must reproduce `c7abf4cd…` byte-for-byte
at plan time. The durable journal is `journal-<new fp>.json`, holding the intent, the pins and per-object IDs.

| State | Experiment tags | Allowed action |
|---|---|---|
| S0 predecessor complete | fp = `c7abf4cd`, `catalog_status=complete`, no extension intent | start: inventory; deep-verify ALL legacy objects against the predecessor plan, before any write |
| S1 extension in progress | fp = `c7abf4cd`, `catalog_status=incomplete`, `zz_prov.dashboard_extension_intent=<new fp>` | create or reuse only bound additions (new fp object tags); resume only if intent and journal = this plan; refuse any other incomplete intent |
| S2 fp committed | fp = `<new fp>`, lineage `c7abf4cd,<new>`, `catalog_status=incomplete`, intent retained | after the additions verify: write the experiment description (new guide) |
| S3 complete | fp = `<new fp>`, `catalog_status=complete`, intent removed | same-plan repeat = deep verify, all noop; any other plan refused |

Rules:
- Legacy objects keep their IDs, `c7abf4cd` object tags, inputs and metric histories.
- The only existing-object edits are on the experiment: the transient `catalog_status`, the extension-intent tag,
  `zz_prov.dashboard_plan_fingerprint`, `zz_prov.dashboard_plan_lineage` and the description.
- `verify` checks legacy against the predecessor plan and additions against the new plan.

**Resume:** no backup is required (research retention). A resume re-verifies that the legacy objects are unchanged
and that the partial additions match the journal.

**Interruption tests** (`--fail-after`, synthetic test fixtures):
- after the S0→S1 intent/status write;
- after some models and versions;
- after some rows;
- before the fp commit;
- after the fp commit, before the description;
- before `complete`.

Each must resume to S3 with the same IDs, be refused for a foreign intent, and be refused when a legacy object has
drifted.

## 7. Code changes (implemented; research-retention delta 2026-10-14 in README "Storage policy")

Files in `IPCCHMLflow/`:
- `extract.py`: `_alphaearth_children` (read-only) + EXTRACTORS entry;
- `naming.py`: family, arms, family-qualified AE dataset names and training-pool feature labels;
- `inventory.py`: §3.3 contract;
- `sources.json`: §3 entry;
- `summary_catalog.py`: §6 extension mode, plan-held EXPECTED, read-only legacy reproduction.

GitNexus impact is run per symbol before editing. `IMPORTER_VERSION` is unchanged; the six families' import
fingerprints and the legacy dashboard fp must reproduce exactly.

Tests: the existing suite unchanged and passing, plus:
- AlphaEarth fixture extraction;
- the ledger adapter's refusals (§3.3);
- digest policy refusals: an unclassified key, a missing required external file, a wrong-path exception;
- the 12-dataset + 8-training freeze;
- the legacy fp and family fingerprints reproduced;
- the §6 state machine with every interruption point.

## 8. Execution (research-retention policy; the B0/scratch/backup steps are dropped — see r1 history)

Every command binds BOTH `--store /home/swl007007/.local/share/ipcch-mlflow` and
`--tracking-uri http://127.0.0.1:5000`; the server is managed only via `manage.sh` (localhost).

1. `import_runs.py plan|import|verify --family alphaearth_partition` (no legacy import/verify).
2. `summary_catalog.py plan --extend alphaearth_partition`, `inventory`, `apply --extend alphaearth_partition
   --plan-fingerprint FP --inventory INV`, `verify --extend alphaearth_partition`.
3. Read back the added metrics, features and datasets; check the expected counts (detailed +21, dashboard +20,
   registered models / versions / logged models +16, evaluation datasets +12, training +8; Forecasting untouched).

Out of scope: retraining, rescoring, loading models, editing historical families, cloud, lifecycle close/archive,
push/merge.

## 9. Evidence (read-only; `p1-formal-d30-logs/mlflow-plan-r2/`, copied to task evidence)

| File | SHA (first 16) | Content |
|---|---|---|
| `dataset_descriptors_v2.py/.json` | `31b1eae8d616d8f4` / `2efda436422300e3` | exact-text identity, 12 registered descriptors |
| `p6_truth_compare.py/.json` | `eb4de4e4637d4c8a` / `29b00b5415aeddc3` | same keys/phase, q3 text diffs vs archived P6 |
| `training_descriptors_v2.py/.json` | `b63bb9e12ac112fc` / `b76c2ab2170d944c` | 8 training descriptors |
| `digest_roles_v2.py/.json` | `de02da46d6b46105` / `bd8e3427d6423929` | all 1,298 leaves, typed |
| `digest_roles_v2_annotations.json` | `7eb7a54abb61ba34` | the 24 non-file/external leaves and 32 context classes, annotated; one completed scan, its sibling-repo indexing was broader than needed and is not repeated |
| `meeting-note.r1-2efc1ab5.md` | — | snapshot of the reviewed note before D1 corrections |
