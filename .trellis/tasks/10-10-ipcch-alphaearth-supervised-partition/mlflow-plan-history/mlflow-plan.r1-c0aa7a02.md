# Offline MLflow registration plan — IPCCH AlphaEarth supervised partition (R15, design §14)

Status: **PLAN ONLY, for supervisor review** (2026-10-14). No adapter code is changed, no store is written, no
server is started. Every count and identity below is to be frozen again from a fresh read-only baseline immediately
before any released write.

## 1. Target

- **Deliverable:** the existing local store (`/home/swl007007/.local/share/ipcch-mlflow/`, MLflow 3.17.0), using the
  existing experiments `IPCCH - detailed runs` (import contract `.trellis/spec/backend/local-mlflow-import.md`) and
  `IPCCH - dashboard` (Summary contract `local-mlflow-summary-catalog.md`). This is one completed family, added
  additively.
- **Not a deliverable:** a separate or isolated dashboard experiment or store. A scratch store restored from a
  backup (own server on port 5001) is used only for rehearsal and evidence, then discarded.
- **Untouched:** every existing object, in all four experiments (including `IPCCH Forecasting - *`), every
  registered model / version / logged model, and every metric value and run ID.

## 2. Baseline (supervisor read-only SQLite query, 2026-10-14; re-freeze before writes)

| Object | Count |
|---|---:|
| exp 1 `IPCCH - detailed runs` (FINISHED, active) | 126 runs |
| exp 2 `IPCCH - dashboard` | 136 runs; plan fp `c7abf4cdc2f5abe8f85300ad41fcf08085ceb14f170b81121e5f1605365158dc`; `catalog_status=complete` |
| exp 3 `IPCCH Forecasting - detailed runs` | 202 runs |
| exp 4 `IPCCH Forecasting - dashboard` | 174 runs |
| global registered models / model versions / logged models | 180 / 226 / 226 |

The IPCCH dashboard plan on disk (`dashboard/plan-c7abf4cd….json`) covers that dashboard's slice:
- 136 rows, 68 registered models, 100 versions;
- 74 datasets (66 evaluation + 8 training);
- 6 families.

The `EXPECTED = {rows 136, registered_models 68, model_versions 100}` in `summary_catalog.py` is that slice, not the
global counts.

**Before/after protection** (the snapshot is taken before writes and compared after):
- per-experiment run counts;
- per-run ID, name, status, tags, metric keys/values/steps and inputs;
- per-registered-model and per-version names, sources, tags and run links;
- per-logged-model ID, name, status and tags.

This covers all four experiments. The only permitted differences are the additions in §4 and the two
experiment-level edits in §5.

## 3. New family (sources.json + naming.py)

- **Family:** key `alphaearth_partition`; source run `p1-formal-d30-20261014`; new `format: "alphaearth"`.
- **Root:** `/home/swl007007/ipcch-alphaearth-runs/p1-formal-d30-20261014` (outside Dropbox).
- **Inputs:** report `report/report.json`; predictions `predict/predictions_h{H:02d}.csv.gz`.
- **Include:** `report/**`, `predict/**`, `develop/**`, `maps/**`, `replay/**`, `descriptors/**`, `prepared/**`
  (manifests and small tables), `archive/**` (failed replay attempt 1 plus the corrupt provider bytes, with
  `ARCHIVE.json`).
- **Bundle:** `models.tar` = `models/**`. That is 61 quartet identity directories (each `q2..q5.ubj` +
  `record.json`, 244 boosters), about 722 MB.
- **Exclude by path+SHA:**
  - `sources/**` (raw inputs, ~6 GB);
  - `prepared/X296_h*.npy` (large matrices);
  - `extraction/pieces.csv.gz` (kept; its SHA is in `extraction/COMPLETE.json`).
- **Extras:**
  - package source `IPCCHAlphaEarthExperiment/{ipcch_alphaearth,config,scripts}/*` and README, bound to the
    analysis code identity `385a8db8…` (with the original identity `57ca2ef1…`, amended only in `replay.py`);
  - the task's `*.md` files and `evidence/p1/formal-d30-20261014/**`.
- **Tags:**
  - `scientific_acceptance`: accepted, receipt `0b050601`;
  - `lifecycle_status`: open, delivery pending;
  - `features`: compact296 / compact296+AlphaEarth128;
  - `maps`: "none selected — B2 fallback at every lead";
  - `training_window`: annual refit, all labels to the fit origin, 24-month half-life;
  - `periods`: main 2023–2025 plus supplement 2026-01..04.
- **Inventory policy** (every `*sha256` key found in the run's JSON):
  - `required_digest_keys`: `booster_sha256`, `X_sha256`, `y_sha256`, `artifacts_sha256`, `report_sha256`,
    `replay_sha256`, `selection_sha256`, `file_sha256`, `map_freeze_sha256`, `prepared_manifest_sha256`,
    `descriptor_table_sha256`, `extraction_complete_sha256`, `plan_sha256`, `pieces_sha256`, `manifest_sha256`,
    `original_sha256`, `reconstructed_sha256`, `after_sha256`, `sha256`;
  - `informational_keys`: `features_sha256` (ordered-name hash), `resolved_config_sha256`, `config_sha256`,
    `compact296_list_sha256`, `request_sha256`, `receipt_sha256`, `regression_test_source_sha256`,
    `replay_diff_sha256`, `supervisor_test_log_sha256` (no single-file form in the run root);
  - scoped `exempt`: `geometry_sha256`, `list_images_sha256`, `source_manifest_sha256` and `sources/**` input SHAs
    (external inputs referenced by path+SHA, not retained);
  - `model_contract: "xgb"`.
- **Arm vocabulary** (naming.py additions; existing entries unchanged):

  | Source | Readable arm | arm_role | Meaning |
  |---|---|---|---|
  | B0 | `pooled` | comparator | compact296 pooled XGB quartet |
  | B1 | `pooled_plus_alphaearth` | feature_arm | same + 128 AlphaEarth descriptors |
  | B2 | `shared_calibration` | calibration_baseline | B1 raw outputs + shared ridge calibration |
  | G | `regional_calibration` | candidate | B1 + shared + regional calibration. No map was selected, so G is bitwise B2 at every lead (stated in tags and description). |
  | persistence | `persistence` | — | existing meaning |

- **Metric mapping:** existing leaves only.
  - `binary.*` and `four_class.*` as they are; `q3_r2_projected` → `share_phase3plus_r2`; n → `n_rows`.
  - Periods: `primary` = main (spans equal to the existing names, e.g. `2023-02..2025-10` at 1-month);
    `holdout` = supplement `2026-01..2026-04`.
  - The detailed child also keeps `q3_r2_raw`, the annual blocks (if the existing `year_YYYY` role accepts
    2023/2024; otherwise they stay as artifacts) and the bootstrap G−B2 record. The primary contrast G−B2 = 0
    stays in detailed records and artifacts.
  - The dashboard keeps only the existing contrast names `minus_persistence` and `minus_pooled`. Pooled means B0
    here; values come from the report's delta block, so there is no interval except where a saved bootstrap exists.
  - No new dashboard metric names.

## 4. Frozen expected additions

| Object | Added | IPCCH-slice total after | Global after |
|---|---:|---:|---:|
| detailed runs (exp 1) | 1 family parent + 20 children (4 leads × B0/B1/B2/G + 4 persistence) | 147 | — |
| dashboard rows (exp 2) | 20 (single run; seed tag 42 = contract XGB seed; persistence `none`) | 156 | — |
| registered models | 16 (`IPCCH AlphaEarth supervised partition \| <arm> \| <lead>`) | 84 | 196 |
| model versions | 16 (one per registered model) | 116 | 242 |
| logged models (external, not loadable) | 16 | — | 242 |
| evaluation datasets | **0 new: all 16 reused** (see below) | 66 | — |
| training-pool datasets | 8 (`compact296` × 4 leads; `compact296+alphaearth128` × 4 leads; B2/G link the B1 pool) | 82 | — |
| exp 3 / exp 4 (Forecasting) | 0 | 202 / 174 unchanged | — |

**Dataset compatibility evidence:** `p1-formal-d30-logs/mlflow_dataset_compat.json`, built with
`mlflow_dataset_compat.py` using `eval_descriptor`'s exact key and truth formula.
- All 16 AlphaEarth cohorts (4 leads × main/supplement × all_scored/persistence_available) have keys and truth
  SHA-256 identical to the existing datasets of the same generated names. For example,
  `IPCCH eval | 1-month | 2023-02..2025-10 | all_scored`: n 17,322, keys `3f9f16e3…`, truth `2bd690a3…`.
- So no name ever gets a second digest, and comparisons with P6, yearly and MLP rows are on identical datasets.
- Persistence F1 on `persistence_available` also equals the existing rows exactly (.7691/.7645/.7657/.7525).

Model descriptors are external composite bundles (design §14; no inference flavor):
- B0/B1 versions list that lead's main+supplement quartet identities: 4 per lead, 3 at 12-month; 15 per arm across
  all leads. B1 versions also list the lead's historical-provider and bridge identities.
- B2/G versions additionally reference `predict/contexts_h*.json` (calibration records) and `maps/MAP_FREEZE.json`
  (`map: null`).
- All of them point to `runs:/<family parent>/models.tar` + its member manifest.

## 5. The global plan-fingerprint lock — smallest additive change

Today `summary_catalog.py` hashes the WHOLE projection (`version, rows, models, datasets`) into one fingerprint. It
refuses any apply whose fingerprint differs from the experiment tag `zz_prov.dashboard_plan_fingerprint`, and it
hard-codes `EXPECTED`. Adding a family therefore always conflicts.

Proposed minimal change (one new mode; existing modes and behavior unchanged):

1. `plan` computes the full projection as now. It also computes the **legacy sub-projection** (families present in
   the predecessor plan file), which must reproduce the predecessor fingerprint `c7abf4cd…` exactly. That is the
   legacy compatibility proof: the code and naming edits change nothing for old families.
   - The plan records `predecessor_fingerprint`, `predecessor_plan_sha256` and the addition set.
   - `EXPECTED` becomes plan data: legacy {136/68/100} + additions {20/16/16}, checked at plan time.
2. `apply --extend` succeeds only when **all** of these hold:
   - the experiment tag equals the predecessor fp;
   - `catalog_status=complete`;
   - a fresh backup matches;
   - `verify` of every legacy object against the predecessor plan passes **before** writing.

   It writes only the addition objects (tagged with the new fp). Then it sets two experiment-level values: the
   `zz_prov.dashboard_plan_fingerprint` experiment tag to the new fp (plus
   `zz_prov.dashboard_plan_lineage = c7abf4cd…,<new>`), and the experiment description (reading guide plus the
   new arm vocabulary). These two edits are the only changes to an existing object. Legacy runs, models, versions,
   logged models and inputs keep their IDs, values and their own `c7abf4cd…` object tags.
3. `verify` checks legacy objects against the predecessor plan and additions against the new plan. A repeated
   `apply --extend` with the new fp is an all-noop. An interrupted extend resumes through a new-fp journal.

Rejected alternatives:
- re-tagging all 136 legacy objects with the new fp (mutates history);
- a full rebuild into a fresh store (new IDs);
- an isolated dashboard (not R15).

The detailed import needs no lock change. Family fingerprints include only the family's own config plus
`IMPORTER_VERSION` (`ipcch-mlflow-import-v2`, kept). Proof: after the edits, `import_runs.py plan` must
reproduce each of the six existing families' stored `zz_prov.import_fingerprint` exactly, and `import` for them
must be a noop.

## 6. Code changes requiring a separate release (not done)

The changes are in `IPCCHMLflow/`:
- `extract.py`: `_alphaearth_children` + EXTRACTORS entry, read-only;
- `naming.py`: family and arm vocabulary;
- `sources.json`: the new entry;
- `summary_catalog.py`: predecessor/extend mode, plan-held EXPECTED.

GitNexus impact is to be run per symbol before editing (`build_plan`, `_dashboard_experiment`, `apply_plan`,
`verify_all`, `check_expected`, `extract`).

Focused tests: the existing `IPCCHMLflow/tests` suite unchanged and passing, plus:
- AlphaEarth fixture extraction: names, NA, G≡B2 note, inventory classification complete;
- legacy plan fingerprint reproduced (`c7abf4cd…`) and six family import fingerprints reproduced after the edits;
- extend in a scratch store:
  - legacy objects byte-identical (`rebuild_check.py` all values unchanged);
  - only the additions created;
  - repeat apply noop;
  - interrupted extend resumes;
  - refusal when the experiment fp is neither predecessor nor new;
  - refusal when a legacy object drifted.

## 7. Execution order after the release (each step gated, evidence saved)

1. Fresh read-only baseline (§2), saved and hashed. `backup_restore.py backup` → restore-check into scratch
   (port 5001).
2. Scratch rehearsal:
   - `import --family alphaearth_partition` → `verify`;
   - Summary `plan` (pin the new fp + predecessor) → `inventory` → `apply --extend` → `verify`;
   - repeat apply = noop;
   - `rebuild_check.py` old vs scratch = every legacy value unchanged;
   - all-experiment before/after diff = additions only.
3. Supervisor review of the rehearsal evidence and the pinned fingerprints, then a live-write release.
4. Live: re-freeze the baseline and take a fresh backup, then repeat step 2 on the live store, then repeat the
   all-experiment before/after protection diff. The service must be stopped/started only via `manage.sh`;
   store jobs run under `setsid nohup` with logs outside `/tmp`.

Out of scope: retraining, rescoring, loading models, changing historical families, cloud, task close/archive,
push/merge.
