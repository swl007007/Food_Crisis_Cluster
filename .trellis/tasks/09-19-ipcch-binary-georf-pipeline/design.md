# IPCCH GeoRF technical design v1.0

Status: planning, against prd.md v1.0. Scientific decisions are approved; final
implementation approval is pending. This document defines the smallest isolated
adaptation, not a completed implementation or measured result.

## Boundary and file responsibilities

Add one experiment directory, `IPCCHGeoRFExperiment/`, with:
- `prepare_data.py`: pinned input/QC, calendar features/history, original-outcome
  split and geographic preparation; plain functions, no adapter class framework.
- `run_pipeline.py`: one CLI, isolated baseline imports, single Stage1, completion,
  monthly Stage3 fitting and durable run status.
- `report_results.py`: shared cohorts, confusion-count metrics and country bootstrap;
  callable by the runner and separately from saved predictions.
- `test_contracts.py`: compact runnable checks for temporal/target/spatial/model/
  reporting invariants; reuse frozen baseline tests for its own mathematical core.
- `README.md`: exact invocation, contracts, observed validation and limits.

No new package architecture, model registry, generic experiment framework, DAG,
parallel scheduler or additional dependency is needed. Keep the root application,
config, batch scripts, historical experiments and released baseline unchanged.
Generated data/models/logs stay under `IPCCHGeoRFExperiment/runs/<run_id>/` and
outside Git. Do not commit raw data or model checkpoints.

Run directory structure (directories, not new service/module abstractions):

```text
runs/<run_id>/
  manifest.json                    # status, source/code/runtime/config identities
  run.log
  baseline/GeoRFBaseline/           # extracted pinned source; local compatibility patch
  data/                           # target QC, feature rows/schema, country/area universe
  geography/                      # original checks, repaired copy, adjacency/provenance
  stage1/                         # split, checkpoints, explicit/complete maps, diagnostics
  stage3/                         # fold records, shared training identities, predictions
  reports/                        # cohorts, metrics/deltas, draws/CIs, limitations
```

## Source and runtime preflight

1. Verify exact CSV/release SHA256 from PRD; never substitute similarly named
   corrected inputs. Hash geometry sidecars, coordinate/country CSVs and executable
   experiment sources. Include CRS, file sizes and module paths in manifest.
2. Verify ZIP CRC, one top-level `GeoRFBaseline/` and its MANIFEST payload hashes
   before/after extraction to a fresh run. Record pristine identity and each local
   patch separately; the original archive/source is immutable.
3. Use the preferred Windows Python3.12 executable. Recorded metadata evidence:
   Python3.12.10, NumPy2.2.6, pandas2.2.3, sklearn1.6.1, XGBoost3.0.0,
   GeoPandas1.0.1, Shapely2.1.0. Verify the actual runtime before execution;
   no silent interpreter/dependency substitution. This is a tested environment,
   not a clean-install or fully transitive lock claim.
4. Validate required columns, canonical integer IDs, unique area-month keys,
   valid calendar months and complete lookup identity. Parse source missing tokens
   deliberately; country codes must not be lost to generic CSV NA interpretation.

The local ETH forecasting guideline supplies useful identity/non-overwrite checks,
but its88features, median imputation, SMOTE, expert anchors and ETH output root are
experiment-specific. IPCCH's explicit PRD supersedes those clauses; no additional
approval is needed to follow the already-approved IPCCH contract.

## Data flow and tables

Use month ordinals for arithmetic and export YYYY-MM strings. The canonical
outcome ID is `(area_id,target_month)`; the expanded model-row ID adds
`horizon_months`. Every join is explicit and validated for cardinality.

1. Read the monthly scaffold and R1 source strings. Build valid truth with decimal
   comparisons, retaining all QC flags/reasons and normalized shares in a target
   ledger. Do not fill missing labels or remove scaffold months before features.
2. Build ordered features only for valid target rows and their four horizons;
   the unlabeled scaffold and full valid-label history remain lookup sources.
   This bounds stored feature rows while retaining all potential supervised/test
   rows, including areas with little history. Missing covariates do not delete rows.
3. Use keyed calendar joins/area-specific reindexing for raw values, sums and lags;
   use same-area as-of history searches with source month<=own origin. No helper
   may shift a sparse row list to stand in for a calendar month.
4. Preserve a pre-imputation feature table, sorted schema and provenance columns.
   Dates/IDs, source-row identities, latest-label month and latest-positive month
   stay outside X. Temporal derivative recipes specify the remaining source months.

Final X has **93 columns**:70raw +15derivatives +2calendar +3latest-label/history
+2crisis-recency +1numeric `horizon_months` indicator. Horizon uses actual1/3/6/12,
not legacy scope indices. Use one ordered schema in both stages; Stage3's horizon
column is constant within a fit. Raw lat/lon predictors use the selected CSV at O;
nearest-donor distance uses the separate keyed reference coordinates.

History feature names: `last_observed_label`, `last_observed_label_age_months`,
`no_observed_label_history`, `months_since_last_observed_crisis`,
`no_prior_observed_crisis`. The first field denotes the latest observed binary
crisis status, including0; it is distinct from the last positive observation.
Metadata: `last_observed_label_month`, `last_observed_crisis_month`.
Use target_month_sin/cos and the exact15names in secondary-predictors.md.

CSV/CSV.gz and JSON suffice for tabular audits and metadata; keep NaN explicit.
Use existing checkpoint serialization for models. No new storage dependency.

## Geographic preparation

Normalize shapefile admin_code and reference/country area_id keys explicitly.
Require the exact6,227ID universe, unique one-feature-per-area Polygon/MultiPolygon,
known CRS, finite bounds and valid reference coordinates. The current source is
EPSG:4326; preserve this source CRS and reference coordinate meaning.

Follow Q8g: Shapely make_valid only on invalid geometries, in the run-local copy.
Keep valid shapes byte/geometry-equivalent where format allows; record original
and resulting geometry hashes, reasons, types and validity. Compare footprint
using a declared geographic/geodesic area calculation, not square degrees as km².
An invalid original's area is only diagnostic, not ground truth. Record adjacency
edge changes where original operations are defined; topology errors mean an
unavailable comparison, not a fabricated zero change. Do not create an arbitrary
unapproved percentage-tolerance gate.

Any empty/invalid/non-polygon result, lost/duplicate ID or ambiguous identity stops
before learning. Do not extract polygon pieces out of GeometryCollections, replace
boundaries or drop areas. No repair method escalation is authorized by Q8g.

Reuse `create_polygon_adjacency_matrix` on this validated local geometry with the
explicit ID field. Check input before invoking its permissive fallback behavior.
Its adjacency is positive-length shared boundary, excluding point contacts.
Convert returned source-ID->index to the index->area-group mapping required by
GeoRF; adjacency keys are polygon indices. Keep one polygon record per group.
Retain its centroid array for inherited polygon refinement; it is not donor-distance
authority. New run-local caches must record geometry component hashes, not merely paths.

Read-only facts:253invalid shapes (212self-intersections,30ring intersections,
5nested shells,6too-few-point components). No repair-success claim exists yet.
Topology validity cannot prove correct original boundary assignment; retain the
upstream unrestricted-nearest-match provenance limitation.

## Stage1 integration and original-outcome support

Generate the original2014–2022outcome manifest before horizon expansion: n>=2areas
use earliest floor(n/2) fit/latest ceil(n/2) validation; singleton outcomes form a
separate supplementary table. Zero-label areas remain in the geographic universe.
Audit no shared original IDs across split sides and exactly four views per included
outcome. No missing-feature filtering; incomplete inputs follow Q6c.

Fit one existing OutOfRangeImputer on genuine fitting views; transform Stage1 and
singleton matrices with the stored values. Reuse the instance/fill parameters
across root and child fits. Never call comp_impute separately on held-out matrices.

Before core imports, put the extracted baseline root first on sys.path, clear any
external PYTHONPATH influence for the run, load its config and disable FEATURE_DROP
(including its alias) because IPCCH already supplies the exact schema. Record and
assert `config`, `src.*` and any `config_visual` module locations resolve into the
extracted baseline. Do not import the production/FEWS feature pipeline or lag validator.
No altered ACTIVE_LAGS is required for direct prepared-array fit.

Call GeoRF with baseline partition depths MIN_DEPTH=1/MAX_DEPTH=6 and unchanged
model defaults, then `fit(X,y,X_group,split={"X_set":...},contiguity_type="polygon",
polygon_contiguity_info=...,feature_names=...,print_to_file=False,
track_partition_metrics=False,VIS_DEBUG_MODE=False)`. Validate lengths,0/1split
values and complete polygon mapping before the call. Use actual non-contiguous
integer IPCCH IDs, not grid-cell offsets.

Preserve trees100, tree depthNone, seed5, F1/q/delta>.01, no SMOTE, class-recovery
pseudo rows and baseline resource-dependent Stage1 RF threading. Stage3 is fixed
to one thread. Record actual estimator settings; outer GeoRF kwargs alone do not
establish them. Preserve and record inherited random seeds/diagnostic behavior.
The existing diagnostic CV path can still run despite its map-disable flag; it
does not learn another partition or add a benchmark arm. Its scores are not final
evidence, and its presence does not relax mandatory artifact checks.

There is one necessary local source patch: in `src/model/GeoRF.py`'s final grid-only
refinement block, restrict `if CONTIGUITY` to non-polygon mode. Training's polygon
refinement stays enabled. Source evidence: `GeoRF.py:438-456`,
`transformation.py:390-407,643,671-730,901-964`. Polygon-refined candidates already
pass the F1 gate before s_branch/checkpoints are saved; a later grid reassignment
can corrupt correspondence without updating those models. Skipping that inapplicable
postprocessing keeps accepted assignments and exports consistent. Match the exact
pristine block/hash, save the local diff, and test the guard. No baseline math changes.
GitNexus impact must precede editing this symbol during implementation; if the new
copy is not indexed, use its indexed lineage plus explicit source/caller evidence
and record that index limitation.

GeoRF ignores constructor output dir in create_dir; run its fit with cwd isolated
under stage1 so result_GeoRF/checkpoints cannot escape into existing workspaces.
Restore cwd reliably. Required model/assignment outputs must exist even when
optional rendering errors were caught by inherited code.

Current effective support gates (config values0) require positive validation
existence/nonempty child subsets; four complete views preserve these predicates.
Flex balancing counts groups. Count original IDs for eligibility/donors/reports;
do not divide q's D/A/error-mass statistics by4 or alter RF internal row-level
tree defaults. F1 combines horizon-view predictions as approved. Evidence is in
convergence-check.md; no general nonzero support-threshold system is needed.

## Map, donor completion and singleton scoring

Build explicit learned mapping from s_branch/branch_table and actual split
participation; compare with saved X_branch_id and model routing. Preserve branch
strings, including leading zeros and genuine root/ancestor membership. Missing
membership must remain unassigned even if an inherited helper defaults to root.
Multiple ancestor memberships are lineage; incompatible terminal assignments
are errors. Do not accept a first-row collision resolution as evidence.

Maintain a stable integer partition code for the existing Stage3 helper plus the
original branch string; unresolved uses-1. This is serialization, not reclustering.
Donors satisfy Q5d, using original outcomes and explicit membership. Search eligible
donors once with haversine/great-circle distance (Earth radius6371km), sorted area
IDs for stable tie ordering, using existing sklearn neighbor machinery. Apply100km
inclusive to both nearest-donor choice and attachment. Never chain through recipients.
Record donor, distance, assignment source and fallback status for every area.

After map freeze, route singleton features to the donor's saved partition checkpoint
with the existing Stage1 imputer; unresolved cases use the saved root/pooled model.
Do not train on singleton truth or fit an additional model. Separate these scores
from learning validation and independent Stage3 results. Model adoption may mean a
branch checkpoint contains a parent RF; retain that provenance.

## Stage3 rolling forecast execution

Build a fold manifest for the PRD schedule:35/33/30/24main target months at h1/3/6/12
(122scheduled folds), plus available target months in2026as a separate period.
An empty test month is recorded with zero support; it is not a zero F1 or failure.
For each nonempty fold, use exactly its36calendar-month target-label pool, all
eligible areas, and historical row-specific feature origins. No test-area-only
training restriction. Empty global training is a reported stop, not a fabricated model.

Fit one common RF imputer, one pooled RF, supported local RFs and one pooled binary
XGB. Reuse the baseline Stage3 fit/probability functions; they already handle RF
single-class probabilities and pooled fallback for unseen partitions. Use the
probability path, because the inherited hard-label helper leaves unseen positive
partition codes at initialized0. Build hard predictions from p1>.5 for all learned
arms. XGB gets its own explicitly approved constructor; the helper's lower_model
argument does not implement XGB. No Stage1 pseudo rows in Stage3.

Training membership is common; local models use subsets of that pool selected by
the frozen completed map. A donor assignment grants partition membership; Stage3
can include the recipient's own available training outcomes. It is not restricted
to the donor area's data. Local<50rows or single class, unresolved or unseen local
models use the SAME pooled RF. Record assignment and model-fallback reasons separately.

Write prediction rows containing area/country/target/origin/horizon, truth, three
probabilities/hard labels, persistence/source date/age, raw branch/partition code,
assignment/donor provenance, actual model route and fold identity. No missing learned
prediction is allowed to shrink a cohort. Save training keys/settings/imputer values
per fold so labels and transforms can be reconstructed without storing every RF.

## Reporting and verification boundary

From saved rows, derive E_all/E_persist and keep main/partial2026 separate. Validate
keys, binary fields, country mapping and probability/prediction consistency before
scoring. Use keyed country names/IDs from the complete lookup even when ISO3 is
missing; no country row can disappear before bootstrap.

Compute the exact PRD confusion metrics/deltas and separate grouped diagnostics.
For each main horizon/cohort, sort countries and initialize default_rng(42), draw
1000paired country samples and preserve multiplicities. Export draws/statistics,
effective counts and defined-only percentile labels. Reuse the small existing
sampling pattern, not the legacy probability-dependent masking/metric implementation.
Use NumPy's standard linear percentile calculation, record it in reporting config.

The reporter accepts saved predictions and writes to a specified fresh report
directory for independent reconstruction. It never fits a model or changes the map.
No new figures/dashboard are needed to satisfy this task.

## Stops, status and rollback

Refuse existing run IDs. On any required check failure, save failed status/reason
and keep diagnostic artifacts; never call that run complete. Stop before fitting
on hash/schema/key/runtime/geometry violations. Stop on inconsistent learned maps,
no eligible donor, leakage, missing predictions or irreproducible reports.
Non-polygon repair outcomes invoke the Q8g stop, not a repair escalation.

Status is a small manifest updated through preparation/Stage1/Stage3/reporting and
complete/failed, not a resumable workflow engine. Use a temporary file and replace
for manifest writes. Failed runs remain immutable evidence; reruns get new IDs.
Rollback affects only the new experiment/run; originals and historical artifacts
have no migration step. Any required scientific change returns to grill before
implementation continues. Runtime, data provenance and geometry limitations remain
explicit even if every computational check passes.
