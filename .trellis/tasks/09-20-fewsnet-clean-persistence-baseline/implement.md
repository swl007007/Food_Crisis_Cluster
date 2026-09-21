# Execution checklist — pending explicit authorization

Do not run task.py start, create product code, fit models, commit or push during
this planning turn. All commands below involving the experiment are planned
interfaces, not currently existing or executed commands.

## 1. Bind the approved context and runtime

- [ ] Follow research/context-index.md and fully read the required contracts;
  automatic context injection intentionally omits oversized research files.
- [ ] Obtain explicit instruction to implement/run after final planning review;
  retain task status planning until then. Load trellis-before-dev and applicable
  package/layer guidelines; check worktree state without overwriting user edits.
- [ ] Pin reviewed plan/source commit, baseline ZIP and every payload hash;
  verify Windows Python 3.12.10 and GeoRFBaseline requirements before fitting.
  Record actual package versions, RF get_params and numerical/thread settings.
- [ ] Create only FEWSNETCleanPersistenceExperiment/ and fresh run-local roots.
  Reuse IPCCH baseline extraction/import guards without automatic IPCCH polygon
  patches. Preserve protected baseline and historical outputs unchanged.
- [ ] Before changing an existing function/class, run GitNexus upstream impact
  and report its scope; HIGH/CRITICAL findings require explicit warning. Keep
  the core algorithm unchanged and prefer its existing explicit-array/split API.

## 2. Implement data preparation and preflight

- [ ] Implement prepare_data.py with pinned source inputs, master/ledger D63
  validation, raw artifact provenance, canonical unique keys and preserved grid.
- [ ] Materialize exact D52-D54 ordered schemas, boolean AEZ encoding and time
  roles. Validate source numeric types/finite values, annual/static agreement,
  source identities and geometry IDs. Never silently coerce unknown tokens into
  scientific missingness or guess source sentinels/units.
- [ ] Join the selected ENSO, D64 WB fields and signed coastline values; preserve
  Bloomberg monthly exports and all 18 fields. Verify WB raw/derived lineage
  before attaching market IDs; explicitly implement deterministic ties.
- [ ] Build approved calendar transforms and source-date lineage on the complete
  grid. Construct A-E/reference without secondary horizon shifts or label-first
  row lags. Keep missing inputs missing until the fitting stage.
- [ ] Save source/schema/role/coverage manifests, run-local geometry/cache identity
  and the exact finite candidate/fold schedule. Reject used independent run roots.

## 3. Implement one Stage 1 worker and the recipe/role schedule

- [ ] Implement run_pipeline.py around the verified baseline core. Each candidate
  is an isolated process with explicit seed/cwd/inputs. Precompute released
  within-area validation assignment, fit max_plus on real fitting rows only,
  transform the rest and pass the unchanged split into GeoRF.fit.
- [ ] Preserve F1 strict >.01, class recovery, q/depth/spatial settings and no
  SMOTE. Record effective RF/thread settings and real versus artificial support.
- [ ] Emit the compatible candidate score/assignment artifacts required by
  released Stage 2 helpers. Bind every job to recipe/schema/window/scope/date.
  Reject incomplete eligible jobs; record unsupported candidates distinctly.
- [ ] Deduplicate only matching jobs across overlapping windows. Schedule the
  663 development upper-bound jobs first; defer the 36 additional final jobs
  until recipe selection. Avoid concurrent full matrices for all recipes.

## 4. Implement consensus and geographic routing

- [ ] Apply D55-D58 eligible-ledger pooling, weights, spatial affinity and top-k;
  select/fill the same core graph. Pin eigsh initialization; retain spectral seed.
- [ ] Keep D19 nonpositive weights, supported automatic nc=1 and failed graph
  states separate. Save graph/node/component/eigengap evidence.
- [ ] Apply D59 fixed-core 100 km completion and D60 no extra smoothing, retaining
  full master-area coverage and reasons. No chained donors or hidden refined maps.

## 5. Implement development predictions and freeze

- [ ] Fit Stage 3 RFs with the D61 full pool/shared imputer and consistent native
  hard/probability fallback. Implement D62 full-pool reuse for valid nc=1.
  Preserve the exclusive-origin training mask; no IPCCH XGBoost/threshold import.
- [ ] Produce all recipes/reference's 2018 and 2020 predictions on their correct
  earlier maps; attach exact-origin valid persistence independently of RF fitting.
- [ ] Reuse the existing month-pool calibration and threshold algorithms with
  explicit new inputs; support fs3 and tau=null without old runner side effects.
- [ ] Independently reproduce 2020 common-key six-cell scores, threshold ties,
  feature-count/manifest tie-breaks and the selected recipe. Freeze identities
  before any final-period model scoring; do not tune from pilot results.

## 6. Run final evidence and report

- [ ] Build only winner/reference final maps and required new candidate jobs;
  verify final target starts and all upstream cutoffs before predicting.
- [ ] Preserve per-row results, training/imputation/map/model-route provenance,
  native RF decisions, raw/calibrated probabilities and correction reasons.
- [ ] Implement report_results.py to recompute paired/full-support metrics,
  common-calendar view, D44 shared bootstrap and leave-one-year-out checks.
- [ ] Report primary/secondary comparisons and D45 complete-pass/complete-fail/
  incomplete status with all source/inference limitations. Do not implement
  fallback research or re-open feature search from final results.

## Validation commands and evidence

Use the verified Windows Python executable (not the planning Linux scanner).
The script interfaces below are to be implemented as a small explicit CLI;
replace NEW_RUN only with a fresh experiment-local path.

    python -m unittest discover -s FEWSNETCleanPersistenceExperiment/tests -v
    python FEWSNETCleanPersistenceExperiment/prepare_data.py --run-dir NEW_RUN --preflight-only
    python FEWSNETCleanPersistenceExperiment/run_pipeline.py --run-dir NEW_RUN --stage pilot
    python FEWSNETCleanPersistenceExperiment/run_pipeline.py --run-dir NEW_RUN --stage development
    python FEWSNETCleanPersistenceExperiment/run_pipeline.py --run-dir NEW_RUN --stage final
    python FEWSNETCleanPersistenceExperiment/report_results.py --run-dir NEW_RUN --verify

The root may be created by preflight and continued by its bound stages; this
does not authorize overwriting a different completed run. Pilot artifacts may
be reused only when their entire recipe/job identity matches the formal schedule.
The pilot is the corrected original-feature reference's full development chain:
51 unique Stage 1 candidate jobs before support exclusions, its two complete
development map ledgers and 18 Stage 3 folds. Start with one eligible candidate
to catch wiring failures, then complete that chain before the other recipes.
Do not declare a partial candidate pool a successful formal map. These pilot
jobs are included in, not additional to, the development budget. The final
stage must refuse an absent/incomplete freeze manifest.

One focused contract test module should exercise meaningful scientific boundaries:
calendar/origin masks and lags; phase/missing persistence; exact source joins,
WB ties and AEZ parsing; training-only/all-missing imputation; real support and
F1/SMOTE invariants; graph selection/completion/no-split states; consistent RF
fallback; correction equality/null thresholds; common keys and joint bootstrap.
Use small synthetic fixtures for boundaries, then pinned real-source preflight.
Do not add tests that merely repeat constants or unrelated security scenarios.

Run existing baseline checks applicable to touched adapters/core boundaries,
including python -m src.tests.sig_test if split logic is changed (not merely
imported), and required package checks loaded by trellis-before-dev. Verify
protected hashes after the bounded chain and final run. Recompute outcome tables
from stored predictions independently; console logs alone are not evidence.

Before declaring experimental completion, fulfill A1-A64, resolve all required
failed evidence, save reproduction commands and an execution handoff. A complete
negative result can close; missing evidence cannot. GitNexus detect_changes is
required before any later separately authorized commit. No commits/pushes or
task closure are authorized by this planning artifact itself.
