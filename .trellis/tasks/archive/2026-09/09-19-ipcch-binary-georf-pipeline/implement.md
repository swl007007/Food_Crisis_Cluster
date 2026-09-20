# IPCCH GeoRF execution plan v1.0

Status: completed and archived. This is the original execution plan; its unchecked
boxes preserve the planning snapshot rather than represent current completion.
Verified A1–A8 coverage, deviations and test evidence are recorded in STATUS.md,
DECISIONS_LOG.md and IPCCHGeoRFExperiment/validation/review.md.

## 0 — Activation and scope guard (A1)

- [ ] After final-summary approval, activate this existing task; preserve unrelated
  baseline/archive/journal changes. Read trellis-before-dev and applicable local specs.
- [ ] Work only in IPCCHGeoRFExperiment and the task's evidence/context artifacts;
  add a narrow generated-run ignore rule if required. Do not edit root modeling
  code/config, existing release or raw sources.
- [ ] Use GitNexus impact before modifying any existing symbol, notably the run-local
  GeoRF.fit compatibility patch. Report direct callers/processes/risk; record
  unindexed-copy limitations and verify actual baseline source/callers. Run
  detect_changes before any eventual commit; do not commit/push implicitly here.
- [ ] Verify hashes, ZIP CRC/MANIFEST, source schemas and runtime. Log exact module
  origins to prevent root src/config shadowing. Record reproducible commands.

## 1 — Data/feature boundary and compact checks (A1–A3, A8)

Own files: prepare_data.py, test_contracts.py; README usage later.

- [ ] Implement decimal R1 target/QC ledger; keep unlabeled scaffold and exact keys.
- [ ] Implement the approved93-column schema: whitelist, calendar windows, sin/cos,
  numeric horizon, latest-label/history and positive recency. Preserve raw NaNs and
  source-month metadata; audit infinity conversion.
- [ ] Split original2014–2022outcomes before four-view expansion. Keep singleton
  and zero-label area roles explicit; generate membership/support/cutoff records.
- [ ] Reuse stored-fit OutOfRangeImputer. Genuine training only, validation/test
  transform only; stage-specific sharing and pseudo-row ordering match Q6c.
- [ ] Add runnable checks with small hand-computable inputs: exact.20/.90/1.10,
  permitted P5-only missingness, invalid components/population, gaps/cross-area
  calendar alignment, future-value exclusion, history at O, no positive history,
  duration origin versus target, odd/even/singleton split and four-view identity.
  Check negative/zero/all-missing RF maxima and held-out extrema not affecting fits.

Gate: label policy reproduces42,695valid/15,206positive and approved split capacity
before any new input discrepancy; all retained rows have the exact schema. A count
mismatch is investigated, not normalized away. No real model training at this step.

## 2 — Geographic audit and approved local repair (A1, A4)

Own: geographic functions in prepare_data.py; keep implementation minimal.

- [ ] Validate/hash shapefile components, reference coordinates and country lookup;
  normalize exact IDs, retain all6,227areas including missing-ISO3 countries.
- [ ] Apply Q8g make_valid only in an experiment-local copy. Save reason/type/
  validity/footprint/adjacency evidence and unavailable-comparison reasons.
- [ ] Stop on invalid/empty/non-polygon output, ambiguous identity or changed key
  universe; no component extraction or alternate repair/source strategy.
- [ ] Reuse adjacency helper on validated geometry, build exact index/group mapping
  and content-bound cache metadata. Keep donor reference coordinates separate.
- [ ] Compact checks: ID normalization, duplicate/missing mapping rejection,
  unchanged valid geometry, successful repair, non-polygon result refusal and
  point-only contacts excluded by the inherited adjacency definition.

Gate: geographic mapping/repair audit passes before any full Stage1 fit. If Q8g
stops, preserve evidence and ask one scope decision; do not proceed to training.

## 3 — Direct Stage1 and frozen map (A1, A3–A4, A8)

Own: run_pipeline.py; only the exact local extracted GeoRF.py block may be patched.

- [ ] Extract verified baseline into a fresh run; apply and save the small guard
  excluding polygon mode from final grid-only refinement. Keep training polygon
  refinement and all F1/q/checkpoint selection unchanged.
- [ ] Set schema-only config override before core imports; validate actual modules
  and effective RF/config values. Keep output cwd under stage1; do not rely on dir=.
- [ ] Call one GeoRF.fit with explicit X_set and polygon mapping, one imputer and
  complete four-view rows. Preserve original-outcome support reporting and active
  zero/nonempty core gates; no altered q counts or new minimum-support rules.
- [ ] Export learned map from accepted state; reconcile s_branch, branch_table,
  row assignments and actual model/checkpoint routing, preserving root/ancestor/
  leading-zero branches. Reject collisions/default-root donor substitutions.
- [ ] Complete unassigned areas with eligible-donor great-circle1-NN<=100km;
  stable donor ordering, no chaining. Save full universe, donor/provenance and
  reversible integer partition codes. Score singleton views separately afterward.
- [ ] Run baseline tests against the local copy, then compact synthetic integration
  checks for non-contiguous groups, export consistency, donor eligibility, exact
  distance boundary, out-of-cap pooled routing and recipient-own-feature use.

Gate: no mandatory artifact silently missing despite inherited caught exceptions.
Map, checkpoint and donor provenance agree; partition cutoff<=2022-12. Synthetic
checks are development evidence; they are not scientific forecast results.

## 4 — Four-arm rolling forecasts (A3–A6, A8)

Own: run_pipeline.py, extending the same compact checks.

- [ ] Create122scheduled main fold records plus separate available2026months;
  retain empty test-month records, but never fit an empty global training pool.
- [ ] Share36-month training keys and predictor information among learned arms.
  Fit one imputer, pooled RF and local RFs per fold; original rows only. Reuse
  baseline probability helpers, including single-class and unseen-group fallbacks.
- [ ] Construct XGB3.0.0 with every approved parameter and native NaN; no search,
  early stopping, calibration, class weights or pseudo rows. Verify actual params.
- [ ] Generate strict p1>.5 hard labels and latest-valid persistence; export complete
  row lineage, class probabilities, model/fallback routes and history source dates.
- [ ] Verify36calendar-month endpoints, own-origin historical features, no target
  or post-origin labels, frozen map availability and E_persist-independent training.
  Check all-negative/positive pooled RF probabilities, missing/unsupported/unseen
  local partitions and ties at.5. Use tiny synthetic fits where necessary.

Gate: all learned predictions exist on every valid scheduled test key. A missing
prediction stops reporting; no arm-specific complete-case mask is allowed.

## 5 — Metrics and uncertainty (A5–A7)

Own: report_results.py; callable from existing saved predictions.

- [ ] Produce E_all/E_persist keys/support/prevalence/coverage, main versus partial
  periods, confusion-count metrics/deltas and separate grouped diagnostics.
- [ ] Implement Q9b country draws and percentile intervals with identical samples
  per arm, preserved duplicates, saved draw identities and explicit undefined rules.
- [ ] Compact hand-computable checks: TP=0with positive denominator gives F1=0;
  all-negative/all-empty zero-denominator is NaN; paired keys exact; missing
  country/prediction fails; country multiplicity retained; deltas use the same draw;
  invalid replicate counts/CI suppression and seeded determinism are reproducible.

Gate: an independent calculation from saved rows reconstructs main confusion
counts/F1/deltas and selected saved bootstrap draws/intervals. No fitting in reporter.

## 6 — Approved full execution and handoff (all acceptance criteria)

- [ ] After implementation/compact checks pass, execute one fresh scientific run
  through preparation, geometry gate, one Stage1, completion, Stage3 and reporting.
  Do not vary protocol based on results. Investigate failures within approved scope;
  changed scientific choices return to grill.
- [ ] Verify raw/release hashes unchanged; required outputs complete, row/fold/map
  reconciliation passes, reports reproduce, and run status is complete only then.
- [ ] Record actual commands, environment, artifacts, exclusions/fallback coverage,
  runtime and limitations in README/task journal. Report null/negative comparisons
  as valid outcomes. No clean-install, real-time availability or geometry-identity
  claim beyond evidence.
- [ ] Final main-agent verification owns acceptance; exploratory agents may inspect
  evidence but cannot replace final verification or edit implementation.

## Commands and review points

These commands are planned, not executed scientific checks. Run Python commands
with the preferred Windows interpreter and Windows-native paths for CLI arguments.
From WSL the interpreter is
`/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe`.

```text
# Existing release verification (from GeoRFBaseline/releases):
sha256sum -c SHA256SUMS

# Existing baseline checks (cwd = extracted baseline root, PYTHONPATH removed):
python3.12.exe -B tests/test_baseline.py

# New compact contract checks, after implementation:
python3.12.exe -B IPCCHGeoRFExperiment/test_contracts.py

# New full runner, fresh run_id; source-root is the pinned assembled_IPCCH folder:
python3.12.exe -B IPCCHGeoRFExperiment/run_pipeline.py --source-root "C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\assembled_IPCCH" --run-id ipcch-v1-<timestamp>

# Independent report reconstruction to a fresh output directory:
python3.12.exe -B IPCCHGeoRFExperiment/report_results.py --run-dir <run> --out-dir <fresh-report-check>

# Task artifact validation:
python3 ./.trellis/scripts/task.py validate .trellis/tasks/09-19-ipcch-binary-georf-pipeline
```

Runner CLI stays limited to source location/run identity; no horizon, target,
threshold, feature or model-search knobs. The reporter only reads existing run
outputs. Source/archive hash+MANIFEST checks use stdlib hashlib/zipfile/json; no
installer or new dependency is needed. Baseline mathematical tests plus focused
new checks suffice; do not rerun unrelated main-pipeline/Stage2 suites.

Rollback: preserve failed run/status/logs, create a new ID for a corrected rerun.
Nothing overwrites raw data, frozen sources or prior results. Do not implement
automatic resume/caching infrastructure. A necessary deviation from PRD is a
planning change, never a silent workaround.
