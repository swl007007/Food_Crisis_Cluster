# Reuse and evidence — planning inspection, 2026-09-21

Read-only inspection, not a new model run. Exact inspected source identities are
in planning-snapshot.json. Other sessions may change the repository; verify the
approved runtime/source snapshot again before implementation.

## Source, targets and features

- prepare_data.py:122 load_source_strings preserves raw tokens; :292
  build_target_ledger(path, verify_hash=True) returns the full scaffold/QC ledger;
  :145 TargetLedger.valid() selects valid labels. These are reusable directly.
- prepare_data.py:236-288,292-298,352-369 establishes valid distributions and
  exact binary labels. :210-222 compares 5*(P3+P4+P5)>S, not a rounded ratio.
  :53-59 uses 100-digit Decimal normalization. Keep label and regression target
  separate; do not overwrite the binary target used by existing feature helpers.
- prepare_data.py:26-33 pins source hash and 42695/15206/27489/6227 valid,
  positive, negative and geographic-area counts. :383 check_target_gate returns
  a result; a caller must explicitly fail when gate_pass is false.
- prepare_data.py:521-532 contains binary history only. :1082 build_feature_matrix
  builds original93 at horizons 1/3/6/12; :877 assemble_feature_matrix uses its
  input label table both as sample rows and history. Never prefilter that table
  to just development/test rows. :943-1006 implements each row's own origin.
- build_feature_matrix does not independently verify that source_path matches
  its supplied ledger; hash/bind them explicitly. The original 93-column schema
  is frozen in feature-schema.json; append new columns externally without editing
  the old experiment's FEATURE_COLUMNS or calling its width-limited audit on new X.
- prepare_data.py:1363 load_country_lookup reads country_area_id_lookup.csv,
  rejects duplicate IDs and all-empty country names, retains missing ISO3, and
  returns a hash audit. Assert coverage for all required area IDs in the new caller.
  It needs no polygon processing. Geographic dependencies are imported lazily.
- load_covariate_panel at prepare_data.py:627-690 converts existing invalid
  nonnumeric/infinite covariates to NaN with audit. Preserve and disclose this
  original behavior; new engineered infinities are errors, not silently coerced.

## Fitting reuse limits

- run_pipeline.py:1600-1604 uses same horizon and training targets in [O-35,O].
  :1634-1652 checks training target<=O, own origin<O, test origin==O and disjoint
  target keys. The new matched-history restriction must be added explicitly.
- run_pipeline.py:1271-1297 persistence lookup has no maximum age/36-month
  truncation, and :1358 uses each row's own origin. Do not use latest row before
  the refit's date for historical training features.
- run_pipeline.py:1198-1203 rejects early origins because of the old partition
  cutoff. Build the new small explicit calendar; do not reuse that guard.
- Existing XGB fits have no global single-class fallback (:1679-1680). The old
  local-RF fallback cannot fix this; the new contract must handle constants.
- Original RF params at run_pipeline.py:105-110; XGB params :115-139. Five XGB
  formulations reuse the common search inventory, with the regressor's objective
  and eval_metric substituted deliberately. Do not execute the old runner.
- baseline_runtime.extract_baseline verifies the ZIP and payload before extracting
  a fresh copy (:81-128). baseline_imports (:178-258) isolates config/src; it imports
  GeoRF classes even when only the imputer is needed. Reuse it without any model
  fitting or polygon patch. Check actual imported module locations.
- GeoRFBaseline/src/customize/customize.py:16-188 OutOfRangeImputer max_plus uses
  training max*100; max==0 ->100; all-missing ->0. Negative-column fills need not
  lie outside the training range. Fresh instance per fit pool; assert fit/width/
  names externally, retain every column. It has a bare config import (:9), so do
  not import through the unpinned repository root.
- The root scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py still has SMOTE
  in a different code path; do not use it as a substitute for the pinned package.

## Reporting reuse limits

- report_results.py:1270-1340 confusion_counts/class1_metrics can be reused.
  Undefined F1 denominator returns NaN/reason, not an invented value. Long-form
  new outputs need their own model/key checks; the old whole validator assumes
  old arm/column names (:679-683). :730-759 supplies useful time/key/country checks.
- :641-661 country keys cannot silently lose missing ISO3 rows; use the validated
  country lookup. Each area must have exactly one stable country key.
- :1674-1784 aggregates country confusion counts efficiently, but reseeds and
  draws within each horizon/cohort. Shared cross-horizon draws must be generated
  once on a union country axis. Same seed alone is insufficient.
- runtime_identity (:376-403) records most package versions, not an enforcement
  gate. New preflight must compare the explicitly declared pinned environment.

## Prior numerical evidence and limitations

Original same-key XGB/persistence F1 by 1/3/6/12 months:
.678929/.681392, .680740/.676817, .667058/.671861, .678704/.675917.
See IPCCHGeoRFExperiment/README.md:18-68 and original reports/main/metrics.csv.
The original geographic model learned no split; differences from pooled RF
include different fitting support. Do not call them a spatial-sharing benefit.

Latest CH/gate ablation has an open audit gate in the recorded planning snapshot;
aggregates and archived task status are not proof of complete acceptance. Existing
2023-2025 scores have already been inspected. This new design is retrospective.
Publication timestamps and upstream covariate/geometry provenance are not proven
by source-month alignment; this task preserves those limitations and does not
depend on geographic model fitting.
