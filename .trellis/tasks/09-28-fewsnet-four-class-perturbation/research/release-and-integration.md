# Release and integration facts — 2026-09-28

Read-only source/ZIP inspection for planning, not execution or validation of a
scientific result. Paths below are relative to GeoRFBaseline unless specified.

## Starting release

`releases/georf-baseline-v0.1.0.zip` SHA256:
`39a26138e3fafb0be2bbd22e9760095d6cdefa7d79b98b4f798cb3aa79b500a0`.
All 46 ZIP files matched their live package counterparts byte-for-byte.
Root repository code is not interchangeable: root Stage 3 and RF helpers may
enable SMOTE, while this release disables it. Start from verified release payload.

## Defaults and contracts

- config.py:174-175: depth 1..6; :214-225: branch/scan minimums both zero.
- config.py:234-244: validation ratio .20; per-group minimum validation 1,
  singleton stays in training, group-split random_state=42.
- src/model/GeoRF.py:45-52: RF 100 trees, depth None, random_state=5;
  Stage 1 N_JOBS is memory-dependent (config.py:178-207), not a pinned constant.
- scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py:65-95: Stage 3 RF
  100 trees, depth None, seed5, n_jobs1; minimum local training rows50.
  :231-262: fewer than2 classes falls back pooled; no need for all4 classes.
  :1150-1166: default hard decision is predict(), not validation threshold tuning.
- Stage 2 step3 :76-80,111-122 fills admin universe0..5717 with s-1;
  step4 :79-105 defaults to the union of in-scope codes, not full universe.
  step4 :164-196 requires coordinates and uses sigma_degrees5.0;
  step5 :16 uses k40; step6 :77-99 obtains the cluster count from the report,
  not a hardcoded4; :183-195 uses spectral seed42 and 1-NN component assignment.
- TESTED_ENVIRONMENT.json:2-21, requirements.txt:1-16: Windows Python3.12.10,
  numpy2.2.6, pandas2.2.3, polars1.27.1, scipy1.15.2, sklearn1.6.1,
  geopandas1.0.1, shapely2.1.0. See full release files for remaining dependencies.
  VALIDATION.md:30-38 explicitly disclaims a prior complete real-data Stage1/3 run.

## Stage 1–2 score handoff

The existing Stage 2 scores are Stage 1 monthly held-out test scores, not the
within-training-window validation scores used for tree splitting. See
app/main_model_GF.py:611-624,735-758; scripts/step1_merge_results.py:95-110,181-189;
scripts/step3_create_linked_tables.py:41-65; step4_similarity_matrix.py:50-64.
Preserve this distinction in the final design; every learning score remains
bounded by D10. Export explicit macro-F1 columns through all consumers.
src/metrics/metrics.py:15-38 currently fills absent classes' PR with other-class
means; replace that metric behavior to satisfy D6, not merely rename outputs.
app/main_model_GF.py:1407-1424 filters to class1 columns and must be adapted.

## Confirmed preprocessing problem — correction approved in D18

src/feature/feature.py:124-131 calls comp_impute on the full feature matrix before
fold splitting; src/preprocess/preprocess.py:87-95 fits impute_missing_values;
src/customize/customize.py:60-105 estimates maxima/means from that supplied X.
Future and validation covariates can therefore influence fill values. This
conflicts with a strict training-only preprocessing contract; merely changing
the outcome to multiclass does not address it. Approved D18 fix: retain max_plus
formula but fit imputation only on each estimator's real fitting rows, before
appending Stage 1 pseudo rows; carry the matching transform with its checkpoint.
The user adopted this correction after the source-based explanation.

Timing rules were subsequently approved in D19; exact feature expansion remains
pending. feature.py:58-65 infers time-variable columns from
full-panel variation; :92-95 and preprocess.py:265-273 use record shifts;
strict_lag.py:19-34 preserves L1 columns and therefore alone cannot prove origin
alignment. These are confirmed code patterns, not a quantified leakage impact.
Resolve the feature-information contract before claiming full temporal safety.

### Feature inventory follow-up

The original panel has 88 source columns; full scan found 1,029,240 rows,
259,440 with observed binary labels, and zero mismatches against raw phase>=3.
There are 69 non-outcome/non-expert raw environmental/economic predictors:
28 plausible static columns and 41 conservatively dynamic columns. Static roles
need explicit names and invariance checks, not full-panel learned role selection.
Observed history originally contains phase and binary versions at record offsets
4/8/12; changing to origin-relative calendar offsets needs an explicit manifest.

preprocess.py:606-670 emits only WFP_Price_m4/_m12, nightlight_m12 and
EVI_l1..l12 due to early returns inside loops. Rolling statistics at :625-626,647
follow grouped shifts with an ungrouped rolling operation and can cross area
boundaries. Preserve intended same-area definitions, not cross-area mixing;
exact retained engineered outputs remain subject to final feature review.

feature.py:99-113 retains FEWSNET_admin_code in X, while config.py:247-250
drops month and fews_ha in GeoRF. Do not silently assume an identifier-free
shared schema or silently add assistance features; inspect both RF consumers
and explicitly disclose the final shared feature contract.
main_model_GF.py:542-550,1264-1269 defaults to single-layer RF consuming X;
L1/L2 partitioning is used by the separate two-layer path, not this default.

## Further implementation checks

- Internal target codes0..3 must match synthetic class rows and probability axes;
  exported labels1,2,3,4or5 remain distinct from missing data.
- Root/child checkpoint provenance: GeoRF.py:398-404, transformation.py:170-176,
  :713-730 and train_branch.py:10-17. Inherited-parent copies must include any
  matching imputer; do not mix candidate child and parent transforms.
- GeoRF.py:444-456 saves X_branch_id before final contiguity processing;
  feature.py:297-368 reads that assignment for correspondence. Verify final
  assignments/predictions agree. merge/terminal.py:73-115 keeps first collision;
  detect same-area conflicting terminal assignments rather than silently choose.
- customize.py:439-448 also restricts training groups to test groups. Describe
  this inherited restriction instead of implying full geographic training coverage.

## Inspected input identities

Root: Analysis/1.Source Data (outside repository). Recheck before execution.

| Relative input | SHA256 |
|---|---|
| FEWSNET_forecast_unadjusted_bm.csv | 611f9e776380e28da3fc845888d66a117626f91b37868e236ce849d44bc8f651 |
| Outcome/FEWSNET_IPC/FEWSNET.csv | 8fdd4cca6f6ba26b84efc209c8eb36492e1257d51e24edd2c2ad4962df7b38d0 |
| FEWSNET_admin_code_lat_lon.csv | a06be85849bb726a4505ed284bed14b100b61f998fb4586a6e14439aca8a4bcb |
| Outcome/FEWSNET_IPC/FEWS NET Admin Boundaries/FEWS_Admin_LZ_v3.shp | 3aba66a6fbf6b2a8beb153df76a67662ce4e0a898fcb11e292a17cc174f5f742 |

The .shx/.dbf exist; remaining sidecars and full key/content validation still need
preflight. No new source, source edit, data migration or experiment is authorized.
