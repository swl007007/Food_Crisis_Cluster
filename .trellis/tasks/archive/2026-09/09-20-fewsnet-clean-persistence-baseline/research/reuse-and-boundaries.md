# Reuse evidence and boundaries — 2026-09-20

Read-only inspection at ffdfd62. Two bounded code scouts supplied the anchors
below; the main session read the foundational documents and spot-checked baseline
contracts, extraction and feature generation. No model was run.

## Shared corrected baseline

- GeoRFBaseline/config.py:342-346 and src/partition/partition_opt.py:885-896:
  true class-1 F1 and exact strict gain above 0.01.
- GeoRFBaseline/src/model/model_RF.py:83-85,110-112: SMOTE=True rejected;
  original fitting rows retained. README.md:42-47 records Stage 1 class recovery.
- scripts/run_stage1.py:31-57: explicit inputs/scope/month/experiment root,
  fresh cwd and Stage 2 handoff. Wrapper does not expose strict-lag/random-seed.
- scripts/step1_merge_results.py:67-72 and step3_create_linked_tables.py:24-35:
  Stage 2 reads the supplied experiment root.
- scripts/step4_similarity_matrix.py:21-38,50-64,79-99,206-231: month-specific
  matrices, positive logit-F1 weights, and explicit admin-universe choices.
- scripts/step6_complete_clustering_pipeline.py:17,23-46,77-129: k=40, cluster
  count from report or explicit option; month suffix normalization needs wiring.
- scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py:810-845,969-988:
  explicit paths/windows/maps, including m2/m6/m10. It accepts existing output
  directories; a new wrapper must reject an already-used run root.
All scripts/src references in this section are inside GeoRFBaseline/.

## IPCCH extraction is reusable; its scientific shortcuts are not shared defaults

- IPCCHGeoRFExperiment/baseline_runtime.py:81-140 verifies ZIP SHA, CRC and
  every payload hash before/after extraction into an empty destination.
- :178-248 isolates imports, but feature_drop_off=True defaults to clearing
  FEATURE_DROP for IPCCH's prepared schema. FEWS must choose this explicitly.
- :143-174 applies a separate run-local polygon patch. Do not apply it to FEWS
  without checking need; it is not implicit in extraction.
- :262-277 provides cwd isolation for relative outputs.
- IPCCH target, sparse-label preprocessing, pooled single Stage 1 and omitted
  Stage 2 are dataset/protocol choices, not this task's requested pipeline.

## Persistence algorithms can be reused; old runners are tied to old runs

- PersistenceCorrectionExperiment/persistencecorrection/persistence.py:178,221,
  389,408 accepts explicit histories/prediction files and checks calendar joins.
  :70 supports only fs1=4 and fs2=8.
- persistencecorrection/calibration.py:414-489 groups dynamically by month and
  partition. Refit for new maps; identical partition integers do not imply
  compatible groups. :219-268,447-458 define fallback behavior.
- persistencecorrection/selection.py:26-69 selects true-F1 thresholds and can
  return tau=None; its freeze metadata hardcodes 2020 and 2021-2024.
- persistencecorrection/override.py:36-39 is structural up-only correction.
- run_calibration.py:50-62,80; run_selection.py:15-16; run_adjudication.py:19-22;
  persistencecorrection/protected.py:48-90 hardcode old inputs/output roots.
- run_adjudication.py:69,97 assumes numerical tau; the new orchestration must
  support a legitimate no-improvement tau=None result.
All bare paths in this section are inside PersistenceCorrectionExperiment/.

## Historical source and time overlap

Historical panel: Analysis/1.Source Data/FEWSNET_forecast_unadjusted_bm.csv.
Recorded SHA: 611f9e776380e28da3fc845888d66a117626f91b37868e236ce849d44bc8f651
(PersistenceCorrectionExperiment/outputs/phase1_20260918/protected_hashes.json:4,41).
This is a historical record, not a fresh hash verification of the external input.
Observed persistence uses Outcome/FEWSNET_IPC/FEWSNET.csv separately
(Step3ExpertCorrectionExperiment/step3correction/protected.py:26-27).

The main Stage 1 batch defaults to scopes 1/2/3, 2018-2020, twelve months
(run_batches_2018_2020_partition_learning_visual_monthly.bat:88,184,202,303-307).
Defaults do not approve this new task's schedule. The old probability manifest
at PersistenceCorrectionExperiment/outputs/phase2_probabilities/
probs_2018_2020_fs1/run_manifest.json:24 calls its maps in-sample for that period.
The generic no-leak sentence nearby does not establish chronological separation.
Archived persistence design.md:90-95 and DECISIONS_LOG.md C5/C6 acknowledge
overlap. Disjoint calibration/threshold windows alone do not remove it.

The old models and calibrators fit all eligible rows; persistence-zero filtering
happens at override application. Onset-only training was an unexecuted redesign.

## Feature inventory and scientific choices

GeoRFBaseline/src/feature/feature.py:55-64,81-100 dynamically identifies varying
fields and builds row-shift lags. Sorting is checked, but shifts denote calendar
months only on a retained complete monthly grid. Verify grid, origin alignment,
strict-lag routing and fixed feature order before adding temporal summaries.

Proposals, not approved candidates: IPC history/age/duration, month sin/cos,
lagged levels/changes/trends, rolling weather/vegetation/price/conflict anomalies,
and missingness/observation age. On 2026-09-20 the user approved NOAA_ENSO,
WB_RTP, Coastline_distance_NOAA and bloomberg_food_and_derivative, and removal
of legacy FAO price and WFP price/standard-deviation families before engineering.
Exact local source identities, columns and merge/availability rules are not yet
verified; the additional source-family boundary is no longer an open question.

GitNexus query had no matching indexed package flow. These references come
from direct source inspection; no indexed hit is not evidence of zero impact.
The Ethiopia local-forecasting spec's SMOTE, 85/88-column schema and strict
publication cutoff are not automatically this FEWS task's scientific contract.

## Approved two-layer fitting and correction contract — D37

The user adopted the historical full-sample crisis-probability model and
one-way correction, alongside the already-approved temporal/schema changes:

- Fit Stage 1/3 RF on all otherwise eligible labeled fitting rows under the
  approved split/window rules. Do not filter training rows to persistence=0
  or require available persistence solely to fit a crisis-probability RF.
  Stage 1's retained class-recovery convention and disabled SMOTE remain as
  approved. The modeled response is the original target-period crisis label.
- Fit each 2018 horizon/month/recipe calibrator on all eligible development
  predictions with valid target labels and RF probabilities, including both
  persistence states and rows with unavailable persistence. D16's pooled
  routing and D17's fit dates stay unchanged. Calibration estimates target
  crisis probability; it is not fitted to a persistence-error indicator.
- Apply the correction only on D36's available-persistence support. For b=0,
  set the corrected prediction to 1 exactly when p>tau. Otherwise keep b.
  For b=1, keep 1 regardless of p. Here p is the raw or calibrated RF crisis
  probability, with the variant's own frozen threshold. Equality does not flip.
- If threshold selection yields a valid no-correction outcome, return b
  unchanged. Missing persistence yields no corrected prediction, not a pooled
  or RF substitute. Raw/calibrated comparisons share predictions and keys.

The main session checked `PersistenceCorrectionExperiment/persistencecorrection/`
`override.py:19-39`: its condition is `(persistence == 0) & (p_cal > tau)`,
strictly greater than, and it has no downward-flip path. `calibration.py:424-470`
fits its supplied eligible target labels/probabilities by calendar month without
a persistence-zero filter. The new orchestration must apply the frozen 2018
window and pooled route, not the historical runner's old defaults.

This decision fixes fitting support and correction direction. D38 fixes threshold
selection and D39 fixes reporting; final scientific success criteria still
require decisions. It adds no onset-only learner,
persistence-error classifier, reverse correction or extra feature recipes.
This approval remains planning-only and does not authorize execution.

## Approved threshold-selection contract — D38

The user adopted reuse of `PersistenceCorrectionExperiment/persistencecorrection/selection.py`
`select_threshold` (:26-52), with the new experiment's approved support and
provenance supplied explicitly rather than copying historical runner defaults:

- Per feature recipe and horizon, select one raw and one calibrated threshold
  on the 2020 February/June/October common valid-persistence support under D36.
  Each threshold is shared across target calendar months and partitions. This
  gives six selections per recipe, distinct from month-specific calibration.
- Scan the sorted unique probabilities observed on that variant's entire
  eligible 2020 support. For each candidate tau, apply D37's strict p>tau
  up-only correction. Do not add a grid, quantiles, interpolation or an extra
  candidate below the smallest observed probability. This deliberately retains
  the existing search domain rather than claiming to search every real tau.
- Pool all three months' TP/FP/FN before calculating class-1 F1; use the existing
  zero-division convention of F1=0 when its denominator is zero. This is not
  an average of per-area or per-month F1 values. Missing required support is
  an incomplete comparison, not a zero-score outcome.
- Initialize the selected result to unchanged persistence. Accept a candidate
  only if its F1 is strictly higher than the current best. Ascending traversal
  therefore retains the smallest tau among candidates tied at the best F1,
  as in the existing selector. This can favor more flips among tied candidates.
- If no candidate strictly improves persistence F1, freeze an explicit
  no-correction outcome (tau=null). Its prediction remains persistence, and
  its development gain is zero in D22's recipe score. Do not remove that cell
  or pretend a numerical threshold was selected. The new orchestration must
  handle this valid outcome without passing null to the old numeric override.

Freeze the selected thresholds/no-correction outcomes with the full candidate
score traces and input identities by 2020-12. No final-period observations may
change them. Choosing no correction is a completed development outcome, not
an execution failure or authorization to search extra recipes. This decision
does not set the final scientific success threshold or uncertainty criterion.

## Approved final comparator and reporting inventory — D39

For each horizon and each of the two final feature arms (original-feature
corrected reference and frozen updated-feature winner), the user adopted these rows:

| Row | Prediction source |
|---|---|
| Persistence | D36's valid exact-O observed crisis label |
| Pooled RF | Released pooled RF's native hard prediction |
| Partitioned RF | Released partitioned RF's native hard prediction, with prescribed fallback provenance |
| Raw persistence correction | D37 override using partitioned RF raw crisis probability and its D38 threshold |
| Calibrated persistence correction | Same partitioned RF probability passed through the D16/D17 calibrator, then D37/D38 override |

Both RF rows use their native classifier prediction rule, without a separate
development-tuned classification threshold or extra standalone-calibration arm.
The probability source for the two correction rows is partitioned RF; pooling
the calibration observations across partitions does not mean fitting a pooled-
RF correction arm. Retain prescribed pooled fallback provenance where it enters
the partitioned route, including D19's valid single-partition outcome.

Use identical D36 paired-evaluation keys across these rows and the two feature
arms within each horizon. Persistence has one common stream and need appear
only once in a combined table; this gives nine unique method/feature streams
per horizon, not two different persistence baselines. Numerical duplicates
from a no-correction outcome or valid unsplit partition retain their method
labels and explanation; do not misrepresent them as independent evidence.

Additionally, report pooled/partitioned standalone RF metrics on their full
eligible labeled-target support in a separate table with sample counts. Reuse
the existing RF predictions; these supplementary metrics do not share a cohort
with unavailable persistence and cannot replace the paired comparison. This
does not add fits, candidates, pooled-RF correction, XGBoost or other model
families. It does not change D22's development feature-selection score.

D40 fixes final evaluation dates and D41 fixes primary/secondary scientific roles;
uncertainty and remaining success/stop conditions still require decisions. This inventory is approved
for planning and does not authorize execution.

## Uncertainty reuse inspection — code facts

A bounded read-only scout traced the historical uncertainty code; a second
independently checked the joint calendar mathematics. The main session checked
the actual bootstrap loop and confusion-count helper. No new predictions,
resamples of actual results or model fits were produced.

- `PersistenceCorrectionExperiment/run_adjudication.py:23,33,55-65,78-83`
  uses default_rng(5), 2,000 target-month block draws per scope, paired override
  and persistence rows, and F1 recomputed from the pooled sampled rows. Scopes
  use different consecutive portions of the RNG stream, not shared date draws.
  It reports 2.5/97.5% NumPy quantiles (default linear interpolation) and
  leave-one-target-month-out results, not leave-one-year-out.
- That file is a top-level runner with directory creation and input/output
  side effects (:20,25,92-94). Do not import it as a utility. Reuse the small
  algorithm and pure metric pieces through the new experiment's orchestration.
- `Step3ExpertCorrectionExperiment/step3correction/expert.py:261-280` provides
  binary confusion counts and F1 with zero denominator returning zero. Verify
  common keys, shape and nonempty support separately; the helper alone does
  not enforce them. Weighted confusion-count recomputation is equivalent to
  repeating all rows from sampled date blocks and avoids large row copies.

Calendar-only checks, assuming nonempty support on each scheduled date:
the 11-date union gives 11/10/9 date blocks for H=4/8/12. Shared multinomial
date weights preserve within-date cross-method, cross-variant, cross-horizon
and spatial co-movement. They do not preserve serial dependence across dates
or capture uncertainty from refitting/feature selection. Treating dates as
exchangeable is an explicit inference assumption, not established by the
calendar or by a positive leave-one-year-out diagnostic.

For 11 draws from the 11-date union, an entire H=12 sample is absent only if
all draws fall in the two 2021 dates: probability (2/11)^11, approximately
7.18e-9 under that calendar-support assumption. Actual row support may differ.
An alternative of four whole-year draws has only four year blocks and 35
distinct multiplicity vectors; the all-2021 draw has no H=12 support with
probability 1/256. More Monte Carlo repetitions do not increase that underlying
year diversity. This is a limitation comparison, not approval of a protocol.
