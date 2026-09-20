# IPCCH binary crisis GeoRF pipeline — specification v1.0

Status: **implemented, completed and archived**; closure/review evidence is in STATUS.md
and IPCCHGeoRFExperiment/validation/review.md. The preregistered requirements below
are retained, with the explicit Q8g amendment recorded during implementation.
This is the authoritative requirement/acceptance document. Technical design is
in `design.md`; the historical execution plan is in `implement.md`.

## Goal and scope

Measure partitioned RF forecasting of IPCCH population-share crisis against
exactly three baselines: pooled RF, pooled binary XGBoost and persistence.
Learn one shared partition from pooled historical data, skip Stage2 consensus,
complete geographic assignment within the approved distance cap, then evaluate
monthly forecasts. A reproducible comparison, including null/negative results,
is success; no winning-model requirement applies.

Preserve the released GeoRF class1-F1/q mathematics, parent/child checkpoint
selection, spatial scan and polygon refinement. At every depth the parent wins
ties and a split requires strict aggregate validation F1 gain >.01; this is not
a statistical significance test. No SMOTE. Retain the approved Stage1 per-class
zero-feature pseudo rows; Stage3 fits original rows only. Narrow IPCCH boundary
adaptations must be documented; the frozen release and prior experiments stay intact.

## Pinned inputs and verified background

- Selected source:
  `C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\assembled_IPCCH\raw\IPCCH_2026_completed.csv`.
  SHA256: `ae696087c3bbb280537ae269a05924133acdb51060d31290523404fa8a717673`.
  The initially named FEWSNET combined CSV is superseded.
- Code baseline: `GeoRFBaseline/releases/georf-baseline-v0.1.0.zip`.
  SHA256: `39a26138e3fafb0be2bbd22e9760095d6cdefa7d79b98b4f798cb3aa79b500a0`.
- Source:1,219,868rows,143columns,6,227areas,53countries; unique
  `(admin_code,year,month)`. Near-complete monthly predictor scaffold but sparse
  labels, median six complete phase distributions per area.
- 43,551rows originally have all five phases;752all-zero distributions,16raw P3+
  sums>1, one phase component>1 and363complete rows with population0. These counts
  can overlap. Missing P5=0 is an approved convention, not established source semantics.
- Under R1:42,695valid labels (15,206positive,27,489negative) in6,224areas/53countries.
  42,611originally complete rows pass;940fail.84only-P5-missing rows are recovered
  and one fails. Normalization changes82labels on the valid cohort (81to1,1to0);
  2,601normalized shares equal.20. Areas1315,1354,3358 have no valid target but
  remain in the scaffold/geographic universe.
- IPC IDs<100000 first have valid labels2017-01; CH IDs>100000 first2014-01.
  Counts24,336IPC +18,349CH +10at boundary ID100000 =42,695. ID100000 is one area,
  first valid2020-01, separately audited without changing model eligibility.
- Approved2014–2022pool capacity before feature construction:19,591original labels;
  18,119in3,264areas with n>=2;1,472singleton areas;1,491zero-label areas.
  R4's split yields8,561fit/9,558validation original outcomes. These are capacities,
  not fitted partitions, stable-partition evidence or successful donor counts.
- Geometry/country/reference keys cover all6,227IDs.253polygons are invalid;
  Q8g below governs repair. Key/topology checks do not establish correct geographic
  identity: the upstream builder allowed unlimited nearest fallback without saved
  per-area match provenance. Reference coordinates are separate from polygon centroids.

Evidence: `research/source-audit.md`, `research/approved_label_policy_audit.json`,
`research/validation_support_by_source.json`, `research/stage1-support.md`,
`research/pooled_split_support_2014_2022.json` and `research/convergence-check.md`.
Decision history, model-setting/runtime evidence and threshold traces:
`research/brainstorm.md`.
Earlier candidate QC/support scenarios remain historical evidence, not policy.

## Requirements

### R1 — Target and provenance (Q1)

Apply in order:

1. Require observed P1–P4. Only missing P5 may become0; flag the fill.
2. Every observed phase share must be in inclusive[0,1]; estimated_population
   must be observed and strictly positive.
3. Filled five-phase total S must be in inclusive[.90,1.10]. Do not clip values
   or rescue other totals. Raw P3+>1 alone is not another exclusion.
4. Normalize all five shares proportionally by S.
5. `ipcch_food_crisis = int(normalized_P3plus > .20)`; exact equality is negative.

Use source-decimal precision at boundaries; equivalent exact comparison is
`int(5*(P3+P4+filled_P5)>S)`. Invalid truth stays missing; never replace it with
overall_phase, persistence or a probability threshold. Preserve raw components,
P5-fill flag, S, normalized components, share, validity reason and label.
Keep unlabeled scaffold rows until calendar features/history have been constructed.

### R2 — Data identity, geography and repair (Q8g)

Canonical area-month identity comes from admin_code/year/month, with a reversible
boundary mapping to area_id and any legacy FEWSNET_admin_code interface. Keep
row identity through sorting/filtering; replace the FEWSNET0..5717 universe with
explicit IPCCH IDs. Supervised training/scoring requires valid observed truth
for that specific area-month. Unlabeled rows may supply historical covariates,
never additional supervised targets. Stage1 has no single target-month area list.

Validate geographic keys, geometry and mapping before adjacency/learning.
**Q8g approved:** apply make_valid only to invalid geometries in an experiment-local
copy; preserve raw source, IDs, reference coordinates and already-valid geometries.
Record original validity reasons and before/after geometry/type/footprint/adjacency
evidence; if invalid originals prevent a comparison, report it unavailable.
Stop on remaining invalid/empty/non-polygon output or ambiguous area identity.
No silent component extraction, area exclusion, boundary-source replacement or
disabling spatial refinement. Topology repair does not prove administrative identity.
Preserve areas with missing ISO3 using the keyed country lookup for reporting.

### R3 — Horizon, availability and features (Q3a/Q3c/Q3d, Q6a–Q6f)

Use actual horizons **1,3,6,12 calendar months**, not inherited FEWSNET/sibling scope
meanings. IPCCH requires no expert alignment. For target T and horizon H, origin
O=T-H is month-end inclusive: source/history/training-label month<=O.
Every historical training row uses its own T-H predictors, not the later fit origin.
Example: h6 target2024-01 has O=2023-07.

Stage3 training targets occupy exactly36calendar months **[O-35,O]**, not36nonmissing
observations. Example: O=2023-07 uses2020-08..2023-07, correcting the inherited
nominal36/actual35month boundary. Stage1 uses R4's pooled period instead.

The following fixed information set is shared by all three learned arms:

- **Q6b:** the exact70raw fields/order in `research/secondary-predictors.md`:
  conflict19, prices/macro9, vegetation/weather/nightlights6, land/soil/access13,
  all21supplied AEZ fields and2coordinates. Use source names/values; area/country/
  text identifiers are metadata. No population-density substitution, automatic
  extra climate blocks, absent-field reconstruction or column auto-discovery.
  Freeze this set and the selected source; WB RTP and other upgrades are a later version.
- **Q6d:** retrieve each raw field at that area's O. Add only15derivatives:
  WFP_Price sums O-3..O/O-11..O; nightlight_mean sum O-11..O; EVI_mean at O-k,
  k=1..12. Per-area calendar windows, all bounds inclusive. Sums require every
  month's value; missing inputs remain NaN. No implicit fill, sparse-row shift,
  cross-area rolling, automatic lag block or target-month covariates.
- **Q6e (user revision):** target month m gets sin(2*pi*(m-1)/12) and
  cos(2*pi*(m-1)/12). No month/year dummies, numeric year or extra origin-month
  input. Dates remain metadata. Keep R4's explicit horizon indicator.
- **Q6a:** latest valid same-area binary target<=O, its calendar-month age and
  missing-history indicator. No history gives NaN label/age before preprocessing.
  Replace legacy binary/ordinal history lag families; exclude historical
  overall_phase and phase shares. Preserve source month as metadata.
- **Q6f:** latest valid same-area positive month C<=O gives months O-C; C=O gives0.
  Search full preceding history, not just36months. If absent: NaN plus
  no_prior_observed_crisis indicator; retain C as metadata. This supplements
  latest-observation age and measures observed-positive recency, not episode
  onset/end or proof of crisis-free time.
- **Q6c:** RF uses baseline max_plus100, fit on genuine fitting rows only:
  finite maximum M!=0 gives100*M; M=0 gives100; all-missing gives0, retaining
  the column. Observed values stay unchanged; negative M need not be out-of-range.
  Stage1 fits one shared imputer on internal fitting views, excluding validation,
  singletons and pseudo rows; append approved pseudo rows only afterward.
  Stage3 fits one shared RF imputer per origin/horizon on the common training pool.
  Held-out rows transform only; root/branches/pooled RF share that transform.
  XGB uses native NaN. Infinities become NaN with an audit. Retain cohorts/schema,
  raw missingness and semantic history flags; no generic missing-indicator block,
  complete-case deletion, scaling or implicit fill.

This is retrospective observation-month availability, not verified publication
timing. Upstream nightlight sum/mean naming and ungrouped EVI/nightlight fills were
found in a candidate2025producer; its link to the selected2026CSV is unverified.
Do not claim source corruption/cleanliness or silently reconstruct/substitute/drop
fields. Exact whitelist, derivative names and evidence live in secondary-predictors.md.

### R4 — One pooled partition, no Stage2 (Q2/Q2r/Q3b, Q5, Q8/Q8r)

**Q2r/Q5b:** pool valid target months2014-01..2022-12 inclusive for ONE GeoRF
partition-learning procedure. This still fits root and candidate child RFs
recursively. No monthly/annual candidate ensemble, spectral k40 or Stage2 consensus.
The2014start supersedes2018; no invented pre2017IPC labels. Expanded INTERNAL
validation supports q/split decisions; it does not expand Stage3 final-test dates.

**Q5h/Q5a:** split original area-month outcomes first. Within every n>=2area,
sort target months; earliest floor(n/2) fit, latest ceil(n/2) validate. Odd counts
put the extra outcome in validation. Expand each outcome to the four horizon views
with explicit horizon indicator; all four stay on the same side, with their own
origins. One shared map, no horizon-specific fit or tuned horizon weights.
Count support/coverage before expansion; report model rows separately. Preserve
baseline q/F1 mathematics and effective support gates, not four independent labels.

The half split supersedes random20percent validation and trades fit size for
validation support. Cutoffs differ by area: some pooled fitting labels can postdate
another area's validation origin. This is internal development validation, not a
globally forward forecast test. Own-row feature cutoffs remain mandatory.

**Q5s/Q5m/Q5v:** a Stage1 singleton is excluded from fitting, q/group statistics
and parent/child F1 gates. After learning and completion, score its four views as
supplementary development diagnostics, with no partition retuning or validation
support inflation. It inherits its eligible nearest donor's partition and uses
that partition RF on its OWN as-of features. No extra area-only RF, copied donor
features/prediction or training on its held-out truth. Preserve donor ID, distance,
partition and model provenance. This leaves Stage3 training eligibility unchanged.

Export hierarchical branch strings directly, retaining root/ancestor lineage.
A default-root placeholder is not learned membership; reconcile final assignments,
actual participation and checkpoint/model provenance before geographic completion.
Code evidence: `GeoRFBaseline/scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py:231-267`.

**Q5d:** donors require >=1original fitting and >=1original validation outcome,
valid coordinates and one unique explicit final learned assignment. Genuine
retained root/ancestor members qualify; inferred/default-root cases do not.
Resolve conflicting assignments; do not choose the first record. No extra
per-area class/F1/minimum-count requirement is approved.

**Q8/Q8r:** reuse geometry/adjacency and the small Stage2 1-NN pattern for unassigned
areas only, using the entire6,227ID universe. Preserve learned labels. Donor
selection AND cutoff use reference-coordinate great-circle distance, **<=100km
inclusive**. No chaining, country restriction, adjacency-only restriction or cap
relaxation. Above the cap, remain unassigned and predict with pooled RF.
Inferred membership is not observed/validated membership. Validate >=1learned seed
and export learned/nearest-donor/unresolved provenance. Reuse anchors:
`GeoRFBaseline/src/adjacency/adjacency_utils.py:18-38`;
`GeoRFBaseline/scripts/step6_complete_clustering_pipeline.py:192-202`.
Do not invoke the full spectral CLI.

Partition availability includes all fitting AND internal-validation labels.
Freeze the2022-12 information cutoff. Every Stage3 origin is at least2023-01;
no final-test outcome selects or adjusts this map. Main target schedule:

| H months | First target | Last main target |
|---|---|---|
| 1 | 2023-02 | 2025-12 |
| 3 | 2023-04 | 2025-12 |
| 6 | 2023-07 | 2025-12 |
| 12 | 2024-01 | 2025-12 |

Apply to all arms; partial2026 is separate. Internal validation, supplementary
singletons, inferred geographic coverage and final-test coverage are separate
reports. Assignment creates no new observed labels.

### R5 — Four-arm forecasting and cohorts (Q3e, Q4/Q4b, Q7a/Q7b)

Each learned model refits from scratch monthly per horizon using R3's window.
Earlier test outcomes may enter later training once available; no current-target/
post-origin labels, test-score tuning or relearning partitions.
All learned arms share eligible training keys and predictor information; inherited
pooled/local group filters must not silently change support.

- Partitioned and pooled RF: retain100trees, unlimited tree depth, seed5;
  Stage3 n_jobs1. Local partition <50training rows or single class falls back
  to the same pooled RF; unassigned/unseen partitions do likewise.
- **Q7b:** XGBoost3.0.0, binary:logistic, gbtree/hist/CPU,400trees, depth6,
  min_child_weight5, learning_rate.05, subsample.8, colsample_bytree.8,
  reg_alpha0/reg_lambda1/gamma0/max_delta_step0, scale_pos_weight1/base_score.5,
  seed5/n_jobs1, max_bin256/grow_policy depthwise/num_parallel_tree1,
  eval_metric logloss/missingNaN. Unit row weights; no search, early stopping,
  pseudo rows, SMOTE or calibration.
- **Q7a:** all learned Stage3 arms predict1 iff p1>.5; ties0. No threshold selection
  or extra threshold arms. Retain Stage1's baseline RF decisions. Fixed thresholds
  do not promise F1 optimality; this is a predeclared, not optimized, XGB benchmark.
- **Q4:** persistence is latest valid same-area R1 label<=O, with no maximum age
  or36-month history truncation. Missing history means missing prediction; export
  source month and age. It refreshes lookup without fitting.

**Q4b:** E_all is every valid test key in the approved schedule; compare the three
learned arms there. E_persist is its history-available subset (both0/1); compare
all four on IDENTICAL keys. Reuse predictions, never retrain/filter training for
the subset. Keep missing persistence outside it; no imputation or silent exclusion
of truth. Missing learned predictions are failures. Never treat subset persistence
F1 versus full-sample learned F1 as paired evidence.

### R6 — Reporting, uncertainty and reproducibility (Q9a/Q9b)

Use a new versioned run under `IPCCHGeoRFExperiment/`. Save source/code hashes,
configs/runtime, target/feature/split/geography audits, partitions, row-level
truth/hard predictions/probabilities where available, persistence dates,
cohort keys, paired metrics and final status. Preserve prior inputs/results.

**Q9a:** separately per horizon/cohort, aggregate observation-level TP/FP/FN/TN
before F1=2TP/(2TP+FP+FN), precision=TP/(TP+FP), recall=TP/(TP+FN).
Zero denominator -> NaN+reason; zero numerator with positive denominator ->0.
No population weighting, country/month mean headline F1 or pooled-horizon headline.
Report target-month/year, country and assignment-provenance breakdowns separately,
on matching keys; no automatic grouping cross-product. Include observation/area/
country support, prevalence, predicted positives, persistence coverage and fallback
counts. Paired deltas are partitioned RF minus each eligible baseline. Undefined
metrics/deltas and empty cohorts remain explicit.

**Q9b:** main Stage3 horizon/cohort summaries only:1000paired country-cluster
bootstrap draws, seed42,95% percentile F1 and F1-difference intervals.
Sort stable country IDs; each draw samples K countries with replacement and keeps
all rows per copy, preserving multiplicity. All models share the draw. Export draw
identities/statistics, valid/undefined counts and reasons. No redraw-until-valid
or NaN-to0. Use defined replicates for2.5/97.5quantiles with explicit labeling if
some are undefined. No CI if the point is undefined, K<2 or <2defined replicates;
these are nondegeneracy checks, not adequacy guarantees. No refitting, subgroup/
partial2026/Stage1 CIs, p-values/stars, simultaneous coverage or multiplicity claims.
Intervals describe country-composition uncertainty conditional on saved predictions,
not future prediction, partition/training uncertainty or cross-country common shocks.
Report all contrasts; no selected favorable result as general-superiority evidence.
Exact metric/bootstrap contracts and helper limitations: `research/evaluation.md`.

Never interpret raw F1 differences between FEWSNET and IPCCH as improvement on the
same task. Observation-month timing and unverified source/geographic provenance
remain limitations in final outputs.

## Amendment — Q8g scope extension, 2026-09-20

R2 above reads "no silent component extraction" and design.md says "do not extract
polygon pieces out of GeometryCollections", both with a mandatory stop. **That stop
triggered during implementation**: `make_valid` on the 253 invalid geometries returns 0
invalid and 0 empty but 217 GeometryCollections. The user was asked the one scope
decision implement.md Phase 2 provides for and narrowly authorized extraction:

> When a GeometryCollection consists of exactly one areal component plus only zero-area
> linear/point components, keep the areal component and discard the zero-area parts.
> Any other composition still stops the run.

R2's prohibition otherwise stands: no boundary replacement, no area deletion, no repair
escalation, no "largest polygon" rule. Evidence, the implemented guard and the resulting
limitations are in `DECISIONS_LOG.md` D1 and D5-D7 and in
`research/geometry-repair-trial.md`. R2's own text is left unedited so the original
requirement and its amendment both stay visible.

## Acceptance criteria

| ID | Observable acceptance |
|---|---|
| A1 | Source/release hashes match; originals unchanged; runtime, local changes and all output lineage recorded. |
| A2 | R1 decimal/fill/bounds/sum/normalization order and exact .20/.90/1.10 boundaries pass; counts, exclusions and flags reconcile. |
| A3 | Key alignment and all own-origin cutoffs pass; Stage3 training is exactly[O-35,O], no current/post-origin truth; partition information precedes test origins. |
| A4 | One pooled Stage1, no Stage2; core F1/q and branch lineage preserved. R4 original-outcome half split/four-view membership/support pass. Singletons only supplementary. Q8g repair and stop rules honored. Learned/eligible-donor/inferred/unresolved maps reconcile; donor distance<=100km, no chaining; correct recipient features/model routing and fallback. |
| A5 | E_persist equals the history-available subset of E_all; appropriate arms share identical keys. Cohort masks never alter training; missing persistence and fallback counts explicit. |
| A6 | Saved rows independently reproduce metrics/deltas/CIs; actual windows, horizons, seeds and parameters match R3–R6; p1>.5 ties0; undefined cases and paired support retained; no outcome-driven retuning. |
| A7 | All specified comparisons reported, including null/negative results; no winning-model requirement or stronger inferential claim than Q9b. |
| A8 | Exact70raw/15derived/calendar/horizon/history/crisis-recency features match R3; no added RTP/year/target leakage. Shared training-only RF fills exclude held-out/pseudo rows; XGB native NaN; rows/schema retained. |

Decision IDs Q1, Q2/Q2r, Q3a/Q3b/Q3c/Q3d/Q3e, Q4/Q4b,
Q5a/Q5b/Q5h/Q5s/Q5m/Q5v/Q5d, Q6a/Q6b/Q6c/Q6d/Q6e/Q6f,
Q7a/Q7b, Q8/Q8r/Q8g and Q9a/Q9b are resolved in the owning requirements.
Q5/Q6/Q9 family questions are resolved by their named subdecisions.
No user-owned scientific question remains open at this revision.

## Exclusions and remaining limits

No upstream target reconstruction, new acquisition/features, expert correction,
extra classifier arms, FEWSNET partitions, monthly/annual candidate ensembles,
Stage2/spectral consensus, new spatial-assignment framework, tuning framework,
remote publication or revision of prior scientific results. Q8g is the sole
approved local geometry-repair exception; incompatible outcomes stop for review.
Do not change scientific behavior to bypass a failed gate.

Source lineage/publication timing and true geographic identity are not proven.
Actual repaired topology, learned donor coverage, partition stability, runtime and
forecast performance remain unmeasured. Design/plan must make these checks and
stop conditions concrete. Final-summary approval is still required before
`task.py start`, implementation or experiment execution.
