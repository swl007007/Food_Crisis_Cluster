# Chronological planning record through D23

This preserves prior draft wording and evidence. Statements that choices remain open
inside earlier decisions describe their historical state; the converged PRD is current.

# FEWS NET four-class forecasting baseline with a three-stage GeoRF pipeline

Status: planning draft, 2026-09-28. Task creation approved; scientific choices below
remain proposals until individually resolved. No implementation or experiment authorized.

## Goal

Build a reliable four-class forecasting baseline from the minimal Stage 1–3
replication package, comparing partitioned RF with expert forecasts, persistence
and pooled RF. Include causal historical-outcome engineering and interactions.
The user explicitly superseded the strict single-perturbation framing in D21;
matching historical binary results is not the purpose. Follow the requested
order: brainstorm, persist a draft spec, then grill to closure.

## Confirmed scope

- Preserve three-stage partition learning, consensus and final RF evaluation.
- fs1/fs2 evaluation arms: partitioned RF, pooled RF, expert forecast, persistence;
  fs3 compares partitioned RF, pooled RF and persistence only (D4).
- Isolate the four-class baseline in a new experiment/package; preserve the original
  baseline release, inputs and historical outputs.
- Historical-outcome engineering and selected interactions are in scope (D21).
  No XGBoost or correction layer is requested; exact feature inventory remains
  to be frozen before training, not selected using final evaluation outcomes.
- Creation only establishes a planning task; it does not start the audit execution run.

## Approved decisions

- D1 (2026-09-28): Merge original phases 4 and 5 into one class, displayed as
  “4或5”. The task has exactly four classes: 1, 2, 3, “4或5”, not five classes.
  Apply this mapping consistently to observed truth, expert forecasts and
  persistence. Retain raw phase values for provenance; no phase-5 exclusion.
  A numeric code 4 represents the merged class; internal 0..3 encoding is only
  an implementation representation. Missing values remain missing.
- D2 (2026-09-28): Use macro F1, equally averaging the four class F1s, as the
  common primary metric for Stage 1 split acceptance, Stage 2 performance
  weighting and Stage 3 comparison. Also report per-class F1/support, confusion
  matrices, accuracy and ordinal error. The candidate scan construction,
  absent-class convention and exact weighting/gate rules remain open; D2
  does not assert that the binary scan decomposition extends to macro F1.
- D3 (2026-09-28): Calendar-align expert forecasts by area and source month:
  fs1 uses fews_proj_near at T-4 months, fs2 uses fews_proj_med at T-8 months.
  Do not use record-count shifts. Missing source rows or phase values remain
  missing, never class 0/noncrisis. Apply D1 to valid raw forecast phases.
  No fs3 expert proxy is approved; D4 defines its evaluation scope.
- D4 (2026-09-28): Retain fs1/fs2/fs3 (4/8/12 months) in Stage 1 partition
  learning and Stage 2 consensus to preserve the candidate partition pool.
  Stage 3 compares all four arms at fs1/fs2, and only partitioned RF, pooled
  RF and persistence at fs3. Expert is unavailable at fs3; do not substitute
  a medium projection as a 12-month forecast. Claims against expert cover
  only the 4- and 8-month horizons.
- D5 (2026-09-28): Persistence for target T uses the same area's observed
  phase at the exact origin T-H, H=4/8/12 months for fs1/fs2/fs3. Apply D1
  and carry the resulting class forward unchanged. Missing source month or
  observed phase means unavailable persistence; no earlier-observation backfill.
  This assumes the origin-month observation is available at origin; source-month
  alignment alone does not establish actual publication-time availability.
- D6 (2026-09-28): Macro F1 always averages the fixed four classes equally.
  For class k, F1_k=2*TP_k/(2*TP_k+FP_k+FN_k); if the denominator is zero,
  assign F1_k=0 rather than dropping the class. Thus perfect predictions on a
  nonempty three-class cohort score .75. Compare parent and candidate split
  predictions on the same complete validation rows of the current parent branch,
  aggregating confusion counts before scoring, not averaging child macro F1s.
  This scoring convention does not authorize accepting an empty validation cohort.
- D7 (2026-09-28): Retain the baseline performance gate at every depth:
  accept a candidate only when its macro F1 exceeds the parent's by strictly
  more than .01 on D6's common validation rows. Parent wins ties; do not tune
  or search the gate. This is a performance rule, not a significance test.
  Different metric scaling may yield few splits; a null partition alone is
  not evidence that spatial heterogeneity is absent.

- D8 (2026-09-28): Approved candidate scan: retain one joint spatial scan with
  four one-vs-rest statistic columns. On the current parent's validation rows,
  define D_gk=2TP_gk+FP_gk+FN_gk and D_k=sum_g D_gk. Pass exposures
  Y_gk=D_gk/(4D_k) and correct mass A_gk=2TP_gk/(4D_k), with zero columns
  when D_k=0. Existing scan then uses C=Y-A, B_gk=sum_g(C_gk)*Y_gk/sum_g(Y_gk),
  and sums class-specific scan scores into one proposed spatial partition.
  This is a parent-normalized candidate-search surrogate, not exact subset
  macro F1 or a calibrated significance test. D6/D7 alone decide acceptance
  after real child predictions. Zero total error mass yields no split candidate.
  Evidence: GeoRFBaseline/src/partition/partition_opt.py:220-225,960-1005;
  the existing scanner already accepts multiple statistic columns. Do not use
  the generic classification statistics at :148-204, which omit false positives,
  or groupby_sum's integer cast (:91-126) for these normalized masses.

- D9 (2026-09-28): Retain the Stage 2 positive logit-gain rule, replacing
  binary class-1 F1 with D2/D6 macro F1:
  w=max(0, logit(clip(F_partitioned))-logit(clip(F_pooled))).
  Scores must use identical Stage 1 scoring keys and truth for the two arms.
  Clip each score to [1e-6,1-1e-6], as in the minimal package. Candidates with
  no positive gain have zero weight. This is a consensus weighting heuristic,
  not a significance test. Handling an all-zero candidate set remains to be
  specified without inventing positive weights.

- D10 (2026-09-28): Stage 1 retains the 2018-2020 learning range across
  fs1/fs2/fs3; Stage 2 pools only those candidates. Partition learning and
  selection may use no information after 2020-12. Stage 3 requires origin
  O=T-H strictly after that cutoff. Schedule target months fs1 2021-05..2024-12,
  fs2 2021-09..2024-12, fs3 2022-01..2024-12. With the inspected Feb/Jun/Oct
  observed-label calendar, first supported targets are respectively 2021-06,
  2021-10 and 2022-02; verify from pinned data rather than fabricate support.
  Record target months with no eligible labels as empty; no interpolation.
  These are horizon-specific cohorts, not identical cross-horizon target periods.
- D11 (2026-09-28): Preserve the inherited rolling training-label window
  O-35 months <= label_month < O, where O=T-H. It contains 35 calendar
  months and excludes the origin month. Retain the original configuration
  semantics for replication but describe the actual inclusion rule explicitly;
  do not silently expand to 36 months or include the origin's label in fitting.
  D5 may still use the observation at O as the persistence prediction under
  its stated availability assumption; that does not change model fitting keys.
- D12 (2026-09-28): Retain Stage 1's inherited class-recovery mechanism:
  append exactly one all-zero-feature artificial fitting row for each of the
  four encoded classes to each applicable Stage 1 RF fit. These four rows
  are fitting-only, never validation observations, scoring support, spatial
  membership or source data. Disclose their possible influence on small
  branches; this is not SMOTE. Stage 3 fits on real rows only, with no
  artificial class-recovery rows. SMOTE remains disabled throughout.
- D13 (2026-09-28): Main fs1/fs2 comparisons use identical keys with valid
  truth, calendar-aligned expert and exact-origin persistence for all four
  arms. Main fs3 comparisons use identical valid truth/persistence keys for
  the three available arms. Supplementary fs1/fs2 comparisons drop only
  the expert-availability requirement and compare both RF arms with persistence.
  Report cohort counts, exclusions and coverage. Expert/persistence missingness
  affects comparison eligibility, not RF fitting membership; do not restrict
  training to the evaluation baseline-available cohort. RF prediction failures
  are not permission to silently shrink a comparison's common keys.
- D14 (2026-09-28): Preserve the baseline within-area random train/validation
  split, including its ratio, seed and small-sample handling. Partition search
  reuses that validation membership; all learning information remains bounded
  by D10. This is not out-of-time validation, and repeated partition selection
  may overfit the validation set. Stage 1 gains are not evidence of forecast
  superiority; assess that only in the separate Stage 3 evaluation period.
- D15 (2026-09-28): When the complete, successfully computed Stage 2 candidate
  inventory has all weights equal to zero, record no effective consensus and
  skip forced clustering. In Stage 3 the partitioned arm routes every row to
  the same pooled RF and reuses its predictions exactly; still evaluate the
  available expert and persistence baselines. This is a valid negative spatial
  result, not a learned spatial partition advantage. Missing candidates,
  failed computations or invalid scores are errors/incomplete evidence, never
  reasons to invoke the all-zero fallback or manufacture positive weights.
- D16 (2026-09-28): Report per-horizon macro F1 and paired delta macro F1
  for partitioned RF versus each available baseline, with 95% intervals from
  2,000 country-cluster bootstrap draws. Reuse country multiplicities across
  models within each draw, preserving each country's area/month observations.
  These intervals condition on already-trained predictions, with no model
  refitting; they do not establish future-year generalization. Expert contrasts
  cover fs1/fs2 only. Point estimates alone do not establish a robust advantage.
- D17 (2026-09-28): Retain Stage 3's local fitting gate: fewer than 50 real
  training rows or fewer than two observed classes routes the partition to
  the same fold's pooled RF. Otherwise fit the local RF even with only two
  or three classes; do not require all four. Align predict_proba columns to
  the fixed four-class axis using fitted classes_, filling absent-class
  probabilities with zero. Local models cannot predict a class absent from
  their real training data. Do not add Stage 3 artificial training rows.
- D18 (2026-09-28): Correct the inherited full-panel imputation fit. Retain
  the baseline fill formula, but fit each estimator's imputer only on its
  own real training rows. Apply that fitted transform to validation/test;
  append the four Stage 1 artificial zero-feature rows only afterwards.
  Save/reload the matching imputer with the estimator, including parent
  checkpoint inheritance and pooled fallback. No validation, test, future
  rows or class-recovery rows contribute imputation statistics. This is an
  explicitly approved preprocessing change beyond the target perturbation.

- D19 (2026-09-28): Preserve predictor sources but make temporal roles explicit
  in a frozen static/dynamic/calendar feature manifest, rather than inferring
  time variation from the complete dataset. Dynamic features use source months
  at or before the row's own origin O=T-H. Construct calendar-keyed lags on
  the complete monthly panel before filtering eligible target labels. Missing
  source observations stay missing until D18's training-only feature imputation;
  never interpolate or forward-fill missing outcomes. Known-in-advance target
  calendar features may remain. Retain dated feature provenance. Together with
  D18 this is a preprocessing correction beyond changing the target, so changes
  versus historical binary results cannot be attributed solely to multiclass.

- D20 (2026-09-28): Approve the eight basic history predictors: observed phase
  and binary phase>=3 at exact months O, O-4, O-8, O-12. Apply D1 to phase
  values. Missing exact-month observations remain missing, not replaced with
  earlier records. Derived feature imputation follows D18; raw validity masks
  and persistence baseline availability remain unchanged.
- D21 (2026-09-28, scope supersession): The final objective is a four-class
  baseline model, not a strict one-variable perturbation relative to the old
  binary baseline. Necessary historical-outcome lag interactions and feature
  engineering beyond D20 are authorized in scope. The exact expansion remains
  a design proposal until frozen. Earlier label-only/minimal-feature rationale
  is superseded; D1-D20's explicitly adopted choices remain binding unless
  separately revised. This does not authorize training or implementation.

- D22 (2026-09-28): Approve four history-engineering blocks in addition to D20:
  exact-lag phase changes; 12/24/36-month observed class-frequency/extrema/
  upgrade-downgrade/support summaries; ages since observed severe states or
  state changes and current observed-run count/span; limited origin-phase
  indicator interactions with recent transition/severity-frequency summaries.
  Frequencies are fractions of observed assessments, never population shares.
  Do not interpolate states, infer continuity between observations, or generate
  all pairwise products. Both RF arms use one frozen feature recipe; no
  feature selection using final evaluation outcomes. Exact columns and edge
  conventions below remain technical proposals for final review.

- D23 (2026-09-28): Exclude FEWSNET_admin_code from both RF predictor matrices.
  Retain it only for joins, geographic lookup, grouping and partition routing.
  Retain latitude/longitude and actual geographic covariates. Do not replace
  the identifier with an encoded identifier or a proxy country-ID predictor.

## Repository evidence

- Starting package: GeoRFBaseline v0.1.0 and its versioned release ZIP.
  GeoRFBaseline/README.md:11-47 defines binary class-1 F1, within-area random
  validation, strict >.01 split gain, disabled SMOTE and retained Stage 1
  synthetic class-recovery observations. No such synthetic rows in Stage 3.
- Stage 1 rejects nonbinary labels explicitly:
  GeoRFBaseline/src/partition/partition_opt.py:134-146. Its exact binary F1
  scan decomposition cannot simply be renamed macro F1; multiclass candidate
  proposal statistics and final split acceptance must both be specified.
- Stage 2 weights use f1(1)/f1_base(1), positive logit gain:
  GeoRFBaseline/scripts/step4_similarity_matrix.py:50-64. Consensus geometry is
  partition-ID based and need not become a different clustering architecture.
- Stage 3 currently supports partitioned/pooled RF only, not expert/persistence:
  GeoRFBaseline/README.md:115-123. Binary metrics and probability extraction:
  scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py:275-406 within that package.
- Original target source is fews_ipc; binary fews_ipc_crisis matches phase>=3
  on all jointly observed rows in the inspected original monthly panel.
  Source selection: GeoRFBaseline/app/main_model_GF.py:95-109; target exclusion
  and raw-phase removal: src/preprocess/preprocess.py:169-197,279-280.
- Initial read-only source inventory found 74 historical phase-5 observations
  in the original panel/release table; none among original release-table
  2021–2024 observed outcomes, but five phase-5 expert projections occur in
  the inspected fs2 calendar-aligned support. These are source counts, not
  frozen model evaluation counts; hash and final-key checks remain required.
- Expert semantics resolved by D3: archived record shifts differ
  from calendar-month joins. Step3ExpertCorrectionExperiment/step3correction/
  expert.py:6-30 documents this difference; :142-215 implements near(T-4)
  and medium(T-8). No native fs3 forecast is established. Do not import the
  disputed ETH selective-correction metric or fill missing expert phases with 0.
- Inherited split uses train_start<=date<train_end with start=end-(36-1):
  GeoRFBaseline/src/customize/customize.py:407-445. Actual 35-month inclusion
  must be disclosed and retained or explicitly changed, not silently called 36.

## Proposed baseline design — exact defaults not yet approved

- Follow approved D1 for the four-class target and display labels; preserve
  the mapping when encoding internally and exporting predictions.
- Retain original covariate sources and three-stage spatial/consensus structure;
  add the bounded history feature inventory authorized in scope by D21.
  Both RF arms must receive the same predictor definitions and column order.
  Relearn Stage 1 and Stage 2 under the multiclass objective; do not reuse binary
  partitions as if they were learned for the new target.
- Follow approved D2/D6 for the primary metric and absent-class convention.
  Follow D8 for candidate scanning and D9 for Stage 2 weighting.
- Historical expansion scope approved in D22 (exact inventory pending): retain D20 and add
  (a) observed direction/magnitude of phase changes across exact lag endpoints;
  (b) 12/24/36-month summaries of genuinely observed phase-class frequencies,
  extrema, upward/downward transition rates and observation support;
  (c) ages since latest observed phase>=3, phase4or5 and observed phase change,
  plus count/span of the current equal-phase observed run;
  (d) origin-phase one-hot interactions with a small declared subset of recent
  transition/severity-frequency summaries. Phase frequencies mean fractions of
  observed assessment records, not population shares. Do not infer uninterrupted
  states between observations, interpolate labels or generate all feature products.
  Ordinal step size is a category distance, not an equal-interval welfare measure.
  Freeze one shared feature recipe for both RF arms before final evaluation.
- RF hard predictions use four-class argmax; no new threshold search proposed.
- Apply D13's main and supplementary evaluation cohorts and coverage reporting.
- Keep missing outcomes/forecasts missing. Source-month joins must be explicit;
  no record-count shift presented as a calendar horizon and no invented fs3 expert.
- Pin release/source/runtime identities; save keyed predictions, phase mapping,
  split/partition provenance and counts sufficient to recompute metrics.

## Open scientific decisions for grilling

1. Exact frozen historical-outcome expansion and interactions under D19-D22;
   final technical defaults and claim wording (paired interval reporting
   approved in D16; local fitting/fallback approved in D17).

## Draft acceptance criteria

- All stages consume the same approved target definition; no current truth or
  expert projection enters model predictors accidentally.
- Stage 1 scoring, child selection and split gate use the approved multiclass
  objective; Stage 2 consumes matching scores and fresh partition candidates.
- RF probabilities retain a consistent four-class axis even with missing classes.
- Expert/persistence provenance is independently reconstructable for each key.
- Four-arm (fs1/fs2) and three-arm (fs3) metrics reproduce on identical keys
  within each comparison, with exclusions and phase-5
  treatment disclosed; final scores do not tune partitions or decision rules.
- A scientifically negative result is acceptable; evidence gaps are incomplete.

## Planning next steps

Resolve one user decision at a time; then converge this PRD and write design.md,
implement.md and curated context manifests. Execution, audit start and commit
remain separate from this planning approval.
