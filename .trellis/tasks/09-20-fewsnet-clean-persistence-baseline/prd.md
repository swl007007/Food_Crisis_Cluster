# FEWS NET clean baseline and final feature-engineering trial

Status: v1.0, final planning review; task remains planning.
Base checkout: ffdfd62c83b52a7e564af60b3ecf2358f92cac64.

## Goal

Establish an isolated FEWS NET baseline by rerunning Stage 1 partition learning,
Stage 2 consensus and Stage 3 forecasting with true class-1 F1 and SMOTE disabled
in both arms, followed by two-layer persistence correction. Then conduct one
final bounded, systematic feature-engineering trial. If it misses the agreed
success criterion, stop this line and move toward forecasting when expert
predictions are unavailable. The user will raise that fallback direction at the
next meeting; implementing that fallback is outside this task.

## Confirmed facts and evidence

The historical two-layer experiment failed its feasibility gate: delta F1
-0.0019605 at 4 months and +0.0131456 at 8 months; both intervals include zero
(PersistenceCorrectionExperiment/RESULTS.md:16-54,60-80). The 2021-2024 outcomes
have already been inspected and must not be called a fresh untouched holdout.

The released baseline already implements aggregate F1 including false positives,
compares parent and children on identical validation rows, and requires a gain
strictly above 0.01. Stage 1 and both Stage 3 arms have SMOTE disabled. Stage 1
still includes its inherited artificial class-recovery rows; disabling SMOTE
does not remove all synthetic observations (GeoRFBaseline/README.md:11-47).

Historical partition learning overlaps 2018-2019 calibration and 2020 threshold
selection. Correcting F1/SMOTE alone does not resolve this (archived persistence
design.md:90-95; DECISIONS_LOG.md:88-126). See research/reuse-and-boundaries.md.

Read-only preflight reconciled all 259,440 observed master targets with the
unadjusted source phases and found no duplicate canonical keys. The single
ledger tail artifact has an explicitly approved handling rule. Static/annual
consistency, WB co-location ties, monthly-source limitations and released
settings are recorded in the research files. No experiment was run.

## Document authority

- Requirements R1-R68 and acceptance criteria A1-A64 below are the task contract.
- design.md specifies the executable data/model/selection/evaluation flow.
- implement.md is the ordered execution and verification checklist.
- research/decision-log.md preserves all 64 user decisions without renumbering.
- research/approved-feature-sources.md owns exact source/feature names, order
  and formulas; its numbered approved contracts supersede earlier proposals.
- Other research files retain source and code evidence. Earlier unresolved
  wording in historical research does not reopen a subsequently approved choice.

## Requirements

- R1: Use a separate FEWSNETCleanPersistenceExperiment/ root for future code and
  fresh run directories. Pin source/release/runtime/configuration identities and
  preserve existing experiments, paper artifacts and baseline release unchanged.
- R2: Reuse the released F1/no-SMOTE implementation. Record any additional
  scientific change; IPCCH-specific preprocessing is not implicitly adopted.
  Preserve its existing class-recovery fitting behavior under D4, while
  excluding artificial rows from D20's real-support counts. No SMOTE must not
  be described as the absence of all artificial observations in Stage 1.
- R3: Build new Stage 1 -> Stage 2 -> Stage 3 artifacts and retain their lineage.
  Derive partition counts and map filenames from this run, not historical names.
- R4: Keep original-feature corrected and feature-engineered results distinct.
  Comparisons must use approved matching cohorts, horizons and training rules.
- R5: Define a finite feature-family inventory, deterministic candidate construction,
  selection rule and run budget before selection. Exhaustive means covering that
  agreed inventory, not an unlimited search in response to test performance.
- R6: Engineer features using only information eligible at each forecast origin;
  fit imputation, reference distributions and supervised selection within the
  allowed fitting data. Save ordered schemas and feature provenance. D33/D49 record
  the retained source-level limitations and fixed-snapshot convention; downstream cutoff compliance alone
  cannot certify historical availability of preconstructed source values.
- R7: Freeze partition/feature/calibration/threshold/evaluation time roles before
  fitting. Enforce D12/R16 and label any reuse of inspected test years.
- R8: Apply D10's raw/calibrated pair and D37's full eligible-sample fitting
  with up-only correction, D38's threshold selection and D39's reporting
  inventory. D41-D45 fix scientific roles and stopping; onset-only
  training is outside this contract.
- R9: Save per-row predictions, persistence source dates, partition assignments,
  candidate-selection evidence and metrics. Independently recompute final results.
- R10: Predefine success and stopping criteria. An unsplit model or negative result
  can be a complete deliverable; do not add variants after the final evaluation.
- R11 (D7): Build the updated feature inventory from the four approved additional
  source families and remove legacy FAO/WFP price families, including derived
  copies, before generating new transformations. Record exact column mappings,
  units, join keys, geographic/time coverage and availability assumptions. Do not
  silently include further external sources. D6's historical-feature reference
  remains separate; the source replacement is not applied retroactively to it.
- R12 (D8): Both the corrected original-feature baseline and the updated-feature
  experiment derive their target/cohort from the same historical master panel.
  Freeze its identity and canonical keys before assembly. Covariate joins must
  preserve master rows and labels; unmatched covariates remain explicit missing
  values rather than silently changing evaluation support.
- R13 (D9): Join WB prices using the approved same-month nearest-market rule.
  Record matched distance and unmatched counts, and preserve the master cohort.
  This spatial rule does not authorize reading a month later than the forecast
  origin or establish historical publication availability. D34 fixes the
  monthly endpoint; D64 fixes deterministic market ties.
- R14 (D10): Construct calibrated and raw correction results from identical model
  predictions and evaluation keys. Select a separate threshold per approved
  horizon and probability variant using development data only, then freeze it
  before final evaluation. Fit calibrators only on their assigned development
  data. Save the calibration and threshold provenance and report both variants;
  both receive equal weight under D41's primary metric; D42-D45 fix the
  effect-size, robustness and outcome rules.
- R15 (D11): Carry scopes 1/2/3 through the full new pipeline and retain scope
  provenance of Stage 1 candidates consumed by Stage 2. Extend the run-local
  persistence-correction orchestration to H=12; the historical helper currently
  supports only H=4/8. Do not silently omit fs3, reinterpret it as 8 months or
  change historical package outputs. Calibrators and thresholds remain separate
  per forecast horizon even when Stage 2 consumes candidates from all scopes.
- R16 (D12): For each reported forecast with origin O=T-H, the information used
  to fit/select every upstream artifact must be eligible at O. Include labels
  used to score/select Stage 1 plans and weight Stage 2 consensus, not merely
  the labels used to fit the local forests. Record information cutoffs and
  lineage for partition maps, feature sets, calibrators, thresholds and fitted
  preprocessing. Reject use of an artifact requiring information after O.
  For this retrospective run, these are simulated information-availability
  cutoffs, not claims that the artifacts were physically built in those years.
  Monthly availability assumptions and historical source-vintage limits must
  be explicit; the cutoff alone does not verify real publication timestamps.
  Under D33/D49, preserve and disclose supplied Gini/nightlight values, source-
  construction concerns and included inherited fixed layers across all origins.
  These source assumptions do not relax fitted-artifact/label cutoffs.
  Do not claim that the downstream cutoff checks
  establish an entirely leakage-free historical source-data pipeline.
- R17 (D13): Freeze and identify the selected upstream artifacts before final
  evaluation, for each approved feature arm and correction variant. Evaluation
  may refit RF and its prescribed fold-local preprocessing on eligible past data
  and compute the frozen feature recipe at the current origin; it may not change
  the feature recipe, partition maps, calibrators or threshold-selection result.
  A newly available earlier evaluation label may enter a later RF training window
  under the fixed rolling rule; this is not permission for further model selection.
- R18 (D14): Record the common latest information cutoff of the pooled Stage 2
  evidence, including all contributing scopes and selection/weighting outcomes.
  Apply it to every forecast using those maps. Derive eligible target months
  from O=T-H and the approved availability convention, never from an individual
  scope's earlier cutoff. Later feature selection, calibration or threshold
  fitting can move the effective freeze later; the partition gap alone does not
  establish eligibility for the complete correction pipeline.
- R19 (D15): Keep temporary-map development predictions and final-map evaluation
  predictions identifiable in the lineage. Verify map, RF, feature-selection and
  fitted-preprocessing cutoffs for each development origin. Calibration and
  threshold development must use the approved temporal folds and complete by
  2020-12. Validate actual early label/validation support; a configured window
  length does not establish that the corresponding history exists. Do not
  transfer partition-specific calibrators between maps by integer ID alone.
- R20 (D16): Fit and freeze one month-pooled calibrator for each horizon and
  target calendar month within each approved feature arm. Route by those keys,
  without local partition-specific overrides. Preserve the existing month-pool
  fitting and documented fallback behavior unless separately changed; record
  the input probability source and intentional pooled route. Raw correction
  remains uncalibrated under D10. This does not authorize adding pooled-RF
  correction arms or combining probabilities from different feature arms.
- R21 (D17): Enforce the approved 2018 calibration and 2020 threshold target
  sets and their distinct temporary-map cutoffs. Both probability variants use
  the same approved 2020 selection keys. Retain the 2018 calibrator fit for
  threshold selection and final evaluation; do not silently refit on 2019/2020
  labels. RF histories remain governed by origin eligibility and the fixed
  rolling-window rule, not a blanket exclusion of intervening years.
- R22 (D18): Identify each Stage 1 candidate by map role, target date and scope,
  using 2014-2016, 2016-2018 and 2018-2020 for the respective map roles. Apply
  the existing origin-dependent training mask within each candidate run. Keep
  evidence reused across overlapping windows tied to the same actual run and
  configuration; window overlap does not authorize mixing feature arms or
  extending a map's information cutoff.
- R23 (D19): Keep data-support exclusion, completed valid no-split evidence and
  execution failure distinguishable. D19's pre-graph unsplit branch requires at least
  one eligible, normally completed candidate and no positive-weight candidate
  contribution, with all required eligible runs accounted for. Reuse pooled RF
  predictions on the same feature schema, fitting rows and prediction keys for
  that map's single-partition arm. R66 separately governs valid automatic nc=1.
  Zero eligible candidates or missing/failed
  required evidence stops the affected build; it does not trigger this fallback.
- R24 (D20): Apply the released 35-calendar-month effective mask and within-area
  validation assignment without adding a minimum complete-history length.
  Check candidate support after applicable preprocessing and split, using only
  real rows: fitting, internal validation and labeled target subsets must each
  be nonempty. Save observed-month, area and class counts and exclusion reasons.
  Do not correct the 36/35 convention, count artificial rows as observations or
  introduce further history/class/area thresholds without an explicit decision.
- R25 (D21): Freeze the finite recipe manifest before new candidate performance
  is inspected. Each candidate must have compatible temporary partitions,
  origin-safe fitted transforms, 2018 calibrators and 2020 threshold evidence.
  Reuse artifacts only where their feature/configuration lineage matches.
  Select one common recipe for all three horizons using the approved 2020-only
  rule and freeze the selection by 2020-12. Final-period results cannot change
  the recipe, candidate inventory or raw/calibrated reporting roles.
- R26 (D22): Compute and retain all six development F1 gains and their equal
  mean. Use aggregate TP/FP/FN on identical per-horizon keys; do not average
  area/month F1 or let candidates improve by dropping evaluation rows. Record
  declared feature counts and fixed manifest order for reproducible exact-tie
  handling. Missing required cells cannot be omitted or assigned invented
  scores; resolve incomplete evidence before ranking.
- R27 (D23): Preserve the monthly covariate grid through calendar transforms
  and origin alignment; filter missing target labels afterwards. Record each
  feature's source month/window and declared time semantics. Fit max_plus
  statistics only on real fitting rows within the applicable split, excluding
  Stage 1 internal-validation rows and artificial class-recovery rows; Stage 3
  fits on its rolling RF training rows. Use the same fitted imputer for the
  corresponding validation/prediction rows. Apply this route to both feature
  arms without using full-panel statistics or treating sparse row shifts as
  calendar lags.
- R28 (D24): Freeze exactly 12 updated-feature recipes in the manifest, in the
  declared order BASE, A, B, C, D, E, ABCDE, BCDE, ACDE, ABDE, ABCE, ABCD for
  deterministic D22 tie resolution. BASE has the approved source replacement
  without additional A-E blocks; the original-feature reference is separate.
  Record explicit block membership and deduplicate identical input columns.
  Do not expand the manifest with further combinations or window variants in
  response to final-period performance. Preserve per-recipe lineage when
  reusing overlapping jobs or deterministic feature construction.
- R29 (D25): Implement the eight-column block A definitions in
  research/approved-feature-sources.md. Calendar sine/cosine uses target month;
  history and crisis recency use only eligible observed records at O=T-H, with
  ages measured in calendar months to O. Keep absent-history numeric values
  missing before D23 imputation; no history sets both history flags to one.
  Place these history flags only in block A, not duplicate copies in block E.
- R30 (D26): Construct block B over exact consecutive calendar windows of
  3/6/12 months ending at the source-specific eligible endpoint. Use mean and
  population standard deviation (ddof=0) for the frozen continuous-series list
  and sum for the frozen count-series list. Require every monthly source value;
  do not compute partial-window statistics, fill missing counts with zero or
  calculate summaries over imputed source history. Apply D23 imputation only
  after missing derived values have been recorded. Avoid duplicate inherited
  features and automatic expansion of static/previously derived columns.
- R31 (D27): Construct block C using the three approved formulas in
  research/approved-feature-sources.md from eligible, pre-imputation source
  values. Require exact calendar endpoints for differences and a complete
  preceding 12-month reference for standardized deviations. Do not substitute
  partial windows, full-panel statistics, an arbitrary denominator epsilon or
  zero for an undefined anomaly. Block C may compute its required summaries
  when B is off, without exposing those intermediates as extra input columns.
- R32 (D28): Construct only the four approved block D columns from their
  specified source fields and eligible pre-imputation operands. The three-month
  conflict operand requires all nine category-month values. Use the retained
  market_distance field, not WB match distance or coastline distance. Preserve
  signs and source units; do not multiply missing-value imputation sentinels.
  Internal B/C computations do not enable their model-input columns when those
  blocks are off, and block D adds no extra recipes.
- R33 (D29): Construct E's missing flags and dynamic-source ages from eligible
  cleaned source values before imputation. Do not shift the scheduled endpoint
  backwards to hide missingness. Age is O minus the latest eligible valid source
  month, with no arbitrary cap; absent history stays missing before D23. Static
  fields receive only a missing flag. Exclude identifiers, target/expert fields,
  calendar encodings, outcome history and inherited or B/C/D-derived columns
  from E's expansion, and deduplicate aliases. Preserve source interpolation
  and vintage limitations without interpreting filled values as observations.
- R34 (D30): Build updated-feature recipes from an explicit source-level BASE
  schema plus only the enabled A-E input columns. Use the same ordered schema
  throughout Stage 1 and Stage 3 for each recipe. Exclude inherited pipeline-
  generated calendar/history, EVI lag, nightlight sum and secondary scope-lag
  copies from updated BASE. Keep keys and labels for joins/evaluation without
  automatically exposing them as predictors. Preserve the separate corrected
  original-feature reference; its frozen schema is not governed by this removal.
- R35 (D31): Include the 22 approved additional BASE fields with explicit source
  mappings. Keep only food_inflation_wb for the WB inflation signal, excluding
  inflation_food_price_index; use soybean-oil last_price without its bid quote.
  Preserve distinct Bloomberg series identities and record unresolved units,
  timing/contract conventions and coastline extraction before execution. The
  22-field count excludes retained legacy sources and engineered block columns.
- R36 (D32): Generate B/C only from the exact 28 continuous and six count
  fields in research/approved-feature-sources.md, applying D26/D27 formulas and
  missingness rules. Expose 186 B and 102 C columns when their respective blocks
  are enabled. Do not recursively expand derived variables or add w5/w10 and
  other unlisted sources. Keep the approved recipe inventory unchanged.
- R37 (D33): Retain the supplied master Gini and nightlight mean/SD values at
  their source keys, applying the same source-value policy to both feature arms.
  Do not overwrite/rebuild the master or silently remove those fields as a
  source-quality remedy. Carry the evidenced construction issues, unresolved
  master lineage and resulting interpretation limits into the source manifest
  and results report. E's flags/ages describe visible valid source-table values;
  previously filled values cannot be relabeled as verified raw observations.
  New source alignment, B/C/D calculations and D23 imputation still follow the
  approved downstream rules; this does not authorize new future/cross-area fills.
- R38 (D34): For monthly covariates, align each target row to its own origin
  month O=T-H and use O as the scheduled source endpoint. Do not introduce an
  extra publication lag or backfill a missing O-month value from an older month.
  Apply the same rule to both feature arms and to historical fitting rows as
  well as prediction rows. Preserve complete-calendar B/C inputs, D's missing
  operand rule and E's separate age calculation. Record that publication timing
  and snapshot-vintage validity remain unverified under this convention.
- R39 (D35): Align GDP/CPI/CC/gini for each row's origin O to reference year
  year(O)-1 without origin-year values or older-year substitution. Validate
  repeated annual-value consistency before collapsing monthly copies; do not
  silently average conflicts. Missing annual values use D23's imputer. For E,
  measure age to O from the latest eligible valid reference year's December,
  independently of whether BASE has a valid immediately preceding-year value.
  Preserve the annual indicator identities and documented availability limits.
- R40 (D36): Attach persistence only from a valid observed IPC record dated
  exactly O=T-H, using its reconciled original binary crisis label. Do not map
  missing observed phase to zero or replace missing O-month records with older
  records. Mark unavailable persistence and retain its exclusion reason in the
  evidence. Across arms/variants, restrict paired persistence comparisons and
  threshold/recipe selection to the same valid target/origin support per horizon,
  recording coverage and class/area/time counts. A may use an earlier valid
  observation at or before O without making exact-O persistence available.
  Empty required scoring cells cannot be silently omitted or scored as zero.
- R41 (D37): Fit RF on all eligible target-labeled training rows under the
  approved partition/split/window rules, and calibrators on all eligible 2018
  labeled RF predictions under D16/D17. Do not condition either fit on the
  persistence state or availability. Correct only b=0 with p>tau; equality and
  b=1 preserve b. Preserve a legitimate no-correction outcome explicitly and
  leave unavailable persistence without a corrected prediction. Do not add
  an onset-only learner, error-indicator target or downward-flip path.
- R42 (D38): Select six thresholds/no-correction outcomes per feature recipe
  from 2020 only: three horizons times raw/calibrated probabilities. On D36's
  shared valid support, scan sorted distinct probabilities with D37's strict
  override; compute F1 from aggregate counts with the existing zero-division
  convention. Update only on strictly better F1, retaining the smallest tau
  among improving ties. If no strict gain exists, preserve persistence with
  explicit tau=null and zero development gain. Save the complete score trace
  and input identities; handle null without invoking a numeric comparison.
- R43 (D39): Produce the approved paired table for both final feature arms:
  common persistence plus pooled RF, partitioned RF and its raw/calibrated
  correction variants. Keep method/source/fallback lineage explicit and share
  D36's evaluation keys within each horizon. Reuse native RF hard predictions
  without standalone threshold tuning or a pooled-RF correction arm. Separately
  report full-support standalone RF metrics and counts; do not compare those
  unmatched rows to persistence or substitute them for the paired results.
- R44 (D40): Evaluate observed February/June/October targets from the approved
  per-horizon starts through 2024-10 inclusive. Apply D36's shared paired keys
  within each horizon without requiring a balanced panel of areas or removing
  rows solely for missing features. Record scheduled versus supported target
  dates and per-date counts. Derive the supplementary 2022-02 to 2024-10 table
  from the same predictions; do not substitute it as a favorable alternative
  main window or extend evaluation to newer data after inspecting outcomes.
- R45 (D41): For the frozen updated-feature winner, calculate each final F1
  gain over persistence from aggregate confusion counts on D36/D40 support,
  then average all six horizon/variant cells equally. Retain all cells and
  their sample counts. Report the original-feature reference's corresponding
  results and matched winner-minus-reference gains separately. Do not promote
  the strongest final cell, reference arm, standalone RF or supplementary
  window to replace the predeclared primary adjudication result.
- R46 (D42): Report actual class-1 F1 levels, each signed absolute-unit F1
  difference and the unrounded six-cell mean. Use +0.01 only as a descriptive
  reference, without a hard +0.01 or +0.02 success cutoff. Do not take the
  mathematical absolute value of a negative gain or substitute relative percent
  change for the signed F1 difference. Judge a smaller positive gain using the
  approved D43 robustness contract; magnitude alone cannot settle it.
- R47 (D43): Require a positive primary point estimate, a paired two-sided 95%
  interval with lower endpoint >0, and positive primary gains in each of four
  leave-one-target-year-out recomputations. Remove the same target year across
  all relevant horizons/variants/comparators, aggregate retained confusion counts,
  then average the six gains. Do not refit or select artifacts during these
  checks. Keep individual-cell outcomes visible; incomplete required evidence
  cannot be substituted with a favorable score or interpreted as a null effect.
- R48 (D44): Implement the joint bootstrap protocol in
  research/fs3-and-time-support.md: 11 shared target-date draws per attempt,
  default_rng(5), 2,000 valid replicates, and linear 2.5/97.5% quantiles of the
  recomputed six-cell means. Use paired weighted confusion counts and retain
  source keys, date multiplicities, bootstrap gains, RNG/runtime identity and
  attempted/invalid draw counts. Reject only undefined empty-horizon draws,
  never draws based on gain sign; stop at 20,000 attempts if evidence remains
  incomplete. Do not refit models or treat this as unconditional training or
  new-location uncertainty. Keep D43's year-exclusion checks separate.
- R49 (D45): Apply the approved three-way outcome mapping only after checking
  required support, lineage, prediction, metric and reproduction evidence.
  Distinguish a completed experiment with insufficient robust-benefit evidence
  from an incomplete experiment. Preserve the baseline, frozen winner, all
  primary/secondary results and any repair provenance. Do not retune scientific
  choices to reverse the verdict or implement the next research direction as
  part of this task. A changed scientific contract needs a separate decision.
- R50 (D46): Exclude Rainf_zscore, Tair_zscore and all their inherited derived
  copies from every model-input schema in both feature arms and all stages.
  Keep source files unchanged and retain the raw rain/temperature fields.
  Distinguish removed inherited scores from D27's permitted C features and D28's
  internal interaction operands. Do not substitute new reference-arm features
  or alter the approved block recipe inventory to compensate for the removal.
- R51 (D47): Remove fews_ha and every predictor derived from it from the
  corrected original-feature and updated-feature schemas for Stage 1 and
  Stage 3. Exclude its missingness/age expansion from E as well. Do not alter
  source records, replace the unadjusted target with fews_ipc_adjusted, or
  reinterpret observed IPC/persistence when enforcing this predictor exclusion.
- R52 (D48): Select population independently for each area's origin year from
  the last valid source month in the preceding year after key validation.
  Apply the same deterministic selection to both arms and retain the exact
  selected source value/date and observed within-year differences. Missing
  required-year population proceeds to D23 without cross-year BASE filling.
  E checks the selected annual value and measures age to O from December of
  the latest eligible year with a valid value; no history leaves age missing.
  Do not treat this age anchor as the actual source date or add population B/C
  derivatives. Disclose snapshot-vintage and demographic-year uncertainty.
- R53 (D49): Preserve included inherited fixed-geography source snapshots for
  both arms throughout the approved calendar, documenting reference/publication
  years and unresolved metadata/generator/master-lineage discrepancies. Do not
  rebuild layers, silently change sources or apply unverified unit conversions.
  Verify repeated values before reducing to one static value per area; surface
  conflicts rather than averaging or choosing a future row. E gives included
  static fields missing flags only. Treat market_distance as monthly under D34,
  including its D28 operand, and keep population under D48. Preserve strict
  temporal cutoffs for labels and all fitted/selected artifacts.
- R54 (D50): Extract coastline_dist using (lon,lat) containing-pixel lookup
  from the specified EPSG:4326 TIFF. Preserve native signed/zero values and
  record coordinates, raster identity, pixel row/column and validity reason.
  Missing/nonfinite/invalid geographic coordinates, out-of-bounds pixels or
  actual invalid raster masks produce missing without removing rows or
  snapping/wrapping coordinates. Do not import another product's sentinel
  rules; investigate unexpected value anomalies in preflight. Use the result
  as the approved fixed updated-BASE covariate and qualify sign/unit provenance
  and retrospective availability rather than claiming verified kilometres.
- R55 (D51): Keep administrative/country/group/partition identifiers outside
  the ordered predictor schemas in both feature arms, including any identity-
  derived copies or E expansions. Preserve them separately for all required
  joining, grouping, validation and routing operations. Both Stage 1 and Stage 3
  model fits exclude FEWSNET_admin_code as well as existing metadata-only group
  IDs. Include lat/lon and the seventeen source AEZ indicators explicitly as
  geographic predictors; do not replace excluded IDs with encoded alternatives.
- R56 (D52): Materialize the exact source whitelist and time roles in
  research/approved-feature-sources.md as ordered schemas: 64 common fields,
  86 updated-BASE fields and 67 original-reference source fields. Apply the
  approved monthly/annual/static rules to both arms and preserve the three
  old reference prices only in that arm. Account for all 88 master header
  fields without automatically including a new column or empirical role
  inference. Source selection does not authorize extra B/C/E expansions or
  change the D54 reference-transform contract.
- R57 (D53): Generate E's exact 140-column schema from D52 BASE order, emitting
  x_missing and then x_age_months for monthly/annual x, and x_missing alone for
  static x. Apply D29/D34/D35/D48/D50 definitions before D23 imputation; a valid
  AEZ zero is not missing. Preserve declared identities even when constant or
  coincidentally equal in a fitting window; deduplicate source/transform
  definitions rather than empirical values. Enforce all twelve approved recipe
  widths without exposing D's internal operands as extra inputs or adding E
  to the original-feature reference.
- R58 (D54): Materialize the reference's 109 ordered columns exactly as listed
  in research/approved-feature-sources.md: source fields, year_2010..year_2024,
  month_{1,2,4,6,7,10}, binary IPC lags 4/8/12, phase IPC lags 4/8/12, EVI lags
  1..12, WFP sums 4/12 and nightlight sum 12. Derive every fitting/prediction
  row from its own O=T-H on the full monthly area grid before label filtering.
  Require valid exact IPC observations, exact EVI endpoints and all source
  months for each O-exclusive sum. Apply D23 after missingness is established;
  no prior-observation substitution, cross-area rolling, second horizon lag,
  unapproved family expansion or new A-E input enters the reference.
- R59 (D55): For each approved recipe/role, pool that window's eligible
  Stage 1 plans over all observed candidate months and all three scopes into
  one general map. Use only the map with the correct recipe, role and cutoff
  for all corresponding Stage 3 month/horizon folds. Set month_ind false
  explicitly, overriding inherited environment settings; do not construct
  seasonal alternatives or substitute future/cross-recipe maps. Keep monthly
  calibration groups separate and preserve the candidate status/weight ledger.
- R60 (D56): Apply the exact consensus weighting, Gaussian, normalization,
  k40 sparsification and eigengap formulas in research/fs3-and-time-support.md
  without bandwidth/k/nc performance search or a manual minimum of two clusters.
  Record eigenvalues/gaps, chosen nc, graph and runtime/initialization identities.
  Preserve valid nc=1 outcomes with their selection reason and use R66's
  full-pool prediction reuse, preserving geographic assignment provenance.
  Failed graph/solver/support checks are incomplete evidence,
  not authorization to reduce nc or label the run a valid unsplit result.
- R61 (D57): Derive positive-edge connected components after D56 sparsification,
  select the largest with canonical area-code tie-breaking, and perform both
  eigengap selection and spectral fitting on the exact same ordered component
  affinity matrix. Record component sizes, selected IDs/matrix identity,
  eigenvalues/nc and assignments. Require at least three component nodes for
  the retained eigensolver formula; failed/invalid builds stop without parameter
  reduction or fabricated no-split evidence. Keep nonselected/isolated/absent
  areas and reasons available for separately specified completion, preserving
  evaluation rows and D19's pre-graph nonpositive-weight branch.
- R62 (D58): Build the sorted graph-node union only from valid non-s-1
  assignments in the approved recipe/window's completed eligible candidate
  ledger, corresponding to the release's full_universe=False behavior.
  Preserve zero-weight-only and isolated support reasons while adding edges
  solely under D56's positive-weight rule. Keep never-assigned/other-component
  areas available for completion and full cohort accounting; do not use future
  target labels, arbitrary discovered files or cross-recipe/window assignments
  to populate learned graph support. Retain D19's pre-graph unsplit distinction.
- R63 (D59): Complete the frozen master area-ID universe using exactly D57's
  fitted component as donors and the approved haversine rule (radius 6371 km,
  distance <=100 km, unrestricted by country, canonical-code exact ties).
  Preserve core labels during completion; do not add IPCCH's separate per-area
  fit/validation donor gate, adjacency-first propagation or newly completed
  donors. Keep other-component and never-in-graph support reasons identifiable.
  Missing/invalid recipient coordinates or excessive distance yield partition
  -1 and pooled prediction without deleting rows or imputing matching coordinates.
  Record coordinate identity, assignment source, donor/distance when defined
  and unresolved reasons; freeze per recipe/role without evaluation-label input.
  A failed required graph remains incomplete, not a geographic pooled fallback.
- R64 (D60): Use the spectral-core assignments and approved geographic completion
  directly as the frozen map for each recipe/role. Explicitly bypass inherited
  post-consensus contiguity-refinement calls and environment/batch defaults;
  do not apply neighbor-vote label changes before or after completion, produce
  alternate refined maps or select maps using forecast results. Keep Stage 1
  spatial operations and D56 affinity intact and document geographic islands
  without claiming single-component polygon partitions.
- R65 (D61): Construct one full eligible real training pool per Stage 3
  recipe/horizon/fold under [O-35 months,O), with each historical row's own
  origin-aligned predictors. Do not filter by target-month area/group presence
  or persistence availability. Fit one max_plus imputer on this pool and reuse
  it unchanged for pooled/local training and prediction. Pool all eligible
  rows; local models use their frozen partition subsets, including completed
  areas, with >=50 real rows and both classes. Route missing local support,
  single-class/unseen partitions and unassigned areas to the same pooled model
  for native predictions and class-1 probabilities. Record map provenance
  separately from actual model routes, and fail on empty global training or
  fit errors rather than converting defects into statistical fallback. Keep
  protected baseline code unchanged; implement the consistent route locally.
- R66 (D62): For a successfully supported automatic-nc=1 map, route its entire
  partitioned stream to the same recipe/horizon/fold pooled RF and reuse both
  output types without an assigned-subset local fit. Preserve map provenance
  and unresolved areas independently of model routing. Record the valid-unsplit
  reason and corresponding pooled model/output identities; D19 retains its
  distinct no-positive-weight reason. Keep all declared correction streams and
  their existing fitting/selection rules. Do not trigger this branch from failed
  graphs, fold support counts, target coverage or observed performance.
- R67 (D63): Validate original master binary and unadjusted phase together,
  accepting only binary 0/1 and integral phases 1..5 with the approved mapping
  and missingness agreement. Reconcile observed master labels with same-key
  ledger phases; check canonical key uniqueness and redundant ledger dates.
  Derive history/persistence binary only from valid observed phases, leaving
  missing observations unavailable under D36. Exclude the single verified
  terminal artifact only in the pinned ledger's parsed view, recording source
  identity, physical line, raw content and reason. Any other malformed record,
  duplicate, invalid nonempty label or source disagreement stops preflight.
  Keep raw files and master rows unchanged; do not substitute adjusted phases
  or projected expert fields for truth.
- R68 (D64): Restore raw market IDs only after source-identity, row-count,
  ordered-key/coordinate and retained-value/missingness reconciliation with
  the selected derived WB file. Preserve its numeric values and use geo_id
  only for deterministic matching/provenance. Resolve exact same-month nearest
  haversine ties by ascending lexical geo_id under D9's inclusive 100 km and
  cross-country rule, copying both retained fields from the chosen row. Keep
  missing fields missing; do not average markets, rely on BallTree's unspecified
  tie return, or choose using values, completeness, confidence or target labels.
  Record the chosen market, distance and tie evidence; reject unverifiable
  lineage or conflicting duplicate market-month keys before assembly.

## Acceptance Criteria

Required evidence for authorized implementation and execution. Planning-only
source scans do not satisfy model/result acceptance criteria:

- A1 (R1-R2): Hashes/configuration are recorded, true-F1/no-SMOTE checks pass and
  protected original files are unchanged.
- A2 (R3): Every evaluation map traces to this run's eligible Stage 1/2 evidence;
  uncovered-area and no-split behavior is explicit.
- A3 (R4): Baseline/feature comparisons reconcile to the approved common support.
- A4 (R5-R7): A frozen candidate manifest and ledger show every attempted feature
  set, origin cutoff, transformation and selection outcome.
- A5 (R8-R9): Persistence alignment, calibration and thresholds are reproducible
  without final evaluation labels entering their fitting/selection.
- A6 (R9-R10): Independent F1/uncertainty/flip accounting supports an explicit
  go/no-go verdict, including null results.
- A7 (R11): The updated raw/engineered schemas reconcile to the approved four
  additions and legacy-price removals, with join coverage and timestamp evidence.
- A8 (R12): Before/after assembly checks prove that covariate additions preserve
  the historical master panel's canonical keys, row count and target values.
- A9 (R13): WB matches reconcile to the nearest eligible same-month market at
  distance <=100 km, without filtering by country; unmatched cases stay missing
  and match-distance/coverage summaries reconcile to the preserved master rows.
- A10 (R14): Raw/calibrated comparisons share the underlying GeoRF predictions
  and keys; their separately frozen thresholds and calibrators are reproducible
  from development data, with both final result rows retained.
- A11 (R15): Stage 2 input manifests include eligible new candidates from fs1,
  fs2 and fs3; final reports include 4/8/12-month forecasts. Per-row origins and
  persistence source months use the correct calendar horizon, including H=12.
- A12 (R16): Per-forecast lineage proves that every fitted/selected upstream
  artifact respects its forecast-origin information cutoff, including fs3.
  A future-dependent map, feature selection, calibrator or threshold fails the
  contract even when that forecast's RF training labels are themselves eligible.
- A13 (R17): The frozen maps/feature recipes/calibrators/thresholds have unchanged
  identities throughout final evaluation. Per-fold RF fitting and preprocessing
  records demonstrate the prescribed rolling update without renewed selection.
- A14 (R18): Every evaluated scope respects the pooled map's common information
  cutoff and any later upstream freeze. The time ledger rejects a prediction
  whose origin precedes evidence contributed by another scope to its map.
- A15 (R19): Every calibration/threshold development row traces to an eligible
  temporary map and origin-safe model inputs. The finalized fold ledger records
  actual support and the approved calibrator routing rule; final evaluation
  cannot silently match a temporary partition ID to a different final partition.
- A16 (R20): Calibration artifacts and per-row routes reconcile to horizon,
  target calendar month and feature arm, without a partition-ID lookup. Required
  target months have an explicit fitted calibrator or documented existing
  fallback; missing month coverage may not silently invent a fitted mapping.
- A17 (R21): The time ledger proves calibration targets are in 2018 and their
  map evidence is no later than 2016-12; selection targets are in 2020 and
  their map/calibrator cutoffs are no later than 2018-12. Calibrator identities
  remain unchanged between threshold selection and final evaluation, and all
  three horizons satisfy the origin checks on these folds.
- A18 (R22): Candidate manifests reconcile to the approved map-role year
  windows and scopes. Consumed candidates trace to compatible run evidence;
  no candidate target outside its map's approved window contributes to that
  map, even if its own RF training labels would predate the map cutoff.
- A19 (R23): Any single-partition result is supported by the complete candidate
  status/weight ledger and matches its corresponding pooled predictions. A
  build with no eligible candidate or failed required evidence cannot produce
  a successful no-split status or substitute a later map.
- A20 (R24): Boundary checks agree with [O-35 months, O), including exclusion
  of origin-month labels. Candidate eligibility reconciles to real-row support
  after preprocessing/splitting; a short but supported history is not rejected
  solely for missing a complete window, and synthetic-only support cannot pass.
- A21 (R25): The selection ledger reproduces the shared winner from the frozen
  candidate inventory and 2020 evidence, with compatible lineage for each
  candidate's development outputs. Final evaluation retains the original-feature
  reference and frozen winner with both correction variants; no final label or
  post-evaluation candidate enters the feature-selection result.
- A22 (R26): Independent recomputation from 2020 predictions reproduces each
  F1 gain, the six-cell mean and the selected recipe, including the declared
  feature-count/manifest-order tie rule. All compared row keys reconcile.
- A23 (R27): Calendar lineage verifies lag/window values against source months
  even when target labels are sparse. Imputation artifacts trace exclusively
  to each split's real fitting rows and are reused unchanged for its validation
  and predictions. Original and updated feature arms both use this contract;
  no full-panel imputation or target-month dynamic value bypasses the origin.
- A24 (R28): The frozen search manifest contains the approved 12 distinct block
  combinations and separate original-feature reference, with stable ordering
  and no undeclared recipe. Final evaluation uses only the frozen winner and
  reference; raw/calibrated variants do not create additional feature recipes.
- A25 (R29): Block A reconciles to the eight frozen column definitions and
  per-row observed source dates. Missing/no-crisis histories yield the declared
  values and flags; adding later observations cannot alter earlier-origin
  history features. No observation age is mistaken for continuous crisis duration.
- A26 (R30): Block B values reconcile to their exact source months and frozen
  whitelist for all three windows. Missing input months yield missing derived
  values before imputation; means, population standard deviations and count
  sums reproduce independently without changing the approved recipe count.
- A27 (R31): Block C values reproduce from the frozen source whitelist and
  calendar endpoints. Missing intermediate months do not invalidate a difference
  with valid endpoints but do invalidate a standardized deviation requiring
  those months. The current endpoint is excluded from reference statistics;
  zero reference variance yields missing before imputation. C remains defined
  with B off and adds no undeclared model-input summary columns.
- A28 (R32): Block D's four products reproduce from the declared source fields
  and origin-eligible operands. Missing or undefined operands leave products
  missing before imputation, including an incomplete conflict window or zero
  reference variance. D alone exposes exactly four additional columns, with
  identical product definitions when B/C are on or off.
- A29 (R33): E flags reconcile to the scheduled endpoint's pre-imputation
  validity and ages to the latest eligible valid source month. A one-month
  source lag yields age one for a valid endpoint; no history yields flag one
  and missing age. Later or imputed values cannot change earlier-origin ages.
  E does not alter source values, relax complete-window rules or duplicate
  outcome-history/derived-feature flags; columns match the frozen whitelist.
- A30 (R34): The updated all-off recipe contains exactly the frozen source-level
  BASE columns; enabling a block adds only its declared inputs. Stage 1 and
  Stage 3 ordered schemas match within each recipe, with no inherited pipeline
  calendar/history/lag/sum columns bypassing the block manifest. The corrected
  original-feature reference remains distinct and reconciles to its own schema.
- A31 (R35): The added-source manifest reconciles to one ENSO, two WB, one
  coastline and 18 Bloomberg fields with the approved exact series identities.
  Excluded WB/soybean-oil quote fields do not enter predictors or derivatives.
  These additions do not create extra feature recipes or certify unresolved
  source availability, units or extraction semantics.
- A32 (R36): Ordered B/C schemas reconcile to their 34 approved source identities
  and transformation types, with 186/102 columns respectively and no undeclared
  fields. Counts remain valid even when source support leaves a column missing;
  do not add/drop transforms based on final-period data. Each value retains the
  calendar lineage and missingness semantics required by A26/A27.
- A33 (R37): The input mapping retains the supplied Gini/nightlight source values
  without the deferred source reconstruction. Provenance and result limitations
  distinguish verified generator behavior from unverified master execution
  lineage, and distinguish downstream origin checks from source-level validity.
  E metadata does not claim recovery of original missingness or measurement age.
- A34 (R38): Monthly feature lineage maps T to O=T-H for each approved horizon
  and each fitting/prediction row. The BASE endpoint is exactly O, with no O+1
  source or extra O-1 shift; missing O values remain missing before D23 even if
  E locates older valid values. Window and difference endpoints reconcile to O.
- A35 (R39): Annual-source lineage uses year(O)-1 for all origins within a
  calendar year, including January, and advances at the January boundary.
  A missing required year does not introduce an older BASE value. E ages agree
  with reference-year December anchors; conflicting annual copies are reported.
  General annual CPI is not confused with monthly food CPI/inflation.
- A36 (R40): Each persistence prediction traces to a valid exact-O observation
  and consistent binary label; absent or missing-phase records stay unavailable.
  Source-key coverage alone does not pass the valid-observation check. Paired
  comparison/selection keys match across recipes and probability variants,
  coverage and exclusions reconcile to the retained master rows, and A's prior
  history does not silently fill unavailable persistence.
- A37 (R41): RF/calibration fitting ledgers contain all otherwise eligible
  labeled rows without persistence-based exclusion. Corrected outputs obey
  b=0 and p>tau exactly, with no 1-to-0 flips or flips at equality. No-correction
  returns b unchanged and unavailable persistence remains unavailable. Raw and
  calibrated variants retain their own thresholds and common prediction keys.
- A38 (R42): Recomputing from 2020 common-support predictions reproduces each
  candidate F1, selected threshold or no-correction result, and tie outcome.
  A no-gain cell remains present with zero recipe-selection gain and unchanged
  persistence predictions. Thresholds are shared across months/partitions and
  remain frozen throughout final evaluation, including explicit null outcomes.
- A39 (R43): Paired results contain the approved five method types for each
  feature arm, with one common persistence stream and matching per-horizon keys.
  Correction probabilities trace to the partitioned route; native RF decisions
  have no extra threshold selection. Supplementary full-support RF results
  reconcile to their own counts and the same stored predictions, without extra
  fits or unapproved arms. Duplicate predictions retain their method provenance.
- A40 (R44): Forecast manifests reconcile to the 11/10/9 scheduled dates and
  per-row eligibility within the approved final windows, with no balanced-area
  or complete-feature filter. Supplementary common-calendar rows are an exact
  date subset of stored predictions and report their own per-horizon support;
  the original full-window results remain present and identified as primary.
- A41 (R45): Independent confusion-count recomputation reproduces all six
  final gains and their equal mean. Primary and secondary comparisons retain
  the predeclared arms, windows and keys. The report distinguishes benefit
  over persistence from incremental benefit over the original-feature reference
  and does not treat the six correlated cells as independent experiments.
- A42 (R46): Reports preserve F1 levels and signed, unrounded gains. Decision
  logic does not reject a positive result solely for being below +0.01 or
  automatically accept one for exceeding it. Negative gains retain their sign;
  +0.01 is identified as advisory and the earlier +0.02 hard gate is absent.
- A43 (R47): The robustness ledger reproduces the positive point-estimate and
  interval-bound tests plus all four year-exclusion gains from frozen outputs.
  Excluding 2021 leaves fs3's 2022-onward rows intact. No required cell disappears
  without an incomplete-status explanation, no additional per-cell-significance
  gate is introduced, and a failed robustness condition is reported accurately.
- A44 (R48): Independent reconstruction from stored counts and shared date
  weights reproduces every accepted bootstrap gain and the final percentiles.
  Horizons/variants/comparators use the same draw weights and original eligible
  support; missing-horizon attempts are traceable and never scored as zero.
  The valid count, attempt limit and seed reconcile, no model fitting occurs,
  and interval limitations are included in the uncertainty report.
- A45 (R49): The final report's evidence-completeness status and scientific
  conclusion reconcile to the stored checks and D43 results. A complete result
  failing robustness can close the bounded experiment; missing/invalid required
  evidence cannot. Primary roles and frozen choices remain unchanged after
  evaluation, repaired artifacts retain provenance, and future-work suggestions
  are separate from the completed deliverable.
- A46 (R50): Frozen schemas and feature lineage confirm that neither inherited
  climate score nor its derived copies enters Stage 1/3 in either arm. Raw
  rainfall/temperature remain included, permitted C features and D operands
  use the approved trailing formula, and the report discloses the intentional
  reference-schema change without asserting unverified old-master leakage.
- A47 (R51): Ordered schemas and feature lineage contain no fews_ha-derived
  predictor in either arm or stage, including E flags/ages. Original targets
  and approved IPC/persistence histories reconcile to the unchanged source
  contract; the report identifies the deliberate assistance-signal exclusion.
- A48 (R52): Population lineage reproduces each preceding-year last-record
  selection and preserves its value/date without averaging or rounding. All
  origins within one calendar year use the same selected population; missing
  required years remain missing before fitting-only imputation. E flags and
  December-anchored ages reconcile, including January year transitions and
  older-year age-only lookups. Both feature arms follow the same source policy.
- A49 (R53): Both arms' included inherited static inputs trace to the preserved
  snapshots across all origins. Conflicting repeated values are reported before
  static reduction, no unauthorized source/unit substitution occurs, and the
  report states the 2015 snapshot and known interpretation/vintage limitations.
  Static E fields have no age; market_distance lineage uses exact O under D34
  and D's interaction never substitutes target-month or WB-match distance.
- A50 (R54): Recorded coordinates/pixel indices reproduce native sampled
  values, including positive, negative and zero examples. Invalid coordinates
  and raster bounds/masks reconcile to missing reasons without cohort loss.
  No interpolation/rescaling/absolute-value transform or source sentinel guess
  appears. The field is static, absent from the original-feature reference,
  and has only its permitted E missing flag; provenance limitations are reported.
- A51 (R55): Model-input schemas and fit lineage contain no administrative,
  country or partition identifiers/encodings in either arm or stage, while
  metadata still resolves every sample's area and required routing assignment.
  Geographic predictors retain lat/lon and the seventeen source AEZ indicators;
  the report identifies the intentional admin-ID removal from the reference.
- A52 (R56): Source schemas reproduce the exact approved field identities,
  order and 64/86/67 counts, with no unaccounted header field or undeclared
  addition. Stage 1/3 schema identities match within each arm. Spatial conflict
  w5/w10 and nearest-conflict distance remain source inputs without unauthorized
  B/C transforms; source counts are not reported as total engineered widths.
- A53 (R57): Ordered E inputs reconcile to 86 missing flags and 54 ages, with
  no ages for the 32 static sources or flags for excluded/engineered inputs.
  All twelve recipe widths match the recorded manifest arithmetic, including
  526 for ABCDE. Values follow the approved temporal/missingness semantics and
  window-constant columns do not silently change the fitted input schema.
- A54 (R58): Reference feature lineage reproduces all 109 names/order and the
  67+21+6+12+2+1 count in every scope/stage. Calendar indicators match T while
  history endpoints/windows match O, including H=12 cases where T-4 is future
  information. Missing endpoints or incomplete sums stay missing before
  fitting-only imputation, and original O-exclusive windows remain distinct
  from updated B's O-inclusive windows. No automatic duplicate lag or expanded
  legacy helper family appears.
- A55 (R59): Each Stage 3 fold resolves to its role/recipe's general map,
  independent of target month/horizon, while map cutoffs and recipe identity
  remain valid. Eligible Jan/Apr/Jul candidates from 2014-2015 are not discarded
  by a 2/6/10 filter. No month_ind/environment setting selects a monthly map;
  calibration retains its approved horizon/month grouping. The planned map
  inventory reconciles to 26 development plus two final builds and records
  actual support/no-split/failure outcomes rather than assuming success.
- A56 (R60): Stored candidate scores and graph evidence reproduce the declared
  weights, sigma=5-degree kernel, normalization, k40 symmetric graph and
  automatic eigengap nc. Valid nc=1 retains its reason and pooled-equivalence
  lineage; unsupported/failed builds cannot claim that outcome. No final RF
  score, historical nc, bandwidth search or alternate-seed selection determines
  the cluster count, and solver/runtime identity is recorded.
- A57 (R61): Stored node IDs and affinity identity prove equal graph support
  for cluster-count selection and fitting. Component choice/ties reproduce
  deterministically; n in the eigengap formula is the selected component size.
  Fewer than three core nodes or solver failure produces incomplete evidence,
  while supported nc=1 and all-nonpositive weights retain their distinct valid
  reasons. Other areas remain traceable for completion and are not dropped
  from evaluation or mislabeled as geographically contiguous.
- A58 (R62): The stored graph-node set equals the eligible candidate-assignment
  union exactly, with canonical ordering and recipe/window provenance.
  Zero-weight-only and missing/s-1 support cannot generate co-membership edges.
  Other-component and never-assigned areas remain accounted for separately;
  later evaluation labels do not change the graph universe or hide cohort loss.
- A59 (R63): The completed map contains exactly one row per frozen master area.
  Core assignments are unchanged by completion; each completed recipient traces
  to the nearest original core donor under the inclusive 100 km/cross-country
  rule and canonical tie order. No chained donor or future outcome determines
  routing. Over-cap/invalid-coordinate cases stay -1 with reasons and pooled
  routing, and all area/support counts reconcile without dropped rows. Map
  identities remain frozen for their prediction role; D19 no-split and failed
  graph builds cannot be disguised as successful spectral completion.
- A60 (R64): Frozen map lineage resolves directly to spectral-core labels and
  D59 completion without an intervening or later majority-vote refinement.
  Core labels and donor-inherited recipient labels remain unchanged through
  prediction routing, and inherited batch settings cannot select refined maps.
  The report identifies the historical-route departure and possible geographic
  islands; Stage 1 spatial logic and D56 weighting retain their approved settings.
- A61 (R65): Training-key ledgers reproduce the full D20 pool, including eligible
  completed and unassigned history, independently of prediction-month coverage.
  Shared imputer identity/statistics trace only to that real pool; each local
  fitting subset matches its frozen partition and support gate. A recipient
  need not have its own history to receive its partition's prediction. Missing,
  small, single-class and unseen local groups have identical hard/probability
  pooled routes with reasons, without silent zero hard predictions. Empty
  global training or actual fit failures cannot produce successful fold output.
- A62 (R66): Every valid automatic-nc=1 fold's partitioned native decisions and
  probabilities exactly match its pooled outputs, including unresolved areas,
  with no assigned-only local fit. Geographic statuses and counts remain intact,
  unsplit reasons distinguish nc=1 from nonpositive weights, and correction
  lineage uses the declared stream. A valid nc>1 map retains D61 regardless
  of how many local models a fold supports; failed builds remain incomplete.
- A63 (R67): Pinned-source preflight reproduces the valid target/key and
  cross-source reconciliation in research/target-label-contract.md, including
  259,440 valid master labels and the one explicitly excluded non-observation
  artifact. Missing phases cannot become zero persistence. Raw hashes and
  master keys/labels remain unchanged; duplicate/invalid/conflicting evidence
  cannot pass through silent deduplication, deletion or relabeling.
- A64 (R68): Source lineage restores the correct market ID without expanding
  rows or altering derived values. Same-coordinate and other exact-distance
  ties reproduce the canonical lexical-ID choice independently of input row
  ordering; both fields share the chosen market even when one is missing.
  Distance/month/country behavior matches D9, IDs stay outside predictors,
  and tie/coverage evidence accounts for conflicting co-located markets.

## Out of scope

Out of scope: building the fallback model itself; overwriting old results;
adopting the IPCCH target or skipping Stage 2; unrestricted model-family searches;
treating the consumed old feasibility gate as a fresh preregistration; training,
committing or publishing during the current planning phase.

## Planning closure and execution checks

No user-owned design decision remains open. Unknown source vintages/units/roll
semantics are disclosed limitations under the approved snapshot policy; they
are not permission to replace sources or invent conversions. Formal runtime
availability, source/geometry hashes, actual candidate support, compute usage,
model results and reproduction checks remain execution-time gates. A failing
gate stops affected work and is not evidence of a scientific null.

All model code and experiments remain unstarted. Review this converged PRD,
design.md and implement.md together. Only a subsequent explicit instruction
to implement/run authorizes leaving planning; individual prior decisions did
not authorize training, commits, pushes or automatic task closure.
