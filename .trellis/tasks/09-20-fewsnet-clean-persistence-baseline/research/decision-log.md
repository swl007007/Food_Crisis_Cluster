# Confirmed decision record — D1-D64

These are the preserved chronological approvals from PRD v0.61. Later numbered
decisions resolve provisional open-language in earlier entries; such language
is historical context, not a current blocker. The converged prd.md requirements
and design.md govern execution. No implementation or run is authorized here.

- D1: Create a new task; brainstorm, persist spec, then grill until convergence.
  Do not begin model execution during planning.
- D2: Use a new experiment folder and rebuild all three stages. Old partitions
  and predictions are historical references, not substitutes for the rerun.
- D3: Fix the recall-biased partition objective and the two-arm SMOTE mismatch.
- D4: Reuse the initial foundation of IPCCH. The matching shared component is
  GeoRFBaseline v0.1.0, not IPCCH's dataset-specific single-stage partition design.
  This includes the release's previously approved Stage 1 class-recovery fitting
  rows (one zero-feature row per class); Stage 3 retains original-row fitting.
  This preservation follows release reuse, not a new resampling experiment.
- D5: Include a final systematic feature-engineering attempt and an explicit
  failure-to-fallback decision. Feature transformations, budget and success bar
  remain open; the added source families are fixed in D7.
- D6: Preserve the original-feature corrected baseline and evaluate engineered
  features separately. The user accepted this two-part comparison on 2026-09-20.
- D7: Before the final feature-engineering attempt, add NOAA_ENSO, WB_RTP,
  Coastline_distance_NOAA and bloomberg_food_and_derivative. Remove the original
  FAO price and WFP price/standard-deviation feature families from this updated
  feature set. Engineer features from that updated set. This new-source scope
  was explicitly approved on 2026-09-20. The exact old fields are `FAO_price`,
  `WFP_Price`, `WFP_Price_std`; remove their derived copies as well. Separate
  `market_distance`, `Food_CPI` and `Food_food_inflation` are not removed by this
  instruction. Source inventory: research/approved-feature-sources.md.
- D8: Retain `Analysis/1.Source Data/FEWSNET_forecast_unadjusted_bm.csv` as
  the target/cohort master panel, as approved on 2026-09-20. Join the four new
  sources onto its area-month framework. Do not switch to the newer assembled
  2025/2026 panel. Evaluation windows remain to be specified; horizons are in D11.
- D9: Follow the existing IPCCH WB price match: nearest market in the same
  calendar month, haversine distance at most 100 km, allowing cross-border
  matches. No same-country restriction. Preserve missing values when no market
  qualifies. The user explicitly chose the IPCCH rule on 2026-09-20.
- D10: Retain calibrated-probability persistence correction and add raw-probability
  correction as a fixed, predeclared comparison, approved on 2026-09-20. Both
  reuse the same GeoRF model predictions, with their own development-selected
  and frozen thresholds. Apply this comparison consistently to the original-
  feature corrected baseline and the updated-feature experiment. Report both;
  do not select the preferred variant after viewing final evaluation results.
- D11: Include fs3, not only fs1/fs2: horizons are fs1=4, fs2=8, fs3=12 months.
  On 2026-09-20 the user explicitly required fs3 to supply enough candidate
  partitions for Stage 2. Generate Stage 1 evidence for all three scopes and
  include it in the Stage 2 candidate pool, then carry fs3 through Stage 3 and
  raw/calibrated persistence-correction evaluation. fs0 is not included.
- D12: Apply the forecast-origin information cutoff to all learned or selected
  artifacts, approved on 2026-09-20: partition maps, feature selection,
  calibrators and thresholds, as well as RF training inputs. A label being
  earlier than the target month is insufficient if it was unavailable at that
  forecast's origin. Evaluation must begin late enough for its horizon and
  artifact cutoff; exact windows remain open.
- D13: One development-and-freeze cycle, approved on 2026-09-20. Before final
  evaluation, fix the partition maps, selected feature recipe, calibrators and
  thresholds. During evaluation, refit RF using the rolling eligible training
  window; do not update partitions, reselect features, recalibrate or retune
  thresholds using evaluation-period outcomes. Deterministic features/history
  still update from information available at each forecast origin.
- D14: Stage 2 pools fs1/fs2/fs3, so forecasts using its consensus maps inherit
  the latest information cutoff across the contributing scopes, including
  scoring/weighting labels. A scope's own earlier training cutoff cannot make
  the pooled map available earlier. On 2026-09-20 the user confirmed that
  "2018-2021" means [2018-01, 2021-01): the three full years 2018-2020,
  excluding 2021. Retain this final partition-learning period and reserve
  horizon-dependent gaps before Stage 3. With a conservative 2020-12 freeze,
  partition eligibility begins at fs1=2021-06, fs2=2021-10, fs3=2022-02 among
  observed target months. This matches the user's intended fs3 start and
  eight-month gap for fs2. D17/D21 fix development time roles; measured early
  support and source availability still need verification. This replaces
  the unapproved proposal to move final partition learning to 2016-2018.
- D15: The user approved additional temporary development partitions to preserve
  the D14 evaluation starts while freezing calibration and thresholds by
  2020-12. Generate development predictions in time order; each fold's
  partition may use only evidence available at that fold's forecast origins,
  including Stage 2 scoring labels across its pooled scopes. Do not apply the
  final 2018-2020 maps retrospectively to those earlier origins. Final evaluation
  still uses the final maps learned from 2018-2020, frozen under D13. D17 fixes
  the development target dates and map cutoffs; D18 fixes candidate windows.
  D20 fixes candidate support gates. Calibration grouping is fixed in D16. This
  approval does not authorize model execution.
- D16: The user approved calibration pooled across all partitions separately
  for each forecast horizon and target calendar month. Fit each feature arm
  separately on its own development predictions. Partitioned RF training is
  unchanged; only probability calibration is pooled. Reuse the existing
  month-pooled calibration logic and omit partition-specific calibration for
  this experiment. Routing must not depend on temporary or final partition IDs.
  This is an explicit change from the historical local-calibration protocol,
  not evidence that temporary and final model probability distributions match.
- D17: The user adopted the development calendar: fit calibration using 2018
  targets (February/June/October), predicted with temporary partitions whose
  information cutoff is at most 2016-12; freeze those calibrators by 2018-12.
  Select raw and calibrated correction thresholds using 2020 targets
  (February/June/October), with temporary partitions frozen by 2018-12 and the
  frozen 2018 calibrators for the calibrated variant. Freeze thresholds by
  2020-12 and retain D14's final evaluation starts. 2019 is outside the
  calibration-fit and threshold-selection target sets but remains eligible
  rolling RF history when available at the origin. Calibration has one target
  year, not multiple independent years. Successful early partition construction
  is not yet verified; D21 fixes feature-selection chronology.
- D18: The user approved the two temporary-map Stage 1 candidate target-year
  windows: 2014-2016 for the map used by 2018 calibration predictions, and
  2016-2018 for the map used by 2020 threshold-selection predictions. The final
  map retains 2018-2020. All three map roles pool fs1/fs2/fs3. These windows
  identify candidate target dates, not the RF training-history window. Early
  candidate eligibility is fixed in D20; D19 defines the no-split/failure
  distinction. Inclusion in a candidate window does not prove map feasibility.
- D19: The user adopted the no-split rule: if eligible candidates complete
  normally but none supplies positive Stage 2 consensus weight, retain a valid
  single-partition result and reuse the corresponding pooled RF predictions,
  explicitly labeled as such. If no candidate meets the sample requirements,
  or execution/artifact failures prevent a valid map build, stop the affected
  map build and report the reason. Do not borrow later maps or present a failed
  run as a valid unsplit result. D20 fixes eligibility; handling positive-weight
  graphs unsupported by consensus settings remains open.
- D20: The user adopted the released training-window and early-candidate rules:
  preserve [O-35 calendar months, O), with O=T-H, despite the configuration
  value 36; exclude origin-month labels from RF fitting. Allow shorter actual
  history inside that window without requiring three complete years. Retain
  the released within-area validation split. A candidate needs nonempty real
  model-fitting, real internal-validation and labeled target samples; otherwise
  record and exclude it. Artificial class-recovery rows cannot establish real
  support. Log actual observed history and support, not just the nominal window.
- D21: The user approved the feature-selection flow: predeclare a finite set of
  feature combinations, run each candidate's own development pipeline with
  temporary partitions, 2018 calibration and 2020 threshold selection, then use
  only 2020 development outcomes to select one feature recipe shared across
  fs1/fs2/fs3. Freeze that choice before final evaluation. Evaluate the selected
  recipe alongside the original-feature corrected baseline, retaining both
  raw and calibrated correction results. D22 fixes scoring and tie rules;
  D24 fixes the recipe budget; exact block contents remain open. The jointly optimized 2020
  feature/threshold results are development evidence, not independent proof
  of an improvement; do not retroactively treat the eventual winner as selected
  at earlier development origins.
- D22: The user adopted the feature-selection score: equal-weight mean of six
  2020 class-1 F1 gains over persistence, crossing fs1/fs2/fs3 with raw/calibrated
  correction. Compute each cell's F1 from pooled confusion counts over the three
  2020 target months, with identical evaluation keys across candidates and
  probability variants within each horizon. Select the greatest mean gain;
  exact ties prefer fewer frozen model-input feature columns, then the fixed
  candidate-manifest order. This selects a feature recipe, not the final
  scientific success/stop criterion, which remains open.
- D23: The user adopted shared preprocessing for the corrected original-feature
  reference and updated-feature candidates: construct calendar-based lags and
  windows on the complete monthly covariate panel before selecting labeled
  targets, align dynamic inputs to the forecast origin, and retain the existing
  max_plus imputation method while fitting its statistics only on the actual
  training subset. Transform validation/target rows with those fitted statistics.
  The original-feature reference preserves source families while correcting
  temporal alignment and the imputation fitting scope; it does not reproduce
  the released default feature matrix unchanged. Exact feature schemas,
  source-availability conventions and inherited-transform definitions remain
  to be frozen. Released RF and partition logic remain the shared model core.
- D24: After considering compute cost, the user approved 12 fixed updated-feature
  recipes across five blocks: A season/IPC history, B trailing levels/volatility,
  C changes/anomalies, D prespecified interactions, E missingness/observation age.
  Include all blocks off, each block alone, all five together, and each of the
  five leave-one-block-out recipes. Keep the original-feature corrected baseline
  separately. This replaces the proposed 32-combination full factorial; it does
  not exhaust two-/three-block combinations. Exact block columns, formulas and
  windows still require freezing. Every recipe retains its own compatible
  development pipeline under D21; only the selected recipe and the reference
  enter final evaluation. This approval remains planning-only.
- D25: The user adopted eight block A features: target-month sin/cos, latest
  eligible observed IPC phase and its original binary crisis label, months
  since that observation, months since the latest eligible observed crisis,
  and flags for no observed history/no previously observed crisis. All history
  must be available at the forecast origin. Missing history remains missing
  with its flags before the common training-fitted imputation. This adds no
  crisis-duration inference or fabricated monthly labels. Source-publication
  conventions and observed phase/label reconciliation remain to be finalized.
- D26: The user adopted block B's fixed 3/6/12-calendar-month windows: means
  and population standard deviations for specified continuous weather,
  vegetation, ENSO, price and nightlight series; cumulative sums for specified
  conflict event/fatality counts. Use only months eligible at the forecast
  origin. A window missing any required month/value remains missing before
  the common training-fitted imputation. No window-size tuning is added. Exact
  source-column whitelist and source-availability endpoints remain to be frozen.
- D27: The user adopted block C's three formulas for whitelisted dynamic source
  series: x(s)-x(s-3), x(s)-x(s-12), and the deviation of x(s) from the preceding
  12-month mean divided by that reference's population standard deviation.
  Here s is the source month eligible at the origin; offsets are calendar months.
  Differences require their two endpoints; standardization requires the current
  value and all 12 preceding values, excluding the current value from the
  reference. Missing required inputs or zero reference standard deviation leave
  the affected feature missing before D23 imputation. Use absolute differences,
  not percentage growth. No additional window or recipe search is authorized;
  exact source whitelists and availability endpoints still require freezing.
- D28: The user adopted exactly four block D products: rain deviation times
  EVI deviation; rain deviation times the three-month conflict-event sum;
  WB food inflation times that conflict sum; and WB food inflation times the
  original market-distance field. Deviations use D27's formula; the conflict
  operand sums battle, explosion and violence event counts over three complete
  eligible calendar months. Compute necessary B/C summaries internally even
  when B/C are off, exposing only the four products. Missing operands yield
  missing products before D23 imputation. No all-pairs expansion or additional
  interaction search is authorized. Exact definitions are in
  research/approved-feature-sources.md.
- D29: The user adopted block E's availability features for a frozen whitelist
  of base source covariates: a pre-imputation missing flag for each field, plus
  months since its latest eligible valid source value for dynamic fields only.
  Check missingness at the scheduled source endpoint; measure age to origin O.
  No eligible history gives a missing flag of one and a missing age, subsequently
  handled by D23. Static fields receive no age. E neither forward-fills sources
  nor duplicates A's history flags or availability flags for B/C/D derivatives.
  Source-value age does not establish original measurement/publication age for
  interpolated or revised data. D53 fixes the source whitelist and column count.
- D30: The user adopted the updated BASE boundary: whitelisted static source
  attributes and origin-aligned dynamic source values, with the approved source
  additions/removals and one explicit schema shared by Stage 1 and Stage 3.
  Do not inherit pipeline-generated calendar encodings, IPC/EVI history lags,
  nightlight cumulative features or automatic secondary horizon-lag copies.
  Add engineered inputs through the approved A-E blocks. The separate original-
  feature corrected reference retains its independently specified inherited
  features; this BASE simplification does not remove them from that reference.
  D52 fixes exact source columns and D54 the reference's inherited transforms.
- D31: The user adopted 22 additional BASE fields: one ENSO anomaly, WB food
  price index and food inflation, one NOAA coastline-distance field, and 18
  Bloomberg series (eight fertiliser, four energy, five staple-crop and one
  soybean-oil last-price series). Exclude the duplicate WB inflation alias and
  the soybean-oil bid quote. Retain the other inventoried Bloomberg series;
  this creates no commodity-subset search or extra recipes. The field list is
  in research/approved-feature-sources.md. Source timing/vintage and some units
  remain unverified; D32 fixes B/C's whitelists and D50 coastline extraction.
- D32: The user adopted B/C's 34-source whitelist: 28 continuous series
  (rainfall, temperature, EVI, GPP, nightlight, original food CPI/inflation,
  ENSO, two WB fields and 18 Bloomberg series), plus six local conflict event/
  fatality counts. B adds 186 columns: three means and three population standard
  deviations per continuous series, and three sums per count series. C adds
  102 columns: all three approved C formulas for each of the 34 series. Do not
  expand w5/w10, inherited z-scores, static attributes or other unlisted fields.
  These limits concern B/C transformations, not the remaining BASE decision.
  B+C together add 288 columns without expanding the 12-recipe search.
- D33: The user chose to retain and document the existing master values instead
  of the proposed source-level reconstruction of Gini and nightlight mean/SD.
  Preserve those values in both feature arms where the fields are included;
  do not reconstruct or drop them solely because of the recorded construction
  concerns. Record the Gini interpolation, ungrouped nightlight-SD filling and
  nightlight zero filling as source-code evidence whose exact incorporation
  into the chosen master remains unverified. D23 still governs all new calendar
  transforms and fitting-only imputation. This experiment enforces downstream
  time cutoffs conditional on the supplied source snapshots, not verified
  end-to-end historical information availability. E describes missingness and
  age visible in supplied values, not recovered raw-observation missingness.
- D34: The user adopted monthly covariate alignment to the end of origin month
  O=T-H, with scheduled source endpoint s_x(O)=O. Take the exact O-month BASE
  value; leave it missing rather than substitute an older value. B/C windows
  end at O and D uses the resulting operands; E may look backwards to calculate
  source-value age without filling BASE. Add no blanket extra month of lag.
  This is a source-month convention for retrospective snapshots, not verified
  release-time availability. Annual/static sources, IPC/persistence eligibility
  and the existing RF-label training cutoff remain separate contracts.
- D35: The user adopted reference year year(O)-1 for GDP, general CPI (annual
  inflation), CC and gini in both feature arms. Keep that reference-year value
  throughout the origin year; missing values remain missing before D23 rather
  than falling back to older years. For E, anchor annual source-value age to
  December of its reference year. Retain supplied Gini values and limitations
  under D33. This retrospective convention does not verify January publication
  of the preceding year's indicators. Food_CPI/Food_food_inflation keep their
  monthly rule; D48 separately fixes population and D49 fixed-layer snapshots.
- D36: The user adopted observed IPC eligibility through origin month O under
  the month-end convention. Persistence keeps the exact-O lookup but requires
  a valid observed phase and its reconciled original binary label. An absent
  record or missing observed phase yields unavailable persistence, not zero
  or an earlier substituted observation. Use identical available-persistence
  labeled-target support for paired comparisons and correction threshold/recipe
  scoring within each horizon, with coverage/exclusions reported. A retains
  its latest-valid-observation lookup at or before O. Master target definitions
  and D20's RF-label training mask are unchanged; publication availability and
  exact phase/key reconciliation remain to be established or qualified.
- D37: The user adopted full eligible-sample crisis-probability RF fitting and
  full eligible 2018 calibration fitting, without filtering either to
  persistence=0 or requiring persistence solely for fitting. On D36's valid
  persistence support, flip only b=0 to one when the raw/calibrated crisis
  probability is strictly greater than that variant's frozen threshold.
  Otherwise retain b, including all b=1 cases; unavailable persistence yields
  no corrected prediction. A valid no-correction selection retains b unchanged.
  This fixes the historical full-sample/up-only structure; D38 fixes threshold
  selection and D39 fixes the reporting inventory. Final success criteria remain open.
- D38: The user adopted the historical threshold-selection algorithm on 2020
  common valid-persistence support. Per recipe and horizon, select separate raw
  and calibrated thresholds shared across months and partitions. Scan observed
  distinct probability values and maximize class-1 F1 from pooled confusion
  counts over February/June/October. Improving ties retain the smallest tau.
  If no candidate strictly improves on persistence, freeze no correction
  (tau=null) and a zero development gain. Freeze before final evaluation with
  no retuning or additional search; this does not set the final success bar.
- D39: The user adopted five reporting rows per final feature arm and horizon:
  persistence, pooled RF, partitioned RF, raw persistence correction and
  calibrated persistence correction. Both correction variants use partitioned
  RF probabilities; standalone RF rows use native classifier predictions without
  extra threshold tuning. Use identical paired keys across all rows and both
  final feature arms; persistence is common and may appear once. Also report
  standalone pooled/partitioned RF metrics on full eligible labeled support
  separately with sample counts. Reuse predictions without adding model fits.
- D40: The user adopted final evaluation through 2024-10 on observed
  February/June/October targets, retaining D14's starts: fs1=2021-06,
  fs2=2021-10 and fs3=2022-02. These are 11/10/9 scheduled target months before
  row-level support checks. Include all eligible area-target observations
  without a balanced-area requirement, and report actual coverage/exclusions.
  Add a supplementary 2022-02 through 2024-10 common-calendar table using stored
  predictions, without replacing the full horizon windows or adding fits.
  Calendar alignment does not imply identical area support across horizons.
- D41: The user adopted the frozen updated-feature winner's equal mean of six
  final-window F1 gains over persistence as the primary adjudication metric:
  three horizons times raw/calibrated correction. Compute each cell from pooled
  confusion counts on its full D40 window and D36 paired support; report all
  six cells without selecting a favorable variant or horizon. Report matched
  gains over the original-feature corrected reference and standalone RF results
  as secondary evidence. A persistence gain alone does not establish incremental
  feature-engineering benefit over that reference. D42 fixes effect-size
  interpretation, D43 fixes robustness conditions and D44 fixes resampling.
  D45 fixes the final research continue/stop interpretation.
- D42: The user lowered the suggested improvement to +0.01 and emphasized
  that, with a strong persistence baseline, any robust positive gain is useful.
  Treat +0.01 as an advisory reference, not a mandatory pass/fail cutoff; the
  earlier proposed +0.02 hard threshold is not adopted. Preserve actual F1
  levels and signed gains in absolute F1 units, including gains below +0.01.
  Neither reject a robust smaller gain solely on magnitude nor declare success
  solely for exceeding +0.01. D43 fixes robustness requirements; the equal
  six-cell primary metric remains unchanged.
- D43: The user adopted two conditions for a robust aggregate benefit: D41's
  mean gain is positive and its paired two-sided 95% interval has lower bound
  strictly above zero; the recomputed mean also remains strictly positive after
  excluding each evaluation target year (2021-2024) in turn. Report all six
  cells without requiring each to be individually significant. Leave-one-year-
  out checks recompute metrics from frozen predictions without refitting.
  Failure means insufficient robust-benefit evidence, not proof of zero effect.
  D44 fixes resampling; D45 fixes the final research continue/stop rule.
- D44: The user adopted a joint target-month block bootstrap with 2,000 valid
  draws and seed 5. Sample the 11-date target union with replacement and share
  date multiplicities across horizons, variants, feature arms and comparators,
  keeping all areas within each date together. Recompute the six-cell mean
  from weighted confusion counts and use linear 2.5/97.5% quantiles. Record and
  redraw attempts lacking an entire required horizon, with at most 20,000
  attempts; insufficient valid draws yield incomplete uncertainty evidence.
  Reuse frozen predictions without refitting. Disclose the conditional
  inference, small-block and remaining temporal-dependence limitations.
- D45: The user adopted the outcome mapping: complete, reproducible evidence
  passing D43 supports further research on the frozen system; complete evidence
  failing any D43 condition ends this bounded feature-engineering cycle with
  insufficient robust-benefit evidence and shifts the next research discussion
  toward expert-unavailable fallback. Incomplete/invalid execution or evidence
  is an unfinished experiment to repair within the approved design, not a
  scientific null. Both complete outcomes can close this experiment after
  review. No post-evaluation feature, threshold or window changes, promotion of
  secondary results to primary, or automatic fallback implementation is allowed.
- D46: The user adopted removal of inherited Rainf_zscore and Tair_zscore,
  including derived copies, from both the corrected original-feature reference
  and all updated-feature recipes. Their master-generation reference windows
  remain unverified; this is not proof that the old values leak. Retain raw
  rainfall and temperature. Updated recipes with C enabled keep D27's specified
  trailing deviations; do not add replacement anomalies to BASE or the reference.
  Disclose this reduction in legacy-feature parity. D33's Gini/nightlight
  retain-and-document decision remains a separate approved source policy.
- D47: The user approved excluding fews_ha and its derived columns from both
  feature arms in every stage, including HA-specific E missingness/age inputs.
  This resolves the released Stage 1/3 inconsistency for the estimated
  assistance-impact phase tag. Preserve source files, original target labels,
  approved phase history and persistence. The experiment deliberately omits
  potentially useful origin-available assistance history rather than adding
  a new HA-history feature family.
- D48: The user adopted a prior-year population snapshot for both feature arms.
  For each origin O, use the area's last valid source-month pop record within
  year(O)-1, preserving the supplied value throughout the origin year. If that
  year has no valid record, retain missing before D23 imputation without older-
  year substitution. Record source dates and within-year discrepancies rather
  than average or round values. E uses the annual missingness convention and
  December source-year age anchor; its older-year age lookup does not fill BASE.
  The sparse source dates/GPW version do not verify demographic reference years
  or historical publication availability. Population remains excluded from B/C.
- D49: The user adopted preserving supplied snapshots of included inherited
  fixed-geography fields in both arms across all development/history/evaluation
  origins, including years earlier than a layer's reference/publication year.
  Disclose the 2015 market-access snapshot, unresolved vintages and documented
  generator/metadata conflicts without reconstructing layers or asserting
  verified physical meanings. This is a retrospective source-snapshot assumption,
  not verified historical availability. Repeated-value conflicts must be surfaced
  before static reduction. market_distance is monthly by inspected construction
  and follows D34; population follows D48. D52 fixes the source whitelist;
  D50 separately fixes new coastline extraction.
- D50: The user adopted sampling the specified coastline raster's containing
  pixel at the master WGS84 lon/lat, preserving its native signed value,
  including positive and zero values. Invalid/out-of-bounds coordinates or
  invalid raster cells yield missing with reasons, preserving master rows.
  Keep one fixed coastline_dist value per validated area coordinate in updated
  BASE only, across dates/recipes; no interpolation, absolute value, unit
  conversion or administrative-area aggregation is introduced. Record raster,
  coordinates, pixel/value provenance and unverified local unit/sign lineage.
  E adds only a static missing flag; B/C do not expand this field. Disclose the
  fixed-snapshot assumption and limitations of point-location representation.
- D51: The user adopted excluding administrative, country and partition IDs
  from model predictors in both arms and all stages, retaining them as metadata
  for joins, within-area validation, spatial mapping/routing and evidence keys.
  Explicitly remove FEWSNET_admin_code rather than inheriting its default RF
  inclusion; do not add replacement identity one-hots. Retain lat/lon and the
  seventeen source AEZ category indicators as geographic predictors under D49.
  Disclose the corrected reference's loss of the legacy admin-ID signal without
  claiming release feature-matrix parity. Spatial keys and routing remain intact.
- D52: The user adopted the complete common source whitelist: 64 retained
  legacy fields, comprising 31 fixed-geography, 28 monthly and five annual
  fields, exactly enumerated in research/approved-feature-sources.md. Retain
  spatial conflict w5/w10 counts and nearest-conflict distance as monthly source
  predictors without adding them to B/C. Updated BASE has 86 source fields
  after the 22 approved additions; the original-feature reference has 67 after
  retaining its three old prices. These widths exclude engineered inputs.
  Preserve inspected header order for retained old fields and D31 order for
  appended additions, with identical ordered source schemas across stages.
  D53 fixes E's expansion and D54 the reference's inherited transforms.
- D53: The user adopted E expansion over all 86 updated-BASE source fields:
  86 missing flags plus ages for 49 monthly and five annual fields, totaling
  140 E columns. The 32 static fields, including lat/lon and source AEZ
  indicators, get missing flags only. Apply the previously approved time and
  imputation rules; do not expand IDs, outcomes/history/calendar or engineered
  inputs. Retain the frozen schema even for window-constant columns. Updated
  recipe widths are 86/94/272/188/90/226/526/518/340/424/522/386 in D24 order;
  the full ABCDE recipe has 526 columns. The reference does not gain block E.
- D54: The user adopted the corrected original-feature reference's full
  109-column schema: 67 source inputs, 21 target-year/month indicators, six
  observed IPC phase/binary lags at O-4/8/12, twelve EVI lags at O-1..O-12,
  WFP price sums over the preceding 4/12 months and a nightlight sum over the
  preceding 12 months, all sums ending at O-1. Use exact calendar endpoints
  and complete windows, leaving missing inputs missing before D23 imputation.
  Calendar indicators use the known target date T. Preserve fixed names/order
  across scopes and stages, without extra automatic horizon shifts, expanding
  early-return helper loops or adding A-E blocks. This deliberately gives the
  reference older origin-relative IPC histories instead of unsafe T-relative
  inputs; it does not recreate the legacy matrix or verify source vintages.
- D55: The user adopted one general consensus map per feature recipe and
  map-role window, pooling all eligible observed target months and fs1/fs2/fs3
  candidates inside that window. Reuse that map for every Stage 3 target month
  and horizon for the same recipe/role, explicitly disabling month_ind. Do not
  build/select monthly alternatives or share maps across recipes/time roles.
  This retains early Jan/Apr/Jul candidate evidence while giving up historical
  seasonal map boundaries. Horizon-by-target-month calibration remains as D16.
  Plan 26 development and two final map builds before support/no-split/failure
  outcomes; the 699 unique Stage 1 candidate-job bound remains unchanged.
- D56: The user adopted the released consensus parameters and automatic
  cluster-count rule: nonnegative clipped-logit F1-gain weights, spatial
  Gaussian sigma=5 degrees, global-maximum normalization, row top-k=40 with
  symmetric union, and largest eigengap selection without Stage 3 tuning.
  k40 is not a forty-partition requirement; a supported automatic nc=1 is a
  valid unsplit result, distinguishable from D19's all-nonpositive weights.
  Retain precomputed-affinity spectral clustering with kmeans assignment and
  random_state=42. Record solver initialization and graph/selection evidence
  for reproduction. D57 fixes selection/fitting support, D58 the node universe
  and D59 geographic completion; final model routing remains to be settled.
- D57: The user adopted selecting nc and fitting spectral clustering on the
  same largest connected component of the positive-affinity similarity graph.
  Select by node count, breaking ties by smallest canonical area code; self
  entries do not connect distinct nodes. Recompute the normalized Laplacian
  from that component's affinity matrix and use its size in the D56 formula.
  A component with fewer than three nodes, invalid graph or solver failure
  stops the affected build as incomplete evidence. Do not reduce nc or invent
  an unsplit result. Preserve other areas for a separate completion/routing
  contract; D19's all-nonpositive branch and valid D56 nc=1 remain distinct.
  This graph connectivity is not polygon geographic contiguity.
- D58: The user adopted the released in-scope graph-node union: include areas
  assigned a valid non-s-1 partition in at least one eligible, normally completed
  candidate for the same recipe/map window, across its approved months/scopes.
  Zero-weight plans retain support records but add no similarity. Missing/s-1
  assignments do not create edges; future evaluation labels or other windows/
  recipes cannot expand learned support. Preserve never-assigned areas and
  nodes outside the selected component for completion/routing without dropping
  master/evaluation rows. D19's nonpositive-weight unsplit branch remains valid
  and distinct from failed/incomplete candidate evidence.
- D59: The user adopted bounded nearest-donor completion for areas outside the
  fitted spectral component. Use only that component's areas as fixed donors,
  preserving their learned labels during completion. For every other master
  area, use validated master WGS84 coordinates, haversine distance <=100 km
  and cross-country matching; exact ties choose the smallest canonical donor
  code. Inherit the donor's partition without chaining through completed areas.
  Beyond-cap or invalid-coordinate recipients remain unassigned and use pooled
  RF, preserving rows and reasons. Freeze completion once per recipe/map role
  without future outcomes. This replaces unrestricted graph-only Euclidean
  completion; graph failures and D19's valid nonpositive-weight branch keep
  their separate contracts. D60 fixes post-consensus refinement and D61 Stage 3
  fitting/fallback; D62 fixes valid single-cluster routing.
- D60: The user adopted freezing the spectral-core assignments plus D59
  completion without additional polygon-neighbor majority-vote smoothing,
  before or after completion. Disable the inherited post-consensus refinement
  route and do not generate competing refined maps. Preserve Stage 1's released
  internal spatial partition operations and D56's spatial affinity. Disclose
  possible geographic islands and the departure from the historical refined-map
  route; similarity connectivity does not establish geographic contiguity.
- D61: The user adopted per-fold RF refitting using the frozen completed map
  and all real eligible history under D20. Completed areas' eligible rows enter
  their assigned partition's fit; unassigned areas' history still enters pooled.
  Fit D23 max_plus once on the full recipe/horizon/fold training pool and share
  it across pooled/local models, without target-month area/group filtering.
  Preserve local eligibility of >=50 real rows and both binary classes.
  Unassigned, insufficient-support, single-class and unseen partitions use
  pooled for both native hard decisions and class-1 probabilities, recording
  actual routes/reasons. Empty global training or fitting failures stop the
  affected fold. Retain the origin-exclusive mask and native RF classification;
  do not import IPCCH's inclusive-origin window, XGBoost or thresholded hard
  decisions. D62 separately fixes supported nc=1 routing.
- D62: The user adopted full-pool prediction reuse for every row when a valid
  positive-weight consensus build selects automatic nc=1, including when some
  areas remain unassigned. Reuse native hard predictions and probabilities;
  omit an assigned-only single-cluster fit. Preserve geographic assignments,
  unresolved status, donors and coverage, recording automatic nc=1 separately
  from D19's all-nonpositive unsplit reason. Apply throughout development and
  final folds without changing correction/calibration/threshold contracts.
  Multi-cluster maps retain D61 even when only one local model is supported in
  a fold; graph failures cannot invoke the valid-unsplit route.
- D63: The user adopted retaining the original master binary with valid
  unadjusted observed phases restricted to integers 1..5: phases 1/2 imply
  zero and 3/4/5 imply one; missing phase remains unavailable. Validate unique
  canonical area-month keys, ledger date consistency and master/ledger phase
  and binary agreement. Exclude only the verified terminal System.IO.MemoryStream
  artifact from the parsed ledger view with hash/line/content/reason evidence,
  preserving raw files. Other malformed records/keys, duplicates, nonempty
  invalid labels or source conflicts stop preflight without silent repair.
  Ordinary missing observations and the full master grid remain preserved.
- D64: The user adopted deterministic WB nearest-market ties: restore raw
  geo_id as metadata through verified raw-to-derived row lineage, retaining
  the approved derived values. Within D9's same-month, <=100 km cross-country
  rule, exact computed haversine-distance ties choose ascending lexical geo_id.
  Copy both approved fields from that market, retaining missing values without
  farther-market substitution, averaging or value/quality-based selection.
  Record market/distance/tie evidence without using IDs as predictors; source
  lineage failures or conflicting duplicate market-month keys stop assembly.
