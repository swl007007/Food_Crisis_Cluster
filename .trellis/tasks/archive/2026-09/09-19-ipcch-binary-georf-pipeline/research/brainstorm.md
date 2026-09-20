# Brainstorm: smallest adaptation that answers the user question

## Objective
Determine how the corrected GeoRF spatial pipeline performs on an IPCCH population-share binary target against pooled RF, pooled binary XGBoost and persistence. The outcome is a reproducible, paired comparison; no model ranking is presumed.

## Evidence-driven direction
1. Start from the pinned GeoRFBaseline source release; use an isolated IPCCH experiment directory. Preserve the mathematical F1/q/partition and RF learning core. User superseded consensus; skip Stage2.
2. Replace the FEWS-specific data boundary: explicit target generation, canonical admin/date keys, feature schema, as-of feature/history construction and train-fitted missing-value handling. Keep core array interfaces X/y/group/split and the correspondence/cluster-map shapes.
3. Pool Stage1 historical samples in one partition-learning procedure with broader internal validation. Directly export branches and reuse geometry/1-NN for unassigned regions within the approved100km donor-distance cap; over-limit areas remain unassigned and receive pooled-RF predictions. FEWSNET partitions and hardcoded0..5717 are unsuitable for6,227 IPCCH IDs.
4. Use one fold/key manifest and the same outcome definition for all four arms. Each learned model gets the same eligible predictor information and training support; differences in missing-value representation must be explicit. Persistence uses the new target's observed history at the same forecast origin.
5. Keep the existing Stage3 pooled fallback and add only the two required benchmarks. No Stage2 consensus, expert correction, GeoDT or new ensemble.

## Code surface established by read-only exploration
- `GeoRFBaseline/src/preprocess/preprocess.py:179-205,266-280`: old target filter and history are FEWS-specific, and filter-before-shift is unsafe for sparse IPCCH observations.
- `GeoRFBaseline/src/feature/feature.py:55-113,124-131`: implicit all-column feature selection, full-panel dynamic/static detection and imputation must be replaced at the new data boundary. Current target components must not leak into predictors.
- `GeoRFBaseline/src/model/GeoRF.py:219-256`: explicit split interface already exists; core need not be rewritten to supply a declared validation split.
- `GeoRFBaseline/scripts/step3_create_linked_tables.py:76-122`: replace hardcoded FEWSNET ID universe with explicit IPCCH IDs while retaining table and clustering logic.
- `GeoRFBaseline/scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py:80-95,220-267`: RF=100 trees, unlimited depth, seed5, n_jobs1; local <50/single-class fallback is reusable.
- Same script `:991-1017`: pooled and partitioned training filters differ in inherited code. A new comparison must explicitly equalize eligible training support before judging spatial partitioning.
- `GeoRFBaseline/src/customize/customize.py:412-420`: nominal 36-month window currently yields `[T-H-35 months,T-H)`; decide deliberately whether to preserve or correct this boundary.
- `GeoRFBaseline/src/utils/lag_schedules.py`: existing scopes fs0=1, fs1=4, fs2=8, fs3=12 months. Do not substitute sibling IPCCH's 0/3/6 meanings by name.
- `EthiopiaForecastingExperiment/run_binary_xgb_comparison.py:78-92`: reusable binary XGB constructor; Ethiopia weighting/tuning policy is not inherited automatically.
- `PersistenceCorrectionExperiment/persistencecorrection/persistence.py:145-167,221-294`: exact calendar-origin persistence helper with key/coverage checks. Latest-observed-as-of persistence needs different lookup, not a record shift.

## Proposed design choices to grill, not approvals
- Label QC is now resolved by user revision: only missing P5 may be zero-filled; raw sum [.90,1.10], observed components [0,1], population>0; normalize all phases before strict >.20 target construction. See PRD R1 and approved_label_policy_audit.json.
- Calendar-safe feature timing; latest valid observed new-target history by forecast-origin cutoff is a candidate for persistence in this sparse panel.
- Preserve core partition mathematics; decide temporal outer periods, inner validation and train-window semantics explicitly.
- Start with fixed model hyperparameters and no extra calibration/threshold arms; decide XGB fairness and tuning before seeing test scores.
- Share evaluation keys, report class1 F1/precision/recall and confusion counts plus coverage/prevalence by horizon/year/country. Any uncertainty or improvement criterion must be fixed before final comparison.

## Required artifact flow after approval
Source/feature/geometry audit -> frozen target+feature+split contracts -> single pooled Stage1 -> existing coordinate1-NN completion -> four-arm Stage3 predictions -> paired metrics and lineage manifest. Raw data and source baseline remain unchanged. No implementation or model execution is authorized by this draft.

## Horizon decision and partition-time audit — 2026-09-19
The Stage2 findings below document the former architecture; current design skips that stage. They do not authorize retaining consensus.
- User approved IPCCH horizons of exactly 1/3/6/12 calendar months, superseding the proposed old schedule; IPCCH need not align with expert forecasts.
- `GeoRFBaseline/config.py:40-67` and `src/utils/lag_schedules.py:24-41` reject 3/6 as legacy values and require exactly 4/8/12. `scripts/run_stage1.py:36` and `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py:817-819` restrict scope indices to 0..3. Scope0 is independently mapped to1 month (`src/utils/lag_schedules.py:45-55`). These are explicit IPCCH adaptation boundaries; changing only the tuple is insufficient. Actual-month metadata must disambiguate artifact identity; no frozen-package changes are authorized.
- Stage2 does not inherently require three horizons: `scripts/step1_merge_results.py:88-100` discovers result files; `scripts/step4_similarity_matrix.py:141-159` iterates candidate plans. Its optional 2/6/10 filter is target month, not horizon.
- Stage1 year/month identifies the target (`scripts/run_stage1.py:43-49`). Fitting receives only the pre-origin split (`app/main_model_GF.py:425-440,546-555`; `src/customize/customize.py:412-430`), including its internal validation.
- However, `app/main_model_GF.py:609-618,738-752` evaluates/saves target-month `f1(1)` and `f1_base(1)`; `scripts/step1_merge_results.py:172-189` matches those scores to partitions, and `scripts/step3_create_linked_tables.py:49-65` propagates them into the index. Stage2 `scripts/step4_similarity_matrix.py:56-60,141-156` uses clipped nonnegative logit-F1 differences to weight/include candidates. Even zero-weight exclusion depends on target labels.
- `scripts/step4_similarity_matrix.py:225-241` consumes the complete index (optional target-month filter only), so horizons share candidate information when placed in one experiment directory. No automatic forecast-origin cutoff is enforced here.
- Consequence: learning target months through2022 may bring 2022-end labels into the frozen consensus. Main evaluation origins in2022 cannot assume that consensus already exists. Q3b proposes origin>=2023-01 while retaining approved learning years; the associated loss of early evaluation target months needs user approval. Publication-date uncertainty remains separately unresolved; this is an observation-month design.

## Approved single pooled Stage1 and utility reuse
- User requires one partition-learning procedure across pooled time samples and broader INTERNAL validation; Stage3 test-period expansion was explicitly not intended. No annual/monthly partition ensemble or Stage2 consensus.
- `GeoRFBaseline/src/model/GeoRF.py:122-154,219-240` accepts pooled arrays and an explicit split. It does not itself enforce dates. Prefer a complete validated `split['X_set']`: the index-form train list only checks overlap and all unspecified rows otherwise default to training. A single fit still trains root/candidate child RFs internally.
- Validation-only areas are accepted; `src/partition/transformation.py:671-708` instead requires each child to have nonempty fitting and validation subsets. Default `src/utils/split.py:63-72` forces singleton areas into training and reserves fitting rows per area, so broad validation needs an explicit split decision.
- `src/helper/helper.py:279-295` and `src/merge/terminal.py:36-105` can export area-to-branch lineage without spectral labels. Audit root/ancestor defaults, duplicate conflicts and actual validation support before deciding which areas are learned seeds. `GeoRF.py:436-471` saves `X_branch_id.npy` before optional global refinement; use a consistent final-map source.
- The actual existing Stage2 completion is `GeoRFBaseline/scripts/step6_complete_clustering_pipeline.py:192-202` (same in root `scripts/`): `KNeighborsClassifier(n_neighbors=1, metric='euclidean')`, fit known coordinates/labels, predict only missing/outlier labels. This is embedded in `main()`, not a separate exported helper. Reuse these few lines at the IPCCH boundary; do not run its spectral-clustering CLI.
- `src/adjacency/adjacency_utils.py:18-38,66-78,93-108` provides adjacency, ID->array index and centroid coordinates; validate the requested ID column because missing columns can silently fall back to row indices. The centroid operation does not reproject. Existing1-NN uses raw-coordinate Euclidean distance, no country/max-distance barrier and no custom tie rule.
- Unlike graph-only propagation, coordinate1-NN assigns islands from available coordinate seeds. Include the complete IPCCH universe: Stage2's old matrix excludes IDs never assigned (`scripts/step4_similarity_matrix.py:79-105`) and its earlier steps hardcode FEWS IDs/coordinate paths.
- Earlier statement that no completion utility existed was too broad: the initial search of `src` found only refinement; the active shared Stage2 script contains the reusable1-NN completion. The user-directed reuse is now authoritative.
- Q5h approved: pool1/3/6/12 horizon views with explicit horizon metadata in one fit and share the resulting partition. Split original area-month units first; all four views stay together. Report independent outcome support separately from expanded model rows.
- Q5s user revision supersedes the independent singleton-validation proposal: predict a singleton area using its geographically nearest eligible donor's model and put it in that donor's partition. Its own held-out outcome cannot fit the model used for that prediction. Q5m is approved: reuse the donor partition's RF with the recipient's own as-of features; no additional area-only RF or copying donor features/predictions. Q5v is now approved: singleton errors are supplementary post-map diagnostics only, excluded from Stage1 fitting/q/split F1. Existing Stage3 per-partition fitting was rechecked at `GeoRFBaseline/scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py:231-267`.

## Required donor-distance cap — 2026-09-19
- Q8r approved: maximum donor distance100km inclusive, measured as great-circle distance between declared reference coordinates. Apply it to singleton and other unassigned areas; preserve unresolved IDs and never bypass the limit through inferred intermediate areas. Beyond the cap, leave the partition unassigned and use existing pooled-RF prediction, recording fallback provenance.
- `research/nearest_seed_distance_audit.json` profiles existing keyed reference coordinates with haversine distance (Earth radius6371km). Candidate donors have >=2 original valid labels, not verified learned partitions; windows are illustrative, not approved learning periods. No fitting or assignment was performed.
- For2017–2022,1,472singleton areas have nearest-candidate coverage48.23%/73.30%/89.27% at50/100/200km. The1,491zero-label areas have86.38% coverage at100km. For2017–2023, singleton/zero-label coverage at100km is85.37%/86.26%; this does not authorize extending the learning cutoff into the reference test period.
- Coordinates are not polygon-edge distances or proof of valid geometry; final eligible donor coverage may differ. No country-only restriction was requested. Q5m donor-model meaning and Q5v supplementary-only singleton scoring are approved.

## Singleton validation-statistics trace — 2026-09-19
- `GeoRFBaseline/src/partition/transformation.py:274-301,326` selects branch validation rows, predicts their own features and passes their labels/predictions/`X_group` into group statistics and `scan()`. `src/partition/partition_opt.py:134-146` aggregates D=2TP+FP+FN and A=2TP by the supplied group identity. These observations affect learning, not just reporting.
- `partition_opt.py:220-225,970-1002` computes per-group scan contributions and a q estimate for the selected group set. A singleton with its own area ID is an independent candidate spatial unit; there is no separate final q parameter per area. `src/helper/helper.py:179,198-208` maps candidate groups back to IDs and uses group membership for train/validation extraction. The checked path has no nearest-donor binding mechanism.
- `transformation.py:671-720` and `partition_opt.py:872-890` evaluate child choices using aggregate confusion counts on concatenated validation rows, not an average of per-area F1. Adding singleton rows can change this gate too. Expanded horizon rows are each processed by the core; the IPCCH adapter must separately enforce the approved original-outcome support accounting.
- `GeoRF.py:1000-1024` prediction uses fitted branches and input features without target labels. Scoring held-out singleton outcomes only after the learned map is fixed and completed therefore keeps them out of the q/split paths above. This remains development/transfer diagnostics, not independent Stage3 final-test evidence.
- Q5v approved on2026-09-20: use singleton outcomes for separate post-map supplementary diagnostics, with no Stage1 fitting, q/split F1 contribution or feedback to partition selection. This does not expand the actual internal validation support used to learn partitions. Singleton status is within the approved Stage1 label pool, before horizon expansion; Stage3 rolling eligibility remains unchanged.

## Approved pooled period — 2026-09-20
- Q2r approved: pool valid target months2014-01..2022-12 inclusive, retaining CH observations from2014 and IPC from2017. This supersedes the old2018–2022 learning range. Q5a now specifies the approved internal split. The existing exact-label audits report1,507labels in2014–2016 and18,084in2017–2022, totaling19,591candidate original outcomes before singleton diagnostics and feature eligibility are separated.
- Evidence: `stage1_support_audit.json` (`annual_2014_2017`, `stage3_remaining`) and `validation_support_by_source.json` (`sources`, `fixed_history`). These are existing planning audits, not a new raw-source scan or fitted-model result. Reusing the2022-12 cutoff retains h12 main target years2024–2025 (14,087raw valid target rows); extending through2023 leaves2025 only (9,792). All counts precede feature/history exclusions and do not establish viable partitions.
- The earlier start retains available labeled history but broadens the time span of CH samples; it creates no pre2017 IPC observations. The later Q5a approval separately chooses the chronological within-area split; neither approval authorizes implementation. Stage3's first origin2023-01 and horizon-specific main target schedule stay unchanged.

## Internal split capacity and approved rule — 2026-09-20
- New read-only source audit in `pooled_split_support_2014_2022.json` matches19,591labels/5,999positives in the approved window, without source-hash recomputation. The6,227area universe has3,264areas with n>=2,1,472singletons and1,491with no label. Only18,119original rows belong to the n>=2candidate learning pool before feature checks.
- Existing `GeoRFBaseline/src/utils/split.py:63-83` selects random rows within an area with n_val=min(n-1,max(1,ceil(r*n))); applying r=.2 yields12,985fit/5,134validation rows and1,204areas with>=2validation rows. A nominal50percent quota gives8,561fit/9,558validation rows and2,287areas with>=2validation rows. Both support3,264areas with at least one row on each side; increased quota adds observations, not regions.
- Random and chronological suffix selection with the same quota have identical counts. Q5a approved on2026-09-20: earlier floor(n/2) outcomes train, latest ceil(n/2) validate within each n>=2area, before horizon expansion. Odd counts put the extra original outcome into validation. This provides chronological ordering within each area while preserving broad geographic capacity; it uses less fitting data. Area cutoffs differ, so internal F1 remains a development score and cannot establish globally forward prediction performance. Stage3 keeps its separate future-evaluation schedule.
- Illustrative common cutoff2014–2020fit/2021–2022validate on that same eligible pool has11,365/6,754rows,1,887areas on both sides,706fit-only and671validation-only. This is not approved; it illustrates the difference from per-area quotas, not a measured performance comparison. No actual split membership, feature panel, model or new partition has been produced.

## Approved donor provenance — 2026-09-20
- `GeoRFBaseline/src/merge/terminal.py:76-90` maps empty branch labels to root and retains the first assignment while recording conflicts. Its`:103-106` also converts invalid labels to root. A nonempty exported mapping alone therefore cannot establish that an area participated in partition learning; inspect actual membership and collisions.
- Q5d approved: >=1original fitting outcome plus >=1original validation outcome under Q5a, valid coordinates and a unique explicit final learned assignment. Retained root/ancestor branches can qualify when their learning participation is real; default-root placeholders and inferred assignments cannot. Add no extra per-area class/F1/count threshold; branch-level core gates stay unchanged. This resolves donor provenance rather than adding a new clustering procedure.

## Approved persistence lookup and paired subsample — 2026-09-20
- Rechecked `PersistenceCorrectionExperiment/persistencecorrection/persistence.py:145-166,221-294`: the existing helper re-keys observations at T=O+H and requires an exact-origin match with100percent coverage; it is not latest-observed-as-of lookup. Its loader at`:178-207` also uses the FEWSNET phase>=3 target and raw-missing-to-zero convention, incompatible with IPCCH's approved R1 target. Reuse key/provenance validation patterns, not these target or lookup semantics.
- Q4 approved: predict the latest valid same-area normalized IPCCH binary label with source month<=O, including O. No extra maximum history age; Stage3's36-month fitting window is not a history-lookup limit. When no valid history exists, prediction stays missing. Save source month and calendar-month age relative to O. This uses sparse observed history but can carry an old label for a long interval.
- Q4b user clarification: persistence baseline uses only observations with a valid history-based prediction. Freeze E_all (valid test truth under the approved schedule) and E_persist (its history-available subset) per horizon. Four-arm paired comparisons use E_persist; three-learned-arm comparisons also retain E_all. Reuse predictions, not subsample-specific refits. Export paired keys, counts/prevalence and coverage; retain rows without history and mark persistence missing. A subset persistence score cannot be paired with a full-sample learned score.
- Q6a approves history-feature inclusion; Q6b freezes secondary raw fields and Q6c now fixes preprocessing. No future-label lookup, missing-truth imputation or hidden change to model-training support is authorized.

## Approved historical predictors — 2026-09-20
- Rechecked the minimal baseline directly: `GeoRFBaseline/src/preprocess/preprocess.py:265-273` constructs both binary and ordinal outcome history with within-area row shifts for every legacy lag; `GeoRFBaseline/config.py:95-102` enables this family by default. IPCCH's approved calendar/origin contract requires a new history lookup, not renaming these columns.
- `GeoRFBaseline/src/feature/feature.py:58-65,92-100` infers dynamic columns on the full supplied panel, adds scope shifts and initially retains original columns. Its`:124-131` imputes the full matrix before downstream splitting. These behaviors do not establish horizon safety or train-only preprocessing; do not inherit them in the IPCCH adapter.
- Q6a approved: learned partitioned RF, pooled RF and XGB use identical latest valid binary history, calendar-month age and missing-history indicator per row origin. Raw missing history label/age remain NaN until the approved model-specific treatment. Replace legacy binary/ordinal outcome lags; do not add historical overall_phase or phase shares. This exposes whether the observed new-target history is stale while dropping ordinal severity detail.
- Source header confirms outcome columns overall_phase/phase1..5_percent/estimated_population and renamed secondary fields EVI_mean/GPP_mean/nightlight_mean/nightlight_std. Name similarity alone does not verify source definitions. Q6b fixes the whitelist and Q6c fixes preprocessing. Broad agent findings about root-entrypoint configuration and debug CSVs were not treated as verified minimal-baseline behavior; future tracing must pin the GeoRFBaseline path.

## Secondary whitelist evidence and approval — 2026-09-20
- `secondary-predictors.md` records newly traced GeoRFBaseline-only behavior and selected input-header/codebook evidence. The baseline uses dynamic column passthrough; Stage1 and Stage3 post-drop differ. Q6b is now APPROVED:70raw fields, the supplied AEZ family and coordinates, area ID metadata-only, without expansion to every IPCCH column. Q6d/Q6e/Q6f fix temporal/calendar/crisis-recency inputs; Q6c now fixes preprocessing.
- Alias evidence exists for EVI/nightlight fields; GPP's same-family mapping is supported only at the concept/codebook level. Codebook resolves CC as corruption-control percentile and conflict _w5/_w10 as spatial neighbor aggregates. No full-panel new-feature inference was performed.
- Follow-up upstream traces found a concrete sum-to-nightlight_mean naming path and ungrouped ffill in a notebook that writes IPCCH_2025_secondary.csv. Its actual link to selected IPCCH_2026_completed.csv remains unverified. The found completed-file organizer moves the file; it does not generate it. Detailed paths, evidence and limits are in secondary-predictors.md. No conclusion of actual selected-source contamination or cleanliness, and no upstream repair/substitution, follows from this audit.
- User additionally requested freezing the current completed-file feature set while leaving future improvements such as WB RTP for a later feature version. This approval pins the70-field whitelist and existing source snapshot; it does not authorize new acquisition or silently changing the baseline when the source later changes.
- Direct recheck of `GeoRFBaseline/src/preprocess/preprocess.py:625-627,647-648,666-670` confirms SUM operations for WFP_Price4/12 and nightlight12, plus12 EVI row lags. Q6d APPROVED on2026-09-20: preserve those15 derivative types with per-area calendar windows at each row's origin, no auto-expanded scope-lag block and explicit missing-month behavior. Exact approved windows and names are in secondary-predictors.md.
- Calendar trace: `GeoRFBaseline/src/preprocess/preprocess.py:282-286` creates target-year/month dummies from available panel values; `config.py:247-251` and `src/feature/feature.py:99-100` do not remove the dummies. Q6e APPROVED WITH USER REVISION on2026-09-20: target month uses sin/cos, superseding the proposed12fixed indicators. Years remain metadata and the approved horizon indicator is retained. Exact formula is in PRD R3 and secondary-predictors.md.
- Q6f APPROVED on2026-09-20: recency relative to each row's origin since the latest observed valid positive, with NaN plus a no-prior-observed-crisis indicator if absent. Keep Q6a's different latest-observation-age feature. Sparse observations do not identify episode onset/end or continuous crisis-free durations. Exact approved semantics/example are in secondary-predictors.md.
- Two read-only explorations established imputation and RF fit boundaries. Direct recheck of `GeoRFBaseline/src/customize/customize.py:60-188` verifies max_plus100 is100*max (zero-max=>100, all-missing=>0), existing fit/transform reuse and the lack of a universal out-of-range guarantee. `src/model/model_RF.py:244-247` confirms pseudo-row insertion before estimator fit. Q6c APPROVED on2026-09-20: training-only shared RF statistics and native XGB NaN; exact approved rules/evidence are in secondary-predictors.md. No data transform/model execution occurred.

## Threshold policy and XGB reference — 2026-09-20

Q7a APPROVED on2026-09-20: use p(class1)>0.5 for every learned Stage3 arm,
including pooled binary XGB; exactly0.5 gives0. Keep Stage1's existing RF
classification path. Do not select thresholds on validation or test scores or
add threshold variants. This preserves the released RF decision policy and a
common threshold across arms; it does not promise F1-optimal classification or
probability calibration. Persistence stays its history-based binary decision.

Evidence, baseline paths relative to GeoRFBaseline:
- README:115-124 prescribes original pooled/partitioned RF and explicitly excludes
  validation-threshold/expert-correction arms. Comparison script
  `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py:827-830` has both threshold
  flags default false;`:270-272,300-326,1156-1166` uses classifier.predict normally.
- Stage1 `src/model/GeoRF.py:402`, `src/model/train_branch.py:15-49` and
  `src/model/model_RF.py:114-122,289-292,404-420` use sklearn RF predict for root,
  children and final branches. With classes[0,1], sklearn argmax tie semantics
  imply p1>0.5; this trace did not execute a fitted-model tie experiment.
- Optional threshold path, not active baseline: comparison`:75-78,403-489,522-547`
  uses the last6calendar months for validation, rounded observed probability
  candidates in[.05,.95], class1-F1 maximization, highest-threshold tie break,
  and fallback.5 on empty/no-positive/no-candidate cases. Decisions use >=, so its
  fallback.5 differs at ties from default RF predict. No additional class/count
  minimum beyond nonempty splits/at least one validation positive was found.
- Optional thresholds are per target/scope/model, not per partition
  (`:965,1059-1123`); fit-only models select them before full-window refitting
  (`:1084-1094,1150-1155`). Approved Q7a leaves these paths disabled.

XGB reference paths are outside GeoRFBaseline, in EthiopiaForecastingExperiment:
- `run_binary_xgb_comparison.py:78-92` sets binary:logistic/logloss/hist,
  learning_rate.05, subsample.8, colsample_bytree.8, seed5 and n_jobs1, but requires
  max_depth/min_child_weight/n_estimators arguments. It is not a fully fixed baseline.
- `:45,132-139,344-350,375-383` searches8 combinations of depth3/6, child weight1/5,
  trees200/400, orders by validation F1 then smaller depth/higher child weight/fewer
  trees, and refits the winner. `:56-75,340-350,375-383` additionally passes normalized
  square-root inverse-frequency sample weights. No early stopping/eval_set is used.
- `run_stage3_aligned.py:195-224` selects thresholds; binary script`:351-352,384-386`
  applies probability>=threshold. Constructor reuse does not authorize this tuning,
  weighting, temporal/sampling policy or FEWS comparisons.
- `outputs/local_partition_experiment/eth_binary_xgb_20260904_seed5_v2/run_metadata.json:7-9`
  records Windows Python3.12.10/XGBoost3.0.0, not current installed-version verification.
  Root requirements/environment specify only xgboost>=2.0.0; the frozen minimal
  package's requirements.txt has no XGBoost entry. Approved Q7b below fixes the
  IPCCH parameters/weights and runtime version explicitly.

## Q7b approved fixed XGB configuration — 2026-09-20

Predeclare one existing capacity combination (depth6, child weight5,400trees)
without running the old eight-candidate search or reading IPCCH validation/test
scores. Retain the constructor's learning rate and stochastic sampling settings;
make regularization/initial intercept/class weighting explicit:

| Parameter | Approved value |
|---|---|
| objective / booster / tree_method / device | binary:logistic / gbtree / hist / cpu |
| n_estimators / max_depth / min_child_weight | 400 / 6 / 5 |
| learning_rate | 0.05 |
| subsample / colsample_bytree | 0.8 / 0.8 |
| reg_alpha / reg_lambda / gamma / max_delta_step | 0 / 1 / 0 / 0 |
| scale_pos_weight / base_score | 1 / 0.5 |
| random_state / n_jobs | 5 / 1 |
| max_bin / grow_policy / num_parallel_tree | 256 / depthwise / 1 |
| eval_metric / missing | logloss / NaN |

Every original fitting row has unit weight: do not pass Ethiopia's square-root
inverse-frequency sample weights. No SMOTE, pseudo rows, class rebalancing,
hyperparameter search, eval_set/early stopping, probability calibration or changed
Q7a threshold. Fit anew for each Stage3 origin/horizon on the approved common
training pool. Fixed depth/child weight regularize capacity but do not establish
optimal XGB performance; this is a reproducible fixed-configuration comparison.
Pin XGBoost3.0.0 and record complete effective parameters/booster configuration
at execution; omitted library settings follow that pinned version. RF retains
the released parameters and approved Stage1-only pseudo-row distinction.

### Current runtime metadata probe — 2026-09-20

Read-only execution of WindowsApps/python3.12.exe using sys and importlib.metadata
reported Python3.12.10, XGBoost3.0.0, scikit-learn1.6.1, NumPy2.2.6 and pandas2.2.3.
Resolved executable:
`C:\Users\swl00\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe`.
This is live installed distribution metadata, unlike the historical Ethiopia run
record. No XGB module import, estimator construction, fitting, installation or
source data transformation was performed. Confirm runtime and import compatibility
again before approved execution; preserve the frozen release/environment files.

## Evaluation design evidence — 2026-09-20

Q7b and Q9a/Q9b evaluation are approved. Two independent
read-only traces found legacy metric edge cases and reusable country-resampling
logic. Approved Q9a/Q9b rules are in research/evaluation.md. No metric
was computed on IPCCH predictions, no bootstrap/model was run, and no old output
or frozen package was modified.
