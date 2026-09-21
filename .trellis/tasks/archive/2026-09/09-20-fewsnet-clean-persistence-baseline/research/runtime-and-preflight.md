# Released settings and source consistency — 2026-09-20

Read-only evidence from two bounded scouts; no modeling or source edits.
The main session inspected the GeoRF explicit-split path, imputer and F1
configuration. These settings implement release reuse, not another search.
The tested-environment manifest is historical release evidence, not a claim
that the current executable has already passed experiment preflight.

## Static and annual fields

A full selected-column scan of the 1,029,240-row master found 5,718 areas
and 85,770 area-years, 2010-01..2024-12. Numeric comparisons used exact parsed
equality, without averaging, rounding or tolerances. SHA identity is the one
verified in target-label-contract.md; this scout did not hash it again.

- All 14 numeric static columns (lat/lon and the 12 non-AEZ geographic layers
  listed in D52) are complete, finite and identical within each area. No
  invalid-coordinate rows or changing coordinate pairs were found.
- Each of 17 AEZ columns contains only lowercase true/false strings, complete
  and constant within area. Every row has exactly one true. These are boolean
  source indicators, not unparseable numeric missing values. Explicitly encode
  true/false as 1/0 before numeric preprocessing and preserve all 17 columns.
- GDP/CPI/CC/gini have no conflicting nonmissing values or mixed missing/valid
  copies within any area-year. Wholly missing area-years are respectively
  8,854 / 15,178 / 5,718 / 44,280; missing rows are 106,248 / 182,136 / 68,616 /
  531,360. No nonnumeric/nonfinite entries were found in those numeric fields.
- Population was not rescanned: its separately approved D48 last-valid-month
  rule remains unchanged. Constancy of other fields does not prove vintage
  or physical meaning; D33/D49 source limitations still apply.

Example: master physical line 2 has area 0, 2010-01, AEZ_32000=true,
AEZ_10000=false, lat=9.551001515338209, lon=29.13029621711373,
GDP empty, CPI=1.169621803, CC=21.4285717, gini=0.459952778081.

## Release identity

GeoRFBaseline/MANIFEST.json:2-4 identifies 0.1.0-f1-nosmote and source commit
2dfa121a9398de9a1918ba9c0af34b31ecbb117a. The scout checked every listed payload
hash: zero mismatches. SOURCE_PROVENANCE.json records original copied-source
hashes separately from release payload hashes; do not interchange them.
releases/SHA256SUMS:1 and the actual ZIP agree on:

    39a26138e3fafb0be2bbd22e9760095d6cdefa7d79b98b4f798cb3aa79b500a0

TESTED_ENVIRONMENT.json records Windows Python 3.12.10, numpy 2.2.6,
pandas 2.2.3, scipy 1.15.2 and scikit-learn 1.6.1; requirements.txt pins the
other packages. It explicitly describes a tested environment, not a historical
environment reconstruction. Verify actual interpreter/packages before fitting;
do not silently use a Linux probe environment for the formal experiment.

## Effective model settings

Paths below are within GeoRFBaseline unless prefixed otherwise.

| Setting | Effective release behavior | Evidence |
|---|---|---|
| Stage 1 validation | Within-area random split, ratio .20, seed 42; min one validation row for n>=2, singleton train-only, no label stratification | config.py:235-244; src/utils/split.py:43-78; src/model/GeoRF.py:241-251 |
| Stage 1 RF | 100 trees, unlimited depth, seed 5, class_weight=None, SMOTE off | src/model/GeoRF.py:50-54,401; src/model/model_RF.py:46-54,252,289-292; app/main_model_GF.py:1267 |
| Stage 3 RF | Same trees/depth/seed; n_jobs=1; pooled and local share factory | scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py:80-108 |
| Other RF parameters | Omitted in source; bind to locked sklearn defaults and record actual get_params | Same constructors; verify in target environment before fitting |
| Stage 1 parallelism | RAM-sensitive n_jobs; fit can reduce threads or retry single-threaded | config.py:178-207; src/model/model_RF.py:265-292,317-320 |
| Partition depth | MIN_DEPTH=1, MAX_DEPTH=6; five candidate-split rounds in range(max_depth-1) | config.py:174-175; src/partition/transformation.py:251-265 |
| Flexible group search | FLEX_OPTION=True, FLEX_RATIO=.1, FLEX_TYPE=n_group; relevant minimum support knobs zero | config.py:214-222; src/partition/partition_opt.py:250-277,842-861 |
| q optimization | Deterministic c/b initialization, 1000 coordinate-descent iterations; random initialization/early-stop code commented | src/partition/partition_opt.py:960-1026 |
| F1 split gate | class_1_f1; strictly >.01 gain, parent wins ties; Fraction boundary arithmetic | config.py:342-346; src/partition/partition_opt.py:867-891 |
| Branch support | Positive validation-class support and nonempty child fitting/validation arrays in active F1 path | src/partition/transformation.py:310-314,706-708 |
| Inactive significance knobs | SIGLVL=.1, ES_THRD=.5, MD_THRD=.0005 exist, but F1 route bypasses sig_test | config.py:225-227; src/partition/transformation.py:715-802 |
| Stage 1 spatial operations | CONTIGUITY=True, MIN_COMPONENT_SIZE=5, polygon adjacency; retained under D60 | config.py:312-325; src/model/GeoRF.py:453-456 |

Stage 1 process Python/NumPy seeds are 42; Stage 3 process seeds are 5.
RF seed remains 5 independently of the Stage 1 process seed. GeoRF.fit only
passes trees/depth into RFmodel, not outer random_state/n_jobs (:401).
The old app calls np.random.seed() without an argument during cleanup
(app/main_model_GF.py:1050); keep one candidate job per isolated process,
initialize its seeds explicitly and record actual effective settings rather
than assuming the outer constructor controls every random source.

GeoRF.fit accepts explicit X_set/split (src/model/GeoRF.py:219-240). This
allows the experiment to determine the released within-area split before
fitting the imputer, then pass the same split into the existing core; no
second split or new partition algorithm is necessary.

Stage 1 FEWS geometry is explicit GEORF_POLYGONS with admin_code
(config.py:326-328; scripts/run_stage1.py:47-54). Existing FEWS source location
is DATA/Outcome/FEWSNET_IPC/FEWS NET Admin Boundaries/FEWS_Admin_LZ_v3.shp
(src/adjacency/adjacency_utils.py:153-155). Bind the full shapefile sidecar
identity and needed ID coverage in preflight; use run-local adjacency/cache
outputs, not an unvalidated IPCCH cache or a path-only cached identity.

## Retained month-pooled calibration defaults

PersistenceCorrectionExperiment/persistencecorrection/calibration.py:444-454
calls the existing fitter on each month pool with min_group_rows=1 (not the
local-group default 50). The distinct-score threshold is 3. With both classes
and >=3 distinct scores, use isotonic; otherwise the existing fitter attempts
Platt as applicable. Failed pool fitting uses identity. Isotonic uses
y_min=0,y_max=1,increasing=True,out_of_bounds=clip (:187-191); Platt is
LogisticRegression(solver=lbfgs,max_iter=1000) on raw probability (:203-207).
No row-count tuning or partition-local calibration is added. Missing a required
month pool raises (:363-370), rather than borrowing another month.

The formal manifest must supplement existing runner manifests with actual
source hashes, package versions, effective parameters, split/row identities,
map identities and seeds. Current Stage 1 command.json records only command,
GEORF_POLYGONS and baseline_version; it is not sufficient reproduction evidence.
