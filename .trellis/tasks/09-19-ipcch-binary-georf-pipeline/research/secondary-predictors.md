# IPCCH predictors — Q6b/Q6c/Q6d/Q6e/Q6f approved

Planning evidence and an approved raw-column whitelist; no panel, feature matrix,
split or model was built. The user selected the existing raw IPCCH source.
Q6b concerns which raw covariates enter the information set, not the number of
eventual derived features. Q6d fixes secondary time summaries and Q6e calendar
representation; Q6f fixes crisis recency and Q6c fixes missing-value handling. History
features and Stage1 horizon metadata are separate from the 70 columns below.

Header check on2026-09-20 against the saved `source_columns.txt`: all70 approved
names exist and are unique; family counts are19/9/6/13/21/2. This checks names,
not field values, temporal validity or the final derived feature width.

User approved Q6b on2026-09-20 and requested freezing the current completed-file
feature set for now. The70 fields below and source hash in PRD R3/input contract
define this version. WB RTP and other improved inputs may be considered later;
no acquisition, addition, replacement, auto-discovery from an updated file, or
overwrite of this baseline is authorized by that future interest.

## Approved raw columns

Use the same fields in partitioned RF, pooled RF and XGBoost. Preserve source
names. Use all21 supplied AEZ categories as one inherited predictor family;
the missing old AEZ_15000 is not fabricated. Coordinates remain predictors as
in the Stage1 baseline. Make admin_code a grouping/join key only, avoiding a
numeric identifier predictor. Country/text labels remain metadata.

### Conflict — 19
```text
distance_to_nearest_acled
event_count_battles
event_count_battles_w5
event_count_battles_w10
event_count_explosions
event_count_explosions_w5
event_count_explosions_w10
event_count_violence
event_count_violence_w5
event_count_violence_w10
sum_fatalities_battles
sum_fatalities_battles_w5
sum_fatalities_battles_w10
sum_fatalities_explosions
sum_fatalities_explosions_w5
sum_fatalities_explosions_w10
sum_fatalities_violence
sum_fatalities_violence_w5
sum_fatalities_violence_w10
```

### Prices and macroeconomic variables — 9
```text
FAO_price
WFP_Price
WFP_Price_std
CPI
GDP
CC
gini
Food_CPI
Food_food_inflation
```

### Vegetation, weather and nightlights — 6
```text
EVI_mean
GPP_mean
Rainf_f_tavg_mean
Tair_f_tavg_mean
nightlight_mean
nightlight_std
```

### Land, terrain, soils and access — 13
```text
crop
range
distance_to_river
elevation
market_distance
market_access
ruggedness
slope
sg_cec_5-15cm
sg_cfvo_5-15cm
sg_nitrogen_5-15cm
sg_phh2o_5-15cm
sg_soc_5-15cm
```

### AEZ — 21
```text
AEZ_4000
AEZ_7000
AEZ_9000
AEZ_10000
AEZ_12000
AEZ_17000
AEZ_19000
AEZ_20000
AEZ_25000
AEZ_28000
AEZ_30000
AEZ_31000
AEZ_32000
AEZ_33000
AEZ_34000
AEZ_35000
AEZ_36000
AEZ_38000
AEZ_40000
AEZ_42000
AEZ_43000
```

### Coordinates — 2
```text
lat
lon
```

Exclude remaining raw columns from this covariate whitelist: IDs/text keys,
outcomes and estimated_population; popdensity has no proven equivalence to old
pop; additional land-surface/weather variables and EVI/GPP dispersion fields
have no retained-baseline counterpart. Year/month remain timing metadata; Q6e
defines the two calendar predictors. Do not recreate absent Tair_zscore or
Rainf_zscore by silently computing full-panel statistics. Names do not establish
availability: every time-indexed feature must obey the approved origin contract.

## Q6d approved temporal inputs — 2026-09-20

Apply the same transformation to all three learned arms and both stages, using
each row's own O=T-H. This changes temporal construction, not Q6b's raw sources:

- For each of the70 raw fields, use the same area's calendar-month O value.
  Missing rows/values remain NaN before Q6c preprocessing; no implicit forward
  or backward fill, future-row retrieval, or full-panel dynamic/static inference.
- Retain15 derived fields from the three baseline helpers' actual feature families:
  `WFP_Price_sum4_asof` sums O-3..O; `WFP_Price_sum12_asof` and
  `nightlight_mean_sum12_asof` sum O-11..O; `EVI_mean_lag{k}_asof`, k=1..12,
  reads O-k. All bounds are inclusive calendar months within the same area.
- A sum requires every month's value in its4/12-month window; otherwise it is
  NaN before imputation. A missing EVI source month yields NaN for that lag.
  Missing months never become zero or get replaced by older observed rows.
- These are temporal SUMS, matching the baseline operation, not means. The
  supplied nightlight field's unresolved spatial statistic remains a separate
  provenance limitation. Using windows ending at O is an explicit adaptation of
  the baseline's row-shifted windows to the approved inclusive origin contract.
- Do not broaden the helpers to other fields, add full-panel auto-detected
  scope-lag copies, or retain any raw target-month T covariate value. The new
  O lookup supplies the horizon alignment. Q6a history remains unchanged;
  Q6e fixes calendar encoding and Q6c fixes imputation below.

This approved contract preserves the baseline's15 preprocessor derivative types
while correcting row/calendar and grouping semantics. No derived features have
been built; task status remains planning.

## Q6e approved with user revision — 2026-09-20

User replaces the proposed month one-hot with sin/cos. Let m be the calendar
month of target T in1..12. Use `target_month_sin=sin(2*pi*(m-1)/12)` and
`target_month_cos=cos(2*pi*(m-1)/12)`, identically in Stage1 and all learned Stage3
arms. This deterministic mapping needs no category discovery or fitted encoding.
T's calendar date is already known at O; no future observed value is retrieved.
Keep Q5h's horizon indicator. Do not add month dummies, separate origin-month
features, numeric years or year dummies. Dates/years remain metadata for joins,
splits, windows and reporting. January and December are adjacent in this encoding.

Evidence: `GeoRFBaseline/src/preprocess/preprocess.py:282-286` creates year/month
dummies from the supplied panel. `config.py:247-251` drops raw year/month/years
but not those dummy columns; `src/feature/feature.py:99-100` also retains them.
The approved encoding keeps cyclic seasonality while removing explicit year effects.
A year category absent from fitting has no learned category-specific effect;
dropping years also discards potentially useful differences among fitting years.
This is a modeling choice, not a claim that known calendar dates leak outcomes.

## Q6f approved crisis-recency feature — 2026-09-20

The user requests months since the last crisis as an additional model feature.
This uses the already selected raw source/approved target, so it does not add a
new external predictor source or reopen the frozen70raw-field whitelist.
Approved meaning for all three learned arms, at every row's own origin O:

- Let C be the latest month <=O in the same area with valid R1
  `ipcch_food_crisis=1`. Use all available preceding history, with no36-month
  truncation or separate maximum age. Invalid/missing labels do not count.
- Set `months_since_last_observed_crisis=12*(year(O)-year(C))+month(O)-month(C)`.
  An observed positive at O yields0; later negative labels do not reset this clock.
- If C does not exist, leave the value NaN and set
  `no_prior_observed_crisis=1`; otherwise set that indicator0. Do not use0 or a
  large fabricated duration for unavailable history. Q6c later handles model NaNs.
- Retain Q6a's latest valid binary label, latest-observation age and history-missing
  indicator. They distinguish no usable label history from observed history with
  no positive. Absence of an observed positive does not prove no historical crisis.
- Save C as `last_observed_crisis_month` provenance, not a numeric model feature.
  Never search past O, including on historical training/validation rows. This
  adds an origin-safe input; no GeoRF partition/F1 mathematics is changed.

Example: last observed positive2024-01, latest valid label0in2024-05 and
origin2024-08 yields crisis-recency7months and latest-observation age3months.
For h3 target2024-11, crisis-recency is still7, measured at O rather than T.
Only an actual positive observation at O yields0: do not infer ongoing crisis
through unobserved months. With sparse labels this is observed-positive recency,
not an identified episode's onset/end date or a proven crisis-free interval.
The user approved these semantics; implementation has not started.

## Q6c approved missing-value representation — 2026-09-20

Preserve eligible rows and the fixed feature schema. Keep pre-imputation values,
missingness and Q6a/Q6f indicators; model-matrix fills are not observed labels,
durations or truth. Convert numerical infinities to NaN for both model families
and record affected columns/counts. No forward/backward fill, scaling,
complete-case deletion or additional generic missing-indicator block is included.

RF: reuse `OutOfRangeImputer(strategy='max_plus', multiplier=100.0)` with
separate fit/transform. For each column use genuine fitting rows only:
- With finite observed maximum M!=0, fill100*M; when M==0, fill100.
- Entirely missing fitting column: retain the column and fill0, matching the
  baseline fallback; audit the all-missing flag. This0 is only a model placeholder.
- Observed values stay unchanged. Negative M can put the fill inside the data
  range: this inherited formula is not a guaranteed high sentinel or M+100.
- Stage1 fits one imputer on the internal fitting cohort's approved horizon
  views, excluding validation, singletons and pseudo rows. Transform the full
  Stage1 matrix and supplementary inputs once; all root/child RFs reuse it.
  The RF adapter appends its approved zero-feature pseudo rows afterwards.
- Stage3 fits one imputer per origin/horizon on the common actual36-month
  fitting pool, shared by pooled and all partition RFs. Evaluation only transforms.
  Any validation split must fit preprocessing solely on its fitting side.
- Record feature order, fit-row identity, fill values and all-missing flags.
  Test/validation extrema cannot alter fitted values.

XGBoost receives the same pre-imputation inputs with native NaN, without RF
sentinels. Approved semantic missing-history indicators appear in all learned
arms. This preserves their information set/cohorts with explicit model-specific
missing-value representation. Raw feature exports retain NaN before RF filling.

Verified evidence, all paths inside GeoRFBaseline:
- `src/feature/feature.py:124-145` calls comp_impute before temporal splitting
  (app`:1342,1398-1399,425-438`; comparison script`:920,990-1004`). Its XGB path
  skips imputation but replaces residual infinity with0. Q6c replaces
  those boundaries; app`:1356-1364` also has a NaN-row deletion to avoid inheriting.
- `src/preprocess/preprocess.py:70-95` and `src/customize/customize.py:251,300,323`
  show comp_impute creates/refits an imputer on each call and discards its state.
  Calling it on test rows refits. The existing class at customize`:85-105,130-149,
  151-188` supplies the exact formula/fallback and stored-value transform above.
- `src/model/GeoRF.py:219-249,287-305,401-420` supplies inner split and the shared
  matrix passed to partitioning; transformation`:172-177` retrains the root from
  that matrix. Only converting a temporary first-root array would miss this path.
- `src/model/model_RF.py:94-112,244-247,352-358` appends one zero row per class
  at every root/child fit. `src/model/train_branch.py:31-50` adds no transformation.
- Comparison script`:220-267` fits sklearn RF directly, without pseudo rows;
  separate pooled/local filters at`:991-1024` must obey approved common support.
  Optional threshold fits at`:1070-1107` have an inner split (disabled under Q7a);
  final-window fits are at`:1151-1162`.
- `requirements.txt:1-6` and `TESTED_ENVIRONMENT.json:2-9,21` record tested
  Python3.12.10/sklearn1.6.1, not historical reconstruction. No native RF-NaN
  behavior or new environment was assumed from the version number.

Q6c is approved; no missing-value transformation or model was executed.

## Frozen baseline evidence

All paths in this section are inside GeoRFBaseline, not root app/src:
- `app/main_model_GF.py:41,45` loads package config; `src/model/GeoRF.py:29,287-290,831-838` applies package `config.FEATURE_DROP`. `config.py:247-251` drops raw year/month/years, fews_ha, IPC_admin_code and ISO_encoded, with no patterns.
- `src/preprocess/preprocess.py:169-177,199-205,279-286` drops FEWS forecast/admin text and current ordinal target, creates ISO_encoded for grouping and year/month dummies. `src/feature/feature.py:100` drops target/date/group metadata, including ISO_encoded. Default Stage1 retains FEWSNET_admin_code, lat/lon and year/month dummies.
- `config_visual.py:156-160` has a different drop declaration, but is not the GeoRF model's active config. Stage3 `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py:920-930,226,247,261` feeds prepared arrays directly to RF without the GeoRF post-drop. This experiment must explicitly harmonize information rather than inherit that discrepancy.
- Baseline input header was read only at `C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_forecast_unadjusted_bm.csv:1`, located through package `app/main_model_GF.py:95,106-107` and comparison script`:65`. Its columns support the families above; it is not a final trained-feature manifest. Root feature_columns_debug.csv was not used as an authoritative manifest.
- Original AEZ codes are10000,12000,15000,17000,19000,25000,31000,32000,33000,34000,36000,38000,4000,40000,43000,7000,9000. The current IPCCH header contains21codes, including five additional supplied categories and lacking15000.
- `src/preprocess/preprocess.py:289-295,625-627,647-648,666-670`: loop-internal returns mean only WFP_Price_m4/m12, nightlight_m12 and EVI_l1..l12 are actually derived by the three helpers. Their grouped shifts use observation rows; rolling after the shift is not grouped. This is evidence for Q6d, not approval to copy unsafe processing or expand the helpers to every listed field.
- `src/feature/feature.py:63-65,92-100` uses full-panel within-area/year variation to classify dynamic columns, adds scope-lag columns and retains originals. Stage1/Stage3 strict-lag-only flags default false (`app/main_model_GF.py:1241`; comparison script`:831`; wrapper `scripts/run_stage1.py:47-49` does not enable it). Final columns are data-dependent; names alone do not prove own-origin availability.

## Mapping and upstream provenance

Codebook: `C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\assembled_IPCCH\metadata\variable_codebook_reorganized.csv`.

| Old name | Approved IPCCH field | Evidence and limit |
|---|---|---|
| EVI | EVI_mean | `../assemble_latest_IPCCH/02_preprocess_and_combine.ipynb`, cell3 (zero-based), line2 explicitly aliases it. Codebook line101 defines EVI, monthly then zonal mean. |
| gpp_mean | GPP_mean | Codebook line103 defines GOSIF-GPP zonal productivity. No explicit old-name mapping or numerical equivalence was established. |
| nightlight | nightlight_mean | Same cell3 alias. Codebook line99 says mean; a candidate assembly route uses sum values under this name, as detailed below. |
| nightlight_sd | nightlight_std | Same cell3 alias; codebook line100 says standard deviation. Complete selected-source production lineage remains unverified. |

Codebook lines110-127 define conflict _w5/_w10 as spatial nearest-station distance
weighting, not5/10month lags. Lines136-139 identify CC as Control of Corruptions
percentile and GDP/gini as World Bank measures; no units/base-year/publication
schedule was established for the latter. Lines20/42 describe crop/range as ASAP
masks. AEZ lines21-41 say indicator OR share without resolving that distinction.
The approved whitelist preserves supplied values without asserting new units.

### Concrete upstream uncertainty, not proof of selected-source corruption

- `../assemble_latest_IPCCH/01_extract_invariables_and_scaffold.ipynb:21-22,865` reads `Step1_Nightlight/output_update/nightlight_sum_extraction_results.csv` and names the melted value nightlight_mean. The CSV header is a monthly wide table with region_id, not a named mean statistic.
- `../../Step1_Nightlight/03_alternative_extract_nightlight.py:61,64,67-75,97-104,116-120` uses zonal SUM/STD and writes the sum unchanged. Its configured output directory at`:14-17` differs from the assembly input path; no intervening copy/generation record was found. This does not numerically prove the current input CSV was made by that script or establish earlier-year statistics.
- Candidate assembly notebook`:875-881,930-943,970-982,1001-1013` sorts by area/time but uses ungrouped ffill then zero fill for nightlights/EVI. This is a concrete possible cross-area/future-value path. Its written output at`:1320` is IPCCH_2025_secondary.csv, not the selected IPCCH_2026_completed.csv.
- `../assemble_latest_IPCCH/organize_ipcch_ml_data_folder.py:323-335` only moves IPCCH_2026_completed.csv into raw. A bounded search of assembly scripts/notebooks and assembled_IPCCH/code did not find the final writer or a verified link from the candidate secondary table to the selected raw file.
- Thus actual inherited preprocessing and affected rows are UNVERIFIED. Do not claim the selected CSV is contaminated, clean, or that nightlight_mean is confirmed to be a spatial mean. The user-selected raw data remains unchanged; this planning work did not authorize source reconstruction, substitution or dropping ambiguous fields on the user's behalf. Any required source repair needs a separate scope decision.

The approved common-family whitelist does not establish real-time publication
availability or complete upstream temporal validity. Preserve this limitation in
design/run provenance and resolve any concrete incompatibility before claiming
stronger validation. Q6b approval is not a claim that upstream timing was verified.
