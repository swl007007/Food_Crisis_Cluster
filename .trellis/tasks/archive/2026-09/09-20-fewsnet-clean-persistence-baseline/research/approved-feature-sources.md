# Approved feature-source additions — 2026-09-20

Planning inspection only: directory/header and bounded source-code reads; no
panel join, feature generation or training. Two scouts located the sources and
assembly code; the main session checked panel/derived-file headers and merge code
anchors. Coverage summaries below are inventory evidence, not a completed join audit.

`DATA` = `Analysis/1.Source Data/` in the user's Google fund directory.
`ASSEMBLY` = `Analysis/2.source_code/Step5_Geo_RF_trial/assemble_latest_IPCCH/`.

## Approved removal boundary

Exact original fields: `FAO_price`, `WFP_Price`, `WFP_Price_std`.
Remove derived copies of those fields from the updated schema as well.
`market_distance`, `Food_CPI`, `Food_food_inflation` are separate fields and are
not included in this price-removal instruction. This matches the separation in
`scripts/create_feature_exclude_datasets_from_unadjusted.py:81-98`.

The original `DATA/FEWSNET_forecast_unadjusted_bm.csv` and the assembled
`FEWSNET_forecast_unadjusted_bm_2025_combined.normalized-v1.csv` both have 88
header fields including the three old prices. The assembled 2025, combined and
normalized-v1 headers do not yet contain the four approved addition families.
The user approved retaining `DATA/FEWSNET_forecast_unadjusted_bm.csv` as the
master panel on 2026-09-20; additions join onto it. The newer assembled panel
is not the chosen source for this experiment.

## Available sources and reuse

| Family | Local source | Shape / existing route | Remaining verification |
|---|---|---|---|
| NOAA_ENSO | `DATA/NOAA_ENSO/nina34.anom.csv` | Monthly global Date/value series; existing loader maps to `nino34_anom` and merges on year/month | Source header names ERSST v6; actual sentinels include -9999 despite header -99.99. Units, historical vintage and origin-month availability still need a stated contract. |
| WB_RTP | `DATA/WB_RTP_price/wb_food_price_index.csv`, derived from `WLD_RTFP_mkt_2026-04-20.csv` | Monthly market coordinates, no FEWS admin key; approved IPCCH rule: nearest same-month market within 100 km, cross-border allowed | Verify source coverage, units, interpolation and historical-vintage limits. |
| Coastline_distance_NOAA | `DATA/Coastline_distance_NOAA/GMT_intermediate_coast_distance_01d.tif` | Raster plus an IPCCH-coordinate table `IPCCH_2026_price_completed_unique_lat_lon_coastline_dist.csv` | FEWS coverage, raster CRS/nodata, signed-distance convention and units need verification. Existing example values are negative; do not silently take absolute values. |
| bloomberg_food_and_derivative | `DATA/Bloomberg_food_and_derivative/bbg_*_monthly_*.csv` | Four global monthly files: fertiliser, oil/gas, soybean-oil futures, staple foods; join keys year/month | Fix series inventory, units, aggregation/continuous-contract construction and availability rules. |

ENSO loader evidence: `ASSEMBLY/add_nino34_to_ipcch.py:40-65,92-97` handles
missing sentinels, rejects duplicate months and uses a many-to-one month join.
The selected file has 948 monthly rows, 1948-01 through 2026-12, but valid
non-sentinel values cover 1950-01 through 2026-04. A second long-series file exists;
do not splice the two sources without resolving their definitions.

WB derived header is exactly `inflation_food_price_index,year,month,lat,lon,`
`food_price_index_WB,food_inflation_wb`. `DATA/WB_RTP_price/filter.ipynb` cell 3
defines the price index as the mean of opening/closing index fields and copies
the inflation column to `food_inflation_wb`; cell 4 retains both inflation aliases.
Do not count duplicate aliases as distinct engineered signals. The raw market
file includes ISO3, coordinates, coverage/confidence and interpolation metadata.
Those fields are absent from the reduced file. Its 2026 snapshot does not by
itself prove the values were available in each historical forecast month.
WB may itself incorporate FAO/WFP inputs; removing old columns is a model-schema
replacement, not a claim of source independence.
Existing join evidence: `ASSEMBLY/add_wb_food_price_to_ipcch.py:13,107-148`.

Bloomberg files inventoried:

- `bbg_fertiliser_monthly_051326.csv`: 2010-01 through 2026-05, ten bbg fields.
- `bbg_oil_and_gas_monthly_050826.csv`: 1996-11 through 2026-05, two bbg fields.
- `bbg_soybean_oil_futures_monthly_050826.csv`: 2000-01 through 2026-05, last/bid.
- `bbg_staple_food_x1_monthly_050826.csv`: 1996-11 through 2026-05, five crops.

Series count is inventory only, not approval to treat every column as an
independent signal. IPCCH codebook commodity labels conflict with some workbook
labels; use underlying series definitions. Do not infer a complete May 2026
monthly observation from a file dated early/mid-May.

## Follow-up research versus user decisions

Resolve source units, sentinels and extraction conventions from source evidence
without asking the user to guess. User-owned choices still include availability
assumptions where vintage evidence cannot be established and the finite
engineered-feature search design.

Decision on 2026-09-20: use the existing IPCCH matching rule, permitting
cross-border markets. The proposed same-country restriction was not adopted.
Use the nearest same-month market within 100 km of the area's representative
coordinate; retain missing when no market qualifies. Country metadata is not
required for this matching rule. No FEWS join or coverage calculation has been
executed during planning. Feature-month matching remains separate from deciding
which source month is available at the forecast origin.

No source or historical result was edited by this inspection.

## Existing feature path: code facts relevant to the new contract

Two further read-only scouts inspected GeoRFBaseline and helper implementations
in IPCCHGeoRFExperiment/EthiopiaForecastingExperiment. The main session checked
the label filter, lag construction and imputation code directly. These are
facts about code defaults, not proof of settings used in any historical run.

- GeoRFBaseline/src/preprocess/preprocess.py:179-197 filters out null targets
  before its IPC lag construction at :265-273. Those lags use row shift, not
  exact calendar-month lookup. Because FEWS labels are sparse, the remaining
  row grid cannot be assumed monthly.
- src/feature/feature.py:58-65 infers dynamic fields from more than two distinct
  values within an admin-year. It then row-shifts those fields by the horizon
  (:92-96) while retaining original values (:99-113). The optional strict-lag
  selector keeps inferred static fields too; low variation does not establish
  static meaning or origin availability.
- prepare_features applies max_plus imputation to the entire supplied matrix
  before temporal splitting (:124-145). src/customize/customize.py:75-105 fits
  column extrema on that input; max_plus uses max*100 (100 when max=0), with
  zero for an all-missing training column (:130-138). The new contract requires
  fitting such statistics on the actual fitting subset and only transforming
  validation/target rows. D23 retains max_plus while correcting its fit scope.
- The three preprocessing engineering helpers return inside their first field
  loop (preprocess.py:623-627,645-648,666-670). Actual inherited derivatives are
  WFP price preceding 4/12-row sums, nightlight preceding 12-row sum and EVI
  lags 1-12. Listed rain/temperature/GPP histories and conflict aggregates do
  not thereby exist. Do not silently repair the loop and call the resulting
  extra signals original-feature parity. Rolling after groupby.shift is not
  itself an explicit grouped calendar rolling implementation.
- GeoRF applies config.py:247-251's exact feature-drop list through
  src/model/GeoRF.py:840-909. config_visual.py has a different list. The Stage 3
  comparison script prepares/splits its matrix (:920-930,993-1017) without an
  equivalent FEATURE_DROP step found by the scout. Freeze actual stage schemas
  rather than assuming all these defaults produce the same feature set.

Header-level signal inventory: weather/vegetation includes EVI, rain, temperature,
GPP and existing rain/temperature z-scores; conflict includes event/fatality counts
with existing spatial w5/w10 variants and nearest-conflict distance; economics
includes nightlight, prices/CPI, GDP/gini and population; geography includes
coordinates, AEZ, land cover, soils, terrain and river/market distances. Existing
z-scores need source-definition verification before being treated as origin-safe
anomalies. The four new source families are not already present by name.

## Reusable helpers and their limits

- IPCCHGeoRFExperiment/prepare_data.py:974-978 encodes target-month sin/cos;
  :808-838 and :980-1006 look up latest observed binary state, observation age
  and time since the last observed crisis at month <= origin. This is recency,
  not a continuous crisis duration. The complete assembler has a fixed IPCCH
  schema; reuse narrow logic rather than importing its entire workflow.
- IPCCH _PanelGrid/:969-972 provides exact calendar lag lookup; _WindowSummer
  at :785-805 requires all months in a window and includes its end month. These
  semantics must be chosen explicitly, not confused with older excluded-end
  rolling sums. Its Stage 1/3 imputation calls fit only on training rows
  (run_pipeline.py:523-547,1644), using the IPCCH imputation method.
- EthiopiaForecastingExperiment/prepare_horizon_aligned_data.py:191-196,216-248
  validates a complete covariate grid, aligns origin rows to T=O+H, then removes
  missing target labels. Its working-panel rolling helpers are usable only on
  a verified complete monthly grid, not on sparse label rows.
- Ethiopia aligned_refit.py:88-140's phase history has a different convention:
  nationally qualified releases, release month strictly before origin, phases
  restricted to 1..4. Do not silently adopt those FEWS-specific restrictions or
  its 90% cohort qualification threshold for this multi-country experiment.
- No general lag-difference/slope, rolling-standard-deviation, newly fitted
  climatology/anomaly, crisis-streak or covariate-observation-age builder was
  found in these two experiment directories. Available label-history flags do
  not constitute a generic covariate missingness feature family.

## Approved shared preprocessing — D23

The user adopted a shared experiment-local feature preparation route for the corrected
original-feature reference and every updated-feature candidate. Reuse released
RF/partition logic while building calendar-based transforms on the complete
covariate panel before restricting to labeled targets. Declare static, dynamic,
calendar-known and outcome-history semantics explicitly; source-availability
rules determine the latest dynamic value eligible at O=T-H. Do not infer that a
slow-moving variable is static just from its distinct-value count.

For minimal method change, retain the existing max_plus imputation formula but
fit it only on real fitting rows in each applicable split, then transform other
rows with the same frozen statistics. Stage 1 internal validation remains
separate from fitting those statistics; Stage 3 fits on its rolling RF training
rows. Artificial class-recovery rows must not supply the imputer's extrema.
Exact schema alignment across stages and inherited lag/window definitions still
need specification. The original-feature reference retains its source families
but corrects time semantics; it cannot be described as reproducing the old
default feature matrix with only two numerical switches changed.

## Approved finite-search scope after compute concern — D24

Keep the approved updated source set as the common base: add ENSO, WB prices,
coastline distance and Bloomberg sources; remove legacy FAO/WFP price families
and their derivatives. The user approved five optional engineering blocks:

| Block | Scope to specify before execution |
|---|---|
| A: Season and IPC history | Calendar encoding and observed crisis/phase history, including crisis recency |
| B: Trailing levels and volatility | Fixed multi-month summaries of appropriate weather, vegetation, price and conflict fields |
| C: Changes and anomalies | Fixed lag differences and deviations from explicitly eligible historical references |
| D: Prespecified interactions | A short, explicit list of cross-variable interactions; no automatic all-pairs expansion |
| E: Missingness and observation age | Covariate availability flags and elapsed time since actual observations |

The initial assistant proposal was all 2^5=32 combinations. Following the user's
compute-cost concern, the user adopted 12 fixed recipes: the all-off source base, five
single-block additions, all five blocks together, and five variants each dropping
one block from the full set. This tests isolated additions and contributions
conditional on the other four blocks; it does not exhaust all two-/three-block
combinations. The original-feature corrected reference remains separate.
Raw/calibrated variants share RF predictions and do not double recipe counts.
D21 still requires each recipe's compatible development partitions/models;
only the frozen winner and reference enter final evaluation. The five blocks
and 12-recipe budget are approved; exact block contents still require decisions.
Manifest order is BASE, A, B, C, D, E, ABCDE, BCDE, ACDE, ABDE, ABCE, ABCD.

Exact input columns, fixed windows, formulas, interaction pairs and the treatment
of overlapping/inherited signals remain undecided. Identical derivatives must
not be duplicated merely because two blocks refer to them. Do not count an
existing source z-score as a newly verified origin-safe anomaly. Imputation is
the common D23 preprocessing step; optional availability features do not change
the imputation method. No nested window/interaction search is included. D24
fixes the search structure and recipe budget, not the eventual column manifest.

## Compute accounting — calendar bounds, not measured runtime

Read-only inspection found no comparable full-run elapsed-time evidence in the
checked global FEWS manifests/logs. GeoRFBaseline/VALIDATION.md:30-34 explicitly
excludes a full real-data Stage 1/3 run or performance benchmark from release
validation. PersistenceCorrectionExperiment's phase2 probability run logs and
manifests record inputs, predictions and RF parameters, but the inspected runs
do not provide elapsed durations. A single timestamp or filesystem modification
time does not establish runtime. No hours/days estimate or new benchmark is claimed.

Using the inspected label calendar and all three scopes, before D20 support
exclusions, each feature recipe has the following candidate counts:

- 2014-2016 temporary window: (4 + 4 + 3) target months * 3 scopes = 33.
- 2016-2018 temporary window: (3 + 3 + 3) * 3 = 27.
- Exact same-recipe 2016 overlap: 9; unique development Stage 1 jobs = 51.
- 2018 calibration plus 2020 threshold prediction folds: 9 + 9 = 18.
- Two temporary consensus-window builds. D55 later fixed a single general map
  for each; no additional month-specific builds are selected.

| Search only, excluding original-feature reference and final evaluation | Earlier 32-recipe proposal | Approved 12 recipes |
|---|---:|---:|
| Unique development Stage 1 candidate jobs | 1,632 | 612 |
| Development target/scope prediction folds | 576 | 216 |
| Temporary consensus windows | 64 | 24 |

These search counts fall by 62.5%; wall-clock time need not scale identically
because feature width, accepted branches, graph structure and parallelism vary.
The separate reference adds 51 development Stage 1 jobs and 18 development
prediction folds in either design. The winner and reference final maps each
need 27 candidate inputs for 2018-2020, of which 9 may reuse identical 2018
development evidence: 18 new jobs each. Thus total unique Stage 1 candidate
counts including reference and final maps would be 1,719 versus 699, before
support exclusions and assuming exact overlap compatibility. Final forecast
folds and calibration/threshold computation are additional.

Jobs are not individual forest fits: the Stage 1 spatial split search fits root
and child forests; released defaults use 100 trees per forest
(GeoRFBaseline/src/model/GeoRF.py:50, src/model/model_RF.py:289-293). Stage 3 fits
a pooled model and then models for eligible partitions
(scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py:220-267), so one prediction
fold also involves multiple fits. Stage 2 additionally constructs spatial
similarity matrices and performs clustering. The accounting alone does not
measure which stage dominates elapsed time on this machine.

Reuse source joins, geometry/adjacency and deterministic individual feature
construction where identities match. Reuse overlapping Stage 1 candidates only
within the same feature recipe and identical training inputs/configuration/seed.
Feature-dependent models, maps, probabilities, calibrators and thresholds are
not interchangeable across different recipes. Cutting recipe count preserves
the approved three-stage comparison and temporal controls.

## Approved block A definition — D25

The user adopted eight columns, calculated for each target T and origin O=T-H:

| Column | Approved definition |
|---|---|
| target_month_sin | sin(2*pi*(month(T)-1)/12) |
| target_month_cos | cos(2*pi*(month(T)-1)/12) |
| last_observed_ipc_phase | Phase from the area's latest eligible valid observed IPC record |
| last_observed_crisis | Original binary crisis label from that same record |
| last_observed_ipc_age_months | Calendar months from that observation month to O |
| months_since_last_observed_crisis | Calendar months from latest eligible observed crisis month to O |
| no_observed_ipc_history | 1 when no eligible valid IPC record exists, otherwise 0 |
| no_prior_observed_crisis | 1 when no eligible crisis observation exists, otherwise 0 |

Missing history leaves corresponding numeric values missing until the common
training-fitted imputation step, with flags retained. If no history exists,
both flags are 1. Recency does not assert continuous crisis duration across
unobserved months. Values reflect eligible real observations, not fabricated
monthly target labels. Exact origin-month/publication availability and alignment
between phase and the original binary label must still be fixed/validated.
IPCCH's as-of history and calendar helpers supply reusable implementation ideas;
its target definition or entire fixed feature schema is not adopted.

This block is additional to the eventual updated base schema. The disposition
of inherited calendar encodings and exact historical lags remains to be frozen
without silently changing the original-feature reference. Block A owns these
two history flags; block E will cover covariate availability, avoiding duplicate
copies of A's flags. No extra candidate combinations are introduced by specifying
the block contents.

## Approved block B window/statistic policy — D26

The user adopted fixed W in {3, 6, 12} calendar months, without an additional window grid.
For explicitly whitelisted continuous weather/vegetation/ENSO, price and
nightlight series, compute trailing mean and population standard deviation
(ddof=0). For explicitly whitelisted conflict event/fatality count series,
compute trailing sums; do not add identical-information mean copies for a
fixed complete window. Static attributes and existing lag/rolling derivatives
are not automatically expanded as fresh source series.

The endpoint is the source month permitted by the still-to-be-frozen source
availability rule at origin O. A W-month feature covers exactly W consecutive
calendar months ending there, all eligible at O. Require a real nonmissing
source value in every month; otherwise leave the feature missing before D23
imputation. Do not shorten a window to the available rows, skip a missing
calendar month, interpret missing conflict as zero, or fit rolling statistics
over fabricated imputed history. An existing source's interpolation/vintage
limits remain part of its separate source contract.

This fixes block B's window/statistic policy. Exact source columns and
per-series units still require a frozen whitelist, including which Bloomberg
series and conflict variants to include. Deduplicate any inherited derivative
with identical inputs/window/availability semantics. Feature counts and memory
cost must be reported for the final manifest; this policy does not authorize
expanding every numeric column or introducing new recipe combinations.

## Approved block C formulas — D27

For each dynamic source series in a still-to-be-frozen whitelist, let s be its
source month permitted at origin O. The user adopted three features:

| Feature | Formula |
|---|---|
| Three-month change | x(s) - x(s-3) |
| Twelve-month change | x(s) - x(s-12) |
| Trailing standardized deviation | [x(s) - mean(x(s-12), ..., x(s-1))] / std_population(x(s-12), ..., x(s-1)) |

All offsets are calendar months. Difference features need both exact endpoint
values, without requiring the intervening months. Standardized deviation
requires x(s) and all 12 preceding monthly values; the reference excludes x(s).
If any required value is missing, leave that feature missing before D23
imputation. Zero reference standard deviation also produces a missing
standardized-deviation feature; do not add an arbitrary epsilon or invent a
zero anomaly. Calculate from eligible pre-imputation source values only.

Absolute changes retain source units and avoid division by zero/negative
baselines. The standardized deviation is a trailing-history comparison, not a
long-term seasonal climatology. It uses no full-panel reference statistics and
does not validate the inherited rain/temperature z-score sources. The exact
whitelist, availability endpoint and handling of any identical inherited
derivatives must be frozen. Block C may compute its own required summaries
when block B is off; those intermediates are not extra model-input columns.
This decision adds no window tuning or recipes beyond D24's fixed 12.

## Approved block D interactions — D28

The user adopted exactly four additional model-input columns, with no all-pairs expansion
or interaction selection within a recipe:

| Column | Product | Intended joint signal |
|---|---|---|
| rain_evi_interaction | Z_rain * Z_EVI | Rainfall and vegetation deviations |
| rain_conflict_interaction | Z_rain * conflict_events_3m | Rainfall deviation and recent conflict |
| inflation_conflict_interaction | food_inflation_wb * conflict_events_3m | Food inflation and recent conflict |
| inflation_market_distance_interaction | food_inflation_wb * market_distance | Food inflation and distance to markets |

Z_rain and Z_EVI use D27's trailing standardized deviation applied respectively
to `Rainf_f_tavg_mean` and `EVI`; these are not the unverified inherited z-score
columns. `conflict_events_3m` is the sum of `event_count_battles`,
`event_count_explosions` and `event_count_violence` over three consecutive
calendar months ending at the eligible conflict source month. All nine monthly
category values are required; do not include w5/w10 variants or fatality counts
in this operand. `food_inflation_wb` uses the single WB inflation field, not its
duplicate alias or the WB price-index level. `market_distance` is the retained
master-panel field, not WB match distance or coastline distance.

Every operand follows its own frozen source-availability contract at O=T-H;
different sources need not share an endpoint month. Retain signs and source
units, without clipping, percentage conversion or additional normalization.
Source units and availability still need verification before schema freeze.
Any missing operand leaves the product missing before D23 imputation; do not
multiply imputed missing-value sentinels.

D computes these fixed intermediate summaries even when B/C are off, but exposes
only its four products. This does not implicitly enable B/C input columns or
add recipes. The approved operand fields must be recorded explicitly in the
eventual whitelist. This approval remains planning-only.

## Approved block E availability features — D29

The user adopted availability features for a frozen whitelist of base source covariates,
before D23 imputation, with no additional windows or recipe combinations:

| Source-field type | Features |
|---|---|
| Dynamic source covariate x | x_missing and x_age_months |
| Static source covariate x | x_missing only |

For a dynamic field, let s_x(O) be its scheduled source-month endpoint permitted
at origin O under the source-availability contract, before inspecting whether
its value is missing. x_missing is 1 if the cleaned source value at that exact
endpoint is absent/invalid, otherwise 0. Do not move the endpoint backwards to
hide a missing current value. Declared sentinels count as missing.

Let m be the latest month at or before s_x(O) with a valid source value eligible
at O in the frozen source history. x_age_months is the calendar-month difference
O-m, not the gap from the target month or from s_x(O). Thus a valid value subject
to a one-month availability lag has age one, not zero. If no eligible valid
history exists, x_missing is 1 and x_age_months stays missing until D23
imputation. Do not add an arbitrary age cap or a separate no-history flag.
For static fields, x_missing records whether the permitted source value is
missing; no artificial observation-age feature is defined.

Calculate ages from eligible pre-imputation source histories, not previously
filled values. These features describe the age of an available source value;
without original-observation metadata they cannot establish the age of a raw
measurement or its historical publication date. Retain source interpolation
and vintage limitations in provenance, particularly for WB. WB ages concern
the area's joined series and do not imply the same market supplied each month.

E does not forward-fill the base value or relax B/C's complete-window rules.
Exclude IDs, target labels, expert forecasts, calendar encodings, A's history
features/flags and inherited or B/C/D-derived columns from this availability
expansion. Deduplicate source aliases. The exact base-source whitelist and
resulting column count still need freezing. This approval remains planning-only.

## Additional schema-boundary evidence

Read-only scouts inspected release-directory source and external source headers;
the main session checked the calendar/history creation, default feature-drop
list, Stage 3 feature preparation and WB alias construction at the anchors below.
No model was run and no source values were joined or transformed. This inspection
does not establish byte identity between the release directory and release ZIP.

- `GeoRFBaseline/src/preprocess/preprocess.py:265-286` creates binary and phase
  history at 4/8/12 row offsets and year/month dummies. There is no inherited
  sine/cosine encoding. Keeping these defaults would leave calendar/history
  information present even when new block A is off.
- `GeoRFBaseline/config.py:247-250` drops exact names `year`, `month`, `fews_ha`,
  `IPC_admin_code`, `ISO_encoded`, `years`, with no wildcard patterns. This does
  not remove year/month dummies or history/assistance derivatives by prefix.
  `GeoRFBaseline/config_visual.py:156-159` declares different drops and cannot
  be assumed to describe the release runner's effective configuration.
- `GeoRFBaseline/scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py:920-930`
  prepares X with an optional strict-lag selector; its fitting path does not
  apply GeoRF's feature-drop method. Freeze an explicit schema shared across
  stages instead of silently inheriting this Stage 1/3 difference.
- Release preprocessing `:169-177` removes projection/adjusted fields, while
  `fews_ha` survives preparation and is subject to the differing later drop
  paths. Feature preparation `:92-103` may generate a second scope lag for
  already-derived fields. The final reference schema must resolve these details
  explicitly rather than infer them from family names or copy both paths.

## Approved updated BASE boundary — D30

The user adopted a source-level updated BASE: retain whitelisted static attributes and
one origin-aligned value per whitelisted dynamic source field, applying the
approved four source additions and FAO/WFP removals. Use one explicit ordered
schema consistently in Stage 1 and Stage 3. Keys and targets remain available
for joins/evaluation but are not automatically predictors.

Do not inherit the old pipeline's generated year/month encodings, fixed IPC
history lags, EVI lags 1-12, nightlight 12-month sum or secondary horizon-lag
copies. A provides the approved calendar/history inputs; B-E provide their
declared engineered inputs. An all-off recipe then contains the approved
source-level base only. Persistence remains available as the correction
baseline even when A is off; this boundary does not remove it from evaluation.

This decision applies to the updated-feature recipes only. Keep the separate
original-feature corrected reference and specify its actual inherited features
and corrected calendar semantics independently. Do not silently remove its
EVI/nightlight/history features as part of this proposal. Source-provided
derived fields such as spatial conflict variants still need individual
whitelist/availability decisions; being present in the master CSV is not
sufficient approval to use them. D46 below excludes inherited climate z-scores.

## Exact additional-source field inventory — not a frozen whitelist

Header evidence is CSV line 1 under `DATA/Bloomberg_food_and_derivative/`:

- `bbg_fertiliser_monthly_051326.csv`: `bbg_GCFPURGB`, `bbg_GCFPDANO`,
  `bbg_GCFPPOBA`, `bbg_GCFPURBS`, `bbg_GCFPAMME`, `bbg_GCFPAMBS`,
  `bbg_GCFPURMG`, `bbg_NGUSHHUB`, `bbg_GCFPDAIN`, `bbg_TZTX2_Comdty`.
- `bbg_oil_and_gas_monthly_050826.csv`:
  `bbg_oilgas_CL1_COMB_Comdty_price`, `bbg_oilgas_NG1_Comdty_price`.
- `bbg_soybean_oil_futures_monthly_050826.csv`:
  `bbg_soybean_oil_futures_last_price`, `bbg_soybean_oil_futures_bid`.
- `bbg_staple_food_x1_monthly_050826.csv`:
  `bbg_staple_food_corn_x1_price`, `bbg_staple_food_hard_wheat_x1_price`,
  `bbg_staple_food_rough_rice_x1_price`, `bbg_staple_food_soft_wheat_x1_price`,
  `bbg_staple_food_soybeans_x1_price`.

The fertiliser workbook `bbg_fertiliser_051326.xlsx`, sheet `main`, pairs labels
in row 1 (row 2 for W) with tickers in row 3:

| Ticker suffix | Column | Workbook label |
|---|---|---|
| GCFPURGB | C | Urea Gulf NOLA (gran) |
| GCFPDANO | E | DAP US Gulf Nola |
| GCFPPOBA | F | Potash Baltic Standard |
| GCFPURBS | G | Urea Black Sea Prill |
| GCFPAMME | J | Ammonia Middle East |
| GCFPAMBS | K | Ammonia Black Sea |
| GCFPURMG | M | Urea Middle East Gran |
| NGUSHHUB | Q | Natural Gas BNGC |
| GCFPDAIN | T | DAP India |
| TZTX2_Comdty | W | ICE endex Dutch TTF Natural gas FUTURES PRICE |

Thus that file contains eight fertiliser and two gas series. Gas appears in
multiple files, but their tickers/pricing measures are not proven equivalent.
Soybean-oil workbook `bbg_soybean_oil_futures_050826.xlsx`, `Worksheet!B1`,
`B7:C7`, identifies one security `BO1 COMB Comdty` with PX_LAST/PX_BID quotes;
these are two measures of one security, not identical values. The `USD` label
at B5 does not verify unit conversion. Oil/gas and staple CSV identifiers are
recorded above; detailed units, contract roll rules and `x1` construction remain
unverified. This inventory does not choose which fields enter BASE or B/C.

WB `filter.ipynb:842-843` defines `food_price_index_WB` as the mean of opening
and closing indices and defines `food_inflation_wb` as an alias of
`inflation_food_price_index`. There are two substantive signals, not three;
full-file equality was not scanned. ENSO's existing loader
`ASSEMBLY/add_nino34_to_ipcch.py:7,45-65` selects `nina34.anom.csv`, names its
value `nino34_anom` and treats both -99.99 and -9999 as missing. The alternative
long-history ENSO file is not implicitly selected.

## Approved additional-source BASE whitelist — D31

The user adopted exactly 22 additional BASE fields from the four approved source families:

| Source subset | Count | Included fields |
|---|---:|---|
| ENSO | 1 | nino34_anom from the selected nina34.anom.csv |
| WB | 2 | food_price_index_WB, food_inflation_wb |
| Coastline | 1 | One distance field extracted from the specified NOAA raster; extraction and units remain unresolved |
| Bloomberg fertiliser prices | 8 | bbg_GCFPURGB, bbg_GCFPDANO, bbg_GCFPPOBA, bbg_GCFPURBS, bbg_GCFPAMME, bbg_GCFPAMBS, bbg_GCFPURMG, bbg_GCFPDAIN |
| Bloomberg energy | 4 | bbg_NGUSHHUB, bbg_TZTX2_Comdty, bbg_oilgas_CL1_COMB_Comdty_price, bbg_oilgas_NG1_Comdty_price |
| Bloomberg staple crops | 5 | bbg_staple_food_corn_x1_price, bbg_staple_food_hard_wheat_x1_price, bbg_staple_food_rough_rice_x1_price, bbg_staple_food_soft_wheat_x1_price, bbg_staple_food_soybeans_x1_price |
| Bloomberg soybean oil | 1 | bbg_soybean_oil_futures_last_price |

Exclude the WB alias `inflation_food_price_index` as duplicate construction.
For soybean oil, choose the last-price measure and exclude
`bbg_soybean_oil_futures_bid`; this is a parsimonious choice between two quote
measures of one security, not a claim that their values are identical. Retain
the geographically/contractually distinct fertiliser and energy series; do not
declare separate gas tickers redundant without supporting source evidence.

The count is 18 Bloomberg fields plus four from the other additions. It is not
the total BASE width or the number of derived A-E features. All retained fields
enter the common updated BASE; this decision adds no commodity-subset search or
recipes beyond the approved 12. B/C transformation whitelists and total model
input counts still need freezing. Source units, aggregation/roll conventions,
historical availability/vintage and coastline extraction remain to be resolved;
this approved field choice does not certify those contracts or authorize a run.

## Approved B/C source whitelist — D32

The user adopted 28 continuous source series and six local conflict-count series:

| Continuous subset | Count | Fields |
|---|---:|---|
| Weather/vegetation | 4 | Rainf_f_tavg_mean, Tair_f_tavg_mean, EVI, gpp_mean |
| Nightlight | 1 | nightlight |
| Existing food economics | 2 | Food_CPI, Food_food_inflation |
| New ENSO/WB | 3 | nino34_anom, food_price_index_WB, food_inflation_wb |
| New Bloomberg | 18 | Exactly the 18 series approved in D31 |

The six count series are `event_count_battles`, `event_count_explosions`,
`event_count_violence`, `sum_fatalities_battles`, `sum_fatalities_explosions`,
and `sum_fatalities_violence`. They are separate inputs, not one combined
conflict index. D's existing three-category event total remains an internal
operand of its approved interactions.

For B, apply the approved 3/6/12-month means and population standard deviations
to each of the 28 continuous series, and the three window sums to each of the
six count series. This specifies 28*6 + 6*3 = 186 additional B input columns.
For C, apply all three approved formulas to these same 34 source series,
specifying 34*3 = 102 additional C input columns. Counts use absolute changes
and the same trailing standardization; a constant reference yields missing,
including a constant zero-conflict reference, under D27.

B and C together add 288 columns. This is a manifest count, not a measured
runtime, and excludes BASE, A, D and E. No additional windows or recipes are
introduced. Eligibility endpoints and missing-value rules remain those approved
for B/C and the still-to-be-frozen source-availability contracts.

Do not expand w5/w10 spatial conflict variants, existing rain/temperature
z-scores, nightlight_sd, general CPI, GDP, gini, population, distances or static
attributes through B/C. This exclusion concerns transformations, not automatic
removal of those fields from BASE; the complete retained-source BASE whitelist
still needs a separate decision. This approval remains planning-only.

## Legacy source-semantics and construction evidence

Two bounded read-only scouts searched source-building code and source metadata.
The main session checked the Gini interpolation, nightlight-SD filling and newer
z-score expressions directly. No master values were transformed or compared
against source tables; this is generator evidence, not a completed proof of
the selected master file's execution lineage.

`CODE` below is `Analysis/2.source_code/`, outside the repository where noted.

| Field | Evidence-backed meaning/time basis | Source anchor and limitation |
|---|---|---|
| CC | Control of Corruption percentile rank, CC.PER.RNK; annual country value | DATA/WBG/CC_percentile.csv:1-2; CODE/Step1_WBG/00_add_WBG_FEWS.ipynb:39,85,250; release timing unknown |
| GDP | GDP per capita PPP, constant 2021 international dollars, NY.GDP.PCAP.PP.KD | DATA/WBG/GDP.csv:2; WBG notebook:70,196; not total GDP |
| CPI | Consumer-price inflation, annual percent, FP.CPI.TOTL.ZG | DATA/WBG/CPI.csv:2; WBG notebook:54,176; not an index level despite its column name |
| gini | PIP country/reporting-year Gini; same-year duplicates averaged, intermediate years interpolated | WBG notebook:302-323; survey basis and release dates remain unresolved |
| market_access | Travel time to cities/ports in 2015, minutes; fixed coordinate-sampled layer | DATA/variable_construction_notes_description.xlsx, sheet1 C49/D49/H49; CODE/Step1_market_access/00_add_market_access_FEWS.ipynb:174-179,201,210,338; private asset binding unresolved |
| crop, range | Fixed ASAP cropland/rangeland masks | CODE/Step1_ASAP/extract_ASAP_FEWSNET.ipynb:65-66,110,238; raster reference years unresolved; codebook percentage semantics not yet reconciled to point extraction |
| pop | Population field joined from FEWS outcome records, with pop_source discarded | CODE/Step2_combine_all_dataset_and_impute/01_FEWSNET-final_process.ipynb:141,150-151,187-188; original estimate years and whole-table source mix unresolved |
| nightlight_sd | Monthly spatial dispersion of nightlight, not trailing temporal SD | CODE/Step2_combine_all_dataset_and_impute/00_combine_all_FEWSNET.ipynb:37,163-185 and final-process notebook:72-73; exact extractor-run binding and early sources unresolved |

Source-level transformations relevant to the clean-preprocessing contract:

- `CODE/Step1_WBG/00_add_WBG_FEWS.ipynb:302-323` expands each country's Gini
  reporting years and calls linear interpolation. An intermediate-year value
  can depend on a later survey endpoint. Month-shifting the result does not
  remove that dependence.
- `CODE/Step2_combine_all_dataset_and_impute/00_combine_all_FEWSNET.ipynb:172-176`
  sorts nightlight SD by region/year/month, then calls ungrouped ffill and fills
  remaining gaps with zero. This can transfer the previous region's ending
  value into the next region's early gaps. Further zero filling occurs at :193;
  nightlight mean also has zero filling at :161. E cannot recover original
  missingness from these filled values alone.
- The master z-score generation provenance is unresolved. The later
  `ASSEMBLY` sibling `assemble_latest_FEWSNET/02_preprocess_and_combine.ipynb`
  reads the old master (:20-21,42-43), drops its z-score columns (:145-146), then
  recomputes them for the 2025 combined output (:414). That newer code uses a
  12-row rolling mean without an explicit area group (:346-349) followed by
  whole-area mean/SD across its input (:361-381). It does not establish how
  the old master columns were created; do not assign the newer bug to the old
  file without lineage evidence.

These facts also prevent treating annual/snapshot source fields as necessarily
available at all earlier origins. Static-snapshot assumptions, annual release
lags and historical vintage limitations still require an explicit contract.

## Approved source-value retention and disclosure — D33

The user chose "先保留并记录" in response to the proposed reconstruction of
Gini and nightlight mean/SD. Retain the current master values for these fields
in both feature arms where included. The proposed source-table reconstruction
is deferred; do not execute it or silently remove the fields as a substitute.
Keep D8's source identity, keys, labels and cohort unchanged.

Record the following in source provenance and the eventual results report:

- Old FEWS Gini construction uses two-sided linear interpolation; an earlier
  value can depend on a later survey endpoint.
- Old nightlight-SD construction uses ungrouped forward filling after sorting
  regions and dates, which can propagate a value across region boundaries.
  Nightlight mean/SD also undergo zero filling in the old assembly code.
- These are verified code behaviors. The selected master has not been matched
  value-by-value to those executions, so do not assert that every relevant
  value is contaminated or that its exact generation lineage is established.

D23's calendar alignment and fitting-only imputation still apply to this
experiment's new processing. Origin checks for models, maps, calibration,
threshold selection and derived features remain mandatory, but are conditional
on the supplied source snapshots. Do not describe their success as verification
of entirely leakage-free historical source data or operational real-time
availability. Recording the upstream concern does not remove it.

For E, missing flags and ages refer to valid values visible in the supplied
source tables. Existing interpolated, forward-filled or zero-filled values
cannot be distinguished from original measurements without metadata; their
original missingness is not recovered. This qualification does not authorize
new source filling or change B/C/D missing-input rules.

D34/D35 fix monthly/annual endpoints and D49 fixes inherited static snapshots;
actual publication availability remains unverified. D33 did not decide the two inherited
climate z-scores; their later exclusion is separately approved in D46 below.
No preprocessing, source reconstruction or training was run.

## Approved monthly source endpoint — D34

For declared monthly covariates, the user adopted scheduled endpoint s_x(O)=O,
where O=T-H represents the end of the origin calendar month. The BASE value
comes from that exact source month; leave it missing if unavailable in the
supplied source table, rather than substituting an older month. B/C windows and
differences end at O under their fixed calendar rules; D uses the resulting
eligible operands. E's latest-valid-source lookup remains a separate age
calculation and does not fill the BASE value.

Add no further universal one-month lag or unverified source-specific publication
delays for these monthly fields. This matches the declared 4/8/12-month
source-to-target offsets, but establishes only source-month alignment using the
available snapshots. Actual historical releases, revisions and interpolation
remain unverified and must be disclosed alongside D33's retained-source issues.
This decision does not settle annual indicator lags, static reference vintages,
IPC/persistence observation eligibility or RF-label fitting cutoffs. It remains
planning-only and does not authorize data processing or model fitting.

## Approved annual indicator endpoint — D35

For GDP, CPI, CC and gini, the user adopted reference year Y=year(O)-1 for every target
row's own origin O. Use that source year's supplied annual value throughout
the origin year: origins in 2020 use the 2019 value, including the January
origin. Do not use origin-year values or automatically fall back to an older
year if Y is missing. Missing annual values proceed to D23's fitting-only
imputation. Apply the same annual alignment to both feature arms.

This uses the existing master/source snapshot; it does not rebuild Gini or
undo its retained interpolation under D33. Validate agreement of repeated
nonmissing annual values within the relevant area/reference year before
collapsing monthly copies. Conflicting values require resolution, not an
invented average or an arbitrary first row. An entirely absent annual value
remains missing.

For E, associate an annual value with December of its reference year. The
scheduled annual endpoint is December Y; a valid Y value has age one month
at the following January origin and 12 months at the December origin. If Y
is missing, E may calculate age from the latest earlier valid annual source
value without substituting that value into BASE. This is age of the supplied
reference-year value, not survey date or actual publication age.

CPI here means the general annual consumer-price inflation field FP.CPI.TOTL.ZG,
not Food_CPI or Food_food_inflation; those food series retain their declared
monthly classification. D48 separately fixes population; fixed geographic/
snapshot fields are not assigned this annual rule merely because they vary slowly.

Prior-year alignment is a stated retrospective convention. It does not prove
that all four indicators were published by January of the origin year, resolve
later revisions or remove future dependence in interpolated Gini. No additional
release-date reconstruction is proposed; those limits stay in the provenance
and report. This approval remains planning-only.

## Historical persistence lookup and missing-phase evidence

The main session read `PersistenceCorrectionExperiment/persistencecorrection/`
`persistence.py:1-296`. Its exact-O behavior is explicit at :5-12 and implemented
by shifting source keys to their target month (:145-159) and an exact calendar
join (:252-259); it is not a latest-observation lookup. It supports H=4/8 only
(:69-70), so H=12 requires the already-planned experiment-local extension.

The historical convention converts a missing raw phase to zero before the join
(:27-33,178-207), retaining `phase_missing` as an audit flag. The coverage gate
at :266-274 checks for an absent source month, not for a missing observed phase
in a matched row. Consequently, a 100% key-match rate does not prove a genuine
IPC observation for every persistence prediction. The historical origin-month
availability assumption is documented at :89-113. These are code facts; no new
source-support counts or model results were computed in this inspection.

## Approved IPC eligibility and persistence support — D36

The user adopted the following rule: at month-end origin O=T-H, treat observed IPC records with source month <=O as
eligible under an explicit source-month assumption. Actual publication/vintage
availability remains unverified. Block A uses the latest eligible valid record,
as already specified in D25. Do not substitute expert projections or adjusted
outcomes for observed IPC, and do not redefine the master target label.

Keep persistence's historical exact-origin lookup: use the observed record
at source month O for that area and the reconciled original binary crisis label
from that record. Require a valid observed phase and binary label under the
still-to-be-frozen value/key reconciliation contract. If the exact record is
absent or the observed phase is missing, persistence is unavailable; do not
convert missing phase to zero, fill it from an earlier record or invent a
prediction. Thus A may have a prior valid history even when persistence is
unavailable for that particular origin. Valid phase-range and source/master
reconciliation details remain to be finalized without importing ETH rules.

All metrics paired with persistence and all correction threshold/recipe scoring
use the same labeled-target keys with available exact-O persistence within
each horizon, across feature arms and probability variants. Report exclusions,
coverage and retained class/area/time support; do not claim full-cohort coverage
from source-key matching alone. Retain unavailable rows and reasons in evidence
rather than silently dropping them from the master. Standalone RF predictions
can still be retained outside that paired support. D37 separately fixes full
eligible-sample RF/calibrator fitting; D39 approves separate full-support RF
reporting as specified in research/reuse-and-boundaries.md.
Required empty development-score cells cannot be omitted or scored as zero.

This intentionally changes the old missing-phase-to-zero and all-key-
coverage contract, while retaining the exact-O persistence timing and original
target definition. RF fitting still excludes origin-month labels under D20;
using an O-month observation as history/persistence does not alter that mask.
This approval remains planning-only, not permission to execute experiments.

## Approved inherited climate z-score removal — D46

The old master's Rainf_zscore/Tair_zscore generation reference windows remain
unverified. The newer assembler's whole-area mean/SD and ungrouped rolling code
at the anchors above are not proof of how the old master was generated. D33
explicitly retains Gini/nightlight; this separate decision settles the two scores.

The user adopted omitting both inherited z-scores and any derived copies from the updated
BASE and the corrected original-feature reference. Keep their raw rainfall and
temperature source fields. Updated recipes with C enabled still receive D27's
specified trailing deviations; do not silently insert replacement anomalies
into BASE or the original-feature reference. This narrows legacy-feature parity
and must be disclosed, but avoids relying on an unverified normalization window.
It neither proves the old scores leaked nor resolves the separately retained
source limitations. Preserve source files and document the model-schema removal;
no source reconstruction is proposed. This approval remains planning-only.

## Assistance-field definition and stage mismatch — inspected evidence

The source metadata `DATA/Outcome/FEWSNET_IPC/Column Descriptions.xlsx`, sheet1
B9/C9, defines `fews_ha` as: "Tag denoting impact of humanitarian assistance.
Phase would be 1 higher without the estimated humanitarian assistance."
The main session verified that cell through the workbook XML. This is a FEWS
assessment of phase impact, not measured aid quantity. Separate near/medium
projection-assistance tags are not this field. B16/C16 defines the adjusted
phase as fews_ipc + fews_ha, assuming missing HA=0 for that adjustment; this
does not establish that a missing raw HA tag is an observed zero.

The main session streamed the chosen master: fews_ha contains 204,150 zeros,
13,209 ones and 811,881 empty values. In release code, this binary field fails
the within-admin/year nunique >2 dynamic test
(`GeoRFBaseline/src/feature/feature.py:62-65`), so no automatic scope lag is
generated and it is retained among L1 features (:116-122). Stage 1's exact drop
normally removes its original column (`GeoRFBaseline/config.py:247-250`,
`src/model/GeoRF.py:885-903`); missing feature names can bypass that drop
(:856-861). Stage 3's preparation only applies the optional strict-lag selector
(`scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py:920-930`), which retains L1
(`src/feature/strict_lag.py:19-25`). Thus binary inference can retain the target-
month HA tag even in strict-lag mode. These are code/source facts, not inspection
of historical fitted-model feature artifacts. Publication timing is unresolved.

Earlier statements that HA derivatives may survive exact drops described a
conditional possibility; the master binary values do not trigger this code's
automatic lag generation. The actual concern here is the original unlagged tag.

## Approved assistance-field exclusion — D47

The user approved excluding fews_ha and any inherited derivatives from model inputs in both
feature arms and all stages, consistently with the released Stage 1 default.
Keep source files, target definitions, phase history and persistence unchanged.
Do not add HA-specific missingness/age columns in E. This gives up a potentially
useful origin-available assistance-history signal; retaining such history could
be legitimate with explicit temporal alignment, but is not silently inherited
from Stage 3's current target-month behavior. This predictor-exclusion decision
is closed; it does not authorize implementation or model execution.

## Population source/support evidence — timing subsequently fixed in D48

A separate read-only scout streamed the master and `DATA/Outcome/FEWSNET_IPC/`
`FEWSNET.csv`; no feature table or model result was produced. Label presence in
these counts means nonempty fews_ipc, not independently validated target values.
In the master, 259,440 labeled rows have pop, 32,076 unlabeled rows also have
pop, and 737,724 unlabeled rows have no pop. No phase-labeled row lacks pop.
These full-panel proportions do not establish missingness at the approved
forecast origins. The field is not a fixed full-history snapshot: 5,709 of
5,716 areas with population have more than one value across time.

Population is nearly annual-constant on its sparse source months: 36 of 85,740
populated area/year groups differ within year, with maximum difference about
0.00005 and no group exceeding 0.001. The ledger shows the same 36-group pattern.
This is a numerical pattern, not a population-construction or rounding contract.
Its 302,948 valid data rows all have pop and pop_source is uniformly
`GPW Version 4 Revision 11 (2018)`; this version is not proof of a reference year
or publication date for each estimate. Ledger row 302950 is a malformed
single-field `System.IO.MemoryStream` record, excluded from those valid counts;
the future data contract must handle it explicitly rather than count it as data.

`CODE/Step2_combine_all_dataset_and_impute/01_FEWSNET-final_process.ipynb:141`
reads a differently named `FEWS October 2024 Update TrueBoundaries_12-09-24.csv`;
identity with FEWSNET.csv has not been verified. It drops pop_source (:150-151),
left-joins on year/month/admin_code (:187-188), and its stored missingness output
(:221) matches the master's 737,724 absent pop values. The next notebook's
population-density conversion concerns df_diffcountries, not demonstrably the
FEWS pop column (`02_add_FAO_prices_and_z_score_adjust_popdensity.ipynb:136-155`).
No annualization, tolerance, filling or alignment policy is approved by this
inspection; D48 below separately fixes the population selection/alignment policy.

The main session verified source metadata in `DATA/Outcome/FEWSNET_IPC/`
`Column Descriptions.xlsx`: B14/C14 defines pop only as "Population count";
B15/C15 defines pop_source as "Data source for population values". Neither
cell specifies an annual reference-year or historical release convention.

## Approved population alignment — D48

The user adopted treating pop as a low-frequency covariate using an explicit prior-year
snapshot convention, consistently in the original-feature reference and updated
BASE. For an origin O, let Y=year(O)-1. Use that area's last nonmissing population
record by source month within Y and retain its exact supplied value throughout
the origin year. Record the selected source month and within-year discrepancies;
do not average, round, interpolate or infer a fixed all-history population.
Existing duplicate-key validation still applies before selecting a dated record.
If Y has no valid population record, BASE stays missing until the approved
fitting-only imputation; do not substitute an earlier year's population.

For E, the scheduled annual endpoint is December Y, as for D35's annual
convention. Flag whether that year's selected value is missing. Measure age
to O from December of the latest eligible year with a valid selected value;
if there is no such year, leave age missing. This lookup for age does not fill
BASE. Preserve the actual selected row's source month separately from the
December age anchor. B/C continue to exclude population under D32.

This chooses how to use the sparse supplied records; their calendar year is
not verified as the demographic estimate's reference year. Prior-year alignment
also cannot establish historical availability of the 2018 GPW source snapshot.
Report these limitations. The trade-off is older population information in
exchange for a consistent annual input without inspecting later origin-year
records or silently carrying values across missing years. This decision is
approved for planning only; no preprocessing or model execution is authorized.

## Fixed-geography inventory and source-definition conflicts

A read-only scout inspected the master header and source code/metadata. The
main session checked the ruggedness ranking, slope asset, river property and
monthly market matching below. These establish candidate generator behavior,
not value-by-value lineage of the current master. `CODE` and `DATA` retain
their meanings above; workbook cells refer to sheet1 of
`DATA/variable_construction_notes_description.xlsx`.

| Master field(s) | Inspected generator/source | Interpretation or vintage limit |
|---|---|---|
| lat, lon | CODE/Step1_elevation/00_add_elevation_FEWS.ipynb:227,255 samples points | Coordinate generation and boundary vintage unverified; do not assert centroids |
| AEZ indicators | CODE/Step1_AEZ/00_add_aez_FEWSNET.ipynb:94,97-122,228 uses ESA/WorldCereal/AEZ/v100 and one-hot encoding | Seventeen fields listed below; paper year 2023 is not proof of geographic reference year |
| crop, range | CODE/Step1_ASAP/extract_ASAP_FEWSNET.ipynb:65-66,107-110,247 samples fixed ASAP v02 assets at points | Reference years unresolved; workbook H45:H46 says area percentages, not established by point sampling |
| elevation | CODE/Step1_elevation/00_add_elevation_FEWS.ipynb:124,203,234 uses GMTED2010 mea | Workbook C31 names SRTM instead; not all original observations necessarily date from 2010 |
| ruggedness | CODE/Step1_ruggedness/00_add_ruggedness_FEWS.ipynb:66-78,257 samples GMTED2010 med then computes percentile rank | Inspected code does not calculate ruggedness; workbook C47:D47 describes Nunn/Puga instead |
| slope | CODE/Step1_slope/00_add_slope_FEWS.ipynb:174-183 samples ALOS_topoDiversity constant; :378-379 fills missing with zero | Workbook C30/H30 calls this Geomorpho90m slope in degrees, conflicting with the asset |
| sg_cec_5-15cm, sg_cfvo_5-15cm, sg_nitrogen_5-15cm, sg_phh2o_5-15cm, sg_soc_5-15cm | CODE/Step1_ISRIC/00_add_ISRIC_data_FEWS.ipynb:228-245 samples SoilGrids property means at 250 m | Release/reference years and application of integer scaling unverified |
| distance_to_river | CODE/Step1_distance_to_rivers/00_add_distance_to_rivers.ipynb:137-159,169,405-406 samples HydroBASINS hybas_12 DIST_MAIN then renames it | Not a point-to-nearest-river calculation; workbook C48/D48 describes another source/meaning; official DIST_MAIN interpretation unverified |
| market_access | Workbook C49/H49 and previously inspected market-access code | Travel time to cities/ports, 2015 reference snapshot; publication cited as Nelson 2019 |

The exact AEZ fields are AEZ_10000, AEZ_12000, AEZ_15000, AEZ_17000,
AEZ_19000, AEZ_25000, AEZ_31000, AEZ_32000, AEZ_33000, AEZ_34000,
AEZ_36000, AEZ_38000, AEZ_4000, AEZ_40000, AEZ_43000, AEZ_7000, AEZ_9000.
There is no generic land_cover field. Geographic candidates are not yet the
frozen predictor whitelist; coordinate and identifier/routing roles remain
to be specified. Constant values alone do not prove source time semantics.

`market_distance` is a separate monthly-source field: the older FAO generator
restricts markets to the same year/month before nearest-market haversine
matching (`CODE/Step1_FAO_price/00_add_FAO.ipynb:991-1016,1033-1044`). Classify
this supplied field as monthly under D34, including D28's interaction operand.
Do not substitute WB match distance or remove it with FAO_price: D7 retains it.
Exact master lineage and historical market-list availability remain unverified.

## Approved fixed-geography snapshot policy — D49

For inherited fixed-geography covariates retained in the eventual explicit
schema, the user adopted preserving supplied snapshots in both arms throughout
development, RF history and evaluation, including origins earlier than a
layer's reference/publication year. Record the known 2015 market-access vintage
and unresolved dates/semantics above. This is a retrospective comparison
conditional on supplied layers, not verified real-time feature availability.

Do not reconstruct layers, silently substitute sources or apply unverified
unit conversions. Preserve inherited names as identifiers without claiming
their conflicting terrain/river descriptions are verified. Validate repeated
values before reducing a field to one static value per area; surface conflicts
rather than silently averaging or choosing a future row. E adds missing flags
only for included static fields. Monthly market_distance and D48 population
keep separate time rules. This policy does not select every geographic
candidate as a predictor or settle the new coastline extraction. It retains
geographic information at the cost of explicit vintage/interpretation limits.
The approval is planning-only; fitted-artifact and label cutoffs remain strict.

## Coastline inspection — extraction subsequently fixed in D50

A separate scout used existing Windows Python 3.12/rasterio for read-only
metadata/point checks; Linux rasterio had a NumPy ABI mismatch. No dependencies
were installed or extraction artifacts created. The TIFF has one int16 band,
36000 by 18000 pixels, EPSG:4326, Affine(0.01,0,-180,0,-0.01,90), global bounds,
nodata=None, all_valid storage mask, scale 1, offset 0, units/description unset.
TIFFTAG_IMAGEDESCRIPTION="distance from coast GMT intermediate" and
AREA_OR_POINT=Area. Pixel spacing is 0.01 degrees. The directory has only the
TIFF and IPCCH coordinate CSV, with no sidecar defining units/sign/sentinels.
The storage mask alone does not establish valid distance values.

Related public metadata at
https://pae-paha.pacioos.hawaii.edu/thredds/dodsC/dist2coast_1deg_land.das
and https://www.pacioos.hawaii.edu/metadata/dist2coast_1deg_land.html
declares km, GSHHS, NASA OBPG and NOAA NOS contributor Richard P. Stumpf,
a 2012-07-12 GeoTIFF-to-NetCDF conversion, and an original 0.04-degree grid
bilinearly interpolated to 0.01 degrees, with point uncertainty up to about
1 km. This is related-product evidence, not identity proof for the local TIFF.
Do not import its land-product _FillValue=0 into the local file or confuse its
upstream interpolation with IPCCH's point-sampling method.

Four rasterio.sample checks returned Addis Ababa (38.7578,8.9806) -500,
central Africa (20,0) -1043, Atlantic (0,0) +572 and (-30,0) +498 (lon,lat).
IPCCH CSV rows 2-4 also have negative inland values. These fit an inland-
negative/ocean-positive kilometre-scale interpretation, but do not prove
authoritative units/sign or justify an absolute-value transformation.

No IPCCH extraction implementation was located in the inspected source/assembly
directories. Existing files consume the field only
(`ASSEMBLY/build_deep_ipcch_features.py:66`,
`build_multiscope_ipcch_features.py:79`,
`03_correct_ipcch_targets.ipynb:191`). Reuse of the raster is supported;
reuse of a verified old extraction algorithm is not established.
These facts informed D50's subsequent extraction decision below. Local unit/sign
lineage remains unverified and qualified. No FEWS-wide extraction or model
fitting occurred.

## Approved coastline extraction contract — D50

The user adopted using the master longitude/latitude as WGS84 point coordinates
and reading the containing pixel from the specified EPSG:4326 TIFF, in (lon,lat)
order. Keep one fixed coastline_dist source value per validated area coordinate
in the updated BASE, reused across dates and recipes. Preserve the native signed
integer value, including zero and positive values, without taking absolute
values, rescaling units, interpolating pixels or deriving an administrative-area
summary. A positive value is retained with its diagnostic record, not treated
as an automatic data error from the empirical inland/ocean sign pattern.

Missing/nonfinite/out-of-range geographic coordinates, or coordinates whose
containing pixel is outside raster bounds, produce missing coastline input with
an explicit reason. Do not clamp/wrap/snap coordinates, invent a point or remove
the master row to hide the missing value. Honor an actual raster invalid mask;
do not import sentinel rules from the related but unverified NetCDF product.
Unexpected source-value anomalies require investigation in preflight rather
than silently inventing sentinel thresholds. Preserve point coordinates, raster
identity, pixel row/column, sampled value and validity reason in provenance.

Treat this as an additional fixed snapshot across the approved origins, with
the same retrospective-vintage qualification as D49. E may add its static
missing flag, never an age; B/C do not expand it. The original-feature reference
does not gain the coastline field. Document units as native raster units, with
the kilometre/sign interpretation supported only by related-product metadata
and the recorded sanity checks until exact source lineage is established.
The limitation is a point-location signal with unverified unit/sign provenance
and potentially limited representation of coastal or elongated administrative
areas. This extraction/risk choice is approved for planning only; no full
extraction, preprocessing or experimental execution is authorized.

## Identifier/predictor routing — inspected release behavior

A bounded read-only scout traced the default single-layer Stage 1 RF and
Stage 3 pooled/partitioned fits. The main session read feature preparation,
the exact-name drop implementation and the Stage 1 fit call. This is current
source evidence, not an audit of historical fitted feature artifacts.

`GeoRFBaseline/src/feature/feature.py:63-65` excludes admin/group IDs from
dynamic detection, but the actual predictor drop at :100 removes only date,
target, AEZ_group, ISO_encoded and AEZ_country_group. FEWSNET_admin_code,
lat/lon and source AEZ indicators remain. Stage 1's configured drop spells
IPC_admin_code instead (`config.py:247-251`), and
`src/model/GeoRF.py:840-909` has no implicit admin/coordinate/AEZ exclusion.
Its optional target/UID-name protection is configuration-dependent, not
evidence that admin identity must be a model predictor.

Stage 1 passes feature names and X through `app/main_model_GF.py:542-550`,
`src/model/GeoRF.py:420-427`, `src/partition/transformation.py:172-176` and
`src/model/model_RF.py:104-115,245-247,305`. The inspected downstream operations
select rows, preserve no-SMOTE input and add the approved recovery rows; they
do not strip another identity column before sklearn fit. Stage 3 fits directly
on the prepared X (`scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py:220-226,
245-261,920-930,993-1022,1151-1155`). Its mapping and X_loc paths (:185-209,
902-908) are separate metadata operations, not removal from predictors.

Thus the default release treats FEWSNET_admin_code as a predictor as well as
a join/group key. ISO_encoded/AEZ_group/AEZ_country_group are already metadata
only; source lat/lon and AEZ indicators remain predictors. Strict-lag mode
retains L1 and does not remove the admin ID (`src/feature/strict_lag.py:19-34`).
User config or nondefault nowcasting/L1-L2 branches can change exact behavior;
none was executed during this inspection. The main session also verified the
master's 88-column header and its seventeen AEZ indicator names above.

## Approved predictor identity boundary — D51

The user adopted using administrative/country/partition identifiers only as metadata
for joins, within-area validation, spatial mapping/routing and evidence keys
in both feature arms and all stages. Explicitly exclude FEWSNET_admin_code
from model inputs instead of inheriting its inclusion or relying on the
IPC_admin_code spelling in the old drop configuration. Do not add country,
area or learned-partition one-hot predictors as a replacement. Existing
ISO_encoded/AEZ_group/AEZ_country_group remain metadata only.

Retain the supplied lat/lon and seventeen AEZ category indicators as geographic
predictors, with D49's snapshot qualification. Their use as X_loc or grouping
metadata does not preclude this explicit predictor role. This changes the
corrected reference's raw admin-ID inclusion as well as updated BASE; report
the lost identity signal rather than claiming release feature-matrix parity.
It does not change polygon adjacency, partition learning or sample keys.
Calendar/history inputs and the remaining full source whitelist are separate
contracts. This modeling choice is approved for planning only.

## Approved complete source-predictor whitelist — D52

The main session reconciled the groups below against the actual
88-column master header, using names only and without constructing features.
Every header field is accounted for exactly once: 64 common retained source
fields, three old prices reserved for the reference, and 21 direct-input
exclusions. The user adopted this common retained inventory for both arms.

| Role | Count | Exact fields |
|---|---:|---|
| Geographic coordinates | 2 | lat, lon |
| AEZ source indicators | 17 | The seventeen exact AEZ names listed in the fixed-geography inventory above |
| Other fixed geographic layers | 12 | crop, range, distance_to_river, elevation, sg_cec_5-15cm, sg_cfvo_5-15cm, sg_nitrogen_5-15cm, sg_phh2o_5-15cm, sg_soc_5-15cm, market_access, ruggedness, slope |
| Monthly conflict counts | 18 | event_count_battles, event_count_explosions, event_count_violence, sum_fatalities_battles, sum_fatalities_explosions, sum_fatalities_violence, each with suffix empty, _w5 and _w10 |
| Other monthly sources | 10 | distance_to_nearest_acled, nightlight, nightlight_sd, EVI, market_distance, Rainf_f_tavg_mean, Tair_f_tavg_mean, gpp_mean, Food_CPI, Food_food_inflation |
| Annual sources | 5 | CPI, GDP, CC, gini under D35; pop under D48 |

The fixed group has 31 fields, the monthly group 28, and the annual group five.
Use D49's supplied geographic snapshots, D34's exact-O monthly values and the
approved annual conventions. This freezes inclusion of spatial conflict w5/w10
and nearest-conflict distance as source predictors; it does not add them to
B/C, whose D32 whitelist remains unchanged. Native units and documented source
construction/vintage limitations remain; do not rebuild sources to make names
match an unverified interpretation.

Approved source widths, before engineered inputs:

- Updated BASE: 64 common fields plus D31's 22 additions = 86 fields
  (32 fixed, 49 monthly, five annual).
- Corrected original-feature reference: 64 common fields plus FAO_price,
  WFP_Price and WFP_Price_std = 67 source fields
  (31 fixed, 31 monthly, five annual). Those three prices use D34. Its inherited
  calendar/history/window features are additional and remain to be frozen.

Retained old fields follow their order in the inspected master header; append
the updated additions in D31 table order (ENSO, two WB fields, coastline, eight
fertiliser, four energy, five crops, soybean-oil last price). The reference
retains its three price fields in master-header order. Materialize explicit
ordered names in the final manifest; do not use a permissive include-all rule
or infer roles from current within-year variation. Stage 1 and Stage 3 use the
same ordered source schema within each arm/recipe.

The 21 direct-input exclusions are unit_name, ADMIN0, ADMIN1, ADMIN2, ADMIN3,
FEWSNET_admin_code, ISO, ISO3, date, month, fews_ipc, fews_ipc_crisis, fews_ha,
fews_proj_near, fews_proj_near_ha, fews_proj_med, fews_proj_med_ha,
fews_ipc_adjusted, fews_proj_med_adjusted, Tair_zscore and Rainf_zscore.
Preserve necessary keys, time fields and outcomes separately for approved
routing/evaluation and declared calendar/history generation. Exclusion as a
direct source predictor does not remove the original response or authorized
IPC history features. D46/D47/D51 also exclude prohibited derived copies.

This approval fixes source schemas, not an E expansion or the reference's
inherited-transform schema. No extra recipe, new data source,
model fit or empirical feature selection was performed by this header check.

## Approved exact E whitelist and recipe widths — D53

The user adopted expanding exactly the 86 updated-BASE source fields fixed in D52:
one pre-imputation missing flag for every source field, plus an age in months
for its 49 monthly and five annual fields. The 32 fixed geographic fields
receive no ages. This gives 86 + 54 = 140 E columns. Source AEZ indicators and
lat/lon are explicitly included in the static missing-flag inventory; a valid
AEZ zero means a nonmember indicator, not missing. The E whitelist excludes
IDs, target/history/calendar columns and all A/B/C/D or legacy engineered
derivatives. It does not add E to the original-feature reference.

Apply the approved D29/D34/D35/D48 formulas without filling BASE: exact-O
missingness and latest eligible source-month age for monthly fields; prescribed
prior-year availability and December anchors for annual fields; missing flags
only for fixed layers. Source-table fills/revisions remain subject to D33's
interpretation limits. No-history age remains missing before D23 imputation.

Freeze column identities/order from BASE order, emitting x_missing followed
by x_age_months for dynamic x, and x_missing alone for static x. Retain the
declared schema even when a column happens to be constant in a fitting window;
deduplicate identical source/transform definitions, not accidental equality of
values in observed data. Do not expand aliases already excluded by D31.

The approved updated recipe widths are:

| Recipe | Input columns |
|---|---:|
| BASE | 86 |
| A | 94 |
| B | 272 |
| C | 188 |
| D | 90 |
| E | 226 |
| ABCDE | 526 |
| BCDE | 518 |
| ACDE | 340 |
| ABDE | 424 |
| ABCE | 522 |
| ABCD | 386 |

These add fixed block widths A=8, B=186, C=102, D=4, E=140 to BASE=86;
internal D operands do not add columns when B/C are disabled. This is manifest
arithmetic, not generated matrices or a measured runtime. The trade-off is
140 extra E inputs, some potentially constant/redundant in observed windows,
for a uniform source-availability inventory without data-dependent screening.
The E whitelist/count is approved for planning only, not model execution.

## Approved corrected-reference inherited transforms — D54

The main session checked `GeoRFBaseline/src/preprocess/preprocess.py:265-286,
288-295,623-627,645-648,666-670`. The IPC phase/binary lag families are 4/8/12;
calendar inputs are year/month indicators. Each engineering helper returns in
its first field iteration, so the actual extra families are WFP_Price preceding
4/12-row sums, nightlight preceding 12-row sum, and EVI lags 1-12. Broader lists
in the calling code do not establish additional generated features. A direct
read-only master scan confirmed original binary-labeled target years 2010-2024
and calendar months {1,2,4,6,7,10}, giving 21 inherited indicator categories.
This checks date/category presence, not target validity or predictive performance.

The user adopted this exact original-feature reference: D52's 67 source fields
plus these 42 inherited inputs, for 109 columns in every scope and stage.
Let T be the row's target month and O=T-H its own forecast origin.

| Family | Count | Approved calendar-corrected definition |
|---|---:|---|
| Target year/month indicators | 21 | year_2010 through year_2024, then month_1, month_2, month_4, month_6, month_7, month_10, evaluated at T |
| IPC phase/binary history | 6 | fews_ipc_crisis_lag_j and fews_ipc_lag_j for j in {4,8,12}, using a valid observed record exactly at O-j months and its reconciled original binary label |
| EVI history | 12 | EVI_l1 through EVI_l12, using exact source months O-1 through O-12 |
| WFP price sums | 2 | WFP_Price_m4=sum(x(O-4),...,x(O-1)); WFP_Price_m12=sum(x(O-12),...,x(O-1)) |
| Nightlight sum | 1 | nightlight_m12=sum(x(O-12),...,x(O-1)) |

Build these on each area's complete monthly source grid before selecting labeled
targets, and compute every historical fitting row relative to its own O.
Calendar indicators use T because the target date is known at forecast time;
their fixed inventory is a declared calendar design, not a fold-fitted estimate.
History offsets use O, not T: for H=12, IPC at T-4 would be after origin and
cannot enter this corrected reference. Missing exact IPC/EVI endpoints remain
missing; do not substitute a nearby observation. Each sum requires every one
of its W source values and excludes O, preserving the original shift(1) intent
at the newly explicit origin endpoint. Missing inputs leave a missing sum
before D23; no across-area rolling or imputed source-history sums are allowed.

Source BASE values still obey their own approved endpoints. The inherited
sums ending at O-1 deliberately differ from updated B windows ending at O;
do not silently make their formulas identical. Referenced IPC records must
pass the same still-to-be-finalized phase/key reconciliation as D36, while
the history features retain exact lookup rather than A's latest-record rule.

Order the reference inputs as its 67 source fields, ascending year indicators,
the six ascending month indicators, binary IPC lags 4/8/12, phase IPC lags
4/8/12, EVI lags 1-12, WFP sums 4/12, then nightlight sum 12. This totals
67+21+6+12+2+1=109. Preserve fixed columns even when inactive in one fitting
window. Do not re-run automatic horizon-lag generation on already aligned
sources/derivatives, retain target-month dynamic copies, repair the early-return
loops to generate more reference families, or add the new A-E blocks to this arm.

The trade-off is a clearly defined origin-safe downstream reference with older
IPC histories than some target-relative legacy fields, not exact reproduction
of the flawed legacy matrix or proof of historical source availability. This
full reference-transform contract is approved for planning only, not execution.
