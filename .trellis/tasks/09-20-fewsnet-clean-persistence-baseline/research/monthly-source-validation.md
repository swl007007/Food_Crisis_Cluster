# Monthly source validation — read-only evidence, 2026-09-20

Two bounded scouts inspected Bloomberg workbooks/exports and WB/ENSO sources.
No FEWSNET panel was joined, feature matrix generated or model fitted. Counts
and reconstructions below are scout-reported; the main session checked WB
conflicting rows/helper selection and independently verified the all-empty
TTF column. Source-generation execution and historical publication availability
are not established by numerical agreement.

Paths: DATA = Analysis/1.Source Data/; B = DATA/Bloomberg_food_and_derivative/;
H = Analysis/2.source_code/Step5_Geo_RF_trial/assemble_latest_IPCCH/.

## Bloomberg monthly construction

Every nonempty numeric CSV value matched the arithmetic mean of available
numeric daily observations for its series/month in the corresponding workbook
(rel_tol=1e-12, abs_tol=1e-10). Missing daily entries were not zero-filled.
This is full numerical reconstruction evidence, not a located export script.

| CSV under B | Workbook / sheet | Months | Range | Nonempty approved values |
|---|---|---:|---|---:|
| bbg_fertiliser_monthly_051326.csv | bbg_fertiliser_051326.xlsx / Value format | 197 | 2010-01..2026-05 | 1,430 |
| bbg_oil_and_gas_monthly_050826.csv | bbg_oil_and_gas_050826.xlsx / FOOD_CRISIS_BLOOMBERG_OIL_GAS_F | 355 | 1996-11..2026-05 | 710 |
| bbg_soybean_oil_futures_monthly_050826.csv | bbg_soybean_oil_futures_050826.xlsx / value and format only | 317 | 2000-01..2026-05 | 317 |
| bbg_staple_food_x1_monthly_050826.csv | bbg_staple_food_050826.xlsx / FOOD_CRISIS_BLOOMBERG_STAPLE_CR | 355 | 1996-11..2026-05 | 1,773 |

No duplicate monthly keys or interior missing calendar rows; no duplicate
daily series/date keys in these workbook tables. CSV values have no nonfinite,
zero or negative entries; missing cells are empty strings. May 2026 is partial:
workbook dates end May 7 (fertiliser/staples), May 6 (oilgas), May 11 (soybean).
Filename dates are not reliable actual final quote dates. These dates are beyond
the approved evaluation endpoint but matter for interpreting source snapshots.

Important coverage within the approved 18 fields:

- bbg_TZTX2_Comdty has zero observations across all 197 months. The main
  session independently checked this with csv.DictReader. D31/D52/D53 freeze
  the field/schema: this discovery does not authorize dropping it or its
  approved derived/missingness inputs. The existing imputer's all-missing
  fallback was subsequently verified below; retain it under D23.
- bbg_GCFPAMBS is valid 2012-10..2022-02, then missing. GCFPPOBA has a
  2022-04..2023-03 gap; GCFPURBS lacks 2022-03/04. Several fertilizer series
  begin 2012-10 or later; DAP India begins 2015-06. Corn lacks 2017-09/10
  (staple CSV physical lines 252-253). The frozen source list is unchanged.
- Fertiliser workbook main!A4 uses BQL px_last, start=2010-01-01,
  end=2026-05-07, per=D, xlfill=B. The inspected Value format sheet supplied
  the numerical reconstruction, not a fresh execution of that formula.
- Soybean workbook Worksheet!B1 is BO1 COMB Comdty; B4=D, B5=USD, B7=PX_LAST.
  Its daily PX_LAST has 13 '#N/A N/A' entries; ignoring nonnumeric daily
  entries reproduces the monthly export. The separate bid is still excluded.

Staple daily contract_month changes are directly observed: Corn workbook
B20/D20 = 1996-11-29 / C Z96, B21/D21 = 1996-12-02 / C H97; the recent series
also switches MAY 26 to JUL 26 at 2026-05-01. Raw oilgas workbook
raw/FOOD_CRISIS_BLOOMBERG_OIL_GAS_FUTURES.xlsx retains contract_month in F;
the processed workbook omits it. No inspected evidence fixes vendor roll-day,
volume/open-interest rules or back/ratio adjustment. The x1 filename and
generic tickers cannot establish those definitions.

Soybean currency USD is explicit, but its quote denominator is not. Fertilizer
metadata contains truncated mt labels; an explicit USD/mt label found on an
unselected column cannot define approved columns. Oilgas/staple units remain
unverified. Preserve field identities and distinguish native quotes from claims
of comparable physical units. The workbook-to-CSV export script and earlier
raw-to-processed workbook execution were not located in the bounded search.

## ENSO validity

DATA/NOAA_ENSO/nina34.anom.csv has 948 unique consecutive months from 1948-01
to 2026-12. All date/numeric entries parse. Its header identifies NOAA ERSST v6
Nino Anom 3.4 and sentinel -99.99, but the actual 32 sentinel rows are -9999:
1948-1949 and 2026-05..12. Valid values cover 1950-01..2026-04, range -2.45..2.72.
H/add_nino34_to_ipcch.py:40-65 handles both sentinels and rejects duplicate
months; :91-92 joins many-to-one on supplied year/month, without computing O.
The caller must enforce D34. Local material does not prove physical units or
historical per-month release timestamps. No alternate ENSO file was adopted.

## WB values, identity and actual co-location ambiguity

DATA/WB_RTP_price/filter.ipynb cells 0/2/3/4/5 (zero-based) read raw
WLD_RTFP_mkt_2026-04-20.csv, select fields, compute (opening+closing food index)/2,
copy inflation to food_inflation_wb, remove OHLC and export. There is no market
aggregation, filtering or interpolation in these cells. Both raw and derived
have 755,647 rows; year/month/lat/lon agree row-for-row. Recomputed retained
values have zero mismatches under NumPy isclose(equal_nan=True); the two
inflation aliases agree, including missingness. This does not prove byte-exact
floating arithmetic or an executed export lineage.

There are 232 months (2007-01..2026-04), 3,325 raw geo_id values and 40 ISO3s.
Raw (geo_id,year,month) is unique. Coordinates are absent on 8,873 rows; all
other coordinates are in range. Valid-coordinate candidates total 746,774.
Source price is missing on 20,045 rows, inflation on 59,945; among coordinate-
valid candidates those counts are 19,971 and 59,391. Numeric columns contain
no infinities, keys contain no fractional year/month values or invalid months.
Price ranges 0.01..555.295; inflation -77.58..950.96. Negative inflation is
valid source content, not a missing code. No -99.99/-9999/-999 values were
found in retained WB fields; the helper defines no WB sentinel substitution.

Valid-coordinate (lat,lon,year,month) has 828 duplicate groups (1,656 rows),
808 with conflicting retained values/missingness, 20 equal. Price conflicts
occur in 656 groups, inflation conflicts in 777. These are distinct markets,
not duplicated raw market IDs. Four coordinate pairs recur:

| Country | Markets | lat,lon |
|---|---|---|
| COG | Ouenze / Total | -4.25,15.28 |
| MLI | Abeibara / Adjelhoc | 18.44,1.42 |
| SEN | Dakar / Tilene | 14.68,-17.45 |
| SYR | Damascus / Shrebishat | 33.51,36.3 |

The main session verified derived CSV physical lines 141788 and 142340:
both 2011-03 at (-4.25,15.28), with prices 0.8 and 0.77. Missing-coordinate
duplicates are excluded from the 828-group count.

H/add_wb_food_price_to_ipcch.py:91-138 filters candidates by valid coordinates
only, groups by source month and queries BallTree haversine k=1. It accepts
distance <=100 km with radius 6371.0088 km. No country filter, co-location
check or explicit distance-tie rule is present. It copies the chosen row's
fields even if missing, without looking for a farther nonmissing value.
Sorting source rows alone cannot be assumed to define BallTree's tie contract.

Raw spatially_interpolated is zero on every row; that flag alone cannot prove
all values are observed or rule out temporal estimation. last_survey_point
ranges 2025-07..2026-04; 5,973 source records postdate their market's stated
last survey. The raw snapshot's exact estimation methods, index base and
inflation period/units remain unverified. The derived file drops market IDs,
survey/confidence metadata and ISO3; no new predictor is authorized from them.

Scout-computed SHA-256 identities:

| Source | SHA-256 |
|---|---|
| nina34.anom.csv | ce67f5c52a2a4695f82ee9acdb4a5dc69eb42f34af6cf1e5801fa7ac20118ca5 |
| wb_food_price_index.csv | bc5f310c008d69cf67539633a4af5ef5da2ba724897269b602aed66771ad523a |
| WB filter.ipynb | beb04a1bd13f4b3cb18ad235b68043113e12dcf70e8c49d9f29e59c1048a0e77 |
| WLD_RTFP_mkt_2026-04-20.csv | cd81efe9ec6c3c1ab7aa14d1a1dbd6092606c015608cec354beccfd960192730 |

## Approved WB distance-tie policy — D64

Restore raw geo_id as metadata only via validated raw/derived row lineage:
check source identities, row count, ordered dates/coordinates and retained
value/missingness agreement before attaching IDs. Do not join on nonunique
coordinates or silently replace the approved derived values with other prices.

Keep D9's same-source-month, <=100 km, cross-country nearest-market rule.
Among candidates with exactly equal computed haversine distance, select the
lexically smallest raw geo_id. Copy both approved value fields from that
single market; retain missing values rather than choosing a different market.
Do not average co-located markets or select by price, model outcomes, confidence
or completeness. Persist chosen ID, distance and tie provenance; IDs never
enter predictors. Unverifiable lineage or conflicting duplicate market-month
keys stops assembly. This is deterministic selection, not a claim of superior
market representativeness. The user adopted this policy for planning; no
assembly/model execution is authorized by this research.

## Existing all-missing imputation behavior — D23 preservation

The main session inspected GeoRFBaseline/src/customize/customize.py:16-149.
OutOfRangeImputer defaults to max_plus, multiplier=100 and fallback_strategy=mean.
For a column with any observed fitting values, it fills missing entries with
100*max, except that max=0 uses 100 (:101-105). A wholly missing fitting column
gets 0.0 under its default fallback (:130-147), with missing column statistics.
The name mean does not mean estimating a mean from validation or future data.

Retaining D23's existing method therefore already defines an all-missing
fitting-column value, including the retained empty TTF source and any missing
derived inputs. This is a model-input sentinel, not an observed price of zero.
Compute B/C/D/E from valid pre-imputation values first. Preserve declared column
widths and record all-missing fitting columns and their fills. No extra feature
drop, source filling, imputer search or user decision is needed for this case.
Do not claim max_plus always places values above the observed range: when its
maximum is negative, multiplication remains the retained formula rather than
the separate extreme_high strategy. This is code inspection, not a new fit.
