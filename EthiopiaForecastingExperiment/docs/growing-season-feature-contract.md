# Ethiopia previous-growing-season feature contract

Status: approved and amended on 2026-09-04. This contract defines a separate
monthly lookup table; implementation remains separately gated.

## Goal and source

- Add previous-growing-season summaries without using observations after the
  forecast origin month.
- Calendar authority: FAO GIEWS *Country Brief on Ethiopia*, reference date
  2025-10-03: <https://www.fao.org/giews/countrybrief/country.jsp?code=ETH>.
- Treat the calendar as a fixed climatological calendar for all years. It does
  not represent year-specific phenology.
- Use the frozen 1,040-admin Ethiopia cohort and `FEWSNET_admin_code`; do not
  substitute another area identifier.
- Monthly non-SPI source: frozen ETH baseline, SHA-256
  `25e458ac6fbdb27c9b264ada111bfe8fc38c6f2c376bd023dc8fb1c4d15eb855`.
- Monthly SPI source: `data/interim/era5_drought_spi/ethiopia_spi_monthly.csv`,
  SHA-256
  `d18e311aba521c680975a39ea8193c3b8f29d47f4ed5bb18fd0a6a0d6e40b547`.

Implementation ownership is fixed: the single-purpose generator is
`EthiopiaForecastingExperiment/prepare_growing_season_lookup.py`, and it writes
`EthiopiaForecastingExperiment/data/interim/growing_season/ethiopia_previous_growing_season_monthly.csv`.
No alternative runtime path or configuration surface is required.

## Calendar profiles

| Profile | Season | Months |
|---|---|---|
| `meher_only` | Meher | June-December |
| `belg_meher_bimodal` | Belg | February-July |
| `belg_meher_bimodal` | Meher | June-December |
| `pastoral_bimodal` | Gu/Genna | March-May |
| `pastoral_bimodal` | Deyr/Hageya | October-December |

Assignment uses exact `ADMIN1 + ADMIN2` labels in the two named sets below;
every unmatched admin uses `meher_only`. Belg and Meher are aggregated
independently even where their fixed calendar months overlap; selection still
depends only on which season most recently ended.

- `pastoral_bimodal`: Oromia/Borena, Guji, West Guji; Somali/Afder, Daawa,
  Doolo, Korahe, Liban, Shabelle.
- `belg_meher_bimodal`: Tigray/Southern, South Eastern; Amhara/North Wello, South Wello,
  North Shewa (AM), Oromia; Oromia/East Hararge, West Hararge, East Shewa,
  Arsi, Bale, East Bale; SNNP/Guraghe, Hadiya, Halaba, Kembata Tibaro, Siltie,
  Yem Special; all Sidama admins.
- Every other admin uses `meher_only`.

The frozen assignment yields 650 `meher_only`, 301 `belg_meher_bimodal`, and
89 `pastoral_bimodal` admins. The fallback includes 39 Afar and 50 Somali
admins not named by the southern-Somali rule; this is an accepted approximation
of the Country Brief mapping.

## Aggregation

Monthly inputs are joined on exact `FEWSNET_admin_code + date` before seasonal
aggregation. Produce these seven values independently:

- mean of `SPI_1`, `SPI_3`, `SPI_6`, and `SPI_12` from the frozen compact SPI
  table;
- mean of monthly `gpp_mean` and `Tair_f_tavg_mean`;
- sum of monthly `EVI`.

For mean features, publish a value when at least `ceil(2/3 * expected_months)`
season months are non-null; otherwise publish null. For `sum(EVI)`, require all
expected months because a partial sum is biased downward. Do not interpolate,
fill, normalize, standardize, or aggregate the four SPI scales together.

## Monthly lookup and no-leak rule

Write one separate table covering 2010-01 through 2024-12 and unique on
`FEWSNET_admin_code + year + month`. For each lookup month `F`, select the most recent season satisfying
`previous_season_end < F`. A season ending in July therefore becomes available
in August, not July. Keep the complete admin-month lookup grid; months before
the first available completed season retain null seasonal values and are not
dropped or filled.

The table contains:

- seven model features named `previous_season_avg_SPI_1`,
  `previous_season_avg_SPI_3`, `previous_season_avg_SPI_6`,
  `previous_season_avg_SPI_12`, `previous_season_avg_gpp_mean`,
  `previous_season_avg_Tair_f_tavg_mean`, and `previous_season_sum_EVI`;
- audit-only fields: `calendar_group`, `previous_season_start`,
  `previous_season_end`, one `expected_months` count, and a separate observed
  month count for each of the seven variables.

Season identity and coverage fields do not enter the model. The seven values
are added alongside existing monthly features; they replace nothing.

For target month `M` and horizon `H`, first compute
`forecast_origin_month = M - H`, then join this table at
`F = forecast_origin_month`. Never choose the previous season relative to `M`.
The existing horizons remain fs0=1, fs1=4, fs2=8, and fs3=12 months.

## Acceptance checks

- Calendar assignment covers exactly the frozen 1,040 admins and reproduces the
  frozen group counts above.
- The lookup covers all 180 months from 2010-01 through 2024-12, yielding
  exactly 187,200 unique admin-month rows.
- The expected 2018-2024 lookup has 87,360 rows and 100% availability for all
  seven features. Across 2010-2024, 10,352 null rows are expected solely from
  the unavailable pre-2010 season history.
- Monthly lookup keys are unique; all source joins preserve the cohort key.
- Every published season statistic uses only dates from its recorded season;
  every recorded season ends strictly before its lookup month.
- Each aligned row's seven seasonal values exactly match the lookup row for its
  `FEWSNET_admin_code + forecast_origin_month` for all four horizons.
- Source hashes, calendar rules, row accounting, coverage counts, output hashes,
  and four alignment steps are appended to `data_lineage.jsonl`.

## Out of scope

FAO ASIS rasters, crop-specific or year-varying calendars, season-category
predictors, replacement/removal of existing features, imputation, and model
fitting are excluded.

`SPI_3/6/12` remain provider-native rolling indices before seasonal averaging;
they are not interpreted as SPI fitted from season-total precipitation.
