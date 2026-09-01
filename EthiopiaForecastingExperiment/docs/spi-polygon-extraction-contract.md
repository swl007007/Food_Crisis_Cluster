# Ethiopia ERA5-Drought SPI polygon contract

Status: Phase A approved; offline preparation only. CDS download remains blocked
until the user approves the exact prepared manifest hash.

## Scope

- Source: Copernicus CDS `derived-drought-historical-monthly`,
  ERA5-Drought/DRYFALL v1.0 reanalysis, reference period 1991–2020.
- Variables: provider-native SPI accumulation periods 1, 3, 6, and 12 months.
- Calendar: source-native monthly values from 2010-01 through 2024-12. No local
  lag, rolling calculation, interpolation, filling, or other feature engineering.
- Cohort key: exact frozen Ethiopia `FEWSNET_admin_code` set joined by calendar
  `date`. Production data and non-Ethiopia polygons are out of scope.

## Spatial and quality contract

- Geometry: the `FEWS_Admin_LZ_v3` shapefile bundle, filtered by the exact 1,040
  cohort codes and transformed to EPSG:4326 if necessary.
- Preserve the provider-native 0.25-degree grid. Do not reproject or interpolate
  raster values.
- Use polygon fractional intersection and spherical grid-cell area weights.
- A cell is valid only when SPI and P0 are finite and `P0 < 0.66`; normality is
  diagnostic only.
- Publish a polygon-month value only when valid intersection coverage is at
  least 0.95. Otherwise retain a null with an explicit missing reason.

## Outputs

- The compact source table is unique on `FEWSNET_admin_code + date` and exposes
  only `SPI_1`, `SPI_3`, `SPI_6`, and `SPI_12` for the working-panel merge.
- A separate long QA sidecar retains scale, coverage, valid areas, P0,
  normality, status, missing reason, source-member hashes, and spatial rules.
  QA fields never enter the working or aligned model tables.
- Raw ZIPs, extracted NetCDF members, download inventory, hashes, completed
  manifest, compact source table, and QA sidecar are retained under the isolated
  Ethiopia SPI raw/interim directories and remain out of Git.
- Actual data addition and fs0/fs1/fs2/fs3 alignment will append new records to
  `data_lineage.jsonl`; Phase A does not mutate that ledger.

## Execution gate

- The prepared campaign has 64 requests: for each of four scales, 15 annual SPI
  requests plus one 2020 calendar-quality request; 816 NetCDF members are
  expected in total.
- Maximum concurrency is two; the minimal runner executes serially. There is no
  automatic retry, cleanup, overwrite, fallback, or resume.
- Failed `.part` files and other evidence are retained. A resume requires a new
  unresolved-request manifest and separate approval.
- `download` requires the exact embedded manifest SHA-256 supplied explicitly on
  the command line. Phase A may run only `prepare`, `validate`, and offline tests.

## Prepared Phase A identity

- Manifest: `manifests/ethiopia_spi_campaign_2010_2024_v1.json`
- Embedded approval SHA-256:
  `80d48c48f017620445a89bcd73af1fef9d94c49378f43d8f80dab799b74f4537`
- Manifest file SHA-256:
  `11b7a26bbebcceaa847a174c0d906f95f2f610dbffb66769d06fff96ee38f42d`
- Request area `[north, west, south, east]`: `[15.0, 32.75, 3.25, 48.0]`.
- This identity records offline preparation only; it does not authorize download.
