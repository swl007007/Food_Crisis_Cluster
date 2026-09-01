# Exact data boundary and current GeoRF lineage

## Boundary used in this audit

The characterized dataset is the parsed `ISO3 == "ETH"` slice of
`FEWSNET_forecast_unadjusted_bm.csv` before `load_and_preprocess_data()` is
called. It has 88 assembled fields. It is not upstream raw-source data: several
fields are already derived or aggregated, including z-scores, IPC binaries,
provider projections, conflict windows, AEZ indicators, prices, and macro data.

## What production currently does after this boundary

1. `load_and_preprocess_data()` drops 11 fields, filters to non-null
   `fews_ipc_crisis`, encodes `ISO`, parses dates, creates target/phase lags at
   4/8/12 months, drops `fews_ipc`, creates year/month dummies, applies three
   feature-engineering helpers, and creates AEZ groups.
2. Each feature-engineering helper currently returns inside its outer feature
   loop. Consequently only the first eligible feature in each list is handled.
3. `prepare_features()` detects time-varying columns, creates a scope-specific
   lagged copy, but retains the original contemporaneous column. It drops the
   target/date/group encodings, then performs max-plus out-of-range imputation.
4. The current feature-drop configuration later removes `month`, `fews_ha`,
   and `years` when present, but does not list `FEWSNET_admin_code`, `lat`, or
   `lon`.

These are observed code behaviors. They create forecast-availability and model
capacity questions but are not, by themselves, measured leakage or confirmed
GeoRF overfitting.
