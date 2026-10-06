# IPCCH GeoXGB climate feature perturbation

## Goal

Swap the monthly climate/vegetation inputs of the IPCCH cumulative-share GeoXGB rich561 recipe for the IPCCH_shared_folder monthly and growing-season climate data, keep every other part of the feature engineering and the pipeline unchanged, rerun the full pipeline, and compare against `p6-formal-20261004b`. Exploratory perturbation; 2023–2026 has already been inspected.

## Decisions (grilled 2026-10-05)

- D1 Replace all four original monthly climate columns: drop `EVI_mean`, `GPP_mean`, `Rainf_f_tavg_mean`, `Tair_f_tavg_mean` and the 12 `EVI_mean_lag{k}_asof` columns.
- D2 Monthly source `IPCCH_shared_folder/climate_monthly_2015_2026_MODELING_READY.csv`. Use only the 14 `*_month_ensmean` columns: prcp_anom, prcp_z, rainy_days, cdd, tmean_anom, tmax_anom, hot_days_p95, gdd, edd, sm_z, spi03, spei03, ndvi_anom, evi_anom. Same engineering as the replaced columns: value at origin month O; the 12 EVI lags move to `evi_anom_month_ensmean_lag{k}_asof` (month O−k). Source-specific, `n_sources`, `ens_sd` and QA columns are not used.
- D3 Growing-season source `IPCCH_shared_folder/climate_2015_2026_MODELING_READY.csv`. For each season type s1 and s2 separately, take the latest season whose last day (`gs_end_date_exclusive` − 1 day) falls in a month ≤ O; use its 14 `*_gs_ensmean` values plus its age in months (O − end month). 30 columns. Seasons in progress at O are never used.
- D4 Keep static `crop`; do not add `crop_fraction`. Keep every other original93 column, history468, and all derivatives unchanged. Resulting recipe: 103 base + 30 growing-season + 468 history = **601** columns.
- D5 Calendar unchanged. The climate data start 2015-01, so earlier origins/lags are NaN under native XGBoost missing handling; report the affected row counts.
- D6 Copy the frozen package into a sibling `IPCCHClimateGeoXGBExperiment/` (namespace `ipcch_climate_geoxgb`). Change only the feature construction, schema, input registry, column counts and an optional run-root override. Same contract constants, recipes, gates, seeds, pinned Windows runtime. Original package and runs untouched.
- D7 Full rerun: preflight, prepare, learn-map (maps re-learned on the new features), predict, report, replay. Run directory outside Dropbox.
- D8 Comparison with P6 on identical keys per H and period: GeoXGB, pooled and persistence panels for both runs; new−old crisis-F1 for GeoXGB and pooled with the original paired country bootstrap (2000 draws, seed 42, main period only); map/recipe changes listed. Supplementary 2026 point-only.
- D9 Commit package, spec and small evidence on a new branch. No trellis-audit, no push, no PR.

## Acceptance Criteria

- [x] New package tests pass under the pinned runtime; original package unchanged.
- [x] Input hashes for both climate files recorded; prepared manifest reports 601 columns and NaN coverage of the new columns.
- [x] All pipeline stages exit 0; replay passes with zero failures.
- [x] Comparison table and bootstrap written to task evidence; results note states limits (exploratory, viewed period, maps re-learned so map and feature effects are confounded in GeoXGB; pooled isolates the feature effect).

## Out of scope

Hyperparameter changes, new seeds, using the audit-flagged per-source columns, MLP task changes, release-vintage availability checks for the climate data (observation-month-end assumption is inherited).

## Outcome (2026-10-05)

Executed as specified; see [results.md](results.md) and `evidence/`. Main-period crisis F1 changed by −0.0007 to +0.0063 (pooled) with every country-bootstrap interval including zero; Geo−pooled stays within ±0.0006.
