# Compact input sources, runtime and availability facts — 2026-10-10

> Consolidation note (2026-10-10): this file preserves the evidence and decision sequence. Earlier wording that a scientific detail was open is historical; accepted D1–D29 are in ../decisions.md and the current final-review contract is ../design.md. Unverified source/asset/resource facts remain unverified, with checkpoints in ../implement.md. No execution is authorized by these notes.

Read-only scouts inspected manifests/code and selected source ID columns. No feature matrices, training, installation, EE calls or source edits. The coordinator checked the inclusive fit-mask/weight code, the compact runtime gate and the explicit observation-vintage limitations. Hashes marked manifest are not freshly verified large-file content hashes.

## Available source identities

Path prefixes (exact spelling retained from source manifests):

- A = `/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/1.Source Data/assembled_IPCCH`
- S = `/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/IPCCH_shared_folder`
- D = `/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Processed_Dataset/IDP_DTM`
- C = `A/model_ready/compact_climate_weather_oracle_v1/compact_climate_weather_oracle_v1_manifest.json`
- P = `A/model_ready/origin_safe_climate_idp_v1/origin_safe_climate_idp_v1_manifest.json`

| Role | Source | SHA256 recorded in manifest | Reference |
|---|---|---|---|
| Ordinary monthly dynamics / reported histories | A/interim/IPCCH_2026_target_corrected_nino34_wbfood.csv | a91719e8e603e11b58e3e44c9a2f4378e4218e1ac408afe03ca54d1b50fb0484 | C:74-75 |
| Monthly climate / anomalies | S/climate_monthly_2015_2026_MODELING_READY.csv | 8082b72ea5fa5c30a7b975b89c1fbbb4530f27a9ce6c5ba4d0e1a4f923320c89 | C:78-79 |
| Seasonal climate | S/climate_2015_2026_MODELING_READY.csv | f024a66c8979fb4a8c66fba1f04e7e69355fa75b499793d29001a146d8c2958e | C:82-83 |
| National IDP | D/output/idp_admin0_monthly.csv | a1ebb82a8e14fc0fb8103d6086217b93da351fc4a8f7fcfbe9eaa44a9df602e1 | P:32-34; current hash matched |
| Area/country mapping | A/country_area_id_lookup.csv | e2baf6ae9481b42b127db5ce81f3c53ad784bb08ae56dfa56cea74e416b44c90 | P:36-38 |
| Coordinates / raw population panel | A/raw/IPCCH_2026_completed.csv | ae696087c3bbb280537ae269a05924133acdb51060d31290523404fa8a717673 | P:40-42; same source as P6 inputs.json:10-15 |

Current hashes of small provenance files: C `456360536a3929e1571d96ba9f16e2f0de54bc1b20019f797bbe62c0c89732ca`; P `3b479c80498e9f2d6d20b88936fce468f297d2e0ccd6f0cb32c61a7470a209bf`; sibling latest extension `results/experiments/compact_climate_weather_oracle_eval_2026_jan_apr_v1/inputs/extension_manifest.json` is `913fae43952ced0982a77abc6fe97b3a0325ac7bf8bfae8f2c98380fb766ab70`. Large-file bytes must still be rehashed and staged before implementation uses them as frozen inputs.

Static29 is inherited from original scope files (P:44-58), checked constant within area by `../IPCCH/scripts/preprocessing/build_origin_safe_climate_idp_inputs.py:107-113`, then retained in the compact tables. It is not regenerated from the monthly climate source. The existing H0 compact table contains static rows for all 6,224 canonical valid-target admins, but popdensity has no finite value for 4,916 admins; do not silently replace it from a different snapshot. The other 28 columns have at least one nonmissing value per canonical admin. Choosing a consistent static source requires per-admin invariance checks; H0's dynamic columns are never a substitute for H1.

## Coverage and old-table incompatibilities

- Read-only scans found no missing canonical admin IDs in interim (1,219,868 rows / 6,227 admins), monthly climate (871,780 / 6,227), seasonal climate (149,448 / 6,227), old H0 compact (52,521 / 6,226), or country lookup. This is area-level coverage, not complete origin-level values.
- The old compact/cohort omits **14 canonical target keys**, including main H1/H3/H6 six keys and H12 five. Main missing keys are `(3076,2023-09)`, `(3161,2024-09)`, `(4686,2025-08)`, `(4687,2025-08)`, `(4688,2025-08)`, `(4697,2025-08)`; H12 excludes the first by schedule. Old compact outputs cannot alone fill this task's main cohort.
- This reflects an inherited reported-phase gate, not permission to delete keys: `build_origin_safe_climate_idp_inputs.py:226,232` filters valid overall_phase and its own share criteria; `build_compact_climate_weather_oracle_inputs.py:281-287,361-365,396-397` inherits it. D15 requires rebuilding on population-derived canonical keys.
- Interim spans 2010-01..2026-04 and climate 2015-01..2026-08 (manifest C:2577-2618). Origins before the climate start occur in 717 of all valid training/target rows for H1/H3/H6 and 978 for H12; this is expected missingness, not a reason to delete old labels. No canonical origin is later than interim 2026-04.
- IDP has 3,487 rows, 1,104 nonmissing reports and 54 report countries. Of canonical admins, 5,683 have an ISO3 represented in report countries and 541 do not (including 31 missing ISO3). This does not establish time-specific availability. Preserve missing values according to the feature recipe.
- Existing compact/origin helpers validate only H0/3/6/12 (`compact_features.py:40,136-147`; `origin_safe.py:23,64-66,152,258`). An explicit H1 adaptation is required; old ready tables are not the new cohort authority.

## Existing numerical and EE environments

Read-only `python -B`/importlib.metadata probes, with no package imports requiring data or installations:

| Interpreter | Python / numpy / pandas / sklearn / XGB | Other relevant packages |
|---|---|---|
| /home/swl007007/.venvs/ipcch-geo/bin/python | 3.12.3 / 2.4.4 / 3.0.3 / 1.8.0 / 3.2.0 | scipy1.17.1, geopandas1.1.3, shapely2.1.2, rasterio1.5.0; no EE/pyarrow |
| /opt/earthengine/bin/python | 3.12.3; no above numerical packages | earthengine-api1.7.36, google-api-python-client2.198.0 |
| Windows Store python3.12.exe | 3.12.10 / 2.2.6 / 2.2.3 / 1.6.1 / 3.0.0 | EE1.5.8, pyarrow19.0.1, geopandas1.0.1, shapely2.1.0, rasterio1.4.3 |

The current ipcch-geo numerical versions match the compact frozen gate (`compact_features.py:468-475`); Windows versions do not. The final-review design selects this existing WSL numerical environment plus the separate EE interpreter through files, without mutating either environment or changing old runs. P0 must recheck and write the complete runtime lock; the table is a read-only probe, not a completed implementation lock or installation.

## Observation-time convention and remaining limitations

- P6 explicitly says `observation-month-end <= origin; not verified release timing` (`IPCCHGeoXGBExperiment/config/experiment-contract.json:30`). Its old 36-month window includes O, but that window is not inherited by D7.
- Sibling `origin_safe.py:313-325` uses label month `<= min(O,T-1)` and weights `0.5**((O-U)/24)` with no lower training cutoff. For every accepted H>0 this includes U=O, with weight1. Twenty-four months is a half-life, not a truncated history window.
- Compact origin-aligned weather/conflict features include O; rolling windows end at O, same-month z references use earlier years (`compact_features.py:204-209,273-296`). Reported histories use latest legal phases with source month<=O (`origin_safe.py:143-190`). These are inputs, not the population-derived class truth.
- IDP validates reportingDate but chooses by its report month<=O (`origin_safe.py:232-282`). Seasons finish when `gs_end_date_exclusive <= first day of O+1`; no added release delay (`compact_features.py:322-367`). The fixed season-duration comparison is calendar information, not future realized weather.
- `build_compact_climate_weather_oracle_inputs.py:225-235` explicitly retains unverified historical releases and supplier climatology-fitting samples and retrospective static snapshots. Thus observation-time safeguards do not certify historical real-time availability or revision-free weather standardization. No oracle means no explicit O-after features; it does not resolve these provenance limits.

Final design uses the D29 month-end/release-proxy interpretation explicitly. Support counts and the accepted D27 bridge calendar use this same convention. No new publication lag is introduced and historical availability is not claimed verified.

## Accepted D29 — observation-month retrospective availability

User accepted on 2026-10-10: retain the existing observation/report-month-end proxy for this first retrospective mechanism experiment. A forecast origin O means the end of month O; for H>0, training outcomes may have target month U<=the fit origin, and each training/prediction row's dynamic inputs may use observation/report months through that row's own origin. Origin month is inclusive. A completed season may have exclusive end date at the first day of O+1. Use reported-phase predictors under the same month proxy, while preserving population-derived truth and QC. The fixed 2017 AlphaEarth annual interval must end before its first permitted 2018 origin; D2's product-version limitation remains separate.

This decision preserves D3–D7/D27 calendar and fitting identities. It does not add a blanket publication lag or certify that month-O IPC, weather, conflict, IDP and just-ended seasons had actually been published at month-end. Published/revised historical data and static snapshots remain retrospective inputs. Supplier anomaly/climatology fitting periods are also unverified; report that limitation rather than asserting the entire upstream data-processing history was blind to later information. New target supervision, feature-window construction, preprocessing and map/recipe selection must still obey their explicit cutoffs; unknown publication provenance does not excuse violating those enforceable boundaries.

The alternative is a historically released-version protocol: first obtain source-specific publication/revision records and historical snapshots, then redefine availability and recompute support/calendar/feature compatibility as needed before freezing a plan. Those records have not been shown available, so a universal one-month lag is not a verified substitute. This would materially expand the data-reconstruction scope. The choice here concerns the scientific interpretation and data contract, not permission to execute either pipeline now.
