# Source evidence — 2026-09-19

Read-only profiling of the exact user-corrected source; no transformed dataset or model was created.

## Pinned source
- Path: `C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\assembled_IPCCH\raw\IPCCH_2026_completed.csv`
- SHA256: `ae696087c3bbb280537ae269a05924133acdb51060d31290523404fa8a717673`
- Size: 1,782,567,753 bytes; 1,219,868 rows, 143 columns.
- Keys: `(admin_code,year,month)`, 0 duplicates; 6,227 areas, 53 countries.
- 2010–2026 monthly scaffold (180–196 rows per area); usable labels are sparse.
- Full header: `source_columns.txt`.

## Target facts
Exact fields are `phase1_percent` through `phase5_percent`, `estimated_population`, `overall_phase`; no preexisting `ipcch_food_crisis` or direct P3+ field.
The assembly target-correction spec documents 0–1 proportions. The raw observed ranges also support this scale, but contain invalid values. Use sum of Phase3/4/5, never substitute `overall_phase >=3`.

Presence patterns Phase1..5:
- 00000: 1,176,039 rows
- 11111: 43,551
- 11000: 107
- 11100: 86
- 11110: 85

All three target components complete: 43,551 rows. Of these, strict decimal P3+ >.20 gives 15,213 positives before quality exclusions, and 2,699 are exactly .20. Each area has 1–25 complete candidates (median 6); full predictor months are not full label months.

Concrete quality findings:
- 752 rows have all five proportions zero. Many have positive estimated_population; this is not a valid observed phase distribution.
- 16 rows have P3+ >1 (2025-12 Lebanon/Palestinian Territory and 2026-03 Haiti).
- One phase1 value exceeds 1: admin_code=100341, 2019-09, Nigeria, phase1=1.4554283798668646.
- 363 complete-candidate rows have estimated_population=0.
- 85 rows have P3/P4 but missing P5 (70 Central African Republic, 12 Nigeria, 2 Chad, 1 Burkina Faso). The file does not establish missing=zero.
- overall_phase includes 0,6,9 and missing values; it is not the target in this task.

Main-thread exact-decimal candidate audit (`candidate_label_policy_audit.json`) tests a PROPOSED, NOT APPROVED rule: all five components present and in [0,1], sum within inclusive [.99,1.01], estimated_population>0. This retains 42,110 labels (14,643 positives, 2,690 exact-.20 negatives), excludes 1,441 complete candidates. Exactly 1,440 complete candidates fail the sum tolerance. Reason counts overlap. The raw source is unchanged; this only writes aggregate planning evidence.

## Geography and country
`admin_code` matches all 6,227 IDs in:
- `assembled_IPCCH/spatial/unique_area_id_lat_lon.csv` (`area_id,lat,lon`)
- `assembled_IPCCH/country_area_id_lookup.csv` (`area_id,iso3,country,country_code,country_en`)
- `assembled_IPCCH/spatial/ipcch_admin_geometry.shp` attributes (`admin_code,lat,lon`).

No missing/extra IDs or duplicate IDs were found in those three files. Polygon correctness is not established by ID coverage. Its builder `assembled_IPCCH/code/build_ipcch_admin_geometry_shapefile.py:191-201` permits unrestricted nearest-neighbor fallback; fallback counts/distances were printed but no saved evidence was located. Geometry needs a bounded audit before partition learning.
Raw ISO3 is missing on 6,076 rows /31 Cote d'Ivoire areas; do not drop these by default. Raw numeric coordinate differences within an area are at most about 5e-11; use the keyed spatial reference, not exact float-string identity.

## Source lineage boundary
User explicitly selected raw; no corrected source is substituted. `../assemble_latest_IPCCH/organize_ipcch_ml_data_folder.py:334-341` classifies this file as raw and keeps target-corrected variants in interim. `../assemble_latest_IPCCH/03_correct_ipcch_targets.ipynb:191-203` identifies a separate coastline-distance baseline, corrected GeoJSON and seven correction fields. Saved output records 21,317 assignments per field with non-targets preserved, but this is not proof of 21,317 differing values in the currently selected raw file. Raw-to-corrected correspondence remains unverified and is not an instruction to switch inputs.

## Evidence method
The profile agent read selected columns as strings and checked numeric domains, missingness, keys, calendar and SHA256. The main thread independently read the actual header and recomputed the proposed label policy with Decimal from CSV strings (full selected-column read). This avoids floating equality ambiguity at .20 and tolerance endpoints. No benchmark performance was evaluated.

## Superseding Q1 decision
The preceding .99–1.01 audit describes the initial rejected proposal, not the
current policy. User broadened the raw-total window to [.90,1.10], allowed only
P5 to be missing (fill zero), retained [0,1] component bounds, and requested
proportional scaling. The population>0 condition is retained. Normalize before
thresholding. Current audit: `approved_label_policy_audit.json`; 42,695 valid /
15,206 positive, 84 P5-fill recoveries and 82 classification changes from scaling.
Current behavior is specified in PRD R1.

## Geometry validity follow-up — 2026-09-20
Read-only check with Windows Python3.12 / GeoPandas1.0.1 / Shapely2.1.0:
6,227geometries, EPSG:4326, none empty/missing,253invalid. Reasons:212
self-intersections,30ring self-intersections,5nested shells,6too-few-points
components. Main thread independently reproduced these counts. No geometry
repair or adjacency generation occurred. Q8g experiment-local repair is approved; see
convergence-check.md. This does not resolve upstream area-boundary provenance.
