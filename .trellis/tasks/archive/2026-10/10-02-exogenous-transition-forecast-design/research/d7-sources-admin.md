# Research: D7 item 1: sources, hashes, administrative keys

- **Query**: Hashes, headers, key uniqueness, admin crosswalk and date coverage for the FEWS NET sources (D7 data readiness)
- **Scope**: internal (local source files, metadata only)
- **Date**: 2026-10-02
- **Protection**: No IPC value, phase, description, HA or projection-value column was loaded from any file. Every CSV that carries outcome columns was read with a `usecols` whitelist, and `rd()` asserts against forbidden tokens. One exception: the non-outcome coordinates file was read with all columns (`d7_admin_keys.py:41`, `cols=None`); it has no IPC fields. The only exemption is the `projection_start`/`projection_end` date fields. GeoJSON files were stream-hashed only. Notebook outputs were not opened; only the source cells were printed. No label distributions or presence counts were computed.

## Probes (all under `research/probes/`)

| Script | Output |
|---|---|
| `d7_hash_files.py`: streaming SHA256 of raw bytes, plus size and mtime | `d7_hash_results.json`, `d7_hash_log.txt` |
| `d7_admin_keys.py`: key, date and name metadata only, including a struct-based DBF reader (geopandas, pyogrio and dbfread are not installed in the scout's Linux probe environment; the frozen Windows environment has geopandas 1.0.1 and pyogrio 0.11.0, see `data-readiness.md` section 1) | `d7_admin_keys_log.txt`, `d7_admin_keys_results.json` |
| Ad hoc commands: `cmp FEWSNET.csv "FEWS October 2024 Update TrueBoundaries_12-09-24.csv"`, a duplicate-key lookup (inline python, keys and names only), and a notebook source-cell printer | shown inline below |

## 1. Hashes (verified)

`$SRC` = `.../Analysis/1.Source Data`; `$IPC` = `$SRC/Outcome/FEWSNET_IPC`.

| File | Bytes | mtime | SHA256 (prefix) | vs pinned |
|---|---:|---|---|---|
| `$SRC/FEWSNET_forecast_unadjusted_bm.csv` | 716,303,754 | 2025-11-05 | 611f9e776380e28d | MATCH |
| `$IPC/FEWSNET.csv` | 51,422,500 | 2025-03-14 | 8fdd4cca6f6ba26b | MATCH |
| `$SRC/FEWSNET_admin_code_lat_lon.csv` | 244,419 | 2026-01-21 | a06be85849bb726a | MATCH |
| `FEWS_Admin_LZ_v3.shp` | 98,094,620 | 2025-04-11 | 3aba66a6fbf6b2a8 | MATCH |
| `assembled/..._2025_combined.normalized-v1.csv` | 773,553,094 | 2026-07-20 | 510375f58cd835e6 | MATCH (sidecar value independently recomputed) |
| `.shx` / `.dbf` / `.prj` / `.cpg` | 45,844 / 7,942,848 / 306 / 5 | | 16061b9a97add122 / 2175dc97d3c6969c / 94ba1b071fb00c93 / 3ad3031f5503a440 | no pin |
| `assembled/..._bm_2025.csv` | 48,907,058 | 2026-05-04 | 5c8aefc116232175 | no pin |
| `assembled/..._bm_2025_combined.csv` | 774,685,964 | 2026-05-04 | 41f02be985d86fbf | no pin |
| `assembled/...normalized-v1.audit.json` | 2,150 | 2026-07-20 | 1c37232629dd11f6 | no pin |
| `$IPC/2025_2026_FEWSNET.csv` | 6,102,353 | 2026-04-24 | a64ed4bb7e3fda95 | no pin |
| `$IPC/FEWS_2025.csv` | 239,478 | 2026-05-01 | 4987356a80bbeff6 | no pin |
| `FEWS October 2024 Update TrueBoundaries_12-09-24.csv` | 51,422,500 | 2025-03-14 | 8fdd4cca6f6ba26b | **byte-identical to FEWSNET.csv** (same hash; `cmp` returned 0) |
| `geoidentifier_fews.csv` | 860,127 | 2025-04-28 | 7881a58b4c253615 | |
| `FEWS_scaffold.csv` / `FEWS_scaffold_fixed.csv` | 157,327,183 / 832,646 | 2025-04-28 | 0bde26af086bae9f / 220fe21e5af8257f | |
| `append_2025_2026.ipynb` / `scrape_fewsnet.py` | 5,952 / 6,021 | 2026-05-01 / 2026-04-29 | 33bbb4d272530a16 / 605db9da13f98ae7 | |
| `fewsnet_ipcphase_2025-01_2026-04.geojson` | 3,287,083,173 | 2026-04-29 | 53652ffc5be90224 | hash only |
| `fewsnet_chunks_2025_2026/` (45 files) | 35 files are 45 bytes, all with the same hash (7d09e532fc380630) (reconciled against `d7_hash_results.json`: 45 total, 10 non-empty; an earlier draft said 33), so they are presumably empty FeatureCollections. Non-empty files: CS 2025-10, CS 2025-12, CS 2026-01, CS 2026-02, ML1 2025-01, ML1 2025-09, ML1 2026-01, ML2 2025-01, ML2 2025-05, ML2 2026-01 | | see `d7_hash_results.json` | hash only |

Full 64-character digests are in `probes/d7_hash_results.json`.

## 2. Headers and key structure (verified, key columns only)

Headers were printed with `head -1` and are recorded in the session log. Key facts:

- The pinned panel's keys are `FEWSNET_admin_code`, `date` (format `YYYY-MM`) and `month`. It has no year column. It carries the label columns `fews_ipc`, `fews_ha`, `fews_proj_*` and `fews_ipc_crisis`.
- `FEWSNET.csv` keys: `country, admin_code, year_month, year, month, admin_name`.
- `..._2025.csv` keys: `admin_code, year, month`. Note the name is `admin_code`, not `FEWSNET_admin_code`.
- The combined and normalized-v1 panels key on `FEWSNET_admin_code`, `date` (`YYYY-MM-DD`) and `month`. They have 88 columns, in a different order from the pinned panel.
- `2025_2026_FEWSNET.csv` (UTF-8 BOM) has the columns `fnid, country, country_code, scenario, reporting_date, projection_start/end, geographic_unit_full_name, id, ...`.
- `FEWS_2025.csv` columns: `admin_code, fews_ipc, year, month, fews_ipc_crisis`. Its keys are written as floats, e.g. `4711.0`.
- The shapefile DBF fields are `cov_start, cov_end, report_mon, unit_name, ADMIN0-3, LZCODE, admin_code(N10), admin_name, ISO, adm0_name, AREA1/2`. The DBF contains no IPC fields. It has 5,718 records, its CRS is WGS84 and its encoding is UTF-8.

| File | Rows | Distinct ids (range) | Duplicate key rows (id, ym) | Months |
|---|---:|---|---:|---|
| pinned panel | 1,029,240 | 5,718 (0–5717) | 0 | 180 (2010-01 to **2024-12**, no gaps), 5,718 rows per month |
| FEWSNET.csv | 302,949 | 5,716; 1 null-id row (country=`System.IO.MemoryStream`) | 0 | 53 tri-annual months, 2009-07 to 2024-10, 5,716 rows per month |
| panel_2025 | 91,490 | 5,718 | **4** | 2025-01 to 2026-04 (16 months); 5,719 rows in 2025-10 and 2026-02 |
| combined | 1,120,730 | 5,718 | **4** (2 in 2025-10, 2 in 2026-02) | 196 months, 2010-01 to 2026-04 |
| normalized-v1 | 1,120,728 | 5,718 | 0 | 196 months, 5,718 rows per month |
| FEWS_2025.csv | 10,086 | 5,716 (+1 null) | 6 (code 2996) | 2025-10: 4,481; 2026-02: 4,367; null year/month: 1,238 |
| scaffold | 1,029,240 | 5,718 | 0 | 2010-01 to 2024-12 |
| coords / geoidentifier / scaffold_fixed / DBF | 5,718 each | 5,718 | 0 (id unique) | n/a |

The duplicated key is `admin_code 2996`, which appears twice in both 2025-10 and 2026-02 in panel_2025, the combined panel and FEWS_2025 (inline lookup).

## 3. Admin crosswalk (verified unless marked)

- **There is one integer code space, 0–5717.** The pinned panel, panel_2025, the combined panel, normalized-v1, the coordinates file, geoidentifier, the scaffolds and the DBF `admin_code` all hold the identical set of 5,718 ids.
  - On all 5,718 codes, `unit_name` is equal between the panel, the DBF and geoidentifier.
  - geoidentifier and the coordinates file agree on lat/lon to within 1.4e-14.
- **FEWSNET.csv and FEWS_2025.csv** hold 5,716 ids, a subset of that space. They are missing **216 and 2786**.
  - FEWSNET.csv `admin_name` equals DBF `admin_name` for 5,714 of 5,716 codes.
  - Codes **1667** and **2996** each carry 2 names in FEWSNET.csv.
  - The name `Northwestern Pastoral Zone, Kerio Delta, Turkana Central, Turkana, Kenya` maps to **both 2995 and 2996**, so there are 5,719 distinct (name, code) pairs.
- **Every 2025 `fnid` is unmatched by direct key.**
  - The `fnid` values are FEWS NET strings such as `TD2022C3070105`. There are 6,807 distinct fnids in 12,086 rows.
  - No fnid equals any `admin_code`, and none appears in DBF `LZCODE` (LZCODE is blank in the rows sampled).
  - geoidentifier_fews.csv has no fnid column, so it cannot bridge the two either. The only bridge is the **name**.
- **How FEWS_2025.csv was derived** (`append_2025_2026.ipynb`, source cells 5–14):
  - Cell 5 takes the distinct `(admin_name, admin_code)` pairs from FEWSNET.csv.
  - Cell 6 filters the 2025 file to `scenario_name == "Current Situation"`.
  - Cell 10 left-joins `previous_df` to `new_df` on `admin_name == geographic_unit_full_name`.
  - Cell 12 derives year and month from `reporting_date`.
  - Cell 14 creates `fews_ipc_crisis = (fews_ipc >= 3)`.
  - No dedup step exists, and no cell writes `FEWS_2025.csv` (no `to_csv` call). *Inference*: the file was written outside the saved cells. Its columns and row count match a keys-only replay: 10,086 rows, 1,238 with no date, and 4 duplicated (id, date) rows.
  - *Code inference, no values read*: unmatched rows get NaN `fews_ipc`, and cell 14 casts those to `fews_ipc_crisis = 0`.
- **Name-join coverage** (CS rows only):
  - 2025-10: 5,573 rows, 4,480 matched. 2026-02: 5,867 rows, 4,366 matched. Across both dates, 1,502 distinct full names are unmatched.
  - Some mismatches come from renamed admin hierarchies, e.g. Ethiopia's `Shebelle` vs `Shabelle` and `Kilbati` vs `Kilbati /Zone2`.
  - Per country, unmatched/total: DRC 345/345, CAR 104/104, El Salvador 59/59, Honduras 50/50, Lebanon 26/26, Syria 14/14, Venezuela 25/25, Ethiopia 665/1161, Burkina Faso 68/133, Niger 43/225, Sudan 30/391, Nigeria 21/305, Mali 20/116, Somalia 19/217, Malawi 8/84, Kenya 3/642, Chad 2/92. AFG, BDI, CMR, GTM, HTI, MDG, MOZ, SSD, YEM and ZWE are fully matched.
  - Uganda (318 FEWSNET.csv codes) has no CS rows. *Inference*: Uganda appears only in the 2024-10 ML1/ML2 rows.
  - DRC (345 codes) has CS rows, but none of them match by name.
- **Boundary-version changes (ambiguous joins)**:
  - 502 full names map to more than one fnid. Examples are Niger `NE2020C…` vs `NE2026C…`, plus Sudan 2025/2026 and Burkina Faso 2021/2026.
  - The fnid vintage year (characters 3–6) is mixed: 2011–2026, plus `IDP*` and `REF*` units.
  - `dataseries_name` carries "(From MM/DD/YYYY)" boundary-validity dates.
  - The DBF boundary coverage is `cov_start` 06/07-2021 (and 09-2020) to `cov_end` 09-2021. *Inference*: the 2021 boundary set predates the 2025/2026 FEWS NET units for several countries.
  - The fnid-to-name direction is unique: 0 fnids map to more than one name.

## 4. Raw 2025 file metadata (dates and scenarios only)

Rows by scenario and reporting date:

| Scenario | Reporting date | Window | Rows |
|---|---|---|---:|
| CS | 2025-10-01 | 2025-10-01 to 10-31 | 5,573 |
| CS | 2026-02-01 | 2026-02-01 to 02-28 | 5,867 |
| ML1 | 2024-10-01 | 2024-10-01 to 2025-01-31 | 323 |
| ML2 | 2024-10-01 | 2025-02-01 to 2025-05-31 | 323 |

- Status: Published/Collected 12,040; Not Available 7; Not Projected 39.
- Classification scale: IPC 3.1 for 12,072 rows; IPC 3.1-aligned for 14.
- Key `(fnid, scenario, reporting_date, projection_start)` has 0 duplicates.
- `scrape_fewsnet.log` reports `merged 23628 unique features` and a final GeoJSON of **1101.1 MB**. The file on disk is **3,287,083,173 bytes (about 3,135 MiB)**. *Inference*: the log does not describe the final GeoJSON that is on disk now. The log's chunk sizes do agree with the chunk files, e.g. CS 2025-10 at 199.3 MB.

## 5. Pinned vs combined panel (verified, keys only)

- The two panels overlap on 180 months (2010-01 to 2024-12). Over that range their (id, month) key sets are **identical** (1,029,240 keys each, 0 differences either way), and `unit_name` is equal for all 5,718 ids.
- The combined panel adds 2025-01 to 2026-04. Its 2025+ keys equal panel_2025's.
- normalized-v1 has the same key set as the combined panel but without the 2 duplicated rows.
- **The pinned panel's last month is 2024-12.**

## Unresolved

1. Values are not certified. For the 2996 duplicates (and the two-name code 1667), it is not established which source row was kept by normalized-v1 or which rows were dropped. The producer's `source_row_number` tie-break rule was not inspected.
2. Whether FEWS_2025.csv was written from exactly these notebook cells (no write cell exists).
3. Codes 216 and 2786: they are in the panels but absent from FEWSNET.csv. Their origin and whether they carry labels in the panels were not checked.
4. The 1,502 unmatched CS names need a proper boundary crosswalk, either spatial overlay or an fnid-vintage table. No local fnid-to-admin_code table was found. The provenance and validity of the DBF boundaries for 2025/2026 are also open.
5. The GeoJSON and chunk files were hashed but not parsed, so feature-level and scenario coverage is known only from the log. The size mismatch between the log and the file on disk is unexplained.
6. Historical label-presence counts per admin and month were not computed: they were optional and skipped for time.
