# P0 foundation evidence — 2026-10-04

State: **awaiting_supervisor_foundation_freeze**. No model was fitted, no label,
feature, map or prediction was built, and no scientific claim is made. This
document reports P0 infrastructure checks only; it is not a task success claim.

- Audit run `7ced754ea36c48c0a6d24ba2a17addec` (active), base_sha
  `6c98f73c34272101ddc124cf20ed5ef338563646`. Lifecycle and the user-authorized
  10-03 native archive are recorded separately in `audit-start-evidence.md`
  (housekeeping commits `8160f68`, `63ca2d7`).
- Executor: Herdr pane `wN:p2`, session `afc97c82-11d4-4a79-ae5c-21d99cac82b5`,
  terminal `term_65d044bdb69412`.
- Numerical runtime: `C:\Users\swl00\AppData\Local\Microsoft\WindowsApps\python3.12.exe`
  (Python 3.12.10, Windows-11-10.0.26200). Linux Python was used only for Git,
  JSON generation of path/hash configs, and hashing of reference sources.

## Package inventory (`IPCCHGeoXGBExperiment/`)

| Path | Responsibility |
|---|---|
| `ipcch_geoxgb/__init__.py`, `errors.py`, `artifacts.py` | package roots; ContractError/NotImplementedPhaseError; sha256, write-once JSON, immutable run dirs |
| `ipcch_geoxgb/contract.py` | load + validate the four configs against accepted R17/R19/R28/R29/R43/R45/R46/R47/R49 values |
| `ipcch_geoxgb/runtime.py` | runtime probe from distribution metadata (no XGBoost import) |
| `ipcch_geoxgb/geography.py` | bounded copies of IPCCH GeoRF ID/lookup/coordinate/geometry loaders; direct frozen-cache validation |
| `ipcch_geoxgb/preflight.py` | read-only input preflight (identities, saved tables, geometry, cache, raw keys, diagnostics) |
| `ipcch_geoxgb/cli.py`, `__main__.py` | `python -m ipcch_geoxgb {validate-config,runtime-probe,preflight,prepare,learn-map,predict,report}` |
| `config/experiment-contract.json` | frozen calendar, quartet/decoder, G1–G4/L1–L2, gates, support, budgets, bootstrap |
| `config/feature-schema.json` | ordered rich561, `schema_version ipcch-geoxgb-rich561-ge020-v1`, `>=0.20` semantics |
| `config/inputs.json` | 13 read-only inputs with bytes/sha256 (generated from task `input-manifest.json`) + structural expectations |
| `config/runtime-lock.json`, `requirements.txt` | Python 3.12.10; numpy 2.2.6, pandas 2.2.3, xgboost 3.0.0, geopandas 1.0.1, pyogrio 0.11.0, shapely 2.1.0, pyproj 3.7.1, pytest 9.1.0 |
| `config/source-provenance.json` | 9 copied/adapted spans (path, sha256, commit, lines, destination, adaptation), 13 pending sources (P1–P5), 3 removed legacy dependencies |
| `tests/test_*.py`, `pytest.ini` | 64 focused infrastructure tests |
| `runs/` | git-ignored run outputs |

The geometry I/O engine is explicitly `pyogrio` (`geography.GEOMETRY_ENGINE`).
`pyproj` is used through geopandas CRS handling, not imported directly. XGBoost
is pinned but not imported by any P0 code path.

## Commands and results (all on the pinned Windows runtime)

| Command | Result | Evidence |
|---|---|---|
| `python -m ipcch_geoxgb validate-config` (cwd repo root) | exit 0, status passed | `evidence/P0-cli.log` |
| `python -m ipcch_geoxgb runtime-probe` | exit 0, `matches_lock: true`, no mismatches | `evidence/P0-cli.log` |
| `prepare` / `learn-map` / `predict` / `report` | each exit 3 `NOT IMPLEMENTED`, nothing written | `evidence/P0-cli.log`; test asserts unchanged `runs/` and cwd |
| `python -m pytest -v` (cwd package root) | **64 passed**, 1 warning (geopandas-internal `shapely.geos` DeprecationWarning) | `evidence/P0-pytest.log` |
| import probe, cwd = repo root | all 7 modules import; repo modules outside package: none; forbidden (`config`, `src`, `prepare_data`, ...): none; xgboost loaded: false | `evidence/P0-import-probe.log` |
| import probe + CLI, cwd = `C:\Users\swl00` with explicit `PYTHONPATH` | same clean result; `validate-config` passed | `evidence/P0-import-probe.log` |
| `python -m ipcch_geoxgb preflight --run-id p0-preflight-20261004` | exit 0, passed, 9.5 s | `evidence/P0-preflight.log`, `evidence/P0-preflight-report.json` (sha256 `b9818b1b…5cc0e`, byte-identical copy of `runs/p0-preflight-20261004/preflight/preflight-report.json`) |
| reference source re-verification (Linux, hashing only) | 14/14 match recorded sha256, bytes and git blob at recorded commit | `evidence/P0-source-verification.log` |

PYTHONPATH is forwarded to Windows Python with `WSLENV=PYTHONPATH/p`.

## Input preflight results (`P0-preflight-report.json`)

- Identities: 13/13 inputs match pinned byte length and SHA256 (raw CSV
  1,782,567,753 bytes `ae696087…`; five repaired-geometry components; cache
  `e6b9562a…`; saved lookup/coordinates; repair/geography audits).
- Saved tables: `geography/country_lookup.csv` equals the source
  `country_area_id_lookup.csv` on iso3/country/country_code/country_en and the
  derived `country_key` (trimmed country_en, else country); saved
  `ref_lat/ref_lon` equal source `lat/lon` exactly for every area ID.
- Geometry (pyogrio): 6,227 features, EPSG:4326, 5,578 Polygon + 649
  MultiPolygon, 0 invalid, unique IDs. Universe 6,227 IDs, 0..101324,
  non-dense — identical across geometry, lookup and reference coordinates.
- Frozen cache (validated directly; old helper not loaded, topology not rebuilt):
  exact key set; `id_column admin_code`; component_sha256 equals the five pinned
  component hashes; `polygon_id_mapping` area→index and `polygon_group_mapping`
  index→area are exact inverses; index order equals shapefile row order;
  `area_ids` consistent; every adjacency index in range, no self loops or
  duplicates, symmetric; 12,411 undirected edges, 1,231 isolated (equal to the
  frozen audit); cached centroids equal the repaired geometry's planar centroids
  (max abs diff 0.0).
- Raw panel keys: 1,219,868 rows, years 2010–2026, no blank/non-integer keys,
  months 1..12, 0 duplicate (admin_code, year, month), 6,227 areas all in the
  universe (none missing), one country per area and 0 disagreements with the
  lookup; all per-row lat/lon within 1e-9° of the reference point.
- Known diagnostics, preserved and not repaired, equal to planning observation:
  53 countries, 31 areas missing ISO3, 15 missing country_code, 735 areas whose
  polygon centroid differs from the reference point by >1e-6°.

## Unresolved issues / items for supervisor review

1. **Panel coordinate tolerance chosen after inspection.** 88,960 raw-panel rows
   print lat/lon at 9 decimals, differing from the 12-decimal reference by at
   most 5.0e-11°. The preflight therefore checks a 1e-9° tolerance and records
   the count/max rather than demanding string equality. Panel lat/lon are
   original93 covariates; P1 will use the panel values as-is.
2. **GitNexus unavailable.** `impact(normalize_area_ids)` and `detect_changes`
   fail with the LadybugDB read-only shadow-page replay error; no reindex was
   run. Bounded fallback: `git diff --stat` shows `.gitignore` (+7 lines) as the
   only modified tracked file outside this task; all new symbols live under
   `IPCCHGeoXGBExperiment/`, and nothing outside it references `ipcch_geoxgb`.
3. **Session-ID observation.** The process env shows
   `CLAUDE_CODE_SESSION_ID=d148c921-…` while Herdr and the controller bind
   `afc97c82-…` to this same pane/terminal; the controller accepted start from
   this pane. Recorded for transparency; no rebind was attempted.
4. **Superseded dev run.** `runs/p0-preflight-dev1/` (git-ignored, report sha256
   `ca5d72e4…`) was produced before the tests; the only later code change
   affecting preflight was the explicit polygon_id_mapping key-set check (which
   turned a raw KeyError into a ContractError). Evidence uses `p0-preflight-20261004`.
5. **Build-time cross-check.** `config/feature-schema.json` original93 names were
   compared once against `IPCCHGeoRFExperiment.prepare_data.FEATURE_COLUMNS` by
   a one-off Windows Python import outside the package (reused the existing
   2026-09-21 `.pyc`; no reference file changed). Not a runtime dependency.
6. **Out of P0 scope by design.** Target QC, labels, rich561 values, F/S split,
   models, maps and reports are pending P1–P5 and are listed as pending in
   `config/source-provenance.json`; their CLI commands fail explicitly.

Reference packages (`IPCCHGeoRFExperiment/`, `IPCCHPopulationHistoryExperiment/`,
`FEWSNETGeoXGBExperiment/`, `GeoRFBaseline/`), sibling `IPCCH` (HEAD `c06bc0f5`,
clean), raw data and the frozen geography were not modified.
