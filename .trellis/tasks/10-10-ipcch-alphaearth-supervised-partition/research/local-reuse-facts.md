# Local reuse facts — 2026-10-10

> Consolidation note (2026-10-10): this file preserves the evidence and decision sequence. Earlier wording that a scientific detail was open is historical; accepted D1–D29 are in ../decisions.md and the current final-review contract is ../design.md. Unverified source/asset/resource facts remain unverified, with checkpoints in ../implement.md. No execution is authorized by these notes.

Read-only scouts inspected code/configs; coordinator read the existing baseline comparison note. Recorded SHA values below are existing manifest values, not newly verified hashes.

## Geometry and extraction entry points

- `IPCCHGeoXGBExperiment/config/inputs.json:31-36`: frozen repaired boundary `IPCCHGeoRFExperiment/runs/ipcch-v1-20260920d/geography/ipcch_admin_geometry_repaired.shp`, recorded SHA256 `9df70f3aa719eaa2458a3ff4a92f058da1f5fe1feeb85432e0dd88e515ccfadb`.
- `IPCCHGeoXGBExperiment/ipcch_geoxgb/geography.py:25-30,46-66`: geometry `admin_code`, country lookup `area_id`, EPSG:4326; identifiers normalized to exact int64. `prepare.py:131-137` performs the country mapping.
- `geography.py:201-211,226-243`: adjacency polygon indices are not admin IDs; preserve explicit mapping and geometry order.
- `IPCCHGeoRFExperiment/runs/ipcch-v1-20260920d/geography/geography_audit.json:8-11`: 6,227 features/areas in the audited source.
- `inputs.json:8`: upstream administrative matching and boundary vintage are not fully verified. Topological validity does not establish identity correctness.
- P6 panel, geometry, adjacency and lookup paths exist and sizes match configuration; no hashes recalculated.
- Local CLI entry points found: `/opt/earthengine/bin/earthengine` via `/usr/local/bin/earthengine`; Windows Google Cloud SDK `gcloud`. Config directories exist. No credentials read or authorization probe performed.
- No AlphaEarth references found among searched nonignored code/docs/configs in this repo and sibling IPCCH. Ignored data roots were not exhaustively searched; this is not proof there are no local embedding assets.

## Baseline comparability

- Existing overview: `docs/notes/2026-10-07_note_IPCCH两套pooled_XGB基线差异.md:7-23`. It predates compact296; use it for semantic differences, not latest feature counts.
- `../IPCCH/src/ipcch/compact_features.py:72,136-147`: compact296 = static29 + ordinary120 + same-month-z56 + stress28 + growing-season29 + identifier27 + history5 + IDP2.
- `../IPCCH/src/ipcch/origin_safe.py:114-136`: reported phase is classification truth; normalized shares form q targets; QC differs from P6.
- `origin_safe.py:328-336` and `scripts/modeling/run_deep_feature_weight_decay_forecasting.py:1115-1131`: raw quartet threshold decoding, without P6 monotonic projection.
- `IPCCHGeoXGBExperiment/ipcch_geoxgb/targets.py:7-14,106-156`: population-derived phase; Decimal inclusive 20% truth; finite positive population and share QC. `projection.py:3-18,71-93`: bounded decreasing isotonic, then unrounded >=.20 decoding.
- compact horizons H0/3/6/12 (`compact_features.py:40`); P6 H1/3/6/12 (`config/experiment-contract.json:20-31`). H0 is not H1.
- Compact annual expanding history, 24-month half-life: `../IPCCH/scripts/modeling/run_deep_feature_weight_decay_forecasting.py:1078-1097,1186-1189`, `origin_safe.py:313-316`. P6 is rolling 36 months (`schedule.py:122-124`).
- `../IPCCH/src/ipcch/weather_oracle.py:5-10`: oracle uses realized O+1..O+min(H,6) precipitation/temperature anomalies. This is a perfect-weather-information scenario, not demonstrated operational forecast skill or a guaranteed performance upper bound.
- Latest compact extension saves keyed raw quartet predictions and normalized shares: `../IPCCH/results/experiments/compact_climate_weather_oracle_eval_2026_jan_apr_v1/runs/compact_baseline/0m/predictions/predictions_2022_2026.csv`. Existing predictions can support re-scoring on intersecting keys; canonical population/QC truth must be joined separately. No re-score or key-intersection calculation performed. No H1 compact predictions exist in the inspected recipe.

## Foundational documents to read when freezing the new baseline

- `../IPCCH/.trellis/tasks/archive/2026-10/10-08-compact-climate-weather-oracle/design.md`
- `../IPCCH/.trellis/tasks/archive/2026-10/10-09-compact-eval-2026-jan-apr/design.md`
- `.trellis/tasks/archive/2026-10/10-04-ipcch-cumulative-share-geoxgb/design.md`

The compact-climate-weather-oracle design was subsequently read in full by the coordinator while preparing the annual-update question. The other two documents remain located references, not newly re-reviewed. No experiments, cloud jobs or live MLflow mutations were run.

## Accepted common compact feature base (D15)

User accepted on 2026-10-10: use the 296-column **non-oracle** compact recipe for B0; B1/B2/G use that same base plus the same fixed AlphaEarth descriptors. This selects a feature recipe, not the sibling run's reported-phase truth, cohort, raw decoder, H0 or runtime by implication. Population-derived truth/QC and the selected H1/3/6/12/projection conventions remain this task's authority.

Coordinator re-read `../IPCCH/src/ipcch/compact_features.py:1-18,43-73,116-147`. Feature groups are static29, ordinary120, same-calendar-month z56, stress28, completed-growing-season29, coordinates/target-calendar27, safe history5 and IDP2. Source details and timing are in the full compact design, already read. Keep actual future-weather oracle columns out of this base.

The five inherited history features are the latest three eligible **reported** phases and two differences (`../IPCCH/.trellis/tasks/archive/2026-10/10-08-compact-climate-weather-oracle/design.md:104-108`). They are predictors from earlier observations, not classification truth for this task. Replacing them with population-derived phases would be a different feature recipe; do not silently do so. The large rich561 share-history block is not appended under D15. Its old outputs remain available as historical references.

The original compact entry point only accepts H0/3/6/12 (`compact_features.py:40,136-147`), so H1 must be explicitly supported by a task-specific input build using O=T−1. Source/feature provenance and missingness must be checked on this task's canonical rows. Do not shrink the new cohort to rows available in old reported-phase-gated compact tables just to make joins easy, and do not substitute H0 feature values/predictions for H1.

Advantage: reuses the user's newer anomaly/climate, season, price/conflict and history inputs with a bounded, documented feature set, helping isolate the addition of AlphaEarth and partitioning. Trade-off: rich561 provides more detailed share-history information, and no unified-cohort experiment yet establishes that compact296 is best for population-derived phase. Exact source hashes, per-horizon coverage, runtime and matched old-result reuse remain unverified. The choice is not authorization to build matrices or fit models yet.

## Read-only geometry coverage check — 2026-10-10

A scout directly read the repaired DBF, existing geography ID CSVs and the canonical valid ledger/horizon keys. The DBF has 6,227 unique nondeleted `admin_code` records (no duplicate IDs). `country_lookup.csv`, `reference_coordinates.csv` and `geometry_repair_audit.csv` each have the same 6,227-ID set. All five shapefile component hashes match `IPCCHGeoXGBExperiment/config/inputs.json`; the SHP hash is the value recorded above. This updates the earlier statement that component hashes had not been freshly checked.

| Cohort | Rows | Distinct admins | Missing-geometry rows/admins | Duplicate-ID ambiguities |
|---|---:|---:|---:|---:|
| All valid targets / each complete H key table | 42695 | 6224 | 0 / 0 | 0 |
| H1 main | 17322 | 5999 | 0 / 0 | 0 |
| H3 main | 16919 | 5999 | 0 / 0 | 0 |
| H6 main | 16413 | 5939 | 0 / 0 | 0 |
| H12 main | 14087 | 5921 | 0 / 0 | 0 |

Main uses `origin_month >= "2023-01" and target_month <= "2025-12"`. Complete H keys match valid-ledger admin/target keys with no duplicates. The saved preflight for these hashed components reports EPSG:4326, 5,578 Polygons + 649 MultiPolygons, invalid=0 (`IPCCHGeoXGBExperiment/runs/p6-preflight-20261004b/preflight/preflight-report.json:107-121`, coordinator checked). Spatial validity was not recomputed in this planning probe.

Existing loaders fail on duplicate/missing/empty/invalid geometry (`ipcch_geoxgb/geography.py:156-168`), and preflight rejects mismatched geometry/country/reference universes (`preflight.py:87-99,120-129`). The existing `unmapped_area_global` prediction route concerns absence from the learned partition map, not missing geometry (`stage3.py:187-193,273-285`). Thus no geometry-driven deletion of the new evaluation cohort is indicated by these counts.

These checks establish file identity, key coverage and saved topological validity, not administrative correctness. `geography.py:8-11` and the source `geography_audit.json:406-409` retain the unresolved boundary-vintage and upstream unrestricted-nearest-neighbour matching limitation, without per-area match provenance. No actual AlphaEarth pixels, assets, coverage or cloud calls were checked.
