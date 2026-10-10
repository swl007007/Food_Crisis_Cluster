# D30 implementation and bounded C pilot proposal (pilot not run; release_id null)

## Exact scientific parameter delta (config/experiment-contract.json; full diff in experiment-contract.D30.diff)

Only `alphaearth.ownership` (first-footprint string) is replaced by `alphaearth.analysis_grid` = {amendment D30,
crs EPSG:6933, affine [10,0,0,0,-10,0], resampling nearest, priority ascending asset id first valid whole 64-band
vector, weights on the same grid, grid-aligned windows only, estimand note}. `validate_frozen` pins the block. All
other contract keys (calendar, targets, features, backbone, correction, evaluation, coverage .80, year, bands) are
byte-identical. Config identity changes, so existing developer runs refuse reuse (intended); future fitting runs need
fresh prepared bindings.

## Implementation (smallest connected change)

- `extraction.py`: `recipe_id`, `Lattice` (fixed lattice, cylindrical check, lattice lines = meridians/parallels),
  `lattice_windows` (frozen geometry ∩ lattice-aligned boxes; no vertex reprojection/repair), `plan_c_requests`
  (ordered candidate ids per window; no-source windows recorded per admin, not request pieces; zero-source admins keep an
  explicit zero row; denominator = whole frozen admin area), recipe binding in `check_piece_binding`, recipe kept by
  `parse_export_csv`. Superseded first-footprint helpers retained for native-pilot comparison only.
- `descriptors.descriptor_table`: refuses plans/exports not built under the frozen D30 recipe id.
- `scripts/ee_pilot_c.py` (new; the native `ee_pilot.py`/`extract_alphaearth.py` stay comparison-only): common
  64-band mask applied to all bands + a `source_rank` band per asset BEFORE `mosaic` over descending ids (last on top
  → lowest valid id wins); one call returns A/S/Q sums, valid-cell count and owned area grouped by source_rank, all on
  EPSG:6933 `crsTransform` [10,0,0,0,-10,0], `bestEffort=false`, per-request band-pixel cap ≤ released ceiling.
- Tests: `tests/test_c_extraction.py` (16) + recipe-bound fixtures in two existing files; full suite 88 OK
  (`unittest-D30.log`); supervisor focused checks 14 PASS (receipt retained; hashes match).

Pinned source hashes: ee_pilot_c.py f16be55ca15c9a7de368371398017cf7fb3bcb8c1154f907025ae35689fb7d06; ee_metadata.py d246a6c507debf167b88c3c28c8a87e708b2e26ccfa234aee8b0cc8661cc7360;
extraction.py de9a3221669c31a88250b828dc9cb2ac365c5ca7bedb1214aeee9684b06d481c; descriptors.py 5f0ea041d3b1a1ef52c61a1da010484f3440732f73af04a0b5f940f97b470368;
contract.py 83a0ca8060f0ce08ba42ab87bc0e6bcc98aa5eff246c3c339253b14d425a8425; experiment-contract.json b75a71066f3b96b2719dc851d61516b453d2fd843805af065419fd96984f1378.

## Bounded C pilot (`c_pilot_request.json`, body sha256 22a593eaf0e061b234c529645217022a418c9786d9c284e31622d6f858d40320)

| id | geometry | candidates | cells | per-request maxPixels (band-px) |
|---|---|---|---|---|
| C01 | admin 1722 overlap piece (same geometry as B1 / mask diagnostic) | 6 (x6bdj…, xbcoe…, xdgzj…, xjlze…, xl600…, xt0jt…) | 2.24e6 | 3.17e8 |
| C02 | admin 3447 whole (single-zone control vs A2) | 1 | 5.12e5 | 7.27e7 |
| C03×6 | admin 3447 as 6 lattice-aligned windows (batching; sum vs C02) | 1 | 4.1e4–1.4e5 each | ≤ 2.0e7 |
| C04 | open Red Sea 2×2 km all-masked fixture | 1 | 3.8e4 | 5.3e6 |
| C05 | one ~4e6-cell lattice-aligned window ∩ admin 3649 (timing) | 1 | 4.0e6 | 5.68e8 |

Limits: ceiling maxPixels 567,600,518 band-px; deadline 120 s; ≤ 13 high-level calls incl. init and retries; sequential;
SDK retries 0; one wrapper retry (memory/timeout only); size rejections stop; outer `timeout 1200`; fresh output dir.
Command: `timeout 1200 /opt/earthengine/bin/python scripts/ee_pilot_c.py --request <released manifest> --out
~/ipcch-alphaearth-runs/p1-c-pilot-<date>`.

Reading: C01 area_by_rank over all 6 candidates (no two-asset expectation), total valid vs the mask-diagnostic union;
C02 vs A2 = measured estimand change for a single-zone admin; Σ C03 vs C02 = batching on one lattice (expect agreement up
to EE summation order and outer-boundary weight quantization; report max relative difference); C04 zero vs null sums on
the analysis grid; C05 seconds per cell. Production export/parse route must be connected to D30 before any bulk release.
