# Q8g repair trial — 2026-09-20 (read-only; nothing written to any source)

Run with the pinned Windows runtime (GeoPandas 1.0.1, Shapely 2.1.0) on
`assembled_IPCCH/spatial/ipcch_admin_geometry.shp`.

## Independent reproduction of the audited state

6,227 features, EPSG:4326, 0 empty, 0 missing, 6,227 unique `admin_code`,
5,446 Polygon + 781 MultiPolygon. **253 invalid**, reasons 212 self-intersection /
30 ring self-intersection / 6 too-few-points / 5 nested shells. This matches the
PRD and STATUS.md counts exactly.

## Trial `make_valid` outcome — the Q8g stop triggers

| result | count |
|---|---|
| still invalid | **0** |
| empty | **0** |
| MultiPolygon | 28 |
| Polygon | 8 |
| **GeometryCollection (non-polygon)** | **217** |

PRD R2 and design.md require a stop on non-polygon output, and forbid extracting
polygon pieces out of GeometryCollections, replacing boundaries, dropping areas or
escalating the repair method. implement.md Phase 2 directs: preserve evidence and
ask one scope decision.

## What the 217 collections actually contain

Every one has the same shape — one areal part plus one linear part:

| composition | count |
|---|---|
| MultiLineString + Polygon | 136 |
| MultiLineString + MultiPolygon | 62 |
| LineString + Polygon | 19 |

**The linear components have exactly zero area** in every case. They are the
degenerate spikes/zero-width slivers that made the original self-intersect.

## Areal change of the polygon part

Against the original's area, which is diagnostic only — an invalid original's area
is not ground truth (design.md):

| statistic | value |
|---|---|
| median relative change | 1.01e-05 |
| cases > 0.1% | 54 / 217 |
| cases > 1% | **35** / 217 (corrected 2026-09-20; this file first said 34) |
| maximum | **29.21%** (`admin_code=1425`) |

Worst cases: 1425 (29.21%), 1578 and 1595 (28.94% each), 1600 and 1635 (24.52%
each), 3845 (17.28%).

Note 1578/1595 and 1600/1635 have **identical original and repaired areas** within
each pair, i.e. distinct `admin_code` values carrying the same geometry. This is
consistent with the recorded upstream limitation that the shapefile builder allowed
unrestricted nearest-neighbour fallback without saved per-area match provenance
(`source-audit.md`). It is a provenance observation, not a repair finding.

## Status

No repaired copy was written. No adjacency matrix was built. Stage1 has not run.
A scope decision is required before geometry can be used.


## Corrections found during implementation — 2026-09-20

**The >1% count is 35, not 34.** Recomputed geodesically (pyproj `Geod(ellps="WGS84")`,
m²) and again planar; both give 35. The values bracketing the 1% line are 0.00884 and
0.01154, a clean gap, so this is an off-by-one in this file's first pass, not a
measure or boundary artifact. The worst case (`admin_code=1425`, 29.21%) is unchanged.

**The largest footprint changes are NOT among the 217 collections.** This file profiled
only the GeometryCollection outcomes. Two *plain* `make_valid` repairs — already covered
by the original Q8g authorization, needing no extension — are far worse:

| admin_code | original | repaired | change |
|---|---|---|---|
| 162 | 2.0013e10 m² (20,013 km²) | 3.0267e8 m² (303 km²) | **-98.49%** |
| 133 | 1.2689e10 m² (12,689 km²) | 4.4289e8 m² (443 km²) | **-96.51%** |
| 3748 | 3.1972e8 m² | 3.1794e8 m² | -0.56% |

Only 2 of the 36 plain repairs exceed 1%, but those two are roughly three times worse
than the worst collection case. An invalid original's area is diagnostic only, so this
is not proof that the repaired shape is wrong — but two administrative areas shrinking
by ~97-98% is a material limitation for any spatial partition learned on this geometry,
and it must appear in the final report.

No stop is triggered: Q8g's stop conditions are remaining-invalid, empty, non-polygon
and ambiguous identity, and none applies. design.md explicitly forbids inventing an
unapproved percentage-tolerance gate, so no area threshold was added.
