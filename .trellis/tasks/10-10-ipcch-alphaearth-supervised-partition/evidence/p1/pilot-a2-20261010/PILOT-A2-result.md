# P1-PILOT-A2-20261010 result (same two geometries, maxPixels 7.0e7 band-pixels)

Release checks passed before execution: ee_pilot.py 24aa9369…, ee_metadata.py d246a6c5…, extraction.py 4e5832a6…,
HEAD a04b40f; manifest body b0b11f13… copied unchanged into fresh `~/ipcch-alphaearth-runs/p1-pilot-a2-20261010`;
`timeout 600`. A1 untouched. Calls: A2 used 3 of 6 (init + 2 reductions, no retry); cumulative pilot calls A1 2 + A2 3 = 5.
No export, no cloud write, no fit, nothing normalized. Raw responses in `responses/`.

| request | seconds | valid_pixels (count) | valid_area m2 (pixelArea sum) | nulls | projection |
|---|---|---|---|---|---|
| admin 3447 piece, asset x7kfwxt1zhh4tk7kr | 7.87 | 511,951 | 51,234,569.36 | 0 of 129 | EPSG:32638, [10,0,500000,0,10,1310720] |
| open Red Sea 2x2 km fixture, asset xzki2wh0491tllh8g | 0.84 | 0 | 0 (all 129 sums exactly 0) | 0 of 129 | EPSG:32637, [10,0,336160,0,10,2129920] |

## Admin 3447 descriptors (offline merge with aemoments, frozen D25 rule)

Local WGS84 geodesic polygon area 51,235,364.86 m2 -> coverage 0.99998447 >= 0.80 -> usable. 128 moments finite,
0 negative variances zeroed. Mean vector L2 norm 0.8768 (unit-norm pixels averaged); mean range -0.315..0.246;
SD range 0.0243..0.1265. Valid pixel count 511,951 vs area/100 = 512,346 (pixelArea != 100 m2 off the UTM central
scale; expected).

## Checks answered

- maxPixels units: band-pixels corroborated (supervisor native-CRS check: 511,944.28 x 129 = 66,040,811.7).
- All-masked behaviour: a completed reduction over fully masked pixels returned 0 (not null) for valid_area and all
  S/Q, with valid_pixels 0 -> genuine no coverage (D25 `no_coverage` route). No null normalization needed here; nulls
  remain technical failures if ever observed.
- Area consistency: EE pixelArea sum vs local geodesic polygon area: -0.0016%. EE `geometry.area(1)` of the planar
  (geodesic=False) request geometry is +0.38% (3447) / +0.30% (fixture) above the local geodesic area: supports the
  design's local geodesic denominator; EE geometry area is a diagnostic only.
- Projection: native UTM with positive scaleY affine, handled by EE without error.

## Timing (single small sample; not a measured envelope)

Overhead proxy (masked fixture) 0.84 s; admin 3447 7.87 s for 512,354 px -> marginal 1.37e-5 s/px. Naive projections:
median piece 3.16e7 px ~ 434 s (> 120 s deadline); 1e8-px piece ~ 23 min; all 4.67e11 px ~ 6.4e6 s (~74 days)
sequential. The masked request may be cheaper than a valid one of equal size; B needs a measured-envelope review
(e.g. a 2-5e6 px piece and a tileScale comparison) before any representative-admin request, and the full extraction
needs a different throughput design (concurrency/high-volume endpoint or batch export — the latter is a cloud write
outside the current authorization). No cap was enlarged.
