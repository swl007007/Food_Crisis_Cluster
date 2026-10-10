# Local checks while the ownership decision is pending (no cloud work, no recipe implemented)

The supervisor's decision note is retained verbatim as `PROPOSAL-ONLY-ipcch-alphaearth-ownership-grid-decision-20261010.md`
(proposal only; not an approval).

## EPSG:6933 projection domain (candidate C lattice, affine [10,0,0,0,-10,0])

- EPSG:6933 (WGS 84 / NSIDC EASE-Grid 2.0 Global), area of use lon −180..180, lat −86..86, metres.
- All 6,227 frozen admins (lon −92.24..127.34, lat −34.83..38.49) and all 1,683 asset footprints (lat −35.53..39.96)
  lie inside the area of use; every projected coordinate is finite. 6933 extent of the footprints:
  x −8.97e6..1.23e7 m, y −4.25e6..4.70e6 m.
- Equal-area consistency (planar area in 6933 vs frozen geodesic area): |rel| median 3.2e-6, p99 5.6e-4, max 1.10e-2.
  Analysis-grid cell total ≈ 4.671e11, equal to the native estimate.
- Six admins (2589, 2757, 3250, 3333, 4044, 101324), valid in EPSG:4326, self-intersect after vertex-wise projection
  to 6933 (near-touching vertices); a buffer(0) repair would change their areas by ≤ 6.3e-5 relative. Recorded, not
  repaired; any 6933 implementation must not reproject-and-repair geometry silently.
- EASE-Grid 2.0 Global is cylindrical: x depends only on longitude and y only on latitude, so analysis-lattice lines
  are exact meridians/parallels (straight edges in planar lon/lat).

## Edge semantics (affects the frozen recipe and both options)

The frozen denominator uses geodesic edges (pyproj); the EE request geometries use planar lon/lat edges
(geodesic=False). Planar-edge vs geodesic-edge area over all admins: |rel| median 2.7e-6, p90 5.5e-5, p99 5.1e-4,
max 1.10e-2 (one admin; 27 admins > 1e-3). Pilot admins are unaffected (3447: −1.2e-6; 101208: 2.8e-9). Proposal: the
amendment freeze should fix one convention for both the reduced region and the denominator (e.g. geodesic edges in
EE, or densified geometry used for both).

## Draft resource pilot (`resource_pilot_draft.json`, release_id null; for the chosen recipe only)

S1 additivity: admin 3447 whole + 4 quadrants cut on analysis-lattice lines (meridian/parallel through the snapped
centroid), so cut-adjacent cells are whole; S2 size scaling: three lattice-aligned squares (~2e6, 8e6, 1.6e7 cells)
∩ admin 3649; S3 one computeFeatures call over the 4 quadrants; S4 concurrency 2/4 only after S1–S3. Comparison rule:
sub-piece sums vs whole are expected equal up to EE summation order and outer-boundary fractional weights (EE ~1/256
quantization); report the maximum relative difference; never claim float-exactness. If B is chosen, cuts follow one
asset's native UTM lattice and geometries are regenerated. Proposed caps: maxPixels = 129 × expected cells × 1.1,
deadline 120 s, ≤ 20 calls, sequential S1–S3, SDK retries 0, one wrapper retry, outer timeout 1,800 s.
