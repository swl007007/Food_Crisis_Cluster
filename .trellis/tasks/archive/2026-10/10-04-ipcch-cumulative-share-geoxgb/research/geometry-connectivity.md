# Geometry reuse and Stage1 connectivity: evidence and accepted contracts

Read-only investigation during grill, 2026-10-04. No new partition fitting or
geometry repair. Spatial smoothing was accepted as R38 in v0.32; exact geometry
artifact adoption and source limitations are accepted as R39 in v0.33. All paths are relative to the
repository unless stated otherwise.

## IPCCH geometry identity and reuse boundary

Authoritative historical run: `IPCCHGeoRFExperiment/runs/ipcch-v1-20260920d/`.
Let G denote its `geography/` directory. Its geometry preparation is separate from
the old Stage1 learned map and subsequent donor assignment.

- `G/geography_audit.json:8-25,50-84` records 6,227 areas, EPSG:4326, non-dense
  integer area IDs, 253 initially invalid geometries, and the original component
  hashes. The source was `assembled_IPCCH/spatial/ipcch_admin_geometry.shp`.
- `IPCCHGeoRFExperiment/prepare_data.py:1254-1303,1419-1449` rejects missing/empty
  geometry, non-4326 CRS, duplicate IDs and mismatched area universes. Geometry
  `admin_code` corresponds to CSV `area_id`; canonical integer conversion rejects
  truncation (`:1191-1216`). MultiPolygon stays one area, not separate observations.
- `prepare_data.py:1458-1518,1541-1609` keeps valid input and repairs invalid input
  with make_valid. A resulting GeometryCollection is accepted only with exactly
  one areal component plus zero-area linear/point components. Other cases stop.
  Recorded outcomes: 5,974 unchanged, 36 polygonal repairs, 217 areal extractions.
- Repaired shapefile identity is the full five-component hash set in
  `G/geography_audit.json:195-199`, not only the .shp hash. The read-only scout
  recomputed these five hashes and matched all of them. The repaired .shp SHA256
  is `9df70f3aa719eaa2458a3ff4a92f058da1f5fe1feeb85432e0dd88e515ccfadb`.
- `G/adjacency_cache.pkl` SHA256 is
  `e6b9562ac3705da1bf26541fb52a8808819710738d9465ecd63a07d0727e387c`.
  Its keys are adjacency_dict, area_ids, component_sha256, id_column,
  polygon_centroids, polygon_group_mapping, polygon_id_mapping, polygons.
  It contains no learned/donor partition fields; cache construction and component
  identity checks are at `prepare_data.py:1983-2009`.
- `G/country_lookup.csv`, `reference_coordinates.csv`, `geometry_repair_audit.csv`
  and `geography_audit.json` are preparation evidence. By contrast,
  `stage1/area_assignments.csv`, eligible_donors, learned_map and partition_codes
  are old model/assignment outputs and cannot be reused as a newly learned map.

Important source limitation: `prepare_data.py:2205-2207` explicitly says topology
repair does not establish administrative identity, and the upstream geometry
builder allowed unrestricted nearest-neighbour matching without saved per-area
match provenance. Boundary vintage and those matches remain unverified. This is
distinct from the model-level donor completion already rejected in R36. Reusing
the completed geometry would preserve this limitation, not repair or certify it.
Some repairs substantially changed invalid source footprints; details remain in
`G/geography_audit.json:117-137,190`. No new areal repair is proposed here.

## Actual adjacency

- Frozen helper `GeoRFBaseline/src/adjacency/adjacency_utils.py:99-108` and
  `prepare_data.py:1796-1806` require touches plus intersection length > 0.
  Point-only contact is excluded; overlapping polygons do not meet touches.
  No distance buffer or country filter occurs in that predicate.
- Saved audit `G/geography_audit.json:204-216`: 6,227 nodes, 12,411 undirected
  edges, 1,231 degree-zero nodes, symmetric graph. These counts describe the
  full prepared universe, not the new eligible Stage1 induced graph.
- Scout-only read-only cache statistics: 1,453 connected components (1,231
  singletons, 222 larger components); 307 cross-country edges across 27 country
  pairs after joining the saved country lookup. No full geometric edge rebuild
  was performed. Isolates are not evidence that all those areas are true islands.
- IPCCH passes an explicit adjacency dictionary; the geometry helper returns it
  directly. Centroid-distance fallback is not that historical path. Centroids
  and reference coordinates also differ for 735 areas (`G/geography_audit.json:224-229`).

## Latest GeoXGB Stage1 is smoothing, not a connected-region algorithm

Let E denote `FEWSNETGeoXGBExperiment/`.

1. `E/src/partition/transformation.py:340-418` scans current parent validation
   groups. Before child support checks or fitting, `:471-489` runs polygon
   refinement; `E/config.py:309-330` sets three rounds.
2. `E/src/partition/partition_opt.py:596-660` uses synchronous neighbor-plus-self
   voting. Current membership switches to the most common other label only when
   its vote share is strictly <4/9. Isolates or no valid neighbors keep membership.
3. `partition_opt.py:772-785` marks parent-external/unassigned polygons -1; those
   do not vote. S-group membership defines the active induced graph; F-only or
   otherwise unassigned areas are not bridges or implicitly filled members.
4. After smoothing, `transformation.py:777-886` checks child support, fits eligible
   models, and compares complete parent validation routing with the F1 gain gate.
   Only accepted routing updates membership (`:1060-1128`).
5. Polygon Stage1 has no connected-component gate. `E/src/model/GeoRF.py:468-474`
   explicitly excludes polygon mode from the final grid component refinement.
   MIN_COMPONENT_SIZE=5 therefore does not guarantee anything on this path.

Thus disconnected pieces and isolates can share a learned label. Smoothing may
also cut a narrow bridge. Names such as contiguity/refinement cannot establish a
single connected component per region. The matching GeoRFBaseline polygon
helpers have the same relevant behavior.

An integration issue to check before any future implementation:
`E/scripts/prepare_fourclass.py:570-579` appears to use polygon ID/index mappings
in the wrong direction when remapping adjacency. Occurrence in real FEWS data
was not established. IPCCH IDs are non-dense; do not copy this FEWS geometry
preparation blindly or silently drop unmatched groups. Expected no-map routing
and corrupt/conflicting mappings remain separate contracts.

## Accepted R38: soft spatial smoothing, no hard connectivity

User-approved policy: use supplied shared-boundary adjacency
for three synchronous candidate-smoothing rounds, retaining the <4/9 rule; run
support and F1 gates on those final candidate memberships. Do not force each
learned region to be one connected component, split it after the gate, or create
distance edges to connect it. Eligible isolated areas may retain learned
membership; unlearned areas still follow R36 global fallback.

Report region connected-component counts and isolate counts on the learned-area
induced graph. Call these spatially smoothed predictive regions, not guaranteed
contiguous territories. This retains sample support and the source algorithm;
the trade-off is that a region may span separate pieces. Strict connectivity
would require a different candidate constraint and could reduce local support
under the adopted 500-key/50-area floors.

Exact geometry artifact adoption and its upstream provenance limitation were
separately accepted as R39 below. Neither R38 nor R39 approves new repairs,
administrative rematching, or reuse of old model assignments.

## Accepted R39: freeze the completed IPCCH geometry

User-approved policy: reuse this specific run's repaired geometry,
area-ID mapping and shared-boundary adjacency, binding the complete component
hash set and verifying key/cache consistency before execution. Preserve the
existing cross-country edges; do not add a country barrier, distance edges,
administrative rematching or fresh geometry repairs. Do not copy old learned or
donor assignments. Carry forward the documented administrative-match/vintage
uncertainty as a source limitation rather than claiming topology repair resolved it.

This isolates the new model comparison on the completed package's geographic
foundation. The alternative, reconstructing and independently verifying upstream
area-to-boundary identity first, expands the work into a separate source-data
project and changes that foundation. Existing documented uncertainty is not an
execution-time exemption for hash mismatches, key conflicts or corrupt artifacts.
