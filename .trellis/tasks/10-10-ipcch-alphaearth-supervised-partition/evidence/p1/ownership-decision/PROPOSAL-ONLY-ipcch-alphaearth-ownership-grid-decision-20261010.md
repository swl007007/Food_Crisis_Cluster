# AlphaEarth ownership amendment for user decision

Status: proposed only; no scientific config changed or new extraction released.

## Observed trigger
On the frozen admin-1722 overlap piece (223.546435 km2), lexicographic footprint ownership retains about 60.8% valid coverage. The donor-only part measures 87.638266 / 87.641812 km2 on owner/donor reference grids. This establishes a large loss on this piece, not on the full admin or cohort.

## Recommended C: a single analysis grid
Use EPSG:6933 and a fixed affine [10,0,0,0,-10,0] (metres), nearest-neighbour sampling, first valid tile in ascending frozen asset ID. One source supplies each complete 64-band vector; validity requires every band unmasked and finite. No bilinear mixing, no vector renormalization, no coarser bestEffort. First-valid ownership is mutually exclusive on this analysis grid, independent of batching. PixelArea and administrative boundary weights use that same grid. Keep the local geodesic whole-polygon denominator, inclusive 80% coverage rule, year 2017, area means/SD and all downstream scientific contracts unchanged.

This changes the estimand: moments describe nearest-neighbour-sampled embeddings on a fixed nominal 10m equal-area grid, rather than integration over each source native UTM grid. Native vectors are selected unchanged, but spatial representation and weights change. Exact single counting refers to analysis-grid cells, not exact continuous polygon integration; EE boundary-weight approximation remains. Grid CRS/affine and recipe enter frozen provenance. Confirm data footprint falls within the projection domain offline before any pilot.

Before bulk: implement only this amendment with focused tests, compare a bounded overlap pilot and single-zone control to retained A2/B1, verify per-cell exclusivity and identical A/S/Q under identical-grid batching. Measure descriptor changes, area differences and resource use; retain raw coverage, no silent clipping. Re-enumerate full requests and cost before bulk release. Existing EE query/download authorization covers bounded live checks; GCS writes remain unauthorized.

## Alternative B: source-native integration with approximate ownership
Retain all admin-intersecting tiles and use each native grid. Exclude prior-valid masks after nearest-neighbour mapping. This preserves native value grids, but cannot guarantee unique physical-area weights across offset grids. A same-size 10m cell shifted 6m may survive because its centre is outside earlier validity while 4m overlaps. The opposite shift can cause a gap. The observed <=0.018% aggregate diagnostic difference is not an upper bound on these errors. Choosing B explicitly accepts this additional approximation and requires gross-overlap/gap and threshold-sensitivity diagnostics; it is not an exact no-double-count solution.

## Decision boundary
Choose C (recommended) or B. This approves a documented extraction-rule amendment and focused validation, not a claim of full extraction success or permission to write GCS objects. Freeze the revised recipe before model fitting; keep all prior pilot evidence unchanged.
