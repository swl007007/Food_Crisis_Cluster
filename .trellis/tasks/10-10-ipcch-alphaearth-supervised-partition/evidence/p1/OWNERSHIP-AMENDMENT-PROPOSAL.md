# Proposed extraction-rule amendment: first VALID tile in lexicographic order (not implemented; for user decision)

Trigger (measured, bounded): on one sampled cross-zone piece the frozen first-footprint rule kept 60.8% and discarded
39.2% of otherwise valid coverage (mask diagnostic). Extent over the cohort is unknown; this proposal does not assume it.

## Recipe B — smallest amendment (native value grids)

For admin polygon P (frozen geometry) and the assets whose footprints intersect P, sorted lexicographically by asset
id a_1 < … < a_n (outcome-independent, unchanged order):

1. Geometric part per asset: G_k = P ∩ F_k (footprint of a_k). Unlike the frozen rule, earlier footprints are NOT
   subtracted geometrically.
2. Validity per asset on its own grid: V_k = all 64 bands unmasked AND finite (|x| < 1e300), 0 outside the footprint
   (unchanged definition).
3. Prior-valid mask on a_k's native grid: Prior_k = max_{j<k, F_j ∩ G_k ≠ ∅} reproject(V_j → grid of a_k) using
   EE's default nearest-neighbour resampling of the 0/1 mask (V_j evaluated at the a_j pixel nearest to each a_k
   pixel centre). Only earlier assets whose footprint meets G_k enter (small sets in practice).
4. Contribution weight on a_k's native 10 m grid: W_k = V_k AND NOT Prior_k, times EE's fractional pixel-in-G_k
   weights (as now). A_k = Σ W_k·pixelArea, S_kj = Σ W_k·pixelArea·e_j, Q_kj = Σ W_k·pixelArea·e_j² — values
   always from a_k's own native pixels (no value resampling, no fill, no new grid).
5. Merge and decide exactly as now: A = Σ_k A_k, S = Σ_k S_k, Q = Σ_k Q_k → 64 means/SDs; coverage = A / area(P)
   (frozen local geodesic denominator); usable iff coverage ≥ 0.80 and 128 finite moments.

Properties: deterministic and label-free; reduces to the frozen rule wherever earlier tiles are valid across the
overlap; equals the union of valid coverage where masks are disjoint; never fills masked pixels of one asset from
another's values; per-piece ledger keeps (admin, asset, prior asset list).

## Cross-UTM mapping, edge and double-count approximation

Masks of earlier assets are compared at a_k's pixel centres via nearest neighbour across different UTM grids (pixel
lattices rotated and offset). Consequences are confined to pixels on the boundary of an earlier asset's valid region
inside overlaps: a location can be counted twice (earlier pixel partly valid and counted in a_j, but its nearest
sample at the a_k centre says invalid) or missed (the reverse). The error band is about one pixel wide along valid-mask
edges in overlaps: per edge length L, at most ~L × 10 m. Empirical bound from the diagnostic: class areas differ by
≤ 0.018% between the two reference grids, and the mutual 'both' area was 1.75% of the piece. Boundary weights versus
P remain EE's documented ~1/256 approximation (unchanged).

## Tests before any extraction (offline, then a bounded pilot)

- Synthetic two-grid fixture (numpy): two 10 m lattices offset/rotated, known masks → B counts each location once
  except within one pixel of mask edges; equals frozen rule when the earlier tile is fully valid; equals union when
  masks are disjoint; all-masked → A = S = Q = 0; deterministic for permuted input order.
- Moment additivity unchanged (merge A/S/Q, never SDs); plan/ledger binding includes prior asset lists.
- Bounded EE pilot: admin-1722 owned piece re-extracted under B — donor contribution must equal the diagnostic's
  donor_only (87.64 km² on the donor grid) within ≤ 0.02%, owner contribution = 135.92 km², total ≤ area(P)(1 + 1e-3);
  plus one single-zone admin (3447) whose result must equal A2 exactly.
- Cost: requests read prior masks (64 bands each) only in overlap zones; the pilot measures the overhead.

## Recipe C — one fixed common 10 m grid (comparison only)

Resample every asset's 64 bands to a single common 10 m grid (e.g. per-admin local UTM or a global equal-area
lattice), mosaic by first valid, then reduce once. Single counting is exact by construction and masks are trivial.
But embedding values are resampled (nearest duplicates/drops native pixels; bilinear mixes unit-norm vectors and
changes the SD descriptors), area weights come from a non-native projection, every pixel is reprojected (higher
EE cost), and it departs from D26's native-grid aggregation more than B. Recommendation: B, which changes only which
native pixels are admitted, not their values.

Both are scientific extraction-rule amendments to D16/D26 operationalization and require the user's decision; the
statistical recipe (64 means + 64 SDs, .80 coverage, local denominator) is unchanged under either.
