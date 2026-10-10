# Proposed extraction-rule amendment options (not implemented; for user decision)

Trigger (measured, bounded): on one sampled cross-zone piece (admin 1722, 223.55 km²) the frozen first-footprint rule
kept 60.8% and excluded 39.2% of area that the next tile reports as valid (mask diagnostic). Extent over admins and the
cohort is unknown; neither option below assumes it. Both are amendments to the D16/D26 operationalization; the
statistical recipe (64 area-weighted means + 64 SDs, .80 coverage, frozen local geodesic denominator) is unchanged.

## Option B — approximate native-grid ownership: first valid tile in lexicographic order

For admin polygon P and the assets whose footprints intersect P, sorted lexicographically a_1 < … < a_n:

1. Candidate part per asset G_k = P ∩ F_k. Earlier footprints are NOT subtracted geometrically, so assets whose parts
   were formerly removed by footprint subtraction become candidates again (work inventory grows; see below).
2. V_k = all 64 bands unmasked AND |x| < 1e300 on a_k's native grid; 0 outside its footprint (unchanged definition).
3. Prior_k on a_k's native grid = max over earlier assets j < k whose footprint meets G_k of V_j sampled at each a_k
   pixel centre by EE default nearest neighbour.
4. W_k = V_k AND NOT Prior_k (× EE fractional pixel-in-G_k weights). A_k, S_k, Q_k summed from a_k's own native
   values only (no value resampling, no fill).
5. Merge A/S/Q over k exactly as now; coverage = A / area(P); usable iff ≥ .80 and 128 finite moments.

What B is and is not: deterministic, label-free, values from native pixels, and identical to the frozen rule where an
earlier tile is valid across the whole overlap. It is NOT exact single counting and does NOT guarantee the union of
valid coverage: ownership is decided at a_k pixel centres against a differently aligned (and differently oriented)
lattice. Counterexample (1-D, 10 m cells): owner cell x ∈ [0,10] valid, owner invalid beyond 10. A donor cell
[6,16] has centre 11 (owner invalid there) and is admitted, so [6,10] (4 m) is counted twice; a donor cell [4,14]
has centre 9 (owner valid) and is rejected, so [10,14] (4 m) is lost. Gross double counting and gaps therefore occur
along every earlier-tile valid-mask edge inside overlaps; no rigorous per-edge bound is claimed here. The ≤ 0.018%
reference-grid difference observed in the diagnostic compares one pair of aggregate class areas on one piece; it is
not a bound on gross gaps/double counting nor on cohort-level error.

Inventory/binding changes under B: the piece plan keeps every candidate part G_k (including formerly subtracted ones)
and binds, per piece, the ordered list of prior asset IDs used in Prior_k; export/replay validation must check that
list; pieces with W_k ≡ 0 return genuine zeros under the frozen mask contract (never dropped).

## Option C — one fixed global equal-area 10 m grid with exact discrete ownership

1. One global equal-area lattice with 10 m × 10 m cells (a single named equal-area projection and origin, fixed in the
   contract before extraction; e.g. an EPSG equal-area CRS with a 10 m affine anchored at a frozen origin).
2. For every common cell c, sample each candidate asset's ENTIRE 64-band vector and its common 64-band validity at the
   cell centre by nearest neighbour (no bilinear or any value mixing).
3. Ownership: c belongs to the first asset in lexicographic order that is valid at c (exact, mutually exclusive,
   discrete); cells with no valid asset are uncovered.
4. A = Σ over cells of (cell area × EE fractional weight of c in P), S_j, Q_j likewise with the owned vector; merge and
   decide exactly as now. Coverage keeps the frozen local geodesic denominator.

What C is and is not: every common cell has a unique weight and at most one owner, so there is no double counting or
ownership gap at the common-grid level. It changes the estimand from integration over native 10 m pixels to
integration over nearest-sampled native vectors on a common lattice: native pixels can be duplicated or skipped by
the nearest sampling (lattice phase/rotation), so per-admin means/SDs differ from native integration by an amount not
yet measured. Cost relative to the native recipe is not measured.

## Tests required before either option is used

- Synthetic two-lattice fixture (numpy, offset and rotated): B — reproduce the counterexample cases (double count
  and loss), reduction to the frozen rule when the earlier tile is fully valid, all-masked zeros, order-permutation
  determinism, prior-ID binding; C — unique ownership per common cell, no double count, nearest sampling of whole
  vectors, sensitivity of means/SDs to lattice phase.
- Moment additivity unchanged (merge A/S/Q, never SDs); plan/export/replay binding of the new inventory fields.
- Bounded EE pilot under the chosen option: admin-1722 piece (compare with the diagnostic classes and report gross
  differences, not only totals) and single-zone admin 3447 (B must equal A2 exactly; C reports its change vs A2).

Supervisor recommendation recorded: C, for clean unique weights. Decision: user.
