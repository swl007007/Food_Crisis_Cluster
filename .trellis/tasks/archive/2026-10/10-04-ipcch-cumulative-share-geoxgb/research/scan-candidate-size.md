# Scan candidate size and deterministic ordering — accepted R46

Planning only, 2026-10-04. The user accepted this rule as R46 / v0.40,
following R45's recursion budget. No fitting, scan execution or data experiments ran.

## Source and reason for an explicit rule

The source `FEWSNETGeoXGBExperiment/src/partition/partition_opt.py:835-858`
sorts group scores and starts with ceil(N/2) groups. Its `optimize_size` at
243-270 uses ceil(.9*ceil(N/2)) and ceil(1.1*ceil(N/2)), an exclusive upper loop
bound, and `optimal_size = size - 1`. If no score crosses zero in that loop, it
keeps the initial half rather than maximizing the cumulative score over all
allowed sizes. Consequently the code is neither an exact symmetric 45%-55%
constraint nor a 10%-minimum-child rule. Plain argsort also provides no explicit
area-ID tie contract. Full source evidence is in recursive-search-budget.md.

The accepted rule below preserves near-half splitting as the intended spatial scale
constraint, while deliberately defining new exact integer boundaries and a
deterministic score-maximizing prefix. It is not a byte-for-byte legacy port.

## Accepted R46 rule

1. N counts distinct learned administrative-area groups of the current parent
   with Stage1 S membership. Include groups with zero crisis-scan mass; they still
   count toward membership and support. Do not count population, repeated months,
   four targets, different H, or outside-parent/unlearned areas as extra groups.
2. Before smoothing, require each side to contain 45%-55% of those N groups.
   For the selected prefix length m, the exact allowed integer range is
   `lo = max(1, ceil(9*N/20))` through
   `hi = min(N-1, floor(11*N/20))`, both inclusive. Compute rational boundaries
   with integer arithmetic. If N<2 or lo>hi, record no feasible size and retain
   the parent; do not expand the range or retry. Example: N=100 permits m=45..55;
   N=101 permits m=46..55. A tiny odd N can have no feasible split, consistently
   with the much larger adopted local support floors.
3. Rank groups by descending score, breaking exactly equal scores by the frozen
   canonical area-ID order. For each allowed m, calculate the sorted-prefix
   score sum. Choose the greatest sum; exact ties prefer smaller abs(2*m-N), then
   smaller m. The complement is the other child. This examines prefix sizes
   within the one scan; it does not fit or validate a local model for each size.
4. Apply this same deterministic grouping function to the existing single-column
   initialization ranks c_g/b_g (zero where b_g=0), and then to each iteration's
   scan scores g_g=c_g*log(rho)+b_g*(1-rho). Initialization prefix sums are merely
   a seed-selection rule, not a likelihood statistic. The existing mass handling
   and rho update remain subject to R32; R45 fixes 1000 iterations and returns the
   last candidate, without best-across-iterations selection or restarts.
5. Apply R38's three synchronous smoothing rounds to that final candidate. The
   45%-55% rule applies before smoothing only; do not rebalance, rerun scan or
   reject solely for a post-smoothing ratio outside the range. Record both sets
   of sizes. Reject an empty resulting side as a non-split; for nonempty sides,
   apply R28/R29's original-key support and complete-routing F1 gate, preserving
   parent routes for unsupported children as already specified. No mandatory
   connectivity, population balancing or country balancing is added.

This keeps the candidate deterministic under a frozen input/schema/environment,
and excludes arbitrary off-by-one inheritance. It may miss a small isolated
high-error pocket because the first proposed cut is close to half; subsequent
accepted splits can refine only within R45's depth-4 budget. Group-count balance
does not imply balanced training rows, crisis cases or population. This is an
engineering search restriction, not a statistical significance guarantee.

R46 fixes this near-half candidate-generation policy alongside R45's accepted
recursion budget. Neither authorizes implementation, training, changes to
completed packages or a new run.
