# P3 evidence — direct Stage1 maps and frozen winners (2026-10-04)

Phase status: implemented and tested on synthetic data (logic tests with
deterministic fake quartets; an end-to-end run with real XGBoost quartets,
the real frozen adjacency and a reduced test-only contract). No Stage1 run on
project data: that is part of the formal P6 run after the implementation-ready
freeze review.

## Implementation

| Module | Content |
|---|---|
| `stage1.py` | R46 `size_bounds`/`choose_prefix` (integer 45-55%, descending score, area-ID ties, max prefix sum, then smaller abs(2m-N), then smaller m); R32 single-column scan (c=Y-A, b=sum(c)Y/sum(Y), init c/b grouped by the same rule, 1000 iterations, last candidate, rho=0 or empty expectation -> 1); R38 synchronous smoothing (`9*same < 4*total`, isolated/no-labelled-neighbour keep, outside-node areas do not vote); R28/R29 support (keys/areas/months + 20/20 crisis classes on S); clarification-1 route gate (parent/parent base; child/parent, parent/child, child/child; strict exact gain > 0; parent wins ties; undefined parent never splits); R45 recursion (depth <= 4, parents at 0..3, one scan per parent, no path cap, inheriting children recurse); children continue the ROOT global once; membership vs provider kept separate; R38 connectivity diagnostics; R44 selection and tie order |
| `learnmap.py` | `learn-map --run-id`: re-hashes every prepared artifact against its manifest, verifies runtime lock and adjacency cache identity, fits each (H, G) root once and shares it across L1/L2 through the R48 store, runs 8 candidates per H, routes the complete S set through terminal providers, saves keyed raw/projected predictions, terminal maps (membership + provider), decisions, selection ledger, and freezes the winner's membership map (`frozen_map_hNN.csv` + `frozen_hNN.json` with map SHA256, recipe, connectivity) |
| `cli.py` | `learn-map`; exit 4 for R41 technical failures |

Membership node IDs are `"r"` + one binary digit per accepted split
(`r0`, `r01`, ...) and are validated on read. A first integration run exposed
that purely binary IDs (`"00"`) lose leading zeros under default CSV dtype
inference; the prefix makes that impossible (this repository has a past
partition-ID string-format bug of exactly this class).

## Tests

`python -m pytest -q`: **175 passed** (P0 70, P1 41, P2 33, P3 31), `evidence/P3-pytest.log`.
P3 covers: R46 bounds (N=100→45..55, 101→46..55, 2→1..1, 3/1/0 infeasible),
area-ID tie ordering, |2m-N| and smaller-m tie rules, max prefix sum; scan
determinism and concentration on the high-error half, zero-mass groups counted
in N; smoothing synchronous behaviour, exact 4/9 keep vs 3/9 switch, outside
and isolated areas; component diagnostics; support equality edges for every
floor key; route gate fixed order, first-improvement tie, ineligible child
skipped, no strict gain, undefined parent; recursion depth/terminal/scan/fit
ceilings, zero-error stop after accepted oracle children, root-only
equivalence (all S rows routed to the root provider), no eligible child, depth
budget 0; node-ID validation; R44 absolute F1 and tie order, NA excluded,
all-NA unavailable. End-to-end: 4 candidates, every candidate's exact F1
recomputed from its saved keyed S predictions equals the ledger, frozen map
covers all learned areas with string IDs and matching SHA256, connectivity
covers every terminal region, each root fitted once per G and shared across L,
a two-regime world produces accepted splits, a tampered prepared artifact stops.

## Compute probe (synthetic, real shape; not a project-data fit)

Pinned runtime, 561 columns, 40% NaN: global quartet G1 7.2 s / G4 24.2 s at
8,561 rows, 9.8 s / 28.2 s at 20,000 rows; L2 local quartet on 3,000 rows
2.1-2.2 s; predicting 8,561 / 20,000 rows 0.34 / 0.68 s. Stage1 per H is
about 1 minute of roots plus at most about 8 minutes of locals
(8 candidates x <= 30 local quartets).
