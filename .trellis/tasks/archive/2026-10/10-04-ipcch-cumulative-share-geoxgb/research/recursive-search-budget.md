# Recursive search budget: source evidence and accepted R45

Planning only, 2026-10-04. The user accepted the budget below as R45 / v0.39,
following R44 configuration selection. No model fitting, scan execution or experiments ran.
Two read-only scouts inspected separate recursion and scan paths. GitNexus
query failed with the read-only shadow-page replay error; no index was rebuilt.
Paths below are relative to `FEWSNETGeoXGBExperiment/`.

## Source recursion and actual local capacity are different counters

- `config.py:174-175` sets MIN_DEPTH=1, MAX_DEPTH=6. The value passes through
  `app/main_model_GF.py:155` and `src/model/GeoRF.py:433` into partition.
  `src/partition/transformation.py:318-334` searches `range(max_depth-1)`;
  `src/helper/helper.py:234-247` constructs branch IDs of length i, and
  transformation.py:1079-1080 appends one character for children. Therefore this
  configuration permits parent depths 0..4 and child membership depth at most 5:
  31 binary search positions and 32 terminal membership leaves before other gates.
  The 64-row allocation at transformation.py:179 is not a 64-leaf result.
- transformation.py:180-189 requires macro_mode and activates only the root.
  The MIN_DEPTH forced-split branch at 893-967 is unreachable in that required
  mode. The new task independently requires strict gain at every accepted split.
- `src/experiment/plan.py:39-44` sets L1=20, L2=40 and PATH_ROUND_CAP=80.
  transformation.py:817-824 checks parent path rounds plus the local increment.
  `src/model/native_xgb.py:345-359` always starts a shared-root child from its
  corresponding saved root, appending only the selected increment, but adds that
  increment to `path_selection_rounds`. `path_rounds()` returns this selection
  counter in root mode (396-401); global rounds are not part of the counter.
- transformation.py:862-867 overwrites an unselected child with the parent model
  and metadata. Such an inherited child can continue recursion after an accepted
  split (1126-1128), without increasing that path's selection-round counter.
  A fitted but discarded child still consumed compute. Thus 80/L = 4 or 2 bounds
  retained local-route increments along a path, not membership depth or total fits.
  Retaining this cap would give L1 and L2 different search opportunities despite
  both final predictors retaining only one local increment.

## Source work per parent and stop conditions

- transformation.py:388-407 skips zero exposure/error mass and otherwise calls
  scan once, returning one binary membership candidate. Empty children (790-795),
  no eligible children (831-836), or failed complete-routing gain end that parent.
  Only accepted splits activate descendant membership nodes (1061,1126-1128).
  There is no same-parent retry loop.
- `src/model/train_branch.py:41-51` visits suffixes 0 and 1 once; each eligible
  child fits once, and an ineligible child loads/predicts/saves its parent route.
  `src/partition/partition_opt.py:880-890` compares the available child/parent,
  parent/child and child/child routes using existing predictions, without new fits.
  This describes the source classifier, not an already implemented regression
  quartet. The new quartet budget must multiply eligible child fits by four.
- Fitting-only source areas outside both candidate groups retain parent routing
  (transformation.py:797-804). Terminal membership leaves should not be confused
  with all stored checkpoints or distinct prediction providers.

## Source scan cost and size-rule warning

- partition_opt.py:967,992-1025 sets 1000 coordinate-descent iterations and returns
  the last candidate, not a best-over-iterations candidate. The convergence break
  is commented out (1008-1009). There is no random restart; random initialization
  is also commented out (980).
- Initialization performs one grouping per input statistic column (970-978),
  before the 1000 iterations. The accepted single crisis column therefore needs
  one initialization, not four target-wise scan restarts. Iterations update scan
  membership/rho; they do not fit XGB models. Sorting and size search occur inside
  grouping, and the adopted three smoothing rounds are additional operations.
- `config.py:216-220` uses FLEX_RATIO=.1, FLEX_OPTION=True, FLEX_TYPE=n_group.
  partition_opt.py:243-268,835-858 varies the split around half the group count;
  it does not impose a 10 percent minimum child size. For N groups, let h=ceil(N/2),
  a=ceil(.9h), b=ceil(1.1h). The source loops `range(a,b)` and can set
  `optimal_size=size-1`; possible sizes are h or a-1 through b-2, subject to
  score conditions and floating-point ceiling. Small groups may produce empties.
  This off-by-one behavior is a source fact, not an adopted new-task rule.
- Smoothing can change these sizes without reapplying flex (transformation.py:
  478-489; partition_opt.py:629-660). Final original-key support checks are
  independent and occur afterward (transformation.py:810-820), as required by R38.
  R46 subsequently fixes new exact size/balance and deterministic score-tie rules
  in scan-candidate-size.md; do not silently copy the legacy index behavior.

## Accepted R45 budget for the first version

1. Set an explicit maximum membership depth of 4 with root depth 0: only parent
   depths 0..3 may be searched; depth-4 children are terminal. Apply the same cap
   to every H and G/L pair. Do not also use the source's 80 selection-round cap.
   XGB tree depth and actual 20/40 appended rounds remain governed by R43.
2. Each eligible parent gets at most one scan call, with one deterministic
   single-column initialization and 1000 coordinate-descent iterations, returning
   the final membership candidate. Do not add seed restarts, alternative scan
   candidates, repeated visits or retries after rejection. Zero scan mass stops
   before scanning; technical errors follow R41 rather than generating retries.
3. Apply the accepted three smoothing rounds, then support checks, at most one
   fit per eligible child quartet, and the adopted complete-routing strict F1
   gate. Only accepted membership splits may recurse. A child that inherits the
   parent after an accepted split still consumes one membership-depth level.
   There is no minimum required split count and no forced split.
4. Record membership depth, attempted scans/fits, accepted routing and stop
   reasons separately from actual appended booster rounds. R46 subsequently fixes
   candidate size and score-tie semantics separately in scan-candidate-size.md.

### Conditional mathematical ceiling, not observed runtime

Under R45 each candidate has at most 1+2+4+8=15 parent scan calls and
16 terminal membership leaves. At most two child quartets per parent gives
30 local quartets = 120 scalar regression fit calls, plus four global fits.
The initialization/sorting/smoothing/predictions are not included in that fit
count; intermediate scan iterations do not multiply it. Failed support/gain
reduces work. Technical failure makes the affected run incomplete under R41.

Across the 32 candidate recipes this is at most 480 scans and 3840 local scalar
fit calls. Counting each candidate's four globals separately gives a conservative
Stage1 ceiling of 3968 scalar fits before reuse. R48 subsequently authorizes the
design of identical global-root reuse across L, reducing that ceiling to 3904,
and specifies Stage3 reuse/total accounting in fit-reuse-budget.md. Exact fitting
keys, configuration, feature/schema identity and provenance must match. These
are planning contracts, not runtime estimates or authorization to train models.

The depth-4 cap limits repeated search on S while allowing several spatial scales.
It may stop a beneficial finer split; it is an engineering budget, not evidence
of an optimal depth or a statistical overfitting guarantee.
