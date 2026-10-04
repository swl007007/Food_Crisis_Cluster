# XGB capacity references, accepted sharing and numeric grid

Planning-only inspection during grill, 2026-10-04. R41 error handling was accepted
in v0.35. Configuration sharing below is accepted as R42 in v0.36; the numeric
grid is accepted as R43 in v0.37. Two
read-only scouts failed with provider usage-limit errors; the parent inspected
the bounded source/config files directly. No search, fitting or experiments ran.

## FEWS GeoXGB parameter-table source evidence

`FEWSNETGeoXGBExperiment/src/experiment/plan.py:23-44` defines:

| Candidate | max_depth | boosting rounds | learning rate |
|---|---:|---:|---:|
| G1 | 3 | 200 | .05 |
| G2 | 3 | 400 | .05 |
| G3 | 4 | 200 | .05 |
| G4 | 4 | 400 | .05 |
| L1 | 1 | 20 appended | .05 |
| L2 | 2 | 40 appended | .05 |

Global common settings: min_child_weight=10, reg_lambda=10, reg_alpha=0,
subsample=.8, colsample_bytree=.8. Local common settings: min_child_weight=20,
reg_lambda=20, reg_alpha=1, subsample=1, colsample_bytree=1.
The source uses gbtree, hist, CPU, seed42, nthread4, num_parallel_tree=1.
Its multi:softprob/num_class=4 backend is not the new squared-error quartet.
Hessian-based min_child_weight also must not be described as the same effective
row-support threshold under different objectives; R28's original-key floors are
separate checks. R43 subsequently adopted the numeric recipes below; their
predictive suitability for this new regression task has not been measured.

`src/model/native_xgb.py:177-199,213-254` uses fixed xgb.train num_boost_round;
those calls supply no validation early stopping. `plan.py:220-223` orders exact
G ties by fewer rounds, shallower trees, then configuration ID. Fixed booster
rounds were accepted in R43; R44 subsequently fixed the joint configuration tie
policy in configuration-selection.md. R45 subsequently fixed the shared recursion
budget in recursive-search-budget.md; R46 fixes scan size/tie semantics separately.

`src/model/native_xgb.py:310-316,338-359` stores one local_config on the Stage1
model adapter. Under shared-root mode every fitted child starts from the saved
root once, while path_selection_rounds accumulates search opportunity. Therefore
PATH_ROUND_CAP=80 is not permission to append all ancestor increments to a final
booster. Partition recursion depth and predictor tree depth are distinct.

The historical source locks G by horizon (`plan.py:60-61`, H4/H8/H12), and its
scenario fixes G and L1 (`plan.py:228-236`); `scripts/run_stage1.py:153-208` reads
G per H. These are FEWS-specific selections, not measured IPCCH winners. The
new H1/H3/H6/H12 winners remain unmeasured; their no-Stage2 selection rule is now
fixed by R44, without adopting the source's selected FEWS winners.

## The sibling IPCCH quartet has a separate q3 recipe

`../IPCCH/CLAUDE.md:72-76` identifies separate phase3 and other-phase JSON files.
`../IPCCH/src/ipcch/launch_nowcasting.py:1294-1322` loads the canonical forecasting
pair and chooses the p3 file only for phase3_worse. The seed is set by the caller.
Each chosen estimator is a separate XGBRegressor.

Current files `../IPCCH/configs/{forecasting,contemporaneous}_hyperparameters{,_p3}.json:1-10`:

| Reference configuration | depth | estimators | eta | subsample | colsample | gamma |
|---|---:|---:|---:|---:|---:|---:|
| forecasting q2/q4/q5 | 11 | 200 | .1 | 1 | .5 | 0 |
| forecasting q3 | 9 | 200 | .1 | .5 | .7 | .1 |
| contemporaneous q2/q4/q5 | 3 | 200 | .1 | 1 | .7 | .1 |
| contemporaneous q3 | 3 | 200 | .01 | 1 | 1 | .1 |

All four JSONs specify min_child_weight=0 and scale_pos_weight=1; they do not
explicitly specify reg_alpha/reg_lambda. Do not invent pinned defaults for
unspecified parameters. The inspected files establish the choices, not a new
empirical justification for them. User R3 adopts the four-regression/decoder
decision flow, not these depths or the separate-q3 tuning policy.

## Population-history pooled q3 reference

`IPCCHPopulationHistoryExperiment/config/candidate-configs.json:3-67` has a base
of 400 trees, depth6, eta=.05, min_child_weight5, lambda1, alpha0, row/column .8,
seed5, CPU hist, base_score=.5, max_bin256 and one job. X0-X5 vary depth3/4/6,
regularization, feature subsampling and an 800-tree eta=.025 option. Its regressor
objective is reg:squarederror. This is a pooled q3 candidate table, not an existing
shared spatial quartet/local-increment selection. The file's retained status
field is not approval authority for the new task.

## Accepted R42: one global/local recipe pair per horizon

User-approved policy: each H selects one global recipe G_H and
one local-increment recipe L_H using the adopted development-only crisis-F1
objective. Within that H, share G_H across q2..q5 global boosters and L_H across
all four targets and learned regions. Use the same frozen recipe pair in Stage1
and all Stage3 historical/current fits; do not retune by target, region or date.
Different H may choose different pairs from the eventual bounded candidate menu.

This shares hyperparameters only. All four boosters, each region's fitted
increments and every legal rolling refit remain separate; learned trees, leaf
values and target-specific global prefixes are not shared across q targets.
No across-target regularization of the learned weights is implied.

This limits search multiplicity and makes spatial adaptation arise from regions
and local residual fitting. The trade-off is reduced flexibility for q3-specific
capacity or rare q4/q5 targets. Per-target recipes, as in sibling IPCCH, would be
a different first-version policy with a larger selection problem.

R42 only fixed sharing scope. R43 subsequently adopted the numeric grid, seed and
fixed-round fitting policy below. R44 subsequently fixed candidate scoring on the
common development S and tie policy; R45 subsequently fixed membership depth 4,
one scan per parent and 1000 iterations without the source's 80-round path cap.

## Accepted R43: bounded numeric G/L menu

User accepted in v0.37: use the four G and two L options from
the FEWS table above, adapted explicitly to four independent reg:squarederror
boosters. The first-version candidate menu has eight G/L pairs per H, 32 recipe
pairs across H1/H3/H6/H12. This counts recipes, not boosters or fitting operations.
These are configuration candidates for selecting one map/recipe per H, not a
consensus-map ensemble; R34 remains in force.

| Configuration | max_depth | fixed boosting rounds | eta | min_child_weight | reg_lambda | reg_alpha | subsample | colsample_bytree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G1 | 3 | 200 | .05 | 10 | 10 | 0 | .8 | .8 |
| G2 | 3 | 400 | .05 | 10 | 10 | 0 | .8 | .8 |
| G3 | 4 | 200 | .05 | 10 | 10 | 0 | .8 | .8 |
| G4 | 4 | 400 | .05 | 10 | 10 | 0 | .8 | .8 |
| L1 | 1 | 20 appended | .05 | 20 | 20 | 1 | 1 | 1 |
| L2 | 2 | 40 appended | .05 | 20 | 20 | 1 | 1 | 1 |

Fixed reproducibility settings: gbtree, reg:squarederror, CPU hist, one tree per scalar
target per boosting round (num_parallel_tree=1), seed42, nthread4, gamma0,
max_delta_step0, max_bin256, grow_policy=depthwise. Explicit global base_score=.5;
local continuation preserves each global base score and trees, never overrides
base_score or refreshes existing trees. The numeric base score is an initialization,
not a constant predictor, fixed target mean or replacement for fitting.

Use the stated fixed rounds, no early stopping or seed selection, and no classifier
num_class/multi:softprob/scale_pos_weight behavior. Tree depths, regularization and
sampling above come from the source GeoXGB recipes; seed42/CPU hist likewise.
Explicit base_score=.5, gamma0, max_delta_step0, max_bin256 and depthwise are the
regression reproducibility choices, also spelled out in the population-
history configuration. They are not claims of a new measured optimum. Pin actual
XGBoost/software versions separately before implementation.

Under shared-root, every eligible local target appends only its selected 20 or
40 rounds to that target's global booster, even at a deep partition node. With
one scalar tree per round the quartet stores 4*(G rounds + L rounds) trees for a
fully local route (880 to 1760 across this menu); this is not the recursive search
fit count. Root-only fallback has 4*G rounds and no invented local increment.

The menu offers two global depths and training lengths, with shallow and more
strongly regularized local increments. It limits search freedom, but may underfit
or suppress small share corrections, especially for rare upper phases. Reusing
classification-origin regularization values does not establish their superiority
for continuous shares. Report all adopted metrics and global/persistence controls;
do not enlarge the menu after examining main-period performance.

R43 freezes this candidate menu and fixed fitting recipe. R44 fixes candidate
scoring/calendar and tie rules in configuration-selection.md. R45 fixes the
recursive scan/depth budget in recursive-search-budget.md; R46 fixes scan size/tie
rules; R47 fixes the prediction schedule and coverage. R48 fixes model reuse and
conservative total fit ceilings in fit-reuse-budget.md. No fitting is authorized.
