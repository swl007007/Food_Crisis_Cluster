# Stage 1 diagnostic summary (saved artifacts, 2018–2020; D4/G4, implement §6)

2026-10-03, Claude executor. Read-only over the accepted run `C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1\stage1_scenario`. No fit, no RUN write, no product/test edit, no 2025 read.

**Command.** From `FEWSNETGeoXGBExperiment/`, pinned Windows Python 3.12.10, 6 min 11 s:

```
$PY -B ../.trellis/tasks/10-02-exogenous-transition-forecast-design/research/probes/stage1_diagnostics.py 'C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1'
```

**What it reuses.** All existing helpers:
- `acceptance.accept_scenario_stage1`;
- `run_stage2.scenario_candidate_row`, which recounts E3 from `target_predictions.csv` against the candidate record and assigns the NA status;
- `run_stage2.crisis_plan_weights`, the existing E4 matched-E3 clipped-logit utility;
- `run_stage2.diagnostics`, the coverage-matched canonical partitions.

**Outputs.** `research/probes/stage1_diagnostics_ledger.csv` (648 rows) and `research/probes/stage1_diagnostics_summary.json`.

**Scope of these figures.** Every number below is a candidate-level diagnostic distribution. The same evaluation keys recur across the 54 candidates of a cell (9 targets × 2 ratios × 3 seeds), so they are **not** independent observations. None of these is the 72-fold selection statistic, which comes from `scen-select` on complete development forecasts.

## Status and NA

- All 648 candidates are `scored`.
- There are 0 `e3_undefined`, 0 `no_e3_target_labels` and 0 `root_insufficient_support`.
- F/S/C/E3 crisis-F1 NA reasons: none in any role (0 NA rows per cell).

## Original-key role support (fold_membership, unique (area, target_month))

Roles are disjoint: 0 duplicate keys within a role and 0 keys shared across roles, in every cell. For B, these are original keys; the k′ variant rows are not counted. That matches the coordinator's B spot check (81,979 keys vs 245,937 rows).

| Cell | F fitting (median) | S validation | C confirmation | E3 target |
|---|---|---|---|---|
| H4 k0/k1 | 51,439 | 15,397.5 | 15,397.5 | 5,365 |
| H8 k0/k1 | 53,113.5 | 15,397.5 | 15,397.5 | 5,365 |
| H4/H8 k2 | 48,931 | 14,221 | 14,221.5 | 5,365 |

A and B have identical medians: they share the same original keys, and B adds weighted variants only. The full ranges per cell are in the summary JSON.

## Partition minus own root crisis F1 by role (median [q10, q90]; share of candidates > 0)

S is the search pool, so its gap is optimistic by construction. C is the post-freeze held-out confirmation (diagnostic, D29). E3 is the out-of-time target and the only E4 input.

| Cell | F | S | C | E3 |
|---|---|---|---|---|
| A h4 k0 | .0092 [.0047,.0156] .98 | .0140 .98 | .0079 [.0011,.0142] .93 | **−.0002 [−.0092,.0059] .48** |
| A h4 k1 | .0141 .98 | .0218 .98 | .0106 .94 | **.0002 [−.0079,.0109] .52** |
| A h4 k2 | .0116 .96 | .0180 .96 | .0112 .96 | **−.0020 [−.0064,.0039] .24** |
| A h8 k0 | .0056 .94 | .0124 .94 | .0049 [.0000,.0093] .87 | **.0019 [−.0038,.0158] .61** |
| A h8 k1 | .0083 .98 | .0173 .98 | .0064 .94 | **−.0007 [−.0071,.0080] .44** |
| A h8 k2 | .0077 .98 | .0196 .98 | .0058 .93 | **−.0007 [−.0094,.0078] .44** |
| B h4 k0 | .0078 .93 | .0136 .93 | .0058 .91 | **.0000 [−.0153,.0144] .48** |
| B h4 k1 | .0164 .98 | .0275 .98 | .0141 .98 | **.0001 [−.0112,.0153] .50** |
| B h4 k2 | .0228 .98 | .0344 .98 | .0199 .98 | **.0015 [−.0096,.0114] .54** |
| B h8 k0 | .0060 .94 | .0153 .94 | .0062 .91 | **.0006 [−.0033,.0101] .59** |
| B h8 k1 | .0077 .91 | .0160 .93 | .0076 .93 | **.0000 [−.0049,.0065] .46** |
| B h8 k2 | .0089 .96 | .0194 .98 | .0081 .91 | **.0050 [−.0020,.0156] .80** |

**Negative result, preserved.** Within the 2018–2020 period, partitions beat their own root on F, S and C in roughly 87–98% of candidates. Out of time, on E3, the median gain is about zero (−.0020 to +.0050), and only 24–80% of candidates gain at all. Within-period partition gains largely do not transfer to the target month. Clear E3 gains appear only in B h8 k2 (median +.0050, 80% > 0); A h4 k2 leans negative (24% > 0). These are candidate distributions, not tests.

## Absolute role levels and F→C / F→E3 gaps (from the saved ledger only)

Computed from `research/probes/stage1_diagnostics_ledger.csv`, with no rerun of acceptance. Values are medians over a cell's 54 candidates: root/final crisis F1 per role, and the per-candidate medians of the F−C and F−E3 differences. These are diagnostic only.

| Cell | F root/final | S root/final | C root/final | E3 root/final | F−C root | F−E3 root | F−C final | F−E3 final |
|---|---|---|---|---|---|---|---|---|
| A h4 k0 | .6664/.6739 | .6607/.6719 | .6543/.6628 | .6221/.6206 | +.0109 | +.0659 | +.0129 | +.0728 |
| A h4 k1 | .5463/.5581 | .5439/.5638 | .5381/.5500 | .5132/.5165 | +.0115 | +.0367 | +.0123 | +.0468 |
| A h4 k2 | .4995/.5113 | .5014/.5211 | .5015/.5105 | .5287/.5274 | +.0039 | −.0173 | +.0058 | −.0052 |
| A h8 k0 | .7256/.7315 | .6930/.7074 | .6924/.6981 | .5208/.5250 | +.0347 | +.2067 | +.0348 | +.2081 |
| A h8 k1 | .6733/.6808 | .6477/.6647 | .6471/.6512 | .5320/.5301 | +.0285 | +.1423 | +.0304 | +.1523 |
| A h8 k2 | .6292/.6367 | .6039/.6250 | .6020/.6103 | .4859/.4863 | +.0272 | +.1398 | +.0284 | +.1464 |
| B h4 k0 | .6554/.6625 | .6496/.6644 | .6447/.6505 | .5579/.5574 | +.0106 | +.1060 | +.0098 | +.1173 |
| B h4 k1 | .5632/.5792 | .5562/.5833 | .5539/.5686 | .4533/.4629 | +.0105 | +.1091 | +.0152 | +.1189 |
| B h4 k2 | .5215/.5442 | .5242/.5607 | .5228/.5440 | .4935/.4831 | +.0070 | +.0475 | +.0092 | +.0652 |
| B h8 k0 | .7075/.7148 | .6806/.6973 | .6770/.6827 | .4651/.4700 | +.0295 | +.2513 | +.0306 | +.2556 |
| B h8 k1 | .6993/.7052 | .6713/.6861 | .6676/.6753 | .4611/.4602 | +.0318 | +.2348 | +.0321 | +.2406 |
| B h8 k2 | .6725/.6829 | .6445/.6651 | .6422/.6485 | .4260/.4340 | +.0350 | +.2436 | +.0355 | +.2460 |

Reading, separating the root model's own overfit from partition-minus-root gains:
- **The root itself scores higher on its fitting rows than on held-out C.** Root F−C is +.004 to +.012 at H4 and +.027 to +.035 at H8. This is the root model's in-sample advantage, present before any partition, and it is about as large as, or larger than, the partition-minus-root gains on C (+.005 to +.020, table above).
- **Final F−C is almost the same as root F−C** (within about .005 per cell). Partitions add little extra in-sample optimism beyond the root's own.
- **The out-of-time drop dominates.** Root F−E3 is up to +.25 (H8, larger for B), against E3 partition-minus-root gains of about 0.
  - The F−E3 gap mixes overfit with temporal shift and a different target-month composition and crisis base rate, so it is not an overfit measure by itself.
  - In A h4 k2, the E3 root is *above* its F level (−.0173).
  - B's E3 levels are lower than A's in every cell (e.g. h8 k0 root .4651 vs .5208), although F/S/C levels are similar.

## Root-only, fallback and maps

| Cell | Root-only (n_terminal = 1) | E4 positive weight | Coverage groups | Same-coverage duplicates | Genuine unique partitions | Max weight share |
|---|---|---|---|---|---|---|
| A h4 k0/k1/k2 | 1/1/2 | 26/28/13 | 4/4/4 | 0/0/0 | 54/54/54 | .171/.199/.289 |
| A h8 k0/k1/k2 | 3/1/1 | 33/24/24 | 4/4/5 | 2/0/0 | 52/54/54 | .135/.114/.166 |
| B h4 k0/k1/k2 | 4/1/1 | 26/27/29 | 4/4/4 | 1/0/0 | 53/54/54 | .099/.122/.107 |
| B h8 k0/k1/k2 | 3/4/1 | 32/25/43 | 4/4/5 | 1/2/0 | 53/52/54 | .116/.124/.060 |
| **A all** | 9 of 324 | 148 | 6 | 5 | 319 | .042 |
| **B all** | 14 of 324 | 182 | 6 | 9 | 315 | .021 |

- **E3 routing fallback.** Rows routed `root_unassigned_test_area` (an area with no learned assignment, routed to the root) make up 1,698–2,592 of about 291,762 E3 rows summed over a cell's 54 candidates, i.e. 0.6–0.9%. All other rows are `terminal_branch`.
- **E4 weights.** Weights are zero for candidates whose E3 gain is not positive: about 20%–76% of candidates per cell (B h8 k2: 11/54 zero, 43/54 positive; A h4 k2: 41/54 zero). A cell's weight mass therefore rests on 13–43 candidates. The maximum single-candidate share is up to .289 (A h4 k2), which indicates concentration, not an effective N.
- **Map pools in scen-develop.** These are per strategy and origin, pooling horizons and k, so the cell rows above describe inputs rather than the maps actually built. Each development map's own record under `scenario_maps/<id>/` holds its route, including any no-scorable-evidence or null fallback.

Limitations:
- **S gaps are selection-inflated.** S is the search pool.
- **Candidate distributions are descriptive only.**
- **E4 weights were recomputed here** with the existing utility on the full 648 ledger. The weights actually used by each development map are those in its `scenario_maps` record, built on its origin-legal subset.
