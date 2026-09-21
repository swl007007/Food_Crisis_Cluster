# Results — IPCCH CH-heterogeneity and split-gate ablation

Six cells, `{all, non-CH, CH-only} x {0.01, 0.005}`, everything else held at the prior
baseline's values. Run directories `IPCCHGeoRFExperiment/runs/abl-C{1..6}-*`.

**Answer: neither factor is the problem.** The gate is the binding constraint on
*partition learning* and nothing else; the cohort split is real and large but does not
make either half learnable.

## A4 reproduction gate

C1 (`all` / 0.01) reproduces `ipcch-v1-20260920d`: target and Stage 1 manifest stages
are identical, Stage 3 differs only in the run-local path and `fit_seconds`
(1018.3 vs 996.4). 81,109 prediction rows in both. Every comparison below rests on that.

## Partition learning

| cell | cohort | gate | accepted splits | terminal branches | max depth | areas |
|---|---|---|---|---|---|---|
| C1 | all | 0.01 | **0** | 1 | 0 | 3,264 |
| C2 | all | 0.005 | **8** | 9 | 5 | 3,264 |
| C3 | non-CH | 0.01 | **0** | 1 | 0 | 2,067 |
| C4 | non-CH | 0.005 | **6** | 7 | 5 | 2,067 |
| C5 | CH-only | 0.01 | **0** | 1 | 0 | 1,197 |
| C6 | CH-only | 0.005 | **3** | 4 | 3 | 1,197 |

At 0.01 **no cohort learns a single split**. At 0.005 every cohort learns several. So
the strict gate, not the data, is what produced the prior baseline's flat map — and
dropping CH does not by itself unlock partition learning.

Gate overrides verified by readback in all three consuming namespaces
(`config`, `src.tests.sig_test`, `src.partition.transformation`), recorded per cell in
`manifest.json -> stages.stage1_fit.split_gate_override`.

## Forecast quality: class-1 F1, cohort `E_persist`, mean over horizons 1/3/6/12

| cell | cohort | gate | partitioned | pooled | xgb | persistence | part − persist |
|---|---|---|---|---|---|---|---|
| C1 | all | 0.01 | 0.6567 | 0.6496 | 0.6764 | 0.6765 | **−0.0198** |
| C2 | all | 0.005 | 0.6486 | 0.6496 | 0.6764 | 0.6765 | **−0.0279** |
| C3 | non-CH | 0.01 | 0.6628 | 0.6723 | 0.6851 | 0.6897 | **−0.0269** |
| C4 | non-CH | 0.005 | 0.6664 | 0.6723 | 0.6851 | 0.6897 | **−0.0233** |
| C5 | CH-only | 0.01 | 0.5396 | 0.5459 | 0.5752 | 0.6045 | **−0.0649** |
| C6 | CH-only | 0.005 | 0.5395 | 0.5459 | 0.5752 | 0.6045 | **−0.0649** |

Persistence is the best arm in **17 of 24** cell-horizon combinations; partitioned RF
wins exactly **1**. Partitioned RF is below persistence on average in every cell.

## The three findings

### 1. Relaxing the gate buys splits, not accuracy

Partitioned-RF change from 0.01 to 0.005, same cohort:

| cohort | h1 | h3 | h6 | h12 | mean |
|---|---|---|---|---|---|
| all | −0.0069 | −0.0324 | +0.0026 | +0.0045 | **−0.0080** |
| non-CH | −0.0118 | +0.0149 | +0.0087 | +0.0024 | **+0.0036** |
| CH-only | +0.0126 | +0.0011 | −0.0099 | −0.0041 | **−0.0000** |

Eight new splits on the full cohort make forecasts slightly *worse*. The splits the
0.01 gate was rejecting were not ones worth accepting — the gate was doing its job.

### 2. Dropping CH raises the numbers, but not the model

non-CH partitioned RF (0.6628 / 0.6664) beats the all-cohort figure (0.6567 / 0.6486).
Read alone that looks like an improvement. But persistence rises with it, 0.6765 to
0.6897, and the deficit against persistence is essentially unchanged: −0.0198 → −0.0269
at gate 0.01, −0.0279 → −0.0233 at 0.005.

**The cohort got easier; the model did not get better.** This is exactly the confound
the like-for-like reference in R5 was specified to expose, and the reason the design is
a factorial rather than the single double-change cell originally proposed.

### 3. CH is a genuinely different and harder regime — and separating it does not help

Every arm loses roughly 0.13 F1 on CH-only against non-CH, persistence included
(0.6045 vs 0.6897). The base rates measured before running confirm the asymmetry: CH
crisis rate 0.137 against 0.621 for the rest, across 42.5% of labeled rows.

So the hypothesis that CH is too heterogeneous to pool is **supported as a description
of the data** and **not supported as an explanation of the modelling failure**. Pulling
CH out gives the worst partitioned-versus-persistence deficit in the whole table
(−0.0649 at both gates), and it is the only cohort where relaxing the gate changes
nothing at all (−0.0000).

## What this rules out

The IPCCH line's failure to beat persistence is not caused by the split gate and not
caused by CH contamination. Both were plausible before this ablation and neither
survives it. Whatever is wrong is upstream of both.

One observation worth carrying forward: **XGB is the strongest learned arm in every
cell** (0.5752 to 0.6851) and wins 6 of the 24 combinations, while partitioned RF wins
1. If any learned model is going to close the gap on this data it is not the current RF.

## Limitations

* Development-window metrics only; no threshold or recipe selection was performed, and
  none of this is a frozen final evaluation.
* The audited-count gates (`check_target_gate`, `check_stage1_split_gate`) validate
  reproduction of the *unrestricted* source. Under a cohort filter the Stage 1 split
  gate is recorded as not-applicable with its realized counts, while the unrestricted
  target gate still runs and passes — so every cell still proves it read the same
  pinned source.
* The 2026 CH anomaly noted during planning (crisis rate 0.9708 on 137 rows, against
  0.13–0.19 in every prior year) is in the `partial_2026` period, which is reported
  separately and is not part of the main-window numbers above.
* Cells differ in cohort size, so their confidence intervals are not directly
  comparable across rows; the country-block bootstrap intervals are in each cell's
  `reports/main/bootstrap_summary.csv`.

## Evidence locations

Per cell, under `IPCCHGeoRFExperiment/runs/abl-C*/`:

* `manifest.json -> stages.target.gate.cohort_filter` — kept/dropped areas, rows and
  class counts, plus the full-source gate totals the restriction was applied after.
* `manifest.json -> stages.stage1_fit.split_gate_override` — requested gate and the
  readback from all three consuming namespaces.
* `manifest.json -> stages.stage1_fit.learned_map` — accepted splits, terminal
  branches, depth, learned areas.
* `data/cohort_filter.json` — the same cohort evidence standalone.
* `stage1/split_gate.json` — realized split counts; for restricted cohorts it carries
  `gate_applicable: false` with the reason.
* `reports/main/bootstrap_summary.csv` — per-arm F1 with country-block intervals.

Cohort counts reconcile exactly to the audited source totals:
24,336 + 18,359 = 42,695 valid rows, and 4,919 + 1,308 = 6,227 areas.
