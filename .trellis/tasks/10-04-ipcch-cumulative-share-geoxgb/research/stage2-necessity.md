# Stage2 necessity investigation for IPCCH

Date: 2026-10-04. Read-only investigation of existing packages and saved artifacts;
no source-data rescan, fitting, clustering or task-lifecycle changes. Three bounded
scouts examined IPCCH execution lineage, package identity, and FEWS Stage2 code.
Parent read the historical brainstorm/support notes and spot-checked manifest,
ablation results and Stage2 source. Paths below are repository-relative unless noted.

## Finding and recommendation

The historical IPCCH spatial package explicitly omitted Stage2: one pooled
partition-learning procedure across time and H=1/3/6/12, followed by geographic
completion and rolling Stage3 prediction. Sparse validation and target-month
scoring support were documented before that decision. This was a design choice,
not a consensus job that ran and then failed for too few accepted maps.

The user accepted omitting consensus Stage2 in the first new cumulative-share
GeoXGB design (v0.28, R34). Each H directly learns/freezes its own recursive
partition map, then uses the approved rolling global/local four-regressor
prediction and adoption gate. This retains the approved horizon-specific maps,
regression/decoder, scan statistic, and local gain/support gates.

The previous logit-weight proposal is not adopted: a candidate pool and its useful
diversity/scoring support have not been established for the new IPCCH design.

## 1. Which IPCCH package actually had no Stage2?

- `.trellis/tasks/archive/2026-09/09-19-ipcch-binary-georf-pipeline/prd.md:12-14,156-160`
  explicitly prescribes one pooled partition and no monthly/annual ensemble,
  spectral k40 or Stage2 consensus.
- Same task `research/brainstorm.md:45-54`: user superseded consensus; one fit pools
  the four horizon views and shares the resulting map. Four views are not four
  candidate maps. A single partition-learning procedure still fits root and child
  models recursively.
- `IPCCHGeoRFExperiment/run_pipeline.py:2261-2263,2399-2418` implements that fit.
  `IPCCHGeoRFExperiment/runs/ipcch-v1-20260920d/manifest.json:2-4` records
  scope=stage1_and_stage3 and status=stage3_complete; no Stage2 output directory.
- The spatial completed package found is GeoRF with a pooled binary XGB benchmark.
  `IPCCHPopulationHistoryExperiment/README.md:12-26` explicitly excludes partition
  learning and describes pooled q3 XGB regression. Sibling `../IPCCH/CLAUDE.md:58-64`
  describes four cumulative-share regressors and the phase cascade, not a spatial
  consensus pipeline.
- Bounded filename/task/config discovery in these two repos and nearby experiment
  directories found no additional completed spatial GeoXGB-on-IPCCH package.
  Older archived GeoXGB directories were located but their entire dataset lineage
  was not exhaustively checked; this is not a claim of filesystem-wide absence.

## 2. Was sparse support the original concern?

Yes, with an important distinction between measured support and candidate-map counts.
The old task `research/stage1-support.md:8-9` records concern that 36-month windows
leave too few early partitions. Its `:99-123` documents two separate sparse-support
problems under the OLD binary-label contract:

| Historical diagnostic | Recorded value |
|---|---|
| 2022 IPC/H12 areas with >=1 possible inner-validation row, monthly median | 1,275 |
| Same with >=2 rows | 36.5 |
| Same with >=3 rows | 0 |
| 2022 IPC target-month scored areas | median 96; min 5; max 635 |
| Target months with fewer than 20 scored areas | 4 of 12 |
| 2022 CH months with observed labels | 2 |

Fractional values are medians across months. These are saved historical support
audits, not fresh counts for the new >=20% decoder, new model or new gates.
The note `:125-134` proposed annual candidates with broader scoring, explicitly
trading fewer candidates for more support and without claiming independence or
stability. That proposal was superseded by one pooled fit/no Stage2 (`:3-6`).
The note `:19-21` explicitly had not evaluated successful splits or positive weights.
Therefore evidence supports the user's recollection of sparse candidate support,
but does not establish a measured minimum number of distinct viable maps or a
formal statistical impossibility of consensus on all future IPCCH designs.

## 3. What did the saved IPCCH run learn?

- `runs/ipcch-v1-20260920d/manifest.json:296-310` under IPCCHGeoRFExperiment:
  accepted_splits=0, terminal_branch_count=1, max_branch_depth=0; 3,264 learned areas.
- `stage1/georf_fit_stdout.txt:3479-3480`: parent F1 .803606, candidate .811040,
  rejected under the old strict >.01 gate. Rejected child checkpoints are not
  accepted maps available to a consensus ensemble.
- `stage1/partition_codes.json:2-6` maps only the root branch to code 0.
  Scout counted `stage1/area_assignments.csv`: 5,631 areas code 0, 596 unresolved
  code -1; donor completion adds area membership, not additional partitions.
- Later `.trellis/tasks/archive/2026-09/09-21-ipcch-ch-gate-ablation/RESULTS.md:18-25`
  records 8/6/3 splits (9/7/4 terminal regions) for all/non-CH/CH-only at .005.
  These are different cohort/gate experiments. Multiple regions inside one map
  are not multiple candidate maps, and these cells cannot be pooled as though
  they shared the new experiment's contract.

The old zero-split outcome must not be used to predict zero splits for the new
four-regressor design: model, truth boundary and Stage1 gain gate have changed.

## 4. Why FEWS Stage2 is not automatically useful with one map

Current FEWS code has a designed multi-candidate schedule. For example,
`FEWSNETGeoXGBExperiment/scripts/prepare_fourclass.py:388-409` and
`src/experiment/plan.py:10,39-48,72-84` combine dates, horizons, split ratios, seeds,
local capacities and gate families. The 648 scheduled random candidates are a
schedule count, not a measured completed or distinct-map count; each actual
consensus pool further filters them. The separate interruption schedule is a
different pool and must not be added to that count.

- `scripts/run_stage2.py:202-224` records candidate/positive-weight/duplicate-map
  counts and weight concentration. It explicitly says the concentration measure
  is not an effective sample size.
- `:247-271` only bypasses construction for no candidates, no scorable evidence,
  or zero positive weights. There is no >=2 positive/distinct-map eligibility gate.
- `:273-298` therefore runs the complete chain even with one positive-weight map.
- `scripts/step4_similarity_matrix.py:141-165,192-207,273-277` aggregates weighted
  co-membership and applies Gaussian distance weights and normalization.
  `step5_sparsification.py:53-92` applies top-40 similarity sparsification and
  eigengap; `step6_complete_clustering_pipeline.py:173-202` performs spectral
  clustering and completion from the largest component.

For identical coverage and identical positive-weight maps, aggregation is just
the same co-membership matrix times a scalar, which normalization removes.
It provides no new co-membership information. Geometry/spectral processing can
still change the result; this is additional spatial processing, not evidence of
consensus across differing learned maps. Useful consensus does not require maps
to be statistically independent, but it does require an informative pool and
defensible evaluation; simply multiplying seeds/files does not prove either.

No FEWSNETGeoXGBExperiment/runs directory or saved candidate/weight ledgers were
found in the scoped package search, including ignored paths. Actual FEWS completed
candidate/positive-weight counts were not established in this investigation.

## 5. Implications for the current draft

1. The user-approved four H-specific maps cannot themselves form a same-H
   candidate ensemble: R24 explicitly keeps H separate.
2. Stage1's candidate child splits build one map; they are not automatically
   independently scored whole-map candidates for Stage2.
3. The new draft has not frozen internal development folds, candidate map
   generation, comparable coverage, or out-of-search weighting support. It has
   no evidence that a useful same-H consensus pool exists.
4. Accepted first version (v0.28, R34): each H directly learns and freezes one map from lawful
   development data; continue using crisis-F1 scanning, global-prefix/local
   increments, combination gates, and identical-key pooled/persistence comparison.
   Internal validation, capacity selection and geography/routing still need design.
5. If consensus is reconsidered later, first establish distinct-map/coverage and
   scoring evidence and compare it with direct maps on held-out development keys.
   Do not extend development into final evaluation or reuse incompatible ablations
   merely to create more maps. No arbitrary candidate-count threshold is proposed.

Conclusion status: investigation complete within the stated scope; the user has
adopted omission of Stage2 (v0.28, R34). The prior weight proposal is not adopted.
Implementation remains unapproved; this acceptance concerns the architecture only.
