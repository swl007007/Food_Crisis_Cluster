# Results and acceptance index — FEWS NET four-class baseline

Authoritative run: `FEWSNETFourClassBaseline/runs/fourclass-v6-20260928/`
(package `FEWSNETFourClassBaseline/`, base SHA 8a56272, audit run
321911af42cc46809237d7a003700ed9). Superseded: `fourclass-v1-20260928` (aborted at
Stage 3 h8 by a coverage gate that departed from the release definition; see
IMPLEMENTATION_LOG.md). Development runs `dev-prep`, `dev-pilot` are wiring pilots
only and are not committed.

## Verdict

A complete negative result. The learned spatial partition does not help: the
partitioned RF is below the pooled RF at all three horizons, and both RF arms are
below exact-origin persistence and the FEWS NET expert. Every one of the eight
specified contrasts is negative with a 95% interval excluding zero.

| horizon | n | partitioned | pooled | expert | persistence |
|---|---|---|---|---|---|
| 4 | 54,330 | 0.6532 | 0.6834 | 0.7651 | 0.7308 |
| 8 | 49,340 | 0.5894 | 0.6102 | 0.6980 | 0.6662 |
| 12 | 43,441 | 0.5337 | 0.5522 | — | 0.6406 |

| contrast (partitioned −) | 4 months | 8 months | 12 months |
|---|---|---|---|
| pooled | −0.0302 [−0.0397, −0.0212] | −0.0208 [−0.0303, −0.0024] | −0.0185 [−0.0322, −0.0102] |
| expert | −0.1119 [−0.1881, −0.0680] | −0.1086 [−0.1472, −0.0413] | not defined (D4) |
| persistence | −0.0776 [−0.1398, −0.0394] | −0.0769 [−0.0975, −0.0179] | −0.1070 [−0.1318, −0.0520] |

Intervals: 2,000 of 2,000 accepted country-cluster draws (22 countries, seed 42),
shared across cohorts and horizons, conditioned on fitted predictions, per-horizon and
marginal (not simultaneous). Source: `report/contrasts.csv`, `report/report.json`.

Per-class F1 (4 months, pooled / persistence / expert): class 1 0.886/0.876/0.894,
class 2 0.740/0.735/0.777, class 3 0.723/0.732/0.766, class 4或5 0.384/0.580/0.624.
The largest RF shortfall is in the rare merged 4或5 class (1,773 of 54,330 keys), which
fixed-four macro F1 weights equally. Full tables: `report/arm_metrics.csv`.

Descriptive only (`report/descriptive_by_country_year.csv`): the partition beats
pooled in 2023 at 4 months (0.716 vs 0.713) and in 2024 at 12 months (0.546 vs
0.537); these are single cells and are not evidence of a partition benefit.

### Reading notes

- Supplementary fs1/fs2 cohorts equal the main cohorts: on the pinned panel every
  key with an exact-origin observation also carries the expert projection published at
  that origin, so dropping the expert requirement adds no key (`coverage` in
  `report/report.json`).
- Excluded keys (no exact-origin observation): 2,434 / 3,039 / 3,964 at 4/8/12
  months, of 56,764 / 52,379 / 47,405 truth keys. Model failures removed none.
- Stage 1 learned splits in 8 of 27 candidates (validation-selected, D14). Only 3 had
  a positive held-out macro-F1 gain over the pooled comparator (weights 0.023, 0.013,
  0.038), so the consensus is driven by three plans. Stage 2 recommended and produced
  13 clusters over 5,506 in-scope areas; 1,177 areas outside the main kNN component
  were assigned by 1-NN on coordinates (inherited).
- Partitioned routing at Stage 3: every mapped cluster fitted a local RF (13 per fold,
  none fell back); 208 evaluated areas were unmapped and used the pooled RF
  (unmapped share 0.46% of labelled panel rows, 1.9-2.1% of evaluated targets).
- The absolute FEWS NET comparison is retrospective: 2021-2024 were inspected in
  earlier work, and source-month alignment is not verified real-time availability.
- Not attributable to the four-class target alone (D21): features, imputation and
  alignment all changed relative to the binary baseline.

## Acceptance index

| criterion | evidence (paths relative to the run) | status |
|---|---|---|
| A1 identities, keys, mapping, calendar joins, source agreement | `prepared/manifests/{sources,runtime,preflight}.json`; `prepared/ledgers/baselines.csv`; tests `test_phase_merge_and_missing`, `test_baseline_calendar_join_and_no_backfill` | pass: 4 pinned source hashes and shapefile sidecars; release ZIP hash plus per-file diff; complete 5,718 x 180 scaffold, unique keys; truth/near/medium agree with FEWSNET.csv on every shared key; binary = phase≥3 on all 259,440 labels; no fs3 expert; missing baselines stay missing |
| A2 schema, hand-computable histories, exclusions, provenance | `prepared/manifests/features.json` (order, provenance, raw NaN/inf per column); `verification.json` (1,200 keys x 17 columns re-derived from raw panel); `HistoryFeatures` tests | pass |
| A3 folds, empty months, windows, cutoff, imputer rows, inheritance, fallback | `prepared/manifests/schedule.json`; `stage1/folds/*/candidate.json` (windows, rows, fit logs, imputer hashes); `stage1/folds/*/fold_membership.csv.gz`; `stage3/h*/folds/*/{fold.json,training_keys.csv.gz,imputer_statistics.csv.gz,local_support.csv}`; verification stage1/stage3 checks; `Imputation`, `Stage3Routing` tests | pass |
| A4 FP+FN statistics, scan masses, strict .01, ties, root rejection, routing agreement | `candidate.json` → `partition.decisions` (exact rational parent/child scores per candidate); `ScanStatistics`, `SplitGate` tests; runtime checks of saved `X_branch_id` vs `s_branch` and correspondence | pass |
| A5 candidate inventory, score provenance, weights, null vs missing, map routes | `stage2/{candidate_ledger.csv,plan_weights.csv,consensus.json}`; `stage2/experiment/knn_sparsification_results/*.{csv,json}`; `similarity_matrices/summary_statistics.json`; verification stage2 checks; `test_stage2_*` | pass (learned-map route) |
| A6 four Stage 1 pseudo rows, none in Stage 3, class axes, fitted params, real support | `candidate.json` → `fits` (pseudo_rows=4 per fit, real class counts); `stage3/*/folds/*/fold.json` (pseudo_rows 0, per-estimator classes_, params, imputer hash) | pass |
| A7 keys, confusion matrices, metrics, intervals recompute | `report/{keyed_evaluation.csv.gz,bootstrap_draws.csv.gz,report.json}`; verification report checks (sklearn recompute; 25 draws recomputed by row replication) | pass |
| A8 independent reconstruction, replay, manifests, originals unchanged | `verification/verification.json` (34/34), `verification/replay/`; first/last fold replay for Stage 3 per horizon (identical incl. probabilities) and Stage 1 per scope (retained checkpoints, local only); `GeoRFBaseline/` untouched (git) | pass |
| A9 lifecycle | spec commit 8a56272; audit start by bound executor; implementation and evidence committed; wrapper close | close pending at time of writing |

## Reproduce

```bash
cd FEWSNETFourClassBaseline
python -B tests/test_baseline.py
./run_all.sh runs/<fresh-id>
python -B scripts/verify_fourclass.py --run-dir runs/<fresh-id>
```

## Audit repair (2026-09-28)

Close audit fa38f19ad8fafc36567f90ac found A01 (Stage 3 discarded fitted forests) and A02
(Stage 1 continuation trusted a completion filename). Repaired in task
`09-28-fourclass-audit-repair`. The authoritative run is now `fourclass-v6-20260928`,
produced by the repaired code; its `report/contrasts.csv` and `report/arm_metrics.csv`
are byte-for-byte equal to v2's, so every number above is unchanged. v3 persists every
Stage 3 estimator bundle (committed, hash-bound in fold.json) and replay now loads them.

## Audit repair round 2 (2026-09-28)

Repair close-audit 026f8908 and spot re-audit 175302c2 were remediated in task
`09-28-fourclass-audit-repair-2`. Authoritative run is now `fourclass-v6-20260928`,
produced from committed code cfbc710 (`code_equals_git_head: true`); report tables are
byte-identical to v2/v3/v5, so every number above is unchanged. Verification 36/36:
run code identity equals committed HEAD, per-estimator training-key digests, and
bit-identical saved-model replay (labels and all four probabilities) for first/last
fitted fold per horizon. Runs v3-v5 are superseded and not committed.
