# Design — Persistence-correction 2-layer architecture

Companion to `prd.md`. Requirement IDs (`R*`) refer to that file.

## Shape of the thing

```
layer 1  persistence:  y_base(T) = 1[ fews_ipc(T-H) >= 3 ]          (no parameters)
layer 2  override:     y(T) = 1  if y_base(T)==0 and p_cal(T) > tau  (one parameter, tau)
                       y(T) = y_base(T)  otherwise
```

`p_cal` is the partitioned GeoRF crisis probability after per-`(month, partition)` calibration.
`tau` is a single scalar per scope, selected once on 2018-2020 and frozen (R13). `1 -> 0` is not
expressible in this form; that is the enforcement mechanism for R15, not a runtime check.

## Boundaries

New package `PersistenceCorrectionExperiment/` (R25). It **reads** from Step 3 and the frozen
package; it writes only under its own output tree (R26).

| Module | Responsibility |
|---|---|
| `persistence.py` | Build the persistence series by calendar join at `T-H`; assert coverage 1.0; carry source-month provenance. |
| `calibration.py` | Cross-fit inside the training window; fit per-`(month, partition_id)` calibrators with pooled fallback; emit reliability tables. |
| `selection.py` | Sweep `tau` on 2018-2020, pick argmax crisis-class F1, freeze. Also compute the R14 consistency threshold. |
| `override.py` | Apply frozen `tau`. Structurally up-only. |
| `adjudicate.py` | Per-scope F1, delta vs persistence, fold bootstrap, leave-one-fold-out, verdict table. |
| `protected.py` | Thin re-export of the Step 3 hash gate over an extended path list. |

Reused by import, never edited: `Step3ExpertCorrectionExperiment/step3correction/expert.py`
(`_calendar_align` at `:142`, its duplicate-key rejection and coverage reporting) and its metric
helper `counts_and_scores` (`correction.py:261`). Note `ExpertTable.for_scope` (`expert.py:93`) has
a hard firewall that structurally rejects non-approved series — persistence must get its own path,
not a reuse of `for_scope`.

Reused for diagnostics: the reliability binning in
`scripts/paper_artifacts/analyze_georf_probability_uncertainty.py:115-148`. No calibration code
exists anywhere in the repo (`sklearn.calibration` is never imported), so the fitting side is new.

## Data flow

```
FEWSNET.csv ──calendar join T-H──> persistence series ─────────────┐
                                                                    ├─> override ─> adjudicate
Stage3 predictions 2021-2024 (y_prob_partitioned) ──> calibrate ───┤
2018-2020 regenerated probabilities ──> calibrate ──> select tau ──┘
```

Two probability sources, both from the same script and the same three month-maps:

- **2021-2024** already exists, frozen, at
  `paper_reproducibility_package/stage3_results/georf_fs{1,2}/predictions_monthly.csv`
  (`y_prob_partitioned`, `partition_id`, 62,189 rows per scope).
- **2018-2020 must be regenerated** (R27). It was never persisted: Stage 1 hard-codes a hard-label
  schema at `app/main_model_GF.py:233` and never calls `predict_proba`.

## Why one parameter

The statistical bar in R19-R20 is the binding constraint, not model capacity. Per fold there are
three observed label months (label months satisfy `m % 4 == 2`), so any design that spends degrees
of freedom per partition or per direction is unidentifiable at `+0.02`. The prior Step 3 experiment
spent them on per-partition thresholds plus two directions and produced two nulls, with validation
gain correlating *negatively* with test gain (Pearson -0.133). One global scalar, frozen out of
sample, is the most-powered version of this hypothesis that exists.

## Calibration design

Stage 3 ran with `--month-ind`, so `partition_id` is not 17 stable units: it is 13 (Feb) / 11 (Jun)
/ 16 (Oct) distinct partitionings selected by calendar month, and 5,350 of 5,713 admins change
partition across months. Calibrating per `partition_id` alone would silently mix three different
partition definitions under one integer.

Therefore: fit per `(calendar month, partition_id)`, 40 groups. Each group draws only from that
month's rows in the 36-month training window — three observed months — so groups will be thin. Any
group with fewer than 50 cross-fit rows falls back to that month's global calibrator (R11), mirroring
the Step 3 abstention threshold rather than inventing a new rule. Count and report fallbacks; a high
fallback rate is itself a finding about whether per-partition calibration was warranted.

Cross-fitting happens strictly inside the training window (R12). The validation label months are
never touched by calibration, because they are the only thing standing between us and the
double-dipping that would void the R19 confidence interval.

## Threshold selection and its known bias

`tau` is chosen on 2018-2020 and frozen (R13), mirroring the project's existing Stage 1 (learn) ->
Stage 3 (evaluate) temporal separation. This removes per-fold selector noise entirely and is what
makes leave-one-fold-out meaningful.

The bias is stated, not hidden: Stage 1 learned the partition maps **on** 2018-2020, so partition
structure is in-sample for the selection window. Two mitigations, both pre-registered (R14):
calibration absorbs most of the probability-scale shift, and we additionally report the threshold
the first six 2021-2024 folds would have chosen. If the two diverge materially, that is a declared
red flag reported in `RESULTS.md` — it does not authorise switching selection windows after the
fact.

## Compatibility and rollback

Read-only with respect to everything that already exists. The Step 3 protected-hash gate (31 hashes,
already proven unchanged across two variants) runs before and after each run, extended to cover the
frozen `refined/` maps this task reads. Rollback is deleting the new package and its output tree;
no frozen artifact, no `src/` module, and no existing test is touched.

The one external side effect is the 2018-2020 regeneration run, which writes to a **new** output
directory and must set `NO_LEAK_PARTITION_LEARNING_YEARS` / `NO_LEAK_EVALUATION_YEARS`, because
`scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py:857-858` otherwise emits a manifest falsely
claiming `evaluation_years: 2021-2024`. That script is not modified (R27).

## What would falsify the whole design

If post-calibration reliability cannot reach R5 in the `persist=1` group, the override's premise —
that `p_cal` is trustworthy where persistence is positive — fails, and layer 2 is wrapping a
miscalibrated model. That is a stop condition on the R4 conditional branch, reported as such, before
any threshold is selected.
