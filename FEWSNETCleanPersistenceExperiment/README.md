# FEWS NET clean persistence baseline

A pre-registered, bounded experiment: does a two-layer system — a spatially partitioned
GeoRF crisis-probability model correcting a persistence base layer — beat persistence
alone on FEWS NET data, and does updated feature engineering add anything over a
corrected original-feature reference?

The scientific design is frozen in `.trellis/tasks/09-20-fewsnet-clean-persistence-baseline/`
(`prd.md` R1–R68 / A1–A64, `design.md`, `implement.md`, `research/` D1–D64). **This
package implements that design; it does not decide it.** Where running the code exposed
a conflict with a contract, the contract was followed and the finding was recorded in
that task's `IMPLEMENTATION_LOG.md` rather than fixed in passing.

## What is being compared

The base layer is **persistence**: the area's last observed IPC crisis label at exactly
the origin month `O = T - H`. It is a strong baseline — in development folds it beats
the RF's own hard predictions in five of six (role, horizon) cells. The experiment does
not try to replace it. It uses the RF's *probability* to selectively flip persistence
`0 -> 1` where the model is confident, and never `1 -> 0`.

## Pipeline

```
prepare_data.py     pinned sources -> validated monthly grid + observed IPC ledger
                    -> frozen schemas, A-E feature blocks, corrected reference
run_pipeline.py     Stage 1  per-candidate GeoRF partition learning (isolated process)
                    Stage 2  per (arm, role) general consensus map + bounded completion
                    Stage 3  rolling pooled/partitioned RF predictions
                    calibrate / thresholds  month-pooled calibrators, frozen taus
                    final    final maps + D40 evaluation windows
report_results.py   D22 recipe selection, D41-D45 adjudication and robustness
```

Stages, in order:

```bash
PY=/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe
RUN=FEWSNETCleanPersistenceExperiment/runs/<fresh-name>

$PY FEWSNETCleanPersistenceExperiment/prepare_data.py --run-dir $PREP --preflight-only
$PY FEWSNETCleanPersistenceExperiment/prepare_data.py --run-dir $PREP
$PY FEWSNETCleanPersistenceExperiment/run_pipeline.py --stage setup        --run-dir $RUN --prepared-dir $PREP
$PY FEWSNETCleanPersistenceExperiment/run_pipeline.py --stage development  --run-dir $RUN --arms BASE,A,B,C,D,E,ABCDE,BCDE,ACDE,ABDE,ABCE,ABCD,reference
$PY FEWSNETCleanPersistenceExperiment/run_pipeline.py --stage consensus    --run-dir $RUN --arms <same>
$PY FEWSNETCleanPersistenceExperiment/run_pipeline.py --stage predictions  --run-dir $RUN --arms <same>
$PY FEWSNETCleanPersistenceExperiment/run_pipeline.py --stage calibrate    --run-dir $RUN --arm <arm>   # per arm
$PY FEWSNETCleanPersistenceExperiment/run_pipeline.py --stage thresholds   --run-dir $RUN --arm <arm>   # per arm
$PY FEWSNETCleanPersistenceExperiment/report_results.py --stage select-recipe --run-dir $RUN
$PY FEWSNETCleanPersistenceExperiment/run_pipeline.py --stage final        --run-dir $RUN --arms <winner>,reference
$PY FEWSNETCleanPersistenceExperiment/report_results.py --stage final --run-dir $RUN --verify
```

`--stage schedule` prints the job inventory without running anything. Every stage is
idempotent at candidate granularity: a completed candidate is skipped, not refitted.

## Ordering that is enforced, not merely documented

* `--stage final` refuses to run without a frozen `report/recipe_selection.json`, so no
  final-window row can be scored before D22's winner is chosen from 2020 evidence.
* Calibrators refuse to overwrite an existing frozen set, because the frozen thresholds
  were selected against a specific probability transformation.
* The threshold freeze guard runs *before* any threshold output is written, so a
  rejected rerun cannot leave fresh traces beside old frozen thresholds.
* Consensus maps refuse to build into a non-empty output directory, and Stage 3 verifies
  each map against its recorded digest before predicting with it.
* Thresholds and calibrators record the SHA-256 of every prediction file and calibrator
  that produced them (R42), not merely a check that today's files still match.

## Reused released code

The partition learner is the pinned `GeoRFBaseline` release (`releases/georf-baseline-v0.1.0.zip`),
extracted per run and payload-hash verified. Every Stage 1 candidate runs in an isolated
process with its own `sys.path`, so the repository root cannot shadow the release.

Two deliberate departures from released behaviour, both recorded:

* The discarded pre-partition CV diagnostic is bypassed through the release's own
  `except ImportError` branch. This is a runtime optimisation only — verified, not
  assumed, by `verify_diagnostic_bypass.py`, whose retained evidence shows every
  scientific artifact byte-identical with the diagnostic on and off.
* `run_stage3_fold` routes a target partition absent from the training pool to the
  pooled model. The released comparator leaves such rows at their zero-initialised
  value, silently predicting "no crisis" (D61.4 directs correcting this in experiment
  code, not in the protected baseline).

`PersistenceCorrectionExperiment` supplies the calibrator, threshold selector and
override under D37/D38. Its partition-specific calibrators are dropped before freezing,
because `CalibratorSet.transform` prefers them over the month pool and R20 forbids
local partition-specific overrides. Its default calibration window `(2018, 2019)` is
always overridden to `(2018,)` per R21.

## Evidence retained per run

```
runs/<name>/
  baseline/            extracted, hash-verified release
  geometry/            adjacency cache, contiguity info, area coordinates
  stage1/<arm>/<cand>/ candidate.json, correspondence_table.csv, s_branch.pkl,
                       branch_table.npy, target_predictions.csv, val_coverage_by_group.csv
  stage2_inputs/       the exact artifacts the released Stage 2 helpers read
  stage2/<arm>/<role>/ consensus_map.csv, consensus_evidence.json, plan_weights.csv
  stage3/<arm>/<fold>/ predictions.csv (partitioned AND pooled streams),
                       training_keys.csv, imputer_statistics.csv, local_support.csv
  calibration/<arm>/   frozen calibrators + their input identities
  thresholds/<arm>/    frozen_thresholds.json + full candidate traces
  verification/        retained control experiments
  report/              recipe selection and adjudication
```

Every reported number is recomputed from these files; console logs are not evidence.

## Tests

```bash
$PY -m unittest discover -s FEWSNETCleanPersistenceExperiment/tests -v
```

Three contract modules covering preparation, Stage 1/2 and Stage 3/reporting. They
exercise scientific boundaries — calendar and origin masks, phase/missing persistence,
source joins and ties, training-only imputation, the F1 gate and SMOTE invariants,
graph selection/completion/no-split states, single-class probability extraction,
correction equality and null thresholds, pooled-versus-averaged F1, and the D43/D45
adjudication table — rather than restating constants.

## Known limitations

These are properties of the frozen design, not defects, and belong in any write-up:

* **Coverage.** The consensus map supplies a learned partition to 63.6% (calibration)
  and 83.6% (selection) of the 5,718-area master cohort; on matched target-row
  denominators, local models actually score 64.5% and 84.1% of rows. The rest route to
  the pooled model. Whole countries can fall outside a window's core.
* **Consensus weighting.** D56's `1e-6` clip gives a plan whose pooled baseline F1 is
  exactly 0 a weight an order of magnitude above well-behaved plans. In the calibration
  window two such plans carry 93.7% of the weight. R60 mandates the released formula, so
  this is reported, not repaired.
* **Gate optimism.** The Stage 1 split gate uses a random within-area validation split,
  not a temporal holdout, so its F1 runs far above the same candidate's target-month F1.
* **Stage 1 group filter.** Training rows are restricted to groups present in the target
  month (inherited release behaviour, retained under R24). It does not expose future
  label values but can change training support and therefore the learned partitions.
* **Untested branches.** Both real maps are multi-cluster, so the `valid_unsplit` and
  unseen-partition routes have unit-test and inspection coverage but no real-run
  coverage.
