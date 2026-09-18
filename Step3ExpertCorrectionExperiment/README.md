# Step 3 partitioned expert selective correction (isolated experiment)

Isolated implementation of the approved design in
`.trellis/tasks/09-18-step3-expert-selective-correction/`.  Nothing in this
directory modifies a production entrypoint: `scripts/`, `app/`, `src/`, the root
batch launchers and every frozen artifact under `paper_reproducibility_package/`,
`archived/`, `other_outputs/` and `result_*/` are read-only inputs.

## What this experiment does

For **fs1 (4-month) and fs2 (8-month)** only:

```
calendar-aligned FEWS NET expert estimate e  +  original main features X
      -> per-partition RF score q = P(expert_wrong = 1)
      -> validation-gated flip rule selected on V = [O-12 months, O)
      -> final prediction e or 1 - e
```

Method identifiers written by the runner:

| Scope | Reported methods | Correction |
|---|---|---|
| fs1, fs2 | `partitioned_selective_correction`, `pooled` | enabled |
| fs3 | `partitioned`, `pooled` | **not applicable, explicitly uncorrected** |

`pooled` (fs1/fs2) and both fs3 methods are **reused frozen archive
predictions**, not newly trained.  Expert-only appears only in selection and
audit artifacts; it is never an additional results series.

The wrong score `q` is not a crisis probability and is never calibrated or named
as one.  It is never written into a `y_prob_partitioned` column.

## The two contract revisions of 2026-09-18

Both were authorized by the user after the first implementation round exposed the
underlying data structure.  Neither relaxes any approved selection gate: the
directional gates remain **20 proposed flips / 2 distinct months / 0.75
correction precision**, and the objective remains a **strictly** greater
validation crisis-class F1 than expert-only with ties keeping expert-only.

### R1. The expert is calendar-aligned at `O = T - H`

`Outcome/FEWSNET_IPC/FEWSNET.csv` is **not** a monthly panel.  All 5,716 admin
units have exactly 53 observations on a shared release grid: quarterly
(Jan/Apr/Jul/Oct) from 2009-07 to 2015-10, then tri-annual
(**February / June / October**) from 2016-02 to 2024-10.

FEWS NET publication semantics: a `fews_proj_near` published in month `D`
targets `D+4`; a `fews_proj_med` published in `D` targets `D+8`.  The expert
estimate for target month `T` is therefore the projection published at the
origin `O = T - H`:

| Scope | Horizon `H` | Source field | Source month |
|---|---|---|---|
| fs1 | 4 months | `fews_proj_near` | `T - 4` |
| fs2 | 8 months | `fews_proj_med` | `T - 8` |

Implemented as a calendar join on `(admin_code, source_month + H == target_month)`.
Unchanged from the historical convention: per-admin ordering, binarisation to
IPC Phase 3+ **before** the join, the raw-missing-phase -> 0 rule,
duplicate-key rejection and per-row provenance.  An origin month that is absent
from the release grid stays **missing** and is never imputed - for example every
pre-2016 quarterly-era fs1 target, because `T-4` never lands on a Jan/Apr/Jul/Oct
grid point.

The previously mandated per-admin **record** shifts (`shift(4)` / `shift(8)`)
were a defect on this source: they resolve to 12-16 (fs1) and 24-32 (fs2)
calendar months, with zero rows at the declared origin.

Independently derived on the 2021-2024 evaluation targets (n = 62,189 rows with
non-missing truth in both scopes):

| Scope | Series | Coverage | Precision | Recall | F1 | tp | fp | fn |
|---|---|---|---|---|---|---|---|---|
| fs1 | calendar `O = T-4` | 1.0000 | 0.8398 | 0.7768 | **0.8070** | 14,120 | 2,694 | 4,058 |
| fs1 | legacy `shift(4)` | 1.0000 | 0.6976 | 0.5642 | 0.6239 | 10,256 | 4,445 | 7,922 |
| fs2 | calendar `O = T-8` | 1.0000 | 0.8077 | 0.7230 | **0.7630** | 13,143 | 3,129 | 5,035 |
| fs2 | legacy `shift(8)` | 1.0000 | 0.6452 | 0.4420 | 0.5246 | 8,035 | 4,419 | 10,143 |

`record shift(1)` and `record shift(2)` reproduce the calendar figures exactly,
confirming that in the tri-annual era one record equals four calendar months.

**The legacy record-shift series is retained, but only as a clearly labelled
pipeline-validation artifact.**  It proves the loader, the binarisation order and
the raw-missing-phase convention are faithful to the frozen historical evaluator
(39/39 archived quarters reproduced per scope at `atol = 1e-12`).  It is
firewalled out of the correction layer structurally:

* every legacy column carries the `legacy_record_shift_` prefix;
* `ExpertTable.for_scope()` - the only path into the correction layer - selects
  an explicit calendar-only column list and re-asserts the absence of the prefix;
* `runner._expert_lookup()` asserts it again before any fitting;
* a test smuggles a legacy column into the lookup and requires an
  `ExpertContractError`.

`require_source_alignment()` now asserts an **exact** `H`-month publication lag
and **has no override parameter**.  A stale lag (`> H`) halts; a
leakage-direction lag (`< H`, i.e. the source post-dates the origin) halts
unconditionally.  The old `--acknowledge-unverified-expert-horizon` CLI flag is
retained only so previously documented commands still parse: it is **inert**,
prints a notice saying so, is wired to nothing, and cannot bypass the gate.
Manifests record `expert_alignment: calendar`, `expert_horizon_verified: true`
and the observed lag distribution (a single bucket at `H`: 154,332 rows at lag 4
for fs1, 148,616 rows at lag 8 for fs2 over the full reconstructed history).

### R2/R8. The validation interval is `V = [O - 12 months, O)`

Label months satisfy `m % 4 == 2` from 2016 on, and because `T` is an observed
month with `H` in `{4, 8}` (both multiples of 4), `O = T - H` also satisfies
`O % 4 == 2`.  The half-open `[O-6, O)` covers `O-6 .. O-1`, of which only `O-4`
is congruent to 2 mod 4 - so it admitted **exactly one** observed label month in
24/24 folds and made the approved `distinct months >= 2` gate structurally
unsatisfiable.

`V = [O - 12 months, O)` admits **exactly three** observed label months -
`O-12`, `O-8`, `O-4` - verified in 24/24 folds.  The outer window is unchanged at
`[O - 35 months, O)` (35 timestamps, historical off-by-one preserved), `V` stays
anchored on `O` rather than on the last observed training month, and it is never
widened further.

Accepted and surfaced consequence: with fit labels isolated strictly before
`V_start - H`, the horizon-isolated first-stage fit retains **4** observed label
months for fs1 and **3** for fs2.  Partitions under the 50-row minimum or with a
single wrong-label class abstain as normal.  `fold_tuning.csv` records the exact
per-fold outer / fit / gap / validation row counts and observed-month counts.

### Deferred: the archived paper FEWS NET baseline

The frozen paper baseline was computed from the legacy record-shift series and
therefore inherits its 12-32 month effective lag.  Per explicit user instruction
this is **out of scope**: nothing under `paper_reproducibility_package/`,
`archived/`, `other_outputs/` or `result_*/` is corrected, regenerated,
relabelled or overwritten.  The discrepancy is recorded here and in every run
manifest (`expert.deferred_known_issue`) as a known deferred issue only.

## Bounded real-data results under the revised contracts

One fold per scope, run 2026-09-18.  Not the full 24-fold experiment, which
remains a separate gate.

| | fs1 2021-02 | fs2 2021-06 |
|---|---|---|
| origin `O` | 2020-10 | 2020-10 |
| outer window | [2017-11, 2020-10) | [2017-11, 2020-10) |
| `V` | [2019-10, 2020-10) | [2019-10, 2020-10) |
| fit cutoff (exclusive) | 2019-06 | 2019-02 |
| outer rows eligible / expert available | 43,200 / 43,200 | 43,200 / 43,200 |
| fit rows / observed months | 21,458 / 4 | 16,093 / 3 |
| gap rows withheld / observed months | 5,365 / 1 | 10,730 / 2 |
| validation rows / observed months | 16,377 / **3** (2019-10, 2020-02, 2020-06) | 16,377 / **3** (same) |
| test rows | 5,425 | 4,385 |
| partitions trained (validation fit / final refit) | 12 / 12 | 9 / 11 |
| threshold candidates | 71 | 86 |
| candidates with any direction enabled | 0 | 6 |
| selection | **no_correction** (no candidate strictly exceeded expert-only) | **corrected**, threshold 0.69, `1->0` only |
| validation F1 expert-only -> selected | 0.656279 -> 0.656279 | 0.560314 -> 0.561772 |
| test rows flipped | 0 | 4 |
| test crisis F1: correction vs pooled | 0.7300 vs 0.5166 | 0.7397 vs 0.4170 |

The `distinct months >= 2` gate is now satisfiable (3 observed months in every
fold) and the fs2 fold selected a real correction: 23 proposed `1->0` flips
across 3 months at 0.913 precision, strictly improving validation F1.  The fs1
fold still selects explicit no-correction, which is the designed behaviour when
no candidate clears the gates and strictly improves F1.

Test-side fix/damage counts in the audit are computed strictly after selection
and are audit-only.  The pooled comparator is unchanged and asymmetric by user
choice; no performance gain is promised or implied by these two folds.

Earlier run directories named `bounded_fs1_2021_02`, `bounded_fs1_2021_02_repeat`
and `bounded_fs2_2021_06` were produced under the **superseded** record-shift and
six-month-`V` contracts.  They are retained as diagnostic evidence (the rollback
policy forbids automatic deletion) and must not be read as current results.

## Environment

`src/preprocess/preprocess.py` requires `polars`, which is absent from the bare
`python3` interpreter.  Use the project-local diagnostic virtual environment for
anything that touches real data:

```bash
.venv-geodt-diagnostic/bin/python   # Python 3.12.3, polars 1.41.2, pandas 2.3.3,
                                    # scikit-learn 1.8.0, numpy 2.2.0
```

This differs from the historical package environment, which is precisely why the
design requires **reusing** archived pooled/fs3 predictions instead of retraining
them.  Every run manifest records the exact interpreter and library versions.

## Commands

All commands are run from the repository root.

### Pre-flight contract evidence (no fitting, no writes outside `logs/`)

```bash
PYTHONPATH="$PWD/Step3ExpertCorrectionExperiment" \
  .venv-geodt-diagnostic/bin/python \
  Step3ExpertCorrectionExperiment/verify_contracts.py \
  --report Step3ExpertCorrectionExperiment/logs/preflight_evidence.json
```

### Bounded real-data check (one scope, one month)

```bash
PYTHONPATH="$PWD/Step3ExpertCorrectionExperiment" \
  .venv-geodt-diagnostic/bin/python \
  Step3ExpertCorrectionExperiment/run_correction_experiment.py \
  --run-id bounded_cal_fs1_2021_02 --scopes 1 --target-months 2021-02
```

### Full authorized experiment (fs1 + fs2, all 12 evaluation months)

This is the exact command for the full 24-fold run.  It is the orchestrator's
gate and was **not** executed by the implementation round.

```bash
PYTHONPATH="$PWD/Step3ExpertCorrectionExperiment" \
  .venv-geodt-diagnostic/bin/python \
  Step3ExpertCorrectionExperiment/run_correction_experiment.py \
  --run-id full_fs1_fs2_<YYYYMMDD> --scopes 1 2 \
  > Step3ExpertCorrectionExperiment/logs/full_fs1_fs2_<YYYYMMDD>.log 2>&1
```

Defaults: `--scopes 1 2` and all twelve Feb/Jun/Oct target months of 2021-2024,
so both flags may be omitted.  No acknowledgement flag is needed or accepted as
meaningful any more.  Run IDs are immutable: an existing run directory raises
`FileExistsError`.

### Tests

```bash
.venv-geodt-diagnostic/bin/python -m pytest Step3ExpertCorrectionExperiment/tests -q
python3 -m pytest Step3ExpertCorrectionExperiment/tests -q   # same 55 tests, no polars needed
```

## Layout

```
Step3ExpertCorrectionExperiment/
  run_correction_experiment.py   # CLI entrypoint
  verify_contracts.py            # pre-flight contract evidence
  step3correction/
    protected.py    # protected-artifact paths and SHA-256 before/after guard
    expert.py       # calendar-aligned expert + firewalled legacy series + alignment gate
    windows.py      # outer / V=[O-12,O) / fit / gap temporal masks
    features.py     # main prepare_features with imputation deferred to the window
    correction.py   # per-partition expert-error RFs and abstention
    selection.py    # threshold candidates, directional gates, F1 selection
    baselines.py    # reused frozen pooled / fs3 predictions and their checks
    runner.py       # fold orchestration, artifacts, manifest
  tests/            # 55 focused tests
  outputs/<run-id>/ # immutable run artifacts (git-ignored)
  logs/             # run logs and pre-flight evidence (git-ignored)
```

## Run artifacts

| File | Contents |
|---|---|
| `predictions_monthly_correction.csv` | per-row test audit: keys, scope, partition, truth, pooled prediction, expert + source provenance, wrong score, eligibility/reason, selected threshold, direction flags, applied flip, final prediction, audit-only fix/damage |
| `metrics_monthly_correction.csv` | monthly crisis-class metrics for `partitioned_selective_correction` and `pooled` on identical support |
| `fold_tuning.csv` | one row per fold: window endpoints, fit/gap/validation/test row counts **and observed-month counts**, the validation observed-month list, partitions trained per stage, expert-only reference F1, selection status and reason |
| `fold_threshold_candidates.csv` | every candidate threshold with both directional gate counts and its validation F1 |
| `validation_rows.csv` | validation-role rows (keys, truth, expert, wrong score, eligibility) so the gates and selection can be independently recomputed; excluded from test metrics |
| `partition_trainability.csv` | per-partition rows/wrong-label counts and abstention reasons for both fitting stages |
| `feature_order_fs{1,2}.csv` | the exact main feature order used |
| `predictions_monthly_fs3_uncorrected.csv`, `metrics_monthly_fs3_uncorrected.csv` | reused fs3, flagged `correction_applicable = False` |
| `run_manifest.json` | sources + hashes, environment, RF parameters, `expert_alignment: calendar`, `expert_horizon_verified`, observed lag distribution, coverage, the firewalled legacy-series declaration and its archived reproduction, the deferred paper-baseline issue, window endpoints and rationale, partition provenance, reuse declaration |
| `protected_hashes.json` | before/after SHA-256 of all 31 protected inputs with an `unchanged` flag |

## Contract gates that halt a run

| Condition | Result |
|---|---|
| Duplicate valid admin-month key in the expert source | `ExpertContractError` |
| Expert source row dated after the declared origin (leakage) | `ExpertContractError`, no override |
| Expert source row older than the declared origin (stale) | `ExpertContractError`, no override |
| A `legacy_record_shift_*` column reaching the correction layer | `ExpertContractError` |
| Missing expert source match on required support | `ExpertContractError` |
| Missing expert estimate on test support | `ExpertContractError` |
| Panel truth disagreeing with source `fews_ipc >= 3` | `ExpertContractError` |
| Frozen month map not reproducing archived partition ids | `RunContractError` |
| Panel target-month support differing from the frozen support | `RunContractError` |
| Archive and package copies of a frozen artifact differing | `BaselineContractError` |
| Recomputed pooled metrics differing from the archive | `RunContractError` |
| Non-finite correction model input | `CorrectionInputError` |
| Output path outside `Step3ExpertCorrectionExperiment/outputs/` | `ValueError` before any directory is created |
| Existing run directory | `FileExistsError` |
| Any protected artifact hash drift | `RuntimeError` |

Partition abstention (unmapped, absent model, fewer than 50 usable rows, single
wrong-label class) is ordinary behaviour, not an error: those rows retain the
expert prediction, stay in the validation F1 support, and can never be proposed
flips.  The crisis-predicting pooled model is never used as a wrong-score source.

## Not authorized by this task

No commits, no paper-artifact promotion, no change to Stage 1/2, the historical
window, pooled thresholds or fs3.  fs0, GeoDT, XGBoost, residual stacking and
additional baselines are out of scope.
