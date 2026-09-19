# Persistence-correction 2-layer feasibility gate (isolated experiment)

Implementation of `.trellis/tasks/09-18-persistence-correction-2layer/`, whose
`DECISIONS_LOG.md` supersedes parts of its `prd.md` / `design.md` / `implement.md`.
**Phases 0-3 are implemented**; Phase 3 ends at a reported stop/go gate and
**Phase 4 is deliberately not started**.

Nothing in this directory modifies a production entrypoint or an existing
experiment. `Step3ExpertCorrectionExperiment/`, `paper_reproducibility_package/`,
`archived/`, `other_outputs/`, `result_*/`, `src/`, `app/` and
`EthiopiaForecastingExperiment/` are read-only inputs; Step 3 utilities are reused
**by import only**.

## Shape of the thing

```
layer 1  persistence:  y_base(T) = 1[ fews_ipc(T-H) >= 3 ]          (no parameters)
layer 2  override:     y(T) = 1  if y_base(T)==0 and p_cal(T) > tau  (one parameter)   [NOT YET BUILT]
```

`H` is 4 months (fs1) and 8 months (fs2). fs0 and fs3 are out of scope (PRD R7).

## Status

| Phase | State |
|---|---|
| 0 - scaffold and protected-hash gate | done |
| 1 - persistence series | done, gate passed |
| 2 - regenerate 2018-2020 probabilities | done (see `DECISIONS_LOG.md` C1/C2; 2021-2024 was regenerated in the same environment) |
| 3 - calibration | done; **gate figures reported, adjudication deferred to the task author** |
| 4 - threshold selection and freeze | **not started, deliberately** |
| 5 - adjudication | not started |
| 6 - close out | not started |

## Phase 1 result (the stop/go gate)

Evaluation support: the frozen Stage 3 2021-2024 predictions,
`paper_reproducibility_package/stage3_results/georf_fs{1,2}/predictions_monthly.csv`,
62,189 rows per scope over the 12 tri-annual target months 2021-02..2024-10.

| scope | H | rows | coverage | precision(1) | recall(1) | F1(1) | reference F1 | abs delta |
|---|---|---|---|---|---|---|---|---|
| fs1 | 4 | 62,189 | **1.0000** | 0.8160 | 0.7399 | **0.776088** | 0.7761 | 1.16e-05 |
| fs2 | 8 | 62,189 | **1.0000** | 0.7662 | 0.6589 | **0.708510** | 0.7085 | 9.57e-06 |

Reference values and their derivation:
`docs/notes/2026-09-18_benchmark_and_direction_review.md` section 3. Tolerance 1e-4.
The two run directories produced under pandas 3.0.0 and pandas 2.3.3 are
byte-identical for both series CSVs.

### Caveat surfaced by the run, not by the reference note

`persistence_phase_missing = 1` on 2,434 fs1 rows (3.91%) and 3,039 fs2 rows
(4.89%): the origin row at `T-H` **exists** - so join coverage is genuinely 1.0 -
but its `fews_ipc` is null, and the historical raw-missing-phase convention maps
that to persistence `0`. Those rows have an actual crisis rate of 0.337 (fs1), so a
non-trivial slice of the `persist = 0` group is a convention, not an observation.
This is inherited from the frozen historical evaluator and is *not* changed here;
it is recorded per row and counted in the manifest so layer 2 can account for it.

## Phase 3 result (calibration, run `phase3_20260918`)

### What was fitted

Per `DECISIONS_LOG.md` C5, which supersedes PRD R12 (the cross-fitted
training-window predictions R12 asked for do not exist and the frozen Stage 3
script cannot emit them), the nine labeled pre-test months split into three
disjoint windows:

| window | labeled months | rows/scope | role |
|---|---|---|---|
| 2018-2019 | 6 | 32,188 | fit the calibrators (in-sample for them) |
| 2020 | 3 | 16,439 | Phase 4 threshold selection; out-of-sample here |
| 2021-2024 | 12 | 62,189 | Phase 5; calibrated and emitted, **never scored in Phase 3** |

Calibrators are isotonic regressions keyed by `(calendar month, partition_id)`,
fitted once and frozen with a SHA-256 digest (C6). `fit_calibrators()` refuses to
start if a single row outside 2018-2019 is present, and the runner additionally
checks that no admin-month key used for fitting reappears in a later window
(0 overlaps, both scopes).

### The group count: 43 observed ids reconcile to PRD R11's 40 groups

`partition_id` has 14 / 12 / 17 distinct values in Feb / Jun / Oct, which is
`nc + 1` for the `nc13_m2` / `nc11_m6` / `nc16_m10` maps. The extra value is the
`-1` unmapped-admin sentinel
(`scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py:270-271, 391-392`), whose
rows are predicted by the pooled model. Excluding it gives exactly the
13 + 11 + 16 = **40** groups R11 specifies, and all 40 are present in the
2018-2019 fit window.

### Fallbacks, counted (identical in both scopes)

| stage | reason | count | note |
|---|---|---|---|
| fit | `min_rows` (< 50) | **0** | the smallest real group has 128 rows; median 719 |
| fit | `single_class` | **3 groups** | `(Feb, 7)` 294 rows, `(Jun, 4)` 226, `(Oct, 14)` 368, all zero crises in 2018-2019 |
| apply | `group_absent_from_fit_window` | 0 / 423 / 2,733 rows | the `-1` bucket; absent from 2018-2019 entirely |
| apply | `single_class` | 888 / 444 / 1,776 rows | the three groups above |

(apply counts are fit / selection / test windows.) The three single-class groups
route to their calendar month's pooled calibrator: isotonic would emit a constant
0 and `LogisticRegression` raises on one class. The Platt attempt is actually made
and its `ValueError` is recorded in `calibration_groups.csv`, so the fallback is an
observed consequence rather than an assumption. 37 of 40 groups therefore carry
their own isotonic calibrator; the three month pools are isotonic on ~10,730 rows
each.

Because the `min_rows` fallback does not fire on real data, a contract test
induces it synthetically and asserts that it is counted and that the rows take the
month pool's own output.

### Reliability by persistence group, pre and post

`mean predicted probability` vs `observed crisis rate`; `|gap|` is the PRD R5
quantity, tolerance 0.05.

| scope | window | sample | group | n | mean pre | mean post | crisis rate | \|gap\| pre | \|gap\| post |
|---|---|---|---|---|---|---|---|---|---|
| fs1 | 2018-2019 | in-sample | persist=0 | 27,039 | 0.2047 | 0.1152 | 0.0763 | 0.1284 | **0.0389** |
| fs1 | 2018-2019 | in-sample | persist=1 | 5,149 | 0.5574 | 0.5199 | 0.7231 | 0.1656 | **0.2032** |
| fs1 | 2020 | out-of-sample | persist=0 | 13,328 | 0.2215 | 0.1609 | 0.0735 | 0.1481 | **0.0874** |
| fs1 | 2020 | out-of-sample | persist=1 | 3,111 | 0.5243 | 0.4489 | 0.5574 | 0.0331 | **0.1085** |
| fs2 | 2018-2019 | in-sample | persist=0 | 27,156 | 0.1914 | 0.1211 | 0.0992 | 0.0922 | **0.0218** |
| fs2 | 2018-2019 | in-sample | persist=1 | 5,032 | 0.5453 | 0.5002 | 0.6145 | 0.0691 | **0.1142** |
| fs2 | 2020 | out-of-sample | persist=0 | 13,159 | 0.2099 | 0.1637 | 0.0899 | 0.1200 | **0.0738** |
| fs2 | 2020 | out-of-sample | persist=1 | 3,280 | 0.5225 | 0.4458 | 0.4665 | 0.0561 | **0.0207** |

Overall (not per group) calibration behaves exactly as isotonic regression should:
in-sample mean predicted probability lands on the base rate (fs1 0.1800 vs 0.1798;
fs2 0.1804 vs 0.1798) and the in-sample Brier score improves (fs1 0.1152 -> 0.0857,
fs2 0.1168 -> 0.0907). Out-of-sample on 2020 the mean stays high (0.2154 / 0.2200
against a 0.1650 rate) and the Brier score is flat for fs1 (0.10865 -> 0.10848) and
**worse** for fs2 (0.10535 -> 0.12005).

### The gate (PRD R5), on 2020 out-of-sample rows

| scope | group | n | mean calibrated p | crisis rate | \|gap\| | vs 0.05 |
|---|---|---|---|---|---|---|
| fs1 | persist=0 | 13,328 | 0.1609 | 0.0735 | 0.0874 | **fail** |
| fs1 | persist=1 | 3,111 | 0.4489 | 0.5574 | 0.1085 | **fail** |
| fs2 | persist=0 | 13,159 | 0.1637 | 0.0899 | 0.0738 | **fail** |
| fs2 | persist=1 | 3,280 | 0.4458 | 0.4665 | 0.0207 | pass |

Three of four groups miss the criterion. **Phase 3 reports this and stops**; the
stop/go decision and the PRD R4 conditional retraining branch are the task
author's, not this package's, and `run_calibration.py` exits 0 either way.

### Why a persistence-agnostic calibrator cannot be expected to pass R5

Isotonic regression preserves the mean of the fit sample, so it drives the
*overall* predicted mean onto the overall base rate. It carries no guarantee for
the mean *within a subgroup it never sees*, and `persistence` is not in its input.
The GeoRF score over-predicts overall (fs1 2018-2019 mean 0.2612 against a 0.1798
rate), so calibration shrinks every score - which helps `persist=0`, where the
model over-predicts, and **hurts `persist=1`, where it already under-predicts**.
In-sample fs1 `persist=1` moves from 0.1656 to 0.2032 and fs2 from 0.0691 to
0.1142: calibration made the group that layer 2's premise depends on measurably
worse, in-sample, in both scopes. This is a property of the design, not a defect
in the fit, and it is the finding the R5 gate is there to surface.

The 2020 `persist=1` rows also show the cleanest pre-calibration figure in the
whole table (fs1 |gap| 0.0331), so a comparison against the 2021-2024 reference
gap of 0.193 quoted in the task brief mixes two different windows and should not
be read as an improvement or a deterioration on its own.

## Why a calendar join and never a record shift

`1.Source Data/Outcome/FEWSNET_IPC/FEWSNET.csv` is not a monthly panel: 302,948
usable rows, 5,716 admin units, exactly 53 records each on a shared release grid -
quarterly (Jan/Apr/Jul/Oct) 2009-07..2015-10, then tri-annual (Feb/Jun/Oct)
2016-02..2024-10. A per-admin `shift(4)`/`shift(8)` resolves to 12-16 / 24-32
calendar months there. `persistence.py` contains no shift, fill, reindex or
resample call at all, and a static AST test enforces that.

## The R6 availability assumption and its recorded contradiction

`fews_ipc(D)` is treated as available to a forecaster at month `D` (PRD R6). The
requirement text is embedded verbatim in `persistence.py` and copied into every
run manifest, together with the repository's own contradiction:
`src/preprocess/preprocess.py:267-273` assumes available, while
`EthiopiaForecastingExperiment/aligned_refit.py:122-127` and
`.trellis/spec/backend/local-forecasting-experiments.md:73` assume not. The
departure from the spec's strictly-before-origin rule is deliberate and scoped to
this package. **Neither the spec nor the Ethiopia experiment is modified.**

## Contract gates that halt a run

| Condition | Result |
|---|---|
| Any target row without an observation at `T-H` (coverage < 1.0) | `PersistenceContractError` |
| A matched source row not dated exactly `T-H` | `PersistenceContractError` |
| Duplicate `(admin_code, target month)` key in the source or the support | `PersistenceContractError` |
| Null or non-integral admin code, or a non-0/1 binarised phase | `PersistenceContractError` |
| Scope other than fs1/fs2 | `PersistenceContractError` |
| A `legacy_record_shift_*` column reaching the persistence series | `ExpertContractError` (Step 3 firewall, reused) |
| Output path outside `PersistenceCorrectionExperiment/outputs/` | `ProtectedPathError` before any directory is created |
| Existing run directory | `FileExistsError` |
| Protected artifact hash drift | `RuntimeError` |

## Protected inputs

`persistencecorrection/protected.py` wraps the Step 3 gate and extends its path
list with this task's frozen reads: `predictions_monthly.csv` and the four
`refined/cluster_mapping_k40_*_refined_contig3.csv` month maps for **both** fs1
and fs2 (the Step 3 list covers fs1 maps only). The union is de-duplicated with
Step 3's order preserved: 31 Step 3 paths + 10 declared additions = **35** unique
files. Baseline digests: `outputs/protected_hashes_baseline.json`.

## Commands

All commands run from the repository root.

```bash
# Phase 1: build, validate and freeze the persistence series into an immutable run dir
PYTHONPATH="$PWD/PersistenceCorrectionExperiment" python3 \
  PersistenceCorrectionExperiment/build_persistence_series.py --run-id phase1_<YYYYMMDD>

# Phase 3: fit, freeze and apply the calibrators, then print the R5 gate figures
PYTHONPATH="$PWD/PersistenceCorrectionExperiment" python3 \
  PersistenceCorrectionExperiment/run_calibration.py --run-id phase3_<YYYYMMDD>

# Contract tests
PYTHONPATH="$PWD/PersistenceCorrectionExperiment" python3 -m pytest PersistenceCorrectionExperiment/tests -q
PYTHONPATH="$PWD/PersistenceCorrectionExperiment" .venv-geodt-diagnostic/bin/python -m pytest PersistenceCorrectionExperiment/tests -q

# Step 3 must stay untouched
python3 -m pytest Step3ExpertCorrectionExperiment/tests -q   # 78 passed
```

`PYTHONPATH` is optional when invoking either entrypoint directly - both bootstrap
their own path - but is required for `pytest`.

`run_calibration.py` is environment-stable: `phase3_20260918` (python3, pandas
3.0.0 / numpy 2.4.2) and `phase3_20260918_venv` (`.venv-geodt-diagnostic`, pandas
2.3.3 / numpy 2.2.0) produce byte-identical `calibrators_fs{1,2}.json`,
`gate_r5_2020_out_of_sample.csv`, `calibration_groups.csv` and every
`calibrated_*.csv`. Only the two reliability tables differ, by at most 1.1e-16 in
one float-formatted column. A same-environment repeat (`phase3_20260918_repro`)
reproduces all 13 non-manifest artifacts byte-for-byte; its manifest differs only
in the timestamp and the run-directory path.

### Phase 3 run artifacts

| File | Contents |
|---|---|
| `calibrators_fs{1,2}.json` | the frozen calibrator set: isotonic knots / Platt coefficients, group key, fit window, thresholds. Hashed in the manifest |
| `calibrated_fit_2018_2019_fs{1,2}.csv` | in-sample calibrated rows, with persistence, route and route reason |
| `calibrated_selection_2020_fs{1,2}.csv` | out-of-sample calibrated rows; the Phase 4 selection input |
| `calibrated_test_2021_2024_fs{1,2}.csv` | calibrated test rows, emitted for Phase 5. **No Phase 3 metric was computed against their labels** |
| `calibration_groups.csv` | per group and per month pool: fit rows, positives, distinct scores, calibrator kind, fallback reason, any Platt failure text |
| `reliability_persistence_groups.csv` | pre/post mean probability, crisis rate and gap by persistence group, for the two measurable windows only |
| `reliability_bins.csv` | ten equal-width reliability bins per scope x window x stage x persistence group |
| `gate_r5_2020_out_of_sample.csv` | the four gate rows |
| `calibration_run_manifest.json` | windows and their roles, the C5/C6 supersession, grouping rationale, fallback order, fit-key overlap check, Brier pre/post, calibrator digests and determinism/round-trip flags, protected-hash summary |
| `protected_hashes.json` | before/after SHA-256 of all 35 protected inputs |

## Layout

```
PersistenceCorrectionExperiment/
  build_persistence_series.py      # Phase 1 entrypoint
  run_calibration.py               # Phase 3 entrypoint
  persistencecorrection/
    __init__.py                    # repo root + step3correction sys.path bootstrap
    protected.py                   # Step 3 hash gate, extended path list, output-path guard
    persistence.py                 # layer-1 persistence series and its contract
    calibration.py                 # frozen (month, partition) calibrators and reliability
  tests/                           # 39 contract tests (13 Phase 0/1 + 26 Phase 3)
  outputs/<run-id>/                # immutable run artifacts (git-ignored)
```

## Reuse from Step 3 (imported, never copied)

`load_expert_history` (source loading, admin-month key normalisation, the all-null
key-row drop, and `source_truth` = binarised observed phase computed before any
join), `assert_no_legacy_expert_columns` (the legacy-series firewall),
`counts_and_scores` (crisis-class metrics), and the whole `protected.py` gate.
The calendar join deliberately does **not** route through `ExpertTable.for_scope`,
whose column firewall structurally rejects any non-expert series.

Phase 3 additionally imports `reliability_bins` from
`scripts/paper_artifacts/analyze_georf_probability_uncertainty.py:115-148` rather
than reimplementing the binning; a contract test asserts the two produce equal
frames, so a drift in the repository's own implementation is caught here.

## Contract gates specific to Phase 3

| Condition | Result |
|---|---|
| A row outside 2018-2019 reaching a calibrator fit | `CalibrationContractError` before anything is fitted |
| A fit admin-month key reappearing in the 2020 or 2021-2024 window | `CalibrationContractError` |
| A score outside `[0, 1]`, or a non-finite score, reaching a calibrator | `CalibrationContractError` |
| A calendar month at apply time that the fit window never saw | `CalibrationContractError` |
| Any row left uncalibrated by `transform()` | `CalibrationContractError` |
| Non-0/1 fit labels | `CalibrationContractError` |
| Persistence coverage below 1.0 on any window | `PersistenceContractError` (Phase 1 contract, unchanged) |
| Existing run directory | `FileExistsError` |
| Output path outside `PersistenceCorrectionExperiment/outputs/` | `ProtectedPathError` before any directory is created |
| Protected artifact hash drift | `RuntimeError` |

Routing a group to its month pool (too few rows, single class, absent from the fit
window) is ordinary behaviour, not an error, and every instance is counted in
`calibration_groups.csv` and the manifest.

## Not authorized yet

Phases 4-6, and the PRD R4 conditional retraining branch. No threshold has been
swept, selected or frozen, and no metric has been computed against the 2021-2024
labels. No commits, no paper-artifact promotion, no change to Stage 1/2/3, and no
edit to `Step3ExpertCorrectionExperiment/`, the frozen packages, `src/`,
`scripts/`, or the Ethiopia experiment and its spec.
