# Acceptance evidence index — A1 to A10

Run `pop-v1`. Paths are relative to the repository root; `RUN` abbreviates
`IPCCHPopulationHistoryExperiment/runs/pop-v1`. Every file listed here is in
the commit — the 731 MB `RUN/data/rich561_X.npy` is the single deliberate
exclusion, hash-bound in `RUN/manifest.json` (`matrix.matrix_sha256`) and
`RUN/freeze.json` (`matrix_sha256`), both
`ce85badb07a37f0ce9df7b2e9d412bc64b651872a9aecd6048cca8079a618648`.

Pinned source `IPCCH_2026_completed.csv`, SHA256
`ae696087c3bbb280537ae269a05924133acdb51060d31290523404fa8a717673`; frozen
inventories `IPCCHPopulationHistoryExperiment/config/feature-schema.json`
(`c660d360…`) and `candidate-configs.json` (`5055e11e…`), copied verbatim into
`RUN/inputs/`.

## A1 — source, keys, QC, country gate, unchanged originals

* `RUN/manifest.json` → `target_gate`: 42,695 valid / 15,206 positive /
  27,489 negative / 6,227 areas, `gate_pass: true`. `prepare` raises on failure
  (`prepare_data.py`, "target gate failed").
* `RUN/data/target_ledger_valid.csv.gz` — exact binary labels and normalized
  components, produced by the earlier experiment's `build_target_ledger`, which
  this module reuses rather than reimplements.
* `RUN/manifest.json` → `inputs.country_lookup` (hash, coverage, areas with a
  blank ISO3 retained). Coverage of every valid area is enforced, not logged.
* Exactly-.20 is non-crisis, and history classification reads the ledger label
  rather than re-deriving it from a share:
  `test_contracts.py::test_exact_threshold_history_uses_the_ledger_label`.
* Nothing under `IPCCHGeoRFExperiment/` or the source tree was modified:
  `git show --stat` for every commit in this task.

## A2 — schema, hand-checkable histories, aliases, origin bound

* Column names and order: `RUN/data/feature_schema.csv` (561 rows, positions and
  block labels). `load_frozen_spec` fails on a duplicate, a collision with
  original93, a count mismatch, or an alias that shadows a materialised column.
* Hand-computed fixtures, 41 tests, all passing:
  `IPCCHPopulationHistoryExperiment/test_contracts.py`. They cover six-slot
  ordering, elapsed months, window edges on both sides, q3 == 0 leaving
  `severe_fraction` undefined, exactly-.20 labels, event dating at the later
  endpoint, missing-versus-zero support, and per-statistic support thresholds.
* Alias deduplication is recomputed and compared exactly per run:
  `prepare_data.check_aliases`, recorded at `RUN/manifest.json` →
  `alias_audit`; negative case in
  `test_contracts.py::test_alias_recomputation_detects_a_mismatch`.
* Each row's own six observation months: `RUN/data/history_source_keys.csv.gz`.
  Every feature's newest source month ≤ its origin is enforced in `prepare`
  ("an observation newer than the row origin was used").
* `RUN/data/feature_audit.json`: 0 infinities, 0 all-NaN columns, 1 constant
  column (`hist_q5_m06_slope`).

## A3 — fold keys, identical matched support, explicit fullpool superset

* `RUN/folds/calendar.csv` — all 266 scheduled folds with `test_rows`,
  `test_rows_with_history`, `full_pool_rows`, `matched_pool_rows`.
* Per-fold realised support and every route: `RUN/development/folds/*.json`
  (144) and `RUN/main/folds/*.json` (122).
* Masks come from keys alone, never from per-model NaN filtering
  (`run_pipeline.fold_support`); disjointness and the training-label cutoff are
  enforced there and tested by
  `test_contracts.py::test_fold_masks_keep_fitting_and_evaluation_disjoint`.
* Identical E_history keys across arms, and identical truth and persistence on
  them, are proved before any metric: `report_results._cohort_audit`, recorded
  at `RUN/reports/summary.json` → `cohort_audit.per_horizon`.

## A4 — correction labels, orientation, cutoffs, fallback

* Score orientation, both flip directions, clipping that leaves the raw output
  intact, `score > t` with equality negative, and the no-flip pair reproducing
  persistence exactly: `test_contracts.py`, tests
  `test_correction_scores_are_oriented_toward_crisis`,
  `test_share_scores_are_clipped_but_the_raw_output_survives`,
  `test_decision_at_the_cutoff_is_negative`,
  `test_no_flip_thresholds_reproduce_persistence`.
* A constant correction target stays conditional on `b`:
  `test_constant_correction_target_stays_conditional_on_b`.
* Missing `b` is never corrected or zero-filled: the correction arm fits and
  scores on matched/E_history rows only (`run_pipeline.run_fold`), and
  `crisis_oriented` raises if any `b` is missing.
* One shared fallback stream, and no fallback probability in a share metric:
  `RUN/reports/metrics_e_all_combined.csv` (labelled per row),
  `RUN/reports/share_diagnostics.csv` (share arm's own E_history only).

## A5 — schedules, candidate inventories, time-causal selection ledgers

* `RUN/development/selection_ledger.csv` — 144 rows, every arm × horizon ×
  candidate with F1, both thresholds, changes-from-b and confusion counts.
* `RUN/freeze.json` — selected candidate and thresholds per arm and horizon,
  the primary family, the four-horizon mean deltas, spec/matrix/code hashes,
  `information_cutoff: 2022-12`.
* Selection consumes only 2020-2022 development predictions
  (`run_pipeline.run_selection`, restricted to `E_history` and development
  months). Calendar causality is tested by
  `test_contracts.py::test_development_and_main_calendars_do_not_overlap`.
* Replayed in a clean clone: `selections`, `primary_family`,
  `primary_mean_delta`, `persistence_development_f1`, spec identity, matrix
  hash, cutoff and cohort all reproduce exactly (see RESULTS.md,
  "Reproduction"). The replay goes to `--replay-into DIR`.
* **Immutable freeze.** `run_selection` refuses to overwrite an existing
  `freeze.json` (`run_pipeline.py`, "Selection is frozen once written");
  `--replay-into` re-derives it elsewhere, `--refreeze` discards one
  deliberately. `RUN/validation/freeze_replay/freeze.json` is the re-derivation,
  matching the committed freeze on all six scientific fields with a different
  `frozen_utc`. Tested by
  `test_contracts.py::test_identity_comparison_separates_code_drift_from_science`.

## A6 — recomputable metrics, shared draws, leave-year-out, conjunctive verdicts

* `RUN/reports/metrics_e_history.csv` (confusion counts per method × horizon),
  `deltas_e_history.csv`, `metrics_e_all_combined.csv`.
* `RUN/reports/bootstrap.json` — seed 42, 52-country axis, 2,000 valid draws in
  2,000 attempts, 0 rejections, the draw ledger and the reason channel.
  `RUN/reports/bootstrap_draws.csv.gz` holds every draw.
* One multiplicity vector is reused by every method and horizon:
  `test_contracts.py::test_bootstrap_shares_one_multiplicity_vector_across_methods`
  (identical predictions must give exactly zero delta in every draw), plus
  `test_bootstrap_is_reproducible_from_its_seed` and
  `test_undefined_f1_rejects_the_whole_draw`.
* `RUN/reports/leave_year_out.json` — 2023/2024/2025 omissions per comparison.
* Verdicts and their four conditions: `RUN/reports/summary.json` → `verdicts`,
  `claims`. Incompleteness is not a pass:
  `test_incomplete_interval_makes_the_verdict_incomplete_not_a_pass`,
  `test_one_negative_horizon_blocks_a_stable_gain`,
  `test_one_bad_omitted_year_blocks_a_stable_gain`.
* All method contrasts stay visible, including the failures:
  `deltas_e_history.csv` carries every ordered pair.
* Strata: `stratified_by_{cohort,country,target_year,history_age,source_support}.csv`.

## A7 — runtime, fitted settings, imputation, routes

* `RUN/manifest.json` → `runtime`: expected versus actual for Python 3.12.10,
  numpy 2.2.6, pandas 2.2.3, sklearn 1.6.1, XGBoost 3.0.0, scipy 1.15.2,
  GeoPandas 1.0.1, Shapely 2.1.0, `pass: true`. `verify_runtime(strict=True)`
  stops the run on any mismatch; it compares rather than logs.
* **Fitted-state readback**, not the requested dictionary:
  `run_pipeline._fitted_identity` records the estimator's own `get_params()`,
  its `save_config()` booster configuration and its realised round count.
  `RUN/main/model_identity/*.json` carries these for all 660 selected models;
  `RUN/development/effective_params.json` carries a real fitted readback for
  each of the 36 (arm, candidate) pairs, which fix every parameter and the seed
  and therefore do not vary by fold. Both the requested and the fitted values
  are kept, so a gap between them is itself visible. Tested by
  `test_contracts.py::test_fitted_identity_reads_the_estimator_back`.
* Per-fit routes, timings, training counts and requested parameters for every
  one of the 5,214 search and main fits: `RUN/{development,main}/folds/*.json`.
* RF fills come only from matched training X, one imputer per fold reused
  across RF candidates: same JSONs → `imputer` (fitted rows, all-missing and
  negative-max column counts, fill values). Behaviour tested against the pinned
  release in `test_contracts.py::test_imputer_fills_from_training_columns_only`,
  including the disclosed negative-column limitation.
* Route accounting: every fit in this run took the `model` route; the
  constant-target and single-class routes are exercised by
  `test_constant_targets_take_the_declared_route`.
* `RUN/inputs/baseline.json` — pristine GeoRFBaseline v0.1.0, patch **not**
  applied, payload re-verified on every attach (`run_pipeline.verify_baseline`).
* No SMOTE, no resampling, no pseudo rows: `candidate-configs.json` declares
  `smote: false`, and no code path constructs synthetic rows.

## A8 — budget reconciliation

| | folds | fits | declared bound |
|---|---:|---:|---:|
| development | 136 non-empty, 8 empty | 4,506 | 5,184 |
| main | 110 non-empty, 12 empty | 660 | 732 |
| verification replay | 8 | 48 | separate, ≤48 |
| refit-for-provenance | 110 | 660 | separate |
| development param readback | 1 | 36 | separate |

**Immutability.** Every fold record carries an `identity` over source, schema,
candidate inventory, key and calendar digests, frozen selections and code
hashes; `completed_folds` reuses only matching records and reports the rest as
refused. `--stage main` refuses to continue under drifted frozen inputs, or
under drifted code without `--allow-code-drift`. The refit stage writes to a
scratch directory, never over `main/folds`. Tested by
`test_contracts.py::test_fold_reuse_refuses_a_record_from_another_identity`.

Counts recomputable from `RUN/{development,main,validation}/folds/*.json` and
summarised in `RUN/stage_{development,main,verify}.json`. The pilot is the
first supported 2020 h=1 development fold and is included in the development
count, not additional to it. Results are immutable: no earlier run was
modified, and nothing outside `IPCCHPopulationHistoryExperiment/` and the task
directory changed.

## A9 — independent replay

* **Fold replay** — `RUN/validation/replay.json`: the first and last non-empty
  main fold at each horizon (8 folds, 48 fits), chosen without reference to any
  score, re-fitted **sequentially with one worker** and compared to the stored
  parallel predictions. `all_identical: true` on all eight, covering 60,229
  prediction rows.
* **Reporter replay** — `RUN/validation/report_replay.json`: all 15 report files
  regenerated into a second directory, `all_identical: true`.
* **Clean-clone replay** — at commit `45b74fb`, from committed evidence only and
  without the feature matrix: 41/41 contract tests pass, selection reproduces
  every scientific field, and `summary.json` reproduces exactly apart from
  `run_dir`. Commands in RESULTS.md, "Reproduction".
* **Retained estimators, full schedule** — `RUN/validation/model_persistence.json`:
  every selected model at every main origin was refitted under the frozen
  choices and kept; all **110/110** folds and **660** models regenerate the
  stored predictions exactly. Identity per model in
  `RUN/main/model_identity/*.json`: SHA256, byte size, `get_params()` readback,
  booster config, round count, and the digest of the ordered fitting keys.
* **Prediction-only replay** — `RUN/validation/model_replay.json`: all **660**
  saved estimators were loaded from disk, their digests verified, and their
  scores reproduced without any fitting. This is the check a refit cannot make.
* The binaries for the first main fold at each horizon are committed (24 models,
  19 MB) so a reviewer can run that replay directly; the remaining 533 MB are
  local and regenerable by `--stage persist`, which verifies them against the
  stored predictions rather than asserting them.
* Fitting-key digests make A3 checkable rather than trusted: the five matched
  arms in a fold share one digest and `fullpool_xgb` has another.
* No claim in RESULTS.md rests on console output.

## A10 — lifecycle

* Audit run `3280efe27b0447a4aa69740d5a2a0948`, base_sha
  `64b3b302b2fd09b11a6a5a3ca471c9c1d99ee0a5`, executor session
  `18826b65-ca2f-4276-b70a-2888ebc93341`, registered repository path
  `/mnt/c/users/swl00/ifpri dropbox/weilun shi/google fund/analysis/2.source_code/step5_geo_rf_trial/food_crisis_cluster`.
* Prior gates were checked clear before starting: `trellis-audit status`
  reported `active_runs: []` and `gate_open: 0` on every historical job.
* Commits in this task: `64b3b30` (frozen inventories), `de6cd41` (package),
  `d930877` (contract tests), `a21cc9e`, `dd89b0c` (reporter coverage and the
  `_cohort_audit` horizon fix), `5c3f204` (run evidence and clone-reproducible
  replays), `45b74fb` (matrix hash binding), plus this documentation commit.
* Completion is queued by `trellis-audit close` from the bound session. A
  launched audit is not a pass; this index is a pre-close deliverable.

## Remediation of close audit 826221934ce1b02cad427dc8

That audit met A1-A4 and A6 — including an independent rebuild of the full
matrix from raw source that matched its SHA256 — and raised two majors, both
valid. Both are now addressed in code, with evidence and with tests:

| finding | fix | evidence |
|---|---|---|
| fitted estimators discarded; requested config labelled `effective_params` | `_fitted_identity` readback; `--stage persist` retains every selected model; `--stage replay-models` predicts from the saved ones | `main/model_identity/*.json`, `validation/model_persistence.json` (110/110), `validation/model_replay.json` (660/660) |
| `select` overwrote an existing freeze; fold reuse checked no identity | freeze written once, `--replay-into` for replay; per-fold identity over inputs/schema/candidates/selections/code; `main` refuses drifted inputs or code | `validation/freeze_replay/`, `identity` in every fold record, `stage_*.json` → `folds_refused_stale` |

Four new contract tests cover the readback, the model round trip and digest
binding, identity-refused reuse, and the code-drift/science separation.

## Remediation of re-audit 4c552551d350bf044879fdb9

That re-audit closed the gate (`gate_open: 0`) and moved A7 and A9 to **met**,
confirming the fitted-state readback and the retained-model replays. It then
found two defects **in the guards themselves**, both real:

**The stale-fold override destroyed evidence.** `--allow-stale-reuse` put
mismatched completed folds into the in-place fitting queue, so the advertised
way to continue a historical run would have overwritten its own prior fold
evidence under the same run id and freeze. Fixed by deleting the override: a
mismatched completed fold now rejects the run and points at a fresh directory,
and a second guard refuses to queue any fold that already has a completed
record, whatever the reuse logic decided.

**A freshly frozen run could not reach main fitting.** The freeze carried two
identity fields while `run_identity` computed five, and the comparison read
"absent on one side" as drift — so `--stage main` refused on a run nothing had
touched. Fixed: the freeze records the complete identity;
`compare_identity` only compares fields present on both sides; absent fields
are reported separately by `missing_identity_fields` and refused by name
(`--allow-legacy-freeze`) rather than passed over or back-filled with a hash
invented after the fact. `check_main_preconditions` is now a function the tests
drive directly.

Three further tests: a freshly frozen run reaches main scheduling and real
key/calendar drift is still refused; a legacy freeze is refused by name and
lists exactly which fields it cannot prove; a stale completed fold stops the
run instead of being requeued. 48/48 pass.

Consequence for `pop-v1`, stated plainly: its fold records predate identity
recording, so `--stage main` now refuses to continue it — correctly, since the
stage is finished and those records cannot prove what they were made under. The
read-only stages (`replay-models`, `select --replay-into`, the reporter) all
still run against it.

## Remediation of re-audit 9c8d9909b25c6e16bd2c4d48

A1-A4, A6, A7 and A9 met. Three majors, all real, all in the guards:

**A partial identity dictionary passed as proof.** `compare_identity` compares
only fields present on both sides — right for the legacy-freeze case, where a
human names the gap and accepts it, wrong for a fold record joining a fitting
queue with no one looking. `completed_folds` now rejects a record before
comparing values if it omits any scientific field this run defines, or
`code_sha256`. Tested with full / partial / empty / code-less records:
only the full one is reusable.

**Selection did not prove its cohort was complete.** It maximised F1 over
whatever predictions were on disk while persistence was scored from the full
calendar, so an interrupted run or a lost artifact would yield a perfectly
valid-looking freeze derived from a smaller cohort than its own baseline.
`reconcile_development_cohort` now runs before anything is optimised: every
supported development fold must have a completed record and contribute
predictions, each arm/candidate/horizon must cover exactly the expected
evaluation keys with no duplicates and nothing outside the window, and all
candidates must share one support. The result is recorded in the freeze as
`cohort_reconciliation`. On pop-v1 it passes — 123 supported folds, all 36
arm/candidate combinations covering 9,668 / 9,667 / 8,887 / 8,168 keys — which
independently confirms the delivered selections were not affected.

**The pilot bypassed the guards.** `stage_pilot` called the executor directly,
so repeating the documented pilot command would refit and replace a completed
fold, even after selection or main fitting. It now obeys the same rule as every
other fold: an exactly matching completed pilot is reused with zero fits, and an
unprovable or mismatched one stops the run.

Three more tests; 51/51 pass.

## Remediation of re-audit 5ec6c8c857a8ef8d6dbb1696

A8 moved to **met**; A1-A4, A6, A7, A9 met. One major left, and it was the
sharp version of the previous one:

**A new freeze did not prove its folds were this run's folds.** The cohort
check accepted any completed record, so if data, keys, candidate parameters or
code changed after development, `--stage select` would attach the *current*
identity to predictions generated under different conditions — and
`check_main_preconditions` would then trust that freshly minted freeze.

Fixed by separating the two things `select` was doing:

* **Creating an authoritative freeze** now validates every consumed fold
  against the expected development identity and refuses any mismatch or any
  record that cannot prove one. The freeze records
  `authoritative: true`, `kind: authoritative_freeze`, and a
  `consumed_predictions_sha256` binding it to the exact prediction files it
  was derived from.
* **Replaying an existing run** (`--replay-into DIR`) is an explicitly labelled
  read-only re-derivation of historical evidence: `authoritative: false`,
  `kind: read_only_replay`, `identity_enforced: false`.
  `check_main_preconditions` refuses such a freeze outright, so a replay can
  never become a commitment.

This is what preserves `pop-v1`: its development folds predate identity
recording, so a new authoritative freeze over them is refused by name — with
the error pointing at `--replay-into` — while the documented replay keeps
working and reproduces `share_xgb` and every selected value.

Two more tests: a complete cohort with one mismatched fold is refused while the
same cohort replays fine; a replay freeze cannot drive a main schedule.
53/53 pass.

## Remediation of re-audit c5499895121d3c7b179654a9

A4, A6, A8 and **A10** met, no violated acceptance criterion, gate closed. One
minor finding against R10 — an interpretation error in my own write-up, and a
correct catch:

RESULTS.md described `fullpool_xgb` and `rich_rf` together as "the two arms with
the least constrained fitting support". That is true of `fullpool_xgb` and false
of `rich_rf`, which trains on exactly the same matched keys and the same 561
columns as the other four matched arms. Its edge is an *algorithm* difference —
forest against boosting on identical rows — not a support advantage, and this
matched-pool design gives no evidence for one.

Corrected in RESULTS.md in both places it appeared: the headline-table commentary
now separates the two reasons explicitly, and "What this does not license" now
says that even `fullpool_xgb`'s support advantage is *consistent with* the data
rather than established by it, since there is no matched-support control for the
extra rows. The same error was corrected in the project memory note, which now
carries a standing warning not to write "more fitting support wins".

No number changed; this was a claim about why the numbers look as they do.

## Known deviations, stated rather than buried

1. **Fold-level parallelism.** `technical-contract.md` §7 asks for sequential
   folds; the run used `--workers 12` with user approval after the pilot
   measured ~21 h sequential. Every estimator kept the frozen `n_jobs=1`. The
   sequential replay in A9 is the evidence that no prediction changed.
2. **`freeze.json` code hashes are pre-edit.** `prepare_data.py` and
   `run_pipeline.py` were edited after selection, for replay plumbing only
   (matrix-free loading, reading the matrix hash from the manifest). The
   clean-clone selection replay reproduces every selected value under the edited
   code, which is what shows the edits were inert.
3. **`manifest.json` matrix hash is backfilled.** It was added after the run,
   computed from the on-disk matrix and cross-checked against the value
   `freeze.json` had already recorded during selection. The two agree.
4. **The earlier experiment's tests are untracked.** `IPCCHGeoRFExperiment`'s
   four `test_*.py` modules are caught by the root `test_*` ignore and are not
   in the repository. This task's tests are carved back in; the older ones are
   left alone as they belong to a closed, already-audited task. Flagged for
   whoever reopens that line.
