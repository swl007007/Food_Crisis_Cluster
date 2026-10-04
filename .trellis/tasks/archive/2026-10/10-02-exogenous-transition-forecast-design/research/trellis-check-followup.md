# Trellis check follow-up: items left NOT REVIEWED by the earlier checkpoint (2026-10-03)

This was a read-only check with a 10-minute bound.
- No product, test or task-state file was edited.
- No tests were run, no fitting was done, and the live run directory was not touched.
- No protected 2025 values were opened.
- Base for the diff: `e843682..ca86bf7` in `FEWSNETGeoXGBExperiment/`.

## 1. Scope and coverage

What was reviewed:
- **`write_scenario_inputs`:** `scripts/prepare_fourclass.py:526-555` (the function) and `:642-709` (how `main` calls it).
- **Availability logic:** all of `src/experiment/availability.py`, including `stage1_input`, `fitting_view`, `key_features`, `label_pool` and `prediction_view`.
- **Stage 1 consumers:**
  - `scripts/run_stage1.py`, diff (the `scenario_entries` and `run_root` command changes);
  - `app/main_model_GF.py:444-567` (`scenario_root`, `scenario_main`).
- **Acceptance:**
  - `src/utils/acceptance.py:31-75` and `:233-297`;
  - the inventory and preparation helpers in `src/utils/run_identity.py`.
- **`scen-*` drivers:** `scripts/run_experiment.py:103-108` (`finish`), `:368-388` (`save_fold`) and `:700-1000` (`scenario_context`, `scen_candidate_ledger`, `_run_or_accept`, `scen_develop`, `scen_select`, `_accept_selection`, `scen_freeze`, `_accept_frozen`, `scen_historical`, `scen_actual`, the head of `scen_report`).
- **Tests:** the list of added classes and methods. The bodies I read were `ScenarioStage1.test_real_scenario_root_splits_original_keys_and_weights_children` (`tests/test_baseline.py:3692-3741`) and the `ScenarioDriverSmoke` fixture and negative paths (`:4120-4280`). I also ran a keyword coverage grep.
- **README:** the full added block.
- **Probes:**
  - `check_scenario_fit_keys.py` and `check_scenario_selection.py`, read in full;
  - `local_gate_reconcile.py` and `dev_ledger_reconcile.py`, docstrings and key lines only (grep).

What I did not reach:
- the bodies of most of the roughly 50 new test methods (judged from their names only);
- the tail of `scen_report`, and `scen_evaluate`;
- a line-by-line reading of `local_gate_reconcile.py` and `dev_ledger_reconcile.py`;
- `run_candidate`'s use of `X_weight` on S rows (earlier work covered this; I did not recheck it).

## 2. Status per scope item

| Item | Status | Evidence |
|---|---|---|
| 1a. Scenario input construction (roles, variants, weights, masks, label window) | PASS | <ul><li>`availability.py:239-259`: fit rows come from `label_pool(O, outer)`, which takes months in [O-59, O), released by O and not in the outer set. A gets k'=0 at weight 1. B gets k'=0/1/2 at 1/3 each, and the variants are grouped by `orig_key`.</li><li>`:215-237`: each variant's IPC history is rebuilt at its own origin from visible(o) minus (outer ∪ hidden(o, k')). The outer exclusion therefore reaches every variant (G2/R4). The variant mask never deletes the supervised label (R4).</li><li>`:290-315`: eval rows are one per original key, with the designated k at the key's own origin plus outer. Target rows are E3 at T under k. Truth is only the genuine labels at T, and empty targets are allowed.</li></ul> |
| 1b. Alignment and features | PASS | <ul><li>`availability.py:131-139`: a real run needs an explicit alignment plus ledger coverage.</li><li>`prepare_fourclass.py:550-552` pins `features.json`.</li><li>`main_model_GF.py:555-563` refuses inputs whose ordered schema columns differ.</li><li>Inputs are 648/6 = 108 distinct (strategy, H, T, k) files (`prepare_fourclass.py:545-551`). This matches the "108 inputs" in the readiness record.</li></ul> |
| 1c. Consumers | PASS | <ul><li>`run_stage1.py` `scenario_entries` refuses any schedule that is not exactly `plan.scenario_stage1_schedule()` on 13 identity fields, or that has an empty `input`.</li><li>`scenario_root` (`main_model_GF.py:461-541`):<ul><li>it performs the label-blind F/S split (`stage1_split` on area/month/origin/seed) and the C split (`confirmation_split`, seed 42) on original keys **before** variants are attached;</li><li>F contributes all of its variants, S and C one eval row each;</li><li>it checks keys lie in [O-59, O) (`:479-480`);</li><li>support is counted on F original keys (`:500`);</li><li>A uses unit weights, so `sample_weight` is None, and B passes its weights to the root and to children;</li><li>C is scored after freeze (frozen digest recorded).</li></ul></li></ul> |
| 2a. `accept_prepared` / `_identity_problems` / `accept_record` | PASS | <ul><li>`acceptance.py:44-66`: code and runtime identity must equal current values. Every recorded output must exist with an unchanged hash, and required files must be recorded.</li><li>Note: `check_inventory` (`run_identity.py:123-130`) does not flag *unrecorded* files, unlike `verify_outputs`. See F4.</li></ul> |
| 2b. `accept_scenario_stage1` | PASS, with one note (F3) | <ul><li>`acceptance.py:239-281`: all 648 entries are checked against the frozen schedule. Each needs `completion.json` with current code/runtime, the same preparation sha, the fixed G record and an exact candidate list.</li><li>`root.json` identity must equal the schedule, and the status set is closed.</li><li>For completed roots, the candidate files and outputs must match their hashes. A missing `candidate.json` raises an exception, so it fails closed.</li></ul> |
| 2c. Legacy `accept_fold` in `scen-*` | PASS (fail-closed for resume); FINDING F1 (select/report call it with partial identity) | <ul><li>`_run_or_accept` (`run_experiment.py:747-754`) passes the **full** identity, which covers phase, `map_id`, `prepared_outputs_sha256` and `availability_inputs_sha256` (plus `gate_k` and extension/table hashes for actual). An incompatible existing fold is refused, not refitted.</li><li>`accept_fold` (`acceptance.py:284-296`) also checks code/runtime identity, the closed status set, and that every output exists with its hash. `predictions.csv.gz` and `gate.json` are required when the status is fitted.</li><li>`scen_select` (`:796`) and `scen_report` (`:997`) check only strategy/horizon/k/target. See F1.</li></ul> |
| 2d. `_accept_selection` / `_accept_frozen` | PASS | <ul><li>`run_experiment.py:806-846`: the selection record is accepted, and each of the 72 fold sha values must equal the current `fold.json`.</li><li>The frozen record is bound to the selection sha. Historical and actual records are bound to the frozen sha.</li><li>`finish` (`:103-108`) hashes every file under the phase directory, so `historical.json` and `selection.json` cover their fold files.</li></ul> |
| 3. Test diff vs contracts | PASS for broad mapping; FINDING F2 (gaps); no vacuous assertion found in the bodies I read | See the mapping below. |
| 4. README | FINDING F5 (stale status) | `README.md` new block. |
| 5a. `check_scenario_fit_keys.py` | PASS, with a limitation (F6) | Independent: no package imports. It rebuilds the lawful pool with explicit filters and recomputes the hashes of keys, labels and weights, plus original support. |
| 5b. `check_scenario_selection.py` | PASS, with a limitation (F7) | It recounts exact-fraction confusion counts, the parity screen, the ranking, A-wins-ties and the unmet reason from the saved predictions. It checks the fold sha binding, output hashes, argmax consistency and that system equals pooled on non-local routes. |
| 5c. `local_gate_reconcile.py` | PASS (partial read) | <ul><li>Its docstring and key lines show independent filters with no package import.</li><li>It recounts gate F1 from the saved gate pairs as exact fractions, with gain strictly > 1/100 compared as a string equal to the Fraction.</li><li>Gate truth is checked against lawful labels (`:157-158`), support floors and routes against independent pools (`:160-194`).</li><li>Its docstring discloses that internal local key lists are not saved.</li></ul> |
| 5d. `dev_ledger_reconcile.py` | PASS as a consistency check; **not independent** (F8) | It imports `scripts.run_experiment` and `src.experiment.stage3` (`:21-22`). |
| Historical, actual-2025 and truth-evaluation outputs | INCOMPLETE | scen-historical is live; scen-actual and scen-evaluate have not run. |

### Test mapping (by class name, plus the bodies I read)

| Contract | Test class(es) |
|---|---|
| R1 / AC1 (release-aware masks, cutoff, monthly/annual alignment, hidden inputs, forecast-only rows) | `ReleaseAwareViews` (11 tests) |
| R4 / AC2 (weights conserved, original support, root prefix) | `WeightedContinuation`, `Stage1Variants`, `ReleaseAwareViews.test_strategy_b_groups_conserve_weight_and_original_support`, `ScenarioStage1.test_real_scenario_root...` |
| R5 (648 schedule, roles, undefined E2, no-target root, pinned features) | `ScenarioStage1` |
| R6 / AC3 (E4 crisis weights and NA, routes, common pool) | `ScenarioStage2` |
| R6 (72 folds) and R3 (selection rule, ties, stop) | `ScenarioDevelopment` |
| R6 / Stage 3 (strict > 0.01 gate including the exact 0.01 boundary, cache isolation, stale-global refusal) | `ScenarioStage3` |
| R3 (bootstrap NA, shared schedule) and R2 (Study 2, onset, country table) | `ScenarioReporting` |
| R4 / R2 (historical calendar, actual refusals, per-country intensity, expert matching) | `ScenarioFinalPath` |
| End-to-end, including truth release bound to the frozen actual record and June unevaluable | `ScenarioDriverSmoke` |

## 3. Actionable findings

None of these findings changes a development number already produced.

### F1. `run_experiment.py:796` and `:997`: partial identity at select and report

- **Severity:** evidence-provenance only.
- **What happens:** `scen_select` and `scen_report` call `accept_fold` with only strategy/horizon/k/target. They do not check `phase`, `map_id`, `prepared_outputs_sha256` or `availability_inputs_sha256`. `scen_select` then copies `map_id` from the record as given.
- **Failure scenario:** a fold directory copied from another run that used the same code would pass at select. One example is a different preparation or ledger in a sibling run. Its predictions would enter the A/B decision.
- **What prevents it today:**
  - code identity;
  - the single-preparation run directory;
  - the fact that scen-develop has already validated every fold with full identity.
- **For later phases:** historical.json's output hashes bind report inputs, but they do not bind fold identity to the frozen `map_id`.
- **Suggested fix:** pass the same expected identity that `_run_or_accept` uses, at least `phase`, `map_id` (frozen) and `prepared_outputs_sha256`.

### F2. Test coverage gaps (`tests/test_baseline.py`)

- **Severity:** required-behaviour (verification gaps only, not defects shown in the code).
- **(a) No negative test for `accept_scenario_stage1`.** It is exercised only on the positive path, through the smoke fixture at `:4145-4190`. That fixture **synthesises** the Stage 1 completion records, so the real `run_stage1` completion writer is never checked against this acceptor.
  - Failure scenario: if the writer's record field names drifted, for example `g_selection`, the unit suite would not detect it. Only the real accept call would.
- **(b) No permutation test of C labels on the scenario path.** The permutation tests at `:1380-1800` belong to the old rootconf path. The scenario test (`:3730-3732`) asserts only that the frozen digests before scoring are equal.
  - This is weaker than the implement.md §5 check ("permuting C labels cannot alter fitting, search, support, maps or routes").
- **(c) No direct test** that `scen_select` and `scen_report` refuse a fold whose `map_id` or inputs disagree. See F1.
- **(d) Five-level and 80-round limits** are not tested again on the scenario path. The scenario test patches `MAX_DEPTH=3`.

### F3. `acceptance.py:239-281`: unexpected root directories are not detected

- **Severity:** evidence-provenance only.
- **What happens:** unlike `accept_stage1` (`:107-109`), `accept_scenario_stage1` never lists `stage1_scenario/roots` to report unexpected directories. Outputs that are present but unrecorded are also not flagged, because `check_inventory` ignores them.
- **Failure scenario:** leftover partial directories from the 21 crashed attempts, or stray roots, sit unreported beside accepted evidence. This does not change the 648-row ledger.

### F4. `run_identity.py:123-130`: `check_inventory` does not flag unrecorded files

- **Severity:** evidence-provenance only.
- **What happens:** every `accept_record`, and therefore every fold, selection and frozen acceptance, tolerates extra files added after the completion record was written.
- **Failure scenario:** a file added later, such as a replaced `pooled_predictions.csv.gz`, would be caught because it is recorded and hashed. A *new* unrecorded file, such as a stray gate file, would be silently present. Consumers read only recorded names, so results are not affected.

### F5. `README.md`, new block: stale status

- **Severity:** evidence-provenance only (documentation).
- **What it says:** "Real fitting remains blocked by the source facts in task `data-readiness.md`".
- **Why it is stale:** for 2018–2024 this is superseded. D7 was resolved through the user-confirmed `reference_month_end` convention, Stage 1 and development have run, and historical is running. "163 regression checks plus two" is accurate as of `b43ef6a`.
- **Failure scenario:** a reader concludes that the development results were produced before D7 cleared, or that no real run exists.
- **Accurate parts:** the command order and interfaces match the code: the prepare flags are required together, scen-report needs historical, scen-actual refuses without a table or extension, and scen-evaluate binds `frozen_actual`.

### F6. `check_scenario_fit_keys.py:35-36`: excluded months are taken from the record

- **Severity:** evidence-provenance only.
- **What happens:** the probe recomputes the record's own hidden cycles independently, but takes `excluded_months` (the outer exclusion inherited by internal gate globals) from the record under test.
- **Failure scenario:** an internal gate global that omitted the outer exclusion would also record it as omitted. The probe would then reconcile, because the label pool it builds would use the same wrong mask.
- **Suggested fix:** derive the expected exclusion from the fold's outer origin and k; the probe does not need the record for this.
- The probe also mirrors the package's hash byte layout (`<i8` columns, (area, month, variant) order). That is acceptable for an exact-reproduction check, but agreement shows equal bytes, not independently derived semantics.

### F7. `check_scenario_selection.py:36-45`: truth and persistence values are not checked independently

- **Severity:** evidence-provenance only.
- **What happens:** `y_true_code` and `persistence_class_code` are read from the saved predictions. They are compared only between system and pooled, not against `observations.csv` or the lawful latest-visible label.
- **What the probe does establish:** the recount and the decision rule are independent.
- **What it does not establish:** the correctness of the comparator and truth values. The run directory and the preparation hash are hard-coded (`:5`, `:18`), which is fine for this one review.
- **Failure scenario:** a persistence comparator that took a hidden or forward label would pass this probe. `dev_ledger_reconcile` reports a "prediction keys/persistence" check (`:122`), but that check uses package code (F8).

### F8. `dev_ledger_reconcile.py:21-22`: not independent

- **Severity:** evidence-provenance only.
- **What happens:** it imports `run_experiment` and `stage3`, so a shared logic error in the package would reproduce in the probe. Its docstring does not claim independence.
- **Action:** the checkpoint should describe it as consistency evidence, not as an independent check.

## 4. Remaining gaps

- Historical (live), actual-2025, truth evaluation, report bootstrap values, expert comparators and the 2025 availability table are all INCOMPLETE.
- Not reviewed:
  - the bodies of most new tests;
  - `scen_report` past `:1000`, and `scen_evaluate`;
  - a line-level read of `local_gate_reconcile.py` and `dev_ledger_reconcile.py`;
  - weighted handling of S rows inside `run_candidate`.
- Earlier checkpoint findings 1–4 and L1, and the executor addendum, still stand. None was re-verified here.

## 5. Statement

This follow-up covers only the items the previous checkpoint marked NOT REVIEWED. It is not a whole-task pass and it is not the independent spot or close audit. No fixes were made: the findings above are reported for the executor to classify.
