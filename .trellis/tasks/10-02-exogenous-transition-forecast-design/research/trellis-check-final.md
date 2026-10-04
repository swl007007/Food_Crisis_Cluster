# Trellis check, final bounded pass (2026-10-03)

This was a read-only check run by the trellis-check sub-agent. I made no product, test or task-state edits and ran no fits, tests, source tracing, audit or commit. The run directories were only read. The only file written is this report. Package code identity fd25e2f7 is taken from the earlier checkpoints, not re-verified here. Paths below are relative to `FEWSNETGeoXGBExperiment/` unless they say otherwise. RUN = `C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1`, REL = `...\scen-b43ef6a-v1.truth-release-v2`.

## 1. Scope and coverage

I reviewed only the five remaining items in the dispatch. Items marked PASS in `trellis-check-progress.md` and `trellis-check-followup.md` were not redone. All five items were reached within the time bound.

- **Code read:**
  - `scripts/run_experiment.py:916-1066` (`_with_pooled`, `_origin_truth`, `_report_entries`, `_expert_for`, `scen_report` to its end, `scen_evaluate`);
  - `scripts/report_fourclass.py:256-410` (`crisis_paired_bootstrap`, `study_rows`, `onset_recall`, `country_table`, `keyed_expert`);
  - `src/experiment/availability.py:80-280` (`hidden`, `hidden_for`, `union`, `gate_dates`, `fitting_view`, `prediction_view`);
  - `src/experiment/stage3.py:150-230` (ScenarioPanel);
  - `app/main_model_GF.py:444-541`;
  - `src/partition/transformation.py:226-235,800-845`;
  - `tests/test_baseline.py:4250-4285` plus a grep across the ScenarioReporting, ScenarioFinalPath and ScenarioDriverSmoke tests.
- **Artifacts recomputed independently** with pandas/hashlib only, no package import:
  - REL `release.json`, truth and crosswalk;
  - RUN `scenario_evaluation/*`, `scenario_actual/h*/*/{predictions,gate_pairs}`, `scenario_globals/*/*/*.json|.ubj`, `prepared/manifests/release_ledger.csv`, `prepared/ledgers/observations.csv`;
  - task `research/launch/actual_availability.csv`.

**Coordinator recount reused, not repeated.** The coordinator's independent actual-metric recount (reported result: 0 problems) is cited here. This check did not repeat it.

- The recount covers truth and exact-origin joins, forecast immutability, argmax, confusion/F1/P/R, the paired 2,000-draw seed-42 bootstrap and the country fields.
- Script: `research/probes/actual_metric_review.py`, sha d73f5e03134ed6a73aa9b4e05333087db1100bbef86a653db3d21702bc39edb5 (hash verified here).
- Result: `research/actual_metric_review.json`, sha 4d60d56fe84207a8574200543566effa2dfc95dae28ad6e6d5964e90db74cdd0 (hashed here, contents not re-read).
- The evaluated record is `evaluation.json`, sha 8c43375a5f258a105c55540245fc0deb4284f36c550dbb1638a5709a6d79d427 (hash verified here).

My checks below are limited to joins, routes, masks and recorded fields. I did no metric or bootstrap recount.

## 2. Status per item

| # | Item | Status | Evidence |
|---|---|---|---|
| 1a | `scen_evaluate` truth-release binding | PASS | <ul><li>`run_experiment.py:1029-1039`: requires all six `TRUTH_RELEASE_FIELDS` and `approved is True`. `frozen_actual` must equal the sha of the current `actual.json`, and the truth bytes must equal `truth_sha256`. Truth keys must be unique with codes on the 0..3 axis.</li><li>Each fold is re-accepted and its `fold.json` sha must equal the frozen record (`:1052-1054`).</li><li>Recomputed: the truth sha (7ccc336f…), `frozen_actual` (c896a583…) and crosswalk sha (ab629b01…) all equal the release v2 values.</li><li>`evaluation.json` `truth_release` holds exactly those six fields, with approved = true.</li></ul> |
| 1b | Truth join without zero fill | PASS (but see F2) | <ul><li>`:1056-1057` uses a left merge, so unmatched keys stay NaN.</li><li>`study_rows` (`report_fourclass.py:303`) keeps only rows with non-NaN `truth_code` before any `astype(int)`.</li><li>Recomputed per case: the keyed `truth_code` equals the release truth on every key, with NaN elsewhere. Oct H4 and H8 each have 4,457 non-null of 5,718 keys, and 0 truth rows went unjoined. June cases have 0 non-null.</li><li>No duplicate keys.</li><li>The keyed `y_pred_code` and persistence equal the frozen `predictions.csv.gz` exactly.</li></ul> |
| 1c | June unevaluable output | VALID-NA (route) | No genuine June-2025 CS source exists. `h4_2025-06` and `h8_2025-06` have `evaluable: false` with reason "no released genuine truth for this target (forecast/coverage only)". Their Study1 key count is 0 and every metric is None. The country tables keep all 22 countries plus `nan`, with `labelled_keys = 0` and `cohort_keys = 5718`. |
| 1d | Study2 exact-origin truth | PASS / VALID-NA | <ul><li>`_origin_truth` (`:926-930`) reindexes exactly (area, T−H) on genuine observations plus released truth, with observations taking precedence (`:1046`). It never uses the latest earlier label.</li><li>The Oct origins (2025-06 for H4, 2025-02 for H8) have no released CS, so origin truth is all NaN.</li><li>Oct Study2: keys = 0, `excluded_missing_origin` = 4,457 (H4 and H8), `excluded_origin_crisis` = 0, onset "no genuine onset".</li><li>June Study2 has `excluded_missing_origin` = 0 because Study1 is empty there. The non-empty origin truth (5,336 keys at H8 origin 2024-10) is genuine and unused.</li></ul> |
| 1e | Expert NA route | VALID-NA (route) | <ul><li>`_expert_for` → `keyed_expert(keys, None)` sets `expert_reason = no_documented_expert_table` on all 5,718 keys in every keyed file, with `expert_class_code` all NaN.</li><li>Study1 Oct `expert_coverage = {no_documented_expert_table: 4457}`, and `vs_expert` n = 0 with CI None.</li><li>`expert_table_sha256` = null.</li></ul> |
| 1f | Country coverage-only rows | PASS (but see F3) | `country_table` groups the whole cohort. Oct: 23 rows (the 22 countries plus `nan`). CD and UG have `labelled_keys = 0` (DRC is wholly unmatched; Uganda has no raw row) and are kept. |
| 1g | `scen_report` tail, `_with_pooled`, `_report_entries` | PASS | <ul><li>`_with_pooled` uses a one-to-one join and refuses a missing pooled key (`:916-923`).</li><li>Report folds go through `accept_fold`, which only checks partial identity; this is the known F1 from the follow-up and is not new.</li><li>Bootstrap rows are matched non-NaN (`report_fourclass.py:268`).</li></ul> |
| 2 | 24 actual per-country internal gate globals vs G2 | PASS (provenance note F4) | <ul><li>Found exactly 24 records with dict `intensity_k`, all linked to cases through `gate_pairs.csv.gz` `global_sha256` (= sha of `.ubj`). None unlinked.</li><li>Each case has 6 internal origins, and V = U − H holds on every pair.</li><li>Recomputed from the ledger (sha 83402d71…), with the cycle list ordered by first release month: `intensity_k` equals the table count for every one of the 22 countries (h4/2025-06 at O 2025-02 → 1; h4/2025-10 at O 2025-06 → 2; h8/2025-06 at O 2024-10 → 0; h8/2025-10 at O 2025-02 → 1).</li><li>`masked_months[c]` equals hidden(V, k_c), the latest k due cycles at V; for example, V 2022-10 with k = 2 gives {2022-06, 2022-10}.</li><li>`excluded_months` = [] (outer k = 0) and `scenario_k` = 0 throughout.</li><li>Gate dates U for O 2025-02 and 2025-06 end at 2024-10, and for O 2024-10 end at 2024-06. These are the latest six below O.</li><li>No numerical or contract error was found.</li><li>Unavailable provenance: internal gate locals have no saved key lists (already disclosed).</li></ul> |
| 3 | Truth mapping and evaluation output | PASS / VALID-NA | <ul><li>Truth has 4,457 unique (area, 2025-10) rows with class counts 1,201/1,796/1,250/210, matching the candidate document.</li><li>Every truth row joins one frozen key in each Oct case (0 unjoined).</li><li>June VALID-NA, Study2 exclusion counts and the expert route are as in rows 1c-1e.</li><li>`evaluation.json` records `truth_release`, `code`, `runtime` and `outputs` hashes.</li><li>I made no new performance claim. The standalone F1 values were read only to confirm they are computed on the 4,457 keyed rows.</li></ul> |
| 4 | S-row versus F-row weighting | PASS | <ul><li>Eval rows (S/C) have weight 1.0 and a single designated-k row (`availability.py:302,310`). F rows carry the strategy weights: A has 1, B has 1/3 for each of k' = 0/1/2 (`:252-253`).</li><li>`scenario_root`: the root fit uses `w_fit = weight[x_set == 0]` (`main_model_GF.py:509,525-526`). `X_weight` is forwarded only when B is non-unit (`:534`).</li><li>In the partition, weights index only child training ids `ids[0]`/`ids[1]` (`transformation.py:843-844`). S rows enter scoring unweighted, and support counts are deduplicated by `X_key` (`:813-816`).</li><li>This matches G4: F weights are conserved, while S and C are single unweighted observations, and this is the same for A and B.</li></ul> |
| 5 | Tests covering `scen_evaluate` | FINDING F5 (gaps only) | `ScenarioDriverSmoke` (`tests/test_baseline.py:4261-4283`) covers the stale-`frozen_actual` refusal, Oct-evaluable/June-unevaluable and coverage-only country rows. `ScenarioReporting` covers `study_rows` exclusion counts (`:3988,3999`). `ScenarioFinalPath` covers the `no_documented_expert_table` route (`:4054`). |

## 3. Findings

None of these findings changes a reported number in the current outputs. Every check of the actual artifacts matched.

- **F1, evidence-provenance only.** `scripts/run_experiment.py:1018,1069`: `TRUTH_RELEASE_FIELDS` records the crosswalk by **name** only. `scen_evaluate` never hashes the crosswalk, even though release v2 carries `crosswalk_sha256`.
  - Failure scenario: the crosswalk file in REL is changed after approval. Evaluation still accepts it and records no crosswalk hash.
  - The truth bytes themselves stay bound by `truth_sha256`, so scores cannot change this way. Only the mapping provenance becomes unverifiable from `evaluation.json`.
- **F2, required behaviour, low severity, not triggered.** `run_experiment.py:1056-1059` silently drops released truth rows whose (area, target_month) does not match a frozen key, and does not count them. The merge has no `validate=`.
  - Failure scenario: a release with a different `target_month` format (e.g. `2025-10-01`) or areas outside the universe joins nothing. That case is then reported "evaluable: false / no released genuine truth" instead of being refused.
  - Verified not to have happened here: 0 unjoined rows, and truth months are exactly `2025-10`.
- **F3, evidence-provenance only.** `report_fourclass.py:341,269` uses `astype(str)` on country. Two `unmapped_area_global` keys have NaN country (`scenario_actual/h4/2025-10/predictions.csv.gz`), so every country table gets a `nan` row.
  - Here that row has no labels, so the bootstrap blocks are unaffected.
  - Failure scenario: if such keys had truth, `nan` would form a pseudo-country bootstrap block and a country-table row.
- **F4, evidence-provenance only.** For the H8 actual cases, `global_sha256` in `gate_pairs` does not identify a unique record. At V 2022-06, 2022-10, 2023-02, 2023-06 and 2023-10, a k = 0 record (h8/2025-06) and a k = 1 record (h8/2025-10) have identical `features_sha256`, `labels_sha256`, `fit_keys_sha256` and booster bytes.
  - This is legitimate. Under strategy A with k = 1, hidden(V, 1) = {V}, which lies outside the fitting window [V−59, V). Fitting-row features (origins ≤ V−8) cannot reference V, so the fitting inputs are identical. The masks differ only in the gate prediction rows at V.
  - Linking a gate pair to its record therefore needs the expected k in addition to the sha. After that disambiguation, all 10 resolve uniquely and correctly.
- **F5, test gaps only, no code defect asserted.** No test exercises these `scen_evaluate` behaviours:
  - the refusals for `approved` not true, missing release fields, a `truth_sha256` mismatch, duplicate truth keys or codes outside 0..3, a missing `--truth-release`, and a rerun (FileExistsError);
  - a value-level check that the joined `truth_code` equals the release with NaN elsewhere (no zero fill);
  - Study2 `excluded_missing_origin` at the `scen_evaluate` level when origin truth is absent;
  - the `truth_release` fields recorded in `evaluation.json`;
  - the F1 crosswalk hash and the F2 unjoined-row count, which no test covers because the code has neither.

## 4. Remaining gaps

- The internal gate local models have no saved per-key lists, so their fit keys can only be checked through support counts (already disclosed).
- The truth mapping is exact name equality. No geometric or boundary continuity check was done (disclosed in the release).
- DRC, 1,093 unmatched rows overall, and Uganda without a raw October row remain coverage-only.
- June truth and same-horizon expert tables are absent. These are valid NA routes, not failures and not a PASS.
- The section 5 synthetic-check boxes in implement.md were not re-verified here. I reused the test evidence and did not rerun tests.
- The test gaps in F5 remain.

## 5. Statement

This is a bounded trellis-check sub-agent report. It is **not** the independent spot audit or close audit and **not** a task close. Nothing was fixed. Findings F1-F5 are for the executor to classify; none changes a reported number.
