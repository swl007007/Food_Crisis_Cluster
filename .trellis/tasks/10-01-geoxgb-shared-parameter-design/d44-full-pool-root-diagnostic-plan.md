# D44 / A18: zero-fit full-pool vs r80-FIT root diagnostic (completed; not adopted; supervisor verification passed)

2026-10-02. Supervisor decision under the user's delegated Stage 1 research authority. Approved after supervisor review with five corrections (fraction cause, schema identity, exact replay, persistence cohort, full producer SHAs); the planning commit precedes any code or results. Same active task, executor, audit run and base. **No new fits**, G selection, thresholds, seeds, partition maps, Stage 2/3, final-period reads, full 648 or audit close. No adoption or further fits are triggered automatically, whatever the result.

## 1. Question

On the 15 H/T pairs shared by v1's G-screen and D34, how do two roots compare on the same E3 keys?
- the saved v1 **full-pool** global root, fitted on every row of the 59-month window `[O−59, O)`;
- the saved D34 **r80-FIT** root, fitted on the D34 fitting rows only.

"r80" is the nominal split name. The actual FIT fraction is about 75%, because the per-area validation allocation rounds up and enforces a minimum. The S/C split only subdivides the held-out validation rows and leaves FIT unchanged. The fraction is recorded per pair. "Full pool" means the same 59-month window, not a longer history. This describes the two saved roots. It does not attribute the difference to sample size.

## 2. Fixed scope

- **Pairs:** exactly H ∈ {4, 8, 12} × T ∈ {2019-02, 2019-06, 2019-10, 2020-02, 2020-06}, 15 pairs, with O = T − H. G = {4: G1, 8: G4, 12: G2}.
- **Full-pool arm:** the original saved v1 global booster `geoxgb-v1-20261001/globals/h{H}/{G}/O{origin}.ubj` and its `.json` record, plus v1 `gscreen/predictions.csv.gz` rows for that H, G and T as a cross-check.
- **r80-FIT arm:** the saved D34 root booster, from the Brier candidate's `xgb_root.ubj` or the root checkpoint whose sha equals `root.json` `root_booster_sha256`. Also `roots/<root>/root_target_predictions.csv` and `roots/<root>/fold_membership.csv.gz`.
- **Cross-checks:** the verified D37/D38 `rows_E3` exports (`p_original`) may be used where they apply.
- **Persistence:** the exact-origin `hist_phase_o00 − 1` of each E3 row, missing values kept and never backfilled.

## 3. Provenance and identity guards (fail closed)

Every check below stops the run if evidence is missing or mismatched.

1. **Snapshots:** read `snapshot_h{H}.parquet` with a pyarrow filter `target_month <= 2020-12`, taking only the needed columns plus the 162 frozen schema features. Do **not** use the current `Panel` constructor, because it reads the whole parquet. The snapshot sha256 must equal both the v1 record's `snapshot_sha256` and the D34 prepared snapshot. The schema's ordered feature list must equal the frozen ordered list, and the schema FILE sha256 must equal the pinned `51b6f8b21b76a78510522c34e2d1f2a648b7aec768bbcac3dd2318669fa13349` (`FEWSNETGeoXGBExperiment/feature-schema.json`).
2. **Keys:** for each pair, the E3 area set and truth must be identical across the v1 G-screen rows, the D34 `heldout_target` membership and `root_target_predictions.csv`. Each v1 row's origin month must equal O.
3. **Full-pool membership:** rebuild the window keys `[O−59, O)` from the filtered snapshot in (area, month) order. Their `keys_sha` must equal the v1 record's `fit_keys_sha256`, and the row count must equal the record's `rows`.
4. **FIT membership:** the D34 fitting keys must give `root.json` `fitting_keys_sha256`. The D34 legal pool (fitting ∪ validation ∪ confirmation) must be a subset of the v1 window.
5. **Recorded differences:** for each pair, record the rows and areas in the window but not in the D34 pool (the Stage 1 target-area restriction; areas absent from E3), and the actual FIT fraction (FIT rows / window rows and / legal-pool rows).
6. **Models:**
   - v1: params and rounds equal `plan.booster_params(G_CONFIGS[G])`, i.e. `XGB_BASE` + G; the record's `booster_sha256` equals the UBJ sha.
   - D34: root UBJ sha equals `root.json` `root_booster_sha256`; root-fit params and rounds are the same.
   - Record both the v1 record runtime and the current environment (Python, numpy, pandas, xgboost).
7. **Raw replay:** for every one of the 30 saved models, call `xgb.Booster.predict` on a fresh `xgb.DMatrix` of the exact E3 matrix, with ±inf converted to NaN and NaN treated as missing. Require EXACT equality, at float32, between the replayed probabilities and the saved ones (v1 G-screen `p_*`; D34 `p_pooled_*`) after round-trip CSV parsing, as in the earlier verified runs. Labels must also be exactly equal. No tolerance is applied: if exact equality fails, the run stops for diagnosis and is never silently relaxed.
8. **Fit APIs excluded:** the script must not call `xgb.train`, `fit_global`, `continue_booster`, `run_candidate` or any fit or continuation API, and must not import production runners that could fit. A static guard in the script asserts this.

## 4. Metrics and reporting

**Arms:** `full_pool_root` and `r80_fit_root`, plus persistence on the same keys.

**Cohorts:**
- all E3 keys;
- the exact-persistence-available cohort, using identical keys for every arm.

**Per arm and cohort:**
- n, TP, FP, FN;
- crisis-positive F1: four-class argmax, then code ≥ 2;
- fixed-four macro-F1;
- crisis Brier: the crisis probability is summed in float64 and the squared error taken against crisis truth;
- one-hot persistence Brier, labelled as a reference.

Persistence is undefined where the exact origin is absent. It is reported only on the persistence-available cohort, never filled and never presented as an all-key result on a smaller denominator.

**Levels:**
- per pair;
- per H;
- pooled over all 15 pairs (summed confusion, row-weighted Brier);
- mean of per-pair differences, labelled separately from pooled.

**Paired, full-pool minus r80-FIT:**
- F1, macro-F1 and Brier deltas;
- changed binary crisis decisions, split into corrected and spoiled;
- TP and FP deltas.

**Not allowed:** arbitrary subsets, replicate-seed claims, significance or independence claims (these are reused development cases), and comparisons with the D42/D43 subsets.

## 5. Confounds to state

- **G selection:** G was chosen on development folds that include these cases (D24/D26). Both arms share it, but the scores are conditional on that choice.
- **Area pool:** the v1 window includes rows from a few areas the Stage 1 pool excludes. Counts are recorded; none of these areas are in E3.
- **Stochastic path:** G uses `subsample = colsample_bytree = 0.8` with seed 42. Fits on different row sets follow different random paths, so each pair's difference mixes data-size or composition effects with fit-to-fit noise. No replicates exist, and none are fitted.
- **Code vintage:** v1 boosters came from git head `268c17b960912b68b5a27c2877fbe05e6857906d` (code sha256 `bfe559236b415f27e1f9c69e316b4a3b498805c0596faa2758f6dfb87998afa0`, 52 files). D34 roots came from producer `7b2bf6fe482d0a77a664f3627e493934d976696d` (code sha256 `cad8439de1d1c838ddd8482c6782f4cfa205e5c28670e45dad2ddbe92e69421b`). Both are taken from the prepared identity records and are scored only by raw replay of their saved bytes.
- **Not a partition result:** this compares two roots. It says nothing about map quality.

## 6. Implementation and evidence

- One external diagnostic script under `C:\Users\swl00\geoxgb_runs\`. It may reuse already verified read/rebuild helpers only if they can neither fit nor read final labels; otherwise it uses pandas/pyarrow with raw XGBoost predict and its own metric code. No production code changes, new framework or new test suite.
- Outputs go to a fresh external directory: bulky keyed joined E3 rows (per pair, both arms, persistence, truth) recorded with their SHA, plus small metrics and identity JSON.
- After the supervisor verifies, the script and small JSON are copied to task research. The bulky rows stay external.
- The executor may run one native read-only scoped check within the 10-minute subagent ceiling. The supervisor checks the results independently.

## 7. Order

Review this plan → align pointers and commit the planning (GitNexus attempt; known LadybugDB failure recorded) → write the external script → run it (30 replays, 0 fits) → supervisor independent check → persist the small evidence → supervisor synthesis. Stop afterwards; nothing is triggered automatically.

## 8. Run, verification and decision (2026-10-02)

**Pre-run review.** A native read-only check of the external script found no material issue. The supervisor found that `hist_phase_o00` was requested twice in the column selection; the duplicate was removed and checks added (it is in the features, no duplicate columns, E3 matrices exactly n × 162). The check's cheap minor items were also applied: uniqueness and month checks, `root.json` pair-field checks, the one-hot persistence-Brier label, and the G-screen predictions, D34 checkpoint and D34 runtime in the identity record.

**Run.** `C:\Users\swl00\geoxgb_runs\d44_full_pool_root_diagnostic.py` (sha256 `f09fb6e7…`) on the frozen Windows Python 3.12.10 / numpy 2.2.6 / pandas 2.2.3 / XGBoost 3.0.0. Exit 0 in 11 s.
- All 30 saved models (both arms × 15 pairs) were replayed with exact float32 equality to the saved probabilities after round-trip parsing, with labels equal. Zero fits.
- All identity, key, pool and truth checks passed.
- FIT/full-window fractions .7485–.7983. The full window has 15–45 extra rows from 3–4 areas per pair, none in E3.
- Output: `C:\Users\swl00\geoxgb_runs\geoxgb-d44-full-pool-root-20261002`. Hashes: `summary.json` sha256 `2a8dafc680dff46c7921ac52ec35639a3a7de5f9438450d8d7f053f3780fb6bc`, `identity.json` `b49abc8c…`, `rows_E3.csv.gz` `9e9f612a…` (kept external).

**Supervisor verification (PASS).**
- The supervisor independently recomputed all 15 original-file comparisons on both cohorts, all fitting digests and pool counts, and per-pair/H/pooled/mean metrics, and raw-replayed 6 models (both arms at T2019-06 for each H): `research/d44_supervisor_check.py` → `research/d44_supervisor_results.json`.
- A separate comparison reconciled all joined probabilities and every reported metric with no discrepancy: `research/d44_compare_checks.py` → `research/d44_comparison_results.json`.
- The executor's script replayed all 30 models; the supervisor independently replayed 6 of them. Assertion counts are not independent statistical tests.
- The initial supervisor script failed on a wrong checkpoint directory and stopped before writing results; it was corrected and rerun.
- Findings: [research/d44-full-pool-root-findings.md](research/d44-full-pool-root-findings.md).

**Supervisor decision.** Do not adopt a full-pool root replacement on the strength of D44.
- Matched F1: r80-FIT .566612, full-pool .562348, persistence .592605.
- On this 15-pair subset the H8 r80-FIT root exceeds persistence.
- Full-pool improves 7/15 pairs on the matched cohort and 8/15 on all keys.
- There is no causal overfitting claim and no ratio or seed grid.
- Stage 1 remains unresolved.
