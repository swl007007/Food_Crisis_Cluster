# D45 / A19: saved-root boosting-prefix learning-curve diagnostic (completed; no prefix policy adopted; supervisor verification passed)

2026-10-02. Supervisor decision under the user's delegated Stage 1 research authority. Approved after supervisor review with three corrections (equivalence wording, explicit targets, persistence metrics); the planning commit precedes any code or results. Same active task, executor, audit run and base.
- **Zero fits:** no new or extra models, no training checkpoints, G/threshold/window/seed/ratio grids, best-round selection or early-stopping policy.
- **Out of scope:** SHAP or other diagnostics, Stage 2/3, final-period reads, full 648 and audit close.
- No default changes and no new fits follow, whatever the result.

## 1. Question

For each saved D34 root, how do in-sample (FIT), in-window historical random holdout (C) and forward target (E3) results change along the exact prefix of its fitted boosting path, at a quarter, half and all of its existing rounds?
- **What it is:** the exact prefix of this one fitted path. Equivalence to separately trained shorter models is not established or claimed.
- **What it is not:** truncation is not scalar shrinkage toward uniform, because individual trees can push in opposite directions.
- **How to read it:** the losses help interpret capacity; crisis F1 remains the scientific target. No result is causal proof of overfitting.

## 2. Frozen scope

- **Roots:** all 21 D34 pairs, H ∈ {4, 8, 12} × T ∈ {2018-06, 2018-10, 2019-02, 2019-06, 2019-10, 2020-02, 2020-06}, with the saved root booster from the Brier candidate checkpoint `xgb_root.ubj`, whose sha equals `root.json` `root_booster_sha256`.
- **Prefix rounds,** using `predict(iteration_range=(0, r))` with r ≥ 1 only. `(0, 0)` means all rounds in XGBoost and is never used.

  | Horizon | G | Rounds |
  |---|---|---|
  | H4 | G1 | 50 / 100 / 200 |
  | H8 | G4 | 100 / 200 / 400 |
  | H12 | G2 | 100 / 200 / 400 |

  Each round is a group of 4 trees, one per class (`multi_strategy=one_output_per_tree`, `num_parallel_tree=1`), and class groups are never cut. The total is 63 prefix evaluations over {FIT, C, E3}, i.e. 189 part predictions, with zero fits.
- **Base score:** the saved uniform `base_score` stays as is; `base_margin` is never supplied.

## 3. Identity and replay gates (fail closed)

1. The prepared identity matches D44's: D34 git head `7b2bf6fe482d0a77a664f3627e493934d976696d`, code sha `cad8439de1d1c838ddd8482c6782f4cfa205e5c28670e45dad2ddbe92e69421b`. The schema file sha256 is `51b6f8b21b76a78510522c34e2d1f2a648b7aec768bbcac3dd2318669fa13349`, with 162 ordered features. The runtime matches the pinned frozen Windows Python 3.12.10 / numpy 2.2.6 / pandas 2.2.3 / xgboost 3.0.0.
2. Snapshots are read with a pyarrow filter `target_month <= 2020-12` and only the needed columns, with no duplicate selection. The snapshot sha equals the D34 prepared one.
3. **Membership:** FIT, C and E3 keys come from each root's `fold_membership.csv.gz`. The FIT `keys_sha` must equal `root.json` `fitting_keys_sha256`. C keys must equal the Brier candidate's `confirmation_predictions.csv.gz` keys, and E3 keys must equal `root_target_predictions.csv`. Truth must agree everywhere, with unique keys, and every matrix must be exactly n × 162.
4. **Model:** the UBJ sha equals `root.json`; params and rounds equal `plan.booster_params(G_CONFIGS[G])`; `num_boosted_rounds()` equals the recorded total; the output has 4 classes.
5. **Full-root replay gate:** before any prefix scoring, the full-model raw replay (fresh DMatrix, ±inf → NaN, missing = NaN) must equal the saved C `p_root_*` and E3 `p_pooled_*` exactly at float32 after round-trip parsing, with argmax equal to the saved `y_root` and `y_pred_pooled_code`. Any mismatch stops the run for diagnosis.
6. **Prefix consistency:** `predict(iteration_range=(0, T))` equals the default full predict exactly.
7. **No fitting:** a static guard rejects any fit or continuation call (`train`, `fit_global`, `continue_booster`, `run_candidate`, `update`, `boost`) and any production-runner import. Only the pure `src.experiment.plan` and `src.metrics.fourclass` modules may be imported.

## 4. Metrics and reporting

**Per part and prefix:**
- n, real label dates, class prevalence;
- four-class negative mean log(p_true), in float64. A non-positive true-class probability or a non-finite loss stops the run; no clipping;
- crisis Brier, with p3 + p4/5 summed in float64;
- argmax-collapse crisis-positive F1 with TP/FP/FN (the target);
- fixed-four macro-F1.

**Levels:**
- per pair;
- per horizon, pooled (summed confusion; row-weighted losses);
- per horizon, mean of per-pair values;
- each labelled with the horizon's absolute round counts.

There is no pooling across horizons as if the rounds were equal, and no grand winner across horizons.

**Differences:** within each horizon and part, quarter minus full and half minus full, pooled and mean-fold, with changed crisis decisions split into corrected and spoiled. FIT/C/E3 gaps are descriptive only.

**E3 cohorts:** all keys, and the exact-origin-persistence-available cohort with persistence scored on the same keys. Persistence is reported only there. FIT and C use all their rows.

**Persistence metrics:** crisis F1, fixed-four macro-F1 and a one-hot crisis Brier (labelled as a reference) only. Four-class log loss is NOT computed for persistence: its one-hot output gives zero true-class probability on every error, so the loss is undefined. No clipping and no pseudo-probabilities.

**Pattern recorded, without a rule:** whether more rounds improve FIT and C while E3 losses or F1 worsen. No significance, adoption or round-selection rule.

## 5. Interpretation limits

- **Development reuse:** E3 targets are exposed development cases, and G was selected on development folds (D24/D26).
- **FIT** is in-sample; each tree saw a subsample.
- **C** is in-window random holdout, largely historical interpolation (D36). It is not forward evidence.
- **Correlation:** the 21 folds overlap and are correlated, so there are no independence claims.
- **Shorter training:** equivalence of a prefix to separately trained shorter models is not established or claimed.

## 6. Implementation and evidence

- **Code:** one external predict-only script under `C:\Users\swl00\geoxgb_runs\`, reusing D44's identity and metric idioms. No production code changes, framework or test suite. A native scoped read-only check before the run, within the 10-minute ceiling.
- **Outputs:** a fresh external directory holding small metrics and identity JSON, plus keyed quarter/half/full C and E3 probabilities kept external with their SHA.
- **FIT probabilities** are scored and then discarded. They can be reproduced from the saved model and the membership keys; FIT provenance is the membership plus checkpoint hashes.
- **Verification:** the supervisor independently recomputes metrics and prefix replays, including FIT by raw replay.
- **Afterwards:** stop after this finite diagnostic.

## 7. Order

Supervisor review → align pointers and commit the planning (GitNexus attempt; LadybugDB failure recorded) → external script → native read-only check → single run → supervisor verification → persist the small evidence → supervisor synthesis.

## 8. Run, verification and decision (2026-10-02)

**Pre-run review.** The native scoped read-only check found one material omission: the snapshot sha was computed but never compared with the D34 prepared record. Before the run, the executor added the outputs-manifest binding (`outputs.json` sha equals `identity.json` `outputs_sha256`) and a per-snapshot comparison with `outputs.json`; only the hash record is read, no ledger file. Minor hardening was also applied: the `src.*` import allowlist, `root.json` row and candidate checks, a decision-semantics note, and extra identity fields.

**Run.** `C:\Users\swl00\geoxgb_runs\d45_root_prefix_diagnostic.py` (sha256 `59f2eac4bed9ed6099405a0568f87dd50089a98a5c55bfe394038d5ec5f0bfa4`) on the frozen Windows Python. Exit 0 in 48 s.
- 21 roots, 63 prefix evaluations, 189 scored part predictions plus the replay gates. Zero fits.
- Every full-root C/E3 replay was exact at float32, and `(0, T)` equalled the default full predict.
- Output: `C:\Users\swl00\geoxgb_runs\geoxgb-d45-root-prefix-20261002`. Hashes: `summary.json` sha256 `2067978d533dd49843c6b3a5cddff0566967f74347bf30c4eddc4344ea341e57`, `identity.json` `ca6f63cc33d68cc7d9e60c6a70b63ada1f536bd565c0d2636afa112242528a61`. The 35.8 MB `rows_C_E3_prefix.csv.gz` (`8b8e59e5…`) stays external, and FIT probabilities were discarded after scoring.

**Supervisor verification (PASS).**
- `research/d45_supervisor_check.py` → `research/d45_supervisor_results.json`: independent `Booster[:r]` slicing for quarter and half and default full prediction, on every FIT/C/E3 row for all 21 roots. It reproduced all 189 role/prefix metric cells and the persistence-matched E3 subsets.
- `research/d45_compare_checks.py` → `research/d45_comparison_results.json`: reconciled per-pair and per-H pooled/mean metrics, differences, support/prevalence and decisions, and raw-replayed every persisted C/E3 prefix probability (321,047 rows), with zero discrepancies. It is bound to the full summary sha above.
- These are numerical checks, not statistical tests.
- Findings: [research/d45-root-prefix-findings.md](research/d45-root-prefix-findings.md).

**Supervisor decision.**
- The diagnostic is complete. No prefix or early-stopping policy is adopted, and no further round grid follows.
- From half to full, all 21 FIT/C pairs improve both proper losses. E3 probability quality often degrades, but H8/H12 crisis F1 increases.
- H4's small benefit at half is insufficient against persistence, and every tested prefix stays below matched persistence at every H.
- Role prevalences differ, so raw FIT–E3 gaps are not causal overfitting estimates.
- Stage 1 remains unresolved.
