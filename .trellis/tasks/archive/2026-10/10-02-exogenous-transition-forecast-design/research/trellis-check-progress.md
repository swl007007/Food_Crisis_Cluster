# Formal trellis-check (progress checkpoint, read-only)

2026-10-03. Executor: trellis-check sub-agent. This check was read-only. It made no code, test or task-state edits, ran no fit, made no writes to the RUN directory and did not open 2025 outcome values. The check ended early on a coordinator time limit; items it did not reach are marked NOT REVIEWED.

**This is not a whole-task pass, and it is not the independent spot or close audit.** It covers the implementation diff plus the current development-phase evidence. Historical, actual-2025 and truth-evaluation evidence do not exist yet; items that depend on them are INCOMPLETE.

## 1. Scope, code identity and test-evidence reuse

- Base `e843682`, head `ca86bf7` (HEAD = `ca86bf7117eda05f129f467c62af2762b74d394f`). The package working tree is clean (`git status` shows no package changes).
- `git diff --stat e843682 ca86bf7 -- FEWSNETGeoXGBExperiment`: 19 files, 3,590 insertions, 131 deletions (README, main_model_GF, prepare/report/run_experiment/run_stage1/run_stage2, availability, plan, stage3, fourclass_features, metrics/fourclass, GeoRF, native_xgb, train_branch, partition_opt, transformation, acceptance, tests).
- `git diff --stat b43ef6a ca86bf7 -- FEWSNETGeoXGBExperiment`: **empty output (rc=0)**. The package code is identical between b43ef6a and ca86bf7.
- **Test-evidence reuse: accepted.** The test evidence recorded at b43ef6a is reused (PROGRESS.md:273–277):
  - `ScenarioDriverSmoke` 2/2;
  - an independent coordinator run of 163/163 tests;
  - 165 total.

  Caveat: after the independent run, one stale smoke-test output filename was changed. It was verified by source-digest reconstruction, not by a rerun (implement.md section 4). That is evidence provenance only.
- **No tests were rerun.** The review found no concrete failure that would justify a rerun while the historical fit is live.

## 2. Requirements-to-evidence checklist

Status key:
- PASS: code inspection plus existing evidence agree with the contract.
- FINDING: see section 3.
- INCOMPLETE: depends on phases that have not run yet.
- NOT REVIEWED: not reached (time limit).

### Stage 1: roles and weights

| Item | Status | Evidence |
|---|---|---|
| F/S/C/E3 separation; label-blind split on original keys before augmentation | PASS | `app/main_model_GF.py` `scenario_root`: `stage1_split` on eval (original) keys, then `confirmation_split` on validation keys. C rows are never in `data`, so they never enter GeoRF.fit; they are scored only after the `frozen_digest` check. Saved evidence: stage1-diagnostics.md (0 duplicate and 0 cross-role keys over all 648). |
| C diagnostic-only (no fit, search, support, retry, gate) | PASS (code); test coverage indirect | C is used only in `score_confirmation` after the freeze, and a digest re-check raises if the candidate changed. The explicit C-label permutation test exists for rootconf (`tests/test_baseline.py:1361`). The scenario path relies on the same structural exclusion plus the frozen-digest assertion (`tests/test_baseline.py:3692`). |
| B variants at w/3, grouped by original key; A unweighted | PASS | `availability.py:47,239–259`; root fit `main_model_GF.py` (weights only when not unit); children `transformation.py` (`X_weight[ids]`) → `train_branch.py:41–49` → `native_xgb.continue_booster(sample_weight)`. Saved: B 81,979 keys vs 245,937 rows; float32(1/3) weight blocks verified on 142 locals (local_gate_reconcile, sha e24e620e…96ea). |
| Original-key support (fit and child validation) | PASS | `transformation.py` `original()` dedups by `X_key`; `FitPool.support` (`stage3.py:122–127`); root_support on original keys. |
| Shared-root single L1 increment; prefix immutable | PASS | `native_xgb.py` continuation prefix checks are unchanged; root mode applies weights (`XGBmodel.train`). Test `tests/test_baseline.py:3740–3744`. |
| E2 crisis F1 gain > 0 strictly; undefined parent cannot accept | PASS | `partition_opt.py:873–890`; `transformation.py:849–890`. Saved: 10,874 E2 decisions across 648 candidates, 0 `rejected_undefined_parent_metric` (read-only count in this check). |
| Fixed schedule of 648 (A/B × H4/H8 × 9 × k × r × seeds); G1/G4/L1; seed 42; no extra grid | PASS | `plan.py:226–267`; `run_stage1.scenario_entries` refuses any deviation; `scen-b43ef6a-v1.stage1_acceptance.json` sha `dffae64d…37ab7` (recomputed, matches). |
| Hard-F1 q-scan, five levels, 80-round path cap, zero-mass no candidate | PASS (inherited, unchanged) | `plan.py:44` PATH_ROUND_CAP=80; config MAX_DEPTH=6; inherited scan tests. |
| 21 native crashes resumed once under identical config | PASS with disclosure | PROGRESS.md:371–382. A deterministic re-execution, not a seed retry. Attribution unknown. |

### Stage 2: crisis-F1 E4 weights and NA/fallback routing

| Item | Status | Evidence |
|---|---|---|
| w = max(0, logit(clip F) − logit(clip F_root)) on matched E3 crisis F1 | PASS | `run_stage2.py` `crisis_plan_weights`; E3 recounted from saved target rows and checked against candidate counts (`scenario_candidate_row`). |
| Undefined → ineligible with a reason; corrupt/missing → error | PASS | Same functions. Missing `completion.json` or `candidate.json` raises; corrupt ledger rows raise. |
| Distinct no_prior / no_scorable / null_consensus routes | PASS | `build_consensus` and `accept_consensus` route checks. Saved: 66 learned_map + 6 no_prior_candidates (H8 2019-02), which matches the strict pool arithmetic. |
| One map per strategy, pooling H4/H8 and k; common origin-legal pool (k_max=2) in development; final freeze at 2020-12 | PASS | `scenario_map_pool`; `run_experiment.py` `scen_develop` (strict, k_max=2) and `scen_freeze` (non-strict, k_max=0, cutoff 2020-12). Frozen map `8965af6d6a724ba5d61d` for both H; frozen.json sha `2344ab59…573e` bound to selection `c0b3c967…b8b` (both recomputed). |

### Stage 3: local gate, masks and cache

| Item | Status | Evidence |
|---|---|---|
| Crisis F1, strict gain > 1/100 (Fraction), equality fails, undefined → disabled | PASS | `stage3.py:323–365`; `plan.py:75` STAGE3_GAIN=Fraction(1,100). |
| Support floors 500/50/6/2 (fit) and 100/20/3/3 (gate); current outer fit must qualify; unsupported dates keep global rows | PASS | `stage3.py:437–486`; `plan.py:78–80`. Saved: 1,194 decisions and 7,160 gate blocks reconciled (local_gate_reconcile). |
| Latest six lawful U<O (outer mask), V=U−H, independent of the fitting window | PASS | `availability.py:189–194`; `stage3.py:381–386`. dev-ledger-reconciliation 66/66. |
| Internal replay: outer exclusions plus own hidden(V,k); gate truth lawful at the outer cutoff and evaluator-only | PASS | `stage3.py:219–226,433–436`; `availability.fitting_view/prediction_view/lawful_labels`. 396 internal globals reconciled. |
| Fitting window [O−59,O) ∩ released ∩ unmasked; hidden labels never re-enter via features | PASS | `availability.py:184–187,215–237` (per-area history rebuilt per (origin, k′); history is per-area, `fourclass_features.py:356–380`). Independent global fit-key check 373/373 (sha 80577d3f…4bd3). |
| Cache identity bound to strategy, k, gate_k, masks, features, labels, weights | PASS | `stage3.py:168–209,245–285`. |
| Forecast-only rows need no truth | PASS | `stage3.py:211–217`; 21,288 unlabelled development rows retained. |

### Selection

| Item | Status | Evidence |
|---|---|---|
| Normal parity on matched persistence ≥ −0.02; rank by mean k1/k2; ties A; undefined does not qualify; no qualifier stops the H | PASS | `run_experiment.py` `ab_select`. Saved selection: H4 A (−0.0133) and H8 A (+0.0050) qualify; B fails both (−0.0413/−0.0495). Independent review sha 3d48f0c8…b36a. |

### Reporting and bootstrap

| Item | Status | Evidence |
|---|---|---|
| Country-block paired bootstrap: 2,000 draws, seed 42, identical multiplicities, linear 95%; undefined draws counted; CI needs defined points, ≥2 countries and all draws defined | PASS (code) | `report_fourclass.py` `crisis_paired_bootstrap`; DRAWS=2000, SEED=42. |
| Study1/Study2 (exact-origin non-crisis, evaluator-only origin truth, onset recall); country tables over the whole cohort; matched persistence/pooled/expert | PASS (code) | `study_rows`, `onset_recall`, `country_table`, `keyed_expert`; `run_experiment._report_entries`. |
| Real historical report values | INCOMPLETE | scen-historical is running (A/h4/k1 in progress at check time); scen-report has not run. |

### Actual and evaluate

| Item | Status | Evidence |
|---|---|---|
| scen-actual never loads truth; refuses without the availability table or extension | PASS (code) | `scenario_context(development_truth=False)`; `actual_gate_intensity` refusals; extension read via `usecols` covariates only. |
| Truth release bound to frozen actual.json hash plus truth hash; unique 0..3 keys; June unevaluable | PASS (code) | `run_experiment.py` `scen_evaluate`. |
| Real actual predictions/evaluation; availability table; 2025 crosswalk; expert comparators | INCOMPLETE | Blocked on availability facts (implement.md section 3). |

### R1–R6 and AC1–AC7

| Item | Status | Note |
|---|---|---|
| R1 availability/cutoff/masks/no imputation | PASS (historical); INCOMPLETE (2025) | Historical uses the reference_month_end reconstruction (ledger sha 83402d71…f960, disclosed). Native NaN only (`nx.clean`). 2025 ACLED coverage shift: see L1. |
| R2 studies/truth/comparators | PASS (code); INCOMPLETE (values, experts) | No expert table exists; keys keep `no_documented_expert_table`. |
| R3 metrics/selection/uncertainty | PASS (selection); INCOMPLETE (bootstrap values) | |
| R4 temporal separation/augmentation | PASS | Historical calendar H4 10 / H8 9 (57 folds), freeze before historical. |
| R5 Stage 1 search/support | PASS | |
| R6 consensus/72 folds/enablement | PASS (development); INCOMPLETE (historical) | D18 conditional map-selection bias must be disclosed in the final report. |
| AC1 | PASS (historical) / INCOMPLETE (2025) | reconcile.json sha 5e9a9716…c4f1. |
| AC2 | PASS | |
| AC3 | PASS | 648/72 ledgers reconciled. |
| AC4 | INCOMPLETE | Historical and 2025 phases pending. |
| AC5 | INCOMPLETE | |
| AC6 | INCOMPLETE | Negative evidence is preserved so far (stage1-diagnostics, B failure). |
| AC7 | INCOMPLETE | Identity is recorded in PROGRESS; the audit is not run (outside this check). |

### Not reviewed (time limit)

- Line-level review of `prepare_fourclass.write_scenario_inputs` beyond its dataflow.
- `acceptance.accept_fold` internals.
- The full 1,287-line test diff.
- `README` changes.
- `research/probes/*` code bodies, except hashes.

## 3. Findings

None of these findings changes a development-phase result already produced.

1. **`FEWSNETGeoXGBExperiment/src/experiment/stage3.py:253`** (evidence-provenance only; already disclosed). `fit_label_months` records the declared window, not the empirical fitted months.
   - Failure scenario: a reviewer reads the record as proof of mask avoidance.
   - Global keys are closed by the independent checker; reuse that checker on the historical globals.
2. **`stage3.py:445–457`** (evidence-provenance only). Internal gate locals save only `local_sha256` plus support counts, not their `fit_keys_sha256` (the `fit_local` record is discarded).
   - Failure scenario: an internal-local fitting-key error would go undetected by saved-record reconciliation; only support counts and decisions can be checked.
3. **`scripts/run_experiment.py:668–689`** (`load_extension`; required-behaviour, actual-only).
   - The overlap check compares all 69 covariate columns, including the 18 excluded sources.
   - The manifest's `first_month`/`last_month` are required fields but are not enforced.
   - Under the adopted two-source splice the overlap check holds by construction, so it is not evidence of identity.
   - Failure scenario: a manifest whose month range disagrees with the file passes. The external probe (verify_actual_extension.py) is the only real evidence. It is already disclosed; record it as a limitation at scen-actual.
4. **Fit-budget accounting** (evidence-provenance only; PROGRESS.md:415). No realized fit count is recorded against the 147,889 ceiling. The 21 crashed partial attempts sit outside it (≤1,323).
   - Failure scenario: the AC fit-bound claim rests on the schedule bound alone.
   - Countable from the saved candidate `fits` and fold records at completion.

Limitation, not a code defect:
- **L1.** At 2025 origins the 19 ACLED features are natively NaN: all 5,718 areas at O 2025-06, and 1,304 areas at O 2025-02. Fitting never had ACLED missing (next-phase-readiness.md:198–208). This is spec-compliant (no imputation), but it is a training/prediction covariate shift that could affect actual-2025 results. It requires a reviewed disclosure before scen-actual.

## 4. Statement

This checkpoint is not a whole-task pass and not the independent spot or close audit. Historical, actual-2025 and truth-evaluation items remain INCOMPLETE, and some areas were NOT REVIEWED because of the time limit.

---

## Executor addendum: factual corrections (2026-10-03; Claude executor, at coordinator request)

The reviewer's text above is left unchanged. These corrections narrow or qualify specific statements. The checkpoint stays **INCOMPLETE / NOT REVIEWED** where marked. It is not a whole-task pass; no product fix was made and no audit was closed.

1. **Finding 4 (fit count).** The statement that actual fits are "countable from the saved candidate `fits` and fold records" is too strong.
   - Development `fold.json` and Stage 1 `completion.json` carry no fit counter.
   - `scenario_globals/` stores global models only. It omits internal gate locals, deployed locals and the fits of the 21 crashed first attempts, whose partial checkpoints are preserved but not a counter.
   - Stage 1 `candidate.json` does hold a per-candidate `fits.fit_log` for **completed** candidates. That does not cover the crashed partial attempts or the development/historical internal and local fits.
   - The **exact realized total is therefore unavailable**. The evidence that applies is the conservative pre-launch schedule bound of 147,889 (Stage 1 ≤ 40,824), plus a separately reported crash overhead of ≤ 21 × 63 = 1,323 (research/next-phase-readiness.md; PROGRESS). No realized count is claimed.
2. **L1 (ACLED).** "Fitting never had ACLED missing" is not supported by the existing probe.
   - That probe checked only the required 2025 source months against the specified 2024 baseline months (2024-01 and 2024-05: 0 missing).
   - Narrowed statement: in the combined panel, the 19 ACLED sources are missing for 1,304 areas at 2025-01 and for all 5,718 at 2025-05, whereas the corresponding 2024 baseline months are fully present.
   - No claim is made about every training input month.
3. **Test provenance.** "Verified by source-digest reconstruction, not by a rerun" omits a post-fix run.
   - The filename-only smoke-test fix **was** covered by the executor's ScenarioDriverSmoke run after the fix: 2/2 OK in 375.880 s (`/tmp/smoke_final.log`, 2026-10-02 22:34).
   - The coordinator's independent 163/163 run came before that fix. The digest reconstruction bridges the two.
   - The 165-test evidence therefore rests on: 163/163 independent (pre-fix) + 2/2 ScenarioDriverSmoke (post-fix) + digest bridge, with package code unchanged b43ef6a → ca86bf7.
4. **Finding 3 (`load_extension`).** This is a loader-validation limitation for arbitrary input, **not a failure of the chosen extension**.
   - The adopted file was verified independently. The executor's verify_actual_extension.py and the coordinator's stdlib scan confirmed exact keys (40,026 unique, 5,718 × 7 months), the month range 2024-12..2025-06, and string equality of each block with its named source.
   - The by-construction overlap is disclosed in manifest v2 (sha 72a766c6…81d6), and the final disclosure is planned.
   - No product identity change is made.
5. **Evidence state.** Partial historical folds exist (scen-historical is running, launcher PID 1439211). Completed historical and report evidence does not yet exist.
6. **Sub-agent status.** The trellis-check sub-agent (handle a887e4af4dd5052fa) is terminal. It finished on its own after about 526 s, within the 10-minute bound; a wrap-up message was sent at about 7.5 min and no stop was needed.
7. **Follow-up.** The NOT REVIEWED areas (line-level `write_scenario_inputs`, `accept_fold` internals, the test diff, the README, the probe bodies) need a bounded follow-up at the final check. This is not a whole-task PASS.
