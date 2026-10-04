# Design: confirmatory pooled-vs-persistence onset test (task 10-03)

## Boundaries

- **Product code is untouched.** `FEWSNETGeoXGBExperiment/` keeps code identity fd25e2f7… (65 files). A task-local driver in `research/` imports the existing modules:
  - `src.experiment.availability`;
  - `src.experiment.stage3` (ScenarioPanel, GlobalStore, run_fold);
  - `src.model.native_xgb`;
  - `scripts.run_experiment` (`scen_dev_fold`, `load_extension`, keyed persistence);
  - `scripts.report_fourclass` (study rows, `crisis_paired_bootstrap`).

  The only new modelling code is the two simple baselines, written in the driver.
- **Locations.** A fresh external run directory: `C:\Users\swl00\geoxgb_runs\confirm-onset-v1\`. It reuses the read-only 10-02 prepared inputs from `scen-b43ef6a-v1\prepared\`; nothing is written into the 10-02 run. Raw sources stay outside the repository and are hash-pinned.
- **Blinding.** No 2026-02 CS value is read before every prediction file is written and hashed (`predictions_frozen.json`). The Feb-2026 truth is then built automatically by the pre-registered rule, and only then is the evaluation run.
- **No tuning.** Every configuration choice is fixed below. Nothing is selected on 2026 outcomes.

## Inputs (all hash-pinned in `inputs_manifest.json`)

| Input | Source | Use |
|---|---|---|
| Observations and release ledger | 10-02 `prepared/ledgers/observations.csv`, `prepared/manifests/release_ledger.csv` (CS through 2024-10, month-end reconstructed) | labels and history up to 2024-10 |
| Oct-2025 truth | 10-02 truth release v2 `truth_oct2025.csv` (7ccc336f…; 4,457 keys) | (a) origin truth defining the onset risk set (evaluator-only); (b) the S0 secondary scenario only: added as observations, with ledger rows for the 2025-10 cycle (release 2025-10-31, `reconstructed`, per country present) |
| Alignment v3 | 10-02 committed alignment (ca9e9a66…) with the 19 ACLED sources set to `excluded` | features: 129 minus the ACLED columns (exact count recorded at build) |
| Covariate extension v3 | splice: pinned 2024-12 overlap + combined panel 2025-01..2025-10 (keys + 69 sources, original strings) | static features at O = 2025-10, monthly at O − 1 = 2025-09, annual reference rows inside the pinned panel |
| Frozen recipe | 10-02 `scenario_final/frozen.json` (2344ab59…): H4 → strategy A, G1, L1, map 8965af6d6a724ba5d61d | the pooled config (G1, A) and the partition comparator (map, gate, L1) |
| Feb-2026 CS (protected) | raw `2025_2026_FEWSNET.csv` (a64ed4bb…), scenario CS, reporting_date 2026-02 | truth, built after the freeze |

**Extension v3 duplicate rule.** The combined panel has two rows for admin 2996 at 2025-10.
- They are identical on all 51 admitted sources and differ only in the excluded `Tair_zscore`/`Rainf_zscore`, which never enter features. The first occurrence is kept.
- This is recorded in the manifest. Like 10-02 v2, the manifest states that the 2024-12 overlap check holds by construction, and that the combined panel's excluded historical z-scores are not certified.

## Scenarios (forecast: H4, target T = 2026-02, origin O = 2025-10, all 5,718 areas)

| Scenario | Outer information | Gate replay intensity | Role |
|---|---|---|---|
| **S3 (primary)** | the prepared ledger without any 2025 cycle, so visible CS ends at 2024-10 (SD 2024-06): Feb, Jun and Oct 2025 all treated as missed | gate_k = 3 for every country (each internal origin masks its own latest 3 cycles plus outer exclusions, per G2) | decision rule |
| **S0 (secondary)** | the ledger plus the Oct-2025 cycle (genuine, released 2025-10-31), so Oct 2025 is visible | gate_k = 0 | descriptive only |

## Models and comparators (each run in S3 and S0)

All hard decisions use argmax over the four classes. Crisis = class ≥ 2 (IPC ≥ 3).

1. **Pooled (model under test).** The fold's outer global booster: `native_xgb.fit_global` with the frozen G1 parameters, strategy A (unweighted, original keys) and the lawful label pool [O − 59, O) visible in the scenario, with native NaN. Its predictions are the pooled arm of `run_fold`.
2. **Partitioned system (reported comparator).** `scen_dev_fold` with the same inputs, the frozen map 8965af6d…, the strict > 0.01 crisis gate and L1 locals: the 10-02 machinery unchanged apart from the feature set.
3. **Persistence.** The latest lawful CS label ≤ O in the scenario, per area (keyed persistence as in 10-02). Under S3 this is 2024-10 (SD 2024-06), with age recorded.
4. **Transition baseline.**
   - Fitted on the same fitting-pool rows as Pooled. The state is (latest observed class from `hist_latest_observed_phase` → min(phase, 4) − 1, age bucket of `hist_latest_observed_age` + H with buckets 0–4 / 5–8 / 9–12 / > 12 months, country).
   - Empirical class frequencies per state. A state with fewer than 30 rows falls back to (class, age bucket) pooled over countries; a still-empty state falls back to the overall frequencies. Predict argmax.
   - Rows with no history get the "no history" state.
5. **IPC-history logistic.**
   - Multinomial `LogisticRegression` (scikit-learn 1.6.1, lbfgs, L2, C = 1.0, max_iter = 5000) on the 75 `hist_*` features of the same fitting-pool rows.
   - Missing values: training-median imputation plus one missing-indicator per feature; standardised with training mean and SD. Predict argmax.

The fitting pool and the prediction rows are identical across Pooled and the two baselines.

## Evaluation (after truth release)

- **Feb-2026 truth.** The 10-02 builder rule: exact full name + country + canonical DBF name, one-to-one, genuine phase 1–5, Kenya 2995/2996-type ambiguity excluded. A new release directory holds the crosswalk, the truth, `release.json` (binding `predictions_frozen.json`) and the source hashes, including the .dbf.
- **Onset risk set.** Keys with Oct-2025 truth class < 2 (from release v2) and genuine Feb-2026 truth, inside the 22 panel countries.
- **Primary metric.** Crisis F1 on the onset risk set, from pooled confusion counts.
- **Paired differences.**
  - Pooled − persistence (H1);
  - Pooled − transition and Pooled − logistic (H2, only if H1 is CONFIRMED);
  - Pooled − partitioned (reported).
- **Uncertainty.** `crisis_paired_bootstrap`: country blocks, 2,000 draws, seed 42, linear 95%, with undefined draws reported.
- **Outcome category per hypothesis.**
  - CONFIRMED: lower bound > 0.
  - INCONCLUSIVE: the interval includes 0.
  - CONTRADICTED: upper bound < 0.
  - NA, with its reason, if the metric is undefined.
- **Secondary (reported only).**
  - Study 1 (all keys with Feb-2026 truth) in S3;
  - all comparisons in S0 (onset-set persistence F1 = 0 labelled mechanical);
  - per-country descriptive onset F1 differences with counts.
- **Disclosures.**
  - single cross-section;
  - about 20 country blocks;
  - reconstructed availability (month-end convention; S3 is a simulated outage);
  - ACLED excluded by availability;
  - exact-name mapping without geometric certification;
  - exposure to historical 10-02 results (hypothesis source).

## Trade-offs and risks

- **Power.** One cross-section with about 20 blocks gives wide intervals, so INCONCLUSIVE is a likely outcome. That is accepted, and inconclusive is never reported as refutation.
- **S3 differs from the historical scenarios.** S3 is a 3-cycle outage, not the historical k = 1/2, so the effect size is not directly comparable. It is the only blind onset test the data allow.
- **ACLED exclusion.** Pooled differs from the 10-02 recipe by the ACLED exclusion (availability-driven). The partition comparator uses the same inputs, so partition-versus-pooled stays clean.
- **Rollback.** Delete the external run directory; the repository and the 10-02 artifacts are never modified.
