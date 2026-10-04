# Confirmatory pooled-vs-persistence onset forecasting under publication interruptions

Status: PLANNING COMPLETE, pending approval of the final summary. No `trellis-audit` registration for this task (user instruction, 2026-10-03). Grilling decisions D1–D12 were made by the user through structured questions on 2026-10-03.

## Goal

Test, as a pre-registered confirmatory study on a genuinely blind window, the hypothesis that came out of task 10-02:

> During FEWS NET CS publication interruptions, a pooled (non-partitioned) global model predicts crisis **onset** (areas non-crisis at the origin) better than carrying forward the last observed label, and better than simple history-based models.

The claim excludes spatial partitioning; in 10-02 the partition added nothing over pooled. The result informs whether a covariate-based fallback has evidence beyond the 10-02 exploratory findings. It is not a policy recommendation.

## Background

- **10-02 historical evidence** (archived at `.trellis/tasks/archive/2026-10/10-02-exogenous-transition-forecast-design`).
  - Simulated 1–2-cycle outages, 2021–2024, Study 2 (exact-origin non-crisis keys), system − persistence crisis F1:
    - H4 k1 +.099 [+.044, +.148]; H4 k2 +.021 [−.004, +.044];
    - H8 k1 +.027 [−.024, +.086]; H8 k2 +.062 [+.018, +.109].
  - Absolute F1 is about .33–.41.
  - Study 1 has no positive CI excluding 0. The system is ≈ pooled.
  - These results have been seen and are hypothesis-generating only.
- **Data facts** (inspection on 2026-10-03; metadata and covariates only; no 2026 CS values read):
  - **Blind window.** The raw `Outcome/FEWSNET_IPC/2025_2026_FEWSNET.csv` (sha a64ed4bb…) has CS only for 2025-10 (5,573 rows; already scored in 10-02) and **2026-02**.
    - Feb 2026: 5,867 rows, 27 countries; Collected 5,844, Not Projected 20, Not Available 3.
    - Six countries are outside the panel universe: CAR, Lebanon, El Salvador, Honduras, Venezuela, Syria.
  - **Onset at H4.** The H4 forecast of 2026-02 has origin 2025-10, where genuine Oct-2025 truth exists (10-02 truth release v2: 4,457 keys, 20 countries), so an exact-origin onset risk set exists. 5,458 unit names appear in both Oct 2025 and Feb 2026.
  - **No onset at H8.** The H8 origin 2025-06 has no CS, so there is no exact-origin onset risk set at H8.
  - **Feb-2026 crosswalk feasibility** (exact full name against FEWSNET.csv history): 4,365 one-to-one, 1 ambiguous, 1,501 unmatched.
  - **Covariates.** All 19 ACLED sources are missing for every area at 2025-09/10; FLDAS (2) and static (28) are complete; annual GDP/CC are read from 2023/2024 reference rows of the pinned panel.
  - **Combined-panel duplicate.** Admin 2996 is duplicated at 2025-10 and 2026-02. The two 2025-10 rows are identical on all 51 admitted sources and differ only in the excluded Tair/Rainf z-scores.

## Requirements

Decision index (user, 2026-10-03):
- D1 primary S3 scenario
- D2 frozen-G1 pooled model
- D3 ACLED excluded
- D4 comparators
- D5 fixed-sequence H1 → H2
- D6 three outcome categories
- D7 10-02 exact-name cohort/truth rule
- D8 secondary set
- D9 partition comparator on the same inputs
- D10 task-local driver, no product edits
- D11 freeze-then-automatic truth release
- D12 country-stratified transition baseline with fallback

- **R1 Pre-registration.** The estimand, cohort, models, comparators, decision rule and reporting are fixed in prd/design/implement **before** any 2026-02 CS value is read.
- **R2 Scenario (D1).**
  - Primary: H4, T = 2026-02, O = 2025-10, with Oct-2025 CS treated as also missed (simulated). Together with the genuinely missed 2025-02/2025-06 that makes 3 consecutive cycles, and visible CS ends at 2024-10 (SD 2024-06).
  - Gate replay intensity is 3 for every country (G2).
  - The real k = 0 scenario (Oct 2025 visible) is secondary.
- **R3 Model under test (D2, D3).**
  - The pooled global XGBoost with the frozen 10-02 H4 G1 configuration, strategy A and native NaN, refit at O on the lawful label pool [O − 59, O) visible in the scenario.
  - No partition, no tuning.
  - Features: the frozen alignment with the 19 ACLED sources excluded. This is an availability-driven change fixed before evaluation and disclosed.
- **R4 Comparators (D4, D9, D12).**
  - stale persistence (the latest lawful label ≤ O in the scenario);
  - an empirical transition-probability baseline (latest observed class × age bucket {0–4, 5–8, 9–12, > 12 months at T} × country, falling back to class × age when a state has fewer than 30 rows, then to the overall frequencies; argmax);
  - a multinomial IPC-history logistic regression (75 `hist_*` features, median imputation + missing indicators, standardised, L2 C = 1; argmax);
  - the 10-02 partitioned system (frozen map 8965af6d…, strict > 0.01 crisis gate, L1) with **the same inputs as pooled**.

  All share the same fitting-pool rows and prediction rows.
- **R5 Decision rule (D5, D6).** A fixed-sequence test on onset-risk-set crisis F1 differences, each with a country-block bootstrap 95% CI (2,000 draws, seed 42):
  - **H1:** pooled − persistence.
  - **H2** (only if H1 is CONFIRMED): pooled − transition **and** pooled − logistic, both required.

  Per hypothesis: CONFIRMED (lower bound > 0), INCONCLUSIVE (CI includes 0), CONTRADICTED (upper bound < 0), or NA with a reason. Partition − pooled is reported outside the sequence.
- **R6 Cohort and truth (D7).**
  - Feb-2026 truth uses the 10-02 approved exact full-name + country + canonical DBF-name one-to-one rule, with a genuine phase 1–5 and class = min(phase, 4) − 1. Ambiguous names are excluded and geometric continuity is not certified.
  - Origin truth: 10-02 Oct-2025 release v2.
  - Onset risk set = Oct-2025 class < 2 ∩ genuine Feb-2026 truth ∩ panel countries.
  - All exclusions are counted and nothing is filled.
- **R7 Secondary (D8)**, reported only:
  - Study 1 (all keys with Feb-2026 truth) in S3;
  - all comparisons in S0 (onset-set persistence F1 = 0 labelled mechanical);
  - descriptive per-country onset F1 differences with counts (no tests).
- **R8 Implementation (D10).**
  - A task-local driver imports the existing package modules. Product code stays byte-identical (identity fd25e2f7…).
  - It writes to a fresh external run directory (`C:\Users\swl00\geoxgb_runs\confirm-onset-v1`) and never to the 10-02 run.
  - Covariate extension v3 = pinned 2024-12 + combined 2025-01..10, with the identical-on-admitted-sources 2996 duplicate deduplicated (first kept, disclosed).
- **R9 Blinding (D11).** Every prediction is written and hashed (`predictions_frozen.json`) before the Feb-2026 truth is built. The truth is then built automatically by the pre-registered rule, with no manual approval gate, and is bound to the freeze hash.
- **R10 Reporting.**
  - All pre-registered cells are reported, including negative or inconclusive ones.
  - Precision/recall context is given.
  - Disclosures: single cross-section; about 20 country blocks; reconstructed availability; simulated S3 outage that differs from the historical k = 1/2; ACLED exclusion; mapping limitation; prior exposure to the 10-02 historical results.

## Acceptance criteria

- **AC1.** Before any 2026-02 CS value is read: prd/design/implement are on disk, and `inputs_manifest.json` plus `predictions_frozen.json` exist, the latter hashing every prediction file for all five models/comparators × two scenarios.
- **AC2.** Product code identity is unchanged (fd25e2f7…). The driver refuses to overwrite and refuses out-of-order steps. Alignment v3 and extension v3 are verified through the real loader path (static at 2025-10, monthly at 2025-09, ACLED absent).
- **AC3.** The scenario inputs are verified:
  - S3: persistence ≤ 2024-10 (SD 2024-06), gate_k = 3 for 22 countries;
  - S0: Oct-2025 visible, gate_k = 0.
- **AC4.** The Feb-2026 release binds the freeze hash and records the source hashes and every exclusion count. Onset risk-set size, onset count and per-country counts are reported.
- **AC5.** `evaluation.json` gives the H1 and H2 outcome categories with point estimates and CIs, the partition comparison, and all secondary cells. An independent recount of the primary confusion counts and F1 from the keyed rows matches.
- **AC6.** `research/results.md` states the pre-registered outcome honestly (INCONCLUSIVE is not refutation; no success claim beyond the rule) with every required disclosure and an artifact hash manifest.

## Out of scope

- Spatial partitioning as a claim; it is only re-tested as a reported comparator.
- Any hyperparameter search or model selection on 2026 data.
- H8 onset (no origin truth).
- New source research beyond the existing local files.
- The six countries outside the panel universe.
- Any modification of 10-02 artifacts or product code.
