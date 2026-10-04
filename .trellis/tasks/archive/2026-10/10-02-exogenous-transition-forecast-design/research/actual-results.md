# Actual 2025 evaluation (frozen predictions vs approved evaluator-only truth release v2)

2026-10-03, Claude executor. Not a task close and not the audit. Nothing was fitted or tuned. The truth was released only after the actual freeze was accepted (`actual.json` c896a583…a84a).

## Run

- **Command:** `python3.12.exe -B scripts/run_experiment.py --run-dir 'C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1' scen-evaluate --truth-release 'C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1.truth-release-v2'`.
  - Run with PYTHONFAULTHANDLER/PYTHONUNBUFFERED, HEAD 482edb4, a clean package tree and code fd25e2f7.
  - rc 0, about 1 s. No expert table (NA route).
- **Release v2:** `release.json` 985b5074…a570, approved by the Codex coordinator under the user-approved spec. Truth 7ccc336f… and crosswalk ab629b01… are byte-identical to v1. The source metadata now cites the consumed `.dbf` (2175dc97…) and qualifies the `.shp` pin (3aba66a6…). v1 is preserved (e80f960d…).
- **Before/after hashes:** all 25 actual/frozen/release files are unchanged.
- **Independent checks:**
  - The coordinator's actual metric recount PASSED, with 0 problems: 22,872 forecast rows, 92 country rows, 24 comparator/bootstrap entries and 8 Study results; no package imports. Byte-exact copies: `research/probes/actual_metric_review.py` (d73f5e03134ed6a73aa9b4e05333087db1100bbef86a653db3d21702bc39edb5) and `research/actual_metric_review.json` (4d60d56fe84207a8574200543566effa2dfc95dae28ad6e6d5964e90db74cdd0).
  - Final bounded trellis-check: `research/trellis-check-final.md` (794c9bee…).
- **Output:** `scenario_evaluation/evaluation.json` sha 8c43375a5f258a105c55540245fc0deb4284f36c550dbb1638a5709a6d79d427, plus `country_h{4,8}_2025-{06,10}.csv` and `keyed_*.csv.gz`.

## Results

October 2025 is a single cross-section.

**Persistence.** The comparator is the latest available lawful CS label per area, no later than 2024-10, from the saved source ages (`probes/actual_postrun_check_summary.json` 41abb0dc…).
- Most areas: 2024-10, age 8 at O 2025-06 (H4) and age 4 at O 2025-02 (H8).
- **SD: 2024-06** (ages 12 / 8).
- Some areas older: Malawi up to 108 / 104 months, Uganda up to 131 / 127, Zimbabwe up to 56 / 52.

No CS was released between 2025-02 and 2025-06, so the per-area comparator is **identical for H4 and H8 on the common October target cohort**.

| Case | Study | Keys with truth (matched) | Model crisis F1 | Persistence F1 | Δ vs persistence [95% CI] | Δ vs pooled [95% CI] | Countries |
|---|---|---|---|---|---|---|---|
| 2025-10, H4 (O 2025-06, k = 2 gate) | Study 1 | 4,457 (4,457) | .7815 | .7879 | −.0064 [−.0179, +.0003] | −.0025 [−.0104, +.0007] | 20 |
| 2025-10, H8 (O 2025-02, k = 1 gate) | Study 1 | 4,457 (4,457) | .7230 | .7879 | −.0649 [−.1412, +.0228] | −.0006 [−.0026, +.0000] | 20 |
| 2025-10, H4/H8 | Study 2 | 0 eligible: all 4,457 excluded for missing exact-origin truth (no released CS at 2025-06/2025-02) | NA | NA | NA (`no eligible rows`) | NA | — |
| 2025-06, H4/H8 | Study 1/2 | 0: **unevaluable**, "no released genuine truth for this target (forecast/coverage only)" | NA | NA | NA | NA | — |

- **Expert comparisons:** every evaluated key has `no_documented_expert_table`, a valid NA route.
- **Negative result:** neither October case has a positive Δ vs persistence. Both CIs include 0 (H4's upper bound is only +.0003). The local models do not beat the pooled global (Δ ≤ 0).
- **Practical parity**, descriptive and using the D4 −0.02 screen as a reference only:
  - H4 Δ −.0064 **meets** Δ ≥ −0.02;
  - H8 Δ −.0649 **does not**.

  The CIs are not equivalence tests: an interval that includes 0 does not prove parity, and H8's interval is wide (−.1412 to +.0228).
- **Coverage:** 4,457 of 5,718 October forecast keys (77.9%), from 20 countries. DRC (all names unmatched) and Uganda (no raw row) contribute no keys; Ethiopia is partial. Each country table has 23 rows: the 22 cohort countries, including coverage-only ones, plus one **"unknown country / coverage only"** row for 2 forecast-only keys that have no country. They are not 23 countries.

## Caveats

- One cross-section; country-block bootstrap over only 20 countries; no multiplicity adjustment.
- Truth comes from an exact-name crosswalk. **Geometric continuity is not certified.** The 44 assistance-flagged published phases are included (approved).
- 2025 availability is a reconstruction from user attestation (0/1/2 missed cycles), not verified vintages. SD persistence is from 2024-06.
- The ACLED features are natively missing at the 2025 origins (all areas at 2025-05; 1,304 areas at 2025-01): a covariate shift relative to fitting.
- D18 map-selection bias applies to the gate replays. Earlier historical exposure is disclosed in historical-results.md.
