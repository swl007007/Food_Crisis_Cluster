# Final report: food-crisis fallback forecasting during FEWS NET interruptions (task 10-02)

2026-10-03, Claude executor. This is the faithful finite execution of the approved spec. It is **not** the independent spot/close audit (AC7 pending), and nothing here was retuned from evaluation scores.

## Scientific summary

**Design.**
- GeoXGBoost three-stage pipeline (Stage 1 partitions → Stage 2 crisis-F1 consensus maps → Stage 3 gated L1 locals), evaluated under simulated interruptions of k = 0/1/2 missed CS publication cycles.
- Two training strategies: A (normal inputs) and B (variant-augmented, w/3).
- Primary metric: pooled crisis F1 (IPC ≥ 3) against matched persistence.

**Development (2018–2020; 648 Stage 1 candidates, 72 folds).**
- **A was selected at both H4 and H8** as the only qualifier. B failed the normal-parity screen (Δ ≥ −0.02) at both horizons:
  - H4 A parity −.0133, B −.0413;
  - H8 A +.0050, B −.0495.
- Within-period partition gains over the root did not transfer to the out-of-time target (E3 median gain ≈ 0; stage1-diagnostics.md).

**Historical 2021–2024 (fixed recipe, 57 folds).**
- **No Study 1 cell has a positive Δ-vs-persistence CI that excludes zero.**
- H8 k0 is significantly worse: Δ −.0463 [−.0961, −.0015].
- H4 k1 is the only positive estimate: +.0073, with a CI that includes 0.
- Local models add about 0 over the pooled global: Δ −.0005 to +.0016.
- Study 2 at k ≥ 1 (matched): Δ CIs exclude 0 only at H4 k1 and H8 k2. The k0 Study 2 persistence F1 = 0 is mechanical.
- Details: historical-results.md.

**Actual 2025 (frozen recipe; user-attested reconstructed availability, 0/1/2 missed cycles).**
- **October 2025:** 4,457 truth keys from 20 countries.
  - H4: model .7815 vs persistence .7879, Δ −.0064 [−.0179, +.0003]; descriptively meets −0.02.
  - H8: .7230 vs .7879, Δ −.0649 [−.1412, +.0228]; does not meet −0.02.
  - CIs are not equivalence proofs. Local models do not beat the pooled global.
- **Valid NA routes:**
  - October Study 2: 0 eligible, because there is no exact-origin truth at the 2025 origins.
  - June 2025: unevaluable, with no truth source.
  - Expert comparator: no documented table.

  These are faithful outputs, not success claims for experts or onsets.
- Details: actual-results.md.

**Conclusion (evidence, not recommendation).** Under the predeclared contract, in the primary **Study 1** comparison the fallback model did not outperform persistence at any horizon or scenario with interval support. In Study 2 (onset keys, matched, k ≥ 1) some cells have positive intervals: historical H4 k1 and H8 k2. These are reported, not generalised. At H4 it stays within a −0.02 descriptive margin in every historical Study 1 cell (−.016 / +.007 / −.004) and in October 2025 (−.0064). At H8 it is worse, significantly so in the historical normal scenario. In primary Study 1, the geographic partitioning (local models) gave no measurable benefit over the pooled global model. The only CI excluding 0 is a small Study 2 gain at historical H8 k2 (+.0026), already disclosed. These are negative results from a completed experiment.

## Acceptance criteria (PRD AC1–AC7)

| AC | Status | Evidence |
|---|---|---|
| AC1 D7 manifests, no hidden-input leakage, raw sources unchanged | **PASS** | `launch-readiness.md`; pre-fit reconcile (no problems); independent global fit-key checks (654/654 + 4 actual outer); local/gate mask links; 24 actual internal records match G2 (trellis-check-final); raw-source hashes unchanged before/after assembly |
| AC2 role isolation, weights, support, prefixes | **PASS** | stage1-diagnostics; C-permutation check; 278 + 232 outer local key digests; 142 B weight blocks; trellis-check PASS |
| AC3 648/72 ledgers, consensus, exact A/B stop rule | **PASS** | Stage 1 acceptance (648/648, 669 attempts); selection.json c0b3c967…; coordinator selection review |
| AC4 frozen recipe/maps → historical and 2025 predictions; gate > 0.01 independent of −0.02 | **PASS** | frozen.json 2344ab59…; historical 57/57; actual.json c896a583…; gate decisions recounted exactly |
| AC5 saved keyed predictions reproduce Study 1/2/country metrics, comparators, uncertainty, exclusions, June unevaluable | **PASS** | historical_metric_review.json (88d21073…); actual_metric_review.json (4d60d56f…); evaluation.json 8c43375a… (June unevaluable, Study 2 exclusions, expert NA) |
| AC6 preserve negative evidence; report generalisation gaps | **PASS** | historical-results.md; actual-results.md; stage1-diagnostics.md; old lineage retained |
| AC7 approved plan, identity, close only with evidence and an accepted audit | **PENDING** | Plan approved, committed and identity verified (audit run 5c4dede7…). The independent spot/close audit has not run; no close |

## Artifact pointers

- **Task research (final packet, pending commit):**
  - launch-readiness.md, next-phase-readiness.md, stage1-diagnostics.md, dev-ledger-reconciliation.md, historical-results.md, actual-results.md, truth-release-candidate.md;
  - trellis-check-progress / -followup / -final.md;
  - scenario_selection_review.json, scenario_fit_keys_review(_final, _actual_outer).json, historical_metric_review.json, actual_metric_review.json;
  - truth_release_v2/ (approved release.json, candidate summary), truth_release_candidate_summary.json;
  - probes/*: the checkers and their summaries, including actual_postrun_check_summary.json, historical_identity_check_summary.json and local_gate_reconcile_*;
  - launch/: alignment.json, release_ledger.csv and actual_availability.csv (plus its superseded versions).
- **External, hash-pinned:** `research/external_evidence_manifest.json` (49844fcf…) covers 27 files:
  - truth-release v1/v2 (truth 7ccc336f…, crosswalk ab629b01…, release v2 985b5074…);
  - the actual covariate extension (CSV 0d6b78ca…, manifest v2 72a766c6…);
  - selection, frozen, historical, report, actual and evaluation records and the evaluation outputs.

  Raw sources stay outside the repository.
- **Run:** `C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1`, code identity fd25e2f7… (65 files), pinned Windows Python 3.12.10 stack.

## Remaining debt and unresolved provenance (explicit)

1. **Fit budget.** No exact realized fit count exists. The schedule bound of 147,889 applies (Stage 1 ≤ 40,824), plus a crash overhead of ≤ 1,323 from 21 native 0xC0000005 failures. The crash attribution is unknown; all 21 completed on one identical resume.
2. **Internal gate locals.** Evidence is support-only (no saved key list or digest).
3. **24 actual per-country internal gate globals.** Their identity and masks were checked (trellis-check-final), but they are not covered by the independent fit-key digest checker, which assumes an int k.
4. Feature values and booster internals are not verified for any model.
5. **Truth mapping.**
   - Exact-name + country + DBF-name only; **geometric continuity not certified**.
   - DRC (wholly unmatched), Ethiopia (645 unmatched), the other unmatched rows (1,093 in total), Kenya 2995/2996 (excluded) and Uganda (no October row) stay coverage-only.
   - Each country table has 23 rows = 22 countries + one "unknown country / coverage only" row (2 forecast-only keys without a country).
6. **2025 availability.** A reconstruction from user attestation, not verified vintages. SD's latest label is 2024-06. ACLED is natively missing at the 2025 origins (a covariate shift).
7. **Historical availability.** The reconstructed `reference_month_end` convention; revised/latest covariates; D18 map-selection bias; earlier historical exposure.
8. **Generic code/test debt** (frozen code not changed):
   - partial-identity acceptance in scen_select/scen_report;
   - load_extension compares all 69 sources and does not enforce the manifest month range;
   - scen-evaluate checks no crosswalk hash (it is externally pinned) and does not count unmatched truth rows (none occurred);
   - missing negative tests for accept_scenario_stage1 and the scen-evaluate refusals;
   - search limits not retested on the scenario path.

   Details: trellis-check-*.md.
9. The bootstrap uses only 20–22 country blocks; there is no multiplicity adjustment.
