# Implementation plan — approved

Status: **approved for execution**, 2026-10-02. User approved the complete final planning summary with “ok implement”. Follow the recorded audit lifecycle before code changes and D7 data readiness before real fitting; protected final outcomes remain isolated. Scientific contracts below are unchanged.

Execution steering, 2026-10-02: user reaffirmed goal mode and made this approved spec/plan's priorities controlling, superseding the older goal text prioritising metric improvement. Complete lawful inputs and engineering checks before the finite development schedule, then freeze, retrospective evaluation, actual predictions and separate final evaluation in section 6 order. The three remaining driver/comparator/evaluator interfaces stay in scope within this dependency order. Synthetic engineering may proceed while D7 is blocked; no tuning beyond the approved plan.

Latest user steering, 2026-10-03: sources have largely been manually verified; stop spending most effort independently corroborating them. Accept existing source identity as user-attested and prioritise continuing the implemented workflow. Stop open-ended web/archive investigations. Section 3 now checks concrete input availability, keys, masks and leakage-relevant temporal semantics needed for the next run; incomplete independent source corroboration alone is not a launch gate. Historical IPC month-end availability remains one explicit scientific convention to resolve directly, without further source hunting. Prepare the existing runner/configuration meanwhile. Later 2025 truth/crosswalk and unavailable expert comparisons do not block unrelated historical development. Keep goal mode active.

Metric priority reaffirmed by user, 2026-10-03: the current PRD R3/design D4/G3–G4 alone govern execution and success. Use pooled crisis F1; screen normal matched persistence delta >= -0.02, rank qualifying A/B by mean k1/k2 F1, exact ties A, stop final release for a horizon with no qualifier. Preserve the separate Stage3 local gain >0.01 and support gates. Do not optimise against the old goal wording or expand tuning to beat persistence/expert/pooled. No additional source tracing.

## 1. Completed planning decisions

- [x] Resolve G1: synchronise simulated missed releases across participating regions; actual 2025 uses verified availability.
- [x] Fix B training weights at w/3 per scenario, total w, without a ratio search.
- [x] Fix primary crisis F1 to pooled equal-weight region/target confusion counts within each H/scenario/study/period.
- [x] Keep country reports descriptive, with support and coverage; retain overall uncertainty and avoid single-date consistency claims.
- [x] Freeze root/local capacities to H4 G1, H8 G4 and L1 for the A/B comparison.
- [x] Freeze Stage3 local enablement to crisis F1 gain strictly >0.01 versus fold-global, with confirmed genuine-support floors and global fallback; latest-six-date construction is fixed in G2, exact dates follow the source ledger.
- [x] Confirm verified-vintage priority, with disclosed release-lag reconstruction where archives are unavailable; exclude features without a defensible availability rule.
- [x] Fix forecast cutoffs at origin-month end, including historical internal origins; disclose nominal month-offset horizons and retain origin-month exclusion from fitting targets.
- [x] Resolve G2/G4 algorithms and internal Stage1 score conventions; converge PRD/design, explicitly deferring source facts to D7.
- [x] Freeze scenario/role/schedule algorithms, evaluation contracts, support, uncertainty and fit bounds; curate real implement.jsonl/check.jsonl entries. Concrete data-dependent keys remain D7 work.
- [x] Confirm G2 historical-gate scenario: same-intensity interruption replay at each internal origin, preserving outer exclusions and the approved local-gain threshold.
- [x] Confirm reuse of original numerical support floors after masks, counting original keys only; see design G4.
- [x] Confirm G3 country-block paired intervals and undefined-metric policy: fixed 2,000 draws/seed42 on saved predictions, crisis-F1 contrasts, no redraw of undefined samples and no CI gate for success; see design G3.
- [x] Confirm Stage1 F/S/C/E3 role reuse: retain existing rootconf separation with C diagnostic-only; the complete fit ceiling is recorded in D7.
- [x] Confirm Stage1 candidate ceiling: A/B × H4/H8 × nine targets × three outer scenarios × two fitting ratios × three split seeds = 648 scheduled candidates, fixed L1/gt0.
- [x] Derive the D7 fit bound:40,824+931*(1+floor(N/50)); compute data-dependent N before launch. New schedule identities cannot bypass old six-root validators.
- [x] Confirm Stage2 pooling: separate A/B general maps, pool horizons/scenarios within each strategy, and use common origin-legal evidence across development scenarios.
- [x] Protect map/recipe provenance with the D2 post-freeze common scenario calendar; verify exact eligible dates from the D7 release ledger before fitting.
- [x] Confirm common historical evaluation calendar: two missed cycles must be after the recipe/map freeze; exclude overlapping early origins by calendar and retain coverage reasons. Exact dates await source-release verification.
- [x] Confirm the 72-fold A/B development comparison on the six2019–2020 target months, using complete three-stage predictions rather than averaged Stage1 E3 scores;2018 remains in candidate production.
- [x] Confirm non-IPC alignment: exact documented source-lag month for monthly variables, native NaN for residual gaps; latest eligible published reference year for annual indicators, with source dates/ages recorded. No lag/missingness grid.
- [x] Confirm E4 reuse on matched E3 crisis F1 versus candidate root, with explicit legitimate-NA eligibility and distinct no-prior/no-scorable/all-zero fallback reasons; missing artifacts remain errors.

## 2. Prepare the approved execution lifecycle

- [x] Present final plan and obtain subsequent execution approval: user “ok implement”, 2026-10-02.
- [x] Commit the approved spec/plan after change-scope verification: d1a4484, whitespace follow-up e843682 (audit base).
- [x] Verify exact registered repository path and actual Claude executor identity in Herdr; active run 5c4dede7bc6e44f0835fbde1ad2f473d, Claude session b0e22913-4f4b-430f-8d19-dee565d26eec / term_65ccaebe9009a8. Reverified on resume; do not rebind/reset the active run. Planning alone does not start an audit run.

## 3. Mandatory data readiness before real fitting

User confirmed “沿用” on 2026-10-03: historical IPC CS is treated as known at the end of its reference month, preserving the original experiment convention. The ledger uses `reference_month_end` and `evidence=reconstructed`; these are assumed availability dates, not verified publication timestamps. No additional source tracing or lag comparison. This resolves the remaining historical-development timing decision; actual-2025 availability and evaluator-only truth retain their separate contracts.

Historical-development readiness (2026-10-03, existing evidence; no new source research):
- the run is `C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1`, prepared at HEAD `ca86bf7`, `accept_prepared` passed with code equal to HEAD, and the prepared outputs sha is `972902bb…5161`;
- committed config: `research/launch/alignment.json` (129 features) and `research/launch/release_ledger.csv` (sha `83402d71…9f960`; `reference_month_end`, `reconstructed`);
- the pre-fit reconcile `scen-b43ef6a-v1.reconcile.json` reports 648 candidates, 108 inputs, 129 features, a certified fit ceiling of 147,889 and no problems;
- see `research/launch-readiness.md` and PROGRESS.

The items below are split into a historical part and an actual-2025 part. Both were completed on 2026-10-03; the independent audit is still pending.

- [x] Historical development: code, the Windows py3.12.10 numerical stack, sources and root/model configuration are pinned by the accepted preparation above. The old incomplete-task lineage is retained.
- [x] Pin the frozen per-horizon maps and recipe at `scen-freeze` (after selection): scenario_final/frozen.json sha 2344ab59…573e, A on map 8965af6d6a724ba5d61d for both H, bound to selection c0b3c967…b8b. Reconciled with §6 (freeze item ticked); reused unchanged by scen-historical and later by scen-actual.
- [x] Actual 2025 prediction artifacts: `scen-actual` ran 2026-10-03 (PID 2504016, 883 s, exit 0, HEAD 482edb4). `scenario_actual/actual.json` sha c896a583…a84a, 4 cases.
  - Post-run check `research/probes/actual_postrun_check_summary.json` (41abb0dc…): 0 problems. Acceptors pass; identities pass (frozen, prepared, table, manifest v2, code/runtime, gate_k 22 × 0/1/2).
  - Truth is all NaN (not loaded). Persistence equals the latest lawful label ≤ O; SD is older (ages 4/8/12).
  - The coordinator accepted the actual freeze. Truth was then released (v2) and evaluated (evaluation.json 8c43375a…).
- [x] Verify 2025 source/administrative mapping and covariate availability using metadata first. Keep final value columns isolated until the evaluation release point.
  - Evidence: metadata-only probes first (next-phase-readiness.md); covariate extension verified; 2025 values read only after the actual freeze was accepted (truth-release-candidate.md).
- [x] Historical development: release rules and vintage status are frozen before fitting. IPC uses the user-confirmed month-end convention, recorded as `reconstructed` (assumed, not verified publication dates). The covariate alignment is committed:
  - 28 static sources;
  - 21 monthly at L=1, including only the two FLDAS raw means; the precomputed climate z-scores are excluded;
  - 2 annual;
  - 18 excluded.

  Revised-value limitations are disclosed in `research/launch-readiness.md`.
- [x] Actual 2025 covariate extension: an explicit two-source splice (pinned 2024-12 + combined 2025-01..06), assembled and verified 2026-10-03. Manifest `C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1.actual-inputs-v1\extension_manifest.v2.json` (sha 72a766c6…81d6; v1 da9b3084…9c9c kept, wording-only supersession). The real `load_extension` → Scaffold → covariate_features path under the frozen alignment matches the expected months. ACLED 2025 gap disclosed (research/next-phase-readiness.md).
- [x] Actual 2025 CS availability table: `research/launch/actual_availability.csv` (sha 185e2c79…4b59; superseded wording v1 67365108…556c kept as `actual_availability.v1_superseded.csv`; 66 rows = 22 countries × origins 2024-10/2025-02/2025-06 with missed service cycles 0/1/2; `evidence=reconstructed`).
  - Sources: the inherited month-end convention (2024-10) and the user attestation '可以确认' of 2026-10-03, given in the coordinator conversation and relayed via Herdr (2025 origins).
  - Ledger reconciliation: the prepared ledger has no CS cycle after 2024-10. SD's last observed cycle is 2024-06 (ages 4/8/12, disclosed; no October label manufactured).
  - Preflight `research/probes/actual_preflight_summary.json` (2a789356…; v1 da5f4301… kept): the code's `actual_gate_intensity` accepts all 22 countries at each origin; extension v2 scaffold through 2025-06; frozen recipe accepted; fresh output; code identity fd25e2f7….
- [x] Specs and modules were read before the slice edits. For the recorded implementation checkpoints (section 4), symbol impact analysis was attempted. Where GitNexus failed (LadybugDB read-only error), the scoped source/caller inspection fallback was used and disclosed in PROGRESS. This is not a claim about every individual edit.
- [x] Historical development D7: the source/schema/environment record, the 648-root schedule, the cycle ledger and the conservative N-based fit ceiling (147,889 total; Stage 1 ≤ 40,824) passed before the real fit.
- [x] Reconcile the saved development and historical fold, role (train/S/C/E3) and internal-gate ledgers against the lawful keys and masks as those runs complete. The schedule and cycle ledger alone do not complete the original exact fold/role/gate/cycle ledger requirement.
  - Done 2026-10-03 (research/dev-ledger-reconciliation.md):
    - development 72/72 and historical 57/57 folds, 0 problems;
    - independent global fit keys 654/654 records;
    - independent outer-local key digests, gate keys/decisions and global mask links;
    - historical identity and inventory 57/57.

    Limit: internal gate locals have support-only evidence (no saved key lists).
- [x] Actual 2025 D7: the source/administrative mapping, the covariate availability and the evaluator-only truth release stay under their own contracts. Source-semantic changes require design review.
  - Evidence: truth release v2 approved under its contract (research/truth_release_v2/release.json, 985b5074…); mapping limitation (exact-name, geometry not certified) recorded; covariate extension manifest v2.

## 4. Implement the smallest compatible change

Reviewed implementation checkpoints:

- `53917be`: weighted native continuation and adapter forwarding; independent 4/4 checks.
- `702888a`: release-aware views, Stage1 648-entry scenario plumbing, original-key support and Stage3 scenario/gate/cache behavior. Independent post-commit `CommittedCode ScenarioStage3 ScenarioStage1`: 15/15 passed; the earlier uncommitted file-count failure is resolved.
- `b43ef6a`: committed the verified slices 5–7 and driver/comparator/evaluator packet described below; 165 tests covered. No real experiment or task closure implied.
- Slices 5–7 and review corrections are verified for an engineering checkpoint: Stage2 crisis E4/NA, exact 72-fold selection, reporting and historical/actual entry points; source/fold identity checks, same-input pooled outputs, country-specific replay and coverage, expert interface and separate truth release. Independent Windows regression: 163/163 passed (99.875s); executor driver smoke: 2/2 passed (375.880s), covering both qualifying and no-qualifier paths. All 165 tests are covered. After the independent run the only Python change was one stale smoke-test output filename, verified by reconstructing the previous source digest. The smoke uses a smaller synthetic development calendar; it is not the real 648-candidate/72-fold experiment.
- User reiterated “continue，包括上面三个”: continue these slices and finish the disclosed engineering gaps (driver-level synthetic validation, lawful expert comparator interface, separate final-truth evaluation input). This does not invent missing source evidence or release protected 2025 outcomes before frozen predictions.

Real historical development is no longer blocked: D7 was resolved for 2018–2024 by the month-end convention above.
- Stage 1 ran on real data: 669 candidate attempts. 21 hit native 0xC0000005 crashes of unknown attribution; all 21 completed in one identical diagnostic resume.
- `accept_scenario_stage1` accepted all 648, saved as `scen-b43ef6a-v1.stage1_acceptance.json` (sha `dffae64d…37ab7`).
- The actual-2025 phases were completed on 2026-10-03 after their inputs were resolved (section 3): availability table, extension, frozen predictions, approved truth release v2 and scen-evaluate.
- Preserve the active audit run; no task closure until the agreed evidence is complete.

- [x] Reuse the existing three-stage experiment; add only the availability/scenario behavior needed by the frozen design. Preserve raw sources and old outputs.
- [x] Rebuild IPC-derived predictors under the as-of boundary; apply the same contract to pooled/local fits and prediction. Preserve origin-specific covariate availability and genuine-label fitting eligibility (synthetic validation; real historical admission via the committed alignment/ledger in section 3; actual-2025 admission via the user-attested availability table and covariate extension v2, section 3).
- [x] Implement A/B training with grouped variants and conserved fitting weight through both global fitting and local continuation; current continuation lacks a weight argument. Compute support from original keys, not expanded arrays. Reuse the existing global weight validation where applicable.
- [x] Select the existing shared-root increment mode explicitly; preserve root-prefix invariance and separate routing parent from the model prefix. Align Stage3 local gate from legacy macro-F1 to crisis F1, requiring gain strictly >0.01 versus fold-global and genuine support, otherwise global fallback.
- [x] Reuse keyed prediction/reporting infrastructure for matched ordinary/prolonged persistence and available experts, Study2 subsets, coverage and country supplements. Interfaces and synthetic paths are verified; real comparator values come from the saved development, historical and actual runs (recounted independently). No actual-2025 expert table exists, so that route is NA.
- [x] Bind both in-memory and disk global/model caches to the lawful input, strategy, scenario, outer exclusion, fitting-key and weight identities. The current GlobalStore memo key (H, origin, G) is insufficient across scenarios.
- [x] Separate forecast keys from evaluator truth so June targets without genuine labels still receive predictions; never fabricate class codes to satisfy the current labelled-only Panel/run_fold interface.

## 5. Verify before real runs

**Status note (2026-10-03, final reconciliation).** The boxes below are ticked on explicit evidence:
- the 165-test evidence at the unchanged code identity fd25e2f7… (no tests rerun);
- three bounded trellis-check reviews (progress, follow-up, final);
- the real-run independent reconciliations and recounts.

The independent spot/close audit has **not** run; AC7 stays pending.

The full synthetic suite above (165 tests at `b43ef6a`) covers the engineering contracts below. That coverage alone was not the formal trellis-check. Three bounded formal trellis-check reviews have since run (research/trellis-check-progress.md, -followup.md, -final.md). The independent audit is still outstanding.

- [x] Minimal synthetic check: changing a hidden recent IPC or future covariate cannot change permitted features/predictions, while changing a permitted older input can.
  - Evidence: 165-test evidence at the unchanged code identity (fd25e2f7…), plus the real-run independent global fit-key and mask checks (654/654, 4 actual outer) and local/gate reconciliation; trellis-check-progress PASS.
- [x] Minimal synthetic check: augmentation variants remain in one split, conserve weight in both root/local training and do not inflate support; the frozen root prefix remains unchanged.
  - Evidence: 165-test evidence; stage1-diagnostics (original-key roles); 142 B local weight blocks = float32(1/3) (local_gate_reconcile); prefix checks in the tests; trellis-check PASS.
- [x] Stage1 role check: S/C are disjoint and exhaust their designated validation pool; permuting C labels cannot alter fitting, search, support eligibility, candidate maps or routes. C scoring occurs only after candidate freeze and cannot trigger retries or filtering.
  - Evidence: stage1-diagnostics (disjoint roles); synthetic C-permutation check PASS (c_permutation_check_summary.json c19924dd…); C scored after freeze (trellis-check).
- [x] Preserve inherited hard-F1 zero-mass no-split behavior and exact E2 comparisons; internal zero conventions cannot become valid reported NA metrics. Verify five-level/80-round search limits separately from the single20-round deployed local increment.
  - Evidence: trellis-check PASS (inherited, unchanged); 10,874 E2 decisions with 0 undefined-parent accepts. The five-level/80-round limits are not retested on the scenario path (accepted via source/config; test debt).
- [x] Calendar/evaluator check: October 2025 fs1/fs2 origins differ; stale projections are not same-horizon experts; missing origin/target truth never becomes a fabricated evaluation row.
  - Evidence: October 2025 origins 2025-06 (H4) and 2025-02 (H8); no expert table (NA route); Study 2 keys lacking exact-origin truth excluded and counted; June unevaluable (evaluation.json 8c43375a…; trellis-check-final).
- [x] Cutoff check: a release after origin-month end is excluded even if its reference month equals the origin; internal historical forecasts obey their own month-end cutoff and the outer information boundary.
  - Evidence: independent reconstructions filter release ≤ origin and match the saved digests and masks (global fit-key checker; local_gate_reconcile; 24 actual internal records in trellis-check-final).
- [x] Gate scenario check: each historical origin masks its own latest k eligible publication cycles plus outer exclusions; outer-lawful validation truth stays evaluator-only. Use the selected A/B strategy and retain fallback rows when local support fails.
  - Evidence: local_gate_reconcile (dev 72, historical 57); 24 actual per-country internal gate globals match G2 (trellis-check-final); evaluator-only gate truth; fallback rows retained.
- [x] Selection check: normal-period 0.02 screen, two-interruption ranking, exact ties and no-qualifier behavior agree with the frozen contract.
  - Evidence: coordinator independent selection review (research/scenario_selection_review.json 3d48f0c8…).
- [x] Reporting check: shared country multiplicities preserve paired keys; zero-event/empty metrics and undefined bootstrap draws produce the specified NA reasons without redraws or zero substitution. Compare a small deterministic confusion-count example with direct row recounting.
  - Evidence: independent historical and actual metric recounts (historical_metric_review.json 88d21073…, actual_metric_review.json 4d60d56f…) recount confusion counts, NA reasons and paired bootstrap directly; trellis-check PASS.
- [x] Stage2 check: matched candidate/root E3 crisis F1 feeds the inherited clipped-logit transform; legitimate NA, valid zero weight and absent/corrupt artifacts retain distinct outcomes; no positive weight yields the specified global fallback.
  - Evidence: trellis-check PASS; stage1-diagnostics (E4 weights recomputed with the existing utility); 66 learned / 6 no_prior routes.
- [x] Local gate check: crisis-F1 gain exactly 0.01 does not enable a local; gain >0.01 enables only with adequate original-key support and lawful matched historical gate rows.
  - Evidence: strict Fraction > 1/100 in code (stage3.py:323–365); every saved decision recounted exactly (1,194 dev + 969 historical regions). No equality case was observed in the data; that case rests on code and tests.
- [x] Reconcile saved predictions with scenario/study counts and comparator coverage. Run relevant existing regression checks for the modules actually changed; no speculative test framework.
  - Evidence: identity/inventory, ledger and metric recounts at every phase; 165-test evidence reused at unchanged code identity (no broad rerun).
- [x] Cache-isolation check: changing strategy/scenario/outer masks cannot return a previously memoized model with incompatible inputs. Forecast-only check: target keys lacking truth produce predictions and coverage, with no accuracy metric or hidden truth dependency.
  - Evidence: trellis-check PASS (cache identity bound to strategy/k/masks/features/weights); forecast-only rows retained (21,288 unlabelled development rows; actual truth all NaN before release).

## 6. Run in the approved order

- [x] Generate the confirmed648 Stage1 candidate schedule over2018–2020 and run the72 complete-pipeline development forecasting folds over2019–2020; preserve train/S/C/E3 diagnostics and report both strategies, including negative results. Internal gate refits count toward the separate fitting budget.
  - Done 2026-10-03:
    - Stage 1: 648/648 accepted (669 attempts, `scen-b43ef6a-v1.stage1_acceptance.json`); diagnostics in `research/stage1-diagnostics.md`.
    - `scen-develop`: 72/72 folds.
    - `scen-select`: `selection.json` sha c0b3c967…b8b, A winner at H4 and H8, B failing normal parity at both; recorded with the negative results in PROGRESS.
    - Three bounded formal trellis-check reviews have since run (progress, follow-up, final). The independent audit is still outstanding.
- [x] Freeze chosen per-horizon recipe/maps and configuration identity before later scores.
  - scenario_final/frozen.json sha 2344ab59…573e: A on map 8965af6d6a724ba5d61d for both horizons; H4 G1/L1, H8 G4/L1; bound to selection c0b3c967…b8b; written before any historical fit.
- [x] Run fixed-recipe 2021–2024 retrospective evaluation; do not retune from its scores.
  - Executed 2026-10-03: 57/57 folds plus scen-report (historical.json 67153c53…, report.json 4c0efad0…). Identity, inventory, fit-key and local/gate reconciliations pass. The coordinator's independent metric recount passes (research/historical_metric_review.json 88d21073…). Results in research/historical-results.md. Nothing was retuned.
  - This covers historical execution and recount only, not a whole-task or audit pass.
- [x] Generate 2025 predictions under verified as-of information, then release evaluator-only truth for the predeclared final evaluation. June without genuine truth is coverage/forecast-only.
  - Evidence: scen-actual (actual.json c896a583…) under the user-attested reconstructed availability (not verified vintages); truth release v2 after freeze acceptance; scen-evaluate (evaluation.json 8c43375a…); June forecast/coverage-only.
- [x] Report practical parity and uncertainty honestly; a negative scientific result can be a completed experiment, never a fabricated success.
  - Evidence: research/historical-results.md and research/actual-results.md (negative results, CIs, practical parity descriptive only, caveats).

## 7. Deliver and close

- [x] Update PRD completion evidence and PROGRESS; preserve all keyed artifacts needed for the agreed conclusions.
  - Evidence: PRD completion-evidence addendum; research/final-report.md; external_evidence_manifest.json.
- [ ] Commit relevant deliverables after change-scope checks; bound executor uses trellis-audit close only when authorised work/evidence is complete. A queued audit is not acceptance.

Record exact commands after the minimal runner changes and before launching the frozen schedule; verify the design D7 environment rather than guessing an interpreter. No product code or test files are changed by this planning pass.
