# Planning progress

2026-10-02: User superseded prior task after supervisor meeting; old task closed incomplete in a818560, active audit run administratively closed with backup, no acceptance audit or success claim. Old experiment/code paths retained; D55 real run absent.

New task explicitly requested and created. Phase: brainstorm only. Initial PRD is a discussion record, not a frozen spec. Confirmed: covariate lags allowed; prediction for regions lacking IPC history; no universal baseline dominance requirement. Next user-owned decision: cold-start validation population. Then transition estimand/metrics, country discovery and evidence split. Formal spec and grill follow brainstorming; no implementation or scoring authorised.

Schema inspected: 75 phase-history columns, 87 other candidate columns. Country selection and origin-label roles remain unresolved. Existing no-phase-input versus true spatial cold-start distinction recorded. GitNexus detect_changes attempted; known LadybugDB read-only replay error; fallback review limited to new planning Markdown/JSON files, no product symbols changed.

User suggested two studies: all-period forecasting without IPC history, then transition evaluation versus expert. Recorded as proposed structure, not frozen architecture; shared versus transition-specific q/split learning remains unresolved.

User confirmed the two-study structure; country-level metrics are secondary supplementary results, not primary country selection. Technical validation scope remains to discuss.

Confirmed Study1 no-history region validation: test regions contribute no IPC labels to any fitting, partition/consensus learning, tuning or selection; labels evaluator-only. No later rolling-fold admission within this evaluation. Exact spatial units/split still open. Next decision: Study2 evaluation population and false-alarm accounting.

2026-10-02: User redefined the primary objective as fallback forecasting during expert/recent-IPC interruptions, prioritising February/June 2025 evaluation. This supersedes the never-labelled-region claim and whole-region label exclusion recorded above. Covariate lags, relaxed baseline superiority and supplementary country reporting remain. Older IPC input permission remains unresolved. Study2 risk-set evaluation (0→0 and 0→1) plus separate onset recall accepted. Rewrote PRD/task metadata; retained stable task ID and planning-only state. Recorded prior metadata/code probe: date-versus-issue ambiguity, raw source ending 2024-10, absent vintage evidence, and forward-filled IPC forbidden as evaluation truth. No implementation, fitting, scoring or 2025 outcome inspection. Next question: target months versus interrupted publication/origin months.

2026-10-02: User clarified interest in later affected targets, suggesting October 2025 and asking whether October 2024 supplies its comparable expert. Recorded calendar distinction: October 2025 fs1/fs2 require June/February 2025 origins; October 2024 projections target February/June 2025 and could only be separately labelled stale-expert carry-forward comparators for October. Actual last release remains unverified. Proposed October primary / June supplementary target coverage awaits user decision.

2026-10-02: User adopted October 2025 as primary interruption target and June 2025 as supplementary partial-interruption target. Updated PRD and task description/confirmed decisions; publication coverage and genuine truth remain unverified. Next recommendation recorded as pending: permit genuinely available older IPC predictors with explicit source dates/ages, changing the claim to missing-recent-IPC resilience. Planning only; no fitting/scoring or implementation.

2026-10-02: User accepted older available IPC predictors and explicitly retained the three-stage design and previous results. Updated active input/architecture contracts; superseded blanket phase-history exclusion. Existing results retain original provenance and scope; changed availability requires compatibility checks/recomputation, and previous overfitting findings remain unresolved. Proposed historical development simulations of one/two missed IPC publication cycles, with normal-availability references, await user decision. No fitting/scoring or implementation performed.

2026-10-02: User adopted historical interruption simulations and relaxed prior overfitting/performance gates to approximate parity with persistence. Recorded replacement of strict superiority/positive-CI-lower-bound requirement while retaining independent validation and availability/no-leak constraints. Checked old PRD R3/D26: primary endpoint was crisis-positive F1, not fixed-four macro-F1. Proposed (unconfirmed) parity tolerance of 0.02 absolute F1 per fs1/fs2 and availability scenario, with uncertainty/instability reported rather than requiring universal wins. No old variant automatically adopted, no scoring or fitting.

2026-10-02: User adopted the 0.02 absolute crisis-F1 tolerance. Recorded delta >= -0.02 versus matched persistence, with fs1/fs2 and availability scenarios separate and uncertainty retained; no statistical equivalence claim. Next pending recommendation: Study2 reuses the same frozen recipe/decision rule and matching predictions, changing evaluation cohort only; no dedicated transition q/split or threshold optimisation. Task remains planning; no experiment run.

2026-10-02: User adopted the shared-model Study2 design: same recipe, decision rule and matching predictions, with evaluation-cohort filtering only; no dedicated transition q/split or threshold optimisation. Next pending recommendation: use genuine exact-origin/target labels for Study2, report missing-origin exclusions and retain eligible rows in Study1; masked historical origin truth is evaluator-only. Expert comparisons use available same-horizon matched subsets. No execution authorised or performed.

2026-10-02: User accepted exact-origin truth for Study2 and missing-origin exclusions with eligible rows retained in Study1. Metadata-only source discovery found October 2025 CS records in external 2025_2026_FEWSNET.csv (5,573 raw rows), and 4,481 dated October rows in FEWS_2025.csv; neither has June 2025 rows. No label values or performance inspected. Recorded research/2025-outcome-metadata.md and pending proposal to keep FEWS NET CS as primary labels, with June prediction/coverage-only if genuine truth remains unavailable. Scout unavailable due to thread limit; direct bounded search used. Planning only.

2026-10-02: User adopted FEWS NET Current Situation as primary evaluation labels and June forecast/coverage-only reporting when genuine truth is unavailable. Checked prior temporal contracts (PIPELINE_WORKFLOW:6–8; old experiment-plan:30,89,136). Recorded pending recommendation: 2018–2020 development/partition+consensus selection; frozen-recipe 2021–2024 retrospective validation; 2025 final interruption case. Distinguish frozen recipe/maps from origin-relative 59-month refitting using genuinely available history; disclose prior historical result exposure. No fitting/scoring/implementation performed.

2026-10-02: User adopted the temporal protocol and 59-month as-of refitting. Preserved the earlier accepted four-class probability/binary-crisis evaluation rule rather than reopening training target choice; hard-class collapse and probability sum distinguished. Added conditional October 2025 horizon example (fs1 June cutoff vs fs2 February cutoff). Next pending recommendation: bounded normal-training vs interruption-augmented-training comparison on 2018–2020 development only, with shared folds/sample weight and lawful training labels. No fitting/scoring/implementation performed.

2026-10-02: User adopted the two-strategy training comparison. Recorded training augmentation as confirmed rather than pending, retaining matched development evaluation and genuine sample support. Next pending recommendation: per horizon, require normal-period practical parity and then rank by equal-weight mean F1 across the two interruption scenarios; report each scenario separately, prefer normal training on exact ties, and report unmet criterion if neither qualifies. Ranking weights are separate from still-unspecified training weights. Planning only; no implementation or experiment run.

2026-10-02: User adopted normal-period practical parity followed by interruption-prioritised selection. Wrote design.md and implement.md as explicitly unapproved drafts, preserving confirmed contracts and grouping remaining grill items G1–G4. Task remains planning with execution_authorized=false; no code/fits/scores. Next grill question: service-wide synchronised missed-release simulation versus region-level missingness (G1).

2026-10-02: User accepted G1 service-wide synchronised masking across participating regions, with actual 2025 following verified country/product coverage. Updated PRD, design, implementation checklist and metadata. Next G2 question: fix B fitting weights at w/3 for each normal/one-cycle/two-cycle variant (total w), without ratio search; this mixture remains proposed, not confirmed. No code, fitting or scoring.

2026-10-02: User accepted equal thirds for B training scenarios, conserving original sample weight and independent support. Updated PRD/design/implement/metadata. Separated outer-outage label exclusion from historical predictor augmentation in the still-pending mask specification. Next G3 recommendation: primary F1 from pooled equal-weight region/target confusion counts within each H/scenario/study, with country/month metrics supplementary; no outcome values or scores inspected. Planning only.

2026-10-02: User adopted pooled equal-weight region/target confusion-count aggregation for primary crisis F1, with horizon/scenario/study/period separated and country/month reports supplementary. Updated all planning documents. Next pending G3 recommendation: descriptive country tables for all eligible countries with support/coverage, no new country-specific significance tests or single-date consistency claim; overall uncertainty retained. Undefined-metric handling drafted for final review. No implementation, fitting, scoring or final-label inspection.

2026-10-02: User adopted descriptive country reports without new country-specific significance tests; aggregate uncertainty retained. Inspected old experiment-plan, GeoXGB README and D54 research decision. Recorded research/reuse-starting-point.md: prior H4=G1/H8=G4, L1 capacity, README stale weight/metric assumptions and outstanding code/map compatibility. Proposed freezing these capacities for A/B rather than renewing parameter search; not yet confirmed. No experiments or final-label access.

2026-10-02: User adopted fixed H4 G1/H8 G4/L1 capacities. Reconciled prior D28/D29 shared-root mechanism with source (explicit root mode required). Found global weights supported but local continuation unweighted, raw-row support inflation risk under augmentation, and Stage3 gate still macro-F1. Added implementation/check requirements; no code edited. Next pending decision: retain +0.01 local-versus-global Stage3 gate on crisis F1, distinct from final −0.02 versus persistence. GitNexus failed with known shadow-page error; direct source used. Planning only.

2026-10-02: User answered “启用” to the Stage3 local gate proposal. Confirmed strict crisis-F1 gain >0.01 versus the fold-global on matched origin-legal historical gate keys, with genuine support and global fallback; equality is insufficient. Synced PRD/design/implementation checklist/task metadata. Exact support thresholds and historical gate construction remain open. This answer approves the gate, not implementation; execution_authorized remains false. Next: metadata-only input availability and release-calendar investigation for G2; no fitting, scoring or 2025 outcome-value inspection.

2026-10-02: Metadata-only probe found 2025 standalone panel dates January 2025–April 2026 and normalised combined panel dates January 2010–April 2026; neither header has obvious release/vintage fields. Selected sidecar metadata exposes ambiguous global-versus-grouped climate rolling semantics, requiring producer verification before reuse. Recorded research/input-availability-metadata.md. Pending G2 decision: permit disclosed retrospective release-lag reconstruction when historical vintages are unavailable, while excluding inputs with no defensible availability rule. No outcome values inspected, no feature source adopted, no product code or experiments changed.

2026-10-02: User adopted the G2 availability evidence policy: verified historical vintages where available, otherwise disclosed retrospective reconstruction using documented release lags/conservative cutoffs; omit features lacking defensible availability rules. Synced PRD/design/implement/task metadata and research status. Source-specific lags remain unverified. Planning-only state retained. A bounded read-only calendar scout could not spawn because the agent thread limit persists; used targeted direct source inspection instead.

2026-10-02: Verified inherited fitting bounds in stage3.py:63–64 and gate scheduling in prepare_fourclass.py:355–370: [O−59,O), six genuine U<O dates, V=U−H. These are month-index checks, not release checks. Recorded pending G2 recommendation to fix forecast cutoff at origin-month end, with actual/reconstructed release eligibility and nominal month-offset horizon disclosure. No changed code, fitted models or inspected outcome values. Exact within-month cutoff remains user-owned.

2026-10-02: User adopted end-of-origin-month forecast cutoffs, including internal historical origins. Synced PRD/design/implement/task metadata, retaining publication eligibility and nominal month-offset horizon disclosure. Fitting labels still use [O−59,O); origin-month IPC input permission does not admit origin-month fitting targets. Task remains planning, execution_authorized=false; no model code, experiments or final outcome values accessed.

2026-10-02: Rechecked stage3.py:232–248: historical gate predictions consume the prebuilt panel without an explicit interruption scenario. Recorded pending G2 proposal: gate locals on historical forecasts under the same missed-cycle intensity, at each internal origin's own month-end, while retaining outer information exclusions and lawful gate truth. This changes the gate's information scenario, not its +0.01 threshold or final persistence criterion. Actual 2025 country/product scenario matching still depends on availability verification. No experiment or outcome inspection.

2026-10-02: User adopted same-intensity historical local-gate interruption replay. Synced PRD/design/implement/task metadata; +0.01 gate, outer information exclusions and planning-only state retained. Verified numerical support floors in plan.py:77–81 and gate_decision in stage3.py:155–174; proposed retaining original floors on lawful original keys after masks, not augmented rows. Located existing shared country-block bootstrap in report_fourclass.py:179–214; metric/undefined-draw alignment remains unresolved, no estimator run. Next user decision: retain support floors despite potentially more frequent global fallback. No code, fitting, scoring or final outcomes touched.

2026-10-02: User adopted original numerical support floors for local fitting, Stage1 child validation and Stage3 historical gating. Synced PRD/design/implement/task metadata, counting lawful original keys after masks and retaining evaluation rows on fallback. No all-four-class requirement; floors are engineering minima, not an effective-sample guarantee. Planning-only state preserved. Next G3 issue is aggregate paired uncertainty and undefined-metric reporting; no product code, fitting, scoring or final outcomes accessed.

2026-10-02: Read report_fourclass.py:153–214 and constants at :39–41. Existing resampling uses shared country multiplicities, 2,000 accepted draws, seed42, macro-F1 and retries of empty required cohorts. Drafted unapproved G3 reuse: paired country-block crisis-F1 intervals from 2,000 fixed draws, preserving full country histories; no fitting or country tests. Proposed explicit NA handling, no silent zero substitution/redraw, and CI=NA if point/support/draw definition requirements fail. Disclose fixed-prediction and cross-country-dependence limitations; uncertainty does not become a success gate. Awaiting one user decision on this reporting protocol.

2026-10-02: User adopted G3 paired country-block uncertainty and undefined-metric reporting. Synced PRD/design/implement/task metadata: 2,000 fixed draws, seed42, paired crisis-F1 differences, 95% percentile interval, NA with reasons for undefined metrics/draws, no redraws and no new CI success gate. G3 reporting is confirmed; internal Stage1 score semantics remain separate G4 work. Task remains planning; no numerical experiments, outcome inspection or implementation authorised.

2026-10-02: Read the full old d29-confirmation-plan.md and current rootconf constants. Existing F/S/C/E3 separation preserves a post-search diagnostic while exposing search-versus-target transfer gaps in retained evidence. Added pending G4 recommendation to retain those roles, with C diagnostic-only and no new filter/retraining/threshold. Candidate ratios/seeds/E2 families, scenario handling and total budget remain to freeze; old six-candidate or full-grid schedules are not automatically adopted. No code or experiments changed.

2026-10-02: User adopted Stage1 F/S/C/E3 role separation with C diagnostic-only. Synced PRD/design/implement/task metadata and added a targeted C-label-isolation verification requirement. Stage1 q and split use S only, with C scored after map/model/routing freeze and E3 distinct. Task remains planning, execution_authorized=false; no code edits, model fitting, scores or final outcome access.

2026-10-02: Inspected plan.py ratios/seeds/threshold families and run_stage1.py's hard-coded rootconf schedule validators. Drafted a new finite candidate proposal: nine targets × three availability scenarios × two ratios × three split seeds =162 per strategy/horizon, 648 across A/B and H4/H8, fixed L1 and D29 gt0; no expanded capacity/threshold grid. This is not the old 648 recipe and not a fit-count estimate. Proposed scenario-specific S/C/E3 with outer exclusions retained; downstream map pooling and total fitting bounds remain open. Candidate quantity is not claimed to establish diversity. Await user decision; no runs.

2026-10-02: User adopted the 648-candidate Stage1 schedule (162 per A/B strategy and H4/H8 horizon): nine targets, three information scenarios, r80/r50, split seeds42/43/44, fixed L1/gt0 and model/confirmation seed42. Synced PRD/design/implement/task metadata. No extra grids or seed retries; map diversity and total booster-fit cost remain distinct from scheduled candidate count. Planning only, no experiment execution or final-label access. Next: Stage2 pooling boundary and shared-map design.

2026-10-02: Inspected run_stage2.py:1–119: existing general all-horizon pooling, coverage-matched duplicate/weight diagnostics and separate no-candidate/all-zero/learned routes. Proposed one general map per A/B strategy, pooling H4/H8 and three scenarios (up to324 identities per final map), with common origin-legal candidate pools for shared development maps. Flagged frozen end-2020 map versus early-2021 simulated outage overlap as an unresolved final-calendar feasibility issue; no silent temporal exclusion waiver. Formula alignment, detailed fitting counts and pooling await review. No fits or final-label access.

2026-10-02: User adopted Stage2 pooling: separate A/B general maps, each pooling its own horizons/scenarios with up to324 candidate identities; no scenario/month-specific or post-selection mixed maps. Synced PRD/design/implement/task metadata. Development uses common origin-legal evidence across scenarios; early-2021 frozen-map/outage compatibility remains unresolved. Planning-only state retained. Next: exact inherited E3 weighting and undefined-score/fallback handling.

2026-10-02: Inspected compute_plan_weights at step4_similarity_matrix.py:50–67: positive logit difference, clipping1e−6, legacy macro-F1 names and fail-on-nonfinite behavior. Drafted pending proposal to retain the formula on matched E3 crisis F1 versus each candidate's own root, with no C/baseline-derived or forced-uniform weights. Legitimate undefined scores stay NA/ineligible with reasons; distinguish no-prior, no-scorable and complete-all-zero global fallbacks from missing/corrupt artifact errors. Plan.py still starts final H4/H8 windows in 2021-05/09; these do not by themselves resolve frozen-map versus outage overlap. No scores, fits or code changes.

2026-10-02: User adopted Stage2 E4 crisis-F1 weighting and distinct fallback/NA handling. Synced PRD/design/implement/task metadata and planned the minimal weight/routing check. Formula uses each candidate's own matched E3 root, not C/persistence; no forced uniform weights or scenario rebalance. Task remains planning with execution_authorized=false. Next calendar decision must protect both fixed-map provenance and recipe selection, since a global-only fallback does not erase selection based on hidden historical labels.

2026-10-02: Ran a stdlib calendar-only check (no source outcome reads): under on-time February/June/October releases, an origin February2021 two-cycle outage deletes October2020/February2021 and overlaps the recipe/map information period. Proposed common scenario calendar begins at originJune2021, hence H4 targetOctober2021 and H8 targetFebruary2022, with 10/9 scheduled target months through2024-10. Relative to the existing final windows, one early genuine scheduled target per horizon is excluded. Recorded this as a pending scope refinement with release-ledger verification; final dates may shift for delayed releases. No model fitting/scoring occurred.

2026-10-02: User adopted the historical evaluation boundary: origin and both simulated missed cycles must follow the 2020-12 map/recipe information freeze; same primary target calendar across scenarios per horizon, early exclusions recorded. Synced PRD/design/implement/task metadata. H4 October2021 and H8 February2022 are conditional calendar examples, not verified release dates. No earlier-map refit or global-only workaround; actual release verification still required. Planning-only state retained.

2026-10-02: Reconciled inherited Stage1 mechanics from source: hard-F1 zero exposure/error mass produces no scan candidate; exact helper's internal denominator-zero score is not the approved reporting NA convention. Existing MAX_DEPTH6 means five binary levels, with80 path-selection rounds distinct from deployed20-round L1; conservative Stage1 ceiling648×63=40,824 fits, not a runtime estimate. Located existing DEV_TARGETS=2019/2020 February/June/October. Drafted pending proposal to select A/B on72 complete-pipeline development folds with prior-only maps/gates, while retaining all nine2018–2020 Stage1 targets. No fits or outcome values accessed. A bounded normalisation-producer string search found only the new planning notes in the inspected paths; producer lineage remains unresolved, no source adopted.

2026-10-02: User adopted the72 complete-pipeline development forecasting folds on six2019–2020 target months, distinct from648 Stage1 candidate jobs across2018–2020 and from internal gate fits. Synced PRD/design/implement/task metadata. Selection uses final pipeline predictions with origin-legal maps and legitimate global fallback; no replacement with averaged candidate E3. Task remains planning/execution_authorized=false. No code, model runs or protected outcomes accessed.

2026-10-02: Verified covariate_features uses exact-origin panel values and aggregates without release-lag alignment. Bounded searches of repository spec/feature/preparation paths and the external assembled_FEWSNET directory did not locate a release-rule/normalisation producer. Read external source-directory AGENTS.md and code-source cells only in FEWSNET_feature_set_construct*.ipynb (no notebook outputs/outcome values); these notebooks did not supply availability rules. Drafted pending D1 input-alignment choice: monthly source-family fixed documented lags with residual NaN; annual latest eligible released reference year with age provenance; no ad-hoc carry-back or lag search. Actual source-family rules remain evidence work, not user-guessed parameters.

2026-10-02: User “ok” adopted fixed documented monthly source/product lags with exact source-month alignment/native NaN, and latest eligible annual reference year with provenance. Synced PRD/design/implementation checklist/task metadata. Completed lossless PRD convergence, resolved algorithmic G2/G4 details and fit bounds, and deferred source facts explicitly to mandatory D7 pre-fit readiness. Curated implementation/check context with task-specific exceptions to Ethiopia/RF rules. Task remains planning/execution_authorized=false, awaiting subsequent approval of the complete final planning summary; no product code, fits, scores, final outcome values, commits or executor dispatch.

Validation found design.md exceeded Trellis's32,768-byte per-file injection limit. Moved its G2–G4 section verbatim into evaluation-contract.md and explicitly included both in each context manifest; this is a document-only split, with no scientific change.

Final planning verification: PRD re-read after convergence; original PRD file:line anchors preserved. task.py validate passes both8-entry context manifests without truncation warnings; JSON/path/size and finite-schedule arithmetic checks pass; git diff --check passes. Live task.json confirms planning and execution_authorized=false. Final summary is presented for subsequent user approval; data readiness and scientific performance remain unverified.

2026-10-02: User explicitly approved the complete final summary with “ok implement”. Execution scope is now authorised, conditional on audit start and D7 pre-fit readiness. No extra planning confirmation required. Preparing plan commit and freshly verified Claude executor; status remains planning until the bound executor starts through trellis-audit.

2026-10-02 (execution, Claude executor; audit run 5c4dede7bc6e44f0835fbde1ad2f473d, base e8436827): Started the run with `trellis-audit --repo <exact path> start`; task status in_progress, verified by Codex. Completed a bounded D7 data-readiness pass using metadata/provenance only (`data-readiness.md`; evidence in `research/d7-*.md` and `research/probes/`). No product code, fits, scores or 2025+ IPC value columns were touched.
- PASS: frozen Windows environment (all pinned packages; xgboost.dll hash equals the prior runs); pinned source hashes; historical admin keys.
- BLOCKED for real fitting: (1) no defensible historical IPC CS release rule, because FDW timestamps are database events and the M+1/M+2 proposals are not adopted; this leaves cycle masks, gate dates, the historical calendar, IPC-derived features and ordinary persistence undefined; (2) the October 2025 truth crosswalk is a name join with 1,093 of 5,573 October 2025 CS rows unmatched (only DRC wholly; Ethiopia 645/1,141; corrected after Codex recount — earlier figures mixed in 2026 rows), and FEWS_2025.csv zero-fills unmatched crisis flags; this blocks the dependent 2025 keys only.
- Findings: the climate z-scores in both the pinned and 2025 panels come from a cross-admin rolling window plus full-sample standardisation (producer `assemble_latest_FEWSNET/02_preprocess_and_combine.ipynb` cells 16–17; reproduction match 1.0000, except pinned Rainf 0.9944 because of ±inf rows), so they are excluded. Covariates: 54 candidates and 30 excluded; none certified pending URL/lineage records and schema approval. Provisional pre-mask fit bound ≤147,889 (keys-only N ≤ 5,714). The raw ML1/ML2 validity windows vs the legacy expert join are unresolved.
- Change boundary recorded, including Codex's GlobalStore memo-key and unlabelled-target traps.
- Not implemented: no scenario/A-B pipeline, release ledger, masks, weighted continuation or crisis-F1 gate exists in code yet.

2026-10-02 (engineering slice 1; Codex-authorised; uncommitted): `FEWSNETGeoXGBExperiment/src/model/native_xgb.py`.
- `continue_booster(..., sample_weight=None)` validates weights with the existing `check_sample_weight`, applies them only to the appended rounds, and adds a `weight_record` block only when weights are given.
- `XGBmodel.train(..., sample_weight=None)` forwards them in both root and parent increment modes.
- Defaults are byte-identical to before.
- New `tests/test_baseline.py::WeightedContinuation` (4 tests, synthetic rows only):
  - the default record and bytes equal a direct `xgb.train` continuation;
  - nonuniform weights equal a direct weighted `xgb.train` reference, while parent bytes, prefix structure and prefix margins stay unchanged;
  - malformed weights (length, shape, NaN, 0, negative, inf, float32 overflow) raise before fitting;
  - `XGBmodel.train` forwards weights in both modes.
- Frozen Windows Python 3.12.10: pre-edit focused baseline 18/18 OK; post-edit focused 22/22 OK; full suite 120/120 OK.
- GitNexus `impact` and `detect_changes` failed with the known LadybugDB shadow-page error; callers were inspected directly: `stage3.fit_local`, `stage1_global_increment.py`, `stage1_map_transfer.py`, `XGBmodel.train`; risk low, additive keyword.
- Still missing end to end: `stage3.fit_local`/global-store weight forwarding and weight-aware cache identity; original-key support counting; release-eligibility ledger and cycle masks; scenario/B-variant construction; the crisis-F1 Stage 3 gate; the new 648/72 schedules; the prediction cohort for unlabelled targets. The A/B pipeline is **not** implemented, and no real data has been run.

2026-10-02 (Codex review): inspected the weighted-continuation diff and independently ran frozen Windows Python `tests/test_baseline.py WeightedContinuation`:4/4 PASS. The dotted `-m unittest tests.test_baseline.WeightedContinuation` invocation failed because tests is not an importable package; direct script invocation is the working command. No real data/final values used in these checks. Authorised the next finite synthetic engineering slice: release/cycle-aware feature construction and grouped A/B fitting views. Real fitting remains blocked by D7; task stays in_progress.

2026-10-02 user clarification (via Codex): FEWSNET.csv is a directly downloaded original source ("这个是直接下载的原文件"), not a local generated file. Its supplied column definitions are its provenance; there is no local producer search. This does not establish historical release dates or prove the legacy row-month to FDW validity-interval mapping. `data-readiness.md` sections 3, 4 and 9 updated.

2026-10-02 slices 2–3 (uncommitted; synthetic only; real data gated by D7).

Slice 2:
- `src/feature/fourclass_features.py`: `covariate_features(alignment=None)` keeps the frozen path. A non-None alignment is validated (`check_alignment`):
  - monthly sources at O−lag;
  - annual sources at the latest eligible reference year;
  - optional actual-release evidence: unreleased → NaN, with no backward search;
  - excluded sources dropped.

  No lag values are adopted.
- New `src/experiment/availability.py`:
  - `ReleaseLedger`: `cycle_id` maps one-to-one to the reference month; publication order must match reference order; real runs refuse synthetic rows or missing coverage; `DUE_RULE` = earliest-country release, recorded as reconstruction metadata;
  - `Availability`: masked visibility, `[O−59, O)` lawful label pools, G2 gate dates (not window-limited), IPC history rebuilt through the existing `history_features`, A/B original-key fitting views (B = k′ 0/1/2 at w/3), prediction views with lawful persistence, a separate evaluator truth view, and provenance.

Slice 3 (`src/experiment/stage3.py`):
- The engine interface is implemented by the frozen `Panel` (behaviour unchanged) and by a new `ScenarioPanel`.
- `GlobalStore` keys both its memo and disk path on the full identity digest: scenario, strategy, masked months, feature and weight hashes. This fixes Codex trap 1.
- Weights reach the global, shared and independent fits; support is counted on original keys.
- Internal gate fits and gate rows inherit the outer hidden cycles, and gate truth must be lawful at the outer cutoff.
- The target prediction cohort needs no labels and carries persistence metadata. This fixes Codex trap 2.
- The crisis-F1 gate requires a gain strictly above 0.01; undefined → not enabled.

Tests (Windows py3.12.10): `ReleaseAwareViews` 11, `ScenarioStage3` 4; mutation checks detected. Full suite 134/135: only `CommittedCode` fails, by file count, until the new module is committed.

Still to do: wire `prepare_fourclass`/`run_experiment` to build `ScenarioPanel` from a real ledger and alignment; the Stage 1 scenario schedule (648) and B-weight plumbing in partition; Stage 2 crisis-F1 E4 weights with NA routing; the 72-fold A/B selection; final reports/bootstrap.

2026-10-02 slice 4 (Stage 1 scenario plumbing; uncommitted; synthetic only):
- `plan.scenario_stage1_schedule()`: the frozen 648 candidates (A/B × H4/H8 × 9 targets × k 0/1/2 × r80/r50 × seeds 42/43/44; G1/G4, L1/gt0, root increments, confirmation seed 42).
- `partition()` gains optional `X_weight`/`X_key`: child support is counted on original keys and child fits receive weights. Plumbed through `GeoRF.fit` and `train_branch`. Defaults are unchanged.
- `Availability.stage1_input` produces fit_variant / eval (designated k plus outer exclusion, one row per original key) / labelled target rows.
- `main_model_GF.scenario_root` and `--scenario-input`: label-blind split on original keys, then the D29 S/C split, a weighted root, and `run_candidate` with weights and keys. C is scored after freeze.
- `run_stage1 --split-mode scen` accepts only the exact frozen 648 with prepared inputs. G is fixed, with no screen.
- `prepare_fourclass --release-ledger --alignment` (real-only) writes `prepared/scenario/*.parquet` and the scenario schedule, covered by the outputs hash.

Tests: `Stage1Variants` 2 and `ScenarioStage1` 4 (one real scenario root end to end with fitted children). Full suite 140/141 (only `CommittedCode` file count).

Still to do: Stage 2 crisis-F1 E4 weights with NA/fallback routing; scenario development maps; 72-fold A/B selection; final/historical/2025 runner and reports. A real run needs the D7 ledger, alignment and 2025 crosswalk.

2026-10-02 (Codex independent slice 2–4 review, fixes pending recheck): independently ran 29 focused synthetic checks on pinned Windows Python (`ScenarioStage3 Stage3Engine ReleaseAwareViews WeightedContinuation ScenarioStage1 Stage1Variants`), all passed. Source review additionally found gate dates incorrectly window-limited, real alignment bypass, empty-pool failure, evaluator truth coupled to input releases, ignored cycle identity, and missing fitting-label cache identity. Sent to bound Claude for fixes and targeted regression; initial availability corrections were inspected. Requested exact aligned schema validation, original assignment counts, undefined E2 handling and retained empty-target outcomes before packet acceptance. Trellis-check dispatch failed at agent thread limit; no alternative reviewer identity invented. GitNexus impact/detect_changes still fail with LadybugDB shadow-page replay error; direct caller/diff inspection used. Real fitting remains blocked by D7. No task closure or audit-pass claim. Updated implement lifecycle evidence and backend scenario interface contract.

2026-10-02 packet 2–4 review fixes (code frozen for the coordinator commit):
- Fitting-label digest (`labels_sha256`) and an immutable `Availability.inputs_sha256` (observations, ledger, alignment; evaluator truth deliberately excluded) are in the store identity.
- A shared-store defect is fixed: the requested `g_config_params` is now separate from the resolved fit `params`, which previously overwrote the identity and broke every disk reopen, including on the frozen Panel. Disk reopen is tested for both panels.
- Evaluator truth is empty unless supplied. Gate truth uses input labels lawful at the outer cutoff, and forecast production needs no truth.
- The pinned `prepared/scenario/features.json` is verified at the CLI.
- Undefined E2 parent crisis F1 rejects the split, with no exported score. This guard is defence in depth: the E1 zero-mass guard already prevents it, and a test forces the call path.
- A root with no labelled E3 target is recorded as `no_e3_target_labels`, with no fit.
- Assignment evidence: `fitting_rows` counts original keys and `fitting_variant_rows` the copies.
- `train_branch.py` CRLF endings preserved.
- Windows py3.12.10 full suite: 147/148; the only failure is the expected `CommittedCode` file count until commit. GitNexus impact/detect_changes unavailable (LadybugDB); fallback source review used.

2026-10-02 (Codex checkpoint): slices 2–4 committed as 702888a1b9bb260016c800d8764a3f2a1ffa3ac3. Independent post-commit frozen Windows check `tests/test_baseline.py CommittedCode ScenarioStage3 ScenarioStage1`: 15/15 passed in 18.835s, including changed-label cache isolation, fresh disk reopen, truth-free forecasts, exact schema, undefined E2 and empty E3 status. Combined with executor full 147/148 pre-commit, the sole file-count failure is resolved. This is component validation, not completed task acceptance. Claude resumed Stage2 and remaining scenario runner/reporting under existing authorisation. Review requested NA reporting in scenario candidate S/C/E3 outputs and keyed original-F diagnostics after freeze; later reporting must not reinterpret undefined as zero or count augmented rows as original support.

2026-10-02 slice 5 (Stage 2 crisis E4/NA + reporting alignment; uncommitted; synthetic only):
- `run_stage2.py` crisis metric (the legacy macro path is unchanged):
  - `scenario_candidate_row` recomputes matched E3 crisis F1 (partitioned vs own root) from the saved `target_predictions.csv` rows and requires agreement with the `candidate.json` counts;
  - `no_e3_target_labels` and `root_insufficient_support` are legitimate ineligible rows with a reason;
  - a missing completion record, file or count mismatch raises.
- `crisis_plan_weights`: clipped-logit E4 on scored rows; NA rows keep weight NaN with a reason; corrupt combinations raise.
- `build_consensus`/`accept_consensus(metric="crisis")`:
  - routes no_prior_candidates / no_scorable_evidence / null_consensus / learned_map, coherent between producer and acceptance;
  - the full ledger is persisted, but only eligible rows enter steps 1–6, with crisis F1 carried in the legacy score columns and disclosed;
  - the metric is part of map acceptance.
- `scenario_map_pool` (common origin-legal pool per strategy) keeps a candidate only when:
  - its E3 target is before O (strict for development; ≤ the 2020-12 cutoff for the final freeze);
  - its target cycle is released by the cutoff in every country (`ReleaseLedger.fully_released`);
  - its full evidence span (`evidence_last_month` = last F/S/C/E3 label month from the saved `fold_membership`; IPC inputs precede their own origins) ends before the first cycle hidden at the cutoff under k_max = 2.

  Late or reordered publication orders are refused at ledger construction.
- Reporting alignment (scenario roots only; legacy modes unchanged):
  - `fourclass.nullable_crisis_summary` (F1 None with a reason) is used for E3, S validation, C confirmation and a new original-F diagnostic (`fit_diagnostic_predictions.csv.gz`: one designated-k row per original fitting key, predict-only after the frozen digest, with support);
  - `run_stage1` copies the file when declared.

Technical choices (no new scientific assumptions):
- crisis-ledger text columns are compared canonically (empty == NaN after a CSV round trip);
- the E3 NA reason names the undefined side;
- the evidence span is taken from saved role lineage, not from label values.

Tests (Windows py3.12.10): `ScenarioStage2` 4, extended `ScenarioStage1`. Full suite 152/152.

Next: development maps per origin and the 72 complete development folds with the A/B stop rule; Study1/Study2/country/bootstrap reporting; the prediction-only final path, which refuses actual 2025 runs without verified country/product availability.

2026-10-02 slice 6 (development folds + A/B rule; uncommitted; synthetic only), in `scripts/run_experiment.py` (legacy phases untouched):
- `scenario_dev_plan()` = the frozen 72 folds.
- `scen_dev_fold` runs one complete fold through `s3.run_fold` with a `ScenarioPanel`, the strategy's origin map (learned → gated L1 locals; other routes → pooled with their reason) and fixed G/L.
- `scen_fold_scores` gives crisis counts for the model on genuine-truth keys, and for the model vs lawful persistence on identical matched keys.
- `ab_select` (exact rationals):
  - qualify = normal parity ≥ −0.02 (defined) and defined one/two-cycle F1;
  - rank by their mean; exact ties → A;
  - no qualifier → no winner, with the unmet criteria;
  - incomplete folds raise.
- Driver phases `scen-develop` and `scen-select`:
  - the 648-row crisis ledger from Stage 1 evidence; per (strategy, origin) the common-pool crisis map via `build_consensus`; folds saved with `save_fold`; selection written once;
  - `scenario_context` builds Availability only from a prepared real ledger and alignment.

Tests: `ScenarioDevelopment` 3 (plan; rule incl. the exact −0.02 boundary, tie, stop, undefined, incomplete; a real fold with key/score recount and the non-learned route). Full suite 155/155.

The driver glue (`scen_develop`/`scen_select`/`scenario_context`) needs a prepared real run and is not exercised end to end. Real runs remain blocked by D7.

2026-10-02 slice 7 (reporting + final path; uncommitted; synthetic only):
- `report_fourclass.py` (legacy functions untouched):
  - `crisis_paired_bootstrap`: G3 shared country-block multiplicities from a fresh `default_rng(42)`, 2,000 fixed draws, pooled crisis counts, undefined draws counted and never redrawn; the CI requires defined points, ≥ 2 countries and all draws defined, with a reason otherwise; event concentration reported;
  - `study_rows`: Study1 = genuine-truth keys; Study2 = genuine exact-origin non-crisis risk set, with missing-origin and origin-crisis exclusions counted;
  - `onset_recall`;
  - `country_table`: descriptive, nullable F1, no tests.
- `stage3.ScenarioPanel(gate_k=...)` with an explicit `internal` flag through `GlobalStore.get`/`fit_pool`. The outer forecast uses k (0 for actual cases on the real ledger) and internal gate replay uses `gate_k`; `intensity_k` is part of the store identity. The frozen Panel accepts and ignores the flag.
- `run_experiment.py`:
  - `historical_targets`: the D2 common calendar, with origin and both k=2 missed cycles strictly after 2020-12 and exclusion reasons. The fixture reproduces the illustrated first targets: H4 2021-10, H8 2022-02;
  - `actual_gate_intensity`: the actual-case contract, which refuses a missing table, synthetic evidence, missing countries or country-specific differing counts; no global synthetic k is substituted;
  - phases `scen-freeze` (winner's final map through 2020-12; no winner → no release), `scen-historical` (k 0/1/2 on the common calendar), `scen-actual` (prediction-only; truth never loaded; refuses unresolved availability) and `scen-report` (Study1/Study2/country/bootstrap vs matched persistence from historical predictions; 2025 truth deferred to a separate release with an approved crosswalk).

Tests: `ScenarioReporting` 2 and `ScenarioFinalPath` 3. Full suite 160/160.

Disclosed gap: the driver glue (`scen-freeze`/`historical`/`actual`/`report`) needs a prepared real run and is not exercised end to end. The expert comparator is not wired, because its source/horizon mapping is unresolved under D7. Real runs remain blocked by D7.

2026-10-02 (Codex review of slices 5–7): user repeated continue including the above three; engineering scope includes end-to-end driver verification, expert comparator input and separately released 2025 truth evaluation. Reviewed Stage2/driver/report diff. Sent concrete fixes for missing candidate/prepared/source identity checks, retained pooled outputs, seasonal historical calendar, exact 72-fold identities, freeze-selection binding, no-winner reporting, country-specific actual replay, country coverage and fresh2025 covariate input. Independent first focused run: 11/12 pass, remaining consensus acceptance fails code identity while executor edits concurrently; rerun only after code freeze. This is not accepted as a clean test result. New README section distinguishes current study from old run_all.sh. No real fitting or final truth inspection; active audit run remains open.

2026-10-02 (Codex continuation): user now prioritises the approved spec/implement sequence while retaining goal mode. The original goal is still active; its old metric-maximisation wording is superseded by this explicit steering, recorded in implement.md and sent to the same Claude session via Herdr. Audit run/base/executor reverified unchanged, controller running. Current review also requires exact-origin evaluator lookup, saved pooled comparisons, input-bound completed-fold reuse, frozen actual-fold hash checks, country exclusions/comparator coverage and non-vacuous driver assertions. Executor is repairing these; no stable packet or independent full-suite result yet. A trellis-check dispatch failed at the agent thread limit, so coordinator review continues directly. Official-page metadata follow-up is in research/d7-web-publication-followup.md; this is a new release-date lead, not a passed D7 calendar. Backend code-spec now records the scenario driver/expert/truth-release contracts. No real fit or 2025 outcome-table access occurred.

2026-10-02/03 packet 5–7 + gaps (FROZEN for the coordinator commit; synthetic only; no real fit; no protected outcome access).

Review fixes:
- `accept_scenario_stage1` (prepared identity, completion code/runtime, exact root/candidate IDs, schedule fields, inventory) runs before any ledger row; `scenario_candidate_row` checks root/candidate identity.
- `scenario_context` accepts the preparation and re-certifies the pinned panel hash on every load, reading only key and covariate columns.
- The hashed 2025 covariate-extension manifest is required at actual entry: overlap equality with the pinned panel, covariate columns only, refusal otherwise. This is a D7 readiness input, so no real extension is admitted yet.
- Per-country intensity and exclusions sit in the shared Availability:
  - a mixed union keeps global exclusions global;
  - the per-row k used is recorded;
  - areas without a country use k = 0;
  - the global fit population is unchanged.
- `ScenarioPanel` keeps the mask dict-of-sets distinct from the intensity dict-of-ints in its cache key and identity. The smoke found the conflation bug; a dedicated fold regression now covers it.
- Actual cases: the outer forecast runs on the real ledger (k = 0) and internal replay uses verified per-country counts, which may differ. A missing, synthetic, incomplete or conflicting table refuses.
- Historical calendar: the frozen Feb/Jun/Oct schedule with no-truth coverage; only `UnsupportedScenario` is caught.
- Folds:
  - computed once, with reuse only via `accept_fold` against an identity that includes the prepared outputs, Availability inputs and actual extension/availability digests;
  - the same-input pooled diagnostic is saved, and scored against the system on matched keys.
- Bindings:
  - the selection records fold hashes;
  - the freeze binds the selection hash; historical binds the frozen recipe;
  - the report checks historical → frozen → selection;
  - evaluate checks the frozen `actual.json` and per-fold hashes.
- `ab_select` validates exact fold identities (duplicates and missing folds refused). The no-qualifier path writes unmet records end to end.
- Expert comparator (`keyed_expert`): requires a documented horizon, origin issue, validity containing T and release ≤ the origin cutoff; class codes integer 0..3; real mode refuses synthetic. Without an expert table, explicit coverage reasons are kept and no proxy is used. The expert table digest is recorded.
- Separate 2025 truth-release evaluation (`scen-evaluate`): an approved release binding the frozen actual predictions and the truth hash; unique keyed truth; evaluator-only exact-origin lookup (historical labels or released truth, never a latest-label substitute); unevaluable targets (June) keep coverage-only country rows.
- Country tables cover the full cohort universe, with Study2 eligibility/exclusion and onset support and all available comparators (persistence, pooled, expert). Keyed evaluator/comparator rows are saved.
- Fixed: an empty-cohort bootstrap crash.

Tests (Windows py3.12.10):
- `ScenarioDriverSmoke` 2/2 (375.9 s, `/tmp/smoke_final.log`): the actual drivers scen-develop → select → freeze → historical → actual (heterogeneous per-country k, extension admitted) → report → evaluate, plus the no-qualifier path, on a synthetic prepared run;
- coordinator: the remaining 163/163 tests (39 classes) on identical product code;
- 165/165 tests in total.

Real runs remain blocked by D7 (historical IPC release rule, 2025 truth crosswalk, extension/expert source facts).

2026-10-03 D7 publication corroboration (bounded ~10 min; read-only; no fit, product edit or 2025 values): `research/d7-publication-corroboration.md`.
- The official pages carry `publicationDate` together with their own report type and period labels: Ethiopia Feb 2020 = 2020-02-06 (Food Security Outlook, "February - September 2020"); Ethiopia Oct 2020 = 2020-10-06 ("October 2020 - May 2021"); Kenya Feb 2020 = 2020-02-28 ("February - September 2020"). This links each date to a named outlook cycle.
- Internet Archive earliest captures are only later upper bounds (Ethiopia Oct 2020: 2020-11-07; Kenya Feb 2020: 2020-05-03). Ethiopia Feb 2020 was not obtained (Archive offline). No PDF date was obtained, because the link is client-side.
- Verdict: not admissible as a verified vintage or a release rule. The FEWSNET.csv row-month ↔ report/CS linkage is unproven, the dates are uncorroborated, and there are only two countries with a 22-day spread within one cycle. No M+0/M+1/M+2 adopted.
- D7 remains blocked. Recommended next decisive facts: FDW historical CS records → document ids → page date, plus a value match against FEWSNET.csv rows on unprotected cycles; and an independent dated copy for Ethiopia Feb 2020.

2026-10-03 D7 historical-cycle linkage probe STOPPED by user steering ("不建议把大部分时间放在核实来源上…现在重点是继续已有工作"). The partial finding is kept as a limitation:
- the FDW CS rows for Ethiopia, reporting 2020-02 (929 rows; the date filter works), all reference `datasourcedocument` 6537 = "Food Security Outlook, Ethiopia" (schedule "Ad Hoc", 49 collections). That is a reusable country product definition, not one dated issue;
- the per-issue `datacollection`/`datacollectionperiod` objects need authentication;
- row `created` = 2021-08-10 (bulk ingestion).

No CSV comparison was performed. Source identity is user-attested; no further web/archive research. The historical IPC release-date convention is being resolved by the coordinator with the user (a scientific alignment question). No dates are invented and no fit happens until it is resolved.

2026-10-03 (Codex, user priority correction): engineering packet committed as b43ef6ac4e787b69d1cee48d72eb400117ba5523; 165 tests covered, scoped tree was clean. User explicitly says sources are largely manually verified and source checking must not dominate continuation. Sent Esc and redirected the bound Claude executor from publication research to minimal existing-runner preparation. Accept source identity as user-attested; no additional web/archive work. PRD/design/implement/data-readiness now narrow D7 to concrete runtime inputs, keys and leakage-relevant time semantics. One direct question is pending: retain the original reference-month-end IPC availability convention as a disclosed reconstruction, or shift to next-month-end. No inferred answer or real fit yet; later 2025 truth/crosswalk and missing expert coverage must not block unrelated historical development. Goal remains active. Attempted read-only scout dispatch hit the existing thread limit; no child ran. FLDAS documentation follow-up was saved before the user steering; it does not activate a feature rule.

2026-10-03 launch preparation (no product/test edit, no fit): `research/launch-readiness.md`.
- `research/launch/alignment.json` built by `probes/build_alignment.py` from the agreed D7 rules: 31 static, 21 monthly at L=1 (ACLED, FLDAS), 2 annual (GDP Y−1 from July; CC Y−2), 15 excluded. It passes `check_alignment(real=True)`, giving 132 of 162 features.
- GDP and CC are constant within calendar year in the pinned panel (2010–2023), so `value_month` = 12.
- `probes/build_release_ledger.py` requires an explicit rule (`reference_month_end` | `following_month_end`) and a citation; there is no default. Dry run: 954 (ISO, cycle) rows, 22 countries, 51 cycles.
- Timing on the real panel: `stage1_input` A 42 s / B 88 s per input, so about 2 h for the 108 inputs.
- Remaining fields: the release rule (user/coordinator decision), the fresh run directory outside Dropbox, and the Stage 1 worker count. The exact commands are in the readiness note.

2026-10-03 (user reaffirmation): actual spec metrics control; the original goal optimisation wording is superseded. No further source tracing, including producer tracing. Sent exact crisis-F1/parity/A-B/stop/local-gate rules to the bound Claude executor and recorded them prominently in implement.md. Continue minimal launch preparation; do not add superiority requirements or tuning.

2026-10-03 launch config corrections (coordinator review; no research, no fit):
- crop, range and market_access are excluded under D1 (unestablished historical as-of semantics). The 28 stable statics are kept as disclosed fixed reconstruction. Alignment = 28 static, 21 monthly, 2 annual, 18 excluded → **129 features**; it passes `check_alignment(real=True)`.
- The annual reference-year convention is inherited with user-attested sources; within-year constancy is consistent with it but not proof.
- Workers set to 1 initially (D7 sequential start); run dir `C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1`.
- `launch/alignment.json` is tracked via a scoped `.gitignore` negation.
- Scratch timing/storage: about 2–3.3 h and 0.7 GB for the 108 inputs. Provisional timing dates stayed scratch-only.
- Metric contract per PRD R3/D4/G3–G4 (binding user steering).
- The only pending input is the historical IPC release convention.

2026-10-03 (user timing decision): User confirmed “沿用” on 2026-10-03: historical IPC CS is treated as known at the end of its reference month, preserving the original experiment convention. The ledger uses `reference_month_end` and `evidence=reconstructed`; these are assumed availability dates, not verified publication timestamps. No additional source tracing or lag comparison. This resolves the remaining historical-development timing decision; actual-2025 availability and evaluator-only truth retain their separate contracts. Coordinator will commit the assumed release ledger/configuration, then direct the bound Claude executor to run preparation and the finite648/72 development in order with workers1. No real results claimed yet. The goal tool still reports blocked; it has no resume action, so work resumes under the user instruction without falsely claiming a goal-state change.
