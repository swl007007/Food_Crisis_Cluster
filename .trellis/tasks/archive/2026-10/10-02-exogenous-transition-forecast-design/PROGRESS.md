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

2026-10-03 (Codex supervision): goal active again with supervision-only objective; actual spec metrics control. PID886199 preparation advancing (32/108 scenario inputs observed), runtime manifest matches pinned Windows environment. Reconciler review corrected Stage1-only N to descriptive and retained full-run conservative147889 fit bound, explicit129 features and per-original-key counts/weight totals. Tried a fresh default/fork-none read-only engineering spot review of702888a..b43ef6a against currentca86bf7 contracts; dispatch failed at agent thread limit, so NO independent spot verdict exists. Controller status: running=true, same active run5c4dede7/basee843682; no current-task audit jobs. Keep task open through real execution; close audit only when evidence is complete. No source tracing or product changes in this monitoring pass.

2026-10-03 LAUNCH (authorised; HEAD ca86bf7; audit run 5c4dede7; pinned Windows py3.12.10). RUN = C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1.
- Ledger: the committed month-end reconstruction (sha256 83402d71…b9f960, 954 rows). Alignment: sha256 ca9e9a66…e543, 129 features.
- Logs and exact commands: `/mnt/c/Users/swl00/geoxgb_runs/scen-b43ef6a-v1.{commands,prepare,stage1}.log`.
- Preparation: PID 886199; exit 0, "preparation complete"; 108 scenario inputs, 707 MB.
- Pre-fit reconciliation (`research/probes/reconcile_prepared.py`; report `scen-b43ef6a-v1.reconcile.json`): exit 0, problems = [].
  - prepared outputs sha256 972902bb…f5161; git_head ca86bf7 with code equal to HEAD; pinned source hashes match;
  - 129 features, 648 schedule entries, 108/108 inputs;
  - keys, masks and per-key weights pass;
  - Stage 1 N = 5,509 (descriptive only); certified fit ceiling 147,889 (conservative N ≤ 5,714).
- Stage 1: launched as PID 1361965 (`run_stage1.py --split-mode scen --workers 1`). The first A/H4 roots took 24–39 s each.
- Coordinator one-candidate numerical spot check (not a full-scope audit or a scientific conclusion), on scenA_h4_2018-02_G1_k0_r50_s42_L1_gt0: the recounted target/confirmation/fit-diagnostic CSVs match `candidate.json` exactly. Crisis F1:
  - E3: partition 0.6720 vs root 0.6746;
  - C: 0.6464 vs 0.6288;
  - F: 0.6512 vs 0.6285.
- Remaining chain: the 648 Stage 1 roots → scen-develop (72) → scen-select. Final trellis-check and the accepted close audit are outstanding; no close.

2026-10-03 Stage 1 in progress (coordinator check after ~5h44m): 494 completed and 21 failed, all with returncode 3221225477 (0xC0000005, native access violation). PID 1361965 is still live.
- Identity retained: Claude session b0e22913-4f4b-430f-8d19-dee565d26eec, audit run 5c4dede7bc6e44f0835fbde1ad2f473d, base e84368274241e6becc67d5a7e69dc39db23bda66, launch HEAD ca86bf7, RUN C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1.
- The failed roots' temp outputs (`%TEMP%\geoxgb_stage1\scen-b43ef6a-v1\<root>\`) are preserved; no retry or concurrent fit.
- scen-develop/select stay blocked until all 648 have accepted completion evidence. Failure classification is in progress.
- Failure classification (no retry yet). All 21 failed temp trees are preserved, copied to `/mnt/c/Users/swl00/geoxgb_runs/scen-b43ef6a-v1.failed_stage1/` (769 MB), because the runner deletes temp before a retry.
  - 20/21 reached the candidate search (`fold_membership.csv.gz` plus `georf/` with 7–39 saved checkpoints). Their GeoRF `log_print` ends mid-print of partition group-id lists, so the crash point varies inside `partition()`.
  - 1/21 (scenB_h4_2019-02_G1_k0_r80_s43) has only command/run.log: it crashed before fold membership.
  - All have code 0xC0000005 (native access violation), with no Python traceback (buffered stdout). The ~4% rate is spread across A/B, H4/H8, k 0/1/2 and r80/r50.
- Smallest remedy: after the current batch exits, rerun the identical command (the runner skips roots that have completion.json; same numerical configuration) with the diagnostic-only environment `PYTHONFAULTHANDLER=1`, `PYTHONUNBUFFERED=1`, propagated to the Windows children via `WSLENV=PYTHONFAULTHANDLER/w:PYTHONUNBUFFERED/w` (`run_root` copies `os.environ`). If a root crashes again, the fault trace attributes it; no package edit is needed to obtain it.
- Approved recovery: `research/probes/stage1_recovery.sh`, launched detached (PID 2850526), waits for batch PID 1361965.
  - Env check: `WSLENV` was empty; with the two entries appended, the Windows interpreter saw PYTHONFAULTHANDLER=1 and PYTHONUNBUFFERED=1 (faulthandler enabled). No fit was involved.
  - On batch exit, in order: preserve newly failed temp trees (`failed_stage1_attempt1/`); snapshot the attempt-1 ledger/log; reconcile completion IDs against the 648 scheduled roots; run ONE identical resume pass (`--workers 1`, plus only the diagnostic env, `WSLENV` appended); write a separate resume ledger slice; preserve resume failures (`failed_stage1_resume1/`); reconcile again.
  - Logs: `scen-b43ef6a-v1.{recovery,stage1.resume1}.log`. Attempts are distinct from unique completed roots; there is no retry loop.
- Recovery re-armed FAIL-CLOSED after coordinator review. The pre-fix watcher (wrapper 2850526 and its bash child 2850532, found still waiting) is stopped; neither produced artifacts. The single resume owner is now PID **2859449**, running `stage1_recovery.sh` sha256 929355ad…feecfb.
  - `set -euo pipefail`.
  - Every failed root must be either copied and verified with `diff -rq` against its temp tree, or verified against the earlier backup in `failed_stage1/`. A missing temp tree without a backup aborts.
  - Ledger/log snapshots are byte-checked; reconcile and `cd` are guarded; the Windows env is checked before the fit. All of these abort before any fit.
  - The resume exit code is captured (nonzero tolerated) so postmortem preservation still runs.
  - Reconcile is informational; acceptance before development uses `accept_scenario_stage1`.
- Recovery script defects fixed after coordinator review. Watcher 2859449 (and its sleep child) was stopped before the edit; no instance ran during it.
  - The earlier `failed_stage1/` backup is reused only for attempt 1; resume 1 always preserves into its own `failed_stage1_resume1/`.
  - The failed-root list comes from Python JSON parsing with an explicit abort: zero failures succeed, a malformed ledger aborts.
  - The log snapshot is now byte-compared (`cmp`) as well as the ledger.
  - Synthetic helper check: 6/6 as expected (zero-failure ok; malformed abort; identical earlier backup reused; mismatched earlier backup abort; repeated failure with different contents saved separately while the earlier backup stayed unchanged; missing temp without backup abort).
  - Re-armed exactly once: PID **2873827**, sha256 1d6a2d8f…770b5b.
- 2026-10-03 RESUME STATE (for compaction):
  - Coordinator reviewed and approved recovery script sha256 1d6a2d8fa87bca63b36fb9e07a138095c223d577afcb3e21ba5fee6e99770b5b as armed. Its independent check (bash syntax; zero-failure, malformed-ledger abort, repeated failure with a new destination and the old backup unchanged) passed, with no fit.
  - Live: batch PID 1361965 (Stage 1, `--workers 1`) and recovery owner PID 2873827 (the sole resume owner; runs ONE diagnostic resume after the batch exits).
  - Then: acceptance of all 648 via `accept_scenario_stage1` → scen-develop (72) → scen-select (spec stop rule).
  - Attempts are preserved: `failed_stage1/` (first 21, 804,609,121 bytes as verified by the coordinator), `failed_stage1_attempt1/`, `failed_stage1_resume1/`. Ledger/log snapshots: `stage1.attempt1.*`, `stage1.resume1.*`.
  - Logs: `/mnt/c/Users/swl00/geoxgb_runs/scen-b43ef6a-v1.{commands,prepare,stage1,recovery,stage1.resume1}.log`.
  - Identity: session b0e22913-4f4b-430f-8d19-dee565d26eec, audit run 5c4dede7bc6e44f0835fbde1ad2f473d, base e8436827, launch HEAD ca86bf7, RUN C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1.
  - Formal trellis-check and the close audit remain later; no close.
- Coordinator second one-candidate numerical spot check (not aggregate evidence, not a formal audit), on scenB_h8_2020-02_G4_k1_r80_s44_L1_gt0: unique original keys, disjoint F/S/C roles with matching coverage, saved-probability argmax, crisis confusion counts/F1 and selected completion hashes all pass.
  - Sizes: F 62,393, S 10,298, C 10,298, E3 5,506.
  - Partition/root crisis F1: F 0.705865/0.701901; S 0.680203/0.672854; C 0.670280/0.665458; E3 0.630685/0.627067.
  - Recovery sha256 1d6a2d8f…770b5b and sole watcher 2873827 were reverified; there are no launch changes.
- 2026-10-03 08:53 MILESTONE: original batch PID 1361965 exited with 627 completed and 21 failed (0xC0000005).
  - Recovery owner 2873827 handled all 21 failed roots, with no ABORT. Each one matched its earlier `failed_stage1/` backup, so that backup was reused and `failed_stage1_attempt1/` is empty by design. It also wrote the attempt-1 ledger and log snapshots; the coordinator independently confirmed the bytes are identical and the ledger prefix is unchanged.
  - after_attempt1 reconcile: 627 exact schedule completions, no unexpected IDs; the 21 missing roots match the failed list.
  - 08:54:47: launched the sole diagnostic resume, child PID 3436146: identical `--split-mode scen --workers 1`, plus PYTHONFAULTHANDLER=1 / PYTHONUNBUFFERED=1. Log: `stage1.resume1.log`.
  - Ledger at ~08:59: 632 completed. The first failed root, scenA_h4_2018-10_G1_k2_r50_s44, completed in 34.1 s; scenA_h4_2020-02/06/10 have also completed.
  - Native crash attribution stays unknown unless a trace supports one. No further launch. Acceptance of all 648 comes before scen-develop.
- 2026-10-03 09:12 Stage 1 RECOVERY TERMINAL. Resume exit was 0, and after_resume1 reconcile gave 648 scheduled, 648 completed, unexpected [], missing [].
  - The resume ledger slice holds 21 rows, all completed: exactly the original 21 failures. `failed_stage1_resume1/` is empty. The resume log has no faulthandler or fatal-error output.
  - The coordinator verified that attempt1 (648 lines) plus resume1 (21 lines) concatenate byte-exactly to the live ledger.
  - Attempt record: **669 candidate attempts** (648 + 21). Attempt 1: 627 completed, 21 failed natively with 0xC0000005. Resume 1: 21 completed, 0 failed.
  - The crash did not recur under the identical configuration. Native failure attribution remains **unknown**: no trace was captured, because the crashes happened before faulthandler was enabled.
- 2026-10-03 accept_scenario_stage1 (pinned Windows Python, existing function) PASSED: 648/648 accepted, all `completed`, in 4m22s.
- 2026-10-03 09:20 DEVELOPMENT LAUNCHED with `research/probes/dev_launch.sh` (sha256 681f49f2b06afd85508cd1159a6d284dfe66d5c3a5bea1b230bfb868c4a0986f), PID 3552757, HEAD ca86bf7.
  - Steps, in order:
    1. Rerun acceptance and save the summary to `scen-b43ef6a-v1.stage1_acceptance.json` (asserts 648/648 completed; records the 669 attempts and the unknown attribution).
    2. `run_experiment.py scen-develop` (72 folds; existing fold records accepted by identity, never recomputed).
    3. `scen-select` (D4 rule, written once).
  - The run stops on the first failure, with no retry. The diagnostic env (PYTHONFAULTHANDLER=1, PYTHONUNBUFFERED=1, appended to WSLENV) has no numerical effect.
  - Logs: `scen-b43ef6a-v1.{develop,select,develop.nohup}.log`. A first launch attempt at 09:19 was a no-op: the path with spaces was unquoted, so bash could not find the script and nothing ran.
- Coordinator independent spot check of the first development fold (A/h4/k0/2019-02) PASSED. This is narrow numerical evidence, not a full audit or the 72-fold selection.
  - Checked: fold output hashes; unique prediction keys; pooled matching; hard argmax; fallback probability equality; paired-count/F1 recount for all 17 cluster gates; exact Fraction gain > 1/100 and the support decision; global routing of unsupported dates.
  - Rows: 5,718 forecast rows, of which 5,365 have genuine truth and matched persistence and 353 are unlabelled but retained. 4 clusters enabled, covering 920 local rows.
  - Single-fold crisis F1: system 0.7913513513513514; pooled 0.7920043219881145; matched persistence 0.7841218053289831.
  - Development process 3573472 live; no retuning.
- Coordinator narrow artifact check PASSED: A/h4/k1/2019-02 versus k0. It verifies the saved records; it is not a fresh numerical replay of the hidden inputs.
  - k1 matches k0: same map_id a3b56f2886757e8e5065 and the same 5,718 forecast/truth keys. All k1 fold output hashes verified.
  - Persistence source months are all strictly before the masked 2018-10. Ages are consistent: 5,364 age 4, 348 absent, 6 older.
  - Six gate internal origins match the saved global SHA records: 2016-06, 2016-10, 2017-02, 2017-06, 2017-10 and 2018-02. Each has scenario_k = intensity_k = 1, excluded=[2018-10] and masked={own origin, 2018-10}.
  - Validation dates fall at internal origin + H4, before the outer origin.
- 2026-10-03 documentation-only reconciliation of implement.md (coordinator request). Section 3 is split into historical items (done, with evidence references) and actual-2025 items (open). The stale blanket statement "Real data stays blocked by D7" is replaced with the Stage 1 and acceptance evidence. Section 5 now separates synthetic coverage from the formal trellis-check and the independent audit, both still outstanding. The section 6 648/72 item stays unticked until all 72 folds and selection finish. No product or test edit, no launch; the sole launcher 3552757 is unchanged.
- implement.md review corrections: added an open item to reconcile the saved fold, role and internal-gate ledgers against lawful keys and masks; narrowed the impact-analysis claim to the recorded checkpoints plus the disclosed fallback.
- Coordinator narrow saved-artifact check PASSED: A/h4/k2/2019-02. This verifies saved metadata and rows only; it is not a numerical replay of the hidden features and not a formal audit.
  - All fold output hashes verified. Same map and the same 5,718 unique forecast/truth keys as k0. Hard argmax holds; the 353 unlabelled rows are retained.
  - Persistence sources are all strictly before the hidden 2018-06. Ages are consistent: 5,364 age 8, 348 missing, 6 older.
  - Six internal origins match the saved booster SHA records: 2016-02, 2016-06, 2016-10, 2017-02, 2017-06 and 2017-10. Each has scenario_k = intensity_k = 2 and exclusions [2018-06, 2018-10]. Each masks its own origin, its own previous four-month cycle and the outer exclusions.
  - Validation dates fall at internal origin + 4, before 2018-06.
  - The checker initially assumed source months were stored as strings; the CSV stores zero-based integer month indices. The checker was corrected and rerun with a PASS; no product defect, no change.
- Coordinator saved-record spot check PASSED: B/h4/k0/2019-02. This verifies metadata and byte-level weights only; it is not a fresh fit and not proof of independent samples.
  - All fold output hashes verified. Root: original-key support 81,979 against 245,937 fitting rows (3×).
  - All 5 saved locals likewise have 3× rows over original support. Weights are float32(1/3) at both min and max; the weight hashes equal directly repeated float32 bytes; total weight conserves the original n within float32 rounding.
  - All 5 locals' parent/prefix metadata point to the same global, with 200 + 20 rounds.
  - The Kish ESS reported on variant rows must not be read as original-key support.
- Next-phase readiness written to research/next-phase-readiness.md (documentation only, no fit): commands, the per-H stop path, expected artifacts (57 historical folds if both horizons are released), pre-launch checks and the missing actual-2025 inputs.
- Readiness review corrections (coordinator): no exact fit count is recorded, so the evidence and the conservative bound are cited instead; the extension's documented contract is separated from its enforced checks (Scaffold enforces the complete grid; the manifest month range, origin-month reach and covariate missingness are not enforced, so they become an explicit pre-actual check); the ledger-ending-2024-10 reconciliation is kept as a pre-actual check. No product edit.
- Fit-budget limitation, for the final check: 147,889 / 40,824 is the pre-launch schedule bound. The 21 failed partial attempts are kept separate, with conservative overhead ≤ 21 × 63 = 1,323 (63 = 40,824 / 648), giving an operational total of ≤ 149,212 in two components. The exact actual fit count is unavailable; no additional fit authorisation is implied.
- Actual-2025 local readiness addendum (metadata/keys only): the extension needs a complete grid only for 2025-01..05. The annual refs (GDP 2023; CC 2023/2022) are inside the pinned panel. The combined panel has all sources but its 2996 duplicates refuse it as-is. A truncated extension is assemblable; it was not assembled in this metadata-only pass and is pending pre-actual checks. The overlap-value equality over all 69 schema sources is unchecked. The 2024-10 availability rows can be built from the ledger; Feb/Jun-2025 missed_cycles evidence is missing (branch stopped). The Oct-2025 crosswalk is unresolved (DRC and Ethiopia unmatched); no June truth or expert table exists locally. See research/next-phase-readiness.md.
- Addendum correction (coordinator review): static sources are read at O (fourclass_features.py:197, 215–217), so the extension scaffold must reach 2025-06 for all 5,718 areas, while monthly dynamic input ends at 2025-05. The missingness check is now split by kind (static at O versus monthly at O−1). The all-69-source overlap issue stays explicit. Wording is now "not assembled in this metadata-only pass; pending pre-actual checks", with no new permission gate.
- Stage 1 diagnostic summary (saved artifacts only; no fit or RUN write): research/stage1-diagnostics.md, probe research/probes/stage1_diagnostics.py (6m11s; reuses accept_scenario_stage1, scenario_candidate_row, crisis_plan_weights and diagnostics), outputs stage1_diagnostics_{ledger.csv,summary.json}. All 648 scored with no NA; F/S/C/E3 roles disjoint on original keys. Partition-minus-root crisis F1 is positive within 2018-2020 (F/S/C positive in 87-98% of candidates) but about zero out of time on E3 (medians -.0020..+.0050; 24-80% > 0). This negative transfer result is preserved. Root-only 9/324 (A), 14/324 (B); E3 root fallback 0.6-0.9% of rows; genuine unique partitions 319 (A), 315 (B); positive E4 weight 148 (A), 182 (B). Candidate distributions only, not the selection statistic.
- 2026-10-03 16:51 DEVELOPMENT + SELECTION COMPLETE. Launcher 3552757 exited 0.
  - scen-develop finished in 26,808 s: 72/72 fold records, no traceback or fatal output. Routes: 66 `learned_map`; 6 `no_prior_candidates` (A and B × k0/1/2 at H8 2019-02, origin 2018-06), which use the pooled global.
  - scen-select finished in 7 s: `scenario_development/selection.json`, sha c0b3c967127c9a91c557343d5d9af3114c4c0975558b4c1a9d70ac4644ab9b8b. All 72 fold_records equal the current fold.json SHAs.
  - D4 decisions, pooled crisis F1 from exact fractions in selection.json. "Parity" is matched normal model minus matched persistence.
    - **H4: winner A.**
      - A: normal .6283 vs persistence .6416, parity −.0133 (qualifies); k1 .5484, k2 .5658, mean .5571.
      - B: normal .6003, parity −.0413. Fails `normal_parity_below_-0.02`; k1 .5017, k2 .5259, mean .5138.
    - **H8: winner A.**
      - A: normal .5478 vs persistence .5428, parity +.0050 (qualifies); k1 .5651, k2 .5483, mean .5567.
      - B: normal .4933, parity −.0495. Fails `normal_parity_below_-0.02`; k1 .4943, k2 .4882, mean .4912.
    - A won both horizons as the only qualifier, not through ranking or a tie.
  - Negative results preserved:
    - B fails the normal-parity screen at both horizons.
    - A's H4 normal F1 is below matched persistence (−.0133, within the −0.02 screen; no superiority claim).
    - Interruption persistence, descriptive only and not part of the rule:
      - H4 k1: A .5484 vs persistence .5428;
      - H4 k2: A .5658 vs .5679;
      - H8 k1: A .5651 vs .5679;
      - H8 k2: A .5483 vs .5378.
    - These are single pooled numbers without uncertainty.
  - Next, after coordinator review of the full selection: scen-freeze (no fit), then scen-historical (57 folds expected), per research/next-phase-readiness.md.
- Coordinator selection review PASSED. This is a bounded selection checkpoint, not the full Trellis audit; the independent sub-agent dispatch failed on the thread limit, so no audit pass is claimed.
  - Independent recompute from all 411,696 forecast rows (21,288 unlabelled rows retained), using stdlib csv/gzip/Fraction and no producer helper.
  - Passed: the 72 IDs and fold hashes; all 354 output hashes; the identities; A/B key/truth/persistence equality; hard argmax; nonlocal = pooled probabilities; shared maps; 66 learned / 6 no-prior folds; every count, rational, tie and qualifier.
  - Exact copies preserved: checker `research/probes/check_scenario_selection.py` (sha 17d7430d…73ec) and report `research/scenario_selection_review.json` (sha 3d48f0c8…b36a).
- Correction to the earlier descriptive interruption numbers. Comparisons against persistence must use MATCHED model counts (the ranking separately uses all-genuine model F1). Matched model minus persistence:
  - A H4: k1 +.0071434384, k2 −.0004691156;
  - A H8: k1 −.0012399598, k2 +.0118776020.

  These supersede the unmatched figures in the 16:51 entry; the signs are unchanged.
- stage1-diagnostics.md corrected: the zero-weight share is about 20%–76% per cell (B h8 k2 43/54 positive), not 52%–76%.
- scen-freeze launched via `research/probes/phase_launch.sh` (sha d65fadf7…9766). Prechecks passed: package HEAD ca86bf7 clean, no live fit, 318 GB free, no scenario_final/historical/report folders. Frozen maps and the historical calendar are verified before scen-historical.
- 2026-10-03 17:03 scen-freeze finished in 353 s with no fit. `scenario_final/frozen.json` was accepted by `_accept_frozen`: it is bound to selection c0b3c967…b8b, with cutoff 2020-12.
  - Both horizons released strategy A on the same map `8965af6d6a724ba5d61d` (learned_map, accepted by `accept_consensus`, 5,510 areas mapped): H4 with G1 + L1, H8 with G4 + L1.
  - The read-only calendar (`historical_targets`) matches design.md:28:
    - H4: 10 targets, 2021-10..2024-10; excluded 2021-02 and 2021-06 (origin_or_missed_cycle_not_after_2020-12_freeze);
    - H8: 9 targets, 2022-02..2024-10; excluded 2021-02, 2021-06 and 2021-10;
    - that is 57 folds.
  - The earlier wait loop self-matched `pgrep -f`, which the coordinator caught. Its shell had already gone; no fit was affected, and watchers now use exact PIDs.
- 2026-10-03 17:10 scen-historical → scen-report launched, single launcher **PID 1439211**, via `phase_launch.sh` (sha d65fadf7…9766). The PID is recorded via exec in `scen-b43ef6a-v1.historical.pid`. Same pinned stack and diagnostic env; stops on the first error, no retry.
  - Logs: `scen-b43ef6a-v1.{historical,report,historical.nohup}.log`. Watcher b6jy9u0gd runs `kill -0 1439211`.
  - Actual-2025 stays blocked on the remaining input checks.
- stage1-diagnostics.md: added the absolute F/S/C/E3 root/final levels and the F→C / F→E3 gaps, from the saved ledger only. The root's own F−C (H4 +.004..+.012; H8 +.027..+.035) is as large as or larger than the partition-minus-root C gains, and final F−C ≈ root F−C. The out-of-time F−E3 drop (up to +.25) mixes overfit with temporal shift. B's E3 levels are below A's in every cell.
- Covariate-only pre-actual check (research/probes/extension_overlap_check.py; summary json; 33 s; usecols keys + 69 sources): all 1,029,240 pinned keys are shared. The 51 admitted sources match exactly. 2 of the 18 excluded (Tair_zscore, Rainf_zscore) mismatch on every key, so load_extension, which compares all 69, would refuse the combined panel: Blocker 1, needing a reviewed resolution with no manufactured agreement. Static (28) and FLDAS (2) are complete at the required months. The 19 ACLED sources are missing for 1,304 areas at 2025-01 and for all 5,718 at 2025-05 (pinned 2024 baseline 0), so Oct-2025 H4 would have no ACLED at all: Blocker/limitation 2, a disclosed coverage shift with no imputation. Details in next-phase-readiness.md.
- Covariate probe CORRECTED (coordinator caught a semantic bug): the comparison now uses np.isclose(equal_nan=True, rtol=0, atol=1e-9), as load_extension's np.allclose does. Rerun on all 1,029,240 shared keys: admitted 51/51 still match exactly. Excluded Tair_zscore and Rainf_zscore still mismatch on every key, with 0 same-signed-inf pairs; 997,920 / 997,380 keys have both values finite and different (max |diff| 13.80 / 12.25), 30,780 / 30,960 involve an inf and 540 / 1,080 differ in NaN pattern. Blocker 1 STANDS on corrected evidence. The ACLED missingness limitation is unchanged.
- Adversarial review of the two-source covariate splice: VALID WITH CONDITIONS, no counterexample. No spec requires a single raw file; load_extension appends only months after 2024-12, so the pinned panel stays authoritative; excluded sources never enter features. A product edit would change code_identity and invalidate every accepted record. Must be disclosed: the in-code overlap check becomes true by construction, so the identity evidence is the external probe. Static boundary identity checked now: 28 sources × 5,718 areas, combined 2025-01..06 = pinned 2024-12, 0 differences. The assembly probe/manifest facts are listed in next-phase-readiness.md. No assembly yet.
- Actual-2025 covariate extension ASSEMBLED and VERIFIED (two-source splice; no fit, no product edit). Directory C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1.actual-inputs-v1: CSV sha 0d6b78ca…8b05 (40,026 keys = 5,718 × 7), manifest da9b3084…9c9c, assembly_report 26884dff…f0b3, verification_report 103a363f…e943; probes assemble_actual_extension.py (50ac8411…) and verify_actual_extension.py (421860df…). Raw hashes unchanged. Blocks equal their named sources string-for-string; static boundary 0 differences. The real load_extension → Scaffold → covariate_features path under the frozen alignment gives static at O through 2025-06, monthly at exactly 2025-01/05 and annual at pinned 2023-12/2022-12. Dimensions: 129 aligned = 54 covariate (51 sources + 3 calendar) + 75 history. Origin 2024-10 is unchanged with the extension. The ACLED NaN counts match. implement.md: extension item ticked; availability table and ledger reconciliation still open.
- Manifest wording correction (coordinator): admitted/static agreement is within rtol 0, atol 1e-9, equal_nan, not exact (e.g. Rainf max finite |diff| 1.6e-27). v1 manifest da9b3084…9c9c is preserved. New extension_manifest.v2.json is 72a766c6…81d6; it differs only in source/supersedes. The binding record manifest_binding_v2.json (5a457939…439a) shows load_extension(v2) gives a panel identical to v1, with the CSV unchanged (0d6b78ca…). Readiness and implement wording updated; use v2 for --actual-scaffold.
- Coordinator independent stdlib scan of the extension (32 s, read-only): raw component SHAs pass; 5,718 overlap rows and 34,308 extension rows are string-equal across all 69 source fields; the full 40,026 source and output key sets are equal with no duplicates. Cited as independent source-block verification, a bounded checkpoint and not a formal trellis-check or close audit.
- Development ledger reconciliation (implement §3 open item, development part): research/dev-ledger-reconciliation.md, probe research/probes/dev_ledger_reconcile.py (ea3f05da…), summary f3e21605…; 17 s, read-only. **72/72 folds, 0 problems.**
  - Outer global identity: 72/72 (origin, strategy/k, masks = hidden(O,k), label range within [O−59, O), A/B rows and weights).
  - Gate calendar: 66 learned folds use the latest six lawful months U < O with internal origin = U − H; the 6 no-prior folds have no gate by design.
  - 396 internal globals: masks = outer ∪ own hidden. 278 locals continue the outer global with 20 rounds.
  - Prediction keys 72/72. Matched persistence: latest lawful label ≤ O, never hidden, age = O − source, 0 mismatches.
  - The expected values come from the same package definitions, so this shows consistency, not independent lineage. Per-key fit lists are not saved (digests only), so per-key mask avoidance rests on the code and its tests.
  - Three intermediate probe errors were corrected before the result: no-gate folds, any-vs-all identity matching, age from T vs O.
  - Historical folds are reconciled after scen-historical finishes.
- Coordinator crosswalk KEY-feasibility note (metadata only; not boundary certification or a truth release). Oct-2025 CS has 5,573 unique fnid/full-name rows.
  - Exact full-name join against all FEWSNET.csv history, excluding names that map to several area codes: 4,479 one-to-one / 1,094 unmatched-or-ambiguous; 0 duplicate matched area keys; all 20 matched-country names agree.
  - The earlier 4,480 included the ambiguous Kenyan name "Northwestern Pastoral Zone, Kerio Delta, Turkana Central, Turkana, Kenya", which maps to both 2995 and 2996.
  - Restricting the lookup to 2024-10 names gives 4,478 / 1,095.
  - Rules: no join is chosen by resulting score; no labels are duplicated across 2995/2996; no outcome values are opened.
- Manifest v2 diff independently verified by the coordinator: only the source disclosure and supersedes changed.
- Reconciliation prose corrected (coordinator): fit_label_months is the DECLARED window [O−59, O−1] set by GlobalStore.get (stage3.py:253), not the empirical min/max of fitted labels. The window check proves only the recorded definition, and the 'endpoint not masked' sub-check is vacuous. Global key identity is pending the coordinator's independent reconstruction of key/label/weight digests against all saved global records (not duplicated here). Local fitted-key provenance remains a separate limitation.
- Coordinator independent global fitting-key check PASSED: 373/373 scenario_globals records present at probe start, 0 problems. Explicit filters (country release ≤ O, window [O−59, O), outer + own-k masks), independent A/B variant ordering, no package imports, features, fits or RUN writes. All saved fit-key, label and weight digests, original-key support, class counts and rows match. Preserved byte-exact: research/probes/check_scenario_fit_keys.py (eed66e0d…695b) and research/scenario_fit_keys_review.json (80577d3f…4bd3). Proves saved global input identities; not local fit keys, feature values, booster internals or the formal audit. Historical-phase globals written later are outside the 373 and get reconciled at completion. dev-ledger-reconciliation.md updated: hash reconstruction does not necessarily test the same code against itself.
- Independent LOCAL and GATE reconciliation, development (probe research/probes/local_gate_reconcile.py 607b1869…; output local_gate_reconcile_scenario_development.json 3aabd903…; pandas/numpy only, no package import, no fit, 31 s): **0 problems** across the 66 learned folds.
  - 278 outer deployed locals: key-digest proof (ordered keys of outer pool ∩ map cluster), plus original support and variant rows.
  - 7,160 gate blocks: evaluator keys, truth and internal origins.
  - Internal local support counts and support decisions (6,754 fitted), support-only evidence because internal key lists are not saved.
  - 1,194 regions: exact-fraction crisis-F1 gains, floors, strict > 1/100, current support, deployed routes and prediction routes.
  - Negative control (window 58) flags 66/66, so the result is not vacuous.
  - Reuse on the 57 historical folds after completion.
- local_gate_reconcile.py strengthened per coordinator (sha 4059c594…; output e24e620e…): every prediction area→cluster checked against the frozen map; region, local and gate-pair cluster sets checked; internal support checked on every row; recorded enabled flag checked against the recomputed gate decision, separately from the deployed route; B native sample_weight blocks (float32 sha/n/sum/min/max) checked against repeated float32(1/3), and A has no weight block. Rerun: 0 problems; 278 locals, 142 B weight blocks, 1,194 regions, 7,160 gate blocks, 377,388 prediction rows. Second negative control (float64 weights) flags all 33 B folds with locals. Outer locals = key-digest proof; internal locals = support counts and decisions only, with no per-key provenance.
- Coordinator accepted the local/gate checkpoint: probe 4059c594…77d9, result e24e620e…96ea; 72 folds, 278 outer local key digests, 142 B weight blocks, 1,194 decisions, 7,160 gate blocks, 377,388 learned-fold prediction routes, 0 problems. The 6 no-prior folds are covered by the earlier selection/fallback check, not by these route totals. The internal-local support-only limitation is retained.
- Completion sequence (no completion claim before all of these):
  1. all 57 historical folds plus scen-report finish;
  2. rerun the saved-ledger check (dev_ledger_reconcile, adapted to the historical calendar), the coordinator's independent global fit-key checker over all records, and local_gate_reconcile on scenario_historical;
  3. reconcile the keyed report metrics and coverage;
  4. the formal whole-task trellis-check and the spot/close audit.

  Actual availability facts are still pending. No further probe expansion and no fit beyond the approved schedule. The exact-PID watcher (b6jy9u0gd, kill -0 1439211) is maintained.
- 2026-10-03T17:41:20-04:00 Formal read-only trellis-check dispatched (native trellis-check sub-agent, background). Scope: diff e843682..ca86bf7 plus current evidence, R1–R6/AC1–AC7 and the Stage 1/2/3/reporting/actual contracts. No fixes, edits, fits or task-state changes. Report goes to research/trellis-check-progress.md. Items needing historical/actual/evaluation evidence are marked INCOMPLETE. This is not a whole-task pass and not the independent spot/close audit.
  - Agent handle: trellis-check background agent a887e4af4dd5052fa (Claude Agent tool). 10-minute limit: if it is unfinished at 10 min, partial findings are collected and the agent is stopped.
- Formal read-only trellis-check (agent a887e4af4dd5052fa) finished in about 526 s, within the 10-minute limit. It received a wrap-up message at about 7.5 min; no stop was needed. Report: research/trellis-check-progress.md. Implementation/current-evidence checkpoint only; NOT a whole-task pass and NOT the independent spot/close audit.
  - Code identity: `git diff --stat b43ef6a ca86bf7 -- FEWSNETGeoXGBExperiment` is empty, so the 165-test evidence is reused (caveat: a stale smoke-filename fix was verified by digest, not rerun). No tests rerun.
  - About 30 PASS items:
    - Stage 1 roles, weights and support; E2 strictness (10,874 decisions, 0 undefined-parent accepts);
    - Stage 2 crisis E4, NA and routes;
    - Stage 3 gate, masks and cache;
    - selection;
    - reporting/bootstrap and actual/evaluate as code.

    All cited hashes were recomputed and match.
  - INCOMPLETE: historical report values, actual 2025, truth evaluation, experts, AC4–AC7, and the D18 map-selection-bias disclosure in the final report.
  - NOT REVIEWED (time limit): line-level prepare_fourclass.write_scenario_inputs, acceptance.accept_fold internals, the full test diff, the README, and the probe bodies.
  - Findings (none changes a development result):
    1. stage3.py:253: the declared label window (provenance; closed for development globals by the independent checker; rerun on historical).
    2. stage3.py:445–457: internal gate locals save no fit-key digest (provenance; support-only).
    3. run_experiment.py:668–689 load_extension: compares all 69 sources; first/last month not enforced; the splice makes the overlap check true by construction (required behaviour, actual-only, disclosed).
    4. Fit budget: no realized count against 147,889; 21 crashed partials ≤ 1,323 outside it (provenance; countable at completion).
  - Limitation L1: ACLED is natively NaN at the 2025 origins, a training/prediction shift that needs a reviewed disclosure before scen-actual.
  - Previously uncited hash: frozen recipe `scenario_final/frozen.json` sha 2344ab59…573e (verified).
- Executor addendum appended to research/trellis-check-progress.md; the reviewer text is unchanged. Corrections:
  1. No exact realized fit count. Stage 1 candidate fit_log exists only for completed candidates; there is no counter for internal/local/crashed fits. Cite the schedule bound 147,889 plus crash overhead ≤ 1,323.
  2. L1 narrowed to the observed 2025-vs-2024-baseline contrast.
  3. Post-fix ScenarioDriverSmoke 2/2 (375.880 s, /tmp/smoke_final.log) recorded alongside the pre-fix 163/163 and the digest bridge.
  4. Finding 3 is a loader limitation for arbitrary input, not a failure of the verified extension.
  5. Partial historical folds exist; completed historical/report evidence does not.
  6. Sub-agent a887e4af4dd5052fa is terminal (about 526 s, within the 10-min bound).
  7. The NOT REVIEWED areas need a bounded follow-up at the final check.
- 2026-10-03T17:53:17-04:00 Follow-up formal read-only trellis-check dispatched: trellis-check background agent a9bf05248ad5de381, started 2026-10-03T17:53:17-04:00, 10-minute bound. Scope: the previously NOT REVIEWED items only (write_scenario_inputs line-level plus callers; scen-* acceptance functions incl. whether the legacy accept_fold is used; test diff coverage against the contracts; README; the 4 relied-on probe bodies). Report: research/trellis-check-followup.md. No fixes or edits; historical/actual outputs stay INCOMPLETE.
- Follow-up trellis-check (agent a9bf05248ad5de381, started 17:53) finished on its own after about 221 s, within the 10-minute bound; no stop was needed. Report: research/trellis-check-followup.md. Read-only; no tests rerun, no fit, run dir untouched. Not a whole-task pass and not the spot/close audit.
  - PASS: write_scenario_inputs plus its dataflow (window, release, B w/3 grouped variants, per-variant history with the outer exclusion, F/S/C split on original keys before augmentation, real-run refusals, pinned feature order). Acceptance (with one finding). README commands and interfaces.
  - Findings (none result-affecting):
    - scen_select (run_experiment.py:796) and scen_report (:997) call accept_fold with a partial identity (strategy/horizon/k/target only). Evidence-only, because scen-develop accepted every fold with the full identity.
    - No negative test for accept_scenario_stage1 against the real completion writer (test gap).
    - No scenario-path C-label permutation test.
    - No select/report mismatched-fold refusal test.
    - Five-level/80-round limits not retested on the scenario path.
    - README still says real fitting is blocked by data-readiness.md (stale for 2018–2024).
    - check_scenario_fit_keys.py takes the inherited outer exclusion from the record under test.
    - check_scenario_selection.py takes truth/persistence from the saved predictions.
    - dev_ledger_reconcile.py imports package code (consistency only).
    - Stage 1 acceptance does not flag unexpected root folders; the inventory check ignores unrecorded files.
  - Executor cross-reference (not a reviewer conclusion):
    - The fit-key probe's inherited-exclusion dependence is covered for development by two other checks. dev_ledger_reconcile checks each internal global's excluded_months = hidden(O,k) (package-based, consistency). local_gate_reconcile derives the internal pool's outer mask from the fold origin independently.
    - For persistence, dev_ledger_reconcile recomputes the latest lawful source month (package-based).
  - Still INCOMPLETE: historical (live), actual 2025, truth evaluation, report bootstrap values, experts.
  - Still NOT REVIEWED (time-bounded): most new test bodies, scen_report past line 1000, scen_evaluate, line-by-line local_gate_reconcile / check_scenario_selection, S-row weight handling inside run_candidate. Carried to a bounded follow-up at the final check.
- Follow-up triage (coordinator):
  - F1/F3/F4: generic acceptance/inventory limitations, no live-run invalidation. The final check must compare each fold's map_id/phase/prepared SHA with the frozen/current expected values.
  - F2a is exercised by the 648 accepted roots; the missing negative test is kept as test debt. F2d: inherited limits via source/config.
  - F5: README status corrected (FEWSNETGeoXGBExperiment/README.md: historical D7 resolved by the user-confirmed reconstructed reference_month_end convention; final-2025 inputs pending). Commands unchanged. README is outside code_identity: identity still fd25e2f7…, 65 files, equal to HEAD and the selection record. The package tree is dirty (README) until committed; phase_launch.sh checks cleanliness only at launch, so the running launcher is unaffected.
- (A) Scenario-path C-label permutation check (implement §5), synthetic fixture only, no real data. Probe research/probes/c_permutation_check.py; summary c_permutation_check_summary.json; exit 0, **PASS** for A and B.
  - C labels changed (109 membership labels per strategy; input cells 218 for A, 436 for B). F/S labels and roles unchanged; C keys never F/S (0 overlap; B's 420 C variant rows unused).
  - 6 child fits exercised per run. All 14 checkpoint model files are byte-identical.
  - The same-input determinism control has 0 differences. The only differing file is confirmation_predictions.csv.gz (the declared C diagnostic). candidate.json is identical apart from the declared 'confirmation'/'timings' blocks.
  - The initial comparator run reported pass=false. It is preserved in research/probes/c_permutation_check_initial_failed.json (transcribed from the console; the file had been overwritten).
  - Diagnosis: the identical input produced the same differences. Decompressed CSV columns were equal; candidate.json differed only in the checkpoints.dir scratch path.
  - Comparator-only corrections: (i) gzip container metadata (compare decompressed content, row order kept); (ii) the scratch checkpoint dir path normalised to relative (checkpoint SHA inventory and model bytes still compared). Plus explicit requirement checks and a nonzero exit.
  - This was a comparator artifact, NOT a model difference.
- (B) local_gate_reconcile.py extended (sha 26d6e261…; output 8acfd935…; exits nonzero on problems; exit 0), all 72 folds, 0 problems.
  - Forecast truth at T: 411,696 rows.
  - Persistence class/source month/age = latest lawful label ≤ O: 387,768 rows.
  - 72 outer globals and 396 internal gate globals linked to saved records with origin/strategy/k/excluded/masked derived independently from the fold origin and k.
  - This closes F6/F7 beyond the package-consistency check. Together with the coordinator's fit-key checker it removes the inherited-exclusion dependence for the development globals.
  - Negative control (persistence < O; internal mask without the outer exclusion): exit 1, 24 k0 persistence folds and 264 internal globals flagged, as expected.
  - Uncovered: internal local per-key provenance, features, booster internals, historical folds.
- Coordinator accepted the bounded verification closures for F2b (C-permutation, summary c19924dd…7ee83) and F6/F7 (extended local/gate, output 8acfd935…74390).
  - Disclosure: the initial comparator outcome (pass=false) was NOT byte-preserved. The original summary file was overwritten by the corrected run before preservation. research/probes/c_permutation_check_initial_failed.json is a transcript/narrative of the console output and diagnosis, not the original file.
  - Package diff: README only.
  - No further probe or test expansion.
  - Next: await all 57 historical folds plus scen-report, then run the queued final reconciliations and the report recount.
  - The formal overall check stays incomplete until those and the actual-2025-dependent evidence are resolved. No task closure.
- Coordinator checkpoint: launcher 1439211 and child 1439316 are live at 21/57; the audit run stays active, no lifecycle change. Completion steps:
  1. Explicitly verify each historical fold's phase, map_id (frozen 8965af6d6a724ba5d61d) and prepared/availability SHAs against the frozen expectations, in addition to the saved acceptors.
  2. Reuse the independent key, local and gate checks.
  3. Recount the metrics.
  The actual availability question is still pending; it is not inferred from absent files.
- 2026-10-03 20:29 HISTORICAL + REPORT COMPLETE. Launcher 1439211 exited; scen-historical took 11,761 s and scen-report 14 s, no traceback.
  - Hashes: historical.json 67153c534d382ed9f30feb0982d37c680d221d243ad819c1dfd8f2a7d5de9d79; report.json 4c0efad070a92accd06d65e0ceb514f0f230af0a40c91f55eb81fe78b6bf7b2b.
  - Post-run reconciliations, all exit 0 with 0 problems:
    1. Identity and inventory: probes/historical_identity_check.py f1269cf9… / summary 4bce3b8c…. 57/57 folds equal calendar × k (H4 10, H8 9, design calendar, all with truth); phase, frozen strategy/map, prepared sha, same-H availability digest and selection code/runtime all match; the binding chain holds; the saved acceptors pass (accept_fold 57/57).
    2. Coordinator fit-key checker (eed66e0d…, unchanged) over ALL globals: scenario_fit_keys_review_final.json 1f02f807…. 654/654 records.
    3. local_gate_reconcile.py a6924fa6… on scenario_historical: output 7751ff1e…. 57/57 folds; 232 outer locals; 969 regions; 5,764 gate blocks; 325,926 forecast rows; 317,612 persistence rows; 57 outer and 342 internal globals linked. The development rerun with the same probe reproduces 8acfd935… byte-for-byte.
  - The probe change is compatibility only: the route is read from consensus.json where historical folds lack map_route.
  - implement.md: ledger-reconciliation and freeze items ticked; the historical item stays open pending the coordinator's metric recount.
  - Limits: internal-local support-only evidence; features and booster internals not verified.
  - Actual availability is still pending.
- Coordinator independent historical metric recount COMPLETE (exit 0, problems = []).
  - Coverage: 325,926 forecast rows, 138 country rows, 36 paired comparisons (12 unavailable-expert routes). Checked hashes and binding, forecast-to-report row equality, truth, argmax, Study1/2 counts and F1/P/R, paired bootstrap CIs and country fields.
  - Byte-exact copies: research/probes/historical_metric_review.py (759a4a69…a1ff) and research/historical_metric_review.json (88d21073…2bf9).
- research/historical-results.md written: Study1/Study2 spec tables, matched n/CI/coverage, negative results, caveats (D18, earlier exposure, availability reconstruction, limited bootstrap, fit budget, evidence limits), and a supplementary per-target-month table derived from saved rows with no fits.
  - Study1 Δ vs persistence: H4 −.015986/+.007265/−.004412; H8 −.046298 (CI [−.0961, −.0015], entirely negative)/−.033383/−.018419.
  - Δ vs pooled from −.000487 to +.001571.
  - The Study2 k0 persistence F1 = 0 is mechanical.
  - No expert table was supplied.
- implement.md: historical-evaluation item ticked (execution plus recount only). Not a whole-task or audit pass. Actual-2025 is still pending its availability facts; no close.
- Documentation clarification only (coordinator review of historical-results.md; no numbers changed, no new analysis):
  1. Study2 denominators labelled. F1/Δ columns are persistence-matched. Model onset recall uses all risk-set keys; persistence recall uses matched keys. The unqualified higher-recall comparison was removed.
  2. Δ vs pooled is stated as all genuine keys, in both tables.
  3. 'Model does not beat persistence' replaced by 'no Study1 cell has a positive Δ CI excluding zero'; the H4 k1 positive estimate is kept.
  4. implement §3 frozen-map pin item reconciled with §6 (ticked: frozen.json 2344ab59…); actual prediction artifacts, availability and truth stay open.
  Coordinator independently recounted all 57 monthly table cells from saved predictions: 0 discrepancies. Historical checkpoint accepted within scope. Waiting for the pending actual country/product availability facts. Overall task and audit incomplete.
  - historical-results.md after the clarifications: sha 431ea61b50ad79d694c394d31911eb4a903dfeace242cd913d8189a03dcc3dff. The Study2 interruption statement is scoped to k ≥ 1, with the k0 CI exclusion noted as mechanical.
- 2026-10-03 ACTUAL-CS AVAILABILITY RESOLVED. The user answered "可以确认" to the explicit all-covered-countries question: no new CS after 2024-10 at the 2025-02 and 2025-06 origins, so missed service cycles are 1 and 2 (0 at 2024-10 under the inherited convention).
  - Table `research/launch/actual_availability.csv`: sha 673651082fd1ee33be390f66ce7ce596e2756971ae4bb66a318f34316acb556c; 66 rows = 22 countries × 3 origins; evidence=reconstructed; the source cites the user attestation, with no verified-vintage claim.
  - SD disclosure: its prepared last observed cycle is 2024-06 (ages 4/8/12). No October label is manufactured, and no extra missed SD publication is inferred.
  - Preflight `research/probes/actual_preflight.py` (388e8e95…) → summary da5f4301…, 0 problems:
    - the code's actual_gate_intensity gives 22 × 0/1/2;
    - the ledger has no cycle after 2024-10;
    - extension v2 (72a766c6…) scaffold through 2025-06;
    - frozen accepted; scenario_actual absent; code fd25e2f7… = selection.
  - Initial preparation outcome preserved: the 63-row draft without SD (aee0df58…) was refused by the code for the missing SD (summary d977878b…). This was a preparation outcome, not a source blocker.
  - D7 addenda added to design.md and prd.md (no metric or scientific contract change); readiness and implement updated.
  - Launcher `research/probes/actual_launch.sh` (bd713324…): 4 cases, fresh output, pinned Python and env. NOT launched; awaiting coordinator review/commit.
  - The README modification must be committed first (the launcher checks for a clean tree). The CSV needs `git add -f`.
  - Truth and expert values stay protected.
- FROZEN PACKET (pre-commit), after the coordinator's provenance and README corrections:
  - **Table** research/launch/actual_availability.csv: sha 185e2c790d0b9d11102bceea0a5925947b99116ad4d3531a1148974eb9804b59. 66 rows; the source wording attributes the attestation to the coordinator conversation, relayed via Herdr (44 rows changed, values unchanged).
    - Superseded v1 kept: launch/actual_availability.v1_superseded.csv 67365108…556c (wording "in the executor session").
    - Earlier draft kept: launch/actual_availability.pending_SD.csv aee0df58….
  - **Preflight** rerun on the final table: research/probes/actual_preflight_summary.json 2a78935695bdc681102b3b9d52ded764a9638378ea5760c927a7adf1406154c9. 66 rows, 22 × 0/1/2, problems [].
    - Superseded: v1 da5f4301… (actual_preflight_summary.v1_superseded.json).
    - Initial 63-row: d977878b… (actual_preflight_summary.initial_63rows.json).
    - Probe actual_preflight.py 388e8e95….
  - **Launcher** research/probes/actual_launch.sh: bd713324cb431ca67bab333a3e037829f84cdd24aeb416d30e80b8e1439b1473. Exact command in next-phase-readiness.md.
  - **Extension** manifest v2 72a766c6…81d6; CSV 0d6b78ca…8b05.
  - **README** (documentation-only status, sha c5ff299a1f3db6e5bc4bc5874ea7314eac1f25636f1d4785731ce6ec00b89941): actual CS availability is a prepared user-attested reconstruction (0/1/2, reconstructed, not verified vintages); frozen 2025 predictions, the truth release/crosswalk and expert coverage are pending.
  - **Code identity** fd25e2f7… (65 files) unchanged.
  - **Commit needs:** README, the task docs (design/prd/implement/PROGRESS/readiness/historical-results/dev-ledger…), the research probes and JSONs, and `git add -f` for the launch CSVs (*.csv is git-ignored).
  - NOT launched; awaiting the coordinator commit and release.
- 2026-10-03 20:54 SCEN-ACTUAL LAUNCHED (coordinator authorisation after commit 482edb4).
  - Pre-launch checks: HEAD 482edb4, package tree clean, code identity fd25e2f7… = HEAD, no scenario_actual, no other fit.
  - Input hashes: launcher bd713324…, table 185e2c79…, manifest v2 72a766c6…, extension CSV 0d6b78ca….
  - Single owner: launcher **PID 2504016** (via exec; scen-b43ef6a-v1.actual.pid), Python child 2504049.
  - Command: `python3.12.exe -B scripts/run_experiment.py --run-dir 'C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1' scen-actual --actual-availability <research\launch\actual_availability.csv> --actual-scaffold <scen-b43ef6a-v1.actual-inputs-v1\extension_manifest.v2.json>`; 4 cases.
  - Logs: scen-b43ef6a-v1.{actual,actual.nohup}.log. Exact-PID watcher on 2504016. No retry on failure. Truth and expert stay unopened.
- 2026-10-03 21:09 SCEN-ACTUAL COMPLETE. Launcher 2504016 exited; scen-actual finished in 883 s, no traceback.
  - `scenario_actual/actual.json`: sha c896a583e4f7c0a82c03a0c07cd9338090fb6390cead6874ee52cfe73bc6a84a.
  - Post-run check: probe research/probes/actual_postrun_check.py (b2bc634b…), summary actual_postrun_check_summary.json (41abb0dc…). Exit 0, 0 problems.
    - accept_record(actual.json) and _accept_frozen pass. actual.json binds frozen.json; code/runtime = selection; exactly the 4 ACTUAL_CASES, all released; fold SHAs match.
    - Each fold: phase scenario_actual; A on map 8965af6d…; scenario_k 0; gate_k = table counts (22 countries: 0 at O 2024-10, 1 at O 2025-02, 2 at O 2025-06); prepared, table (185e2c79…) and manifest v2 (72a766c6…) SHAs; truth "not loaded"; 5,718 rows; outputs hashed; accept_record passes.
    - Predictions: 5,718 unique areas per case; y_true_code all NaN. Persistence source/class/age = the latest lawful label ≤ O (5,716 rows with persistence per case).
    - SD age 4/8/12 at O 2024-10/2025-02/2025-06. The minimum age elsewhere is 0/4/8; some areas in other countries have older latest labels (up to 123–131 months, area-level coverage).
    - Routes per case: local_model 1,912 (h4 2025-06), 810 (h4 2025-10), 841 (h8 2025-06), 183 (h8 2025-10); the rest global fallback or unmapped (208).
  - Output hashes are recorded in the summary JSON (fold.json, predictions, pooled, gate, gate_pairs per case).
  - Truth and expert values are still unopened, pending the coordinator's actual-freeze acceptance. No retune, commit or close.
- ACTUAL FREEZE ACCEPTED by the coordinator. Independently checked: actual.json c896a583…a84a; frozen/selection bindings; exactly H4/H8 × Jun/Oct 2025; fold output hashes; A/map/code/runtime; table 185e2c79 and extension 72a766c6; gate_k 22 × 0/1/2; 5,718 unique keys per case; truth absent. Prediction bytes preserved and never modified. Next: evaluator-only truth release preparation (2025 CS values may now be read, evaluator-only, never fed to fitting or selection).
- Fit-key checker coverage for the actual run. The run now has 682 global records (654 + 28 actual):
  - 4 actual outer globals (int k = 0; origins 2024-10/2025-02/2025-06): VALID for the unchanged coordinator checker (eed66e0d…). Run via a scratch run dir containing only these 4 records plus the byte-identical prepared observations/ledger → research/scenario_fit_keys_review_actual_outer.json (2ab40231…): 4/4, 0 problems. The historical evidence was not overwritten.
  - 24 actual internal gate globals: per-country dict intensity_k and per-country mask dicts. NOT covered, because the checker assumes an int k and list masks. Left as a stated limitation, with no checker expansion.
- Evaluator-only truth release CANDIDATE: research/truth-release-candidate.md. Packet in scen-b43ef6a-v1.truth-release-v1:
  - Files: crosswalk ab629b01…, truth 7ccc336f…, release.json e80f960d… (approved: false; frozen_actual c896a583…), summary b9cf0cdd…; builder ab96e1a7….
  - Source: the raw 2025_2026_FEWSNET.csv (a64ed4bb…), NOT FEWS_2025.csv.
  - Exact-name one-to-one crosswalk. Result: 4,457 admitted (77.9% of October keys); 1,093 unmatched names (DRC 345 all; Ethiopia 645); Kenya 2995/2996 ambiguity excluded (2 rows); 21 with no genuine phase; Uganda has no raw October row.
  - Class counts 1,201 / 1,796 / 1,250 / 210.
  - Decision point: include the 44 admitted allowing-for-assistance rows (proposed; consistent with fews_ipc).
  - June has no source, so it is forecast-only. No expert table, so it gets the NA route.
  - No scores computed; the crosswalk was the probe build and nothing fed to fitting.
- A first build attempt failed while writing its summary (tuple JSON keys) after writing crosswalk/truth/release. That just-created directory was deleted and fully rebuilt by the fixed builder.
- 2026-10-03T22:20:50-04:00 Final bounded read-only trellis-check dispatched: agent a3b178d92faa9e797, 10-minute bound. Scope: scen_report tail/scen_evaluate; the 24 actual per-country internal gate globals vs G2; truth mapping/evaluation outputs; S-row weighting; relevant tests. Report: research/trellis-check-final.md.
- Truth release v2 APPROVED (coordinator) and scen-evaluate RUN.
  - v2 dir scen-b43ef6a-v1.truth-release-v2: release.json 985b5074…a570 (approved by the Codex coordinator under the user-approved spec; binds c896a583…).
  - Truth 7ccc336f… and crosswalk ab629b01… are byte-identical to v1. Source metadata adds the consumed .dbf 2175dc97… and qualifies the .shp pin 3aba66a6…; the .dbf names re-read match the crosswalk on 4,479 matched rows. v1 preserved.
  - Decisions recorded in release.json: 44 assistance-flagged published phases included (flag kept); exact-name + country + DBF-name mapping, with geometry NOT certified; exclusions and NA routes unchanged.
  - scen-evaluate: command recorded in commands.log; rc 0; about 1 s; no expert table. Before/after hashes of 25 actual/frozen/release files are unchanged. evaluation.json 3f9c62b9….
  - Results (research/actual-results.md):
    - Oct H4: 4,457 keys / 20 countries; model .7815 vs persistence .7879; Δ −.0064 [−.0179, +.0003]; vs pooled −.0025.
    - Oct H8: .7230 vs .7879; Δ −.0649 [−.1412, +.0228]; vs pooled −.0006.
    - Oct Study 2: 0 eligible (4,457 lacking exact-origin truth), valid NA.
    - June H4/H8: unevaluable, with the reason recorded. Expert: NA.
  - No fitting or tuning; no commit or close.
  - CORRECTION: evaluation.json sha is 8c43375a5f258a105c55540245fc0deb4284f36c550dbb1638a5709a6d79d427, not 3f9c62b9…. 3f9c62b9… is the shared sha of the identical coverage-only country_h4_2025-06.csv and country_h8_2025-06.csv. Fixed in actual-results.md.
- Final bounded trellis-check (agent a3b178d92faa9e797) terminal (about 257 + 41 s, within bound). Report research/trellis-check-final.md (794c9bee…).
  - Items 1–4 PASS: scen_evaluate/report code; the 24 actual per-country internal gate globals match G2; truth mapping/evaluation outputs; S-row weighting. Item 5 has test gaps only. June and expert are VALID-NA.
  - Coordinator triage:
    - F1: crosswalk hash externally pinned (release v2 plus recount).
    - F2: not triggered (4,457 exact-month keys all matched).
    - F3: the 23 country rows = 22 countries + "unknown country / coverage only"; wording fixed.
    - F4: identical boosters disambiguated by the expected intensity.
    - F5: test debt retained.
    - No product or model change.
- Coordinator actual metric recount PASS: 22,872 forecast rows, 92 country rows, 24 comparator/bootstrap entries, 8 Study results, 0 problems. Byte-exact copies: research/probes/actual_metric_review.py (d73f5e03…) and research/actual_metric_review.json (4d60d56f…).
- FINAL PACKET (documentation only; no code, tests, fits or reruns):
  - research/final-report.md: scientific summary, AC1–AC6 PASS, AC7 PENDING audit, artifact pointers, remaining debt and unresolved provenance (no exact realized fit count, etc.).
  - research/actual-results.md: persistence qualified with saved source ages; practical parity H4 meets / H8 misses (descriptive; CIs are not equivalence); recount cited; 23-row country tables.
  - research/historical-results.md: 138 = 6 × 23 rows.
  - prd.md: completion-evidence addendum.
  - implement.md: all items except the commit/close item ticked on explicit evidence (§5 note: 165 tests at the unchanged identity + three bounded trellis-checks + real recounts; no reruns; audit not run).
  - README: status updated (README-only package diff; code identity fd25e2f7… unchanged).
  - research/truth_release_v2/: approved release.json and candidate summary.
  - research/external_evidence_manifest.json: 27 external files hash-pinned (truth/crosswalk, extension, run records, evaluation outputs); raw data outside the repo.
  - Commit note: these evidence files match .gitignore '*.json' and need `git add -f`: research/truth_release_v2/release.json, research/truth_release_v2/release_summary_v1_candidate.json, research/probes/actual_postrun_check_summary.json.
- Final scope corrections (coordinator review), documentation only:
  - final-report.md (d4dc0c5b…): the conclusion is confined to primary Study 1, with the positive Study 2 k ≥ 1 intervals (H4 k1, H8 k2) noted. The partition 'no measurable benefit' claim is confined to Study 1, with the small Study 2 H8 k2 +.0026 disclosed. Task research is labelled 'final packet, pending commit'.
  - implement.md (0bb9d71f…): stale current-status claims reconciled with the completed phases (actual blocked/admission open → completed via table/extension/v2/evaluate; 'formal trellis-check outstanding' → three bounded reviews run; independent audit still outstanding).
  - Reviewer reports unchanged; AC7 still PENDING. PROGRESS history retained. PACKET FROZEN.

---

## Post-close operational addendum (2026-10-03; not part of the pinned close snapshot)

This addendum was written after the bound close. It does **not** alter the pinned task snapshot (sha256 9e1b2f2b363c3783048e3a733ac57f9aff3f3f010dd5b917871fabcac1620fae) or any scientific evidence. **No audit has passed. AC7 remains pending.**

- **Close.** `trellis-audit --repo <repo> close` was run from the bound session b0e22913-4f4b-430f-8d19-dee565d26eec (term_65ccaebe9009a8) on a clean HEAD b71ce98, rc 0.
  - Close-audit job `e356444e8f195f33fe13cbb7`.
  - audited_sha = completion_sha = b71ce983eda54376f5308c96c27a18980ed21976.
  - base_sha e84368274241e6becc67d5a7e69dc39db23bda66 preserved.
  - Audit run 5c4dede7bc6e44f0835fbde1ad2f473d left active_runs, closed by the wrapper.
- **Archive move (performed by the wrapper; `task.py archive` was not invoked separately).**
  - The task moved from `.trellis/tasks/10-02-exogenous-transition-forecast-design` to `.trellis/tasks/archive/2026-10/10-02-exogenous-transition-forecast-design`.
  - All 97 previously tracked files are present at the archive path. Every one is byte-identical to HEAD except `task.json`, whose status changed in_progress → completed and completedAt to 2026-10-03.
  - 22 tracked evidence files are git-ignored at the new path, because the old path's ignore negations do not apply there, so the archive commit needs `git add -f` for them: research/launch/*.csv, alignment.json, the probe summary JSONs and truth_release_v2/*.json.
  - 5 untracked local files sit in the archive and were never tracked: research/probes/__pycache__/*.pyc (4) and research/probes/stage1_diagnostics_ledger.csv. They are not part of the evidence set.
- **Attempt 1 of the close audit FAILED operationally.** It failed during the TUI bootstrap: account/read workspace routing discovery returned **401 unauthorized** (pane wR:p1 left at a terminal shell, no result). The controller status was "attention".
  - `codex login status` still reported "Logged in using ChatGPT".
  - This is an authentication/bootstrap failure, not an audit verdict.
- **Retry.** The coordinator requested ONE bounded retry of the same job through the wrapper, with the same audited SHA and base. At the time of writing, attempt 2 is `launching` (pane wS:p1, agent audit-e356444e8f195f33-2). No new audit was created, reset or rebound by the executor, and the executor did not retry anything itself.
- **Scientific evidence unchanged.** research/final-report.md and all results and hashes are as committed in b71ce98.

- **Post-close update (2026-10-03).**
  - Attempt 2 of close-audit job e356444e8f195f33fe13cbb7 also failed at the same TUI bootstrap with a 401 (pane wS:p1 left at a terminal shell, no result). Both failed attempts are retained.
  - USER-WAIVED (2026-10-03): the user explicitly authorized skipping the close and spot audits (“准许跳过close audit和spot audit，因为supervisor已经完成了相关内容”). Close-audit job e356444e8f195f33fe13cbb7 failed operationally twice at TUI bootstrap (401 unauthorized; attempts 1 and 2), with no audit result. The waiver is recorded by the coordinator. This is not an audit PASS. The completed science and checks were accepted by the supervisor, not by an independent audit.
  - No further audits or retries.
  - Earlier 'AC7 pending' statements stay as chronology. The current closure status is USER-WAIVED for the audit; AC1–AC6 PASS on the recorded evidence.
  - The pinned snapshot, the scientific results and the evidence hashes are unchanged. The executor made no controller DB edits.
