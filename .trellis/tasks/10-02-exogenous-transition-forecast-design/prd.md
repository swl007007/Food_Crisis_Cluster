# Food-crisis fallback forecasting during FEWS NET interruptions

Status: **approved for execution**, 2026-10-02. User approved the complete final planning summary with “ok implement”. Follow the recorded audit lifecycle before code changes and D7 data readiness before real fitting; protected final outcomes remain isolated. Scientific contracts below are unchanged.

## Goal and user value

Provide credible fs1/fs2 predictions when FEWS NET expert releases and recent IPC observations used by ordinary persistence are interrupted. October 2025 is the primary final case; June 2025 is supplementary. Aim for useful predictive accuracy approaching available persistence/expert performance, without requiring universal superiority. Accuracy means comparison with genuine outcomes, not merely agreement with experts.

## Background and retained evidence

This interruption-fallback claim supersedes the earlier never-labelled-region/spatial-cold-start claim and blanket exclusion of IPC history. Genuinely available older IPC and covariate lags are permitted. PROGRESS.md retains superseded answers only as chronology.

The prior task `10-01-geoxgb-shared-parameter-design` remains **incomplete** (closure `a818560`). Preserve D26–D54 results, including negative findings, under their original conditions. D55 producer `097d98b` has synthetic checks but no released real run and is not an additional arm. Reuse code/maps/checkpoints/keyed predictions only when fitting, selection and input lineage are compatible; changed availability requires recomputation. Relaxing the final performance criterion does not establish that overfitting is resolved.

## Requirements

### R1. Information available at prediction time — design D1, G2, D7

A key is region/target T/horizon H/scenario, with origin O=T−H; fs1=H4 and fs2=H8. Cutoff is origin-month end, including internal historical origins. Retain the fitting-label window [O−59,O), intersected with release eligibility and outage masks; never move it backwards to replace missing labels. H4/H8 are month offsets, not full-month leads before target-month start.

Prefer verified vintages; allow disclosed retrospective reconstruction with documented conservative release rules when archives are unavailable. Revised/latest values are not exact real-time replay. Exclude unverifiable features. Monthly non-IPC inputs use the exact source month implied by one fixed documented source/product lag, including aligned trailing features; residual gaps stay native NaN. Annual inputs use the latest eligible published reference year with source date/year/age. No backward search for monthly nonmissing values, interpolation, lag tuning or missingness-policy search. Static features still require provenance and valid semantics.

Recompute every affected IPC lag, roll, change, distribution, interaction, run and missingness feature after masking. Exact missing lags stay missing; existing latest-observed-phase/age fields may carry lawful older IPC at its true date. Experts are comparators only. No new SMOTE, imputation or synthetic-class path. Hidden releases must not re-enter pooled/local fitting, validation or selection through labels or another region.

For October 2025, H4/H8 origins are June/February 2025; for June 2025 they are February 2025/October 2024. If October 2024 is the latest IPC, October-target H4/H8 origin ages are eight/four months and target age is twelve months. These examples do not verify the actual outage; June-origin covariates cannot enter February-origin predictions.

### R2. Studies, truth and comparators — design D5, G3

Study1 covers all eligible targets. Study2 reuses exactly the same recipe, predictions and decision rule on genuine exact-origin non-crisis rows, including 0→0 and 0→1; report onset recall separately. Recovery is optional. No separate transition training/q/split/threshold optimisation. Rows without exact-origin truth are excluded from Study2; retain them in Study1 when genuine target truth exists. Masked historical origin truth is evaluator-only, never a routing feature. Last-observed IPC is not exact-origin truth. Onset-only persistence recall is mechanically zero and does not establish useful warning.

Use FEWS NET Current Situation truth; never use projections, forward fills or model outputs as truth. June without genuine labels remains forecast/coverage-only. Other IPC/CH sources require a separate comparability contract. Match ordinary or last-available prolonged-lag persistence and genuine same-horizon experts on identical keys, reporting coverage and persistence source date/age. No fs3 expert proxy. October 2024 near/medium forecasts target February/June 2025, not October 2025; a stale-expert carry-forward comparator is optional and separately named.

Country results are descriptive supplements for every eligible country, including negative results: rows/regions/dates, crisis/onset support, computable metrics and matched differences. Show excluded countries/rows in coverage. No winner-country selection, new country significance tests or consistency claims from one 2025 date.

### R3. Metrics, selection and uncertainty — design D4, G3

Retain four-class probability training. Hard crisis = four-class argmax collapsed at IPC>=3; p_crisis=p3+p4/5 is for probability diagnostics. Fixed-four macro-F1 is supplementary. Primary crisis F1 pools equal-weight region/target TP/FP/FN separately by horizon/scenario/study/period; country/month F1 is not the selection average. Model/baseline comparisons use matched keys.

Practical parity is F1(model)−F1(matched persistence)>=−0.02, descriptive rather than an equivalence/non-inferiority claim. For each H, screen A/B by normal-availability development parity, then maximise the equal-weight mean of one-cycle and two-cycle interruption F1; exact ties favour A. Report each interruption score/deficit separately. Undefined comparisons do not qualify. If neither qualifies, report the unmet criterion and stop final-model release for that H; no automatic winner or expanded search. The inherited old PRD R3/D26 crisis endpoint supersedes earlier macro-F1/H12/positive-CI gates.

Use paired country-block bootstrap of fixed matched predictions: 2,000 fixed draws, seed42, full country histories and identical model/comparator multiplicities, 95% percentile interval. Undefined metrics/draws remain NA with reasons; no discarding/redrawing. F1=0 requires positive denominator and TP=0. Numerical CI requires defined point scores, >=2 countries and all 2,000 defined differences; otherwise preserve defined points/support and report valid/undefined counts. No refitting or CI success gate. Disclose limited country count, event concentration, absence of training/selection uncertainty and shared-cross-country-shock guarantees.

### R4. Temporal separation and adaptation — design D2, G2

Stage1 candidates and Stage2 recipe/map selection use 2018–2020 evidence through 2020-12. Freeze recipe/maps before 2021–2024 retrospective Stage3 reporting; origin-relative refitting remains permitted. Disclose previous historical-result exposure. Do not retune using historical final scores or 2025 outcomes/countries. Freeze 2025 predictions before opening evaluator-only final truth.

Historical primary comparisons use common target months across normal/one/two-cycle scenarios within H, requiring origin and both missed cycles strictly after 2020-12. Record early exclusions; no global-only workaround or earlier-map refit. Under on-time February/June/October releases, starts are H4 October2021 and H8 February2022 (10/9 targets through2024-10); release-ledger verification may reduce coverage.

Simulate the latest one/two ordinarily due publication cycles synchronously across all regions; actual2025 uses verified country/product availability. A trains on normal historical inputs subject to outer exclusions. B uses normal/one/two-cycle historical predictor variants, each at w/3 for original weight w. Split original keys first; conserve total weight in global and local fits and count support on originals. Augmentation cannot recover outer-hidden labels, but its historical input masks do not delete otherwise outer-lawful supervised targets.

### R5. Stage1 search and support — design D3, G4

Retain F fitting; S q-scan and parent/child evaluation; C post-freeze diagnostic only; E3 future development-target evaluation. C cannot filter, retrain, trigger retries, gate or contribute fitting/search support, and is not a wholly unseen temporal holdout. Keep the deterministic label-blind S/C split and grouped augmentation roles.

Fix H4 G1 depth3/200 rounds, H8 G4 depth4/400 and local L1 depth1/20; model/confirmation seed42. Each Stage1 child starts from the shared candidate root plus one L1, retaining routing parent as split comparator/fallback; no ancestor-local stack. Stage3 locals similarly extend their fold-global once.

Schedule 648 candidates: A/B × H4/H8 × nine February/June/October2018–2020 targets × three scenarios × r80/r50 × split seeds42/43/44 (162 per strategy/H). Retain hard-F1 q, five binary levels, 80 path-selection rounds and E2 gain strictly >0 versus current parent, parent wins ties. No exposure/error mass means no q candidate. Internal denominator-zero arithmetic cannot establish an undefined gain or a valid reporting score. No extra capacity/threshold/seed grid; record distinct maps, support failures and root-only cases without retries.

Support after masks: local fit >=500 rows/50 areas/6 genuine dates/2 observed four-class categories; Stage1 child validation >=100 rows/20 areas/3 dates; Stage3 gate >=100 rows/20 areas/3 dates including >=3 with eligible local fits. These are partition totals, not per-area/all-four-class requirements or effective-sample guarantees. Unsupported locals preserve evaluation rows via parent/global fallback.

### R6. Consensus, full-pipeline comparison and enablement — design G2, G4

Each strategy gets one general map pooling its H4/H8 and three scenarios (up to324 identities/map). No A/B mixing, month/scenario maps or post-selection fusion. Development uses the common origin-legal pool across scenarios, with candidate E3 target before origin and all evidence lawful. Final maps freeze at2020-12. Retain kNN40, spatial sigma5, eigengap/connected components and seed42; k40 is not40 clusters.

E3 weight = max(0, logit(clip(F_partitioned))−logit(clip(F_own_root))), crisis F1 on matched candidate keys, clip=[1e−6,1−1e−6]. No C/persistence weights or scenario balancing. Genuine undefined metrics are ineligible with reasons; missing/corrupt artifacts are errors. Distinguish no-prior, no-scorable and complete-all-zero global fallback. Never force uniform weights or claim fallback proves partition benefit.

Select A/B on 72 complete-pipeline development forecasting folds: two strategies × two H × three scenarios × six February/June/October2019–2020 targets. Keep legal global-fallback folds; averaged candidate E3 scores cannot substitute for system-level predictions.2018 contributes candidates/diagnostics.

Stage3 local enablement requires crisis-F1 gain strictly >0.01 over fold-global on matched historical gate keys plus support; equality fails. Replay the same missed-cycle intensity at each internal origin, retaining outer exclusions and selected A/B. Use the latest six globally lawful genuine dates U<O, internal V=U−H; no local date cherry-picking. The outer-known map may organise historical replays, per prior D18, with conditional map-selection bias disclosed. Unsupported dates retain global routes; current local fit must also qualify. Retain fold-global predictions as the same-input diagnostic without an extra arm.

## Out of scope

No required phase-free/covariate-only/spatial-cold-start ablation, H12 arm, binary-training comparison, dedicated transition model, winner-country discovery, broad tuning, forced partition benefit, new tracking service, or automatic D55 adoption. Optional recovery/stale-expert analyses are not mandatory initial deliverables. No consistent superiority claim or fabricated accuracy for unlabeled June targets.

## Acceptance criteria

- **AC1 (R1/R4):** Source/schema/key/mask manifests pass design D7 before real fits; hidden IPC/future releases cannot affect permitted predictions and raw sources remain unchanged.
- **AC2 (R4/R5):** Original-key role isolation, conserved augmentation weights, support floors and immutable shared-root prefixes pass focused checks; retain F/S/C/E3 gaps and failed-support cases.
- **AC3 (R3/R6):** Reconciled 648-candidate/72-development-fold ledgers implement lawful consensus and the exact A/B selection/stop rule, with no silent missing-artifact fallback.
- **AC4 (R4/R6):** Frozen recipe/maps and origin-specific inputs support the predeclared historical and2025 predictions; historical enablement obeys its own >0.01 gate, independent of final−0.02 tolerance.
- **AC5 (R2/R3):** Saved keyed predictions reproduce Study1/Study2/country metrics, matched comparators, uncertainty, support/exclusions and June unevaluable status where appropriate.
- **AC6 (Background/R5):** Preserve old incomplete-task/negative evidence and report generalisation gaps; completion means faithful finite execution/reporting, not guaranteed scientific success.
- **AC7 (all):** Final planning summary approved subsequently, then approved plan committed and actual Claude/audit identity verified before implementation; close only with required evidence and accepted audit status.

## Technical dependencies and execution limits

Latest execution clarification (user, 2026-10-03): sources are largely manually verified; prioritise continuation over independent source re-verification. Accept user-attested source identity, preserve uncertainty disclosures and retain concrete key/temporal-leakage checks. Do not let unresolved final-2025 mapping or expert coverage block unrelated historical development. See design D7 and implement.md for this narrowing of the readiness gate.

No user-owned planning question remains. Deferred source facts are mandatory pre-fit work under design D7: byte hashes/admin joins, genuine CS and actual release/outage calendars, fixed source-family lags/vintages, climate producer lineage, exact schema/keys and frozen Windows numerical environment. Unverifiable covariates are excluded; missing IPC calendar or ambiguous administrative mapping blocks dependent fitting. Final truth values remain isolated until frozen predictions; invalid final labels mean unevaluable reporting.

Conservative fit budget:648 candidates/<=40,824 Stage1 fits;72 development, <=57 historical and <=4 actual2025 forecasting folds. With N frozen maximum distinct mapped fitting areas, total <=40,824+931×(1+floor(N/50)); compute N before launch. Sequential initially, immutable scratch runs outside Dropbox, exact-identity cache reuse only, no grid expansion. Stop on leakage/key/identity/schedule errors. Exact commands follow the minimal implementation; no model command is executed during planning.

The package guideline `.trellis/spec/backend/local-forecasting-experiments.md` contributes keyed snapshots, source identities, immutable runs and evidence reconciliation. Its Ethiopia85/88-feature schema, median imputation/SMOTE, residual-expert lags, macro-primary metric and old coverage/NA gates do not govern this task; this approved GeoXGB design controls those differences.

## Repository evidence and availability gaps

- Existing calendar conventions: fs1=4 months and fs2=8 months (`src/utils/lag_schedules.py:45–55`; `Step3ExpertCorrectionExperiment/step3correction/expert.py:6–11,45–46`). Expert fs1 uses near projection at T−4; fs2 uses medium projection at T−8. These joins do not prove actual publication availability. Do not inherit that legacy module's raw-missing-to-zero convention.
- Calendar implications only: target 2025-02 maps to fs1 origin 2024-10 and fs2 origin 2024-06; target 2025-06 maps to 2025-02 and 2024-10; target 2025-10 maps to 2025-06 and 2025-02. If the latter two publications are absent, October lacks the ordinary same-horizon experts. October 2024 near/medium projections map to February/June 2025, not October 2025. Carrying such a projection into October is a proposed stale-expert carry-forward comparator with a different original target, never a same-horizon expert forecast. Actual country/product availability and whether October 2024 is the last available release remain unverified.
- Prior read-only metadata probe of external `Outcome/FEWSNET_IPC/FEWSNET.csv` found dates 2009-07 through 2024-10, no 2025 dates and no publication/vintage/as-of field. This file alone cannot verify outage coverage or supply 2025 truth. Other truth sources and actual outage/resumption dates remain to verify. No 2025 outcome values were inspected.
- New metadata-only discovery: external `Outcome/FEWSNET_IPC/2025_2026_FEWSNET.csv` has 5,573 records with reporting month 2025-10, scenario CS, Current Situation, Published, covering 2025-10-01 through 2025-10-31. `FEWS_2025.csv` has 4,481 rows dated 2025-10. Neither inspected candidate CSV has 2025-06 records. Counts are not usable-label or unique-evaluation-unit counts; source-to-admin mapping, label validity and publication-vintage provenance remain unverified. See [research/2025-outcome-metadata.md](research/2025-outcome-metadata.md).
- `src/preprocess/preprocess.py:151–156,226–259` has default-off gap filling that copies older IPC labels into 2025-02/06. The implementation can use any earlier labelled month despite a one-month-cap comment. Such values may represent labelled persistence predictions, **never evaluation truth**. Missing targets may otherwise be dropped unless prediction-target retention is explicit (`:179–197`).
- Secondary covariates are external conflict/nightlight/market/price/etc. variables (`scripts/create_feature_exclude_datasets_from_unadjusted.py:109–138`). Existing feature alignment uses exact-origin rows and EVI lags (`FEWSNETFourClassBaseline/src/feature/fourclass_features.py:93–103`); mixing fresher covariates with older history needs an explicit new as-of contract. Source month alone does not establish publication availability, and revised data need vintage treatment.
- Existing GeoXGB schema has 162 columns: 75 IPC-history, 28 static, 41 dynamic, 15 covariate lag/aggregate and 3 calendar. Dropping the 75 leaves 87 candidates, not a selected new schema. IPC history includes levels, crisis indicators, changes, distributions, missingness/age, runs and interactions. D51 retained history/calendar and removed other covariates: it was not a phase-free ablation.

Evidence: `PIPELINE_WORKFLOW.md:6–8` separates Stage1 2018–2020 from Stage3 2021–2024. The previous task's `experiment-plan.md:30,89,136` specifies selection through 2020-12, origin-relative 59-month fitting pools, and frozen subsequent evaluation. These are reusable starting contracts, not an automatic release of old execution scope. The trade-off is less recent data for choosing the recipe in exchange for compatibility and preserved temporal evaluation; later fitting can still use recently published observations under the frozen rule.
