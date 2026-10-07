# IPCCH fixed-map yearly pooled regional XGB

## Goal

Determine whether frozen IPCCH spatial partitions provide useful regional XGB adaptation when fitting/testing follows an explicitly selected yearly pooled protocol. Separate the geographic increment from changes to the pooled baseline.

The user explicitly identifies GeoXGB as the main model for this experiment; this is not another MLP experiment.

## Status and authorization

- User authorized creation on 2026-10-07 and requested: create a Trellis task, grill, write a specification, then execute.
- On 2026-10-07 the user approved the complete final summary: “确认，可以进入执行。” Design v1.0 and the execution plan are approved, including Claude Opus 5.5 1M execution under Codex supervision and the P0 checkpoint before formal fitting. Lifecycle start follows the planning commit and live executor verification.
- No experiment package, source data, saved model or existing result has been changed.

## Confirmed context

- The comparison note is `docs/notes/2026-10-07_note_IPCCH两套pooled_XGB基线差异.md`.
- A later split2024 sensitivity also exists; R7 identifies the selected map source so these frozen maps are not treated as interchangeable.
- P6 pooled uses rich561 features, population-share-derived truth, bounded isotonic projection, monthly refitting on the previous 36 calendar months, equal fitting weights, and shallow selected global XGB recipes. Sources: `IPCCHGeoXGBExperiment/config/experiment-contract.json`, `ipcch_geoxgb/targets.py`, `stage3.py`.
- The latest adjacent-repository global experiment is `../IPCCH/.trellis/tasks/archive/2026-10/10-06-global-origin-safe-climate-idp/`. Its annual model for target year Y fits all eligible historical labels through Jan(Y)−max(H,1), with age weights 0.5**(age/24) anchored at Jan(Y)−H. It uses different features, reported-phase truth, raw-score threshold decoding, deeper XGB models, and H0/H3/H6/H12 evaluation in 2022–2025. It is not a feature-only alternative to P6.
- Previously inspected evaluation years are exploratory. Frozen maps must not be treated as available before their learning cutoff; annual fitting and historical gate replay require explicit timing rules.

## Requirements

- R1: Reuse explicitly identified frozen IPCCH memberships; do not learn or alter partitions during this experiment.
- R2: Define a matched yearly pooled baseline and regional system on identical evaluation keys and truth. Attribute geographic benefit only to their paired difference.
- R3: User adopted protocol-only transfer: fit once per annual test block, use all eligible history through its safe cutoff, and apply 24-month half-life age weights. Retain P6 rich561 features, population-share-derived truth/QC, bounded isotonic projection and unrounded >=0.20 decoding, and frozen P6 XGB recipes. Do not import adjacent-repository deep features, reported-phase truth, raw-score decoding or deeper XGB recipes. Annual block boundaries follow R9, historical model replay R11, and weights R13.
- R4: Freeze map-to-horizon mapping, label/QC and feature semantics, annual fitting cutoff/window/weights, global and regional learning recipes, gate/fallback, test calendar, metrics, reproducibility and computation limits in the design.
- R5: Preserve original experiment packages and results. No scientific choices may be retuned after seeing this experiment's evaluation outcomes.
- R6: Produce PRD/design/implementation plan and curated spec/research contexts before release. Bind execution to the agreed executor and current lifecycle authorization; do not inherit a task-specific audit waiver or executor identity silently.
- R7: User adopted the original P6 frozen maps from `IPCCHGeoXGBExperiment/runs/p6-formal-20261004b`, learned through 2022, for H1/H3/H6/H12: 3,264 mapped areas each, 9/7/6/9 terminal regions. Preserve membership and horizon assignment exactly; exclude the split2024 maps.
- R8: GeoXGB is the primary regional model. Retain its four scalar global XGB regressors and fresh local continuation from each matching immutable global root, with the frozen P6 global/local recipes. No MLP, separate neural residual network, or local-to-local ancestry. Compare regional GeoXGB with its matched yearly pooled XGB baseline; annual fitting, validation and route timing follow R9–R11.
- R9: User adopted the original evaluation calendar with a partial first-year block per horizon: H1/H3/H6 start at 2023-02/2023-04/2023-07, and H12 at 2024-01. Anchor each first block at that scheduled target month A, with fitting cutoff/origin O_block=A−H=2023-01; do not shift the anchor to the first nonempty observed month. Subsequent calendar-year blocks anchor at January and fit through January−H. Fit the current global/local models once for a block and reuse them throughout it; each predicted row retains features built at its own lawful forecast origin. Retain original missing-truth folds and matched evaluation keys. Report 2026-01..04 separately. This explicit initial-block adaptation keeps current-model fitting cutoffs after the map-learning period; it does not establish historical gate replay as free of map-selection lookahead.
- R10: User adopted one gate decision per region at the annual block origin, held fixed throughout that block. Only historical outcomes available then may enter the decision; do not refresh the gate with intra-block outcomes. Retain original fitting/validation support conditions and projected/decoded crisis-F1 gain strictly greater than 0.01 against the matched pooled XGB. Statistical support/gain failure falls back to pooled; technical failures must not be hidden as statistical fallback. Historical model replay follows R11.
- R11: User adopted annual historical model replay for the gate. Select the latest six globally observed target months U strictly before O_block. Predict each U with a global/local pair fitted for its historical annual block, using only labels through that block's safe origin; never use the current model to score labels it may have fitted. Reuse a pair across all validation dates in its block. Use the R9 partial first-year block for dates that fall within it; dates outside the adopted evaluation blocks use the corresponding calendar-year January anchor. Apply the same full-history/24-month decay fitting protocol and frozen P6 recipes. Historical regional support failure retains every validation key and uses the corresponding historical pooled prediction on the local side. Preserve the original minimum of three successfully locally supported validation dates: count distinct target months, not unique fitted models, and report both counts to avoid implying independent refits. This is conditional replay on maps/recipes selected with through-2022 development information; it is not independent validation of map/recipe selection or proof that the system was deployable at those earlier dates.

- R12: User adopted ungated current-local diagnostics. Fit and predict each mapped region with eligible current fitting support and evaluation keys regardless of its annual gate decision. Compare its local XGB continuation with matched yearly pooled XGB on exactly the same supported keys. Keep gated GeoXGB versus pooled as the primary full-cohort contrast; diagnostic scores cannot select recipes, change maps, relax gates or alter routing. Report eligible counts and support/gain/adoption groups, retain unsupported rows in the full-cohort pooled fallback, and include additional required local fits in the pre-execution inventory with exact-model reuse.
- R13: User adopted identical unnormalized global/local decay weights. For fitting target month t at block origin O, set w_t=0.5**((O−t)/24) on the full lawful global fitting pool; each local subset takes those exact row weights for all four targets. Do not normalize to mean one, rescale regional totals, or re-anchor ages to the region's latest observation. Retain original count-based support gates. Record weight sums and effective sample size diagnostically; smaller weight mass can interact with unchanged XGB regularization and min_child_weight and does not authorize retuning.
- R14: Retain the original requested metric panel and NA semantics: four-class accuracy/macro-F1; binary accuracy/F1/precision/recall/F2; projected q3 R², with raw q3 R² diagnostic. Match all comparison keys/truth. Report main and 2026 separately; use E_all for the primary geographic contrast, E_persist for persistence and the supported local cohort for ungated diagnostics. Preserve paired country bootstrap (2,000 draws, seed42) for main gated GeoXGB minus pooled and minus persistence crisis F1. Report original-P6 comparisons as bundled protocol sensitivities, not isolated effects of any one changed ingredient.
- R15: Bound execution to the independently enumerated 21 global plus 160 local quartets (724 scalar fits), with original frozen recipes and exact reuse. P0 must reproduce the inventory, verify source/runtime identities and pass synthetic weighted-fit/replay checks before supervisor release of scientific fitting. Missing/corrupt artifacts or numerical failures stop with durable incomplete evidence; they cannot trigger silent refitting or tuning. Deliver keyed predictions, matched reports, zero-fit saved-model replay and a final output inventory. Positive improvement is not required for completion.

## Out of scope and interpretation

No map relearning, Stage2, MLP, SMOTE, learned edges, feature redesign, added hyperparameter/seed search, or changes to the original packages/runs. This experiment changes annual refitting, history length, decay and annual gate timing together. Already-viewed test years and historical replay conditioned on development-selected maps limit claims about independent generalization. The within-protocol G−P comparison measures the geographic increment under this fixed design.

## Delivery and execution boundary

Design: `design.md`; ordered plan: `implement.md`; workload and reuse evidence: `research/findings.md` and `research/fit-enumeration.json`. Both context manifests bind these artifacts and applicable isolation/provenance guidance. The approved executor is Claude Opus 5.5 1M with Codex supervision; live identity must be verified before lifecycle binding. P0 implementation/preflight is followed by a supervisor checkpoint, then one bounded formal run and report/replay. Technical verification does not authorize scientific scope changes.

## Acceptance Criteria

- [x] Transfer scope adopted: yearly full-history fitting plus 24-month half-life weighting, with P6 features/targets/decoding/model recipes retained.
- [x] Original P6 maps and H1/H3/H6/H12 are adopted; GeoXGB is the main model with its original shared-global-root local continuation.
- [x] Annual evaluation blocks adopted: original partial first-year starts, subsequent January anchors, and separate 2026 supplementary reporting.
- [x] Gates are decided at each block origin and fixed within the block, retaining original support rules and strict F1 gain >0.01.
- [x] Historical gate predictions use annual model replay on the latest six observed pre-origin months, with frozen-map conditioning explicitly disclosed.
- [x] Ungated current-local predictions are included on the supported mapped cohort, without changing the primary gated system.
- [x] Global/local decay weights use the same block origin and absolute scale, without local renormalization.
- [x] Scientific choices R1–R13 are resolved; complete design/plan specifies cutoffs, map-conditioning limits, comparators and delivery checks (R4, R6, R14–R15).
- [x] User approves the complete final planning summary, including execution responsibility and P0 checkpoint (R6); approval quoted above.
- [ ] Implementation passes relevant synthetic checks; preflight reproduces 15 current blocks, 724 scalar fits, frozen inputs and exact supported/evaluation counts (R9–R15).
- [ ] Main full-cohort keys are 17,322/16,919/16,413/14,087 for H1/H3/H6/H12, and 2026 has 4,092 per H; all 138 calendar folds, including 12 empty folds, remain represented (R2, R9).
- [ ] Main ungated local diagnostic keys are 9,943/9,866/9,522/7,904, with 1,829 per H in 2026; statistical fallback retains full-cohort rows and gates remain fixed within each block (R10–R12).
- [ ] Approved runs deliver keyed predictions, fitted-model/weight provenance, matched metrics, bootstrap evidence, passing saved-model replay and complete final inventory, including negative outcomes (R5, R13–R15).
- [ ] Results and limitations are documented for discussion without attributing pooled protocol changes to geography (R2, R14).
