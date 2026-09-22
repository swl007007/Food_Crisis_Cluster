# IPCCH population-history and pooled XGBoost objective comparison

Status: v0.9 final review. D1-D21 are approved; the exact technical inventory
is awaiting the user's final confirmation. Task remains planning. No product
code, experiment, commit, wrapper start or task closure is authorized this turn.

## Goal and authority

Test whether rich historical population distributions improve IPCCH crisis
forecasting, and whether learning persistence errors or continuous P3+ shares
improves on direct binary XGBoost. Preserve generality by fitting pooled IPC+CH
models and using one development-selected main method across horizons.

This PRD defines requirements/acceptance; technical-contract.md fixes executable
formulas and selection rules, feature-schema.json the ordered predictors, and
candidate-configs.json the finite search. design.md and implement.md describe
implementation and verification. research/decision-log.md retains D1-D21 and
all historical evidence anchors; its formerly open wording is chronological.
New technical defaults are proposed, not retroactively labeled user approvals.

## Confirmed evidence and limits

- Current IPCCH features use binary history, not continuous phase history:
  IPCCHGeoRFExperiment/prepare_data.py:521-532. Target QC retains normalized
  shares and exact binary truth (:236-288,292-298,352-369).
- Original same-key XGB/persistence F1 at 1/3/6/12m is
  .678929/.681392, .680740/.676817, .667058/.671861, .678704/.675917.
  Source: original reports/main/metrics.csv and IPCCHGeoRFExperiment/README.md.
  Archived geographic-baseline results do not establish a spatial-sharing gain.
- On original main evaluation rows, only 34-38% have at least six prior valid
  observations; missing lags must not remove samples. Development years have
  usable history at all four horizons. Exact support evidence is preserved in
  research/decision-log.md; these counts are not new model results.
- 2023-2025 outcomes have been inspected. This is a retrospective design, not an
  untouched holdout. Source-month alignment does not establish publication-time
  availability. Preserve upstream feature provenance limitations. A FEWS NET
  negative feature-search result is not an IPCCH impossibility result.

## Requirements

- R1 (D1): Primary truth remains normalized P3+ population share strictly >.20.
  Reuse the pinned IPCCH source, original target validation and country mapping;
  do not replace truth with official overall_phase, interpolate missing outcomes,
  change the phase5-only fill or normalization rules, or modify original sources.
- R2 (D2-D3,D18-D20): Append the approved rich history to the original93 schema.
  Exactly eight distribution series, six observed slots, elapsed-time changes,
  finite trailing/all-history summaries, distribution-shape and observed crisis
  features; 468 added columns, 561 total. Follow technical-contract.md and the
  ordered feature-schema.json. Preserve missingness and source-date provenance.
  No feature subset search after final outcomes, automatic column dropping,
  additional source, spatial neighbors, identifiers or invented monthly states.
- R3 (D11): Every historical row uses its own origin o=T-h. A refit at O uses
  labels from [O-35,O], same horizon. Feature history can extend before that
  fitting window but never beyond the row's o. No target-first history truncation.
  Fitting and evaluation keys must be disjoint within each fold.
- R4 (D9-D10,D14): Fit the six learned arms listed below. Five matched arms use
  identical ordered fitting keys with >=1 valid history at own origin; fullpool
  XGB uses all otherwise eligible fitting keys. Rich arms share identical X
  information. Primary comparison/selection uses common history-available test
  keys; missing derived values never exclude rows. No-history combined binary
  predictions use the same fullpool XGB fallback and are reported separately.
- R5 (D4-D8): Correction trains one pooled XGB on e=1[y!=b], permits both flip
  directions and retains raw error score/b. Share regression uses squared error,
  equal row weights, clipped shares for decisions/errors and retained raw outputs.
  Also report fixed predicted-share>.20 and continuous MAE/RMSE. No claim that
  mean predicted share equals crisis probability or official overall_phase.
- R6 (D12-D13): Pool all IPC/CH areas; separate models by h=1/3/6/12. Development
  targets are 2020-2022. Main dates remain h1 2023-02..2025-12, h3 2023-04..2025-12,
  h6 2023-07..2025-12, h12 2024-01..2025-12. No geographic partitions or country/
  IPC-CH-specific models or thresholds. Their results are stratified reports.
- R7 (D6-D7,D15): Give each learned arm six declared configurations per horizon,
  using candidate-configs.json. Select configuration and two history-state
  thresholds jointly through pooled development F1; no per-subgroup/month F1
  optimization, test tuning, separate calibration, early stopping, SMOTE or
  repeated-seed search. technical-contract.md fixes candidate cutoffs and ties.
- R8 (D16): Before final predictions, select one of rich direct/correction/share
  XGB by equal mean four-horizon development delta F1 vs persistence. Freeze
  primary family, all per-horizon parameters/thresholds, inputs/schema/code and
  choices. Final rolling refits may use newly available past labels but never
  change these choices or mix methods using final scores. Report every arm.
- R9 (D17,D21): Stable gain for each required baseline needs mean delta F1>0,
  paired95% lower>0, all four horizon point deltas>=0 and all three leave-target-
  year-out mean deltas>0. No +.01 hard floor or separate per-horizon significance
  requirement. Share country draws across methods/horizons; use2000 valid draws,
  seed42 and frozen missing-draw rules. No pseudo-independent row bootstrap.
- R10 (D17): Distinguish prediction gain vs RF/persistence, formulation advantage
  additionally vs matched/fullpool direct XGB, and rich-vs-binary-history input
  gain. Apply required comparisons conjunctively, retain failures and all subgroup
  results; do not choose the successful claim/primary family after seeing final
  scores. A negative complete study is acceptable completion, not permission for
  another model search. Undefined required evidence is incomplete.
- R11 (D9,D14-D15): Maintain pinned runtime, equal real-row fitting weights,
  native XGB NaNs, training-only RF imputation and width. Record constants,
  empty folds, failures, effective estimator parameters and source/code identities.
  The proposed exact fallback/imputation rules are in technical-contract.md;
  do not substitute root helpers that reintroduce SMOTE or IPCCH partition logic.
- R12 (D15): Use a fresh IPCCHPopulationHistoryExperiment/runs/<run_id>/ root.
  Development upper bound5184 fits and main upper bound732; pilot is included,
  not another experiment. No required partial2026 run or uncontrolled retries.
  Log actual resources and all failed/skipped attempts; preserve existing runs.
- R13 (D1): Deliver independently recomputable keyed evidence, not just headline
  tables. Source/QC ledger, feature schema and dated histories, ordered training
  membership, candidate out-of-time predictions, threshold scores, freeze identity,
  final probabilities/hard decisions/routes, bootstrap draws and reproduction
  commands must be retained. Promote reviewable evidence before completion audit.
- R14 (D1): Opus5 implements only after this final review and a separate execution
  instruction. Follow the enrolled audit-wrapper lifecycle; respect existing gates,
  commit the approved spec before freezing base_sha and commit completed evidence
  before close. An archived task, successful commit or launched reviewer is not
  audit acceptance. Do not use native start/archive to bypass the controller.

### Required model inventory

| ID | Features | Fitting labels/support |
|---|---|---|
| binary_history_xgb | original93 | y, matched |
| rich_direct_xgb | rich561 | y, matched |
| correction_xgb | rich561 | e, matched |
| share_xgb | rich561 | q3, matched |
| rich_rf | rich561 | y, matched |
| fullpool_xgb | rich561 | y, full pool |
| persistence | latest b | no model; history-available keys |

## Acceptance criteria

- A1 (R1): Source/hash/keys/QC/country gates pass; exact binary labels and
  normalized regression targets reproduce the original source contract. Invalid
  history never becomes non-crisis. No raw data or original experiment changed.
- A2 (R2-R3): Original93 and rich561 column names/order match the JSON inventory.
  Hand-checkable sparse histories verify six-slot ordering, elapsed months,
  window edges, zero-P3+ ratios, exact-.20 labels, events, missing support and
  alias deduplication. Each feature's latest source month<=its row origin.
- A3 (R3-R4): Every fold's ordered train/test keys and source-role masks reproduce;
  five matched arms have identical training/evaluation support, and fullpool
  supersets are explicit. No row disappears due to missing engineered features.
- A4 (R4-R5): Error labels, score orientation, both flip directions, equality at
  cutoffs and no-flip options reproduce. Missing b is not corrected or zero-filled.
  Fallback binary outputs are identical across combined methods; no fallback
  probability enters a share-error metric.
- A5 (R6-R8): Complete schedules, six candidate inventories and selection ledgers
  reproduce; time-causal development predictions use2020-2022 only. Selected
  parameters, thresholds and one primary family reconstruct before final work.
  No final metric, partial2026 sample or subgroup result selected these choices.
- A6 (R9-R10): All F1s/deltas recompute from stored confusion counts and common
  keys. Shared country multiplicities reproduce mean-delta intervals and reject
  logs; all leave-year-out results and conjunctive verdicts reconstruct. Suppressed
  or undefined evidence is labeled incomplete; all method contrasts remain visible.
- A7 (R11): Required environment and actual fitted settings match the declared
  inventory; RF fills use only matched training X. Empty/single-class/constant
  routes are correct, declared and tested. No pseudo rows, SMOTE or hidden models.
- A8 (R12): Actual candidate/fold counts reconcile to the upper bounds and pilot
  reuse ledger, with immutable fresh results and no modifications outside scope.
- A9 (R13): Independent reporter replay from saved predictions and an independently
  selected first/last nonempty main-fold replay per horizon reproduce selected
  predictions under the pinned environment. Saved XGB/RF model identities, inputs,
  masks and transforms explain each prediction; console-only evidence is rejected.
- A10 (R14, lifecycle verification after close): Final handoff links acceptance
  evidence, commits, runtime/artifact manifests and the actual audit status/result.
  Prior gates are resolved by accepted re-audit before ordinary wrapper start;
  completion queues an independent Codex audit. A1-A9 are pre-close deliverables;
  A10 never demands an audit result before the operation that queues that audit.

## Out of scope

GeoRF/GeoXGBoost partitioning or parameter sharing; four-output phase distribution
models; new external covariates; IPC/CH-specific models or tuning; expanded fitting
windows; arbitrary feature/model search; causal/equally-spaced-welfare interpretation
of severity indices; proven real-time publication availability; repairs to unrelated
active work; model execution or publishing during this planning turn.

## Final-review choices and execution gate

D1-D21 resolve the research choices. Final review must adopt or amend the concrete
technical defaults in technical-contract.md, feature-schema.json and
candidate-configs.json: original93 unchanged, rich561 expansion, six actual search
configurations, bounded threshold grid/ties, uncalibrated scores, .5 no-history
fallback, original RF max_plus behavior and explicit constant-target handling.
These defaults are not claimed individually approved in the previous grill.

Planning snapshot: HEAD2dae94e; no active audit run; controller stopped; old
ipcch-ch-gate-ablation job b0945b254eac1c8756430272 has an open major gate.
This state is recorded evidence, not a prediction of later state. Recheck before
execution, follow the prior task's repair/re-audit lifecycle and never relabel this
new research task as remediation. Other-session working-tree changes must remain
untouched. No new wrapper start or audit registration is performed during planning.
