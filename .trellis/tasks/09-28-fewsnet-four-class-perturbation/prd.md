# FEWS NET four-class forecasting baseline

Status: v0.9 approved, 2026-09-28. User confirmed D1-D23, the exact feature
expansion and all technical defaults. Audit registration/controller setup is
authorized; execution is handed to the bound Claude executor. Task remains
planning until that executor commits the approved plan and starts via the wrapper.

## Goal and authority

Build a reliable four-class baseline from the minimal three-stage GeoRF release,
comparing spatial partitioned RF with pooled RF, persistence and FEWS NET experts.
Include causal historical-outcome engineering. This is no longer a strict
single-variable perturbation of the old binary experiment (D21).

This PRD governs requirements and acceptance. feature-contract.md plus
feature-schema.json define the approved exact predictors; design.md describes
stage integration and technical defaults; implement.md sequences implementation
and evidence. research/decision-log.md preserves D1-D23 and all earlier source
anchors losslessly. research/release-and-integration.md records inspected release,
source identities, current code behavior and inherited pitfalls. Historical open
wording in that chronological record is not a new pending decision.

## Requirements

- R1 (D1): Four ordered classes1,2,3,4or5; map raw4/5 to class4 and display
  it as “4或5”. Preserve raw phases outside X. Missing is never a class.
- R2 (D2,D6): Use fixed-four macro F1 in all stages. Per-class F1 is
  2TP/(2TP+FP+FN), with zero denominator scored0; always average four classes.
  Aggregate counts on the scoring cohort before F1, not mean child/month F1.
  Report class supports/F1, confusion matrices, accuracy and category-step error.
- R3 (D3-D5): fs1/fs2/fs3 mean4/8/12 calendar months. Expert at target T is
  same-area near(T-4) for fs1 and medium(T-8) for fs2, missing if source/value
  missing. No fs3 expert proxy. Persistence is observed phase at exact T-H;
  do not backfill an older observation. Apply R1 consistently to all baselines.
- R4 (D4,D10): Stage1 uses2018-2020 monthly schedules for all three horizons;
  Stage2 pools only these candidates. Partition learning/selection information
  cutoff is2020-12. Stage3 target schedules: fs1 2021-05..2024-12, fs2
  2021-09..2024-12, fs3 2022-01..2024-12. Empty targets are recorded, not filled.
  First supported months expected from the inspected calendar are2021-06,
  2021-10 and2022-02; source preflight establishes actual support.
- R5 (D11,D14): Preserve train labels in [O-35 months,O), O=T-H:35 calendar
  months, excluding origin. Retain within-area random validation, original ratio,
  seed and small-sample rules; this is not temporal internal validation. Repeated
  partition search can overfit this validation, so Stage1 gain is not final proof.
- R6 (D18-D19): Freeze explicit static/dynamic/calendar roles. Align every
  historical fitting row at its own origin using calendar keys on the complete
  monthly scaffold before target filtering. No row-count lags, cross-area rolling
  or global variation-based feature-role discovery. Fit each estimator's imputer
  only on its real fitting rows; preserve the fill formula and matched transform
  through checkpoint inheritance, prediction and fallback. No future/validation
  observations or artificial rows enter imputation statistics.
- R7 (D20,D22): Include eight exact-origin history features (phase/binary at
  O,O-4,O-8,O-12) plus the four approved blocks: changes, observed12/24/36-month
  summaries, event ages/observed runs and limited origin-state interactions.
  No outcome interpolation or inferred continuous states. Frequencies count
  assessments, not people. Both RF arms use the same frozen feature recipe;
  final outcomes cannot select features. Exact expansion is frozen in the
  feature contract; no external data or exhaustive feature search is requested.
- R8 (D23): Area ID is only a key/geometry/group/routing field, never an RF
  predictor or replaced with an encoded ID. Geographic covariates remain.
- R9 (D8): Preserve one joint spatial scan with four normalized columns.
  On parent validation rows let D_gk=2TP_gk+FP_gk+FN_gk, D_k=sum_g D_gk;
  Y_gk=D_gk/(4D_k), A_gk=2TP_gk/(4D_k). D_k=0 gives zero columns.
  C=Y-A; B_gk=sum_g C_gk *Y_gk/sum_g Y_gk, zero if exposure is zero.
  Feed floating masses to the existing scan and combine class scores into
  one candidate. This surrogate is neither exact subset macro F1 nor a
  calibrated significance test. Zero total error gives no scan candidate.
- R10 (D7): Compare parent and inherited child/parent combinations on identical
  complete parent validation keys. Accept only strict macro-F1 gain>.01 at
  every depth; parent wins ties. No threshold search or legacy class1 gate.
- R11 (D9,D15): Stage2 uses max(0,logit(F_partitioned)-logit(F_pooled)), clipping
  each macro F1 to[1e-6,1-1e-6] on identical Stage1 scoring keys. Preserve the
  distinction between internal split validation and monthly held-out scores.
  A complete successful candidate ledger with all-zero weights yields no
  effective consensus; partitioned Stage3 reuses pooled predictions exactly.
  Missing candidates/failed scores are incomplete, never this valid fallback.
- R12 (D12,D17): No SMOTE. Stage1 appends one zero-feature fitting row per
  encoded class only after real-data imputation; synthetic rows never become
  validation support or source observations. Stage3 uses real fitting rows only.
  Local partitions below50 rows or2 observed classes use the pooled RF;2/3-class
  local fits remain allowed. Align probability axes to four classes via classes_,
  absent class probability0. Save all model/imputer routes and class support.
- R13 (D13): Main fs1/fs2 comparisons use identical valid truth/expert/persistence
  keys for four arms; main fs3 uses truth/persistence keys for three arms.
  Supplementary fs1/fs2 compares both RF arms and persistence without requiring
  expert availability. Report exclusions and coverage. Baseline missingness
  does not restrict RF fitting membership; model failure cannot shrink cohorts.
- R14 (D16): Per-horizon paired delta macro F1 and95% intervals use2000 country
  bootstrap draws shared across models, preserving country-level dependence.
  Condition on fitted predictions, no refit bootstrap. No expert contrast at fs3.
  Do not infer robust superiority from ranks alone or generalize to future years.
- R15 (D21): Keep an isolated package/run derived from the verified release.
  Original sources/results are immutable. The study is a four-class baseline
  with explicit preprocessing/feature changes, not causal attribution of gains
  solely to changing the binary target. No correction layer/XGBoost requested.

## Acceptance criteria

- A1 (R1,R3,R15): Source/release/runtime identities, unique keys, raw phase
  mapping and calendar joins reproduce. Missing baselines stay missing; expert
  record shifts and fs3 proxies are absent. Source truth/history agreements
  and any source coverage differences are explicitly checked.
- A2 (R6-R8): Ordered shared feature schema matches the approved manifest.
  Sparse hand-computable histories verify exact offsets, window edges, class
  frequencies, events, run spans, interactions, area isolation and no future
  source months. IDs/current outcomes/experts cannot enter X. Missing engineered
  values do not remove rows. Date provenance and source missingness are saved.
- A3 (R4-R6): Every scheduled fold and skipped-empty reason reconstructs; each
  train row uses its own origin, training windows reproduce, partition information
  stops by2020-12. Imputer statistics use only the estimator's real fitting rows.
  Matched checkpoint/imputer inheritance and pooled fallback replay correctly.
- A4 (R2,R9-R10): Four-class statistics include both FP and FN; scan normalized
  masses/zero guards and fixed-four F1 reproduce. Strict .01 boundary, ties and
  root/non-root rejection work. Final assignments, checkpoints and routing agree.
- A5 (R4,R11): Complete candidate inventory, score provenance and Stage2 weights
  reproduce. Only learning-period scores select consensus. All-zero valid
  consensus is distinguished from missing/failed candidates and uses exact
  pooled fallback. Real cluster count/coverage/assignment routes are retained.
- A6 (R12): Stage1 has exactly four permitted synthetic fitting rows; Stage3
  has none. Class-axis alignment and local/global missing-class behavior pass.
  Every RF's fitted parameter/transform identity and real row support are saved.
- A7 (R2,R13-R14): Main/supplementary keys, confusion matrices, metrics and paired
  intervals recompute from keyed artifacts and shared country multiplicities.
  Undefined/empty support is disclosed; no favorable-cell selection is hidden.
- A8 (R15): Independent report reconstruction and representative saved-model
  replay reproduce predictions. Fresh run manifests enumerate actual fits,
  skips and failures; original package/results remain unchanged. Reviewable
  evidence must be accessible to the independent auditor, not just local logs.
- A9 (lifecycle): Final planning approval precedes separately authorized execution.
  Approved spec commit, actual bound executor/start and base SHA are verified;
  accepted prior audit gates precede ordinary start. Commit completed evidence,
  close via wrapper and retain the real audit result. Archival is not audit pass.

## Approved defaults and limits

The approved inventory is162 unique predictors:28 static,41 origin dynamic,15
calendar/area-corrected inherited covariate-derived,3 known calendar,75 outcome
history. The user's final confirmation approves this precise inventory and
fixed RF100 trees/depthNone/seed5/no class weights, argmax decisions
without threshold tuning, general consensus only with release cluster selection,
seed42 shared bootstrap with explicit empty-draw handling, and per-horizon
marginal/descriptive inference. See design.md and feature-contract.md.

Source-month alignment is not verified real-time release availability; inherited
secondary data may contain revisions. Stage1 pseudo rows can influence small
branches. Static-role validation and finite-input checks cannot prove historical
publication dates. Stage3 years were previously inspected and are retrospective.
A complete negative result is successful study completion; missing evidence is not.

No scientific choices remain unresolved. Codex completes registration/controller
setup only; the bound executor establishes the implementation run. Any material
review finding reopens the affected design choice.
