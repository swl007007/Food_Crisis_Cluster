# Exogenous food-crisis transition forecasting: scope and evaluation design

Status: **planning / initial brainstorm**, 2026-10-02. Discussion record, not an approved implementation spec. User requested a new task and brainstorm → written spec → grill. No new fitting, scoring, Stage 3 inspection or implementation.

## Goal and user direction

Explore forecasting for regions without IPC history, emphasizing non-crisis → crisis transitions and potentially countries with consistent, statistically supported gains over expert/persistence. Binary or four-class training remain options; two-way transitions are optional and onset has priority.

**User confirmations (2026-10-02):** lagged covariates remain allowed; the principal claim concerns prediction in regions without prior IPC history. Evaluation may be relaxed: universal superiority over persistence/expert is NOT required. Narrow the scientific scope and discuss useful-performance criteria; do not inherit the previous all-horizon strict screen or CI gate.

The user requested closure of `10-01-geoxgb-shared-parameter-design` as **incomplete**. Preserve its experiments/code/checks as evidence; its scientific target was not attained. D55 producer `097d98b` exists with synthetic checks, but real fitting was never released. New feature, endpoint, country and acceptance choices require their own spec.

## Repository facts

- Ordered schema: 162 columns = 75 IPC-history + 28 static + 41 dynamic-at-origin + 15 lagged/aggregated covariates + 3 calendar (`FEWSNETGeoXGBExperiment/feature-schema.json`).
- The 75 history columns include IPC levels, crisis indicators, changes, rolling distributions, observation/missingness ages, runs and interactions. Dropping only the most recent phase is insufficient.
- Covariate lags are separate: price/nightlight aggregates and lagged EVI (`FEWSNETGeoXGBExperiment/src/feature/fourclass_features.py:93`). Removing phase-history leaves 87 candidate columns, not yet an approved schema.
- D51 retained history/calendar and removed other covariates: the opposite ablation. D38/D52/D54 used phase inputs or derived transformations; their scores are not evidence for a phase-free model. Their lineage/replay infrastructure may be reused after review.

## Brainstorm: input boundary and claim

Confirmed direction: retain non-IPC covariate lags and remove phase-history prediction inputs. Recommended full exclusion boundary, still to specify: exclude all phase-derived predictors and phase-based prediction priors, residual corrections, target encodings and prediction fallbacks from every fitted model. Retain non-IPC covariate lags available at forecast origin. Historical labels remain supervised targets. Keep origin IPC separate as evaluator metadata for transition cohorts and persistence comparison.

This supports “past IPC is not required as a prediction input.” It does not establish causal exogeneity: prices/conflict may be endogenous. Prefer “covariate-only” or “past-IPC-free” pending causal justification. If origin phase defines q/split evaluation populations, disclose its use in partition learning despite its exclusion from inference inputs.

## Brainstorm: cold-start claim

Removing IPC input columns does not alone test a region with no IPC training history: its earlier outcome labels could still influence learned parameters or partitions. Recommended main test, not yet selected: hold out whole administrative regions from all supervised fitting, partition learning, tuning and selection; retain lawful origin-time covariates. Keep withheld IPC observations accessible only to the independent retrospective evaluator. This simulates no-history deployment; genuinely never-labelled regions cannot provide observed accuracy estimates. It need not mean leaving out whole countries, which tests a harder cross-country claim. Spatial grouping/buffers and time isolation need a written design.

A secondary same-region, no-IPC-input temporal benchmark can help attribute representation versus transfer difficulty; it is not automatically authorised as an extra experiment. If true no-history deployment makes origin IPC unknown, transition cohorts are retrospective evaluator strata, not an operational routing requirement.

## Brainstorm: transition estimand

Let c_O = I[origin IPC ≥ 3] and c_T = I[target IPC ≥ 3].

- **Actual onset only**, c_O=0,c_T=1: event recall, no non-events or false-positive penalty. Persistence always predicts zero; its failure is mechanical. Precision/F1 computed after this filtering lose ordinary false-alarm interpretation.
- **Origin non-crisis risk set**, all c_O=0, including 0→0 and 0→1 (recommended primary, unconfirmed): onset detection plus false alarms and probability quality. Persistence still has crisis F1=0 when events exist, so beating that alone is weak; expert/pooled comparisons and false-alarm/probability metrics matter.
- **Reverse-transition context**, c_O=1, including 1→1 and 1→0 (optional): distinguish crisis-status prediction from transition-occurrence prediction before specifying metrics.

Proposed joint report: risk-set evaluation primary, true-onset recall diagnostic, reverse transitions optional. Nothing selected. Unknown origin requires explicit exclusion/coverage, no silent latest-label substitution.

Stage 1 q-scan, split acceptance, development selection, local enablement and Stage 3 reporting require separate population/metric contracts. Historical validation labels may define internal event diagnostics; current held-out target outcomes must not enter fitting/search/routing. A new population changes q normalization/support and empty/single-class behavior; no automatic formula reuse.

## Brainstorm: country scope

Recommendation, unconfirmed: discover countries with pre-final development evidence, freeze eligibility/selection, then confirm on untouched final data. Report all eligible discovery countries, including failures. Define “consistent,” event/date support, multiplicity control and paired uncertainty before scoring. Within-country claims require temporal/spatial dependence handling; an across-country bootstrap cannot simply be reused for one country. No country selected and no final-country scores inspected.

## Open decisions, in order

1. Main cold-start validation: whole-region label exclusion versus only removing IPC input features.
2. Primary estimand: actual transitions only, origin non-crisis risk set, or primary risk-set evaluation plus true-event recall?
3. Cold-start validation: whether evaluation regions contribute NO IPC labels to any fitting, partition/consensus learning, tuning or selection; origin-label access for evaluator only. Covariate lags are allowed (confirmed).
4. Binary versus four-class target/rule and whether partition/consensus remains the starting architecture.
5. Country discovery/confirmation, horizons, split, support and useful-performance criteria (no universal baseline superiority requirement).
6. Reuse versus refits/recomputed candidates, finite budget and stop rule.

## Planning acceptance

- [ ] Brainstorm resolves scientific estimand and claim boundaries.
- [ ] Written spec defines every fit/predict feature exclusion, evaluator-only metadata, evaluation layers and support/fallback rules.
- [ ] Country discovery and final information isolation explicit.
- [ ] Grill stress-tests spec one decision at a time before execution approval.
- [ ] Old incomplete closure and reusable evidence linked without calling old phase-dependent scores new-model evidence.
