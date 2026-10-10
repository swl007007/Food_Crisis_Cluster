# Scientific evaluation criteria — accepted decisions and pending details

> Consolidation note (2026-10-10): this file preserves the evidence and decision sequence. Earlier wording that a scientific detail was open is historical; accepted D1–D29 are in ../decisions.md and the current final-review contract is ../design.md. Unverified source/asset/resource facts remain unverified, with checkpoints in ../implement.md. No execution is authorized by these notes.

The user wants statistically stable partition benefit over a strong unpartitioned IPCCH model, evaluated and interpreted separately for each H. IPCCH has no expert forecast comparator. The accepted primary contrast is G−B2; B1, B0 and persistence remain relevant controls. D28 explicitly recognizes benefit at only some H as a valuable contribution; there is no experiment-level requirement that all four H win. A scientific null result is reportable and does not mean a faithfully implemented experiment must be retuned until positive. Horizon-specific findings and completion of the authorized research workflow are separate.

## Existing evidence and inherited boundaries

- D11 defines H1/3/6/12 and notes that the same outcome viewed at several H is not independent data.
- D4 uses origins >=2023-01 and main targets <=2025-12, with 2026 targets reported separately; H12 starts at target2024-01. These periods have been examined in earlier experiments.
- D12 fixes common projection and unrounded inclusive .20 decoding; D13 uses 2021/2022 to select a recipe. That development selector is not a test of main-period statistical success.
- The previous P6 contract uses paired country-bootstrap infrastructure: `IPCCHGeoXGBExperiment/config/experiment-contract.json:94-96` specifies 2,000 draws, seed42 and 2.5/97.5 percentiles. This establishes a reusable implementation starting point, not sufficient proof of an appropriate uncertainty model for the new claim. Dependence assumptions and multiple-H correction need an explicit final specification.
- D2 and D21 retain product/pretraining and inherited-tuning provenance limitations. New intervals cannot make a repeatedly viewed retrospective period an untouched prospective confirmation set.

## Accepted primary criterion (D22)

For each H separately, define the main point contrast as crisis `F1(G)-F1(B2)` using identical canonical evaluation keys/truth and ordinary unweighted confusion counts pooled over that H's main target period. It is not an average of annual F1 scores and is not weighted by the temporal fitting weights. D13's equal-year development selection remains a distinct estimator/procedure.

User accepted on 2026-10-10 that an H passes the primary partition-benefit criterion only if all hold:

1. The main-period point gain is at least `.005` on the 0–1 F1 scale (0.5 percentage point, not 0.5 percent relative improvement).
2. The paired lower confidence bound is strictly above zero under the D24 country-bootstrap/Bonferroni procedure for the four-H primary family at nominal 95% family coverage/error level. Its approximate inference and dependence limitations are explicit below; four separate unadjusted 95% intervals do not implement this criterion.
3. Every main target-year panel that exists under the frozen schedule has a nonnegative point gain. This is a sign-consistency requirement, not a requirement of significance in each year. H12 has no 2023 main target panel; do not invent one or score it as zero. Undefined metrics in a required nonempty panel are unresolved/inconclusive evidence, not automatic passes.

Report all H, including no-partition fallbacks and negative or inconclusive results. Under the user's D28 correction, **one or more qualifying H is a valuable horizon-specific retrospective finding even when others do not qualify**. Do not mark the experiment unsuccessful merely because fewer than four H benefit. A report may identify the qualifying subset after evaluating all four under the fixed corrected procedure; it must also show the complete other-H results. No four-H average or winner-only table may replace that account. All-horizon language is justified only if all actually qualify; it is not a target or acceptance gate for this experiment. 2026 is separate supplementary evidence and cannot rescue that H's failed main criterion or select the map/recipe.

Here "evaluate independently" means separate estimands, decisions and scientific interpretations per H. It does not assert statistical independence of overlapping outcomes, and does not remove D24's four-H multiplicity correction. The correction controls discovery claims across the four prespecified horizons without requiring them to pass jointly.

The result table must show each H's point gain, corrected lower bound, annual signs and individual guard outcomes. A nonqualifying H is not automatically evidence of zero effect: distinguish a small gain below the practical floor, imprecision, annual reversal, a secondary-metric trade-off and undefined evidence. Significance at one H but not another does not itself prove a statistically significant difference between their effect sizes; no such cross-H interaction test is currently specified.

The `.005` floor is the accepted practical-effect preference, not a data-derived power estimate. The alternative was no minimum magnitude: retain positive corrected lower bounds and year-sign consistency but accept any positive effect size. That permits statistically precise but negligible gains. The accepted floor may reject a smaller useful effect; it must not be relaxed after observing this experiment's evaluation scores.

## Additional protections for each H's positive finding

This primary rule alone is not the whole acceptance standard for a positive finding at a given H. D23 below freezes guards against a weak B2, secondary metric tolerances and comparisons with persistence on its available-key cohort; D24 freezes the uncertainty procedure and its scope. Do not call that H a benefit passing all protections just because G−B2 passes while stronger pooled performance or the user's other metrics are substantially harmed. The accepted intervals condition on frozen predictions and do not estimate model-selection/training uncertainty.

No metric was recomputed, no bootstrap was run, and no result from this new experiment was used to choose the `.005` value.

## Accepted strong-baseline and secondary-metric guards (D23)

Existing metric inventory was checked in `IPCCHGeoXGBExperiment/ipcch_geoxgb/metrics.py:94-105` and `IPCCHYearlyGeoXGBExperiment/ipcch_yearly_xgb/report.py:28-47`. It contains seven classification scores and both raw/projected q3 R². Thus the user's eight core outcomes are crisis F1, six other classification scores, and projected q3 R²; retaining raw q3 R² as an additional diagnostic gives nine reported numeric metrics. The raw score must not substitute for projected q3 R² when judging the deployed decoded/projection pipeline.

User accepted the following guard policy on 2026-10-10, additional to D13/D22:

1. Crisis F1: require G's point estimate to be at least B1's and a strictly matched B0's. Passing G−B2 alone is insufficient if uncalibrated pooled predictions perform better. These are point guards, not a new claim of statistically significant superiority to B0/B1.
2. For each comparator B1 and B2, require `metric(G)-metric(comparator) >= -.01` separately for four-class accuracy, four-class macro F1, binary accuracy, binary precision, binary recall and binary F2. All six guards apply; high accuracy does not compensate for a large macro-F1 decline.
3. For each comparator B1 and B2, require projected q3 R² difference >=`-.02`. Retain raw q3 R² and full confusion/per-class/NA diagnostics, but do not use them to override a failing guard.
4. Apply these practical point guards within each of the two development validation years before D13 ranks eligible recipes. Apply them again to each H's aggregate main-period panel for a scientific-success claim. Main year-specific secondary metrics remain reported diagnostics; D22's annual nonnegative F1 requirement still applies to G−B2. Undefined required metrics do not silently count as passes.

All guarded comparisons use the same canonical keys, truth and transformation protocol. To complete the B0 guard, matched predictions are needed; old incompatible results may remain historical references under D9 but cannot satisfy that guard. If reuse fails compatibility, obtain B0 through the accepted fixed D21 recipe under this task's own protocol after execution is authorized. Do not shrink the evaluation cohort to old-file availability or substitute a historical aggregate score. B2/G share the same fitted B1 backbone.

Persistence remains a secondary comparator, with all eight core metrics reported for every model on its common available-key cohort and with availability coverage shown. It is not an added mandatory significance gate under D23; neither gains nor deficits against persistence may be hidden. This keeps the primary objective focused on partition value, as the user clarified, and does not presume that the new pooled protocol will outperform persistence.

These tolerances are practical point-estimate rules, not formal statistical noninferiority/equivalence tests. Report uncertainty without asserting that a passing point guard proves absence of deterioration. D22's corrected significance claim remains specifically G−B2; passing point guards against other models does not extend that claim to them. A horizon passing the primary criterion but failing a guard is reported as a trade-off at that H, with the failing metric named; this does not negate protected positive findings at other H. Valid negative or mixed results remain deliverables, not instructions to retune on the evaluation period.

The unselected, more permissive alternative allowed classification declines up to `.02` and projected R² declines up to `.05`, retaining the same F1/baseline and cohort rules. The accepted `.01/.02` margins protect all secondary scores more tightly than a majority-only rule, but can reject useful precision/recall trade-offs. Neither margin pair has been justified by a task-specific utility or power calculation; these are the user's chosen performance tolerances.

## Reusable uncertainty implementation — inspected 2026-10-10

A read-only scout traced the P6 and yearly report helpers; the coordinator checked the P6 implementation and interpretation text. `IPCCHGeoXGBExperiment/ipcch_geoxgb/report.py:65-119` aggregates each country's entire cohort into TP/FP/FN/TN, samples K countries uniformly with replacement K times, applies identical country multiplicities to both arms, and computes F1 from summed counts. It does not average country F1 scores or resample individual rows. The country universe is sorted from that particular H/cohort. All regions and dates within a sampled country share its multiplicity.

The existing constants are 2000 draws, seed42, `numpy.random.default_rng`, and linear 2.5/97.5 percentiles. A defined interval requires K>=2, a defined point contrast and every draw defined; failed draws are retained, not dropped/redrawn. The helper saves country counts, multiplicities and both arm scores. It resets the seed per call; this is not an explicit cross-H common sampling contract (H12 in this task has one fewer country). The yearly copy is at `IPCCHYearlyGeoXGBExperiment/ipcch_yearly_xgb/report.py:56-108`. These are reuse sources; no existing package is to be changed merely to adjust this new experiment's report.

The P6 report explicitly limits its intervals to frozen predictions and the observed cohort, excluding training/map-selection, shared-shock and future-year uncertainty (`report.py:228-230`). The helper is suitable as a narrow starting point, not an already sufficient four-H procedure. The current main support probe in `temporal-support.md` finds 52–53 countries but only 2–3 target years, with about 40% of rows in three countries and 57–70% in 2025. Country clusters are unequal in size and have sparse date coverage. No independence assumption or actual interval coverage has been validated by these counts.

## Accepted paired country bootstrap and limited inference (D24)

User accepted the following first-experiment procedure on 2026-10-10:

1. For each H's main G/B2 common-key cohort, reuse the whole-country paired count resampling above with **20,000 draws**, seed42, `numpy.random.default_rng`. Within a draw both arms must receive identical multiplicities. The point estimand remains unweighted pooled-row F1 difference, not country-average F1. No fitting or map/recipe selection occurs within a draw.
2. For D22, fix the primary comparison family at **four H**, even if some select B2 fallback. Use Bonferroni one-sided alpha `.05/4=.0125`: each H's lower bound is the **1.25th percentile** of its paired delta draws, `method="linear"` (nominal one-sided 98.75% bound). Require that bound >0 as well as D22's effect floor and annual signs and D23's guards. This is an approximate family-95% bootstrap procedure under its sampling assumptions, not exact finite-sample coverage. It is not four unadjusted two-sided 95% intervals. Bonferroni does not require H independence or identical country sets.
3. Retain all draws and the existing no-drop/no-redraw NA policy: insufficient country support, undefined point or any undefined replicate makes the required bound inconclusive, never a pass. An exact G=B2 fallback with defined F1 has zero gain and cannot pass D22; it does not shrink the four-H family. Any ordinary 95% intervals reported for context must be labeled descriptive and cannot replace the corrected lower bound.
4. Keep D22's nonnegative gain in each existing main target year, plus country/year sample counts and concentration diagnostics. Conclude only retrospective benefit on this calendar under country-composition resampling assumptions, conditional on fitted predictions and the frozen recipe. Do not infer transfer to unseen countries, independent future years, shared cross-country shocks, or selection-adjusted confirmatory significance. Increasing bootstrap draws improves Monte Carlo resolution, not the amount of independent data.

The unselected alternative was to add a prespecified, cross-country-synchronous calendar-block sensitivity analysis in the first experiment. With only 2–3 years it would remain a sensitivity diagnostic, not a cure for limited temporal evidence. It is outside the accepted first-experiment scope. No bootstrap or time-block experiment has been run; the user chose the country procedure plus the already accepted year-sign check, with its claim boundary stated clearly.
