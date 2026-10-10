# Supervised AlphaEarth partition mechanism — planning decisions and open details

> Consolidation note (2026-10-10): this file preserves the evidence and decision sequence. Earlier wording that a scientific detail was open is historical; accepted D1–D29 are in ../decisions.md and the current final-review contract is ../design.md. Unverified source/asset/resource facts remain unverified, with checkpoints in ../implement.md. No execution is authorized by these notes.

Status: high-level architecture accepted in D8; explicit no-AlphaEarth control/result-reuse requirement accepted in D9. Details below still require convergence before a final execution spec.

## Evidence and boundaries

Coordinator read `docs/notes/2026-10-05_未来方向讨论稿_IPCCH空间分区与局部增量.md` in full. Relevant evidence:

- Lines 22–30: original local models can learn useful conditional corrections on the development S set; near-zero mean residuals do not imply absence of useful conditional signal. S reuse prevents treating those improvements as independent transfer evidence.
- Lines 64–75: old deployment was limited by map coverage and fitting/validation support, among other factors. Coverage-only or support-only explanations are not established.
- Lines 77–95: residual-mean splitting captures level differences but can miss equal-mean/different-slope regions. It is not a proven repair and should not be represented as new split-score mathematics.
- Lines 129–135: replacing local learners on the old maps or changing fitting schedules did not establish partition gains. This does not test the proposed AlphaEarth-supervised map and does not prove it will work.
- `research/alphaearth-documentation.md`: the provider supports use of embeddings in tree learners and aggregation; this does not provide IPCCH-specific effectiveness evidence.

## Accepted first architecture (D8)

Keep the strong pooled XGB as the base predictor, with AlphaEarth descriptors also available to the principal pooled comparator. Learn a shallow partition rule from fixed admin-level AlphaEarth descriptors. Candidate splits are supervised by the reduction in the predictive loss of regularized regional response corrections, not by visual embedding compactness, reported phase classes, or mean residual alone.

Conceptually, for a target and horizon:

`prediction = pooled_base(X, E) + shared_correction(z) + regional_deviation_R(E)(z)`

`regional_deviation_R(z) = a_R + b_R^T z`

Here E is the fixed geographic representation, and z is the small prespecified vector accepted in D17. Penalize region-specific deviations toward zero, so the regional response approaches the shared response as support weakens. This shrinks response **coefficients**, not each prediction to the regional outcome mean. D11/D12/D14/D18/D19 subsequently settle target sharing, loss/projection, tree limits, joint correction fitting and regularization/preprocessing. Support, split search and map coverage remain open.

At map-learning time, fit candidate correction models on historical base predictions whose labels were held out of their base fits. At inner validation, the map itself and every estimated component must be learned before that block's origin. Once the final map is frozen, update the predictive coefficients annually using eligible historical supervision per D7; never regenerate old pseudo-residuals from a base model trained on their own labels.

The principal unpartitioned comparator must have an appropriate shared calibration/correction, with the same base inputs and historical supervision, to distinguish regional heterogeneity from generic extra calibration/capacity. A stronger global nonlinear correction control and its budget remain to be settled; merely comparing a calibrated regional model to an uncalibrated base is not sufficient attribution.

## Why start here

- Retains cross-region training information in the base and shared correction, while estimating a smaller regional deviation.
- Allows response-slope heterogeneity even if regional mean residuals are similar.
- A rule based on E can assign an admin with no development labels, provided its representation is valid; that does not establish extrapolation reliability or waive support requirements.
- Keeps the first mechanism narrower than separate high-capacity XGB/MLP models in every region.

## Trade-offs and unresolved choices

- Linear regional corrections can miss nonlinear regional response. Independently fitted regional XGB models allow richer changes but estimate more local structure; existing evidence does not rank these architectures for newly learned maps.
- An embedding-based tree does not automatically yield connected geographical regions. D10 permits disconnected/cross-country groups; this is a response grouping, not a claim of geographically contiguous regions. Exact routing/missing-data rules still need specification.
- An axis-aligned shallow tree may poorly capture geometry in the embedding space; embeddings do not guarantee easy partition rules.
- A powerful pooled model with the same embeddings may absorb all available signal. Better fit on historical training residuals is not evidence of partition value.
- Map sharing is now fixed by D11; shrinkage estimation, no-split fallback, missing representation handling and success criteria remain open.

No classifier/tree/linear model was fitted, and no new dependency or implementation was added.

## Control roles and reuse (D9)

| Role | Inputs/mechanism | Comparison purpose |
|---|---|---|
| B0 | Strong pooled XGB, no AlphaEarth | Required no-embedding reference |
| B1 | Strong pooled XGB, with AlphaEarth | Added representation value vs B0 under matched protocol |
| B2 | B1 plus unpartitioned global calibration | Shared correction value vs B1; main partition comparator |
| G | Same base plus shared correction and regional deviations | Partition contribution vs B2; total practical increment vs B1 |

persistence remains a separate secondary reference. These are conceptual roles; exact calibration capacity and fair tuning budget remain open. Main success must be evaluated against the appropriate strong unpartitioned model, not merely against the weakest row of the table.

User explicitly permits reuse of existing trained results. Report whether each reference is a fresh fit, a reused model/prediction, a keyed re-score, or only a historical aggregate. Verify input/training eligibility, feature recipe, horizon, fit origins/history/weights, learner recipe, output transformation, canonical truth and paired evaluation keys before claiming a strict matched comparison. Hashes/run identities and original results remain preserved. Incompatible references are still useful as historical context, clearly labelled; no speculative retraining merely to erase history.

The sibling compact artifacts contain raw predictions but use reported-phase truth, different training QC, no isotonic projection and H0 instead of H1 (`research/local-reuse-facts.md:18-26`). Joining canonical truth and re-decoding saved predictions can align scoring on intersecting keys; it cannot retrospectively change which data trained those models. Current compatibility and key intersections have not yet been computed. A matched B0 fit may therefore be needed for pure feature attribution; final plan will enumerate only required fits and verified reuses.

## Accepted partition spatial meaning (D10)

User accepted: let each admin have one group determined by the fixed AlphaEarth rule, allowing a group to include disconnected and cross-country admins. This tests shared environmental response rather than contiguous service regions. It enables distant but similar areas to share the small response correction and avoids fragmenting every tree leaf into many separately fitted connected components.

The rule alone does not enforce geographical continuity. Clipping a completed tree leaf into connected components later would change the fitted model and the estimand; it is outside D10. Geographic maps and per-group country composition should still be reported. Similar embeddings do not ensure comparable institutions, conflict processes or food-market exposure; temporal validation and appropriate simple-group controls remain necessary.

## Accepted horizon/target sharing (D11)

Existing evidence: `IPCCHGeoXGBExperiment/config/experiment-contract.json:5-11,20-26` defines q2–q5 and H1/3/6/12. `ipcch_geoxgb/learnmap.py:143` iterates horizons and freezes each selected map; `learnmap.py:277-301` records membership-only maps used by later quartets. The sibling compact uses H0/3/6/12; its H0 result is not a substitute for H1.

User accepted: retain H1/3/6/12. Learn one map per H and share that map across q2–q5, while fitting separate regression and correction parameters for each target. This gives four final maps, rather than sixteen target-specific maps. Inner validation maps are fold-specific reconstructions of the same per-H procedure and are not extra deployment maps.

Rationale: short- and long-horizon predictions may have different conditional residual structures, while sharing the map across the four cumulative targets limits search complexity and preserves one group identity for the quartet. Separate coefficients still allow the four cumulative outcomes to respond differently. A single map across all H would further reduce search variance and complexity but impose stronger common-region structure; existing evidence has not established which sharing pattern is optimal.

Choosing this structure did not itself settle the multi-target split loss, target weights or decoder; those were subsequently accepted in D12. Tree depth and shrinkage details remain open. Cross-H copies of one outcome are not independent observations and must not inflate sample counts or uncertainty. H1 compact inputs would require explicit supported reconstruction; old H0 predictions must remain separately labelled.

## Accepted continuous split criterion and final label objective (D12)

User accepted: generate candidate maps by fitting the four corrected cumulative shares under equally weighted squared error, with the D7 temporal row weights and a penalty on regional coefficient deviations. Target weights are fixed equally, not selected by a new search. Conceptually, the data-fit term is `sum_i w_i * (1/4) * sum_{k=2}^5 (q_ik - qhat_raw_ik)^2`; the normalization and penalty units must be frozen together later. This is a candidate-generation proxy, not the definition of scientific success.

Choose complexity/regularization recipes on the two forward validation blocks using crisis F1 after one common bounded monotone projection and unrounded inclusive 20% decoding. The existing P6 implementation `ipcch_geoxgb/projection.py:1-18,71-93` defines projection into `1 >= q2 >= q3 >= q4 >= q5 >= 0`, then phase thresholding; `config/experiment-contract.json:5-11` defines crisis as phase>=3 and four-class labels 1/2/3/4-or-5. Once monotone, crisis is equivalent to projected q3>=.20. Other quartet predictions can change projected q3, so raw-q3 threshold alone is not equivalent.

Rationale: squared error allows stable coefficient fitting and incremental split comparisons even when small prediction changes do not flip labels. Keeping all four targets in the criterion acknowledges the user's four-class and continuous-share outcomes; final recipe selection still follows the requested binary objective. Equal nominal weights do not equalize error variance, and may favor higher-variance targets. A q3-only or q3-upweighted criterion is an alternative design choice, not an empirically justified improvement.

Prior evidence does not allow equating MSE improvement with F1 improvement: the notes report development candidates improving both, while later transfer gains were unclear. No claim that the surrogate is optimal for F1. Fixed output thresholds are not tuned. Exact aggregation across 2021/2022, conservative selection/ties, secondary-metric guardrails, base-XGB capacity, and model-success tests remain open. These rules must apply consistently to the matched arms; historical raw-decoded outputs remain labelled by their original recipe when shown as historical references.

## Accepted aggregation of the two development validation years (D13)

User accepted, applied separately within each H:

1. Compute matched `d_2021 = F1(G_2021) - F1(B2_2021)` and `d_2022 = F1(G_2022) - F1(B2_2022)` from each block's own forward predictions and common keys/truth.
2. A regional recipe is eligible only if both gains are nonnegative and their equal-year mean is strictly positive. Do not require the old Stage3 +.01 gate or per-year significance.
3. Among eligible recipes, maximize `(d_2021 + d_2022)/2`; prefer simpler maps on a tie (complete deterministic tie order to be frozen with the candidate grid).
4. If none qualify, retain B2 as the no-regional-correction result for that H, rather than forcing a split or selecting an adverse recipe. A null outcome is not a successful partition effect.

The existing valid cohorts have 3,246 versus 4,153 rows in these two years (`research/temporal-support.md`). Giving years equal weight expresses the intended temporal-consistency preference. This development selector is not the pooled confusion-count F1 estimator that may be reported on the main evaluation; report both year-specific metrics and any pooled metric with their own definitions.

Trade-off: nonnegative gains in both years are conservative and can reject a truly beneficial recipe when one year's point estimate is slightly negative because of noise. Two nonnegative point estimates do not establish statistical significance or future stability. This is one whole-recipe selection rule inside development, not a reintroduced per-region deployment gate. The historical-data fit penalties remain unchanged; equal weighting of validation years does not imply replacing the 24-month training weights.

This rule compares to B2 under the same base/validation setup; it does not license claiming success if G only repairs a badly calibrated B2 while still underperforming strong B1. Strong-baseline guards, other metric tolerances, treatment of undefined metrics, full statistical acceptance and prospective confirmation remain to be specified. An unconstrained mean-gain selector is the simpler alternative with less protection against year-to-year reversal.

## Accepted bounded partition-tree capacity (D14)

User accepted: compare maximum-leaf budgets `{2, 4, 8}` and impose maximum depth 3 (root depth 0) on the partition rule, retaining the explicit B2 no-partition option from D13. These limits apply to the AlphaEarth partition tree, not to the strong pooled XGB base trees. A leaf budget is a ceiling; actual leaf count can be smaller when no admissible beneficial split remains. The concrete split/support/pruning algorithm and deterministic search budget still need specification.

Evidence for restraint: `research/temporal-support.md:74-81` gives only 3,683 candidate historical supervision rows for the earliest H12 validation map before actual feature joins/fit checks; even larger row counts overstate independent temporal support. `docs/notes/2026-10-05_未来方向讨论稿_IPCCH空间分区与局部增量.md:64-75` documents the tension between regional granularity and fitting/validation support in the old recipe. These observations justify considering a bounded first search, not any claim that 8 or depth 3 is optimal.

Separate target-specific regional coefficient vectors add parameters within every leaf, making complexity depend on the forthcoming low-dimensional correction inputs as well as K. Minimum support must be checked on eligible historical supervision, including distinct admins/dates, not on unlabelled mapped polygon counts. Exact thresholds and coefficient penalties are still open and must be compatible with D6's early folds.

Trade-off: these caps can miss more intricate environmental response groups. Raising them allows richer grouping and more adaptive search, with more opportunity to fit development noise. No empirical performance comparison has been made; D14 records a first-experiment complexity budget, not an optimal partition count.

## Accepted low-dimensional correction inputs (D17)

User accepted on 2026-10-10: for target k, use its own unprojected B1 prediction p_k and three existing compact296 inputs:

- `prcp_anom_month_ensmean__ma3_at_origin`;
- `tmean_anom_month_ensmean__ma3_at_origin`;
- `event_count_violence__ma3_at_origin`.

This gives four substantive slope inputs plus an intercept for each target's shared and regional correction. Shared B2 and regional G must use the same input definitions and eligible historical out-of-time supervision. D19 subsequently adds three missing indicators and specifies fit-only scaling and penalty normalization, for eight coefficient slots including intercept. No 296/424-dimensional regional regression or extra raw-data feature is proposed. AlphaEarth determines the fixed admin group, while weather/conflict inputs vary by row origin; weather must not dynamically change group membership.

Evidence rechecked: `../IPCCH/src/ipcch/compact_features.py:44-48,56-57,91-92` defines the ordinary source families and three-month trailing names; `../IPCCH/.trellis/tasks/archive/2026-10/10-08-compact-climate-weather-oracle/design.md:25,34,38,56-61` specifies these source fields and O-2..O means with at least two nonmissing months. They are already part of D15, not realized future-weather oracle inputs. This inspection verifies definitions, not actual support or predictive value on this task's canonical keys.

The own-prediction slope permits region-specific calibration; the other slopes permit different responses to recent weather and conflict even at a similar base prediction. These are predictive associations, not identified causal effects. The simpler alternative was intercept plus own-prediction slope only, which tests regional affine calibration but cannot express separate weather/conflict residual responses conditional on the same p_k. The selected variables may still omit useful regional variation in prices, seasons or history; those remain available to the strong pooled base. No feature-subset/window search is included in the first experiment, and no improvement is established.

## Accepted shared/regional fitting arrangement (D18)

D18 resolves the joint-versus-sequential choice left open by D8/D9/D17. The shared coefficients in G are not required to be numerically identical to B2's separately fitted calibration coefficients.

User accepted on 2026-10-10: keep the pooled XGB base fixed during correction/map fitting, and jointly estimate the shared linear correction and the shrunk regional coefficient deviations on eligible historical out-of-time base predictions. For one target, with v containing an intercept and the common transformed D17 inputs:

- B2 raw prediction: `p + v^T beta_B2`.
- G raw prediction: `p + v^T (beta_G + u_R)`.
- Identify the decomposition with `sum_R pi_R * u_R = 0` coefficient by coefficient, where pi_R is the region's fraction of eligible mapped correction-training row weight. These weights use D7 and fitting data only; they are not evaluation-year or human-population weights.

Use the same shared-parameter penalty and supervision protocol in B2/G, plus an additional positive penalty on regional deviations in G. Exact normalization, penalty strengths, intercept treatment, missing-map rows and sparse-region fallback remain to be specified. Under a single eligible region the constraint forces u_R=0, so the procedure must reduce exactly to B2, including preprocessing/support/penalty semantics; the implementation should reuse the B2 result for an actual no-partition decision. This prevents a no-partition outcome from gaining another unrestricted calibration layer. For multiple regions beta_G may differ from beta_B2; report this explicitly. The comparison tests the whole nested regional-response model, not a claim that all nonregional fitted coefficients are numerically identical.

The zero-mean constraint identifies coefficients; it does not require average prediction corrections to be zero, because v differs across regions. With a freely fitted beta_G it does not prevent all regions from sharing an overall calibration shift: that shift belongs in beta_G. Regional coefficients describe predictive associations, not causal effects. This structural nesting is not a statistical test of partition value or a guarantee that a shallow linear B2 is the strongest possible unpartitioned control.

The alternative was to first fit/freeze B2 calibration and then estimate regional corrections. It preserves literal shared-model identity, but constrains the joint optimum; without an explicit nesting/centering rule, a second correction layer can also relearn common calibration. Neither fitting strategy has been empirically compared here. D18 retains the fixed pooled backbone and uses a single low-dimensional correction fit rather than alternating updates of XGB and regions.

## Accepted regularization and bounded search (D19)

User accepted on 2026-10-10: use weighted ridge with a prespecified small regional grid, keeping q2-q5 on the same penalty recipe within each H. This is a prespecified recipe for the new experiment, not a claim that these strengths were optimal in an earlier run.

Preprocessing for the correction layer only:

- For the four core D17 inputs, estimate weighted means and population standard deviations from the eligible correction-training rows, using observed entries and D7 weights only. Use the same per-target transform across regions and B2/G in a given fit; never estimate a region-specific scale or use validation/test rows.
- Standardize finite entries; replace a missing dynamic input with its training mean (zero after scaling) and retain one unscaled 0/1 missing indicator for each of the three dynamic fields. No row deletion for these missing inputs. A nonfinite base prediction is a technical error, not a fourth imputable field.
- A wholly missing or constant training column has no estimable continuous slope: encode its standardized value as zero for that fitted model, including at inference. Save that inactive-column decision. Constant indicator coefficients are likewise fixed to zero, avoiding an unidentified indicator/intercept duplicate. The shape remains four core inputs plus three indicators plus intercept (eight coefficient slots per target).
- These operations do not alter the compact296 inputs supplied to XGB or add predictors to B0/B1. No log transform, winsorization, nonlinear expansion or imputation search is included in D19.

For each q, with total eligible training weight W, raw residual r_i=q_i-p_i, transformed inputs v_i and D18's centering constraint, minimize:

`sum_i (w_i/W) * (r_i - v_i^T(beta + u_R(i)))^2 + .01 * ||beta_nonintercept||^2 + lambda_region * sum_R ||u_R||^2`.

The quartet objective is the equal mean of these four objectives. The shared intercept is unpenalized; all regional deviations, including intercept and missing-indicator deviations, are penalized. Sum the regional penalties without multiplying them by regional sample fractions: smaller regions then contribute less data-fit weight relative to their penalty and receive stronger shrinkage. This normalization is essential; the numeric lambda grid must not be copied into an unnormalized sum-of-errors implementation. Missing-map routing/support rules remain a separate pending decision and must preserve B2 nesting.

Freeze shared lambda at `.01` for B2/G. Compare regional lambda `{.01, .1, 1}` crossed with D14 leaf ceilings `{2,4,8}`: nine regional recipes per H, plus the existing B2 fallback, with no target-specific lambda search. Evaluate the same nine recipes in each of the two D6 development blocks, using D13; this is up to 72 horizon/block/recipe evaluations, not 72 unique XGB fits or a complete compute enumeration. Reuse each block's eligible raw pooled predictions across correction candidates. Once selected, freeze the recipe for final-map learning and annual parameter refits. No grid expansion triggered by main-period scores.

The smaller alternative was to fix regional lambda at `1` and compare only the three leaf budgets. It reduces development selection opportunities but may over-shrink a useful response. The selected three strengths span two orders of magnitude; neither they nor shared `.01` are empirically validated choices. Evidence for a bounded first grid is the 3,683 candidate supervision keys before coverage/support filtering in the earliest H12 block (`research/temporal-support.md:74-81`) and the already accepted four-core-input, shallow-tree architecture. Row counts are not independent-sample counts; ridge does not replace the forthcoming distinct-admin/date support checks or statistical evaluation.

## Accepted minimum regional support and fallback (D20)

Evidence: the old XGB local fitter requires 500 rows, 50 admins and 6 target months (`IPCCHGeoXGBExperiment/config/experiment-contract.json:88-92`; `ipcch_geoxgb/stage1.py:167-180`, coordinator spot-checked). Its separate validation gate requires 100 rows, 20 admins, 3 dates and 20 crisis/20 noncrisis rows; Stage3 also requires three successful local-fit dates (`stage3.py:144-170`). These are different fit and validation requirements, not one combined training floor. The yearly variant retains the count floors and records, but does not gate on, weighted ESS (`IPCCHYearlyGeoXGBExperiment/ipcch_yearly_xgb/engine.py:123-141`). None of these sources validates a threshold for the new eight-coefficient-per-target ridge layer.

User accepted on 2026-10-10: require each regional correction-training pool to have weighted row `n_eff=(sum w)^2/sum(w^2) >= 200`, at least 50 distinct admins, and at least 6 distinct target months. Use D7 weights and actual eligible historical OOT base-prediction keys with valid map routing, not all mapped polygons, imputed synthetic rows, or future validation outcomes. No target-class-balance gate is added for continuous ridge fitting. Preserve raw counts, weight sums, n_eff, dates/span and largest-month weight share as diagnostics. The thresholds are fixed across H and are not added to the hyperparameter grid.

When learning a split, both proposed children must pass before fitting/accepting that split; otherwise do not split that node. During later annual coefficient refits keep the frozen map unchanged, but set an unsupported leaf's entire regional deviation vector to zero for that block. Its rows remain in the shared correction fit and receive `p + v^T beta_G`, not a forced neighbouring region, a dropped prediction, or an invented new region. D18 permits beta_G != beta_B2, so this is a fallback to G's shared response, not necessarily byte-identical B2 prediction. Record that routing distinction.

The D18 centering constraint continues to include zeroed deviations. If fewer than two leaves have support, all regional deviations are zero and the full model must reuse B2 exactly. No per-region F1 acceptance gate is introduced; development recipe selection remains D13. AlphaEarth validity/coverage and completely missing-map routing remain separate pending rules and must preserve the same no-partition nesting.

Rationale and trade-off: compared with the old large XGB learner, this small shrunk correction motivates a lower row floor while retaining admin/date diversity. Weighted n_eff catches unequal row-weight concentration but not within-admin or cross-area dependence. The early H12 probe has 3,683 potential rows, 2,065 admins and 12 observed target months; it establishes feasibility only at the unsplit pool, not for candidate leaves. The more conservative alternative was n_eff>=500 with the same 50-admin/6-month requirements, potentially leaving fewer eligible splits. Neither 200 nor 500 is an empirically established power or stability guarantee. Temporal-concentration diagnostics and the two development years remain necessary; no support check can guarantee later partition benefit.
