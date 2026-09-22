# Technical contract — final-review proposal

D1-D21 are approved. The exact expansion, candidate values and edge-case rules
below are proposed for one final user review; they are not permission to execute.
After that review, this file and the two JSON inventories are the implementation
contract. Ask about a material conflict rather than silently inventing a variant.

## 1. Fixed inputs and predictor boundary

Source root remains Analysis/1.Source Data/assembled_IPCCH; use
raw/IPCCH_2026_completed.csv, SHA256
ae696087c3bbb280537ae269a05924133acdb51060d31290523404fa8a717673.
Read raw strings and reuse existing target QC: P1-P4 required, missing P5=0,
each share in [0,1], positive estimated_population, sum S in [.90,1.10];
normalize by S. Keep invalid outcomes missing, including as history. Preserve
the exact binary label 5*(P3+P4+P5)>S and normalized share as separate targets.
Duplicate or malformed area/month keys fail. Source files are never rewritten.

All variants retain the original93 features in feature-schema.json, in that
order. No new external source, country/area identifier, source-system predictor,
spatial neighbor feature or automatic feature dropping. The rich variants append
the 468 specified history columns, giving 561. The old WFP/macroeconomic/secondary
feature definitions remain as originally implemented; FEWS NET's replacement
feature recipe is not imported. Dates, keys and provenance never enter X except
the explicitly listed original time/horizon and new elapsed-time features.
Use original raw-feature missing-token/nonnumeric audit behavior and disclose it;
new derived infinity, duplicate column or ordering mismatch fails preflight.

Country lookup uses country_area_id_lookup.csv; pin its hash before scoring,
validate unique IDs and complete required coverage. Use its nonempty country
key even if ISO3 is missing. IPC/CH reporting uses admin_code <100000 / >=100000
only as the inherited dataset cohort convention. No polygon/adjacency is needed.

## 2. Exact history construction

For area a and row origin o=T-h, take valid distributions at month <=o, sorted
newest first; obs1 is latest. No interpolation, forward-fill, cross-area borrowing,
observation-age cap, restriction to fitting-window dates or target-first filtering.
Obtain histories from the full valid ledger, not a filtered fit/test subset.

The eight series, in order, are:
q2=p2+p3+p4+p5; q3=p3+p4+p5; q4=p4+p5; q5=p5;
severity_index=sum(k*p_k); entropy=-sum(p_k*ln(p_k))/ln(5), 0*ln(0)=0;
concentration=sum(p_k^2); severe_fraction=q4/q3 if q3>0, missing otherwise.
Numeric summaries use float64 representations of normalized shares; classification
history always uses the exact ledger binary, never a rounded q3 comparison.

Feature names, order, original93 and aliases are frozen in feature-schema.json:

| Appended block | Width | Definition |
|---|---:|---|
| observation_levels | 48 | 8 series x obs1..6 |
| observation_timing | 10 | ages obs2..6; gaps obs1-2 through obs5-6 |
| changes | 80 | 8 series x 5 adjacent pairs x raw difference/per-month rate |
| observation_trends | 16 | OLS slopes on last 3 and last 6 observation slots |
| window_statistics | 240 | 8 series x 5 windows x 6 statistics |
| window_support | 29 | count/span/age, shared common mask and ratio-specific mask |
| prior_binary_states | 5 | observed binary states obs2..6 |
| threshold_distance | 12 | latest signed/absolute margin; mean/min absolute per window |
| window_crisis | 20 | fraction/entries/exits/eligible pairs x 5 windows |
| event_and_run | 8 | noncrisis/entry/exit ages+absence, current run count/span |

Age=o-observation_month; gap=newer_month-older_month>0. Difference=newer-older;
rate=difference/gap. Missing either slot/value => missing difference and rate.
Do not compress ratio history around undefined q3=0 observations. Last3/last6
slopes use those original slots, >=3 finite values, and their actual month ordinals.
OLS slope=sum((t-mean(t))*(x-mean(x)))/sum((t-mean(t))^2); no extrapolated values.

Windows m06/m12/m24/m36 are [o-W+1,o], all includes every valid month<=o.
Statistics: mean, std (population ddof=0), min, max, latest finite value minus
mean, OLS slope. Mean/extrema/latest-minus-mean require >=1 finite value; std
requires >=2; slope >=3. No values => missing statistics, count=0, missing span
and age. One value => span=0; insufficient slope/std is missing, not zero.
The seven complete series share common support; severe_fraction uses its own
finite-value support. Support span=latest-earliest usable month, age=o-latest.
Aliases reuse original latest-label age, last binary state, last crisis age and
no-prior-crisis indicator; do not materialize duplicate columns for those aliases.

Latest signed margin=q3-.20; absolute margin=abs(margin). Each window's margin
summary uses its actually observed q3 values. Historical binary states remain
strict-.20 ledger labels, including exact equality as non-crisis. Window crisis
fraction needs >=1 observed state; entries/exits count adjacent state changes
whose two endpoints lie within the window. Need >=2 records for these counts;
otherwise missing, while eligible-pair count=max(n-1,0). Pair count cannot imply
that long gaps were continuously observed.

Noncrisis age uses the most recent observed noncrisis record; entry/exit ages
use the most recent observed transition of each type across all available history.
A change date is its later observed endpoint, not the true event date.
No such record => missing age
and absence flag=1; otherwise flag=0. Current run is the maximal newest prefix
of equal observed binary states: count and first-to-last observed month span,
not elapsed duration through o. No history => missing run count/span; one
observation => count=1, span=0. Existing binary no-history and crisis-absence
flags remain. These definitions apply identically to all methods.

## 3. Model and support inventory

| ID | Estimator/target | X | Fit support |
|---|---|---|---|
| binary_history_xgb | XGBClassifier, y | 93 | matched |
| rich_direct_xgb | XGBClassifier, y | 561 | matched |
| correction_xgb | XGBClassifier, e=1[y!=b] | 561 | matched |
| share_xgb | XGBRegressor squared error, q3 | 561 | matched |
| rich_rf | RandomForestClassifier, y | 561 | matched |
| fullpool_xgb | XGBClassifier, y | 561 | full pool |
| persistence | b, no fitting | latest valid binary | evaluation history available |

For fold (h,T), O=T-h, full fitting pool has valid labels at targets [O-35,O]
and the same h. Matched pool additionally has b available at each row's own
origin target-h. The five matched learners have byte-identical ordered fitting
keys and eligible test keys. Sample weights are equal per eligible area-month;
no population/class weighting, SMOTE, pseudo rows or other resampling.
Feature histories may predate the fitting window. Prior test outcomes may later
enter rolling fitting once target<=O; final choices never get reselected.

Main selection/claims use E_history, the common test keys with b. E_no_history
uses fullpool_xgb for every combined binary stream, at fixed crisis-probability
cutoff .5 (strict >); no separate no-history tuning or calibrator. E_all combines
these disjoint supports and is explicitly labeled as a method+shared-fallback
stream. Persistence is unavailable in E_no_history, not a zero prediction.
The share regressor's continuous/fixed-.20 reports use only its own E_history
outputs; fullpool_xgb probability is never substituted for a population share.

## 4. Missing values, constants and runtime

XGB uses native NaN handling. RF retains the pinned baseline max_plus imputer:
training column max*100, max==0 ->100, all-missing ->0; all columns retained.
This rule is retained for baseline continuity; for negative columns it need not
be outside their range. Disclose that limit, do not silently change strategy.
Use a fresh fitted imputer per matching fold pool, reused across RF candidates;
fit only on real training X. Assert names/order/width/finite transformed values,
save fill values and all-missing/negative-max counts. No imputation fitted on
development/test or full pool can be reused in the matched RF.

Empty test => recorded skipped fold without fitting. Nonempty test but empty
required full/matched pool => incomplete run, stop; do not substitute a model
trained on later labels or drop the fold. A nonempty classifier pool with one
target class uses an explicit constant 0/1 score for that target, without fake
rows or a failed XGB fit. Correction constant e is converted conditional on b.
A constant regression target uses that constant share. Record support, reason
and actual estimator/constant route. Nonconstant real fit errors stop the run.

Formal runtime: Windows Python 3.12.10; numpy2.2.6, pandas2.2.3, sklearn1.6.1,
XGBoost3.0.0; original baseline import path also requires scipy1.15.2,
GeoPandas1.0.1, Shapely2.1.0. Compare, do not merely log, required versions.
Use installed original Windows Python; no Linux numerical substitution.
RF imputer extraction uses the pristine GeoRFBaseline v0.1.0 ZIP with SHA256
39a26138e3fafb0be2bbd22e9760095d6cdefa7d79b98b4f798cb3aa79b500a0.
Reuse verified extraction/import isolation; no polygon patch or GeoRF fitting.
Record all effective estimator params and fitted XGB booster config/round count.

## 5. Search, score orientation and deterministic decisions

candidate-configs.json defines six XGB configurations and six RF configurations.
Resolve each by overlaying its candidate values on its common values; XGB
classifier/regressor objective blocks are separate. Record all other effective
library defaults under the pinned environment. No early stopping, additional
probability calibration, repeated-seed search or undeclared candidate expansion.
Training remains logloss for y/e classifiers and squared error for q3 regression.
No claim of calibrated probability follows from a good F1 decision threshold.

Convert every learned output to a score increasing toward crisis: direct/RF p1;
correction r if b=0, 1-r if b=1; share clipped to [0,1]. Keep raw share, raw error
probability and b for audits. Choose a crisis-score threshold t0/t1 for each b.
Decision is always score>t_b, equality is negative. Thus correction flips 0->1
when r>t0 and flips 1->0 when r>=1-t1; the latter inclusive boundary follows
the unified crisis-score convention. Report both threshold representations.
No-flip options exist through t0=+infinity and t1=-infinity.

For each method/candidate/horizon, pool common E_history development predictions
over all 2020-2022 target months. In each b subgroup, form candidate thresholds
from 101 empirical quantiles at 0,.01,...,1 (numpy linear method), add .5
(.20 for share), and +/-infinity; sort and deduplicate. At most104 thresholds
per subgroup, 10,816 pairs; compute joint F1 from pooled TP/FP/FN, not separate
subgroup F1 or mean monthly F1. Save threshold candidates and score ledger.
No b subgroup support => its no-flip threshold only, marked unsupported.

Within a configuration, maximize F1; tie -> fewer changes from b across pooled
development keys; tie -> lexicographically smallest (t0,t1), extended-real order.
Across configurations, maximize each one's best F1; exact tie -> earlier JSON
candidate order. Exact F1 ties can be compared from integer confusion counts.
Undefined denominators are unavailable, not zero; if selection cannot be made,
stop with incomplete evidence. Persist infinities as named JSON strings, not
nonstandard numeric Infinity. Development selections are not out-of-sample gains.

For final main method choose rich_direct_xgb/correction_xgb/share_xgb by equal
mean of four selected development delta F1 vs persistence. Exact tie order:
rich_direct_xgb, correction_xgb, share_xgb. No per-horizon method mixing. Save
all six methods' selected configurations/thresholds, schema/source/code identity
and the primary family in a freeze manifest before computing final predictions.
Final state-dependent thresholds never change, including when new outcomes
enter the rolling training pool. Fullpool_xgb is selected on E_history too;
its E_no_history routing retains .5 regardless of selected t0/t1.

## 6. Inference, reporting and success

For each horizon, compute aggregate confusion-count F1/precision/recall and
paired deltas on exact common keys. Report support and prediction coverage,
IPC/CH, country, target year, horizon, history-age bins and source support.
For correction also report beneficial/harmful flips by direction. For share
report MAE/RMSE, raw clipping counts and fixed predicted-share>.20 binary results.
The fixed rule uses model float outputs; actual binary truth remains Decimal exact.
No alternative supplemental score selects the primary method after freeze.

Bootstrap seed42, 2,000 valid paired draws, max20,000 attempts. Sort the union
country keys across all four E_history main cohorts; sample that many countries
with replacement, reuse multiplicities for all seven methods and horizons.
Recompute weighted confusion counts, each F1/delta and their equal mean. Reject
the entire draw if any required method/horizon has empty support or undefined
F1; do not selectively keep different draws for different contrasts. Save all
draws/rejections/reasons. Fewer than2,000 valid draws => incomplete interval
evidence, not a scientific failure. Use linear 2.5/97.5 percentiles; report
defined-draw conditioning. This interval is conditional on trained predictions,
not a refit bootstrap or a guarantee about future years/countries.

For each required comparison, stable gain needs mean delta>0, 95% lower>0,
every horizon point delta>=0, and mean delta>0 after omitting each target year
2023/2024/2025 in turn (no refits/retuning). After each omission, recompute each
horizon's pooled F1 and baseline delta on its remaining common keys, then take
the equal mean of the four horizon deltas. Empty/undefined required cell =>
incomplete. No +.01 minimum. All required baseline tests must pass conjunctively;
secondary per-method/group intervals are descriptive, not simultaneous claims.
No selection of the most favorable baseline or subgroup to declare success.

Report three claims separately: (1) primary method vs rich_rf AND persistence;
(2) if primary is correction/share, formulation advantage additionally vs
rich_direct_xgb AND fullpool_xgb; (3) information gain, rich_direct_xgb vs
binary_history_xgb, always reported under the same stability rule. Claim (3)
concerns a tuned feature pipeline, not a fixed-hyperparameter causal ablation.
The rich schema adds both continuous shares and longer binary history; this
contrast does not isolate the contribution of continuous shares alone.
If direct XGB is selected, formulation advantage is not applicable. A complete
negative or mixed result is valid completion; missing evidence is incomplete.
Do not start GeoXGB/phase-vector research or another feature search automatically.

## 7. Calendar, budget and supplementary scope

Development: 36 targets Jan2020-Dec2022 per horizon, 144 scheduled folds. Main:
h1 Feb2023-Dec2025 (35), h3 Apr2023-Dec2025 (33), h6 Jul2023-Dec2025 (30),
h12 Jan2024-Dec2025 (24), total122. Empty months remain in the ledger. Data
available through Dec2022 selects decisions; earliest main origin Jan2023.
These already-inspected outcome years are retrospective comparison evidence.

Upper bounds: 6 methods*6 configs*144 development folds=5,184 estimator fits;
6 selected models*122 main folds=732. Constant/empty folds reduce actual fits.
No SMOTE or imputer fitting counts as extra estimator candidates. A bounded
pilot is the first supported 2020 development target at h1 across all six
methods/configs (<=36 fits), included in this budget; record time/memory before
continuing. Do not infer total runtime without measurement or alter science
based on pilot/final scores. Reuse only exact identities, preserve failed evidence.

Optional partial-2026 report is supplementary and disabled in the required run.
If later explicitly requested, fit only already-frozen configurations/thresholds,
record its additional budget separately, and never use it for method selection.
No GPU switch or multiworker scheduler in the initial implementation; retain
n_jobs=1 for predictable resource use, sequential folds/candidates and shared
precomputed feature matrices. A resource problem returns for a bounded adjustment.
