# Planning record D1-D21 — preserved before convergence

This is the chronological planning record, including formerly open wording.
D1-D21 are approved. The converged PRD and final-review technical contract
resolve implementation details only after the user approves that final review.
No historical pending proposal below is permission to implement.

# IPCCH population-history and pooled XGBoost objective comparison

## Goal

Plan a controlled IPCCH experiment testing continuous phase-share history and comparing direct binary XGBoost, persistence correction and P3+ share regression, with Opus execution and independent Codex audit after planning approval.

Status: planning / grill in progress. This is an initial scope record, not the
closed execution spec. No implementation, model run or audit start is authorized.

## Confirmed scope — D1

The user accepted steps 1 and 2 of the proposed research sequence:

1. Test whether adding historically available continuous phase-share information
   improves pooled XGBoost over the current binary-history feature representation.
2. On controlled information and evaluation support, compare direct binary XGBoost,
   persistence correction, and direct P3+ population-share regression.

Keep the primary target as normalized population share in phase 3+ strictly above
0.20. Continuous shares are candidate predictors/supervision, not a replacement
of the primary binary outcome by official overall_phase.

Do not interpret steps 1/2 as approval to rewrite GeoXGBoost. Spatial parameter
sharing, four-output cumulative phase models, new external covariates and an
unbounded feature/hyperparameter search are outside this task's initial scope.
The executor will be Opus 5 after design closure and separate execution approval;
completion should receive independent Codex audit through the existing controller.

## Verified starting evidence

- Existing code retains normalized five-phase shares and derives binary truth
  from valid P3+ share; missing labels are not replaced by official overall_phase.
  `IPCCHGeoRFExperiment/prepare_data.py:236-288,292-298,352-369`.
- Existing history features contain last binary status, its age/missing indicator
  and crisis recency, not continuous phase-share history.
  `IPCCHGeoRFExperiment/prepare_data.py:521-532`.
- Existing horizons are 1/3/6/12 months; XGB is a pooled binary classifier with
  fixed settings. Source/runtime and original evaluation design are documented in
  `IPCCHGeoRFExperiment/README.md` and the archived IPCCH design. Their reuse in
  this new experiment must be explicitly resolved below, not presumed wholesale.
- The current original-baseline same-key XGB/persistence F1 pairs at 1/3/6/12m are
  .678929/.681392, .680740/.676817, .667058/.671861, .678704/.675917.
  `IPCCHGeoRFExperiment/runs/ipcch-v1-20260920d/reports/main/metrics.csv`.
- Previously examined test years are retrospective development evidence for a new
  direction, not a newly untouched holdout. FEWS NET feature-search failure is not
  proof that IPCCH continuous history or XGB correction cannot help.

## Decisions to close through grill

### D2 — rich history scope approved; exact schema pending

The user rejected restricting the first round to a single historical distribution.
Include multiple historical phase-share observations and a substantial, explicitly
defined set of engineered history features. Do not interpret this as permission
to choose features using final-period scores or to promise statistical significance.
The exact lag count/windows/transforms and fair input contrasts remain to be grilled.

Read-only support check against original run `ipcch-v1-20260920d/data/` on
2026-09-21: count valid same-area observations dated <= each row's origin; include
all original main-period labeled evaluation rows, without requiring persistence.
Read `target_ledger_valid.csv.gz` and `feature_metadata.csv.gz`; horizon is target
month minus origin month. No model fitting or outcome-score selection was used.

| Horizon | Rows | >=1 observed | >=3 observed | >=6 observed | >=12 observed |
|---|---:|---:|---:|---:|---:|
| 1 | 17,322 | 92.4% | 65.6% | 37.3% | 14.0% |
| 3 | 16,919 | 94.0% | 65.9% | 37.9% | 14.3% |
| 6 | 16,413 | 91.1% | 63.6% | 37.2% | 12.9% |
| 12 | 14,087 | 88.1% | 58.4% | 33.7% | 12.1% |

### D3 — six-observation history and five feature groups approved

The user confirmed the following feature scope:

1. Last six valid observed full phase distributions represented by four cumulative
   shares (phase 2+/3+/4+/5+), each observation's age and inter-observation gaps.
2. Successive differences, changes per elapsed calendar month and multi-observation
   trends.
3. Trailing 6/12/24/36-calendar-month and all-available-history summaries: means,
   variability, extrema, latest-value position and observation counts.
4. Distribution shape: population-weighted phase index, concentration and severe
   phase shares and their changes.
5. Crisis boundary: distance of P3+ from .20, historical crisis fraction, observed
   entries/exits and runs of observed crisis status with their elapsed span.

Observation lags are not monthly lags. Missing slots remain missing, never
interpolated labels, and must not shrink the common comparison cohort. Observed
runs do not establish uninterrupted crisis between observations. Earlier history
beyond the six explicit observations still contributes through summary features.
The binary-history XGB remains a reference; the three new methods use identical
rich historical information so gains from inputs and modelling can be separated.
Exact formulas, eligibility counts and missing-value rules remain to be frozen.

### D4 — two-way persistence correction approved

The user approved correction in both directions: latest observed non-crisis to
future crisis (0->1), and latest observed crisis to future non-crisis (1->0).
Do not inherit FEWS NET's onset-only correction restriction. This decision does
not require separate models for the two directions; estimator and decision
threshold policy remain open.

Original baseline main-period E_persist counts from
`IPCCHGeoRFExperiment/runs/ipcch-v1-20260920d/reports/main/metrics.csv`:

| Horizon | Persistence FN (last 0, target 1) | Persistence FP (last 1, target 0) |
|---|---:|---:|
| 1 | 1,765 | 2,144 |
| 3 | 1,753 | 2,195 |
| 6 | 1,623 | 2,041 |
| 12 | 1,292 | 1,861 |

These are observed endpoint disagreements, not exact transition dates. They
motivate two-way correction but do not prove either direction is predictable.

### D5 — single pooled error classifier approved

The user approved one pooled binary XGBoost learning
e = 1[y != b], where b is latest valid binary history as of the row's origin,
with b and the common rich feature set as inputs. Predict flip probability r;
retain b or flip to 1-b under a development-selected decision policy. The implied
crisis probability is r when b=0 and 1-r when b=1. This is an alternative target
parameterization with the same information, not inherently more expressive than
direct binary prediction conditional on b. Missing-history fitting/deployment
and calibration remain separate unresolved contracts; D6 fixes the classifier
threshold principle below.

### D6 — matched two-threshold classifier policy approved

The user confirmed separate flip thresholds for b=0 and b=1,
selected jointly to maximize overall crisis-class F1 on common temporal
development predictions, including a no-flip option for either direction.
Do not optimize each subgroup's F1 independently or tune on final evaluation.
Direct binary classifier comparators receive matching history-state-conditional
threshold flexibility and the same development protocol, so threshold freedom
alone cannot be attributed to correction. Candidate grids, tie rules, support
and freeze dates remain open. D7 extends the decision-policy principle to the
share regressor.

### D7 — matched regression decisions plus fixed-rule report approved

The user approved giving the P3+ share regressor the same two
history-state-conditional decision thresholds and temporal development selection
for the primary binary comparison. Retain the literal predicted-share > .20
decision as a supplementary fixed-rule result using the very same predictions,
not an extra fitted model. Ground-truth y remains actual normalized P3+ > .20 in
both reports. A tuned cutoff is an empirical classifier based on the predicted
share, not a change to the outcome definition or an official phase reconstruction.
Report continuous-share prediction error separately; neither improved regression
error nor tuned binary F1 alone establishes improvement on the other objective.

### D8 — standard squared-error share regression approved

The user approved fitting a standard XGBoost regressor to normalized
P3+ share using squared-error loss, with equal weight per eligible area-month.
Clip predicted shares to [0,1] before threshold selection, prediction and share
error reporting; retain raw predictions and clipping counts for interpretation.
Do not add bespoke threshold-neighborhood weighting or a custom mixed loss in
this round. Report MAE and RMSE separately from binary F1. Squared-error regression
targets a conditional mean share, not the probability that the share exceeds .20;
D7's development-selected decision thresholds address the classification mapping
without asserting that the mean alone captures the full conditional distribution.

### D9 — matched training support and full-pool comparator approved

The user approved, for attribution, fitting the binary-history XGB,
rich-history direct XGB, error XGB, share regressor and matched pooled RF on the
same eligible area-month/horizon training keys with valid persistence at each
row's own historical forecast origin. Evaluate/select on common history-available
keys too. Require at least one valid history observation, not all six lag slots;
missing engineered features do not exclude rows. Keep an additional rich-history
direct XGB trained on all otherwise eligible labeled rows as a full-training-pool
reference, scored on the same evaluation keys. This prevents a matched-support
win from being represented as a win against a direct model using all its data.
Operational no-history prediction routing is fixed by D10 below.

### D10 — no-history direct-XGB fallback approved

The user approved reusing D9's full-training-pool direct binary XGB
for binary predictions when a test row has no valid observed history at origin.
In particular correction cannot define e or flip a missing b; do not fill b with
zero or fabricate a persistence prediction. Keep the primary matched-history
comparison unchanged. Report no-history support/performance separately and label
any all-row combined stream explicitly as the method plus shared direct-XGB
fallback, not the pure correction/regression/RF model. This adds no fitted model.
Never interpret fallback crisis probability as a predicted population share;
continuous regression metrics concern actual share-model outputs only. Exact
no-history binary decision threshold remains to be frozen with development rules.

### D11 — rolling 36-month fitting, unrestricted prior feature history approved

The user approved preserving the original 36-calendar-month rolling
training-label window, inclusive [O-35 months, O], for this controlled experiment.
History feature construction is not truncated to that fitting window: at each
historical row's own origin o = target-horizon it may use all permitted earlier
observations through o, including observations before the fitting window. Do not
construct a past training row's features using the later refit's origin O.
This retains temporal comparability while testing richer history/objectives;
expanding-window fitting would be a separate experimental factor, not silently
combined with the approved feature and objective changes. Source publication
timing remains an explicit retrospective observation-month assumption.

### D12 — development and evaluation calendar approved

The user approved using target months 2020-01..2022-12
for temporal rolling development predictions, hyperparameter/threshold selection;
freeze all such choices by 2022-12. Keep the original main evaluation calendar:
h1 2023-02..2025-12, h3 2023-04..2025-12, h6 2023-07..2025-12,
h12 2024-01..2025-12. All main origins are >=2023-01. Each candidate/fold uses only
labels available through its origin and historical features through each row's
own origin. Development performance used for selection is not an unbiased estimate
of the selected procedure. During final rolling evaluation, refit model parameters
using newly eligible labels under D11, but never retune the frozen schema,
hyperparameters or decision thresholds. Already-viewed 2023-2025 outcomes make
this a retrospectively designed comparison, not a pristine holdout. Any partial
2026 results stay supplementary and cannot select the method or thresholds.

Read-only development support check (original prepared metadata, all cohorts):
2020/2021/2022 each contain 4,111/3,246/4,153 valid outcome rows per horizon;
history-available support across h1/h3/h6/h12 is respectively
3,585/3,585/3,222/2,913 in 2020, 2,809/2,809/2,800/2,523 in 2021,
and 3,274/3,273/2,865/2,732 in 2022. These are evaluation-support counts,
not numbers of independent observations across horizons or evidence of model gain.

### D13 — pooled IPC+CH cohort and stratified reporting approved

The user approved retaining the full IPC+CH cohort and fitting
pooled models per horizon, with no geographic partitions or separate IPC/CH
models in this round. Select the two history-state thresholds on the pooled
development cohort, not independently by country or IPC/CH subgroup. Report
IPC/non-CH and CH, country, target-year and horizon breakdowns on the same stored
predictions to expose concentrated gains or harms. Reuse the existing admin-code
cohort convention (<100000 non-CH; >=100000 CH) for diagnostic comparability;
it is a dataset convention, not a new inference about official system identity.
This cohort policy introduces no new group-ID predictor by itself. Separate
group models or group-specific thresholds would add another experimental factor.

### D14 — rich-history pooled RF comparator approved

The user approved giving the D9 matched pooled RF
the same rich historical feature schema as the rich direct XGB, error XGB and
share regressor, with the matched training support and D6 decision flexibility.
Keep the binary-history direct XGB as the deliberate information ablation. Prior
archived RF/XGB results are contextual references, not substitutes for the new
same-key, matched-protocol baselines. Missing-value handling remains appropriate
to each estimator and fitted only on its training data; shared information does
not require identical model internals. This resolves the input of an already
approved RF arm, not an additional fitted model. Exact raw-feature inventory,
imputation and hyperparameter budgets remain to be frozen.

### D15 — six-configuration development budget approved

The user approved capping development at six frozen
hyperparameter configurations per learned method per horizon, including the
existing parameter configuration where applicable. All five XGB variants use
the same declared search budget; RF also receives six configurations. Freeze
candidate lists before scored model runs; use only D12 development predictions
to select configuration plus D6/D7 thresholds by overall binary crisis-class F1
within each horizon. The regressor still fits squared error under D8; its model
selection score follows the primary binary task. No test-driven grid expansion
or indefinite search. Exact candidate values and tie rules remain open.

Budget accounting for D15: six learned methods times six candidates
times 36 development target months times four horizons = at most 5,184
development estimator fits, before empty/unsupported-fold exclusions. Six frozen
methods times the original 122 scheduled main folds = at most 732 main evaluation
fits. Threshold searches and the fixed-.20 regression report reuse stored
predictions; partial-2026 refits and any additional experiments are not included
in these counts. Counts are not runtime estimates or counts of individual trees;
implementation should measure a bounded development-only pilot before the full
search, without using final-period scores to change the experiment.

### D16 — one development-selected primary method across horizons approved

The user approved, after per-method/per-horizon
configuration and threshold selection on development predictions, select one
method family from rich-history matched direct XGB, error-correction XGB and
P3+ share regression. Rank them by equal-weight mean of the four horizons'
development delta F1 versus persistence on common keys. Each horizon F1 uses
pooled confusion counts, not a mean of monthly F1. With common persistence/support
this ranking equals ranking by mean F1. Freeze that family across all four
horizons before final evaluation; horizon-specific selected configurations and
thresholds remain allowed under D15. Do not assemble a final-score-selected mix
of methods or change the primary method after final results. Report all three
methods and all baselines, distinguishing the frozen primary method from
secondary comparisons. Prefer the direct classifier on an exact best-score tie;
the remaining deterministic tie rule will be fixed before execution.

### D17 — stable-gain and formulation-advantage criteria approved

The user approved distinguishing a useful forecasting
gain from evidence for the correction/regression formulation itself. For a claim
of stable improvement against a required baseline, require equal-weight mean
four-horizon delta F1 > 0 with a paired 95% interval whose lower bound > 0,
and no negative point-estimate delta in any of the four horizons. Do not require
each horizon to be individually significant or impose a minimum +.01 effect.
Apply this first to the frozen primary method versus rich-history matched RF and
persistence on identical history-available evaluation keys. If correction or
share regression is selected, a stronger claim of formulation advantage also
requires passing the comparison against both rich-history matched direct XGB
and the full-training-pool rich-history direct XGB. A selected direct classifier
can support an information-gain result against the binary-history XGB; it cannot
establish a correction/regression advantage. Publish all contrasts even on failure.
Exact interval resampling, temporal robustness and multiplicity/reporting details
remain to be specified; intervals must not be described as proof of future-domain
generalization. No primary-method reselection from final scores is allowed.

### D18 — observed-record summaries and calendar-time trends approved

The user approved computing features only over
valid observed distributions, with equal weight per observation; do not forward
fill a monthly label series or interpret absent observations as non-crisis.
For each row origin o, a W-month trailing window contains observation months
[o-W+1, o], W in {6,12,24,36}; the all-history window contains all valid months <=o.
Apply the common observed-data eligibility to each of the four cumulative shares.
For each window compute mean, population standard deviation (ddof=0), min, max,
latest in-window value minus its window mean, and OLS slope against actual
calendar-month ordinal. Count actual observations and retain the observed time
span and latest observation age, so observation density is visible to the model.
Mean/min/max/latest-minus-mean need >=1 observation; standard deviation needs >=2;
OLS slope needs >=3. Empty windows have count 0 and missing value statistics;
insufficient support produces missing statistics, never a zero trend. Differences
between successive observations are retained both raw and divided by positive
elapsed calendar months. Observed crisis fractions are observation-weighted,
not estimates of the fraction of all calendar months in crisis. This contract
does not add gap-based interpolation, time weighting or decay hyperparameters.

### D19 — distribution-shape series and history transforms approved

The user approved, on each valid normalized
five-phase observation p1..p5, derive (a) phase severity index sum(k*p_k),
(b) normalized Shannon entropy -sum(p_k*ln(p_k))/ln(5), with 0*ln(0)=0,
(c) concentration sum(p_k**2), and (d) severe fraction among P3+ population,
q4/q3 where q3>0, otherwise missing. The severity index is an ordinal-derived
descriptive feature, not official overall_phase or a cardinal welfare measure.
Apply the approved six observation slots, adjacent changes/rates and D18 window
summaries to these four derived series too. The observation slots still refer to
the same valid phase-distribution dates; do not skip a zero-q3 observation to
backfill a ratio slot. Ratios with missing endpoints have missing changes.
Window statistics use finite defined values within the original time window and
enforce D18's support requirements per derived series; preserve corresponding
count/span/age so undefined ratios cannot masquerade as complete support.
Raw q4/q5 histories already represent severe phase shares; do not duplicate them
under new predictor names. All calculations use only observations available at
the row's own origin, with no target-period distribution as an input.

### D20 — crisis-boundary and observed-state features approved

The user approved retaining last-observation signed
and absolute distance of q3 from .20; for each D18 window retain minimum/mean
absolute distance to .20. Do not multiply columns by repeating every linear
shift of an existing q3 statistic or copying existing recency fields. Retain the
last six observed binary states (reuse the existing latest-state column) and,
in each D18 window, observed crisis fraction and counts of 0->1 and 1->0 changes
between successive observed states, requiring both endpoints inside the window.
Fractions require >=1 observation; transition counts require >=2, otherwise
missing; retain the number of eligible adjacent pairs as support metadata/features.
Use the ledger's exact strict-.20 binary states, never reclassify rounded share
floats. Across available history, record latest observed crisis/non-crisis ages
(reusing existing crisis age), latest observed 0->1/1->0 change ages, and current
observed-state run length and first-to-last observation span. A transition's
reference date is the later endpoint, not an estimated real event date; a run
means consecutive observed records, not known uninterrupted monthly conditions.
No prior event produces missing age with an explicit absence indicator; zero
means an event observed at the origin, not absence. Changes across a window's
left boundary are not counted in that window. These are finite descriptive
features, not an imputed latent monthly state sequence.

### D21 — joint country bootstrap and leave-year-out checks approved

The user approved using 2,000 paired country-cluster
bootstrap replicates with seed 42, retaining each sampled country's full saved
trajectory. Draw from the joint country universe and share each replicate's
country multiplicities across all methods and horizons; recompute each horizon's
F1 from confusion counts and then its equal-weight mean delta. Do not bootstrap
rows independently or average independently calculated horizon confidence limits.
The percentile 95% interval describes country-composition uncertainty conditional
on fitted predictions and observed years, not all future-time/training uncertainty.
Additionally omit target year 2023, 2024 and 2025 one at a time from saved main
predictions and recompute the four-horizon mean delta without refitting or
retuning. Require all three leave-year-out means >0 for each comparison needed
by a D17 stable-gain/formulation-advantage claim. An empty required horizon makes
the corresponding check unavailable, not passed. Exact empty-bootstrap handling
and quantile conventions will be specified. These checks add no model fits.

### Remaining decisions

1. Exact continuous-history feature set, observation count, source cutoff and age.
2. Correction decision policy: safeguards for scarce transitions and missing
   history; thresholds/calibration for the approved two directions and estimator.
3. Exact common raw-feature schema and estimator-specific missing-value handling;
   the six learned arms and RF history scope follow D9/D14.
4. Exact fold and reporting contracts under D11-D13's fixed training window,
   historical cutoffs, horizons, calendar and pooled IPC/CH cohort.
5. Exact decision thresholds, calibration and candidate configuration values
   within D15's approved budget and the approved training objectives.
6. Primary paired estimand, stability/uncertainty checks and continuation criteria.
7. Fresh artifact directory, reproduction evidence, execution and audit handoff.

Unresolved defaults above are not approved. Populate final testable
requirements and acceptance criteria after resolving the scientific choices;
then produce design.md and implement.md and converge the final spec.

## Audit handoff prerequisite — live check 2026-09-21

Repository is already registered; no audit run is active. The controller reports
an OPEN major gate for archived `ipcch-ch-gate-ablation`, job
`b0945b254eac1c8756430272`, status `needs_evidence`, against commit `608d5da`.
Current HEAD is `2dae94e`; subsequent fixes are not equivalent to a cleared gate.
The controller is stopped. This does not prevent planning, but ordinary wrapper
start must wait for the prior gate's repair/re-audit resolution. Do not bypass it,
mark this research task as remediation, or claim the prior audit passed.
Before implementation, recheck live status, executor identity and exact registered
path spelling; commit approved planning and use the wrapper to freeze base_sha.
