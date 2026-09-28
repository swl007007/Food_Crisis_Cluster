# Feature contract — approved exact defaults

D19/D20/D22 approve causal alignment, eight historical levels and the four
engineering blocks. User confirmation on 2026-09-28 approves their precise
realization here and the ordered schema in `feature-schema.json`, including
administrative-ID exclusion and all exact defaults.

## Inputs and time

Each row has area a, target T, horizon H and origin O=T-H. Use the full keyed
monthly scaffold to obtain features before selecting valid training targets.
Keys must be unique and valid; target and expert phases must be integral1..5 or
missing. Map 4/5 to4, export that class as `4或5`; never map missing to a class.
Keep raw phase and source-month provenance outside X. Verify observed raw
binary agrees with phase>=3 rather than trusting inconsistent representations.

The approved schema has 162 unique columns: 28 static +41 origin dynamic +15
existing covariate-derived +3 calendar +75 outcome-history. Both RF arms share
the definitions, order and missingness; each fits its own training-only imputer.
No current outcome, expert projection, assistance field or geographic identifier
is proposed as an input; identifiers remain keys and geometry/group metadata.

## Original covariate sources

The exact source names are in the JSON. Static roles are declared from geography
and soil/ecological meaning, never inferred from full-panel variation. Validate
within-area invariance of finite repeated values; a conflict stops preflight
for a documented source decision, not silent reclassification. Record the static
source/vintage; fixed geographic values do not establish historical publication
availability. Dynamic fields use the exact source row at O, with no earlier-row
substitution. Unknown market/crop/access timing is conservatively dynamic.

Retain only the 15 existing emitted covariate-derived columns, correcting calendar
and area grouping rather than expanding the old early-return loops:

- WFP_Price_m4/m12: sum the same area's monthly values over [O-W,O-1], W=4/12.
- nightlight_m12: corresponding 12-month sum.
- EVI_l1..l12: exact EVI at O-k. Never lag an aligned row a second time.

Require all W real monthly values for these sums; otherwise NaN. These are sums,
not means despite the old comments. No rolling window crosses an area boundary.
Use target_year and sin/cos(2*pi*(target_month-1)/12) as known calendar features;
do not discover dummy categories from future data. Do not retain duplicated
contemporaneous/lagged columns or old automatic L1/L2 discovery. The inherited
single-layer model consumes this full schema; do not enable the two-layer path.

## Historical levels and changes: 16 columns

At exact O-k, k=0/4/8/12, retain merged phase and phase>=3 binary (8 columns).
No exact observation means NaN. For (newer,older) offsets (0,4),(4,8),(8,12),
(0,12), compute phase difference and its sign (8 columns); either missing endpoint
gives NaN. Differences express category steps, not equal welfare distances.

## Observed-window summaries: 36 columns

For W=12/24/36, use valid same-area assessments in [O-W+1,O]. Each observed
assessment has equal weight; never expand it across unobserved months.
For each window export four class frequencies, minimum/maximum phase, upward/
downward transition fractions, observation count, eligible pair count, observed
span and latest observation age. The four frequencies are record frequencies,
not population shares. Min/max/frequencies need n>=1. Transitions use adjacent
observations with both endpoints inside the window; rates need n>=2 and divide
by n-1, including unchanged pairs in the denominator. These are fractions of
observed transitions, not monthly hazards. n_pairs=max(n-1,0).

No observations: counts0; other summariesNaN. One observation: span0,
transition fractionsNaN. Span=latest-earliest month; age=O-latest month.

## Events and observed run: 11 columns

Use all valid same-area observations through O. Export latest observed phase,
its age and a no-history flag. This is an additional feature, not a replacement
for missing exact-origin persistence or an alteration of D13 evaluation keys.
For crisis (phase>=3), phase4or5 and a phase-change event, export age since the
most recent qualifying observation/event and an absence flag. A change event
is a pair of consecutive observations with unequal merged phase; its timestamp
is the later endpoint, not the actual unobserved transition time. No event means
ageNaN, absence1; otherwise absence0. Unknown observation history is not safety.

The current observed run is the maximal equal-phase suffix of actual records.
Export its record count and first-to-last observed month span, not duration
through O. No history gives NaN/NaN; a single observation gives1/0.

## Limited interactions: 12 columns

Multiply each of four exact-origin phase indicators by each of the 12-month
upward fraction, downward fraction and phase4or5 frequency. If the origin phase
or summary is missing, the interaction is NaN even when an indicator might be0.
Do not use latest-observed phase as a substitute. Indicators are intermediate
calculations, not four extra feature columns. No arbitrary pairwise products.

## Fitting transforms

Fit the release max_plus imputer per estimator on real fitting rows only;
max*100, max0->100, all-missing->0, with release numeric coercion audited.
Negative maxima do not necessarily yield an out-of-range sentinel; disclose this
retained limitation. Keep every declared column. Transform validation/test with
the matching estimator's imputer. Add four Stage1 zero-feature synthetic rows
afterwards. Parent checkpoint inheritance copies the transform too. Stage3
pooled fallback reuses the pooled model AND transform, never local fills.

Record source keys, raw feature missingness and fill values. Imputed history
features never turn a missing observation into valid truth, baseline support or
an event. Focused fixtures must verify exact-month gaps, sparse windows, area
boundaries, origin cutoffs, no-history cases and feature-order consistency.
