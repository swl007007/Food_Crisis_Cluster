# FEWS NET clean persistence experiment — technical design

Status: final planning review; no implementation or experiment execution.
Authority: prd.md R1-R68/A1-A64 and research/decision-log.md D1-D64.
Research files retain exact schemas, formulas and code/source evidence. Later
numbered decisions supersede provisional open-language in earlier discussion.

## Boundary and minimal architecture

Create FEWSNETCleanPersistenceExperiment/ only after explicit execution
authorization. Keep the master, old experiments and released GeoRFBaseline
unchanged. Use a fresh run directory for every independent run; never overwrite
successful old outputs. Pin the baseline ZIP/payload, input snapshots, geometry,
runtime, ordered schemas and settings before fitting.

Keep ordinary Python entrypoints for preparation, orchestration and reporting:
prepare_data.py, run_pipeline.py and report_results.py, plus a focused contract
test module and README. Prefer existing pure helpers and the IPCCH verified
baseline extraction/import-isolation utilities; no plugin framework, generic
experiment engine, new model family or separate scheduling service is needed.
Adapt only inside this experiment or its run-local extracted baseline copy.

The data flow is:

    pinned sources -> validated monthly grid + observed IPC ledger
      -> frozen source schemas + A-E blocks / corrected reference
      -> recipe-specific Stage 1 candidate evidence
      -> recipe/role Stage 2 general maps + bounded completion
      -> rolling pooled/partitioned Stage 3 predictions
      -> 2018 month-pooled calibration -> 2020 thresholds + recipe choice
      -> frozen final evaluation -> paired metrics and robustness evidence

The original-feature corrected reference is independent of the updated-source
recipes. Both use the same corrected F1/no-SMOTE core and approved preprocessing.
Its matrix intentionally differs from the historical release defaults.

## Inputs and source contracts

Use DATA/FEWSNET_forecast_unadjusted_bm.csv as the master and
DATA/Outcome/FEWSNET_IPC/FEWSNET.csv for observed history. Do not adopt a newer
assembled panel. Exact identities/counts and D63 validation are in
research/target-label-contract.md. Preserve all master keys/labels. Valid truth
is original binary 0/1 consistent with unadjusted integer phase 1..5; missing
phase never means zero. Only the verified terminal artifact is excluded from
the parsed ledger with evidence. Unexpected malformed/duplicate/conflicting
data fails preflight. Earlier ledger observations may support eligible history;
they do not extend the master target cohort.

The complete ordered feature schemas are authoritative in
research/approved-feature-sources.md: 64 common old source columns, 86 updated
BASE columns, 67 old reference source columns and 109 total reference inputs.
Parse AEZ true/false explicitly to 1/0. Administrative/country/partition IDs are
metadata only. Remove fews_ha and inherited Rainf_zscore/Tair_zscore derivatives
from both arms. Remove FAO_price/WFP_Price/WFP_Price_std and derivatives only
from the updated arm; retain market_distance and food CPI/inflation.

Add exactly the selected ENSO file, two WB fields, coastline_dist and 18
Bloomberg fields. Retain supplied monthly values, units and series identities;
do not rebuild futures contracts, invent conversions or add new sources.
Bloomberg monthly exports match means of available daily numeric quotes, not
last quotes; missing source months remain missing. TTF is entirely missing but
stays in the approved schema. Native units/roll conventions and historical
publication timing remain qualified as documented, not silently guessed.

WB follows D9/D64: same source month, haversine radius 6371.0088 km, <=100 km,
cross-country allowed; exact distance ties choose lexical raw geo_id after
verified raw/derived lineage restoration. Both fields come from one market;
no averaging or farther-market substitution for a missing field. ENSO treats
the two documented sentinel encodings as missing. Coastline uses the containing
pixel at validated master WGS84 coordinates, preserving native signed/zero
values and invalidity reasons. No absolute-value conversion or interpolation.

Retain supplied Gini/nightlight and included old static snapshots under D33/D49,
with known generator, vintage and semantic limitations. Validate repeated static
and annual values before reducing them; never average conflicts. Read-only
consistency evidence and boolean parsing are in research/runtime-and-preflight.md.

## Calendar transforms and missing data

For every target row, including historical fitting rows, O=T-H, H in 4/8/12.
Transform the complete monthly grid before selecting labeled targets. Monthly
BASE uses exact O; B/C end at O and D uses those operands. Missing O is not
filled from an older source; E looks backwards only to measure age. No second
automatic horizon shift, data-driven role inference or sparse row-count lag.

GDP/general CPI/CC/gini use year(O)-1 throughout the origin year. Population
uses the last valid source month within that preceding year, preserving its
exact value/date. Missing required-year values stay missing. Annual E age uses
December of the latest eligible valid reference year; older-year lookup is
age-only. Included static snapshots stay fixed across all origins.

A: eight season/history features; B: 186 complete-window means/population-SDs
or sums over 3/6/12 months; C: 102 differences and preceding-12-month standardized
deviations; D: four specified products; E: 86 missing flags +54 dynamic ages.
Exact fields/order/formulas are in the source-schema research. Zero reference
variance gives missing, without epsilon; no partial-window statistics or source
imputation before feature construction. Reference history uses exact O-4/8/12
IPC, O-1..12 EVI, and O-exclusive approved sums plus target-date dummies.

Preserve max_plus x100: fit only real fitting rows, then transform validation
and targets. max=0 uses 100; an all-missing fitting column uses the existing
zero fallback. Do not drop constant/all-missing columns or call the fill an
observed zero. Stage 1 excludes internal validation and recovery rows from
imputer fitting; Stage 3 uses one full-pool imputer per recipe/horizon/fold,
shared by pooled and local RFs. Record ordered fill statistics and row identities.

## Time roles, finite search and job reuse

| Map role | Stage 1 target years | Map evidence cutoff | Prediction role |
|---|---|---|---|
| calibration | 2014-2016 | 2016-12 | 2018-02/06/10 |
| selection | 2016-2018 | 2018-12 | 2020-02/06/10 |
| final | 2018-2020 | 2020-12 | final horizon-specific dates |

For every Stage 1/3 RF, training targets lie in [O-35 calendar months,O).
Retain the effective 35-month mask despite config=36. Short real history is
allowed. Candidate eligibility requires nonempty real fitting, validation and
labeled target support; record exclusions. Artificial rows never supply support.
2019 is not a calibration/selection target year but may be eligible RF history.

Stage 1 enumerates only observed candidate target months within these windows:
2014/15 Jan/Apr/Jul/Oct and 2016 onward Feb/Jun/Oct, across all three scopes.
Pool all eligible months/scopes per recipe/role into one general map; explicitly
disable month_ind and inherited environment overrides. No seasonal alternatives.
The map cutoff includes candidate scoring labels, not only RF training labels.

Recipe order and widths are fixed:

| Recipe | BASE | A | B | C | D | E | ABCDE | BCDE | ACDE | ABDE | ABCE | ABCD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Columns | 86 | 94 | 272 | 188 | 90 | 226 | 526 | 518 | 340 | 424 | 522 | 386 |

Keep the separate 109-column reference. Every recipe/reference has compatible
development maps, predictions, calibrators and thresholds; no cross-recipe reuse.
Deduplicate only identical same-recipe candidate jobs in overlapping windows:
13*(33+27-9)=663 development jobs, plus 2*(27-9)=36 new final jobs, upper bound
699 before support exclusions. There are 26 development and 2 final map builds.
Stage 3 has 234 planned development folds (13 arms x 2 roles x 3 scopes x
3 dates) and 60 final folds (2 arms x 30 scope/date pairs); each fold's local
RF count depends on its map/support. These are job counts, not measured runtime,
total individual forest fits or guarantees of successful support.
Run bounded sequential jobs by default; reuse deterministic blocks and matching
artifacts without materializing all recipes/horizons simultaneously. Preserve
completed evidence; repair only affected jobs under the same scientific design.

## Stage 1 core and runtime

Reuse released GeoRF with class-1 F1 including FP, identical parent/child
validation support and exact strict gain >.01. SMOTE stays off throughout.
Retain one zero-feature recovery row per class inside Stage 1 RF fitting only.
Keep the released within-area .20 split, seed 42 and singleton handling.
Determine that split before imputation and supply it through GeoRF.fit's
existing explicit split/X_set interface; do not let the core split again.

Pin release/runtime/settings from research/runtime-and-preflight.md. RF uses
100 trees, unlimited depth and seed 5. Preserve partition/q and Stage 1 spatial
defaults; do not activate legacy p-value gates in the F1 route. Feed prepared
arrays/metadata directly rather than reusing legacy full-panel imputation,
automatic feature/drop inference or row-based lag construction.

Use a process per candidate, with explicit process/split/RF seeds, clean cwd
and run-local outputs. Record actual RF parameters and adaptive Stage 1 thread
settings; Stage 3 n_jobs=1. The locked Windows numerical stack must pass preflight.
Bind the FEWS shapefile and sidecars, required administrative-ID coverage and
run-local adjacency cache to source hashes. Do not import an IPCCH geometry
patch or cache without demonstrating need and compatibility.

## Stage 2 consensus and completion

Use the eligible completed candidate ledger. No eligible candidate or failed
required evidence stops that build. When all completed eligible weights are
nonpositive, record D19 valid unsplit and reuse full pooled predictions.

Otherwise use the D58 sorted valid-assignment union for that recipe/window.
Missing/s-1 is never shared co-membership. Clip F1 and baseline F1 to
[1e-6,1-1e-6]; weight=max(logit(F1)-logit(base),0). Sum weighted co-membership,
multiply by the haversine Gaussian sigma=5 degrees, normalize by global maximum.
Keep row top-k=40 with symmetric union, the retained self convention and all
nodes when n<=40; k40 does not mean 40 partitions. No pair-coverage normalization.

Select the largest positive off-diagonal connected component, breaking size
ties by its smallest canonical area code. Both normalized-Laplacian eigengap
and spectral fitting use this exact component. Require n>=3; eigen request
k=max(2,min(20,n-2)), sorted eigenvalues, first argmax(diff)+1. Pin eigsh's
initial vector with a local default_rng(42), recording it and matrix identity;
no alternate seed/nc search. Preserve spectral precomputed affinity, kmeans
assignment and random_state=42. Invalid support/solver failure is incomplete,
not permission to cap nc or fabricate a no-split result.

Complete all remaining master areas using only original spectral-core donors:
haversine radius 6371 km, <=100 km, cross-country, exact ties by smallest
canonical area code. No chaining or adjacency-first fill. Invalid-coordinate
or over-cap recipients remain -1 with pooled routing. Preserve core, completed,
other-component and never-in-graph provenance. No extra polygon-vote smoothing
before/after completion; geographic islands may remain. Freeze maps per role.

## Stage 3 and probability correction

Use the full eligible rolling labeled history, independently of target-month
area/group coverage and persistence availability. Pooled fits all rows; each
local RF fits its assigned subset, including completed areas' eligible labels.
Unassigned rows contribute to pooled. Local support is >=50 real rows and both
classes; all other local routes, including unseen groups, use pooled for both
native hard prediction and class-1 probability. Empty global training or actual
fit errors fail the fold, not statistical fallback. Save map and model routes
separately. For valid automatic nc=1, reuse pooled for every row without a
local subset fit, preserving geographic coverage and the distinct unsplit reason.

Persistence is valid observed binary at exact O. No latest-value substitution;
A independently may use latest valid <=O. Unavailable persistence stays
unavailable and excludes only paired correction/comparison support, not RF fits.
All methods/arms/recipes share the same valid-persistence labeled keys within
each horizon for threshold/recipe scoring and paired final comparisons.

Fit only month-pooled calibrators for each recipe/horizon/calendar month from
all eligible 2018 labeled predictions, including unavailable persistence.
Preserve existing isotonic/Platt/identity behavior documented in runtime research;
no partition-ID calibration transfer. Freeze by 2018-12 and retain those fits.

For each recipe/horizon choose separate raw/calibrated thresholds from 2020
Feb/Jun/Oct: scan sorted distinct observed probabilities, strict p>tau up-only
override of b=0, pooled class-1 F1, retaining smallest improving tied tau.
No strict gain gives tau=null and unchanged persistence. No extra below-min
candidate or threshold grid. b=1 always stays one. Missing b has no correction.

Select the updated recipe by the equal mean of six 2020 correction-minus-
persistence F1 gains (three horizons x two variants). Exact ties choose fewer
columns then fixed manifest order. Freeze winner/thresholds by 2020-12. Final
maps use only winner/reference; RF may refit on later eligible history, while
maps, recipe, calibrators and thresholds stay fixed.

## Final evaluation, uncertainty and stop rule

Final observed targets end 2024-10: fs1 starts 2021-06 (11 dates), fs2 2021-10
(10), fs3 2022-02 (9). Preserve all eligible areas without a balanced-area
requirement. Report a supplementary common-calendar 2022-02..2024-10 view from
the same predictions. No new fits or alternative primary endpoint.

Nine unique paired streams per horizon: common persistence, and each arm's
pooled RF, partitioned RF, raw correction and calibrated correction. RF rows
use native classifier decisions; no pooled-correction/XGBoost arm. Also report
full-support standalone RF metrics separately, reusing predictions.

Primary adjudication is the frozen updated winner's equal mean of six final
signed absolute-unit F1 gains over persistence. Report original-reference
increments and individual cells secondarily. +.01 is advisory, not a magnitude
gate. Robust benefit requires positive mean, paired two-sided 95% CI lower>0,
and positive recomputed mean after dropping each target year 2021..2024.

Use D44 joint target-date bootstrap: default_rng(5), 11 draws from the sorted
11-date union per attempt, shared date multiplicities across all methods/arms/
horizons, all areas within a date together. Recompute weighted confusion counts
then F1 and mean gains; 2,000 valid replicates, cap 20,000 attempts. Redraw only
empty required-horizon samples, never based on gain; nonempty zero-denominator
F1=0. Linear .025/.975 quantiles. Year exclusions do not refit anything.

Complete evidence passing the robustness rule supports further research;
complete evidence failing it ends this bounded feature search and informs
the next expert-unavailable fallback discussion. Incomplete evidence requires
repair under the same design, not a scientific null. Neither outcome permits
post-evaluation retuning or automatic fallback implementation.

## Artifacts, reproduction and limitations

Save source/release/runtime/config/schema manifests; candidate status, support,
split, scoring and correspondence evidence; Stage 2 weights/graphs/component/
eigenvalues/maps; per-fold training/imputation/model-route evidence and per-row
predictions; calibrators, threshold traces, recipe scores and freeze manifests;
paired keys, exclusions, counts, bootstrap weights/gains and final reports.
Record synthetic recovery separately from real support. Failed work gets an
explicit status, never a successful empty metric table.

Recompute metrics/selection/uncertainty from stored evidence independently of
the orchestration. Validate at least one bounded complete development chain
under the locked runtime before the full manifest, without changing design
from its outcomes. Fresh-source preflight and measured early support are execution
checks; planning-only source scans are not model feasibility or result evidence.

Retrospective outcomes were already inspected historically. Source snapshot
vintages, Gini/nightlight construction, static-layer semantics, new-source units/
futures rolls and publication timing remain documented limitations. Bootstrap
inference is conditional on the frozen pipeline/observed geography with only
9-11 date blocks; it does not cover full selection uncertainty or arbitrary
serial dependence. No claim of untouched holdout or verified real-time data.

Rollback means leaving protected sources/packages/results untouched and using
a new run root or recorded affected-job repair. No destructive cleanup, publication,
commit or push is included in this planning phase.
