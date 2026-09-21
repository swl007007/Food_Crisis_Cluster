# Three-scope support and time constraints — 2026-09-20

Planning-only read-only inspection. Two scouts traced code and streamed source
date/label-presence fields; no training, performance scoring or model selection.
The main session checked the common split mask and Stage 2 weight formula.

## The existing consensus pools scopes

`spatial_weighted_consensus_clustering.bat:174-175,190-191,257` collects fs1/fs2/fs3
into one results/correspondence pool. General consensus runs once (`:296`);
month-specific consensus filters months 2/6/10, not scopes (`:351,366`).

The released equivalent retains scope in plan identity:
`GeoRFBaseline/scripts/step1_merge_results.py:172-198` and
`scripts/step3_create_linked_tables.py:62-63`. Its step4 filters only calendar
month for month-specific matrices (`scripts/step4_similarity_matrix.py:229-265`).
Weights are `max(logit(f1)-logit(f1_base),0)` (`:50-64`); nonpositive weights
are skipped (`:141-145`). Including fs3 increases eligible candidate evidence,
but does not guarantee positive-weight plans or a nontrivial consensus partition.

## Persistence extension must cover every layer

In `PersistenceCorrectionExperiment/persistencecorrection/persistence.py`,
`:70` permits only fs1=4/fs2=8; `require_horizon` (`:169-175`) rejects fs3.
The reference F1 dictionary (`:122,344`) also has no fs3 value. Builder defaults
and historical run_selection/run_adjudication/protected-input lists assume two
scopes. New isolated orchestration must not inherit these loops unchanged or
invent a historical reference F1 for fs3.

`fit_calibrators` accepts an integer scope without a two-scope whitelist
(`calibration.py:414-435,509`); the old calibration runner nevertheless reaches
the rejecting persistence helper first (`run_calibration.py:84-89`).

Persistence history uses an exact calendar match to T-H (`persistence.py:145-154,
252-285`), not latest-observation fallback. Missing row coverage is fatal in the
old mechanism. Coverage at H=12 still needs explicit validation for the chosen
cohort and windows.

## Source label support

Scanned `Analysis/1.Source Data/FEWSNET_forecast_unadjusted_bm.csv` and
`Analysis/1.Source Data/Outcome/FEWSNET_IPC/FEWSNET.csv` using csv.DictReader,
counting nonempty labels and dates only. The master has 68,616 rows per year in
2014-2024; its monthly skeleton is not a monthly observed-label series.

| Year | Nonempty observed labels in each source | Labeled months |
|---|---:|---|
| 2014 | 19,589 | 01,04,07,10 |
| 2015 | 20,036 | 01,04,07,10 |
| 2016 | 15,684 | 02,06,10 |
| 2017 | 15,942 | 02,06,10 |
| 2018 | 16,093 | 02,06,10 |
| 2019 | 16,095 | 02,06,10 |
| 2020 | 16,439 | 02,06,10 |
| 2021 | 14,784 | 02,06,10 |
| 2022 | 14,850 | 02,06,10 |
| 2023 | 15,825 | 02,06,10 |
| 2024 | 16,730 | 02,06,10 |

Master fews_ipc/fews_ipc_crisis presence and ledger fews_ipc counts agree by
year/month; this is not a per-key/value reconciliation or publication-date audit.
The observed ledger has one missing-year_month record outside these counts.

Later read-only inspection corrected the earlier assumption that labels begin
in 2014. The table above covered only 2014-2024. The current master has 1,029,240
rows covering 2010-2024 (68,616 per year), with both observed phase and original
binary label present on 19,062 / 19,434 / 19,437 / 19,440 rows in 2010 / 2011 /
2012 / 2013 respectively. The main session independently streamed both label
fields: their nonempty status agrees on every master row. This is a presence
check, not a valid-phase or value reconciliation. Earlier master rows remain
eligible RF history under D20; the approved candidate target windows do not
change. Earlier master/ledger per-key/value agreement remains to be checked.

## Existing training cutoff and freeze consequences

Stage 1 wrapper year/month denotes target month (`GeoRFBaseline/scripts/
run_stage1.py:43-49`). Stage 1 and Stage 3 call the same split function
(`app/main_model_GF.py:426-430`; `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py:
993-998,1011-1016`). In `src/customize/customize.py:408-421`:

    origin = target_month_start - active_lag_months
    train_start = origin - (train_window_months - 1) months
    train_mask = (date >= train_start) & (date < origin)

Thus the configured 36-month window contains 35 calendar months and excludes
origin-month labels. D20 explicitly retains this convention. This training mask
is distinct from historical persistence allowing an observation at origin;
the persistence availability convention still needs to be fixed separately.

If all selected artifacts are deemed available at month F and require F<=origin,
the earliest eligible target among February/June/October is:

| F | H=4 | H=8 | H=12 | Earliest shared target |
|---|---|---|---|---|
| 2020-12 | 2021-06 | 2021-10 | 2022-02 | 2022-02 |
| 2021-12 | 2022-06 | 2022-10 | 2023-02 | 2023-02 |
| 2022-12 | 2023-06 | 2023-10 | 2024-02 | 2024-02 |

This is conditional calendar arithmetic, not a selected protocol or measured
artifact release date. All labels used in feature selection, partition scoring,
calibration or threshold selection contribute to the artifact's information
cutoff. A source target month and a Stage 1 training-window end are not the same.

On 2026-09-20 the user approved the forecast-origin cutoff for all upstream
learned/selected artifacts (PRD D12) and one development/freeze cycle followed
by rolling RF refits during evaluation (D13). Final partition learning and the
calibration/threshold time roles are now confirmed below; feature-selection
chronology and early partition support still need resolution.

Previously inspected 2021-2024 outcomes cannot become a fresh unseen holdout by
renaming the split. A retrospective evaluation can still enforce within-run
forecast-origin discipline, with its scientific evidence limits stated.

## User-directed calendar revision — learning boundary confirmed

On 2026-09-20 the user preferred retaining "2018-2021" partition learning and
reserving a gap before Stage 3: fs3 should start at 2022-02, with an eight-month
gap for fs2. The user emphasized that Stage 2 pools fs1/fs2/fs3, so all forecasts
using the resulting map inherit information contributed by the other scopes.
The earlier unapproved 2016-2018 partition-learning proposal is superseded.

Use one common latest information cutoff across the evidence contributing to
the pooled maps, including validation and consensus-scoring/weighting labels.
Do not assess a map's eligibility using only the evaluated scope's training
cutoff. Sharing evidence across scopes is permitted; using that evidence before
it is available at the forecast origin is not.

The user subsequently confirmed that "2018-2021" means [2018-01, 2021-01):
2018-2020 inclusive, excluding 2021. With a conservative 2020-12 partition
freeze, the first eligible observed targets are fs1 2021-06, fs2 2021-10 and
fs3 2022-02. The alternative including all of 2021 is not the selected window.

These dates establish partition eligibility. D17 also fixes the development
time roles below; they do not establish measured source-publication dates.
Stage 1 target dates, training-label dates and dates of labels used to score
candidates must remain distinguishable in the evidence ledger. Feature
selection must also respect the approved freeze. Any later information would
supersede the partition cutoff for the complete correction pipeline and require
an explicit revision, not silent retention of the earlier evaluation start.

## Temporary development partitions — APPROVED in principle

Keeping the above starts for final correction evaluation requires freezing
feature selection, calibration and thresholds by 2020-12 as well. Reusing the
final 2018-2020 maps for earlier development predictions would retain the old
development overlap: those predictions would use partitions informed by labels
later than their origins. A later final evaluation can still be origin-safe,
but it does not make those internal development predictions temporally valid.

The user approved additional temporary partition runs to generate
earlier development predictions in time order, with each fold's map using only
evidence available at that fold's origins. Final maps still pool the
approved 2018-2020 evidence once and remain fixed during evaluation. This
approval did not by itself settle fold dates. D17 subsequently fixes the target
dates and cutoffs below; historical support and feature-selection nesting remain
open. Calibration grouping was approved separately; integer partition IDs do
not establish matching spatial groups.

## Calibration routing inspection and approved grouping

Read-only code inspection of PersistenceCorrectionExperiment/persistencecorrection/
calibration.py confirms one CalibratorSet per scope, with month-pooled and
(calendar_month, partition_id) local calibrators (:276-288). The application
looks up local integer IDs first (:363-382), without map identity; matching
numbers from different maps can silently select the wrong local calibrator.
The current month pool (:442-470) fits all partitioned-RF probabilities for that
calendar month within the scope. It is not calibration of pooled-RF predictions.
The old runner fits scopes separately (run_calibration.py:131-174).

The user approved using the existing month-pooled fitting/application
logic as the only calibration route, separately by horizon and target calendar
month within each feature arm. RF training remains partitioned. This removes
the need to align temporary and final partition IDs, and increases calibration
support relative to local groups. It deliberately gives up partition-specific
calibration and must not be presented as identical to the historical protocol.
Temporary-to-final model probability shifts can still affect calibration quality;
compatible group keys do not prove statistical transferability.

The existing CLI has no pooled-only switch. New isolated orchestration can reuse
the underlying month-pool Calibrator.transform; record an intentional pooled
route instead of disguising it as an accidental missing-group fallback. Do not
edit the protected historical experiment. Retaining pooled fitting also retains
its existing isotonic/Platt/identity behavior unless explicitly changed; missing
calendar-month coverage currently raises rather than inventing a calibrator.

## Early development support — code constraints, not run evidence

A second read-only inspection confirms no complete-35-month-history gate in
GeoRFBaseline/src/customize/customize.py:412-420: the mask uses whatever rows
exist. Its actual_train_months at :451 is a theoretical calendar count, not an
observed support count. The Stage 1 wrapper has no lower-year guard
(scripts/run_stage1.py:34-49), but missing-label target months are removed by
src/preprocess/preprocess.py:196-197 and skipped by app/main_model_GF.py:447-451.
Default within-area validation sends singleton groups entirely to training
(src/utils/split.py:63-80), so a nonempty early history need not provide usable
validation support.

Stage 2 has no minimum of two candidate plans (scripts/step4_similarity_matrix.py:
229-245), but nonpositive weights contribute no edges (:143-145), and graph
size/connectivity still constrain spectral clustering. Weights use Stage 1
target-month F1 (app/main_model_GF.py:615,738-752; scripts/
step3_create_linked_tables.py:53-65), so temporary-map cutoffs must include those
target labels. Early area/class support, positive weights and connectivity have
not been measured; no runnable development calendar is established by this check.

## Approved development calendar — D17

After approving D16 grouping, the user adopted the following time-role assignment:

| Role | Target months | Temporary-map information cutoff | Calibrator information cutoff |
|---|---|---|---|
| Fit calibration | 2018-02, 2018-06, 2018-10 | 2016-12 | Fitting here; freeze by 2018-12 |
| Select raw/calibrated thresholds | 2020-02, 2020-06, 2020-10 | 2018-12 | 2018-12 for calibrated variant |
| Final evaluation | D14 horizon-specific starts onward | Final maps use 2018-2020, freeze by 2020-12 | Retain the development fit; thresholds freeze by 2020-12 |

Calendar arithmetic was checked for all scopes using O=T-H. For the earliest
2018-02 calibration target, fs1/fs2/fs3 origins are 2017-10, 2017-06 and 2017-02,
all after the approved 2016-12 temporary-map cutoff. For the earliest 2020-02
threshold target, origins are 2019-10, 2019-06 and 2019-02, all after both the
approved 2018-12 temporary-map cutoff and calibrator freeze. Later June/October
targets also pass. These checks establish date ordering only, not actual source
availability, sufficient geographic/class support or successful partition runs.

2019 is outside the calibration-fit and threshold-selection target sets; its
labels can still enter RF rolling histories once eligible at each origin.
Both probability variants select thresholds on the same approved 2020 keys.
The approved calibration period supplies only one target date per calendar-month
pool, albeit many area rows; this limits temporal representativeness. Do not
reuse 2019 calibration labels for 2020 fs3 origins before those labels exist.

Stage 1 candidate-year windows are fixed in D18, D19 fixes the no-split rule,
and D20 fixes support gates. D21 fixes feature-selection chronology. Measured
support, source availability and the exact selection rule still need resolution.
No feature winner selected using 2020 outcomes may be retroactively treated as
already selected at an earlier development origin. The finite candidate and
selection protocol must reconcile this before the complete calendar is frozen.

## Approved candidate-year windows — D18

The user approved three-year Stage 1 target-year candidate windows ending at
each approved map cutoff:

| Map role | Candidate target years | Status |
|---|---|---|
| Temporary map for 2018 calibration predictions | 2014-2016 | Approved in D18 |
| Temporary map for 2020 threshold-selection predictions | 2016-2018 | Approved in D18 |
| Final evaluation map | 2018-2020 | Already approved in D14 |

All retain fs1/fs2/fs3 pooling. The window denotes candidate target months whose
labels score Stage 1 plans; it does not replace the per-origin RF training mask.
The master has observed labels from 2010, so a 2014 candidate is not automatically
history-free. Actual per-origin fitting/internal-validation support must still
be measured under D20; candidate-year inclusion does not prove that all
month/scope combinations can run. Preflight
eligibility is now fixed in D20 below; actual early feasibility remains to be
measured. D19 fixes the no-split outcome. No future-map substitution is permitted.

## Approved no-split/failure distinction — D19

The user adopted the distinction below; D20 subsequently fixes the support gates:

- A candidate lacks the predeclared training/validation/target support: record
  its support counts and exclusion reason under D20. Do not invent thresholds
  from eventual evaluation performance.
- Eligible candidates complete normally but none contributes positive Stage 2
  consensus weight: record a valid unsplit map, represented as one partition.
  Its prediction arm uses the corresponding pooled RF predictions on the same
  feature schema, training rows and evaluation keys. In development this means
  the calibration input can be pooled RF because the temporary map did not
  support a useful split; it must not be described as a learned multi-partition
  model. Transfer to a later split model may still change probability behavior.
- No candidate is eligible, or an execution/artifact failure prevents producing
  the required evidence: stop the affected map build and report the reason.
  Do not convert program failures into statistical no-split evidence or borrow
  a future map to fill the gap.

This is an approved scientific no-split rule, not current built-in Stage 2
behavior. It does not authorize silent graph-parameter changes, reduction of
support requirements or new fallback models. Positive-weight graphs that are
too small/disconnected for the selected consensus settings still need an
explicit handling rule before execution.

## Approved history/support rules — D20

The user adopted preservation of [O-35 calendar months, O), with O=T-H. Its configured
value is 36 but its effective calendar span is 35 months and origin-month labels
are excluded. Use the real history available within that span without adding a
complete-35-month or a new 12/24-month minimum-coverage requirement. This is
preservation of the existing mask, not correction of its off-by-one convention.

Retain the released within-area random validation split. Direct inspection of
GeoRFBaseline/src/utils/split.py:63-80 confirms singleton groups stay in training,
while groups with at least two rows retain at least one training row and receive
max(1, ceil(n * val_ratio)) validation rows, capped at n-1. A candidate must
have nonempty real model-fitting, real internal-validation and labeled target
samples. Artificial class-recovery rows cannot satisfy these real-support gates;
their existing retention follows the release-reuse decision D4. Record actual observed months,
area coverage and class counts, rather than relying on the theoretical calendar
count logged as actual_train_months. These approved support gates are evaluated
after the applicable preprocessing and split; extra class-count or minimum-area
thresholds are not implicitly introduced here.

The release explicitly calls class-recovery fitting user-approved
(GeoRFBaseline/README.md:42-47). Its RF fitting method appends pseudo rows at
src/model/model_RF.py:244-247, generated as one zero-feature row per class at
:352-358. Reusing that release preserves this behavior; it is not SMOTE and
does not authorize treating those rows as observed history. Stage 3 uses real
rows under its existing small/single-class pooled fallback. No additional
retention confirmation is needed for a behavior already approved and included
in the selected release.

## Approved feature-selection chronology — D21

The user adopted this flow: predeclare a finite inventory of feature recipes before viewing
new candidate performance. Each candidate produces its own origin-safe
development evidence with the approved temporary-map, calibration and threshold
windows. Data-dependent transforms are fitted inside each allowed training
window. Compare candidate recipes using 2020 development outcomes, select one
engineered recipe shared across the three horizons, and freeze that choice by
2020-12 before final evaluation.
The selected feature recipe's final map still uses only its approved 2018-2020
partition-learning window. The original-feature corrected reference is retained
separately, and raw/calibrated probability comparisons remain predeclared.

Earlier development predictions are predictions of specified candidates, not
evidence that the eventual 2020-selected system had already been selected at
those origins. Do not use the eventual winner to adapt earlier folds or train
supervised transforms with future outcomes. Selecting recipes and thresholds
on 2020 outcomes can overfit that development set; only the later frozen
evaluation adjudicates performance. Exact finite feature inventory and candidate
budget remain open; D22 fixes scoring and tie-breaking. Each candidate needs
its own compatible development partitions. This approval does not authorize trying additional recipes
after final-period results are seen.

## Approved feature-selection score — D22

The user adopted the following score. For candidate c, horizon h in {4, 8, 12},
and correction variant v in {raw, calibrated}, compute:

    delta(c,h,v) = F1_2020(corrected_prediction(c,h,v)) - F1_2020(persistence(h))
    score(c) = sum(delta(c,h,v) for all six h,v cells) / 6

Each cell aggregates TP/FP/FN across its 2020 February/June/October rows before
computing class-1 F1. Do not average per-area or per-month F1 values. Within a
horizon, all candidates, both variants and persistence must use identical
predeclared evaluation keys; a candidate cannot improve its score by dropping
hard rows. Each horizon and each probability variant receives equal weight,
rather than letting a horizon with more rows dominate the recipe choice.
Both thresholds remain separately fitted on the approved 2020 selection data.

Choose the greatest score; exact ties prefer fewer frozen model-input feature
columns, then the earlier candidate in the predeclared manifest. Required
missing cells cannot be silently omitted or assigned a fabricated score; resolve
incomplete evidence before ranking. Cohort construction and missing-feature
preprocessing still need their own concrete contract. This score chooses a
feature recipe; it does not yet define the final go/no-go success criterion,
minimum useful gain or uncertainty rule.

## Approved final evaluation windows — D40

The user adopted D14's horizon-specific starts and a common end of 2024-10,
the last observed target month in the historical master inventory above.
Use the existing February/June/October observed-target schedule:

| Horizon | Start target | End target | Scheduled target months before support checks |
|---|---|---|---:|
| fs1 / 4 months | 2021-06 | 2024-10 | 11 |
| fs2 / 8 months | 2021-10 | 2024-10 | 10 |
| fs3 / 12 months | 2022-02 | 2024-10 | 9 |

For paired evaluation, apply D36's valid exact-O persistence and valid-target
support consistently across D39's method/feature rows within each horizon.
Retain all eligible area-target observations; do not require a balanced set of
areas observed in every target month. Missing features use the approved
preprocessing rather than removing rows. Report per-month/horizon support and
exclusion reasons; a scheduled date is not proof of actual nonempty support.
Do not add unlabeled November/December dates or extend to a newer source panel.

The main horizon-specific tables use each full window above. Additionally,
report a supplementary common-calendar view for 2022-02 through 2024-10 (nine
target months each) from the same stored predictions. This aligns calendar
windows, not necessarily area-target support across horizons: exact-O IPC
availability can still differ. State that distinction and each cohort size;
do not interpret cross-horizon F1 differences as an isolated horizon effect.

The supplementary view adds no fits, recipe selection, threshold tuning or
alternative success rule after results are observed. The original full windows
remain primary reporting windows; the final scientific adjudication rule still
requires an explicit decision. All dates remain a retrospective evaluation of
already-inspected historical years under the documented source assumptions.

## Approved primary final adjudication metric — D41

The user adopted the same equal-weight six-cell structure as D22, evaluated with the
frozen updated-feature winner and D40's full per-horizon final windows:

    delta_final(h,v) = F1(winner_correction(h,v)) - F1(persistence(h))
    primary_gain = sum(delta_final(h,v) for h in {4,8,12}
                       for v in {raw,calibrated}) / 6

Compute each F1 from pooled confusion counts over its whole approved horizon
window and D36's identical within-horizon paired support. Do not average local
F1 values, weight horizons by sample size, select the best correction variant,
or replace the full windows with a favorable common-calendar result. Report
all six cell gains, not only the mean. The variants share predictions/labels
and the horizons overlap in targets, so these are not six independent studies.

Persistence is the primary comparator for deciding whether the frozen proposed
correction system adds value. Report the original-feature corrected reference's
same six-cell score and matched winner-minus-reference gains to identify the
increment attributable to the updated sources/engineering. Keep these and the
standalone RF comparisons as secondary evidence, without selecting an alternate
primary system after observing final results. Passing a persistence comparison
alone does not establish an incremental feature-engineering benefit over the
corrected reference; those conclusions must be separated in the report.

This explicitly replaces the historical fs2-primary/fs1-directional gate
with a predeclared three-horizon, two-variant summary. D42 makes the effect-size
reference advisory, D43 fixes robustness conditions and D44 fixes resampling.
D45 fixes the final continue/stop interpretation below. An
incomplete pipeline or missing required evidence is not a scientific null.
This approval remains planning-only and does not authorize execution.

## Approved advisory effect-size reference — D42

The user proposed relaxing the suggested improvement to +0.01 and emphasized
that any robust positive gain can be useful against the strong baseline.
Accordingly, use +0.01 as a descriptive reference, not an absolute success
cutoff. The previous +0.02 hard-gate proposal is superseded without adoption;
the historical experiment's numerical gate is not carried into this trial.

Report actual model and persistence F1 levels, all six signed F1 differences,
their equal mean, and the matched winner-minus-reference results. Values are
in absolute F1 units: +0.01 is one F1 percentage point, not a one-percent
relative change. Preserve full precision for calculations. Absolute units do
not mean taking abs(delta): a negative difference remains a decline.

A gain such as +0.004 is not automatically a failure; it can be useful if the
predeclared robustness evidence supports it. Conversely, a gain above +0.01 is
not automatically robust or a success. The result may be described relative
to the reference line, but that description does not replace uncertainty and
stability checks. D41's primary metric and secondary attribution comparisons
are unchanged. No new search, model fitting or source reconstruction is authorized.

## Approved robustness decision rule — D43

The user adopted two requirements for calling D41's aggregate gain robust:

1. Its point estimate is positive and its paired, two-sided 95% interval has a
   lower endpoint strictly above zero.
2. Recompute the same six-cell mean after excluding each target calendar year
   (2021, 2022, 2023, 2024) in turn; every such mean remains strictly positive.

For the leave-one-year-out calculation, drop that target year from all relevant
horizons, variants and comparator rows together, retain each horizon's remaining
approved support, and recompute pooled confusion counts before averaging the
six gains. Do not refit models, maps, calibrators or thresholds. Excluding 2021
does not remove fs3 rows because its final window begins in 2022. A missing
required cell makes the check incomplete, not a positive or zero substitute.

D44 specifies the resampling unit, paired multi-horizon draws, replicate count
and interval construction below. The two variants and
overlapping horizons cannot be treated as six independent replications.
Interval precision is limited by the short retrospective evaluation period;
the check does not erase source-vintage or already-inspected-outcome limitations.

Report all six individual gains, including any negative cells. Do not require
each individual cell to be significant as an additional unapproved gate, and
do not describe an aggregate gain as a demonstrated gain at every horizon or
for both variants. If the aggregate criteria fail, report insufficient evidence
of a robust overall benefit rather than proof of zero effect. D45 fixes the final
research continue/stop rule below. This approval remains planning-only.

## Approved joint bootstrap protocol — D44

The user adopted adaptation of the historical target-month block bootstrap to one shared
joint draw, retaining 2,000 replicates and seed 5 without RF refitting:

1. Freeze D36/D39's paired keys and predictions. Aggregate n/TP/FP/FN/TN for
   every horizon, method/feature arm and target date. Preserve the structural
   absence of pre-start dates rather than inventing observations.
2. Use the sorted union of D40's 11 scheduled target dates. Each attempted
   draw samples 11 dates with replacement using numpy.random.default_rng(5).
   A date's multiplicity is shared by all horizons, variants and comparators;
   all eligible areas in that date block remain together. Each horizon uses
   only its own approved window and cohort.
3. Sum the weighted confusion counts, calculate each method's pooled F1, form
   the six paired gains over persistence and average them equally. Produce one
   primary_gain per draw. Do not average month F1 scores, separately bootstrap
   horizons/variants, average six CI endpoints or resample individual area rows.
4. An attempted draw with zero total observations for any required horizon is
   undefined: record it and draw again. Do not treat missing horizons as zero
   gains or average fewer cells. Collect 2,000 valid draws in at most 20,000
   attempts; otherwise report incomplete uncertainty evidence. A nonempty
   cell with zero F1 denominator follows the established F1=0 convention and
   is not rejected. Gain sign or size never determines draw acceptance.
5. Use linear-interpolated 2.5th/97.5th percentiles of the valid joint gains
   for the two-sided 95% interval. Preserve the original point estimate,
   bootstrap gains, shared date multiplicities, attempt/rejection counts,
   seed and NumPy identity so the interval can be independently recomputed.

Shared draws can also supply per-cell and matched secondary intervals without
new fits; only D41/D43's declared aggregate interval has primary decision power.
D43's four leave-one-target-year-out checks remain separate metric recomputations.
No feature, threshold, calibration or model selection is repeated inside the
bootstrap. These intervals are conditional on the frozen pipeline and observed
geographic support; they do not quantify the entire selection/training process
or generalization to newly sampled locations.

Rejecting empty-horizon draws conditions the empirical resampling distribution
on complete horizon support. Report the rejection rate instead of concealing
this detail; its actual frequency cannot be inferred from calendar counts alone.
Only 9-11 target blocks are available per horizon. Date-block exchangeability
and remaining serial dependence limit coverage claims, and year-exclusion
stability does not prove that assumption. Do not present the nominal 95% interval
as verified coverage under arbitrary temporal dependence or as fresh untouched
holdout inference. This protocol is approved for planning only.

## Approved research continue/stop contract — D45

The user adopted the following interpretation after the frozen experiment is completed
and its required lineage, predictions, metrics and robustness evidence pass
the implementation/reproduction checks:

| Evidence state | Scientific interpretation | Research consequence |
|---|---|---|
| Complete and all D43 conditions pass | Robust positive aggregate benefit under the stated retrospective/source/resampling assumptions; +0.01 remains advisory | Preserve the baseline and frozen winner as evidence supporting further work on this system; plan any follow-up separately |
| Complete but any D43 condition fails | Insufficient evidence of a robust aggregate benefit, not proof of zero effect | End this bounded feature-search cycle; direct the next research discussion toward forecasting when expert predictions are unavailable |
| Required support, execution, lineage or metric/interval evidence is incomplete/invalid | The experiment has not delivered an assessable result | Resolve the specific failure within the approved design; do not classify it as a scientific null or fabricate substitute outputs |

The primary decision concerns the frozen updated-feature winner under D41.
Report favorable or unfavorable original-feature/reference and individual-cell
results honestly as secondary evidence; do not swap them into the primary role
after seeing final results to reverse its verdict. Any claim that engineering
itself improved the clean reference requires the separate matched comparison.

Both complete outcome branches close this predeclared experiment after review
and reproducibility evidence; a null result is a valid completed deliverable.
Neither branch authorizes a new feature recipe, retuned threshold, shifted
window or automatic new task/model run. The fallback direction is a next-task
research discussion, not implementation included in the current scope.

Implementation defects may be repaired and their affected artifacts rebuilt
under unchanged scientific contracts with the previous outputs and repair
provenance preserved. If a repair requires changing a frozen scientific choice,
return to the user for that explicit design decision rather than treating it
as routine bug fixing. This outcome mapping is approved for planning only;
it does not authorize implementation or experimental execution.

## General/month-specific map routing — inspected behavior

Two bounded read-only scouts inspected current released-directory Stage 2/3
code and selected historical manifests. The main session checked month filters,
target-month routing, eigengap and largest-component fitting. No map or model
was built. Current directory behavior is not a verified byte comparison with
the published release ZIP.

- `GeoRFBaseline/scripts/step4_similarity_matrix.py:23-28,229-237` accepts
  month 2/6/10 or no month. No filter means all eligible plans; monthly mode
  keeps only exact candidate calendar-month matches and fails if none remain.
  The 2014-2015 Jan/Apr/Jul candidates therefore enter general maps but not
  m2/m6/m10; their October candidates can enter m10. Do not rename seasons.
- `spatial_weighted_consensus_clustering.bat:174-175,190-191,296,366` pools
  fs1/fs2/fs3, then builds general and month-specific matrices separately.
  Scope pooling and month filtering are independent choices.
- `GeoRFBaseline/scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py:893-900,
  965-988` uses one supplied general map throughout if month_ind is false.
  With month_ind true, target calendar month 2/6/10 selects m2/m6/m10; other
  months use general. A missing selected monthly CSV is not a general-map
  fallback: :987 calls its reader before the per-fold try block.
- `run_partition_k40_comparison_unified.bat:134-135,296-307` can also enable
  month routing from MONTH_IND=1. Experiment-local orchestration must supply
  the chosen routing explicitly, not inherit that environmental choice.
- Historical `PersistenceCorrectionExperiment/outputs/phase2_probabilities/`
  `probs_2018_2020_fs1/run_manifest.json:35-39` and
  `probs_2021_2024_fs2/run_manifest.json:35-44` enable month_ind and name
  refined m2/m6/m10 maps (nc13/11/16 respectively; general nc17). These two
  manifests' corresponding map hashes match across scopes. Their old paths
  and in-sample development maps are comparison evidence, not permitted new
  artifacts or proof of origin-safe development.

## Approved map granularity — D55

The user adopted one general consensus map per feature recipe and map-role window,
pooling all eligible observed target months and all fs1/fs2/fs3 candidates
inside that window. Reuse that role's single map for every February/June/
October Stage 3 target and all three horizons; disable month_ind explicitly.
Do not build alternative monthly maps or select granularity on performance.
This applies equally to temporary calibration maps, temporary selection maps
and final maps, for the reference and updated recipes. Preserve D18's distinct
year windows, map identities and information cutoffs; general does not mean
one map shared across feature recipes or time roles.

This retains 2014-2015 Jan/Apr/Jul candidate evidence in the early general
consensus instead of concentrating m2/m6 on 2016 alone. It gives up seasonal
partition boundaries used by the historical run. D16's horizon-by-target-month
probability calibration remains unchanged; general RF maps do not pool those
calibrators. D19/D20's no-split, eligibility and execution-failure distinctions
still apply. Successful or sufficiently connected early maps are not established
by this design choice.

The plan requires 26 development map builds (13 recipes/reference times
two windows) and two final builds (winner/reference), 28 in total before
valid-no-split/execution outcomes; Stage 1's 699 unique candidate-job bound is
unchanged. These counts are planning arithmetic, not executed map counts or
runtime estimates. Map granularity is approved for planning only, not execution.

## Other Stage 2 code facts — settings not yet frozen

Paths below are within GeoRFBaseline/. These facts inform later decisions,
not automatic adoption of every current default:

- `scripts/step4_similarity_matrix.py:16,188-203,265-269` multiplies accumulated
  plan co-membership weights by a haversine Gaussian with sigma=5 degrees,
  then divides by the global maximum. It does not normalize by each pair's
  joint plan-coverage count. `:79-105,236-261` discovers a node universe from
  all retained plan files, including zero-weight plans, defaulting to areas
  assigned somewhere other than s-1. Thus some nodes can have zero graph rows.
- `scripts/step5_sparsification.py:16,53-69` uses top k=40 per row, including
  self when selected, retaining all indices for n<=40. Symmetric maximum takes
  the union of directional edges; final degree need not be <=40. Equal-value
  tie ordering is not explicitly specified by its argpartition call.
- `compute_eigengap` (:72-92) uses the whole graph normalized Laplacian,
  replacing zero degrees with 1e-10 and requesting
  k_eigen=max(2,min(20,n-2)) eigenvalues with eigsh(which='SM', maxiter=5000).
  The largest consecutive gap recommends argmax(gaps)+1 clusters, i.e.
  1..19 for sufficiently large graphs. No component-count lower bound is set.
  eigsh has no explicit v0/seed, unlike the later spectral clustering.
- `scripts/step6_complete_clustering_pipeline.py:29,68-99` uses explicit
  --n-clusters if given, otherwise report recommended_clusters; validation
  checks positive integers but not compatibility with the fitted component.
  It fits only the largest component (:173-190), with whole-graph recommended
  nc, SpectralClustering affinity=precomputed, assign_labels=kmeans,
  random_state=42, n_jobs=-1. Equal-size components use the first argmax result.
- Other graph nodes are assigned using Euclidean lat/lon 1NN to that component
  (:192-202), with is_outlier in the map. This does not assign areas absent
  from the graph, impose a distance limit or use haversine neighbour distance.
- There is no general retry/reduce-nc rule for a positive-weight graph that is
  too small or incompatible. Step 5's n=1/2 eigenvalue formula requests k>=n;
  wrapper exceptions fail the process. D19's all-nonpositive single-partition
  rule is an approved new orchestration contract, not a released code branch.

The scout's optional in-memory SciPy probe used Linux SciPy 1.17.0, not the
requirements-pinned SciPy 1.15.2/sklearn 1.6.1. Its small-graph error observations
are not frozen-environment validation. Cluster selection, graph support,
deterministic solver initialization, graph universe and completion/routing
remain to be settled; do not silently import IPCCH choices.

## Approved consensus parameter/cluster-selection contract — D56

The user adopted retaining the released consensus formula and automatic cluster-count
rule for every recipe and map role, with no added parameter search:

1. Clip each candidate's class-1 F1 and pooled reference F1 to
   [1e-6,1-1e-6], then use max(logit(F1)-logit(F1_base),0) as its weight
   (`GeoRFBaseline/scripts/step4_similarity_matrix.py:50-64`). D19 separately
   governs the case in which every eligible completed candidate has zero weight.
2. Sum weighted co-membership, multiply by the existing haversine Gaussian
   with sigma=5 degrees, and divide by the matrix-wide maximum when positive
   (:137-161,188-203). Do not add pair-coverage normalization or tune bandwidth.
3. Retain row-wise top-k sparsification with k=40 and symmetric union, including
   the existing self-entry convention and all-node selection for n<=40. k is
   a graph-neighbour parameter, not the required number of partitions.
4. Select nc by the existing normalized-Laplacian largest consecutive eigengap
   formula: request k_eigen=max(2,min(20,n-2)) eigenvalues, sort ascending and
   take argmax(diff)+1, retaining the first equal maximum. For sufficiently
   large graphs the candidate counts are 1..19. Do not force nc=40, impose
   nc>=2, use an old map's nc, or compare nc choices using Stage 3 performance.
   The graph population on which selection/fitting operate is a separate open
   support decision; this approval does not adopt the inspected mismatch of
   whole-graph selection versus largest-component-only fitting.
5. Preserve precomputed-affinity spectral clustering with kmeans assignment
   and random_state=42 after graph/support requirements are satisfied. Persist
   eigenvalues, eigengaps, selected nc, graph/input identities and runtime
   settings. Pin eigensolver initialization in the implementation design for
   reproducibility rather than searching multiple seeds for a preferred map.

A supported, successful automatic nc=1 is a valid unsplit result. Record that
it arose from eigengap selection, distinct from D19's all-nonpositive weights;
once the complete routing is a single partition on matching rows/features,
reuse corresponding pooled RF predictions instead of pretending a spatial
split was learned. Positive-weight graph/support/solver failures remain
incomplete builds, not nc=1 evidence; do not cap/reduce nc on failure to force
a successful run. Their exact support/repair rules remain open.

This retains a data-adaptive partition count and the released weighting/graph
parameters, with the cost that some recipes/windows may learn few or no splits.
No alternate cluster count is selected after seeing forecasts. This parameter
contract is approved for planning only, not experimental execution.

## Approved common graph support for selection/fitting — D57

The user adopted preserving the released largest-component clustering approach while
making eigengap selection use that same component. For a normally completed
positive-weight build, construct the D56 sparse symmetric affinity graph over
the separately declared node universe. Find connected components using positive
off-diagonal affinities; self-entries do not join distinct nodes. These are
similarity-graph components, not a claim of polygon geographic contiguity.

Choose the largest component by node count, breaking an equal-size tie by its
smallest canonical FEWSNET_admin_code (nodes ordered by canonical code). Slice
its affinity matrix once, then compute the normalized Laplacian/eigengap and
fit spectral clustering on exactly those same ordered nodes. The eigengap
formula's n is that component's size. Keep D56's graph/weight parameters; do
not add edges to connect components or select a component by forecast scores.
Record all component sizes, chosen node IDs, affinity identity, selection
eigenvalues/nc and fitted assignments so equal support can be checked directly.

Require at least three component nodes for the retained eigensolver formula
to have k_eigen<n. A smaller largest component, invalid graph, solver failure
or incompatible output stops the affected map build as incomplete evidence.
Do not reduce k/nc, invent spectral evidence, or classify such failures as
D19's statistical no-split case. A supported eigengap nc=1 remains valid under
D56. D19's all-nonpositive-weight branch precedes these graph checks and is
not changed by this minimum numerical-support condition.

Other components, isolated nodes and areas absent from the graph do not
contribute to the chosen component's eigengap or spectral fit. Preserve their
identities and reasons for the separate completion/routing contract; do not
drop their evaluation rows or automatically inherit unrestricted Euclidean
1NN assignment from the release. The node universe and geographic completion
policy still need their own decisions.

This deliberately changes the release's whole-graph eigengap to match its
largest-component fit. The trade-off is that separate components do not
determine cluster count or learn their own spectral partitions, while the
selected nc corresponds to the matrix actually fitted. This support policy
is approved for planning only, not experimental execution.

## Approved graph-node universe contract — D58

The user adopted retaining the released in-scope union for each recipe/map-role window:
include a canonical area exactly when at least one of that window's eligible,
normally completed candidate plans assigns it a valid non-s-1 partition.
Use the same candidate-status ledger already required by D19/D20. Take the
sorted union across the window's approved months and fs1/fs2/fs3; do not borrow
assignments from another recipe, a later map window or future evaluation labels.
This corresponds to discover_admin_codes(full_universe=False) in
GeoRFBaseline/scripts/step4_similarity_matrix.py:79-105, applied to the explicitly
approved candidate set rather than every file discovered in an output folder.

Retain nodes contributed only by zero-weight candidates in this support ledger,
but their candidate contributes no similarity under D56. A node with no positive
off-diagonal edges is isolated and supplies no artificial connection. Missing
plan assignments and s-1 are not a shared residual partition and cannot create
co-membership edges. D57's largest-component rule determines which of the
resulting graph nodes participate in eigengap selection and spectral fitting.

Areas never assigned in any eligible candidate, and graph nodes outside the
selected component, keep their separate provenance for completion/routing.
Do not delete their master/evaluation rows, add zero rows for every future
target merely to make the matrix look complete, or infer learned partitions
for them from absence alone. Their usable geometry may come from the approved
snapshot convention, but geometry availability is not candidate evidence.

D19's all-nonpositive-weight rule still permits a valid unsplit result after
all required eligible candidates complete; do not run an all-zero spectral
pipeline or invent a graph component for that branch. Its one-partition/pooled
route must still document supported versus otherwise unassigned areas under
the eventual completion/routing contract. No eligible candidate or invalid
required artifacts remain failed/incomplete builds, not enlarged node unions.

This choice preserves the released node-eligibility convention and prevents
evaluation-period labels from defining the learned graph. Its trade-off is
that early windows may cover fewer areas directly; the later completion rule
must expose rather than conceal that coverage. The node-universe contract
is approved for planning only, not experimental execution.

## Geographic completion and Stage 3 routing — inspected behavior

Two bounded read-only scouts inspected the existing IPCCH completion and the
released-directory FEWSNET completion/refinement/forecast paths. The main
session checked IPCCH's distance/donor/completion functions and the contrasting
hard/probability prediction loops. These are code facts, not new approvals or
runtime verification; no model was fitted and no source code was changed.

- IPCCHGeoRFExperiment/run_pipeline.py:779-791,841-865 implements haversine
  nearest-donor lookup, with Earth radius 6371 km (:70), sorted donors and
  first-minimum tie resolution. complete_assignments (:868-974) fixes the donor
  set before lookup, preserves learned assignments, allows distance <=100 km
  (:72,941-957), and does not restrict countries or use polygon adjacency.
  Its donor table (:794-838) additionally requires a genuine learned assignment,
  valid reference coordinates and at least one original fit and validation row.
  Empty donors or missing/nonfinite recipient coordinates raise (:886-909).
  Out-of-range recipients instead retain partition_code=-1 and pooled_root.
- IPCCH's Stage 3 maps frozen partitions onto the panel (:1280-1336), then
  fits with all eligible rows in the current fold (:1580-1584,1634-1655).
  Completed areas' own eligible labels may therefore enter their assigned
  partition's fit. They do not merely borrow a saved Stage 1 donor model.
  local_model_routes (:1524-1563) distinguishes map provenance from per-fold
  local-model availability, including the 50-row/two-class support fallback.
- GeoRFBaseline/scripts/step6_complete_clustering_pipeline.py:173-202 uses
  largest-component nodes as fixed donors for unrestricted Euclidean lat/lon
  1NN completion of other graph nodes. Its output (:199-228) never adds areas
  absent from the graph. This is not IPCCH's bounded great-circle lookup.
- GeoRFBaseline/scripts/refine_partitions_contiguity.py:154-187,211-221 can
  subsequently alter both core and completed assignments using polygon-neighbor
  votes. src/partition/partition_opt.py:603-669 within that package performs
  synchronous swaps when own-label support is strictly below 4/9; it neither
  fills -1 nodes nor guarantees single-component geographic partitions.
  The root batch defaults to three iterations, but invokes root scripts rather
  than this release directory (run_partition_k40_comparison_unified.bat:
  93-94,284-285,394-412). Refinement is not implicit in the Step 6 Python call.
- GeoRFBaseline/scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py:185-209
  rejects more than 2% unmatched panel rows, otherwise assigns -1. Its map
  reader (:158-175) does not reject duplicate area keys. train_partitioned_model
  (:237-262) fits nonnegative groups with >=50 rows and two classes; smaller
  or single-class groups use pooled. predict_partitioned (:307-328) handles
  dictionary entries and -1 only, leaving unseen nonnegative groups at the
  initial zero. predict_partitioned_probability (:338-356) sends every unhandled
  row to pooled. The hard/probability routing inconsistency must be resolved
  when specifying Stage 3, without editing the protected baseline package.
- The release's shared split helper additionally restricts training to groups
  appearing in the prediction subset (GeoRFBaseline/src/customize/customize.py:
  439-445). With one partition and unresolved -1 areas, local training may
  still differ from the full pooled fit. A single learned cluster alone does
  not establish D56's matching-input condition for pooled prediction reuse.

## Approved geographic completion — D59

The user adopted IPCCH's bounded nearest-donor geometry rule with this
experiment's own frozen consensus support:

1. For a valid positive-weight graph build, donors are exactly D57's fitted
   component areas, with their learned labels and validated master WGS84
   coordinates. Preserve those labels during completion. Do not add IPCCH's
   distinct per-area fit/validation donor gate or use donors from another
   recipe/window. A missing/invalid core needed for the graph is a D57 failure,
   not permission to route a failed map to pooled.
2. Recipients are all remaining canonical area IDs in the frozen master
   metadata universe, including graph outliers and never-in-graph areas.
   Retain these distinct support reasons. Geometry uses the approved supplied
   snapshot convention; future outcome presence/value cannot determine the
   donor set, recipient set or nearest assignment.
3. Find 1NN using haversine distance with radius 6371 km. Allow cross-country
   matches and distance <=100 km, resolving exact ties by smallest canonical
   donor code. Assign that donor's partition. Newly completed areas never become
   donors. Do not add adjacency-first routing, iterative propagation or a
   geographic-threshold search.
4. Missing/invalid recipient coordinates or nearest distance >100 km leave
   the area unassigned (partition -1), retaining rows for pooled prediction.
   This missing-coordinate behavior is an explicit approved difference from
   IPCCH's failure rule. Record assignment source, support status, donor/distance
   when defined, coordinate identity and unresolved reason; do not impute
   coordinates for matching or force a remote assignment.
5. Freeze this completed map once per recipe/role before its prediction folds.
   Completion supplies geographic routing, not extra learned graph evidence.
   D19's pre-graph all-nonpositive case retains the approved pooled result;
   it does not fabricate spectral donors. D60 omits additional post-consensus
   refinement; D61 fixes Stage 3 fitting/fallback and D62 valid nc=1 routing.

The benefit is one transparent completion rule covering the whole retained
cohort; the cost is that remote/unlocatable areas use pooled rather than a
spatial model. This is a separate approval from D9's WB market matching rule.
No implementation or experimental execution is authorized.

## Approved omission of post-consensus smoothing — D60

The user adopted using the D57 spectral-core assignments plus D59 completion directly
as the frozen map, without additional polygon-neighbor majority-vote refinement
before or after completion. Explicitly bypass inherited batch refinement and
do not produce competing refined/unrefined maps or choose between them by
forecast scores. Stage 1's released internal spatial partition operations and
D56's spatial weighting remain unchanged.

The main session inspected GeoRFBaseline/src/partition/partition_opt.py:580-669
and scripts/refine_partitions_contiguity.py:176-221. The helper includes the
area itself and valid neighbors in its vote; if its own label has strictly
less than 4/9 of those votes, it switches to the most popular other label.
Updates within each iteration are synchronous. There is no protection of
spectral-core or donor-completed labels and no guarantee that the resulting
partition is one geographic component. The root comparison batch defaults to
CONTIGUITY=1 and REFINE_ITERS=3 (:93-94), whereas the refinement Python CLI
defaults to two iterations. These settings describe inspected code, not a new
experiment run or implicit scientific approval.

Skipping this operation preserves the learned core and each recipient's
exact donor assignment, avoiding additional unscored label changes. It can
also leave geographic islands and intentionally departs from the historical
refined-map route. Do not claim geographic contiguity from similarity-graph
connectivity or from the absence of smoothing. D61 fixes Stage 3 fitting and
fallback; D62 fixes valid nc=1 behavior. Planning approval does
not authorize implementation or experiment execution.

## Approved Stage 3 fitting and fallback — D61

The main session checked IPCCHGeoRFExperiment/run_pipeline.py:1524-1655 and
GeoRFBaseline/scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py:220-262,
plus GeoRFBaseline/src/customize/customize.py:408-449. IPCCH passes the full
eligible fold history and frozen partition codes to its pooled/local helpers,
using one training-fitted imputer. Completed areas are not excluded from local
fitting. Its route ledger explicitly includes unseen-partition fallback. The
released split helper instead filters training groups to those present in the
target subset; that can also change preprocessing statistics. These facts do
not authorize copying IPCCH's inclusive-origin window or classification rule.

The user adopted the following training/routing contract for all development
and final folds, separately for each recipe and horizon:

1. Select all real eligible labeled rows under D20's [O-35 months,O) mask,
   preserving every row's own origin-aligned predictors. Do not restrict this
   global pool by target-month area/group coverage or persistence availability.
   Fit D23 max_plus once on the pool and share it across pooled/local RF fits.
2. Train pooled on the full pool. Train each needed local RF on its frozen
   partition's subset, including both core and geographically completed areas'
   eligible history. Unassigned (-1) rows contribute only to pooled. Recipient
   labels cannot change the frozen map, donor set or thresholds; their temporal
   eligibility for rolling RF fits is distinct from map-learning eligibility.
3. Preserve the released local-support gate: at least 50 real training rows
   and both binary classes. A recipient without its own history can still use
   its assigned partition's model. No artificial Stage 3 rows or SMOTE are added.
4. Unassigned areas and partitions with too few rows, one class or no training
   rows use pooled for both native classifier decisions and class-1 probabilities.
   Record the actual route/reason alongside the unchanged map provenance.
   Correct the known unhandled-hard-prediction route in isolated experiment
   code; do not modify the protected baseline. Global empty training or an
   actual fitting error is incomplete execution, not a statistical fallback.

This reuses existing RF fits and support rules while making completed areas'
training role and the common preprocessing pool explicit. It does not borrow
saved donor-area models, change D20 to IPCCH's inclusive-origin history, adopt
IPCCH's probability-threshold hard decisions or add its XGBoost comparator.
The cost is that completed areas can influence their partition's later fitted
RF once their labels become eligible; they are not permanently donor-only
prediction sites. D62 separately fixes supported nc=1 plus unresolved areas,
where an assigned-only local training set would differ from pooled.
This is planning approval only; no implementation or model fitting has occurred.

## Approved valid single-cluster routing — D62

D61 fits a local RF using only rows assigned to that partition. If a valid
automatic nc=1 map still has unresolved areas, its assigned-only local pool can
differ from the full pooled training set. Existing code evidence is recorded
above (GeoRFBaseline/scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py:220-262).
Thus nc=1 alone cannot prove identical predictions from two separately fitted
models. D56 previously conditioned pooled reuse on matching complete inputs.

The user adopted this no-split route: whenever a valid completed positive-weight
map selects automatic nc=1, use that recipe/horizon/fold's full-pool RF directly
for all rows in the partitioned prediction stream, including assigned and
unassigned areas. Reuse native hard predictions and class-1 probabilities; omit
the assigned-only local fit. This is an explicit routing choice, not a claim
that the two potential training subsets were already equal. D19's approved
all-nonpositive-weight no-split branch also retains its full-pool route, with
a distinct reason. Failed or unsupported graphs remain incomplete builds.

Preserve the original graph, spectral-core/completed/unassigned map statuses,
donor/distance evidence and coverage. Do not relabel remote areas as learned
members merely because their actual model route is pooled. Record the valid
unsplit reason alongside per-row actual routing. Apply the same rule in
development and final evaluation for every recipe; correction continues to
use this partitioned-stream probability source with the already approved
calibrators/thresholds and method labels. Identical RF streams remain reported
with their provenance, without adding a separately tuned pooled-correction arm.

For valid maps with nc>1 retain D61, even when a fold has only one supported
local RF or all local routes fall back. Do not reclassify a multi-cluster map
based on a fold's target-group coverage or final scores. The trade-off is
forgoing an assigned-subset fit when no split is learned; this makes the
single-partition result a consistent full-pool baseline. This is planning
approval only, not authorization to implement or run the experiment.

## Updated source-label evidence

A subsequent full read-only scan reconciles all master binary/phase values
and overlapping ledger phases exactly, with no duplicate canonical keys.
The single malformed ledger record is at physical line 303003; embedded
newlines account for the difference from logical CSV record numbering.
See research/target-label-contract.md for source hashes, exact support counts,
the distinction between missing rows and missing phases, and the pending
validation/explicit-artifact-exclusion decision. No models were run.
