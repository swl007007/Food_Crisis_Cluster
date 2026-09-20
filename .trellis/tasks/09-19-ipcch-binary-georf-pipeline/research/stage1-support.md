# Stage1 support and period trade-off — planning audit

Current-design note: the user subsequently selected one time-pooled Stage1 fit,
broader INTERNAL validation, no Stage2 consensus, and reuse of existing geometry/
coordinate1-NN completion. Annual-candidate proposals below are superseded;
the audited counts remain evidence. See PRD v0.11 and research/brainstorm.md.

User approved monthly rolling refits but questioned whether a36-month training
window leaves too few early partitions, suggesting more Stage1 and less Stage3.
No period replacement or training-area rule has been approved.

## Evidence and method

`stage1_support_audit.json` and `stage1_support_folds.csv` are read-only support
audits, not transformed modeling data or experiment results. Exact Decimal100
target reconstruction matches42,695 valid labels and15,206 positives. The source
hash was previously verified, not recomputed in this audit. The336 folds cover
2018-01 through2024-12 for H=1/3/6/12; O=T-H; train label months [O-35,O].
All summaries below use months with at least one valid target label. No feature,
geometry, actual validation-class support, successful split or positive consensus
weight has been evaluated.

The raw source has460/518/529 valid labels in2014/2015/2016, spanning5/6/6
countries and261/264/270 areas. In2017 coverage expands to1,726 labels,
411 positives,35 countries and1,062 areas. Thus2017 is a coverage expansion,
not the first year with any valid label.

Subsequent user clarification: IPC areas (`area_id<100000`) start receiving
labels in2017; all pre2017 labels are CH. The initial tables below pool IPC and
CH, so their early support cannot establish IPC-specific coverage. The subsequent
source-stratified audit below independently verifies this timing clarification.

## Code facts

Paths below are relative to `GeoRFBaseline/`.

- `src/customize/customize.py:407-421` selects observations within a calendar
  range; it does not require36nonmissing labels or36months of coverage per area.
  Its old interval excludes O; the approved IPCCH interval includes O.
- `src/customize/customize.py:439-448` further filters historical training rows
  to groups present in the target-month test set. In polygon mode groups are
  admin IDs (`src/preprocess/preprocess.py:429-431`).
- `src/utils/split.py:63-83` gives singleton areas no validation observation;
  for n>=2, n_val=min(n-1,max(1,ceil(.2*n))), randomly selected within each area,
  not stratified by class. Reported validation capacity uses this formula only.
- `src/partition/transformation.py:299-314` creates group stats from validation
  rows and requires a parent validation positive under current zero minimums.
  `:693-708` requires nonempty child train/validation subsets; fewer than two
  validation groups cannot yield a viable split (`src/partition/partition_opt.py:842-847`).
- `src/partition/partition_opt.py:867-890` requires strict aggregate F1 gain>.01.
  This performance gate is not a statistical-support guarantee.
- `src/feature/feature.py:88-96` only shifts by the horizon, with no second36-month
  history requirement. The inherited row-shift must still become calendar-safe
  at the IPCCH boundary; historical training features use their own origins.

## Training-area filter effect

U uses all historical labeled areas; R additionally requires target-month labels
for area membership. Values are monthly medians; a fractional median is not a
fractional observation. A2 counts areas with at least two training observations.

| Target year | H | U train rows | R train rows | U A2 | R A2 |
|---|---:|---:|---:|---:|---:|
| 2018 | 12 | 2,328 | 13.5 | 381 | 0 |
| 2022 | 12 | 9,697 | 272 | 2,320 | 71 |
| 2023 | 12 | 10,616.5 | 199 | 2,317 | 44 |

The recommendation is to remove this target-month-area training restriction in
the new Stage1 data boundary, while retaining within-area validation and the
partition core. Target-month scoring still uses observed labels only. This is
pending Q5a; handling target areas unseen in training remains a separate Q5
contract. Removing the restriction alone does not prove successful partitions.
All-area predictor retention does not mean fitting on unlabelled rows: retain
the monthly scaffold to construct calendar/as-of features, then fit supervised
models only where target labels are valid. Historical persistence inputs never
replace missing truth. Evaluation eligibility is a valid area-month pair, not
all months in an area that happens to have a label somewhere in the test period.

## Period trade-off

Stage1 target-period counts start2018. Evaluation applies the previously approved
rule that origin is at least January after the partition-freeze year, with main
target months ending2025-12. Counts precede feature/history/support exclusions.

| Stage1 through | Candidate target labels | Candidate areas | h12 main targets | h12 labels |
|---|---:|---:|---|---:|
| 2022 | 16,358 | 4,498 | 2024–2025 | 14,087 |
| 2023 | 21,283 | 5,210 | 2025 | 9,792 |
| 2024 | 25,578 | 5,779 | None before2026 | 0 |

Through2023 adds4,925 target labels (+30.1%) and712 distinct candidate areas,
but h12 loses the2024 evaluation year. Through2024 leaves h12 only the incomplete
2026 source, which has4,092 labels/1,923 positives across3,710areas. Extending
the learning period cannot compensate for an unresolved training-area filter.
First decide Q5a, then revisit Q2r; do not choose dates based on predictive scores.

## Subsequent validation-support challenge: Q5b now precedes Q5a/Q2r

The user challenged monthly validation coverage and likely unstable partitions.
`validation_support_by_source.json` is an exact-Decimal source scan, matching
all42,695labels/15,206positives. No model fits or measured instability are claimed.

| Source IDs | First valid month | All-period labels | Areas |
|---|---|---:|---:|
| IPC <100000 | 2017-01 | 24,336 | 4,916 |
| CH >100000 | 2014-01 | 18,349 | 1,307 |
| =100000, counted separately | 2020-01 | 10 | 1 |

Two distinct sources of sparse support:

1. Inner within-area validation uses historical-window observations, not only
   the score month. In2022 IPC/h12, monthly medians across IPC-labeled target
   months are4,547.5historical labels,1,275areas with>=1possible validation row,
   but only36.5areas with>=2possible validation rows; none has>=3. Broad
   geographic presence does not establish reliable per-area F1/error estimates.
2. Outer target-month scoring drives consensus weights. In2022 IPC, monthly
   scored areas have median96,min5,max635;4of12months have<20areas. The whole
   year covers1,595distinct areas/2,081label rows. CH has only2label months with
   much larger coverage that year; combined monthly statistics obscure this.

All2017–2023 IPC labels provide10,904rows/4,141areas, with20percent within-area
holdout capacity distributed as:1,564areas=0,2,253=1,290=2,34=3rows. Even longer
pooled history cannot by itself guarantee stable q or splits.

Proposed response (not approval): annual Stage1 candidate cycles, expanding
pre-origin history, existing within-area validation and F1/q/split core, and
whole-year pooled out-of-sample predictions for each candidate's consensus-weight
F1. Candidate fitting must precede or include only the earliest month-end origin
of that score year; annual score labels remain outside that fit and prediction
features obey their own origins. Stage3 keeps monthly36-month refits and frozen
partitions. This trades fewer, better-supported score units for reduced candidate
count; it does not create independent evidence or prove stability. Minimum
support, split membership, unseen-area routing, exact years/history start and
area-filter policy still need agreement. No model change has been implemented.
