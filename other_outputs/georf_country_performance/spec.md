# GeoRF main-results country performance

## Approved scope

Aggregate existing main GeoRF Stage 3 predictions by country and forecasting
horizon. Compare GeoRF (partitioned), Pooled RF, and FEWS NET expert forecasts
for binary IPC Phase 3+ using Precision, Recall, and F1. No fitting or threshold
selection is needed. All new files stay in this directory; the existing final
artifacts and reproducibility package remain untouched by this analysis.

## Inputs and evaluation contract

- Main predictions: `archived/release_20260624_reproducibility_inputs/`
  `result_partition_k40_compare_GF_fs{1,2,3}/predictions_monthly.csv`.
  Use saved `y_true`, `y_pred_partitioned`, and `y_pred_pooled` directly.
  fs1/fs2/fs3 correspond to 4/8/12-month horizons.
- Country mapping and expert projections: the original `FEWSNET.csv` in the
  source-data `Outcome/FEWSNET_IPC` directory. Join on normalized admin code
  and evaluation month, require unique keys, complete country mapping, and
  agreement of raw `fews_ipc >= 3` with saved `y_true`.
- Preserve the original FEWS NET evaluator exactly as requested: sort each
  area's complete history by year/month; convert near/medium phases to binary
  using the original >=3 rule (original missing-phase conversion is retained);
  apply the original per-area record shifts of 4 and 8 respectively BEFORE
  selecting evaluation rows. Do not substitute a calendar join or same-row
  forecasts. Check these reconstructed expert series against the archived
  fs1/fs2 baseline summaries on the original full baseline support.
- Then evaluate experts on precisely the saved GeoRF area-month support,
  shared with Pooled RF. This common-support country analysis need not equal
  the full-support expert totals in the previous main figure.
- FEWS NET at 12 months is unavailable: literal `N/A` in the main table and no
  expert points in the corresponding figure row. No 8-month proxy.
- The main evaluation schedule is February/June/October in 2021-2024.
  Each scope contains 62,189 observations across 22 countries; keys and truth
  agree across scopes. No missing aligned expert labels were observed at fs1/fs2.
  Unexpected duplicate keys, missing aligned experts, or truth mismatches stop
  generation rather than silently changing the common sample.

## Aggregation and missingness

Calculate country-month confusion counts and positive-class Precision, Recall,
and F1 (zero denominator returns zero, matching the existing evaluator).
Country summaries are the arithmetic mean of monthly metrics, with equal
weight per observed month. Compute monthly F1 before averaging it.

Do not fabricate observations for an absent country-month or count it as a
zero score. Coverage inspection found 7-12 observed months per country:
Afghanistan 7; South Sudan 8; Burkina Faso and Somalia 10; Burundi, Sudan,
and Yemen 11; all other countries 12. Show observed months and total area-month
sample counts in the main table; retain monthly counts and metrics for audit.

## Deliverables

1. `country_performance.xlsx` (primary): 66 rows, one country x horizon per
   row, alphabetic country order. Group the nine score columns by model;
   include horizon, area-month sample count, and observed evaluation-month
   count. Display scores to three decimals and retain full precision.
   Supporting worksheets contain monthly confusion counts/metrics and concise
   calculation/source notes including source hashes. Use formulas for monthly
   ratios and country averages and recalculate them in installed Excel;
   verify cached values independently after saving. Literal `N/A` is not an
   Excel error. Format with readable fonts, frozen headers and filtering.
2. `country_performance_3x3.png` with a matching vector PDF of the same figure:
   rows are 4/8/12 months, columns are Precision/Recall/F1. All panels share
   alphabetical country order and a 0-1 metric scale. Use offset colored
   markers with distinct shapes for the models; do not connect nominal country
   categories. Provide readable country labels and one shared legend.

Figure purpose: compare how country-level performance varies across the three
models and forecast horizons without presupposing which model wins.
Archetype: quantitative grid. Backend: the existing Python/matplotlib workflow;
export a readable landscape grid (approximately 18 x 13 inches), PNG >=300 dpi
and PDF with editable text. Monthly means are descriptive; no error bars,
significance tests, or country ranks are requested. Use the same values as the
Excel main sheet. Explain IPC Phase 3+, observed-month averaging, and the
unavailable 12-month expert forecast in a concise caption/footnote.

## Implementation and acceptance

Use one local Python generator and only the small Excel recalculation helper
needed for this installed environment. Reuse existing metric/label conventions;
avoid importing old plotting modules with unrelated geospatial dependencies.
Keep the source spec and reproducible command here. Do not mutate shared
pipeline code or the unrelated active Ethiopia Trellis task.

Verify saved model labels reproduce the archived monthly main model metrics;
verify the historical expert calculation reproduces archived baseline metrics;
then check common-support confusion counts, country averages, coverage, and
table/figure agreement independently. Reopen the XLSX to check calculated
values and zero Excel errors, inspect the full 3x3 figure for overlap/clipping,
and verify final-artifact/package files are unchanged from this task's start.
