# GeoRF Humanitarian Population Metrics Design

## Goal

Respond to the reviewer request for population-weighted humanitarian metrics by adding a GeoRF-only appendix diagnostic for the evaluated 2021-2024 polygon-month predictions.

## Scope

This diagnostic uses only:

- GeoRF Stage 3 pooled and partitioned prediction outputs from `result_partition_k40_compare_GF_fs1/`, `result_partition_k40_compare_GF_fs2/`, and `result_partition_k40_compare_GF_fs3/`.
- Raw FEWSNET population from `C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWSNET.csv`.
- The raw FEWSNET `pop` column joined by `admin_code` and target month.

The diagnostic does not add FEWSNET baseline metrics, GeoDT metrics, GeoXGB metrics, new threshold tuning, or unique-person exposure estimates.

## Metric Definitions

All population totals are population-month totals over evaluated polygon-month observations, not unique people.

- `true_alert_population`: sum of `pop` where `y_true = 1` and `y_pred = 1`.
- `missed_crisis_population`: sum of `pop` where `y_true = 1` and `y_pred = 0`.
- `false_alert_population`: sum of `pop` where `y_true = 0` and `y_pred = 1`.
- `true_noncrisis_population`: sum of `pop` where `y_true = 0` and `y_pred = 0`.
- `population_weighted_recall`: `true_alert_population / (true_alert_population + missed_crisis_population)`.
- `population_weighted_precision`: `true_alert_population / (true_alert_population + false_alert_population)`.

Deltas are reported as `partitioned - pooled`. Lower missed-crisis and false-alert population totals are better. Higher population-weighted recall and precision are better.

## Outputs

Create `final_artifacts_in_paper_updated/09_humanitarian_metrics/` containing:

- `georf_humanitarian_population_compact_table.csv`: one row per forecasting horizon with pooled, partitioned, and delta values.
- `georf_humanitarian_population_compact_table.md`: Markdown rendering of the compact appendix table.
- `georf_humanitarian_population_by_month.csv`: month-level diagnostic table by horizon and model.
- `georf_humanitarian_population_summary.csv`: long-format horizon-level diagnostic table by horizon and model.
- `georf_humanitarian_population_bars.png`: compact bar figure comparing missed-crisis and false-alert population by horizon and model.
- `georf_humanitarian_population_note.md`: Chinese reviewer-facing note and English appendix text.

Update `final_artifacts_in_paper_updated/README.md` with the new folder and file index.

## Validation

- Verify raw FEWSNET population join coverage is complete for all three GeoRF prediction files.
- Unit-test population metric formulas, compact table formatting, and duplicate-key validation.
- Run script compile checks and `git diff --check`.
