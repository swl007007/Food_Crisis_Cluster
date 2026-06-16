# GeoRF validation-selected max-F1 threshold design

## Purpose

Improve partitioned GeoRF crisis-class F1, especially recall, without changing
the learned partitions or using test-set information to choose thresholds.
Existing artifacts show that partitioned GeoRF probabilities contain useful
ranking information below the default 0.5 hard-classification threshold. The
new experiment should therefore evaluate validation-selected probability
thresholding as a Stage 3 extension.

## Scope

This design applies only to GeoRF Stage 3 partitioned RF evaluation.

In scope:

- Add a `partitioned_thresholded` evaluation variant.
- Select thresholds from validation predictions inside each rolling training
  window.
- Optimize class-1 F1 on validation data.
- Apply the selected threshold only to the held-out target-month test data.
- Preserve existing pooled and partitioned hard-prediction outputs for
  comparison.
- Write threshold provenance and compact reviewer-facing artifacts.

Out of scope:

- GeoDT, GeoXGB, or non-GeoRF model families.
- Test-set-selected thresholds.
- Retuning partitions, k-nearest-neighbour sparsification, adjacency
  refinement, or Stage 1 recursive partitioning.
- Changing RF feature sets, lag schedules, or FEWSNET baseline handling.
- Replacing all paper artifacts before the thresholded results are inspected.

## Threshold Selection Rule

For each forecasting horizon and evaluated target month:

1. Build the same rolling training window used by the current Stage 3
   comparison.
2. Split the training window into a model-training subset and a validation
   subset using a deterministic temporal holdout from the end of the training
   window. The validation subset must remain before the target-month test data
   and must respect the active lag.
3. Train the pooled fallback model and local partitioned RF models on the
   model-training subset.
4. Generate class-1 probabilities for the validation subset using the same
   partitioned/pooling fallback rules used for test prediction.
5. Search candidate thresholds on validation probabilities and select the
   threshold that maximizes binary class-1 F1.
6. Break ties by choosing the highest threshold among tied maximum-F1
   candidates, preserving precision when recall gains are equivalent.
7. Refit the pooled fallback model and local partitioned RF models on the full
   Stage 3 training window.
8. Generate class-1 probabilities for the target-month test data and convert
   them to binary predictions using the selected validation threshold.

The baseline `partitioned` model should continue to use the current hard
prediction rule. The new thresholded predictions should be stored separately as
`partitioned_thresholded`.

## Candidate Thresholds

Use validation probability scores as the default candidate threshold set,
optionally rounded to two decimal places for stable provenance. Thresholds are
bounded to `[0.05, 0.95]` so all-positive or all-negative edge cases do not
dominate unless the validation data genuinely support them.

If a validation subset contains no positive class-1 observations, fall back to
the default threshold `0.5` and record `fallback_reason =
no_validation_positive_cases`.

If no candidate produces a finite F1, fall back to `0.5` and record
`fallback_reason = no_finite_validation_f1`.

## Outputs

Do not overwrite the existing Stage 3 result folders in the first pass. Write
new thresholded result folders:

- `result_partition_k40_compare_GF_thresholded_fs1/`
- `result_partition_k40_compare_GF_thresholded_fs2/`
- `result_partition_k40_compare_GF_thresholded_fs3/`

Each folder should preserve the current Stage 3 output pattern and add:

- `metrics_monthly.csv` with `pooled`, `partitioned`, and
  `partitioned_thresholded` rows.
- `predictions_monthly.csv` with `y_prob_partitioned`,
  `y_pred_partitioned`, `selected_threshold`, and
  `y_pred_partitioned_thresholded`.
- `threshold_provenance.csv` with one row per evaluated target month.
- `run_manifest.json` documenting the validation threshold rule.

Add a new paper-artifact folder:

- `final_artifacts_in_paper_updated/12_thresholded_georf_results/`

Initial paper-facing outputs:

- `georf_thresholded_compact_table.csv`: horizon-level pooled vs partitioned
  vs partitioned-thresholded precision, recall, and F1.
- `georf_thresholded_threshold_provenance.csv`: selected thresholds and
  validation/test metrics by horizon and target month.
- `georf_thresholded_monthly_metrics.csv`: long-format monthly metrics.
- `georf_thresholded_note.md`: Chinese reviewer-facing note plus English
  appendix text explaining validation-selected max-F1 thresholding.

## Artifact Regeneration Impact

After thresholded results are inspected and accepted as the main GeoRF result,
the following existing artifact groups should be regenerated because they
depend on binary predictions or hard-label metrics:

- `01_main_results/`: main workbook, monthly performance figure, season table,
  region table, partitioned/pooled/FEWSNET comparison table, selected prediction
  maps, and monthly performance manifest.
- `04_error_analysis/`: seasonal error-rate source tables and figures.
- `06_cluster_profiles/`: cluster profile table, similarity/error-mode figure,
  and note if TP/FP/FN/TN or error-mode composition remain included.
- `07_probability_uncertainty/`: bootstrap precision/recall/F1 intervals and
  any note text that references the binary decision rule. Brier/reliability
  probability summaries are unchanged in principle, but should be refreshed if
  the table is presented alongside thresholded hard metrics.
- `09_humanitarian_metrics/`: missed-crisis population, false-alert
  population, and population-weighted recall/precision.
- `10_false_negative_error_modes/`: false-negative hotspot summaries and
  descriptive error-mode tables.
- `11_threshold_free_metrics/`: PR-AUC and fixed operating-point values do not
  change because probabilities do not change, but the note should be updated to
  distinguish threshold-free diagnostics from the new validation-selected
  binary rule.

The following groups should not need regeneration unless their text references
the old default hard-prediction rule:

- `02_methods_and_temporal_scope/`: update method notes and temporal split
  wording, but static schematic images do not need regeneration.
- `03_class_prevalence/`: actual class prevalence is model-independent.
- `05_partition_diagnostics/`: partition stability and adjacency refinement are
  unaffected by thresholding.
- `08_geodt_diagnostics/`: GeoDT diagnostics are unaffected.

## Validation

Implementation should include focused tests for:

- max-F1 threshold selection from validation probabilities.
- tie-breaking toward the higher threshold.
- fallback to 0.5 when validation positives are absent.
- conversion of probabilities to binary labels with the selected threshold.

Before promoting thresholded results to main paper artifacts, compare:

- horizon-level precision, recall, F1;
- monthly precision/recall/F1;
- false-alert population increase;
- missed-crisis population decrease;
- threshold distribution by horizon and target month.

Promotion should only happen after confirming that max-F1 thresholding improves
F1 without producing an unacceptable false-alert or precision penalty.
