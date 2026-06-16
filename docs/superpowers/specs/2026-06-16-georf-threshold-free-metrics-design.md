# GeoRF threshold-free and fixed-operating-point metrics design

## Purpose

Reviewer comment 2 asks the manuscript to report recall at fixed precision,
precision at fixed recall, and PR-AUC. The goal is to answer that request using
already generated GeoRF prediction probabilities, without retraining models or
introducing a new threshold-selection procedure.

## Scope

This diagnostic covers GeoRF only, comparing pooled and partitioned predictions
for the three existing forecasting horizons:

- `fs1`: 4-month lag
- `fs2`: 8-month lag
- `fs3`: 12-month lag

Inputs are the existing Stage 3 files:

- `result_partition_k40_compare_GF_fs1/predictions_monthly.csv`
- `result_partition_k40_compare_GF_fs2/predictions_monthly.csv`
- `result_partition_k40_compare_GF_fs3/predictions_monthly.csv`

Each input already contains `y_true`, `y_prob_pooled`, and
`y_prob_partitioned`, so no model rerun is needed.

## Metrics

For each horizon and model, compute:

- PR-AUC using average precision from the predicted crisis probability.
- Recall at fixed precision thresholds of 0.75 and 0.80.
- Precision at fixed recall thresholds of 0.50 and 0.60.

The operating-point metrics are computed from the precision-recall curve. For
recall at fixed precision, report the maximum recall among thresholds where
precision meets or exceeds the fixed precision target. For precision at fixed
recall, report the maximum precision among thresholds where recall meets or
exceeds the fixed recall target.

If no threshold satisfies a target, record the metric as missing and state that
the requested operating point is not attainable from the existing probability
ranking for that model and horizon.

## Outputs

Create a new artifact folder:

- `final_artifacts_in_paper_updated/11_threshold_free_metrics/`

Write:

- `georf_threshold_free_metrics.csv`: full horizon-model metric table.
- `georf_threshold_free_metrics_compact_table.md`: compact appendix-ready
  table comparing pooled and partitioned GeoRF.
- `georf_threshold_free_metrics_note.md`: Chinese reviewer-facing note plus
  English appendix wording.

The README in `final_artifacts_in_paper_updated/` should register the new
folder and files.

## Interpretation

These metrics are threshold-free or post hoc operating-point diagnostics. They
describe ranking quality and feasible precision-recall tradeoffs under existing
probability outputs. They do not imply that the main experiment tuned thresholds
or used a different decision rule. The main binary results remain the current
hard-prediction precision, recall, and F1 tables.

If the threshold-free metrics are weaker or mixed for partitioned GeoRF, the
note should present that directly and frame it as complementary evidence: the
partitioned model can improve hard-label crisis recall while the probability
ranking and high-precision operating points may vary by horizon.

## Tests and validation

Add focused tests for metric helpers using small synthetic arrays:

- PR-AUC computation matches scikit-learn average precision.
- recall-at-fixed-precision chooses the maximum feasible recall.
- precision-at-fixed-recall chooses the maximum feasible precision.
- unattainable operating points are returned as missing values.

Run script compilation and the focused unit test before committing generated
artifacts.
