# No-Leak Partition Learning Output Archive

Archived on 2026-06-16 during workspace cleanup.

This folder preserves root-level Stage 1 partition-learning summary and
prediction-detail CSV files that were generated for the no-temporal-leak
workflow.

## Contents

- `GeoRFResults/`
  - `results_df_gp_fs{1,2,3}_{2018,2019,2020}_{2018,2019,2020}.csv`
  - `y_pred_test_gp_fs{1,2,3}_{2018,2019,2020}_{2018,2019,2020}.csv`
- `GeoDTResults/`
  - `results_df_dt_gp_fs{1,2,3}_2020_2020.csv`
  - `y_pred_test_dt_gp_fs{1,2,3}_2020_2020.csv`

## Notes

- The root-level `results_df...` files matched same-name copies under
  `GeoRFExperiment/GeoRFResults/` or `GeoDTExperiment/GeoDTResults/` before
  archiving.
- The root-level `y_pred_test...` files did not have same-name copies in those
  experiment result directories at cleanup time, so they were moved here to
  preserve the prediction-detail outputs while cleaning the repository root.
- CSV files in this archive are ignored by the repository-wide `*.csv` rule and
  are intended as local generated artifacts, not source files.
