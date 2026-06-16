# Fixed-Partition Feature-Exclude Ablation Design

## Goal

Regenerate `ablation_feature_exclude.xlsx` under
`final_artifacts_in_paper_updated/` using the current no-leak GeoRF workflow,
while isolating the measured effect of each feature group.

## Scope

This ablation is Stage 3 only. Stage 1 partition learning and Stage 2
consensus clustering are not rerun for each excluded feature group. All
feature-exclude datasets are evaluated against the same current main GeoRF
partition maps so the comparison reflects the performance drop from removing a
feature group, not changes in learned partition structure.

The feature-exclude groups are:

- `weather_exclude`
- `agri_exclude`
- `conflict_exclude`
- `econ_exclude`
- `food_prices_exclude`
- `geographic_exclude`

## Fixed Experimental Conditions

Each ablation run uses:

- base model: GeoRF / GF
- stage: Stage 3 partitioned-vs-pooled comparison
- scopes: fs1, fs2, fs3
- lags: 4, 8, 12 months
- evaluation window: 2021-01 through 2024-12
- training window: 36 months
- partition maps: current main GeoRF maps from `GeoRFExperiment/knn_sparsification_results`
- month-specific maps: general, m2, m6, and m10
- month indicator: enabled
- contiguity refinement: `cont3`
- source datasets: existing feature-exclude CSVs under
  `C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data`

## Output Layout

New run outputs should be isolated under:

`main_ablation_exclude_updated_stage3_fixed_partitions/`

with one subfolder per excluded feature group. The final paper workbook should
be written to:

`final_artifacts_in_paper_updated/ablation_feature_exclude.xlsx`

The old `final_artifacts_in_paper/ablation_feature_exclude.xlsx` remains a
layout reference only. It should not be treated as the updated source of truth.

## Workbook Semantics

The workbook keeps the old paper-facing shape where practical:

- one block per feature-exclude group
- one row per lag/scope: 4, 8, and 12 months
- partitioned/split model precision, recall, and F1
- pooled model precision, recall, and F1
- improvement and comparison columns derived from the updated runs

The main comparison baseline should come from the current updated main
`main_month_ind_cont3.xlsx` artifact or its upstream Stage 3 outputs, not from
old April ablation folders.

## Validation

Before promoting the workbook, verify:

- all six feature-exclude groups have fs1, fs2, and fs3 outputs
- every generated `metrics_monthly.csv` has partitioned and pooled model rows
- the evaluation months match the current Stage 3 window
- the workbook preserves the expected column layout
- output values are derived from the new run root and not copied from old
  `main_ablation_exclude/` folders

## Non-Goals

This work does not estimate partition-learning sensitivity to feature groups.
It does not rerun Stage 1 or Stage 2 per excluded dataset, and it does not
modify the current main partition maps.
