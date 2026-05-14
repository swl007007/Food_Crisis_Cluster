# Contract: Monthly Performance Plotting Script

## Entry Point

`python scripts/plot_monthly_performance_metrics.py [options]`

The script is a lightweight diagnostic utility. It must not train models, run batch launchers, modify `ACTIVE_LAGS`, read partition maps, perform shapefile joins, or execute notebooks.

## Default Inputs

Model metrics root:

`main_ablation_results/march2026_main_backup_month_ind_cont3`

Required model metrics files:

- `result_partition_k40_compare_DT_fs1/metrics_monthly.csv`
- `result_partition_k40_compare_DT_fs2/metrics_monthly.csv`
- `result_partition_k40_compare_DT_fs3/metrics_monthly.csv`
- `result_partition_k40_compare_GF_fs1/metrics_monthly.csv`
- `result_partition_k40_compare_GF_fs2/metrics_monthly.csv`
- `result_partition_k40_compare_GF_fs3/metrics_monthly.csv`

Required FEWSNET baseline files:

- `fewsnet_baseline_results_backup/fewsnet_baseline_results_fs1.csv`
- `fewsnet_baseline_results_backup/fewsnet_baseline_results_fs2.csv`

## Default Outputs

Output directory:

`main_ablation_results/march2026_main_backup_month_ind_cont3/monthly_performance_plots/`

Generated full-run artifacts:

- `geodt_monthly_performance.png`
- `georf_monthly_performance.png`
- `monthly_performance_manifest.json`

Generated artifacts are diagnostic outputs and should not be committed unless explicitly promoted.

## Options

| Option | Expected behavior |
|--------|-------------------|
| `--ablation-root PATH` | Override the default model metrics root. |
| `--fewsnet-root PATH` | Override the default FEWSNET baseline root. |
| `--output-dir PATH` | Override the default diagnostic output directory. |
| `--model {geodt,georf,all}` | Restrict plotting to one model for smoke testing or generate both figures. Default: `all`. |
| `--smoke` | Validate a reduced run without batch jobs; may generate only the selected model figure and manifest summary for smoke validation. |
| `--dry-run` | Validate inputs and report planned artifacts without writing figures. |

## File and Column Contract

Model `metrics_monthly.csv` files must contain:

- `test_month`
- `model`
- `precision`
- `recall`
- `f1`

FEWSNET baseline files must contain:

- `year`
- `quarter`
- `precision(1)`
- `recall(1)`
- `f1(1)`

## Selection Contract

- GeoDT is selected only from `result_partition_k40_compare_DT_fs*`.
- GeoRF is selected only from `result_partition_k40_compare_GF_fs*`.
- GeoXGB/XGB paths are ignored even when present.
- `pooled` and `partitioned` series come from the model metrics `model` column.
- fs1, fs2, and fs3 come from source directory suffixes and appear as figure rows in that order.

## FEWSNET Contract

- fs1 uses `fewsnet_baseline_results_fs1.csv`.
- fs2 uses `fewsnet_baseline_results_fs2.csv`.
- fs3 reuses `fewsnet_baseline_results_fs2.csv` as a comparison proxy.
- The fs3 FEWSNET label must be `FEWSNET baseline (fs2 reused for fs3)` or an equivalently explicit label.
- The script must not imply that FEWSNET has a native fs3 baseline.

## Time Alignment Contract

FEWSNET quarter rows align to model test months as follows:

- Quarter 1 -> `YYYY-02`
- Quarter 2 -> `YYYY-06`
- Quarter 4 -> `YYYY-10`
- Quarter 3 -> ignored for the current plotted cadence

All x-axis values must be sorted by chronological `test_month`.

## Missing Data Contract

- Missing plotted model or FEWSNET values appear as gaps.
- Each missing plotted point is recorded in `monthly_performance_manifest.json`.
- Missing values are never fabricated, interpolated, or imputed.

## Smoke Validation Contract

A smoke run must validate at minimum:

- Expected source files can be found for the selected model.
- GeoXGB is excluded.
- Required columns exist.
- Each applicable subplot has three planned series.
- Test months are chronological.
- fs3 FEWSNET label states that fs2 is reused.
- No full batch jobs, model training, or notebooks are invoked.
