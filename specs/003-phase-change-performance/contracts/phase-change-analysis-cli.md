# Contract: Phase-Change Analysis CLI

## Purpose

Define the user-facing command contract for generating phase-change-only monthly performance diagnostics from existing row-level prediction files.

## Command

```bash
python3 scripts/plot_phase_change_monthly_performance.py [options]
```

## Options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--ablation-root PATH` | No | `main_ablation_results/march2026_main_backup_month_ind_cont3` | Source root containing `result_partition_k40_compare_{GF,DT}_fs{1,2,3}/predictions_monthly.csv` files. |
| `--output-dir PATH` | No | `<ablation-root>/phase_change_monthly_performance` | Destination for phase-change-only generated outputs. Must not be `monthly_performance_plots`. |
| `--model {georf,geodt,all}` | No | `all` | Model family selection. `all` includes GeoRF and GeoDT only. |
| `--scope {fs1,fs2,fs3,all}` | No | `all` | Forecasting scope selection. Full mode should include fs1, fs2, and fs3. |
| `--smoke` | No | false | Run a reduced validation path, rendering one reduced plot and draft summary row without producing the full artifact set. |
| `--dry-run` | No | false | Validate input discovery, column contracts, filtering counts, and planned outputs without writing generated artifacts. |
| `--write-audit` | No | true | Write filtered row-level audit output in full mode. |

## Input Contract

For each included source file, required columns are:

- `FEWSNET_admin_code`
- `month_start`
- `y_true`
- `y_pred_pooled`
- `y_pred_partitioned`

Optional audit column:

- `partition_id`

Included source folders:

- `result_partition_k40_compare_GF_fs1`
- `result_partition_k40_compare_GF_fs2`
- `result_partition_k40_compare_GF_fs3`
- `result_partition_k40_compare_DT_fs1`
- `result_partition_k40_compare_DT_fs2`
- `result_partition_k40_compare_DT_fs3`

Excluded source folders:

- Any `result_partition_k40_compare_XGB_fs*` folder.
- Any GeoXGB/XGBoost tokened source.

## Output Contract

Full mode writes the following to `--output-dir`:

- `georf_phase_change_monthly_performance.png`
- `geodt_phase_change_monthly_performance.png`
- `metrics_monthly_phase_change.csv`
- `summary_phase_change.xlsx`
- `filtered_predictions_phase_change.csv` when audit output is enabled
- `phase_change_manifest.json`

Smoke mode must demonstrate:

- one included model/scope subset,
- first-observation exclusion,
- before/after row counts,
- recomputed precision, recall, and F1,
- one reduced plot in memory or a clearly labeled smoke output,
- one draft summary row.

Dry-run mode must not write generated artifacts.

## Validation Contract

The command must fail with a clear message when:

- an included file is missing required columns,
- no GeoRF or GeoDT prediction files can be found,
- `--output-dir` resolves to the existing `monthly_performance_plots` directory,
- `--dry-run` and `--smoke` are both requested.

The command must record in the manifest when:

- XGB files are found and excluded,
- duplicate `(model, scope, FEWSNET_admin_code, month_start)` rows are detected,
- a month has no retained phase-change rows,
- any metric is blank/NA because of a zero denominator.
