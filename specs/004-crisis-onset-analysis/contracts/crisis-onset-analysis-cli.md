# Contract: Crisis-Onset Analysis CLI

## Purpose

Define the user-facing command contract for generating crisis-onset-only monthly performance diagnostics from existing row-level prediction files while preserving the previous broader phase-change mode.

## Command

```bash
python3 scripts/plot_phase_change_monthly_performance.py [options]
```

## Options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--ablation-root PATH` | No | `main_ablation_results/march2026_main_backup_month_ind_cont3` | Source root containing `result_partition_k40_compare_{GF,DT}_fs{1,2,3}/predictions_monthly.csv` files. |
| `--output-dir PATH` | No | Mode-specific default | Destination for generated outputs. For `crisis_onset`, defaults to `<ablation-root>/crisis_onset_analysis`. Must not be `monthly_performance_plots` or the broader phase-change output directory unless the active mode matches that directory. |
| `--filter-mode {any_phase_change,crisis_onset}` | No | `any_phase_change` | Row filter to apply after previous true-label fields are computed. `crisis_onset` retains only `0 -> 1` true-label transitions. |
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

## Filter Contract

| Filter mode | Retained rows |
|-------------|---------------|
| `any_phase_change` | Non-first observations where current `y_true` differs from `previous_y_true`. |
| `crisis_onset` | Non-first observations where `previous_y_true = 0` and current `y_true = 1`. |

Both modes use previous available test month within each `(model_key, scope, FEWSNET_admin_code)` series, sorted by `month_start`.

## Output Contract

For `--filter-mode crisis_onset`, full mode writes the following to `--output-dir`:

- `georf_crisis_onset_monthly_performance.png`
- `geodt_crisis_onset_monthly_performance.png`
- `metrics_monthly_crisis_onset.csv`
- `summary_crisis_onset.xlsx`
- `filtered_predictions_crisis_onset.csv` when audit output is enabled
- `crisis_onset_manifest.json`

Smoke mode must demonstrate:

- one included model/scope subset,
- first-observation exclusion,
- at least one retained `0 -> 1` crisis-onset row when present in the sampled data,
- exclusion of `1 -> 0` crisis-recovery rows from crisis-onset mode when present in the sampled data,
- before/after row counts,
- recomputed precision, recall, and F1,
- one reduced plot in memory or a clearly labeled smoke output,
- one draft summary row.

Dry-run mode must not write generated artifacts.

## Validation Contract

The command must fail with a clear message when:

- an included file is missing required columns,
- no GeoRF or GeoDT prediction files can be found,
- `--output-dir` resolves to the existing standard `monthly_performance_plots` directory,
- `--filter-mode crisis_onset` output resolves to the broader `phase_change_monthly_performance` directory,
- `--dry-run` and `--smoke` are both requested,
- `--smoke` is requested without selecting exactly one model and one scope.

The command must record in the manifest when:

- XGB files are found and excluded,
- duplicate `(model, scope, FEWSNET_admin_code, month_start)` rows are detected,
- a month has no retained rows for the selected filter mode,
- any metric is blank/NA because of a zero denominator,
- crisis-onset mode retains row counts by model and scope.
