# Quickstart: Monthly Performance Plots

This feature plots existing metrics only. Do not run training, full batch workflows, or notebooks for this validation path.

## Inputs

Default model metrics root:

```text
main_ablation_results/march2026_main_backup_month_ind_cont3
```

Default FEWSNET baseline root:

```text
fewsnet_baseline_results_backup
```

The Windows paths from the feature request correspond to these repository-relative directories when running from WSL under `/mnt/c/...`. Quote paths that contain spaces if passing absolute overrides.

## Smoke Test / Dry Run

Validate input discovery and contracts without writing full outputs:

```bash
python3 scripts/plot_monthly_performance_metrics.py --model geodt --dry-run
```

Expected smoke checks:

- Finds GeoDT fs1/fs2/fs3 `metrics_monthly.csv` files.
- Ignores GeoXGB files.
- Finds FEWSNET fs1 and fs2 baseline CSVs.
- Confirms required columns exist.
- Confirms x-axis test months are chronological.
- Confirms fs3 FEWSNET label says fs2 is reused for fs3.
- Does not invoke batch jobs, model training, or notebooks.

Optional single-model smoke figure:

```bash
python3 scripts/plot_monthly_performance_metrics.py --model geodt --smoke
```

Smoke outputs, if written, are diagnostic only and should not be committed unless explicitly requested.

## Full Diagnostic Figure Generation

Generate the two required figures and manifest:

```bash
python3 scripts/plot_monthly_performance_metrics.py
```

Expected generated artifacts:

```text
main_ablation_results/march2026_main_backup_month_ind_cont3/monthly_performance_plots/geodt_monthly_performance.png
main_ablation_results/march2026_main_backup_month_ind_cont3/monthly_performance_plots/georf_monthly_performance.png
main_ablation_results/march2026_main_backup_month_ind_cont3/monthly_performance_plots/monthly_performance_manifest.json
```

## Validation Checklist

After generation, verify:

- Exactly two full-run figures exist: GeoDT and GeoRF.
- Each figure has a 3x3 subplot layout.
- Rows are fs1, fs2, fs3.
- Columns are precision, recall, F1.
- Each applicable subplot has partitioned, pooled, and FEWSNET lines.
- Partitioned lines are solid.
- Pooled lines are dashed in the model color family.
- FEWSNET lines use a distinct comparison color.
- Test months are chronological.
- GeoXGB is absent from figures and manifest.
- fs3 FEWSNET is labeled as `FEWSNET baseline (fs2 reused for fs3)` or equivalently explicit wording.
- Missing plotted values appear as gaps and are listed in the manifest.
- Generated outputs are not committed unless explicitly promoted.

## Implementation Validation Notes

Validated commands:

```bash
python3 scripts/plot_monthly_performance_metrics.py --model geodt --dry-run
python3 scripts/plot_monthly_performance_metrics.py --model geodt --smoke
python3 scripts/plot_monthly_performance_metrics.py
python3 -m py_compile scripts/plot_monthly_performance_metrics.py
```

The full run produced exactly two PNG figures and one manifest in `main_ablation_results/march2026_main_backup_month_ind_cont3/monthly_performance_plots/`. Git status did not show those generated PNG/JSON outputs because they are ignored generated diagnostics.
