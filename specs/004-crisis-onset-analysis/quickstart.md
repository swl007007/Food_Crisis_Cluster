# Quickstart: Crisis Onset Analysis

## Preconditions

- Work from the repository root.
- Use the existing Python 3.12 environment with pandas, numpy, matplotlib, and Excel-writing support installed.
- Do not run model training, batch evaluation, notebooks, standalone prediction workflows, or scenario workflows for this feature.
- Use existing row-level source files under:

```text
main_ablation_results/march2026_main_backup_month_ind_cont3
```

## Smoke-test path

Run a reduced crisis-onset validation before generating full outputs:

```bash
python3 scripts/plot_phase_change_monthly_performance.py --filter-mode crisis_onset --model georf --scope fs1 --smoke
```

Expected smoke evidence:

- exactly one GeoRF fs1 `predictions_monthly.csv` source is loaded,
- required columns are validated,
- first observed row per `FEWSNET_admin_code` is excluded,
- retained crisis-onset rows satisfy `previous_y_true = 0` and `y_true = 1`,
- `1 -> 0` crisis-recovery rows are excluded from crisis-onset mode when present in the sampled data,
- precision, recall, and F1 are recomputed from retained rows for pooled and partitioned series,
- one reduced plot and one draft summary row are reported or written to a clearly labeled smoke output,
- no standard output directory or broader phase-change output directory is overwritten.

## Dry run

Validate full input discovery and planned outputs without writing generated artifacts:

```bash
python3 scripts/plot_phase_change_monthly_performance.py --filter-mode crisis_onset --dry-run
```

Expected dry-run evidence:

- six included source files are identified: GF and DT for fs1, fs2, and fs3,
- XGB files are listed as excluded,
- output directory resolves to `main_ablation_results/march2026_main_backup_month_ind_cont3/crisis_onset_analysis`,
- planned filenames are crisis-onset-specific,
- no files are written.

## Full exploratory analysis

Generate crisis-onset-only plots, tables, audit files, and manifest:

```bash
python3 scripts/plot_phase_change_monthly_performance.py --filter-mode crisis_onset
```

Expected generated outputs:

```text
main_ablation_results/march2026_main_backup_month_ind_cont3/crisis_onset_analysis/
├── georf_crisis_onset_monthly_performance.png
├── geodt_crisis_onset_monthly_performance.png
├── metrics_monthly_crisis_onset.csv
├── summary_crisis_onset.xlsx
├── filtered_predictions_crisis_onset.csv
└── crisis_onset_manifest.json
```

## Manual validation checklist

After full generation, confirm:

1. `monthly_performance_plots/` timestamps and contents are unchanged.
2. `phase_change_monthly_performance/` timestamps and contents are unchanged.
3. `crisis_onset_manifest.json` records included GF/DT source files and excluded XGB files.
4. Manifest row counts show before-filter, first-observation-excluded, non-onset-excluded, and retained counts for each model/scope.
5. Filtered audit rows all satisfy `previous_y_true = 0` and `y_true = 1`.
6. Monthly metrics are recomputed from retained rows, not from existing `metrics_monthly.csv` or broader phase-change metric files.
7. Blank/NA metrics caused by zero denominators are listed in the manifest.
8. `summary_crisis_onset.xlsx` includes only GeoRF(crisis onset) and GeoDT(crisis onset) for fs1, fs2, and fs3.
9. Plots are clearly labeled as crisis-onset-only and contain no FEWSNET or XGB series.
