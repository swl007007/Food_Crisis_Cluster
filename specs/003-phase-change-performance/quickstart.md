# Quickstart: Phase-Change Monthly Performance

## Preconditions

- Work from the repository root.
- Use the existing Python 3.12 environment with pandas, numpy, matplotlib, and Excel-writing support installed.
- Do not run model training, batch evaluation, notebooks, or standalone prediction workflows for this feature.
- Use existing row-level source files under:

```text
main_ablation_results/march2026_main_backup_month_ind_cont3
```

## Smoke-test path

Run a reduced validation before generating full outputs:

```bash
python3 scripts/plot_phase_change_monthly_performance.py --model georf --scope fs1 --smoke
```

Expected smoke evidence:

- exactly one GeoRF fs1 `predictions_monthly.csv` source is loaded,
- required columns are validated,
- first observed row per `FEWSNET_admin_code` is excluded,
- phase-change rows are counted after comparing `y_true` with the previous available test month,
- precision, recall, and F1 are recomputed from filtered rows for pooled and partitioned series,
- one reduced plot and one draft summary row are reported or written to a clearly labeled smoke output,
- no standard output directory is overwritten.

## Dry run

Validate full input discovery and planned outputs without writing generated artifacts:

```bash
python3 scripts/plot_phase_change_monthly_performance.py --dry-run
```

Expected dry-run evidence:

- six included source files are identified: GF and DT for fs1, fs2, and fs3,
- XGB files are listed as excluded,
- output directory resolves to `main_ablation_results/march2026_main_backup_month_ind_cont3/phase_change_monthly_performance`,
- no files are written.

## Full exploratory analysis

Generate phase-change-only plots, tables, audit files, and manifest:

```bash
python3 scripts/plot_phase_change_monthly_performance.py
```

Expected generated outputs:

```text
main_ablation_results/march2026_main_backup_month_ind_cont3/phase_change_monthly_performance/
├── georf_phase_change_monthly_performance.png
├── geodt_phase_change_monthly_performance.png
├── metrics_monthly_phase_change.csv
├── summary_phase_change.xlsx
├── filtered_predictions_phase_change.csv
└── phase_change_manifest.json
```

## Manual validation checklist

After full generation, confirm:

1. `monthly_performance_plots/` timestamps and contents are unchanged.
2. `phase_change_manifest.json` records included GF/DT source files and excluded XGB files.
3. Manifest row counts show before-filter, first-observation-excluded, and after-filter counts for each model/scope.
4. Filtered audit rows all satisfy `y_true != previous_y_true`.
5. Monthly metrics are recomputed from filtered rows, not from existing `metrics_monthly.csv` files.
6. Blank/NA metrics caused by zero denominators are listed in the manifest.
7. `summary_phase_change.xlsx` includes only GeoRF(phase change) and GeoDT(phase change) for fs1, fs2, and fs3.
8. Plots are clearly labeled as phase-change-only and contain no FEWSNET or XGB series.
