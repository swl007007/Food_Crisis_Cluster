# Quickstart: Actual Predicted Dashboard

## Purpose

Generate and smoke-test one exploratory static dashboard comparing actual crisis labels with partitioned predicted crisis labels for GeoRF and GeoDT fs1-fs3 outputs.

## Preconditions

- Run from the repository root.
- Use the existing project Python/geospatial environment with pandas and GeoPandas available.
- Do not run model training, full batch workflows, or notebooks for this feature.
- Confirm the source result directory exists:
  `main_ablation_results/march2026_main_backup_month_ind_cont3`
- Confirm the global FEWSNET shapefile is available:
  `C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp`

## Planned Generator Command

```bash
python scripts/generate_actual_predicted_dashboard.py \
  --input-dir "main_ablation_results/march2026_main_backup_month_ind_cont3" \
  --shapefile "C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp" \
  --output-dir "main_ablation_results/march2026_main_backup_month_ind_cont3/actual_predicted_dashboard"
```

If the active shell exposes Python as `python3`, use `python3` instead of `python`. If WSL `python3` does not have GeoPandas, use the documented Windows Python fallback from this workstation:

```bash
"/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe" scripts/generate_actual_predicted_dashboard.py \
  --input-dir "main_ablation_results/march2026_main_backup_month_ind_cont3" \
  --shapefile "C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp" \
  --output-dir "main_ablation_results/march2026_main_backup_month_ind_cont3/actual_predicted_dashboard"
```

If running from Windows CMD, keep output ASCII-safe.

## Expected Outputs

```text
main_ablation_results/march2026_main_backup_month_ind_cont3/actual_predicted_dashboard/
├── actual_predicted_dashboard.html
├── manifest.json
└── assets/                  # Only if needed for size/performance
```

The HTML should be self-contained unless asset extraction is needed for size or performance. Any supporting assets must remain in the same output directory and be listed in `manifest.json`.

## Smoke-Test Path

1. Generate the dashboard using the command above.
2. Confirm `manifest.json` reports:
   - `workflow_mode = exploratory diagnostics/tooling`
   - `production_status = exploratory`
   - included model tokens are only `GF` and `DT`
   - excluded source patterns include `result_partition_k40_compare_XGB_fs*`
   - `join_key = FEWSNET_admin_code`
   - shapefile path ends in `FEWS_Admin_LZ_v3.shp`
   - threshold contract is `none; existing binary labels only`
3. Open `actual_predicted_dashboard.html` in a desktop browser.
4. Select a known complete combination such as GeoRF, fs2, `2024-06-01`.
5. Verify the left panel is labeled as actual/reference and uses `y_true` values.
6. Verify the right panel is labeled as predicted/comparison and uses `y_pred_partitioned` values.
7. Verify model choices exclude GeoXGB/XGBoost.
8. Verify date and scope selectors populate from available data.
9. Verify at least one selected combination has a non-empty shapefile join in the manifest smoke-test section.
10. Verify no standard forecast, standalone prediction, or synthetic scenario deliverable was overwritten.

## Failure Handling Checks

- Temporarily point to an output-only copy with one model-scope file omitted to verify the missing combination is omitted or reported visibly.
- Verify an invalid or Nigeria-only shapefile path fails validation rather than silently rendering the wrong geography.
- Verify unexpected non-binary `y_true` or `y_pred_partitioned` values are reported and not silently reinterpreted.

## Non-Goals

- Do not change `ACTIVE_LAGS`.
- Do not change partition maps, spatial clustering, or model code.
- Do not execute notebooks.
- Do not create a backend server, Dash app, or full GIS frontend.
- Do not promote the output as a production forecast deliverable.
