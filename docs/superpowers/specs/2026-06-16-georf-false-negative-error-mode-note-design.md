# GeoRF False-Negative Error-Mode Note Design

## Goal

Create a GeoRF partitioned-model false-negative error analysis note for crisis-state observations that the model missed, with emphasis on Sudan, South Sudan, Somalia, Chad, Niger, northern Afghanistan, and central/southern Mozambique.

## Scope

This diagnostic uses only GeoRF partitioned predictions:

- `result_partition_k40_compare_GF_fs1/predictions_monthly.csv`
- `result_partition_k40_compare_GF_fs2/predictions_monthly.csv`
- `result_partition_k40_compare_GF_fs3/predictions_monthly.csv`

The analysis filters to `y_true = 1` and `y_pred_partitioned = 0`. It does not compare pooled, GeoDT, or GeoXGB false negatives.

## Evidence Sources

The script joins predictions to the current FEWSNET model input panel:

`C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_forecast_unadjusted_bm_phase_change.csv`

It uses available columns as descriptive proxies:

- conflict: ACLED distance, event-count, and fatality columns;
- market prices: FAO price, WFP price, WFP standardized price, CPI, Food CPI, and food inflation columns;
- seasonal timing: evaluated target month / season;
- lagged outcome: `fews_overall_phase_lagone`, `fews_overall_phase_lagtwo`, and `fews_overall_phase_lagthree`;
- boundary context: adjacent polygon crisis/prediction mix from `src/adjacency/polygon_adjacency_cache.pkl` when available;
- humanitarian impact: raw `pop` already present in the FEWSNET model input panel.

The note must use cautious language: these are descriptive error modes and proxy evidence, not causal attribution.

## Hotspot Definitions

- Sudan, South Sudan, Somalia, Chad, and Niger use exact `ADMIN0` matches.
- Northern Afghanistan is Afghanistan polygons with latitude greater than or equal to the Afghanistan median latitude in the model input panel.
- Central/southern Mozambique is Mozambique polygons with latitude less than or equal to the Mozambique median latitude in the model input panel.

## Outputs

Create `final_artifacts_in_paper_updated/10_false_negative_error_modes/` containing:

- `georf_partitioned_false_negative_hotspot_summary.csv`: hotspot by horizon false-negative counts, population-months, actual crisis population, missed-crisis share, mean probability, near-threshold share, and dominant month.
- `georf_partitioned_false_negative_error_modes.csv`: hotspot-level descriptive evidence table across horizons.
- `georf_partitioned_false_negative_hotspot_compact_table.md`: compact appendix-ready Markdown table.
- `georf_partitioned_false_negative_note.md`: Chinese reviewer-facing note and English appendix text.

Update `final_artifacts_in_paper_updated/README.md` with the new folder and file index.

## Validation

- Unit-test hotspot assignment, false-negative filtering, lagged-outcome proxy logic, neighbor-context proxy logic, and compact table rendering.
- Verify the prediction-to-feature join has no duplicate feature keys and no missing feature rows for evaluated predictions.
- Run script compile checks and `git diff --check`.
