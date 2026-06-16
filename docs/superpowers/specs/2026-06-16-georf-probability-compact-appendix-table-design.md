# GeoRF Probability Compact Appendix Table Design

## Goal

Add a small appendix-ready table that summarizes the country-clustered paired
bootstrap results without exposing the full diagnostic CSV in the manuscript.

## Scope

- Use the existing `georf_probability_bootstrap_ci.csv` output.
- Keep the reliability figure as the primary visual artifact.
- Add one compact table with one row per forecasting horizon.
- Keep full bootstrap and region-specific CSVs as diagnostic artifacts only.

## Table Format

Columns:

- `forecasting_horizon`
- `delta_precision`
- `delta_recall`
- `delta_f1`
- `delta_brier`

Each metric cell is formatted as:

`point [95% CI low, high]`

Values are rounded to three decimals. Delta is always `partitioned - pooled`.
For Brier score, negative values favor the partitioned model because lower
Brier score is better.

## Outputs

- `final_artifacts_in_paper_updated/georf_probability_bootstrap_compact_table.csv`
- `final_artifacts_in_paper_updated/georf_probability_bootstrap_compact_table.md`

Update `final_artifacts_in_paper_updated/README.md` to register both files.

## Validation

- Unit test compact-cell formatting.
- Unit test the 3-row horizon table from synthetic bootstrap rows.
- Regenerate the compact table from existing diagnostics and inspect shape.
