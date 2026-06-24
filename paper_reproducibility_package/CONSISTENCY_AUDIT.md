# Consistency Audit

## Aligned With Manuscript

- Main model: GeoRF is the paper-facing model; GeoDT is auxiliary appendix /
  interpretability comparison.
- Temporal split: Stage 1 learns partitions on 2018-2020; Stage 2 builds fixed
  consensus maps; Stage 3 evaluates 2021-2024.
- Horizons: `fs1`, `fs2`, and `fs3` correspond to 4-, 8-, and 12-month forecasts.
- Stage 3 result manifests record `n_test_months_evaluated=12`, matching
  February, June, and October evaluation months for 2021-2024.
- Stage 3 GeoRF manifests record `5,716` polygons and `62,189` predictions per
  horizon.

## Accepted Differences

- The manuscript's `200,060` observations describe the analysis subset, not the
  full assembled CSV.
- The full source CSV is `1,029,240 x 88` and is intentionally not copied into
  this package.
- Predictor-count wording differs depending on whether base predictors or
  engineered lag/rolling/encoded features are counted.

## Excluded Experimental Workflows

GeoXGB, fs0 lag-1, and 2026-2027 forward/scenario prediction are not part of
this package. They remain in their current repo paths for future release-version
migration.
