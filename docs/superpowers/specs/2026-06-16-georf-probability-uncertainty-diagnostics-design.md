# GeoRF Probability and Uncertainty Diagnostics Design

## Goal

Address reviewer comments 12 and 13 by exporting GeoRF probabilities and
reporting uncertainty diagnostics for the existing pooled vs partitioned GeoRF
evaluation.

## Scope

- Model family: GeoRF only.
- Forecasting horizons: existing `fs1`, `fs2`, and `fs3` GeoRF Stage 3 result
  folders.
- Models compared: pooled RF and partitioned GeoRF local RF.
- Data unit: evaluated polygon-month observations from `predictions_monthly.csv`.
- Output folder: `final_artifacts_in_paper_updated/`.

Out of scope:

- GeoDT, GeoXGB, and FEWSNET Brier scores.
- New calibration model fitting such as Platt scaling or isotonic regression.
- Post-hoc threshold optimization.

## Stage 3 Probability Export

Patch `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py` so future GeoRF
Stage 3 runs write two additional columns:

- `y_prob_pooled`: pooled RF class-1 probability.
- `y_prob_partitioned`: partitioned local RF class-1 probability, using the
  same partition-specific / pooled-fallback dispatch as the hard prediction.

The existing hard prediction columns remain unchanged:

- `y_pred_pooled`
- `y_pred_partitioned`

This keeps the existing binary classification workflow intact while enabling
probability diagnostics.

## Threshold and Calibration Framing

The appendix note will state:

- The standard Stage 3 evaluation uses the classifier default threshold / hard
  prediction rule and does not perform separate threshold tuning.
- Probability columns are exported for Brier score, reliability, and uncertainty
  diagnostics.
- No additional probability calibration model is fit in this artifact; the
  reliability bins are diagnostic summaries of raw RF probabilities.

## Bootstrap Confidence Intervals

Create a GeoRF-only diagnostics script that reads fs1/fs2/fs3 prediction
artifacts with probability columns and computes paired bootstrap confidence
intervals.

Bootstrap design:

- Paired bootstrap: each replicate evaluates pooled and partitioned predictions
  on the same resampled observations.
- Country-clustered uncertainty: resample countries with replacement and keep
  all polygon-month observations within selected countries.
- Region-specific uncertainty: repeat the country-clustered bootstrap within
  each region when the region has enough countries.
- Random seed fixed for reproducibility.
- Default replicates: 1000.
- Confidence interval: percentile 2.5% and 97.5%.

Metrics:

- Precision, recall, and F1 from hard predictions.
- Brier score from class-1 probabilities.
- Delta metrics as `partitioned - pooled`, except Brier where lower is better
  and delta is still reported as `partitioned - pooled`.

## Reliability and Probability Uncertainty

For each horizon and model, compute reliability bins from raw class-1
probabilities:

- Bin probabilities into 10 equal-width bins from 0 to 1.
- Report bin count, mean predicted probability, observed crisis rate, and
  calibration gap.

Also compute simple uncertainty summaries:

- Mean prediction entropy or threshold-distance uncertainty by horizon and
  model.
- These are descriptive uncertainty summaries, not predictive intervals for
  future outcomes.

## Output Artifacts

Write:

- `georf_probability_bootstrap_ci.csv`
- `georf_probability_bootstrap_region_ci.csv`
- `georf_probability_brier_reliability.csv`
- `georf_probability_uncertainty_summary.csv`
- `georf_probability_reliability.png`
- `georf_probability_uncertainty_note.md`

Update `final_artifacts_in_paper_updated/README.md` to register these files.

## Validation

- Unit tests for probability extraction from binary classifiers, including
  single-class models.
- Unit tests for paired country-clustered bootstrap sampling.
- Unit tests for Brier score and reliability-bin summaries.
- Smoke run against the current GeoRF fs1/fs2/fs3 prediction folders after
  probability columns are available.
