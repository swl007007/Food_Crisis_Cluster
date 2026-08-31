# Ethiopia Forecasting Experiment Contract — Working Draft

**Status**: Design discussion in progress. This is not an implementation plan
and does not authorize data acquisition, code changes, model runs, or artifact
generation.

## Objective

Establish a defensible Ethiopia-only forecasting baseline, diagnose and reduce
unsupported generalization or overfitting, and then measure the prediction
change produced by forecast-derived weather covariates under a matched design.

## Current stage gates

### Gate 0 — Evidence inventory

Record authoritative inputs, current code behavior, released artifact lineage,
and unresolved assumptions without changing them.

### Gate 1 — Baseline diagnosis

Define and execute an approved audit covering temporal availability,
train/validation/test generalization, calibration, class imbalance and
thresholding, spatial partition complexity, branch/cluster support, and
feature stability.

### Gate 2 — Baseline improvement and freeze

Accept only improvements supported by matched temporal evaluation. Freeze the
Ethiopia cohort, target, feature availability rules, model configuration,
threshold policy, partitions, random seeds, and artifact schema.

### Gate 3 — CDS acquisition and harmonization

Select the formal CDS product and record forecast origin, issue time, valid
month, lead, units, ensemble statistic, spatial aggregation, missingness, and
provenance. Aggregate to the frozen FEWS NET polygons.

### Gate 4 — Forward comparison

Compare the frozen baseline and matched CDS-enhanced predictions. Before target
labels exist, report prediction deltas only.

### Gate 5 — Delayed outcome evaluation

When labels become available, evaluate the pre-registered predictions without
refitting or retroactively changing the weather vintage.

## Locked contracts

### Geography

- `ISO3 == "ETH"` defines membership in the raw authoritative panel.
- `FEWSNET_admin_code` is the canonical spatial key.
- Reference `area_id` systems are non-authoritative.

### Time

- Production forecast horizons remain 4, 8, and 12 months.
- Lead 0 is the forecast-origin month.
- Lead 0 through lead 6 is inclusive: seven monthly values.
- `target_relative_lag = forecast_horizon - forecast_lead`.

### Comparison

- A valid CDS comparison must be matched to the same Ethiopia cohort, target
  rows, split, model family, threshold policy, and non-CDS features.
- Forward prediction differences are not performance differences.
- Precision, recall, F1, accuracy, or superiority claims require observed target
  labels and delayed outcome evaluation.

### Isolation

- Existing source data, production entry points, release archives, paper
  artifacts, and reproducibility manifests are read-only.
- Experiment outputs must use a dedicated future output root and may not reuse
  production result names.

## Open decisions for continued grilling

1. What operational definition and evidence threshold will establish that the
   baseline is overfitting?
2. Is the first audit pooled-RF only, fixed-partition GeoRF only, or both?
3. Are the global fixed partitions retained, or is an Ethiopia-specific
   partition design allowed after baseline characterization?
4. Which target months and temporal folds define the baseline audit?
5. Which corrections are permitted before the baseline is re-frozen?
6. What formal CDS product and vintage contract will be used after Gate 2?

