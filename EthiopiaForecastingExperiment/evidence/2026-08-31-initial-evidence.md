# Initial Evidence Record — 2026-08-31

## Evidence status

This note records read-only inspection and accepted design decisions. It does
not claim that overfitting has been demonstrated, that a baseline has been
improved, or that CDS data have been acquired.

## Executed read-only evidence

### Authoritative FEWS NET panel

- Source:
  `Analysis/1.Source Data/FEWSNET_forecast_unadjusted_bm.csv`.
- Recorded release metadata: SHA-256
  `611f9e776380e28da3fc845888d66a117626f91b37868e236ce849d44bc8f651`,
  1,029,240 rows by 88 columns, 2010-01 through 2024-12, 22 ISO3
  countries, and 5,718 FEWS NET admin codes.
- The header contains `ISO3` and `FEWSNET_admin_code`; it does not contain
  `area_id`.
- A fresh read-only scan of `ISO3 == "ETH"` found 187,200 rows, 1,040 unique
  `FEWSNET_admin_code` values, and 180 months from 2010-01 through 2024-12.

### Reference weather table

- Reference only:
  `assembled_IPCCH/spatial/cds_api_tif_values_by_area_time.csv`.
- Header:
  `area_id,time,lat,lon,Rainf_f_tavg_mean,Tair_f_tavg_mean`.
- The inspected file contains six valid-month labels only: 2026-05 through
  2026-10. It does not contain 2026-11 through 2027-02.
- The table has no forecast-origin, issue-date, lead, ensemble, or vintage
  field. It cannot by itself prove forecast-time availability.
- Its `area_id` system and Ethiopia area count are not authoritative for this
  experiment.
- The associated codebook describes `Rainf_f_tavg_mean` and
  `Tair_f_tavg_mean` as FLDAS/GLDAS land-surface climate variables aggregated
  monthly and then by zone. Formal CDS provenance therefore remains open.

### Current pipeline behavior relevant to the baseline audit

- `config.py` fixes the production horizon schedule at 4, 8, and 12 months and
  rejects alternative schedules.
- `prepare_features()` creates scope-specific shifted columns but retains the
  original contemporaneous time-varying columns in the feature matrix. This is
  verified code behavior and creates an availability question for forward
  forecasting; it is not, by itself, proof of measured leakage or overfitting.
- The rolling split computes `train_start = train_end - (window - 1)` while
  applying an exclusive `dates < train_end` mask. With a configured 36-month
  window, the effective date range is 35 months. This requires characterization
  before changing any code.
- `feature_engineering_3()` returns from inside the outer feature loop. The
  current code therefore processes only the first eligible feature in its list.
  Its effect on released results has not yet been quantified.

## Accepted design decisions

1. The experiment workspace is isolated from production and paper artifacts.
2. Ethiopia membership is derived from the authoritative FEWS NET panel using
   `ISO3 == "ETH"`; `FEWSNET_admin_code` is the canonical spatial key.
3. Formal CDS data will be downloaded anew and aggregated to the FEWS NET
   polygons. Reference `area_id` values will not be joined to the experiment.
4. Weather lead 0 through lead 6 is inclusive and contains seven monthly values.
5. For forecast horizon `H` and weather lead `j`, target-relative lag is
   `H - j`. The agreed feature windows are:
   - horizon 4: leads 0-4, target-relative lags 4-0;
   - horizon 8: leads 0-6, target-relative lags 8-2;
   - horizon 12: leads 0-6, target-relative lags 12-6.
6. Before outcomes are available, comparison is limited to prediction classes,
   probabilities, and spatial differences. Precision, recall, F1, and accuracy
   claims require delayed outcome evaluation.
7. Baseline diagnosis and improvement precede formal CDS acquisition and model
   integration.

## Advisor input recorded as design rationale

The advisor discussion concluded that adding future weather forecasts has a
theoretical performance ceiling no higher than a model supplied with perfect
nowcasting-period weather information. This motivates resolving baseline
overfitting and generalization concerns first. The ceiling is a conceptual
argument, not an executed empirical result.

## Open evidence questions

- Is the suspected overfitting primarily temporal, spatial, partition-related,
  threshold-related, calibration-related, or feature-related?
- What train-versus-validation-versus-test evidence is available from current
  artifacts, and is it comparable across horizons?
- Do small branch or cluster sample sizes inflate apparent in-sample gains?
- How much of the current result depends on contemporaneous covariates that are
  unavailable at a genuine forecast origin?
- What is the effect of the effective 35-month rolling window and the early
  return in `feature_engineering_3()`?
- Which baseline improvements can be evaluated without altering the frozen
  paper result lineage?
- Which formal CDS product, forecast system, initialization schedule, units,
  ensemble statistic, and release latency will be used later?

## Explicitly not executed

- No production source-data or production code file was edited.
- No CDS request or download was made.
- No Ethiopia model was trained or evaluated.
- No baseline metric or artifact was regenerated.
- No overfitting diagnosis or correction has yet been accepted.
