# Final comparisons and uncertainty reporting — accepted R49

Planning only, 2026-10-04. The user accepted this reporting policy as R49 / v0.43,
following R48. No predictions, bootstrap draws or training ran.

## Source evidence: the two completed packages use different procedures

- `IPCCHGeoRFExperiment/report_results.py:63-71,1762-1784` uses 1000 fixed
  country-cluster draws, seed42. Within each H/cohort, draw K countries with
  replacement from its sorted K-country list; country multiplicities weight
  each arm's TP/FP/FN. Paired arms use the same multiplicities. It reports crisis
  F1 and paired F1 differences, not a mean of monthly F1. Its interval uses the
  finite draws' 2.5/97.5 percentiles with linear interpolation, with point/cluster
  and minimum-defined-draw checks (`:1786-1807`). Undefined draws are not replaced.
  The main-period runner excludes 2026, Stage1 and subgroup intervals (`:1893-1912`).
- `IPCCHPopulationHistoryExperiment/report_results.py:45-47,387-458` instead
  seeks 2000 valid joint country draws, with up to 20000 attempts and seed42.
  It rejects a draw if any required method/H is undefined, and averages paired
  differences across H. It uses the history-only cohort. Its multi-arm and
  leave-year-out stable-gain rules (`:518-591`) answer that task's questions.
  Those cohort, cross-H averaging and success rules are not inherited here.
- `IPCCHGeoRFExperiment/README.md:183-184` explicitly limits interpretation to
  country-composition uncertainty conditional on saved predictions, excluding
  model-training, partition-learning and future-year uncertainty.

## Accepted R49 report and interval contract

1. For each H separately, report the full R8/R25 metric panel on the main
   2023-2025 observed-key cohort. Compare the new routed GeoXGB quartet against
   the matched pooled quartet on all common valid model keys (E_all), and against
   persistence on the paired keys where persistence exists (E_persist). Report
   both sides' metrics, their differences and cohort coverage; do not compare
   scores from different keys. The matched pooled model is R44's same selected
   G_H, not a separately optimized claim of the best possible pooled model.
2. Add paired uncertainty intervals for the primary crisis-positive F1
   difference only: GeoXGB minus matched pooled on E_all, and GeoXGB minus
   persistence on E_persist, independently for each H. This gives eight main
   contrast intervals. All other required metrics, including raw/projected q3
   R2, retain point estimates and paired differences; they are not omitted.
3. In each H/cohort let K be its number of represented countries. With
   numpy.default_rng(seed=42), generate exactly 2000 draws of K countries with
   replacement from the frozen sorted country-ID list. Reset the RNG per
   H/cohort and save its identity. Each selected country brings all that cohort's
   area/month rows; a country selected m times receives multiplicity m.
   Use the same draw for both compared arms, sum TP/FP/FN across those rows,
   compute each F1, then take their difference. Do not average country/month F1.
   Uniform country sampling does not change the original point estimand into a
   country-equal metric: the point score remains the adopted row-pooled score.
4. For each contrast, if K>=2, the observed paired F1 difference is defined and
   all 2000 paired replicate differences are finite, report the 2.5 and 97.5
   percentiles with linear interpolation. Otherwise report interval=NA with the
   reason and all available draws/defined/undefined counts. Empty cohorts have
   no draws; K=1 may retain its constant draws but has no interval. Do not fill
   undefined F1 with zero, discard draws and silently renormalize, or keep drawing
   until the interval is available. One unavailable interval does not suppress
   another valid H/contrast and is not a model technical failure.
5. Save the exact paired key lists, sorted countries, per-country confusion
   counts, all country multiplicities, per-arm and paired replicate scores,
   RNG/seed/B/percentile method, interval/NA reasons and the source prediction
   identities. Reporting must reproduce from frozen saved predictions without
   refitting models, relearning maps or rerunning model selection. This adds no
   scientific model fits to the R48 envelope.
6. Name these pointwise, descriptive 95% country-cluster bootstrap intervals,
   conditional on the stored predictions and observed cohort. Country blocks
   preserve within-country spatial/temporal dependence, but do not model shared
   shocks or dependence between countries, unseen years/countries, training/map
   selection uncertainty, or source-vintage uncertainty. Country independence /
   exchangeability is only a working approximation. There is no multiplicity
   adjustment or joint all-H confidence statement, no IID area-row resampling,
   and no use of interval results to tune the model.
7. 2026 remains separate and descriptive: report the full point-metric panel,
   paired deltas and coverage, without formal intervals in the first version.
   Country/month diagnostic tables likewise remain descriptive. Do not merge
   periods or H to obtain a favorable interval.
8. Separate scientific gain from successful execution. Report every H's signed
   delta and interval, including ties, losses and unavailable intervals. Positive
   delta means a higher observed score; a positive lower interval endpoint gives
   positive evidence under the limited country-resampling interpretation only.
   Do not require all H/countries/months to beat persistence as a completion gate,
   invent a cross-H score, or turn an interval into a future-performance claim.
   Complete reproducible evidence with a null/negative result is a legitimate
   research deliverable; technical failures or missing required evidence still
   cannot be called a completed run. No additional search follows a disappointing
   main-period result without a separately revised scientific design.

This accepted policy keeps the GeoRF package's per-H paired country-block structure,
raises the fixed Monte Carlo count to 2000 and uses a stricter unavailable-draw
policy than either old report. Suppressing an interval if any draw is undefined
may yield NA for sparse crisis cohorts; it is a transparent first-version rule,
not evidence that the model failed or that a different bootstrap was run.
The intervals measure one limited source of variation; they do not establish
universal generalization or confirmatory significance across the eight contrasts.
