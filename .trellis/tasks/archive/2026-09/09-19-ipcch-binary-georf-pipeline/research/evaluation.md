# IPCCH evaluation — Q9a/Q9b approved

Planning evidence only. Q4b already fixes E_all/E_persist, Q7 fixes models and
thresholds, and class1 F1 is the primary metric. Q9a/Q9b below are approved.
This document does not report experiment results.

## Q9a approved aggregation and metric definitions — 2026-09-20

- Main rows are separate for each horizon1/3/6/12 and cohort. E_all compares the
  three learned arms; E_persist compares all four on exactly matching keys.
  Aggregate TP/FP/FN/TN over all scheduled observed test area-months in that cohort
  before computing class1 metrics. Each observation has unit weight. There is
  no population weight, country equal-weighting, mean monthly F1, or combined
  four-horizon headline F1. Regions/countries with more observed labels contribute
  more rows; this estimates performance over available labeled observations.
- Report class1 F1=2TP/(2TP+FP+FN), precision=TP/(TP+FP), recall=TP/(TP+FN).
  A positive denominator with zero numerator gives0. A zero denominator gives
  NaN plus a reason, without borrowing another class's score. For example,
  TP=0,FP=1,FN=1 gives F1=0; TP=FP=FN=0 gives undefined class1 F1 even with many TN.
  Keep support, predicted positives, observed positive-label count, prevalence and all
  four confusion cells so all-negative/no-positive cases remain interpretable.
- Export target-month, target-year, country and partition-assignment-provenance
  breakdowns separately within each horizon/cohort, applying the same formulas.
  No automatic cross-product of all groupings or equal-country/month averaged
  headline score. Provenance distinguishes learned, nearest-donor-assigned and
  unresolved areas; retain separate pooled-model fallback reasons in row outputs.
  These subgroup comparisons use the same selected keys for every eligible arm.
- Main paired deltas are partitioned RF minus each eligible baseline within the
  same horizon/cohort. Do not subtract a full-sample score from a persistence
  subsample score. If either metric is undefined, its delta stays undefined.
  Include observation/area/country counts, prevalence, persistence coverage and
  fallback counts. Keep partial2026 and Stage1 supplementary diagnostics separate
  from the approved main Stage3 period.
- Validate unique keys and binary truth/predictions before counting. Missing
  learned hard predictions are run/report failures under Q4b, not per-model
  complete-case masks. Hard-label F1 does not require a probability column, so
  persistence remains eligible without a fabricated probability. No empty cohort
  is presented as a zero score; record empty support/undefined metrics explicitly.

This changes IPCCH reporting semantics only. Retain the corrected GeoRF F1/q/
split core, frozen release and previous outputs. Saved predictions must reproduce
the reports. Q9b fixes uncertainty and claims below before scores exist.

## Q9b approved uncertainty and claim scope — 2026-09-20

- Apply only to main Stage3 summary rows, separately per horizon/cohort. For
  E_all include the three learned arms and partitioned-minus-pooled/XGB F1;
  for E_persist include all four arms and all three partitioned-minus-baseline
  F1 contrasts. Do not add CIs for month/year/country/routing subgroups, partial2026
  or Stage1 diagnostics in this baseline. PR/confusion/support remain point reports.
- Use1000 draws and `numpy.random.default_rng(42)` separately for each horizon/
  cohort. Sort stable country IDs first. If the cohort has K countries, draw K
  countries uniformly with replacement and retain all available area-month rows
  for each selected copy. Preserve duplicates/multiplicity. Validate every row's
  country assignment before sampling; missing country IDs are errors, not silently
  dropped rows. This reuses the small existing country-resampling pattern.
- All models share the identical sampled rows in a replicate. Recompute Q9a
  confusion-count F1 on that sample, then subtract within that same replicate.
  Take2.5/97.5percentiles of valid draws for each F1 and paired difference. Do not
  subtract two marginal interval endpoints or use independent draws across arms.
- Save requested count1000, per-statistic valid/undefined counts/reasons, seed,
  country count and replicate statistics/draw identities sufficient to reproduce
  intervals. Keep NaN when Q9a makes a metric undefined; a paired delta requires
  both F1 values. Do not replace undefined scores with0 or resample until1000valid
  replicates. If any draws are undefined, explicitly label resulting quantiles
  as using only defined replicates and give their count/fraction.
- Emit no interval if the original point is undefined, K<2, or fewer than2
  replicates are defined; keep point/support and the failure reason. These are
  non-degeneracy checks, not a claim that two countries/draws give reliable inference.
- No model fitting, monthly training replay or partition learning occurs during
  bootstrap. Intervals describe country-composition variation conditional on the
  observed cohort and saved fitted predictions. They are not future-event prediction
  intervals and omit training/map estimation, cross-country common shocks and
  shifts in future years. Country-level resampling preserves within-country rows'
  observed spatial/time dependence; it does not make Q9a country-equal-weighted F1.
- Report all specified contrasts with point estimates and pointwise95% intervals.
  No p-value/significance-star table, simultaneous coverage or multiple-comparison
  correction is included. Do not select a favorable horizon/cohort/contrast to
  claim general superiority. Success remains a valid reproducible comparison,
  including a null or negative result; no winning-model condition is imposed.

Choice/tradeoff: this reuses the existing cluster logic and preserves within-country
dependence without a costly retraining experiment. It answers a conditional test-set
comparison, not uncertainty of the entire model-building process. Q9b is approved.

## Existing metric behavior — verified code, not IPCCH policy

Let B=`GeoRFBaseline/scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py` and
M=`GeoRFBaseline/src/metrics/metrics.py`.

- B:365-400 `compute_binary_metrics` yields class1 PRF, n and TP/FP/FN/TN. It
  calls M:68-103 `get_class_wise_accuracy`, dependent on NUM_CLASS, and M:15-38
  `get_prf(..., nan_option='mean')`. That fills undefined precision/recall from
  other classes before calculating F1, while returning raw precision/recall.
  For an all-zero truth/prediction sample, class0 scores1 and class1 F1 becomes1.
  Direct inspection confirms the same implementation in both baseline and root.
- B:1164-1166,1207-1229,1279-1283 exports per-month metrics. B:563-594,1302-1305
  recomputes polygon metrics across time but omits polygon confusion-count fields.
  B:1353-1367 console summaries average monthly F1; model SD uses pandas ddof1,
  delta SD uses NumPy ddof0, with different NaN behavior. No global concatenated
  cross-month/area confusion-count PRF table is exported by that script.
- B:1051-1053 skips empty test months. Row exports at B:1231-1250 depend on an
  available admin-code array; metric existence alone does not prove row lineage.

Let P=`scripts/paper_artifacts/analyze_georf_probability_uncertainty.py`
(root paper artifact utility, NOT inside GeoRFBaseline).

- P:72-112 `compute_model_metrics` independently masks each arm by truth,
  hard-prediction AND probability completeness. It computes F1 through P/R and
  safe-divide, making TP=0 cases NaN even when direct confusion F1 would be0.
  Do not import its filtering/formula wholesale for paired four-arm IPCCH F1.
- Hard predictions are converted to int without strict{0,1}validation. A common
  sampled DataFrame does not guarantee shared model support when masks differ.

## Existing uncertainty utility — evidence for approved Q9b

- P:154-165 `resample_clusters(df, cluster_col, rng)` draws the same number of
  unique nonmissing clusters with replacement, retaining every row for each draw
  and marking `_bootstrap_draw`. Default cluster is ADMIN0 country. This is one
  country-cluster bootstrap, not an independent area-month or two-stage sample.
- P:184-203 uses1000replicates and NumPy default_rng(seed42); one sample per
  replicate supplies both arms, then partitioned-minus-pooled differences. It
  already includes F1/precision/recall/Brier but is hardcoded to two arms.
- P:175-181 takes2.5/97.5percentiles after silently dropping NaNs; all-NaN yields
  NaN bounds. P:205-226 lacks effective-replicate counts and reports original
  DataFrame support even if an arm filtered more rows. IPCCH must not inherit
  silent arm-specific exclusions/replicate losses.
- P:354-362 runs each horizon separately using the same seed. It does not
  perform joint horizon/multiple-comparison inference. Region routines require
  >=3countries (P:229-245), but full-sample bootstrap has no minimum check.
- P:272-314,489-524 full CLI depends on FEWSNET keys/shapefile and fs1/2/3;
  reuse only the small sampling/percentile pattern, not the CLI/paper outputs.
  Tests at src/tests/test_georf_probability_uncertainty.py:47-66,85-119 cover
  ordinary F1/cluster retention/CI structure, not the edge cases described above.

These CIs resample fixed predictions, not training, monthly model refits or
partition estimation. Country draws preserve observed dependence within each
country across areas/months but do not model cross-country shocks or uncertainty
about future years. Equal country sampling probabilities do not equalize countries'
contributions to pooled observation-level F1. Apply the approved Q9b rules above.
