# Evidence for Step 3 selective-correction design

Inspected on 2026-09-18. These are source/document observations, not results from a new model run or fresh numerical reproduction. Paths are repository-relative.

- `PIPELINE_WORKFLOW.md:5-10,165-206`: main Stage 1/2 partition provenance, Stage 3 2021–2024 evaluation, general/month-specific partition comparison.
- `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py:72-101`: configured 36-month window, six-month validation default, RF parameters, 50-row partition minimum.
- Same file `:296-335`: original partition fitting uses SMOTE and falls back to pooled for small/single-class partitions. This fallback cannot serve the new expert-error target.
- Same file `:1037-1119`: scope/month-specific maps and separate pooled/partitioned splitting; baseline uses dummy groups to preserve its full window.
- `src/customize/customize.py:407-445`: actual outer range starts at `train_end-(train_window_months-1)` and uses exclusive `dates < train_end`; configured 36 yields 35 monthly timestamps. User explicitly confirmed retaining this behavior. Existing group eligibility must also be preserved.
- `archived/release_20260624_nonpaper_pipelines/legacy_misc/app_final/fewsnet_baseline_evaluation.py:67-75`: binarize near/medium projections, raw missing → 0, sort full admin history, record shift(4/8).
- `other_outputs/georf_country_performance/spec.md:13-37`: approved main expert lineage, archived-baseline equality, common support, fs3 N/A (no proxy), main release-month schedule.
- `other_outputs/georf_country_performance/generate.py:105-145`: existing implementation of original expert convention, 39-quarter fs1/fs2 archived checks at 1e-12 tolerance, main-key/truth checks. This verification was inspected, not executed in this design session.
- `EthiopiaForecastingExperiment/run_fewsnet_selective_correction_xgb.py:46-49,58-89,114-129,383-385,489,536-610`: ETH wrong-label classifier, input expert anchor, directional 20/2/75% gates, strict threshold comparison, validation F1 selection, refit.
- `EthiopiaForecastingExperiment/run_fewsnet_residual_xgb.py:54-59,350-396`: ETH calendar-based expert mapping, including fs3 T-12 medium anchor. Not an approved source for the new expert definition.
- `.trellis/tasks/09-04-ethiopia-fewsnet-selective-correction-xgb/design.md`: prior selective mechanism and no-test-retuning contract; its input lineage and XGBoost grid do not transfer.
- `.trellis/tasks/09-04-ethiopia-fewsnet-residual-xgb/prd.md`: distinct additive residual architecture. User explicitly chose selective flipping instead.

## Remaining empirical checks

Raw source dates versus declared origins; archived metric reproduction; raw data hashes and environment; new correction preprocessing fit boundaries; unchanged pooled/fs3 predictions. These are required before acceptance of a future implementation. Do not infer that record shifts establish calendar-horizon availability merely because the historical metric calculation matches.
