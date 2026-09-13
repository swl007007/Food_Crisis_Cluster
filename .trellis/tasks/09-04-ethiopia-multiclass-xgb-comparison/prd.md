# ETH multicategory XGBoost comparison

## Goal

Fit a simple four-class Ethiopia IPC-phase XGBoost on the aligned v5 inputs and compare it fairly with GeoRF v5 and FEWS NET expert forecasts.

## Confirmed facts

- Input provider: the four immutable 88-predictor snapshots from
  `eth_aligned_refit_20260904_seed5_v5` (fs0/fs1/fs2/fs3 = 1/4/8/12 months).
- The snapshots contain binary `fews_ipc_crisis`; multiclass truth must be joined
  one-to-one from the ETH working panel by
  `FEWSNET_admin_code + target_month=date` using raw `fews_ipc`.
- Observed labels are phases 1-4. Phase 5 is absent from this data contract.
- The 2021-06 target has only one observed row and cannot support multiclass
  evaluation; it must remain an explicitly suppressed audit fold.
- GeoRF v5 predicts only crisis/non-crisis (`phase >= 3`). FEWS NET exposes
  multiclass expert phases only for fs1 near-term at T-4 and fs2 medium-term at
  T-8; fs0/fs3 have no expert baseline.
- Windows Python 3.12 has XGBoost 3.0.0. The archived GeoXGB workflow is not a
  suitable four-class baseline and will not be revived.

## Requirements

1. Train one pooled ETH `XGBClassifier` multiclass model per frozen temporal
   fold; do not add GeoRF partitions or change production code.
2. Use the same 88 ordered predictors and forecast origins as v5. Fit any
   preprocessing and hyperparameter choice using training/validation data only;
   test labels must never select settings.
3. Encode phases 1-4 as classes 0-3 for fitting and map predictions back to
   phases 1-4. Do not synthesize missing classes or targets.
4. Tune exactly eight candidates per fold from
   `max_depth={3,6} × min_child_weight={1,5} × n_estimators={200,400}` with
   `learning_rate=0.05`, `subsample=0.8`, and `colsample_bytree=0.8` fixed.
   Select by macro-F1 on the latest six eligible training months; break ties by
   shallower depth, higher `min_child_weight`, then fewer trees. Refit the chosen
   candidate on the complete 36-month training fold before testing.
5. Do not use SMOTE. Fit XGBoost with fold-local square-root inverse-frequency
   sample weights `sqrt(n / (4*n_k))`, normalized to mean one. Compute weights
   from the fit subset during tuning and from the complete training fold during
   final refit; validation and test metrics remain unweighted.
6. Use identical admin-month support within every reported model comparison.
   Retain per-fold predictions, support, chosen parameters, metrics, confusion
   matrices, source hashes, and a compact comparison figure/table under a new
   dedicated output directory.
7. Use two explicit comparison tracks:
   - native four-class XGBoost versus FEWS NET for fs1/fs2;
   - phase-3+ collapsed XGBoost versus binary GeoRF v5 and FEWS NET for fs1/fs2,
     with fs0/fs3 limited to XGBoost versus GeoRF.
   Never present binary GeoRF as a four-class model.
8. Do not modify the aligned v5 run, the 2026-09-01 GeoRF run, canonical aligned
   inputs, global XGBoost configuration, or the production Stage 1-3 workflow.
9. Treat each horizon separately as the primary result and average eligible target
   months with equal weight. Exclude the suppressed 2021-06 fold. Use common
   support for three-way FEWS NET comparisons (currently ten months for fs1 and
   ten for fs2); report an equal-weight cross-horizon summary only as secondary.
10. For the native four-class track, headline macro-F1 and also report exact
    accuracy, ordinal MAE, quadratic-weighted Cohen's kappa, per-class
    precision/recall/F1/support, and the 4x4 confusion matrix. For the collapsed
    binary track, headline crisis precision/recall/F1 and also report balanced
    accuracy and confusion counts.
11. Pass missing predictors directly to XGBoost and use its native learned missing
    directions; do not median-impute them. Document that the comparison therefore
    measures the full method, including native missing-value handling, rather than
    an imputation-controlled classifier ablation.
12. Use random seed 5 for the single experiment run; do not add repeated-seed
    aggregation unless the bounded run later provides evidence of instability.
13. Write only the row-level predictions and four probabilities, selected
    parameters by fold, monthly and equal-weight summary metrics, confusion
    matrices, one compact comparison figure, source hashes, and run metadata to a
    new dedicated run directory. Do not add dashboards, maps, SHAP, or extra
    explanation figures.

## Acceptance Criteria

- [ ] Every XGBoost fold uses the exact v5 scope snapshot, 36-month temporal
      training boundary, 88-feature order, and an untouched held-out target month.
- [ ] Raw phase labels join one-to-one; 2021-06 is retained only as suppressed
      audit output; all other evaluation folds contain phases 1-4.
- [ ] Tuning is reproducible, training-only, and small enough to audit directly.
- [ ] Multiclass predictions are valid phases 1-4 and exported with four class
      probabilities summing to one.
- [ ] GeoRF/XGBoost/FEWS NET comparisons use a documented common-support and
      task-compatible metric contract; unsupported comparisons are not implied.
- [ ] Existing runs and production files remain unchanged by hash.

## Out of Scope

- Reintroducing GeoXGB, adding local spatial partitions, broad hyperparameter
  search, Optuna, calibration, ensembling, or changing the v5 feature contract.
