# Design

## Boundary

Add one isolated runner,
`EthiopiaForecastingExperiment/run_fewsnet_selective_correction_xgb.py`, and one
focused test file. Reuse the existing residual runner's calendar join, temporal
split, coverage, metrics, hashing, and run-directory helpers. Do not modify the
old runner or create shared abstractions for one new consumer.

## Data flow

1. Load and hash the frozen v5 snapshots, binary comparison, FEWS NET source,
   aligned-input contract, and imported predecessor runner.
2. Validate the same 88 predictors, fs0-fs3 horizons, admin-month keys, expert
   calendar mapping, truth equality, and common-support coverage.
3. Within each outer fold, fit each of the eight `XGBClassifier` candidates on
   `expert_wrong = (y != expert)` using the 88 predictors plus expert anchor.
   Use `binary:logistic`, the predecessor's fixed learning rate/subsampling,
   seed 5, native missing-value handling, and no sample weights.
4. Search rounded unique validation scores as correction thresholds. For each
   threshold, assess `0->1` and `1->0` separately; enable a direction only when
   it has at least 20 flips across at least two validation months and at least
   75% correction precision. A raw threshold below `0.5` is valid because the
   score is not declared calibrated.
5. After direction masks, retain a corrected candidate only when its final
   validation crisis F1 strictly exceeds expert-only on identical support.
   Otherwise select zero correction. Expert-only wins F1 ties; remaining ties
   use the fixed grid and threshold enumeration order.
6. Refit the selected classifier on fit plus validation rows, apply the frozen
   threshold and direction masks to test, and never revisit selection using test
   outcomes.
7. Export the five approved artifacts and print the equal-month scope summary.

## Audit contract

Predictions and tuning retain the expert call, wrong-score, threshold, enabled
directions, proposed/applied flip flags, and whether each applied flip is a fix
or damage. Monthly and summary outputs compare FEWS NET, GeoRF v5, plain binary
XGBoost, and selective correction on identical support. Metadata records source
hashes, candidate rules, fold choices, and protected-input equality.

## Rollback

The runner uses the fixed run ID
`eth_fewsnet_selective_correction_xgb_20260904_seed5_v1` and refuses an existing
directory. Failure leaves all frozen sources and predecessor outputs untouched;
delete only the new incomplete directory before an explicitly authorized rerun.
