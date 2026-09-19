# Implementation

1. Add the isolated selective-correction runner and import the predecessor's
   stable calendar/split/coverage helpers; do not edit predecessor code.
2. Implement the unweighted wrong-call classifier, threshold search,
   direction-specific safety masks, expert-only candidate, deterministic
   ordering, refit, comparison, and five-file export.
3. Add focused tests for independent direction gates, the 75% and 20-row/two-
   month rules, strict-F1 fallback to expert-only, and deterministic ties.
4. Run the new focused tests with the Windows Python 3.12 environment used by
   the prior XGBoost experiments.
5. Execute the fixed seed-5 run
   `eth_fewsnet_selective_correction_xgb_20260904_seed5_v1` under
   `EthiopiaForecastingExperiment/outputs/local_partition_experiment/`.
6. Independently recompute the equal-month summary; verify source hashes,
   row/key counts, and unchanged predecessor artifacts.
7. Report the result even if it is expert-only everywhere or loses to expert on
   untouched test months. Do not commit without separate authorization.
