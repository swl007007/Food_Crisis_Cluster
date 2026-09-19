# Implementation

1. Audit and load the immutable v5 snapshots, phase truth, GeoRF outputs, and
   correctly lagged FEWS NET baselines; assert keys, feature order, and hashes.
2. For each scope and eligible fold, tune the eight candidates on the latest six
   training months, refit the selected candidate on all 36 training months, and
   export phase predictions and probabilities using seed 5 and native missing
   handling.
3. Build common-support four-class and collapsed-binary comparisons, write the
   minimal output bundle, render one compact figure, and verify metrics, support,
   no-leak boundaries, probability sums, and unchanged protected inputs.
