# Implementation

1. Run GitNexus upstream impact analysis for every existing symbol to be edited;
   stop and report any HIGH or CRITICAL result.
2. Add focused tests for qualifying release months, strict-before-origin phase
   histories, 88-column ordering, unchanged canonical aligned files, and
   train-only imputation including an all-null column.
3. Add the smallest run-local input builder to the Ethiopia runner. Validate and
   hash the four 88-predictor snapshots before launching any fit.
4. Add Ethiopia-specific Stage 1 and Stage 3 entrypoints derived from the current
   GeoRF logic. Load aligned arrays directly, apply fold-local imputation before
   SMOTE, and preserve the frozen split/seed/threshold behavior.
5. Point the Ethiopia runner at the dedicated Stage 1/3 entrypoints and the
   scope-specific snapshots. Leave Stage 2 command construction unchanged.
6. Suppress 2021-06 metrics and plot points while retaining its fold/audit
   records. Preserve the existing fs1/fs2 lagged FEWS NET comparison.
7. Run focused tests and a bounded smoke run before the full 36-plan/48-fold
   experiment. Verify manifests, hashes, feature names, partition coverage,
   thresholds, metric recomputation, and protected-artifact hashes.
8. Run the full experiment once with a new run ID. Keep generated inputs and
   outputs uncommitted unless separately requested.
9. Before any commit, run GitNexus `detect_changes()` and the final Trellis
   quality gate.

## Execution evidence

- Successful run: `eth_aligned_refit_20260904_seed5_v5`.
- Completed 36 Stage 1 plans, four Stage 2 mappings, and 48 Stage 3 folds.
- Exported 144 monthly metrics, 96 fixed-0.5 diagnostics, and 48 thresholds.
- Independently recomputed all 24 fs1/fs2 FEWS NET rows with zero differences.
- Verified 12 suppressed 2021-06 rows and no suppressed plot points.
- Refreshed only the v5 figure title/legend layout; the run manifest records the
  plot-only refresh and figure SHA-256 while reusing unchanged model outputs.
- Compared monthly macro means against final 2026-09-01 v5 on common valid
  target months; 2026-09-01 v4 is byte-identical for the compared artifacts.
- Focused suite: 25 tests passed; all changed Python files passed `py_compile`;
  `git diff --check` passed; GitNexus comparison risk was low.

## Rollback

Delete only the new run directory after validating its exact path. Code rollback
is limited to the Ethiopia-specific runner/entrypoints/tests; production Stage
1/2/3 files and canonical aligned artifacts must remain unchanged.
