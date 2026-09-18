# Implementation and experiment checklist

User authorized implementation and execution after design review. All experiment code, logs and results belong under `Step3ExpertCorrectionExperiment/`. Only Phase 3 may run; reuse the frozen Phase 2-derived month maps. No commits or replacement of original outputs.

1. Read PRD, design, research, and backend spec index. Capture baseline git state and protected artifact hashes. Confirm the exact main source, environment, saved comparison support, and original expert baselines.
2. Before editing symbols, run GitNexus upstream impact and report direct callers, affected processes, and risk. Inspect the complete affected functions and feature/preprocessing flow. Keep shared splitter behavior unchanged.
3. Reconstruct the original expert history with source-row provenance. Verify all archived fs1/fs2 quarterly baseline metrics using the existing country-performance verification pattern (without invoking its document-writing entrypoint). Validate source timing and main truth/keys; halt discrepancies.
4. Add an isolated fs1/fs2 correction runner under the new experiment directory, reusing existing RF construction, maps, metrics, and persistence. Preserve pooled/fs3 behavior. Isolate correction fitting from the existing SMOTE and pooled fallback paths; add no dependency, general estimator framework, or XGBoost migration.
5. Implement the exact outer/inner boundaries, expert-error target, abstention, candidate gates, deterministic selection, refit, and audit exports defined in design.md. Provide a run command in the experiment README without changing the production launcher. Keep Windows batch echo parentheses escaped.
6. Add one focused runnable test module covering record-shift versus calendar-shift semantics and missingness; exact temporal masks; wrong target/input; 50-row/single-class abstention; strict score threshold; each directional gate boundary; F1 ties/no correction; and recomputation of final predictions. A meaningful leakage check ensures changing withheld validation-gap/test labels cannot change the fitting stage or selected test rule respectively.
7. Verify pooled and fs3 unchanged predictions on bounded deterministic fixtures. Run a bounded real-data check, then the authorized full fs1/fs2 experiment, compare original pooled/fs3 arrays and metrics, and recompute correction metrics and audit fields from saved rows. Preserve every original artifact hash. Do not claim successful experiment verification from unit tests alone.
8. Document the isolated command, method labels, and paths in the experiment README; production launcher behavior remains unchanged. Final implementation review must check all affected paths. Run `git diff --check`, focused tests under the approved environment, and GitNexus detect_changes before any separately authorized commit.

## Rollback

Original branch behavior and artifacts remain available. New correction output is isolated. On failure stop the new run, retain its diagnostic evidence, and fix only the new path; no automatic deletions, overwrites, or rollback of unrelated work.
