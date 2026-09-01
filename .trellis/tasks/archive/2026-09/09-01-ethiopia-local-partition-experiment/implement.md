# Implementation Plan

1. **Impact and regression baseline**
   - Run GitNexus upstream impact for every shared symbol to be edited.
   - Record clean Git status and current default CLI dry-run/output behavior.

2. **Cohort and strict feature contract**
   - Add focused tests for exact Ethiopia filtering, source hashes, key equality,
     duplicate/null rejection, and strict lag-only column selection.
   - Add the default-off strict feature hook and optional Stage 1 data path.
   - Verify production defaults produce the original feature-column contract.

3. **Experiment entrypoint and Stage 1**
   - Create the isolated run directory and Ethiopia panel/manifest.
   - Execute 36 Stage 1 runs for fs0-fs3, 2018-2020, February/June/October only.
     Do not fill missing target months.
   - Validate plan counts, filenames, cohort codes, and lag provenance.

4. **Shared Stage 2 consensus**
   - Feed all four-scope Stage 1 plans into existing step1/3/4/5/6 scripts.
   - Produce and validate general/m2/m6/m10 mappings and manifest using spectral
     seed 42.

5. **Matched Stage 3 validation**
   - Add symmetric pooled/partitioned validation-threshold selection behind the
     experiment flag.
   - Run fs0-fs3 on the same cohort, mappings, folds, model seed 5, and strict
     feature policy.
   - Export predictions, monthly metrics, fixed-0.5 diagnostics, and thresholds.

6. **Corrected FEWS NET baseline**
   - Filter exact `country == "Ethiopia"`; assert the 1,040-code set matches.
   - Calendar-join near(T-4)/medium(T-8), preserve missing values, intersect model
     keys, enforce 90% coverage, and add metrics/coverage status to the monthly
     metrics rows.

7. **Aggregation and figure**
   - Independently recompute monthly metrics from predictions.
   - Generate the confirmed 4x3 PNG and run manifest with source/config hashes.

8. **Quality gate**
   - Run focused unit tests, production-default regression checks, full artifact
     schema/key/count/hash validation, and `git diff --check`.
   - Run GitNexus `detect_changes()` against `main`; verify only approved symbols
     and flows changed.
   - Report exact commands, pass counts, output path, unavailable FEWS NET points,
     and any incomplete runtime checks.

## Rollback

Shared hooks are default-off and may be reverted without touching experiment
outputs. Never delete or overwrite prior run directories; create a new run ID.
