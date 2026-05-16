# Research: Phase-Change Monthly Performance

## Decision: Use a dedicated phase-change analysis script

**Rationale**: The existing `scripts/plot_monthly_performance_metrics.py` reads already aggregated monthly metrics and includes FEWSNET baseline series. The phase-change feature must read row-level predictions first, exclude XGB and FEWSNET from the summary, and avoid overwriting standard monthly plots. A dedicated script keeps the exploratory contract isolated while reusing the existing script's layout conventions.

**Alternatives considered**:

- Extend the existing monthly plotter with a phase-change mode: rejected because it would mix row-level and aggregate-metric inputs in one standard diagnostics script.
- Add a Windows batch launcher: rejected because the feature is exploratory and no full batch workflow is required.

## Decision: Explicitly include GF/GeoRF and DT/GeoDT fs1-fs3 prediction folders

**Rationale**: The inspected source root contains `result_partition_k40_compare_{DT,GF,XGB}_fs{1,2,3}/predictions_monthly.csv`. The spec requires GeoRF and GeoDT only. Explicit model-token mapping reduces accidental inclusion of XGB and gives predictable provenance.

**Alternatives considered**:

- Glob all `predictions_monthly.csv` files and filter later: rejected because it increases the chance of XGB leakage into metric outputs.
- Require manual file paths only: rejected because the folder naming is already consistent and discoverable.

## Decision: Use previous available test month for phase-change chronology

**Rationale**: The user specified previous available test month, and the inspected monthly outputs contain February, June, and October test months rather than every calendar month. Strict previous calendar month would classify most rows as lacking a predecessor.

**Alternatives considered**:

- Strict previous calendar month: rejected by the feature definition and source cadence.
- Previous row regardless of sort order: rejected because chronological ordering by `month_start` is required.

## Decision: Treat undefined precision, recall, and F1 as blank/NA

**Rationale**: Undefined metrics caused by zero denominators should not be coerced into false zero or perfect scores. Blank/NA values preserve mathematical meaning, and the manifest records the zero-denominator reason for auditability.

**Alternatives considered**:

- Record undefined values as 0.0: rejected because it conflates undefined with poor performance.
- Drop the affected month/scope rows: rejected because missingness itself is useful diagnostic evidence.

## Decision: Summary table aggregates by pooling all filtered phase-change rows across months

**Rationale**: The summary table should behave like an overall ablation-style report for each model, scope, and series, while monthly plots show month-by-month variation. Recomputing from pooled filtered rows avoids averaging instability when some months have very small phase-change counts.

**Alternatives considered**:

- Simple average of monthly metrics: rejected because small-sample months would carry the same weight as larger months.
- Report both pooled and average-monthly metrics: rejected for v1 because the requested table should stay focused and avoid unrelated baseline-style rows.

## Decision: Write generated artifacts under `phase_change_monthly_performance/`

**Rationale**: A new clearly labeled directory under the existing ablation root satisfies the no-overwrite requirement and keeps exploratory outputs near their source data.

**Alternatives considered**:

- Reuse `monthly_performance_plots/`: rejected because it would risk overwriting or confusing standard diagnostics.
- Write under `deliverables/`: rejected because the feature is exploratory unless explicitly promoted.

## Decision: Produce audit CSVs plus a JSON manifest

**Rationale**: Row-count provenance and filtered-row auditability are central success criteria. A filtered row-level CSV and recomputed monthly metrics CSV make smoke testing and reviewer validation straightforward; a JSON manifest captures included/excluded files, row counts, zero-denominator points, and generated artifacts.

**Alternatives considered**:

- Manifest only: rejected because reviewers may need row-level evidence for phase-change filtering.
- README only: rejected because machine-readable row counts and missing-metric details are useful for tests.
