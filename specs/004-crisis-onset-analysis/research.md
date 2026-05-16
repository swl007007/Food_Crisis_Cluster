# Research: Crisis Onset Analysis

## Decision: Extend the existing phase-change analysis script

**Rationale**: The previous feature already implemented the required row-level source discovery, GeoRF/GeoDT inclusion, XGB exclusion, data-contract validation, previous available test-month fields, metric recomputation, plotting layout, summary-table generation, manifest structure, dry-run behavior, and smoke-test path. Extending that script with a filter-mode/output-profile seam satisfies the lightweight follow-up requirement and avoids redesigning or duplicating logic.

**Alternatives considered**:

- Create a new `plot_crisis_onset_monthly_performance.py` script: rejected because it would duplicate discovery, validation, metric, plotting, and manifest logic that the user explicitly asked to reuse.
- Modify the standard monthly plotter: rejected because it reads aggregate metrics and standard outputs, while this feature must filter row-level predictions first.

## Decision: Add `crisis_onset` as a named filter mode alongside `any_phase_change`

**Rationale**: The prior phase-change feature's retained-row definition is the broader `any_phase_change` mode: current `y_true` differs from `previous_y_true`. The follow-up requires a narrower diagnostic mode: previous true crisis indicator equals 0 and current true crisis indicator equals 1. Naming both modes keeps the previous behavior explicit and prevents accidental replacement of broader phase-change outputs.

**Alternatives considered**:

- Replace the existing phase-change filter in place: rejected because the broader outputs must not be overwritten or semantically changed.
- Implement crisis-onset as a post-filter on generated phase-change artifacts: rejected because metrics must be recomputed from row-level predictions, not precomputed or broader filtered metrics.

## Decision: Use previous available test month for onset chronology

**Rationale**: The previous phase-change contract and this follow-up both depend on previous available test rows within a spatial unit/model/scope series. Existing monthly outputs may not represent every calendar month for every scope, so strict previous calendar month would incorrectly discard valid comparisons.

**Alternatives considered**:

- Strict previous calendar month: rejected because it conflicts with the established phase-change contract and source cadence.
- Previous row without chronological sorting: rejected because onset direction requires ordered `month_start` values.

## Decision: Write crisis-onset outputs under `crisis_onset_analysis/`

**Rationale**: The requested folder name keeps crisis-onset diagnostics separate from both standard `monthly_performance_plots/` and broader `phase_change_monthly_performance/` outputs. Mode-specific filenames and labels reduce reviewer confusion.

**Alternatives considered**:

- Reuse `phase_change_monthly_performance/`: rejected because it risks overwriting or confusing broader `any_phase_change` outputs.
- Write under `deliverables/`: rejected because the feature remains exploratory diagnostics unless explicitly promoted later.

## Decision: Preserve GeoRF/GeoDT-only inclusion and XGB exclusion

**Rationale**: The feature is a follow-up to the prior GeoRF/GeoDT-only analysis. Continuing the same model-family contract enables direct comparison between broader phase-change and crisis-onset diagnostics while avoiding accidental XGB leakage.

**Alternatives considered**:

- Include XGB for completeness: rejected because the user explicitly excluded XGB.
- Make model-family inclusion fully open-ended: rejected because the source contract and summary acceptance criteria are limited to GeoRF and GeoDT.

## Decision: Recompute metrics from retained row-level predictions

**Rationale**: Monthly precision, recall, and F1 must reflect only crisis-onset rows. Recomputing from `y_true`, `y_pred_pooled`, and `y_pred_partitioned` preserves the class-1 crisis metric contract and avoids averaging or filtering already-aggregated results.

**Alternatives considered**:

- Filter existing phase-change metrics: rejected because those metrics are already aggregated and cannot be narrowed to onset rows correctly.
- Average monthly metrics for the summary table: rejected because pooled row-level recomputation is already established by the prior feature and avoids small-month instability.

## Decision: Keep undefined metric handling as blank/NA plus provenance

**Rationale**: Crisis-onset subsets can be small. Undefined precision, recall, or F1 values caused by zero denominators should remain blank/NA and be documented in the manifest so reviewers can distinguish mathematical undefinedness from poor model performance.

**Alternatives considered**:

- Coerce undefined metrics to 0.0: rejected because it misrepresents zero-denominator cases.
- Drop months with undefined metrics: rejected because missingness and low support are diagnostic evidence.
