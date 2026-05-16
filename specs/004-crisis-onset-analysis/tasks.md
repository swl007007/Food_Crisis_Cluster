# Tasks: Crisis Onset Analysis

**Input**: Design documents from `/specs/004-crisis-onset-analysis/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/crisis-onset-analysis-cli.md, quickstart.md

**Tests**: No separate automated test suite was requested. Validation tasks use the required dry-run, smoke-test, and manual artifact checks from quickstart.md.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete same-file edits)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Every task includes exact file paths

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare the existing phase-change analysis entry point for a lightweight mode extension without creating a duplicate script.

- [X] T001 Review current constants and output naming in `scripts/plot_phase_change_monthly_performance.py`
- [X] T002 Add filter-mode and output-profile constants for `any_phase_change` and `crisis_onset` in `scripts/plot_phase_change_monthly_performance.py`
- [X] T003 Update argparse to accept `--filter-mode {any_phase_change,crisis_onset}` with default `any_phase_change` in `scripts/plot_phase_change_monthly_performance.py`
- [X] T004 Update mode-specific default output directory resolution so `crisis_onset` defaults to `main_ablation_results/march2026_main_backup_month_ind_cont3/crisis_onset_analysis` in `scripts/plot_phase_change_monthly_performance.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Generalize existing phase-change primitives so all user stories can reuse one filter-mode-aware workflow.

**CRITICAL**: No user story work can begin until this phase is complete.

- [X] T005 Generalize output directory validation to reject `monthly_performance_plots` for all modes and reject `phase_change_monthly_performance` when `--filter-mode crisis_onset` in `scripts/plot_phase_change_monthly_performance.py`
- [X] T006 Preserve existing GeoRF/GeoDT source discovery and XGB exclusion behavior for both filter modes in `scripts/plot_phase_change_monthly_performance.py`
- [X] T007 Preserve required-column, binary-label, date parsing, and duplicate detection validation for both filter modes in `scripts/plot_phase_change_monthly_performance.py`
- [X] T008 Rename or wrap phase-change-specific helper names so shared code operates on retained rows for the active filter mode in `scripts/plot_phase_change_monthly_performance.py`
- [X] T009 Generalize `OUTPUT_FILES`, plot filename templates, smoke filename templates, and workbook sheet names by filter mode in `scripts/plot_phase_change_monthly_performance.py`
- [X] T010 Generalize manifest construction to include `filter_mode`, mode-specific filter definition, mode-specific row-count labels, output profile, and generated artifact names in `scripts/plot_phase_change_monthly_performance.py`
- [X] T011 Keep `any_phase_change` behavior backward compatible by ensuring its default directory, file names, labels, and retained-row definition remain equivalent to the previous phase-change workflow in `scripts/plot_phase_change_monthly_performance.py`

**Checkpoint**: Foundation is ready; the existing broader phase-change workflow can still run while crisis-onset-specific behavior has a clean insertion point.

---

## Phase 3: User Story 1 - Isolate crisis-onset rows (Priority: P1) MVP

**Goal**: Filter row-level prediction outputs to rows where the true crisis label changes from non-crisis to crisis (`0 -> 1`) for the same spatial unit, model, and scope.

**Independent Test**: Run the smoke path on GeoRF fs1 and verify first-observation exclusion, retained `0 -> 1` rows, excluded `1 -> 0` rows when present, before/after row counts, and XGB exclusion.

### Implementation for User Story 1

- [X] T012 [US1] Keep previous available test-month fields `previous_month_start`, `previous_y_true`, and `is_first_observation` in `scripts/plot_phase_change_monthly_performance.py`
- [X] T013 [US1] Add explicit `any_phase_change` and `crisis_onset` boolean fields after chronological sorting in `scripts/plot_phase_change_monthly_performance.py`
- [X] T014 [US1] Implement retained-row selection for `--filter-mode crisis_onset` using `previous_y_true == 0` and `y_true == 1` in `scripts/plot_phase_change_monthly_performance.py`
- [X] T015 [US1] Ensure retained crisis-onset rows never include first observations or `1 -> 0` crisis-recovery rows in `scripts/plot_phase_change_monthly_performance.py`
- [X] T016 [US1] Add `filter_mode` and `retained_by_filter` fields to retained rows for auditability in `scripts/plot_phase_change_monthly_performance.py`
- [X] T017 [US1] Update row-count generation to report before-filter rows, first-observation exclusions, non-onset exclusions, and retained rows by model and scope for crisis-onset mode in `scripts/plot_phase_change_monthly_performance.py`
- [X] T018 [US1] Update no-retained-month detection to report months with no `crisis_onset` rows instead of only no phase-change rows in `scripts/plot_phase_change_monthly_performance.py`
- [X] T019 [US1] Extend smoke-mode console evidence to report retained `0 -> 1` counts and excluded `1 -> 0` counts for `--filter-mode crisis_onset --model georf --scope fs1 --smoke` in `scripts/plot_phase_change_monthly_performance.py`

**Checkpoint**: User Story 1 is independently functional and can prove the crisis-onset filter is correct from row-level predictions.

---

## Phase 4: User Story 2 - Recompute onset-only metrics (Priority: P2)

**Goal**: Recompute monthly and summary precision, recall, and F1 from crisis-onset rows only, preserving fs1-fs3 and pooled versus partitioned result series.

**Independent Test**: Use a retained month/scope/model subset and confirm reported precision, recall, and F1 match manual calculations from `y_true`, `y_pred_pooled`, and `y_pred_partitioned`.

### Implementation for User Story 2

- [X] T020 [US2] Update monthly metric aggregation to use retained rows for the active filter mode and include `filter_mode` in `scripts/plot_phase_change_monthly_performance.py`
- [X] T021 [US2] Rename metric row-count output from phase-change-specific wording to retained-row wording where needed in `scripts/plot_phase_change_monthly_performance.py`
- [X] T022 [US2] Preserve separate pooled and partitioned series calculations from `y_pred_pooled` and `y_pred_partitioned` in `scripts/plot_phase_change_monthly_performance.py`
- [X] T023 [US2] Update summary metric aggregation so `crisis_onset` summaries pool all retained onset rows across months by model, scope, and series in `scripts/plot_phase_change_monthly_performance.py`
- [X] T024 [US2] Set crisis-onset summary labels to `GeoRF(crisis onset)` and `GeoDT(crisis onset)` in `scripts/plot_phase_change_monthly_performance.py`
- [X] T025 [US2] Preserve blank/NA undefined precision, recall, and F1 handling with zero-denominator provenance for crisis-onset mode in `scripts/plot_phase_change_monthly_performance.py`
- [X] T026 [US2] Ensure crisis-onset metric generation does not read or reuse `metrics_monthly.csv`, `metrics_monthly_phase_change.csv`, or any precomputed summary workbook in `scripts/plot_phase_change_monthly_performance.py`
- [X] T027 [US2] Update smoke-mode draft summary output to identify `filter_mode=crisis_onset` and show recomputed pooled and partitioned metrics in `scripts/plot_phase_change_monthly_performance.py`

**Checkpoint**: User Story 2 is independently functional and can prove metrics are recomputed from retained crisis-onset rows rather than existing aggregate metrics.

---

## Phase 5: User Story 3 - Produce separated onset diagnostics (Priority: P3)

**Goal**: Generate clearly labeled crisis-onset-only plots, tables, audit files, and manifest in `crisis_onset_analysis/` without overwriting standard or broader phase-change outputs.

**Independent Test**: Run the full crisis-onset command and verify outputs are in `crisis_onset_analysis/`, labels state crisis-onset-only, XGB/FEWSNET are absent, and standard plus broader phase-change output directories are unchanged.

### Implementation for User Story 3

- [X] T028 [US3] Update plot rendering to use mode-specific titles, empty-state text, legend labels, and filenames for crisis-onset-only outputs in `scripts/plot_phase_change_monthly_performance.py`
- [X] T029 [US3] Ensure crisis-onset plots preserve the existing GeoRF/GeoDT colors and fs1-fs3 by precision/recall/F1 layout where practical in `scripts/plot_phase_change_monthly_performance.py`
- [X] T030 [US3] Write `metrics_monthly_crisis_onset.csv` to `main_ablation_results/march2026_main_backup_month_ind_cont3/crisis_onset_analysis/` in `scripts/plot_phase_change_monthly_performance.py`
- [X] T031 [US3] Write `summary_crisis_onset.xlsx` with a crisis-onset-specific sheet name to `main_ablation_results/march2026_main_backup_month_ind_cont3/crisis_onset_analysis/` in `scripts/plot_phase_change_monthly_performance.py`
- [X] T032 [US3] Write `filtered_predictions_crisis_onset.csv` with previous-value fields, `filter_mode`, and `retained_by_filter` to `main_ablation_results/march2026_main_backup_month_ind_cont3/crisis_onset_analysis/` when audit output is enabled in `scripts/plot_phase_change_monthly_performance.py`
- [X] T033 [US3] Write `crisis_onset_manifest.json` with included sources, excluded XGB sources, filter definition, row counts, zero-denominator metrics, duplicate flags, and generated artifact paths in `scripts/plot_phase_change_monthly_performance.py`
- [X] T034 [US3] Ensure full crisis-onset mode writes no files under `main_ablation_results/march2026_main_backup_month_ind_cont3/monthly_performance_plots/` in `scripts/plot_phase_change_monthly_performance.py`
- [X] T035 [US3] Ensure full crisis-onset mode writes no files under `main_ablation_results/march2026_main_backup_month_ind_cont3/phase_change_monthly_performance/` in `scripts/plot_phase_change_monthly_performance.py`
- [X] T036 [US3] Update ASCII-safe console summary text to identify the active filter mode and generated crisis-onset artifacts in `scripts/plot_phase_change_monthly_performance.py`

**Checkpoint**: User Story 3 is independently functional and produces traceable crisis-onset artifacts without touching standard or broader phase-change outputs.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Validate the complete workflow, confirm artifact hygiene, and align usage guidance.

- [X] T037 Run dry-run validation with `python3 scripts/plot_phase_change_monthly_performance.py --filter-mode crisis_onset --dry-run` and verify planned outputs in `specs/004-crisis-onset-analysis/quickstart.md`
- [X] T038 Run smoke validation with `python3 scripts/plot_phase_change_monthly_performance.py --filter-mode crisis_onset --model georf --scope fs1 --smoke` and verify evidence against `specs/004-crisis-onset-analysis/quickstart.md`
- [X] T039 Capture pre-run timestamps or checksums for files under `main_ablation_results/march2026_main_backup_month_ind_cont3/monthly_performance_plots/` before full crisis-onset generation
- [X] T040 Capture pre-run timestamps or checksums for files under `main_ablation_results/march2026_main_backup_month_ind_cont3/phase_change_monthly_performance/` before full crisis-onset generation
- [X] T041 Run full exploratory analysis with `python3 scripts/plot_phase_change_monthly_performance.py --filter-mode crisis_onset` and generate artifacts under `main_ablation_results/march2026_main_backup_month_ind_cont3/crisis_onset_analysis/`
- [X] T042 Verify `main_ablation_results/march2026_main_backup_month_ind_cont3/monthly_performance_plots/` was not modified by comparing against T039 timestamps or checksums
- [X] T043 Verify `main_ablation_results/march2026_main_backup_month_ind_cont3/phase_change_monthly_performance/` was not modified by comparing against T040 timestamps or checksums
- [X] T044 Verify `main_ablation_results/march2026_main_backup_month_ind_cont3/crisis_onset_analysis/filtered_predictions_crisis_onset.csv` rows satisfy `previous_y_true = 0`, `y_true = 1`, and `is_first_observation` is false
- [X] T045 Verify `main_ablation_results/march2026_main_backup_month_ind_cont3/crisis_onset_analysis/summary_crisis_onset.xlsx` contains only GeoRF(crisis onset) and GeoDT(crisis onset) rows for fs1, fs2, and fs3
- [X] T046 Verify `main_ablation_results/march2026_main_backup_month_ind_cont3/crisis_onset_analysis/crisis_onset_manifest.json` records six included GF/DT source files, XGB exclusions, row counts, zero-denominator metrics, duplicate flags, and generated artifact paths
- [X] T047 Verify `main_ablation_results/march2026_main_backup_month_ind_cont3/crisis_onset_analysis/georf_crisis_onset_monthly_performance.png` and `main_ablation_results/march2026_main_backup_month_ind_cont3/crisis_onset_analysis/geodt_crisis_onset_monthly_performance.png` are clearly labeled crisis-onset-only
- [X] T048 Run backward-compatibility smoke or dry-run for default `any_phase_change` mode with `python3 scripts/plot_phase_change_monthly_performance.py --dry-run` and confirm previous broader mode paths remain unchanged
- [X] T049 Update `specs/004-crisis-onset-analysis/quickstart.md` if final command names, filenames, or validation observations differ from the planned contract
- [X] T050 Review `git status --short` output to confirm generated crisis-onset artifacts under `main_ablation_results/march2026_main_backup_month_ind_cont3/crisis_onset_analysis/` are not accidentally staged unless explicitly promoted

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies; can start immediately.
- **Foundational (Phase 2)**: Depends on Setup completion; blocks all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational completion; delivers MVP crisis-onset row filtering.
- **User Story 2 (Phase 4)**: Depends on User Story 1 retained rows.
- **User Story 3 (Phase 5)**: Depends on User Story 2 metrics for final plots, tables, and manifest artifacts.
- **Polish (Phase 6)**: Depends on all selected user stories being complete.

### User Story Dependencies

- **User Story 1 (P1)**: No dependency on other stories after Phase 2; delivers MVP row filtering.
- **User Story 2 (P2)**: Depends on User Story 1 because metrics require retained crisis-onset rows.
- **User Story 3 (P3)**: Depends on User Story 2 because final artifacts require recomputed metrics.

### Within Each User Story

- Validate source contracts before filtering.
- Compute previous available true-label fields before applying filter modes.
- Filter row-level predictions before recomputing metrics.
- Recompute metrics before rendering plots or summary tables.
- Run smoke/dry-run validation before full artifact generation.
- Verify artifact hygiene before considering the feature complete.

## Parallel Execution Examples

No implementation tasks are currently marked `[P]` because the implementation is intentionally concentrated in `scripts/plot_phase_change_monthly_performance.py`; splitting those tasks across agents would create same-file edit conflicts.

### Validation parallel checks after full generation

```text
Task: "Verify crisis_onset_manifest.json records included and excluded sources in main_ablation_results/march2026_main_backup_month_ind_cont3/crisis_onset_analysis/crisis_onset_manifest.json"
Task: "Verify summary_crisis_onset.xlsx contains only GeoRF(crisis onset) and GeoDT(crisis onset) rows in main_ablation_results/march2026_main_backup_month_ind_cont3/crisis_onset_analysis/summary_crisis_onset.xlsx"
Task: "Verify georf_crisis_onset_monthly_performance.png and geodt_crisis_onset_monthly_performance.png are clearly labeled crisis-onset-only in main_ablation_results/march2026_main_backup_month_ind_cont3/crisis_onset_analysis/"
```

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup.
2. Complete Phase 2: Foundational mode generalization.
3. Complete Phase 3: User Story 1 crisis-onset row filtering.
4. Stop and validate with `python3 scripts/plot_phase_change_monthly_performance.py --filter-mode crisis_onset --model georf --scope fs1 --smoke`.

### Incremental Delivery

1. Setup + Foundational: mode-aware script remains backward compatible with `any_phase_change`.
2. User Story 1: crisis-onset rows can be identified and audited.
3. User Story 2: onset-only metrics and summary values can be recomputed.
4. User Story 3: final crisis-onset plots, tables, audit file, and manifest are generated in `crisis_onset_analysis/`.
5. Polish: dry-run, smoke, full generation, no-overwrite checks, and git hygiene validation complete.

## Notes

- `[P]` tasks are omitted because most implementation tasks edit the same script.
- Story labels map tasks to user stories for traceability.
- Generated outputs should remain unstaged unless explicitly promoted.
- Do not change model training, prediction generation, ACTIVE_LAGS, lag mapping, threshold semantics, partition maps, shapefile scope, notebooks, or batch launchers for this feature.
