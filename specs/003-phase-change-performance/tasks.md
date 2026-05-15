# Tasks: Phase-Change Monthly Performance

**Input**: Design documents from `/specs/003-phase-change-performance/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/phase-change-analysis-cli.md, quickstart.md

**Tests**: No separate automated test suite was requested. Validation tasks use the required smoke-test, dry-run, and manual artifact checks from quickstart.md.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Every task includes exact file paths

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare the dedicated exploratory analysis entry point without touching existing standard outputs.

- [X] T001 Create `scripts/plot_phase_change_monthly_performance.py` with argparse options from `specs/003-phase-change-performance/contracts/phase-change-analysis-cli.md`
- [X] T002 Define constants for model tokens, scopes, required columns, metric names, output filenames, and default paths in `scripts/plot_phase_change_monthly_performance.py`
- [X] T003 Add path resolution helpers for relative, absolute, and Windows-style paths in `scripts/plot_phase_change_monthly_performance.py`
- [X] T004 Add output-directory guard that rejects `main_ablation_results/march2026_main_backup_month_ind_cont3/monthly_performance_plots` in `scripts/plot_phase_change_monthly_performance.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build shared source discovery, validation, and metric primitives that all user stories depend on.

**CRITICAL**: No user story work can begin until this phase is complete.

- [X] T005 Implement GeoRF/GeoDT source discovery for `main_ablation_results/march2026_main_backup_month_ind_cont3/result_partition_k40_compare_{GF,DT}_fs*/predictions_monthly.csv` in `scripts/plot_phase_change_monthly_performance.py`
- [X] T006 Implement XGB/GeoXGB exclusion discovery for `main_ablation_results/march2026_main_backup_month_ind_cont3/result_partition_k40_compare_XGB_fs*/predictions_monthly.csv` in `scripts/plot_phase_change_monthly_performance.py`
- [X] T007 Implement required-column validation for `FEWSNET_admin_code`, `month_start`, `y_true`, `y_pred_pooled`, and `y_pred_partitioned` in `scripts/plot_phase_change_monthly_performance.py`
- [X] T008 Implement binary-label and date parsing validation for source prediction rows in `scripts/plot_phase_change_monthly_performance.py`
- [X] T009 Implement precision, recall, and F1 helper calculations with blank/NA output for zero-denominator cases in `scripts/plot_phase_change_monthly_performance.py`
- [X] T010 Implement manifest data structure for workflow mode, source files, excluded files, column contract, row counts, duplicate flags, zero-denominator metrics, and generated artifacts in `scripts/plot_phase_change_monthly_performance.py`
- [X] T011 Implement dry-run mode that validates discovery, exclusions, source columns, output path, and planned artifacts without writing files in `scripts/plot_phase_change_monthly_performance.py`

**Checkpoint**: Foundation ready; source files can be discovered and validated without generating final artifacts.

---

## Phase 3: User Story 1 - Identify phase-change prediction rows (Priority: P1) MVP

**Goal**: Filter row-level prediction outputs to rows whose true crisis indicator changed from the previous available test month for the same spatial unit, model, and scope.

**Independent Test**: Run the smoke path on GeoRF fs1 and verify first-observation exclusion, before/after row counts, XGB exclusion, and retained rows where `y_true != previous_y_true`.

### Implementation for User Story 1

- [X] T012 [US1] Implement source loading that attaches `model_key`, `model_label`, `source_token`, `scope`, and `source_file` to each row in `scripts/plot_phase_change_monthly_performance.py`
- [X] T013 [US1] Implement duplicate `(model_key, scope, FEWSNET_admin_code, month_start)` detection and manifest reporting in `scripts/plot_phase_change_monthly_performance.py`
- [X] T014 [US1] Implement chronological sorting by `month_start` within each `(model_key, scope, FEWSNET_admin_code)` series in `scripts/plot_phase_change_monthly_performance.py`
- [X] T015 [US1] Implement previous available test-month comparison fields `previous_month_start`, `previous_y_true`, `is_first_observation`, and `phase_change` in `scripts/plot_phase_change_monthly_performance.py`
- [X] T016 [US1] Implement phase-change row filtering that excludes first observations and retains only rows where `y_true != previous_y_true` in `scripts/plot_phase_change_monthly_performance.py`
- [X] T017 [US1] Add before-filter, first-observation-excluded, non-change-excluded, and after-filter row counts by model and scope to `phase_change_manifest.json` generation in `scripts/plot_phase_change_monthly_performance.py`
- [X] T018 [US1] Implement `--model georf --scope fs1 --smoke` validation output for first-row exclusion and phase-change row counts in `scripts/plot_phase_change_monthly_performance.py`

**Checkpoint**: User Story 1 is independently functional and can prove the row-level phase-change filter is correct.

---

## Phase 4: User Story 2 - Recompute phase-change monthly metrics (Priority: P2)

**Goal**: Recompute monthly and summary precision, recall, and F1 from phase-change rows only, preserving fs1-fs3 and pooled versus partitioned result series.

**Independent Test**: Use a filtered month/scope/model subset and confirm the reported precision, recall, and F1 match manual calculations from `y_true`, `y_pred_pooled`, and `y_pred_partitioned`.

### Implementation for User Story 2

- [X] T019 [US2] Implement monthly metric aggregation by `model_key`, `model_label`, `scope`, `month_start`, and series in `scripts/plot_phase_change_monthly_performance.py`
- [X] T020 [US2] Implement pooled and partitioned series expansion from `y_pred_pooled` and `y_pred_partitioned` in `scripts/plot_phase_change_monthly_performance.py`
- [X] T021 [US2] Record true-positive, false-positive, false-negative, support, and phase-change row counts for each monthly metric row in `scripts/plot_phase_change_monthly_performance.py`
- [X] T022 [US2] Record blank/NA precision, recall, or F1 values and zero-denominator reasons in `phase_change_manifest.json` from `scripts/plot_phase_change_monthly_performance.py`
- [X] T023 [US2] Implement summary-table metric aggregation from all filtered phase-change rows pooled across months for each model, scope, and series in `scripts/plot_phase_change_monthly_performance.py`
- [X] T024 [US2] Write `metrics_monthly_phase_change.csv` and `summary_phase_change.xlsx` under `main_ablation_results/march2026_main_backup_month_ind_cont3/phase_change_monthly_performance/` from `scripts/plot_phase_change_monthly_performance.py`
- [X] T025 [US2] Extend `--smoke` output to include one draft summary row and recomputed pooled and partitioned precision, recall, and F1 in `scripts/plot_phase_change_monthly_performance.py`

**Checkpoint**: User Story 2 is independently functional and can prove metrics are recomputed from phase-change rows rather than existing aggregate metrics.

---

## Phase 5: User Story 3 - Produce labeled exploratory outputs (Priority: P3)

**Goal**: Generate clearly labeled phase-change-only plots, tables, audit files, and manifest in a new output directory without overwriting standard diagnostics.

**Independent Test**: Run the full command and verify outputs are in `phase_change_monthly_performance/`, plots and tables are labeled phase-change-only, XGB/FEWSNET are absent from result series, and the manifest documents provenance and row counts.

### Implementation for User Story 3

- [X] T026 [US3] Implement GeoRF and GeoDT phase-change plot rendering using the existing fs1-fs3 by precision/recall/F1 layout in `scripts/plot_phase_change_monthly_performance.py`
- [X] T027 [US3] Ensure plot titles, legends, axis labels, and filenames identify phase-change-only analysis in `scripts/plot_phase_change_monthly_performance.py`
- [X] T028 [US3] Exclude FEWSNET and XGB series from phase-change plots and summary outputs in `scripts/plot_phase_change_monthly_performance.py`
- [X] T029 [US3] Write `filtered_predictions_phase_change.csv` audit output with previous true-value fields under `main_ablation_results/march2026_main_backup_month_ind_cont3/phase_change_monthly_performance/` from `scripts/plot_phase_change_monthly_performance.py`
- [X] T030 [US3] Honor `--write-audit` behavior so `filtered_predictions_phase_change.csv` is written when audit output is enabled and skipped when disabled in `scripts/plot_phase_change_monthly_performance.py`
- [X] T031 [US3] Write final `phase_change_manifest.json` with included sources, excluded XGB sources, filter definition, row counts, zero-denominator metrics, duplicate flags, and generated artifacts in `scripts/plot_phase_change_monthly_performance.py`
- [X] T032 [US3] Ensure full mode writes only to `main_ablation_results/march2026_main_backup_month_ind_cont3/phase_change_monthly_performance/` and never overwrites `main_ablation_results/march2026_main_backup_month_ind_cont3/monthly_performance_plots/` in `scripts/plot_phase_change_monthly_performance.py`
- [X] T033 [US3] Add concise ASCII-safe console summary for dry-run, smoke, and full modes in `scripts/plot_phase_change_monthly_performance.py`
- [X] T034 [US3] Implement smoke-mode reduced plot output for `python3 scripts/plot_phase_change_monthly_performance.py --model georf --scope fs1 --smoke` in `scripts/plot_phase_change_monthly_performance.py`

**Checkpoint**: User Story 3 is independently functional and produces traceable exploratory artifacts without touching standard outputs.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Validate the complete workflow, update usage guidance, and confirm artifact hygiene.

- [X] T035 Run dry-run validation with `python3 scripts/plot_phase_change_monthly_performance.py --dry-run`, inspect console/manifest preview, and verify no generated files are written under `main_ablation_results/march2026_main_backup_month_ind_cont3/phase_change_monthly_performance/`
- [X] T036 Run smoke validation with `python3 scripts/plot_phase_change_monthly_performance.py --model georf --scope fs1 --smoke` and verify evidence against `specs/003-phase-change-performance/quickstart.md`
- [X] T037 Capture pre-run timestamps or checksums for files in `main_ablation_results/march2026_main_backup_month_ind_cont3/monthly_performance_plots/` before full exploratory analysis
- [X] T038 Run full exploratory analysis with `python3 scripts/plot_phase_change_monthly_performance.py` and generate artifacts under `main_ablation_results/march2026_main_backup_month_ind_cont3/phase_change_monthly_performance/`
- [X] T039 Verify `main_ablation_results/march2026_main_backup_month_ind_cont3/monthly_performance_plots/` was not modified by comparing against the T037 pre-run timestamps or checksums
- [X] T040 Verify `phase_change_manifest.json` records six included GF/DT source files, XGB exclusions, row counts, zero-denominator metrics, duplicate flags, and generated artifact paths in `main_ablation_results/march2026_main_backup_month_ind_cont3/phase_change_monthly_performance/phase_change_manifest.json`
- [X] T041 Verify `filtered_predictions_phase_change.csv` rows satisfy `y_true != previous_y_true` and `is_first_observation` is false in `main_ablation_results/march2026_main_backup_month_ind_cont3/phase_change_monthly_performance/filtered_predictions_phase_change.csv`
- [X] T042 Verify `summary_phase_change.xlsx` contains only GeoRF(phase change) and GeoDT(phase change) rows for fs1, fs2, and fs3 in `main_ablation_results/march2026_main_backup_month_ind_cont3/phase_change_monthly_performance/summary_phase_change.xlsx`
- [X] T043 Verify `georf_phase_change_monthly_performance.png` and `geodt_phase_change_monthly_performance.png` are clearly labeled phase-change-only in `main_ablation_results/march2026_main_backup_month_ind_cont3/phase_change_monthly_performance/`
- [X] T044 Update `specs/003-phase-change-performance/quickstart.md` with any final command/output name adjustments discovered during validation
- [X] T045 Review `git status --short` output to confirm generated phase-change artifacts under `main_ablation_results/march2026_main_backup_month_ind_cont3/phase_change_monthly_performance/` are not accidentally staged unless explicitly promoted

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies; can start immediately.
- **Foundational (Phase 2)**: Depends on Setup completion; blocks all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational completion; MVP phase-change filter.
- **User Story 2 (Phase 4)**: Depends on User Story 1 filtered rows.
- **User Story 3 (Phase 5)**: Depends on User Story 2 metrics for final plots/tables.
- **Polish (Phase 6)**: Depends on all selected user stories being complete.

### User Story Dependencies

- **User Story 1 (P1)**: No dependency on other stories after Phase 2; delivers MVP row filtering.
- **User Story 2 (P2)**: Depends on User Story 1 because metrics require filtered rows.
- **User Story 3 (P3)**: Depends on User Story 2 because final artifacts require recomputed metrics.

### Within Each User Story

- Validate source contracts before filtering.
- Filter row-level predictions before recomputing metrics.
- Recompute metrics before rendering plots or summary tables.
- Run smoke/dry-run validation before full artifact generation.
- Verify artifact hygiene before considering the feature complete.

## Parallel Execution Examples

No setup or story tasks are currently marked `[P]` because the implementation is intentionally concentrated in `scripts/plot_phase_change_monthly_performance.py`; splitting these tasks across agents would create same-file edit conflicts.

### Validation parallel checks after full generation

```text
Task: "Verify phase_change_manifest.json records included and excluded sources in main_ablation_results/march2026_main_backup_month_ind_cont3/phase_change_monthly_performance/phase_change_manifest.json"
Task: "Verify summary_phase_change.xlsx contains only GeoRF(phase change) and GeoDT(phase change) rows in main_ablation_results/march2026_main_backup_month_ind_cont3/phase_change_monthly_performance/summary_phase_change.xlsx"
Task: "Verify georf_phase_change_monthly_performance.png and geodt_phase_change_monthly_performance.png are clearly labeled phase-change-only in main_ablation_results/march2026_main_backup_month_ind_cont3/phase_change_monthly_performance/"
```

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 setup for `scripts/plot_phase_change_monthly_performance.py`.
2. Complete Phase 2 source discovery, validation, metric helper, manifest skeleton, and dry-run support.
3. Complete Phase 3 row-level phase-change filtering.
4. Stop and validate with `python3 scripts/plot_phase_change_monthly_performance.py --model georf --scope fs1 --smoke`.

### Incremental Delivery

1. Deliver US1 row filtering and row-count evidence.
2. Add US2 recomputed monthly and summary metrics.
3. Add US3 final plots, audit files, manifest, and output isolation.
4. Complete Phase 6 validation against `specs/003-phase-change-performance/quickstart.md`.

### Artifact Hygiene

- Keep generated artifacts under `main_ablation_results/march2026_main_backup_month_ind_cont3/phase_change_monthly_performance/`.
- Do not overwrite `main_ablation_results/march2026_main_backup_month_ind_cont3/monthly_performance_plots/`.
- Do not stage generated CSV, XLSX, PNG, or JSON artifacts unless the user explicitly promotes them as deliverables.
