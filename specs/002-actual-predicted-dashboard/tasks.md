# Tasks: Actual Predicted Dashboard

**Input**: Design documents from `/specs/002-actual-predicted-dashboard/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: The feature specification requests a smoke-test/dry-run validation path, not a formal TDD suite. Tasks therefore include validation and manual browser smoke checks rather than separate automated test-first tasks.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare the generator script location and establish constants that reflect the plan contract.

- [X] T001 Create `scripts/generate_actual_predicted_dashboard.py` with argparse entry point accepting `--input-dir`, `--shapefile`, and `--output-dir` defaults from `specs/002-actual-predicted-dashboard/quickstart.md`
- [X] T002 Add model/scope/source constants in `scripts/generate_actual_predicted_dashboard.py` for included tokens `GF` and `DT`, excluded token `XGB`, scopes `fs1`/`fs2`/`fs3`, required columns, and output filenames
- [X] T003 [P] Add HTML/CSS/JavaScript template constants in `scripts/generate_actual_predicted_dashboard.py` for two side-by-side panels, selectors, legend, alpha control, and no-data message regions

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Implement reusable data loading, validation, geometry preparation, and manifest helpers that all user stories need.

**CRITICAL**: No user story work can begin until this phase is complete.

- [X] T004 Implement result-folder discovery in `scripts/generate_actual_predicted_dashboard.py` that includes only `result_partition_k40_compare_GF_fs*` and `result_partition_k40_compare_DT_fs*` and records `result_partition_k40_compare_XGB_fs*` as excluded
- [X] T005 Implement prediction CSV loading and validation in `scripts/generate_actual_predicted_dashboard.py` for `FEWSNET_admin_code`, `month_start`, `y_true`, and `y_pred_partitioned` with chronological date normalization
- [X] T006 Implement binary-label validation in `scripts/generate_actual_predicted_dashboard.py` that rejects or records unexpected non-0/1 values without applying any threshold
- [X] T007 Implement global shapefile loading in `scripts/generate_actual_predicted_dashboard.py` requiring `FEWS_Admin_LZ_v3.shp`, normalizing accepted admin-code aliases to `FEWSNET_admin_code`, repairing invalid geometries when possible, and rejecting Nigeria-only shapefile paths
- [X] T008 Implement geometry simplification/projection-to-browser payload preparation in `scripts/generate_actual_predicted_dashboard.py` using the global FEWSNET polygons and preserving no-data polygons
- [X] T009 Implement availability index construction in `scripts/generate_actual_predicted_dashboard.py` keyed by model, scope, and normalized `month_start`
- [X] T010 Implement manifest assembly and JSON writing in `scripts/generate_actual_predicted_dashboard.py` matching `specs/002-actual-predicted-dashboard/contracts/manifest.schema.json`

**Checkpoint**: The generator can discover valid inputs, validate contracts, prepare geometry/data payloads, and write provenance metadata without rendering the final dashboard.

---

## Phase 3: User Story 1 - Compare actual and predicted crisis maps (Priority: P1) MVP

**Goal**: A user can open one generated dashboard, choose a complete model-scope-month selection, and compare actual labels on the left with partitioned predicted labels on the right over the same geography.

**Independent Test**: Generate the dashboard, open the HTML, select GeoRF fs2 2024-06-01, and verify the left panel displays `y_true` actual crisis distribution while the right panel displays `y_pred_partitioned` predicted crisis distribution for the same geographic canvas.

### Implementation for User Story 1

- [X] T011 [US1] Implement dashboard data payload generation in `scripts/generate_actual_predicted_dashboard.py` that maps each `FEWSNET_admin_code` to `y_true` and `y_pred_partitioned` values per model/scope/month
- [X] T012 [US1] Implement static HTML writing in `scripts/generate_actual_predicted_dashboard.py` that embeds geometry, availability, and label payloads into `main_ablation_results/march2026_main_backup_month_ind_cont3/actual_predicted_dashboard/actual_predicted_dashboard.html` by default
- [X] T013 [US1] Implement browser-side selection state and map rendering JavaScript in `scripts/generate_actual_predicted_dashboard.py` so both panels update from the same active model/scope/date selection
- [X] T014 [US1] Implement prediction overlay alpha behavior in `scripts/generate_actual_predicted_dashboard.py` so the left panel defaults to actual-only/prediction alpha 0, the right panel defaults to predicted-visible/prediction alpha 1, and any alpha control updates both panels clearly
- [X] T015 [US1] Implement fixed crisis color legend and no-data styling in `scripts/generate_actual_predicted_dashboard.py` with `0` as non-crisis, `1` as crisis, and unmatched polygons as no data
- [X] T016 [US1] Implement explicit panel labels and summary counts in `scripts/generate_actual_predicted_dashboard.py` identifying left as actual/reference and right as predicted/comparison
- [X] T017 [US1] Run the generator command from `specs/002-actual-predicted-dashboard/quickstart.md` and verify `main_ablation_results/march2026_main_backup_month_ind_cont3/actual_predicted_dashboard/actual_predicted_dashboard.html` exists
- [ ] T018 [US1] Manually open `main_ablation_results/march2026_main_backup_month_ind_cont3/actual_predicted_dashboard/actual_predicted_dashboard.html` and verify GeoRF fs2 2024-06-01 renders actual left and predicted right without running training, batch workflows, or notebooks

**Checkpoint**: User Story 1 is independently functional as the MVP dashboard comparison.

---

## Phase 4: User Story 2 - Inspect available dates, scopes, and models safely (Priority: P2)

**Goal**: A user can switch among available dates, GeoRF/GeoDT models, and fs1/fs2/fs3 scopes while unavailable or incomplete combinations are omitted or shown with visible no-data messaging.

**Independent Test**: Compare selector options with source folders and `month_start` values, verify GeoXGB/XGBoost never appears, and verify a missing or simulated unavailable combination does not display stale maps.

### Implementation for User Story 2

- [X] T019 [US2] Implement model selector population in `scripts/generate_actual_predicted_dashboard.py` from the availability index with exactly GeoRF and GeoDT when available and no GeoXGB/XGBoost option
- [X] T020 [US2] Implement scope selector population in `scripts/generate_actual_predicted_dashboard.py` for available fs1/fs2/fs3 combinations only
- [X] T021 [US2] Implement date selector population in `scripts/generate_actual_predicted_dashboard.py` using normalized `month_start` values sorted chronologically
- [X] T022 [US2] Implement visible no-data and incomplete-data messages in `scripts/generate_actual_predicted_dashboard.py` for missing rows, unmatched joins, or unavailable selections, clearing stale map state before showing the message
- [X] T023 [US2] Update manifest warnings in `scripts/generate_actual_predicted_dashboard.py` to record omitted folders, missing required columns, unmatched polygons, and excluded XGB source paths
- [X] T024 [US2] Run the generator and inspect `main_ablation_results/march2026_main_backup_month_ind_cont3/actual_predicted_dashboard/manifest.json` to verify available models, scopes, dates, XGB exclusions, and no-data warnings are recorded
- [ ] T025 [US2] Manually verify selector behavior in `main_ablation_results/march2026_main_backup_month_ind_cont3/actual_predicted_dashboard/actual_predicted_dashboard.html`, including no GeoXGB/XGBoost option and no stale maps after unavailable selections

**Checkpoint**: User Story 2 is independently functional for safe selector-driven exploration.

---

## Phase 5: User Story 3 - Preserve provenance and output boundaries (Priority: P3)

**Goal**: A maintainer can review generated artifacts and confirm source files, shapefile, join key, model exclusions, value meanings, threshold contract, smoke-test result, and exploratory output boundaries.

**Independent Test**: Read the generated manifest and verify it documents all included prediction files, XGB exclusions, global shapefile path, `FEWSNET_admin_code` join key, binary label meaning, no-threshold contract, output path, and exploratory workflow status.

### Implementation for User Story 3

- [X] T026 [US3] Implement smoke-test selection logic in `scripts/generate_actual_predicted_dashboard.py` that validates at least one GeoRF or GeoDT fs1/fs2/fs3 month has both actual/predicted data and a non-empty shapefile join
- [X] T027 [US3] Write manifest fields in `scripts/generate_actual_predicted_dashboard.py` for workflow mode, production status, source directory, included source files, excluded XGB sources, shapefile path, join key, available models/scopes/dates, label contract, threshold contract, output HTML, optional assets, smoke-test result, and warnings
- [X] T028 [US3] Add output-boundary validation in `scripts/generate_actual_predicted_dashboard.py` to keep generated dashboard files under `main_ablation_results/march2026_main_backup_month_ind_cont3/actual_predicted_dashboard/` by default and avoid overwriting source CSVs or standard deliverable directories
- [X] T029 [US3] Add optional asset manifesting in `scripts/generate_actual_predicted_dashboard.py` so any extracted asset files remain under the exploratory output directory and are listed in `manifest.json`
- [X] T030 [US3] Validate `main_ablation_results/march2026_main_backup_month_ind_cont3/actual_predicted_dashboard/manifest.json` against `specs/002-actual-predicted-dashboard/contracts/manifest.schema.json` using a local JSON-schema-capable validator if available or by manual field inspection if not
- [X] T031 [US3] Verify generated outputs overwrite zero files under `deliverables/`, `prediction_pipeline/`, standard forecast result directories, standalone prediction deliverables, and synthetic scenario deliverables

**Checkpoint**: User Story 3 is independently functional for provenance and artifact-hygiene review.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation, documentation alignment, and cleanup across all user stories.

- [X] T032 [P] Update `specs/002-actual-predicted-dashboard/quickstart.md` if the final generator command, output filenames, or validation steps differ from the planned command
- [X] T033 [P] Review `scripts/generate_actual_predicted_dashboard.py` for ASCII-safe console output, no notebook dependencies, no batch launcher assumptions, and no model-training calls
- [X] T034 Run a final smoke generation using `scripts/generate_actual_predicted_dashboard.py` and record the validated output paths and selected smoke-test combination in `main_ablation_results/march2026_main_backup_month_ind_cont3/actual_predicted_dashboard/manifest.json`
- [ ] T035 Open `main_ablation_results/march2026_main_backup_month_ind_cont3/actual_predicted_dashboard/actual_predicted_dashboard.html` in a browser and verify dashboard opens with selectors populated in under 30 seconds
- [X] T036 Inspect `git status --short` and verify generated exploratory outputs remain ignored or intentionally untracked unless explicitly promoted by the user

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies; can start immediately.
- **Foundational (Phase 2)**: Depends on Setup completion; blocks all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational completion; MVP scope.
- **User Story 2 (Phase 4)**: Depends on Foundational completion and can be developed after or alongside US1, but final selector behavior should be checked against US1 rendering.
- **User Story 3 (Phase 5)**: Depends on Foundational completion and can be developed after or alongside US1/US2, but final manifest should include all implemented behavior.
- **Polish (Phase 6)**: Depends on all desired user stories being complete.

### User Story Dependencies

- **US1 - Compare actual and predicted crisis maps**: Independent after Foundation; delivers MVP dashboard comparison.
- **US2 - Inspect available dates, scopes, and models safely**: Independent after Foundation; enhances selectors and missing-data behavior.
- **US3 - Preserve provenance and output boundaries**: Independent after Foundation; enhances manifest, smoke validation, and artifact hygiene.

### Within Each User Story

- Data payload tasks precede HTML/browser rendering tasks.
- Selector population tasks precede manual selector validation.
- Manifest field tasks precede schema/manual manifest validation.
- Generator execution precedes browser and artifact-boundary validation.

## Parallel Opportunities

- T003 can run in parallel with T001-T002 after the file exists.
- T004-T010 are mostly independent helper implementations in the same file; split only if multiple agents coordinate edits carefully.
- After Phase 2, US1, US2, and US3 can be implemented in parallel by different agents if they coordinate edits to `scripts/generate_actual_predicted_dashboard.py`.
- T032 and T033 can run in parallel during polish because they touch different files.

## Parallel Example: User Story 1

```text
Task: "Implement dashboard data payload generation in scripts/generate_actual_predicted_dashboard.py"
Task: "Implement fixed crisis color legend and no-data styling in scripts/generate_actual_predicted_dashboard.py"
```

## Parallel Example: User Story 2

```text
Task: "Implement model selector population in scripts/generate_actual_predicted_dashboard.py"
Task: "Implement scope selector population in scripts/generate_actual_predicted_dashboard.py"
Task: "Implement date selector population in scripts/generate_actual_predicted_dashboard.py"
```

## Parallel Example: User Story 3

```text
Task: "Implement smoke-test selection logic in scripts/generate_actual_predicted_dashboard.py"
Task: "Add output-boundary validation in scripts/generate_actual_predicted_dashboard.py"
```

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 setup tasks.
2. Complete Phase 2 foundational discovery, validation, geometry, availability, and manifest helpers.
3. Complete Phase 3 User Story 1 tasks.
4. Stop and validate the generated HTML with GeoRF fs2 2024-06-01.

### Incremental Delivery

1. Setup + Foundation creates a reusable generator skeleton.
2. US1 delivers the core actual-vs-predicted dashboard.
3. US2 adds robust selector and missing-data behavior.
4. US3 adds complete provenance, smoke-test, and artifact-hygiene guarantees.
5. Polish validates browser behavior, quickstart accuracy, and git/artifact boundaries.

### Notes

- Every task uses project-relative file paths for implementation targets.
- Do not add formal test tasks unless the user later requests TDD or a test suite.
- Do not create a backend server, Dash app, notebook, or new batch launcher.
- Do not run model training or full batch workflows.
