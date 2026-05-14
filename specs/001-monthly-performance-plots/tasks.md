# Tasks: Monthly Performance Plots

**Input**: Design documents from `/specs/001-monthly-performance-plots/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/plot-monthly-performance-cli.md, quickstart.md

**Tests**: No formal test suite was requested. Validation is handled through the required dry-run/smoke paths and figure/manifest checks.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Every task includes an exact file path

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish the lightweight plotting entry point and confirm the existing input/output locations without changing pipeline behavior.

- [X] T001 Create the lightweight plotting script skeleton with argparse options from the contract in `scripts/plot_monthly_performance_metrics.py`
- [X] T002 Add constants for allowed model families, scopes, metrics, default input roots, and default output paths in `scripts/plot_monthly_performance_metrics.py`
- [X] T003 [P] Verify the required source CSV paths listed in `specs/001-monthly-performance-plots/plan.md` exist before implementation proceeds
- [X] T004 [P] Verify `.gitignore` or repository status keeps generated outputs under `main_ablation_results/march2026_main_backup_month_ind_cont3/monthly_performance_plots/` out of source control unless explicitly promoted

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Implement shared data-loading and validation behavior required by all user stories.

**CRITICAL**: No user story work can begin until this phase is complete.

- [X] T005 Implement repository-relative path resolution and optional path overrides for `--ablation-root`, `--fewsnet-root`, and `--output-dir` in `scripts/plot_monthly_performance_metrics.py`
- [X] T006 Implement explicit GeoDT/GeoRF allowlist selection and GeoXGB exclusion for model metrics paths in `scripts/plot_monthly_performance_metrics.py`
- [X] T007 Implement model metrics CSV loading with required column validation for `test_month`, `model`, `precision`, `recall`, and `f1` in `scripts/plot_monthly_performance_metrics.py`
- [X] T008 Implement FEWSNET baseline CSV loading with required column validation for `year`, `quarter`, `precision(1)`, `recall(1)`, and `f1(1)` in `scripts/plot_monthly_performance_metrics.py`
- [X] T009 Implement FEWSNET quarter-to-test-month alignment where Q1 maps to `YYYY-02`, Q2 maps to `YYYY-06`, Q4 maps to `YYYY-10`, and Q3 is ignored in `scripts/plot_monthly_performance_metrics.py`
- [X] T010 Implement chronological `test_month` parsing and sorting for all plotted series in `scripts/plot_monthly_performance_metrics.py`
- [X] T011 Implement missing-point detection that preserves plotted gaps and records missing model or FEWSNET values in `scripts/plot_monthly_performance_metrics.py`

**Checkpoint**: Foundation ready - user story implementation can now begin.

---

## Phase 3: User Story 1 - Compare monthly model performance (Priority: P1) MVP

**Goal**: Generate one GeoDT and one GeoRF monthly performance figure with 3x3 subplots and three comparison lines per applicable subplot.

**Independent Test**: Run the full script and verify exactly two figures exist, each with rows fs1/fs2/fs3, columns precision/recall/F1, chronological x-axis values, and partitioned/pooled/FEWSNET lines.

### Implementation for User Story 1

- [X] T012 [US1] Implement 3x3 subplot figure construction for one model family using rows `fs1`, `fs2`, `fs3` and columns `precision`, `recall`, `f1` in `scripts/plot_monthly_performance_metrics.py`
- [X] T013 [US1] Implement partitioned solid-line and pooled dashed-line plotting from `model` column values in `scripts/plot_monthly_performance_metrics.py`
- [X] T014 [US1] Implement FEWSNET baseline plotting with a distinct comparison color for fs1 and fs2 in `scripts/plot_monthly_performance_metrics.py`
- [X] T015 [US1] Implement fs3 FEWSNET plotting by reusing fs2 FEWSNET values and labeling the line `FEWSNET baseline (fs2 reused for fs3)` in `scripts/plot_monthly_performance_metrics.py`
- [X] T016 [US1] Add clear subplot titles, row/scope labels, metric/column labels, y-axis metric labels, rotated chronological x-axis labels, and readable legends in `scripts/plot_monthly_performance_metrics.py`
- [X] T017 [US1] Save full-run figures as `geodt_monthly_performance.png` and `georf_monthly_performance.png` under `main_ablation_results/march2026_main_backup_month_ind_cont3/monthly_performance_plots/` in `scripts/plot_monthly_performance_metrics.py`
- [X] T018 [US1] Run `python3 scripts/plot_monthly_performance_metrics.py` and verify the two generated figure paths listed in `specs/001-monthly-performance-plots/quickstart.md`

**Checkpoint**: User Story 1 is independently functional and testable.

---

## Phase 4: User Story 2 - Understand baseline comparison provenance (Priority: P2)

**Goal**: Generate a concise manifest that records source files, file selection logic, column contract, missing data, generated artifacts, and FEWSNET fs3 proxy assumption.

**Independent Test**: Read the manifest and confirm it identifies all sources, excludes GeoXGB, names pooled/partitioned/FEWSNET series rules, records missing points, and clearly states that FEWSNET fs2 is reused for fs3.

### Implementation for User Story 2

- [X] T019 [US2] Implement manifest assembly with workflow mode, exploratory status, source paths, excluded GeoXGB paths, and generated artifact paths in `scripts/plot_monthly_performance_metrics.py`
- [X] T020 [US2] Add model-family, scope, pooled-versus-partitioned, metric-column, and FEWSNET-column contracts to the manifest in `scripts/plot_monthly_performance_metrics.py`
- [X] T021 [US2] Add explicit FEWSNET fs3 proxy text stating fs2 is reused for fs3 and no native fs3 FEWSNET baseline is implied in `scripts/plot_monthly_performance_metrics.py`
- [X] T022 [US2] Add missing-point records and validation summary fields to `monthly_performance_manifest.json` from `scripts/plot_monthly_performance_metrics.py`
- [X] T023 [US2] Save the manifest as `main_ablation_results/march2026_main_backup_month_ind_cont3/monthly_performance_plots/monthly_performance_manifest.json` in `scripts/plot_monthly_performance_metrics.py`
- [X] T024 [US2] Manually inspect `main_ablation_results/march2026_main_backup_month_ind_cont3/monthly_performance_plots/monthly_performance_manifest.json` against the manifest requirements in `specs/001-monthly-performance-plots/data-model.md`

**Checkpoint**: User Story 2 is independently functional and testable.

---

## Phase 5: User Story 3 - Validate with a lightweight smoke test (Priority: P3)

**Goal**: Provide a dry-run and smoke path that validates input contracts, line counts, legend labels, chronological ordering, GeoXGB exclusion, and fs3 FEWSNET labeling without long batch workflows.

**Independent Test**: Run the dry-run and optional single-model smoke commands from quickstart.md and confirm they do not invoke training, batch jobs, notebooks, or unrelated deliverable generation.

### Implementation for User Story 3

- [X] T025 [US3] Implement `--dry-run` behavior that validates input discovery, required columns, planned artifact paths, chronological ordering, GeoXGB exclusion, and fs3 FEWSNET labeling without writing figures in `scripts/plot_monthly_performance_metrics.py`
- [X] T026 [US3] Implement `--model {geodt,georf,all}` behavior for restricted smoke validation and full two-model generation in `scripts/plot_monthly_performance_metrics.py`
- [X] T027 [US3] Implement `--smoke` behavior that can generate or validate a reduced single-model run without invoking batch jobs, model training, or notebooks in `scripts/plot_monthly_performance_metrics.py`
- [X] T028 [US3] Ensure CLI status output from `scripts/plot_monthly_performance_metrics.py` is ASCII-safe for Windows/GBK terminals
- [X] T029 [US3] Run `python3 scripts/plot_monthly_performance_metrics.py --model geodt --dry-run` and verify the checks listed in `specs/001-monthly-performance-plots/quickstart.md`
- [X] T030 [US3] Run `python3 scripts/plot_monthly_performance_metrics.py --model geodt --smoke` and verify the smoke behavior described in `specs/001-monthly-performance-plots/contracts/plot-monthly-performance-cli.md`

**Checkpoint**: User Story 3 is independently functional and testable.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation, documentation, and artifact hygiene across all user stories.

- [X] T031 [P] Update `specs/001-monthly-performance-plots/quickstart.md` if implementation command names, output names, or smoke behavior differ from the planned contract
- [X] T032 [P] Update `specs/001-monthly-performance-plots/contracts/plot-monthly-performance-cli.md` if any final CLI option behavior changes during implementation
- [X] T033 Run final full validation command `python3 scripts/plot_monthly_performance_metrics.py` and verify success criteria SC-001 through SC-009 from `specs/001-monthly-performance-plots/spec.md`
- [X] T034 Verify `git status --short` does not include generated PNGs, JSON manifests, caches, notebooks, pickles, shapefile caches, or unrelated deliverables from `main_ablation_results/march2026_main_backup_month_ind_cont3/monthly_performance_plots/`
- [X] T035 Record any final artifact-hygiene or validation notes in `specs/001-monthly-performance-plots/quickstart.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately.
- **Foundational (Phase 2)**: Depends on Setup completion and blocks all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational phase; delivers MVP figures.
- **User Story 2 (Phase 4)**: Depends on Foundational phase; can be implemented after or alongside US1 once output artifact paths are stable.
- **User Story 3 (Phase 5)**: Depends on Foundational phase; can be implemented alongside US1/US2 after core data loading exists.
- **Polish (Phase 6)**: Depends on desired user stories being complete.

### User Story Dependencies

- **US1 (P1)**: MVP; no dependency on US2 or US3 after Foundational.
- **US2 (P2)**: No dependency on US3; uses source/series information from Foundational and artifact names from US1.
- **US3 (P3)**: No dependency on US2; uses source/series validation from Foundational and plotting behavior from US1 for smoke figure mode.

### Within Each User Story

- Data-loading and validation helpers before plotting or manifest output.
- FEWSNET fs3 proxy labeling before any fs3 figure or manifest is accepted.
- Dry-run/smoke validation before final full generation.
- Artifact hygiene check before considering implementation complete.

---

## Parallel Opportunities

- T003 and T004 can run in parallel after T001/T002 because they inspect existing paths and artifact hygiene separately.
- Documentation updates T031 and T032 can run in parallel after implementation behavior stabilizes.
- After Phase 2, US1 plotting, US2 manifest fields, and US3 CLI validation can be developed incrementally, but tasks editing `scripts/plot_monthly_performance_metrics.py` should be coordinated because they touch the same file.

## Parallel Example: User Story 2

```bash
Task: "Add model-family, scope, pooled-versus-partitioned, metric-column, and FEWSNET-column contracts to the manifest in scripts/plot_monthly_performance_metrics.py"
Task: "Manually inspect main_ablation_results/march2026_main_backup_month_ind_cont3/monthly_performance_plots/monthly_performance_manifest.json against specs/001-monthly-performance-plots/data-model.md"
```

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 setup.
2. Complete Phase 2 foundational loading, validation, alignment, sorting, and missing-point behavior.
3. Complete Phase 3 US1 plotting.
4. Stop and validate the two-figure output before adding manifest and smoke refinements.

### Incremental Delivery

1. Deliver US1: figures can be reviewed visually.
2. Deliver US2: manifest makes the comparison provenance auditable.
3. Deliver US3: dry-run/smoke commands make future validation cheap and safe.
4. Complete Polish: update docs if implementation differs and verify no generated outputs are accidentally staged.

### Artifact Hygiene Rule

Commit only source and Spec Kit planning/documentation changes unless the user explicitly asks to promote generated diagnostic artifacts. Do not commit generated PNGs, JSON manifests, caches, notebooks, pickles, shapefile caches, or unrelated deliverables.
