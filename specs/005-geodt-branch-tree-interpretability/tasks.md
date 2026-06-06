# Tasks: GeoDT Branch-Specific Local Tree Interpretability Figure

**Input**: Design documents from `/specs/005-geodt-branch-tree-interpretability/`  
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `quickstart.md`, `contracts/geodt-branch-tree-diagnostic-cli.md`, `specs/_evidence/005-geodt-branch-tree-interpretability.evidence.md`  
**Status**: Draft task plan; feature implementation not started

**Tests**: Required for this Brownfield feature. Tasks include characterization, test harness, focused regression, feature implementation, migration, validation, and documentation work. Behavior-changing implementation tasks depend on characterization, contract, smoke, or focused regression tasks when legacy behavior is evidence-dependent.

Task-level `Depends:` lines are authoritative for execution order; phase and wave sections summarize the same graph for human readability.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel with other ready tasks after listed dependencies are satisfied
- **[Story]**: User-story label for story phases only: `[US1]`, `[US2]`, `[US3]`
- Every task has a `Reqs:` line mapping it to formal requirement IDs from `spec.md`
- Every task has a `Depends:` line; `Depends: none` is used only for the graph root
- Existing GeoDT training, prediction dispatch, branch training, batch launchers, `dt_rules` export behavior, checkpoints, partition artifacts, deliverables, and evaluation outputs are read-only unless a future spec revision explicitly authorizes changes

## Artifact-of-Record Policy

`tasks.md` is the execution plan and should not be used as the primary runtime evidence log. Characterization findings, validation outputs, migration notes, optional real-archive observations, known baseline failures, and rollback confirmations MUST be written to one of the following artifacts instead:

- `specs/_evidence/005-geodt-branch-tree-interpretability.evidence.md`, when preserving brownfield evidence;
- a generated audit summary under the configured diagnostic output directory;
- the diagnostic metadata JSON;
- a structured failure summary;
- task-run comments/checkmarks in the task runner, if the runner supports them.

Only intentional task-plan edits should modify `tasks.md`.

Current `spec.md` References are authoritative for this feature. Older evidence-file sections labeled as recommendations, such as "Copy into spec.md", are historical guidance and MUST NOT override the current `spec.md` reference list. If evidence references need refresh, create a future evidence refresh task or note; do not change task execution based solely on stale recommendation text.

`<focused-test-path>` = the focused test file path selected by T002. Until T002 selects and documents the focused test path in the designated evidence/audit artifact, downstream test tasks MUST refer to `<focused-test-path>` rather than hard-coding `src/tests/test_geodt_branch_tree_diagnostic.py`.

## Task Categories

- **Characterization tasks**: confirm legacy behavior, artifact availability, and evidence-dependent assumptions before implementation
- **Test harness tasks**: create synthetic fixtures and focused tests for the new diagnostic without relying on large production artifacts
- **Focused regression tasks**: guard brownfield boundaries such as root/global `dt_rules`, no-overwrite behavior, and read-only production artifacts
- **Feature implementation tasks**: add the standalone exploratory diagnostic CLI and internal logic
- **Migration tasks**: confirm no data/runtime migration is required and that rollback is deletion of new diagnostic files/outputs only
- **Validation tasks**: run audit-only, figure-generation, reproduction, and boundary checks
- **Documentation tasks**: keep Speckit contracts, quickstart, and context pointers aligned with the implemented interface

---

## Phase 1: Setup and Brownfield Scope Boundary

**Purpose**: Confirm where new work may happen and establish the read-only boundaries before creating tests or implementation files.

- [X] T001 Characterization: Verify the active feature directory and design artifacts exist at `specs/005-geodt-branch-tree-interpretability/spec.md`, `specs/005-geodt-branch-tree-interpretability/plan.md`, and `specs/005-geodt-branch-tree-interpretability/contracts/geodt-branch-tree-diagnostic-cli.md`
  Reqs: FR-000, FR-000D, FR-031, SC-013
  Depends: none
- [X] T002 Characterization: Inspect existing test locations `src/tests/` and `tests/`, select the focused test path, and write the selected path plus rationale to the designated evidence/audit artifact
  Reqs: FR-AUDIT-004, FR-EVID-004, SC-018, SC-020
  Depends: T001
- [X] T003 Characterization: Verify `scripts/plot_geodt_branch_tree_comparison.py` does not already exist or, if it exists, document whether it is unrelated or superseded in the designated evidence/audit artifact
  Reqs: FR-035, FR-OUT-003, SC-009, SC-019
  Depends: T001
- [X] T004 Migration: Confirm no data migration, config migration, dependency migration, or production artifact migration is required for this feature in the designated migration/evidence artifact or final implementation notes
  Reqs: FR-033, FR-035, SC-009
  Depends: T001
- [X] T005 Documentation: Confirm `CLAUDE.md` and `AGENTS.md` context blocks either reference `specs/005-geodt-branch-tree-interpretability/plan.md` where appropriate or are intentionally generic Speckit guidance; verify no active context pointer references a different 005 feature path, and treat stale evidence recommendations as historical notes rather than current source of truth
  Reqs: FR-EVID-001, FR-EVID-004, SC-020
  Depends: T001

---

## Phase 2: Foundational Characterization, Test Harness, and Regression Gates

**Purpose**: Build the evidence and test foundation that blocks all behavior-changing implementation tasks.

**CRITICAL**: No implementation task may edit or create `scripts/plot_geodt_branch_tree_comparison.py` until T018A is complete. T018A gates T006 through T018.

### Characterization Tasks

- [X] T006 [P] Characterization: Confirm current root/global `dt_rules` export behavior by inspecting `app/main_model_DT.py`, `src/utils/dt_rule_export.py`, and `config.py`
  Reqs: FR-008, FR-032, FR-EVID-002, FR-EVID-004, SC-002, SC-016, SC-020
  Depends: T001, T005
- [X] T007 [P] Characterization: Confirm existing GeoDT checkpoint naming and loader behavior by inspecting `src/model/model_DT.py`, `src/model/GeoRF_DT.py`, and `src/model/train_branch.py`
  Reqs: FR-004, FR-CHK-001, FR-CHK-002, FR-CHK-003, FR-CHK-005, FR-EVID-002, SC-013, SC-020
  Depends: T001
- [X] T008 [P] Characterization: Confirm branch assignment semantics for `space_partitions/X_branch_id.npy`, `space_partitions/s_branch.pkl`, and terminal correspondence tables by inspecting `src/helper/helper.py` and `src/merge/terminal.py`
  Reqs: FR-006B, FR-EVID-002, FR-EVID-004, SC-013, SC-020
  Depends: T001
- [X] T009 [P] Characterization: Inspect available local GeoDT archive/source folders under the explicit archive path, explicit archive list, or bounded archive root used for the first smoke run, and write archive availability, missing artifacts, and fallback candidate observations to the designated evidence/audit artifact
  Reqs: FR-000F, FR-000G, FR-000H, FR-000I, FR-000L, FR-003A, FR-003B, FR-EVID-004, SC-014, SC-017, SC-020, SC-022
  Depends: T001, T003
- [X] T010 [P] Characterization: Identify candidate selected-run feature-name sources in available archives or source folders and write feature-source compatibility, feature-count evidence, and run/scope tie findings to the designated evidence/audit artifact
  Reqs: FR-006D, FR-006E, FR-006F, FR-006G, FR-006H, FR-EVID-002, FR-EVID-004, SC-020, SC-023
  Depends: T001

### Test Harness Tasks

- [X] T011 Test harness: Create synthetic fixture builder scaffolding in `<focused-test-path>` after T002 selects the focused test location; the scaffolding must expose either a pytest fixture returning archive/output paths for subprocess tests or a helper function/command that creates a temporary GeoDT-like archive and prints or returns the path
  Reqs: FR-AUDIT-004, SC-001, SC-010, SC-018
  Depends: T001, T002
- [X] T012 [P] Test harness: Add a synthetic complete GeoDT-like archive fixture with branch checkpoints, `space_partitions/`, feature names, and correspondence-table variants in `<focused-test-path>`; the fixture archive must be usable by CLI subprocess tests, not only in-process unit tests
  Reqs: FR-000I, FR-006A, FR-006B, FR-006D, FR-CHK-001, SC-001, SC-013, SC-017
  Depends: T011
- [X] T013 [P] Test harness: Add a synthetic visual-archive plus same-run artifact-provider fixture in `<focused-test-path>`
  Reqs: FR-000F, FR-000I, FR-000M, FR-000N, FR-003A, FR-003B, FR-EVID-004, SC-014, SC-020
  Depends: T011
- [X] T014 [P] Test harness: Add synthetic negative fixtures for root/global-only `dt_rules`, feature-name count mismatch, out-of-bounds split index, unreadable pairs, and missing reproduction artifacts in `<focused-test-path>`
  Reqs: FR-000C, FR-000L, FR-006F, FR-006G, FR-016E, FR-036A, FR-REPRO-006, SC-008, SC-021
  Depends: T011

### Focused Regression Tasks

- [X] T015 Focused regression: Add a regression test that a root/global-only `dt_rules` archive is incomplete and is never used as a branch-tree rule source in `<focused-test-path>`
  Reqs: FR-008, FR-000L, FR-032, FR-EVID-002, SC-002, SC-016, SC-020
  Depends: T006, T012, T014
- [X] T016 Focused regression: Add a no-overwrite regression test for existing checkpoints, `space_partitions/`, `dt_rules/`, `other_outputs/`, and `deliverables/` paths in `<focused-test-path>`
  Reqs: FR-035, FR-OUT-003, SC-009, SC-019
  Depends: T003, T012
- [X] T017 Focused regression: Add a bounded archive-discovery regression test that rejects unbounded filesystem search and unrelated user directories in `<focused-test-path>`
  Reqs: FR-000F, FR-000G, FR-000H, SC-022
  Depends: T009, T012
- [X] T018 Focused regression: Add a same-run artifact-provider compatibility regression test that records both selected archive and provider paths in `<focused-test-path>`
  Reqs: FR-000F, FR-000M, FR-000N, FR-003B, FR-EVID-001, FR-EVID-004, SC-014, SC-020
  Depends: T009, T013
- [X] T018A Foundation gate: Confirm T006 through T018 are complete, characterization findings are written to the designated evidence/audit artifact, the focused test path is selected, archive discovery/fallback observations from T009 are captured, feature-name source findings from T010 are captured, evidence-dependent assumption status from T006-T010 is captured, brownfield regression gates are ready, and no implementation task may create or edit `scripts/plot_geodt_branch_tree_comparison.py` before this gate
  Reqs: FR-000, FR-000D, FR-035, FR-EVID-001, FR-EVID-004, SC-009, SC-013, SC-020
  Depends: T006, T007, T008, T009, T010, T011, T012, T013, T014, T015, T016, T017, T018

**Checkpoint**: The test harness can represent complete, incomplete, fallback, and mismatch archives without using production artifacts.

---

## Phase 3: User Story 1 - Generate Branch-Tree Comparison Figure (Priority: P1) MVP

**Goal**: Generate a 1x2 PNG/PDF figure comparing two eligible terminal branch-specific local DecisionTree checkpoints from the same selected GeoDT monthly run.

**Independent Test**: Run the diagnostic on a synthetic or eligible archived GeoDT monthly result folder and verify that both plotted trees are loaded from branch-specific checkpoints, share plotted depth/style, satisfy readability constraints, and write metadata without modifying source artifacts.

### Tests for User Story 1

- [X] T019 [P] [US1] Add contract tests for default figure-generation CLI arguments and standalone CLI behavior from `specs/005-geodt-branch-tree-interpretability/contracts/geodt-branch-tree-diagnostic-cli.md` in `<focused-test-path>`, including an assertion that parsing/running diagnostic arguments does not mutate training config values such as `SAVE_DT_RULES`, `SAVE_DT_NODE_DUMP`, or `ACTIVE_LAGS`
  Reqs: FR-001, FR-018, FR-029, FR-030, FR-035, FR-OUT-001, FR-OUT-004, US1, SC-001, SC-009
  Depends: T018A
- [X] T020 [P] [US1] Add preflight tests for archive discovery, fallback selection, minimum required artifact completeness, and structured failure summaries in `<focused-test-path>`
  Reqs: FR-000, FR-000A, FR-000C, FR-000F, FR-000G, FR-000H, FR-000I, FR-000J, FR-000K, FR-000L, FR-003A, FR-003B, FR-036A, FR-036B, SC-013, SC-014, SC-017, SC-021, SC-022
  Depends: T012, T013, T014, T017, T018, T018A
- [X] T021 [P] [US1] Add checkpoint classification and branch eligibility tests covering root/global exclusion, load failures, unused checkpoints, and assignment-count reporting in `<focused-test-path>`
  Reqs: FR-004, FR-005, FR-006A, FR-006B, FR-006C, FR-007, FR-017, FR-017A, FR-017B, FR-017C, FR-017D, FR-CHK-001, FR-CHK-002, FR-CHK-003, FR-CHK-004, FR-CHK-005, SC-015
  Depends: T012, T014, T015, T018A
- [X] T022 [P] [US1] Add feature-name source precedence and split-index bounds tests in `<focused-test-path>`
  Reqs: FR-006D, FR-006E, FR-006F, FR-006G, FR-006H, SC-023
  Depends: T010, T012, T014, T018A
- [X] T023 [P] [US1] Add figure-rendering smoke tests for 1x2 layout, minimum dimensions, 300 DPI PNG metadata, optional PDF output, neutral class labels, and documented abbreviations in `<focused-test-path>`
  Reqs: FR-018, FR-019, FR-020, FR-021, FR-022, FR-022A, FR-022B, FR-022C, FR-023, FR-023A, FR-023B, FR-023C, FR-024, FR-024A, FR-024B, FR-024C, FR-024D, FR-024E, FR-025, FR-025A, FR-025B, FR-027, FR-028, FR-028A, FR-028B, FR-029, FR-030, SC-001, SC-011, SC-012
  Depends: T012, T014, T018A

### Implementation for User Story 1

- [X] T024 [US1] Feature implementation: Create standalone CLI skeleton with `--archive-path`, `--archive-list`, `--archive-root`, `--artifact-provider-path`, `--output-dir`, `--overwrite`, `--png-only`, `--pdf-only`, `--k`, and `--max-plot-depth` in `scripts/plot_geodt_branch_tree_comparison.py`
  Reqs: FR-001, FR-000F, FR-000G, FR-OUT-001, FR-OUT-003, US1
  Depends: T003, T006, T007, T008, T009, T010, T018A, T019
- [X] T025 [US1] Feature implementation: Implement bounded archive resolver and deterministic fallback candidate decisions in `scripts/plot_geodt_branch_tree_comparison.py`
  Reqs: FR-000F, FR-000G, FR-000H, FR-002, FR-003, FR-003A, FR-003B, SC-014, SC-022
  Depends: T020, T024
- [X] T026 [US1] Feature implementation: Implement same-run/source artifact-provider resolution and compatibility evidence recording in `scripts/plot_geodt_branch_tree_comparison.py`
  Reqs: FR-000I, FR-000M, FR-000N, FR-EVID-001, FR-EVID-003, FR-EVID-004, SC-017, SC-020
  Depends: T018, T025
- [X] T027 [US1] Feature implementation: Implement artifact characterizer for `checkpoints/dt_*`, `space_partitions/X_branch_id.npy`, `space_partitions/s_branch.pkl`, `branch_table.npy`, `correspondence_table_*.csv`, feature-name candidates, and root/global `dt_rules` boundary notes in `scripts/plot_geodt_branch_tree_comparison.py`
  Reqs: FR-000, FR-000A, FR-000B, FR-000C, FR-000D, FR-000E, FR-000I, FR-000L, FR-004, FR-008, FR-032, SC-013, SC-016, SC-017
  Depends: T015, T020, T026
- [X] T028 [US1] Feature implementation: Implement branch assignment source selector with precedence and rejected-source reasons in `scripts/plot_geodt_branch_tree_comparison.py`
  Reqs: FR-005, FR-006, FR-006A, FR-006B, FR-006C, FR-EVID-002, FR-EVID-004, SC-015, SC-020
  Depends: T008, T021, T027
- [X] T029 [US1] Feature implementation: Implement feature-name source selector with feature-count validation and split-index bounds checks in `scripts/plot_geodt_branch_tree_comparison.py`
  Reqs: FR-006D, FR-006E, FR-006F, FR-006G, FR-006H, SC-023
  Depends: T010, T022, T027
- [X] T030 [US1] Feature implementation: Implement checkpoint parser and loader that classifies root/global, branch-specific candidate, unusable, and unknown checkpoints without trusting filename pattern alone in `scripts/plot_geodt_branch_tree_comparison.py`
  Reqs: FR-004, FR-007, FR-CHK-001, FR-CHK-002, FR-CHK-003, FR-CHK-004, FR-CHK-005, SC-015
  Depends: T007, T021, T027
- [X] T031 [US1] Feature implementation: Implement branch eligibility builder with assigned admin/group counts, prediction-row counts when available, unavailable training-sample count markers, and exclusion reasons in `scripts/plot_geodt_branch_tree_comparison.py`
  Reqs: FR-006A, FR-017, FR-017A, FR-017B, FR-017C, FR-017D, SC-015
  Depends: T021, T028, T029, T030
- [X] T032 [US1] Feature implementation: Implement figure renderer for matched 1x2 shallow DecisionTree panels, branch titles, safe class labels, label wrapping or unique abbreviations, 300 DPI PNG, optional PDF, and caption text in `scripts/plot_geodt_branch_tree_comparison.py`
  Reqs: FR-018, FR-019, FR-020, FR-021, FR-022, FR-022A, FR-022B, FR-022C, FR-023, FR-023A, FR-023B, FR-023C, FR-024, FR-024A, FR-024B, FR-024C, FR-024D, FR-024E, FR-025, FR-025A, FR-025B, FR-027, FR-028, FR-028A, FR-028B, FR-029, FR-030, SC-001, SC-011, SC-012
  Depends: T023, T031, T041, T042, T043
- [X] T033 [US1] Feature implementation: Implement metadata JSON writer with selected archive, artifact-provider evidence, checkpoint provenance, branch counts, feature-name provenance, readability result, output paths, and root/global `dt_rules` distinction in `scripts/plot_geodt_branch_tree_comparison.py`
  Reqs: FR-031, FR-032, FR-034, FR-EVID-004, SC-004, SC-010, SC-016, SC-020
  Depends: T016, T027, T028, T029, T030, T031, T032, T041, T042, T043
- [X] T034 [US1] Feature implementation: Implement no-overwrite output path handling and structured failure-summary writing/display in `scripts/plot_geodt_branch_tree_comparison.py`
  Reqs: FR-035, FR-036, FR-036A, FR-036B, FR-OUT-001, FR-OUT-002, FR-OUT-003, FR-OUT-004, SC-009, SC-019, SC-021
  Depends: T016, T020, T033

### Validation for User Story 1

- [X] T035 [US1] Validation: Run focused tests for archive preflight, checkpoint classification, feature-name validation, figure rendering, no-overwrite behavior, metadata output, and shared selection-pipeline behavior in `<focused-test-path>`
  Reqs: SC-001, SC-002, SC-003, SC-004, SC-008, SC-009, SC-011, SC-012, SC-013, SC-015, SC-017, SC-021, SC-023
  Depends: T024, T025, T026, T027, T028, T029, T030, T031, T032, T033, T034, T041, T042, T043
- [X] T036 [US1] Validation: Run a pytest-managed CLI subprocess smoke test, or a standalone synthetic figure-generation smoke command using the reusable fixture path/helper from T011/T012, with fixture output under a temporary diagnostics directory
  Reqs: US1, SC-001, SC-002, SC-004, SC-009, SC-011, SC-012
  Depends: T035

**Checkpoint**: User Story 1 is independently complete when a synthetic or eligible real archive produces a valid branch-specific figure and metadata, or fails clearly without partial/misleading output.

---

## Phase 4: User Story 2 - Audit Candidate Branch Selection Without Rendering the Final Figure (Priority: P2)

**Goal**: Provide an audit-only mode that performs the same branch eligibility, scoring, readability, and selection decisions as figure-generation mode without rendering PNG/PDF output.

**Independent Test**: Run audit-only mode and verify that it writes or displays eligible/ineligible branches, rejected/skipped candidates, pairwise scores, readability decisions, selected pair, fallback decision, and root/global exclusion decision without creating a figure.

### Tests for User Story 2

- [X] T037 [P] [US2] Add top-K signature extraction tests for split features at depths `0` through `K-1`, actual available depth, and neutral leaf class summaries in `<focused-test-path>`
  Reqs: FR-010, FR-011, FR-012, FR-014A, US2
  Depends: T012, T031
- [X] T038 [P] [US2] Add deterministic Jaccard scoring tests proving threshold/direction supplemental fields do not alter default ranking in `<focused-test-path>`
  Reqs: FR-013, FR-014, FR-014A, FR-014B, FR-015, SC-003
  Depends: T012, T037
- [X] T039 [P] [US2] Add tie-break and readability-gate tests for rejected higher-scoring pairs, no-readable-pair failure, and deterministic selected-pair ordering in `<focused-test-path>`
  Reqs: FR-016A, FR-016B, FR-016C, FR-016D, FR-016E, FR-016F, FR-016G, FR-016H, SC-003, SC-008
  Depends: T012, T014, T038
- [X] T040 [P] [US2] Add audit-only contract tests verifying no PNG/PDF outputs are rendered and selection matches figure-generation mode for the same inputs in `<focused-test-path>`
  Reqs: FR-AUDIT-001, FR-AUDIT-002, FR-AUDIT-003, FR-AUDIT-004, SC-018
  Depends: T019, T035, T039

### Implementation for User Story 2

- [X] T041 [US2] Feature implementation: Implement top-K branch signature extraction and supplemental threshold/direction audit fields in `scripts/plot_geodt_branch_tree_comparison.py`
  Reqs: FR-010, FR-011, FR-012, FR-015
  Depends: T031, T037
- [X] T042 [US2] Feature implementation: Implement deterministic Jaccard scoring, contrast descriptor assignment, tie-break fields, and rejected-pair recording in `scripts/plot_geodt_branch_tree_comparison.py`
  Reqs: FR-013, FR-014, FR-014A, FR-014B, FR-016A, FR-016G, FR-016H, FR-028B, SC-003
  Depends: T038, T041
- [X] T043 [US2] Feature implementation: Implement readability gate checks for non-leaf splits, matched plotted depth/style, label wrapping or abbreviation, minimum font size, and no silent truncation in `scripts/plot_geodt_branch_tree_comparison.py`
  Reqs: FR-016B, FR-016C, FR-016D, FR-016E, FR-016F, FR-024C, FR-024D, FR-024E, SC-012
  Depends: T039, T042
- [X] T044 [US2] Feature implementation: Implement `--audit-only` mode and audit summary output in `scripts/plot_geodt_branch_tree_comparison.py`
  Reqs: FR-AUDIT-001, FR-AUDIT-002, FR-AUDIT-003, FR-AUDIT-004, SC-018
  Depends: T040, T041, T042, T043

### Validation for User Story 2

- [X] T045 [US2] Validation: Run focused tests for signature extraction, scoring, tie-breaks, readability decisions, audit-only output, and no-figure behavior in `<focused-test-path>`
  Reqs: US2, SC-003, SC-008, SC-018
  Depends: T041, T042, T043, T044
- [X] T046 [US2] Validation: Run the synthetic audit-only quickstart command from `specs/005-geodt-branch-tree-interpretability/quickstart.md` adapted to the fixture archive and verify no PNG/PDF file is created
  Reqs: US2, FR-AUDIT-004, SC-018
  Depends: T045

**Checkpoint**: User Story 2 is independently complete when audit-only mode selects the same pair as figure mode and records all candidate decisions without rendering a figure.

---

## Phase 5: User Story 3 - Reproduce a Previously Generated Branch-Tree Figure from Metadata (Priority: P3)

**Goal**: Reproduce the selected branch comparison from a prior metadata summary without reselecting a different branch pair by default.

**Independent Test**: Run reproduction mode with prior metadata and verify that selected archive, branch IDs, checkpoint paths, feature-name source, top-K features, default dissimilarity score, and figure inputs match the metadata summary. Bitwise-identical image output is not required.

### Tests for User Story 3

- [X] T047 [P] [US3] Add reproduction metadata schema tests for recorded archive input, selected archive, artifact-provider path, branches, checkpoint paths, feature-name source, K, plotted depths, top-K features, score, and formula in `<focused-test-path>`
  Reqs: FR-REPRO-001, FR-REPRO-002, FR-031, SC-010, US3
  Depends: T033, T045
- [X] T048 [P] [US3] Add reproduction mismatch tests for missing checkpoint, changed top-K features, changed score, missing feature-name source, and missing artifact-provider path in `<focused-test-path>`
  Reqs: FR-REPRO-003, FR-REPRO-006, FR-036A, SC-010, SC-021
  Depends: T014, T047
- [X] T049 [P] [US3] Add CLI contract tests for `--reproduce-from <metadata.json>` default non-reselection behavior in `<focused-test-path>`
  Reqs: FR-REPRO-001, FR-REPRO-004, FR-REPRO-005, US3
  Depends: T019, T047

### Implementation for User Story 3

- [X] T050 [US3] Feature implementation: Implement `--reproduce-from` metadata loading and recorded artifact resolution in `scripts/plot_geodt_branch_tree_comparison.py`
  Reqs: FR-REPRO-001, FR-REPRO-002
  Depends: T047, T049
- [X] T051 [US3] Feature implementation: Implement reproduction verification for recovered branch IDs, checkpoint paths, top-K feature sets, default dissimilarity score, and figure inputs without default reselection in `scripts/plot_geodt_branch_tree_comparison.py`
  Reqs: FR-REPRO-003, FR-REPRO-004, FR-REPRO-005, SC-010
  Depends: T048, T050
- [X] T052 [US3] Feature implementation: Implement reproduction report and mismatch failure summary output in `scripts/plot_geodt_branch_tree_comparison.py`
  Reqs: FR-REPRO-006, FR-036A, FR-036B, SC-021
  Depends: T048, T051

### Validation for User Story 3

- [X] T053 [US3] Validation: Run focused reproduction and mismatch tests in `<focused-test-path>`
  Reqs: US3, SC-010, SC-021
  Depends: T050, T051, T052
- [X] T054 [US3] Validation: Run a pytest-managed reproduction subprocess test, or a standalone reproduction command using fixture metadata produced by the reusable fixture path/helper from T011/T012, and verify the same selected pair is recovered
  Reqs: US3, FR-REPRO-001, FR-REPRO-005, SC-010
  Depends: T053

**Checkpoint**: User Story 3 is independently complete when reproduction mode recovers recorded selection inputs or fails with a mismatch report, never silently selecting replacements.

---

## Phase 6: Polish, Documentation, and Brownfield Validation

**Purpose**: Finalize docs, validate boundaries, and keep optional real-archive smoke testing separate from synthetic test success.

- [X] T055 Documentation: Update `specs/005-geodt-branch-tree-interpretability/quickstart.md` if implemented CLI behavior differs from the planned examples, preserving audit-only, figure-generation, artifact-provider, discovery-root, reproduction, and safety-check sections
  Reqs: FR-AUDIT-001, FR-REPRO-001, FR-OUT-001, SC-018, SC-019
  Depends: T005, T036, T046, T054
- [X] T056 Documentation: Update `specs/005-geodt-branch-tree-interpretability/contracts/geodt-branch-tree-diagnostic-cli.md` if any implemented CLI flag or output behavior differs from the planned contract
  Reqs: FR-OUT-001, FR-OUT-004, FR-AUDIT-001, FR-REPRO-001
  Depends: T005, T036, T046, T054
- [X] T057 [Optional] Validation: Run an optional real-archive audit-only dry run using `scripts/plot_geodt_branch_tree_comparison.py --audit-only` only after an explicit bounded archive path/root is available, and write optional real-archive dry-run findings to the audit summary or evidence artifact
  Reqs: FR-000F, FR-000G, FR-AUDIT-001, FR-EVID-004, SC-018, SC-020, SC-022
  Depends: T046
  Blocking: no
- [X] T058 Validation: Run a source-boundary check confirming `app/main_model_DT.py`, `src/model/GeoRF_DT.py`, `src/model/model_DT.py`, `src/model/train_branch.py`, `src/helper/helper.py`, `src/merge/terminal.py`, `src/utils/dt_rule_export.py`, `run_batches_2021_2024_visual_monthly.bat`, and production artifact directories were not modified for this feature
  Reqs: FR-035, SC-009
  Depends: T035, T045, T053
- [X] T059 Validation: Run focused tests for `<focused-test-path>` and write known baseline failures and new regressions to the validation summary or evidence artifact
  Reqs: SC-001, SC-003, SC-008, SC-010, SC-013, SC-018, SC-020, SC-021, SC-023, FR-EVID-004
  Depends: T035, T045, T053
- [X] T060 Migration: Document rollback in the designated migration/evidence artifact or final implementation notes as removal of `scripts/plot_geodt_branch_tree_comparison.py`, `<focused-test-path>`, and generated diagnostics outputs only; confirm no migration rollback is needed for existing GeoDT artifacts
  Reqs: FR-033, FR-035, FR-EVID-004, SC-009, SC-020
  Depends: T004, T058

T057 is useful for brownfield confidence but is not required for MVP completion because the spec allows synthetic fixture validation and real archive availability is environment-dependent.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1: Setup and Brownfield Scope Boundary**: T001 is the graph root; T002-T005 depend on T001.
- **Phase 2: Foundational Characterization, Test Harness, and Regression Gates**: Depends on Phase 1 and ends at T018A; T018A blocks all behavior-changing implementation.
- **Phase 3: User Story 1 (P1 MVP)**: Depends on T018A and delivers figure-generation mode.
- **Phase 4: User Story 2 (P2)**: Depends on US1 preflight/eligibility foundations and adds audit-only selection.
- **Phase 5: User Story 3 (P3)**: Depends on metadata, audit, scoring, and figure-input provenance from US1/US2.
- **Phase 6: Polish, Documentation, and Brownfield Validation**: Depends on completed desired user stories; T057 is optional and non-blocking.

### User Story Dependencies

- **US1 Generate Branch-Tree Comparison Figure**: Starts after T018A; no dependency on US2 or US3.
- **US2 Audit Candidate Branch Selection**: Starts after US1 eligibility/preflight implementation because audit-only uses the same selection pipeline.
- **US3 Reproduce from Metadata**: Starts after US1 metadata and US2 scoring/audit fields are implemented.

### Brownfield Dependency Rules

- Implementation tasks T024-T034, T041-T044, and T050-T052 must not begin before T018A is complete.
- Any task touching `scripts/plot_geodt_branch_tree_comparison.py` depends on verified allowed change surface and focused tests.
- The diagnostic CLI must remain standalone. It may read existing code or constants only when safe, but it must not mutate `SAVE_DT_RULES`, `SAVE_DT_NODE_DUMP`, `ACTIVE_LAGS`, or other training/runtime configuration values.
- No task edits read-only files listed in `plan.md`: `app/main_model_DT.py`, `src/model/GeoRF_DT.py`, `src/model/model_DT.py`, `src/model/train_branch.py`, `src/helper/helper.py`, `src/merge/terminal.py`, `src/utils/dt_rule_export.py`, or `run_batches_2021_2024_visual_monthly.bat`.
- Existing model artifacts under `result_*`, `checkpoints/`, `space_partitions/`, `dt_rules/`, `other_outputs/`, and `deliverables/` are read-only inputs.
- Completion does not require T057 unless a bounded real archive path/root is explicitly available.

---

## Execution Wave DAG

### Wave 1: Scope and Location Verification

- T001
- Then parallel: T002, T003, T004, T005

### Wave 2: Legacy Evidence Characterization

- T006, T007, T008, T009, T010

### Wave 3: Test Harness Fixtures

- T011
- Then parallel: T012, T013, T014

### Wave 4: Brownfield Regression Gates

- T015, T016, T017, T018
- Then T018A

### Wave 5: US1 Test Specifications

- T019, T020, T021, T022, T023

### Wave 6: US1 Preflight and Artifact Implementation

- Requires T018A complete
- T024
- Then sequential dependency chain with parallel review opportunities: T025, T026, T027, T028, T029, T030, T031

### Wave 7: US1 Selection Pipeline, Figure, Metadata, Failure Handling, and Validation

- Selection pipeline tests: T037, then T038, then T039
- Selection pipeline implementation: T041, then T042, then T043
- Figure and metadata tasks: T032, T033, T034 after T041-T043 and their task-level dependencies
- Then T035, T036

### Wave 8: US2 Audit-Only Contract and Wrapper

- T040 runs only after T035 and T039
- Then T044 wraps the completed shared selection pipeline from T041-T043
- Then T045, T046

### Wave 9: US3 Reproduction Tests and Logic

- T047
- Then parallel: T048, T049
- Then T050
- Then T051
- Then T052
- Then T053, T054

### Wave 10: Final Documentation, Optional Real Archive Smoke, and Boundary Validation

- T055, T056, T058, T059, T060
- Optional/non-blocking: T057, if bounded real archive path/root is available

---

## Parallel Opportunities

- T006-T010 can run in parallel after their Phase 1 dependencies because they inspect separate evidence areas.
- T012-T014 can run in parallel after T011 because they add distinct fixture scenarios in the same focused test file; coordinate edits to avoid conflicts.
- T015-T018 can run in parallel after fixture tasks because they validate separate brownfield boundaries.
- T019-T023 can run in parallel after T018A.
- T037-T039 follow their explicit dependencies; T040 waits for T035 and T039. Do not treat all US2 tests as parallel-ready.
- T048 and T049 can run in parallel after T047.
- Documentation tasks T055-T056 can run in parallel with validation tasks T058-T060 after story validation is complete; T057 can run separately only when a bounded real archive path/root is available.

---

## Implementation Strategy

### MVP First: User Story 1

1. Complete Phase 1 scope/boundary verification.
2. Complete Phase 2 characterization, fixtures, focused regression gates, and T018A.
3. Complete Phase 3 US1 tests and implementation.
4. Stop and validate figure generation on the synthetic fixture before using any real archive.
5. Treat T057 as optional; run a bounded real-archive audit only if an explicit path/root is available.

### Incremental Delivery

1. Deliver US1: preflight plus branch-specific figure and metadata.
2. Deliver US2: audit-only mode using the same selection pipeline without rendering.
3. Deliver US3: metadata-driven reproduction and mismatch reports.
4. Finish with documentation, optional real-archive dry run, and brownfield boundary validation.

### Validation Commands

Use the project Python 3.12 environment documented in `CLAUDE.md` and `AGENTS.md`. Candidate commands to run after implementation tasks reach their validation points:

```bash
python -m pytest <focused-test-path> -p no:cacheprovider
python scripts/plot_geodt_branch_tree_comparison.py --audit-only --archive-path <synthetic-fixture-archive> --output-dir <temporary-diagnostics-output>
python scripts/plot_geodt_branch_tree_comparison.py --archive-path <synthetic-fixture-archive> --output-dir <temporary-diagnostics-output>
python scripts/plot_geodt_branch_tree_comparison.py --reproduce-from <temporary-diagnostics-output>/<metadata.json>
```

Replace `<focused-test-path>` with the path selected by T002 and documented in the designated evidence/audit artifact. If pytest is unavailable in the active environment, record the environment gap separately from feature regressions in the validation summary or evidence artifact and run the narrowest available Python smoke command instead.

---

## Requirement Traceability Matrix

| Requirement | Primary Task IDs | Notes |
|---|---|---|
| FR-000-FR-000N | T009, T013, T018, T020, T025, T026, T027, T034, T035 | Preflight, archive discovery, artifact-provider compatibility, minimum artifact completeness, failure handling |
| FR-001-FR-003B | T019, T024, T025, T035, T036 | Diagnostic creation, preferred/fallback run selection |
| FR-004-FR-006C | T007, T008, T021, T027, T028, T030, T031 | Checkpoint discovery, branch assignment, eligibility |
| FR-006D-FR-006H | T010, T022, T029, T035 | Feature-name source precedence and validation |
| FR-CHK-001-FR-CHK-005 | T007, T021, T030, T035 | Checkpoint parsing and classification |
| FR-007-FR-009 | T006, T015, T027, T030, T035 | Root/global exclusion and branch checkpoint use |
| FR-010-FR-016H | T037, T038, T039, T041, T042, T043, T045 | Signature extraction, scoring, readability, tie-break |
| FR-017-FR-017D | T021, T031, T035 | Branch coverage counts |
| FR-018-FR-030 | T023, T032, T035, T036 | Figure generation and visual requirements |
| FR-AUDIT-001-FR-AUDIT-004 | T040, T044, T045, T046 | Audit-only mode |
| FR-REPRO-001-FR-REPRO-006 | T047, T048, T049, T050, T051, T052, T053, T054 | Reproduction mode |
| FR-OUT-001-FR-OUT-004 | T003, T016, T034, T055, T056 | Output locations, no overwrite, output path recording |
| FR-031-FR-036B | T033, T034, T035, T047, T052, T059 | Metadata, provenance, failure handling |
| FR-EVID-001-FR-EVID-004 | T002, T005, T006, T008, T009, T010, T018, T018A, T026, T033, T057, T059, T060 | Evidence validation and status recording |
| SC-001-SC-012 | T011, T012, T019, T023, T032, T035, T036, T037-T045, T047-T054 | Figure, scoring, reproduction, readability |
| SC-013-SC-023 | T009, T015-T018, T020-T023, T033-T035, T044-T046, T052-T060 | Preflight, bounded discovery, evidence, failure, output validation |
| US1 | T019-T036 | Figure-generation MVP |
| US2 | T037-T046 | Audit-only branch selection |
| US3 | T047-T054 | Reproduction from metadata |

---

## Task Plan Self-Audit Criteria

Before implementation starts, verify:

- Every task T001 through T060 plus T018A has a `Reqs:` line.
- Every task T001 through T060 plus T018A has a `Depends:` line.
- Every task except T001 has at least one dependency, unless explicitly marked Optional and non-blocking.
- Every requirement group in the traceability matrix has at least one primary task.
- Every story has tests, implementation tasks, and validation tasks.
- No implementation task touching `scripts/plot_geodt_branch_tree_comparison.py` can start before T018A.
- T057 is marked optional and non-blocking.
- No task requires editing read-only GeoDT training, prediction dispatch, branch training, or existing `dt_rules` export files.
- No task instructs agents to write runtime findings directly into `tasks.md`; runtime findings use the Artifact-of-Record Policy.
- Test tasks use `<focused-test-path>` until T002 selects and documents the focused test path.

---

## Readiness Gate

This task plan is ready for implementation only when:

- T001-T018A are complete;
- every task has `Reqs:` and `Depends:`;
- the requirement traceability matrix has no empty requirement groups;
- no implementation task is unblocked before T018A;
- optional T057 is not treated as required for MVP completion;
- T002, T009, and T010 outputs are written to the designated evidence/audit artifact;
- the diagnostic CLI is constrained not to mutate training/runtime config values;
- fixture helpers from T011/T012 can support CLI subprocess smoke tests.

Expected readiness after these corrections: audit-ready for implementation.

---

## Notes

- Do not implement code in read-only GeoDT training, prediction dispatch, branch training, or existing `dt_rules` export files.
- Do not add a batch launcher unless a future spec revision authorizes launcher integration.
- Do not search arbitrary user directories or filesystem root during archive discovery.
- Do not use `dt_rules_*.csv` as a fallback source for branch-specific local tree rules.
- Do not label classes as crisis/non-crisis unless class-label mapping evidence is confirmed and recorded.
- Keep generated figures, metadata, audit summaries, reproduction reports, failure summaries, caches, and fixtures out of production artifact directories unless explicitly configured as diagnostics output.
