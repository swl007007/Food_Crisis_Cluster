# Implementation Micro-Plan: GeoDT Branch-Specific Local Tree Interpretability Figure

## Artifact Inputs

| Artifact | Path | Status | Notes |
|---|---|---|---|
| spec.md | `specs/005-geodt-branch-tree-interpretability/spec.md` | present | Source for requirements, acceptance criteria, non-goals, and Brownfield preservation rules. |
| plan.md | `specs/005-geodt-branch-tree-interpretability/plan.md` | present | Source for architecture, change surface, test strategy, and allowed/read-only files. |
| tasks.md | `specs/005-geodt-branch-tree-interpretability/tasks.md` | present | Primary task graph and requirement mapping. Task-level `Depends:` lines are authoritative. |
| analyze findings / resolutions | conversation analysis report | present | Latest analysis found no blocking inconsistencies after B1-B10 remediation; T001-T018A remain required before implementation. |
| evidence pack | `specs/_evidence/005-geodt-branch-tree-interpretability.evidence.md` | present | Evidence source only; current `spec.md` References are authoritative and stale evidence recommendations must not override current artifacts. |
| supplementary docs | `research.md`, `data-model.md`, `quickstart.md`, `contracts/geodt-branch-tree-diagnostic-cli.md` | present | Source for CLI contract, examples, entities, and research decisions. |
| existing micro-plan | `implementation-micro-plan.md` | absent before generation | This generated file is historical output only on future regenerations. |

## Source-of-Truth Order

1. Resolved analyze findings / human resolution notes
2. `tasks.md`
3. `plan.md`
4. `spec.md`
5. Evidence pack / repo evidence
6. Supplementary Speckit docs
7. This micro-plan

## Conflict Policy

If this micro-plan conflicts with source artifacts, stop and report the conflict. Do not silently resolve conflicts by changing task intent, adding behavior absent from the spec, or overriding `tasks.md` dependencies.

## Brownfield Enforcement

This micro-plan additionally enforces:

- No opportunistic refactor.
- No dependency upgrade unless explicitly in scope; none is currently in scope.
- No broad formatting or unrelated cleanup.
- No editing candidate paths until verified by T001-T003 and T002's focused-test-path selection.
- No changing public API / CLI / config / data artifact behavior unless explicitly authorized by current Speckit artifacts.
- Characterization, smoke, contract, or focused regression tests before behavior-changing edits in legacy areas.
- Known baseline failures must be separated from new regressions in validation/evidence outputs.
- Risky changes require rollback/revert notes; for this feature rollback is deletion of the new diagnostic script, focused tests, and generated diagnostics outputs only.
- Existing GeoDT training, prediction dispatch, branch training, batch launchers, `dt_rules` export behavior, checkpoints, partition artifacts, deliverables, and evaluation outputs are read-only.

## Code Block Policy

The micro-plan may describe the smallest implementation change, but must not include speculative production code blocks. Use discovery steps or tests when exact code cannot be grounded.

## TDD Policy Summary

`superpowers:test-driven-development` was loaded before generating this plan.

- Behavior-changing code requires strict TDD unless a per-task exception is explicitly approved by the user or recorded in resolved human notes.
- Bugfixes require a reproducing failing test first.
- Refactors require existing coverage or a characterization test first.
- CLI/API behavior requires a contract or focused smoke test first.
- Data/artifact behavior requires a schema, path, row-count, fixture, metadata, or deterministic artifact check first.
- Docs / non-behavioral config / deterministic generated artifacts may skip strict TDD only with a per-task validation-only exception statement.
- Behavior-changing implementation tasks T024-T034, T041-T044, and T050-T052 must not start before T018A.

## Blocked Status

- Status: not blocked to start T001.
- Missing required planning artifacts: none currently known.
- Runtime/archive artifacts: pending characterization through T009 and T010.
- Focused test path: pending T002.
- Blocking conflicts: none currently known in the planning artifacts.
- Required human input: none before executing T001. T057 requires an explicit bounded real archive path/root if the optional real-archive dry run is attempted.

## Warnings and Degraded Confidence

- `specs/005-geodt-branch-tree-interpretability/` and `specs/_evidence/` were untracked in the repository baseline before this micro-plan was written.
- Existing local archive completeness is unknown. Mitigation: T009 and T010 must record local archive and feature-source findings before implementation.
- The focused test location is intentionally unknown until T002. Downstream tasks must use `<focused-test-path>` until T002 writes the selected path to the designated evidence/audit artifact.

## Repository Baseline

Baseline before writing this file:

```text
?? specs/005-geodt-branch-tree-interpretability/
?? specs/_evidence/
```

Post-write verification must confirm the only newly introduced file from this micro-plan step is:

```text
specs/005-geodt-branch-tree-interpretability/implementation-micro-plan.md
```

Because the feature directory was already untracked at baseline, `git status --short` may continue to show the directory-level untracked entry rather than this file individually.

## Execution Waves

### Wave 1: Scope and Location Verification

Safe to run in parallel: no for T001; yes for T002-T005 after T001.  
Prerequisites: none.

Tasks:
- T001 Verify active feature artifacts.
- T002 Select focused test path.
- T003 Verify diagnostic script candidate path.
- T004 Confirm no migration is required.
- T005 Confirm Speckit context pointers and stale-evidence handling.

### Wave 2: Legacy Evidence Characterization

Safe to run in parallel: T006-T009 can run in parallel after their Phase 1 dependencies. T010 can run in parallel only when explicit archive/source paths are already available; otherwise it follows T009.  
Prerequisites: T001 plus T005 for T006, T003 for T009.

Tasks:
- T006 Root/global `dt_rules` export behavior.
- T007 GeoDT checkpoint naming and loader behavior.
- T008 Branch assignment semantics.
- T009 Bounded local archive/source-folder availability.
- T010 Feature-name source candidates. T010 may run after T001 only if explicit archive/source paths are already provided; otherwise it waits for T009.

### Wave 3: Test Harness Fixtures

Safe to run in parallel: T012-T014 after T011, with edit coordination in `<focused-test-path>`.  
Prerequisites: T001, T002.

Tasks:
- T011 Fixture builder scaffolding.
- T012 Complete synthetic GeoDT-like archive fixture.
- T013 Visual archive plus same-run artifact-provider fixture.
- T014 Negative fixtures.

### Wave 4: Brownfield Regression Gates

Safe to run in parallel: T015-T018 after dependencies; T018A is sequential gate.  
Prerequisites: Wave 2 and Wave 3 dependencies.

Tasks:
- T015 Root/global-only `dt_rules` incomplete regression.
- T016 No-overwrite regression.
- T017 Bounded archive-discovery regression.
- T018 Same-run artifact-provider compatibility regression.
- T018A Foundation gate.

### Wave 5: US1 Test Specifications

Safe to run in parallel: yes after T018A.  
Prerequisites: T018A plus task-specific fixture dependencies.

Tasks:
- T019 Figure-generation CLI contract and config non-mutation tests.
- T020 Preflight/fallback/failure tests.
- T021 Checkpoint classification and branch eligibility tests.
- T022 Feature-name precedence and split-index bounds tests.
- T023 Figure-rendering smoke tests.

### Wave 6: US1 Preflight and Artifact Implementation

Safe to run in parallel: no; follow task-level dependencies.  
Prerequisites: T018A and US1 RED tests.

Tasks:
- T024 CLI skeleton.
- T025 Bounded archive resolver and fallback decisions.
- T026 Same-run/source artifact-provider compatibility.
- T027 Artifact characterizer.
- T028 Branch assignment source selector.
- T029 Feature-name source selector.
- T030 Checkpoint parser/loader.
- T031 Branch eligibility builder.

### Wave 7: US1 Selection Pipeline, Figure, Metadata, Failure Handling, and Validation

Safe to run in parallel: no where dependencies chain; selection pipeline must complete before figure rendering.  
Prerequisites: T031 and US1/selection test tasks.

Tasks:
- T037 Signature extraction tests.
- T038 Jaccard scoring tests.
- T039 Tie-break/readability tests.
- T041 Signature extraction implementation.
- T042 Scoring/tie-break implementation.
- T043 Readability gate implementation.
- T032 Figure renderer.
- T033 Metadata JSON writer.
- T034 No-overwrite output handling and failure summary.
- T035 Focused US1 validation.
- T036 Synthetic figure-generation CLI smoke.

### Wave 8: US2 Audit-Only Contract and Wrapper

Safe to run in parallel: no; audit-only mode wraps the selection pipeline implemented by US1.  
Prerequisites: T035 and completed selection pipeline T041-T043.

Tasks:
- T040 Audit-only contract tests.
- T044 Audit-only mode implementation.
- T045 US2 validation.
- T046 Synthetic audit-only smoke.

### Wave 9: US3 Reproduction Tests and Logic

Safe to run in parallel: T048 and T049 only after T047.  
Prerequisites: T033 and T045 as specified.

Tasks:
- T047 Reproduction metadata schema tests.
- T048 Reproduction mismatch tests.
- T049 `--reproduce-from` CLI contract tests.
- T050 Reproduction metadata loading and resolution.
- T051 Reproduction verification.
- T052 Reproduction report and mismatch failure summary.
- T053 US3 validation.
- T054 Reproduction CLI smoke.

### Wave 10: Final Documentation, Optional Real Archive Smoke, and Boundary Validation

Safe to run in parallel: documentation and validation can run in parallel after story validation; T060 waits for T004 and T058.  
Prerequisites: T036, T046, T054; T057 also requires explicit bounded real archive path/root.

Tasks:
- T055 Quickstart update if needed.
- T056 CLI contract update if needed.
- T057 Optional real-archive audit-only dry run.
- T058 Source-boundary check.
- T059 Focused tests and baseline/new regression summary.
- T060 Rollback/no-migration documentation.

## Task Micro-Plans

### Shared Command Conventions

Replace placeholders only after the prerequisite task resolves them:

- `<focused-test-path>`: path selected by T002.
- `<synthetic-fixture-archive>`: fixture path returned or printed by helpers from T011/T012.
- `<temporary-diagnostics-output>`: temporary output directory created by pytest/tmp fixture or standalone smoke command.
- `<metadata.json>`: metadata generated by successful fixture run.

Candidate validation commands:

```bash
python -m pytest <focused-test-path> -p no:cacheprovider
python scripts/plot_geodt_branch_tree_comparison.py --audit-only --archive-path <synthetic-fixture-archive> --output-dir <temporary-diagnostics-output>
python scripts/plot_geodt_branch_tree_comparison.py --archive-path <synthetic-fixture-archive> --output-dir <temporary-diagnostics-output>
python scripts/plot_geodt_branch_tree_comparison.py --reproduce-from <temporary-diagnostics-output>/<metadata.json>
```

If pytest is unavailable, record the environment gap separately from feature regressions and use the narrowest runnable Python smoke command.

### T001: Verify active feature artifacts

Source:
- Requirement IDs: FR-000, FR-000D, FR-031, SC-013
- Plan section: Change Surface; Project Structure

Dependencies:
- none

Files:
- Inspect: `specs/005-geodt-branch-tree-interpretability/spec.md`, `plan.md`, `tasks.md`, `contracts/geodt-branch-tree-diagnostic-cli.md`
- Edit: none
- Tests: none

TDD classification:
- validation-only exception

TDD exception:
- Required: yes
- Reason: Artifact presence verification has no runtime behavior change.
- Why strict TDD does not apply: No production or test behavior is being implemented.
- Validation method: Inspect/read required artifacts and record result in task-run/evidence notes.
- Command or review criterion: confirm all listed files exist and align with feature `005-geodt-branch-tree-interpretability`.
- Risk: Low; wrong feature selection would invalidate all downstream work.
- Approval source: User invoked micro-plan generation and tasks.md classifies T001 as characterization.

RED:
- Write/update: none
- Command: `test -f "specs/005-geodt-branch-tree-interpretability/spec.md" && test -f "specs/005-geodt-branch-tree-interpretability/plan.md" && test -f "specs/005-geodt-branch-tree-interpretability/tasks.md" && test -f "specs/005-geodt-branch-tree-interpretability/contracts/geodt-branch-tree-diagnostic-cli.md"`
- Expected failure: only if an artifact is missing or feature path drift recurs.

GREEN:
- Smallest implementation change: none; record characterization result in evidence/task-run notes.
- Command: same as RED command.
- Expected passing result: command exits 0.

REFACTOR:
- Allowed cleanup: none.
- Command: not applicable.

Validation:
- Commands: same as RED command.
- Expected result: all required artifacts present.

Done criteria:
- [ ] Required artifacts are present.
- [ ] Any mismatch is reported before downstream tasks.
- [ ] No files are modified except permitted evidence/task-run notes.

Commit suggestion:
- message: `verify geodt branch tree feature artifacts`

### T002: Select focused test path

Source:
- Requirement IDs: FR-AUDIT-004, FR-EVID-004, SC-018, SC-020
- Plan section: Test Strategy; Project Structure

Dependencies:
- T001

Files:
- Inspect: `src/tests/`, `tests/`, `AGENTS.md`, `CLAUDE.md`
- Edit: designated evidence/audit artifact only; no test file until path selected
- Tests: selected `<focused-test-path>` after this task

TDD classification:
- characterization test / validation-only exception

TDD exception:
- Required: yes
- Reason: This task chooses a test location; it does not implement behavior.
- Why strict TDD does not apply: No code behavior is changed.
- Validation method: Document selected test path and rationale in the designated evidence/audit artifact.
- Command or review criterion: list `src/tests/` and `tests/`, choose path consistent with observed conventions.
- Risk: Medium; wrong path causes downstream test discovery friction.
- Approval source: tasks.md T002 explicitly requires path selection before test creation.

RED:
- Write/update: none
- Command: `ls "src/tests" "tests"`
- Expected failure: acceptable if one candidate directory is absent; the failure documents available test layout.

GREEN:
- Smallest implementation change: record `<focused-test-path>` and rationale in the designated evidence/audit artifact or task-run notes.
- Command: review the recorded path.
- Expected passing result: downstream tasks have a concrete test path.

REFACTOR:
- Allowed cleanup: none.
- Command: not applicable.

Validation:
- Commands:
  - Verify that the designated evidence/audit artifact or task-run note contains a concrete focused test path and rationale.
  - If using an environment variable during execution: `test -n "$FOCUSED_TEST_PATH" && test "$FOCUSED_TEST_PATH" != "<focused-test-path>"`
- Expected result: selected path exists or parent path is intentionally selected for new test file creation, and the literal placeholder `<focused-test-path>` is not treated as a resolved path.

Done criteria:
- [ ] Focused test path recorded outside `tasks.md`.
- [ ] Downstream tasks can replace `<focused-test-path>`.
- [ ] The literal placeholder `<focused-test-path>` is not treated as a resolved path.
- [ ] No candidate path is edited before this task completes.

Commit suggestion:
- message: `record focused diagnostic test path`

### T003: Verify diagnostic script candidate path

Source:
- Requirement IDs: FR-035, FR-OUT-003, SC-009, SC-019
- Plan section: Change Surface

Dependencies:
- T001

Files:
- Inspect: `scripts/plot_geodt_branch_tree_comparison.py`, `scripts/`
- Edit: designated evidence/audit artifact only
- Tests: none

TDD classification:
- characterization / validation-only exception

TDD exception:
- Required: yes
- Reason: Candidate path verification only.
- Why strict TDD does not apply: No runtime behavior change.
- Validation method: Inspect whether candidate script exists and record status.
- Command or review criterion: `ls "scripts/plot_geodt_branch_tree_comparison.py"` and inspect if present.
- Risk: Medium; editing an unrelated existing script would violate Brownfield constraints.
- Approval source: tasks.md T003.

RED:
- Write/update: none
- Command: `ls "scripts/plot_geodt_branch_tree_comparison.py"`
- Expected failure: expected if the script does not exist yet; this confirms T024 will create it rather than edit an unknown file.

GREEN:
- Smallest implementation change: record absent/present/unrelated/superseded status in evidence/task-run notes.
- Command: same inspection command.
- Expected passing result: status is documented.

REFACTOR:
- Allowed cleanup: none.
- Command: not applicable.

Validation:
- Commands: review evidence/task-run note.
- Expected result: T024 has verified edit/create target.

Done criteria:
- [ ] Candidate path status recorded.
- [ ] No script is created or edited.

Commit suggestion:
- message: `verify diagnostic script path`

### T004: Confirm no migration is required

Source:
- Requirement IDs: FR-033, FR-035, SC-009
- Plan section: Migration / rollback requirements

Dependencies:
- T001

Files:
- Inspect: `plan.md`, `spec.md`, `tasks.md`
- Edit: designated migration/evidence artifact or final implementation notes
- Tests: none

TDD classification:
- validation-only exception

TDD exception:
- Required: yes
- Reason: Documentation of no migration requirement.
- Why strict TDD does not apply: No runtime behavior change.
- Validation method: Confirm no data/config/dependency/production-artifact migration is in scope.
- Command or review criterion: review source artifacts for migration scope.
- Risk: Low, but missing rollback note weakens Brownfield safety.
- Approval source: tasks.md T004 and user Brownfield instructions.

RED:
- Write/update: none
- Command: review `plan.md` Change Surface.
- Expected failure: not applicable; this is review-only.

GREEN:
- Smallest implementation change: record no-migration confirmation.
- Command: review recorded note.
- Expected passing result: migration status and rollback basis are documented.

REFACTOR:
- Allowed cleanup: none.
- Command: not applicable.

Validation:
- Commands: review note for no data/config/dependency migration.
- Expected result: no migration required.

Done criteria:
- [ ] No-migration status recorded.
- [ ] Risky change rollback basis is available for T060.

Commit suggestion:
- message: `document no migration requirement`

### T005: Confirm context pointers and stale-evidence handling

Source:
- Requirement IDs: FR-EVID-001, FR-EVID-004, SC-020
- Plan section: References; Evidence Validation Plan

Dependencies:
- T001

Files:
- Inspect: `CLAUDE.md`, `AGENTS.md`, `specs/_evidence/005-geodt-branch-tree-interpretability.evidence.md`, `spec.md`
- Edit: evidence/task-run notes only unless a future explicit task authorizes doc edits
- Tests: none

TDD classification:
- documentation characterization / validation-only exception

TDD exception:
- Required: yes
- Reason: Context-pointer review has no runtime behavior change.
- Why strict TDD does not apply: It validates source-of-truth alignment.
- Validation method: Confirm no active pointer references a different 005 feature and stale evidence recommendations are historical.
- Command or review criterion: inspect named docs.
- Risk: Medium; stale context can mislead implementation agents.
- Approval source: tasks.md T005.

RED:
- Write/update: none
- Command: inspect named files.
- Expected failure: only if a stale pointer references a wrong feature path.

GREEN:
- Smallest implementation change: record findings in evidence/task-run notes.
- Command: review recorded findings.
- Expected passing result: current plan/spec references are authoritative.

REFACTOR:
- Allowed cleanup: none.
- Command: not applicable.

Validation:
- Commands: review note for `CLAUDE.md`, `AGENTS.md`, stale evidence treatment.
- Expected result: no source-of-truth conflict remains.

Done criteria:
- [ ] Context pointer status recorded.
- [ ] Stale evidence recommendation status recorded.
- [ ] No Speckit artifact is changed unless separately authorized.

Commit suggestion:
- message: `verify geodt branch tree context pointers`

### T006: Characterize root/global dt_rules export behavior

Source:
- Requirement IDs: FR-008, FR-032, FR-EVID-002, FR-EVID-004, SC-002, SC-016, SC-020
- Plan section: Evidence Validation Plan

Dependencies:
- T001, T005

Files:
- Inspect: `app/main_model_DT.py`, `src/utils/dt_rule_export.py`, `config.py`
- Edit: evidence/task-run notes only
- Tests: none yet; T015 encodes regression

TDD classification:
- characterization

TDD exception:
- Required: yes
- Reason: Read-only evidence gathering before regression test.
- Why strict TDD does not apply: No code behavior is changed in this task.
- Validation method: Record evidence that current `dt_rules` are root/global-only or mark unknown.
- Command or review criterion: inspect export path and config flags.
- Risk: High if mischaracterized; T015 must lock this boundary before implementation.
- Approval source: tasks.md T006.

RED:
- Write/update: none
- Command: inspect files for `dt_rules`, `SAVE_DT_RULES`, and root/global branch load behavior.
- Expected failure: not applicable; this is evidence collection.

GREEN:
- Smallest implementation change: record confirmed/rejected/unknown status for root/global `dt_rules` boundary.
- Command: review evidence note.
- Expected passing result: T015 can turn finding into regression.

REFACTOR:
- Allowed cleanup: none.
- Command: not applicable.

Validation:
- Commands: review evidence status for FR-EVID-002 item 1.
- Expected result: assumption status is recorded.

Done criteria:
- [ ] Root/global `dt_rules` evidence status recorded.
- [ ] No read-only source file modified.

Commit suggestion:
- message: `record dt rules boundary evidence`

### T007: Characterize checkpoint naming and loader behavior

Source:
- Requirement IDs: FR-004, FR-CHK-001-FR-CHK-003, FR-CHK-005, FR-EVID-002, SC-013, SC-020
- Plan section: Checkpoint Parsing Plan

Dependencies:
- T001

Files:
- Inspect: `src/model/model_DT.py`, `src/model/GeoRF_DT.py`, `src/model/train_branch.py`
- Edit: evidence/task-run notes only
- Tests: T021/T030 later

TDD classification:
- characterization

TDD exception:
- Required: yes
- Reason: Read-only legacy evidence gathering.
- Why strict TDD does not apply: No behavior change.
- Validation method: Record checkpoint naming, loader, and branch-ID parsing evidence.
- Command or review criterion: inspect named files and branch-id handling.
- Risk: High; filename-only inference is forbidden.
- Approval source: tasks.md T007.

RED:
- Write/update: none
- Command: inspect named files for checkpoint path/load/save behavior.
- Expected failure: not applicable.

GREEN:
- Smallest implementation change: record confirmed/rejected/unknown status.
- Command: review evidence note.
- Expected passing result: T021/T030 can use grounded expectations.

REFACTOR:
- Allowed cleanup: none.
- Command: not applicable.

Validation:
- Commands: review evidence note for checkpoint parsing assumptions.
- Expected result: parsing method evidence recorded.

Done criteria:
- [ ] Checkpoint naming and loader evidence recorded.
- [ ] No read-only source file modified.

Commit suggestion:
- message: `record geodt checkpoint loader evidence`

### T008: Characterize branch assignment semantics

Source:
- Requirement IDs: FR-006B, FR-EVID-002, FR-EVID-004, SC-013, SC-020
- Plan section: Branch Assignment Source Plan

Dependencies:
- T001

Files:
- Inspect: `src/helper/helper.py`, `src/merge/terminal.py`
- Edit: evidence/task-run notes only
- Tests: T021/T028 later

TDD classification:
- characterization

TDD exception:
- Required: yes
- Reason: Read-only evidence gathering.
- Why strict TDD does not apply: No behavior change.
- Validation method: Record semantics and uncertainty for `X_branch_id.npy`, `s_branch.pkl`, correspondence tables.
- Command or review criterion: inspect branch mapping functions and terminal table fields.
- Risk: High; assignment source precedence depends on this evidence.
- Approval source: tasks.md T008.

RED:
- Write/update: none
- Command: inspect named files for branch assignment behavior.
- Expected failure: not applicable.

GREEN:
- Smallest implementation change: record assumption status.
- Command: review evidence note.
- Expected passing result: T028 can implement selector from documented precedence.

REFACTOR:
- Allowed cleanup: none.
- Command: not applicable.

Validation:
- Commands: review evidence note for FR-EVID-002 items 2-4.
- Expected result: assignment semantics status recorded.

Done criteria:
- [ ] Branch assignment source semantics recorded.
- [ ] No read-only source file modified.

Commit suggestion:
- message: `record branch assignment evidence`

### T009: Characterize bounded local archive/source folders

Source:
- Requirement IDs: FR-000F-H, FR-000I, FR-000L, FR-003A-B, FR-EVID-004, SC-014, SC-017, SC-020, SC-022
- Plan section: Data and Artifact Discovery Strategy

Dependencies:
- T001, T003

Files:
- Inspect: explicit archive path/list/root selected for first smoke run; do not inspect arbitrary directories
- Edit: evidence/audit artifact only
- Tests: T017/T020 later

TDD classification:
- characterization

TDD exception:
- Required: yes
- Reason: Archive availability discovery is pre-implementation evidence gathering.
- Why strict TDD does not apply: No production behavior change.
- Validation method: Record bounded discovery input, candidates, missing artifacts, fallback observations.
- Command or review criterion: use only explicit/configured bounded root; never filesystem root or unrelated user directories.
- Risk: High; unbounded search violates spec.
- Approval source: tasks.md T009 and user Brownfield command args.

RED:
- Write/update: none
- Command: bounded `find`/listing under explicit archive input only.
- Expected failure: expected if no archive input/root exists; record failure reason instead of guessing.

GREEN:
- Smallest implementation change: record archive availability and fallback observations.
- Command: review evidence/audit artifact.
- Expected passing result: T017/T020 can test bounded behavior using fixtures.

REFACTOR:
- Allowed cleanup: none.
- Command: not applicable.

Validation:
- Commands: review evidence for discovery input, candidate count, missing artifacts.
- Expected result: bounded archive status recorded.

Done criteria:
- [ ] No unbounded filesystem search performed.
- [ ] Archive/source folder findings recorded outside `tasks.md`.

Commit suggestion:
- message: `record bounded archive discovery evidence`

### T010: Characterize feature-name sources

Source:
- Requirement IDs: FR-006D-H, FR-EVID-002, FR-EVID-004, SC-020, SC-023
- Plan section: Feature-Name Source Plan

Dependencies:
- T001
- T009, unless explicit archive/source paths were provided before T010 begins

Files:
- Inspect: candidate selected-run feature-name artifacts only after bounded archive/source paths are known
- Edit: evidence/audit artifact only
- Tests: T022/T029 later

TDD classification:
- characterization

TDD exception:
- Required: yes
- Reason: Feature-source discovery before implementation.
- Why strict TDD does not apply: No behavior change.
- Validation method: Record source candidates, compatibility, feature-count evidence, run/scope tie.
- Command or review criterion: inspect selected archive/source candidates only.
- Risk: High; wrong feature names make figure misleading.
- Approval source: tasks.md T010.

RED:
- Write/update: none
- Command: bounded inspection of selected archive/source candidates.
- Expected failure: acceptable if no compatible source exists; record failure reason.

GREEN:
- Smallest implementation change: record selected/rejected feature-source evidence or unknowns.
- Command: review evidence/audit note.
- Expected passing result: T022/T029 have expected precedence and mismatch cases.

REFACTOR:
- Allowed cleanup: none.
- Command: not applicable.

Validation:
- Commands: review evidence for FR-EVID-002 item 6 and FR-006D-H.
- Expected result: feature-name source status recorded.

Done criteria:
- [ ] Feature-source candidates and compatibility recorded.
- [ ] If T010 ran before T009, the explicit archive/source path used for feature-source inspection is recorded in the designated evidence/audit artifact.
- [ ] No unverified path edited.

Commit suggestion:
- message: `record geodt feature source evidence`

### T011: Create synthetic fixture builder scaffolding

Source:
- Requirement IDs: FR-AUDIT-004, SC-001, SC-010, SC-018
- Plan section: Test Strategy; Fixture-based tests

Dependencies:
- T001, T002

Files:
- Inspect: `<focused-test-path>` parent selected by T002
- Edit: `<focused-test-path>`
- Tests: `<focused-test-path>`

TDD classification:
- test harness; validation-first

TDD exception:
- Required: yes
- Reason: This creates test scaffolding rather than production behavior.
- Why strict TDD does not apply: It is test infrastructure, not product code.
- Validation method: Run the focused test file and ensure fixture helper can create/return/print archive and output paths.
- Command or review criterion: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Risk: Medium; weak fixture interface can block CLI subprocess smoke tests.
- Approval source: tasks.md T011.

RED:
- Write/update: `<focused-test-path>` helper test for fixture path creation.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: helper/scaffold absent or placeholder assertion fails until fixture scaffolding exists.

GREEN:
- Smallest implementation change: add fixture builder scaffolding returning or printing temporary archive/output paths.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected passing result: scaffolding test passes.

REFACTOR:
- Allowed cleanup: only test helper naming/duplication inside `<focused-test-path>`.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`

Validation:
- Commands: focused pytest command.
- Expected result: fixture helper usable by subprocess tests.

Done criteria:
- [ ] Fixture path/helper is available for CLI tests.
- [ ] No production code is edited.

Commit suggestion:
- message: `add geodt diagnostic fixture scaffolding`

### T012: Add complete synthetic GeoDT-like archive fixture

Source:
- Requirement IDs: FR-000I, FR-006A-B, FR-006D, FR-CHK-001, SC-001, SC-013, SC-017
- Plan section: Fixture-based tests

Dependencies:
- T011

Files:
- Inspect: `<focused-test-path>`
- Edit: `<focused-test-path>`
- Tests: `<focused-test-path>`

TDD classification:
- test harness

TDD exception:
- Required: yes
- Reason: Fixture data creation only.
- Why strict TDD does not apply: It supports later behavior tests and does not change production runtime.
- Validation method: Focused pytest verifies created fixture contains branch checkpoints, `space_partitions/`, feature names, and correspondence variants.
- Command or review criterion: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Risk: Medium; fixture must represent enough behavior for CLI subprocesses.
- Approval source: tasks.md T012.

RED:
- Write/update: fixture-structure assertion in `<focused-test-path>`.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: complete fixture artifacts are absent.

GREEN:
- Smallest implementation change: extend fixture builder with complete synthetic GeoDT-like archive artifacts.
- Command: focused pytest command.
- Expected passing result: fixture structure and subprocess usability assertions pass.

REFACTOR:
- Allowed cleanup: test helper duplication only.
- Command: focused pytest command.

Validation:
- Commands: focused pytest command.
- Expected result: complete fixture can be used by CLI tests.

Done criteria:
- [ ] Fixture contains minimum required artifact set.
- [ ] Fixture path can be consumed by subprocess tests.

Commit suggestion:
- message: `add complete geodt diagnostic fixture`

### T013: Add same-run artifact-provider fixture

Source:
- Requirement IDs: FR-000F, FR-000I, FR-000M-N, FR-003A-B, FR-EVID-004, SC-014, SC-020
- Plan section: Data and Artifact Discovery Strategy

Dependencies:
- T011

Files:
- Edit: `<focused-test-path>`
- Tests: `<focused-test-path>`

TDD classification:
- test harness

TDD exception:
- Required: yes
- Reason: Fixture scenario creation only.
- Why strict TDD does not apply: No production behavior change.
- Validation method: Focused pytest checks visual archive lacks artifacts and matching provider supplies them with compatibility evidence fields.
- Command or review criterion: focused pytest.
- Risk: Medium; provider compatibility is central to clarified spec.
- Approval source: tasks.md T013.

RED:
- Write/update: fixture assertion for visual archive + provider pair.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: provider fixture is absent.

GREEN:
- Smallest implementation change: add fixture pair with run identity evidence.
- Command: focused pytest.
- Expected passing result: fixture pair exists and exposes both paths.

REFACTOR:
- Allowed cleanup: helper reuse inside tests.
- Command: focused pytest.

Validation:
- Commands: focused pytest.
- Expected result: provider fixture supports T018/T026 tests.

Done criteria:
- [ ] Visual archive/provider fixture exists.
- [ ] Both paths can be passed to CLI subprocess tests.

Commit suggestion:
- message: `add geodt artifact provider fixture`

### T014: Add negative fixtures

Source:
- Requirement IDs: FR-000C, FR-000L, FR-006F-G, FR-016E, FR-036A, FR-REPRO-006, SC-008, SC-021
- Plan section: Fixture-based tests; Failure Handling Strategy

Dependencies:
- T011

Files:
- Edit: `<focused-test-path>`
- Tests: `<focused-test-path>`

TDD classification:
- test harness

TDD exception:
- Required: yes
- Reason: Negative fixture data only.
- Why strict TDD does not apply: No production behavior change.
- Validation method: Focused pytest verifies fixture variants can be created.
- Command or review criterion: focused pytest.
- Risk: Medium; missing negative fixtures weakens Brownfield failure guarantees.
- Approval source: tasks.md T014.

RED:
- Write/update: assertions for negative fixture variants.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: variants absent.

GREEN:
- Smallest implementation change: add root/global-only, feature mismatch, out-of-bounds, unreadable, and missing reproduction artifact fixture variants.
- Command: focused pytest.
- Expected passing result: variants created deterministically.

REFACTOR:
- Allowed cleanup: fixture helper factoring only.
- Command: focused pytest.

Validation:
- Commands: focused pytest.
- Expected result: all negative fixtures available for later RED tests.

Done criteria:
- [ ] Negative fixtures cover all listed cases.
- [ ] No production artifacts used or modified.

Commit suggestion:
- message: `add geodt diagnostic negative fixtures`

### T015: Add root/global-only dt_rules regression

Source:
- Requirement IDs: FR-008, FR-000L, FR-032, FR-EVID-002, SC-002, SC-016, SC-020
- Plan section: Test Strategy item 4

Dependencies:
- T006, T012, T014

Files:
- Edit: `<focused-test-path>`
- Production edit: none

TDD classification:
- focused regression test

RED:
- Write/update: test asserting root/global-only `dt_rules` archive is incomplete and never used as branch-tree source.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: diagnostic implementation/import/CLI behavior does not exist yet, or currently cannot reject fixture.

GREEN:
- Smallest implementation change: none in this task; leave failing test for T027/T034 if implementation is absent.
- Command: focused pytest.
- Expected passing result: may remain failing until implementation tasks; record as expected RED gate.

REFACTOR:
- Allowed cleanup: test naming only after green.
- Command: focused pytest.

Validation:
- Commands: focused pytest.
- Expected result: fails before implementation for expected missing behavior, later passes after implementation.

Done criteria:
- [ ] RED failure observed for expected reason.
- [ ] Test references root/global `dt_rules` boundary.

Commit suggestion:
- message: `test geodt dt rules are not branch source`

### T016: Add no-overwrite regression

Source:
- Requirement IDs: FR-035, FR-OUT-003, SC-009, SC-019
- Plan section: Output Artifact Strategy

Dependencies:
- T003, T012

Files:
- Edit: `<focused-test-path>`
- Production edit: none

TDD classification:
- focused regression test

RED:
- Write/update: test creating protected existing `checkpoints/`, `space_partitions/`, `dt_rules/`, `other_outputs/`, `deliverables/` sentinels and asserting diagnostic writes only diagnostics output or fails on conflict.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: diagnostic no-overwrite behavior is not implemented yet.

GREEN:
- Smallest implementation change: none in this task.
- Command: focused pytest.
- Expected passing result: later after T034.

REFACTOR:
- Allowed cleanup: test helper reuse only.
- Command: focused pytest.

Validation:
- Commands: focused pytest.
- Expected result: RED failure now, green after T034.

Done criteria:
- [ ] Sentinel paths covered.
- [ ] RED failure is due to missing diagnostic behavior, not fixture error.

Commit suggestion:
- message: `test geodt diagnostic output boundaries`

### T017: Add bounded archive-discovery regression

Source:
- Requirement IDs: FR-000F-H, SC-022
- Plan section: Archive Discovery and Fallback Planning

Dependencies:
- T009, T012

Files:
- Edit: `<focused-test-path>`

TDD classification:
- focused regression / contract test

RED:
- Write/update: test asserting no archive path/list/root without configured root fails clearly and discovery never scans unrelated directories.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: archive resolver not implemented.

GREEN:
- Smallest implementation change: none in this task.
- Command: focused pytest.
- Expected passing result: later after T025.

REFACTOR:
- Allowed cleanup: test helper only.
- Command: focused pytest.

Validation:
- Commands: focused pytest.
- Expected result: RED failure for missing bounded resolver.

Done criteria:
- [ ] Test encodes bounded deterministic search.
- [ ] No real unbounded filesystem search is performed.

Commit suggestion:
- message: `test bounded geodt archive discovery`

### T018: Add artifact-provider compatibility regression

Source:
- Requirement IDs: FR-000F, FR-000M-N, FR-003B, FR-EVID-001, FR-EVID-004, SC-014, SC-020
- Plan section: Same-run/source artifact-provider decision

Dependencies:
- T009, T013

Files:
- Edit: `<focused-test-path>`

TDD classification:
- focused regression / data-artifact test

RED:
- Write/update: test passing visual archive and provider fixture and asserting metadata records both paths and compatibility evidence.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: provider resolution not implemented.

GREEN:
- Smallest implementation change: none in this task.
- Command: focused pytest.
- Expected passing result: later after T026/T033.

REFACTOR:
- Allowed cleanup: test helper only.
- Command: focused pytest.

Validation:
- Commands: focused pytest.
- Expected result: RED failure for missing provider compatibility behavior.

Done criteria:
- [ ] Test requires selected archive path and provider path in metadata.
- [ ] Test does not rely on real archives.

Commit suggestion:
- message: `test geodt artifact provider compatibility`

### T018A: Foundation gate

Source:
- Requirement IDs: FR-000, FR-000D, FR-035, FR-EVID-001, FR-EVID-004, SC-009, SC-013, SC-020
- Plan section: Required characterization before edit

Dependencies:
- T006-T018

Files:
- Inspect: evidence/task-run notes; `<focused-test-path>`
- Edit: gate note in designated evidence/audit artifact or task runner
- Tests: focused RED gates from T015-T018

TDD classification:
- validation gate

TDD exception:
- Required: yes
- Reason: Gate verification only.
- Why strict TDD does not apply: No code behavior is implemented.
- Validation method: Confirm all dependencies complete and outputs recorded.
- Command or review criterion: run focused tests and review evidence outputs.
- Risk: Critical; implementation must not start before this gate.
- Approval source: tasks.md T018A.

RED:
- Write/update: none
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: tests may fail because implementation is absent; gate checks that RED tests exist and fail for expected missing behavior, not typos.

GREEN:
- Smallest implementation change: none; record gate status.
- Command: review T006-T018 completion and focused test outcomes.
- Expected passing result: all characterization findings recorded, focused path selected, RED gates ready.

REFACTOR:
- Allowed cleanup: none.
- Command: not applicable.

Validation:
- Commands: review dependency completion and test RED outcomes.
- Expected result: implementation tasks are allowed to begin only after gate is complete.

Done criteria:
- [ ] T006-T018 complete.
- [ ] T002/T009/T010 outputs recorded.
- [ ] Evidence-dependent assumptions from T006-T010 recorded.
- [ ] No implementation task has edited the diagnostic script before this gate.

Commit suggestion:
- message: `complete geodt diagnostic foundation gate`

### T019: Add figure-generation CLI contract and config non-mutation tests

Source:
- Requirement IDs: FR-001, FR-018, FR-029, FR-030, FR-035, FR-OUT-001, FR-OUT-004, US1, SC-001, SC-009
- Plan section: CLI contract; Runtime config touched

Dependencies:
- T018A

Files:
- Edit: `<focused-test-path>`
- Tests: `<focused-test-path>`

TDD classification:
- contract test

RED:
- Write/update: tests for CLI args and assertion that parsing/running diagnostic does not mutate `SAVE_DT_RULES`, `SAVE_DT_NODE_DUMP`, or `ACTIVE_LAGS`.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: script/CLI parser absent.

GREEN:
- Smallest implementation change: none in this task; T024 makes it pass.
- Command: focused pytest.
- Expected passing result: later after T024.

REFACTOR:
- Allowed cleanup: test organization only.
- Command: focused pytest.

Validation:
- Commands: focused pytest.
- Expected result: RED failure for missing CLI, not test syntax.

Done criteria:
- [ ] CLI contract tests exist.
- [ ] Runtime config non-mutation is asserted.

Commit suggestion:
- message: `test geodt diagnostic cli contract`

### T020: Add preflight/fallback/failure tests

Source:
- Requirement IDs: FR-000*, FR-003A-B, FR-036A-B, SC-013, SC-014, SC-017, SC-021, SC-022
- Plan section: Preflight and Failure Handling Strategy

Dependencies:
- T012, T013, T014, T017, T018, T018A

Files:
- Edit: `<focused-test-path>`

TDD classification:
- data/artifact behavior test

RED:
- Write/update: tests for archive discovery, fallback selection, minimum required artifact completeness, and structured failure summaries.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: preflight resolver/summary behavior absent.

GREEN:
- Smallest implementation change: none in this task; T025/T027/T034 make tests pass.
- Command: focused pytest.
- Expected passing result: later after implementation.

REFACTOR:
- Allowed cleanup: test helper only.
- Command: focused pytest.

Validation:
- Commands: focused pytest.
- Expected result: RED failure for missing preflight.

Done criteria:
- [ ] Fallback candidate decisions and failure summaries tested.
- [ ] Tests use fixtures, not real production artifacts.

Commit suggestion:
- message: `test geodt preflight and fallback behavior`

### T021: Add checkpoint classification and branch eligibility tests

Source:
- Requirement IDs: FR-004-FR-007, FR-017*, FR-CHK-001-FR-CHK-005, SC-015
- Plan section: Branch Eligibility Plan

Dependencies:
- T012, T014, T015, T018A

Files:
- Edit: `<focused-test-path>`

TDD classification:
- data/artifact behavior test

RED:
- Write/update: tests covering root/global exclusion, load failures, unused checkpoints, assignment-count reporting.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: classification/eligibility behavior absent.

GREEN:
- Smallest implementation change: none in this task; T030/T031 pass it later.
- Command: focused pytest.
- Expected passing result: later after implementation.

REFACTOR:
- Allowed cleanup: test helper only.
- Command: focused pytest.

Validation:
- Commands: focused pytest.
- Expected result: RED failure for missing classifier/eligibility.

Done criteria:
- [ ] Root/global exception is explicit.
- [ ] Counts remain separate.

Commit suggestion:
- message: `test geodt checkpoint eligibility`

### T022: Add feature-name precedence and split-index tests

Source:
- Requirement IDs: FR-006D-H, SC-023
- Plan section: Feature-Name Source Plan

Dependencies:
- T010, T012, T014, T018A

Files:
- Edit: `<focused-test-path>`

TDD classification:
- data/artifact behavior test

RED:
- Write/update: tests for feature source precedence, feature-count mismatch, and split index out of bounds.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: feature selector absent.

GREEN:
- Smallest implementation change: none in this task; T029 passes it later.
- Command: focused pytest.
- Expected passing result: later after T029.

REFACTOR:
- Allowed cleanup: test helper only.
- Command: focused pytest.

Validation:
- Commands: focused pytest.
- Expected result: RED failure for missing feature-name selector.

Done criteria:
- [ ] Selected and rejected feature sources are asserted.
- [ ] Mismatch marks checkpoint unusable.

Commit suggestion:
- message: `test geodt feature source validation`

### T023: Add figure-rendering smoke tests

Source:
- Requirement IDs: FR-018-FR-030, SC-001, SC-011, SC-012
- Plan section: Figure Generation Plan

Dependencies:
- T012, T014, T018A

Files:
- Edit: `<focused-test-path>`

TDD classification:
- smoke / artifact behavior test

RED:
- Write/update: smoke tests for 1x2 layout, dimensions, 300 DPI PNG metadata, optional PDF, neutral class labels, documented abbreviations.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: renderer absent.

GREEN:
- Smallest implementation change: none in this task; T032 passes it later.
- Command: focused pytest.
- Expected passing result: later after T032/T034.

REFACTOR:
- Allowed cleanup: test helper only.
- Command: focused pytest.

Validation:
- Commands: focused pytest.
- Expected result: RED failure for missing renderer.

Done criteria:
- [ ] Smoke test checks artifact metadata/dimensions.
- [ ] Neutral class-label behavior covered.

Commit suggestion:
- message: `test geodt tree figure rendering`

### T024: Implement CLI skeleton

Source:
- Requirement IDs: FR-001, FR-000F-G, FR-OUT-001, FR-OUT-003, US1
- Plan section: Component Design; CLI contract

Dependencies:
- T003, T006, T007, T008, T009, T010, T018A, T019

Files:
- Inspect: `contracts/geodt-branch-tree-diagnostic-cli.md`, `<focused-test-path>`
- Edit: `scripts/plot_geodt_branch_tree_comparison.py`
- Tests: `<focused-test-path>`

TDD classification:
- strict TDD / CLI contract behavior

RED:
- Write/update: already T019; rerun CLI contract tests.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: script is missing or CLI args unsupported.

GREEN:
- Smallest implementation change: create standalone script with argparse skeleton for required flags and no mutation of training/runtime config.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected passing result: T019 contract tests pass; unrelated later behavior tests may still fail if not selected/narrowed.

REFACTOR:
- Allowed cleanup: internal CLI parser naming only; no helper extraction unless needed immediately.
- Command: focused CLI contract subset or full focused test path.

Validation:
- Commands: `python scripts/plot_geodt_branch_tree_comparison.py --help`
- Expected result: help lists required flags and exits 0.

Done criteria:
- [ ] T019 RED was observed before implementation.
- [ ] CLI skeleton passes T019.
- [ ] No production config values are mutated.

Commit suggestion:
- message: `add geodt branch tree diagnostic cli skeleton`

### T025: Implement bounded archive resolver and fallback decisions

Source:
- Requirement IDs: FR-000F-H, FR-002, FR-003A-B, SC-014, SC-022
- Plan section: Archive Resolver; Fallback order

Dependencies:
- T020, T024

Files:
- Edit: `scripts/plot_geodt_branch_tree_comparison.py`
- Tests: `<focused-test-path>`

TDD classification:
- strict TDD / data-artifact behavior

RED:
- Write/update: T017/T020 tests already cover behavior; rerun bounded discovery/fallback subset.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: resolver/fallback behavior missing.

GREEN:
- Smallest implementation change: implement deterministic resolver for explicit path/list/root and configured root failure handling; do not scan arbitrary directories.
- Command: focused pytest.
- Expected passing result: bounded discovery/fallback tests pass.

REFACTOR:
- Allowed cleanup: local helper functions in script only.
- Command: focused pytest.

Validation:
- Commands: audit-only CLI against synthetic archive once available.
- Expected result: candidate decisions recorded or clear failure.

Done criteria:
- [ ] Bounded discovery tests pass.
- [ ] No arbitrary filesystem search.

Commit suggestion:
- message: `implement bounded geodt archive resolver`

### T026: Implement same-run/source artifact-provider resolution

Source:
- Requirement IDs: FR-000I, FR-000M-N, FR-EVID-001, FR-EVID-003-FR-EVID-004, SC-017, SC-020
- Plan section: Artifact-provider resolution

Dependencies:
- T018, T025

Files:
- Edit: `scripts/plot_geodt_branch_tree_comparison.py`
- Tests: `<focused-test-path>`

TDD classification:
- strict TDD / data-artifact behavior

RED:
- Write/update: T018 provider compatibility test.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: provider resolution/metadata absent.

GREEN:
- Smallest implementation change: resolve explicit provider path, verify same model family/year-month/fs/run identity evidence, record both paths.
- Command: focused pytest.
- Expected passing result: provider compatibility test passes.

REFACTOR:
- Allowed cleanup: small provider evidence helper only.
- Command: focused pytest.

Validation:
- Commands: synthetic audit-only CLI with `--artifact-provider-path`.
- Expected result: metadata/audit records selected archive and provider.

Done criteria:
- [ ] Provider tests pass.
- [ ] Incompatible provider fails clearly.

Commit suggestion:
- message: `implement geodt artifact provider validation`

### T027: Implement artifact characterizer

Source:
- Requirement IDs: FR-000A-E, FR-000I, FR-000L, FR-004, FR-008, FR-032, SC-013, SC-016, SC-017
- Plan section: Artifact Characterizer

Dependencies:
- T015, T020, T026

Files:
- Edit: `scripts/plot_geodt_branch_tree_comparison.py`
- Tests: `<focused-test-path>`

TDD classification:
- strict TDD / data-artifact behavior

RED:
- Write/update: T015/T020 preflight and root/global-only tests.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: artifact classification/completeness behavior absent.

GREEN:
- Smallest implementation change: inspect archive/provider for checkpoint, partition, correspondence, feature candidate, and root/global `dt_rules` boundary notes; return structured completeness result.
- Command: focused pytest.
- Expected passing result: artifact completeness/root-global-only tests pass.

REFACTOR:
- Allowed cleanup: local dataclass/dict helper if immediately used.
- Command: focused pytest.

Validation:
- Commands: synthetic audit-only CLI.
- Expected result: preflight record identifies artifacts and missing/unusable items.

Done criteria:
- [ ] Root/global-only archive incomplete.
- [ ] Minimum artifact completeness recorded.

Commit suggestion:
- message: `implement geodt archive artifact characterizer`

### T028: Implement branch assignment source selector

Source:
- Requirement IDs: FR-005, FR-006, FR-006A-C, FR-EVID-002, FR-EVID-004, SC-015, SC-020
- Plan section: Assignment Source Selector

Dependencies:
- T008, T021, T027

Files:
- Edit: `scripts/plot_geodt_branch_tree_comparison.py`
- Tests: `<focused-test-path>`

TDD classification:
- strict TDD / data-artifact behavior

RED:
- Write/update: T021 assignment/eligibility tests for precedence and rejected-source reasons.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: assignment selector absent.

GREEN:
- Smallest implementation change: select `X_branch_id.npy`, `s_branch.pkl`, correspondence table, or documented equivalent in spec order; record rejected reasons.
- Command: focused pytest.
- Expected passing result: assignment selector tests pass.

REFACTOR:
- Allowed cleanup: local normalization helper only.
- Command: focused pytest.

Validation:
- Commands: synthetic audit-only CLI.
- Expected result: assignment source path/type/status recorded.

Done criteria:
- [ ] Source precedence honored.
- [ ] Rejected sources documented.

Commit suggestion:
- message: `implement geodt branch assignment selector`

### T029: Implement feature-name source selector

Source:
- Requirement IDs: FR-006D-H, SC-023
- Plan section: Feature-Name Source Selector

Dependencies:
- T010, T022, T027

Files:
- Edit: `scripts/plot_geodt_branch_tree_comparison.py`
- Tests: `<focused-test-path>`

TDD classification:
- strict TDD / data-artifact behavior

RED:
- Write/update: T022 feature-source tests.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: feature selector absent.

GREEN:
- Smallest implementation change: implement source precedence, count compatibility, split-index bounds check, and rejected-source records.
- Command: focused pytest.
- Expected passing result: feature-source tests pass.

REFACTOR:
- Allowed cleanup: local feature-count helper only.
- Command: focused pytest.

Validation:
- Commands: synthetic audit-only CLI.
- Expected result: feature-name source and compatibility recorded.

Done criteria:
- [ ] Feature count mismatch marks checkpoint unusable.
- [ ] Out-of-bounds split index marks checkpoint unusable.

Commit suggestion:
- message: `implement geodt feature source selector`

### T030: Implement checkpoint parser and loader

Source:
- Requirement IDs: FR-004, FR-007, FR-CHK-001-FR-CHK-005, SC-015
- Plan section: Checkpoint Parser and Loader

Dependencies:
- T007, T021, T027

Files:
- Edit: `scripts/plot_geodt_branch_tree_comparison.py`
- Read-only inspect only if needed: `src/model/model_DT.py`
- Tests: `<focused-test-path>`

TDD classification:
- strict TDD / data-artifact behavior

RED:
- Write/update: T021 checkpoint classification tests.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: parser/loader absent.

GREEN:
- Smallest implementation change: parse branch IDs with documented loader convention, classify root/global/branch/unusable/unknown, load tree objects through safe existing behavior or fixture-compatible loader.
- Command: focused pytest.
- Expected passing result: checkpoint classification tests pass.

REFACTOR:
- Allowed cleanup: local checkpoint record helper only.
- Command: focused pytest.

Validation:
- Commands: synthetic audit-only CLI.
- Expected result: checkpoint records include filename, parsed branch ID, classification, parsing method, mismatch reason.

Done criteria:
- [ ] Filename pattern alone is not terminal evidence.
- [ ] Root/global checkpoint excluded unless explicitly terminal.

Commit suggestion:
- message: `implement geodt checkpoint classification`

### T031: Implement branch eligibility builder

Source:
- Requirement IDs: FR-006A, FR-017-FR-017D, SC-015
- Plan section: Branch Eligibility Builder

Dependencies:
- T021, T028, T029, T030

Files:
- Edit: `scripts/plot_geodt_branch_tree_comparison.py`
- Tests: `<focused-test-path>`

TDD classification:
- strict TDD / data-artifact behavior

RED:
- Write/update: T021 branch eligibility/count tests.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: eligibility builder absent.

GREEN:
- Smallest implementation change: join assignment, checkpoint, feature-source results; compute assigned admin/group count, prediction-row count when available, unavailable training count marker, exclusion reasons.
- Command: focused pytest.
- Expected passing result: eligibility tests pass.

REFACTOR:
- Allowed cleanup: local branch record helper only.
- Command: focused pytest.

Validation:
- Commands: synthetic audit-only CLI.
- Expected result: eligible/ineligible/failed/skipped branches recorded.

Done criteria:
- [ ] Counts are not conflated.
- [ ] Exclusion reasons documented.

Commit suggestion:
- message: `implement geodt branch eligibility records`

### Shared Selection Pipeline Tasks Required Before US1 Figure Rendering

T037, T038, and T039 are tests for the shared selection pipeline. T041, T042, and T043 implement the shared selection pipeline. These tasks are required before T032/T033/T035 and are reused by US2 audit-only mode. Do not duplicate selection logic inside T032 or T044.

### T037: Add top-K signature extraction tests

Source:
- Requirement IDs: FR-010, FR-011, FR-012, FR-014A, US1, US2
- Plan section: Signature and Pair Selector
- Note: This is a US1/foundational selection test, not only a US2 audit-only test.

Dependencies:
- T012, T031

Files:
- Edit: `<focused-test-path>`

TDD classification:
- strict RED test for shared selection pipeline implementation

RED:
- Write/update: tests for split features at depths `0` through `K-1`, actual available depth, neutral leaf summaries.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: signature extraction absent.

GREEN:
- Smallest implementation change: none in this test task; T041 passes it later.
- Command: focused pytest.
- Expected passing result: later after T041.

REFACTOR:
- Allowed cleanup: test helper only.
- Command: focused pytest.

Validation:
- Commands: focused pytest.
- Expected result: RED failure for missing signature extractor.

Done criteria:
- [ ] Test captures K default and shallow tree handling.
- [ ] Test is written as shared selection-pipeline coverage required before US1 rendering and reused by US2.

Commit suggestion:
- message: `test geodt branch signatures`

### T038: Add deterministic Jaccard scoring tests

Source:
- Requirement IDs: FR-013, FR-014, FR-014A-B, FR-015, SC-003
- Plan section: Branch Signature and Scoring Plan
- Note: This is shared selection-pipeline coverage required before US1 rendering and reused by US2.

Dependencies:
- T012, T037

Files:
- Edit: `<focused-test-path>`

TDD classification:
- strict RED test for shared selection pipeline implementation

RED:
- Write/update: tests for Jaccard formula and threshold/direction supplemental fields not altering ranking.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: scorer absent.

GREEN:
- Smallest implementation change: none in this task; T042 passes it later.
- Command: focused pytest.
- Expected passing result: later after T042.

REFACTOR:
- Allowed cleanup: test naming only.
- Command: focused pytest.

Validation:
- Commands: focused pytest.
- Expected result: RED failure for missing scorer.

Done criteria:
- [ ] Formula is exact.
- [ ] Supplemental threshold/direction cannot change ranking.
- [ ] Test is written as shared selection-pipeline coverage required before US1 rendering and reused by US2.

Commit suggestion:
- message: `test geodt branch pair scoring`

### T039: Add tie-break and readability-gate tests

Source:
- Requirement IDs: FR-016A-H, SC-003, SC-008
- Plan section: Readability Gate Plan
- Note: This is shared selection-pipeline coverage required before US1 rendering and reused by US2.

Dependencies:
- T012, T014, T038

Files:
- Edit: `<focused-test-path>`

TDD classification:
- strict RED test for shared selection pipeline implementation

RED:
- Write/update: tests for rejected higher-scoring unreadable pair, no-readable-pair failure, deterministic tie-break ordering.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: tie-break/readability gate absent.

GREEN:
- Smallest implementation change: none in this task; T042/T043 pass it later.
- Command: focused pytest.
- Expected passing result: later after T043.

REFACTOR:
- Allowed cleanup: test helper only.
- Command: focused pytest.

Validation:
- Commands: focused pytest.
- Expected result: RED failure for missing readability/tie-break behavior.

Done criteria:
- [ ] Tie-break order matches spec.
- [ ] No-readable-pair fails clearly.
- [ ] Test is written as shared selection-pipeline coverage required before US1 rendering and reused by US2.

Commit suggestion:
- message: `test geodt readability and tie breaks`

### T041: Implement top-K branch signature extraction

Source:
- Requirement IDs: FR-010, FR-011, FR-012, FR-015
- Plan section: Signature and Pair Selector
- Note: Implements shared selection-pipeline behavior required before T032/T033/T035 and reused by US2 audit-only mode.

Dependencies:
- T031, T037

Files:
- Edit: `scripts/plot_geodt_branch_tree_comparison.py`
- Tests: `<focused-test-path>`

TDD classification:
- strict TDD

RED:
- Write/update: T037 already exists.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: signature extractor absent.

GREEN:
- Smallest implementation change: extract non-leaf split feature names at depths `0..K-1`, actual available depth, neutral leaf summaries when safe, supplemental threshold/direction fields only.
- Command: focused pytest.
- Expected passing result: T037 signature tests pass.

REFACTOR:
- Allowed cleanup: local traversal helper only.
- Command: focused pytest.

Validation:
- Commands: focused pytest and synthetic audit-only CLI after T044.
- Expected result: signatures recorded in shared selection records, audit output, and metadata.

Done criteria:
- [ ] Signature tests pass.
- [ ] Threshold/direction fields do not affect ranking.
- [ ] Shared selection-pipeline record can be consumed by T032/T033/T044.

Commit suggestion:
- message: `implement geodt branch signatures`

### T042: Implement Jaccard scoring and tie-break fields

Source:
- Requirement IDs: FR-013, FR-014, FR-014A-B, FR-016A, FR-016G-H, FR-028B, SC-003
- Plan section: Branch Signature and Scoring Plan
- Note: Implements shared selection-pipeline behavior required before T032/T033/T035 and reused by US2 audit-only mode.

Dependencies:
- T038, T041

Files:
- Edit: `scripts/plot_geodt_branch_tree_comparison.py`
- Tests: `<focused-test-path>`

TDD classification:
- strict TDD

RED:
- Write/update: T038 scoring tests.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: scorer/tie-break fields absent.

GREEN:
- Smallest implementation change: implement exact Jaccard formula, empty-set ineligibility, contrast descriptor, deterministic tie-break fields, rejected-pair recording hook, and selected highest-ranked readable-pair record consumed by renderer/metadata/audit wrappers.
- Command: focused pytest.
- Expected passing result: scoring tests pass; T039 may still fail until T043.

REFACTOR:
- Allowed cleanup: local score record helper only.
- Command: focused pytest.

Validation:
- Commands: focused pytest.
- Expected result: deterministic score/tie-break behavior.

Done criteria:
- [ ] Formula and tie-break tests pass.
- [ ] Supplemental fields remain audit-only.
- [ ] Shared selection-pipeline record can be consumed by T032/T033/T044.

Commit suggestion:
- message: `implement geodt branch pair scoring`

### T043: Implement readability gate checks

Source:
- Requirement IDs: FR-016B-F, FR-024C-E, SC-012
- Plan section: Readability Gate Plan
- Note: Implements shared selection-pipeline behavior required before T032/T033/T035 and reused by US2 audit-only mode.

Dependencies:
- T039, T042

Files:
- Edit: `scripts/plot_geodt_branch_tree_comparison.py`
- Tests: `<focused-test-path>`

TDD classification:
- strict TDD

RED:
- Write/update: T039 readability tests.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: readability gate absent.

GREEN:
- Smallest implementation change: implement non-leaf split requirement, same plotted depth/style fields, label wrapping/abbreviation map, minimum font size metadata, no silent truncation failure, rejected higher-scoring-pair records, and final selected highest-ranked readable-pair record.
- Command: focused pytest.
- Expected passing result: readability/tie-break tests pass.

REFACTOR:
- Allowed cleanup: local readability record helper only.
- Command: focused pytest.

Validation:
- Commands: focused pytest and synthetic audit-only once T044 exists.
- Expected result: unreadable pairs rejected with reasons and selected-pair record available to T032/T033/T044.

Done criteria:
- [ ] Readability tests pass.
- [ ] No misleading figure can be generated for unreadable pair.
- [ ] Shared selection-pipeline record can be consumed by T032/T033/T044.

Commit suggestion:
- message: `implement geodt readability gate`

### T032: Implement figure renderer

Source:
- Requirement IDs: FR-018-FR-030, SC-001, SC-011, SC-012
- Plan section: Figure Renderer

Dependencies:
- T023
- T031
- T041
- T042
- T043

Rationale: figure renderer must receive selected highest-ranked readable pair rather than arbitrary eligible branches.

Files:
- Edit: `scripts/plot_geodt_branch_tree_comparison.py`
- Tests: `<focused-test-path>`

TDD classification:
- strict TDD / artifact behavior

RED:
- Write/update: T023 rendering smoke tests.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: renderer absent.

GREEN:
- Smallest implementation change: add matplotlib/sklearn 1x2 renderer that consumes the selected highest-ranked readable pair produced by the shared selection pipeline, renders matched depth/style, safe labels, wrapping/abbreviations, PNG/PDF output hooks, and caption text.
- Command: focused pytest.
- Expected passing result: rendering smoke tests pass or advance to output-path behavior covered by T034.

REFACTOR:
- Allowed cleanup: local render helpers only; no broad visualization module unless repetition justifies and scope remains allowed.
- Command: focused pytest.

Validation:
- Commands: synthetic figure-generation CLI.
- Expected result: figure satisfies dimensions/DPI/readability metadata.

Done criteria:
- [ ] No crisis/non-crisis labels without confirmed mapping.
- [ ] Figure uses branch-specific checkpoints only.
- [ ] T032 does not implement or duplicate branch-pair scoring or readability selection logic; it consumes selection records from T041-T043.

Commit suggestion:
- message: `implement geodt branch tree figure renderer`

### T033: Implement metadata JSON writer

Source:
- Requirement IDs: FR-031, FR-032, FR-034, FR-EVID-004, SC-004, SC-010, SC-016, SC-020
- Plan section: Metadata / Provenance Strategy

Dependencies:
- T016
- T027
- T028
- T029
- T030
- T031
- T032
- T041
- T042
- T043

Rationale: metadata must include selected pair score, contrast descriptor, tie-break/readability results, and rejected-pair summaries.

Files:
- Edit: `scripts/plot_geodt_branch_tree_comparison.py`
- Tests: `<focused-test-path>`

TDD classification:
- strict TDD / data-artifact behavior

RED:
- Write/update: metadata assertions in existing US1 tests or a focused metadata test in `<focused-test-path>`.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: metadata writer absent/incomplete.

GREEN:
- Smallest implementation change: write canonical metadata JSON with FR-031 fields currently available from implemented components, including selected pair score, contrast descriptor, tie-break/readability results, and rejected-pair summaries from T041-T043.
- Command: focused pytest.
- Expected passing result: metadata assertions pass.

REFACTOR:
- Allowed cleanup: local metadata assembly helper only.
- Command: focused pytest.

Validation:
- Commands: synthetic figure-generation CLI and inspect metadata.
- Expected result: metadata includes provenance, output paths, root/global distinction, selected-pair scoring, readability, tie-break, and rejected-pair records.

Done criteria:
- [ ] Metadata supports reproduction inputs.
- [ ] Evidence validation statuses recorded.
- [ ] Selected score, contrast descriptor, tie-break/readability results, and rejected-pair summaries are recorded from the shared selection pipeline.

Commit suggestion:
- message: `implement geodt diagnostic metadata writer`

### T034: Implement no-overwrite output handling and failure summary

Source:
- Requirement IDs: FR-035, FR-036, FR-036A-B, FR-OUT-001-FR-OUT-004, SC-009, SC-019, SC-021
- Plan section: Output Artifact Strategy; Failure Handling Strategy

Dependencies:
- T016
- T020
- T033

Files:
- Edit: `scripts/plot_geodt_branch_tree_comparison.py`
- Tests: `<focused-test-path>`

TDD classification:
- strict TDD / artifact behavior

RED:
- Write/update: T016/T020 failure and no-overwrite tests.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: no-overwrite/failure summary absent.

GREEN:
- Smallest implementation change: implement output dir resolution, conflict handling, `--overwrite`, unique/fail behavior, structured failure summary display/write.
- Command: focused pytest.
- Expected passing result: no-overwrite and failure summary tests pass.

REFACTOR:
- Allowed cleanup: output path helper only.
- Command: focused pytest.

Validation:
- Commands: synthetic CLI with existing output conflict.
- Expected result: no production artifact overwritten; structured failure on invalid inputs.

Done criteria:
- [ ] Existing artifacts are not overwritten by default.
- [ ] Failure never creates misleading partial figure.

Commit suggestion:
- message: `implement geodt diagnostic output safeguards`

### T035: Run focused US1 validation

Source:
- Requirement IDs: SC-001, SC-002, SC-003, SC-004, SC-008, SC-009, SC-011, SC-012, SC-013, SC-015, SC-017, SC-021, SC-023
- Plan section: Validation commands
- Note: T035 validates US1 including selected-pair scoring/readability behavior, so it requires T041-T043 to be complete before it can pass.

Dependencies:
- T024
- T025
- T026
- T027
- T028
- T029
- T030
- T031
- T032
- T033
- T034
- T041
- T042
- T043

Rationale: US1 validation includes SC-003 and therefore requires selection pipeline implementation.

Files:
- Inspect: `<focused-test-path>`, generated temp outputs
- Edit: validation summary/evidence only

TDD classification:
- validation

TDD exception:
- Required: yes
- Reason: Validation task only.
- Why strict TDD does not apply: Tests already exist; this task runs them.
- Validation method: Execute focused tests including selected-pair scoring/readability behavior and record known baseline failures separately from new regressions.
- Command or review criterion: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Risk: Medium; unexecuted validation means task is not complete.
- Approval source: tasks.md T035.

RED:
- Write/update: none
- Command: focused pytest.
- Expected failure: any remaining failure must be classified as known baseline, environment gap, or new regression.

GREEN:
- Smallest implementation change: fix only scoped implementation defects if failures are new regressions; otherwise record unexecuted/blocked status.
- Command: focused pytest.
- Expected passing result: US1 focused tests pass, including selected-pair scoring/readability behavior.

REFACTOR:
- Allowed cleanup: none unless tests are green and cleanup is local/scoped.
- Command: focused pytest.

Validation:
- Commands: focused pytest.
- Expected result: pass, or not complete with recorded reason.

Done criteria:
- [ ] Focused US1 tests pass.
- [ ] Highest-ranked readable pair selection is validated through the shared selection pipeline.
- [ ] Any rejected higher-scoring pair or tie-break result is reflected in metadata.
- [ ] Known baseline failures separated from new regressions.

Commit suggestion:
- message: `validate geodt branch tree figure workflow`

### T036: Run synthetic figure-generation CLI smoke

Source:
- Requirement IDs: US1, SC-001, SC-002, SC-004, SC-009, SC-011, SC-012
- Plan section: Quickstart figure generation

Dependencies:
- T035

Files:
- Inspect: fixture output, metadata, figure files
- Edit: validation summary/evidence only

TDD classification:
- smoke validation

TDD exception:
- Required: yes
- Reason: Smoke command execution only.
- Why strict TDD does not apply: Behavior was already test-first implemented.
- Validation method: Run pytest-managed subprocess or standalone synthetic CLI.
- Command or review criterion: `python scripts/plot_geodt_branch_tree_comparison.py --archive-path <synthetic-fixture-archive> --output-dir <temporary-diagnostics-output>`
- Risk: Medium; smoke may reveal integration issues not in unit tests.
- Approval source: tasks.md T036.

RED:
- Write/update: none
- Command: figure-generation smoke command.
- Expected failure: if environment or implementation integration incomplete; classify before fixing.

GREEN:
- Smallest implementation change: fix only scoped integration defects, preserving tests.
- Command: smoke command.
- Expected passing result: figure and metadata generated in temp diagnostics output.

REFACTOR:
- Allowed cleanup: none unless tests/smoke green.
- Command: focused pytest plus smoke command.

Validation:
- Commands: smoke command.
- Expected result: valid figure metadata and no source artifact modifications.

Done criteria:
- [ ] Smoke passes or task remains incomplete with reason.
- [ ] Outputs are under temporary diagnostics directory.

Commit suggestion:
- message: `smoke test geodt branch tree figure cli`

### T040: Add audit-only contract tests

Source:
- Requirement IDs: FR-AUDIT-001-FR-AUDIT-004, SC-018
- Plan section: Audit-Only Mode Plan
- Note: T039 is already implemented through T043 before T040's audit-only contract expectations are finalized.

Dependencies:
- T019
- T035
- T039

Files:
- Edit: `<focused-test-path>`

TDD classification:
- contract test

RED:
- Write/update: tests for `--audit-only` no PNG/PDF and same selection as figure-generation mode, expecting reuse of the completed shared selection pipeline.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: audit-only mode absent.

GREEN:
- Smallest implementation change: none in this task; T044 passes it later.
- Command: focused pytest.
- Expected passing result: later after T044.

REFACTOR:
- Allowed cleanup: test helper only.
- Command: focused pytest.

Validation:
- Commands: focused pytest.
- Expected result: RED failure for missing audit-only mode.

Done criteria:
- [ ] Test asserts no figure output.
- [ ] Test asserts selection parity through shared selection-pipeline records.

Commit suggestion:
- message: `test geodt audit-only mode`

### T044: Implement audit-only mode and audit summary

Source:
- Requirement IDs: FR-AUDIT-001-FR-AUDIT-004, SC-018
- Plan section: Audit-Only Mode Plan

Dependencies:
- T040, T041, T042, T043

Files:
- Edit: `scripts/plot_geodt_branch_tree_comparison.py`
- Tests: `<focused-test-path>`

TDD classification:
- strict TDD / CLI contract behavior

RED:
- Write/update: T040 audit-only tests.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: audit-only mode absent.

GREEN:
- Smallest implementation change: wire `--audit-only` through existing preflight, eligibility, feature selection, signature extraction, scoring, and readability records produced by the shared selection pipeline; write/display metadata/audit output and suppress PNG/PDF rendering.
- Command: focused pytest.
- Expected passing result: audit-only tests pass.

REFACTOR:
- Allowed cleanup: shared workflow orchestration helper inside script only.
- Command: focused pytest.

Validation:
- Commands: synthetic audit-only CLI.
- Expected result: audit summary created/displayed and no figure files.

Done criteria:
- [ ] Audit-only selection matches figure mode.
- [ ] Audit-only mode reuses the shared selection pipeline from T041-T043.
- [ ] No duplicate scoring/readability implementation is introduced in the audit-only wrapper.
- [ ] No PNG/PDF rendered in audit-only.

Commit suggestion:
- message: `implement geodt audit-only mode`

### T045: Run US2 focused validation

Source:
- Requirement IDs: US2, SC-003, SC-008, SC-018
- Plan section: Validation tasks

Dependencies:
- T040
- T044

Files:
- Inspect: `<focused-test-path>`, temp outputs
- Edit: validation summary/evidence only

TDD classification:
- validation-only exception

TDD exception:
- Required: yes
- Reason: Runs tests; no new behavior.
- Why strict TDD does not apply: Behavior was implemented through prior RED/GREEN tasks.
- Validation method: Execute focused tests and record baseline/new regressions separately.
- Command or review criterion: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Risk: Medium.
- Approval source: tasks.md T045.

RED:
- Write/update: none
- Command: focused pytest.
- Expected failure: any failure must be classified.

GREEN:
- Smallest implementation change: fix scoped implementation defects only if new regressions.
- Command: focused pytest.
- Expected passing result: US2 focused tests pass.

REFACTOR:
- Allowed cleanup: none unless green.
- Command: focused pytest.

Validation:
- Commands: focused pytest.
- Expected result: US2 tests pass.

Done criteria:
- [ ] US2 tests pass.
- [ ] Known baseline failures separated.

Commit suggestion:
- message: `validate geodt audit-only workflow`

### T046: Run synthetic audit-only smoke

Source:
- Requirement IDs: US2, FR-AUDIT-004, SC-018
- Plan section: Quickstart audit-only

Dependencies:
- T045

Files:
- Inspect: audit output, output directory
- Edit: validation summary/evidence only

TDD classification:
- smoke validation exception

TDD exception:
- Required: yes
- Reason: Smoke execution only.
- Why strict TDD does not apply: No new behavior is implemented.
- Validation method: Run adapted quickstart against fixture archive.
- Command or review criterion: `python scripts/plot_geodt_branch_tree_comparison.py --audit-only --archive-path <synthetic-fixture-archive> --output-dir <temporary-diagnostics-output>`
- Risk: Medium.
- Approval source: tasks.md T046.

RED:
- Write/update: none
- Command: audit-only smoke command.
- Expected failure: integration/environment issue if present; classify before fixing.

GREEN:
- Smallest implementation change: fix scoped integration defects only.
- Command: audit-only smoke command.
- Expected passing result: audit summary/metadata output, no PNG/PDF.

REFACTOR:
- Allowed cleanup: none unless smoke and tests green.
- Command: focused pytest plus smoke command.

Validation:
- Commands: audit-only smoke command.
- Expected result: no figure files created.

Done criteria:
- [ ] Smoke passes or remains incomplete with reason.
- [ ] No PNG/PDF present.

Commit suggestion:
- message: `smoke test geodt audit-only cli`

### T047: Add reproduction metadata schema tests

Source:
- Requirement IDs: FR-REPRO-001-FR-REPRO-002, FR-031, SC-010, US3
- Plan section: Reproduction Mode Plan

Dependencies:
- T033, T045

Files:
- Edit: `<focused-test-path>`

TDD classification:
- strict RED test for later implementation

RED:
- Write/update: tests for metadata fields required to recover archive input, provider path, branches, checkpoints, feature source, K, plotted depths, top-K features, score, formula.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: reproduction schema checks/loading absent.

GREEN:
- Smallest implementation change: none in this task; T050 passes it later.
- Command: focused pytest.
- Expected passing result: later after T050.

REFACTOR:
- Allowed cleanup: test helper only.
- Command: focused pytest.

Validation:
- Commands: focused pytest.
- Expected result: RED failure for missing reproduction handling.

Done criteria:
- [ ] Metadata schema covers reproduction inputs.

Commit suggestion:
- message: `test geodt reproduction metadata schema`

### T048: Add reproduction mismatch tests

Source:
- Requirement IDs: FR-REPRO-003, FR-REPRO-006, FR-036A, SC-010, SC-021
- Plan section: Reproduction mismatch handling

Dependencies:
- T014, T047

Files:
- Edit: `<focused-test-path>`

TDD classification:
- strict RED test for later implementation

RED:
- Write/update: tests for missing checkpoint, changed top-K features, changed score, missing feature-name source, missing provider path.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: mismatch detection absent.

GREEN:
- Smallest implementation change: none in this task; T051/T052 pass it later.
- Command: focused pytest.
- Expected passing result: later after T052.

REFACTOR:
- Allowed cleanup: test helper only.
- Command: focused pytest.

Validation:
- Commands: focused pytest.
- Expected result: RED failure for missing mismatch behavior.

Done criteria:
- [ ] Missing/changed artifacts fail rather than reselect.

Commit suggestion:
- message: `test geodt reproduction mismatches`

### T049: Add `--reproduce-from` CLI contract tests

Source:
- Requirement IDs: FR-REPRO-001, FR-REPRO-004, FR-REPRO-005, US3
- Plan section: CLI contract; Reproduction Mode Plan

Dependencies:
- T019, T047

Files:
- Edit: `<focused-test-path>`

TDD classification:
- CLI contract test

RED:
- Write/update: tests for `--reproduce-from <metadata.json>` and default non-reselection behavior.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: reproduction CLI behavior absent.

GREEN:
- Smallest implementation change: none in this task; T050/T051 pass it later.
- Command: focused pytest.
- Expected passing result: later after reproduction implementation.

REFACTOR:
- Allowed cleanup: test helper only.
- Command: focused pytest.

Validation:
- Commands: focused pytest.
- Expected result: RED failure for missing CLI behavior.

Done criteria:
- [ ] Test asserts no default reselection.

Commit suggestion:
- message: `test geodt reproduce cli contract`

### T050: Implement reproduction metadata loading and artifact resolution

Source:
- Requirement IDs: FR-REPRO-001-FR-REPRO-002
- Plan section: Reproduction Mode Plan

Dependencies:
- T047, T049

Files:
- Edit: `scripts/plot_geodt_branch_tree_comparison.py`
- Tests: `<focused-test-path>`

TDD classification:
- strict TDD / CLI behavior

RED:
- Write/update: T047/T049 tests.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: metadata loading/resolution absent.

GREEN:
- Smallest implementation change: load metadata path, resolve recorded archive/provider/branches/checkpoints/feature source/K/formula inputs.
- Command: focused pytest.
- Expected passing result: metadata loading and CLI contract tests advance/pass where verification not yet required.

REFACTOR:
- Allowed cleanup: local metadata load helper only.
- Command: focused pytest.

Validation:
- Commands: reproduction CLI with fixture metadata once available.
- Expected result: recorded artifacts resolved.

Done criteria:
- [ ] `--reproduce-from` accepts metadata path.
- [ ] Recorded inputs are resolved, not reselected.

Commit suggestion:
- message: `implement geodt reproduction metadata loading`

### T051: Implement reproduction verification

Source:
- Requirement IDs: FR-REPRO-003-FR-REPRO-005, SC-010
- Plan section: Reproduction verification

Dependencies:
- T048, T050

Files:
- Edit: `scripts/plot_geodt_branch_tree_comparison.py`
- Tests: `<focused-test-path>`

TDD classification:
- strict TDD / data-artifact behavior

RED:
- Write/update: T048 mismatch tests and T049 non-reselection tests.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: verification absent.

GREEN:
- Smallest implementation change: verify recovered branch IDs, checkpoint paths, top-K feature sets, score, and figure inputs match metadata; no default reselection.
- Command: focused pytest.
- Expected passing result: verification tests pass or fail only on report formatting pending T052.

REFACTOR:
- Allowed cleanup: verification helper only.
- Command: focused pytest.

Validation:
- Commands: reproduction CLI with fixture metadata.
- Expected result: same selected pair recovered.

Done criteria:
- [ ] Changed artifacts produce mismatch.
- [ ] Bitwise-identical image is not required.

Commit suggestion:
- message: `implement geodt reproduction verification`

### T052: Implement reproduction report and mismatch failure summary

Source:
- Requirement IDs: FR-REPRO-006, FR-036A-B, SC-021
- Plan section: Failure Handling Strategy; Reproduction Mode Plan

Dependencies:
- T048, T051

Files:
- Edit: `scripts/plot_geodt_branch_tree_comparison.py`
- Tests: `<focused-test-path>`

TDD classification:
- strict TDD / artifact behavior

RED:
- Write/update: T048 mismatch report assertions.
- Command: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Expected failure: mismatch report/failure summary absent.

GREEN:
- Smallest implementation change: write/display reproduction report and structured mismatch summary without replacement selection.
- Command: focused pytest.
- Expected passing result: reproduction mismatch tests pass.

REFACTOR:
- Allowed cleanup: reuse existing failure summary helper only.
- Command: focused pytest.

Validation:
- Commands: reproduction CLI against missing-artifact fixture.
- Expected result: clear mismatch report and nonzero/clear failure according to tests.

Done criteria:
- [ ] Missing/changed artifacts do not trigger replacement selection.
- [ ] Report paths recorded when output dir available.

Commit suggestion:
- message: `implement geodt reproduction mismatch reports`

### T053: Run focused reproduction and mismatch tests

Source:
- Requirement IDs: US3, SC-010, SC-021
- Plan section: Validation tasks

Dependencies:
- T050, T051, T052

Files:
- Inspect: `<focused-test-path>`, temp reports
- Edit: validation summary/evidence only

TDD classification:
- validation-only exception

TDD exception:
- Required: yes
- Reason: Test execution only.
- Why strict TDD does not apply: Behavior already implemented by strict TDD tasks.
- Validation method: Run focused reproduction tests.
- Command or review criterion: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Risk: Medium.
- Approval source: tasks.md T053.

RED:
- Write/update: none
- Command: focused pytest.
- Expected failure: any failure must be classified.

GREEN:
- Smallest implementation change: fix scoped reproduction defects only.
- Command: focused pytest.
- Expected passing result: reproduction tests pass.

REFACTOR:
- Allowed cleanup: none unless green.
- Command: focused pytest.

Validation:
- Commands: focused pytest.
- Expected result: US3 tests pass.

Done criteria:
- [ ] Reproduction tests pass.
- [ ] Known baseline failures separated.

Commit suggestion:
- message: `validate geodt reproduction workflow`

### T054: Run reproduction CLI smoke

Source:
- Requirement IDs: US3, FR-REPRO-001, FR-REPRO-005, SC-010
- Plan section: Quickstart reproduction mode

Dependencies:
- T053

Files:
- Inspect: fixture metadata and reproduction output/report
- Edit: validation summary/evidence only

TDD classification:
- smoke validation exception

TDD exception:
- Required: yes
- Reason: Smoke execution only.
- Why strict TDD does not apply: Behavior already covered by tests.
- Validation method: Run reproduction command using fixture metadata.
- Command or review criterion: `python scripts/plot_geodt_branch_tree_comparison.py --reproduce-from <temporary-diagnostics-output>/<metadata.json>`
- Risk: Medium.
- Approval source: tasks.md T054.

RED:
- Write/update: none
- Command: reproduction smoke command.
- Expected failure: integration/environment issue if present; classify before fixing.

GREEN:
- Smallest implementation change: fix scoped integration defects only.
- Command: reproduction smoke command.
- Expected passing result: same selected pair recovered.

REFACTOR:
- Allowed cleanup: none unless green.
- Command: focused pytest plus smoke command.

Validation:
- Commands: reproduction smoke command.
- Expected result: selected pair recovered from metadata.

Done criteria:
- [ ] Reproduction smoke passes or remains incomplete with reason.
- [ ] No replacement pair selected.

Commit suggestion:
- message: `smoke test geodt reproduction cli`

### T055: Update quickstart if implemented behavior differs

Source:
- Requirement IDs: FR-AUDIT-001, FR-REPRO-001, FR-OUT-001, SC-018, SC-019
- Plan section: Quickstart

Dependencies:
- T005, T036, T046, T054

Files:
- Inspect/Edit: `specs/005-geodt-branch-tree-interpretability/quickstart.md`
- Tests: validation by command review only

TDD classification:
- documentation validation-only exception

TDD exception:
- Required: yes
- Reason: Documentation-only update if needed.
- Why strict TDD does not apply: No runtime behavior change.
- Validation method: Compare quickstart commands against implemented CLI and smoke commands.
- Command or review criterion: review `--help` and smoke commands.
- Risk: Low; stale docs mislead users.
- Approval source: tasks.md T055.

RED:
- Write/update: none unless mismatch found.
- Command: `python scripts/plot_geodt_branch_tree_comparison.py --help`
- Expected failure: not expected; if CLI unavailable, task is blocked.

GREEN:
- Smallest implementation change: update only mismatched quickstart lines; preserve planned sections.
- Command: review quickstart commands.
- Expected passing result: docs match implemented CLI behavior.

REFACTOR:
- Allowed cleanup: none beyond targeted doc correction.
- Command: not applicable.

Validation:
- Commands: compare quickstart examples to smoke commands.
- Expected result: quickstart remains accurate.

Done criteria:
- [ ] Quickstart updated or confirmed current.
- [ ] No scope expansion.

Commit suggestion:
- message: `align geodt diagnostic quickstart`

### T056: Update CLI contract if implemented behavior differs

Source:
- Requirement IDs: FR-OUT-001, FR-OUT-004, FR-AUDIT-001, FR-REPRO-001
- Plan section: Contract

Dependencies:
- T005, T036, T046, T054

Files:
- Inspect/Edit: `specs/005-geodt-branch-tree-interpretability/contracts/geodt-branch-tree-diagnostic-cli.md`
- Tests: docs review and CLI `--help`

TDD classification:
- documentation validation-only exception

TDD exception:
- Required: yes
- Reason: Contract doc update if implementation deviates within allowed spec.
- Why strict TDD does not apply: Documentation-only unless it reveals a source-artifact conflict.
- Validation method: Compare implemented flags/outputs to contract.
- Command or review criterion: CLI `--help` and smoke outputs.
- Risk: Medium; contract drift misleads future agents.
- Approval source: tasks.md T056.

RED:
- Write/update: none unless mismatch found.
- Command: `python scripts/plot_geodt_branch_tree_comparison.py --help`
- Expected failure: not expected; if behavior conflicts with contract, stop rather than silently documenting unauthorized change.

GREEN:
- Smallest implementation change: update contract only for authorized implemented behavior differences; otherwise fix implementation.
- Command: review contract vs help/output.
- Expected passing result: contract matches source-authorized implementation.

REFACTOR:
- Allowed cleanup: none beyond targeted doc correction.
- Command: not applicable.

Validation:
- Commands: compare contract to CLI and output artifacts.
- Expected result: no contract drift.

Done criteria:
- [ ] Contract updated or confirmed current.
- [ ] Unauthorized CLI changes are not documented as acceptable.

Commit suggestion:
- message: `align geodt diagnostic cli contract`

### T057: Optional real-archive audit-only dry run

Source:
- Requirement IDs: FR-000F, FR-000G, FR-AUDIT-001, FR-EVID-004, SC-018, SC-020, SC-022
- Plan section: Optional real archive smoke

Dependencies:
- T046

Files:
- Inspect: explicit bounded real archive path/root only
- Edit: audit summary/evidence artifact only

TDD classification:
- optional smoke validation exception

TDD exception:
- Required: yes
- Reason: Optional environment-dependent smoke.
- Why strict TDD does not apply: No behavior implementation; synthetic validation is sufficient for MVP.
- Validation method: Run audit-only with explicit bounded archive input.
- Command or review criterion: `python scripts/plot_geodt_branch_tree_comparison.py --audit-only --archive-path <real-archive-path> --output-dir <diagnostics-output>`
- Risk: Medium; real archive availability varies.
- Approval source: tasks.md marks T057 Optional and non-blocking.

RED:
- Write/update: none
- Command: real-archive audit-only command only if explicit bounded path/root is available.
- Expected failure: acceptable if artifacts incomplete; record structured result.

GREEN:
- Smallest implementation change: none unless smoke reveals scoped implementation defect reproducible in fixtures.
- Command: same audit-only command.
- Expected passing result: audit output or clear structured preflight failure.

REFACTOR:
- Allowed cleanup: none.
- Command: not applicable.

Validation:
- Commands: optional audit-only command.
- Expected result: evidence/audit record, no PNG/PDF, no production artifact modification.

Done criteria:
- [ ] If run, findings recorded outside `tasks.md`.
- [ ] If not run, MVP completion is not blocked.

Commit suggestion:
- message: `record optional geodt real archive audit`

### T058: Run source-boundary check

Source:
- Requirement IDs: FR-035, SC-009
- Plan section: Change Surface read-only files

Dependencies:
- T035, T045, T053

Files:
- Inspect: git diff/status for read-only files and production artifact dirs
- Edit: validation summary/evidence only

TDD classification:
- validation-only exception

TDD exception:
- Required: yes
- Reason: Boundary verification only.
- Why strict TDD does not apply: No runtime behavior change.
- Validation method: Confirm read-only source files and production artifacts are unchanged.
- Command or review criterion: `git status --short` and `git diff -- <read-only paths>`
- Risk: High; accidental Brownfield edits must be caught.
- Approval source: tasks.md T058.

RED:
- Write/update: none
- Command: `git status --short`
- Expected failure: not applicable; unexpected modified read-only files are blockers.

GREEN:
- Smallest implementation change: revert only unauthorized accidental changes after user-safe review; do not use destructive commands without confirmation.
- Command: status/diff commands.
- Expected passing result: no read-only files or production artifacts modified.

REFACTOR:
- Allowed cleanup: none.
- Command: not applicable.

Validation:
- Commands: status/diff review.
- Expected result: only allowed new diagnostic script/tests/docs and generated ignored diagnostics outputs changed.

Done criteria:
- [ ] Read-only files unchanged.
- [ ] Production artifact directories unchanged.

Commit suggestion:
- message: `verify geodt diagnostic source boundaries`

### T059: Run focused tests and summarize baselines/regressions

Source:
- Requirement IDs: SC-001, SC-003, SC-008, SC-010, SC-013, SC-018, SC-020, SC-021, SC-023, FR-EVID-004
- Plan section: Test Strategy; Known baseline failures

Dependencies:
- T035, T045, T053

Files:
- Inspect: focused test output
- Edit: validation summary/evidence only

TDD classification:
- validation-only exception

TDD exception:
- Required: yes
- Reason: Test execution/reporting only.
- Why strict TDD does not apply: Tests already created and implementation complete.
- Validation method: Run focused tests and record known baseline failures separately from new regressions.
- Command or review criterion: `python -m pytest <focused-test-path> -p no:cacheprovider`
- Risk: Medium.
- Approval source: tasks.md T059 and user Brownfield requirement.

RED:
- Write/update: none
- Command: focused pytest.
- Expected failure: any failure must be triaged.

GREEN:
- Smallest implementation change: fix scoped new regressions only; record known baseline/environment gaps separately.
- Command: focused pytest.
- Expected passing result: focused tests pass or task remains incomplete with documented blocker.

REFACTOR:
- Allowed cleanup: none unless green.
- Command: focused pytest.

Validation:
- Commands: focused pytest.
- Expected result: pass with clean summary.

Done criteria:
- [ ] Test results recorded.
- [ ] Known baseline failures separated from new regressions.

Commit suggestion:
- message: `record geodt diagnostic validation results`

### T060: Document rollback/no-migration conclusion

Source:
- Requirement IDs: FR-033, FR-035, FR-EVID-004, SC-009, SC-020
- Plan section: Migration / rollback requirements

Dependencies:
- T004, T058

Files:
- Edit: designated migration/evidence artifact or final implementation notes
- Inspect: generated files and output directories

TDD classification:
- documentation validation-only exception

TDD exception:
- Required: yes
- Reason: Rollback documentation only.
- Why strict TDD does not apply: No runtime behavior change.
- Validation method: Document rollback as deletion of new diagnostic script, `<focused-test-path>`, and generated diagnostics outputs only.
- Command or review criterion: review changed files and output directories.
- Risk: Medium; missing rollback note weakens Brownfield readiness.
- Approval source: tasks.md T060.

RED:
- Write/update: none
- Command: review git status and output directories.
- Expected failure: not applicable; if unauthorized changes exist, block until resolved.

GREEN:
- Smallest implementation change: write rollback/no-migration note.
- Command: review note.
- Expected passing result: rollback is clear and existing GeoDT artifacts need no migration rollback.

REFACTOR:
- Allowed cleanup: none.
- Command: not applicable.

Validation:
- Commands: review note plus T058 boundary result.
- Expected result: rollback note matches actual change surface.

Done criteria:
- [ ] Rollback note recorded.
- [ ] No migration rollback needed for existing artifacts.

Commit suggestion:
- message: `document geodt diagnostic rollback path`

## Global Validation

- lint: not found in current Speckit artifacts.
- typecheck: not found in current Speckit artifacts.
- unit tests: `python -m pytest <focused-test-path> -p no:cacheprovider`
- integration/smoke tests:
  - `python scripts/plot_geodt_branch_tree_comparison.py --audit-only --archive-path <synthetic-fixture-archive> --output-dir <temporary-diagnostics-output>`
  - `python scripts/plot_geodt_branch_tree_comparison.py --archive-path <synthetic-fixture-archive> --output-dir <temporary-diagnostics-output>`
  - `python scripts/plot_geodt_branch_tree_comparison.py --reproduce-from <temporary-diagnostics-output>/<metadata.json>`
- build: not applicable for this standalone Python diagnostic.
- deterministic checks: compare metadata candidate decisions, selected pair, output paths, and no-overwrite behavior across repeated fixture runs.
- source-boundary check: `git status --short` plus targeted diff review for read-only paths.
- speckit verify command, if available: not found in current artifacts.

## Final Self-Audit

- Every task maps to an existing Speckit task ID: yes, T001-T060 plus T018A.
- Every behavior-changing task has RED/GREEN/REFACTOR steps: yes.
- Every RED step has target, command, and expected failure: yes; placeholder targets depend on T002 as required by `tasks.md`.
- Every validation-only exception has a reason, validation method, risk, and approval source: yes.
- No task adds behavior absent from source artifacts: yes.
- No task changes architecture decisions: yes.
- Dependent tasks are not grouped as parallel: yes.
- Unknown paths are discovery steps or placeholders gated by T002/T009/T010: yes.
- Shared selection pipeline T041-T043 is completed before T032/T033/T035 and reused by T044; no duplicate selection logic exists.
- Brownfield constraints from the user command are enforced: yes.
- Only selected output file should be added by this planning step: `implementation-micro-plan.md`.

## Implementation Start Gate

Safe to start immediately:
- T001-T018A characterization, fixture, and RED gate work.

Not safe to start until prerequisites are complete:
- Any behavior-changing implementation task, especially T024 and later.
- T032/T033/T035 until T041-T043 shared selection pipeline is complete.

The micro-plan is implementation-ready only if:
- T018A is complete before any behavior-changing script edit;
- T041-T043 are completed before T032/T033/T035;
- T044 reuses T041-T043 rather than duplicating selection logic;
- T010 either follows T009 or records the explicit archive/source path used when it runs early.

## Implementation Constraint Prompt

Paste this before implementation:

```text
Before implementation, read specs/005-geodt-branch-tree-interpretability/implementation-micro-plan.md.

Treat resolved analyze findings/human notes, tasks.md, plan.md, spec.md, evidence artifacts, and supplementary Speckit docs as the source of truth, in that order.

Use implementation-micro-plan.md only as an execution guide. If it conflicts with source artifacts, stop and report the conflict.

Brownfield Mode is ON:
- no opportunistic refactor;
- no dependency upgrade unless explicitly in scope;
- no broad formatting or unrelated cleanup;
- no editing candidate paths until verified;
- no changing public API / CLI / config / data artifact behavior unless explicitly authorized;
- characterization / smoke / contract tests before behavior-changing edits in legacy areas;
- known baseline failures must be separated from new regressions;
- rollback or revert notes are required for risky changes.

Do not edit read-only GeoDT training, prediction dispatch, branch training, batch launcher, or dt_rules export files. Existing checkpoints, space_partitions, dt_rules, deliverables, and evaluation outputs are read-only inputs.

Do not create or edit scripts/plot_geodt_branch_tree_comparison.py until T018A is complete.

Signature extraction, Jaccard scoring, tie-breaks, and readability gating are shared selection-pipeline behavior required before US1 figure rendering. Implement them once in T041-T043 and reuse them for both figure-generation and audit-only modes.

For behavior-changing tasks, follow strict TDD:
1. write or update the failing test first,
2. run it and confirm the expected failure,
3. implement the smallest change,
4. run the relevant test until it passes,
5. refactor only after green.

For behavior-changing tasks, do not skip strict TDD unless a per-task TDD exception has been explicitly approved by the user or recorded in resolved human notes. If no approval exists, stop and request approval before editing.

Use this exception format when strict TDD is skipped:

TDD exception for <task ID>:
Reason:
Why strict TDD does not apply:
Validation method:
Command or review criterion:
Risk:
Approval source:

Run validation before marking work complete. If validation is not run, do not mark the task complete; state Validation Status: Not Executed, explain why, and ask for user acceptance or a runnable validation path.
```
