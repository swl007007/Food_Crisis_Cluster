# Implementation Plan: GeoDT Branch-Specific Local Tree Interpretability Figure

**Branch**: `005-geodt-branch-tree-interpretability` | **Date**: 2026-06-05 | **Spec**: `specs/005-geodt-branch-tree-interpretability/spec.md`  
**Input**: Feature specification from `/specs/005-geodt-branch-tree-interpretability/spec.md`

## Summary

Implement a separate exploratory GeoDT diagnostic that reads one archived/source monthly GeoDT run, validates branch-specific DecisionTree checkpoints and terminal branch assignments, ranks eligible branch pairs by deterministic top-K split-feature Jaccard dissimilarity, applies readability gates, and writes a 1x2 comparison figure plus metadata/audit outputs. The diagnostic must preserve existing GeoDT training, prediction dispatch, branch training, checkpoint artifacts, and root/global `dt_rules_*.csv` export behavior.

MVP completion means audit/figure mode can run on a complete GeoDT-like archive or fixture, prove both plotted trees came from branch-specific checkpoints, generate metadata proving archive/branch/feature/provenance decisions, and fail clearly when required artifacts are unavailable.

## 1. Technical Context

**Language/Version**: Python 3.12+ per project guidance.  
**Repository runtime**: WSL and Windows-compatible Python analysis environment; Windows CMD-facing text must be ASCII-safe.  
**Current GeoDT training/evaluation entry points**: `app/main_model_DT.py --start_year --end_year --forecasting_scope`; batch launcher `run_batches_2021_2024_visual_monthly.bat geodt`; existing `--no-dt-rules` / `SAVE_DT_RULES` behavior must remain unchanged.  
**DecisionTree implementation/library**: scikit-learn `DecisionTreeClassifier` based on existing `dt_rules` exporter and GeoDT evidence.  
**Checkpoint serialization format**: NEEDS EVIDENCE VALIDATION during Phase 1 by inspecting/loading through existing `DTmodel.load()` behavior; plan must not assume raw pickle/joblib internals beyond using the existing loader where feasible.  
**Plotting backend**: matplotlib plus scikit-learn tree plotting is sufficient for MVP; Graphviz is not required unless evidence later proves matplotlib cannot satisfy readability constraints.  
**Primary dependencies**: Python standard library (`argparse`, `json`, `pathlib`, `itertools`), numpy, pandas, matplotlib, scikit-learn; optional existing `src/model/model_DT.py` loader. No new framework dependency is planned.  
**Storage**: Read-only selected GeoDT archives/source run folders; generated ignored diagnostics outputs under selected archive `diagnostics/output/` or configured diagnostics root.  
**Testing**: pytest-style unit/fixture tests if test harness exists; otherwise focused Python smoke scripts under existing test conventions. Validation must include at least one synthetic GeoDT-like archive fixture and optionally one real archive dry-run if available.  
**Target Platform**: Local research environment in WSL/Python 3.12, with Windows path and CMD output compatibility.  
**Project Type**: Python research pipeline with exploratory diagnostic script.  
**Performance Goals**: Lightweight archive scan and plotting; bounded deterministic archive discovery; no model retraining; no full batch run required.  
**Constraints**: Read-only access to archived checkpoints, partition artifacts, existing `dt_rules`, evaluation outputs, and production deliverables; no unbounded filesystem search; no fallback from branch checkpoints to root/global `dt_rules`.  
**Scale/Scope**: One selected monthly GeoDT run, all eligible terminal branches in that run, one selected branch pair, one figure, metadata, audit summary, and optional reproduction report.

## 2. Constitution / Spec Compliance Check

- **Pipeline Contract**: Affected workflow is exploratory diagnostics only. Planned canonical entry point is a new script under `scripts/` with figure, audit-only, and reproduction modes. Inputs are archive path/list/root, optional artifact-provider source run folder, optional output directory, and optional metadata path for reproduction. Outputs are ignored diagnostics artifacts, not production forecasts or evaluation outputs.
- **Temporal Integrity**: No forecasting logic, lag schedule, `ACTIVE_LAGS`, feature-month mapping, target-month mapping, synthetic rows, or labels are changed. Preferred run identity is `2024-10 fs1`; fallback records exact selected year/month/scope.
- **Spatial Partitioning**: Spatial partitioning behavior is not changed. Terminal branch identity is read from selected-run dispatch artifacts only after evidence validation. Checkpoint filename presence is insufficient.
- **Threshold & Prediction Contract**: No prediction threshold is introduced or changed. Class tendency is display-only and only uses confirmed class-label mapping or neutral labels.
- **Geographic Scope**: No map rendering or shapefile join is planned. Geographic scope is inherited from the selected archive and recorded only if available.
- **Crisis-Class Validation**: This diagnostic is not model validation and does not compute class-1 metrics. Crisis/non-crisis labels are forbidden unless class-label mapping is confirmed from selected-run artifacts, pipeline constants, or another documented source.
- **Operational Hygiene**: Diagnostic outputs remain generated/ignored unless explicitly promoted. Console output must be ASCII-safe when intended for Windows CMD. The implementation must avoid notebook outputs, scratch artifacts, and writes into production result artifacts.
- **Brownfield Behavior & Archive Characterization**: Existing root/global `dt_rules` semantics, GeoDT training, prediction dispatch, and checkpoint naming behavior are preserved. First implementation phase is read-only evidence validation and archive preflight.

**Gate Result**: PASS. No constitution violation is required. If Phase 1 evidence proves that required artifacts cannot be discovered or safely loaded, implementation must remain blocked at preflight rather than modifying production code to compensate.

## Change Surface

- **Allowed files / modules**:
  - Add a focused diagnostic entry point under `scripts/`, likely `scripts/plot_geodt_branch_tree_comparison.py`.
  - Add tests under the repository's existing test area or a focused `tests/`/`src/tests/` location, depending on current conventions confirmed during implementation.
  - Add optional small reusable helper modules only if needed under `src/utils/` or `src/vis/`; avoid helper extraction until repetition justifies it.
  - Update Speckit docs under `specs/005-geodt-branch-tree-interpretability/`.
- **Read-only files / modules**:
  - `app/main_model_DT.py`
  - `src/model/GeoRF_DT.py`
  - `src/model/model_DT.py`
  - `src/model/train_branch.py`
  - `src/helper/helper.py`
  - `src/merge/terminal.py`
  - `src/utils/dt_rule_export.py`
  - `run_batches_2021_2024_visual_monthly.bat`
  - Existing `result_*`, `dt_rules/`, `checkpoints/`, `space_partitions/`, `other_outputs/`, and `deliverables/` artifacts.
- **Forbidden files / modules**:
  - Do not modify GeoDT training, prediction dispatch, branch training, lag schedules, batch launchers, production prediction scripts, or root/global `dt_rules` export behavior.
  - Do not alter archived checkpoints, partition artifacts, existing evaluation CSVs/workbooks, production deliverables, or existing `dt_rules` CSVs.
- **Public contracts touched**:
  - New exploratory CLI contract only. Existing public CLI/config/data contracts remain unchanged: `run_batches_2021_2024_visual_monthly.bat geodt`, `--no-dt-rules`, `SAVE_DT_RULES`, `SAVE_DT_NODE_DUMP`, `app/main_model_DT.py` arguments, `dt_rules/*.csv`, `checkpoints/dt_*`, and `space_partitions/*`.
- **Runtime config touched**:
  - None for production config. The diagnostic may accept command-line arguments but must not change `config.py` defaults or environment-controlled training behavior.
- **Data artifacts touched**:
  - Read: selected archive/source run folders, branch checkpoints, branch assignments, feature-name sources, correspondence tables, optional existing logs/manifests.
  - Write: new diagnostics output directory only, containing PNG/PDF, metadata JSON, audit summary, reproduction report, and failure summary.
- **Migration / rollback requirements**:
  - No data migration. Rollback is removal of the new diagnostic script/tests/docs and generated diagnostic outputs. Existing model artifacts are read-only and unaffected.
- **Compatibility risks**:
  - Checkpoint serialization may be environment-sensitive.
  - Visual archives may lack `checkpoints/` or `space_partitions/`, requiring same-run/source artifact-provider validation.
  - Feature-name source mismatch can make otherwise loadable checkpoints unusable.
  - Misidentifying root/global checkpoint as branch-specific would invalidate the figure.
- **Required characterization or regression coverage before edit**:
  - Dry-run evidence validation of selected archive/source artifact availability.
  - Root/global `dt_rules` boundary check.
  - Checkpoint parsing/load smoke check.
  - Branch assignment source precedence characterization.
  - Feature-name compatibility characterization.
  - Artifact write-boundary check proving no production artifacts are overwritten.

## 3. Evidence Validation Plan

Phase 1 implementation must begin with a read-only dry-run evidence validation and archive preflight. The implementation must validate these assumptions before relying on them:

1. Existing `dt_rules_*.csv` exports represent root/global rules when generated from `branch_id=''`.
   - Validate via repository code path evidence and metadata note; do not use `dt_rules` as branch rule source.
2. `space_partitions/X_branch_id.npy`, when present, is a final prediction dispatch artifact and has higher precedence than `s_branch.pkl`.
   - Validate shape/type and relationship to selected archive/source run; if not mappable, reject or lower precedence.
3. `space_partitions/s_branch.pkl` maps admin groups or prediction rows to terminal branch IDs for the selected run.
   - Validate object schema/columns and branch ID values; reject if not tied to selected run.
4. `correspondence_table_*.csv` contains terminal branch IDs for the same run/month/scope before it is used as an assignment source.
   - Validate presence of `branch_id` or equivalent terminal lineage; reject display-only partition labels.
5. Checkpoint filenames such as `dt_`, `dt_0`, `dt_1`, `dt_00`, and `dt_01` can be parsed into branch IDs consistently with the selected run's checkpoint loader.
   - Validate by invoking existing loader behavior where feasible and comparing parsed branch IDs to dispatch branch IDs.
6. Any configured feature-column file corresponds to the selected run/month/scope.
   - Validate source provenance and count against checkpoint `n_features_in_` when available; reject mismatch.

If any assumption cannot be validated, the diagnostic must use a lower-precedence confirmed source, mark the affected artifact/candidate unusable, or fail preflight with a structured evidence-mismatch report. Evidence status must be recorded as confirmed, rejected, or not applicable in metadata.

## 4. Implementation Phases

1. **Phase 1: Read-only preflight and evidence validation**
   - Implement bounded archive discovery, fallback candidate enumeration, same-run/source artifact-provider resolution, artifact completeness checks, and structured failure summaries.
   - No plotting until this phase confirms at least two eligible branch-specific checkpoints and reliable assignment/feature sources.
2. **Phase 2: Candidate loading and branch eligibility**
   - Classify checkpoints, validate branch-ID parsing, select assignment source, load candidate DecisionTree checkpoints, validate feature-name mapping, and record eligible/ineligible branches.
3. **Phase 3: Signature extraction, scoring, readability gating, and audit-only mode**
   - Extract top-K split-feature signatures, compute deterministic Jaccard scores, apply tie-breaks and readability checks, write audit output, and prove audit-only selection matches figure mode.
4. **Phase 4: Figure generation and metadata output**
   - Render the 1x2 shallow tree comparison, write PNG/PDF as configured, produce metadata JSON and optional audit markdown, and enforce no-overwrite behavior.
5. **Phase 5: Reproduction mode and regression coverage**
   - Implement metadata-driven reproduction checks, mismatch reports, and tests/smoke commands against fixture and available real archive.
6. **Phase 6: Polish and brownfield boundary verification**
   - Verify ASCII-safe output, generated artifact hygiene, no production code path changes, no `dt_rules` reinterpretation, and documented MVP/deferred items.

## 5. Data and Artifact Discovery Strategy

- Support four discovery inputs: explicit archive path, explicit archive list, archive discovery root, configured project result/archive root.
- Reject unbounded recursive search of arbitrary user directories, temporary folders, or filesystem root.
- Prefer exact archive `result_GeoDT_2024_fs1_2024-10_visual` for `2024-10 fs1`.
- If the selected visual archive lacks branch artifacts, allow a same-run/source artifact-provider folder only when preflight confirms compatibility by model family, year/month, forecasting scope, and run identity.
- Fallback order:
  1. Exact preferred archive if complete.
  2. Same `fs1`, smallest absolute month distance from `2024-10`, if complete.
  3. Any GeoDT monthly archive, smallest absolute month distance from `2024-10`, if complete.
  4. Tie-break by most complete minimum required artifact set.
  5. Fail preflight if none complete.
- Minimum required artifact checks:
  - selected archive directory exists;
  - at least two loadable branch-specific DecisionTree checkpoints;
  - one reliable branch assignment source;
  - one reliable feature-name source;
  - tree objects expose non-leaf and leaf structure;
  - assignment data can compute assigned admin/group or prediction-row counts.
- Class-to-label mapping and training-sample count are optional; root/global-only `dt_rules` are insufficient.
- Metadata records discovery input, root/path/list, candidate count, all fallback candidates, accept/reject reasons, final selected archive, source artifact-provider path if used, and same-run compatibility evidence.

## 6. Component Design

### Archive Resolver

- CLI-facing component that normalizes archive inputs and produces deterministic candidate list.
- Does not scan outside explicit/configured roots.
- Computes preferred/fallback ordering and records candidate provenance.

### Artifact Characterizer

- Inspects selected archive and optional artifact-provider source folder.
- Checks for `checkpoints/dt_*`, `space_partitions/X_branch_id.npy`, `space_partitions/s_branch.pkl`, `branch_table.npy`, `correspondence_table_*.csv`, feature-name candidates, and existing `dt_rules` only as boundary evidence.
- Produces minimum-artifact completeness result.

### Assignment Source Selector

- Applies precedence: `X_branch_id.npy`, `s_branch.pkl`, reliable correspondence table, documented equivalent artifact, failure.
- Records selected/rejected sources and evidence validation status.

### Feature-Name Source Selector

- Applies precedence: checkpoint/model bundle names, pipeline feature column file, recoverable training matrix column order, compatible configured feature list, failure.
- Validates feature-name count, checkpoint feature count, and split index bounds.

### Checkpoint Parser and Loader

- Classifies root/global, branch-specific candidate, unusable, or unknown.
- Parses filenames using existing loader semantics where feasible.
- Loads branch-specific `DecisionTreeClassifier` objects through existing `DTmodel` behavior or a documented compatible loader.
- Never treats filename pattern alone as terminal branch evidence.

### Branch Eligibility Builder

- Joins dispatch branch IDs, counts, checkpoint classification, load result, and feature mapping result.
- Records eligible, ineligible, failed, skipped branches and exclusion reasons.

### Signature and Pair Selector

- Extracts split feature sets at depths `0` through `K-1`, default `K=3`.
- Computes Jaccard distance and optional threshold/direction supplemental audit fields.
- Applies deterministic tie-breaks and readability gating.

### Figure Renderer

- Uses matplotlib/scikit-learn plotting for 1x2 layout.
- Enforces minimum 10x4 inches, 300 DPI PNG, PDF vector text when supported, matched depth/style, label wrapping/unique abbreviations, and safe class labeling.

### Metadata/Audit/Reproduction Writer

- Writes JSON metadata, audit markdown or JSON, reproduction report, and failure summary.
- Uses no-overwrite or unique filename behavior.

## 7. Failure Handling Strategy

- All failures after preflight begins produce or display a structured failure summary with failure stage, missing/invalid artifacts, evidence mismatch, candidate archive decisions, and recommended next action.
- If output directory is unavailable, failure summary may be console-only.
- Failure never produces a misleading partial figure.
- Failure never silently falls back to root/global `dt_rules`.
- Candidate-specific failures mark individual branches/artifacts unusable when enough alternatives remain; global failures stop preflight.
- Reproduction mismatch fails with recorded mismatch details and does not reselect a replacement pair unless explicitly requested.

## 8. Test Strategy

### Unit tests

1. Archive discovery is bounded and deterministic.
2. Preferred archive exact match is selected when complete.
3. Fallback archive selection follows FR-003A.
4. Archive with only root/global `dt_rules` is incomplete.
5. Branch assignment source precedence works.
6. Feature-name source precedence works.
7. Feature count mismatch marks checkpoint unusable.
8. Out-of-bounds split feature index marks checkpoint unusable.
9. Root/global checkpoint is excluded unless explicitly terminal.
10. Checkpoint filename parsing records branch IDs and mismatches.
11. Eligible/ineligible branches are documented.
12. Jaccard scoring is deterministic.
13. Threshold/direction supplemental fields do not affect ranking.
14. Tie-break rules are deterministic.
15. Highest-scoring unreadable pair is rejected and next readable pair is selected.
16. No readable pair produces structured failure.

### Fixture-based tests

- Synthetic GeoDT-like archive with `checkpoints/`, `space_partitions/`, feature names, and correspondence table variations.
- Fixture where visual archive lacks branch artifacts but matching source folder provides them.
- Fixture with only root/global `dt_rules`.
- Fixture with feature-name mismatch and out-of-bounds split index.
- Fixture with class mapping unavailable to verify neutral labels.

### Smoke tests

- Audit-only smoke test against a synthetic archive.
- Figure-generation smoke test verifying PNG/PDF creation, dimensions/DPI/readability metadata, and no overwrite behavior.
- Reproduction smoke test using prior metadata.
- Optional real-archive dry run if a complete GeoDT monthly archive/source folder exists locally.
- Regression boundary check that existing GeoDT artifacts are not overwritten and source read-only files remain unchanged.

## 9. Metadata / Provenance Strategy

Write structured JSON metadata as the canonical machine-readable provenance artifact. Include at minimum:

- workflow mode;
- archive discovery input;
- archive discovery root or explicit archive path/list;
- discovered archive candidate count;
- selected archive folder;
- artifact-provider source run folder, if different;
- artifacts provided by each folder and same-run compatibility evidence, if used;
- selected run year/month;
- forecasting scope;
- minimum required artifact completeness result;
- fallback selection reason;
- all considered fallback candidates and rejection reasons;
- model family;
- branch assignment source path/type;
- rejected assignment sources and reasons;
- evidence validation status for branch assignment source;
- feature-name source path;
- rejected feature-name sources and reasons;
- feature-name count;
- checkpoint feature count, if available;
- feature-name compatibility result;
- evidence validation status for feature-name source;
- selected branch IDs;
- checkpoint paths used;
- checkpoint filename, parsed branch ID, classification, parsing method, mismatch reason;
- evidence validation status for checkpoint branch-ID parsing;
- root/global exclusion decision;
- evidence validation status for root/global `dt_rules` boundary;
- K value and actual plotted depth for each tree;
- dissimilarity formula name/version;
- selected pair score and contrast descriptor;
- top-K split features per selected branch;
- supplemental threshold/direction fields, if computed;
- assigned admin/group count, prediction-row count if available, training-sample count if available;
- class-label mapping source if used;
- class output availability and leaf color encoding status;
- readability check result;
- rejected higher-scoring pair summary;
- tie-break result;
- failed/skipped candidate branch summary;
- output figure paths;
- audit summary paths;
- reproduction report paths, if any;
- failure summary path, if applicable;
- package/checkpoint loading notes when relevant.

Audit-only mode may write a markdown summary for human inspection, but JSON metadata remains the reproducibility source.

## 10. Output Artifact Strategy

- Accept an output directory argument.
- If absent, write under `diagnostics/output/` inside selected archive or a configured project diagnostic output root.
- Do not overwrite existing files unless explicitly requested/configured.
- If paths conflict, create deterministic unique filenames or fail clearly.
- Suggested filenames:
  - `geodt_branch_tree_compare_<YYYY-MM>_<fs>_<branchA>_vs_<branchB>.png`
  - `geodt_branch_tree_compare_<YYYY-MM>_<fs>_<branchA>_vs_<branchB>.pdf`
  - `geodt_branch_tree_compare_<YYYY-MM>_<fs>_<branchA>_vs_<branchB>_metadata.json`
  - `geodt_branch_tree_compare_<YYYY-MM>_<fs>_<branchA>_vs_<branchB>_audit.md`
- Generated outputs remain ignored unless explicitly reviewed/promoted.
- No output may be written into `checkpoints/`, `space_partitions/`, `dt_rules/`, production `deliverables/`, or standard evaluation output locations.

## 11. Risks and Open Questions

### Risks

- Preferred archive may not exist or may lack branch artifacts.
- Source artifact-provider folder may be hard to match to visual archive without strong run identity evidence.
- Checkpoint loading may fail due to serialization/package-version mismatch.
- Feature-name source may be absent or incompatible with checkpoint feature count.
- `X_branch_id.npy` may not be mappable to admin/group counts without additional run context.
- Tree labels may be too long for readable 1x2 plotting; readability gate may reject high-scoring pairs.
- Class-label semantics may be unavailable, requiring neutral labels.

### Mitigations

- Make archive preflight the first phase and fail before plotting.
- Record artifact-provider compatibility evidence when source folders supply missing artifacts.
- Use existing loader behavior instead of raw deserialization where feasible.
- Treat feature-name mismatch and split-index bounds as checkpoint unusability, not a plotting fallback.
- Keep Graphviz out of MVP unless matplotlib/sklearn cannot satisfy readability.
- Use audit-only mode to validate candidate selection before rendering.

### Open Questions Before Coding

- Which actual local archive/source folder, if any, will be used for the first real smoke run?
- Where are selected-run feature-name artifacts stored in complete GeoDT outputs? NEEDS EVIDENCE VALIDATION.
- Does `X_branch_id.npy` align with rows, groups, or another artifact in archived monthly outputs? NEEDS EVIDENCE VALIDATION.
- Which existing test framework/location should hold fixture tests if no current pytest layout is confirmed? NEEDS EVIDENCE VALIDATION.

## 12. Implementation Task Breakdown Preview

1. Add synthetic GeoDT-like archive fixtures and expected metadata examples.
2. Add archive discovery and fallback selection tests.
3. Add read-only archive preflight and evidence validation component.
4. Add assignment-source and feature-name-source selector tests.
5. Add checkpoint parser/load characterization tests.
6. Add branch eligibility builder and branch exclusion reporting.
7. Add top-K signature extraction, deterministic Jaccard scoring, supplemental threshold/direction audit fields, and tie-break logic.
8. Add readability gate and failure-summary behavior.
9. Add audit-only mode and audit output.
10. Add 1x2 figure renderer with safe labels, dimensions/DPI checks, and no-overwrite output behavior.
11. Add metadata JSON writer and validation checks.
12. Add reproduction mode and mismatch report tests.
13. Add smoke tests for synthetic archive, optional real archive, and source artifact-provider folder.
14. Verify brownfield boundaries: no production code path, batch launcher behavior, checkpoint artifacts, partition artifacts, or `dt_rules` behavior changed.

**MVP includes**: archive preflight, branch checkpoint loading, branch eligibility, deterministic scoring, readability-gated pair selection, audit output, one PNG/PDF figure, metadata JSON, structured failure summary, and synthetic archive smoke test.

**Deferred unless needed**: Graphviz rendering, interactive notebooks, map/geographic labels, custom reusable visualization module, support for non-GeoDT model families, and promoted deliverable packaging.

## Project Structure

### Documentation (this feature)

```text
specs/005-geodt-branch-tree-interpretability/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── geodt-branch-tree-diagnostic-cli.md
└── tasks.md
```

### Source Code (planned)

```text
scripts/
└── plot_geodt_branch_tree_comparison.py     # New exploratory diagnostic entry point

src/utils/                                  # Optional, only if helper reuse is justified
src/vis/                                    # Optional, only if renderer reuse is justified

tests/ or src/tests/                         # Exact test location to validate during implementation
└── test_geodt_branch_tree_diagnostic*.py
```

**Structure Decision**: Prefer one focused script under `scripts/` for MVP. Keep existing GeoDT training/dispatch/export modules read-only unless a later, explicitly approved spec revision changes the feature scope.

## Complexity Tracking

No constitution violations are required. Complexity is managed by preflight-first implementation, strict read-only artifact boundaries, and synthetic fixtures before real archive smoke testing.
