# Release Cleanup and Non-Paper Pipeline Archive Design

## Goal

Prepare a cleaner release-facing repository view for the current paper
reproducibility version while preserving provenance for experimental and legacy
workflows. The release should make the GeoRF paper reproduction path obvious,
keep GeoDT as an artifact-only appendix source, and move non-paper entry points
out of the main workflow surface.

## Current Context

- The paper-facing workflow is the no-leak three-stage GeoRF workflow:
  2018-2020 partition learning, fixed consensus maps, and 2021-2024 evaluation.
- `paper_reproducibility_package/` already provides a fast audit package with
  Stage 2 maps, Stage 3 core outputs, final paper artifacts, and lightweight
  ablation provenance.
- `final_artifacts_in_paper_updated/` is the active final paper artifact
  directory.
- `scripts/verify_current_results_reproducibility.py` is the acceptance verifier
  for the current result bundle.
- GeoXGB, fs0 lag-1, and 2026-2027 forward/scenario prediction are explicitly
  non-paper workflows.

## Approved Direction

Use a combined cleanup and release approach:

1. Clean the current `main` repository surface in place.
2. Preserve non-paper workflows under an archive directory.
3. Keep verifier-dependent results in place.
4. Create a release tag/package from the cleaned state.

## Release Scope

### Included in the release main view

- `README.md` quick start for paper reproduction.
- `paper_reproducibility_package/` validation path.
- GeoRF full regeneration path:
  - `run_batches_2018_2020_partition_learning_visual_monthly.bat georf`
  - `spatial_weighted_consensus_clustering.bat georf`
  - `run_partition_k40_comparison_unified.bat georf --visual --month-ind`
- `final_artifacts_in_paper_updated/`.
- Current result directories needed by `verify_current_results_reproducibility.py`.
- GeoDT outputs and paper artifacts as appendix provenance, without making GeoDT
  a release quickstart workflow.

### Excluded from the release main view

- GeoXGB workflow entry points and GeoXGB experiment-facing scripts.
- fs0 lag-1 workflow commands and launch guidance.
- 2026-2027 forward prediction and scenario prediction launchers.
- Legacy notebooks under `scripts/`.
- Regional exploratory entry points.
- Legacy baseline or demo entry points not referenced by the paper release path.

## Archive Design

Create a new archive root:

```text
archived/release_20260624_nonpaper_pipelines/
```

The archive is historical provenance, not a maintained runnable surface. Moved
scripts are not required to keep working from their archived location.

Proposed archive subdirectories:

```text
archived/release_20260624_nonpaper_pipelines/
|-- README.md
|-- MANIFEST.csv
|-- prediction_pipeline/
|-- scripts_prediction/
|-- geoxgb_workflow/
|-- notebooks_legacy/
`-- legacy_misc/
```

Archive categories:

- `prediction_pipeline/`: the 2026-2027 forward/scenario `.bat` launchers.
- `scripts_prediction/`: prediction-only Python implementations such as
  `predict_partitioned_2026_2027.py` and `predict_scenario_2026_2027.py`.
- `geoxgb_workflow/`: `app/main_model_XGB.py`, GeoXGB-specific comparison or
  rename scripts, and any GeoXGB workflow notes identified during dry-run.
- `notebooks_legacy/`: legacy `.ipynb` files currently under `scripts/`.
- `legacy_misc/`: demo, baseline, regional exploratory, or old helper entry
  points that are not part of the paper release workflow.

The implementation plan must include a dry-run inventory before moving files.
That inventory becomes the archive `MANIFEST.csv` and should record old path,
new archive path, category, and reason.

## Paper Artifact Script Layout

Do not archive scripts that generate or audit paper artifacts. Move them into a
dedicated subpackage-like directory:

```text
scripts/paper_artifacts/
```

This keeps provenance close to the release while reducing noise in the root
`scripts/` directory.

Initial candidates:

- `scripts/analyze_georf_*.py`
- `scripts/plot_georf_*.py`
- `scripts/plot_global_cluster_map_2x2_refined.py`
- `scripts/plot_predictions_2024.py`
- `scripts/plot_region_class_prevalence.py`
- `scripts/build_georf_*.py`
- `scripts/build_feature_exclude_ablation_workbook.py`
- `scripts/create_region_performance_partitioned_pooled_fewsnet.py`
- `scripts/audit_final_artifact_sources.py`
- `scripts/relabel_final_artifact_horizons.py`
- `scripts/paper_horizon_labels.py`

Any moved script referenced by tests, package docs, final artifact audit
metadata, or `verify_current_results_reproducibility.py` must have references
updated. If reference updates would create high risk, the implementation may use
thin compatibility wrappers in `scripts/` for the affected paper-artifact
commands, but wrappers should be avoided unless tests or documented commands
need them.

## Files and Results to Keep In Place

Keep these stable to avoid unnecessary verifier churn:

- `paper_reproducibility_package/`
- `final_artifacts_in_paper_updated/`
- `GeoRFExperiment/` and `GeoDTExperiment/` Stage 2/Stage 1 handoff artifacts
  needed by the verifier.
- `result_partition_k40_compare_GF_fs{1,2,3}/`
- `result_partition_k40_compare_DT_fs{1,2,3}/`
- `result_partition_k40_compare_GF_thresholded_fs{1,2,3}/`
- `main_ablation_exclude_updated_stage3_fixed_partitions/`
- `archived/no_leak_partition_learning_2018_2020_20260616/`
- `src/model/*XGB*` and `src/utils/lag_schedules.py` compatibility code.

Keeping these directories in place means the release cleanup focuses on visible
entry points and workflow documentation, not on result storage migration.

## Documentation Updates

Update the release-facing documentation so the first-read surface matches the
release scope:

- `README.md`
  - Show only the package validation and GeoRF regeneration quickstart.
  - State that GeoDT is retained as appendix/artifact provenance.
  - Remove fs0, GeoXGB, and 2026-2027 prediction commands from main sections.
  - Link to the archive README for non-paper workflows.
- `PIPELINE_WORKFLOW.md`
  - Make GeoRF the primary active pipeline.
  - Move fs0 and prediction-only details into archive/reference notes or a
    clearly marked non-release section.
  - Keep GeoDT as appendix provenance rather than a quickstart workflow.
- `CURRENT_RESULTS_REPRODUCTION.md`
  - Keep verifier/result-bundle instructions aligned with the retained
    directories.
- `paper_reproducibility_package/README.md` and generated package text
  - If text currently says non-paper workflows remain in current paths, update
    it to point to the new archive.
- `scripts/build_paper_reproducibility_package.py`
  - Update generated documentation strings that reference archived non-paper
    workflow paths.

Add a release manifest:

```text
RELEASE_MANIFEST.md
```

The manifest should list release scope, active commands, archived categories,
verification commands, and the planned release tag.

## Release Tag and Package

After cleanup and verification, create an annotated tag:

```text
v1.0-paper-reproducibility-20260624
```

The release package can be the git tag plus `RELEASE_MANIFEST.md`. A separate
zip or tarball is optional and should not be produced unless explicitly
requested during implementation review.

## Verification Requirements

The implementation is acceptable only if these checks pass:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest src.tests.test_paper_reproducibility_package -v
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_paper_reproducibility_package.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_current_results_reproducibility.py
git diff --check
git status --short
```

Additional release-scope checks:

- `README.md` quickstart does not expose GeoXGB, fs0, forward prediction, or
  scenario commands.
- `PIPELINE_WORKFLOW.md` does not present non-paper workflows as release
  quickstart paths.
- The archive manifest contains every moved file.
- `paper_reproducibility_package/MANIFEST.csv` still matches tracked package
  files.
- No file required by `scripts/verify_current_results_reproducibility.py` is
  moved into archive.
- If `scripts/verify_current_results_reproducibility.py` rewrites only
  `final_artifacts_in_paper_updated/artifact_source_audit.md`, restore that file
  before committing final cleanup.

## Risks and Mitigations

- Risk: moving paper artifact scripts breaks tests or audit references.
  Mitigation: move paper artifact scripts separately from archive moves, update
  references immediately, and run focused tests after the move.
- Risk: archive moves accidentally include verifier-required artifacts.
  Mitigation: generate a dry-run manifest and compare it against
  `scripts/verify_current_results_reproducibility.py` expectations before
  moving files.
- Risk: release docs still expose non-paper workflows as active commands.
  Mitigation: add keyword checks for `GeoXGB`, `fs0`, `prediction_pipeline`,
  `scenario`, and `2026-2027` in quickstart sections.
- Risk: `.gitignore` hides archived CSV/PNG/XLSX assets from commits.
  Mitigation: use explicit tracked-file checks for the archive manifest and use
  forced add only for files intentionally included in the release cleanup commit.

## Out of Scope

- Removing low-level GeoXGB compatibility modules from `src/model/`.
- Removing fs0 support from `src/utils/lag_schedules.py` or shared CLIs.
- Migrating verifier-dependent result directories into archive.
- Guaranteeing archived scripts remain runnable after movement.
- Creating a separate minimal public repository.
- Uploading a remote GitHub release.
