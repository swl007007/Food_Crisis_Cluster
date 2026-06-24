# Clean-Root Paper Reproducibility Release Design

## Purpose

Create a post-release cleanup that keeps the existing
`v1.0-paper-reproducibility-20260624` tag unchanged, then prepares a cleaner
root directory for a follow-up release:

- Proposed tag: `v1.0.1-paper-reproducibility-clean-root-20260624`
- Primary goal: make the repository root visually and operationally cleaner.
- Non-goal: rerun Stage 1, Stage 2, Stage 3, ablations, or paper artifact
  generation.

The current release entry points stay easy to find:

- `README.md`
- `PIPELINE_WORKFLOW.md`
- `CURRENT_RESULTS_REPRODUCTION.md`
- `RELEASE_MANIFEST.md`
- `final_artifacts_in_paper_updated/`
- `paper_reproducibility_package/`
- `scripts/validate_paper_reproducibility_package.py`
- `scripts/verify_current_results_reproducibility.py`

## Current Problem

The tracked release surface is already cleaner after the v1.0 cleanup, but the
working directory root still contains many local outputs and legacy workspaces:

- large verifier inputs such as `GeoRFExperiment/`, `GeoDTExperiment/`,
  `main_ablation_exclude_updated_stage3_fixed_partitions/`, and
  `result_partition_k40_compare_*`
- local residue such as `writing/`, `dt_rules/`, `monthly_results/`, old images
  under `other_outputs/`, and temporary files under `scripts/`
- legacy or regional working folders such as `demo/`, `prediction_pipeline/`,
  `regional_ablation_results/`, and `fewsnet_baseline_results/`
- legacy helper scripts that are not part of the paper package builder,
  package validator, current-result verifier, or documented no-leak rerun path

Some of these are ignored and untracked, but they still make the local project
root hard to inspect. Some are still required by the live verifier and package
builder, so moving them requires path updates.

## Archive Classes

### 1. Reproducibility Inputs Archive

Create:

```text
archived/release_20260624_reproducibility_inputs/
```

This archive is for directories that remain part of the local reproducibility
contract but should not stay in the root:

- `GeoRFExperiment/`
- `GeoDTExperiment/`
- `main_ablation_exclude_updated_stage3_fixed_partitions/`
- `result_partition_k40_compare_GF_fs1/`
- `result_partition_k40_compare_GF_fs2/`
- `result_partition_k40_compare_GF_fs3/`
- `result_partition_k40_compare_DT_fs1/`
- `result_partition_k40_compare_DT_fs2/`
- `result_partition_k40_compare_DT_fs3/`
- `result_partition_k40_compare_GF_thresholded_fs1/`
- `result_partition_k40_compare_GF_thresholded_fs2/`
- `result_partition_k40_compare_GF_thresholded_fs3/`
- `fewsnet_baseline_results/`

Large ignored directories must not be force-added to git. The implementation
should track only lightweight documentation and manifests when practical.

### 2. Legacy Script And Workspace Archive

Create:

```text
archived/release_20260624_legacy_workspace/
```

This archive is for tracked helper scripts and script-owned outputs that are not
part of the quick release validation path:

- `other_outputs/` as a whole, including the tracked aggregation helpers and
  historical aggregate workbooks/images
- root-level exploratory helper scripts that a dependency scan confirms are not
  imported by the package builder, package validator, current-result verifier,
  `scripts/paper_artifacts/`, or the documented three-stage no-leak rerun path
- obsolete temporary script snapshots such as `scripts/*.tmp.*`

The implementation should keep these release-facing scripts in `scripts/`:

- `scripts/build_paper_reproducibility_package.py`
- `scripts/validate_paper_reproducibility_package.py`
- `scripts/verify_current_results_reproducibility.py`
- `scripts/paper_artifacts/`
- Stage 1/2/3 scripts that are still called by the documented batch workflow

The script move must be conservative. If a script's dependency status is
unclear, leave it in place and record the reason in the archive manifest rather
than breaking a reproducibility path.

### 3. Local Workspace Residue Archive

Create:

```text
archived/local_workspace_residue_20260624/
```

This archive is for material that is not required for the release verifier:

- `writing/`
- `dt_rules/`
- `monthly_results/`
- empty or ignored local residue under `demo/` and `prediction_pipeline/`
- legacy regional exploratory outputs under `regional_ablation_results/`

This archive is a local cleanup archive, not a runnable release workflow. It may
contain ignored files. The implementation should not force-add large ignored
artifacts to git.

### 4. Keep In Root

Keep these in the root for review ergonomics:

- `final_artifacts_in_paper_updated/`
- `paper_reproducibility_package/`
- active source/config/test directories
- release documentation and manifest files

## Required Code Changes

### Centralized Path Resolution

Add a small path-resolution layer instead of scattering new archive paths across
scripts. At minimum, `scripts/verify_current_results_reproducibility.py` and
`scripts/build_paper_reproducibility_package.py` need a single source of truth
for the archived reproducibility-input root.

The path layer should:

- default to `archived/release_20260624_reproducibility_inputs/`
- keep `final_artifacts_in_paper_updated/` and
  `paper_reproducibility_package/` in root
- provide clear names for Stage 2 experiment roots, Stage 3 result roots,
  thresholded result roots, ablation root, and FEWS NET baseline root
- fail with useful messages if a required archived directory is missing
- keep script/archive paths explicit enough that docs can distinguish release
  quick validation from historical full-rerun workspaces

### Verifier Updates

Update `scripts/verify_current_results_reproducibility.py` so it reads:

- Stage 1/2 handoff artifacts from the archived `GeoRFExperiment/` and
  `GeoDTExperiment/`
- Stage 3 outputs from archived `result_partition_k40_compare_*`
- fixed-partition ablation outputs from archived
  `main_ablation_exclude_updated_stage3_fixed_partitions/`
- thresholded GeoRF provider manifests from archived thresholded result folders
- FEWS NET baseline source CSVs from archived `fewsnet_baseline_results/` for
  baseline artifact audits

The verifier should still write artifact source audit outputs to
`final_artifacts_in_paper_updated/`.

### Package Builder Updates

Update `scripts/build_paper_reproducibility_package.py` so it rebuilds the
package from archived reproducibility inputs while preserving the existing
package layout and validator behavior.

The package root remains:

```text
paper_reproducibility_package/
```

### Documentation Updates

Update:

- `README.md`
- `PIPELINE_WORKFLOW.md`
- `CURRENT_RESULTS_REPRODUCTION.md`
- `RELEASE_MANIFEST.md`

These docs should make clear that heavy local reproducibility inputs live under
`archived/release_20260624_reproducibility_inputs/`, while the quick validation
path remains the root-level `paper_reproducibility_package/`.

Docs should no longer present `other_outputs/` as a root-level current-release
deliverable. Paper-facing aggregate tables should point to
`final_artifacts_in_paper_updated/01_main_results/` and the package copy.

## Manifest Requirements

Each archive should include a lightweight manifest:

- `archived/release_20260624_reproducibility_inputs/README.md`
- `archived/release_20260624_reproducibility_inputs/MANIFEST.csv`
- `archived/release_20260624_legacy_workspace/README.md`
- `archived/release_20260624_legacy_workspace/MANIFEST.csv`
- `archived/local_workspace_residue_20260624/README.md`
- `archived/local_workspace_residue_20260624/MANIFEST.csv`

The manifests should record:

- old path
- archive path
- category
- whether the item is verifier/package dependency or local residue
- for moved scripts, the dependency-scan reason for archiving
- tracked status before the move
- size in bytes where cheap to compute
- SHA-256 for small files and manifest-relevant files

For very large directories, directory-level size and file count are sufficient;
do not hash every file in multi-GB trees unless the implementation can do it
quickly.

## Acceptance Criteria

1. Root no longer contains the large reproducibility-input directories listed in
   this spec.
2. Root no longer contains `other_outputs/` or local residue directories listed
   in this spec.
3. `final_artifacts_in_paper_updated/` and `paper_reproducibility_package/`
   remain in root.
4. `scripts/verify_current_results_reproducibility.py` passes from the archived
   reproducibility-input layout.
5. `scripts/build_paper_reproducibility_package.py` rebuilds the package from
   the archived reproducibility-input layout.
6. `scripts/validate_paper_reproducibility_package.py` passes.
7. Release cleanup contract tests are extended or new contract tests are added
   so future root-clutter regressions are caught.
8. A script dependency scan is recorded before archiving tracked helper scripts,
   and release-facing scripts remain callable.
9. `git diff --check` passes.
10. `git status --short` is clean except for intentionally ignored local archive
   payloads that are too large to track.
11. A local annotated tag
    `v1.0.1-paper-reproducibility-clean-root-20260624` is created only after all
    verification passes.

## Non-Goals

- Do not rewrite or move the existing `v1.0-paper-reproducibility-20260624` tag.
- Do not force-add multi-GB ignored result trees to git.
- Do not remove source code needed for the paper quickstart.
- Do not delete local artifact trees without first archiving them.
- Do not move `final_artifacts_in_paper_updated/` or
  `paper_reproducibility_package/` out of the root.

## Risks and Mitigations

- Risk: moving verifier dependencies breaks reproducibility checks.
  Mitigation: update path resolution first, then run verifier before and after
  moves where practical.
- Risk: package builder resets an ignored package tree and produces many
  apparent deletions.
  Mitigation: always force-add regenerated package files and run package
  validator before committing.
- Risk: archives become too large for git.
  Mitigation: track lightweight manifests/docs only for large ignored payloads.
- Risk: old docs keep pointing at root artifact paths.
  Mitigation: run targeted path scans over release docs and package generated
  docs before tagging.
