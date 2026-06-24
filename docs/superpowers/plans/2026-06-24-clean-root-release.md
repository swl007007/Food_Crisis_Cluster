# Clean-Root Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare `v1.0.1-paper-reproducibility-clean-root-20260624` by archiving heavy reproducibility inputs and local workspace residue while preserving the root-level paper package, final artifacts, verifier, and package builder.

**Architecture:** Add a centralized release path module first, then update the verifier, package builder, and paper artifact audit to read archived reproducibility inputs. Use a manifest-aware archive utility for deterministic moves, update docs/package metadata to describe the new layout, and verify before tagging.

**Tech Stack:** Python 3.12, `pathlib`, `csv`, `json`, `hashlib`, `shutil`, `pytest`, `unittest`, Git.

---

## File Structure

- Create: `scripts/release_paths.py`
  - Single source of truth for root paths, archive roots, Stage 2 experiment paths, Stage 3 result paths, thresholded GeoRF paths, ablation paths, FEWS NET baseline paths, and legacy manifest path resolution.
- Create: `src/tests/test_release_paths.py`
  - Pure unit tests for archive path resolution and fallback behavior.
- Modify: `scripts/verify_current_results_reproducibility.py`
  - Replace root-hardcoded live result paths with `ReleasePaths`.
- Modify: `scripts/paper_artifacts/audit_final_artifact_sources.py`
  - Audit provider manifests from archived result folders while continuing to write audit reports under `final_artifacts_in_paper_updated/`.
- Modify: `src/tests/test_artifact_source_audit.py`
  - Update expected provider/default paths after path resolution changes.
- Modify: `scripts/build_paper_reproducibility_package.py`
  - Build the same package layout from archived input roots.
- Modify: `src/tests/test_paper_reproducibility_package.py`
  - Add path-layer/package-builder contract checks.
- Create: `scripts/release_tools/archive_clean_root_release_inputs.py`
  - Manifest-aware move utility for reproducibility inputs, legacy workspace files, and local residue.
- Create: `scripts/release_tools/__init__.py`
  - Package marker for release utility imports in tests.
- Modify: `.gitignore`
  - Allow archive README/MANIFEST files and tracked lightweight legacy scripts, while keeping large moved payloads ignored.
- Modify: `src/tests/test_release_cleanup_contract.py`
  - Add clean-root v1.0.1 archive expectations and root-clutter guards.
- Modify: `README.md`, `PIPELINE_WORKFLOW.md`, `CURRENT_RESULTS_REPRODUCTION.md`, `RELEASE_MANIFEST.md`
  - Document root quick validation, archived reproducibility inputs, archived legacy workspace, archived local residue, and new tag.
- Regenerate: `paper_reproducibility_package/`
  - Rebuilt by `scripts/build_paper_reproducibility_package.py` after path/doc updates.
- Create after archive execution:
  - `archived/release_20260624_reproducibility_inputs/README.md`
  - `archived/release_20260624_reproducibility_inputs/MANIFEST.csv`
  - `archived/release_20260624_legacy_workspace/README.md`
  - `archived/release_20260624_legacy_workspace/MANIFEST.csv`
  - `archived/local_workspace_residue_20260624/README.md`
  - `archived/local_workspace_residue_20260624/MANIFEST.csv`

## Task 1: Add Centralized Release Paths

**Files:**
- Create: `scripts/release_paths.py`
- Create: `src/tests/test_release_paths.py`

- [ ] **Step 1: Write failing path tests**

Add `src/tests/test_release_paths.py`:

```python
from pathlib import Path

from scripts.release_paths import ReleasePaths


def test_default_archive_roots_are_under_archived() -> None:
    paths = ReleasePaths(repo_root=Path("/repo"))

    assert paths.reproducibility_inputs_root == Path(
        "/repo/archived/release_20260624_reproducibility_inputs"
    )
    assert paths.legacy_workspace_root == Path(
        "/repo/archived/release_20260624_legacy_workspace"
    )
    assert paths.local_residue_root == Path(
        "/repo/archived/local_workspace_residue_20260624"
    )
    assert paths.final_artifacts_root == Path("/repo/final_artifacts_in_paper_updated")
    assert paths.package_root == Path("/repo/paper_reproducibility_package")


def test_model_and_result_paths_resolve_to_reproducibility_archive() -> None:
    paths = ReleasePaths(repo_root=Path("/repo"))

    assert paths.experiment_root("GF") == Path(
        "/repo/archived/release_20260624_reproducibility_inputs/GeoRFExperiment"
    )
    assert paths.experiment_root("DT") == Path(
        "/repo/archived/release_20260624_reproducibility_inputs/GeoDTExperiment"
    )
    assert paths.stage3_root("GF", 2) == Path(
        "/repo/archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_fs2"
    )
    assert paths.stage3_root("DT", 3) == Path(
        "/repo/archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_DT_fs3"
    )
    assert paths.thresholded_georf_root(1) == Path(
        "/repo/archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_thresholded_fs1"
    )
    assert paths.ablation_root == Path(
        "/repo/archived/release_20260624_reproducibility_inputs/main_ablation_exclude_updated_stage3_fixed_partitions"
    )
    assert paths.fewsnet_baseline_root == Path(
        "/repo/archived/release_20260624_reproducibility_inputs/fewsnet_baseline_results"
    )


def test_resolve_repo_reference_prefers_existing_root_file(tmp_path: Path) -> None:
    root_file = tmp_path / "final_artifacts_in_paper_updated" / "artifact_source_audit.md"
    root_file.parent.mkdir(parents=True)
    root_file.write_text("audit\n", encoding="utf-8")

    paths = ReleasePaths(repo_root=tmp_path)

    assert paths.resolve_repo_reference("final_artifacts_in_paper_updated/artifact_source_audit.md") == root_file


def test_resolve_repo_reference_falls_back_to_archived_input(tmp_path: Path) -> None:
    archived_file = (
        tmp_path
        / "archived"
        / "release_20260624_reproducibility_inputs"
        / "result_partition_k40_compare_GF_fs1"
        / "run_manifest.json"
    )
    archived_file.parent.mkdir(parents=True)
    archived_file.write_text("{}", encoding="utf-8")

    paths = ReleasePaths(repo_root=tmp_path)

    assert paths.resolve_repo_reference("result_partition_k40_compare_GF_fs1/run_manifest.json") == archived_file


def test_resolve_repo_reference_handles_windows_drive_paths() -> None:
    paths = ReleasePaths(repo_root=Path("/repo"))

    resolved = paths.resolve_repo_reference(
        r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_forecast_unadjusted_bm.csv"
    )

    assert resolved == Path(
        "/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/1.Source Data/FEWSNET_forecast_unadjusted_bm.csv"
    )


def test_resolve_repo_reference_returns_root_candidate_for_unknown_relative_path() -> None:
    paths = ReleasePaths(repo_root=Path("/repo"))

    assert paths.resolve_repo_reference("docs/example.md") == Path("/repo/docs/example.md")
```

- [ ] **Step 2: Run path tests and confirm failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest src/tests/test_release_paths.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'scripts.release_paths'`.

- [ ] **Step 3: Add `scripts/release_paths.py`**

Create `scripts/release_paths.py`:

```python
"""Central path contract for the paper reproducibility release layout."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPRODUCIBILITY_INPUT_NAMES = frozenset(
    {
        "GeoRFExperiment",
        "GeoDTExperiment",
        "main_ablation_exclude_updated_stage3_fixed_partitions",
        "fewsnet_baseline_results",
        "result_partition_k40_compare_GF_fs1",
        "result_partition_k40_compare_GF_fs2",
        "result_partition_k40_compare_GF_fs3",
        "result_partition_k40_compare_DT_fs1",
        "result_partition_k40_compare_DT_fs2",
        "result_partition_k40_compare_DT_fs3",
        "result_partition_k40_compare_GF_thresholded_fs1",
        "result_partition_k40_compare_GF_thresholded_fs2",
        "result_partition_k40_compare_GF_thresholded_fs3",
    }
)


@dataclass(frozen=True)
class ReleasePaths:
    """Resolve release artifact paths after the clean-root archive move."""

    repo_root: Path

    @classmethod
    def default(cls) -> "ReleasePaths":
        return cls(repo_root=Path(__file__).resolve().parents[1])

    @property
    def reproducibility_inputs_root(self) -> Path:
        return self.repo_root / "archived" / "release_20260624_reproducibility_inputs"

    @property
    def legacy_workspace_root(self) -> Path:
        return self.repo_root / "archived" / "release_20260624_legacy_workspace"

    @property
    def local_residue_root(self) -> Path:
        return self.repo_root / "archived" / "local_workspace_residue_20260624"

    @property
    def final_artifacts_root(self) -> Path:
        return self.repo_root / "final_artifacts_in_paper_updated"

    @property
    def package_root(self) -> Path:
        return self.repo_root / "paper_reproducibility_package"

    @property
    def ablation_root(self) -> Path:
        return self.reproducibility_inputs_root / "main_ablation_exclude_updated_stage3_fixed_partitions"

    @property
    def fewsnet_baseline_root(self) -> Path:
        return self.reproducibility_inputs_root / "fewsnet_baseline_results"

    def experiment_root(self, token: str) -> Path:
        normalized = token.upper()
        if normalized == "GF":
            return self.reproducibility_inputs_root / "GeoRFExperiment"
        if normalized == "DT":
            return self.reproducibility_inputs_root / "GeoDTExperiment"
        raise ValueError(f"Unknown experiment token: {token}")

    def stage3_root(self, token: str, scope: int) -> Path:
        normalized = token.upper()
        if normalized not in {"GF", "DT"}:
            raise ValueError(f"Unknown Stage 3 token: {token}")
        if scope not in {1, 2, 3}:
            raise ValueError(f"Stage 3 scope must be 1, 2, or 3, got {scope}")
        return self.reproducibility_inputs_root / f"result_partition_k40_compare_{normalized}_fs{scope}"

    def thresholded_georf_root(self, scope: int) -> Path:
        if scope not in {1, 2, 3}:
            raise ValueError(f"Thresholded GeoRF scope must be 1, 2, or 3, got {scope}")
        return self.reproducibility_inputs_root / f"result_partition_k40_compare_GF_thresholded_fs{scope}"

    def resolve_repo_reference(self, value: str | Path) -> Path:
        """Resolve manifest paths from old root layout or new archive layout."""
        raw = str(value).replace("\\", "/")
        if len(raw) >= 3 and raw[1:3] == ":/":
            if os.name == "nt":
                return Path(raw)
            return Path("/mnt") / raw[0].lower() / raw[3:]

        path = Path(raw)
        if path.is_absolute():
            return path

        root_candidate = self.repo_root / path
        if root_candidate.exists():
            return root_candidate

        if path.parts and path.parts[0] in REPRODUCIBILITY_INPUT_NAMES:
            archived_candidate = self.reproducibility_inputs_root / path
            if archived_candidate.exists():
                return archived_candidate
            return archived_candidate

        return root_candidate

    def require_file(self, path: Path, label: str) -> Path:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
        return path

    def require_dir(self, path: Path, label: str) -> Path:
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {label}: {path}")
        return path
```

- [ ] **Step 4: Run path tests and confirm pass**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest src/tests/test_release_paths.py -q
```

Expected: `6 passed`.

- [ ] **Step 5: Commit path layer**

Run:

```bash
git add scripts/release_paths.py src/tests/test_release_paths.py
git commit -m "add clean release path contract"
```

## Task 2: Point Verifier And Artifact Audit At Archived Inputs

**Files:**
- Modify: `scripts/verify_current_results_reproducibility.py`
- Modify: `scripts/paper_artifacts/audit_final_artifact_sources.py`
- Modify: `src/tests/test_artifact_source_audit.py`

- [ ] **Step 1: Add failing audit/path expectations**

In `src/tests/test_artifact_source_audit.py`, extend the import and add a provider path test:

```python
from scripts.paper_artifacts.audit_final_artifact_sources import (
    CLEAN_PANEL_BASENAME,
    PHASE_CHANGE_BASENAME,
    PROVIDER_MANIFESTS,
    audit_artifact_manifest,
    audit_provider_manifest,
    audit_script_default,
    classify_source_text,
    write_audit_outputs,
)
```

Add this test near the provider-manifest tests:

```python
def test_provider_manifests_resolve_from_reproducibility_archive():
    provider_paths = {label: path for label, path in PROVIDER_MANIFESTS}

    assert provider_paths["result_partition_k40_compare_GF_fs1"].as_posix().endswith(
        "archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_fs1/run_manifest.json"
    )
    assert provider_paths["result_partition_k40_compare_DT_fs3"].as_posix().endswith(
        "archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_DT_fs3/run_manifest.json"
    )
    assert provider_paths["result_partition_k40_compare_GF_thresholded_fs2"].as_posix().endswith(
        "archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_thresholded_fs2/run_manifest.json"
    )
```

- [ ] **Step 2: Run artifact audit test and confirm failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest src/tests/test_artifact_source_audit.py::test_provider_manifests_resolve_from_reproducibility_archive -q
```

Expected: fail because provider manifests still point to root result folders.

- [ ] **Step 3: Update `audit_final_artifact_sources.py` provider paths**

Add this import after `Path` imports:

```python
from scripts.release_paths import ReleasePaths
```

Replace root constants with:

```python
REPO_ROOT = Path(__file__).resolve().parents[2]
PATHS = ReleasePaths(repo_root=REPO_ROOT)
FINAL_ARTIFACT_ROOT = PATHS.final_artifacts_root
```

Replace `PROVIDER_MANIFESTS` with:

```python
PROVIDER_MANIFESTS = [
    ("result_partition_k40_compare_GF_fs1", PATHS.stage3_root("GF", 1) / "run_manifest.json"),
    ("result_partition_k40_compare_GF_fs2", PATHS.stage3_root("GF", 2) / "run_manifest.json"),
    ("result_partition_k40_compare_GF_fs3", PATHS.stage3_root("GF", 3) / "run_manifest.json"),
    ("result_partition_k40_compare_DT_fs1", PATHS.stage3_root("DT", 1) / "run_manifest.json"),
    ("result_partition_k40_compare_DT_fs2", PATHS.stage3_root("DT", 2) / "run_manifest.json"),
    ("result_partition_k40_compare_DT_fs3", PATHS.stage3_root("DT", 3) / "run_manifest.json"),
    ("result_partition_k40_compare_GF_thresholded_fs1", PATHS.thresholded_georf_root(1) / "run_manifest.json"),
    ("result_partition_k40_compare_GF_thresholded_fs2", PATHS.thresholded_georf_root(2) / "run_manifest.json"),
    ("result_partition_k40_compare_GF_thresholded_fs3", PATHS.thresholded_georf_root(3) / "run_manifest.json"),
]
```

Keep `SCRIPT_DEFAULTS` pointing to current script locations, including `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py`, because Stage 3 still calls that script.

- [ ] **Step 4: Update verifier path wiring**

In `scripts/verify_current_results_reproducibility.py`, add:

```python
from scripts.release_paths import ReleasePaths
```

After `REPO_ROOT` setup, add:

```python
PATHS = ReleasePaths(repo_root=REPO_ROOT)
```

Change `MAIN_RESULTS` experiment entries:

```python
"experiment": PATHS.experiment_root("GF"),
```

and:

```python
"experiment": PATHS.experiment_root("DT"),
```

In `repo_path_from_manifest`, replace the body with:

```python
def repo_path_from_manifest(value: str | Path) -> Path:
    """Resolve Windows, repo-relative, or archived manifest paths."""
    return PATHS.resolve_repo_reference(value)
```

In `verify_main_stage3`, replace:

```python
out_dir = REPO_ROOT / f"result_partition_k40_compare_{token}_fs{scope}"
```

with:

```python
out_dir = PATHS.stage3_root(token, scope)
```

In `verify_ablation_outputs`, replace:

```python
root = REPO_ROOT / "main_ablation_exclude_updated_stage3_fixed_partitions"
```

with:

```python
root = PATHS.ablation_root
```

In `verify_final_artifacts`, replace:

```python
current_final = REPO_ROOT / "final_artifacts_in_paper_updated"
```

with:

```python
current_final = PATHS.final_artifacts_root
```

Also replace monthly manifest source checks:

```python
verifier.check((REPO_ROOT / source).is_file(), f"monthly performance source exists: {source}")
```

with:

```python
verifier.check(PATHS.resolve_repo_reference(source).is_file(), f"monthly performance source exists: {source}")
```

In `verify_artifact_source_audit`, replace:

```python
audit_csv, audit_md = write_audit_outputs(rows, REPO_ROOT / "final_artifacts_in_paper_updated")
```

with:

```python
audit_csv, audit_md = write_audit_outputs(rows, PATHS.final_artifacts_root)
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest src/tests/test_release_paths.py src/tests/test_artifact_source_audit.py -q
```

Expected: all tests pass except live-file checks are not run in these unit tests.

- [ ] **Step 6: Commit verifier/audit path update**

Run:

```bash
git add scripts/verify_current_results_reproducibility.py scripts/paper_artifacts/audit_final_artifact_sources.py src/tests/test_artifact_source_audit.py
git commit -m "point release verifier at archive paths"
```

## Task 3: Update Package Builder To Read Archived Inputs

**Files:**
- Modify: `scripts/build_paper_reproducibility_package.py`
- Modify: `src/tests/test_paper_reproducibility_package.py`

- [ ] **Step 1: Add failing package path contract test**

In `src/tests/test_paper_reproducibility_package.py`, extend the import:

```python
from scripts.build_paper_reproducibility_package import (
    ManifestRow,
    PackageBuildError,
    ensure_package_relative,
    sha256_file,
    stage2_specs,
    stage3_result_dirs,
    write_manifest,
    write_sha256sums,
)
```

Add:

```python
    def test_package_builder_sources_archived_release_inputs(self) -> None:
        stage2_sources = [source.as_posix() for source, _, _ in stage2_specs()]
        stage3_sources = [source.as_posix() for _, source in stage3_result_dirs()]

        self.assertTrue(
            any(
                "archived/release_20260624_reproducibility_inputs/GeoRFExperiment" in source
                for source in stage2_sources
            )
        )
        self.assertTrue(
            any(
                "archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_fs1" in source
                for source in stage3_sources
            )
        )
```

- [ ] **Step 2: Run package test and confirm failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest src.tests.test_paper_reproducibility_package.PaperReproducibilityPackageTests.test_package_builder_sources_archived_release_inputs -v
```

Expected: fail because `stage2_specs()` and `stage3_result_dirs()` still point to root folders.

- [ ] **Step 3: Wire package builder to `ReleasePaths`**

In `scripts/build_paper_reproducibility_package.py`, add:

```python
from scripts.release_paths import ReleasePaths
```

After `REPO_ROOT`, add:

```python
PATHS = ReleasePaths(repo_root=REPO_ROOT)
```

Change:

```python
PACKAGE_ROOT = REPO_ROOT / "paper_reproducibility_package"
```

to:

```python
PACKAGE_ROOT = PATHS.package_root
```

Replace every `REPO_ROOT / "GeoRFExperiment"` source in `stage2_specs()` with `PATHS.experiment_root("GF")`.

Replace every `REPO_ROOT / "GeoDTExperiment"` source in `stage2_specs()` with `PATHS.experiment_root("DT")`.

Replace `stage3_result_dirs()` with:

```python
def stage3_result_dirs() -> list[tuple[str, Path]]:
    return [
        ("georf_fs1", PATHS.stage3_root("GF", 1)),
        ("georf_fs2", PATHS.stage3_root("GF", 2)),
        ("georf_fs3", PATHS.stage3_root("GF", 3)),
        ("geodt_fs1", PATHS.stage3_root("DT", 1)),
        ("geodt_fs2", PATHS.stage3_root("DT", 2)),
        ("geodt_fs3", PATHS.stage3_root("DT", 3)),
    ]
```

In `canonical_refined_files`, replace:

```python
source = normalized if normalized.is_absolute() else REPO_ROOT / normalized
```

with:

```python
source = normalized if normalized.is_absolute() else PATHS.resolve_repo_reference(normalized)
```

In `add_final_artifacts`, replace:

```python
source_root = REPO_ROOT / "final_artifacts_in_paper_updated"
```

with:

```python
source_root = PATHS.final_artifacts_root
```

In `add_ablation_provenance`, replace `REPO_ROOT / "main_ablation_exclude_updated_stage3_fixed_partitions"` with `PATHS.ablation_root`.

Update generated package text:

```python
Experimental and legacy entry points are archived under
`archived/release_20260624_nonpaper_pipelines/` and historical release
workspaces are archived under `archived/release_20260624_legacy_workspace/` in
the source repository.
```

and:

```python
Their entry points are preserved as historical provenance under
`archived/release_20260624_nonpaper_pipelines/`; local result workspaces used to
rebuild this package are under
`archived/release_20260624_reproducibility_inputs/`.
```

- [ ] **Step 4: Run package unit tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest src.tests.test_paper_reproducibility_package -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit package path update**

Run:

```bash
git add scripts/build_paper_reproducibility_package.py src/tests/test_paper_reproducibility_package.py
git commit -m "build package from archived release inputs"
```

## Task 4: Add Archive Utility, Gitignore Exceptions, And Clean-Root Contracts

**Files:**
- Create: `scripts/release_tools/__init__.py`
- Create: `scripts/release_tools/archive_clean_root_release_inputs.py`
- Modify: `.gitignore`
- Modify: `src/tests/test_release_cleanup_contract.py`

- [ ] **Step 1: Add failing archive utility contract tests**

In `src/tests/test_release_cleanup_contract.py`, add tests that validate the
archive utility's item lists without requiring the root move to have happened:

```python
def archive_old_paths(items) -> set[str]:
    return {item.old_path for item in items}


def test_clean_root_archive_item_lists_are_explicit() -> None:
    from scripts.release_tools.archive_clean_root_release_inputs import (
        LEGACY_WORKSPACE_ITEMS,
        LOCAL_RESIDUE_ITEMS,
        REPRODUCIBILITY_ITEMS,
    )

    assert len(REPRODUCIBILITY_ITEMS) == 13
    assert len(LEGACY_WORKSPACE_ITEMS) == 11
    assert len(LOCAL_RESIDUE_ITEMS) == 6


def test_clean_root_repro_items_cover_verifier_inputs() -> None:
    from scripts.release_tools.archive_clean_root_release_inputs import REPRODUCIBILITY_ITEMS

    old_paths = archive_old_paths(REPRODUCIBILITY_ITEMS)
    expected = {
        "GeoRFExperiment",
        "GeoDTExperiment",
        "main_ablation_exclude_updated_stage3_fixed_partitions",
        "fewsnet_baseline_results",
        "result_partition_k40_compare_GF_fs1",
        "result_partition_k40_compare_GF_fs2",
        "result_partition_k40_compare_GF_fs3",
        "result_partition_k40_compare_DT_fs1",
        "result_partition_k40_compare_DT_fs2",
        "result_partition_k40_compare_DT_fs3",
        "result_partition_k40_compare_GF_thresholded_fs1",
        "result_partition_k40_compare_GF_thresholded_fs2",
        "result_partition_k40_compare_GF_thresholded_fs3",
    }
    assert old_paths == expected


def test_clean_root_legacy_items_include_workspace_and_script_candidates() -> None:
    from scripts.release_tools.archive_clean_root_release_inputs import LEGACY_WORKSPACE_ITEMS

    old_paths = archive_old_paths(LEGACY_WORKSPACE_ITEMS)
    assert "other_outputs" in old_paths
    assert "scripts/config_visual.py" in old_paths
    for item in LEGACY_WORKSPACE_ITEMS:
        assert item.dependency_scan_reason.strip(), item
```

These tests are expected to fail before the archive utility exists, then pass
after Step 3.

- [ ] **Step 2: Create release utility package marker**

Create `scripts/release_tools/__init__.py`:

```python
"""Release maintenance utilities."""
```

- [ ] **Step 3: Add archive utility**

Create `scripts/release_tools/archive_clean_root_release_inputs.py`:

```python
"""Move clean-root release inputs into archive folders and write manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from scripts.release_paths import ReleasePaths


REPO_ROOT = Path(__file__).resolve().parents[2]
PATHS = ReleasePaths(repo_root=REPO_ROOT)
MANIFEST_FIELDS = [
    "old_path",
    "archive_path",
    "category",
    "dependency",
    "dependency_scan_reason",
    "tracked_status_before_move",
    "size_bytes",
    "file_count",
    "sha256",
]


@dataclass(frozen=True)
class ArchiveItem:
    old_path: str
    archive_root: Path
    category: str
    dependency: str
    dependency_scan_reason: str

    @property
    def archive_path(self) -> Path:
        return self.archive_root / self.old_path


REPRODUCIBILITY_ITEMS = [
    ArchiveItem("GeoRFExperiment", PATHS.reproducibility_inputs_root, "stage1_stage2_georf", "verifier_package_dependency", "Stage 1/2 GeoRF handoff read by verifier and package builder"),
    ArchiveItem("GeoDTExperiment", PATHS.reproducibility_inputs_root, "stage1_stage2_geodt", "verifier_package_dependency", "Stage 1/2 GeoDT appendix handoff read by verifier and package builder"),
    ArchiveItem("main_ablation_exclude_updated_stage3_fixed_partitions", PATHS.reproducibility_inputs_root, "fixed_partition_ablation", "verifier_package_dependency", "Ablation manifests and outputs read by verifier and package builder"),
    ArchiveItem("fewsnet_baseline_results", PATHS.reproducibility_inputs_root, "fewsnet_baseline", "verifier_package_dependency", "Baseline source CSVs used by paper artifact audit context"),
    ArchiveItem("result_partition_k40_compare_GF_fs1", PATHS.reproducibility_inputs_root, "stage3_georf", "verifier_package_dependency", "Stage 3 GeoRF fs1 read by verifier and package builder"),
    ArchiveItem("result_partition_k40_compare_GF_fs2", PATHS.reproducibility_inputs_root, "stage3_georf", "verifier_package_dependency", "Stage 3 GeoRF fs2 read by verifier and package builder"),
    ArchiveItem("result_partition_k40_compare_GF_fs3", PATHS.reproducibility_inputs_root, "stage3_georf", "verifier_package_dependency", "Stage 3 GeoRF fs3 read by verifier and package builder"),
    ArchiveItem("result_partition_k40_compare_DT_fs1", PATHS.reproducibility_inputs_root, "stage3_geodt", "verifier_package_dependency", "Stage 3 GeoDT fs1 appendix provenance read by verifier and package builder"),
    ArchiveItem("result_partition_k40_compare_DT_fs2", PATHS.reproducibility_inputs_root, "stage3_geodt", "verifier_package_dependency", "Stage 3 GeoDT fs2 appendix provenance read by verifier and package builder"),
    ArchiveItem("result_partition_k40_compare_DT_fs3", PATHS.reproducibility_inputs_root, "stage3_geodt", "verifier_package_dependency", "Stage 3 GeoDT fs3 appendix provenance read by verifier and package builder"),
    ArchiveItem("result_partition_k40_compare_GF_thresholded_fs1", PATHS.reproducibility_inputs_root, "thresholded_georf", "verifier_dependency", "Thresholded GeoRF fs1 provider manifest audited by verifier"),
    ArchiveItem("result_partition_k40_compare_GF_thresholded_fs2", PATHS.reproducibility_inputs_root, "thresholded_georf", "verifier_dependency", "Thresholded GeoRF fs2 provider manifest audited by verifier"),
    ArchiveItem("result_partition_k40_compare_GF_thresholded_fs3", PATHS.reproducibility_inputs_root, "thresholded_georf", "verifier_dependency", "Thresholded GeoRF fs3 provider manifest audited by verifier"),
]

LEGACY_WORKSPACE_ITEMS = [
    ArchiveItem("other_outputs", PATHS.legacy_workspace_root, "legacy_workspace_output", "historical_workspace", "Current release docs now point to final_artifacts_in_paper_updated and package copies"),
    ArchiveItem("scripts/config_visual.py", PATHS.legacy_workspace_root, "legacy_script", "historical_workspace", "Root config_visual.py is the imported runtime config; scripts/config_visual.py has no release dependency"),
    ArchiveItem("scripts/check_country_columns.py", PATHS.legacy_workspace_root, "legacy_script", "historical_workspace", "No release verifier/package/paper_artifact/batch dependency found in scan"),
    ArchiveItem("scripts/check_shapefile.py", PATHS.legacy_workspace_root, "legacy_script", "historical_workspace", "No release verifier/package/paper_artifact/batch dependency found in scan"),
    ArchiveItem("scripts/diagnose_partition_lineage.py", PATHS.legacy_workspace_root, "legacy_script", "historical_workspace", "Diagnostic helper not called by release verifier/package/paper artifacts or batch workflow"),
    ArchiveItem("scripts/enable_visual_debug.py", PATHS.legacy_workspace_root, "legacy_script", "historical_workspace", "Interactive debug toggle not called by release verifier/package/paper artifacts or batch workflow"),
    ArchiveItem("scripts/plot_cluster_map.py", PATHS.legacy_workspace_root, "legacy_script", "historical_workspace", "Referenced as historical plotting approach only; paper artifacts use scripts/paper_artifacts"),
    ArchiveItem("scripts/plot_cluster_map_3x4.py", PATHS.legacy_workspace_root, "legacy_script", "historical_workspace", "Legacy plotting helper superseded by scripts/paper_artifacts"),
    ArchiveItem("scripts/plot_cluster_map_3x4_refined.py", PATHS.legacy_workspace_root, "legacy_script", "historical_workspace", "Legacy plotting helper superseded by scripts/paper_artifacts"),
    ArchiveItem("scripts/render_scope_images.py", PATHS.legacy_workspace_root, "legacy_script", "historical_workspace", "No release verifier/package/paper_artifact/batch dependency found in scan"),
    ArchiveItem("scripts/structure_change.py", PATHS.legacy_workspace_root, "legacy_script", "historical_workspace", "No release verifier/package/paper_artifact/batch dependency found in scan"),
]

LOCAL_RESIDUE_ITEMS = [
    ArchiveItem("writing", PATHS.local_residue_root, "local_residue", "not_release_dependency", "Manuscript drafting residue outside verifier/package path"),
    ArchiveItem("dt_rules", PATHS.local_residue_root, "local_residue", "not_release_dependency", "Generated rule exports not used as branch-specific paper explanations"),
    ArchiveItem("monthly_results", PATHS.local_residue_root, "local_residue", "not_release_dependency", "Empty or stale monthly output folder outside verifier/package path"),
    ArchiveItem("demo", PATHS.local_residue_root, "local_residue", "not_release_dependency", "Ignored demo residue outside release quickstart"),
    ArchiveItem("prediction_pipeline", PATHS.local_residue_root, "local_residue", "not_release_dependency", "Forward-prediction residue already represented by archived non-paper pipeline provenance"),
    ArchiveItem("regional_ablation_results", PATHS.local_residue_root, "local_residue", "not_release_dependency", "Regional exploratory output outside paper verifier/package path"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_status(relative_path: str) -> str:
    result = subprocess.run(
        ["git", "ls-files", relative_path],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    tracked = [line for line in result.stdout.splitlines() if line.strip()]
    if tracked:
        return f"tracked_files={len(tracked)}"
    return "untracked_or_ignored"


def size_and_count(path: Path) -> tuple[int, int]:
    if path.is_file():
        return path.stat().st_size, 1
    total = 0
    count = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
            count += 1
    return total, count


def small_file_hash(path: Path, size_bytes: int) -> str:
    if path.is_file() and size_bytes <= 50 * 1024 * 1024:
        return sha256_file(path)
    return ""


def move_item(item: ArchiveItem, *, execute: bool) -> dict[str, str]:
    source = REPO_ROOT / item.old_path
    destination = item.archive_path
    status = tracked_status(item.old_path)
    if not source.exists():
        size_bytes = 0
        file_count = 0
        checksum = ""
    else:
        size_bytes, file_count = size_and_count(source)
        checksum = small_file_hash(source, size_bytes)

    if execute and source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Archive destination already exists: {destination}")
        shutil.move(str(source), str(destination))

    return {
        "old_path": item.old_path,
        "archive_path": destination.relative_to(REPO_ROOT).as_posix(),
        "category": item.category,
        "dependency": item.dependency,
        "dependency_scan_reason": item.dependency_scan_reason,
        "tracked_status_before_move": status,
        "size_bytes": str(size_bytes),
        "file_count": str(file_count),
        "sha256": checksum,
    }


def write_archive_docs(archive_root: Path, rows: list[dict[str, str]], title: str, description: str) -> None:
    archive_root.mkdir(parents=True, exist_ok=True)
    with (archive_root / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# {title}",
        "",
        description,
        "",
        "This archive was created for `v1.0.1-paper-reproducibility-clean-root-20260624`.",
        "Large ignored payloads are preserved locally but are not force-added to git.",
        "",
        "See `MANIFEST.csv` for old paths, archive paths, dependency category, tracked status, size, file count, and small-file checksums.",
        "",
    ]
    (archive_root / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def archive_group(items: list[ArchiveItem], title: str, description: str, *, execute: bool) -> None:
    rows = [move_item(item, execute=execute) for item in items]
    write_archive_docs(items[0].archive_root, rows, title, description)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Move files. Without this flag, only manifests are written.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive_group(
        REPRODUCIBILITY_ITEMS,
        "Release 20260624 Reproducibility Inputs",
        "Archived Stage 1/2, Stage 3, ablation, thresholded, and FEWS NET baseline inputs used by local verifier/package workflows.",
        execute=args.execute,
    )
    archive_group(
        LEGACY_WORKSPACE_ITEMS,
        "Release 20260624 Legacy Workspace",
        "Archived helper scripts and script-owned outputs outside the quick release validation path.",
        execute=args.execute,
    )
    archive_group(
        LOCAL_RESIDUE_ITEMS,
        "Local Workspace Residue 20260624",
        "Archived local residue that is not a verifier or package dependency.",
        execute=args.execute,
    )
    mode = "executed" if args.execute else "dry-run manifests written"
    print(f"Clean-root archive {mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Update `.gitignore` archive exceptions**

Append below the existing `archived/*` block:

```gitignore
!archived/release_20260624_reproducibility_inputs/
!archived/release_20260624_reproducibility_inputs/README.md
!archived/release_20260624_reproducibility_inputs/MANIFEST.csv
!archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_thresholded_fs1/
!archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_thresholded_fs2/
!archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_thresholded_fs3/
!archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_thresholded_fs1/run_manifest.json
!archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_thresholded_fs2/run_manifest.json
!archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_thresholded_fs3/run_manifest.json
!archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_thresholded_fs1/threshold_provenance.csv
!archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_thresholded_fs2/threshold_provenance.csv
!archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_thresholded_fs3/threshold_provenance.csv
!archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_thresholded_fs1/metrics_monthly.csv
!archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_thresholded_fs2/metrics_monthly.csv
!archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_thresholded_fs3/metrics_monthly.csv
!archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_thresholded_fs1/metrics_polygon_overall.csv
!archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_thresholded_fs2/metrics_polygon_overall.csv
!archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_thresholded_fs3/metrics_polygon_overall.csv
!archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_thresholded_fs1/predictions_monthly.csv
!archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_thresholded_fs2/predictions_monthly.csv
!archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_thresholded_fs3/predictions_monthly.csv

!archived/release_20260624_legacy_workspace/
!archived/release_20260624_legacy_workspace/**/
!archived/release_20260624_legacy_workspace/**/*.py
!archived/release_20260624_legacy_workspace/**/*.txt
!archived/release_20260624_legacy_workspace/README.md
!archived/release_20260624_legacy_workspace/MANIFEST.csv

!archived/local_workspace_residue_20260624/
!archived/local_workspace_residue_20260624/README.md
!archived/local_workspace_residue_20260624/MANIFEST.csv
```

- [ ] **Step 5: Run focused contract tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest src/tests/test_release_cleanup_contract.py -q
```

Expected: existing tests and new archive utility list tests pass.

- [ ] **Step 6: Commit archive utility and contracts**

Run:

```bash
git add .gitignore scripts/release_tools/__init__.py scripts/release_tools/archive_clean_root_release_inputs.py src/tests/test_release_cleanup_contract.py
git commit -m "add clean root archive utility"
```

## Task 5: Execute Archive Move

**Files:**
- Move: root reproducibility input folders into `archived/release_20260624_reproducibility_inputs/`
- Move: legacy workspace outputs/scripts into `archived/release_20260624_legacy_workspace/`
- Move: local residue folders into `archived/local_workspace_residue_20260624/`
- Create/update: archive README/MANIFEST files

- [ ] **Step 1: Confirm clean starting state**

Run:

```bash
git status --short
```

Expected: clean before executing moves.

- [ ] **Step 2: Dry-run manifest generation**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/release_tools/archive_clean_root_release_inputs.py
```

Expected: prints `Clean-root archive dry-run manifests written`; archive README/MANIFEST files exist, but root input folders still exist.

- [ ] **Step 3: Inspect dry-run manifest categories**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
import csv
from pathlib import Path

for manifest in [
    Path("archived/release_20260624_reproducibility_inputs/MANIFEST.csv"),
    Path("archived/release_20260624_legacy_workspace/MANIFEST.csv"),
    Path("archived/local_workspace_residue_20260624/MANIFEST.csv"),
]:
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    print(manifest, len(rows))
    for row in rows[:5]:
        print(" ", row["old_path"], row["tracked_status_before_move"], row["size_bytes"], row["file_count"])
PY
```

Expected: reproducibility manifest has 13 rows, legacy manifest has 11 rows, residue manifest has 6 rows.

- [ ] **Step 4: Execute the move**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/release_tools/archive_clean_root_release_inputs.py --execute
```

Expected: prints `Clean-root archive executed`.

- [ ] **Step 5: Inspect root and archive state**

Run:

```bash
find . -maxdepth 1 -type d -printf '%f\n' | sort
```

Expected: root no longer lists `GeoRFExperiment`, `GeoDTExperiment`, `main_ablation_exclude_updated_stage3_fixed_partitions`, `fewsnet_baseline_results`, `result_partition_k40_compare_*`, `other_outputs`, `writing`, `dt_rules`, `monthly_results`, `demo`, `prediction_pipeline`, or `regional_ablation_results`.

Run:

```bash
find archived/release_20260624_reproducibility_inputs -maxdepth 1 -mindepth 1 -printf '%f\n' | sort
find archived/release_20260624_legacy_workspace -maxdepth 2 -mindepth 1 -printf '%P\n' | sort | head -80
find archived/local_workspace_residue_20260624 -maxdepth 1 -mindepth 1 -printf '%f\n' | sort
```

Expected: archive roots contain the moved names and README/MANIFEST files.

- [ ] **Step 6: Add clean-root archive manifest and root-absence tests**

In `src/tests/test_release_cleanup_contract.py`, add constants:

```python
CLEAN_ROOT_REPRO_ARCHIVE = REPO_ROOT / "archived" / "release_20260624_reproducibility_inputs"
CLEAN_ROOT_LEGACY_ARCHIVE = REPO_ROOT / "archived" / "release_20260624_legacy_workspace"
CLEAN_ROOT_RESIDUE_ARCHIVE = REPO_ROOT / "archived" / "local_workspace_residue_20260624"

CLEAN_ROOT_FORBIDDEN_DIRS = [
    "GeoRFExperiment",
    "GeoDTExperiment",
    "main_ablation_exclude_updated_stage3_fixed_partitions",
    "fewsnet_baseline_results",
    "result_partition_k40_compare_GF_fs1",
    "result_partition_k40_compare_GF_fs2",
    "result_partition_k40_compare_GF_fs3",
    "result_partition_k40_compare_DT_fs1",
    "result_partition_k40_compare_DT_fs2",
    "result_partition_k40_compare_DT_fs3",
    "result_partition_k40_compare_GF_thresholded_fs1",
    "result_partition_k40_compare_GF_thresholded_fs2",
    "result_partition_k40_compare_GF_thresholded_fs3",
    "other_outputs",
    "writing",
    "dt_rules",
    "monthly_results",
    "regional_ablation_results",
]
```

Add:

```python
def load_manifest(path: Path) -> list[dict[str, str]]:
    assert path.is_file(), f"Missing manifest: {path}"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_clean_root_archives_have_readmes_and_manifests() -> None:
    for archive in [CLEAN_ROOT_REPRO_ARCHIVE, CLEAN_ROOT_LEGACY_ARCHIVE, CLEAN_ROOT_RESIDUE_ARCHIVE]:
        assert (archive / "README.md").is_file(), archive
        assert (archive / "MANIFEST.csv").is_file(), archive


def test_clean_root_forbidden_dirs_are_not_in_root() -> None:
    for relative in CLEAN_ROOT_FORBIDDEN_DIRS:
        assert not (REPO_ROOT / relative).exists(), relative


def test_clean_root_repro_manifest_contains_expected_inputs() -> None:
    rows = load_manifest(CLEAN_ROOT_REPRO_ARCHIVE / "MANIFEST.csv")
    old_paths = {row["old_path"] for row in rows}
    expected = {
        "GeoRFExperiment",
        "GeoDTExperiment",
        "main_ablation_exclude_updated_stage3_fixed_partitions",
        "fewsnet_baseline_results",
        "result_partition_k40_compare_GF_fs1",
        "result_partition_k40_compare_GF_fs2",
        "result_partition_k40_compare_GF_fs3",
        "result_partition_k40_compare_DT_fs1",
        "result_partition_k40_compare_DT_fs2",
        "result_partition_k40_compare_DT_fs3",
        "result_partition_k40_compare_GF_thresholded_fs1",
        "result_partition_k40_compare_GF_thresholded_fs2",
        "result_partition_k40_compare_GF_thresholded_fs3",
    }
    assert expected.issubset(old_paths)
    for row in rows:
        assert row["archive_path"].startswith("archived/release_20260624_reproducibility_inputs/")
        assert row["category"]
        assert row["dependency"]
        assert row["tracked_status_before_move"]


def test_clean_root_legacy_manifest_records_dependency_scan_reason() -> None:
    rows = load_manifest(CLEAN_ROOT_LEGACY_ARCHIVE / "MANIFEST.csv")
    old_paths = {row["old_path"] for row in rows}
    assert "other_outputs" in old_paths
    assert "scripts/config_visual.py" in old_paths
    for row in rows:
        assert row["dependency_scan_reason"].strip(), row
```

- [ ] **Step 7: Stage move metadata carefully**

Run:

```bash
git add -A
git status --short
```

Expected:
- tracked thresholded GeoRF files show as renames or delete/add under `archived/release_20260624_reproducibility_inputs/`
- tracked `other_outputs/*.py` and `other_outputs/new_results_append/.../audit_only_failure_summary.txt` show as moved under `archived/release_20260624_legacy_workspace/`
- large untracked archive payloads remain ignored
- archive README/MANIFEST files are staged

- [ ] **Step 8: Run clean-root contract tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest src/tests/test_release_cleanup_contract.py -q
```

Expected: all release cleanup contract tests pass.

- [ ] **Step 9: Commit archive move**

Run:

```bash
git commit -m "archive clean root release inputs"
```

## Task 6: Update Docs And Rebuild Package

**Files:**
- Modify: `README.md`
- Modify: `PIPELINE_WORKFLOW.md`
- Modify: `CURRENT_RESULTS_REPRODUCTION.md`
- Modify: `RELEASE_MANIFEST.md`
- Regenerate: `paper_reproducibility_package/`

- [ ] **Step 1: Update docs to remove root-level archived paths**

Make these exact documentation changes:

- In `README.md`, change Stage 3 output text from root `result_partition...` and `other_outputs/...` to:

```markdown
Output: archived Stage 3 result folders under
`archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_fsN/`.
Paper-facing aggregate tables are in
`final_artifacts_in_paper_updated/01_main_results/` and the
`paper_reproducibility_package/` copy.
```

- In the `README.md` directory tree, replace root `GeoRFExperiment/` and `GeoDTExperiment/` entries with:

```markdown
├── final_artifacts_in_paper_updated/ # Paper-facing figures/tables/PDF
├── paper_reproducibility_package/    # Fast paper artifact audit package
├── archived/
│   ├── release_20260624_reproducibility_inputs/ # Heavy local verifier/package inputs
│   ├── release_20260624_legacy_workspace/       # Historical helper scripts and outputs
│   └── release_20260624_nonpaper_pipelines/     # Non-paper workflow provenance
```

- In `PIPELINE_WORKFLOW.md`, replace `other_outputs/Table_Format.xlsx` and `other_outputs/Model_Comparison_Table.xlsx` references with `final_artifacts_in_paper_updated/01_main_results/main_month_ind_cont3.xlsx` and package copies.
- In `CURRENT_RESULTS_REPRODUCTION.md`, rename "Live Stage 3 main results" to "Archived local Stage 3 main results" and point paths under `archived/release_20260624_reproducibility_inputs/`.
- In `CURRENT_RESULTS_REPRODUCTION.md`, update the verifier checklist to say it checks archived Stage 2/3/ablation inputs plus root final artifacts.
- In `RELEASE_MANIFEST.md`, add a new section:

```markdown
## Clean-Root Follow-Up Release

- Planned tag: `v1.0.1-paper-reproducibility-clean-root-20260624`
- Keeps `v1.0-paper-reproducibility-20260624` unchanged.
- Keeps `final_artifacts_in_paper_updated/` and `paper_reproducibility_package/` in the repository root.
- Moves heavy local verifier/package inputs to `archived/release_20260624_reproducibility_inputs/`.
- Moves legacy helper workspaces to `archived/release_20260624_legacy_workspace/`.
- Moves non-release local residue to `archived/local_workspace_residue_20260624/`.
```

- [ ] **Step 2: Scan docs for stale root references**

Run:

```bash
rg -n "other_outputs/|^[- ]*`?GeoRFExperiment/|^[- ]*`?GeoDTExperiment/|result_partition_k40_compare_|main_ablation_exclude_updated_stage3_fixed_partitions" README.md PIPELINE_WORKFLOW.md CURRENT_RESULTS_REPRODUCTION.md RELEASE_MANIFEST.md
```

Expected: every remaining result/experiment/ablation reference is either inside `archived/release_20260624_reproducibility_inputs/`, a reproduction command/output explanation, or historical text explicitly labeled archived.

- [ ] **Step 3: Rebuild the paper reproducibility package**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/build_paper_reproducibility_package.py
```

Expected: prints `Built paper_reproducibility_package`, file count, and byte count.

- [ ] **Step 4: Force-add regenerated package files that are normally ignored**

Run:

```bash
git add README.md PIPELINE_WORKFLOW.md CURRENT_RESULTS_REPRODUCTION.md RELEASE_MANIFEST.md
git add -f paper_reproducibility_package
```

Expected: package manifest/docs/final artifact copies staged as needed; large root archive payloads are not force-added.

- [ ] **Step 5: Run package validator**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_paper_reproducibility_package.py
```

Expected: `Package validation passed: ... files checked.`

- [ ] **Step 6: Commit docs and package rebuild**

Run:

```bash
git commit -m "document clean root release layout"
```

## Task 7: Full Verification And Tag

**Files:**
- No source edits expected unless verification exposes a concrete missed path.
- Create tag: `v1.0.1-paper-reproducibility-clean-root-20260624`

- [ ] **Step 1: Run focused tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  src/tests/test_release_paths.py \
  src/tests/test_release_cleanup_contract.py \
  src/tests/test_artifact_source_audit.py \
  src/tests/test_partition_comparison_contract.py \
  src/tests/test_georf_stage3_manifest_provenance.py \
  src/tests/test_build_georf_partitioned_shap_heatmap.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run package unit tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest src.tests.test_paper_reproducibility_package -v
```

Expected: `OK`.

- [ ] **Step 3: Validate package**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_paper_reproducibility_package.py
```

Expected: `Package validation passed: ... files checked.`

- [ ] **Step 4: Run live result verifier**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_current_results_reproducibility.py
```

Expected: ends with:

```text
Verification passed: current result bundle is organized and reproducibility metadata is present.
```

- [ ] **Step 5: Check tracked/ignored state**

Run:

```bash
git diff --check
git status --short
git status --ignored --short archived/release_20260624_reproducibility_inputs archived/local_workspace_residue_20260624 | head -120
```

Expected:
- `git diff --check` has no output.
- `git status --short` is clean after commits.
- ignored archive payloads are visible only in ignored status, not unstaged tracked status.

- [ ] **Step 6: Confirm root remains clean**

Run:

```bash
find . -maxdepth 1 -type d -printf '%f\n' | sort
```

Expected: root still contains `final_artifacts_in_paper_updated`, `paper_reproducibility_package`, source/config/docs/app/scripts, and `archived`, but does not contain the archived heavy/result/residue folders listed in Task 5.

- [ ] **Step 7: Create annotated tag**

Run:

```bash
git tag -a v1.0.1-paper-reproducibility-clean-root-20260624 -m "Paper reproducibility clean-root release"
git show --stat --oneline --decorate v1.0.1-paper-reproducibility-clean-root-20260624
```

Expected: tag points to the clean verified HEAD and does not move `v1.0-paper-reproducibility-20260624`.

- [ ] **Step 8: Final status report**

Report:

- final commit hash
- tag name
- verifier result
- package validator result
- root folders removed
- archives created
- any intentionally ignored local archive payloads
