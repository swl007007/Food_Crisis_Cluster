# Paper Reproducibility Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lightweight repo-internal paper reproducibility package and align root `README.md` with the current manuscript-facing GeoRF no-leak workflow.

**Architecture:** Add two small Python utilities: one deterministic builder that assembles `paper_reproducibility_package/` from existing artifacts and one validator that checks package manifest/checksums. Keep source data and large ablation outputs outside the package; use package docs to record paths, checksums, and commands. Update documentation only for experimental workflows; do not move GeoXGB, fs0, or prediction scripts.

**Tech Stack:** Python 3 standard library (`csv`, `hashlib`, `json`, `pathlib`, `shutil`, `subprocess`, `unittest`), Markdown, existing repo artifacts and verifier.

---

## File Structure

- Create: `scripts/build_paper_reproducibility_package.py`
  - Deterministically rebuilds `paper_reproducibility_package/`, copies selected artifacts, and writes package docs plus `MANIFEST.csv` and `SHA256SUMS.txt`.
- Create: `scripts/validate_paper_reproducibility_package.py`
  - Validates package file existence and SHA-256 checksums from `MANIFEST.csv` and `SHA256SUMS.txt`.
- Create: `src/tests/test_paper_reproducibility_package.py`
  - Unit tests for checksum, manifest writing, package-path safety, and checksum validation.
- Modify: `README.md`
  - Make the repo first-read paper-facing, update geography, separate main workflow from GeoXGB/fs0/forward prediction extensions, and point to package README.
- Generate: `paper_reproducibility_package/`
  - Contains copied Stage 2 maps, Stage 3 summaries/refined maps, final paper artifacts, lightweight ablation provenance, package docs, manifest, and checksums.
- Do not modify or move:
  - `app/main_model_XGB.py`
  - `GeoXGBExperiment/`
  - `prediction_pipeline/`
  - `src/utils/lag_schedules.py`
  - source data under `Analysis/1.Source Data/`
  - full `main_ablation_exclude_updated_stage3_fixed_partitions/` output tree beyond reading two manifest files

## Task 1: Add Package Utility Tests

**Files:**
- Create: `src/tests/test_paper_reproducibility_package.py`
- Read: `docs/superpowers/specs/2026-06-24-paper-reproducibility-package-design.md`

- [ ] **Step 1: Create the test file**

Use `apply_patch` to add `src/tests/test_paper_reproducibility_package.py`:

```python
"""Tests for paper reproducibility package builder and validator helpers."""

from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.build_paper_reproducibility_package import (
    ManifestRow,
    PackageBuildError,
    ensure_package_relative,
    sha256_file,
    write_manifest,
    write_sha256sums,
)
from scripts.validate_paper_reproducibility_package import validate_package


class PaperReproducibilityPackageTests(unittest.TestCase):
    def test_sha256_file_matches_hashlib(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.txt"
            path.write_text("abc\n", encoding="utf-8")

            expected = hashlib.sha256(b"abc\n").hexdigest()
            self.assertEqual(sha256_file(path), expected)

    def test_ensure_package_relative_rejects_absolute_and_parent_paths(self) -> None:
        self.assertEqual(ensure_package_relative("stage2/file.csv"), Path("stage2/file.csv"))

        with self.assertRaises(PackageBuildError):
            ensure_package_relative("/tmp/file.csv")

        with self.assertRaises(PackageBuildError):
            ensure_package_relative("../outside.csv")

    def test_write_manifest_and_sha256sums_are_validator_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package_root = Path(tmp) / "paper_reproducibility_package"
            package_root.mkdir()
            data_path = package_root / "stage2_cluster_maps" / "georf" / "cluster_mapping.csv"
            data_path.parent.mkdir(parents=True)
            data_path.write_text("admin_code,cluster\nA,1\n", encoding="utf-8")
            checksum = sha256_file(data_path)
            size = data_path.stat().st_size

            rows = [
                ManifestRow(
                    package_path="stage2_cluster_maps/georf/cluster_mapping.csv",
                    source_path="GeoRFExperiment/knn_sparsification_results/cluster_mapping.csv",
                    role="stage2_georf_cluster_map",
                    size_bytes=size,
                    sha256=checksum,
                    notes="test file",
                )
            ]

            write_manifest(package_root, rows)
            write_sha256sums(package_root, rows)

            with (package_root / "MANIFEST.csv").open(newline="", encoding="utf-8") as handle:
                loaded = list(csv.DictReader(handle))
            self.assertEqual(loaded[0]["package_path"], "stage2_cluster_maps/georf/cluster_mapping.csv")
            self.assertEqual(loaded[0]["sha256"], checksum)

            result = validate_package(package_root)
            self.assertEqual(result.checked_files, 1)
            self.assertEqual(result.failures, [])

    def test_validator_reports_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package_root = Path(tmp) / "paper_reproducibility_package"
            package_root.mkdir()
            file_path = package_root / "README.md"
            file_path.write_text("changed\n", encoding="utf-8")
            bad_checksum = "0" * 64

            rows = [
                ManifestRow(
                    package_path="README.md",
                    source_path="generated",
                    role="package_documentation",
                    size_bytes=file_path.stat().st_size,
                    sha256=bad_checksum,
                    notes="bad hash test",
                )
            ]
            write_manifest(package_root, rows)
            write_sha256sums(package_root, rows)

            result = validate_package(package_root)
            self.assertEqual(result.checked_files, 1)
            self.assertTrue(any("checksum mismatch" in item for item in result.failures))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest src.tests.test_paper_reproducibility_package -v
```

Expected: import failure because `scripts/build_paper_reproducibility_package.py` and `scripts/validate_paper_reproducibility_package.py` do not exist yet.

- [ ] **Step 3: Commit the failing tests**

Run:

```bash
git add src/tests/test_paper_reproducibility_package.py
git commit -m "test paper reproducibility package utilities"
```

Expected: commit records only the new test file.

## Task 2: Implement Package Builder and Validator Helpers

**Files:**
- Create: `scripts/build_paper_reproducibility_package.py`
- Create: `scripts/validate_paper_reproducibility_package.py`
- Test: `src/tests/test_paper_reproducibility_package.py`

- [ ] **Step 1: Add the builder utility**

Use `apply_patch` to add `scripts/build_paper_reproducibility_package.py`:

```python
"""Build the lightweight paper reproducibility package.

The package is intentionally generated from existing artifacts. It does not
rerun models, does not copy raw source data, and does not move experimental
workflows.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "paper_reproducibility_package"


class PackageBuildError(RuntimeError):
    """Raised when the package cannot be built safely."""


@dataclass(frozen=True)
class ManifestRow:
    package_path: str
    source_path: str
    role: str
    size_bytes: int
    sha256: str
    notes: str


def ensure_package_relative(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise PackageBuildError(f"Package path must be relative, got {value!s}")
    if any(part == ".." for part in path.parts):
        raise PackageBuildError(f"Package path must not escape package root, got {value!s}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def copy_file(package_root: Path, source: Path, package_path: str, role: str, notes: str) -> ManifestRow:
    if not source.is_file():
        raise PackageBuildError(f"Required source file missing: {source}")
    relative = ensure_package_relative(package_path)
    destination = package_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return ManifestRow(
        package_path=relative.as_posix(),
        source_path=str(source.relative_to(REPO_ROOT)) if source.is_relative_to(REPO_ROOT) else str(source),
        role=role,
        size_bytes=destination.stat().st_size,
        sha256=sha256_file(destination),
        notes=notes,
    )


def write_manifest(package_root: Path, rows: list[ManifestRow]) -> None:
    manifest_path = package_root / "MANIFEST.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["package_path", "source_path", "role", "size_bytes", "sha256", "notes"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "package_path": row.package_path,
                    "source_path": row.source_path,
                    "role": row.role,
                    "size_bytes": row.size_bytes,
                    "sha256": row.sha256,
                    "notes": row.notes,
                }
            )


def write_sha256sums(package_root: Path, rows: list[ManifestRow]) -> None:
    content = "".join(f"{row.sha256}  {row.package_path}\n" for row in sorted(rows, key=lambda item: item.package_path))
    write_text(package_root / "SHA256SUMS.txt", content)


def reset_package_root(package_root: Path) -> None:
    resolved = package_root.resolve()
    expected = (REPO_ROOT / "paper_reproducibility_package").resolve()
    if resolved != expected:
        raise PackageBuildError(f"Refusing to reset unexpected package root: {package_root}")
    if package_root.exists():
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def add_generated_doc(rows: list[ManifestRow], package_root: Path, package_path: str, role: str, content: str, notes: str) -> None:
    relative = ensure_package_relative(package_path)
    destination = package_root / relative
    write_text(destination, content)
    rows.append(
        ManifestRow(
            package_path=relative.as_posix(),
            source_path="generated by scripts/build_paper_reproducibility_package.py",
            role=role,
            size_bytes=destination.stat().st_size,
            sha256=sha256_file(destination),
            notes=notes,
        )
    )


def stage2_specs() -> list[tuple[Path, str, str]]:
    return [
        (
            REPO_ROOT / "GeoRFExperiment" / "knn_sparsification_results" / "cluster_mapping_manifest.json",
            "stage2_cluster_maps/georf/cluster_mapping_manifest.json",
            "GeoRF Stage 2 manifest",
        ),
        (
            REPO_ROOT / "GeoRFExperiment" / "knn_sparsification_results" / "cluster_mapping_k40_nc17_general.csv",
            "stage2_cluster_maps/georf/cluster_mapping_k40_nc17_general.csv",
            "GeoRF general consensus map",
        ),
        (
            REPO_ROOT / "GeoRFExperiment" / "knn_sparsification_results" / "cluster_mapping_k40_nc13_m2.csv",
            "stage2_cluster_maps/georf/cluster_mapping_k40_nc13_m2.csv",
            "GeoRF February consensus map",
        ),
        (
            REPO_ROOT / "GeoRFExperiment" / "knn_sparsification_results" / "cluster_mapping_k40_nc11_m6.csv",
            "stage2_cluster_maps/georf/cluster_mapping_k40_nc11_m6.csv",
            "GeoRF June consensus map",
        ),
        (
            REPO_ROOT / "GeoRFExperiment" / "knn_sparsification_results" / "cluster_mapping_k40_nc16_m10.csv",
            "stage2_cluster_maps/georf/cluster_mapping_k40_nc16_m10.csv",
            "GeoRF October consensus map",
        ),
        (
            REPO_ROOT / "GeoDTExperiment" / "knn_sparsification_results" / "cluster_mapping_manifest.json",
            "stage2_cluster_maps/geodt/cluster_mapping_manifest.json",
            "GeoDT Stage 2 manifest",
        ),
        (
            REPO_ROOT / "GeoDTExperiment" / "knn_sparsification_results" / "cluster_mapping_k40_nc15_general.csv",
            "stage2_cluster_maps/geodt/cluster_mapping_k40_nc15_general.csv",
            "GeoDT general consensus map",
        ),
        (
            REPO_ROOT / "GeoDTExperiment" / "knn_sparsification_results" / "cluster_mapping_k40_nc16_m2.csv",
            "stage2_cluster_maps/geodt/cluster_mapping_k40_nc16_m2.csv",
            "GeoDT February consensus map",
        ),
        (
            REPO_ROOT / "GeoDTExperiment" / "knn_sparsification_results" / "cluster_mapping_k40_nc15_m6.csv",
            "stage2_cluster_maps/geodt/cluster_mapping_k40_nc15_m6.csv",
            "GeoDT June consensus map",
        ),
        (
            REPO_ROOT / "GeoDTExperiment" / "knn_sparsification_results" / "cluster_mapping_k40_nc18_m10.csv",
            "stage2_cluster_maps/geodt/cluster_mapping_k40_nc18_m10.csv",
            "GeoDT October consensus map",
        ),
    ]


def stage3_result_dirs() -> list[tuple[str, Path]]:
    return [
        ("georf_fs1", REPO_ROOT / "result_partition_k40_compare_GF_fs1"),
        ("georf_fs2", REPO_ROOT / "result_partition_k40_compare_GF_fs2"),
        ("georf_fs3", REPO_ROOT / "result_partition_k40_compare_GF_fs3"),
        ("geodt_fs1", REPO_ROOT / "result_partition_k40_compare_DT_fs1"),
        ("geodt_fs2", REPO_ROOT / "result_partition_k40_compare_DT_fs2"),
        ("geodt_fs3", REPO_ROOT / "result_partition_k40_compare_DT_fs3"),
    ]


def canonical_refined_files(source_dir: Path, manifest: dict) -> list[Path]:
    refined_dir = source_dir / "refined"
    if not refined_dir.is_dir():
        raise PackageBuildError(f"Missing refined directory: {refined_dir}")

    selected: dict[str, Path] = {}
    partition_path = manifest.get("partition_map_path")
    if partition_path:
        normalized = Path(str(partition_path).replace("\\", "/"))
        source = normalized if normalized.is_absolute() else REPO_ROOT / normalized
        if source.is_file():
            selected[source.name] = source

    for source in sorted(refined_dir.glob("cluster_mapping_k40_nc*_refined_contig3.csv")):
        if "_refined_contig3_refined" in source.name:
            continue
        selected[source.name] = source

    for source in sorted(refined_dir.glob("refine_summary_cluster_mapping_k40_nc*.txt")):
        if "_refined_contig3" in source.name:
            continue
        selected[source.name] = source

    return [selected[name] for name in sorted(selected)]


def add_stage2_files(rows: list[ManifestRow], package_root: Path) -> None:
    for source, package_path, notes in stage2_specs():
        rows.append(copy_file(package_root, source, package_path, "stage2_cluster_map", notes))


def add_stage3_files(rows: list[ManifestRow], package_root: Path) -> None:
    core_files = ("run_manifest.json", "metrics_monthly.csv", "metrics_polygon_overall.csv", "predictions_monthly.csv")
    for package_dir, source_dir in stage3_result_dirs():
        manifest = load_json(source_dir / "run_manifest.json")
        for filename in core_files:
            rows.append(
                copy_file(
                    package_root,
                    source_dir / filename,
                    f"stage3_results/{package_dir}/{filename}",
                    "stage3_result_core",
                    f"{source_dir.name}/{filename}",
                )
            )
        for source in canonical_refined_files(source_dir, manifest):
            rows.append(
                copy_file(
                    package_root,
                    source,
                    f"stage3_results/{package_dir}/refined/{source.name}",
                    "stage3_refined_partition_map",
                    f"refined map or refinement summary from {source_dir.name}",
                )
            )


def add_final_artifacts(rows: list[ManifestRow], package_root: Path) -> None:
    source_root = REPO_ROOT / "final_artifacts_in_paper_updated"
    if not source_root.is_dir():
        raise PackageBuildError(f"Missing final artifact directory: {source_root}")
    for source in sorted(source_root.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(source_root)
        rows.append(
            copy_file(
                package_root,
                source,
                f"paper_artifacts/final_artifacts_in_paper_updated/{relative.as_posix()}",
                "paper_final_artifact",
                "active manuscript artifact",
            )
        )


def add_ablation_provenance(rows: list[ManifestRow], package_root: Path) -> None:
    sources = [
        (
            REPO_ROOT / "main_ablation_exclude_updated_stage3_fixed_partitions" / "ablation_run_manifest.json",
            "ablation/ablation_run_manifest.json",
            "fixed-partition ablation run manifest",
        ),
        (
            REPO_ROOT
            / "main_ablation_exclude_updated_stage3_fixed_partitions"
            / "input_datasets"
            / "feature_exclude_dataset_manifest.json",
            "ablation/feature_exclude_dataset_manifest.json",
            "feature-exclude input dataset manifest",
        ),
    ]
    for source, package_path, notes in sources:
        rows.append(copy_file(package_root, source, package_path, "ablation_provenance", notes))


def package_readme() -> str:
    return """# FEWS NET GeoRF Paper Reproducibility Package

This package supports fast audit and replication of the current manuscript-facing
GeoRF no-leak workflow. It bundles frozen Stage 2 consensus maps, Stage 3 result
summaries, final paper artifacts, and lightweight ablation provenance.

## What This Package Is

- Paper scope: 22 FEWS NET monitored countries across Africa, the Middle East,
  Asia, and Latin America.
- Main paper model: GeoRF with fixed partitions learned from 2018-2020 and
  evaluated on 2021-2024.
- Forecast horizons: 4, 8, and 12 months (`fs1`, `fs2`, `fs3`).
- Evaluation months: February, June, and October releases in 2021-2024.
- Fast path: inspect packaged artifacts or rerun Stage 3 using the packaged
  Stage 2 maps, without rerunning Stage 1 partition learning.

## What This Package Is Not

- It does not include raw source data.
- It does not include the full 4.7 GB ablation output tree.
- It does not include experimental GeoXGB, fs0 lag-1, or 2026-2027 forward
  prediction/scenario workflows.
- It does not move or archive experimental scripts; release-version code
  migration is a separate future task.

## Quick Validation

Run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_paper_reproducibility_package.py
```

Expected result: all files listed in `MANIFEST.csv` and `SHA256SUMS.txt` exist
and match their SHA-256 checksums.

## Full Repo Verification

Run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_current_results_reproducibility.py
```

This checks the live Stage 2/Stage 3/final-artifact bundle outside this copied
package.
"""


def source_data_note() -> str:
    return """# Source Data Note

The raw assembled panel is not copied into this package.

Expected source file:

- Windows: `C:\\Users\\swl00\\IFPRI Dropbox\\Weilun Shi\\Google fund\\Analysis\\1.Source Data\\FEWSNET_forecast_unadjusted_bm.csv`
- WSL: `/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/1.Source Data/FEWSNET_forecast_unadjusted_bm.csv`

Recorded metadata:

- SHA-256: `611f9e776380e28da3fc845888d66a117626f91b37868e236ce849d44bc8f651`
- Rows x columns: `1,029,240 x 88`
- Coverage: `2010-01` to `2024-12`; 22 ISO3 countries; 5,718 FEWSNET admin codes
- Local modified time: `2025-11-05 09:41:30 -0500`

The paper analysis subset is smaller than the assembled panel because the
manuscript uses the valid FEWS NET release-month and area subset. This package
does not make public redistribution claims for restricted or third-party source
data.
"""


def artifact_map() -> str:
    return """# Paper Artifact Map

| Manuscript item | Package path | Notes |
|---|---|---|
| Manuscript PDF | `paper_artifacts/final_artifacts_in_paper_updated/Forecasting_FEWS_NET_Food_Security_Crises_Using_a_Geo_Aware_Spatial_Clustering_Model (1).pdf` | Reference text checked against repo artifacts. |
| Table 1 main performance | `paper_artifacts/final_artifacts_in_paper_updated/01_main_results/main_month_ind_cont3.xlsx` | GeoRF/GeoDT/FEWS NET summary workbook. |
| Figure 1 GeoRF partition maps | `paper_artifacts/final_artifacts_in_paper_updated/01_main_results/global_cluster_map_2x2_georf_refined.png` | Refined general and month-specific GeoRF maps. |
| Figure 3 monthly performance | `paper_artifacts/final_artifacts_in_paper_updated/01_main_results/georf_monthly_performance.png` | Uses Stage 3 GeoRF metrics and FEWS NET baselines. |
| Figure 4 SHAP attribution | `paper_artifacts/final_artifacts_in_paper_updated/01_main_results/georf_partitioned_shap_group_heatmap.png` | Grouped SHAP diagnostic. |
| Figure 5 GeoDT branch comparison | `paper_artifacts/final_artifacts_in_paper_updated/08_geodt_diagnostics/geodt_branch_tree_compare_2024-10_fs1_001_vs_1.png` | Appendix/interpretability comparison. |
| Figure 6 2024 prediction maps | `paper_artifacts/final_artifacts_in_paper_updated/01_main_results/predictions_2024_feb_jun_oct.png` | GeoRF 8-month prediction comparison. |
| Figure 7 precision-recall curves | `paper_artifacts/final_artifacts_in_paper_updated/11_threshold_free_metrics/georf_precision_recall_curves.png` | Threshold-free GeoRF diagnostic. |
| Appendix error maps | `paper_artifacts/final_artifacts_in_paper_updated/04_error_analysis/` | Seasonal all/crisis/non-crisis error panels. |
| Appendix class prevalence | `paper_artifacts/final_artifacts_in_paper_updated/03_class_prevalence/` | Test-period class prevalence and context figure. |
| Appendix probability diagnostics | `paper_artifacts/final_artifacts_in_paper_updated/07_probability_uncertainty/` | Reliability, Brier, and bootstrap summaries. |
| Appendix thresholded GeoRF | `paper_artifacts/final_artifacts_in_paper_updated/12_thresholded_georf_results/` | Thresholded GeoRF provenance and compact tables. |
| Nature-style data manifest | `paper_artifacts/final_artifacts_in_paper_updated/02_methods_and_temporal_scope/fewsnet_data_manifest_nature_portfolio.md` | Minimum dataset and derived asset manifest. |
"""


def consistency_audit() -> str:
    return """# Consistency Audit

## Aligned With Manuscript

- Main model: GeoRF is the paper-facing model; GeoDT is auxiliary appendix /
  interpretability comparison.
- Temporal split: Stage 1 learns partitions on 2018-2020; Stage 2 builds fixed
  consensus maps; Stage 3 evaluates 2021-2024.
- Horizons: `fs1`, `fs2`, and `fs3` correspond to 4-, 8-, and 12-month forecasts.
- Stage 3 result manifests record `n_test_months_evaluated=12`, matching
  February, June, and October evaluation months for 2021-2024.
- Stage 3 GeoRF manifests record `5,716` polygons and `62,189` predictions per
  horizon.

## Accepted Differences

- The manuscript's `200,060` observations describe the analysis subset, not the
  full assembled CSV.
- The full source CSV is `1,029,240 x 88` and is intentionally not copied into
  this package.
- Predictor-count wording differs depending on whether base predictors or
  engineered lag/rolling/encoded features are counted.

## Excluded Experimental Workflows

GeoXGB, fs0 lag-1, and 2026-2027 forward/scenario prediction are not part of
this package. They remain in their current repo paths for future release-version
migration.
"""


def add_generated_docs(rows: list[ManifestRow], package_root: Path) -> None:
    docs = [
        ("README.md", "package_documentation", package_readme(), "package overview and quick validation"),
        ("SOURCE_DATA.md", "source_data_note", source_data_note(), "external source data path and checksum"),
        ("PAPER_ARTIFACT_MAP.md", "paper_artifact_map", artifact_map(), "manuscript item to artifact mapping"),
        ("CONSISTENCY_AUDIT.md", "consistency_audit", consistency_audit(), "PDF-to-repo consistency summary"),
    ]
    for package_path, role, content, notes in docs:
        add_generated_doc(rows, package_root, package_path, role, content, notes)


def build_package(package_root: Path = PACKAGE_ROOT) -> list[ManifestRow]:
    reset_package_root(package_root)
    rows: list[ManifestRow] = []
    add_generated_docs(rows, package_root)
    add_stage2_files(rows, package_root)
    add_stage3_files(rows, package_root)
    add_final_artifacts(rows, package_root)
    add_ablation_provenance(rows, package_root)
    write_manifest(package_root, rows)
    manifest_row = ManifestRow(
        package_path="MANIFEST.csv",
        source_path="generated by scripts/build_paper_reproducibility_package.py",
        role="package_manifest",
        size_bytes=(package_root / "MANIFEST.csv").stat().st_size,
        sha256=sha256_file(package_root / "MANIFEST.csv"),
        notes="package file inventory",
    )
    all_rows = rows + [manifest_row]
    write_sha256sums(package_root, all_rows)
    return all_rows


def main() -> int:
    rows = build_package()
    total_bytes = sum(row.size_bytes for row in rows)
    print(f"Built {PACKAGE_ROOT.relative_to(REPO_ROOT)}")
    print(f"Files: {len(rows)}")
    print(f"Bytes: {total_bytes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Add the validator utility**

Use `apply_patch` to add `scripts/validate_paper_reproducibility_package.py`:

```python
"""Validate the lightweight paper reproducibility package."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE_ROOT = REPO_ROOT / "paper_reproducibility_package"


@dataclass(frozen=True)
class ValidationResult:
    checked_files: int
    failures: list[str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(package_root: Path) -> dict[str, str]:
    manifest_path = package_root / "MANIFEST.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing package manifest: {manifest_path}")
    checksums: dict[str, str] = {}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            checksums[row["package_path"]] = row["sha256"]
    return checksums


def load_sha256sums(package_root: Path) -> dict[str, str]:
    sums_path = package_root / "SHA256SUMS.txt"
    if not sums_path.is_file():
        raise FileNotFoundError(f"Missing checksum file: {sums_path}")
    checksums: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        checksum, package_path = line.split(None, 1)
        checksums[package_path.strip()] = checksum
    return checksums


def validate_package(package_root: Path = DEFAULT_PACKAGE_ROOT) -> ValidationResult:
    manifest = load_manifest(package_root)
    sums = load_sha256sums(package_root)
    failures: list[str] = []
    all_paths = sorted(set(manifest) | set(sums))

    for package_path in all_paths:
        manifest_checksum = manifest.get(package_path)
        sums_checksum = sums.get(package_path)
        if manifest_checksum is None:
            failures.append(f"{package_path}: missing from MANIFEST.csv")
            continue
        if sums_checksum is None:
            failures.append(f"{package_path}: missing from SHA256SUMS.txt")
            continue
        if manifest_checksum != sums_checksum:
            failures.append(f"{package_path}: manifest/checksum file disagree")
            continue

        file_path = package_root / package_path
        if not file_path.is_file():
            failures.append(f"{package_path}: listed file missing")
            continue
        actual = sha256_file(file_path)
        if actual != manifest_checksum:
            failures.append(f"{package_path}: checksum mismatch expected {manifest_checksum} got {actual}")

    return ValidationResult(checked_files=len(all_paths), failures=failures)


def main() -> int:
    result = validate_package()
    if result.failures:
        print(f"Package validation failed for {len(result.failures)} issue(s):")
        for failure in result.failures:
            print(f"- {failure}")
        return 1
    print(f"Package validation passed: {result.checked_files} files checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest src.tests.test_paper_reproducibility_package -v
```

Expected: all four tests pass.

- [ ] **Step 4: Commit utilities**

Run:

```bash
git add scripts/build_paper_reproducibility_package.py scripts/validate_paper_reproducibility_package.py
git commit -m "add paper reproducibility package utilities"
```

Expected: commit includes only the two scripts.

## Task 3: Generate the Lightweight Package

**Files:**
- Generate: `paper_reproducibility_package/`
- Read: `GeoRFExperiment/knn_sparsification_results/`
- Read: `GeoDTExperiment/knn_sparsification_results/`
- Read: `result_partition_k40_compare_{GF,DT}_fs{1,2,3}/`
- Read: `final_artifacts_in_paper_updated/`
- Read: `main_ablation_exclude_updated_stage3_fixed_partitions/ablation_run_manifest.json`
- Read: `main_ablation_exclude_updated_stage3_fixed_partitions/input_datasets/feature_exclude_dataset_manifest.json`

- [ ] **Step 1: Build the package**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/build_paper_reproducibility_package.py
```

Expected output contains:

```text
Built paper_reproducibility_package
Files:
Bytes:
```

- [ ] **Step 2: Validate the package**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_paper_reproducibility_package.py
```

Expected output:

```text
Package validation passed: <N> files checked.
```

- [ ] **Step 3: Inspect package size and exclusions**

Run:

```bash
du -sh paper_reproducibility_package
find paper_reproducibility_package -maxdepth 3 -type d | sort
find paper_reproducibility_package/ablation -type f | sort
```

Expected:

- Package size is far below the 4.7 GB full ablation tree.
- `paper_reproducibility_package/ablation/` contains only `ablation_run_manifest.json` and `feature_exclude_dataset_manifest.json`.
- No `GeoXGBExperiment`, `prediction_pipeline`, or `result_partition_k40_compare_*_fs0` directory appears under the package.

- [ ] **Step 4: Commit generated package**

Run:

```bash
git add paper_reproducibility_package
git commit -m "add lightweight paper reproducibility package"
```

Expected: commit includes the package files only.

## Task 4: Update Root README for Paper-First Scope

**Files:**
- Modify: `README.md`
- Read: `paper_reproducibility_package/README.md`
- Read: `docs/superpowers/specs/2026-06-24-paper-reproducibility-package-design.md`

- [ ] **Step 1: Update the top-level summary and Quick Start**

Use `apply_patch` to edit the start of `README.md` so the title and overview state:

````markdown
# GeoRF Food Crisis Forecasting Paper Reproducibility

This repository contains the manuscript-facing FEWS NET food-crisis forecasting
workflow built around GeoRF, a geo-aware Random Forest that learns spatially
heterogeneous predictive regimes. The current paper results use a no-leak
three-stage workflow: Stage 1 learns candidate partitions on 2018-2020, Stage 2
builds fixed consensus maps, and Stage 3 evaluates fixed partitions on
2021-2024 FEWS NET release months.

The paper scope covers 22 FEWS NET monitored countries across Africa, the
Middle East, Asia, and Latin America. It is not limited to Sub-Saharan Africa.
The main model is GeoRF; GeoDT is retained as an auxiliary appendix and
interpretability comparison. GeoXGB, fs0 lag-1, and 2026-2027 forward/scenario
prediction workflows are experimental extensions and are not part of the paper
reproducibility package.

## Quick Start for Paper Reproduction

For fast review, start with the lightweight package:

```text
paper_reproducibility_package/README.md
paper_reproducibility_package/MANIFEST.csv
paper_reproducibility_package/SHA256SUMS.txt
```

Validate it from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_paper_reproducibility_package.py
```

Then verify the live repository result bundle:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_current_results_reproducibility.py
```
````

Remove the existing `This project applies the GeoRF framework to **FEWSNET food crisis prediction** in Sub-Saharan Africa` sentence and the old Quick Start positioning that puts fs0 and 2026-2027 prediction before the paper workflow.

- [ ] **Step 2: Rewrite the complete workflow section**

Keep the three-stage commands, but make them the slow complete regeneration path:

````markdown
## Complete No-Leak Regeneration Path

The full regeneration path is slower than the package audit path because Stage 1
partition learning is expensive. Use it when you need to regenerate the complete
workflow rather than audit the packaged paper artifacts.

### Stage 1: Learn partition candidates on 2018-2020

```batch
run_batches_2018_2020_partition_learning_visual_monthly.bat georf
run_batches_2018_2020_partition_learning_visual_monthly.bat geodt
```

### Stage 2: Generate fixed consensus partitions

```batch
spatial_weighted_consensus_clustering.bat georf
spatial_weighted_consensus_clustering.bat geodt
```

### Stage 3: Evaluate 2021-2024 fixed partitions

```batch
run_partition_k40_comparison_unified.bat georf --visual --month-ind
run_partition_k40_comparison_unified.bat geodt --visual --month-ind
```

GeoRF is the paper-facing model. GeoDT is kept for appendix comparison and
branch-level interpretability. `run_partition_k40_comparison_unified.bat all`
is still available for local convenience, but paper reproduction should treat
GeoRF as the main model family.
````

- [ ] **Step 3: Move experimental workflows into a separate section**

Move the existing fs0 and 2026-2027 prediction content under:

````markdown
## Experimental and Extension Workflows

These workflows remain in the repository for development continuity but are not
part of the current paper reproducibility package. They should not be interpreted
as manuscript main results.

### GeoXGB

`app/main_model_XGB.py`, `GeoXGBExperiment/`, and XGBoost comparison scripts are
legacy/experimental. They are not included in the no-leak paper workflow or the
lightweight package.

### fs0 Lag-1 Extension

fs0 is a stand-alone lag-1 extension. It is orthogonal to the paper's fs1/fs2/fs3
4-, 8-, and 12-month workflow and is not included in the paper reproducibility
package.

```batch
run_batches_2018_2020_partition_learning_visual_monthly.bat <model> --fs0-only
spatial_weighted_consensus_clustering.bat <model> --fs0-only
run_partition_k40_comparison_unified.bat <model> --fs0-only
```

Keep `<model>` as `georf` or `geodt`. Running fs0 requires the flag on all three
stages and writes separate fs0 workbooks, so it must not be mixed with the
paper fs1/fs2/fs3 outputs.

### 2026-2027 Forward and Scenario Prediction

The `prediction_pipeline/` launchers support GeoRF-only forward prediction for
June 2026 and February 2027 plus a separate synthetic scenario overlay. These
outputs are operational extensions, not manuscript backtest results.

```batch
prediction_pipeline\spatial_weighted_consensus_clustering_predict.bat georf
prediction_pipeline\run_predict_2026_2027.bat georf
prediction_pipeline\run_partition_predict_unified.bat georf
prediction_pipeline\run_scenario_predict_jun2026_feb2027.bat
```

Standard forward outputs go to `deliverables\predict_2026_2027\`. Synthetic
scenario outputs go to `deliverables\predict_scenario_jun2026_feb2027\` and
should not be described as standard manuscript forecasts.
````

Keep the practical commands for fs0 and prediction under these subsections, but state that paths and scripts are not moved in this task.

- [ ] **Step 4: Fix performance wording and config summary**

Replace the old line:

```markdown
- **Crisis Prediction F1**: 0.70-0.80 (partitioned XGBoost)
```

with:

```markdown
- **GeoRF paper summary**: In the current manuscript table, partitioned GeoRF
  improves crisis-class F1 over pooled RF at 4-, 8-, and 12-month horizons.
  Use `final_artifacts_in_paper_updated/01_main_results/main_month_ind_cont3.xlsx`
  and `paper_reproducibility_package/PAPER_ARTIFACT_MAP.md` as the source of
  paper-facing numbers.
```

Update the config block to include:

```python
ACTIVE_LAGS = (4, 8, 12)
TRAIN_WINDOW_MONTHS = 36
DATA_MODE = "unadjusted"
DATA_PATH = "FEWSNET_forecast_unadjusted_bm.csv"
CONTIGUITY = True
USE_ADJACENCY_MATRIX = True
K = 40
SIGMA = 5.0
RF_STAGE3 = {"n_estimators": 100, "max_depth": None, "random_state": 5, "n_jobs": 1}
```

Add prose that `K=40` is graph-neighbor count, not cluster count, and that Stage 3 artifacts use contiguity-refined maps with three refinement iterations.

- [ ] **Step 5: Run README scope checks**

Run:

```bash
rg -n "Sub-Saharan Africa|partitioned XGBoost|Crisis Prediction F1" README.md
rg -n "Experimental and Extension Workflows|GeoXGB|fs0 Lag-1|2026-2027 Forward" README.md
```

Expected:

- First command exits non-zero with no matches.
- Second command finds the experimental section and subsections.

- [ ] **Step 6: Commit README update**

Run:

```bash
git add README.md
git commit -m "align readme with paper reproducibility scope"
```

Expected: commit includes only `README.md`.

## Task 5: Full Verification and Final Cleanups

**Files:**
- Read/verify: `paper_reproducibility_package/`
- Read/verify: `README.md`
- Read/verify: current result bundle

- [ ] **Step 1: Run unit tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest src.tests.test_paper_reproducibility_package -v
```

Expected: all tests pass.

- [ ] **Step 2: Validate package checksums**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_paper_reproducibility_package.py
```

Expected:

```text
Package validation passed: <N> files checked.
```

- [ ] **Step 3: Run existing reproducibility verifier**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_current_results_reproducibility.py
```

Expected: output ends with:

```text
Verification passed: current result bundle is organized and reproducibility metadata is present.
```

If this command rewrites only line endings in `final_artifacts_in_paper_updated/artifact_source_audit.md`, restore that file before committing any final cleanup:

```bash
git restore final_artifacts_in_paper_updated/artifact_source_audit.md
```

- [ ] **Step 4: Confirm experimental scripts were not moved into the package**

Run:

```bash
find paper_reproducibility_package -maxdepth 4 -type d | rg "GeoXGB|prediction_pipeline|fs0|predict_2026|scenario" || true
git status --short
```

Expected:

- No package directory match for GeoXGB/fs0/prediction/scenario.
- No unexpected modified files.

- [ ] **Step 5: Run final diff checks**

Run:

```bash
git diff --check
git status --short
git log --oneline -5
```

Expected:

- `git diff --check` exits 0.
- `git status --short` is clean.
- Recent commits show tests/utilities, package, README, and plan/spec commits.
