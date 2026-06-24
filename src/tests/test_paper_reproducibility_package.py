"""Tests for paper reproducibility package builder and validator helpers."""

from __future__ import annotations

import csv
import hashlib
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_paper_reproducibility_package as package_builder
from scripts.build_paper_reproducibility_package import (
    ManifestRow,
    PackageBuildError,
    canonical_refined_files,
    ensure_package_relative,
    sha256_file,
    stage2_specs,
    stage3_result_dirs,
    write_manifest,
    write_sha256sums,
)
from scripts.release_paths import ReleasePaths
from scripts.validate_paper_reproducibility_package import validate_package


class PaperReproducibilityPackageTests(unittest.TestCase):
    def test_script_loads_from_direct_path_without_running_main(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "scripts" / "build_paper_reproducibility_package.py"
        scripts_dir = repo_root / "scripts"
        package_root = repo_root / "paper_reproducibility_package"
        before_exists = package_root.exists()
        before_mtime = package_root.stat().st_mtime_ns if before_exists else None
        original_path = list(sys.path)
        saved_modules = {
            name: sys.modules.get(name)
            for name in ("scripts", "scripts.release_paths")
        }

        def is_repo_root_entry(entry: str) -> bool:
            if entry == "":
                return True
            try:
                return Path(entry).resolve() == repo_root.resolve()
            except OSError:
                return False

        try:
            sys.path = [str(scripts_dir)] + [
                entry
                for entry in original_path
                if not is_repo_root_entry(entry) and Path(entry or ".") != scripts_dir
            ]
            for name in saved_modules:
                sys.modules.pop(name, None)

            loaded = runpy.run_path(str(script_path), run_name="not_main")
        finally:
            sys.path = original_path
            for name, module in saved_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        self.assertIn("build_package", loaded)
        self.assertEqual(package_root.exists(), before_exists)
        if before_exists:
            self.assertEqual(package_root.stat().st_mtime_ns, before_mtime)

    def test_canonical_refined_files_resolves_absolute_old_repo_manifest_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            filename = "cluster_mapping_k40_nc17_refined_contig3.csv"
            old_root_file = (
                repo_root
                / "GeoRFExperiment"
                / "knn_sparsification_results"
                / "refined"
                / filename
            )
            archived_file = (
                repo_root
                / "archived"
                / "release_20260624_reproducibility_inputs"
                / "GeoRFExperiment"
                / "knn_sparsification_results"
                / "refined"
                / filename
            )
            archived_file.parent.mkdir(parents=True)
            archived_file.write_text("admin_code,cluster\nA,1\n", encoding="utf-8")
            source_dir = (
                repo_root
                / "archived"
                / "release_20260624_reproducibility_inputs"
                / "result_partition_k40_compare_GF_fs1"
            )
            (source_dir / "refined").mkdir(parents=True)

            with patch.object(package_builder, "PATHS", ReleasePaths(repo_root=repo_root)):
                selected = canonical_refined_files(
                    source_dir,
                    {"partition_map_path": str(old_root_file)},
                )

        self.assertEqual(selected, [archived_file])

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
