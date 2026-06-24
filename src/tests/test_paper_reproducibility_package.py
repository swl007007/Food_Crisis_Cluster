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
