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
        if manifest_checksum is None and package_path != "MANIFEST.csv":
            failures.append(f"{package_path}: missing from MANIFEST.csv")
            continue
        if sums_checksum is None:
            failures.append(f"{package_path}: missing from SHA256SUMS.txt")
            continue

        expected_checksum = manifest_checksum or sums_checksum
        if manifest_checksum is not None and manifest_checksum != sums_checksum:
            failures.append(f"{package_path}: manifest/checksum file disagree")
            continue

        file_path = package_root / package_path
        if not file_path.is_file():
            failures.append(f"{package_path}: listed file missing")
            continue
        actual = sha256_file(file_path)
        if actual != expected_checksum:
            failures.append(f"{package_path}: checksum mismatch expected {expected_checksum} got {actual}")

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
