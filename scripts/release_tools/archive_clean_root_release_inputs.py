"""Move clean-root release inputs into archive folders and write manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.release_paths import ReleasePaths

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


@dataclass(frozen=True)
class ArchiveGroupSpec:
    items: list[ArchiveItem]
    title: str
    description: str


REPRODUCIBILITY_ITEMS = [
    ArchiveItem(
        "GeoRFExperiment",
        PATHS.reproducibility_inputs_root,
        "stage1_stage2_georf",
        "verifier_package_dependency",
        "Stage 1/2 GeoRF handoff read by verifier and package builder",
    ),
    ArchiveItem(
        "GeoDTExperiment",
        PATHS.reproducibility_inputs_root,
        "stage1_stage2_geodt",
        "verifier_package_dependency",
        "Stage 1/2 GeoDT appendix handoff read by verifier and package builder",
    ),
    ArchiveItem(
        "main_ablation_exclude_updated_stage3_fixed_partitions",
        PATHS.reproducibility_inputs_root,
        "fixed_partition_ablation",
        "verifier_package_dependency",
        "Ablation manifests and outputs read by verifier and package builder",
    ),
    ArchiveItem(
        "fewsnet_baseline_results",
        PATHS.reproducibility_inputs_root,
        "fewsnet_baseline",
        "verifier_package_dependency",
        "Baseline source CSVs used by paper artifact audit context",
    ),
    ArchiveItem(
        "result_partition_k40_compare_GF_fs1",
        PATHS.reproducibility_inputs_root,
        "stage3_georf",
        "verifier_package_dependency",
        "Stage 3 GeoRF fs1 read by verifier and package builder",
    ),
    ArchiveItem(
        "result_partition_k40_compare_GF_fs2",
        PATHS.reproducibility_inputs_root,
        "stage3_georf",
        "verifier_package_dependency",
        "Stage 3 GeoRF fs2 read by verifier and package builder",
    ),
    ArchiveItem(
        "result_partition_k40_compare_GF_fs3",
        PATHS.reproducibility_inputs_root,
        "stage3_georf",
        "verifier_package_dependency",
        "Stage 3 GeoRF fs3 read by verifier and package builder",
    ),
    ArchiveItem(
        "result_partition_k40_compare_DT_fs1",
        PATHS.reproducibility_inputs_root,
        "stage3_geodt",
        "verifier_package_dependency",
        "Stage 3 GeoDT fs1 appendix provenance read by verifier and package builder",
    ),
    ArchiveItem(
        "result_partition_k40_compare_DT_fs2",
        PATHS.reproducibility_inputs_root,
        "stage3_geodt",
        "verifier_package_dependency",
        "Stage 3 GeoDT fs2 appendix provenance read by verifier and package builder",
    ),
    ArchiveItem(
        "result_partition_k40_compare_DT_fs3",
        PATHS.reproducibility_inputs_root,
        "stage3_geodt",
        "verifier_package_dependency",
        "Stage 3 GeoDT fs3 appendix provenance read by verifier and package builder",
    ),
    ArchiveItem(
        "result_partition_k40_compare_GF_thresholded_fs1",
        PATHS.reproducibility_inputs_root,
        "thresholded_georf",
        "verifier_dependency",
        "Thresholded GeoRF fs1 provider manifest audited by verifier",
    ),
    ArchiveItem(
        "result_partition_k40_compare_GF_thresholded_fs2",
        PATHS.reproducibility_inputs_root,
        "thresholded_georf",
        "verifier_dependency",
        "Thresholded GeoRF fs2 provider manifest audited by verifier",
    ),
    ArchiveItem(
        "result_partition_k40_compare_GF_thresholded_fs3",
        PATHS.reproducibility_inputs_root,
        "thresholded_georf",
        "verifier_dependency",
        "Thresholded GeoRF fs3 provider manifest audited by verifier",
    ),
]

LEGACY_WORKSPACE_ITEMS = [
    ArchiveItem(
        "other_outputs",
        PATHS.legacy_workspace_root,
        "legacy_workspace_output",
        "historical_workspace",
        (
            "Current release docs now point to final_artifacts_in_paper_updated "
            "and package copies"
        ),
    ),
    ArchiveItem(
        "scripts/config_visual.py",
        PATHS.legacy_workspace_root,
        "legacy_script",
        "historical_workspace",
        (
            "Root config_visual.py is the imported runtime config; "
            "scripts/config_visual.py has no release dependency"
        ),
    ),
    ArchiveItem(
        "scripts/check_country_columns.py",
        PATHS.legacy_workspace_root,
        "legacy_script",
        "historical_workspace",
        "No release verifier/package/paper_artifact/batch dependency found in scan",
    ),
    ArchiveItem(
        "scripts/check_shapefile.py",
        PATHS.legacy_workspace_root,
        "legacy_script",
        "historical_workspace",
        "No release verifier/package/paper_artifact/batch dependency found in scan",
    ),
    ArchiveItem(
        "scripts/diagnose_partition_lineage.py",
        PATHS.legacy_workspace_root,
        "legacy_script",
        "historical_workspace",
        (
            "Diagnostic helper not called by release verifier/package/paper "
            "artifacts or batch workflow"
        ),
    ),
    ArchiveItem(
        "scripts/enable_visual_debug.py",
        PATHS.legacy_workspace_root,
        "legacy_script",
        "historical_workspace",
        (
            "Interactive debug toggle not called by release verifier/package/paper "
            "artifacts or batch workflow"
        ),
    ),
    ArchiveItem(
        "scripts/plot_cluster_map.py",
        PATHS.legacy_workspace_root,
        "legacy_script",
        "historical_workspace",
        "Referenced as historical plotting approach only; paper artifacts use scripts/paper_artifacts",
    ),
    ArchiveItem(
        "scripts/plot_cluster_map_3x4.py",
        PATHS.legacy_workspace_root,
        "legacy_script",
        "historical_workspace",
        "Legacy plotting helper superseded by scripts/paper_artifacts",
    ),
    ArchiveItem(
        "scripts/plot_cluster_map_3x4_refined.py",
        PATHS.legacy_workspace_root,
        "legacy_script",
        "historical_workspace",
        "Legacy plotting helper superseded by scripts/paper_artifacts",
    ),
    ArchiveItem(
        "scripts/render_scope_images.py",
        PATHS.legacy_workspace_root,
        "legacy_script",
        "historical_workspace",
        "No release verifier/package/paper_artifact/batch dependency found in scan",
    ),
    ArchiveItem(
        "scripts/structure_change.py",
        PATHS.legacy_workspace_root,
        "legacy_script",
        "historical_workspace",
        "No release verifier/package/paper_artifact/batch dependency found in scan",
    ),
]

LOCAL_RESIDUE_ITEMS = [
    ArchiveItem(
        "writing",
        PATHS.local_residue_root,
        "local_residue",
        "not_release_dependency",
        "Manuscript drafting residue outside verifier/package path",
    ),
    ArchiveItem(
        "dt_rules",
        PATHS.local_residue_root,
        "local_residue",
        "not_release_dependency",
        "Generated rule exports not used as branch-specific paper explanations",
    ),
    ArchiveItem(
        "monthly_results",
        PATHS.local_residue_root,
        "local_residue",
        "not_release_dependency",
        "Empty or stale monthly output folder outside verifier/package path",
    ),
    ArchiveItem(
        "demo",
        PATHS.local_residue_root,
        "local_residue",
        "not_release_dependency",
        "Ignored demo residue outside release quickstart",
    ),
    ArchiveItem(
        "prediction_pipeline",
        PATHS.local_residue_root,
        "local_residue",
        "not_release_dependency",
        (
            "Forward-prediction residue already represented by archived non-paper "
            "pipeline provenance"
        ),
    ),
    ArchiveItem(
        "regional_ablation_results",
        PATHS.local_residue_root,
        "local_residue",
        "not_release_dependency",
        "Regional exploratory output outside paper verifier/package path",
    ),
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


def path_label(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def existing_archive_docs(archive_root: Path) -> list[Path]:
    return [
        path
        for path in (archive_root / "MANIFEST.csv", archive_root / "README.md")
        if path.exists()
    ]


def archive_collision_paths(items: list[ArchiveItem]) -> list[Path]:
    collisions = []
    for item in items:
        source = REPO_ROOT / item.old_path
        if source.exists() and item.archive_path.exists():
            collisions.append(item.archive_path)

    archive_roots = {item.archive_root for item in items}
    for archive_root in sorted(archive_roots):
        collisions.extend(existing_archive_docs(archive_root))
    return collisions


def raise_archive_collisions(collisions: list[Path]) -> None:
    collision_list = "\n".join(f"- {path_label(path)}" for path in collisions)
    raise FileExistsError(
        "Archive destination or documentation already exists:\n"
        f"{collision_list}\n"
        "Use --force to overwrite existing archive files."
    )


def preflight_archive_items(items: list[ArchiveItem], *, force: bool = False) -> None:
    if force:
        return

    collisions = archive_collision_paths(items)
    if collisions:
        raise_archive_collisions(collisions)


def preflight_archive_plan(
    groups: list[ArchiveGroupSpec],
    *,
    force: bool = False,
) -> None:
    if force:
        return

    collisions = []
    for group in groups:
        collisions.extend(archive_collision_paths(group.items))
    if collisions:
        raise_archive_collisions(collisions)


def remove_existing_destination(destination: Path) -> None:
    if destination.is_dir() and not destination.is_symlink():
        shutil.rmtree(destination)
    else:
        destination.unlink()


def move_item(
    item: ArchiveItem,
    *,
    execute: bool,
    force: bool = False,
) -> dict[str, str]:
    source = REPO_ROOT / item.old_path
    destination = item.archive_path
    status = tracked_status(item.old_path)

    if source.exists():
        size_bytes, file_count = size_and_count(source)
        checksum = small_file_hash(source, size_bytes)
    else:
        size_bytes = 0
        file_count = 0
        checksum = ""

    if execute and source.exists():
        if destination.exists():
            if not force:
                raise FileExistsError(
                    f"Archive destination already exists: {destination}"
                )
            remove_existing_destination(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
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


def write_archive_docs(
    archive_root: Path,
    rows: list[dict[str, str]],
    title: str,
    description: str,
    *,
    force: bool = False,
) -> None:
    existing_docs = existing_archive_docs(archive_root)
    if existing_docs and not force:
        collision_list = "\n".join(f"- {path_label(path)}" for path in existing_docs)
        raise FileExistsError(
            "Archive documentation already exists:\n"
            f"{collision_list}\n"
            "Use --force to overwrite existing archive documentation."
        )

    archive_root.mkdir(parents=True, exist_ok=True)
    with (archive_root / "MANIFEST.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=MANIFEST_FIELDS,
            lineterminator="\n",
        )
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
        (
            "See `MANIFEST.csv` for old paths, archive paths, dependency category, "
            "tracked status, size, file count, and small-file checksums."
        ),
        "",
    ]
    (archive_root / "README.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )


def archive_group(
    items: list[ArchiveItem],
    title: str,
    description: str,
    *,
    execute: bool,
    force: bool = False,
    preflight: bool = True,
) -> None:
    if preflight:
        preflight_archive_items(items, force=force)
    rows = [move_item(item, execute=execute, force=force) for item in items]
    write_archive_docs(items[0].archive_root, rows, title, description, force=force)


def archive_plan() -> list[ArchiveGroupSpec]:
    return [
        ArchiveGroupSpec(
            REPRODUCIBILITY_ITEMS,
            "Release 20260624 Reproducibility Inputs",
            (
                "Archived Stage 1/2, Stage 3, ablation, thresholded, and FEWS NET "
                "baseline inputs used by local verifier/package workflows."
            ),
        ),
        ArchiveGroupSpec(
            LEGACY_WORKSPACE_ITEMS,
            "Release 20260624 Legacy Workspace",
            (
                "Archived helper scripts and script-owned outputs outside the quick "
                "release validation path."
            ),
        ),
        ArchiveGroupSpec(
            LOCAL_RESIDUE_ITEMS,
            "Local Workspace Residue 20260624",
            "Archived local residue that is not a verifier or package dependency.",
        ),
    ]


def run_archive_plan(
    groups: list[ArchiveGroupSpec],
    *,
    execute: bool,
    force: bool = False,
) -> None:
    preflight_archive_plan(groups, force=force)
    for group in groups:
        archive_group(
            group.items,
            group.title,
            group.description,
            execute=execute,
            force=force,
            preflight=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Move files. Without this flag, only manifests are written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing archive destinations and README/MANIFEST files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_archive_plan(
        archive_plan(),
        execute=args.execute,
        force=args.force,
    )
    mode = "executed" if args.execute else "dry-run manifests written"
    print(f"Clean-root archive {mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
