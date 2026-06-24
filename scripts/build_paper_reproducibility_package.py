"""Build the lightweight paper reproducibility package.

The package is intentionally generated from existing artifacts. It does not
rerun models, does not copy raw source data, and does not move experimental
workflows.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.release_paths import ReleasePaths

PATHS = ReleasePaths(repo_root=REPO_ROOT)
PACKAGE_ROOT = PATHS.package_root


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
            lineterminator="\n",
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
    content = "".join(
        f"{row.sha256}  {row.package_path}\n" for row in sorted(rows, key=lambda item: item.package_path)
    )
    write_text(package_root / "SHA256SUMS.txt", content)


def remove_tree_posix(path: Path) -> None:
    """Remove a package tree on mounted filesystems where Python rmdir can fail."""
    file_result = subprocess.run(
        ["find", str(path), "(", "-type", "f", "-o", "-type", "l", ")", "-delete"],
        check=False,
        capture_output=True,
        text=True,
    )
    if file_result.returncode != 0:
        detail = file_result.stderr.strip() or "unknown error"
        raise PackageBuildError(f"Failed to reset package root: {detail}")

    detail = "unknown error"
    for _ in range(10):
        dir_result = subprocess.run(
            ["find", str(path), "-mindepth", "1", "-depth", "-type", "d", "-exec", "rmdir", "{}", ";"],
            check=False,
            capture_output=True,
            text=True,
        )
        if dir_result.returncode != 0:
            detail = dir_result.stderr.strip() or detail

        root_result = subprocess.run(["rmdir", str(path)], check=False, capture_output=True, text=True)
        if root_result.returncode == 0 or not path.exists():
            return
        detail = root_result.stderr.strip() or detail
        time.sleep(0.1)

    raise PackageBuildError(f"Failed to reset package root: {detail}")


def reset_package_root(package_root: Path) -> None:
    resolved = package_root.resolve()
    expected = (REPO_ROOT / "paper_reproducibility_package").resolve()
    if resolved != expected:
        raise PackageBuildError(f"Refusing to reset unexpected package root: {package_root}")
    if package_root.exists():
        if os.name == "posix":
            remove_tree_posix(package_root)
        else:
            shutil.rmtree(package_root)
    package_root.mkdir(parents=True)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def add_generated_doc(
    rows: list[ManifestRow],
    package_root: Path,
    package_path: str,
    role: str,
    content: str,
    notes: str,
) -> None:
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
            PATHS.experiment_root("GF") / "knn_sparsification_results" / "cluster_mapping_manifest.json",
            "stage2_cluster_maps/georf/cluster_mapping_manifest.json",
            "GeoRF Stage 2 manifest",
        ),
        (
            PATHS.experiment_root("GF") / "knn_sparsification_results" / "cluster_mapping_k40_nc17_general.csv",
            "stage2_cluster_maps/georf/cluster_mapping_k40_nc17_general.csv",
            "GeoRF general consensus map",
        ),
        (
            PATHS.experiment_root("GF") / "knn_sparsification_results" / "cluster_mapping_k40_nc13_m2.csv",
            "stage2_cluster_maps/georf/cluster_mapping_k40_nc13_m2.csv",
            "GeoRF February consensus map",
        ),
        (
            PATHS.experiment_root("GF") / "knn_sparsification_results" / "cluster_mapping_k40_nc11_m6.csv",
            "stage2_cluster_maps/georf/cluster_mapping_k40_nc11_m6.csv",
            "GeoRF June consensus map",
        ),
        (
            PATHS.experiment_root("GF") / "knn_sparsification_results" / "cluster_mapping_k40_nc16_m10.csv",
            "stage2_cluster_maps/georf/cluster_mapping_k40_nc16_m10.csv",
            "GeoRF October consensus map",
        ),
        (
            PATHS.experiment_root("DT") / "knn_sparsification_results" / "cluster_mapping_manifest.json",
            "stage2_cluster_maps/geodt/cluster_mapping_manifest.json",
            "GeoDT Stage 2 manifest",
        ),
        (
            PATHS.experiment_root("DT") / "knn_sparsification_results" / "cluster_mapping_k40_nc15_general.csv",
            "stage2_cluster_maps/geodt/cluster_mapping_k40_nc15_general.csv",
            "GeoDT general consensus map",
        ),
        (
            PATHS.experiment_root("DT") / "knn_sparsification_results" / "cluster_mapping_k40_nc16_m2.csv",
            "stage2_cluster_maps/geodt/cluster_mapping_k40_nc16_m2.csv",
            "GeoDT February consensus map",
        ),
        (
            PATHS.experiment_root("DT") / "knn_sparsification_results" / "cluster_mapping_k40_nc15_m6.csv",
            "stage2_cluster_maps/geodt/cluster_mapping_k40_nc15_m6.csv",
            "GeoDT June consensus map",
        ),
        (
            PATHS.experiment_root("DT") / "knn_sparsification_results" / "cluster_mapping_k40_nc18_m10.csv",
            "stage2_cluster_maps/geodt/cluster_mapping_k40_nc18_m10.csv",
            "GeoDT October consensus map",
        ),
    ]


def stage3_result_dirs() -> list[tuple[str, Path]]:
    return [
        ("georf_fs1", PATHS.stage3_root("GF", 1)),
        ("georf_fs2", PATHS.stage3_root("GF", 2)),
        ("georf_fs3", PATHS.stage3_root("GF", 3)),
        ("geodt_fs1", PATHS.stage3_root("DT", 1)),
        ("geodt_fs2", PATHS.stage3_root("DT", 2)),
        ("geodt_fs3", PATHS.stage3_root("DT", 3)),
    ]


def canonical_refined_files(source_dir: Path, manifest: dict) -> list[Path]:
    refined_dir = source_dir / "refined"
    if not refined_dir.is_dir():
        raise PackageBuildError(f"Missing refined directory: {refined_dir}")

    selected: dict[str, Path] = {}
    partition_path = manifest.get("partition_map_path")
    if partition_path:
        normalized = Path(str(partition_path).replace("\\", "/"))
        source = PATHS.resolve_repo_reference(normalized)
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
                    f"canonical refined map or refinement summary from {source_dir.name}",
                )
            )


def add_final_artifacts(rows: list[ManifestRow], package_root: Path) -> None:
    source_root = PATHS.final_artifacts_root
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
            PATHS.ablation_root / "ablation_run_manifest.json",
            "ablation/ablation_run_manifest.json",
            "fixed-partition ablation run manifest",
        ),
        (
            PATHS.ablation_root / "input_datasets" / "feature_exclude_dataset_manifest.json",
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
- Experimental and legacy entry points are archived under
  `archived/release_20260624_nonpaper_pipelines/` and historical release
  workspaces are archived under `archived/release_20260624_legacy_workspace/` in
  the source repository.

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

This checks the current archived Stage 2/Stage 3/final-artifact verifier bundle
outside this copied package.
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

GeoXGB, fs0 lag-1 launch guidance, and 2026-2027 forward/scenario prediction
are not part of this package. Their entry points are preserved as historical
provenance under `archived/release_20260624_nonpaper_pipelines/`; local result
workspaces used to rebuild this package are under
`archived/release_20260624_reproducibility_inputs/`.
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
