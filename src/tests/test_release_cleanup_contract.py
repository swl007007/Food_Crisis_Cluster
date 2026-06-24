"""Contract tests for the paper release cleanup layout."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = REPO_ROOT / "archived" / "release_20260624_nonpaper_pipelines"
PAPER_ARTIFACT_ROOT = REPO_ROOT / "scripts" / "paper_artifacts"

PAPER_ARTIFACT_SCRIPTS = [
    "audit_final_artifact_sources.py",
    "analyze_georf_false_negative_error_modes.py",
    "analyze_georf_humanitarian_population_metrics.py",
    "analyze_georf_m2_cluster_profiles.py",
    "analyze_georf_partition_stability.py",
    "analyze_georf_probability_uncertainty.py",
    "analyze_georf_threshold_free_metrics.py",
    "build_feature_exclude_ablation_workbook.py",
    "build_georf_partitioned_shap_heatmap.py",
    "build_georf_thresholded_artifacts.py",
    "create_region_performance_partitioned_pooled_fewsnet.py",
    "paper_horizon_labels.py",
    "plot_error_rate_grids.py",
    "plot_fewsnet_crisis_stack_2018.py",
    "plot_geodt_branch_1_vs_011_locations.py",
    "plot_geodt_branch_tree_comparison.py",
    "plot_georf_m2_adjacency_refinement.py",
    "plot_georf_precision_recall_curves.py",
    "plot_global_cluster_map_2x2_refined.py",
    "plot_monthly_performance_metrics.py",
    "plot_predictions_2024.py",
    "plot_region_class_prevalence.py",
    "plot_seasonal_performance.py",
    "relabel_final_artifact_horizons.py",
]

ARCHIVED_OLD_PATHS = {
    "prediction_pipeline/run_partition_predict_unified.bat": "prediction_pipeline",
    "prediction_pipeline/run_predict_2026_2027.bat": "prediction_pipeline",
    "prediction_pipeline/run_scenario_predict_jun2026_feb2027.bat": "prediction_pipeline",
    "prediction_pipeline/spatial_weighted_consensus_clustering_predict.bat": "prediction_pipeline",
    "scripts/predict_partitioned_2026_2027.py": "scripts_prediction",
    "scripts/predict_scenario_2026_2027.py": "scripts_prediction",
    "app/main_model_XGB.py": "geoxgb_workflow",
    "scripts/compare_partitioned_vs_pooled_xgb_k40_nc4.py": "geoxgb_workflow",
    "scripts/rename_xgb_results.py": "geoxgb_workflow",
    "scripts/CH.ipynb": "notebooks_legacy",
    "scripts/baseline_comparison.ipynb": "notebooks_legacy",
    "scripts/baseline_comparison_2021.ipynb": "notebooks_legacy",
    "scripts/compare_unsplit_and_split_f1.ipynb": "notebooks_legacy",
    "scripts/examine_scope.ipynb": "notebooks_legacy",
    "scripts/genenerate_new_test.ipynb": "notebooks_legacy",
    "scripts/generate_test_data.ipynb": "notebooks_legacy",
    "scripts/seasonality.ipynb": "notebooks_legacy",
    "scripts/see_polygons.ipynb": "notebooks_legacy",
    "scripts/trial_phase_change.ipynb": "notebooks_legacy",
    "app/final/baseline_probit_regression.py": "legacy_misc",
    "app/final/fewsnet_baseline_evaluation.py": "legacy_misc",
    "app/final/georf_vs_baseline_comparison_plot.py": "legacy_misc",
    "demo/GeoRF_demo.py": "legacy_misc",
    "demo/data.py": "legacy_misc",
    "scripts/generate_actual_predicted_dashboard.py": "legacy_misc",
    "scripts/plot_phase_change_monthly_performance.py": "legacy_misc",
    "scripts/plot_f1_improvement_comparison.py": "legacy_misc",
    "scripts/plot_f1_improvement_comparison_by_country.py": "legacy_misc",
    "scripts/plot_score_details_maps.py": "legacy_misc",
    "scripts/run_georf_2024.sh": "legacy_misc",
    "regional_ablation_results/Nigeria_experiments_global/filter_nigeria_and_aggregate.py": "legacy_misc",
    "regional_ablation_results/actual_predicted_dashboard/actual_predicted_dashboard_Nigeria.html": "legacy_misc",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_old_paths(items) -> set[str]:
    return {item.old_path for item in items}


def assert_path_is_under(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise AssertionError(f"{path} is not under {root}") from exc


def load_archive_manifest() -> dict[str, dict[str, str]]:
    manifest = ARCHIVE_ROOT / "MANIFEST.csv"
    assert manifest.is_file(), f"Missing archive manifest: {manifest}"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row["old_path"]: row for row in rows}


def release_quickstart_text() -> str:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("## Quick Start for Paper Reproduction")
    end = readme.index("## Complete No-Leak Regeneration Path")
    return readme[start:end]


def test_paper_artifact_scripts_are_moved_under_paper_artifacts() -> None:
    assert (PAPER_ARTIFACT_ROOT / "__init__.py").is_file()
    for script in PAPER_ARTIFACT_SCRIPTS:
        assert (PAPER_ARTIFACT_ROOT / script).is_file(), script
        assert not (REPO_ROOT / "scripts" / script).exists(), script


def test_nonpaper_archive_manifest_contains_expected_moves() -> None:
    rows = load_archive_manifest()
    assert set(rows) >= set(ARCHIVED_OLD_PATHS)
    for old_path, expected_category in ARCHIVED_OLD_PATHS.items():
        row = rows[old_path]
        archive_path_value = Path(row["archive_path"])
        assert not archive_path_value.is_absolute(), row
        archive_path = REPO_ROOT / archive_path_value
        assert_path_is_under(archive_path, ARCHIVE_ROOT)
        assert archive_path.is_file(), row
        assert row["category"] == expected_category, row
        assert row["reason"].strip(), row
        assert int(row["size_bytes"]) == archive_path.stat().st_size
        assert row["sha256"] == file_sha256(archive_path)
        assert not (REPO_ROOT / old_path).exists(), old_path


def test_release_quickstart_is_paper_only() -> None:
    quickstart = release_quickstart_text()
    forbidden = [
        "GeoXGB",
        "fs0",
        "--fs0-only",
        "prediction_pipeline",
        "predict_partitioned_2026_2027.py",
        "predict_scenario_2026_2027.py",
        "spatial_weighted_consensus_clustering_predict",
        "run_partition_predict_unified",
        "run_predict_2026_2027",
        "run_scenario_predict_jun2026_feb2027",
        "2026-2027",
    ]
    for token in forbidden:
        assert token not in quickstart
    assert "validate_paper_reproducibility_package.py" in quickstart
    assert "verify_current_results_reproducibility.py" in quickstart


def test_release_manifest_names_archive_and_tag() -> None:
    manifest = REPO_ROOT / "RELEASE_MANIFEST.md"
    assert manifest.is_file(), f"Missing release manifest: {manifest}"
    text = manifest.read_text(encoding="utf-8")
    assert "v1.0-paper-reproducibility-20260624" in text
    assert "archived/release_20260624_nonpaper_pipelines/" in text
    assert "GeoRF" in text
    assert "GeoDT appendix" in text


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
    from scripts.release_tools.archive_clean_root_release_inputs import (
        REPRODUCIBILITY_ITEMS,
    )

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
    from scripts.release_tools.archive_clean_root_release_inputs import (
        LEGACY_WORKSPACE_ITEMS,
    )

    old_paths = archive_old_paths(LEGACY_WORKSPACE_ITEMS)
    assert "other_outputs" in old_paths
    assert "scripts/config_visual.py" in old_paths
    for item in LEGACY_WORKSPACE_ITEMS:
        assert item.dependency_scan_reason.strip(), item


def test_archive_move_item_dry_run_keeps_source_and_reports_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.release_tools import archive_clean_root_release_inputs as archive

    source = tmp_path / "source.txt"
    source.write_text("source payload", encoding="utf-8")
    item = archive.ArchiveItem(
        "source.txt",
        tmp_path / "archive",
        "test_category",
        "test_dependency",
        "test reason",
    )
    monkeypatch.setattr(archive, "REPO_ROOT", tmp_path)

    row = archive.move_item(item, execute=False)

    assert source.is_file()
    assert not item.archive_path.exists()
    assert row["old_path"] == "source.txt"
    assert row["archive_path"] == "archive/source.txt"
    assert row["size_bytes"] == str(source.stat().st_size)
    assert row["file_count"] == "1"
    assert row["sha256"] == file_sha256(source)


def test_archive_group_preflights_destination_collisions_before_moving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.release_tools import archive_clean_root_release_inputs as archive

    first_source = tmp_path / "first.txt"
    second_source = tmp_path / "second.txt"
    archive_root = tmp_path / "archive"
    first_source.write_text("first", encoding="utf-8")
    second_source.write_text("second", encoding="utf-8")
    archive_root.mkdir()
    (archive_root / "second.txt").write_text("existing destination", encoding="utf-8")

    items = [
        archive.ArchiveItem(
            "first.txt",
            archive_root,
            "test_category",
            "test_dependency",
            "test reason",
        ),
        archive.ArchiveItem(
            "second.txt",
            archive_root,
            "test_category",
            "test_dependency",
            "test reason",
        ),
    ]
    monkeypatch.setattr(archive, "REPO_ROOT", tmp_path)

    with pytest.raises(FileExistsError, match="second.txt"):
        archive.archive_group(
            items,
            "Temp Archive",
            "Temporary archive test.",
            execute=True,
        )

    assert first_source.is_file()
    assert second_source.is_file()
    assert not (archive_root / "first.txt").exists()
    assert (archive_root / "second.txt").read_text(encoding="utf-8") == (
        "existing destination"
    )
    assert not (archive_root / "MANIFEST.csv").exists()


def test_write_archive_docs_refuses_existing_docs_without_force(tmp_path: Path) -> None:
    from scripts.release_tools import archive_clean_root_release_inputs as archive

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "MANIFEST.csv").write_text("existing manifest", encoding="utf-8")
    (archive_root / "README.md").write_text("existing readme", encoding="utf-8")

    with pytest.raises(FileExistsError, match="MANIFEST.csv"):
        archive.write_archive_docs(
            archive_root,
            [],
            "Temp Archive",
            "Temporary archive test.",
            force=False,
        )

    assert (archive_root / "MANIFEST.csv").read_text(encoding="utf-8") == (
        "existing manifest"
    )
    assert (archive_root / "README.md").read_text(encoding="utf-8") == (
        "existing readme"
    )


def test_write_archive_docs_force_allows_overwriting_existing_docs(
    tmp_path: Path,
) -> None:
    from scripts.release_tools import archive_clean_root_release_inputs as archive

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "MANIFEST.csv").write_text("existing manifest", encoding="utf-8")
    (archive_root / "README.md").write_text("existing readme", encoding="utf-8")

    archive.write_archive_docs(
        archive_root,
        [],
        "Temp Archive",
        "Temporary archive test.",
        force=True,
    )

    assert "old_path,archive_path,category" in (
        archive_root / "MANIFEST.csv"
    ).read_text(encoding="utf-8")
    assert "# Temp Archive" in (archive_root / "README.md").read_text(
        encoding="utf-8"
    )


def test_main_preflights_later_group_collision_before_moving_first_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.release_tools import archive_clean_root_release_inputs as archive

    first_source = tmp_path / "first_group.txt"
    later_source = tmp_path / "later_group.txt"
    first_archive_root = tmp_path / "archive_first"
    later_archive_root = tmp_path / "archive_later"
    residue_archive_root = tmp_path / "archive_residue"
    first_source.write_text("first group payload", encoding="utf-8")
    later_source.write_text("later group payload", encoding="utf-8")
    later_archive_root.mkdir()
    (later_archive_root / "later_group.txt").write_text(
        "existing later archive payload",
        encoding="utf-8",
    )

    first_items = [
        archive.ArchiveItem(
            "first_group.txt",
            first_archive_root,
            "test_category",
            "test_dependency",
            "test reason",
        )
    ]
    later_items = [
        archive.ArchiveItem(
            "later_group.txt",
            later_archive_root,
            "test_category",
            "test_dependency",
            "test reason",
        )
    ]
    residue_items = [
        archive.ArchiveItem(
            "missing_residue.txt",
            residue_archive_root,
            "test_category",
            "test_dependency",
            "test reason",
        )
    ]
    monkeypatch.setattr(archive, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(archive, "REPRODUCIBILITY_ITEMS", first_items)
    monkeypatch.setattr(archive, "LEGACY_WORKSPACE_ITEMS", later_items)
    monkeypatch.setattr(archive, "LOCAL_RESIDUE_ITEMS", residue_items)
    monkeypatch.setattr(
        archive,
        "parse_args",
        lambda: SimpleNamespace(execute=True, force=False),
    )

    with pytest.raises(FileExistsError, match="later_group.txt"):
        archive.main()

    assert first_source.is_file()
    assert later_source.is_file()
    assert not (first_archive_root / "first_group.txt").exists()
    assert not (first_archive_root / "MANIFEST.csv").exists()
