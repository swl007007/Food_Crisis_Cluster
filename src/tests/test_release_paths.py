import os
from pathlib import Path

import pytest

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


def test_lowercase_model_tokens_are_accepted() -> None:
    paths = ReleasePaths(repo_root=Path("/repo"))

    assert paths.experiment_root("gf") == Path(
        "/repo/archived/release_20260624_reproducibility_inputs/GeoRFExperiment"
    )
    assert paths.experiment_root("dt") == Path(
        "/repo/archived/release_20260624_reproducibility_inputs/GeoDTExperiment"
    )
    assert paths.stage3_root("gf", 1) == Path(
        "/repo/archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_GF_fs1"
    )
    assert paths.stage3_root("dt", 2) == Path(
        "/repo/archived/release_20260624_reproducibility_inputs/result_partition_k40_compare_DT_fs2"
    )


def test_invalid_experiment_token_raises_value_error() -> None:
    paths = ReleasePaths(repo_root=Path("/repo"))

    with pytest.raises(ValueError, match="Unknown experiment token: XGB"):
        paths.experiment_root("XGB")


def test_invalid_stage3_token_raises_value_error() -> None:
    paths = ReleasePaths(repo_root=Path("/repo"))

    with pytest.raises(ValueError, match="Unknown Stage 3 token: XGB"):
        paths.stage3_root("XGB", 1)


def test_invalid_stage3_scope_raises_value_error() -> None:
    paths = ReleasePaths(repo_root=Path("/repo"))

    with pytest.raises(ValueError, match="Stage 3 scope must be 1, 2, or 3, got 0"):
        paths.stage3_root("GF", 0)


def test_invalid_thresholded_georf_scope_raises_value_error() -> None:
    paths = ReleasePaths(repo_root=Path("/repo"))

    with pytest.raises(
        ValueError, match="Thresholded GeoRF scope must be 1, 2, or 3, got 4"
    ):
        paths.thresholded_georf_root(4)


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

    if os.name == "nt":
        expected = Path(
            "C:/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/1.Source Data/FEWSNET_forecast_unadjusted_bm.csv"
        )
    else:
        expected = Path(
            "/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/1.Source Data/FEWSNET_forecast_unadjusted_bm.csv"
        )

    assert resolved == expected


def test_resolve_repo_reference_returns_root_candidate_for_unknown_relative_path() -> None:
    paths = ReleasePaths(repo_root=Path("/repo"))

    assert paths.resolve_repo_reference("docs/example.md") == Path("/repo/docs/example.md")


def test_require_file_returns_existing_file(tmp_path: Path) -> None:
    existing_file = tmp_path / "run_manifest.json"
    existing_file.write_text("{}", encoding="utf-8")
    paths = ReleasePaths(repo_root=tmp_path)

    assert paths.require_file(existing_file, "manifest") == existing_file


def test_require_file_raises_for_missing_file(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing.json"
    paths = ReleasePaths(repo_root=tmp_path)

    with pytest.raises(FileNotFoundError, match="Missing manifest:"):
        paths.require_file(missing_file, "manifest")


def test_require_dir_returns_existing_directory(tmp_path: Path) -> None:
    existing_dir = tmp_path / "GeoRFExperiment"
    existing_dir.mkdir()
    paths = ReleasePaths(repo_root=tmp_path)

    assert paths.require_dir(existing_dir, "experiment") == existing_dir


def test_require_dir_raises_for_missing_directory(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"
    paths = ReleasePaths(repo_root=tmp_path)

    with pytest.raises(FileNotFoundError, match="Missing experiment:"):
        paths.require_dir(missing_dir, "experiment")
