from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

try:
    import pytest
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("pytest is required for GeoDT diagnostic tests") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_SCRIPT = REPO_ROOT / "scripts" / "plot_geodt_branch_tree_comparison.py"


def _fit_tree(offset: float = 0.0) -> DecisionTreeClassifier:
    x = np.array(
        [
            [0.0 + offset, 0.0, 0.0],
            [0.1 + offset, 1.0, 0.0],
            [1.0 + offset, 0.0, 1.0],
            [1.1 + offset, 1.0, 1.0],
            [0.2 + offset, 0.2, 0.0],
            [1.2 + offset, 0.8, 1.0],
        ]
    )
    y = np.array([0, 0, 1, 1, 0, 1])
    clf = DecisionTreeClassifier(max_depth=2, random_state=7)
    clf.fit(x, y)
    return clf


def _write_checkpoint(path: Path, clf: DecisionTreeClassifier) -> None:
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(clf, handle)


def _write_feature_names(path: Path, names: list[str]) -> None:
    path.write_text("\n".join(names) + "\n", encoding="utf-8")


def build_complete_geodt_archive(
    root: Path, name: str = "result_GeoDT_2024_fs1_2024-10"
) -> Path:
    archive = root / name
    checkpoints = archive / "checkpoints"
    space = archive / "space_partitions"
    archive.mkdir(parents=True)
    checkpoints.mkdir()
    space.mkdir()

    _write_checkpoint(checkpoints / "dt_", _fit_tree(0.0))
    _write_checkpoint(checkpoints / "dt_0", _fit_tree(0.0))
    _write_checkpoint(checkpoints / "dt_1", _fit_tree(0.5))

    s_branch = pd.DataFrame({"0": [101, 102, -1], "1": [201, 202, -1]})
    s_branch.to_pickle(space / "s_branch.pkl")
    np.save(space / "X_branch_id.npy", np.array(["0", "0", "1", "1"], dtype="U2"))
    np.save(space / "branch_table.npy", np.array([[0, 1]], dtype=object))

    pd.DataFrame(
        {
            "FEWSNET_admin_code": ["101", "102", "201", "202"],
            "partition_id": ["0", "0", "1", "1"],
            "branch_id": ["0", "0", "1", "1"],
        }
    ).to_csv(archive / "correspondence_table_2024-10_fs1.csv", index=False)
    _write_feature_names(
        archive / "feature_names.txt", ["rainfall", "price", "conflict"]
    )
    return archive


@pytest.fixture
def complete_geodt_archive(tmp_path: Path) -> Path:
    return build_complete_geodt_archive(tmp_path)


@pytest.fixture
def visual_archive_with_provider(tmp_path: Path) -> tuple[Path, Path]:
    visual = tmp_path / "result_GeoDT_2024_fs1_2024-10_visual"
    provider = build_complete_geodt_archive(tmp_path, "result_GeoDT_2024_fs1_2024-10")
    visual.mkdir()
    (visual / "vis").mkdir()
    pd.DataFrame(
        {
            "FEWSNET_admin_code": ["101", "102", "201", "202"],
            "partition_id": ["0", "0", "1", "1"],
            "branch_id": ["0", "0", "1", "1"],
        }
    ).to_csv(visual / "correspondence_table_2024-10_fs1.csv", index=False)
    return visual, provider


@pytest.fixture
def negative_archives(tmp_path: Path) -> dict[str, Path]:
    root_only = tmp_path / "result_GeoDT_2024_fs1_2024-10_visual"
    (root_only / "dt_rules").mkdir(parents=True)
    (root_only / "dt_rules" / "dt_rules_2024_fs1_2024-10.csv").write_text(
        "Rule,Class,Confidence,Samples\nROOT,class_0,1.0,10\n",
        encoding="utf-8",
    )

    mismatch = build_complete_geodt_archive(tmp_path, "result_GeoDT_2024_fs1_2024-11")
    _write_feature_names(mismatch / "feature_names.txt", ["only_one_feature"])

    out_of_bounds = build_complete_geodt_archive(
        tmp_path, "result_GeoDT_2024_fs1_2024-12"
    )
    _write_feature_names(out_of_bounds / "feature_names.txt", ["rainfall"])

    return {
        "root_only": root_only,
        "feature_mismatch": mismatch,
        "out_of_bounds": out_of_bounds,
    }


def _run_diagnostic(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(DIAGNOSTIC_SCRIPT), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _load_diagnostic_module():
    spec = importlib.util.spec_from_file_location(
        "geodt_branch_tree_diagnostic", DIAGNOSTIC_SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_builder_creates_cli_usable_complete_archive(
    complete_geodt_archive: Path,
) -> None:
    assert (complete_geodt_archive / "checkpoints" / "dt_0").is_file()
    assert (complete_geodt_archive / "checkpoints" / "dt_1").is_file()
    assert (complete_geodt_archive / "space_partitions" / "s_branch.pkl").is_file()
    assert (complete_geodt_archive / "space_partitions" / "X_branch_id.npy").is_file()
    assert (complete_geodt_archive / "feature_names.txt").is_file()


def test_visual_archive_provider_fixture_records_separate_paths(
    visual_archive_with_provider: tuple[Path, Path],
) -> None:
    visual, provider = visual_archive_with_provider
    assert visual != provider
    assert not (visual / "checkpoints").exists()
    assert (provider / "checkpoints" / "dt_0").is_file()


def test_negative_fixture_set_covers_root_only_mismatch_and_out_of_bounds(
    negative_archives: dict[str, Path],
) -> None:
    assert (negative_archives["root_only"] / "dt_rules").is_dir()
    assert (negative_archives["feature_mismatch"] / "feature_names.txt").read_text(
        encoding="utf-8"
    ).strip() == "only_one_feature"
    assert (negative_archives["out_of_bounds"] / "feature_names.txt").read_text(
        encoding="utf-8"
    ).strip() == "rainfall"


def test_root_global_only_dt_rules_archive_is_incomplete_and_not_used_as_branch_source(
    negative_archives: dict[str, Path], tmp_path: Path
) -> None:
    result = _run_diagnostic(
        "--audit-only",
        "--archive-path",
        str(negative_archives["root_only"]),
        "--output-dir",
        str(tmp_path / "diagnostics"),
    )

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "dt_rules" in combined
    assert "branch" in combined.lower()
    assert "incomplete" in combined.lower() or "root/global" in combined.lower()


def test_no_overwrite_regression_for_existing_production_artifact_paths(
    complete_geodt_archive: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "diagnostics"
    output_dir.mkdir()
    protected = (
        output_dir / "geodt_branch_tree_compare_2024-10_fs1_0_vs_1_metadata.json"
    )
    protected.write_text(json.dumps({"existing": True}), encoding="utf-8")

    result = _run_diagnostic(
        "--archive-path",
        str(complete_geodt_archive),
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode != 0
    assert json.loads(protected.read_text(encoding="utf-8")) == {"existing": True}
    failure_files = list(
        output_dir.glob(
            "geodt_branch_tree_compare_2024-10_fs1_*_vs_*_failure_summary.json"
        )
    )
    assert failure_files
    failure_summary = json.loads(failure_files[0].read_text(encoding="utf-8"))
    assert failure_summary["failure_stage"] == "output-conflict"
    assert str(protected) in failure_summary["conflicting_output_paths"]
    combined = result.stdout + result.stderr
    assert "overwrite" in combined.lower() or "exists" in combined.lower()


def test_bounded_archive_discovery_rejects_unbounded_or_unrelated_search(
    tmp_path: Path,
) -> None:
    result = _run_diagnostic(
        "--audit-only", "--archive-root", str(tmp_path / "missing-root")
    )

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "archive" in combined.lower()
    assert "root" in combined.lower() or "bounded" in combined.lower()


def test_same_run_artifact_provider_records_selected_archive_and_provider_paths(
    visual_archive_with_provider: tuple[Path, Path], tmp_path: Path
) -> None:
    visual, provider = visual_archive_with_provider
    output_dir = tmp_path / "diagnostics"

    result = _run_diagnostic(
        "--audit-only",
        "--archive-path",
        str(visual),
        "--artifact-provider-path",
        str(provider),
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0
    metadata_files = list(output_dir.glob("*_metadata.json")) + list(
        output_dir.glob("*_audit.json")
    )
    assert metadata_files
    combined_metadata = "\n".join(
        path.read_text(encoding="utf-8") for path in metadata_files
    )
    assert str(visual) in combined_metadata
    assert str(provider) in combined_metadata


def test_default_figure_cli_accepts_contract_arguments_without_training_config_mutation(
    complete_geodt_archive: Path, tmp_path: Path
) -> None:
    import config

    before = {
        "SAVE_DT_RULES": config.SAVE_DT_RULES,
        "SAVE_DT_NODE_DUMP": config.SAVE_DT_NODE_DUMP,
        "ACTIVE_LAGS": tuple(config.ACTIVE_LAGS),
    }

    result = _run_diagnostic(
        "--archive-path",
        str(complete_geodt_archive),
        "--output-dir",
        str(tmp_path / "diagnostics"),
        "--k",
        "3",
        "--max-plot-depth",
        "3",
        "--png-only",
    )

    assert result.returncode == 0
    assert config.SAVE_DT_RULES == before["SAVE_DT_RULES"]
    assert config.SAVE_DT_NODE_DUMP == before["SAVE_DT_NODE_DUMP"]
    assert tuple(config.ACTIVE_LAGS) == before["ACTIVE_LAGS"]


def test_preflight_metadata_records_archive_fallback_and_minimum_completeness(
    tmp_path: Path,
) -> None:
    incomplete = tmp_path / "result_GeoDT_2024_fs1_2024-09_visual"
    incomplete.mkdir()
    complete = build_complete_geodt_archive(tmp_path, "result_GeoDT_2024_fs1_2024-10")
    output_dir = tmp_path / "diagnostics"

    result = _run_diagnostic(
        "--audit-only",
        "--archive-list",
        f"{incomplete},{complete}",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0
    metadata_files = list(output_dir.glob("*_metadata.json")) + list(
        output_dir.glob("*_audit.json")
    )
    assert metadata_files
    metadata_text = "\n".join(
        path.read_text(encoding="utf-8") for path in metadata_files
    )
    assert "candidate" in metadata_text.lower()
    assert "minimum" in metadata_text.lower()
    assert str(incomplete) in metadata_text
    assert str(complete) in metadata_text


def test_figure_metadata_records_all_archive_list_candidate_decisions(
    tmp_path: Path,
) -> None:
    incomplete = tmp_path / "result_GeoDT_2024_fs1_2024-09_visual"
    incomplete.mkdir()
    complete = build_complete_geodt_archive(tmp_path, "result_GeoDT_2024_fs1_2024-10")
    output_dir = tmp_path / "diagnostics"

    result = _run_diagnostic(
        "--archive-list",
        f"{incomplete},{complete}",
        "--output-dir",
        str(output_dir),
        "--k",
        "3",
        "--max-plot-depth",
        "3",
    )

    assert result.returncode == 0
    metadata_files = list(output_dir.glob("*_metadata.json"))
    assert metadata_files
    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    decisions = metadata["candidate_archive_decisions"]
    assert metadata["discovered_archive_candidate_count"] == 2
    assert [decision["path"] for decision in decisions] == [
        str(incomplete),
        str(complete),
    ]
    assert decisions[0]["decision"] == "rejected"
    assert decisions[0]["minimum_required_artifact_completeness"] is False
    assert decisions[0]["missing_or_invalid_artifacts"]
    assert decisions[1]["decision"] == "accepted"
    assert decisions[1]["minimum_required_artifact_completeness"] is True


def test_figure_generation_writes_structured_failure_when_no_archive_is_complete(
    negative_archives: dict[str, Path], tmp_path: Path
) -> None:
    output_dir = tmp_path / "diagnostics"

    result = _run_diagnostic(
        "--archive-list",
        str(negative_archives["root_only"]),
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "failure_stage" in combined
    assert "candidate_archive_decisions" in combined
    assert "recommended_next_action" in combined
    failure_summary_files = list(output_dir.glob("*_failure_summary.json"))
    assert failure_summary_files
    failure_summary = json.loads(failure_summary_files[0].read_text(encoding="utf-8"))
    assert failure_summary["failure_stage"] == "preflight"
    assert failure_summary["missing_or_invalid_artifacts"]
    assert failure_summary["candidate_archive_decisions"]
    assert failure_summary["recommended_next_action"]


def test_checkpoint_classification_and_branch_eligibility_exclude_root_global(
    complete_geodt_archive: Path, tmp_path: Path
) -> None:
    result = _run_diagnostic(
        "--audit-only",
        "--archive-path",
        str(complete_geodt_archive),
        "--output-dir",
        str(tmp_path / "diagnostics"),
    )

    assert result.returncode == 0
    combined = result.stdout + result.stderr
    assert "dt_0" in combined
    assert "dt_1" in combined
    assert "dt_" in combined
    assert "root/global" in combined.lower() or "root" in combined.lower()
    assert "eligible" in combined.lower()
    assert "assigned" in combined.lower()


def test_feature_name_source_precedence_and_split_index_bounds_are_reported(
    negative_archives: dict[str, Path], tmp_path: Path
) -> None:
    result = _run_diagnostic(
        "--audit-only",
        "--archive-path",
        str(negative_archives["feature_mismatch"]),
        "--output-dir",
        str(tmp_path / "diagnostics"),
    )

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "feature" in combined.lower()
    assert "count" in combined.lower() or "bounds" in combined.lower()
    assert "unusable" in combined.lower() or "mismatch" in combined.lower()


def test_feature_name_reference_csv_is_accepted_as_selected_run_feature_source(
    complete_geodt_archive: Path, tmp_path: Path
) -> None:
    feature_txt = complete_geodt_archive / "feature_names.txt"
    feature_names = [
        line.strip()
        for line in feature_txt.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    feature_txt.unlink()
    pd.DataFrame(
        {
            "feature_index": list(range(len(feature_names))),
            "feature_name": feature_names,
        }
    ).to_csv(complete_geodt_archive / "feature_name_reference.csv", index=False)

    result = _run_diagnostic(
        "--audit-only",
        "--archive-path",
        str(complete_geodt_archive),
        "--output-dir",
        str(tmp_path / "diagnostics"),
    )

    assert result.returncode == 0
    combined = result.stdout + result.stderr
    assert "feature_name_reference.csv" in combined
    assert "feature_names.txt" not in combined


def test_figure_generation_writes_readable_png_metadata_and_neutral_labels(
    complete_geodt_archive: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "diagnostics"

    result = _run_diagnostic(
        "--archive-path",
        str(complete_geodt_archive),
        "--output-dir",
        str(output_dir),
        "--k",
        "3",
        "--max-plot-depth",
        "3",
    )

    assert result.returncode == 0
    png_files = list(
        output_dir.glob("geodt_branch_tree_compare_2024-10_fs1_*_vs_*.png")
    )
    metadata_files = list(
        output_dir.glob("geodt_branch_tree_compare_2024-10_fs1_*_vs_*_metadata.json")
    )
    assert png_files
    assert metadata_files
    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    metadata_text = json.dumps(metadata)
    assert "branch-specific" in metadata_text.lower()
    assert "readability" in metadata_text.lower()
    assert "class 0" in metadata_text.lower() or "class_0" in metadata_text.lower()
    assert "crisis" not in metadata_text.lower()
    assert metadata["selected_archive"] == str(complete_geodt_archive)
    assert metadata["workflow_mode"] == "figure-generation"
    assert metadata["selected_run_month"] == "2024-10"
    assert metadata["forecasting_scope"] == "fs1"
    assert metadata["minimum_required_artifact_completeness"] is True
    assert metadata["branch_assignment_source"]["type"] == "X_branch_id.npy"
    assert metadata["feature_name_source"]["feature_name_count"] == 3
    assert metadata["root_global_exclusion_decision"]
    assert metadata["selected_pair_score"] >= 0
    assert metadata["contrast_descriptor"] in {"low", "moderate", "high"}
    assert metadata["readability_result"]["passes"] is True
    assert metadata["figure_layout"]["width_inches"] >= 24
    assert metadata["figure_layout"]["height_inches"] >= 8
    assert metadata["figure_layout"]["tree_fontsize"] <= 6
    assert metadata["figure_layout"]["uses_constrained_layout"] is True
    assert "tie_break_result" in metadata
    assert "rejected_higher_scoring_pairs" in metadata
    assert set(metadata["selected_branch_ids"]) == set(
        metadata["checkpoint_paths_used"]
    )
    assert all(Path(path).is_file() for path in metadata["output_figure_paths"])


def test_figure_metadata_contains_reproduction_schema_inputs(
    complete_geodt_archive: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "diagnostics"

    result = _run_diagnostic(
        "--archive-path",
        str(complete_geodt_archive),
        "--output-dir",
        str(output_dir),
        "--k",
        "3",
        "--max-plot-depth",
        "3",
    )

    assert result.returncode == 0
    metadata_files = list(output_dir.glob("*_metadata.json"))
    assert metadata_files
    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))

    assert metadata["archive_discovery_input"]["type"] == "archive-path"
    assert metadata["archive_discovery_input"]["value"] == str(complete_geodt_archive)
    assert metadata["selected_archive"] == str(complete_geodt_archive)
    assert "artifact_provider_path" in metadata
    assert metadata["selected_branch_ids"]
    assert set(metadata["selected_branch_ids"]) == set(
        metadata["checkpoint_paths_used"]
    )
    assert metadata["feature_name_source"]["path"]
    assert metadata["k"] == 3
    assert metadata["actual_plotted_depth"] == 3
    assert metadata["selected_pair_score"] >= 0
    assert (
        metadata["dissimilarity_formula_name"] == "top_k_split_feature_jaccard_distance"
    )
    assert metadata["dissimilarity_formula_version"] == "1"
    assert metadata["discovered_archive_candidate_count"] >= 1
    assert metadata["candidate_archive_decisions"]
    assert (
        metadata["evidence_validation_status"]["branch_assignment_source"]
        == "confirmed"
    )
    assert metadata["evidence_validation_status"]["feature_name_source"] == "confirmed"
    assert (
        metadata["evidence_validation_status"]["checkpoint_branch_id_parsing"]
        == "confirmed"
    )
    assert (
        metadata["evidence_validation_status"]["root_global_dt_rules_boundary"]
        == "confirmed"
    )
    assert metadata["checkpoint_loading_notes"]
    assert metadata["feature_name_source"]["checkpoint_feature_count"] == 3
    assert (
        metadata["feature_name_source"]["feature_name_compatibility_result"]
        == "selected"
    )
    for branch_id in metadata["selected_branch_ids"]:
        assert metadata["selected_signatures"][branch_id]["split_feature_set"]


def test_reproduce_from_metadata_fails_when_recovered_top_k_features_change(
    complete_geodt_archive: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "diagnostics"
    reproduce_dir = tmp_path / "reproduce"

    figure_result = _run_diagnostic(
        "--archive-path",
        str(complete_geodt_archive),
        "--output-dir",
        str(output_dir),
        "--k",
        "3",
        "--max-plot-depth",
        "3",
    )
    assert figure_result.returncode == 0
    metadata_files = list(output_dir.glob("*_metadata.json"))
    assert metadata_files
    metadata_path = metadata_files[0]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_name_path = Path(metadata["feature_name_source"]["path"])
    _write_feature_names(
        feature_name_path,
        ["changed_rainfall", "changed_price", "changed_conflict"],
    )

    reproduce_result = _run_diagnostic(
        "--reproduce-from", str(metadata_path), "--output-dir", str(reproduce_dir)
    )

    assert reproduce_result.returncode != 0
    combined = reproduce_result.stdout + reproduce_result.stderr
    assert "mismatch" in combined.lower()
    assert "top-k" in combined.lower() or "signature" in combined.lower()
    failure_summary_files = list(
        reproduce_dir.glob("*_reproduction_failure_summary.json")
    )
    assert failure_summary_files
    failure_summary = json.loads(failure_summary_files[0].read_text(encoding="utf-8"))
    assert failure_summary["failure_stage"] == "reproduction-mismatch"
    assert failure_summary["reselected_pair"] is False
    assert failure_summary["evidence_mismatches"]
    assert not list(reproduce_dir.glob("*.png"))
    assert not list(reproduce_dir.glob("*.pdf"))


def test_reproduce_from_metadata_cli_recovers_recorded_selection_without_reselection(
    complete_geodt_archive: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "diagnostics"
    reproduce_dir = tmp_path / "reproduce"

    figure_result = _run_diagnostic(
        "--archive-path",
        str(complete_geodt_archive),
        "--output-dir",
        str(output_dir),
        "--k",
        "3",
        "--max-plot-depth",
        "3",
    )
    assert figure_result.returncode == 0
    metadata_files = list(output_dir.glob("*_metadata.json"))
    assert metadata_files
    metadata_path = metadata_files[0]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    reproduce_result = _run_diagnostic(
        "--reproduce-from", str(metadata_path), "--output-dir", str(reproduce_dir)
    )

    assert reproduce_result.returncode == 0
    report_files = list(reproduce_dir.glob("*_reproduction_report.json"))
    assert report_files
    report = json.loads(report_files[0].read_text(encoding="utf-8"))
    assert report["workflow_mode"] == "reproduction"
    assert report["source_metadata_path"] == str(metadata_path)
    assert report["selected_archive"] == metadata["selected_archive"]
    assert report["selected_branch_ids"] == metadata["selected_branch_ids"]
    assert report["selected_pair_score"] == pytest.approx(
        metadata["selected_pair_score"]
    )
    assert report["reselected_pair"] is False
    assert not list(reproduce_dir.glob("*.png"))
    assert not list(reproduce_dir.glob("*.pdf"))


def test_reproduce_from_metadata_fails_on_missing_recorded_checkpoint_without_reselection(
    complete_geodt_archive: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "diagnostics"

    figure_result = _run_diagnostic(
        "--archive-path",
        str(complete_geodt_archive),
        "--output-dir",
        str(output_dir),
        "--k",
        "3",
        "--max-plot-depth",
        "3",
    )
    assert figure_result.returncode == 0
    metadata_files = list(output_dir.glob("*_metadata.json"))
    assert metadata_files
    metadata_path = metadata_files[0]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    first_checkpoint_path = Path(next(iter(metadata["checkpoint_paths_used"].values())))
    first_checkpoint_path.unlink()

    reproduce_result = _run_diagnostic(
        "--reproduce-from",
        str(metadata_path),
        "--output-dir",
        str(tmp_path / "reproduce"),
    )

    assert reproduce_result.returncode != 0
    combined = reproduce_result.stdout + reproduce_result.stderr
    assert "mismatch" in combined.lower()
    assert "missing" in combined.lower()
    assert str(first_checkpoint_path) in combined
    failure_summary_files = list(
        (tmp_path / "reproduce").glob("*_reproduction_failure_summary.json")
    )
    assert failure_summary_files
    failure_summary = json.loads(failure_summary_files[0].read_text(encoding="utf-8"))
    assert failure_summary["failure_stage"] == "reproduction-mismatch"
    assert failure_summary["reselected_pair"] is False
    assert str(first_checkpoint_path) in failure_summary["missing_artifacts"]
    assert not list((tmp_path / "reproduce").glob("*.png"))
    assert not list((tmp_path / "reproduce").glob("*.pdf"))


def test_audit_only_writes_selection_parity_metadata_without_rendering_figures(
    complete_geodt_archive: Path, tmp_path: Path
) -> None:
    figure_output_dir = tmp_path / "figure-diagnostics"
    audit_output_dir = tmp_path / "audit-diagnostics"

    figure_result = _run_diagnostic(
        "--archive-path",
        str(complete_geodt_archive),
        "--output-dir",
        str(figure_output_dir),
        "--k",
        "3",
        "--max-plot-depth",
        "3",
    )
    audit_result = _run_diagnostic(
        "--audit-only",
        "--archive-path",
        str(complete_geodt_archive),
        "--output-dir",
        str(audit_output_dir),
        "--k",
        "3",
        "--max-plot-depth",
        "3",
    )

    assert figure_result.returncode == 0
    assert audit_result.returncode == 0
    assert not list(audit_output_dir.glob("*.png"))
    assert not list(audit_output_dir.glob("*.pdf"))

    figure_metadata_files = list(figure_output_dir.glob("*_metadata.json"))
    audit_files = list(audit_output_dir.glob("*_audit.json"))
    assert figure_metadata_files
    assert audit_files
    figure_metadata = json.loads(figure_metadata_files[0].read_text(encoding="utf-8"))
    audit_metadata = json.loads(audit_files[0].read_text(encoding="utf-8"))

    assert audit_metadata["workflow_mode"] == "audit-only"
    assert audit_metadata["selected_archive"] == figure_metadata["selected_archive"]
    assert (
        audit_metadata["selected_branch_ids"] == figure_metadata["selected_branch_ids"]
    )
    assert audit_metadata["selected_pair_score"] == pytest.approx(
        figure_metadata["selected_pair_score"]
    )
    assert audit_metadata["readability_result"] == figure_metadata["readability_result"]
    assert audit_metadata["tie_break_result"] == figure_metadata["tie_break_result"]
    assert audit_metadata["output_figure_paths"] == []
    assert audit_metadata["selection_pipeline_source"] == "shared"


def test_top_k_signature_extraction_records_depth_features_and_neutral_leaf_summaries() -> (
    None
):
    diagnostic = _load_diagnostic_module()
    clf = _fit_tree()

    signature = diagnostic.extract_branch_signature(
        clf,
        feature_names=["rainfall", "price", "conflict"],
        k=3,
    )

    assert signature["k"] == 3
    assert signature["actual_available_depth"] <= 2
    assert signature["split_features_by_depth"]
    assert signature["split_feature_set"]
    assert "rainfall" in signature["split_feature_set"]
    assert "leaf_class_summaries" in signature
    assert all(
        "crisis" not in str(item).lower() for item in signature["leaf_class_summaries"]
    )
    assert any(
        "class" in str(item).lower() for item in signature["leaf_class_summaries"]
    )


def test_jaccard_scoring_is_deterministic_and_ignores_supplemental_threshold_direction_fields() -> (
    None
):
    diagnostic = _load_diagnostic_module()
    signatures = {
        "0": {
            "branch_id": "0",
            "split_feature_set": {"rainfall", "price"},
            "threshold_direction_summary": {"rainfall": ["<= 0.5"]},
        },
        "1": {
            "branch_id": "1",
            "split_feature_set": {"price", "conflict"},
            "threshold_direction_summary": {"price": ["> 0.1"]},
        },
        "00": {
            "branch_id": "00",
            "split_feature_set": {"rainfall", "price"},
            "threshold_direction_summary": {"rainfall": ["> 0.9"]},
        },
    }

    scored = diagnostic.score_branch_pairs(signatures)

    pair_scores = {
        tuple(record["branch_pair"]): record["jaccard_distance"] for record in scored
    }
    assert pair_scores[("0", "1")] == pytest.approx(2 / 3)
    assert pair_scores[("0", "00")] == pytest.approx(0.0)
    assert scored[0]["branch_pair"] == ("0", "1")
    assert scored[0]["ranking_formula"] == "top_k_split_feature_jaccard_distance"
    assert scored[0]["threshold_direction_fields_used_for_ranking"] is False


def test_pair_selection_applies_tie_breaks_readability_gate_and_no_readable_failure() -> (
    None
):
    diagnostic = _load_diagnostic_module()
    scored_pairs = [
        {
            "branch_pair": ("0", "1"),
            "jaccard_distance": 1.0,
            "min_assigned_count": 3,
            "total_assigned_count": 8,
            "min_prediction_row_count": 3,
        },
        {
            "branch_pair": ("00", "01"),
            "jaccard_distance": 0.75,
            "min_assigned_count": 5,
            "total_assigned_count": 12,
            "min_prediction_row_count": 5,
        },
        {
            "branch_pair": ("10", "11"),
            "jaccard_distance": 0.75,
            "min_assigned_count": 5,
            "total_assigned_count": 10,
            "min_prediction_row_count": 5,
        },
    ]
    readability = {
        ("0", "1"): {
            "passes": False,
            "reason": "no non-leaf split within plotted depth",
        },
        ("00", "01"): {"passes": True, "reason": "readable"},
        ("10", "11"): {"passes": True, "reason": "readable"},
    }

    selection = diagnostic.select_readable_pair(scored_pairs, readability)

    assert selection["selected_pair"] == ("00", "01")
    assert selection["selected_pair_score"] == pytest.approx(0.75)
    assert selection["tie_break_result"]["min_assigned_count"] == 5
    assert selection["rejected_higher_scoring_pairs"][0]["branch_pair"] == ("0", "1")
    assert (
        "readability" in selection["rejected_higher_scoring_pairs"][0]["reason"].lower()
    )

    with pytest.raises(ValueError, match="No readable branch pair"):
        diagnostic.select_readable_pair(
            scored_pairs,
            {pair["branch_pair"]: {"passes": False} for pair in scored_pairs},
        )
