import json
import subprocess
from pathlib import Path

import pandas as pd

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
from scripts.verify_current_results_reproducibility import audit_rows_are_clean


def test_classify_source_text_detects_clean_panel():
    status, reason = classify_source_text(
        rf"C:\data\{CLEAN_PANEL_BASENAME}",
        requires_panel=True,
    )
    assert status == "clean"
    assert "clean panel" in reason


def test_classify_source_text_detects_phase_change_panel():
    status, reason = classify_source_text(
        rf"C:\data\{PHASE_CHANGE_BASENAME}",
        requires_panel=True,
    )
    assert status == "invalid_phase_change"
    assert "phase-change" in reason


def test_classify_source_text_fails_closed_when_panel_required_and_missing():
    status, reason = classify_source_text("", requires_panel=True)
    assert status == "needs_regeneration"
    assert "no source text" in reason


def test_provider_manifest_records_invalid_phase_change(tmp_path: Path):
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(
        json.dumps({"data_path": rf"C:\data\{PHASE_CHANGE_BASENAME}"}),
        encoding="utf-8",
    )

    row = audit_provider_manifest("thresholded fs1", manifest)

    assert row["status"] == "invalid_phase_change"
    assert row["source_path"].endswith(PHASE_CHANGE_BASENAME)


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


def test_audit_script_help_runs_when_executed_directly():
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        ["python3", "scripts/paper_artifacts/audit_final_artifact_sources.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "--output-dir" in result.stdout


def test_write_audit_outputs_writes_csv_and_markdown(tmp_path: Path):
    rows = [
        {
            "artifact_group": "result_partition_k40_compare_GF_fs1",
            "artifact_type": "provider_manifest",
            "path": "result_partition_k40_compare_GF_fs1/run_manifest.json",
            "status": "clean",
            "source_path": rf"C:\data\{CLEAN_PANEL_BASENAME}",
            "reason": "manifest uses clean panel",
        }
    ]

    csv_path, md_path = write_audit_outputs(rows, tmp_path)

    assert csv_path.is_file()
    assert md_path.is_file()
    df = pd.read_csv(csv_path)
    assert df.loc[0, "status"] == "clean"
    assert "result_partition_k40_compare_GF_fs1" in md_path.read_text(encoding="utf-8")


def test_artifact_manifest_accepts_source_data_paths(tmp_path: Path):
    manifest = tmp_path / "artifact_source_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_data_paths": {
                    "fs1": rf"C:\data\{CLEAN_PANEL_BASENAME}",
                    "fs2": rf"C:\data\{CLEAN_PANEL_BASENAME}",
                },
                "provider_manifests": {
                    "fs1": "result_partition_k40_compare_GF_thresholded_fs1/run_manifest.json"
                },
            }
        ),
        encoding="utf-8",
    )

    row = audit_artifact_manifest("12_thresholded_georf_results", manifest)

    assert row["status"] == "clean"
    assert CLEAN_PANEL_BASENAME in row["source_path"]


def test_paper_facing_script_defaults_do_not_reference_phase_change():
    repo_root = Path(__file__).resolve().parents[2]
    scripts = [
        repo_root / "scripts" / "compare_partitioned_vs_pooled_rf_k40_nc4.py",
        repo_root
        / "archived"
        / "release_20260624_nonpaper_pipelines"
        / "geoxgb_workflow"
        / "scripts"
        / "compare_partitioned_vs_pooled_xgb_k40_nc4.py",
        repo_root
        / "scripts"
        / "paper_artifacts"
        / "analyze_georf_m2_cluster_profiles.py",
        repo_root
        / "scripts"
        / "paper_artifacts"
        / "analyze_georf_false_negative_error_modes.py",
    ]
    for script in scripts:
        row = audit_script_default(script.as_posix(), script)
        assert row["status"] == "clean", row


def test_audit_rows_are_clean_rejects_phase_change_and_needs_regeneration():
    rows = [
        {"artifact_group": "06_cluster_profiles", "artifact_type": "final_artifact_group", "status": "clean"},
        {
            "artifact_group": "12_thresholded_georf_results",
            "artifact_type": "final_artifact_group",
            "status": "invalid_phase_change",
        },
        {
            "artifact_group": "10_false_negative_error_modes",
            "artifact_type": "final_artifact_group",
            "status": "needs_regeneration",
        },
    ]

    ok, failures = audit_rows_are_clean(rows)

    assert not ok
    assert any("12_thresholded_georf_results" in failure for failure in failures)
    assert any("10_false_negative_error_modes" in failure for failure in failures)


def test_audit_rows_are_clean_accepts_static_and_clean_rows():
    rows = [
        {"artifact_group": "01_main_results", "artifact_type": "final_artifact_group", "status": "clean"},
        {
            "artifact_group": "05_partition_diagnostics",
            "artifact_type": "final_artifact_group",
            "status": "static_or_shape_only",
        },
    ]

    ok, failures = audit_rows_are_clean(rows)

    assert ok
    assert failures == []
