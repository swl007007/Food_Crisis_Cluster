import json
from pathlib import Path

import pandas as pd

from scripts.audit_final_artifact_sources import (
    CLEAN_PANEL_BASENAME,
    PHASE_CHANGE_BASENAME,
    audit_provider_manifest,
    classify_source_text,
    write_audit_outputs,
)


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
