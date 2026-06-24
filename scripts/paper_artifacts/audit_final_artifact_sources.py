#!/usr/bin/env python3
"""Audit paper artifact and provider sources for phase-change contamination."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.release_paths import ReleasePaths

PATHS = ReleasePaths(repo_root=REPO_ROOT)
FINAL_ARTIFACT_ROOT = PATHS.final_artifacts_root
CLEAN_PANEL_BASENAME = "FEWSNET_forecast_unadjusted_bm.csv"
PHASE_CHANGE_BASENAME = "FEWSNET_forecast_unadjusted_bm_phase_change.csv"

PAPER_ARTIFACT_GROUPS = {
    "01_main_results": "clean",
    "02_methods_and_temporal_scope": "static_or_shape_only",
    "03_class_prevalence": "clean",
    "04_error_analysis": "clean",
    "05_partition_diagnostics": "static_or_shape_only",
    "06_cluster_profiles": "requires_artifact_manifest",
    "07_probability_uncertainty": "clean",
    "08_geodt_diagnostics": "static_or_shape_only",
    "09_humanitarian_metrics": "clean",
    "10_false_negative_error_modes": "requires_artifact_manifest",
    "11_threshold_free_metrics": "clean",
    "12_thresholded_georf_results": "requires_artifact_manifest",
}

PROVIDER_MANIFESTS = [
    ("result_partition_k40_compare_GF_fs1", PATHS.stage3_root("GF", 1) / "run_manifest.json"),
    ("result_partition_k40_compare_GF_fs2", PATHS.stage3_root("GF", 2) / "run_manifest.json"),
    ("result_partition_k40_compare_GF_fs3", PATHS.stage3_root("GF", 3) / "run_manifest.json"),
    ("result_partition_k40_compare_DT_fs1", PATHS.stage3_root("DT", 1) / "run_manifest.json"),
    ("result_partition_k40_compare_DT_fs2", PATHS.stage3_root("DT", 2) / "run_manifest.json"),
    ("result_partition_k40_compare_DT_fs3", PATHS.stage3_root("DT", 3) / "run_manifest.json"),
    (
        "result_partition_k40_compare_GF_thresholded_fs1",
        PATHS.thresholded_georf_root(1) / "run_manifest.json",
    ),
    (
        "result_partition_k40_compare_GF_thresholded_fs2",
        PATHS.thresholded_georf_root(2) / "run_manifest.json",
    ),
    (
        "result_partition_k40_compare_GF_thresholded_fs3",
        PATHS.thresholded_georf_root(3) / "run_manifest.json",
    ),
]

SCRIPT_DEFAULTS = [
    (
        "scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py",
        REPO_ROOT / "scripts" / "compare_partitioned_vs_pooled_rf_k40_nc4.py",
    ),
    (
        "archived/release_20260624_nonpaper_pipelines/geoxgb_workflow/scripts/compare_partitioned_vs_pooled_xgb_k40_nc4.py",
        REPO_ROOT
        / "archived"
        / "release_20260624_nonpaper_pipelines"
        / "geoxgb_workflow"
        / "scripts"
        / "compare_partitioned_vs_pooled_xgb_k40_nc4.py",
    ),
    (
        "scripts/paper_artifacts/analyze_georf_m2_cluster_profiles.py",
        REPO_ROOT / "scripts" / "paper_artifacts" / "analyze_georf_m2_cluster_profiles.py",
    ),
    (
        "scripts/paper_artifacts/analyze_georf_false_negative_error_modes.py",
        REPO_ROOT / "scripts" / "paper_artifacts" / "analyze_georf_false_negative_error_modes.py",
    ),
]

ARTIFACT_MANIFESTS = {
    "06_cluster_profiles": "artifact_source_manifest.json",
    "10_false_negative_error_modes": "artifact_source_manifest.json",
    "12_thresholded_georf_results": "artifact_source_manifest.json",
}


def classify_source_text(source_text: str, *, requires_panel: bool) -> tuple[str, str]:
    """Classify source text with phase-change detection taking precedence."""
    text = str(source_text or "")
    if PHASE_CHANGE_BASENAME in text:
        return "invalid_phase_change", "source text references phase-change panel"
    if CLEAN_PANEL_BASENAME in text:
        return "clean", "source text references clean panel"
    if requires_panel:
        return "needs_regeneration", "no source text proves clean panel provenance"
    return "static_or_shape_only", "no model-panel source required"


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_from_manifest(data: dict[str, Any]) -> str:
    values: list[str] = []
    for key in (
        "data_path",
        "panel_path",
        "source_data_path",
        "clean_panel_source",
        "source_path",
    ):
        value = data.get(key)
        if value:
            values.append(str(value))
    for key in ("provider_manifests", "source_data_paths", "sources"):
        value = data.get(key)
        if isinstance(value, dict):
            values.extend(str(item) for item in value.values())
        elif isinstance(value, list):
            values.extend(str(item) for item in value)
    return "\n".join(values)


def audit_provider_manifest(group: str, manifest_path: Path) -> dict[str, str]:
    """Audit a model-provider manifest."""
    if not manifest_path.is_file():
        return {
            "artifact_group": group,
            "artifact_type": "provider_manifest",
            "path": _rel(manifest_path),
            "status": "needs_regeneration",
            "source_path": "",
            "reason": "missing provider manifest",
        }
    data = _read_json(manifest_path)
    source = str(data.get("data_path", ""))
    status, reason = classify_source_text(source, requires_panel=True)
    return {
        "artifact_group": group,
        "artifact_type": "provider_manifest",
        "path": _rel(manifest_path),
        "status": status,
        "source_path": source,
        "reason": reason,
    }


def audit_script_default(label: str, path: Path) -> dict[str, str]:
    """Audit a script's paper-facing source default."""
    if not path.is_file():
        return {
            "artifact_group": label,
            "artifact_type": "script_default",
            "path": _rel(path),
            "status": "needs_regeneration",
            "source_path": "",
            "reason": "script missing",
        }
    text = path.read_text(encoding="utf-8", errors="replace")
    status, reason = classify_source_text(text, requires_panel=True)
    return {
        "artifact_group": label,
        "artifact_type": "script_default",
        "path": _rel(path),
        "status": status,
        "source_path": CLEAN_PANEL_BASENAME if CLEAN_PANEL_BASENAME in text else "",
        "reason": reason,
    }


def audit_artifact_manifest(group: str, manifest_path: Path) -> dict[str, str]:
    """Audit a final artifact group's own provenance manifest."""
    if not manifest_path.is_file():
        return {
            "artifact_group": group,
            "artifact_type": "artifact_manifest",
            "path": _rel(manifest_path),
            "status": "needs_regeneration",
            "source_path": "",
            "reason": "missing artifact source manifest",
        }
    data = _read_json(manifest_path)
    source = _source_from_manifest(data)
    status, reason = classify_source_text(source, requires_panel=True)
    return {
        "artifact_group": group,
        "artifact_type": "artifact_manifest",
        "path": _rel(manifest_path),
        "status": status,
        "source_path": source.replace("\n", "; "),
        "reason": reason,
    }


def audit_artifact_group(
    group: str,
    mode: str,
    manifest_rows: list[dict[str, str]],
) -> dict[str, str]:
    """Audit a final artifact group with strict provenance requirements."""
    path = FINAL_ARTIFACT_ROOT / group
    if not path.exists():
        return {
            "artifact_group": group,
            "artifact_type": "final_artifact_group",
            "path": _rel(path),
            "status": "needs_regeneration",
            "source_path": "",
            "reason": "artifact group missing",
        }
    if mode in {"clean", "static_or_shape_only"}:
        return {
            "artifact_group": group,
            "artifact_type": "final_artifact_group",
            "path": _rel(path),
            "status": mode,
            "source_path": "",
            "reason": f"classified by artifact contract as {mode}",
        }

    relevant = [row for row in manifest_rows if row["artifact_group"] == group]
    statuses = {row["status"] for row in relevant}
    if "invalid_phase_change" in statuses:
        status = "invalid_phase_change"
        reason = "artifact source manifest references phase-change panel"
    elif "needs_regeneration" in statuses or not relevant:
        status = "needs_regeneration"
        reason = "artifact source manifest missing or not clean"
    else:
        status = "clean"
        reason = "artifact source manifest proves clean panel provenance"
    return {
        "artifact_group": group,
        "artifact_type": "final_artifact_group",
        "path": _rel(path),
        "status": status,
        "source_path": "",
        "reason": reason,
    }


def build_audit_rows() -> list[dict[str, str]]:
    """Build all provider, generator, artifact-manifest, and artifact-group rows."""
    provider_rows = [audit_provider_manifest(group, path) for group, path in PROVIDER_MANIFESTS]
    script_rows = [audit_script_default(label, path) for label, path in SCRIPT_DEFAULTS]
    artifact_manifest_rows = [
        audit_artifact_manifest(group, FINAL_ARTIFACT_ROOT / group / filename)
        for group, filename in ARTIFACT_MANIFESTS.items()
    ]
    artifact_rows = [
        audit_artifact_group(group, mode, artifact_manifest_rows)
        for group, mode in PAPER_ARTIFACT_GROUPS.items()
    ]
    return provider_rows + script_rows + artifact_manifest_rows + artifact_rows


def write_audit_outputs(rows: list[dict[str, str]], output_dir: Path) -> tuple[Path, Path]:
    """Write CSV and Markdown audit reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "artifact_source_audit.csv"
    md_path = output_dir / "artifact_source_audit.md"
    fieldnames = ["artifact_group", "artifact_type", "path", "status", "source_path", "reason"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Artifact source audit",
        "",
        "| artifact_group | artifact_type | status | reason |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['artifact_group']} | {row['artifact_type']} | {row['status']} | {row['reason']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FINAL_ARTIFACT_ROOT)
    parser.add_argument("--fail-on-unclean", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = build_audit_rows()
    csv_path, md_path = write_audit_outputs(rows, args.output_dir)
    unclean = [row for row in rows if row["status"] in {"invalid_phase_change", "needs_regeneration"}]
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    if args.fail_on_unclean and unclean:
        for row in unclean:
            print(f"UNCLEAN: {row['artifact_group']} [{row['status']}] {row['reason']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
