# Artifact Source Audit Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove phase-change contamination from paper-facing artifacts, regenerate affected GeoRF outputs from the clean panel, and add fail-closed provenance verification.

**Architecture:** Add one focused audit script that classifies paper artifact groups and provider manifests as `clean`, `invalid_phase_change`, `needs_regeneration`, or `static_or_shape_only`. Patch unsafe script defaults to the clean panel, regenerate contaminated provider/artifact folders, and make `verify_current_results_reproducibility.py` fail if the final bundle is not clean.

**Tech Stack:** Python 3.12, pandas/csv/json/pathlib, existing GeoRF Stage 3 comparison scripts, existing `src/tests` pytest-style tests, Windows Python at `/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe`.

---

## File Structure

- Create: `scripts/audit_final_artifact_sources.py`
  - Owns artifact/provider provenance classification and writes `artifact_source_audit.csv` and `artifact_source_audit.md`.
- Create: `src/tests/test_artifact_source_audit.py`
  - Covers strict source-path matching, phase-change detection, missing-provider fail-closed behavior, and audit summary output.
- Modify: `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py`
  - Change `DEFAULT_DATA_PATH` from phase-change to clean panel.
- Modify: `scripts/compare_partitioned_vs_pooled_xgb_k40_nc4.py`
  - Change the same default to clean panel so future non-paper runs do not accidentally inherit the invalid source.
- Modify: `scripts/analyze_georf_m2_cluster_profiles.py`
  - Change `DEFAULT_PANEL` to clean panel.
- Modify: `scripts/analyze_georf_false_negative_error_modes.py`
  - Change `DEFAULT_DATA` to clean panel.
- Modify: `scripts/verify_current_results_reproducibility.py`
  - Invoke/read the audit output and fail on paper-facing `invalid_phase_change` or unresolved `needs_regeneration` rows.
- Regenerate:
  - `result_partition_k40_compare_GF_thresholded_fs1/`
  - `result_partition_k40_compare_GF_thresholded_fs2/`
  - `result_partition_k40_compare_GF_thresholded_fs3/`
  - `final_artifacts_in_paper_updated/06_cluster_profiles/`
  - `final_artifacts_in_paper_updated/10_false_negative_error_modes/`
  - `final_artifacts_in_paper_updated/12_thresholded_georf_results/`
  - `final_artifacts_in_paper_updated/artifact_source_audit.csv`
  - `final_artifacts_in_paper_updated/artifact_source_audit.md`

### Task 1: Add Fail-Closed Artifact Source Audit

**Files:**
- Create: `scripts/audit_final_artifact_sources.py`
- Create: `src/tests/test_artifact_source_audit.py`

- [ ] **Step 1: Write tests for source classification**

Create `src/tests/test_artifact_source_audit.py` with these concrete tests:

```python
import json
from pathlib import Path

import pandas as pd

from scripts.audit_final_artifact_sources import (
    CLEAN_PANEL_BASENAME,
    PHASE_CHANGE_BASENAME,
    classify_source_text,
    audit_provider_manifest,
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
```

- [ ] **Step 2: Run the new tests and confirm they fail**

Run:

```bash
/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe -m pytest src/tests/test_artifact_source_audit.py -q
```

Expected: FAIL because `scripts.audit_final_artifact_sources` does not exist yet.

- [ ] **Step 3: Implement `scripts/audit_final_artifact_sources.py`**

Create the script with these public functions and behavior:

```python
#!/usr/bin/env python3
"""Audit paper artifact and provider sources for phase-change contamination."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_ARTIFACT_ROOT = REPO_ROOT / "final_artifacts_in_paper_updated"
CLEAN_PANEL_BASENAME = "FEWSNET_forecast_unadjusted_bm.csv"
PHASE_CHANGE_BASENAME = "FEWSNET_forecast_unadjusted_bm_phase_change.csv"

PAPER_ARTIFACT_GROUPS = {
    "01_main_results": "clean",
    "02_methods_and_temporal_scope": "static_or_shape_only",
    "03_class_prevalence": "clean",
    "04_error_analysis": "clean",
    "05_partition_diagnostics": "static_or_shape_only",
    "06_cluster_profiles": "needs_panel_audit",
    "07_probability_uncertainty": "clean",
    "08_geodt_diagnostics": "static_or_shape_only",
    "09_humanitarian_metrics": "clean",
    "10_false_negative_error_modes": "needs_panel_audit",
    "11_threshold_free_metrics": "clean",
    "12_thresholded_georf_results": "needs_provider_audit",
}

PROVIDER_MANIFESTS = [
    ("result_partition_k40_compare_GF_fs1", REPO_ROOT / "result_partition_k40_compare_GF_fs1" / "run_manifest.json"),
    ("result_partition_k40_compare_GF_fs2", REPO_ROOT / "result_partition_k40_compare_GF_fs2" / "run_manifest.json"),
    ("result_partition_k40_compare_GF_fs3", REPO_ROOT / "result_partition_k40_compare_GF_fs3" / "run_manifest.json"),
    ("result_partition_k40_compare_DT_fs1", REPO_ROOT / "result_partition_k40_compare_DT_fs1" / "run_manifest.json"),
    ("result_partition_k40_compare_DT_fs2", REPO_ROOT / "result_partition_k40_compare_DT_fs2" / "run_manifest.json"),
    ("result_partition_k40_compare_DT_fs3", REPO_ROOT / "result_partition_k40_compare_DT_fs3" / "run_manifest.json"),
    ("result_partition_k40_compare_GF_thresholded_fs1", REPO_ROOT / "result_partition_k40_compare_GF_thresholded_fs1" / "run_manifest.json"),
    ("result_partition_k40_compare_GF_thresholded_fs2", REPO_ROOT / "result_partition_k40_compare_GF_thresholded_fs2" / "run_manifest.json"),
    ("result_partition_k40_compare_GF_thresholded_fs3", REPO_ROOT / "result_partition_k40_compare_GF_thresholded_fs3" / "run_manifest.json"),
]

SCRIPT_DEFAULTS = [
    ("scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py", REPO_ROOT / "scripts" / "compare_partitioned_vs_pooled_rf_k40_nc4.py"),
    ("scripts/compare_partitioned_vs_pooled_xgb_k40_nc4.py", REPO_ROOT / "scripts" / "compare_partitioned_vs_pooled_xgb_k40_nc4.py"),
    ("scripts/analyze_georf_m2_cluster_profiles.py", REPO_ROOT / "scripts" / "analyze_georf_m2_cluster_profiles.py"),
    ("scripts/analyze_georf_false_negative_error_modes.py", REPO_ROOT / "scripts" / "analyze_georf_false_negative_error_modes.py"),
]


def classify_source_text(source_text: str, *, requires_panel: bool) -> tuple[str, str]:
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


def audit_provider_manifest(group: str, manifest_path: Path) -> dict[str, str]:
    if not manifest_path.is_file():
        return {
            "artifact_group": group,
            "artifact_type": "provider_manifest",
            "path": _rel(manifest_path),
            "status": "needs_regeneration",
            "source_path": "",
            "reason": "missing provider manifest",
        }
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
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


def audit_artifact_group(group: str, mode: str, provider_rows: list[dict[str, str]], script_rows: list[dict[str, str]]) -> dict[str, str]:
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
    if group == "06_cluster_profiles":
        relevant = [row for row in script_rows if row["artifact_group"].endswith("analyze_georf_m2_cluster_profiles.py")]
    elif group == "10_false_negative_error_modes":
        relevant = [row for row in script_rows if row["artifact_group"].endswith("analyze_georf_false_negative_error_modes.py")]
    elif group == "12_thresholded_georf_results":
        relevant = [row for row in provider_rows if "thresholded" in row["artifact_group"]]
    else:
        relevant = []
    statuses = {row["status"] for row in relevant}
    if "invalid_phase_change" in statuses:
        status = "invalid_phase_change"
        reason = "dependent provider or generator references phase-change panel"
    elif "needs_regeneration" in statuses or not relevant:
        status = "needs_regeneration"
        reason = "dependent provenance is incomplete or not clean"
    else:
        status = "clean"
        reason = "dependent provider or generator provenance is clean"
    return {
        "artifact_group": group,
        "artifact_type": "final_artifact_group",
        "path": _rel(path),
        "status": status,
        "source_path": "",
        "reason": reason,
    }


def build_audit_rows() -> list[dict[str, str]]:
    provider_rows = [audit_provider_manifest(group, path) for group, path in PROVIDER_MANIFESTS]
    script_rows = [audit_script_default(label, path) for label, path in SCRIPT_DEFAULTS]
    artifact_rows = [
        audit_artifact_group(group, mode, provider_rows, script_rows)
        for group, mode in PAPER_ARTIFACT_GROUPS.items()
    ]
    return provider_rows + script_rows + artifact_rows


def write_audit_outputs(rows: list[dict[str, str]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "artifact_source_audit.csv"
    md_path = output_dir / "artifact_source_audit.md"
    fieldnames = ["artifact_group", "artifact_type", "path", "status", "source_path", "reason"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Artifact source audit",
        "",
        "| artifact_group | artifact_type | status | reason |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['artifact_group']} | {row['artifact_type']} | {row['status']} | {row['reason']} |")
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
```

- [ ] **Step 4: Run audit tests and commit**

Run:

```bash
/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe -m pytest src/tests/test_artifact_source_audit.py -q
```

Expected: PASS.

Commit:

```bash
git add scripts/audit_final_artifact_sources.py src/tests/test_artifact_source_audit.py
git commit -m "add artifact source audit"
```

### Task 2: Patch Unsafe Phase-Change Defaults

**Files:**
- Modify: `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py`
- Modify: `scripts/compare_partitioned_vs_pooled_xgb_k40_nc4.py`
- Modify: `scripts/analyze_georf_m2_cluster_profiles.py`
- Modify: `scripts/analyze_georf_false_negative_error_modes.py`
- Test: `src/tests/test_artifact_source_audit.py`

- [ ] **Step 1: Add a regression test for script defaults**

Append this test to `src/tests/test_artifact_source_audit.py`:

```python
from scripts.audit_final_artifact_sources import audit_script_default


def test_paper_facing_script_defaults_do_not_reference_phase_change():
    repo_root = Path(__file__).resolve().parents[2]
    scripts = [
        repo_root / "scripts" / "compare_partitioned_vs_pooled_rf_k40_nc4.py",
        repo_root / "scripts" / "compare_partitioned_vs_pooled_xgb_k40_nc4.py",
        repo_root / "scripts" / "analyze_georf_m2_cluster_profiles.py",
        repo_root / "scripts" / "analyze_georf_false_negative_error_modes.py",
    ]
    for script in scripts:
        row = audit_script_default(script.as_posix(), script)
        assert row["status"] == "clean", row
```

- [ ] **Step 2: Run the regression test and confirm it fails before patching**

Run:

```bash
/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe -m pytest src/tests/test_artifact_source_audit.py::test_paper_facing_script_defaults_do_not_reference_phase_change -q
```

Expected: FAIL while the four scripts still contain phase-change defaults.

- [ ] **Step 3: Patch script defaults**

Replace these exact strings:

```python
r"\FEWSNET_forecast_unadjusted_bm_phase_change.csv"
```

and:

```python
DEFAULT_DATA_PATH = r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_forecast_unadjusted_bm_phase_change.csv"
```

with the clean source:

```python
r"\FEWSNET_forecast_unadjusted_bm.csv"
```

and:

```python
DEFAULT_DATA_PATH = r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_forecast_unadjusted_bm.csv"
```

- [ ] **Step 4: Run script-default tests and commit**

Run:

```bash
/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe -m pytest src/tests/test_artifact_source_audit.py -q
```

Expected: PASS.

Commit:

```bash
git add scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py scripts/compare_partitioned_vs_pooled_xgb_k40_nc4.py scripts/analyze_georf_m2_cluster_profiles.py scripts/analyze_georf_false_negative_error_modes.py src/tests/test_artifact_source_audit.py
git commit -m "use clean panel defaults for paper artifacts"
```

### Task 3: Wire Audit Into Reproducibility Verification

**Files:**
- Modify: `scripts/verify_current_results_reproducibility.py`
- Test: `src/tests/test_artifact_source_audit.py`

- [ ] **Step 1: Add audit-gate helper tests**

Append these tests to `src/tests/test_artifact_source_audit.py`:

```python
from scripts.verify_current_results_reproducibility import audit_rows_are_clean


def test_audit_rows_are_clean_rejects_phase_change_and_needs_regeneration():
    rows = [
        {"artifact_group": "06_cluster_profiles", "artifact_type": "final_artifact_group", "status": "clean"},
        {"artifact_group": "12_thresholded_georf_results", "artifact_type": "final_artifact_group", "status": "invalid_phase_change"},
        {"artifact_group": "10_false_negative_error_modes", "artifact_type": "final_artifact_group", "status": "needs_regeneration"},
    ]

    ok, failures = audit_rows_are_clean(rows)

    assert not ok
    assert any("12_thresholded_georf_results" in failure for failure in failures)
    assert any("10_false_negative_error_modes" in failure for failure in failures)


def test_audit_rows_are_clean_accepts_static_and_clean_rows():
    rows = [
        {"artifact_group": "01_main_results", "artifact_type": "final_artifact_group", "status": "clean"},
        {"artifact_group": "05_partition_diagnostics", "artifact_type": "final_artifact_group", "status": "static_or_shape_only"},
    ]

    ok, failures = audit_rows_are_clean(rows)

    assert ok
    assert failures == []
```

- [ ] **Step 2: Run the helper tests and confirm they fail**

Run:

```bash
/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe -m pytest src/tests/test_artifact_source_audit.py::test_audit_rows_are_clean_rejects_phase_change_and_needs_regeneration src/tests/test_artifact_source_audit.py::test_audit_rows_are_clean_accepts_static_and_clean_rows -q
```

Expected: FAIL because `audit_rows_are_clean` is not implemented.

- [ ] **Step 3: Add verifier helper and check**

Modify `scripts/verify_current_results_reproducibility.py`:

```python
from scripts.audit_final_artifact_sources import build_audit_rows, write_audit_outputs
```

Add this helper near the other standalone helper functions:

```python
def audit_rows_are_clean(rows: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for row in rows:
        status = row.get("status")
        artifact_group = row.get("artifact_group", "<unknown>")
        artifact_type = row.get("artifact_type", "<unknown>")
        if status in {"invalid_phase_change", "needs_regeneration"}:
            failures.append(f"{artifact_group} ({artifact_type}) is {status}: {row.get('reason', '')}")
    return not failures, failures
```

Add this verifier function:

```python
def verify_artifact_source_audit(verifier: Verifier) -> None:
    rows = build_audit_rows()
    audit_csv, audit_md = write_audit_outputs(rows, REPO_ROOT / "final_artifacts_in_paper_updated")
    verifier.check(audit_csv.is_file(), "artifact source audit CSV exists")
    verifier.check(audit_md.is_file(), "artifact source audit Markdown exists")

    ok, failures = audit_rows_are_clean(rows)
    verifier.check(ok, "paper-facing artifact source audit has no phase-change or unresolved regeneration rows")
    for failure in failures:
        verifier.fail(f"artifact source audit failure: {failure}")

    thresholded = [
        row for row in rows
        if row.get("artifact_type") == "provider_manifest"
        and "result_partition_k40_compare_GF_thresholded" in row.get("artifact_group", "")
    ]
    verifier.check(len(thresholded) == 3, "all three thresholded GeoRF provider manifests audited")
    for row in thresholded:
        verifier.check(
            str(row.get("source_path", "")).endswith("FEWSNET_forecast_unadjusted_bm.csv"),
            f"{row.get('artifact_group')} uses clean FEWSNET source",
        )
```

Call `verify_artifact_source_audit(verifier)` from `main()` after final artifact checks and before printing the summary.

- [ ] **Step 4: Run helper tests and commit**

Run:

```bash
/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe -m pytest src/tests/test_artifact_source_audit.py -q
```

Expected: PASS.

Commit:

```bash
git add scripts/verify_current_results_reproducibility.py src/tests/test_artifact_source_audit.py
git commit -m "verify final artifact source provenance"
```

### Task 4: Regenerate Thresholded GeoRF Providers From Clean Panel

**Files:**
- Regenerate: `result_partition_k40_compare_GF_thresholded_fs1/`
- Regenerate: `result_partition_k40_compare_GF_thresholded_fs2/`
- Regenerate: `result_partition_k40_compare_GF_thresholded_fs3/`

- [ ] **Step 1: Remove only the invalid thresholded provider outputs**

Run:

```bash
rm -rf result_partition_k40_compare_GF_thresholded_fs1 result_partition_k40_compare_GF_thresholded_fs2 result_partition_k40_compare_GF_thresholded_fs3
```

Expected: only the thresholded provider folders are removed.

- [ ] **Step 2: Regenerate fs1 thresholded provider**

Run:

```bash
/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py \
  --data "C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_forecast_unadjusted_bm.csv" \
  --partition-map result_partition_k40_compare_GF_fs1/refined/cluster_mapping_k40_nc17_general_refined_contig3.csv \
  --out-dir result_partition_k40_compare_GF_thresholded_fs1 \
  --start-month 2021-01 \
  --end-month 2024-12 \
  --train-window 36 \
  --forecasting-scope 1 \
  --lower-model rf \
  --enable-validation-threshold
```

Expected: `result_partition_k40_compare_GF_thresholded_fs1/run_manifest.json` exists and `data_path` ends with `FEWSNET_forecast_unadjusted_bm.csv`.

- [ ] **Step 3: Regenerate fs2 thresholded provider**

Run:

```bash
/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py \
  --data "C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_forecast_unadjusted_bm.csv" \
  --partition-map result_partition_k40_compare_GF_fs2/refined/cluster_mapping_k40_nc17_general_refined_contig3.csv \
  --out-dir result_partition_k40_compare_GF_thresholded_fs2 \
  --start-month 2021-01 \
  --end-month 2024-12 \
  --train-window 36 \
  --forecasting-scope 2 \
  --lower-model rf \
  --enable-validation-threshold
```

Expected: `result_partition_k40_compare_GF_thresholded_fs2/run_manifest.json` exists and `data_path` ends with `FEWSNET_forecast_unadjusted_bm.csv`.

- [ ] **Step 4: Regenerate fs3 thresholded provider**

Run:

```bash
/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py \
  --data "C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_forecast_unadjusted_bm.csv" \
  --partition-map result_partition_k40_compare_GF_fs3/refined/cluster_mapping_k40_nc17_general_refined_contig3.csv \
  --out-dir result_partition_k40_compare_GF_thresholded_fs3 \
  --start-month 2021-01 \
  --end-month 2024-12 \
  --train-window 36 \
  --forecasting-scope 3 \
  --lower-model rf \
  --enable-validation-threshold
```

Expected: `result_partition_k40_compare_GF_thresholded_fs3/run_manifest.json` exists and `data_path` ends with `FEWSNET_forecast_unadjusted_bm.csv`.

- [ ] **Step 5: Check thresholded provider schemas**

Run:

```bash
for d in result_partition_k40_compare_GF_thresholded_fs1 result_partition_k40_compare_GF_thresholded_fs2 result_partition_k40_compare_GF_thresholded_fs3; do
  test -f "$d/metrics_monthly.csv"
  test -f "$d/predictions_monthly.csv"
  test -f "$d/threshold_provenance.csv"
  test -f "$d/run_manifest.json"
done
```

Expected: command exits with status 0.

### Task 5: Regenerate Paper Artifact Groups 06, 10, and 12

**Files:**
- Regenerate: `final_artifacts_in_paper_updated/06_cluster_profiles/`
- Regenerate: `final_artifacts_in_paper_updated/10_false_negative_error_modes/`
- Regenerate: `final_artifacts_in_paper_updated/12_thresholded_georf_results/`

- [ ] **Step 1: Regenerate cluster profiles from clean panel**

Run:

```bash
rm -rf final_artifacts_in_paper_updated/06_cluster_profiles
/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe scripts/analyze_georf_m2_cluster_profiles.py \
  --panel "C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_forecast_unadjusted_bm.csv" \
  --output-dir final_artifacts_in_paper_updated/06_cluster_profiles
```

Expected files:

```text
final_artifacts_in_paper_updated/06_cluster_profiles/georf_m2_cluster_profile_table.csv
final_artifacts_in_paper_updated/06_cluster_profiles/georf_m2_cluster_profile_similarity_matrices.csv
final_artifacts_in_paper_updated/06_cluster_profiles/georf_m2_cluster_profile_cohesion.csv
final_artifacts_in_paper_updated/06_cluster_profiles/georf_m2_cluster_profile_similarity.png
final_artifacts_in_paper_updated/06_cluster_profiles/georf_m2_cluster_profile_note.md
```

- [ ] **Step 2: Regenerate false-negative error modes from clean panel**

Run:

```bash
rm -rf final_artifacts_in_paper_updated/10_false_negative_error_modes
/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe scripts/analyze_georf_false_negative_error_modes.py \
  --data "C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_forecast_unadjusted_bm.csv" \
  --output-dir final_artifacts_in_paper_updated/10_false_negative_error_modes
```

Expected files:

```text
final_artifacts_in_paper_updated/10_false_negative_error_modes/georf_partitioned_false_negative_hotspot_summary.csv
final_artifacts_in_paper_updated/10_false_negative_error_modes/georf_partitioned_false_negative_error_modes.csv
final_artifacts_in_paper_updated/10_false_negative_error_modes/georf_partitioned_false_negative_hotspot_compact_table.md
final_artifacts_in_paper_updated/10_false_negative_error_modes/georf_partitioned_false_negative_note.md
```

- [ ] **Step 3: Regenerate thresholded appendix artifacts**

Run:

```bash
rm -rf final_artifacts_in_paper_updated/12_thresholded_georf_results
/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe scripts/build_georf_thresholded_artifacts.py \
  --source-dir . \
  --output-dir final_artifacts_in_paper_updated/12_thresholded_georf_results
```

Expected files:

```text
final_artifacts_in_paper_updated/12_thresholded_georf_results/georf_thresholded_horizon_metrics.csv
final_artifacts_in_paper_updated/12_thresholded_georf_results/georf_thresholded_monthly_metrics.csv
final_artifacts_in_paper_updated/12_thresholded_georf_results/georf_thresholded_threshold_provenance.csv
final_artifacts_in_paper_updated/12_thresholded_georf_results/georf_thresholded_compact_table.csv
final_artifacts_in_paper_updated/12_thresholded_georf_results/georf_thresholded_compact_table.md
final_artifacts_in_paper_updated/12_thresholded_georf_results/georf_thresholded_note.md
```

- [ ] **Step 4: Generate the audit reports**

Run:

```bash
/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe scripts/audit_final_artifact_sources.py --output-dir final_artifacts_in_paper_updated --fail-on-unclean
```

Expected: PASS after regeneration, and these files exist:

```text
final_artifacts_in_paper_updated/artifact_source_audit.csv
final_artifacts_in_paper_updated/artifact_source_audit.md
```

- [ ] **Step 5: Commit regenerated clean artifacts**

Run:

```bash
git add result_partition_k40_compare_GF_thresholded_fs1 result_partition_k40_compare_GF_thresholded_fs2 result_partition_k40_compare_GF_thresholded_fs3 final_artifacts_in_paper_updated/06_cluster_profiles final_artifacts_in_paper_updated/10_false_negative_error_modes final_artifacts_in_paper_updated/12_thresholded_georf_results final_artifacts_in_paper_updated/artifact_source_audit.csv final_artifacts_in_paper_updated/artifact_source_audit.md
git commit -m "regenerate clean GeoRF paper artifacts"
```

### Task 6: Final Verification and Push-Ready Commit State

**Files:**
- Verify all modified scripts and regenerated artifacts.

- [ ] **Step 1: Run focused test suite**

Run:

```bash
/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe -m pytest \
  src/tests/test_artifact_source_audit.py \
  src/tests/test_georf_validation_threshold.py \
  src/tests/test_georf_thresholded_artifacts.py \
  src/tests/test_georf_m2_cluster_profiles.py \
  src/tests/test_georf_false_negative_error_modes.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run reproducibility verifier**

Run:

```bash
/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe scripts/verify_current_results_reproducibility.py
```

Expected: PASS with no artifact source audit failures.

- [ ] **Step 3: Run phase-change contamination scan**

Run:

```bash
rg --no-ignore -n "FEWSNET_forecast_unadjusted_bm_phase_change|phase_change" \
  final_artifacts_in_paper_updated \
  result_partition_k40_compare_GF_thresholded_fs1 \
  result_partition_k40_compare_GF_thresholded_fs2 \
  result_partition_k40_compare_GF_thresholded_fs3
```

Expected: no output. If output appears in paper-facing generated content, trace the file and regenerate or patch the generator.

- [ ] **Step 4: Inspect final git diff**

Run:

```bash
git status --short
git diff --stat HEAD
```

Expected: clean if all task commits were made, or only intentional uncommitted final verification logs if the worker kept verification artifacts outside git.

- [ ] **Step 5: Push after user approval**

Run only after the user asks to push:

```bash
git push
```

Expected: current branch updates `origin/main`.
