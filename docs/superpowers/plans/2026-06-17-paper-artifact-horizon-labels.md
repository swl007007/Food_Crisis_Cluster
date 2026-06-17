# Paper Artifact Horizon Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change paper-facing final artifacts and artifact generators from `4/8/12-month lag` wording to `4/8/12-month horizon` wording without rerunning models or changing metrics.

**Architecture:** Add one small shared label helper for paper-facing horizon display strings, update the scripts that currently hard-code old labels, then run a deterministic relabel pass over existing final artifacts. Regenerate only PNGs whose labels are embedded pixels, and validate that numeric data and allowed technical lag terminology are preserved.

**Tech Stack:** Python 3.12, pandas, openpyxl, matplotlib/geopandas for existing figure scripts, unittest-based tests under `src/tests/`, final artifacts under `final_artifacts_in_paper_updated/`.

---

### File Structure

- Create: `scripts/paper_horizon_labels.py`
  - Owns paper-facing `fs1/fs2/fs3` display labels and exact old-to-new replacements.
  - Does not own implementation variables such as `lag_months` or `ACTIVE_LAGS`.
- Create: `scripts/relabel_final_artifact_horizons.py`
  - Applies label-only replacements to final CSV/Markdown/JSON/XLSX artifacts.
  - Verifies CSV row counts and numeric columns are unchanged.
  - Verifies workbook `Lag Exclude` remains unchanged.
- Create: `src/tests/test_paper_horizon_labels.py`
  - Unit tests for label mapping and the relabeler’s guard behavior.
- Modify these generator scripts:
  - `scripts/plot_monthly_performance_metrics.py`
  - `scripts/plot_seasonal_performance.py`
  - `scripts/create_region_performance_partitioned_pooled_fewsnet.py`
  - `scripts/plot_error_rate_grids.py`
  - `scripts/analyze_georf_probability_uncertainty.py`
  - `scripts/analyze_georf_humanitarian_population_metrics.py`
  - `scripts/analyze_georf_false_negative_error_modes.py`
  - `scripts/analyze_georf_threshold_free_metrics.py`
  - `scripts/build_georf_thresholded_artifacts.py`
  - `scripts/analyze_georf_partition_stability.py`
  - `scripts/plot_global_cluster_map_2x2_refined.py`
  - `scripts/plot_geodt_branch_1_vs_011_locations.py`
  - `scripts/build_feature_exclude_ablation_workbook.py`
  - `other_outputs/generate_table.py`
  - `other_outputs/plot_model_comparison.py`
- Modify final artifacts under `final_artifacts_in_paper_updated/` only through the relabeler or targeted figure regeneration.

---

### Task 1: Shared Horizon Label Helper

**Files:**
- Create: `scripts/paper_horizon_labels.py`
- Create: `src/tests/test_paper_horizon_labels.py`

- [ ] **Step 1: Write the failing label-helper tests**

Add this initial content to `src/tests/test_paper_horizon_labels.py`:

```python
import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "paper_horizon_labels.py"
spec = importlib.util.spec_from_file_location("paper_horizon_labels", SCRIPT_PATH)
labels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labels)


class PaperHorizonLabelTests(unittest.TestCase):
    def test_label_for_scope_uses_horizon_wording(self):
        self.assertEqual(labels.label_for_scope("fs1"), "4-month horizon")
        self.assertEqual(labels.label_for_scope("fs2"), "8-month horizon")
        self.assertEqual(labels.label_for_scope("fs3"), "12-month horizon")

    def test_label_for_scope_preserves_unknown_scope(self):
        self.assertEqual(labels.label_for_scope("fs0"), "fs0")

    def test_replace_paper_horizon_terms_updates_only_display_phrases(self):
        text = (
            "Forecasting horizon / lag: 4-month lag, 8-month lag, 12-month lag. "
            "Forecasting horizon (month lag). Lag Exclude and lagged outcomes remain."
        )

        updated = labels.replace_paper_horizon_terms(text)

        self.assertIn("Forecasting horizon: 4-month horizon, 8-month horizon, 12-month horizon", updated)
        self.assertIn("Forecasting horizon.", updated)
        self.assertIn("Lag Exclude", updated)
        self.assertIn("lagged outcomes", updated)
        self.assertNotIn("4-month lag", updated)
        self.assertNotIn("8-month lag", updated)
        self.assertNotIn("12-month lag", updated)
        self.assertNotIn("horizon / lag", updated.lower())
        self.assertNotIn("month lag)", updated.lower())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest src.tests.test_paper_horizon_labels -v
```

Expected: FAIL with `FileNotFoundError` for `scripts/paper_horizon_labels.py`.

- [ ] **Step 3: Implement the shared helper**

Create `scripts/paper_horizon_labels.py`:

```python
#!/usr/bin/env python3
"""Paper-facing forecasting horizon display labels."""

from __future__ import annotations


HORIZON_MONTHS_BY_SCOPE = {
    "fs1": 4,
    "fs2": 8,
    "fs3": 12,
}

HORIZON_LABELS = {
    scope: f"{months}-month horizon"
    for scope, months in HORIZON_MONTHS_BY_SCOPE.items()
}

OLD_TO_NEW_DISPLAY_REPLACEMENTS = {
    "4-month-lag": "4-month-horizon",
    "8-month-lag": "8-month-horizon",
    "12-month-lag": "12-month-horizon",
    "4-month lag": "4-month horizon",
    "8-month lag": "8-month horizon",
    "12-month lag": "12-month horizon",
    "Forecasting horizon / lag": "Forecasting horizon",
    "forecasting horizon / lag": "forecasting horizon",
    "horizon / lag": "horizon",
    "Forecasting Horizon / Lag": "Forecasting Horizon",
    "Forecasting horizon (month lag)": "Forecasting horizon",
}


def label_for_scope(scope: str) -> str:
    """Return the paper-facing forecasting horizon label for a scope token."""
    return HORIZON_LABELS.get(str(scope), str(scope))


def replace_paper_horizon_terms(text: str) -> str:
    """Replace paper-facing forecast interval labels without touching true lag terms."""
    updated = text
    for old, new in OLD_TO_NEW_DISPLAY_REPLACEMENTS.items():
        updated = updated.replace(old, new)
    return updated
```

- [ ] **Step 4: Run the focused test to verify it passes**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest src.tests.test_paper_horizon_labels -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add scripts/paper_horizon_labels.py src/tests/test_paper_horizon_labels.py
git commit -m "add paper horizon label helper"
```

---

### Task 2: Generator Script Label Updates

**Files:**
- Modify: `scripts/plot_monthly_performance_metrics.py`
- Modify: `scripts/plot_seasonal_performance.py`
- Modify: `scripts/create_region_performance_partitioned_pooled_fewsnet.py`
- Modify: `scripts/plot_error_rate_grids.py`
- Modify: `scripts/analyze_georf_probability_uncertainty.py`
- Modify: `scripts/analyze_georf_humanitarian_population_metrics.py`
- Modify: `scripts/analyze_georf_false_negative_error_modes.py`
- Modify: `scripts/analyze_georf_threshold_free_metrics.py`
- Modify: `scripts/build_georf_thresholded_artifacts.py`
- Modify: `scripts/analyze_georf_partition_stability.py`
- Modify: `scripts/plot_global_cluster_map_2x2_refined.py`
- Modify: `scripts/plot_geodt_branch_1_vs_011_locations.py`
- Modify: `scripts/build_feature_exclude_ablation_workbook.py`
- Modify: `other_outputs/generate_table.py`
- Modify: `other_outputs/plot_model_comparison.py`
- Modify tests that assert old labels:
  - `src/tests/test_georf_probability_uncertainty.py`
  - `src/tests/test_georf_thresholded_artifacts.py`
  - `src/tests/test_georf_humanitarian_population_metrics.py`
  - `src/tests/test_georf_false_negative_error_modes.py`
  - `src/tests/test_georf_threshold_free_metrics.py`
  - `src/tests/test_georf_partition_stability.py`

- [ ] **Step 1: Update tests that currently expect old paper labels**

In the listed tests, replace expected display values only:

```python
"4-month lag" -> "4-month horizon"
"8-month lag" -> "8-month horizon"
"12-month lag" -> "12-month horizon"
"Forecasting horizon / lag" -> "Forecasting horizon"
```

Do not replace `Lag Exclude`, `lagged outcomes`, `lagged non-crisis states`, `lag_months`, or feature-name suffixes.

- [ ] **Step 2: Run the focused tests and verify they fail before script updates**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  src.tests.test_georf_probability_uncertainty \
  src.tests.test_georf_thresholded_artifacts \
  src.tests.test_georf_humanitarian_population_metrics \
  src.tests.test_georf_false_negative_error_modes \
  src.tests.test_georf_threshold_free_metrics \
  src.tests.test_georf_partition_stability \
  -v
```

Expected: FAIL where scripts still emit `*-month lag` or `Forecasting horizon / lag`.

- [ ] **Step 3: Import the shared helper where scripts produce paper labels**

For scripts under `scripts/`, add the import near existing local imports:

```python
from paper_horizon_labels import HORIZON_LABELS, HORIZON_MONTHS_BY_SCOPE, label_for_scope
```

For scripts under `other_outputs/`, add a repo-root import shim near the top:

```python
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from paper_horizon_labels import HORIZON_MONTHS_BY_SCOPE, label_for_scope
```

If a script already defines `REPO_ROOT`, reuse that variable and only add the `SCRIPTS_DIR` block.

- [ ] **Step 4: Replace label dictionaries and label functions**

Apply these exact patterns:

```python
HORIZONS = {
    "fs1": "4-month lag",
    "fs2": "8-month lag",
    "fs3": "12-month lag",
}
```

becomes:

```python
HORIZONS = HORIZON_LABELS
```

Any function like:

```python
def scope_label(scope: str) -> str:
    horizon = SCOPE_TO_HORIZON.get(str(scope))
    return f"{horizon}-month lag" if horizon is not None else str(scope)
```

becomes:

```python
def scope_label(scope: str) -> str:
    return label_for_scope(str(scope))
```

Keep numeric mappings when they are used as month counts:

```python
SCOPE_TO_HORIZON = {"fs1": 4, "fs2": 8, "fs3": 12}
```

may remain as numeric metadata, but paper display strings must use `label_for_scope()`.

- [ ] **Step 5: Update explanatory strings that describe the FEWSNET baseline display proxy**

In `scripts/plot_monthly_performance_metrics.py` and `scripts/create_region_performance_partitioned_pooled_fewsnet.py`, replace these paper-facing phrases:

```python
"FEWSNET baseline (8-month lag reused for 12-month lag)"
"Reuse FEWSNET 8-month baseline values for the 12-month lag as an explicitly labeled diagnostic."
"FEWSNET 8-month baseline reused for 12-month lag"
"FEWSNET has no native 12-month baseline; 8-month values are reused for the 12-month lag as a labeled comparison proxy."
"FEWSNET has no native 12-month baseline and is not plotted for the 12-month lag."
"8-month lag"
```

with:

```python
"FEWSNET baseline (8-month horizon reused for 12-month horizon)"
"Reuse FEWSNET 8-month baseline values for the 12-month horizon as an explicitly labeled diagnostic."
"FEWSNET 8-month baseline reused for 12-month horizon"
"FEWSNET has no native 12-month baseline; 8-month values are reused for the 12-month horizon as a labeled comparison proxy."
"FEWSNET has no native 12-month baseline and is not plotted for the 12-month horizon."
"8-month horizon"
```

Leave `12_month_fewsnet_source` as a JSON key if present; key names are machine-readable provenance, not paper display text.

- [ ] **Step 6: Update figure titles and axis labels**

Replace paper-facing strings:

```python
"Forecasting horizon / lag" -> "Forecasting horizon"
"4-month lag Global Refined Partition Mapping (k=40)" -> "4-month horizon Global Refined Partition Mapping (k=40)"
"4-month lag mappings:" -> "4-month horizon mappings:"
"Global spatial locations of GeoDT branch-specific local DecisionTree comparison pair (2024-10, 4-month lag)" -> "Global spatial locations of GeoDT branch-specific local DecisionTree comparison pair (2024-10, 4-month horizon)"
```

Do not change comments or variables that describe training lag mechanics.

- [ ] **Step 7: Update workbook header generation**

In `scripts/build_feature_exclude_ablation_workbook.py`, change:

```python
"Forecasting horizon (month lag)"
```

to:

```python
"Forecasting horizon"
```

Keep `SCOPE_TO_LAG` if it is used to parse the existing input columns, and keep the ablation row label `Lag Exclude`.

- [ ] **Step 8: Run focused tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  src.tests.test_paper_horizon_labels \
  src.tests.test_georf_probability_uncertainty \
  src.tests.test_georf_thresholded_artifacts \
  src.tests.test_georf_humanitarian_population_metrics \
  src.tests.test_georf_false_negative_error_modes \
  src.tests.test_georf_threshold_free_metrics \
  src.tests.test_georf_partition_stability \
  -v
```

Expected: PASS.

- [ ] **Step 9: Scan generator scripts for forbidden display labels**

Run:

```bash
rg -n "4-month lag|8-month lag|12-month lag|Forecasting horizon / lag|Forecasting horizon \\(month lag\\)" scripts other_outputs -g '*.py'
```

Expected: no hits in paper-facing display strings. If hits remain in comments describing historical lag mechanics, review manually and either keep them with a note in final reporting or revise them if they are presentation text.

- [ ] **Step 10: Commit**

Run:

```bash
git add scripts other_outputs src/tests/test_georf_probability_uncertainty.py src/tests/test_georf_thresholded_artifacts.py src/tests/test_georf_humanitarian_population_metrics.py src/tests/test_georf_false_negative_error_modes.py src/tests/test_georf_threshold_free_metrics.py src/tests/test_georf_partition_stability.py
git commit -m "use horizon labels in paper generators"
```

---

### Task 3: Deterministic Final Artifact Relabeler

**Files:**
- Create: `scripts/relabel_final_artifact_horizons.py`
- Modify: `src/tests/test_paper_horizon_labels.py`

- [ ] **Step 1: Add relabeler tests**

Append these tests to `src/tests/test_paper_horizon_labels.py`:

```python
class FinalArtifactRelabelerTests(unittest.TestCase):
    def test_is_allowed_remaining_lag_line_accepts_lagged_covariates(self):
        self.assertTrue(labels.is_allowed_remaining_lag_line("Lag Exclude"))
        self.assertTrue(labels.is_allowed_remaining_lag_line("lagged outcomes from prior FEWSNET phase columns"))
        self.assertTrue(labels.is_allowed_remaining_lag_line("feature suffix _lag4m"))
        self.assertFalse(labels.is_allowed_remaining_lag_line("4-month lag"))

    def test_forbidden_paper_lag_terms_reports_only_display_labels(self):
        text = "4-month lag\nLag Exclude\nlagged non-crisis states\n12-month horizon\n"

        hits = labels.forbidden_paper_lag_terms(text)

        self.assertEqual(hits, ["4-month lag"])
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest src.tests.test_paper_horizon_labels -v
```

Expected: FAIL with missing `is_allowed_remaining_lag_line` and `forbidden_paper_lag_terms`.

- [ ] **Step 3: Extend `scripts/paper_horizon_labels.py` with scan helpers**

Add this code below `replace_paper_horizon_terms`:

```python
FORBIDDEN_PAPER_LAG_TERMS = (
    "4-month lag",
    "8-month lag",
    "12-month lag",
    "Forecasting horizon / lag",
    "forecasting horizon / lag",
    "horizon / lag",
    "Forecasting horizon (month lag)",
)

ALLOWED_REMAINING_LAG_MARKERS = (
    "Lag Exclude",
    "lagged outcome",
    "lagged outcomes",
    "lagged non-crisis",
    "lagged covariate",
    "lagged covariates",
    "_lag",
    "lag_months",
    "ACTIVE_LAGS",
    "forecasting_scope_to_lag",
    "SCOPE_TO_LAG",
    "FS_TO_LAG",
)


def is_allowed_remaining_lag_line(line: str) -> bool:
    """Return whether a remaining lag mention is a true technical or covariate term."""
    return any(marker in line for marker in ALLOWED_REMAINING_LAG_MARKERS)


def forbidden_paper_lag_terms(text: str) -> list[str]:
    """Return forbidden paper-facing lag terms present in text."""
    hits: list[str] = []
    for term in FORBIDDEN_PAPER_LAG_TERMS:
        if term in text:
            hits.append(term)
    return hits
```

- [ ] **Step 4: Run helper tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest src.tests.test_paper_horizon_labels -v
```

Expected: PASS.

- [ ] **Step 5: Create the artifact relabeler**

Create `scripts/relabel_final_artifact_horizons.py`:

```python
#!/usr/bin/env python3
"""Relabel final paper artifacts from lag wording to horizon wording."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import openpyxl

from paper_horizon_labels import (
    forbidden_paper_lag_terms,
    is_allowed_remaining_lag_line,
    replace_paper_horizon_terms,
)


TEXT_SUFFIXES = {".csv", ".md", ".json"}
WORKBOOK_SUFFIXES = {".xlsx"}


@dataclass(frozen=True)
class RelabelResult:
    path: Path
    changed: bool
    details: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="")


def _csv_numeric_signature(path: Path) -> tuple[int, dict[str, list[str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    numeric_columns: dict[str, list[str]] = {}
    for name in fieldnames:
        values: list[str] = []
        numeric = True
        for row in rows:
            value = row.get(name, "")
            if value == "":
                values.append(value)
                continue
            try:
                float(value)
            except ValueError:
                numeric = False
                break
            values.append(value)
        if numeric:
            numeric_columns[name] = values
    return len(rows), numeric_columns


def relabel_text_file(path: Path, dry_run: bool) -> RelabelResult:
    before = _read_text(path)
    before_signature = _csv_numeric_signature(path) if path.suffix.lower() == ".csv" else None
    after = replace_paper_horizon_terms(before)
    if before == after:
        return RelabelResult(path, False, "no display-label changes")
    if before_signature is not None:
        tmp_path = path.with_suffix(path.suffix + ".relabel_tmp")
        try:
            _write_text(tmp_path, after)
            after_signature = _csv_numeric_signature(tmp_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        if before_signature != after_signature:
            raise ValueError(f"{path} numeric CSV signature changed")
    if not dry_run:
        _write_text(path, after)
    return RelabelResult(path, True, "updated display labels")


def relabel_workbook(path: Path, dry_run: bool) -> RelabelResult:
    workbook = openpyxl.load_workbook(path)
    changed_cells: list[str] = []
    lag_exclude_seen = False
    for sheet in workbook.worksheets:
        for row in sheet.iter_rows():
            for cell in row:
                value = cell.value
                if value == "Lag Exclude":
                    lag_exclude_seen = True
                if isinstance(value, str):
                    updated = replace_paper_horizon_terms(value)
                    if updated != value:
                        cell.value = updated
                        changed_cells.append(f"{sheet.title}!{cell.coordinate}")
    if not lag_exclude_seen:
        raise ValueError(f"{path} did not contain expected Lag Exclude row")
    if changed_cells and not dry_run:
        workbook.save(path)
    return RelabelResult(path, bool(changed_cells), ", ".join(changed_cells) or "no display-label changes")


def iter_artifacts(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES | WORKBOOK_SUFFIXES:
            yield path


def scan_for_forbidden_terms(root: Path) -> list[str]:
    failures: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = _read_text(path)
        for line_number, line in enumerate(text.splitlines(), start=1):
            hits = forbidden_paper_lag_terms(line)
            if hits and not is_allowed_remaining_lag_line(line):
                failures.append(f"{path}:{line_number}: {', '.join(hits)}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "final_artifacts_in_paper_updated",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    results: list[RelabelResult] = []
    for path in iter_artifacts(args.root):
        if path.suffix.lower() in TEXT_SUFFIXES:
            results.append(relabel_text_file(path, args.dry_run))
        elif path.suffix.lower() in WORKBOOK_SUFFIXES:
            results.append(relabel_workbook(path, args.dry_run))

    for result in results:
        if result.changed:
            print(f"UPDATED {result.path}: {result.details}")

    failures = scan_for_forbidden_terms(args.root)
    if failures:
        print("Forbidden paper-facing lag labels remain:", file=sys.stderr)
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1

    print(f"Checked {len(results)} artifacts; changed {sum(result.changed for result in results)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run relabeler dry-run**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/relabel_final_artifact_horizons.py --dry-run
```

Expected: It prints `UPDATED ...` lines and exits 0, proving the replacements are deterministic and no forbidden final-artifact terms remain after simulated replacement.

- [ ] **Step 7: Commit**

Run:

```bash
git add scripts/paper_horizon_labels.py scripts/relabel_final_artifact_horizons.py src/tests/test_paper_horizon_labels.py
git commit -m "add final artifact horizon relabeler"
```

---

### Task 4: Apply Text, CSV, JSON, and Workbook Relabels

**Files:**
- Modify: `final_artifacts_in_paper_updated/**/*.csv`
- Modify: `final_artifacts_in_paper_updated/**/*.md`
- Modify: `final_artifacts_in_paper_updated/**/*.json`
- Modify: `final_artifacts_in_paper_updated/01_main_results/main_month_ind_cont3.xlsx`
- Modify: `final_artifacts_in_paper_updated/01_main_results/ablation_feature_exclude.xlsx`

- [ ] **Step 1: Capture pre-change final artifact signature**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
from pathlib import Path
import csv
import json
root = Path("final_artifacts_in_paper_updated")
summary = {}
for path in sorted(root.rglob("*.csv")):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        summary[str(path)] = {"rows": len(rows), "columns": reader.fieldnames}
Path(".tmp").mkdir(exist_ok=True)
Path(".tmp/paper_horizon_csv_signature_before.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(f"wrote {len(summary)} CSV signatures")
PY
```

Expected: A `.tmp/paper_horizon_csv_signature_before.json` file is written.

- [ ] **Step 2: Apply the relabeler**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/relabel_final_artifact_horizons.py
```

Expected: The command prints updated artifact paths and exits 0.

- [ ] **Step 3: Verify CSV row and column signatures are unchanged**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
from pathlib import Path
import csv
import json
before = json.loads(Path(".tmp/paper_horizon_csv_signature_before.json").read_text(encoding="utf-8"))
after = {}
for path in sorted(Path("final_artifacts_in_paper_updated").rglob("*.csv")):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        after[str(path)] = {"rows": len(rows), "columns": reader.fieldnames}
if before != after:
    missing = sorted(set(before) - set(after))
    extra = sorted(set(after) - set(before))
    changed = sorted(path for path in set(before) & set(after) if before[path] != after[path])
    raise SystemExit(f"CSV signature changed: missing={missing}, extra={extra}, changed={changed[:20]}")
print(f"CSV row/column signatures unchanged for {len(after)} files")
PY
```

Expected: `CSV row/column signatures unchanged...`.

- [ ] **Step 4: Verify workbook labels**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
from pathlib import Path
from openpyxl import load_workbook
paths = [
    Path("final_artifacts_in_paper_updated/01_main_results/main_month_ind_cont3.xlsx"),
    Path("final_artifacts_in_paper_updated/01_main_results/ablation_feature_exclude.xlsx"),
]
for path in paths:
    wb = load_workbook(path, read_only=True, data_only=False)
    strings = []
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, str):
                    strings.append(cell.value)
    if "Forecasting horizon" not in strings:
        raise SystemExit(f"{path} missing Forecasting horizon")
    if any("Forecasting horizon (month lag)" in value for value in strings):
        raise SystemExit(f"{path} still has old workbook header")
    if path.name == "ablation_feature_exclude.xlsx" and "Lag Exclude" not in strings:
        raise SystemExit(f"{path} lost Lag Exclude row")
    print(f"{path}: workbook labels OK")
PY
```

Expected: both workbooks print `workbook labels OK`.

- [ ] **Step 5: Commit**

Run:

```bash
git add final_artifacts_in_paper_updated
git commit -m "relabel final artifact tables as horizons"
```

---

### Task 5: Regenerate Figures with Embedded Labels

**Files:**
- Modify: final PNGs under `final_artifacts_in_paper_updated/01_main_results/`
- Modify: final PNGs under `final_artifacts_in_paper_updated/04_error_analysis/`
- Modify: final PNGs under `final_artifacts_in_paper_updated/05_partition_diagnostics/`
- Modify: final PNGs under `final_artifacts_in_paper_updated/07_probability_uncertainty/`
- Modify: final PNGs under `final_artifacts_in_paper_updated/09_humanitarian_metrics/`
- Modify: final PNGs under `final_artifacts_in_paper_updated/08_geodt_diagnostics/` if the source script can regenerate the current final filename

- [ ] **Step 1: Capture current PNG dimensions**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
from pathlib import Path
from PIL import Image
import json
paths = [
    Path("final_artifacts_in_paper_updated/01_main_results/georf_monthly_performance.png"),
    Path("final_artifacts_in_paper_updated/01_main_results/global_cluster_map_2x2_georf_refined.png"),
    Path("final_artifacts_in_paper_updated/01_main_results/global_cluster_map_2x2_geodt_refined.png"),
    Path("final_artifacts_in_paper_updated/04_error_analysis/error_rate_seasonal_3x3.png"),
    Path("final_artifacts_in_paper_updated/04_error_analysis/error_rate_seasonal_3x3_crisis.png"),
    Path("final_artifacts_in_paper_updated/04_error_analysis/error_rate_seasonal_3x3_noncrisis.png"),
    Path("final_artifacts_in_paper_updated/05_partition_diagnostics/georf_stage1_partition_stability.png"),
    Path("final_artifacts_in_paper_updated/07_probability_uncertainty/georf_probability_reliability.png"),
    Path("final_artifacts_in_paper_updated/09_humanitarian_metrics/georf_humanitarian_population_bars.png"),
    Path("final_artifacts_in_paper_updated/08_geodt_diagnostics/geodt_branch_1_vs_001_locations_2024-10_fs1_global.png"),
]
summary = {}
for path in paths:
    if path.exists():
        with Image.open(path) as image:
            summary[str(path)] = {"size": image.size}
Path(".tmp").mkdir(exist_ok=True)
Path(".tmp/paper_horizon_png_dimensions_before.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"captured {len(summary)} PNG dimensions")
PY
```

Expected: Dimensions are captured. If PIL is unavailable, use `python3 -c "import matplotlib.image as mpimg"` as the fallback in the implementation session.

- [ ] **Step 2: Regenerate monthly performance figure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/plot_monthly_performance_metrics.py \
  --model georf \
  --output-dir final_artifacts_in_paper_updated/01_main_results
```

Expected: `final_artifacts_in_paper_updated/01_main_results/georf_monthly_performance.png` and its manifest are regenerated with `*-month horizon` display labels.

- [ ] **Step 3: Regenerate season and region tables/figures if their scripts own existing outputs**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/plot_seasonal_performance.py \
  --output-dir final_artifacts_in_paper_updated/01_main_results
```

Expected: `table1_season_performance.csv` keeps the same row count and now uses `*-month horizon`.

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/create_region_performance_partitioned_pooled_fewsnet.py \
  --output-dir final_artifacts_in_paper_updated/01_main_results
```

Expected: region performance CSVs keep the same row count and now use `*-month horizon`.

- [ ] **Step 4: Regenerate error-rate grids**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/plot_error_rate_grids.py \
  --output-dir final_artifacts_in_paper_updated/04_error_analysis
```

Expected: `error_rate_seasonal_3x3.png`, `error_rate_seasonal_3x3_crisis.png`, and `error_rate_seasonal_3x3_noncrisis.png` are regenerated with horizon labels.

- [ ] **Step 5: Regenerate analysis-derived figures and compact tables**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/analyze_georf_partition_stability.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/analyze_georf_probability_uncertainty.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/analyze_georf_humanitarian_population_metrics.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/analyze_georf_false_negative_error_modes.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/analyze_georf_threshold_free_metrics.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/build_georf_thresholded_artifacts.py
```

Expected: the corresponding final artifact sections are regenerated, metrics remain unchanged, and display labels use `*-month horizon`.

- [ ] **Step 6: Regenerate global cluster maps**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/plot_global_cluster_map_2x2_refined.py \
  --output-dir final_artifacts_in_paper_updated/01_main_results
```

Expected: `global_cluster_map_2x2_georf_refined.png` and `global_cluster_map_2x2_geodt_refined.png` use `4-month horizon` in titles.

- [ ] **Step 7: Regenerate or inspect GeoDT branch-location diagnostic**

First inspect the script CLI:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/plot_geodt_branch_1_vs_011_locations.py --help
```

If it supports the current output folder and branch pair, run it with output under `final_artifacts_in_paper_updated/08_geodt_diagnostics/`. If the script is for branch `011` while the final artifact is `001`, do not force a mismatched regeneration; instead patch the script label for future reruns and leave the existing final PNG unchanged unless a matching generator is found.

Expected: no mismatched branch-pair figure overwrites.

- [ ] **Step 8: Verify PNG dimensions still exist and are nonzero**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
from pathlib import Path
from PIL import Image
import json
before_path = Path(".tmp/paper_horizon_png_dimensions_before.json")
before = json.loads(before_path.read_text(encoding="utf-8")) if before_path.exists() else {}
for raw_path, old in before.items():
    path = Path(raw_path)
    if not path.exists():
        raise SystemExit(f"missing regenerated PNG: {path}")
    with Image.open(path) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        raise SystemExit(f"invalid PNG dimensions: {path} {width}x{height}")
    print(f"{path}: {width}x{height} (before {old['size']})")
PY
```

Expected: every listed existing PNG has positive dimensions.

- [ ] **Step 9: Commit**

Run:

```bash
git add final_artifacts_in_paper_updated scripts
git commit -m "regenerate paper figures with horizon labels"
```

---

### Task 6: Final Verification

**Files:**
- Read-only validation across `final_artifacts_in_paper_updated/`, `scripts/`, `other_outputs/`, and `src/tests/`

- [ ] **Step 1: Run all affected tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  src.tests.test_paper_horizon_labels \
  src.tests.test_georf_probability_uncertainty \
  src.tests.test_georf_thresholded_artifacts \
  src.tests.test_georf_humanitarian_population_metrics \
  src.tests.test_georf_false_negative_error_modes \
  src.tests.test_georf_threshold_free_metrics \
  src.tests.test_georf_partition_stability \
  -v
```

Expected: PASS.

- [ ] **Step 2: Run the final artifact relabeler in dry-run mode**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/relabel_final_artifact_horizons.py --dry-run
```

Expected: exits 0. It may print no `UPDATED` lines if all artifacts are already relabeled.

- [ ] **Step 3: Scan final artifacts for forbidden paper-facing labels**

Run:

```bash
rg -n "4-month lag|8-month lag|12-month lag|month lag|horizon / lag|forecasting horizon / lag" final_artifacts_in_paper_updated
```

Expected: no hits for forecast-horizon display labels. Hits containing `Lag Exclude`, `lagged outcomes`, `lagged non-crisis states`, or feature suffixes are allowed only if manually reviewed and listed in the final report.

- [ ] **Step 4: Scan scripts for old paper-facing labels**

Run:

```bash
rg -n "4-month lag|8-month lag|12-month lag|Forecasting horizon / lag|Forecasting horizon \\(month lag\\)" scripts other_outputs -g '*.py'
```

Expected: no hits in display strings. Technical lag mechanics may remain under names like `lag_months`, `SCOPE_TO_LAG`, `FS_TO_LAG`, and `ACTIVE_LAGS`.

- [ ] **Step 5: Verify workbook acceptance criteria**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
from pathlib import Path
from openpyxl import load_workbook
for path in [
    Path("final_artifacts_in_paper_updated/01_main_results/main_month_ind_cont3.xlsx"),
    Path("final_artifacts_in_paper_updated/01_main_results/ablation_feature_exclude.xlsx"),
]:
    wb = load_workbook(path, read_only=True, data_only=False)
    values = []
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, str):
                    values.append(cell.value)
    assert "Forecasting horizon" in values, path
    assert "Forecasting horizon (month lag)" not in values, path
    if path.name == "ablation_feature_exclude.xlsx":
        assert "Lag Exclude" in values, path
    print(f"{path}: OK")
PY
```

Expected: both workbooks print `OK`.

- [ ] **Step 6: Run reproducibility bundle checker if available**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_current_results_reproducibility.py
```

Expected: PASS. If it fails only because it asserts old display labels, update the checker wording to horizon labels and rerun.

- [ ] **Step 7: Review git diff for accidental model/data changes**

Run:

```bash
git diff --stat HEAD~3..HEAD
git diff --name-only HEAD~3..HEAD
```

Expected: changed files are limited to label helper/relabeler/tests, paper generators, and final paper artifacts. There should be no Stage 1/2/3 model output reruns outside `final_artifacts_in_paper_updated/`.

- [ ] **Step 8: Final commit if verification changed any files**

If verification required small checker/test wording fixes, run:

```bash
git add scripts src/tests
git commit -m "verify paper horizon label relabel"
```

Expected: no commit is needed if verification is read-only and the worktree is already clean.

---

### Implementation Notes

- Do not change `config.py`, `src/utils/lag_schedules.py`, `src/preprocess/preprocess.py`, or model entrypoints for this task.
- Do not rename files or directories containing `fs1`, `fs2`, or `fs3`.
- Do not change `Lag Exclude`; it is a feature-ablation condition.
- Do not run `.bat` Stage 1/2/3 workflows for this label-only update.
- If a figure generator cannot reproduce the exact final PNG without changing branch pair, data source, or output contract, patch the script label for future runs and leave the current PNG unchanged with that limitation reported.
