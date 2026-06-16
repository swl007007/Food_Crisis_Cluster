# Fixed-Partition Feature-Exclude Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate `final_artifacts_in_paper_updated/ablation_feature_exclude.xlsx` from Stage 3 GeoRF runs that remove one feature group at a time while holding current partitions fixed.

**Architecture:** Add a small dedicated runner for feature-exclude Stage 3 runs instead of changing `run_partition_k40_comparison_unified.bat`. Add a focused workbook builder that reads the new run root, derives class-1 precision/recall/F1 summaries, and writes the old paper-facing workbook shape. Keep the main no-leak Stage 1/2/3 workflow untouched.

**Tech Stack:** Python 3.12, pandas, openpyxl, existing `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py`, existing contiguity refiner `scripts/refine_partitions_contiguity.py`, pytest.

---

### Task 1: Add Workbook Builder Unit Tests

**Files:**
- Create: `src/tests/test_build_feature_exclude_ablation_workbook.py`
- Create later: `scripts/build_feature_exclude_ablation_workbook.py`

- [ ] **Step 1: Write tests for metrics aggregation and workbook rows**

```python
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from scripts.build_feature_exclude_ablation_workbook import (
    FEATURE_GROUPS,
    build_ablation_rows,
    write_workbook,
)


def _write_metrics(path: Path, partitioned_f1: float, pooled_f1: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"test_month": "2021-02", "model": "partitioned", "precision": 0.7, "recall": 0.5, "f1": partitioned_f1},
            {"test_month": "2021-02", "model": "pooled", "precision": 0.6, "recall": 0.4, "f1": pooled_f1},
            {"test_month": "2021-03", "model": "partitioned", "precision": 0.9, "recall": 0.7, "f1": partitioned_f1 + 0.2},
            {"test_month": "2021-03", "model": "pooled", "precision": 0.8, "recall": 0.6, "f1": pooled_f1 + 0.2},
        ]
    ).to_csv(path, index=False)


def test_build_ablation_rows_uses_new_run_root(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    for group in FEATURE_GROUPS:
        for scope in (1, 2, 3):
            _write_metrics(run_root / group / f"result_partition_k40_compare_GF_fs{scope}" / "metrics_monthly.csv", 0.50 + scope / 10, 0.40 + scope / 10)

    main_by_lag = {
        4: {"partitioned_f1": 0.80, "fewsnet_f1": 0.30},
        8: {"partitioned_f1": 0.70, "fewsnet_f1": 0.20},
        12: {"partitioned_f1": 0.60, "fewsnet_f1": 0.10},
    }
    rows = build_ablation_rows(run_root, main_by_lag)

    assert len(rows) == 18
    assert rows[0]["Feature Group"] == "Weather Exclude"
    assert rows[0]["lag(months)"] == 4
    assert rows[0]["F1"] == 0.7
    assert rows[0]["Pooled F1"] == 0.6
    assert rows[0]["F1 Improvement Percentage"] == (0.7 - 0.6) / 0.6
    assert rows[0]["F1-Compare with main"] == -0.1
    assert rows[0]["F1-compare with baseline"] == 0.4


def test_write_workbook_preserves_paper_shape(tmp_path: Path) -> None:
    rows = []
    for group in FEATURE_GROUPS:
        for lag in (4, 8, 12):
            rows.append(
                {
                    "Feature Group": group.replace("_", " ").title(),
                    "lag(months)": lag,
                    "Precision": 0.7,
                    "Recall": 0.5,
                    "F1": 0.6,
                    "Pooled precision": 0.6,
                    "Pooled recall": 0.4,
                    "Pooled F1": 0.5,
                    "F1 Improvement Percentage": 0.2,
                    "F1-Compare with main": -0.1,
                    "F1-compare with main %": -0.1428571429,
                    "F1-compare with baseline": 0.3,
                }
            )

    out_path = tmp_path / "ablation_feature_exclude.xlsx"
    write_workbook(rows, out_path)

    ws = load_workbook(out_path, data_only=False).active
    assert ws.max_row == 20
    assert ws.max_column == 12
    assert ws["C1"].value == "Split Model"
    assert ws["F1"].value == "Pooled Model(Non-split)"
    assert ws["A3"].value == "Weather Exclude"
    assert ws["B3"].value == 4
```

- [ ] **Step 2: Run tests to verify they fail before implementation**

Run:

```bash
python3 -m pytest src/tests/test_build_feature_exclude_ablation_workbook.py -q
```

Expected: import failure because `scripts.build_feature_exclude_ablation_workbook` does not exist yet.

### Task 2: Implement Workbook Builder

**Files:**
- Create: `scripts/build_feature_exclude_ablation_workbook.py`

- [ ] **Step 1: Implement constants, metric loading, baseline loading, and workbook writing**

Create a CLI with defaults:

```bash
python scripts/build_feature_exclude_ablation_workbook.py \
  --run-root main_ablation_exclude_updated_stage3_fixed_partitions \
  --main-workbook final_artifacts_in_paper_updated/main_month_ind_cont3.xlsx \
  --out final_artifacts_in_paper_updated/ablation_feature_exclude.xlsx
```

Implementation requirements:

- `FEATURE_GROUPS` maps folder names to display labels in this order: weather, agri, conflict, econ, food prices, geographic.
- `SCOPE_TO_LAG = {1: 4, 2: 8, 3: 12}`.
- `summarize_metrics(path)` reads `metrics_monthly.csv`, filters `model == partitioned` and `model == pooled`, and returns mean precision/recall/F1 for each.
- `load_main_by_lag(path)` reads `main_month_ind_cont3.xlsx` and extracts updated main partitioned F1 by lag plus any FEWSNET/baseline F1 row if present; if a FEWSNET/baseline row is absent for lag 12, leave the value blank rather than extrapolating.
- `build_ablation_rows(run_root, main_by_lag)` reads only `run_root`, computes partitioned minus pooled F1 percentage, partitioned minus main F1, percent against main F1, and partitioned minus baseline F1.
- `write_workbook(rows, out_path)` writes two header rows and the 18 ablation data rows with the same 12-column paper layout.

- [ ] **Step 2: Run focused tests**

Run:

```bash
python3 -m pytest src/tests/test_build_feature_exclude_ablation_workbook.py -q
```

Expected: both tests pass.

### Task 3: Add Dedicated Stage 3 Ablation Runner

**Files:**
- Create: `scripts/run_feature_exclude_stage3_fixed_partitions.py`

- [ ] **Step 1: Implement the runner**

The runner must:

- Use Windows Python-compatible paths while being callable from WSL.
- Resolve repo root from `__file__`.
- Use source data folder `C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data`.
- Use datasets `weather_exclude.csv`, `agri_exclude.csv`, `conflict_exclude.csv`, `econ_exclude.csv`, `food_prices_exclude.csv`, `geographic_exclude.csv`.
- Use current GeoRF Stage 2 manifest `GeoRFExperiment/knn_sparsification_results/cluster_mapping_manifest.json`.
- Refine general, m2, m6, and m10 maps into each scope output folder under `refined/` using `--iters 3`.
- Call `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py` with `--month-ind`, `--start-month 2021-01`, `--end-month 2024-12`, `--train-window 36`, and scopes 1, 2, 3.
- Skip an output only when its `metrics_monthly.csv` already exists, so interrupted runs can resume.
- Write a small `ablation_run_manifest.json` under the run root with dataset paths, partition map paths, and completed scope folders.

Default CLI:

```bash
python scripts/run_feature_exclude_stage3_fixed_partitions.py
```

Useful rerun CLI:

```bash
python scripts/run_feature_exclude_stage3_fixed_partitions.py --force weather_exclude --scope 1
```

### Task 4: Smoke Test Runner Without Full Experiment

**Files:**
- Modify: `scripts/run_feature_exclude_stage3_fixed_partitions.py`

- [ ] **Step 1: Add `--dry-run`**

`--dry-run` should print the exact refine and comparison commands without running them.

- [ ] **Step 2: Verify command generation**

Run:

```bash
python3 scripts/run_feature_exclude_stage3_fixed_partitions.py --dry-run --group weather_exclude --scope 1
```

Expected: output contains `weather_exclude.csv`, `--forecasting-scope 1`, `--month-ind`, and all four refined partition map paths.

### Task 5: Run Full Stage 3 Ablation

**Files:**
- Generated: `main_ablation_exclude_updated_stage3_fixed_partitions/**`

- [ ] **Step 1: Launch full run**

Run with the Windows Store Python executable:

```bash
/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe scripts/run_feature_exclude_stage3_fixed_partitions.py
```

Expected: 18 completed Stage 3 outputs, one per feature group and scope.

- [ ] **Step 2: If runtime is long, keep the process alive and monitor only warnings/errors**

Poll the session periodically. If the output shows a warning/error, inspect it immediately. Otherwise allow the run to continue without frequent user-facing noise.

### Task 6: Build and Validate Final Workbook

**Files:**
- Generated: `final_artifacts_in_paper_updated/ablation_feature_exclude.xlsx`

- [ ] **Step 1: Build workbook**

Run:

```bash
/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe scripts/build_feature_exclude_ablation_workbook.py
```

Expected: `final_artifacts_in_paper_updated/ablation_feature_exclude.xlsx` exists.

- [ ] **Step 2: Validate outputs**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
import pandas as pd
from openpyxl import load_workbook

root = Path("main_ablation_exclude_updated_stage3_fixed_partitions")
groups = ["weather_exclude", "agri_exclude", "conflict_exclude", "econ_exclude", "food_prices_exclude", "geographic_exclude"]
for group in groups:
    for scope in (1, 2, 3):
        path = root / group / f"result_partition_k40_compare_GF_fs{scope}" / "metrics_monthly.csv"
        df = pd.read_csv(path)
        assert set(df["model"]) == {"pooled", "partitioned"}
        assert df["test_month"].nunique() >= 1

wb = load_workbook("final_artifacts_in_paper_updated/ablation_feature_exclude.xlsx", data_only=False)
ws = wb.active
assert ws.max_row == 20
assert ws.max_column == 12
print("validated", ws.max_row, ws.max_column)
PY
```

Expected: prints `validated 20 12`.
