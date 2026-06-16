# GeoRF Threshold-Free Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate GeoRF PR-AUC and fixed precision/recall operating-point diagnostics from existing Stage 3 probability outputs.

**Architecture:** Add one focused analysis script under `scripts/`, one focused unittest module under `src/tests/`, and one new artifact folder under `final_artifacts_in_paper_updated/11_threshold_free_metrics/`. The script reads existing `predictions_monthly.csv` files, computes threshold-free and post hoc operating-point metrics for pooled and partitioned GeoRF, writes CSV/Markdown/note outputs, and registers them in the final-artifact README.

**Tech Stack:** Python 3.12, pandas, numpy, scikit-learn `average_precision_score`, unittest, existing GeoRF Stage 3 prediction CSVs.

---

### File Structure

- Create: `scripts/analyze_georf_threshold_free_metrics.py`
  - Loads `result_partition_k40_compare_GF_fs*/predictions_monthly.csv`.
  - Computes PR-AUC, recall at fixed precision, and precision at fixed recall.
  - Writes paper artifacts and reviewer note.
- Create: `src/tests/test_georf_threshold_free_metrics.py`
  - Imports the script module directly and tests metric helper behavior.
- Create directory through script output:
  - `final_artifacts_in_paper_updated/11_threshold_free_metrics/`
- Modify: `final_artifacts_in_paper_updated/README.md`
  - Adds folder index and file descriptions for the new metrics.

### Task 1: Metric Helper Tests

**Files:**
- Create: `src/tests/test_georf_threshold_free_metrics.py`

- [ ] **Step 1: Write failing tests for metric helpers**

Create `src/tests/test_georf_threshold_free_metrics.py` with:

```python
import importlib.util
import math
import unittest
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_georf_threshold_free_metrics.py"
spec = importlib.util.spec_from_file_location("analyze_georf_threshold_free_metrics", SCRIPT_PATH)
threshold_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(threshold_metrics)


class GeoRFThresholdFreeMetricsTests(unittest.TestCase):
    def test_average_precision_matches_sklearn(self):
        y_true = pd.Series([0, 1, 1, 0])
        y_prob = pd.Series([0.10, 0.70, 0.40, 0.20])

        observed = threshold_metrics.pr_auc(y_true, y_prob)

        self.assertAlmostEqual(observed, average_precision_score(y_true, y_prob))

    def test_recall_at_fixed_precision_uses_max_feasible_recall(self):
        y_true = pd.Series([1, 1, 0, 0])
        y_prob = pd.Series([0.90, 0.40, 0.80, 0.10])

        points = threshold_metrics.precision_recall_points(y_true, y_prob)
        observed = threshold_metrics.recall_at_fixed_precision(points, 0.75)

        self.assertAlmostEqual(observed, 0.5)

    def test_precision_at_fixed_recall_uses_max_feasible_precision(self):
        y_true = pd.Series([1, 1, 0, 0])
        y_prob = pd.Series([0.90, 0.40, 0.80, 0.10])

        points = threshold_metrics.precision_recall_points(y_true, y_prob)
        observed = threshold_metrics.precision_at_fixed_recall(points, 1.0)

        self.assertAlmostEqual(observed, 2.0 / 3.0)

    def test_unattainable_operating_points_return_nan(self):
        y_true = pd.Series([1, 0, 0])
        y_prob = pd.Series([0.20, 0.80, 0.70])

        points = threshold_metrics.precision_recall_points(y_true, y_prob)

        self.assertTrue(math.isnan(threshold_metrics.recall_at_fixed_precision(points, 1.01)))
        self.assertTrue(math.isnan(threshold_metrics.precision_at_fixed_recall(points, 1.01)))

    def test_build_compact_table_reports_partitioned_minus_pooled_deltas(self):
        metrics = pd.DataFrame(
            [
                {
                    "scope": "fs1",
                    "forecasting_horizon": "4-month lag",
                    "model": "pooled",
                    "support": 4,
                    "positive_cases": 2,
                    "pr_auc": 0.50,
                    "recall_at_precision_0_75": 0.25,
                    "recall_at_precision_0_80": 0.20,
                    "precision_at_recall_0_50": 0.70,
                    "precision_at_recall_0_60": 0.60,
                },
                {
                    "scope": "fs1",
                    "forecasting_horizon": "4-month lag",
                    "model": "partitioned",
                    "support": 4,
                    "positive_cases": 2,
                    "pr_auc": 0.60,
                    "recall_at_precision_0_75": 0.50,
                    "recall_at_precision_0_80": 0.30,
                    "precision_at_recall_0_50": 0.80,
                    "precision_at_recall_0_60": 0.55,
                },
            ]
        )

        compact = threshold_metrics.build_compact_table(metrics)
        row = compact.iloc[0]

        self.assertAlmostEqual(row["delta_pr_auc"], 0.10)
        self.assertAlmostEqual(row["delta_recall_at_precision_0_75"], 0.25)
        self.assertAlmostEqual(row["delta_precision_at_recall_0_60"], -0.05)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail because the script does not exist**

Run:

```bash
python3 -m unittest src.tests.test_georf_threshold_free_metrics
```

Expected: FAIL with `FileNotFoundError` for `scripts/analyze_georf_threshold_free_metrics.py`.

### Task 2: Threshold-Free Metrics Script

**Files:**
- Create: `scripts/analyze_georf_threshold_free_metrics.py`
- Test: `src/tests/test_georf_threshold_free_metrics.py`

- [ ] **Step 1: Implement the analysis script**

Create `scripts/analyze_georf_threshold_free_metrics.py` with these public helpers and CLI behavior:

```python
#!/usr/bin/env python3
"""Build GeoRF threshold-free and fixed-operating-point diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "final_artifacts_in_paper_updated" / "11_threshold_free_metrics"
HORIZONS = {
    "fs1": "4-month lag",
    "fs2": "8-month lag",
    "fs3": "12-month lag",
}
MODEL_PROB_COLUMNS = {
    "pooled": "y_prob_pooled",
    "partitioned": "y_prob_partitioned",
}
FIXED_PRECISION_TARGETS = (0.75, 0.80)
FIXED_RECALL_TARGETS = (0.50, 0.60)


def _clean_probability_inputs(y_true: pd.Series, y_prob: pd.Series) -> tuple[pd.Series, pd.Series]:
    y = pd.to_numeric(y_true, errors="coerce")
    p = pd.to_numeric(y_prob, errors="coerce")
    valid = y.notna() & p.notna()
    y = y.loc[valid].astype(int)
    p = p.loc[valid].clip(0, 1).astype(float)
    return y.reset_index(drop=True), p.reset_index(drop=True)


def pr_auc(y_true: pd.Series, y_prob: pd.Series) -> float:
    y, p = _clean_probability_inputs(y_true, y_prob)
    if y.empty or int(y.sum()) == 0:
        return np.nan
    return float(average_precision_score(y, p))


def precision_recall_points(y_true: pd.Series, y_prob: pd.Series) -> pd.DataFrame:
    y, p = _clean_probability_inputs(y_true, y_prob)
    if y.empty or int(y.sum()) == 0:
        return pd.DataFrame(columns=["threshold", "precision", "recall", "predicted_positive"])

    rows = []
    positives = int(y.sum())
    for threshold in sorted(p.unique(), reverse=True):
        pred = p >= threshold
        predicted_positive = int(pred.sum())
        if predicted_positive == 0:
            continue
        tp = int(((y == 1) & pred).sum())
        fp = int(((y == 0) & pred).sum())
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(tp / (tp + fp)) if (tp + fp) else np.nan,
                "recall": float(tp / positives) if positives else np.nan,
                "predicted_positive": predicted_positive,
            }
        )
    return pd.DataFrame(rows)


def recall_at_fixed_precision(points: pd.DataFrame, fixed_precision: float) -> float:
    feasible = points[points["precision"] >= fixed_precision]
    if feasible.empty:
        return np.nan
    return float(feasible["recall"].max())


def precision_at_fixed_recall(points: pd.DataFrame, fixed_recall: float) -> float:
    feasible = points[points["recall"] >= fixed_recall]
    if feasible.empty:
        return np.nan
    return float(feasible["precision"].max())
```

Continue the same file with:

```python
def _target_label(value: float) -> str:
    return f"{value:.2f}".replace(".", "_")


def compute_model_metrics(df: pd.DataFrame, prob_col: str) -> dict[str, float]:
    y_true, y_prob = _clean_probability_inputs(df["y_true"], df[prob_col])
    points = precision_recall_points(y_true, y_prob)
    metrics: dict[str, float] = {
        "support": int(len(y_true)),
        "positive_cases": int(y_true.sum()) if not y_true.empty else 0,
        "pr_auc": pr_auc(y_true, y_prob),
    }
    for target in FIXED_PRECISION_TARGETS:
        metrics[f"recall_at_precision_{_target_label(target)}"] = recall_at_fixed_precision(points, target)
    for target in FIXED_RECALL_TARGETS:
        metrics[f"precision_at_recall_{_target_label(target)}"] = precision_at_fixed_recall(points, target)
    return metrics


def load_prediction_files(source_dir: Path, scopes: list[str]) -> pd.DataFrame:
    required = {"FEWSNET_admin_code", "month_start", "y_true", "y_prob_pooled", "y_prob_partitioned"}
    frames = []
    for scope in scopes:
        path = source_dir / f"result_partition_k40_compare_GF_{scope}" / "predictions_monthly.csv"
        df = pd.read_csv(path)
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        df["scope"] = scope
        df["forecasting_horizon"] = HORIZONS.get(scope, scope)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scope, horizon), sub in predictions.groupby(["scope", "forecasting_horizon"], sort=True):
        for model, prob_col in MODEL_PROB_COLUMNS.items():
            rows.append(
                {
                    "scope": scope,
                    "forecasting_horizon": horizon,
                    "model": model,
                    **compute_model_metrics(sub, prob_col),
                }
            )
    return pd.DataFrame(rows)


def build_compact_table(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "pr_auc",
        "recall_at_precision_0_75",
        "recall_at_precision_0_80",
        "precision_at_recall_0_50",
        "precision_at_recall_0_60",
    ]
    rows = []
    for (scope, horizon), sub in metrics.groupby(["scope", "forecasting_horizon"], sort=True):
        pooled = sub[sub["model"] == "pooled"].iloc[0]
        partitioned = sub[sub["model"] == "partitioned"].iloc[0]
        row: dict[str, object] = {"scope": scope, "forecasting_horizon": horizon}
        for metric in metric_columns:
            row[f"pooled_{metric}"] = pooled[metric]
            row[f"partitioned_{metric}"] = partitioned[metric]
            row[f"delta_{metric}"] = partitioned[metric] - pooled[metric]
        rows.append(row)
    return pd.DataFrame(rows)
```

Continue the same file with:

```python
def format_compact_for_paper(compact: pd.DataFrame) -> pd.DataFrame:
    table = compact.copy()
    for column in table.columns:
        if column not in {"scope", "forecasting_horizon"}:
            table[column] = table[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    return table


def write_markdown_table(table: pd.DataFrame, output_path: Path) -> None:
    columns = list(table.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        values = ["" if pd.isna(row[column]) else str(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_note(output_dir: Path) -> Path:
    path = output_dir / "georf_threshold_free_metrics_note.md"
    path.write_text(
        "\n".join(
            [
                "# GeoRF Threshold-Free and Fixed Operating-Point Metrics",
                "",
                "中文审查说明：",
                "",
                "该 appendix 只针对 GeoRF pooled 和 partitioned/local RF 模型。",
                "所有指标都从现有 Stage 3 `y_prob_pooled` 和 `y_prob_partitioned` 概率输出计算，不重跑模型，也不改变主文 binary prediction rule。",
                "PR-AUC 使用 average precision，衡量 crisis probability ranking 的整体 precision-recall 表现。",
                "Recall at fixed precision 和 precision at fixed recall 是 post hoc operating-point diagnostics，用于展示现有概率排序在指定 precision 或 recall 约束下可达到的 tradeoff。",
                "这些指标不表示本文已经进行了 threshold tuning；主结果仍然使用当前 hard predictions 的 precision、recall 和 F1。",
                "如果某个 operating point 不可达到，表中保留空值。",
                "",
                "Appendix text (English):",
                "",
                "We report additional threshold-free and fixed operating-point diagnostics for the GeoRF pooled and partitioned RF models.",
                "All metrics are computed from existing Stage 3 crisis-class probabilities and do not require model retraining or a different threshold-selection procedure.",
                "PR-AUC is computed as average precision and summarizes the probability ranking across the precision-recall curve.",
                "Recall at fixed precision and precision at fixed recall are post hoc operating-point diagnostics showing feasible tradeoffs under the existing probability scores.",
                "The main binary results remain based on the implemented hard predictions; these appendix metrics are complementary ranking and sensitivity diagnostics.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scopes", nargs="+", default=["fs1", "fs2", "fs3"], choices=["fs1", "fs2", "fs3"])
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = load_prediction_files(args.source_dir, args.scopes)
    metrics = build_metric_table(predictions)
    compact = build_compact_table(metrics)
    compact_paper = format_compact_for_paper(compact)

    metrics.to_csv(args.output_dir / "georf_threshold_free_metrics.csv", index=False)
    compact.to_csv(args.output_dir / "georf_threshold_free_metrics_compact_table.csv", index=False)
    write_markdown_table(compact_paper, args.output_dir / "georf_threshold_free_metrics_compact_table.md")
    write_note(args.output_dir)

    print(f"Wrote GeoRF threshold-free metrics to {args.output_dir}")
    print(f"Rows: predictions={len(predictions)}, metrics={len(metrics)}, compact={len(compact)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run focused unit tests**

Run:

```bash
python3 -m unittest src.tests.test_georf_threshold_free_metrics
```

Expected: `Ran 5 tests` and `OK`.

- [ ] **Step 3: Compile the script**

Run:

```bash
python3 -m py_compile scripts/analyze_georf_threshold_free_metrics.py
```

Expected: exit code 0 with no output.

### Task 3: Generate Artifacts

**Files:**
- Create: `final_artifacts_in_paper_updated/11_threshold_free_metrics/georf_threshold_free_metrics.csv`
- Create: `final_artifacts_in_paper_updated/11_threshold_free_metrics/georf_threshold_free_metrics_compact_table.csv`
- Create: `final_artifacts_in_paper_updated/11_threshold_free_metrics/georf_threshold_free_metrics_compact_table.md`
- Create: `final_artifacts_in_paper_updated/11_threshold_free_metrics/georf_threshold_free_metrics_note.md`

- [ ] **Step 1: Run the analysis script**

Run:

```bash
python3 scripts/analyze_georf_threshold_free_metrics.py
```

Expected output:

```text
Wrote GeoRF threshold-free metrics to /mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/2.source_code/Step5_Geo_RF_trial/Food_Crisis_Cluster/final_artifacts_in_paper_updated/11_threshold_free_metrics
Rows: predictions=186567, metrics=6, compact=3
```

- [ ] **Step 2: Inspect the compact table**

Run:

```bash
cat final_artifacts_in_paper_updated/11_threshold_free_metrics/georf_threshold_free_metrics_compact_table.md
```

Expected: one header row and three horizon rows. Confirm columns include:

```text
pooled_pr_auc
partitioned_pr_auc
delta_pr_auc
pooled_recall_at_precision_0_75
partitioned_recall_at_precision_0_75
pooled_precision_at_recall_0_50
partitioned_precision_at_recall_0_50
```

### Task 4: Register README and Verify

**Files:**
- Modify: `final_artifacts_in_paper_updated/README.md`

- [ ] **Step 1: Add the folder index entry**

In `final_artifacts_in_paper_updated/README.md`, add this item after the `10_false_negative_error_modes/` folder index item:

```markdown
- `11_threshold_free_metrics/`: GeoRF PR-AUC and fixed precision/recall
  operating-point diagnostics from existing probability outputs.
```

- [ ] **Step 2: Add the artifact section**

In `final_artifacts_in_paper_updated/README.md`, add this section before `## Reproduction Checks`:

```markdown
## 11 Threshold-Free Metrics

- `11_threshold_free_metrics/georf_threshold_free_metrics.csv`: long-format
  GeoRF pooled and partitioned PR-AUC and fixed operating-point metrics by
  forecasting horizon / lag.
- `11_threshold_free_metrics/georf_threshold_free_metrics_compact_table.csv`:
  compact appendix-ready comparison table with partitioned-minus-pooled deltas.
- `11_threshold_free_metrics/georf_threshold_free_metrics_compact_table.md`:
  Markdown rendering of the compact threshold-free metrics table.
- `11_threshold_free_metrics/georf_threshold_free_metrics_note.md`: Chinese
  reviewer-facing note and English appendix text for PR-AUC and fixed
  precision/recall diagnostics.
```

- [ ] **Step 3: Verify README paths**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
import re
root = Path('final_artifacts_in_paper_updated')
text = (root / 'README.md').read_text(encoding='utf-8')
paths = [m for m in re.findall(r'`([^`]+)`', text) if '/' in m and not m.endswith('/') and not m.startswith('main_')]
missing = [p for p in paths if not (root / p).exists()]
print(f'readme_paths={len(paths)} missing={len(missing)}')
for p in missing:
    print(p)
PY
```

Expected: `missing=0`.

- [ ] **Step 4: Run final verification**

Run:

```bash
python3 -m unittest src.tests.test_georf_threshold_free_metrics
python3 -m py_compile scripts/analyze_georf_threshold_free_metrics.py
git diff --check
```

Expected: unittest reports `Ran 5 tests` and `OK`; py_compile and diff check exit 0.

- [ ] **Step 5: Stage ignored artifacts and commit**

Run:

```bash
git add -A
git add -f final_artifacts_in_paper_updated/11_threshold_free_metrics
git diff --cached --stat
git commit -m "add GeoRF threshold-free metrics"
```

Expected: staged files include the new script, test, README update, and four files under `11_threshold_free_metrics/`.

### Self-Review

- Spec coverage: the plan computes PR-AUC, recall at fixed precision 0.75/0.80, and precision at fixed recall 0.50/0.60 for GeoRF pooled and partitioned models across fs1/fs2/fs3 from existing probability outputs.
- Scope check: the plan does not add GeoDT, GeoXGB, model reruns, threshold tuning, or calibration.
- Output check: the plan creates a new `11_threshold_free_metrics/` folder and registers it in the active final artifact README.
- Validation check: the plan includes focused helper tests, script compilation, README path verification, `git diff --check`, and a final commit.
