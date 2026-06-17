# GeoRF Validation Max-F1 Threshold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a GeoRF Stage 3 `partitioned_thresholded` evaluation variant that selects a class-1 probability threshold on validation data by max F1 and applies it to held-out target-month predictions.

**Architecture:** Extend `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py` with deterministic validation-threshold helpers, CLI flags, threshold provenance output, and an additional metrics/prediction variant while preserving existing pooled and partitioned outputs. Add a small artifact builder that summarizes the new thresholded result folders into `final_artifacts_in_paper_updated/12_thresholded_georf_results/` without overwriting the existing 01-11 artifact groups.

**Tech Stack:** Python 3.12, pandas, numpy, scikit-learn RandomForest already used by Stage 3, existing GeoRF rolling split utilities, unittest.

---

### File Structure

- Modify: `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py`
  - Add validation threshold helper functions.
  - Add `--enable-validation-threshold`, `--threshold-validation-months`, and threshold-bound CLI flags.
  - Add `partitioned_thresholded` metrics rows and prediction columns when the flag is enabled.
  - Write `threshold_provenance.csv` and manifest threshold settings.
- Create: `src/tests/test_georf_validation_threshold.py`
  - Focused unit tests for threshold selection, tie-breaking, fallback, binary conversion, and temporal validation split.
- Create: `scripts/build_georf_thresholded_artifacts.py`
  - Reads the three thresholded Stage 3 result folders and writes compact paper-facing outputs under `12_thresholded_georf_results/`.
- Create: `src/tests/test_georf_thresholded_artifacts.py`
  - Focused tests for compact table generation.
- Modify: `final_artifacts_in_paper_updated/README.md`
  - Register the new `12_thresholded_georf_results/` folder after it is generated.

### Task 1: Add Validation-Threshold Unit Tests

**Files:**
- Create: `src/tests/test_georf_validation_threshold.py`
- Modify later: `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py`

- [ ] **Step 1: Write failing tests for threshold helpers**

Create `src/tests/test_georf_validation_threshold.py`:

```python
import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "compare_partitioned_vs_pooled_rf_k40_nc4.py"
spec = importlib.util.spec_from_file_location("compare_partitioned_vs_pooled_rf_k40_nc4", SCRIPT_PATH)
stage3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage3)


class GeoRFValidationThresholdTests(unittest.TestCase):
    def test_apply_probability_threshold_uses_greater_equal(self):
        observed = stage3.apply_probability_threshold(np.array([0.49, 0.50, 0.51]), 0.50)
        np.testing.assert_array_equal(observed, np.array([0, 1, 1]))

    def test_select_max_f1_threshold_returns_best_validation_threshold(self):
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.90, 0.40, 0.80, 0.10])

        result = stage3.select_max_f1_threshold(y_true, y_prob, bounds=(0.05, 0.95), default_threshold=0.5)

        self.assertAlmostEqual(result["selected_threshold"], 0.4)
        self.assertAlmostEqual(result["validation_precision"], 2.0 / 3.0)
        self.assertAlmostEqual(result["validation_recall"], 1.0)
        self.assertAlmostEqual(result["validation_f1"], 0.8)
        self.assertEqual(result["fallback_reason"], "")

    def test_select_max_f1_threshold_tie_breaks_to_higher_threshold(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.90, 0.80, 0.70, 0.60])

        result = stage3.select_max_f1_threshold(
            y_true,
            y_prob,
            candidate_thresholds=np.array([0.7, 0.6]),
            bounds=(0.05, 0.95),
            default_threshold=0.5,
        )

        self.assertAlmostEqual(result["selected_threshold"], 0.7)
        self.assertAlmostEqual(result["validation_f1"], 0.8)

    def test_select_max_f1_threshold_falls_back_without_positive_cases(self):
        y_true = np.array([0, 0, 0])
        y_prob = np.array([0.20, 0.40, 0.80])

        result = stage3.select_max_f1_threshold(y_true, y_prob, default_threshold=0.5)

        self.assertAlmostEqual(result["selected_threshold"], 0.5)
        self.assertTrue(math.isnan(result["validation_f1"]))
        self.assertEqual(result["fallback_reason"], "no_validation_positive_cases")

    def test_split_training_validation_by_dates_uses_latest_calendar_months(self):
        X = np.arange(12).reshape(12, 1)
        y = np.arange(12)
        groups = np.arange(12) % 2
        dates = pd.to_datetime(
            [
                "2020-01-01",
                "2020-01-15",
                "2020-02-01",
                "2020-02-15",
                "2020-03-01",
                "2020-03-15",
                "2020-04-01",
                "2020-04-15",
                "2020-05-01",
                "2020-05-15",
                "2020-06-01",
                "2020-06-15",
            ]
        )

        split = stage3.split_training_validation_by_dates(X, y, groups, dates, validation_months=2)

        np.testing.assert_array_equal(split["X_fit"].ravel(), np.arange(8))
        np.testing.assert_array_equal(split["X_val"].ravel(), np.arange(8, 12))
        np.testing.assert_array_equal(split["y_val"], np.arange(8, 12))
        np.testing.assert_array_equal(split["group_val"], np.array([0, 1, 0, 1]))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python3 -m unittest src.tests.test_georf_validation_threshold
```

Expected: FAIL because `apply_probability_threshold`, `select_max_f1_threshold`, and `split_training_validation_by_dates` do not exist yet.

### Task 2: Implement Threshold Helpers

**Files:**
- Modify: `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py`
- Test: `src/tests/test_georf_validation_threshold.py`

- [ ] **Step 1: Add constants near existing defaults**

Add after `PARTITION_UNMAPPED_THRESHOLD_PCT = 2.0`:

```python
DEFAULT_THRESHOLD_VALIDATION_MONTHS = 6
DEFAULT_THRESHOLD_LOWER_BOUND = 0.05
DEFAULT_THRESHOLD_UPPER_BOUND = 0.95
DEFAULT_PARTITIONED_THRESHOLD = 0.5
```

- [ ] **Step 2: Add threshold helper functions after `compute_binary_metrics`**

Add these helpers after the existing metric function block:

```python
def apply_probability_threshold(y_prob: np.ndarray, threshold: float) -> np.ndarray:
    """Convert class-1 probabilities to binary labels using >= threshold."""
    probabilities = np.asarray(y_prob, dtype=float)
    return (probabilities >= float(threshold)).astype(int)


def candidate_thresholds_from_probabilities(
    y_prob: np.ndarray,
    bounds: Tuple[float, float] = (DEFAULT_THRESHOLD_LOWER_BOUND, DEFAULT_THRESHOLD_UPPER_BOUND),
) -> np.ndarray:
    """Return stable candidate thresholds from validation probabilities."""
    probabilities = np.asarray(y_prob, dtype=float)
    probabilities = probabilities[~np.isnan(probabilities)]
    if probabilities.size == 0:
        return np.array([], dtype=float)
    lower, upper = bounds
    candidates = np.unique(np.round(probabilities, 2))
    candidates = candidates[(candidates >= lower) & (candidates <= upper)]
    return np.sort(candidates)[::-1]


def select_max_f1_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    candidate_thresholds: Optional[np.ndarray] = None,
    bounds: Tuple[float, float] = (DEFAULT_THRESHOLD_LOWER_BOUND, DEFAULT_THRESHOLD_UPPER_BOUND),
    default_threshold: float = DEFAULT_PARTITIONED_THRESHOLD,
) -> Dict[str, Any]:
    """Select the validation threshold that maximizes class-1 F1.

    Ties are resolved toward the higher threshold, preserving precision when
    validation F1 is equivalent.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    valid = ~np.isnan(y_prob)
    y_true = y_true[valid]
    y_prob = y_prob[valid]

    base = {
        "selected_threshold": float(default_threshold),
        "validation_precision": np.nan,
        "validation_recall": np.nan,
        "validation_f1": np.nan,
        "validation_support": int(len(y_true)),
        "validation_positive_cases": int((y_true == 1).sum()),
        "fallback_reason": "",
    }
    if len(y_true) == 0:
        base["fallback_reason"] = "no_validation_observations"
        return base
    if int((y_true == 1).sum()) == 0:
        base["fallback_reason"] = "no_validation_positive_cases"
        return base

    candidates = candidate_thresholds_from_probabilities(y_prob, bounds=bounds) if candidate_thresholds is None else np.asarray(candidate_thresholds, dtype=float)
    candidates = np.sort(np.unique(np.round(candidates, 2)))[::-1]
    if candidates.size == 0:
        base["fallback_reason"] = "no_candidate_thresholds"
        return base

    rows = []
    for threshold in candidates:
        y_pred = apply_probability_threshold(y_prob, threshold)
        metrics = compute_binary_metrics(y_true, y_pred)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )
    metrics_df = pd.DataFrame(rows)
    finite = metrics_df[np.isfinite(metrics_df["f1"])]
    if finite.empty:
        base["fallback_reason"] = "no_finite_validation_f1"
        return base

    max_f1 = float(finite["f1"].max())
    tied = finite[np.isclose(finite["f1"], max_f1, rtol=1e-12, atol=1e-12)]
    selected = tied.sort_values("threshold", ascending=False).iloc[0]
    return {
        **base,
        "selected_threshold": float(selected["threshold"]),
        "validation_precision": float(selected["precision"]),
        "validation_recall": float(selected["recall"]),
        "validation_f1": float(selected["f1"]),
    }


def split_training_validation_by_dates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_group_train: np.ndarray,
    train_dates: pd.Series | np.ndarray,
    validation_months: int = DEFAULT_THRESHOLD_VALIDATION_MONTHS,
) -> Dict[str, np.ndarray]:
    """Split training rows into earlier fit rows and latest-month validation rows."""
    train_dates = pd.to_datetime(pd.Series(train_dates)).reset_index(drop=True)
    if len(train_dates) != len(y_train):
        raise ValueError(f"train_dates length {len(train_dates)} does not match y_train length {len(y_train)}")
    if validation_months <= 0:
        raise ValueError(f"validation_months must be positive, got {validation_months}")
    if len(y_train) == 0:
        raise ValueError("Cannot split empty training data")

    latest_month = train_dates.max().to_period("M").to_timestamp()
    validation_start = latest_month - pd.DateOffset(months=validation_months - 1)
    validation_mask = train_dates >= validation_start
    fit_mask = ~validation_mask

    if int(validation_mask.sum()) == 0 or int(fit_mask.sum()) == 0:
        raise ValueError(
            f"Validation split failed: fit={int(fit_mask.sum())}, validation={int(validation_mask.sum())}, "
            f"validation_months={validation_months}"
        )

    return {
        "X_fit": X_train[fit_mask],
        "y_fit": y_train[fit_mask],
        "group_fit": X_group_train[fit_mask],
        "X_val": X_train[validation_mask],
        "y_val": y_train[validation_mask],
        "group_val": X_group_train[validation_mask],
        "fit_start": train_dates[fit_mask].min(),
        "fit_end": train_dates[fit_mask].max(),
        "validation_start": train_dates[validation_mask].min(),
        "validation_end": train_dates[validation_mask].max(),
    }
```

- [ ] **Step 3: Run helper tests**

Run:

```bash
python3 -m unittest src.tests.test_georf_validation_threshold
```

Expected: `Ran 5 tests` and `OK`.

### Task 3: Integrate Thresholding Into Stage 3 Evaluation

**Files:**
- Modify: `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py`
- Test: `src/tests/test_georf_validation_threshold.py`

- [ ] **Step 1: Add CLI flags**

In `main()`, after `--lower-model`, add:

```python
    parser.add_argument('--enable-validation-threshold', action='store_true',
                        help='Add partitioned_thresholded results using validation-selected max-F1 thresholds')
    parser.add_argument('--threshold-validation-months', type=int, default=DEFAULT_THRESHOLD_VALIDATION_MONTHS,
                        help='Number of latest training-window months held out for threshold validation')
    parser.add_argument('--threshold-lower-bound', type=float, default=DEFAULT_THRESHOLD_LOWER_BOUND,
                        help='Lower bound for candidate probability thresholds')
    parser.add_argument('--threshold-upper-bound', type=float, default=DEFAULT_THRESHOLD_UPPER_BOUND,
                        help='Upper bound for candidate probability thresholds')
```

- [ ] **Step 2: Initialize threshold provenance collection**

After:

```python
    monthly_metrics = []
    all_predictions = []
```

add:

```python
    threshold_provenance = []
```

- [ ] **Step 3: Retrieve train row indices for date-aware validation**

Inside the monthly loop, after the existing `baseline_split` and `partition_split` calls succeed, add index split calls:

```python
            row_indices = np.arange(len(y))
            baseline_index_split = train_test_split_rolling_window(
                X, y, X_loc, baseline_groups, years, dates,
                test_month=test_month,
                active_lag=active_lag,
                train_window_months=args.train_window,
                admin_codes=row_indices,
            )
            partition_index_split = train_test_split_rolling_window(
                X, y, X_loc, X_group, years, dates,
                test_month=test_month,
                active_lag=active_lag,
                train_window_months=args.train_window,
                admin_codes=row_indices,
            )
            baseline_train_indices = baseline_index_split[8]
            partition_train_indices = partition_index_split[8]
            baseline_train_dates = pd.to_datetime(pd.Series(dates).iloc[baseline_train_indices]).reset_index(drop=True)
            partition_train_dates = pd.to_datetime(pd.Series(dates).iloc[partition_train_indices]).reset_index(drop=True)
```

This deliberately uses the existing rolling split helper and its group-filtering behavior instead of reimplementing the temporal masks.

- [ ] **Step 4: Select validation threshold before fitting the full-window final models**

Before the existing full-window training block:

```python
        # Train pooled model (full window)
```

add:

```python
        threshold_record: Dict[str, Any] = {
            "test_month": str(test_month),
            "forecasting_scope": args.forecasting_scope,
            "active_lag_months": active_lag,
            "threshold_enabled": bool(args.enable_validation_threshold),
            "selected_threshold": DEFAULT_PARTITIONED_THRESHOLD,
            "fallback_reason": "thresholding_disabled",
        }
        if args.enable_validation_threshold:
            try:
                baseline_tv = split_training_validation_by_dates(
                    Xtrain_pooled,
                    ytrain_pooled,
                    np.zeros_like(ytrain_pooled),
                    baseline_train_dates,
                    validation_months=args.threshold_validation_months,
                )
                partition_tv = split_training_validation_by_dates(
                    Xtrain_partitioned,
                    ytrain_partitioned,
                    Xtrain_group_partitioned,
                    partition_train_dates,
                    validation_months=args.threshold_validation_months,
                )
                threshold_pooled_model = train_pooled_model(
                    baseline_tv["X_fit"],
                    baseline_tv["y_fit"].astype(int),
                    args.lower_model,
                )
                threshold_partitioned_models = train_partitioned_model(
                    partition_tv["X_fit"],
                    partition_tv["y_fit"].astype(int),
                    partition_tv["group_fit"],
                    args.lower_model,
                )
                y_prob_val_partitioned = predict_partitioned_probability(
                    threshold_partitioned_models,
                    threshold_pooled_model,
                    partition_tv["X_val"],
                    partition_tv["group_val"],
                )
                threshold_record.update(
                    select_max_f1_threshold(
                        partition_tv["y_val"].astype(int),
                        y_prob_val_partitioned,
                        bounds=(args.threshold_lower_bound, args.threshold_upper_bound),
                        default_threshold=DEFAULT_PARTITIONED_THRESHOLD,
                    )
                )
                threshold_record.update(
                    {
                        "fit_start": str(pd.Timestamp(partition_tv["fit_start"]).date()),
                        "fit_end": str(pd.Timestamp(partition_tv["fit_end"]).date()),
                        "validation_start": str(pd.Timestamp(partition_tv["validation_start"]).date()),
                        "validation_end": str(pd.Timestamp(partition_tv["validation_end"]).date()),
                    }
                )
            except Exception as exc:
                threshold_record.update(
                    {
                        "selected_threshold": DEFAULT_PARTITIONED_THRESHOLD,
                        "fallback_reason": f"validation_threshold_error: {exc}",
                    }
                )
```

- [ ] **Step 5: Compute thresholded predictions and metrics**

After:

```python
        metrics_partitioned = compute_binary_metrics(ytest, y_pred_partitioned)
```

add:

```python
        y_pred_partitioned_thresholded = None
        metrics_partitioned_thresholded = None
        if args.enable_validation_threshold:
            y_pred_partitioned_thresholded = apply_probability_threshold(
                y_prob_partitioned,
                threshold_record["selected_threshold"],
            )
            metrics_partitioned_thresholded = compute_binary_metrics(ytest, y_pred_partitioned_thresholded)
            threshold_record.update(
                {
                    "test_precision": metrics_partitioned_thresholded["precision"],
                    "test_recall": metrics_partitioned_thresholded["recall"],
                    "test_f1": metrics_partitioned_thresholded["f1"],
                    "test_tp": metrics_partitioned_thresholded["tp"],
                    "test_fp": metrics_partitioned_thresholded["fp"],
                    "test_fn": metrics_partitioned_thresholded["fn"],
                    "test_tn": metrics_partitioned_thresholded["tn"],
                }
            )
            threshold_provenance.append(threshold_record)
```

After the existing `partitioned` `monthly_metrics.append(...)`, add:

```python
        if args.enable_validation_threshold and metrics_partitioned_thresholded is not None:
            monthly_metrics.append({
                'test_month': str(test_month),
                'model': 'partitioned_thresholded',
                **metrics_partitioned_thresholded,
            })
```

- [ ] **Step 6: Add prediction columns**

In the prediction dataframe construction, after `y_prob_partitioned`, add conditional columns:

```python
            if args.enable_validation_threshold and y_pred_partitioned_thresholded is not None:
                pred_df["selected_threshold"] = float(threshold_record["selected_threshold"])
                pred_df["y_pred_partitioned_thresholded"] = y_pred_partitioned_thresholded
```

- [ ] **Step 7: Save `threshold_provenance.csv` and update manifest**

In Step 5 saving, after monthly metrics are saved, add:

```python
    if threshold_provenance:
        threshold_df = pd.DataFrame(threshold_provenance)
        threshold_path = out_dir / 'threshold_provenance.csv'
        threshold_df.to_csv(threshold_path, index=False)
        print(f"  Saved: {threshold_path}")
    else:
        threshold_df = pd.DataFrame()
```

In `manifest`, change:

```python
        'n_test_months_evaluated': len(monthly_metrics) // 2,
```

to:

```python
        'n_test_months_evaluated': int(metrics_df['test_month'].nunique()) if not metrics_df.empty else 0,
```

and add:

```python
        'validation_threshold_enabled': bool(args.enable_validation_threshold),
        'threshold_selection_metric': 'class_1_f1' if args.enable_validation_threshold else None,
        'threshold_validation_months': args.threshold_validation_months if args.enable_validation_threshold else None,
        'threshold_candidate_bounds': [args.threshold_lower_bound, args.threshold_upper_bound] if args.enable_validation_threshold else None,
        'threshold_default': DEFAULT_PARTITIONED_THRESHOLD if args.enable_validation_threshold else None,
```

- [ ] **Step 8: Print thresholded summary**

After the existing partitioned print statement, add:

```python
        if args.enable_validation_threshold and metrics_partitioned_thresholded is not None:
            print(
                f"  Thresholded: Precision={metrics_partitioned_thresholded['precision']:.4f}, "
                f"Recall={metrics_partitioned_thresholded['recall']:.4f}, "
                f"F1={metrics_partitioned_thresholded['f1']:.4f}, "
                f"Threshold={threshold_record['selected_threshold']:.2f}"
            )
```

- [ ] **Step 9: Run focused tests and compile**

Run:

```bash
python3 -m unittest src.tests.test_georf_validation_threshold
python3 -m py_compile scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py
```

Expected: tests pass and compile exits 0.

### Task 4: Add Thresholded Artifact Builder

**Files:**
- Create: `scripts/build_georf_thresholded_artifacts.py`
- Create: `src/tests/test_georf_thresholded_artifacts.py`

- [ ] **Step 1: Write artifact-builder tests**

Create `src/tests/test_georf_thresholded_artifacts.py`:

```python
import importlib.util
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_georf_thresholded_artifacts.py"
spec = importlib.util.spec_from_file_location("build_georf_thresholded_artifacts", SCRIPT_PATH)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class GeoRFThresholdedArtifactsTests(unittest.TestCase):
    def test_build_compact_table_pivots_three_models(self):
        metrics = pd.DataFrame(
            [
                {"scope": "fs1", "forecasting_horizon": "4-month lag", "model": "pooled", "precision": 0.7, "recall": 0.5, "f1": 0.58},
                {"scope": "fs1", "forecasting_horizon": "4-month lag", "model": "partitioned", "precision": 0.75, "recall": 0.6, "f1": 0.67},
                {"scope": "fs1", "forecasting_horizon": "4-month lag", "model": "partitioned_thresholded", "precision": 0.70, "recall": 0.7, "f1": 0.70},
            ]
        )

        compact = builder.build_compact_table(metrics)
        row = compact.iloc[0]

        self.assertAlmostEqual(row["pooled_f1"], 0.58)
        self.assertAlmostEqual(row["partitioned_f1"], 0.67)
        self.assertAlmostEqual(row["partitioned_thresholded_f1"], 0.70)
        self.assertAlmostEqual(row["delta_thresholded_minus_partitioned_recall"], 0.10)
        self.assertAlmostEqual(row["delta_thresholded_minus_partitioned_f1"], 0.03)

    def test_format_compact_rounds_numeric_columns(self):
        compact = pd.DataFrame(
            [{"scope": "fs1", "forecasting_horizon": "4-month lag", "partitioned_thresholded_f1": 0.70321}]
        )

        formatted = builder.format_for_markdown(compact)

        self.assertEqual(formatted.loc[0, "partitioned_thresholded_f1"], "0.703")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail because the builder does not exist**

Run:

```bash
python3 -m unittest src.tests.test_georf_thresholded_artifacts
```

Expected: FAIL with `FileNotFoundError`.

- [ ] **Step 3: Create the artifact builder**

Create `scripts/build_georf_thresholded_artifacts.py`:

```python
#!/usr/bin/env python3
"""Build paper-facing artifacts for GeoRF validation-selected thresholding."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "final_artifacts_in_paper_updated" / "12_thresholded_georf_results"
HORIZONS = {
    "fs1": "4-month lag",
    "fs2": "8-month lag",
    "fs3": "12-month lag",
}


def load_thresholded_results(source_dir: Path, scopes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_frames = []
    threshold_frames = []
    for scope in scopes:
        result_dir = source_dir / f"result_partition_k40_compare_GF_thresholded_{scope}"
        metrics = pd.read_csv(result_dir / "metrics_monthly.csv")
        thresholds = pd.read_csv(result_dir / "threshold_provenance.csv")
        metrics["scope"] = scope
        metrics["forecasting_horizon"] = HORIZONS[scope]
        thresholds["scope"] = scope
        thresholds["forecasting_horizon"] = HORIZONS[scope]
        metrics_frames.append(metrics)
        threshold_frames.append(thresholds)
    return pd.concat(metrics_frames, ignore_index=True), pd.concat(threshold_frames, ignore_index=True)


def _aggregate_model_metrics(group: pd.DataFrame) -> pd.Series:
    tp = group["tp"].sum()
    fp = group["fp"].sum()
    fn = group["fn"].sum()
    tn = group["tn"].sum()
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    return pd.Series(
        {
            "support": int(group["n"].sum()),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


def build_horizon_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        metrics.groupby(["scope", "forecasting_horizon", "model"], sort=True)
        .apply(_aggregate_model_metrics)
        .reset_index()
    )
    return grouped


def build_compact_table(horizon_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_names = ["precision", "recall", "f1"]
    for (scope, horizon), sub in horizon_metrics.groupby(["scope", "forecasting_horizon"], sort=True):
        row = {"scope": scope, "forecasting_horizon": horizon}
        by_model = {model: frame.iloc[0] for model, frame in sub.groupby("model")}
        for model in ["pooled", "partitioned", "partitioned_thresholded"]:
            model_row = by_model.get(model)
            for metric in metric_names:
                row[f"{model}_{metric}"] = model_row[metric] if model_row is not None else float("nan")
        for metric in metric_names:
            row[f"delta_thresholded_minus_partitioned_{metric}"] = (
                row[f"partitioned_thresholded_{metric}"] - row[f"partitioned_{metric}"]
            )
            row[f"delta_thresholded_minus_pooled_{metric}"] = (
                row[f"partitioned_thresholded_{metric}"] - row[f"pooled_{metric}"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def format_for_markdown(table: pd.DataFrame) -> pd.DataFrame:
    formatted = table.copy()
    for column in formatted.columns:
        if column not in {"scope", "forecasting_horizon"}:
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    return formatted


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


def write_note(output_dir: Path) -> None:
    (output_dir / "georf_thresholded_note.md").write_text(
        "\n".join(
            [
                "# GeoRF Validation-Selected Max-F1 Thresholding",
                "",
                "中文审查说明：",
                "",
                "该 appendix 只针对 GeoRF partitioned/local RF 模型的 thresholded diagnostic。",
                "Threshold 在每个 rolling training window 的 validation subset 上选择，目标是最大化 class-1 F1。",
                "选出的 threshold 只应用于随后 held-out target month 的 test probabilities；没有使用 test labels 选择 threshold。",
                "原始 pooled 和 partitioned hard-prediction 结果保留，用于对照。",
                "这些结果先写入独立 artifact folder，尚不覆盖主文 01-11 artifacts。",
                "",
                "Appendix text (English):",
                "",
                "We evaluate a validation-selected probability threshold for the GeoRF partitioned RF model.",
                "For each forecasting horizon and target month, the threshold is selected on a validation subset from the rolling training window by maximizing class-1 F1, then applied to the held-out target-month probabilities.",
                "The procedure does not use test labels for threshold selection.",
                "Pooled and original partitioned hard-prediction results are retained as comparators.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scopes", nargs="+", default=["fs1", "fs2", "fs3"], choices=["fs1", "fs2", "fs3"])
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics, thresholds = load_thresholded_results(args.source_dir, args.scopes)
    horizon_metrics = build_horizon_metrics(metrics)
    compact = build_compact_table(horizon_metrics)

    horizon_metrics.to_csv(args.output_dir / "georf_thresholded_horizon_metrics.csv", index=False)
    metrics.to_csv(args.output_dir / "georf_thresholded_monthly_metrics.csv", index=False)
    thresholds.to_csv(args.output_dir / "georf_thresholded_threshold_provenance.csv", index=False)
    compact.to_csv(args.output_dir / "georf_thresholded_compact_table.csv", index=False)
    write_markdown_table(format_for_markdown(compact), args.output_dir / "georf_thresholded_compact_table.md")
    write_note(args.output_dir)

    print(f"Wrote GeoRF thresholded artifacts to {args.output_dir}")
    print(f"Rows: monthly_metrics={len(metrics)}, thresholds={len(thresholds)}, compact={len(compact)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run artifact-builder tests**

Run:

```bash
python3 -m unittest src.tests.test_georf_thresholded_artifacts
python3 -m py_compile scripts/build_georf_thresholded_artifacts.py
```

Expected: tests pass and compile exits 0.

### Task 5: Generate Thresholded Stage 3 Results

**Files:**
- Create: `result_partition_k40_compare_GF_thresholded_fs1/`
- Create: `result_partition_k40_compare_GF_thresholded_fs2/`
- Create: `result_partition_k40_compare_GF_thresholded_fs3/`

- [ ] **Step 1: Run fs1 thresholded comparison**

Run:

```bash
python3 scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py \
  --forecasting-scope 1 \
  --start-month 2021-01 \
  --end-month 2024-12 \
  --partition-map result_partition_k40_compare_GF_fs1/refined/cluster_mapping_k40_nc17_general_refined_contig3.csv \
  --out-dir result_partition_k40_compare_GF_thresholded_fs1 \
  --enable-validation-threshold \
  --threshold-validation-months 6
```

Expected:

- `result_partition_k40_compare_GF_thresholded_fs1/metrics_monthly.csv`
- `result_partition_k40_compare_GF_thresholded_fs1/predictions_monthly.csv`
- `result_partition_k40_compare_GF_thresholded_fs1/threshold_provenance.csv`
- `metrics_monthly.csv` contains `partitioned_thresholded` rows.

- [ ] **Step 2: Run fs2 thresholded comparison**

Run:

```bash
python3 scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py \
  --forecasting-scope 2 \
  --start-month 2021-01 \
  --end-month 2024-12 \
  --partition-map result_partition_k40_compare_GF_fs2/refined/cluster_mapping_k40_nc17_general_refined_contig3.csv \
  --out-dir result_partition_k40_compare_GF_thresholded_fs2 \
  --enable-validation-threshold \
  --threshold-validation-months 6
```

Expected equivalent fs2 outputs.

- [ ] **Step 3: Run fs3 thresholded comparison**

Run:

```bash
python3 scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py \
  --forecasting-scope 3 \
  --start-month 2021-01 \
  --end-month 2024-12 \
  --partition-map result_partition_k40_compare_GF_fs3/refined/cluster_mapping_k40_nc17_general_refined_contig3.csv \
  --out-dir result_partition_k40_compare_GF_thresholded_fs3 \
  --enable-validation-threshold \
  --threshold-validation-months 6
```

Expected equivalent fs3 outputs.

- [ ] **Step 4: Validate generated result schemas**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
import pandas as pd
for scope in ["fs1", "fs2", "fs3"]:
    root = Path(f"result_partition_k40_compare_GF_thresholded_{scope}")
    metrics = pd.read_csv(root / "metrics_monthly.csv")
    preds = pd.read_csv(root / "predictions_monthly.csv")
    thresholds = pd.read_csv(root / "threshold_provenance.csv")
    print(scope, "models", sorted(metrics["model"].unique()), "months", metrics["test_month"].nunique(), "pred_rows", len(preds), "threshold_rows", len(thresholds))
    assert "partitioned_thresholded" in set(metrics["model"])
    assert "selected_threshold" in preds.columns
    assert "y_pred_partitioned_thresholded" in preds.columns
    assert thresholds["test_month"].nunique() == metrics["test_month"].nunique()
PY
```

Expected: each scope reports three models, 12 evaluated months, nonzero predictions, and 12 threshold rows.

### Task 6: Build Paper-Facing Thresholded Artifacts

**Files:**
- Create: `final_artifacts_in_paper_updated/12_thresholded_georf_results/`
- Modify: `final_artifacts_in_paper_updated/README.md`

- [ ] **Step 1: Run artifact builder**

Run:

```bash
python3 scripts/build_georf_thresholded_artifacts.py
```

Expected output:

```text
Wrote GeoRF thresholded artifacts to .../final_artifacts_in_paper_updated/12_thresholded_georf_results
Rows: monthly_metrics=108, thresholds=36, compact=3
```

- [ ] **Step 2: Inspect compact result**

Run:

```bash
cat final_artifacts_in_paper_updated/12_thresholded_georf_results/georf_thresholded_compact_table.md
```

Expected: one header and three horizon rows. Check whether `partitioned_thresholded_f1` exceeds `partitioned_f1` and whether recall gain is acceptable relative to precision loss.

- [ ] **Step 3: Register README folder index**

In `final_artifacts_in_paper_updated/README.md`, add after the `11_threshold_free_metrics/` folder index item:

```markdown
- `12_thresholded_georf_results/`: validation-selected max-F1 thresholded
  GeoRF partitioned results and threshold provenance.
```

- [ ] **Step 4: Register README section**

Before `## Reproduction Checks`, add:

```markdown
## 12 Thresholded GeoRF Results

- `12_thresholded_georf_results/georf_thresholded_compact_table.csv`:
  compact horizon-level pooled, partitioned, and partitioned-thresholded
  precision, recall, and F1 comparison.
- `12_thresholded_georf_results/georf_thresholded_compact_table.md`:
  Markdown rendering of the compact thresholded GeoRF comparison table.
- `12_thresholded_georf_results/georf_thresholded_horizon_metrics.csv`:
  long-format horizon-level metrics for pooled, partitioned, and
  partitioned-thresholded models.
- `12_thresholded_georf_results/georf_thresholded_monthly_metrics.csv`:
  monthly metrics from the thresholded Stage 3 result folders.
- `12_thresholded_georf_results/georf_thresholded_threshold_provenance.csv`:
  selected validation thresholds and validation/test metrics by horizon and
  evaluated target month.
- `12_thresholded_georf_results/georf_thresholded_note.md`: Chinese
  reviewer-facing note and English appendix text for validation-selected
  max-F1 thresholding.
```

- [ ] **Step 5: Verify README paths**

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

### Task 7: Final Verification and Commit

**Files:**
- All files from previous tasks.

- [ ] **Step 1: Run focused tests and compilation**

Run:

```bash
python3 -m unittest src.tests.test_georf_validation_threshold src.tests.test_georf_thresholded_artifacts
python3 -m py_compile scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py scripts/build_georf_thresholded_artifacts.py
git diff --check
```

Expected: all tests pass, both scripts compile, and diff check exits 0.

- [ ] **Step 2: Inspect git status**

Run:

```bash
git status --short --branch
git diff --stat
```

Expected: changes include only:

- `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py`
- `scripts/build_georf_thresholded_artifacts.py`
- `src/tests/test_georf_validation_threshold.py`
- `src/tests/test_georf_thresholded_artifacts.py`
- `docs/superpowers/plans/2026-06-16-georf-validation-max-f1-threshold.md`
- new thresholded result folders
- `final_artifacts_in_paper_updated/12_thresholded_georf_results/`
- `final_artifacts_in_paper_updated/README.md`

- [ ] **Step 3: Force-add ignored artifacts and commit**

Run:

```bash
git add -A
git add -f final_artifacts_in_paper_updated/12_thresholded_georf_results
git add -f result_partition_k40_compare_GF_thresholded_fs1 result_partition_k40_compare_GF_thresholded_fs2 result_partition_k40_compare_GF_thresholded_fs3
git diff --cached --stat
git commit -m "add GeoRF validation-selected threshold results"
```

Expected: commit succeeds. Do not push unless the user asks.

### Self-Review

- Spec coverage: covers validation-selected max-F1 thresholding, separate `partitioned_thresholded` variant, threshold provenance, isolated thresholded result folders, and `12_thresholded_georf_results/`.
- Scope control: does not alter Stage 1 partitions, Stage 2 consensus, GeoDT, GeoXGB, FEWSNET baseline, feature sets, or lag schedules.
- Leakage control: threshold selection uses training-window validation rows only and applies the selected threshold to the held-out target month.
- Artifact safety: first pass writes isolated thresholded outputs and does not replace 01-11 paper artifacts.
- Tests: helper tests cover max-F1 selection, tie-breaking, fallback, threshold application, and temporal validation splitting; artifact tests cover compact summary construction.
