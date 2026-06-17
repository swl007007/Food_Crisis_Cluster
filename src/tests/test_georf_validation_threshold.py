import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


def _install_import_stubs() -> None:
    """Stub heavy pipeline imports that are not needed by helper tests."""
    def get_class_wise_accuracy(y_true, y_pred, prf=True):
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        true_class = {}
        total_class = {}
        pred_total = {}
        for cls in [0, 1]:
            true_class[cls] = int(((y_true == cls) & (y_pred == cls)).sum())
            total_class[cls] = int((y_true == cls).sum())
            pred_total[cls] = int((y_pred == cls).sum())
        return true_class, total_class, pred_total

    def get_prf(true_class, total_class, pred_total, nan_option="mean"):
        precision = {}
        recall = {}
        f1 = {}
        for cls in [0, 1]:
            precision[cls] = true_class[cls] / pred_total[cls] if pred_total[cls] else 0.0
            recall[cls] = true_class[cls] / total_class[cls] if total_class[cls] else 0.0
            denom = precision[cls] + recall[cls]
            f1[cls] = 2 * precision[cls] * recall[cls] / denom if denom else 0.0
        return precision, recall, f1, None

    module_specs = {
        "src.preprocess.preprocess": {"load_and_preprocess_data": lambda *args, **kwargs: None},
        "src.feature.feature": {"prepare_features": lambda *args, **kwargs: None},
        "src.customize.customize": {"train_test_split_rolling_window": lambda *args, **kwargs: None},
        "src.utils.lag_schedules": {"forecasting_scope_to_lag": lambda scope, lags=None: {0: 1, 1: 4, 2: 8, 3: 12}[scope]},
        "src.metrics.metrics": {
            "get_prf": get_prf,
            "get_class_wise_accuracy": get_class_wise_accuracy,
        },
    }
    for name, attrs in module_specs.items():
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        for attr_name, value in attrs.items():
            setattr(module, attr_name, value)
        sys.modules[name] = module


_install_import_stubs()
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
