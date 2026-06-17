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
    def test_scope_display_labels_use_horizon_wording(self):
        self.assertEqual(
            threshold_metrics.HORIZONS,
            {
                "fs1": "4-month horizon",
                "fs2": "8-month horizon",
                "fs3": "12-month horizon",
            },
        )

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
                    "forecasting_horizon": "4-month horizon",
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
                    "forecasting_horizon": "4-month horizon",
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
