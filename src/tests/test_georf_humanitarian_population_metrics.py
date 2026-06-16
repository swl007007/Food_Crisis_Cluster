import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_georf_humanitarian_population_metrics.py"
spec = importlib.util.spec_from_file_location("analyze_georf_humanitarian_population_metrics", SCRIPT_PATH)
humanitarian = importlib.util.module_from_spec(spec)
spec.loader.exec_module(humanitarian)


class GeoRFHumanitarianPopulationMetricsTests(unittest.TestCase):
    def test_compute_population_metrics_uses_population_confusion_totals(self):
        df = pd.DataFrame(
            {
                "y_true": [1, 1, 0, 0],
                "y_pred_partitioned": [1, 0, 1, 0],
                "pop": [100.0, 200.0, 300.0, 400.0],
            }
        )

        metrics = humanitarian.compute_population_metrics(df, "y_pred_partitioned")

        self.assertEqual(metrics["support"], 4)
        self.assertAlmostEqual(metrics["population_at_risk"], 1000.0)
        self.assertAlmostEqual(metrics["true_alert_population"], 100.0)
        self.assertAlmostEqual(metrics["missed_crisis_population"], 200.0)
        self.assertAlmostEqual(metrics["false_alert_population"], 300.0)
        self.assertAlmostEqual(metrics["true_noncrisis_population"], 400.0)
        self.assertAlmostEqual(metrics["population_weighted_recall"], 100.0 / 300.0)
        self.assertAlmostEqual(metrics["population_weighted_precision"], 100.0 / 400.0)

    def test_build_summary_reports_pooled_partitioned_and_delta(self):
        df = pd.DataFrame(
            {
                "scope": ["fs1"] * 4,
                "forecasting_horizon": ["4-month lag"] * 4,
                "month_start": pd.to_datetime(["2021-02-01"] * 4),
                "y_true": [1, 1, 0, 0],
                "y_pred_pooled": [1, 0, 1, 0],
                "y_pred_partitioned": [1, 1, 0, 0],
                "pop": [100.0, 200.0, 300.0, 400.0],
            }
        )

        summary = humanitarian.build_summary_table(df)
        compact = humanitarian.build_compact_table(summary)

        self.assertEqual(set(summary["model"]), {"pooled", "partitioned"})
        self.assertEqual(compact.shape, (1, 14))
        row = compact.iloc[0]
        self.assertEqual(row["scope"], "fs1")
        self.assertEqual(row["forecasting_horizon"], "4-month lag")
        self.assertAlmostEqual(row["delta_missed_crisis_population"], -200.0)
        self.assertAlmostEqual(row["delta_false_alert_population"], -300.0)
        self.assertAlmostEqual(row["delta_population_weighted_recall"], 2.0 / 3.0)
        self.assertAlmostEqual(row["delta_population_weighted_precision"], 0.75)

    def test_load_population_lookup_rejects_duplicate_admin_month_keys(self):
        raw = pd.DataFrame(
            {
                "admin_code": [10, 10],
                "year": [2021, 2021],
                "month": [2, 2],
                "pop": [100.0, 100.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "FEWSNET.csv"
            raw.to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "Duplicate population keys"):
                humanitarian.load_population_lookup(path)

    def test_join_population_rejects_missing_population(self):
        predictions = pd.DataFrame(
            {
                "FEWSNET_admin_code": [10, 20],
                "month_start": pd.to_datetime(["2021-02-01", "2021-02-01"]),
                "scope": ["fs1", "fs1"],
                "forecasting_horizon": ["4-month lag", "4-month lag"],
                "y_true": [1, 0],
                "y_pred_pooled": [1, 0],
                "y_pred_partitioned": [1, 0],
            }
        )
        population = pd.DataFrame(
            {
                "admin_code": ["10"],
                "month_start": pd.to_datetime(["2021-02-01"]),
                "pop": [100.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "Missing population"):
            humanitarian.join_population(predictions, population)

    def test_write_markdown_table_has_no_optional_dependency(self):
        table = pd.DataFrame(
            {
                "forecasting_horizon": ["4-month lag"],
                "partitioned_population_weighted_recall": [0.75],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "table.md"
            humanitarian.write_markdown_table(table, path)

            text = path.read_text(encoding="utf-8")
            self.assertIn("| forecasting_horizon | partitioned_population_weighted_recall |", text)
            self.assertIn("| 4-month lag | 0.75 |", text)


if __name__ == "__main__":
    unittest.main()
