import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "paper_artifacts"
    / "analyze_georf_probability_uncertainty.py"
)
spec = importlib.util.spec_from_file_location("analyze_georf_probability_uncertainty", SCRIPT_PATH)
probability = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probability)


class GeoRFProbabilityUncertaintyTests(unittest.TestCase):
    def test_scope_display_labels_use_horizon_wording(self):
        self.assertEqual(
            probability.HORIZONS,
            {
                "fs1": "4-month horizon",
                "fs2": "8-month horizon",
                "fs3": "12-month horizon",
            },
        )

    def test_resolve_path_keeps_windows_drive_paths_under_windows_python(self):
        raw = Path(r"C:\Users\swl00\data\file.shp")

        resolved = probability.resolve_path(raw, platform_name="nt")

        self.assertEqual(str(resolved), str(raw))

    def test_brier_score_uses_probabilities_not_hard_labels(self):
        y_true = pd.Series([0, 1, 1, 0])
        y_prob = pd.Series([0.1, 0.8, 0.4, 0.2])

        score = probability.brier_score(y_true, y_prob)

        self.assertAlmostEqual(score, ((0.1 - 0) ** 2 + (0.8 - 1) ** 2 + (0.4 - 1) ** 2 + (0.2 - 0) ** 2) / 4)

    def test_compute_model_metrics_returns_hard_metrics_and_brier(self):
        df = pd.DataFrame(
            {
                "y_true": [1, 1, 0, 0],
                "y_pred_partitioned": [1, 0, 1, 0],
                "y_prob_partitioned": [0.9, 0.4, 0.7, 0.2],
            }
        )

        metrics = probability.compute_model_metrics(df, "y_pred_partitioned", "y_prob_partitioned")

        self.assertEqual(metrics["support"], 4)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)
        self.assertAlmostEqual(metrics["brier"], probability.brier_score(df["y_true"], df["y_prob_partitioned"]))

    def test_reliability_bins_report_observed_rate_and_gap(self):
        df = pd.DataFrame(
            {
                "y_true": [0, 1, 1, 0],
                "y_prob_partitioned": [0.05, 0.25, 0.75, 0.95],
            }
        )

        bins = probability.reliability_bins(df, "y_prob_partitioned", n_bins=4)

        self.assertEqual(bins["bin_id"].tolist(), [0, 1, 3])
        first = bins[bins["bin_id"].eq(0)].iloc[0]
        self.assertEqual(first["n"], 1)
        self.assertAlmostEqual(first["mean_predicted_probability"], 0.05)
        self.assertAlmostEqual(first["observed_crisis_rate"], 0.0)
        self.assertAlmostEqual(first["calibration_gap"], -0.05)

    def test_country_clustered_bootstrap_resamples_whole_countries(self):
        df = pd.DataFrame(
            {
                "ADMIN0": ["A", "A", "B", "B"],
                "value": [1, 2, 10, 20],
            }
        )
        rng = np.random.default_rng(7)

        sample = probability.resample_clusters(df, "ADMIN0", rng)

        self.assertEqual(len(sample), 4)
        country_counts = sample["ADMIN0"].value_counts()
        self.assertTrue(set(country_counts.index).issubset({"A", "B"}))
        for country, count in country_counts.items():
            self.assertIn(count, {2, 4})

    def test_build_bootstrap_ci_includes_delta_metrics(self):
        df = pd.DataFrame(
            {
                "ADMIN0": ["A", "A", "B", "B", "C", "C"],
                "y_true": [1, 0, 1, 0, 1, 0],
                "y_pred_pooled": [1, 0, 0, 0, 1, 1],
                "y_pred_partitioned": [1, 0, 1, 0, 1, 0],
                "y_prob_pooled": [0.8, 0.2, 0.4, 0.2, 0.9, 0.6],
                "y_prob_partitioned": [0.8, 0.2, 0.7, 0.2, 0.9, 0.3],
            }
        )

        ci = probability.build_bootstrap_ci(df, n_bootstrap=20, seed=3)

        self.assertIn("delta_point", ci.columns)
        self.assertEqual(set(ci["metric"]), {"precision", "recall", "f1", "brier"})
        brier = ci[ci["metric"].eq("brier")].iloc[0]
        self.assertLessEqual(brier["delta_ci_low"], brier["delta_ci_high"])

    def test_format_ci_cell_rounds_point_and_interval(self):
        row = pd.Series({"delta_point": 0.040839, "delta_ci_low": 0.019371, "delta_ci_high": 0.063455})

        cell = probability.format_ci_cell(row)

        self.assertEqual(cell, "0.041 [0.019, 0.063]")

    def test_build_compact_bootstrap_table_returns_one_row_per_horizon(self):
        bootstrap = pd.DataFrame(
            [
                {
                    "scope": "fs1",
                    "forecasting_horizon": "4-month horizon",
                    "metric": metric,
                    "delta_point": value,
                    "delta_ci_low": value - 0.01,
                    "delta_ci_high": value + 0.01,
                }
                for metric, value in [
                    ("precision", -0.01),
                    ("recall", 0.08),
                    ("f1", 0.04),
                    ("brier", -0.006),
                ]
            ]
            + [
                {
                    "scope": "fs2",
                    "forecasting_horizon": "8-month horizon",
                    "metric": metric,
                    "delta_point": value,
                    "delta_ci_low": value - 0.01,
                    "delta_ci_high": value + 0.01,
                }
                for metric, value in [
                    ("precision", -0.02),
                    ("recall", 0.06),
                    ("f1", 0.03),
                    ("brier", -0.004),
                ]
            ]
        )

        compact = probability.build_compact_bootstrap_table(bootstrap)

        self.assertEqual(compact.shape, (2, 6))
        self.assertEqual(compact["forecasting_horizon"].tolist(), ["4-month horizon", "8-month horizon"])
        self.assertIn("delta_precision", compact.columns)
        self.assertIn("delta_brier", compact.columns)
        self.assertEqual(compact.loc[0, "delta_recall"], "0.080 [0.070, 0.090]")

    def test_write_markdown_table_does_not_require_optional_tabulate(self):
        table = pd.DataFrame(
            {
                "forecasting_horizon": ["4-month horizon"],
                "delta_f1": ["0.041 [0.019, 0.063]"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "table.md"
            probability.write_markdown_table(table, path)

            text = path.read_text(encoding="utf-8")
            self.assertIn("| forecasting_horizon | delta_f1 |", text)
            self.assertIn("| 4-month horizon | 0.041 [0.019, 0.063] |", text)

    def test_region_specific_ci_skips_regions_with_too_few_countries(self):
        df = pd.DataFrame(
            {
                "region": ["East", "East", "West", "West"],
                "ADMIN0": ["A", "B", "C", "C"],
                "y_true": [1, 0, 1, 0],
                "y_pred_pooled": [1, 0, 1, 1],
                "y_pred_partitioned": [1, 0, 1, 0],
                "y_prob_pooled": [0.8, 0.2, 0.8, 0.7],
                "y_prob_partitioned": [0.8, 0.2, 0.8, 0.2],
            }
        )

        ci = probability.build_region_bootstrap_ci(df, n_bootstrap=10, seed=4, min_countries=2)

        self.assertEqual(ci["region"].unique().tolist(), ["East"])


if __name__ == "__main__":
    unittest.main()
