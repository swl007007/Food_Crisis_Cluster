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
                {
                    "scope": "fs1",
                    "forecasting_horizon": "4-month lag",
                    "model": "pooled",
                    "precision": 0.7,
                    "recall": 0.5,
                    "f1": 0.58,
                },
                {
                    "scope": "fs1",
                    "forecasting_horizon": "4-month lag",
                    "model": "partitioned",
                    "precision": 0.75,
                    "recall": 0.6,
                    "f1": 0.67,
                },
                {
                    "scope": "fs1",
                    "forecasting_horizon": "4-month lag",
                    "model": "partitioned_thresholded",
                    "precision": 0.70,
                    "recall": 0.7,
                    "f1": 0.70,
                },
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
