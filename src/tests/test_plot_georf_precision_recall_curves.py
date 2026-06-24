import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "paper_artifacts"
    / "plot_georf_precision_recall_curves.py"
)
spec = importlib.util.spec_from_file_location("plot_georf_precision_recall_curves", SCRIPT_PATH)
pr_curves = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr_curves)


class GeoRFPrecisionRecallCurvePlotTests(unittest.TestCase):
    def test_writes_curve_points_and_plot_outputs_from_stage3_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "artifacts"
            for scope in ("fs1", "fs2", "fs3"):
                prediction_dir = root / f"result_partition_k40_compare_GF_{scope}"
                prediction_dir.mkdir(parents=True)
                pd.DataFrame(
                    {
                        "FEWSNET_admin_code": [1, 2, 3, 4],
                        "month_start": ["2021-01-01"] * 4,
                        "y_true": [1, 1, 0, 0],
                        "y_prob_pooled": [0.90, 0.50, 0.60, 0.10],
                        "y_prob_partitioned": [0.95, 0.65, 0.45, 0.05],
                    }
                ).to_csv(prediction_dir / "predictions_monthly.csv", index=False)

            outputs = pr_curves.write_precision_recall_curve_artifacts(
                source_dir=root,
                output_dir=output_dir,
                dpi=80,
            )

            curve_points = pd.read_csv(outputs["curve_points"])
            self.assertEqual(
                set(curve_points.columns),
                {
                    "scope",
                    "forecasting_horizon",
                    "model",
                    "recall",
                    "precision",
                    "threshold",
                    "average_precision",
                },
            )
            self.assertEqual(sorted(curve_points["scope"].unique().tolist()), ["fs1", "fs2", "fs3"])
            self.assertEqual(
                sorted(curve_points["forecasting_horizon"].unique().tolist()),
                ["12-month horizon", "4-month horizon", "8-month horizon"],
            )
            self.assertFalse(curve_points["forecasting_horizon"].str.contains("lag", case=False).any())
            self.assertEqual(sorted(curve_points["model"].unique().tolist()), ["partitioned", "pooled"])
            self.assertTrue(outputs["png"].exists())
            self.assertTrue(outputs["pdf"].exists())


if __name__ == "__main__":
    unittest.main()
