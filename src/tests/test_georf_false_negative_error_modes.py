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
    / "analyze_georf_false_negative_error_modes.py"
)
spec = importlib.util.spec_from_file_location("analyze_georf_false_negative_error_modes", SCRIPT_PATH)
false_modes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(false_modes)


class GeoRFFalseNegativeErrorModeTests(unittest.TestCase):
    def test_scope_display_labels_use_horizon_wording(self):
        self.assertEqual(
            false_modes.HORIZONS,
            {
                "fs1": "4-month horizon",
                "fs2": "8-month horizon",
                "fs3": "12-month horizon",
            },
        )

    def test_assign_hotspots_uses_country_and_latitude_rules(self):
        df = pd.DataFrame(
            {
                "ADMIN0": [
                    "Sudan",
                    "Afghanistan",
                    "Afghanistan",
                    "Mozambique",
                    "Mozambique",
                    "Kenya",
                ],
                "lat": [12.0, 36.0, 32.0, -20.0, -12.0, 1.0],
            }
        )

        assigned = false_modes.assign_hotspots(df)

        self.assertEqual(assigned["hotspot"].tolist()[0], "Sudan")
        self.assertEqual(assigned["hotspot"].tolist()[1], "Northern Afghanistan")
        self.assertTrue(pd.isna(assigned["hotspot"].tolist()[2]))
        self.assertEqual(assigned["hotspot"].tolist()[3], "Central/southern Mozambique")
        self.assertTrue(pd.isna(assigned["hotspot"].tolist()[4]))
        self.assertTrue(pd.isna(assigned["hotspot"].tolist()[5]))

    def test_filter_partitioned_false_negatives_uses_partitioned_prediction_only(self):
        df = pd.DataFrame(
            {
                "y_true": [1, 1, 0, 1],
                "y_pred_pooled": [0, 0, 0, 1],
                "y_pred_partitioned": [0, 1, 0, 0],
            }
        )

        out = false_modes.filter_partitioned_false_negatives(df)

        self.assertEqual(out.index.tolist(), [0, 3])

    def test_add_lag_proxy_maps_scope_to_lag_phase_column(self):
        df = pd.DataFrame(
            {
                "scope": ["fs1", "fs2", "fs3", "fs1"],
                "fews_overall_phase_lagone": [2.0, 4.0, 4.0, np.nan],
                "fews_overall_phase_lagtwo": [4.0, 2.0, 4.0, 4.0],
                "fews_overall_phase_lagthree": [4.0, 4.0, 2.0, 4.0],
            }
        )

        out = false_modes.add_lag_proxy(df)

        self.assertEqual(out["active_lag_phase"].tolist()[:3], [2.0, 2.0, 2.0])
        self.assertEqual(out["lagged_noncrisis"].tolist()[:3], [True, True, True])
        self.assertTrue(pd.isna(out["active_lag_phase"].iloc[3]))
        self.assertEqual(out["lagged_missing"].tolist(), [False, False, False, True])

    def test_add_neighbor_context_computes_mixed_neighbor_share(self):
        df = pd.DataFrame(
            {
                "FEWSNET_admin_code": ["10", "20", "30"],
                "scope": ["fs1", "fs1", "fs1"],
                "month_start": pd.to_datetime(["2021-02-01"] * 3),
                "y_true": [1, 1, 0],
                "y_pred_partitioned": [0, 1, 0],
            }
        )
        adjacency = {0: [1, 2], 1: [0], 2: [0]}
        code_to_index = {"10": 0, "20": 1, "30": 2}

        out = false_modes.add_neighbor_context(df, adjacency, code_to_index)

        first = out[out["FEWSNET_admin_code"].eq("10")].iloc[0]
        self.assertAlmostEqual(first["neighbor_actual_crisis_share"], 0.5)
        self.assertAlmostEqual(first["neighbor_predicted_crisis_share"], 0.5)
        self.assertTrue(first["mixed_neighbor_actual_state"])

    def test_build_hotspot_summary_counts_false_negative_population(self):
        df = pd.DataFrame(
            {
                "hotspot": ["Sudan", "Sudan", "Sudan"],
                "scope": ["fs1", "fs1", "fs1"],
                "forecasting_horizon": ["4-month horizon"] * 3,
                "month_start": pd.to_datetime(["2021-02-01", "2021-02-01", "2021-06-01"]),
                "y_true": [1, 1, 1],
                "y_pred_partitioned": [0, 1, 0],
                "y_prob_partitioned": [0.45, 0.70, 0.20],
                "pop": [100.0, 200.0, 300.0],
            }
        )
        false_negative = false_modes.filter_partitioned_false_negatives(df)

        summary = false_modes.build_hotspot_summary(df, false_negative)

        row = summary.iloc[0]
        self.assertEqual(row["false_negative_observations"], 2)
        self.assertAlmostEqual(row["false_negative_population"], 400.0)
        self.assertAlmostEqual(row["actual_crisis_population"], 600.0)
        self.assertAlmostEqual(row["missed_crisis_population_share"], 400.0 / 600.0)
        self.assertAlmostEqual(row["near_threshold_share"], 0.5)

    def test_write_markdown_table_avoids_optional_tabulate(self):
        table = pd.DataFrame({"hotspot": ["Sudan"], "false_negative_population": ["400"]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "table.md"
            false_modes.write_markdown_table(table, path)

            text = path.read_text(encoding="utf-8")
            self.assertIn("| hotspot | false_negative_population |", text)
            self.assertIn("| Sudan | 400 |", text)


if __name__ == "__main__":
    unittest.main()
