import importlib.util
import inspect
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_georf_partition_stability.py"
spec = importlib.util.spec_from_file_location("analyze_georf_partition_stability", SCRIPT_PATH)
stability = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stability)


class GeoRFPartitionStabilityTests(unittest.TestCase):
    def test_partition_stability_plot_uses_horizon_axis_label(self):
        source = inspect.getsource(stability.render_figure)
        old_axis_label = "Forecasting horizon" + " / " + "lag"

        self.assertIn('"Forecasting horizon"', source)
        self.assertNotIn(old_axis_label, source)

    def test_write_note_uses_horizon_only_paper_wording(self):
        summary = pd.DataFrame(
            {
                "comparison_group": ["across_years"],
                "n_pairs_total": [1],
                "n_pairs_with_metric": [1],
            }
        )
        cluster_summary = pd.DataFrame(
            {
                "plan": ["GeoRF_2018_02_fs1"],
                "year": [2018],
                "month": [2],
                "forecasting_horizon_months": [4],
                "source_scope": ["fs1"],
                "n_polygons": [10],
                "n_clusters": [2],
                "median_cluster_size": [5.0],
                "largest_cluster_share": [0.5],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            note_path = stability.write_note(Path(tmp), summary, cluster_summary)
            text = note_path.read_text(encoding="utf-8")
        old_lower = "forecasting horizon" + " / " + "lag"
        old_title = "Forecasting Horizon" + " / " + "Lag"

        self.assertIn("forecasting horizon", text)
        self.assertIn("## Cluster-Size Summary by Forecasting Horizon", text)
        self.assertNotIn(old_lower, text)
        self.assertNotIn(old_title, text)

    def test_pairwise_metrics_align_common_valid_admin_units(self):
        left = pd.DataFrame(
            {
                "FEWSNET_admin_code": ["1", "2", "3", "4"],
                "partition_id": ["a", "a", "b", "s-1"],
            }
        )
        right = pd.DataFrame(
            {
                "FEWSNET_admin_code": ["1", "2", "3", "5"],
                "partition_id": ["x", "x", "y", "z"],
            }
        )

        row = stability.pairwise_stability_row(
            {"name": "GeoRF_2018_02_fs1", "year": 2018, "month": 2, "forecasting_scope": "fs1"},
            {"name": "GeoRF_2019_02_fs1", "year": 2019, "month": 2, "forecasting_scope": "fs1"},
            left,
            right,
        )

        self.assertEqual(row["comparison_group"], "across_years")
        self.assertEqual(row["n_common_valid"], 3)
        self.assertEqual(row["adjusted_rand_index"], 1.0)
        self.assertEqual(row["normalized_mutual_information"], 1.0)

    def test_cluster_size_summary_excludes_out_of_scope_labels(self):
        plan = pd.DataFrame(
            {
                "FEWSNET_admin_code": ["1", "2", "3", "4", "5"],
                "partition_id": ["a", "a", "b", "b", "s-1"],
            }
        )

        row = stability.cluster_size_summary_row(
            {"name": "GeoRF_2018_02_fs1", "year": 2018, "month": 2, "forecasting_scope": "fs1"},
            plan,
        )

        self.assertEqual(row["n_polygons"], 4)
        self.assertEqual(row["n_clusters"], 2)
        self.assertEqual(row["min_cluster_size"], 2)
        self.assertEqual(row["median_cluster_size"], 2)
        self.assertEqual(row["max_cluster_size"], 2)
        self.assertTrue(math.isclose(row["largest_cluster_share"], 0.5))

    def test_markdown_table_does_not_require_tabulate(self):
        df = pd.DataFrame({"comparison_group": ["across_years"], "n_pairs": [3]})

        table = stability.markdown_table(df)

        self.assertIn("| comparison_group | n_pairs |", table)
        self.assertIn("| across_years | 3 |", table)

    def test_pairwise_summary_separates_metric_eligible_pairs(self):
        pairwise = pd.DataFrame(
            {
                "comparison_group": ["across_years", "across_years"],
                "n_common_valid": [0, 5],
                "adjusted_rand_index": [float("nan"), 0.25],
                "normalized_mutual_information": [float("nan"), 0.5],
            }
        )

        row = stability.summarize_pairwise_stability(pairwise).iloc[0]

        self.assertEqual(row["n_pairs_total"], 2)
        self.assertEqual(row["n_pairs_with_metric"], 1)
        self.assertEqual(row["n_pairs_without_common_valid"], 1)
        self.assertEqual(row["n_common_valid_min"], 5)
        self.assertEqual(row["adjusted_rand_index_mean"], 0.25)

    def test_appendix_table_keeps_compact_stability_fields(self):
        summary = pd.DataFrame(
            {
                "comparison_group": ["across_years"],
                "n_pairs_total": [21],
                "n_pairs_with_metric": [18],
                "n_pairs_without_common_valid": [3],
                "n_common_valid_median": [5358.0],
                "adjusted_rand_index_median": [0.0180238],
                "adjusted_rand_index_p25": [0.00232386],
                "adjusted_rand_index_p75": [0.199617],
                "normalized_mutual_information_median": [0.0869823],
                "normalized_mutual_information_p25": [0.0188972],
                "normalized_mutual_information_p75": [0.238686],
            }
        )

        appendix = stability.build_appendix_stability_table(summary)

        self.assertEqual(
            appendix.columns.tolist(),
            [
                "Comparison axis",
                "Total pairs",
                "Pairs used for metrics",
                "Median common polygons",
                "ARI median [IQR]",
                "NMI median [IQR]",
            ],
        )
        self.assertEqual(appendix.loc[0, "Comparison axis"], "Across years")
        self.assertEqual(appendix.loc[0, "Pairs used for metrics"], "18/21")
        self.assertEqual(appendix.loc[0, "ARI median [IQR]"], "0.018 [0.002, 0.200]")
        self.assertEqual(appendix.loc[0, "NMI median [IQR]"], "0.087 [0.019, 0.239]")


if __name__ == "__main__":
    unittest.main()
