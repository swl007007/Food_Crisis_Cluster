import importlib.util
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "paper_artifacts"
    / "plot_region_class_prevalence.py"
)
spec = importlib.util.spec_from_file_location("plot_region_class_prevalence", SCRIPT_PATH)
region_prevalence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(region_prevalence)


class RegionClassPrevalenceTests(unittest.TestCase):
    def test_summarizes_counts_and_prevalence_by_region_and_month(self):
        predictions = pd.DataFrame(
            [
                {"FEWSNET_admin_code": "1", "month_start": "2021-02-01", "y_true": 1},
                {"FEWSNET_admin_code": "2", "month_start": "2021-02-01", "y_true": 0},
                {"FEWSNET_admin_code": "3", "month_start": "2021-02-01", "y_true": 1},
                {"FEWSNET_admin_code": "1", "month_start": "2021-06-01", "y_true": 0},
            ]
        )
        region_lookup = pd.DataFrame(
            [
                {"FEWSNET_admin_code": "1", "region": "East Africa"},
                {"FEWSNET_admin_code": "2", "region": "East Africa"},
                {"FEWSNET_admin_code": "3", "region": "West Africa"},
            ]
        )

        table = region_prevalence.build_prevalence_table(predictions, region_lookup)

        east_feb = table[
            (table["region"] == "East Africa") & (table["target_month"] == "2021-02")
        ].iloc[0]
        self.assertEqual(int(east_feb["crisis_count"]), 1)
        self.assertEqual(int(east_feb["non_crisis_count"]), 1)
        self.assertEqual(int(east_feb["total"]), 2)
        self.assertEqual(float(east_feb["crisis_prevalence"]), 0.5)

        west_feb = table[
            (table["region"] == "West Africa") & (table["target_month"] == "2021-02")
        ].iloc[0]
        self.assertEqual(int(west_feb["crisis_count"]), 1)
        self.assertEqual(int(west_feb["non_crisis_count"]), 0)
        self.assertEqual(float(west_feb["crisis_prevalence"]), 1.0)

    def test_marks_middle_east_unvalidated_window(self):
        predictions = pd.DataFrame(
            [
                {"FEWSNET_admin_code": "1", "month_start": "2021-06-01", "y_true": 1},
                {"FEWSNET_admin_code": "1", "month_start": "2021-10-01", "y_true": 1},
                {"FEWSNET_admin_code": "1", "month_start": "2023-02-01", "y_true": 0},
                {"FEWSNET_admin_code": "1", "month_start": "2023-06-01", "y_true": 0},
            ]
        )
        region_lookup = pd.DataFrame(
            [{"FEWSNET_admin_code": "1", "region": "Middle East"}]
        )

        table = region_prevalence.build_prevalence_table(predictions, region_lookup)
        statuses = dict(zip(table["target_month"], table["validation_status"]))

        self.assertEqual(statuses["2021-06"], "validated")
        self.assertEqual(statuses["2021-10"], "data not validated")
        self.assertEqual(statuses["2023-02"], "data not validated")
        self.assertEqual(statuses["2023-06"], "validated")


if __name__ == "__main__":
    unittest.main()
