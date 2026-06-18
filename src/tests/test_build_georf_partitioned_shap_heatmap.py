import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_georf_partitioned_shap_heatmap.py"
spec = importlib.util.spec_from_file_location("build_georf_partitioned_shap_heatmap", SCRIPT_PATH)
shap_heatmap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shap_heatmap)


class GeoRFPartitionedShapHeatmapTests(unittest.TestCase):
    def test_feature_groups_exclude_secondary_and_preserve_display_order(self):
        self.assertEqual(
            list(shap_heatmap.FEATURE_GROUPS.keys()),
            ["weather", "agri", "conflict", "econ", "food_prices", "geographic", "lag"],
        )
        self.assertEqual(
            [meta["display"] for meta in shap_heatmap.FEATURE_GROUPS.values()],
            ["Weather", "Agri", "Conflict", "Econ", "Food Prices", "Geographic", "Lag"],
        )
        self.assertNotIn("secondary", shap_heatmap.FEATURE_GROUPS)

    def test_assign_feature_group_maps_base_columns_and_target_lags(self):
        self.assertEqual(shap_heatmap.assign_feature_group("Rainf_zscore"), "weather")
        self.assertEqual(shap_heatmap.assign_feature_group("Rainf_zscore_lag4m"), "weather")
        self.assertEqual(shap_heatmap.assign_feature_group("event_count_battles_w5_lag_8"), "conflict")
        self.assertEqual(shap_heatmap.assign_feature_group("WFP_Price_std_lag12m"), "food_prices")
        self.assertEqual(shap_heatmap.assign_feature_group("sg_soc_5-15cm_lag_4"), "geographic")
        self.assertEqual(shap_heatmap.assign_feature_group("fews_ipc_lag_4"), "lag")
        self.assertEqual(shap_heatmap.assign_feature_group("fews_ipc_crisis_lag_8"), "lag")
        self.assertIsNone(shap_heatmap.assign_feature_group("unmatched_feature"))

    def test_resolve_feature_group_matches_reports_missing_base_columns(self):
        feature_names = [
            "Rainf_zscore_lag4m",
            "Tair_zscore",
            "crop_lag_4",
            "event_count_battles_w5",
            "GDP_lag8m",
            "WFP_Price_std",
            "slope",
            "fews_ipc_crisis_lag_4",
            "unmatched_feature",
        ]

        resolved = shap_heatmap.resolve_feature_group_matches(feature_names)

        self.assertEqual(resolved.feature_to_group["Rainf_zscore_lag4m"], "weather")
        self.assertEqual(resolved.feature_to_group["crop_lag_4"], "agri")
        self.assertEqual(resolved.feature_to_group["event_count_battles_w5"], "conflict")
        self.assertEqual(resolved.feature_to_group["GDP_lag8m"], "econ")
        self.assertEqual(resolved.feature_to_group["WFP_Price_std"], "food_prices")
        self.assertEqual(resolved.feature_to_group["slope"], "geographic")
        self.assertEqual(resolved.feature_to_group["fews_ipc_crisis_lag_4"], "lag")
        self.assertEqual(resolved.unmatched_features, ["unmatched_feature"])
        self.assertGreater(len(resolved.missing_base_columns["weather"]), 0)

    def test_collapse_shap_values_handles_binary_list_and_three_dimensional_layouts(self):
        class0 = np.array([[1.0, -2.0], [3.0, -4.0]])
        class1 = np.array([[5.0, -6.0], [7.0, -8.0]])
        observed_list = shap_heatmap.collapse_shap_values([class0, class1], n_samples=2, n_features=2)
        np.testing.assert_allclose(observed_list, np.array([[3.0, -4.0], [5.0, -6.0]]))

        raw_samples_features_classes = np.stack([class0, class1], axis=2)
        observed_sfc = shap_heatmap.collapse_shap_values(raw_samples_features_classes, n_samples=2, n_features=2)
        np.testing.assert_allclose(observed_sfc, np.array([[3.0, -4.0], [5.0, -6.0]]))

        raw_classes_samples_features = np.stack([class0, class1], axis=0)
        observed_csf = shap_heatmap.collapse_shap_values(raw_classes_samples_features, n_samples=2, n_features=2)
        np.testing.assert_allclose(observed_csf, np.array([[-0.5, -0.5], [-0.5, -0.5]]))

    def test_collapse_shap_values_uses_deterministic_three_dimensional_shape_priority(self):
        class0 = np.array([[1.0, -2.0], [3.0, -4.0], [5.0, -6.0]])
        class1 = np.array([[7.0, -8.0], [9.0, -10.0], [11.0, -12.0]])

        raw_samples_features_classes = np.stack([class0, class1], axis=2)
        observed_sfc = shap_heatmap.collapse_shap_values(raw_samples_features_classes, n_samples=3, n_features=2)
        np.testing.assert_allclose(observed_sfc, np.array([[4.0, -5.0], [6.0, -7.0], [8.0, -9.0]]))

        raw_classes_samples_features = np.stack([class0, class1], axis=0)
        observed_csf = shap_heatmap.collapse_shap_values(raw_classes_samples_features, n_samples=3, n_features=2)
        np.testing.assert_allclose(observed_csf, np.array([[4.0, -5.0], [6.0, -7.0], [8.0, -9.0]]))

    def test_group_mean_abs_and_share_normalization(self):
        shap_values = np.array(
            [
                [1.0, -3.0, 2.0, 4.0],
                [-1.0, 5.0, -2.0, 6.0],
            ]
        )
        feature_names = ["Rainf_zscore", "Tair_zscore", "GDP", "fews_ipc_lag_4"]
        resolved = shap_heatmap.resolve_feature_group_matches(feature_names)

        row = shap_heatmap.build_monthly_group_rows(
            scope="fs1",
            horizon_months=4,
            target_month="2021-02",
            shap_values=shap_values,
            feature_names=feature_names,
            resolved=resolved,
            fallback_samples=1,
            evaluated_samples=3,
        )

        monthly = pd.DataFrame(row)
        weather = monthly[monthly["group"] == "weather"].iloc[0]
        econ = monthly[monthly["group"] == "econ"].iloc[0]
        lag = monthly[monthly["group"] == "lag"].iloc[0]

        self.assertAlmostEqual(weather["raw_mean_abs_shap"], 5.0)
        self.assertAlmostEqual(econ["raw_mean_abs_shap"], 2.0)
        self.assertAlmostEqual(lag["raw_mean_abs_shap"], 5.0)
        self.assertAlmostEqual(monthly["group_share"].sum(), 1.0)
        self.assertAlmostEqual(weather["group_share"], 5.0 / 12.0)
        self.assertEqual(int(weather["fallback_samples"]), 1)
        self.assertEqual(int(weather["evaluated_samples"]), 3)

    def test_summarize_group_shares_uses_sample_standard_deviation(self):
        monthly = pd.DataFrame(
            [
                {"scope": "fs1", "horizon_months": 4, "target_month": "2021-02", "group": "weather", "display_group": "Weather", "group_share": 0.10},
                {"scope": "fs1", "horizon_months": 4, "target_month": "2021-06", "group": "weather", "display_group": "Weather", "group_share": 0.20},
                {"scope": "fs1", "horizon_months": 4, "target_month": "2021-02", "group": "agri", "display_group": "Agri", "group_share": 0.30},
                {"scope": "fs1", "horizon_months": 4, "target_month": "2021-06", "group": "agri", "display_group": "Agri", "group_share": 0.50},
            ]
        )

        summary = shap_heatmap.summarize_group_shares(monthly, expected_month_count=2)
        weather = summary[summary["group"] == "weather"].iloc[0]
        agri = summary[summary["group"] == "agri"].iloc[0]

        self.assertAlmostEqual(weather["mean_share"], 0.15)
        self.assertAlmostEqual(weather["sd_share"], math.sqrt(0.005))
        self.assertEqual(int(weather["n_months"]), 2)
        self.assertAlmostEqual(agri["mean_share"], 0.40)

    def test_summarize_group_shares_rejects_extra_months(self):
        monthly = pd.DataFrame(
            [
                {"scope": "fs1", "horizon_months": 4, "target_month": "2021-02", "group": "weather", "display_group": "Weather", "group_share": 0.10},
                {"scope": "fs1", "horizon_months": 4, "target_month": "2021-06", "group": "weather", "display_group": "Weather", "group_share": 0.20},
                {"scope": "fs1", "horizon_months": 4, "target_month": "2021-10", "group": "weather", "display_group": "Weather", "group_share": 0.30},
            ]
        )

        with self.assertRaisesRegex(ValueError, "Expected 2 months"):
            shap_heatmap.summarize_group_shares(monthly, expected_month_count=2)

    def test_build_heatmap_matrix_preserves_rows_and_columns(self):
        rows = []
        for group, meta in shap_heatmap.FEATURE_GROUPS.items():
            for scope, months in shap_heatmap.SCOPE_TO_HORIZON_MONTHS.items():
                rows.append(
                    {
                        "group": group,
                        "display_group": meta["display"],
                        "scope": scope,
                        "forecasting_horizon": f"{months}-month horizon",
                        "mean_share": 0.10,
                        "sd_share": 0.02,
                        "n_months": 12,
                    }
                )
        summary = pd.DataFrame(rows)

        values, annotations = shap_heatmap.build_heatmap_matrices(summary)

        self.assertEqual(list(values.index), ["Weather", "Agri", "Conflict", "Econ", "Food Prices", "Geographic", "Lag"])
        self.assertEqual(list(values.columns), ["4-month horizon", "8-month horizon", "12-month horizon"])
        self.assertEqual(annotations.loc["Weather", "4-month horizon"], "10.0%\n+/- 2.0")

    def test_evaluation_months_are_february_june_october_for_each_year(self):
        observed = shap_heatmap.evaluation_months("2021-01", "2024-12")
        self.assertEqual(len(observed), 12)
        self.assertEqual(str(observed[0]), "2021-02")
        self.assertEqual(str(observed[-1]), "2024-10")
        self.assertEqual(sorted(set(month.month for month in observed)), [2, 6, 10])

    def test_select_partition_map_uses_month_specific_maps(self):
        maps = {
            "general": Path("general.csv"),
            "m2": Path("feb.csv"),
            "m6": Path("jun.csv"),
            "m10": Path("oct.csv"),
        }

        self.assertEqual(shap_heatmap.select_partition_map(pd.Period("2021-02", freq="M"), maps), Path("feb.csv"))
        self.assertEqual(shap_heatmap.select_partition_map(pd.Period("2021-06", freq="M"), maps), Path("jun.csv"))
        self.assertEqual(shap_heatmap.select_partition_map(pd.Period("2021-10", freq="M"), maps), Path("oct.csv"))
        self.assertEqual(shap_heatmap.select_partition_map(pd.Period("2021-03", freq="M"), maps), Path("general.csv"))


if __name__ == "__main__":
    unittest.main()
