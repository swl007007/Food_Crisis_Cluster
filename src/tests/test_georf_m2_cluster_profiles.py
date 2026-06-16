import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_georf_m2_cluster_profiles.py"
spec = importlib.util.spec_from_file_location("analyze_georf_m2_cluster_profiles", SCRIPT_PATH)
cluster_profiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cluster_profiles)


class GeoRFM2ClusterProfilesTests(unittest.TestCase):
    def test_attach_clusters_normalizes_admin_codes_and_inner_joins(self):
        records = pd.DataFrame(
            {
                "FEWSNET_admin_code": [101.0, " 102 ", "103.0", "999"],
                "value": [1, 2, 3, 4],
            }
        )
        mapping = pd.DataFrame(
            {
                "FEWSNET_admin_code": ["101", "102.0", 103],
                "cluster_id": [7, 8, 9],
            }
        )

        attached = cluster_profiles.attach_clusters(records, mapping)

        self.assertEqual(attached["FEWSNET_admin_code"].tolist(), ["101", "102", "103"])
        self.assertEqual(attached["cluster_id"].tolist(), [7, 8, 9])
        self.assertEqual(attached["value"].tolist(), [1, 2, 3])

    def test_filter_february_predictions_adds_target_month_and_numeric_labels(self):
        predictions = pd.DataFrame(
            {
                "FEWSNET_admin_code": ["1", "2", "3"],
                "month_start": ["2021-02-01", "2021-06-01", "2022-02-28"],
                "y_true": ["1", 0.0, "0"],
                "y_pred_partitioned": [1.0, "1", "0"],
            }
        )

        filtered = cluster_profiles.filter_february_predictions(predictions)

        self.assertEqual(filtered["target_month"].tolist(), ["2021-02", "2022-02"])
        self.assertEqual(filtered["FEWSNET_admin_code"].tolist(), ["1", "3"])
        self.assertTrue(pd.api.types.is_numeric_dtype(filtered["y_true"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(filtered["y_pred_partitioned"]))
        self.assertEqual(filtered["y_true"].tolist(), [1, 0])
        self.assertEqual(filtered["y_pred_partitioned"].tolist(), [1, 0])

    def test_filter_february_predictions_preserves_fractional_labels_for_invalid_modes(self):
        predictions = pd.DataFrame(
            {
                "FEWSNET_admin_code": ["1", "2", "3", "4"],
                "month_start": ["2021-02-01", "2021-02-01", "2021-02-01", "2021-02-01"],
                "y_true": [1, 0, 0.5, 1.9],
                "y_pred_partitioned": [1, 0, 1, 1],
            }
        )

        filtered = cluster_profiles.filter_february_predictions(predictions)
        labeled = cluster_profiles.add_error_modes(filtered)

        self.assertEqual(labeled["error_mode"].tolist(), ["TP", "TN", "invalid", "invalid"])

    def test_filter_february_panel_aligns_to_prediction_target_months(self):
        panel = pd.DataFrame(
            {
                "FEWSNET_admin_code": ["1", "1", "1", "1"],
                "date": ["2020-02", "2021-02", "2021-06", "2024-02"],
                "market_access": [10, 20, 30, 40],
            }
        )

        filtered = cluster_profiles.filter_february_panel(panel, ["2021-02", "2024-02"])

        self.assertEqual(filtered["date"].tolist(), ["2021-02", "2024-02"])
        self.assertEqual(filtered["market_access"].tolist(), [20, 40])

    def test_add_error_modes_labels_tp_fp_fn_tn(self):
        df = pd.DataFrame(
            {
                "y_true": [1, 0, 1, 0],
                "y_pred_partitioned": [1, 1, 0, 0],
            }
        )

        labeled = cluster_profiles.add_error_modes(df)

        self.assertEqual(labeled["error_mode"].tolist(), ["TP", "FP", "FN", "TN"])

    def test_add_error_modes_labels_fractional_values_invalid(self):
        df = pd.DataFrame(
            {
                "y_true": [0.5, 1, 0, 1.9],
                "y_pred_partitioned": [1, 0.25, 0, 1],
            }
        )

        labeled = cluster_profiles.add_error_modes(df)

        self.assertEqual(labeled["error_mode"].tolist(), ["invalid", "invalid", "TN", "invalid"])

    def test_build_error_summary_returns_cluster_counts_and_shares(self):
        predictions = pd.DataFrame(
            {
                "cluster_id": [1, 1, 1, 2],
                "y_true": [1, 0, 1, 0],
                "y_pred_partitioned": [1, 1, 0, 0],
            }
        )

        summary = cluster_profiles.build_error_summary(predictions)
        row = summary[summary["cluster_id"].eq(1)].iloc[0]
        tn_row = summary[summary["cluster_id"].eq(2)].iloc[0]

        self.assertEqual(row["n_observations"], 3)
        self.assertEqual(row["tp_count"], 1)
        self.assertEqual(row["fp_count"], 1)
        self.assertEqual(row["fn_count"], 1)
        self.assertEqual(row["tn_count"], 0)
        self.assertAlmostEqual(row["crisis_prevalence"], 2 / 3)
        self.assertAlmostEqual(row["tp_share"], 1 / 3)
        self.assertAlmostEqual(row["fp_share"], 1 / 3)
        self.assertAlmostEqual(row["fn_share"], 1 / 3)
        self.assertAlmostEqual(row["tn_share"], 0.0)
        self.assertEqual(row["main_error_mode"], "FN/FP/TP")
        self.assertEqual(tn_row["main_error_mode"], "TN-dominant")

    def test_build_error_summary_returns_expected_columns_for_empty_input(self):
        predictions = pd.DataFrame(columns=["cluster_id", "y_true", "y_pred_partitioned"])

        summary = cluster_profiles.build_error_summary(predictions)

        self.assertTrue(summary.empty)
        self.assertEqual(
            summary.columns.tolist(),
            [
                "cluster_id",
                "n_observations",
                "crisis_prevalence",
                "tp_count",
                "fp_count",
                "fn_count",
                "tn_count",
                "main_error_mode",
                "tp_share",
                "fp_share",
                "fn_share",
                "tn_share",
            ],
        )

    def test_build_error_summary_excludes_missing_cluster_id_rows(self):
        predictions = pd.DataFrame(
            {
                "cluster_id": [1, None, pd.NA],
                "y_true": [1, 0, 1],
                "y_pred_partitioned": [1, 1, 0],
            }
        )

        summary = cluster_profiles.build_error_summary(predictions)

        self.assertEqual(summary["cluster_id"].tolist(), [1])
        self.assertEqual(summary["n_observations"].tolist(), [1])
        self.assertEqual(summary["tp_count"].tolist(), [1])

    def test_build_error_summary_returns_empty_for_all_missing_cluster_ids(self):
        predictions = pd.DataFrame(
            {
                "cluster_id": [None, pd.NA],
                "y_true": [0, 1],
                "y_pred_partitioned": [1, 0],
            }
        )

        summary = cluster_profiles.build_error_summary(predictions)

        self.assertTrue(summary.empty)
        self.assertEqual(summary.columns.tolist(), cluster_profiles.ERROR_SUMMARY_COLUMNS)

    def test_build_dominant_aez_selects_largest_cluster_mean_and_reports_share(self):
        panel = pd.DataFrame(
            {
                "cluster_id": [2, 2, 2, 5, 5],
                "AEZ_arid": [1, 1, 0, 0, 0],
                "AEZ_humid": [0, 0, 1, 1, 1],
                "AEZ_temperate": [0, 0, 0, 0, 0],
            }
        )

        result = cluster_profiles.build_dominant_aez(panel, ["AEZ_arid", "AEZ_humid", "AEZ_temperate"])

        self.assertEqual(result["cluster_id"].tolist(), [2, 5])
        cluster_2 = result[result["cluster_id"].eq(2)].iloc[0]
        cluster_5 = result[result["cluster_id"].eq(5)].iloc[0]
        self.assertEqual(cluster_2["dominant_aez"], "AEZ_arid")
        self.assertAlmostEqual(cluster_2["dominant_aez_share"], 2 / 3)
        self.assertEqual(cluster_5["dominant_aez"], "AEZ_humid")
        self.assertAlmostEqual(cluster_5["dominant_aez_share"], 1.0)

    def test_tertile_label_maps_high_moderate_and_low(self):
        self.assertEqual(cluster_profiles.tertile_label(0.8), "high")
        self.assertEqual(cluster_profiles.tertile_label(0.5), "moderate")
        self.assertEqual(cluster_profiles.tertile_label(0.1), "low")

    def test_cluster_percentile_scales_values_across_zero_to_one(self):
        percentiles = cluster_profiles.cluster_percentile(pd.Series([10, 20, 30]))

        self.assertAlmostEqual(percentiles.iloc[0], 0.0)
        self.assertAlmostEqual(percentiles.iloc[1], 0.5)
        self.assertAlmostEqual(percentiles.iloc[2], 1.0)

    def test_cluster_percentile_returns_neutral_for_constant_values(self):
        percentiles = cluster_profiles.cluster_percentile(pd.Series([10, 10, 10]))

        self.assertEqual(percentiles.tolist(), [0.5, 0.5, 0.5])

    def test_build_market_profile_labels_low_moderate_high_for_three_clusters(self):
        panel = pd.DataFrame(
            {
                "cluster_id": [1, 1, 2, 2, 3, 3],
                "market_access": [10, 10, 20, 20, 30, 30],
            }
        )

        profile = cluster_profiles.build_market_profile(panel)

        self.assertEqual(profile["cluster_id"].tolist(), [1, 2, 3])
        self.assertEqual(profile["market_market_access_percentile"].tolist(), [0.0, 0.5, 1.0])
        self.assertEqual(profile["market_market_access_label"].tolist(), ["low", "moderate", "high"])

    def test_load_region_map_uses_absolute_script_path(self):
        region_map = cluster_profiles.load_region_map()

        self.assertEqual(region_map["Kenya"], "East Africa")
        self.assertEqual(region_map["Niger"], "West Africa")

    def test_build_market_profile_returns_neutral_labels_for_missing_columns(self):
        panel = pd.DataFrame({"cluster_id": [1, 1, 2]})

        profile = cluster_profiles.build_market_profile(panel)

        self.assertEqual(profile["cluster_id"].tolist(), [1, 2])
        self.assertEqual(profile["market_market_access_label"].tolist(), ["moderate", "moderate"])
        self.assertEqual(profile["market_market_distance_label"].tolist(), ["moderate", "moderate"])

    def test_build_conflict_profile_returns_neutral_labels_for_missing_columns(self):
        panel = pd.DataFrame({"cluster_id": [1, 1, 2]})

        profile = cluster_profiles.build_conflict_profile(panel)

        self.assertEqual(profile["cluster_id"].tolist(), [1, 2])
        self.assertEqual(profile["conflict_distance_to_nearest_acled_label"].tolist(), ["moderate", "moderate"])
        self.assertEqual(profile["conflict_event_count_battles_label"].tolist(), ["moderate", "moderate"])

    def test_build_country_region_summary_exposes_countries_or_regions_included(self):
        mapping = pd.DataFrame(
            {
                "FEWSNET_admin_code": ["1", "2", "3"],
                "cluster_id": [4, 4, 9],
            }
        )
        region_lookup = pd.DataFrame(
            {
                "FEWSNET_admin_code": ["1.0", "2", "3"],
                "ADMIN0": ["Kenya", "Somalia", "Niger"],
                "region": ["East Africa", "East Africa", "West Africa"],
            }
        )

        summary = cluster_profiles.build_country_region_summary(mapping, region_lookup)

        self.assertIn("countries_or_regions_included", summary.columns)
        self.assertNotIn("country_region_summary", summary.columns)
        cluster_4 = summary[summary["cluster_id"].eq(4)].iloc[0]
        self.assertEqual(cluster_4["n_polygons"], 2)
        self.assertEqual(cluster_4["dominant_region"], "East Africa")
        self.assertEqual(cluster_4["countries_or_regions_included"], "Kenya, Somalia")

    def test_build_intercluster_similarity_returns_long_format_and_diagonal_one(self):
        df = pd.DataFrame(
            {
                "cluster_id": [1, 1, 2, 2],
                "x": [1.0, 1.0, 0.0, 0.0],
                "y": [0.0, 0.0, 1.0, 1.0],
            }
        )

        matrix = cluster_profiles.build_intercluster_similarity(df, ["x", "y"], "market")

        self.assertEqual(matrix.columns.tolist(), ["profile_type", "cluster_i", "cluster_j", "similarity"])
        self.assertEqual(len(matrix), 4)
        diag = matrix[matrix["cluster_i"].eq(matrix["cluster_j"])]
        self.assertTrue((diag["similarity"].round(6) == 1.0).all())
        self.assertEqual(set(matrix["profile_type"]), {"market"})

    def test_upper_triangle_mask_hides_only_duplicate_cells(self):
        matrix = pd.DataFrame(
            [[1.0, 0.2, 0.3], [0.2, 1.0, 0.4], [0.3, 0.4, 1.0]],
            index=[0, 1, 2],
            columns=[0, 1, 2],
        )

        mask = cluster_profiles.upper_triangle_mask(matrix)

        self.assertFalse(mask.iloc[0, 0])
        self.assertTrue(mask.iloc[0, 1])
        self.assertTrue(mask.iloc[0, 2])
        self.assertFalse(mask.iloc[1, 0])
        self.assertFalse(mask.iloc[1, 1])
        self.assertTrue(mask.iloc[1, 2])
        self.assertFalse(mask.iloc[2, 0])
        self.assertFalse(mask.iloc[2, 1])
        self.assertFalse(mask.iloc[2, 2])

    def test_build_feature_cohesion_is_between_zero_and_one_and_sets_profile_type(self):
        df = pd.DataFrame(
            {
                "cluster_id": [1, 1, 2, 2],
                "x": [1.0, 1.1, 0.0, 0.1],
                "y": [0.0, 0.1, 1.0, 1.1],
            }
        )

        cohesion = cluster_profiles.build_feature_cohesion(df, ["x", "y"], "market")

        self.assertEqual(cohesion.columns.tolist(), ["profile_type", "cluster_id", "cohesion"])
        self.assertTrue(cohesion["cohesion"].between(0, 1).all())
        self.assertEqual(set(cohesion["profile_type"]), {"market"})

    def test_build_error_cohesion_uses_dominant_error_mode_share(self):
        df = pd.DataFrame(
            {
                "cluster_id": [1, 1, 1, 2],
                "error_mode": ["TN", "TN", "FP", "FN"],
            }
        )

        cohesion = cluster_profiles.build_error_cohesion(df)

        self.assertAlmostEqual(cohesion.loc[cohesion["cluster_id"].eq(1), "cohesion"].iloc[0], 2 / 3)
        self.assertAlmostEqual(cohesion.loc[cohesion["cluster_id"].eq(2), "cohesion"].iloc[0], 1.0)

    def test_plot_similarity_figure_handles_empty_profile_panels(self):
        similarities = pd.DataFrame(
            {
                "profile_type": ["error_mode"],
                "cluster_i": [1],
                "cluster_j": [1],
                "similarity": [1.0],
            }
        )
        cohesion = pd.DataFrame(
            {
                "profile_type": ["error_mode"],
                "cluster_id": [1],
                "cohesion": [1.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "empty_profile_guard.png"
            cluster_profiles.plot_similarity_figure(similarities, cohesion, output_path, dpi=72)

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
