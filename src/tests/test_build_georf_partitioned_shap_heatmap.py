import importlib.util
import io
import math
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
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

    def test_write_heatmap_creates_non_empty_png_and_pdf(self):
        rows = []
        for group, meta in shap_heatmap.FEATURE_GROUPS.items():
            for scope in shap_heatmap.SCOPE_TO_HORIZON_MONTHS:
                rows.append(
                    {
                        "group": group,
                        "display_group": meta["display"],
                        "scope": scope,
                        "forecasting_horizon": shap_heatmap.HORIZON_LABELS[scope],
                        "mean_share": 0.10,
                        "sd_share": 0.02,
                        "n_months": 12,
                    }
                )
        summary = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmp:
            outputs = shap_heatmap.write_heatmap(summary, Path(tmp), dpi=72)

            self.assertTrue(outputs["png"].exists())
            self.assertTrue(outputs["pdf"].exists())
            self.assertGreater(outputs["png"].stat().st_size, 0)
            self.assertGreater(outputs["pdf"].stat().st_size, 0)

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

    def test_resolve_path_converts_windows_backslash_and_slash_paths(self):
        original_mount_root = shap_heatmap.WSL_MOUNT_ROOT
        with tempfile.TemporaryDirectory() as tmp:
            mount_root = Path(tmp)
            expected = mount_root / "c" / "Users" / "swl00" / "example.csv"
            expected.parent.mkdir(parents=True)
            expected.write_text("ok", encoding="utf-8")
            shap_heatmap.WSL_MOUNT_ROOT = mount_root
            try:
                self.assertEqual(
                    shap_heatmap.resolve_path(r"C:\Users\swl00\example.csv"),
                    expected,
                )
                self.assertEqual(
                    shap_heatmap.resolve_path("C:/Users/swl00/example.csv"),
                    expected,
                )
            finally:
                shap_heatmap.WSL_MOUNT_ROOT = original_mount_root

    def test_default_partition_maps_for_scope_uses_current_refined_stage3_filenames(self):
        maps = shap_heatmap.default_partition_maps_for_scope(Path("stage3"), "fs1")

        self.assertEqual(
            maps["general"].name,
            "cluster_mapping_k40_nc17_general_refined_contig3.csv",
        )
        self.assertEqual(
            maps["m2"].name,
            "cluster_mapping_k40_nc13_m2_refined_contig3.csv",
        )
        self.assertEqual(
            maps["m6"].name,
            "cluster_mapping_k40_nc11_m6_refined_contig3.csv",
        )
        self.assertEqual(
            maps["m10"].name,
            "cluster_mapping_k40_nc16_m10_refined_contig3.csv",
        )

    def test_validate_partition_maps_raises_on_missing_files_and_passes_existing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing = root / "existing.csv"
            existing.write_text("admin_code,cluster\n1,1\n", encoding="utf-8")
            missing = root / "missing.csv"

            with self.assertRaisesRegex(FileNotFoundError, "fs1:m2"):
                shap_heatmap.validate_partition_maps(
                    {"fs1": {"general": existing, "m2": missing}}
                )

            shap_heatmap.validate_partition_maps({"fs1": {"general": existing}})

    def test_write_summary_outputs_creates_csv_manifest_and_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            monthly = pd.DataFrame(
                [
                    {
                        "scope": "fs1",
                        "horizon_months": 4,
                        "forecasting_horizon": "4-month horizon",
                        "target_month": "2021-02",
                        "group": "weather",
                        "display_group": "Weather",
                        "raw_mean_abs_shap": 2.0,
                        "group_share": 1.0,
                        "matched_feature_count": 2,
                        "normalization_denominator": 2.0,
                        "fallback_samples": 0,
                        "evaluated_samples": 4,
                    }
                ]
            )
            summary = pd.DataFrame(
                [
                    {
                        "scope": "fs1",
                        "horizon_months": 4,
                        "forecasting_horizon": "4-month horizon",
                        "group": "weather",
                        "display_group": "Weather",
                        "mean_share": 1.0,
                        "sd_share": 0.0,
                        "n_months": 1,
                    }
                ]
            )
            manifest = {
                "source_csv": "source.csv",
                "scope_to_horizon_months": {"fs1": 4},
                "evaluated_months": ["2021-02"],
                "partition_maps": {"fs1": {"m2": "map.csv"}},
                "rf_params": {"n_estimators": 100},
                "feature_group_matches": {"weather": ["Rainf_zscore"]},
                "feature_group_missing_base_columns": {"weather": []},
                "fallback_sample_counts": [{"scope": "fs1", "target_month": "2021-02", "fallback_samples": 0}],
            }

            outputs = shap_heatmap.write_tabular_outputs(
                monthly=monthly,
                summary=summary,
                manifest=manifest,
                output_dir=output_dir,
            )

            self.assertTrue(outputs["monthly_csv"].exists())
            self.assertTrue(outputs["summary_csv"].exists())
            self.assertTrue(outputs["manifest_json"].exists())
            self.assertTrue(outputs["note_md"].exists())
            self.assertIn("relative SHAP attribution shares", outputs["note_md"].read_text(encoding="utf-8"))
            loaded_manifest = pd.read_json(outputs["manifest_json"], typ="series")
            self.assertEqual(loaded_manifest["source_csv"], "source.csv")
            self.assertEqual(
                sorted(loaded_manifest["output_paths"]),
                ["manifest_json", "monthly_csv", "note_md", "summary_csv"],
            )

    def test_sampled_indices_are_deterministic_and_sorted(self):
        observed = shap_heatmap.sampled_indices(
            n_rows=10,
            max_samples=4,
            random_state=7,
        )

        self.assertEqual(observed.tolist(), [5, 6, 8, 9])
        self.assertEqual(
            shap_heatmap.sampled_indices(3, max_samples=4, random_state=7).tolist(),
            [0, 1, 2],
        )
        self.assertEqual(
            shap_heatmap.sampled_indices(3, max_samples=0, random_state=7).tolist(),
            [0, 1, 2],
        )

    def test_shap_values_for_partitioned_models_dispatches_local_models_and_counts_fallbacks(self):
        class FakeExplainer:
            def __init__(self, model):
                self.model = model

            def shap_values(self, X_partition):
                base = float(self.model)
                return np.full((X_partition.shape[0], X_partition.shape[1]), base)

        fake_shap = types.SimpleNamespace(TreeExplainer=FakeExplainer)
        original_shap = sys.modules.get("shap")
        sys.modules["shap"] = fake_shap
        try:
            X_test = np.arange(15, dtype=float).reshape(5, 3)
            X_group_test = np.array([2, -1, 1, 99, 2])

            shap_values, diagnostics = shap_heatmap.shap_values_for_partitioned_models(
                models={1: 10, 2: 20},
                X_test=X_test,
                X_group_test=X_group_test,
                feature_names=["a", "b", "c"],
                max_samples_per_month=0,
                random_state=5,
            )
        finally:
            if original_shap is None:
                sys.modules.pop("shap", None)
            else:
                sys.modules["shap"] = original_shap

        np.testing.assert_allclose(
            shap_values,
            np.array(
                [
                    [10.0, 10.0, 10.0],
                    [20.0, 20.0, 20.0],
                    [20.0, 20.0, 20.0],
                ]
            ),
        )
        self.assertEqual(
            diagnostics,
            {
                "selected_samples": 5,
                "evaluated_samples": 3,
                "fallback_samples": 2,
                "missing_model_samples": 1,
                "unmapped_samples": 1,
            },
        )

    def test_sort_panel_for_feature_alignment_orders_by_admin_code_and_date(self):
        unsorted = pd.DataFrame(
            {
                "FEWSNET_admin_code": [2, 1, 1, 2],
                "date": ["2021-02-01", "2021-02-01", "2021-01-01", "2021-01-01"],
                "latitude": [20.0, 11.0, 10.0, 21.0],
                "longitude": [120.0, 111.0, 110.0, 121.0],
            }
        )

        sorted_df = shap_heatmap.sort_panel_for_feature_alignment(unsorted)

        self.assertEqual(sorted_df["FEWSNET_admin_code"].tolist(), [1, 1, 2, 2])
        self.assertEqual(
            sorted_df["date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2021-01-01", "2021-02-01", "2021-01-01", "2021-02-01"],
        )
        self.assertEqual(sorted_df["latitude"].tolist(), [10.0, 11.0, 21.0, 20.0])
        self.assertEqual(sorted_df.index.tolist(), [0, 1, 2, 3])

    def test_prepare_scope_context_sorts_shared_runtime_inputs(self):
        unsorted = pd.DataFrame(
            {
                "FEWSNET_admin_code": [2, 1, 1, 2],
                "date": ["2021-02-01", "2021-02-01", "2021-01-01", "2021-01-01"],
                "latitude": [20.0, 11.0, 10.0, 21.0],
                "longitude": [120.0, 111.0, 110.0, 121.0],
                "fews_ipc_crisis": [0, 1, 0, 1],
            }
        )
        captured = {}

        def fake_load_and_preprocess_data(_path):
            return unsorted.copy()

        def fake_prepare_features(df, X_group, X_loc, forecasting_scope):
            captured["df"] = df.copy()
            captured["X_group"] = X_group.copy()
            captured["X_loc"] = X_loc.copy()
            captured["forecasting_scope"] = forecasting_scope
            return (
                np.arange(len(df) * 2, dtype=float).reshape(len(df), 2),
                df["fews_ipc_crisis"].to_numpy(),
                [],
                [],
                df["date"].dt.year.to_numpy(),
                np.ones(len(df), dtype=int),
                df["date"],
                ["feature_a", "feature_b"],
            )

        context = shap_heatmap.prepare_scope_context(
            "dummy.csv",
            2,
            load_data_fn=fake_load_and_preprocess_data,
            prepare_features_fn=fake_prepare_features,
        )

        expected_admin_codes = [1, 1, 2, 2]
        expected_dates = ["2021-01-01", "2021-02-01", "2021-01-01", "2021-02-01"]
        expected_x_loc = np.array(
            [
                [10.0, 110.0],
                [11.0, 111.0],
                [21.0, 121.0],
                [20.0, 120.0],
            ]
        )

        self.assertEqual(
            context["df"]["FEWSNET_admin_code"].tolist(),
            expected_admin_codes,
        )
        self.assertEqual(
            context["df"]["date"].dt.strftime("%Y-%m-%d").tolist(),
            expected_dates,
        )
        np.testing.assert_allclose(context["X_loc"], expected_x_loc)
        self.assertEqual(context["admin_codes"].tolist(), expected_admin_codes)
        self.assertEqual(
            captured["df"]["FEWSNET_admin_code"].tolist(),
            expected_admin_codes,
        )
        self.assertEqual(
            captured["df"]["date"].dt.strftime("%Y-%m-%d").tolist(),
            expected_dates,
        )
        np.testing.assert_allclose(captured["X_loc"], expected_x_loc)
        np.testing.assert_array_equal(captured["X_group"], np.zeros(4, dtype=int))
        self.assertEqual(captured["forecasting_scope"], 2)

    def test_run_scope_month_passes_string_month_to_splitter(self):
        captured = {}
        context = {
            "df": pd.DataFrame({"FEWSNET_admin_code": [1, 2, 3]}),
            "X": np.arange(6, dtype=float).reshape(3, 2),
            "y": np.array([0, 1, 0]),
            "X_loc": np.zeros((3, 2), dtype=float),
            "years": np.array([2021, 2021, 2021]),
            "dates": pd.to_datetime(["2021-01-01", "2021-02-01", "2021-03-01"]),
            "feature_columns": ["Rainf_zscore", "GDP"],
            "admin_codes": np.array([1, 2, 3]),
        }
        resolved = shap_heatmap.resolve_feature_group_matches(
            context["feature_columns"]
        )

        def fake_load_partition_mapping(_path):
            return pd.DataFrame(
                {"FEWSNET_admin_code": [1, 2, 3], "cluster_id": [1, 1, 1]}
            )

        def fake_create_partition_group_array(df, partition_df):
            return np.array([1, 1, 1]), df.assign(cluster_id=[1, 1, 1])

        def fake_forecasting_scope_to_lag(scope, lags):
            captured["forecasting_scope"] = scope
            captured["lags"] = tuple(lags)
            return 4

        def fake_split_fn(*args, **kwargs):
            captured["split_test_month"] = kwargs["test_month"]
            X_train = np.array([[0.0, 1.0], [2.0, 3.0]])
            y_train = np.array([0, 1])
            X_group_train = np.array([1, 1])
            X_test = np.array([[4.0, 5.0]])
            y_test = np.array([1])
            X_group_test = np.array([1])
            return (
                X_train,
                y_train,
                np.zeros((2, 2)),
                X_group_train,
                X_test,
                y_test,
                np.zeros((1, 2)),
                X_group_test,
                np.array([1, 2]),
                np.array([3]),
            )

        def fake_train_partitioned_model(X_train, y_train, X_group_train, **kwargs):
            return {1: object()}

        def fake_shap_values(**kwargs):
            return (
                np.array([[1.0, 2.0]]),
                {
                    "selected_samples": 1,
                    "evaluated_samples": 1,
                    "fallback_samples": 0,
                    "missing_model_samples": 0,
                    "unmapped_samples": 0,
                },
            )

        rows, diag = shap_heatmap.run_scope_month(
            context=context,
            scope="fs1",
            test_month=pd.Period("2021-02", freq="M"),
            partition_map=Path("partition.csv"),
            train_window_months=36,
            max_samples_per_month=25,
            random_state=5,
            resolved=resolved,
            lags_months=(4, 8, 12),
            min_partition_samples=50,
            load_partition_mapping_fn=fake_load_partition_mapping,
            create_partition_group_array_fn=fake_create_partition_group_array,
            split_fn=fake_split_fn,
            train_partitioned_model_fn=fake_train_partitioned_model,
            shap_values_fn=fake_shap_values,
            forecasting_scope_to_lag_fn=fake_forecasting_scope_to_lag,
        )

        self.assertEqual(captured["split_test_month"], "2021-02")
        self.assertIsInstance(captured["split_test_month"], str)
        self.assertEqual(rows[0]["target_month"], "2021-02")
        self.assertEqual(diag["target_month"], "2021-02")

    def test_run_analysis_manifest_records_feature_resolution_by_scope(self):
        captured = {}
        original_functions = {
            "evaluation_months": shap_heatmap.evaluation_months,
            "validate_partition_maps": shap_heatmap.validate_partition_maps,
            "prepare_scope_context": shap_heatmap.prepare_scope_context,
            "run_scope_month": shap_heatmap.run_scope_month,
            "summarize_group_shares": shap_heatmap.summarize_group_shares,
            "write_heatmap": shap_heatmap.write_heatmap,
            "write_tabular_outputs": shap_heatmap.write_tabular_outputs,
        }
        stage3_module = "scripts.compare_partitioned_vs_pooled_rf_k40_nc4"
        original_stage3 = sys.modules.get(stage3_module)

        def fake_prepare_scope_context(_data_path, forecasting_scope):
            return {
                "feature_columns": [
                    "Rainf_zscore",
                    "crop",
                    "event_count_battles",
                    f"GDP_lag{forecasting_scope}m",
                    "WFP_Price_std",
                    "slope",
                    f"fews_ipc_crisis_lag_{forecasting_scope}",
                    f"unmatched_scope_{forecasting_scope}",
                ]
            }

        def fake_run_scope_month(**kwargs):
            return (
                [
                    {
                        "scope": kwargs["scope"],
                        "horizon_months": (
                            shap_heatmap.SCOPE_TO_HORIZON_MONTHS[kwargs["scope"]]
                        ),
                        "forecasting_horizon": (
                            shap_heatmap.HORIZON_LABELS[kwargs["scope"]]
                        ),
                        "target_month": str(kwargs["test_month"]),
                        "group": "weather",
                        "display_group": "Weather",
                        "group_share": 1.0,
                    }
                ],
                {"scope": kwargs["scope"], "target_month": str(kwargs["test_month"])},
            )

        def fake_write_tabular_outputs(*, monthly, summary, manifest, output_dir):
            captured["manifest"] = manifest
            return {"manifest_json": Path(output_dir) / "manifest.json"}

        try:
            shap_heatmap.evaluation_months = lambda _start, _end: [
                pd.Period(f"2021-{month:02d}", freq="M")
                for month in range(1, 13)
            ]
            shap_heatmap.validate_partition_maps = lambda _maps: None
            shap_heatmap.prepare_scope_context = fake_prepare_scope_context
            shap_heatmap.run_scope_month = fake_run_scope_month
            shap_heatmap.summarize_group_shares = (
                lambda monthly, expected_month_count: monthly
            )
            shap_heatmap.write_heatmap = lambda summary, output_dir, dpi: {
                "png": Path(output_dir) / "plot.png"
            }
            shap_heatmap.write_tabular_outputs = fake_write_tabular_outputs
            sys.modules[stage3_module] = types.SimpleNamespace(
                RF_PARAMS={"n_estimators": 1}
            )

            with tempfile.TemporaryDirectory() as tmp:
                args = types.SimpleNamespace(
                    output_dir=tmp,
                    stage3_root=tmp,
                    start_month="2021-01",
                    end_month="2021-12",
                    data="source.csv",
                    train_window=36,
                    max_shap_samples=25,
                    random_state=5,
                    dpi=72,
                )
                with redirect_stdout(io.StringIO()):
                    shap_heatmap.run_analysis(args)
        finally:
            for name, value in original_functions.items():
                setattr(shap_heatmap, name, value)
            if original_stage3 is None:
                sys.modules.pop(stage3_module, None)
            else:
                sys.modules[stage3_module] = original_stage3

        by_scope = captured["manifest"]["feature_group_resolution_by_scope"]
        self.assertEqual(set(by_scope), {"fs1", "fs2", "fs3"})
        self.assertIn("GDP_lag1m", by_scope["fs1"]["feature_group_matches"]["econ"])
        self.assertIn("GDP_lag2m", by_scope["fs2"]["feature_group_matches"]["econ"])
        self.assertIn("GDP_lag3m", by_scope["fs3"]["feature_group_matches"]["econ"])
        self.assertEqual(by_scope["fs1"]["unmatched_features"], ["unmatched_scope_1"])


if __name__ == "__main__":
    unittest.main()
