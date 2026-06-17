import importlib.util
import json
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_georf_thresholded_artifacts.py"
spec = importlib.util.spec_from_file_location("build_georf_thresholded_artifacts", SCRIPT_PATH)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class GeoRFThresholdedArtifactsTests(unittest.TestCase):
    def test_scope_display_labels_use_horizon_wording(self):
        self.assertEqual(
            builder.HORIZONS,
            {
                "fs1": "4-month horizon",
                "fs2": "8-month horizon",
                "fs3": "12-month horizon",
            },
        )

    def test_build_horizon_metrics_uses_monthly_macro_mean(self):
        metrics = pd.DataFrame(
            [
                {
                    "scope": "fs1",
                    "forecasting_horizon": "4-month horizon",
                    "model": "partitioned",
                    "n": 100,
                    "tp": 90,
                    "fp": 10,
                    "fn": 10,
                    "tn": 0,
                    "precision": 0.90,
                    "recall": 0.90,
                    "f1": 0.90,
                },
                {
                    "scope": "fs1",
                    "forecasting_horizon": "4-month horizon",
                    "model": "partitioned",
                    "n": 100,
                    "tp": 1,
                    "fp": 0,
                    "fn": 99,
                    "tn": 0,
                    "precision": 1.00,
                    "recall": 0.01,
                    "f1": 0.019801980198019802,
                },
            ]
        )

        horizon = builder.build_horizon_metrics(metrics)
        row = horizon.iloc[0]

        self.assertEqual(row["support"], 200)
        self.assertEqual(row["tp"], 91)
        self.assertEqual(row["fp"], 10)
        self.assertEqual(row["fn"], 109)
        self.assertAlmostEqual(row["precision"], 0.95)
        self.assertAlmostEqual(row["recall"], 0.455)
        self.assertAlmostEqual(row["f1"], 0.4599009900990099)

    def test_build_compact_table_pivots_three_models(self):
        metrics = pd.DataFrame(
            [
                {
                    "scope": "fs1",
                    "forecasting_horizon": "4-month horizon",
                    "model": "pooled",
                    "precision": 0.7,
                    "recall": 0.5,
                    "f1": 0.58,
                },
                {
                    "scope": "fs1",
                    "forecasting_horizon": "4-month horizon",
                    "model": "partitioned",
                    "precision": 0.75,
                    "recall": 0.6,
                    "f1": 0.67,
                },
                {
                    "scope": "fs1",
                    "forecasting_horizon": "4-month horizon",
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
            [{"scope": "fs1", "forecasting_horizon": "4-month horizon", "partitioned_thresholded_f1": 0.70321}]
        )

        formatted = builder.format_for_markdown(compact)

        self.assertEqual(formatted.loc[0, "partitioned_thresholded_f1"], "0.703")


def test_load_provider_manifests_keeps_partition_and_runtime_details(tmp_path: Path):
    provider = tmp_path / "result_partition_k40_compare_GF_thresholded_fs1"
    provider.mkdir()
    manifest = {
        "data_path": r"C:\data\FEWSNET_forecast_unadjusted_bm.csv",
        "month_ind_enabled": True,
        "partition_map_path": "general.csv",
        "partition_map_m2_path": "m2.csv",
        "partition_map_m6_path": "m6.csv",
        "partition_map_m10_path": "m10.csv",
        "partition_map_hashes": {"general": "a" * 64, "m2": "b" * 64, "m6": "c" * 64, "m10": "d" * 64},
        "smote_available": True,
        "imblearn_version": "0.14.1",
        "python_executable": "python3.12.exe",
    }
    (provider / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    details = builder.load_provider_manifests(tmp_path, ["fs1"])

    assert details["fs1"]["month_ind_enabled"] is True
    assert details["fs1"]["partition_map_m2_path"] == "m2.csv"
    assert details["fs1"]["smote_available"] is True
    assert details["fs1"]["partition_map_hashes"]["m10"] == "d" * 64


if __name__ == "__main__":
    unittest.main()
