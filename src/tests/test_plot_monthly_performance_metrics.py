import importlib.util
import math
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "plot_monthly_performance_metrics.py"
spec = importlib.util.spec_from_file_location("plot_monthly_performance_metrics", SCRIPT_PATH)
plot_monthly = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_monthly)


class MonthlyPerformanceMetricsTests(unittest.TestCase):
    def _model_df(self):
        rows = []
        for scope in plot_monthly.SCOPES:
            for model in ("partitioned", "pooled"):
                rows.append(
                    {
                        "model_key": "georf",
                        "model_family": "GeoRF",
                        "scope": scope,
                        "test_month": "2024-02",
                        "model": model,
                        "precision": 0.7,
                        "recall": 0.6,
                        "f1": 0.65,
                    }
                )
        return pd.DataFrame(rows)

    def _baselines(self):
        return {
            "fs1": pd.DataFrame(
                [{"aligned_test_month": "2024-02", "precision": 0.1, "recall": 0.2, "f1": 0.3}]
            ),
            "fs2": pd.DataFrame(
                [{"aligned_test_month": "2024-02", "precision": 0.4, "recall": 0.5, "f1": 0.6}]
            ),
        }

    def test_no_extend_fewsnet_leaves_fs3_baseline_blank(self):
        payload, missing = plot_monthly.build_plot_payload(
            self._model_df(),
            self._baselines(),
            ["georf"],
            extend_fewsnet=False,
        )

        fs3_f1 = payload["georf"]["fs3"]["metrics"]["f1"]["fewsnet"]
        self.assertEqual(len(fs3_f1), 1)
        self.assertTrue(math.isnan(fs3_f1[0]))
        self.assertTrue(
            any(
                point["source_kind"] == "fewsnet"
                and point["scope"] == "fs3"
                and point["reason"] == "FEWSNET fs3 extension disabled"
                for point in missing
            )
        )

    def test_manifest_records_model_selection(self):
        payload, missing = plot_monthly.build_plot_payload(
            self._model_df(),
            self._baselines(),
            ["georf"],
            extend_fewsnet=False,
        )
        manifest = plot_monthly.make_manifest(
            Path("."),
            Path("fewsnet_baseline_results"),
            Path("final_artifacts_in_paper_updated"),
            ["georf"],
            ["result_partition_k40_compare_GF_fs1/metrics_monthly.csv"],
            ["fewsnet_baseline_results/fewsnet_baseline_results_fs1.csv"],
            missing,
            payload,
            "dry-run",
            extend_fewsnet=False,
        )

        self.assertEqual(manifest["model_selection"], ["georf"])
        self.assertIsNone(manifest["validation_summary"]["fewsnet_fs3_label"])
        self.assertEqual(manifest["fewsnet_fs3_assumption"], "FEWSNET fs3 is not plotted.")


if __name__ == "__main__":
    unittest.main()
