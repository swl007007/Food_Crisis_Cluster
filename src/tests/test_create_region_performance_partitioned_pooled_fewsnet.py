import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "create_region_performance_partitioned_pooled_fewsnet.py"


def load_script_module():
    if "geopandas" not in sys.modules:
        sys.modules["geopandas"] = types.SimpleNamespace(read_file=lambda *_args, **_kwargs: None)
    spec = importlib.util.spec_from_file_location("create_region_performance_partitioned_pooled_fewsnet", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RegionPerformanceFewsnetTests(unittest.TestCase):
    def test_no_extend_fewsnet_leaves_fs3_expert_metrics_blank(self):
        module = load_script_module()
        helpers = types.SimpleNamespace(
            _confusion=lambda y_true, y_pred: {
                "tp": int(((np.asarray(y_true) == 1) & (np.asarray(y_pred) == 1)).sum()),
                "fp": int(((np.asarray(y_true) == 0) & (np.asarray(y_pred) == 1)).sum()),
                "fn": int(((np.asarray(y_true) == 1) & (np.asarray(y_pred) == 0)).sum()),
                "tn": int(((np.asarray(y_true) == 0) & (np.asarray(y_pred) == 0)).sum()),
                "support": len(y_true),
            },
            _prf=lambda tp, fp, fn, tn: (
                tp / (tp + fp) if (tp + fp) else np.nan,
                tp / (tp + fn) if (tp + fn) else np.nan,
                2 * (tp / (tp + fp)) * (tp / (tp + fn)) / ((tp / (tp + fp)) + (tp / (tp + fn)))
                if (tp + fp) and (tp + fn) and (tp / (tp + fp) + tp / (tp + fn))
                else np.nan,
                (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else np.nan,
            ),
        )
        model_df = pd.DataFrame(
            [
                {
                    "FEWSNET_admin_code": "1",
                    "date": pd.Timestamp("2024-02-01"),
                    "scope": "fs3",
                    "y_true": 1,
                    "y_pred_partitioned": 1,
                    "y_pred_pooled": 0,
                }
            ]
        )
        fewsnet_df = pd.DataFrame(
            columns=["FEWSNET_admin_code", "date", "y_pred_fewsnet", "scope"]
        )
        region_lookup = pd.DataFrame(
            [{"FEWSNET_admin_code": "1", "ADMIN0": "Country", "region": "Region"}]
        )

        table = module.build_table(model_df, fewsnet_df, region_lookup, ["fs3"], helpers)

        self.assertTrue(pd.isna(table.loc[0, "fewsnet_valid_support"]))
        self.assertTrue(pd.isna(table.loc[0, "fewsnet_expert_f1"]))
        self.assertTrue(pd.isna(table.loc[0, "delta_partitioned_minus_fewsnet_expert_f1"]))

    def test_load_fewsnet_expert_predictions_can_disable_fs3_extension(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "FEWSNET.csv"
            pd.DataFrame(
                [
                    {
                        "country": "Country",
                        "admin_code": 1,
                        "year": 2023,
                        "month": month,
                        "fews_proj_near": 4,
                        "fews_proj_med": 4,
                    }
                    for month in range(1, 13)
                ]
            ).to_csv(csv_path, index=False)

            df = module.load_fewsnet_expert_predictions(
                csv_path,
                ["fs1", "fs2", "fs3"],
                extend_fewsnet=False,
            )

            self.assertEqual(sorted(df["scope"].unique().tolist()), ["fs1", "fs2"])


if __name__ == "__main__":
    unittest.main()
