import unittest
import sys
import types

import pandas as pd


def _stub_module(name: str, **attrs) -> None:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


_stub_module("src.preprocess.preprocess", load_and_preprocess_data=None)
_stub_module("src.feature.feature", prepare_features=None)
_stub_module("src.customize.customize", train_test_split_rolling_window=None)
_stub_module("src.utils.lag_schedules", forecasting_scope_to_lag=None)
_stub_module("src.metrics.metrics", get_prf=None, get_class_wise_accuracy=None)

from scripts import compare_partitioned_vs_pooled_rf_k40_nc4 as comparison


class PartitionComparisonContractTests(unittest.TestCase):
    def test_small_global_boundary_gap_uses_unmapped_fallback(self):
        total = 1000
        mapped = 985
        df = pd.DataFrame({"FEWSNET_admin_code": [f"A{i:04d}" for i in range(total)]})
        partition_df = pd.DataFrame(
            {
                "FEWSNET_admin_code": [f"A{i:04d}" for i in range(mapped)],
                "cluster_id": [i % 3 for i in range(mapped)],
            }
        )

        x_group, df_with_partition = comparison.create_partition_group_array(df, partition_df)

        self.assertEqual(comparison.PARTITION_UNMAPPED_THRESHOLD_PCT, 2.0)
        self.assertEqual((x_group == -1).sum(), total - mapped)
        self.assertEqual(df_with_partition["cluster_id"].isna().sum(), 0)

    def test_large_partition_gap_still_fails_fast(self):
        total = 1000
        mapped = 970
        df = pd.DataFrame({"FEWSNET_admin_code": [f"A{i:04d}" for i in range(total)]})
        partition_df = pd.DataFrame(
            {
                "FEWSNET_admin_code": [f"A{i:04d}" for i in range(mapped)],
                "cluster_id": [i % 3 for i in range(mapped)],
            }
        )

        with self.assertRaisesRegex(ValueError, "threshold: 2.0%"):
            comparison.create_partition_group_array(df, partition_df)


if __name__ == "__main__":
    unittest.main()
