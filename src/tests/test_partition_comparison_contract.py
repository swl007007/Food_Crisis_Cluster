import unittest
import sys
import types

import numpy as np
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


class FakeProbabilityModel:
    def __init__(self, classes, probabilities, labels=None):
        self.classes_ = np.array(classes)
        self._probabilities = np.array(probabilities, dtype=float)
        self._labels = np.array(labels if labels is not None else [int(p >= 0.5) for p in self._probabilities])

    def predict_proba(self, X):
        n = len(X)
        if len(self.classes_) == 1:
            return np.ones((n, 1), dtype=float)
        probabilities = np.resize(self._probabilities, n)
        return np.column_stack([1.0 - probabilities, probabilities])

    def predict(self, X):
        return np.resize(self._labels, len(X))


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

    def test_predict_class1_probability_extracts_positive_class_column(self):
        model = FakeProbabilityModel(classes=[0, 1], probabilities=[0.2, 0.8, 0.6])
        X = np.zeros((3, 2))

        probabilities = comparison.predict_class1_probability(model, X)

        np.testing.assert_allclose(probabilities, np.array([0.2, 0.8, 0.6]))

    def test_predict_class1_probability_handles_single_class_models(self):
        positive_model = FakeProbabilityModel(classes=[1], probabilities=[1.0])
        negative_model = FakeProbabilityModel(classes=[0], probabilities=[1.0])
        X = np.zeros((2, 2))

        positive = comparison.predict_class1_probability(positive_model, X)
        negative = comparison.predict_class1_probability(negative_model, X)

        np.testing.assert_allclose(positive, np.ones(2))
        np.testing.assert_allclose(negative, np.zeros(2))

    def test_predict_partitioned_probability_uses_partition_and_pooled_fallback(self):
        pooled = FakeProbabilityModel(classes=[0, 1], probabilities=[0.1])
        models = {
            2: FakeProbabilityModel(classes=[0, 1], probabilities=[0.7]),
            3: None,
        }
        X = np.zeros((4, 2))
        X_group = np.array([2, 3, -1, 99])

        probabilities = comparison.predict_partitioned_probability(models, pooled, X, X_group)

        np.testing.assert_allclose(probabilities, np.array([0.7, 0.1, 0.1, 0.1]))


if __name__ == "__main__":
    unittest.main()
