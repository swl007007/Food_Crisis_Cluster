"""Focused checks for the Ethiopia binary XGBoost comparison."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from EthiopiaForecastingExperiment.run_binary_xgb_comparison import (
    candidate_order,
    class_weights,
    split_fold,
)


class BinaryXgbComparisonTests(unittest.TestCase):
    def test_weights_are_normalized(self) -> None:
        weights, audit = class_weights([0, 0, 0, 1])
        self.assertTrue(np.isclose(weights.mean(), 1.0))
        self.assertEqual((audit["class_0_n"], audit["class_1_n"]), (3, 1))

    def test_parameter_tie_break(self) -> None:
        rows = [
            {"validation_f1": 0.5, "max_depth": 6, "min_child_weight": 5, "n_estimators": 200},
            {"validation_f1": 0.5, "max_depth": 3, "min_child_weight": 1, "n_estimators": 200},
            {"validation_f1": 0.5, "max_depth": 3, "min_child_weight": 5, "n_estimators": 400},
        ]
        selected = min(rows, key=candidate_order)
        self.assertEqual((selected["max_depth"], selected["min_child_weight"]), (3, 5))

    def test_fold_uses_horizon_boundary_and_latest_six_months(self) -> None:
        months = pd.period_range("2017-01", "2021-02", freq="M").to_timestamp()
        frame = pd.DataFrame(
            {
                "target_month": np.repeat(months, 2),
                "fews_ipc_crisis": np.tile((0, 1), len(months)),
            }
        )
        fit, validation, test, audit = split_fold(frame, target_month="2021-02", horizon=4)
        self.assertEqual(audit["train_end_exclusive"], "2020-10")
        self.assertEqual(audit["validation_months"], ["2020-04", "2020-05", "2020-06", "2020-07", "2020-08", "2020-09"])
        self.assertEqual(len(test), 2)
        self.assertLess(frame.iloc[np.r_[fit, validation]]["target_month"].max(), pd.Timestamp("2020-10-01"))


if __name__ == "__main__":
    unittest.main()
