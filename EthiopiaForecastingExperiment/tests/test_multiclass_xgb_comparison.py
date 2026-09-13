"""Focused checks for the Ethiopia multiclass XGBoost experiment."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from EthiopiaForecastingExperiment.run_multiclass_xgb_comparison import (
    candidate_order,
    class_weights,
    split_fold,
)


class MulticlassXgbComparisonTests(unittest.TestCase):
    def test_missing_class_is_audited_without_synthesis(self) -> None:
        weights, audit = class_weights([0, 0, 1, 2])
        self.assertTrue(np.isclose(weights.mean(), 1.0))
        self.assertEqual(audit[4]["n"], 0)
        self.assertIsNone(audit[4]["normalized_weight"])

    def test_candidate_tie_break_is_frozen(self) -> None:
        candidates = [
            {"validation_macro_f1": 0.5, "max_depth": 6, "min_child_weight": 1, "n_estimators": 200},
            {"validation_macro_f1": 0.5, "max_depth": 3, "min_child_weight": 5, "n_estimators": 400},
            {"validation_macro_f1": 0.5, "max_depth": 3, "min_child_weight": 5, "n_estimators": 200},
        ]
        selected = min(candidates, key=candidate_order)
        self.assertEqual((selected["max_depth"], selected["min_child_weight"], selected["n_estimators"]), (3, 5, 200))

    def test_fold_ends_at_forecast_origin_and_uses_latest_six_months(self) -> None:
        months = pd.period_range("2017-01", "2021-02", freq="M").to_timestamp()
        frame = pd.DataFrame(
            {
                "target_month": np.repeat(months, 4),
                "fews_ipc": np.tile((1, 2, 3, 4), len(months)),
            }
        )
        fit, validation, test, audit = split_fold(frame, target_month="2021-02", horizon=4)
        self.assertEqual(audit["train_end_exclusive"], "2020-10")
        self.assertEqual(audit["validation_months"], ["2020-04", "2020-05", "2020-06", "2020-07", "2020-08", "2020-09"])
        self.assertEqual(len(test), 4)
        self.assertLess(frame.iloc[np.r_[fit, validation]]["target_month"].max(), pd.Timestamp("2020-10-01"))


if __name__ == "__main__":
    unittest.main()
