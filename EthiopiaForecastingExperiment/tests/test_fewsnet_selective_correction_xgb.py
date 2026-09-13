"""Focused checks for the conservative FEWS NET correction gate."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "run_fewsnet_selective_correction_xgb.py"
)


def _load_module():
    if not SCRIPT.is_file():
        raise AssertionError("Selective-correction runner is not implemented")
    spec = importlib.util.spec_from_file_location("eth_fewsnet_selective", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DirectionGateTests(unittest.TestCase):
    def test_directions_are_gated_independently(self) -> None:
        module = _load_module()
        expert = np.r_[np.zeros(20, dtype=int), np.ones(20, dtype=int)]
        truth = np.r_[np.ones(20, dtype=int), np.ones(15, dtype=int), np.zeros(5, dtype=int)]
        result = module.evaluate_correction_rule(
            truth,
            expert,
            np.full(40, 0.9),
            np.tile(["2020-01", "2020-02"], 20),
            threshold=0.8,
        )

        self.assertTrue(result["enable_0_to_1"])
        self.assertFalse(result["enable_1_to_0"])
        self.assertEqual(result["applied_flips_0_to_1"], 20)
        self.assertEqual(result["applied_flips_1_to_0"], 0)
        np.testing.assert_array_equal(result["prediction"], np.ones(40, dtype=int))

    def test_each_direction_requires_precision_count_and_month_support(self) -> None:
        module = _load_module()

        def evaluate(correct: int, count: int, month_count: int):
            expert = np.zeros(count, dtype=int)
            truth = np.r_[np.ones(correct, dtype=int), np.zeros(count - correct, dtype=int)]
            months = np.resize(np.array(["2020-01", "2020-02"]), count)
            if month_count == 1:
                months[:] = "2020-01"
            return module.evaluate_correction_rule(
                truth,
                expert,
                np.ones(count),
                months,
                threshold=0.99,
            )

        self.assertTrue(evaluate(15, 20, 2)["enable_0_to_1"])
        self.assertFalse(evaluate(14, 20, 2)["enable_0_to_1"])
        self.assertFalse(evaluate(15, 19, 2)["enable_0_to_1"])
        self.assertFalse(evaluate(15, 20, 1)["enable_0_to_1"])

    def test_score_must_strictly_exceed_threshold(self) -> None:
        module = _load_module()
        prediction, proposed, applied = module.apply_correction_rule(
            np.array([0, 0]),
            np.array([0.8, 0.81]),
            threshold=0.8,
            enable_0_to_1=True,
            enable_1_to_0=False,
        )

        np.testing.assert_array_equal(proposed, np.array([False, True]))
        np.testing.assert_array_equal(applied, np.array([False, True]))
        np.testing.assert_array_equal(prediction, np.array([0, 1]))


class CandidateSelectionTests(unittest.TestCase):
    def test_expert_only_wins_exact_f1_tie(self) -> None:
        module = _load_module()
        candidates = [
            {"validation_f1": 0.8, "expert_only": False, "grid_order": 0, "threshold_order": 0},
            {"validation_f1": 0.8, "expert_only": True, "grid_order": -1, "threshold_order": -1},
        ]

        selected = min(candidates, key=module.candidate_order)

        self.assertTrue(selected["expert_only"])

    def test_nonexpert_ties_follow_grid_then_threshold_order(self) -> None:
        module = _load_module()
        candidates = [
            {"validation_f1": 0.81, "expert_only": False, "grid_order": 1, "threshold_order": 0},
            {"validation_f1": 0.81, "expert_only": False, "grid_order": 0, "threshold_order": 1},
            {"validation_f1": 0.81, "expert_only": False, "grid_order": 0, "threshold_order": 0},
        ]

        selected = min(candidates, key=module.candidate_order)

        self.assertEqual((selected["grid_order"], selected["threshold_order"]), (0, 0))

    def test_lower_f1_correction_falls_back_to_expert_only(self) -> None:
        module = _load_module()
        candidates = [
            {"validation_f1": 0.79, "expert_only": False, "grid_order": 0, "threshold_order": 0},
            {"validation_f1": 0.80, "expert_only": True, "grid_order": -1, "threshold_order": -1},
        ]

        selected = min(candidates, key=module.candidate_order)

        self.assertTrue(selected["expert_only"])


if __name__ == "__main__":
    unittest.main()
