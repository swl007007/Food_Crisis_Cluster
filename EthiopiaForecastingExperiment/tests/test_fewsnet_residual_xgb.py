"""Focused checks for the Ethiopia FEWS NET residual XGBoost experiment."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "run_fewsnet_residual_xgb.py"


def _load_module():
    if not SCRIPT.is_file():
        raise AssertionError("FEWS NET residual XGBoost runner is not implemented")
    spec = importlib.util.spec_from_file_location("eth_fewsnet_residual_xgb", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExpertCalendarTests(unittest.TestCase):
    def test_expert_anchor_uses_explicit_scope_calendar_mapping(self) -> None:
        module = _load_module()
        fewsnet = pd.DataFrame(
            {
                "country": ["Ethiopia"] * 5,
                "admin_code": [1] * 5,
                "year_month": ["2020_06", "2020_10", "2021_02", "2021_05", "2021_06"],
                "fews_ipc": [1, 1, 1, 1, 3],
                "fews_proj_near": [2, 1, 3, 1, 1],
                "fews_proj_med": [5, 4, 1, 1, 1],
            }
        )
        snapshot = pd.DataFrame(
            {"FEWSNET_admin_code": ["1"], "target_month": ["2021-06-01"]}
        )

        expected = {
            0: ("2021-02", 3.0),
            1: ("2021-02", 3.0),
            2: ("2020-10", 4.0),
            3: ("2020-06", 5.0),
        }
        for scope, (source_month, phase) in expected.items():
            with self.subTest(scope=scope):
                joined = module.attach_expert_anchor(snapshot, fewsnet, scope)
                self.assertIn("fewsnet_target_phase", joined.columns)
                self.assertIn("fewsnet_truth", joined.columns)
                self.assertEqual(joined.loc[0, "expert_source_month"], source_month)
                self.assertEqual(joined.loc[0, "expert_projection_phase"], phase)
                self.assertEqual(joined.loc[0, "expert_anchor"], 1.0)
                self.assertEqual(joined.loc[0, "fewsnet_target_phase"], 3.0)
                self.assertEqual(joined.loc[0, "fewsnet_truth"], 1.0)


class TemporalOofTests(unittest.TestCase):
    def test_oof_fold_stops_before_pseudo_origin_and_keeps_month_grouped(self) -> None:
        module = _load_module()
        self.assertTrue(
            hasattr(module, "select_temporal_oof_fold"),
            "temporal OOF fold selector is not implemented",
        )
        months = pd.period_range("2017-01", "2021-02", freq="M").to_timestamp()
        frame = pd.DataFrame(
            {
                "target_month": np.repeat(months, 2),
                "fews_ipc_crisis": np.tile((0, 1), len(months)),
                "expert_anchor": 0.0,
            }
        )

        train, test, audit = module.select_temporal_oof_fold(
            frame,
            target_month="2021-02",
            horizon=4,
        )

        self.assertEqual(audit["train_start"], "2017-10")
        self.assertEqual(audit["train_end_exclusive"], "2020-10")
        self.assertEqual(len(train), 72)
        self.assertEqual(len(test), 2)
        self.assertLess(
            frame.iloc[train]["target_month"].max(),
            pd.Timestamp("2020-10-01"),
        )
        self.assertEqual(
            frame.iloc[test]["target_month"].dt.to_period("M").unique().tolist(),
            [pd.Period("2021-02", freq="M")],
        )

    def test_second_layer_target_uses_oof_first_layer_prediction(self) -> None:
        module = _load_module()
        self.assertTrue(
            hasattr(module, "second_residual_target"),
            "second residual target builder is not implemented",
        )
        target = module.second_residual_target(
            y_true=np.array([0.0, 1.0]),
            expert=np.array([0.0, 0.0]),
            first_layer_oof=np.array([0.25, 0.75]),
        )

        np.testing.assert_allclose(target, np.array([-0.25, 0.25]))
        self.assertFalse(np.array_equal(target, np.array([0.0, 0.0])))

    def test_temporal_oof_predictions_exist_only_for_held_out_month(self) -> None:
        module = _load_module()
        self.assertTrue(
            hasattr(module, "temporal_oof_predictions"),
            "temporal OOF prediction routine is not implemented",
        )
        months = pd.period_range("2017-01", "2021-02", freq="M").to_timestamp()
        frame = pd.DataFrame(
            {
                "target_month": np.repeat(months, 2),
                "fews_ipc_crisis": np.tile((0, 1), len(months)),
                "expert_anchor": 0.0,
            }
        )
        features = np.c_[np.tile((0.0, 1.0), len(months)), np.zeros(len(frame))]
        desired = np.flatnonzero(
            frame["target_month"].eq(pd.Timestamp("2021-02-01")).to_numpy()
        )

        predictions, audits = module.temporal_oof_predictions(
            frame,
            features,
            desired,
            horizon=4,
            model_parameters={"max_depth": 1, "min_child_weight": 1, "n_estimators": 2},
        )

        self.assertTrue(np.isfinite(predictions[desired]).all())
        self.assertTrue(np.isnan(np.delete(predictions, desired)).all())
        self.assertEqual(audits[0]["train_end_exclusive"], "2020-10")

    def test_outer_fold_excludes_degenerate_month_before_validation_split(self) -> None:
        module = _load_module()
        self.assertTrue(
            hasattr(module, "split_residual_fold"),
            "residual outer fold splitter is not implemented",
        )
        months = pd.period_range("2019-10", "2022-10", freq="4M").to_timestamp()
        frame = pd.DataFrame(
            {
                "target_month": np.repeat(months, 2),
                "fews_ipc_crisis": np.tile((0, 1), len(months)),
                "expert_anchor": 0.0,
            }
        )

        fit, validation, test, audit = module.split_residual_fold(
            frame,
            target_month="2022-10",
            horizon=1,
        )

        selected_months = frame.iloc[np.r_[fit, validation]]["target_month"].dt.to_period("M")
        self.assertNotIn(pd.Period("2021-06", freq="M"), set(selected_months))
        self.assertEqual(
            audit["validation_months"],
            ["2020-06", "2020-10", "2021-02", "2021-10", "2022-02", "2022-06"],
        )
        self.assertEqual(len(test), 2)


class ScoreThresholdTests(unittest.TestCase):
    def test_threshold_search_retains_scores_outside_zero_one(self) -> None:
        module = _load_module()
        self.assertTrue(
            hasattr(module, "select_score_threshold"),
            "residual score threshold selector is not implemented",
        )

        selected = module.select_score_threshold(
            np.array([0, 1]),
            np.array([-0.20, 1.20]),
        )

        self.assertEqual(selected["selected_threshold"], 1.20)
        self.assertEqual(selected["validation_f1"], 1.0)

    def test_candidate_tie_break_prefers_simpler_residual_model(self) -> None:
        module = _load_module()
        self.assertTrue(
            hasattr(module, "candidate_order"),
            "residual candidate ordering is not implemented",
        )
        candidates = [
            {"validation_f1": 0.7, "layer_count": 2, "max_depth": 3, "n_estimators": 200, "min_child_weight": 5},
            {"validation_f1": 0.7, "layer_count": 1, "max_depth": 6, "n_estimators": 200, "min_child_weight": 5},
            {"validation_f1": 0.7, "layer_count": 1, "max_depth": 3, "n_estimators": 400, "min_child_weight": 5},
            {"validation_f1": 0.7, "layer_count": 1, "max_depth": 3, "n_estimators": 200, "min_child_weight": 1},
        ]

        selected = min(candidates, key=module.candidate_order)

        self.assertEqual(selected["layer_count"], 1)
        self.assertEqual(selected["max_depth"], 3)
        self.assertEqual(selected["n_estimators"], 200)


class CoverageTests(unittest.TestCase):
    def test_low_target_or_expert_coverage_sets_single_suppression_flag(self) -> None:
        module = _load_module()
        self.assertTrue(
            hasattr(module, "fold_coverage"),
            "fold coverage gate is not implemented",
        )
        low_expert = pd.DataFrame(
            {
                "fews_ipc_crisis": [0, 1] * 5,
                "expert_anchor": [0.0] * 8 + [np.nan, np.nan],
            }
        )
        low_target = pd.DataFrame(
            {
                "fews_ipc_crisis": [0, 1] * 4 + [np.nan, np.nan],
                "expert_anchor": [0.0] * 10,
            }
        )

        expert_audit = module.fold_coverage(low_expert, cohort_n=10)
        target_audit = module.fold_coverage(low_target, cohort_n=10)

        self.assertEqual(expert_audit["expert_coverage"], 0.8)
        self.assertEqual(target_audit["target_coverage"], 0.8)
        self.assertTrue(expert_audit["suppressed"])
        self.assertTrue(target_audit["suppressed"])
        self.assertNotIn("suppression_reason", expert_audit)


class PlotTests(unittest.TestCase):
    def test_summary_plot_writes_four_metric_panels(self) -> None:
        module = _load_module()
        self.assertTrue(
            hasattr(module, "plot_summary"),
            "scope comparison plotter is not implemented",
        )
        rows = []
        for scope in range(4):
            for model in ("fewsnet", "binary_xgboost", "georf_v5", "residual_xgboost"):
                rows.append(
                    {
                        "scope": f"fs{scope}",
                        "model": model,
                        "precision": 0.50 + scope / 100,
                        "recall": 0.60,
                        "f1": 0.55,
                        "balanced_accuracy": 0.58,
                    }
                )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "comparison.png"
            panel_count = module.plot_summary(pd.DataFrame(rows), output)

            self.assertEqual(panel_count, 4)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


class CliTests(unittest.TestCase):
    def test_help_exposes_only_required_experiment_paths(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--v5-run", result.stdout)
        self.assertIn("--binary-run", result.stdout)
        self.assertIn("--fewsnet", result.stdout)
        self.assertIn("--output-root", result.stdout)
        self.assertIn("--run-id", result.stdout)


if __name__ == "__main__":
    unittest.main()
