import importlib
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _load_selector():
    spec = importlib.util.find_spec("src.feature.strict_lag")
    if spec is None:
        raise AssertionError("strict lag feature selector is not implemented")
    return importlib.import_module("src.feature.strict_lag").select_strict_lag_features


class StrictLagFeatureTests(unittest.TestCase):
    def test_keeps_static_and_active_lag_columns_in_original_order(self):
        select_strict_lag_features = _load_selector()

        X = np.arange(18).reshape(3, 6)
        columns = [
            "FEWSNET_admin_code",
            "rain",
            "latitude",
            "rain_lag4m",
            "price_lag4m",
            "price",
        ]

        result = select_strict_lag_features(
            X,
            columns,
            l1_index=[0, 2],
            l2_index=[1, 3, 4, 5],
            lag_months=4,
        )

        np.testing.assert_array_equal(result[0], X[:, [0, 2, 3, 4]])
        self.assertEqual(result[1], [0, 1])
        self.assertEqual(result[2], [2, 3])
        self.assertEqual(
            result[3],
            ["FEWSNET_admin_code", "latitude", "rain_lag4m", "price_lag4m"],
        )

    def test_rejects_a_scope_without_active_lag_columns(self):
        select_strict_lag_features = _load_selector()

        with self.assertRaisesRegex(ValueError, "No active lag columns"):
            select_strict_lag_features(
                np.ones((2, 2)),
                ["latitude", "rain"],
                l1_index=[0],
                l2_index=[1],
                lag_months=8,
            )


class Stage1ExperimentCliTests(unittest.TestCase):
    def test_georf_adapter_import_does_not_require_archived_demo_package(self):
        repo_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                "from src.model.adapters import GFAdapter; GFAdapter()",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dry_run_accepts_default_off_experiment_inputs(self):
        script = Path(__file__).resolve().parents[2] / "app" / "main_model_GF.py"
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(script),
                    "--dry-run",
                    "--data",
                    str(Path(tmp) / "ethiopia_panel.csv"),
                    "--strict-lag-only",
                    "--random-seed",
                    "5",
                ],
                cwd=tmp,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Strict lag-only features: True", result.stdout)
            self.assertIn("Random seed: 5", result.stdout)
            self.assertIn("ethiopia_panel.csv", result.stdout)


if __name__ == "__main__":
    unittest.main()
