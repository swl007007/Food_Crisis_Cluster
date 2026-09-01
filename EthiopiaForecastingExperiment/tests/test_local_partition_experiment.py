import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "run_local_partition_experiment.py"


def _load_module():
    if not SCRIPT.is_file():
        raise AssertionError("local Ethiopia experiment runner is not implemented")
    spec = importlib.util.spec_from_file_location("eth_local_experiment", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CohortTests(unittest.TestCase):
    def test_exact_eth_filter_sorts_and_preserves_unique_admin_month_keys(self):
        module = _load_module()
        panel = pd.DataFrame(
            {
                "ISO3": ["ETH", "KEN", "ETH"],
                "FEWSNET_admin_code": [2, 9, 1],
                "date": ["2020-02-01", "2020-01-01", "2020-01-01"],
                "value": [20, 90, 10],
            }
        )

        eth = module.filter_ethiopia_panel(panel)

        self.assertEqual(eth["FEWSNET_admin_code"].tolist(), [1, 2])
        self.assertEqual(eth["value"].tolist(), [10, 20])
        self.assertEqual(eth["ISO3"].unique().tolist(), ["ETH"])

    def test_duplicate_eth_admin_month_is_rejected(self):
        module = _load_module()
        panel = pd.DataFrame(
            {
                "ISO3": ["ETH", "ETH"],
                "FEWSNET_admin_code": [1, 1],
                "date": ["2020-01-01", "2020-01-20"],
            }
        )

        with self.assertRaisesRegex(ValueError, "Duplicate Ethiopia admin-month"):
            module.filter_ethiopia_panel(panel)


class FewsnetCalendarTests(unittest.TestCase):
    def test_fs1_joins_near_projection_from_exact_t_minus_four_month(self):
        module = _load_module()
        fewsnet = pd.DataFrame(
            {
                "country": ["Ethiopia"] * 6,
                "admin_code": [1, 2, 1, 2, 1, 2],
                "year_month": [
                    "2021_01",
                    "2021_01",
                    "2021_04",
                    "2021_04",
                    "2021_05",
                    "2021_05",
                ],
                "fews_ipc": [1, 1, 1, 1, 3, 2],
                "fews_proj_near": [4, np.nan, 1, 4, 1, 1],
                "fews_proj_med": [1, 1, 1, 1, 1, 1],
            }
        )

        joined, coverage = module.calendar_join_fewsnet(
            fewsnet,
            target_month="2021-05",
            scope=1,
            cohort_codes=[1, 2],
        )

        self.assertEqual(joined.loc[joined["FEWSNET_admin_code"].eq("1"), "y_pred_fewsnet"].iloc[0], 1)
        self.assertTrue(pd.isna(joined.loc[joined["FEWSNET_admin_code"].eq("2"), "y_pred_fewsnet"].iloc[0]))
        self.assertEqual(coverage, {"available": 1, "total": 2, "fraction": 0.5})

    def test_low_coverage_suppresses_fewsnet_but_uses_common_model_support(self):
        module = _load_module()
        predictions = pd.DataFrame(
            {
                "FEWSNET_admin_code": [1, 2],
                "month_start": ["2021-05-01", "2021-05-01"],
                "y_true": [1, 0],
                "y_pred_pooled": [0, 0],
                "y_pred_partitioned": [1, 0],
                "y_pred_pooled_thresholded": [1, 0],
                "y_pred_partitioned_thresholded": [1, 0],
            }
        )
        fewsnet = pd.DataFrame(
            {
                "country": ["Ethiopia"] * 4,
                "admin_code": [1, 2, 1, 2],
                "year_month": ["2021-01", "2021-01", "2021-05", "2021-05"],
                "fews_ipc": [1, 1, 3, 2],
                "fews_proj_near": [4, np.nan, 1, 1],
                "fews_proj_med": [1, 1, 1, 1],
            }
        )

        final_rows, fixed_rows = module.evaluate_scope_month(
            predictions,
            scope=1,
            lag_months=4,
            fewsnet_eth=fewsnet,
            cohort_codes=[1, 2],
            coverage_threshold=0.9,
        )

        final = pd.DataFrame(final_rows).set_index("model")
        fixed = pd.DataFrame(fixed_rows).set_index("model")
        self.assertEqual(final.loc["pooled", "n"], 1)
        self.assertEqual(final.loc["partitioned", "n"], 1)
        self.assertEqual(final.loc["fewsnet", "status"], "suppressed_low_coverage")
        self.assertTrue(pd.isna(final.loc["fewsnet", "f1"]))
        self.assertEqual(fixed.loc["pooled", "n"], 1)


class OrchestrationTests(unittest.TestCase):
    def test_stage1_plan_has_36_labeled_month_cells(self):
        module = _load_module()

        cells = [
            (scope, year, month)
            for scope in module.SCOPES
            for year in range(2018, 2021)
            for month in module.STAGE1_MONTHS
        ]

        self.assertEqual(module.STAGE1_MONTHS, (2, 6, 10))
        self.assertEqual(len(cells), 36)

    def test_stage1_command_is_one_scope_one_month_with_strict_lag(self):
        module = _load_module()
        self.assertTrue(hasattr(module, "build_stage1_command"), "Stage 1 command builder is not implemented")
        command = module.build_stage1_command(
            python_executable=Path("python.exe"),
            repo_root=Path("repo"),
            panel_path=Path("eth.csv"),
            scope=0,
            year=2018,
            month=2,
        )

        self.assertEqual(command[0], "python.exe")
        self.assertIn("repo/app/main_model_GF.py", command[2].replace("\\", "/"))
        self.assertEqual(command[command.index("--forecasting_scope") + 1], "0")
        self.assertEqual(command[command.index("--desired_terms") + 1], "2018-02")
        self.assertEqual(command[command.index("--random-seed") + 1], "5")
        self.assertIn("--strict-lag-only", command)

    def test_stage2_commands_build_general_and_three_month_specific_mappings(self):
        module = _load_module()
        self.assertTrue(hasattr(module, "build_stage2_commands"), "Stage 2 command builder is not implemented")
        commands = module.build_stage2_commands(
            python_executable=Path("python.exe"),
            repo_root=Path("repo"),
            stage2_dir=Path("stage2"),
        )

        self.assertEqual(len(commands), 14)
        step4 = [command for command in commands if command[2].endswith("step4_similarity_matrix.py")]
        self.assertEqual(len(step4), 4)
        self.assertEqual(sum("--month" not in command for command in step4), 1)
        self.assertEqual(
            sorted(command[command.index("--month") + 1] for command in step4 if "--month" in command),
            ["10", "2", "6"],
        )

    def test_run_directory_must_be_new(self):
        module = _load_module()
        self.assertTrue(hasattr(module, "create_run_directory"), "Run-directory guard is not implemented")
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            first = module.create_run_directory(output_root, "run1")
            self.assertTrue(first.is_dir())
            with self.assertRaises(FileExistsError):
                module.create_run_directory(output_root, "run1")

    def test_run_directory_rejects_repo_paths_outside_experiment_root(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            module.__file__ = str(
                repo_root / "EthiopiaForecastingExperiment" / "run_local_partition_experiment.py"
            )
            forbidden_root = repo_root / "result_GeoRF_protected_test"

            with self.assertRaisesRegex(ValueError, "approved experiment output root"):
                module.create_run_directory(forbidden_root, "run1")

    def test_collect_stage1_outputs_copies_only_stage2_contract_artifacts(self):
        module = _load_module()
        self.assertTrue(hasattr(module, "collect_stage1_outputs"), "Stage 1 collector is not implemented")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cell = root / "cell"
            model = cell / "result_GeoRF_3"
            stage2_results = root / "stage2" / "GeoRFResults"
            model.mkdir(parents=True)
            pd.DataFrame(
                {"year": [2018], "month": [2], "f1(1)": [0.6], "f1_base(1)": [0.5]}
            ).to_csv(cell / "results_df_gp_fs0_2018_2018.csv", index=False)
            pd.DataFrame(
                {"FEWSNET_admin_code": [1, 2], "partition_id": ["0", "1"]}
            ).to_csv(model / "correspondence_table_2018-02.csv", index=False)

            metrics = module.collect_stage1_outputs(
                cell_dir=cell,
                stage2_results_dir=stage2_results,
                scope=0,
                year=2018,
                month=2,
            )

            self.assertEqual(metrics.loc[0, "month"], 2)
            copied = (
                stage2_results
                / "result_GeoRF_2018_fs0_2018-02_visual"
                / "correspondence_table_2018-02.csv"
            )
            self.assertTrue(copied.is_file())
            self.assertFalse((stage2_results / "result_GeoRF_3").exists())

    def test_stage3_command_runs_one_fold_with_symmetric_thresholds(self):
        module = _load_module()
        self.assertTrue(hasattr(module, "build_stage3_command"), "Stage 3 command builder is not implemented")
        command = module.build_stage3_command(
            python_executable=Path("python.exe"),
            repo_root=Path("repo"),
            panel_path=Path("eth.csv"),
            partition_map=Path("m2.csv"),
            out_dir=Path("fold"),
            scope=3,
            target_month="2024-02",
        )

        self.assertEqual(command[command.index("--start-month") + 1], "2024-02")
        self.assertEqual(command[command.index("--end-month") + 1], "2024-02")
        self.assertEqual(command[command.index("--forecasting-scope") + 1], "3")
        self.assertIn("--strict-lag-only", command)
        self.assertIn("--enable-symmetric-validation-threshold", command)

    def test_plot_has_four_scope_rows_and_three_metric_columns(self):
        module = _load_module()
        self.assertTrue(hasattr(module, "plot_monthly_metrics"), "4x3 plotter is not implemented")
        rows = []
        for scope in range(4):
            for model in ("pooled", "partitioned", "fewsnet"):
                rows.append(
                    {
                        "scope": f"fs{scope}",
                        "test_month": "2021-02",
                        "model": model,
                        "precision": 0.5,
                        "recall": 0.6,
                        "f1": 0.55,
                        "status": "unavailable_for_scope" if model == "fewsnet" and scope in (0, 3) else "available",
                    }
                )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "figure.png"
            panel_count = module.plot_monthly_metrics(pd.DataFrame(rows), output)
            self.assertEqual(panel_count, 12)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
