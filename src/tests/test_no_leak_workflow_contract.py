from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


def read_text(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8", errors="ignore")


class NoLeakWorkflowContractTests(unittest.TestCase):
    def test_active_stage1_entrypoint_uses_partition_learning_window(self):
        new_path = REPO_ROOT / "run_batches_2018_2020_partition_learning_visual_monthly.bat"
        old_path = REPO_ROOT / "run_batches_2021_2024_visual_monthly.bat"

        self.assertTrue(new_path.exists(), "No-leak Stage 1 entrypoint is missing")
        self.assertFalse(old_path.exists(), "Old leaking Stage 1 entrypoint must not remain active")

        content = new_path.read_text(encoding="utf-8", errors="ignore").lower()
        self.assertIn("2018-2020", content)
        self.assertIn("for /l %%y in (2018,1,2020)", content)
        self.assertNotIn("for /l %%y in (2021,1,2024)", content)
        self.assertNotIn("geoxgb", content)
        self.assertNotIn("main_model_xgb.py", content)

    def test_stage2_main_workflow_excludes_geoxgb_and_points_to_new_stage1(self):
        content = read_text("spatial_weighted_consensus_clustering.bat").lower()

        self.assertIn("run_batches_2018_2020_partition_learning_visual_monthly.bat", content)
        self.assertNotIn("run_batches_2021_2024_visual_monthly.bat", content)
        self.assertNotIn("geoxgb", content)
        self.assertNotIn("geoxgbexperiment", content)

    def test_stage2_archive_validation_enumerates_directories(self):
        content = read_text("spatial_weighted_consensus_clustering.bat").lower()

        self.assertIn("for /d %%d in (%archive_glob%) do (", content)
        self.assertRegex(
            content,
            re.compile(
                r"set \"has_archives=\"\s+"
                r"for /d %%d in \(%archive_glob%\) do \(\s+"
                r"if exist \"%target_results_dir%\\%%~nxd\" set \"has_archives=1\"",
                re.MULTILINE,
            ),
        )

    def test_stage3_evaluates_2021_2024_but_all_mode_excludes_geoxgb(self):
        content = read_text("run_partition_k40_comparison_unified.bat").lower()

        self.assertIn("set start_month=2021-01", content)
        self.assertIn("set end_month=2024-12", content)
        self.assertIn("partition learning", content)
        self.assertIn("2018-2020", content)
        self.assertRegex(content, r"for %%m in \(georf geodt\) do")
        self.assertNotIn("for %%m in (georf geoxgb geodt) do", content)
        self.assertNotIn("compare_partitioned_vs_pooled_xgb", content)
        self.assertNotIn("result_partition_k40_compare_xgb", content)

    def test_stage3_all_mode_exits_nonzero_when_any_combo_fails(self):
        content = read_text("run_partition_k40_comparison_unified.bat").lower()

        self.assertRegex(
            content,
            re.compile(
                r"if \"%all_fail%\"==\"1\" \(\s+"
                r"echo some runs failed\. check output above for details\.\s+"
                r"pause\s+"
                r"exit /b 1\s+"
                r"\)",
                re.MULTILINE,
            ),
        )

    def test_main_aggregators_exclude_geoxgb(self):
        aggregate = read_text("other_outputs/aggregate_results.py")
        legacy_table = read_text("other_outputs/generate_table.py")

        self.assertNotIn("GeoXGB", aggregate)
        self.assertNotIn('_candidates("XGB"', aggregate)
        self.assertNotIn("GeoXGB", legacy_table)
        self.assertNotIn("GeoXGBExperiment", legacy_table)

    def test_main_aggregate_table_uses_horizon_display_labels(self):
        aggregate = read_text("other_outputs/aggregate_results.py")

        self.assertIn("Forecasting horizon", aggregate)
        self.assertIn("label_for_scope", aggregate)
        self.assertNotIn("lag" + "(months)", aggregate)
        self.assertNotIn("8-month predictions used as " + "12-month proxy", aggregate)

    def test_user_docs_do_not_advertise_old_stage1_or_geoxgb_main_workflow(self):
        docs = "\n".join(
            read_text(path)
            for path in ("README.md", "PIPELINE_WORKFLOW.md", "INSTALL.md")
            if (REPO_ROOT / path).exists()
        ).lower()

        self.assertNotIn("run_batches_2021_2024_visual_monthly.bat", docs)
        self.assertIn("run_batches_2018_2020_partition_learning_visual_monthly.bat", docs)
        self.assertNotRegex(docs, re.compile(r"model type[s]?:.*geoxgb"))
        self.assertNotIn("geoxgb partitions", docs)


if __name__ == "__main__":
    unittest.main()
