"""Contract tests for FEWSNETCleanPersistenceExperiment/run_pipeline.py (Stage 1).

These exercise the scientific boundaries a defect could silently cross in the Stage 1
worker and its schedule: the D18/D22 candidate inventory and its deduplication, the
D20 [O-35 months, O) mask and the released target-month group filter, fitting-row-only
imputation, the release pin and its two import traps, and the exact artifacts the
released Stage 2 helpers read. Synthetic fixtures are reused from the prepare-stage
module; nothing here fits a model.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import prepare_data as pdat  # noqa: E402
import run_pipeline as rp  # noqa: E402
from test_prepare_contracts import (  # noqa: E402
    SyntheticPanel, SyntheticPanelTestCase, prepared_from_master,
)


# --------------------------------------------------------------------------------------
# D18/D22/D55: the finite candidate inventory
# --------------------------------------------------------------------------------------


class TestCandidateSchedule(unittest.TestCase):
    def test_unique_development_jobs_per_arm(self) -> None:
        jobs = rp.development_jobs(pdat.REFERENCE_ARM)
        self.assertEqual(len(jobs), 51)
        self.assertEqual(len({job.name for job in jobs}), 51)

    def test_overlapping_window_jobs_run_once_under_both_roles(self) -> None:
        jobs = rp.development_jobs("BASE")
        shared = [job for job in jobs if len(job.roles) > 1]
        self.assertEqual(len(shared), 9)
        for job in shared:
            self.assertEqual(job.year, 2016)
            self.assertEqual(set(job.roles), {"calibration", "selection"})
        self.assertEqual(len(rp.role_jobs("BASE", "calibration")), 33)
        self.assertEqual(len(rp.role_jobs("BASE", "selection")), 27)
        # Deduplication is by identical same-arm job only; the union is not 33 + 27.
        self.assertEqual(len(jobs), 33 + 27 - 9)

    def test_role_windows_and_observed_candidate_months(self) -> None:
        calibration = rp.role_jobs(pdat.REFERENCE_ARM, "calibration")
        selection = rp.role_jobs(pdat.REFERENCE_ARM, "selection")
        self.assertEqual({job.year for job in calibration}, {2014, 2015, 2016})
        self.assertEqual({job.year for job in selection}, {2016, 2017, 2018})
        early = {job.month for job in calibration if job.year <= 2015}
        late = {job.month for job in calibration if job.year >= 2016}
        # A55: the 2014-2015 Jan/Apr/Jul candidates must not be dropped by a 2/6/10 filter.
        self.assertEqual(early, {1, 4, 7, 10})
        self.assertEqual(late, {2, 6, 10})

    def test_every_scope_is_carried_with_its_own_horizon(self) -> None:
        jobs = rp.development_jobs(pdat.REFERENCE_ARM)
        self.assertEqual({job.scope for job in jobs}, {1, 2, 3})
        for job in jobs:
            self.assertEqual(job.horizon, {1: 4, 2: 8, 3: 12}[job.scope])

    def test_arm_inventory_matches_the_frozen_663_job_bound(self) -> None:
        arms = [name for name, _ in pdat.RECIPE_MANIFEST] + [pdat.REFERENCE_ARM]
        self.assertEqual(len(arms), 13)
        total = sum(len(rp.development_jobs(arm)) for arm in arms)
        schedule = pdat.build_schedule()
        self.assertEqual(total, schedule["stage1"]["development_jobs_upper_bound"])
        self.assertEqual(total, 663)

    def test_job_names_are_unique_across_arms(self) -> None:
        names = [
            job.name
            for arm in ("reference", "BASE", "ABCDE")
            for job in rp.development_jobs(arm)
        ]
        self.assertEqual(len(names), len(set(names)))


# --------------------------------------------------------------------------------------
# D20: the released training mask and target-month group filter
# --------------------------------------------------------------------------------------


class TestRowSelection(SyntheticPanelTestCase):
    def months_of(self, rows: np.ndarray) -> List[int]:
        return sorted({int(m) for m in self.sources.target_month_idx[rows]})

    def test_training_window_is_o_minus_35_to_o_exclusive(self) -> None:
        selection = rp.select_candidate_rows(self.sources, 4, 2020, 2)
        origin = pdat.month_index(2020, 2) - 4
        months = self.months_of(selection.train_rows)
        self.assertTrue(months, "expected nonempty synthetic training support")
        self.assertLess(max(months), origin, "origin-month labels must be excluded")
        self.assertGreaterEqual(min(months), origin - 35)
        self.assertNotIn(origin, months)
        self.assertNotIn(origin - 36, months)

    def test_horizon_shifts_the_window_by_exactly_the_lag(self) -> None:
        fs1 = rp.select_candidate_rows(self.sources, 4, 2020, 2)
        fs3 = rp.select_candidate_rows(self.sources, 12, 2020, 2)
        self.assertEqual(
            max(self.months_of(fs1.train_rows)) - max(self.months_of(fs3.train_rows)), 8
        )
        # The target cohort is the same month regardless of horizon.
        np.testing.assert_array_equal(fs1.target_rows, fs3.target_rows)

    def test_target_rows_are_exactly_the_candidate_month(self) -> None:
        selection = rp.select_candidate_rows(self.sources, 8, 2018, 10)
        self.assertEqual(self.months_of(selection.target_rows),
                         [pdat.month_index(2018, 10)])
        self.assertEqual(selection.evidence["target_month"], "2018-10")
        self.assertEqual(selection.evidence["origin_month"], "2018-02")

    def test_evidence_records_real_month_area_and_class_counts(self) -> None:
        evidence = rp.select_candidate_rows(self.sources, 4, 2020, 2).evidence
        self.assertEqual(
            evidence["train_observed_month_count"], len(evidence["train_observed_months"])
        )
        self.assertGreater(evidence["train_areas"], 0)
        self.assertEqual(
            evidence["train_class_counts"]["0"] + evidence["train_class_counts"]["1"],
            evidence["rows_after_group_filter"],
        )
        self.assertEqual(
            evidence["rows_in_temporal_window"] - evidence["rows_after_group_filter"],
            evidence["rows_removed_by_group_filter"],
        )


class TestWindowBoundaries(unittest.TestCase):
    """The exact [O-35, O) edges, with a label planted on the boundary month.

    The standard fixture only labels Feb/Jun/Oct, and with H in {4, 8, 12} the month
    35 before the origin is never one of those, so an extra observed label is planted
    at that exact month to make the inclusive edge observable.
    """

    TARGET = (2021, 6)
    HORIZON = 4

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        origin = pdat.month_index(*self.TARGET) - self.HORIZON
        self.origin_label = pdat.month_label(origin)             # 2021-02, labeled
        self.included = pdat.month_label(origin - 35)            # 2018-03, planted
        self.excluded = pdat.month_label(origin - 36)            # 2018-02, labeled
        overrides = {
            (area, self.included): {"fews_ipc": 2.0, "fews_ipc_crisis": 0}
            for area in (7, 11, 23)
        }
        panel = SyntheticPanel(Path(self._tmp.name), master_overrides=overrides)
        master, ledger = panel.load()
        self.sources = prepared_from_master(master, ledger)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_o_minus_35_is_included_and_o_minus_36_and_o_are_excluded(self) -> None:
        selection = rp.select_candidate_rows(self.sources, self.HORIZON, *self.TARGET)
        months = list(selection.evidence["train_observed_months"])
        self.assertEqual(self.included, "2018-03")
        self.assertEqual(self.excluded, "2018-02")
        self.assertEqual(self.origin_label, "2021-02")
        self.assertIn(self.included, months)
        self.assertNotIn(self.excluded, months)
        self.assertNotIn(self.origin_label, months)
        self.assertEqual(
            selection.evidence["training_window"],
            "[2018-03, 2021-02) = 35 calendar months, origin month excluded",
        )


class TestTargetMonthGroupFilter(unittest.TestCase):
    """An area absent from the target month loses its training rows (released rule)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        blanked = {
            (7, "2018-02"): {"fews_ipc": "", "fews_ipc_crisis": ""},
        }
        panel = SyntheticPanel(Path(self._tmp.name), master_overrides=blanked)
        master, ledger = panel.load()
        self.sources = prepared_from_master(master, ledger)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_area_without_a_target_label_is_dropped_from_training(self) -> None:
        selection = rp.select_candidate_rows(self.sources, 4, 2018, 2)
        areas = self.sources.areas
        target_areas = set(int(a) for a in areas[
            self.sources.target_area_idx[selection.target_rows]
        ])
        train_areas = set(int(a) for a in areas[
            self.sources.target_area_idx[selection.train_rows]
        ])
        self.assertNotIn(7, target_areas)
        self.assertNotIn(7, train_areas)
        self.assertGreater(selection.evidence["rows_removed_by_group_filter"], 0)
        self.assertIn("target month", selection.evidence["train_group_filter"])

    def test_the_removed_rows_are_still_counted_in_the_temporal_window(self) -> None:
        selection = rp.select_candidate_rows(self.sources, 4, 2018, 2)
        self.assertGreater(
            selection.evidence["rows_in_temporal_window"],
            selection.evidence["rows_after_group_filter"],
        )


# --------------------------------------------------------------------------------------
# Ordered schemas reaching the fit
# --------------------------------------------------------------------------------------


class TestCandidateSchema(SyntheticPanelTestCase):
    def test_every_recipe_is_an_ordered_subset_of_the_single_superset(self) -> None:
        superset = list(pdat.updated_superset_columns())
        for recipe, _ in pdat.RECIPE_MANIFEST:
            columns = list(pdat.recipe_columns(recipe))
            self.assertEqual(len(columns), pdat.DECLARED_RECIPE_WIDTHS[recipe])
            positions = [superset.index(name) for name in columns]
            self.assertEqual(positions, sorted(positions), f"{recipe} reorders the superset")
            self.assertEqual(len(set(columns)), len(columns))

    def test_build_candidate_matrix_matches_the_declared_widths(self) -> None:
        rows = np.arange(6)
        for arm in ("BASE", "ABCD", pdat.REFERENCE_ARM):
            matrix, columns = rp.build_candidate_matrix(self.sources, arm, 4, rows)
            expected = (
                len(pdat.REFERENCE_COLUMNS) if arm == pdat.REFERENCE_ARM
                else pdat.DECLARED_RECIPE_WIDTHS[arm]
            )
            self.assertEqual(matrix.shape, (rows.size, expected))
            self.assertEqual(list(columns), list(
                pdat.REFERENCE_COLUMNS if arm == pdat.REFERENCE_ARM
                else pdat.recipe_columns(arm)
            ))

    def test_recipe_slice_equals_a_direct_superset_slice(self) -> None:
        rows = np.arange(5)
        superset, superset_columns = pdat.build_updated_superset(self.sources, 8, rows=rows)
        matrix, columns = rp.build_candidate_matrix(self.sources, "ACDE", 8, rows)
        index_of = {name: i for i, name in enumerate(superset_columns)}
        for position, name in enumerate(columns):
            np.testing.assert_allclose(
                matrix[:, position], superset[:, index_of[name]], equal_nan=True
            )

    def test_identifiers_never_enter_the_model_input_schema(self) -> None:
        forbidden = {"FEWSNET_admin_code", "ISO", "ISO3", "IPC_admin_code", "ISO_encoded",
                     "partition_id", "unit_name", "ADMIN0", "ADMIN1", "ADMIN2", "ADMIN3"}
        for arm in [name for name, _ in pdat.RECIPE_MANIFEST] + [pdat.REFERENCE_ARM]:
            columns = set(
                pdat.REFERENCE_COLUMNS if arm == pdat.REFERENCE_ARM
                else pdat.recipe_columns(arm)
            )
            self.assertFalse(columns & forbidden, f"{arm} exposes an identifier")


# --------------------------------------------------------------------------------------
# D23: imputation fitted on real fitting rows only
# --------------------------------------------------------------------------------------


class TestStage1Imputation(unittest.TestCase):
    def test_validation_and_target_rows_never_influence_the_fill(self) -> None:
        columns = ["a", "b"]
        train = np.array([[1.0, np.nan], [2.0, np.nan], [np.nan, np.nan]])
        fitting = np.array([True, True, False])  # third row is internal validation
        target = np.array([[np.nan, np.nan], [500.0, 7.0]])

        imputer = pdat.MaxPlusImputer()
        imputer.fit(train, fitting_mask=fitting, columns=columns)
        filled_train = imputer.transform(train)
        filled_target = imputer.transform(target)

        # max over real fitting rows of column a is 2.0 -> 200.0; never 500.0 from a
        # target row, and never influenced by the validation row.
        self.assertEqual(imputer.fill_values_[0], 200.0)
        self.assertEqual(filled_train[2, 0], 200.0)
        self.assertEqual(filled_target[0, 0], 200.0)
        self.assertEqual(filled_target[1, 0], 500.0)
        self.assertEqual(imputer.n_fitting_rows_, 2)

    def test_all_missing_fitting_column_keeps_the_released_zero_sentinel(self) -> None:
        imputer = pdat.MaxPlusImputer()
        imputer.fit(
            np.array([[1.0, np.nan], [2.0, np.nan]]),
            fitting_mask=np.array([True, True]), columns=["a", "b"],
        )
        self.assertEqual(imputer.fill_values_[1], 0.0)
        manifest = imputer.manifest()
        self.assertEqual(manifest["columns_all_missing_in_fitting_rows"], ["b"])


# --------------------------------------------------------------------------------------
# Split-decision evidence
# --------------------------------------------------------------------------------------


class TestSplitGateTrace(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "stdout.txt"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_rejected_and_accepted_decisions_are_both_recovered(self) -> None:
        self.path.write_text(
            "noise\n"
            "F1 performance gate: parent=0.736369, candidate=0.743770, accepted=False\n"
            "= Branch  not split\n"
            "F1 performance gate: parent=0.500000, candidate=0.600000, accepted=True\n",
            encoding="utf-8",
        )
        trace = rp.parse_split_gate_trace(self.path)
        self.assertEqual(len(trace), 2)
        self.assertFalse(trace[0]["accepted"])
        self.assertAlmostEqual(trace[0]["gain"], 0.007401, places=6)
        self.assertTrue(trace[1]["accepted"])
        self.assertAlmostEqual(trace[1]["gain"], 0.1, places=9)

    def test_a_missing_log_yields_an_empty_trace_not_an_error(self) -> None:
        self.assertEqual(rp.parse_split_gate_trace(self.path / "absent"), [])


# --------------------------------------------------------------------------------------
# D56: the Stage 2 weight transform reported alongside each candidate
# --------------------------------------------------------------------------------------


class TestStage2Weight(unittest.TestCase):
    def test_equal_partitioned_and_base_f1_gives_zero_weight(self) -> None:
        value = 0.3172413793103448
        self.assertEqual(rp._logit_clip(value) - rp._logit_clip(value), 0.0)

    def test_weight_matches_the_released_clipped_logit_difference(self) -> None:
        weight = max(rp._logit_clip(0.6) - rp._logit_clip(0.5), 0.0)
        self.assertAlmostEqual(weight, float(np.log(0.6 / 0.4)), places=12)

    def test_zero_and_one_are_clipped_rather_than_infinite(self) -> None:
        self.assertTrue(np.isfinite(rp._logit_clip(0.0)))
        self.assertTrue(np.isfinite(rp._logit_clip(1.0)))


# --------------------------------------------------------------------------------------
# The artifacts the released Stage 2 helpers actually read
# --------------------------------------------------------------------------------------


class TestStage2Inputs(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "run"
        (self.root / "geometry").mkdir(parents=True)
        pd.DataFrame({
            "FEWSNET_admin_code": [0, 1, 2], "lat": [1.0, 2.0, 3.0], "lon": [4.0, 5.0, 6.0],
        }).to_csv(self.root / "geometry" / "FEWSNET_admin_code_lat_lon.csv", index=False)
        rp.write_json(self.root / "run_manifest.json", {
            "experiment": "test", "completed_stages": ["setup"],
        })
        self.context = rp.RunContext(self.root, create=False)
        self.jobs = rp.role_jobs(pdat.REFERENCE_ARM, "selection")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_candidate(self, job: rp.CandidateJob, status: str, **scores) -> None:
        job_dir = self.root / "stage1" / job.arm / job.name
        job_dir.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, object] = {"identity": job.identity(), "status": status}
        if status == "completed":
            payload["scores"] = {"f1(1)": scores["f1"], "f1_base(1)": scores["f1_base"]}
            payload["support"] = {"real_fitting_rows": 10, "real_validation_rows": 2,
                                  "labeled_target_rows": 3}
            payload["partitions"] = {"n_terminal_partitions": 2}
            pd.DataFrame({
                "FEWSNET_admin_code": [0, 2], "partition_id": ["0", "1"],
            }).to_csv(job_dir / "correspondence_table.csv", index=False)
        else:
            payload["exclusion_reason"] = "D20 requires nonempty real support"
            payload["support"] = {"real_fitting_rows": 0, "real_validation_rows": 0,
                                  "labeled_target_rows": 0}
        rp.write_json(job_dir / "candidate.json", payload)

    def test_only_completed_eligible_candidates_enter_the_index(self) -> None:
        self._write_candidate(self.jobs[0], "completed", f1=0.4, f1_base=0.3)
        self._write_candidate(self.jobs[1], "excluded_insufficient_support")
        summary = rp.write_stage2_inputs(self.context, pdat.REFERENCE_ARM, "selection")

        target = self.root / "stage2_inputs" / pdat.REFERENCE_ARM / "selection"
        index = pd.read_csv(target / "linked_tables" / "main_index.csv")
        self.assertEqual(list(index.columns), [
            "name", "variant", "year", "month", "forecasting_scope", "f1(1)", "f1_base(1)",
        ])
        self.assertEqual(len(index), 1)
        self.assertEqual(index.loc[0, "name"], self.jobs[0].name)

        status = pd.read_csv(target / "candidate_status.csv")
        self.assertEqual(len(status), len(self.jobs))
        self.assertIn("excluded_insufficient_support", set(status["status"]))
        self.assertIn("not_run", set(status["status"]))
        self.assertEqual(summary["completed_eligible_candidates"], 1)
        self.assertEqual(summary["candidates_with_positive_stage2_weight"], 1)
        self.assertFalse(summary["stage2_ready"])
        self.assertFalse(summary["month_ind"])

    def test_partition_tables_cover_the_full_admin_range_with_s_minus_one(self) -> None:
        self._write_candidate(self.jobs[0], "completed", f1=0.4, f1_base=0.4)
        rp.write_stage2_inputs(self.context, pdat.REFERENCE_ARM, "selection")
        path = (self.root / "stage2_inputs" / pdat.REFERENCE_ARM / "selection"
                / "linked_tables" / "partitions" / f"{self.jobs[0].name}_partition.csv")
        table = pd.read_csv(path, dtype={"partition_id": str})
        self.assertEqual(len(table), 5718)
        self.assertEqual(table.loc[table["FEWSNET_admin_code"] == 0, "partition_id"].iloc[0], "0")
        self.assertEqual(table.loc[table["FEWSNET_admin_code"] == 1, "partition_id"].iloc[0], "s-1")
        self.assertEqual(table.loc[table["FEWSNET_admin_code"] == 5717, "partition_id"].iloc[0], "s-1")

    def test_equal_f1_candidates_contribute_no_positive_weight(self) -> None:
        self._write_candidate(self.jobs[0], "completed", f1=0.4, f1_base=0.4)
        summary = rp.write_stage2_inputs(self.context, pdat.REFERENCE_ARM, "selection")
        self.assertEqual(summary["completed_eligible_candidates"], 1)
        self.assertEqual(summary["candidates_with_positive_stage2_weight"], 0)

    def test_lat_lon_accompanies_the_linked_tables(self) -> None:
        self._write_candidate(self.jobs[0], "completed", f1=0.4, f1_base=0.3)
        rp.write_stage2_inputs(self.context, pdat.REFERENCE_ARM, "selection")
        target = self.root / "stage2_inputs" / pdat.REFERENCE_ARM / "selection"
        self.assertTrue((target / "FEWSNET_admin_code_lat_lon.csv").is_file())


# --------------------------------------------------------------------------------------
# Release pin and import isolation
# --------------------------------------------------------------------------------------


class TestBaselinePin(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.destination = Path(cls._tmp.name) / "baseline"
        cls.runtime = rp.extract_baseline(rp.BASELINE_ZIP, cls.destination)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_the_pinned_zip_is_the_released_artifact(self) -> None:
        self.assertEqual(rp.sha256_file(rp.BASELINE_ZIP), rp.RELEASE_SHA256)
        self.assertEqual(self.runtime.manifest_version, rp.RELEASE_VERSION)
        self.assertEqual(self.runtime.manifest_source_commit, rp.RELEASE_SOURCE_COMMIT)
        self.assertGreater(self.runtime.payload_files_verified, 0)

    def test_extraction_refuses_a_used_destination(self) -> None:
        with self.assertRaises(rp.PipelineError):
            rp.extract_baseline(rp.BASELINE_ZIP, self.destination)

    def test_a_modified_payload_is_detected_on_disk(self) -> None:
        target = self.runtime.root / "config.py"
        original = target.read_bytes()
        try:
            target.write_bytes(original + b"\n# local edit\n")
            with self.assertRaises(rp.PipelineError):
                rp.verify_extracted_baseline(self.runtime.root)
        finally:
            target.write_bytes(original)
        rp.verify_extracted_baseline(self.runtime.root)

    def test_imports_resolve_inside_the_pinned_copy_with_drops_disabled(self) -> None:
        root = self.runtime.root.resolve()
        with rp.baseline_imports(self.runtime) as (config, georf_module):
            # Both names, in both namespaces: `from config import *` copied the alias
            # into the GeoRF module, which is where the drop list is actually read.
            for module in (config, georf_module):
                for attribute in ("FEATURE_DROP", "feature_drop"):
                    self.assertFalse(getattr(module, attribute)["enable"])
                    self.assertEqual(getattr(module, attribute)["cols"], [])

            import src  # noqa: PLC0415
            self.assertEqual(src.__path__, [str(root / "src")])

            for name, location in self.runtime.module_locations.items():
                self.assertTrue(
                    Path(location).resolve().is_relative_to(root),
                    f"{name} escaped the pinned copy: {location}",
                )
            self.assertIn("src.partition.partition_opt", self.runtime.module_locations)
            self.assertIn("src.customize.customize", self.runtime.module_locations)
            self.assertEqual(config.GOVERNING_METRIC, "class_1_f1")
            self.assertEqual(config.MIN_CLASS_1_IMPROVEMENT_THRESHOLD, 0.01)
            self.assertEqual(config.MIN_DEPTH, 1)
            self.assertEqual(config.MAX_DEPTH, 6)
            self.assertEqual(config.VAL_RATIO, 0.20)
            self.assertTrue(config.GROUP_SPLIT["enable"])
            self.assertEqual(config.GROUP_SPLIT["random_state"], 42)
        self.assertNotIn("config", sys.modules)
        self.assertNotIn(rp.DIAGNOSTIC_MODULE, sys.modules)

    def test_smote_stays_rejected_by_the_released_model_wrapper(self) -> None:
        with rp.baseline_imports(self.runtime):
            from src.model.model_RF import RFmodel  # noqa: PLC0415
            with self.assertRaises(ValueError):
                RFmodel("ckpt", 100, use_smote=True)
            self.assertFalse(RFmodel("ckpt", 100).use_smote)

    def test_released_within_area_split_keeps_singletons_in_training(self) -> None:
        with rp.baseline_imports(self.runtime):
            from src.utils.split import group_aware_train_val_split  # noqa: PLC0415
            groups = np.array([1, 1, 1, 1, 1, 2])
            result = group_aware_train_val_split(
                groups=groups, val_ratio=0.20, min_val_per_group=1,
                random_state=42, skip_singleton_groups=True,
            )
            x_set = result["X_set"]
            self.assertEqual(int((x_set[groups == 1] == 1).sum()), 1)
            self.assertEqual(int((x_set[groups == 2] == 1).sum()), 0)
            # Determinism: the same inputs must reproduce the same assignment.
            repeat = group_aware_train_val_split(
                groups=groups, val_ratio=0.20, min_val_per_group=1,
                random_state=42, skip_singleton_groups=True,
            )
            np.testing.assert_array_equal(x_set, repeat["X_set"])

    def test_the_f1_gate_requires_a_strict_gain_above_one_point(self) -> None:
        with rp.baseline_imports(self.runtime):
            from src.partition.partition_opt import select_f1_children  # noqa: PLC0415

            def decide(child_hits: int) -> bool:
                truth = np.array([1] * 50 + [0] * 50)
                parent = np.array([1] * 25 + [0] * 75)
                child = np.array([1] * child_hits + [0] * (100 - child_hits))
                accepted, _, _, base, best = select_f1_children(
                    truth[:50], truth[50:], parent[:50], parent[50:],
                    child[:50], child[50:],
                )
                self.assertLessEqual(base, best)
                return bool(accepted)

            self.assertFalse(decide(25))   # identical to the parent: parent wins ties
            self.assertTrue(decide(45))    # a clear improvement is accepted


# --------------------------------------------------------------------------------------
# Run-root discipline
# --------------------------------------------------------------------------------------


class TestRunRoot(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "run"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_a_fresh_root_is_created_and_a_used_one_is_rejected(self) -> None:
        rp.RunContext(self.root, create=True)
        self.assertTrue((self.root / "run_manifest.json").is_file())
        with self.assertRaises(rp.PipelineError):
            rp.RunContext(self.root, create=True)

    def test_an_uninitialized_root_cannot_be_continued(self) -> None:
        with self.assertRaises(rp.PipelineError):
            rp.RunContext(self.root / "never", create=False)

    def test_standing_limitations_are_recorded_in_the_manifest(self) -> None:
        context = rp.RunContext(self.root, create=True)
        manifest = json.loads((self.root / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["limitations"], list(rp.STANDING_LIMITATIONS))
        self.assertEqual(context.manifest["scope"], "stage1_candidates_and_schedule")


if __name__ == "__main__":
    unittest.main(verbosity=2)
