"""Vocabulary tests: every old metric pattern maps, the mapping is one-to-one per family, and
names respect MLflow character rules.

Run: /home/swl007007/.venvs/ipcch-mlflow/bin/python -m unittest discover -s IPCCHMLflow/tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import naming  # noqa: E402
from extract import SourceConflict  # noqa: E402

M = ("binary.f1", "binary.count.tp", "four_class.macro_f1", "q3_r2_projected", "q3_r2_raw")


class MetricNameTest(unittest.TestCase):
    def m(self, family, old):
        return naming.metric_name(family, old)

    def test_panels_periods_and_cohorts(self):
        self.assertEqual(self.m("p6_geoxgb", "main.E_all.binary.f1"), "primary.all_scored.binary.f1")
        self.assertEqual(self.m("p6_geoxgb", "supplementary.E_persist.q3_r2_projected"),
                         "holdout.persistence_available.share_phase3plus_r2")
        self.assertEqual(self.m("p6_geoxgb", "main.E_persist.q3_r2_raw"),
                         "primary.persistence_available.share_phase3plus_r2_raw")
        self.assertEqual(self.m("p6_geoxgb", "main.E_all.n"), "primary.all_scored.n_rows")
        self.assertEqual(self.m("yearly_geoxgb", "main.local_eligible.n_keys"),
                         "primary.regional_model_fitted.n_rows_reported")
        self.assertEqual(self.m("yearly_geoxgb", "main.local_persist_matched.binary.f1"),
                         "primary.regional_model_fitted_and_persistence_available.binary.f1")
        self.assertEqual(self.m("mlp_fixed_map", "supplementary.local_eligible.historical_support_rejected.binary.f1"),
                         "holdout.regional_model_fitted.gate_rejected_too_little_validation.binary.f1")
        self.assertEqual(self.m("split2024_sensitivity", "y2026.E_all.binary.f1"), "year_2026.all_scored.binary.f1")
        self.assertEqual(self.m("split2024_sensitivity", "combined.E_persist.n"), "combined.persistence_available.n_rows")

    def test_window_probe(self):
        f = "history_window_sensitivity"
        self.assertEqual(self.m(f, "selected_dates.new_local_support.binary.f1"),
                         "selected_months.regional_model_fitted_full_history_only.binary.f1")
        self.assertEqual(self.m(f, "selected_dates.mapped.n"), "selected_months.in_partition_map.n_rows")
        self.assertEqual(self.m(f, "selected_date_2024-09.all.binary.f1"), "target_month_2024-09.all_scored.binary.f1")
        self.assertEqual(self.m(f, "selected_date_2025-01.fit_n"), "target_month_2025-01.training_rows")

    def test_reference_panels_deltas_and_bootstrap(self):
        self.assertEqual(self.m("yearly_geoxgb", "main.E_all.comparator.p6geo.binary.f1"),
                         "primary.all_scored.reference_partitioned_gated.binary.f1")
        self.assertEqual(self.m("climate_perturbation", "main.E_all.matched_old.p6pool.n"),
                         "primary.all_scored.reference_pooled.n_rows")
        self.assertEqual(self.m("p6_geoxgb", "main.E_all.delta.geo_minus_pool.binary.f1"),
                         "primary.all_scored.delta.partitioned_gated_minus_pooled.binary.f1")
        self.assertEqual(self.m("climate_perturbation", "main.E_all.delta.new_minus_old_pool.binary.f1"),
                         "primary.all_scored.delta.pooled_minus_reference_pooled.binary.f1")
        self.assertEqual(self.m("mlp_fixed_map", "main.E_all.delta.base_minus_xgbpool.binary.f1"),
                         "primary.all_scored.delta.global_base_minus_reference_pooled.binary.f1")
        self.assertEqual(self.m("mlp_fixed_map", "main.E_all.delta.pool_minus_base.binary.f1"),
                         "primary.all_scored.delta.pooled_minus_global_base.binary.f1")
        self.assertEqual(self.m("yearly_geoxgb", "main.E_persist.comparator_delta.p6geo_minus_persistence.binary.f1"),
                         "primary.persistence_available.delta.reference_partitioned_gated_minus_persistence.binary.f1")
        self.assertEqual(self.m("climate_perturbation",
                                "main.E_persist.matched_old.delta.p6geo_minus_persistence.binary.f1"),
                         "primary.persistence_available.delta.reference_partitioned_gated_minus_persistence.binary.f1")
        self.assertEqual(self.m("yearly_geoxgb", "main.local_eligible.adopted.delta.local_minus_pool.binary.f1"),
                         "primary.regional_model_fitted.gate_used_regional.delta.regional_ungated_minus_pooled.binary.f1")
        self.assertEqual(self.m("p6_geoxgb", "main.E_all.contrast.geo_vs_pool.ci_lower"),
                         "primary.all_scored.bootstrap.partitioned_gated_minus_pooled.binary.f1.ci_low")
        self.assertEqual(self.m("mlp_fixed_map", "main.E_persist.contrast.geo_minus_persistence.point_delta"),
                         "primary.persistence_available.bootstrap.partitioned_gated_minus_persistence.binary.f1.delta")
        self.assertEqual(self.m("climate_perturbation", "main.E_persist.matched_old.contrast.p6geo_minus_persistence.K"),
                         "primary.persistence_available.bootstrap.reference_partitioned_gated_minus_persistence"
                         ".binary.f1.countries")

    def test_diagnostic_trees(self):
        self.assertEqual(self.m("p6_geoxgb", "main.coverage.E_persist_keys"),
                         "primary.coverage.persistence_available_rows")
        self.assertEqual(self.m("p6_geoxgb", "main.routes.rows.local"), "primary.routes.rows.local")
        self.assertEqual(self.m("mlp_fixed_map", "main.label_flips_geo_vs_pool.crisis"),
                         "primary.label_flips.partitioned_gated_vs_pooled.crisis")
        self.assertEqual(self.m("climate_perturbation", "main.label_flips_vs_p6.geo.crisis_changed"),
                         "primary.label_flips.partitioned_gated_vs_reference.crisis_changed")
        self.assertEqual(self.m("split2024_sensitivity", "y2025.local_rows"), "year_2025.regional_routed_rows")
        self.assertEqual(self.m("mlp_fixed_map", "main.E_all.q3_mse.star"),
                         "primary.all_scored.share_phase3plus_mse")
        self.assertEqual(self.m("mlp_fixed_map", "seed_summary.main_h03.geo_minus_pool_f1.mean"),
                         "seed_summary.primary.lead_03.partitioned_gated_minus_pooled.binary.f1.mean")
        self.assertEqual(self.m("mlp_fixed_map", "seed_summary.supplementary_h12.f1_xgbgeo.max"),
                         "seed_summary.holdout.lead_12.reference_partitioned_gated.binary.f1.max")

    def test_unknown_names_refused(self):
        for fam, old in (("p6_geoxgb", "main.E_weird.binary.f1"), ("p6_geoxgb", "decade.E_all.binary.f1"),
                         ("p6_geoxgb", "main.E_all.delta.foo_minus_pool.binary.f1"), ("nope", "main.E_all.n")):
            with self.assertRaises(SourceConflict):
                naming.metric_name(fam, old)

    def test_one_to_one_on_a_full_pattern_grid(self):
        olds = []
        for p in ("main", "supplementary"):
            for c in ("E_all", "E_persist", "local_eligible", "local_persist_matched"):
                olds += [f"{p}.{c}.{x}" for x in M + ("n",)]
                olds += [f"{p}.{c}.comparator.p6geo.{x}" for x in M]
                olds += [f"{p}.{c}.delta.geo_minus_pool.{x}" for x in M]
            olds += [f"{p}.local_eligible.n_keys", f"{p}.E_persist.comparator_delta.p6geo_minus_persistence.binary.f1"]
        new = [naming.metric_name("yearly_geoxgb", o) for o in olds]
        self.assertEqual(len(set(new)), len(olds))
        naming.check_one_to_one("yearly_geoxgb", olds)
        with self.assertRaisesRegex(SourceConflict, "same new name"):
            naming.check_one_to_one("climate_perturbation",
                                    ["main.E_persist.matched_old.delta.p6geo_minus_persistence.binary.f1",
                                     "main.E_persist.comparator_delta.p6geo_minus_persistence.binary.f1"])


class NamesTest(unittest.TestCase):
    def test_arms_and_roles(self):
        self.assertEqual(naming.arm("p6_geoxgb", "geo"), ("partitioned_gated", None, "candidate"))
        self.assertEqual(naming.arm("p6_geoxgb", "persistence"), ("persistence", None, "baseline"))
        self.assertEqual(naming.arm("mlp_fixed_map", "base"), ("global_base", None, "candidate"))
        self.assertEqual(naming.arm("mlp_fixed_map", "local")[0], "regional_ungated")
        self.assertEqual(naming.arm("history_window_sensitivity", "exp_global"), ("pooled", "full_history", "candidate"))
        with self.assertRaises(SourceConflict):
            naming.arm("p6_geoxgb", "local")

    def test_long_titles_start_with_the_short_title(self):
        for fam, f in naming.FAMILIES.items():      # truncated run names must still read the same
            self.assertTrue(f["long"].startswith(f["short"] + " ("), fam)

    def test_lead(self):
        self.assertEqual((naming.lead_label("3"), naming.lead_tag("3"), naming.lead_label("12")),
                         ("3-month", "03", "12-month"))

    def test_run_and_model_names_are_mlflow_safe(self):
        rn = naming.run_name("mlp_fixed_map", "pool", "6", "43")
        self.assertEqual(rn, "MLP residual on fixed maps | pooled | 6-month | seed 43")
        self.assertEqual(naming.run_name("p6_geoxgb", "geo", "3", "42"), "GeoXGB reference | partitioned_gated | 3-month")
        self.assertEqual(naming.run_name("p6_geoxgb", "persistence", "3", "none"), "GeoXGB reference | persistence | 3-month")
        self.assertEqual(naming.run_name("history_window_sensitivity", "base_local", "12", "42"),
                         "GeoXGB window probe | regional_ungated | 36-month window | 12-month")
        for fam in naming.FAMILIES:
            for old_arm in naming.ARMS[fam]:
                if old_arm == "persistence":
                    continue
                rm = naming.registered_model_name(fam, old_arm, "6")
                lm = naming.logged_model_name(fam, old_arm, "6", "42")
                self.assertFalse(set("/:") & set(rm), rm)
                self.assertFalse(set("/:.%\"'") & set(lm), lm)
        self.assertEqual(naming.registered_model_name("p6_geoxgb", "geo", "3"),
                         "IPCCH GeoXGB reference | partitioned_gated | 3-month")

    def test_dataset_names(self):
        self.assertEqual(naming.eval_dataset_name("p6_geoxgb", "3", "main", "E_persist", None),
                         "IPCCH eval | 3-month | 2023-04..2025-10 | persistence_available")
        self.assertEqual(naming.eval_dataset_name("split2024_sensitivity", "3", "main", "E_all", None),
                         "IPCCH eval | 3-month | 2025-04..2025-10 | all_scored")
        self.assertEqual(naming.eval_dataset_name("history_window_sensitivity", "12", "selected_dates", "all",
                                                  ["2024-09", "2025-01"]),
                         "IPCCH eval | 12-month | target months 2024-09, 2025-01 | all_scored")
        self.assertEqual(naming.eval_dataset_name("yearly_geoxgb", "1", "main", "local_eligible", None),
                         "IPCCH eval | 1-month | 2023-02..2025-10 | regional_model_fitted (GeoXGB yearly refit)")
        with self.assertRaisesRegex(SourceConflict, "no scored months"):
            naming.eval_dataset_name("split2024_sensitivity", "12", "main", "E_all", None)


if __name__ == "__main__":
    unittest.main()
