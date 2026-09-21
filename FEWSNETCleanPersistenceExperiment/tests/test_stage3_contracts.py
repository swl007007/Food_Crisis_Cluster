"""Contract tests for Stage 3, calibration, thresholds and reporting (D36-D45, D61/D62).

These cover the boundaries where a defect changes a reported number rather than
crashing: the D61/R65 training pool, the local-support gate, the pooled fallback routes
including the one the released comparator gets wrong, single-class probability
extraction, the up-only correction and its null-threshold case, and the pooled-counts
F1 that D22/D41 require instead of an average of per-month values.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "PersistenceCorrectionExperiment"))

import prepare_data as pdat  # noqa: E402
import report_results as rr  # noqa: E402
import run_pipeline as rp  # noqa: E402


class _SingleClassModel:
    """Minimal stand-in for a forest fitted on one class."""

    classes_ = [0]

    @staticmethod
    def predict_proba(X):
        return np.ones((len(X), 1))


class _TwoClassModel:
    classes_ = [0, 1]

    @staticmethod
    def predict_proba(X):
        column = np.linspace(0.0, 1.0, len(X))
        return np.column_stack([1.0 - column, column])


# --------------------------------------------------------------------------------------
# D61/D62: fold inventory, probability extraction and routing labels
# --------------------------------------------------------------------------------------


class TestFoldInventory(unittest.TestCase):
    def test_development_is_eighteen_folds(self) -> None:
        folds = rp.development_folds(pdat.REFERENCE_ARM)
        self.assertEqual(len(folds), 18)
        self.assertEqual({fold.role for fold in folds}, set(rp.DEVELOPMENT_ROLES))

    def test_final_windows_match_d40(self) -> None:
        """fs1 11, fs2 10, fs3 9 target months; the windows are deliberately unequal."""
        folds = rp.final_folds(pdat.REFERENCE_ARM)
        counts = {
            horizon: sum(1 for fold in folds if fold.horizon == horizon)
            for horizon in pdat.HORIZONS
        }
        self.assertEqual(counts, {4: 11, 8: 10, 12: 9})
        self.assertEqual(len(folds), 30)

    def test_final_candidate_jobs_reuse_overlapping_dates(self) -> None:
        """2018 repeats between the selection and final windows, so only the genuinely
        new dates are scheduled; refitting them would waste work and risk divergence."""
        new = rp.new_final_jobs(pdat.REFERENCE_ARM)
        self.assertEqual(len(new), 18)
        existing = {job.name for job in rp.development_jobs(pdat.REFERENCE_ARM)}
        self.assertFalse({job.name for job in new} & existing)


class TestClassOneProbability(unittest.TestCase):
    def test_single_class_model_yields_zero_not_indexerror(self) -> None:
        """D61.3 gates only *local* models on both classes, so a single-class global
        pool is legitimate. The naive predict_proba(...)[:, 1] raises IndexError."""
        X = np.zeros((4, 2))
        with self.assertRaises(IndexError):
            _SingleClassModel.predict_proba(X)[:, 1]
        self.assertTrue(np.array_equal(
            rp.class1_probability(_SingleClassModel, X), np.zeros(4)
        ))

    def test_two_class_model_matches_the_naive_column(self) -> None:
        X = np.zeros((5, 2))
        self.assertTrue(np.allclose(
            rp.class1_probability(_TwoClassModel, X),
            _TwoClassModel.predict_proba(X)[:, 1],
        ))


class TestMalformedEstimators(unittest.TestCase):
    """A legitimately single-class fit returns zeros; a malformed one must raise."""

    def test_missing_class_metadata_raises(self) -> None:
        class NoClasses:
            @staticmethod
            def predict_proba(X):
                return np.ones((len(X), 1))

        with self.assertRaises(rp.PipelineError):
            rp.class1_probability(NoClasses, np.zeros((3, 2)))

    def test_non_binary_labels_raise(self) -> None:
        class Weird:
            classes_ = [2]

            @staticmethod
            def predict_proba(X):
                return np.ones((len(X), 1))

        with self.assertRaises(rp.PipelineError):
            rp.class1_probability(Weird, np.zeros((3, 2)))

    def test_column_count_disagreeing_with_classes_raises(self) -> None:
        class Mismatched:
            classes_ = [0, 1]

            @staticmethod
            def predict_proba(X):
                return np.ones((len(X), 1))

        with self.assertRaises(rp.PipelineError):
            rp.class1_probability(Mismatched, np.zeros((3, 2)))


class TestLeaveOneYearOut(unittest.TestCase):
    """Exercises the production leave_one_year_out, not a restatement of its rule."""

    @staticmethod
    def _frame(years) -> "object":
        import pandas as pd

        rows = []
        for year in years:
            for horizon in pdat.HORIZONS:
                for index in range(4):
                    rows.append({
                        "horizon_months": horizon,
                        "target_month": f"{year}-02",
                        "FEWSNET_admin_code": index,
                        "target_label": index % 2,
                        "persistence": 0,
                        "p_raw": 0.9 if index % 2 else 0.1,
                        "p_calibrated": 0.9 if index % 2 else 0.1,
                    })
        return pd.DataFrame(rows)

    @staticmethod
    def _thresholds():
        return {
            f"{variant}_h{horizon}": {"tau": 0.5}
            for horizon in pdat.HORIZONS for variant in rp.PROBABILITY_VARIANTS
        }

    def test_all_four_years_are_always_attempted(self) -> None:
        """A frame missing 2021 must still produce four checks, not three."""
        result = rr.leave_one_year_out(self._frame([2022, 2023, 2024]), self._thresholds())
        self.assertEqual(
            sorted(result["by_excluded_year"]), ["2021", "2022", "2023", "2024"]
        )
        self.assertEqual(result["by_excluded_year"]["2021"]["rows_excluded"], 0)

    def test_excluding_an_absent_year_changes_nothing(self) -> None:
        frame = self._frame([2022, 2023, 2024])
        result = rr.leave_one_year_out(frame, self._thresholds())
        full = rr.six_cell_gains(frame, self._thresholds())[1]
        self.assertAlmostEqual(
            result["by_excluded_year"]["2021"]["primary_gain"], full, places=12
        )

    def test_removing_every_row_makes_the_check_incomplete(self) -> None:
        result = rr.leave_one_year_out(self._frame([2022]), self._thresholds())
        self.assertIsNone(result["by_excluded_year"]["2022"]["primary_gain"])
        self.assertFalse(result["all_defined"])


class TestBootstrapCalendar(unittest.TestCase):
    """The date universe is D40's schedule, not whatever dates have support."""

    def test_scheduled_dates_are_the_union_of_the_three_windows(self) -> None:
        dates = rr.scheduled_target_dates()
        self.assertEqual(len(dates), 11)
        self.assertEqual(dates[0], "2021-06")
        self.assertEqual(dates[-1], "2024-10")

    def test_a_shrunken_calendar_is_rejected(self) -> None:
        import pandas as pd

        frame = pd.DataFrame([{
            "horizon_months": horizon, "target_month": "2022-02",
            "FEWSNET_admin_code": 0, "target_label": 1, "persistence": 0,
            "p_raw": 0.9, "p_calibrated": 0.9,
        } for horizon in pdat.HORIZONS])
        thresholds = {
            f"{variant}_h{horizon}": {"tau": 0.5}
            for horizon in pdat.HORIZONS for variant in rp.PROBABILITY_VARIANTS
        }
        with self.assertRaises(rr.ReportError):
            rr.joint_bootstrap(frame, thresholds, target_dates=["2022-02"])

    def test_dates_outside_the_schedule_are_rejected(self) -> None:
        import pandas as pd

        frame = pd.DataFrame([{
            "horizon_months": horizon, "target_month": "2025-02",
            "FEWSNET_admin_code": 0, "target_label": 1, "persistence": 0,
            "p_raw": 0.9, "p_calibrated": 0.9,
        } for horizon in pdat.HORIZONS])
        thresholds = {
            f"{variant}_h{horizon}": {"tau": 0.5}
            for horizon in pdat.HORIZONS for variant in rp.PROBABILITY_VARIANTS
        }
        with self.assertRaises(rr.ReportError):
            rr.joint_bootstrap(frame, thresholds)


class TestEvaluationCohort(unittest.TestCase):
    """D22 forbids a candidate improving its score by altering its cohort."""

    @staticmethod
    def _frame(rows):
        import pandas as pd

        return pd.DataFrame(rows)

    def test_duplicate_keys_are_rejected(self) -> None:
        frame = self._frame([
            {"horizon_months": 4, "FEWSNET_admin_code": 1, "target_month": "2020-02",
             "target_label": 1, "persistence": 0},
            {"horizon_months": 4, "FEWSNET_admin_code": 1, "target_month": "2020-02",
             "target_label": 1, "persistence": 0},
        ])
        with self.assertRaises(rr.ReportError):
            rr.evaluation_cohort(frame)

    def test_a_flipped_label_changes_the_cohort_signature(self) -> None:
        """A key-set comparison would miss this; the cohort signature must not."""
        base = [
            {"horizon_months": 4, "FEWSNET_admin_code": 1, "target_month": "2020-02",
             "target_label": 1, "persistence": 0},
            {"horizon_months": 4, "FEWSNET_admin_code": 2, "target_month": "2020-02",
             "target_label": 0, "persistence": 0},
        ]
        flipped = [dict(row) for row in base]
        flipped[1]["target_label"] = 1
        first = rr._cohort_signature(rr.evaluation_cohort(self._frame(base))[4])
        second = rr._cohort_signature(rr.evaluation_cohort(self._frame(flipped))[4])
        self.assertNotEqual(first, second)

    def test_a_changed_persistence_value_changes_the_signature(self) -> None:
        base = [{"horizon_months": 4, "FEWSNET_admin_code": 1,
                 "target_month": "2020-02", "target_label": 1, "persistence": 0}]
        altered = [dict(base[0], persistence=1)]
        self.assertNotEqual(
            rr._cohort_signature(rr.evaluation_cohort(self._frame(base))[4]),
            rr._cohort_signature(rr.evaluation_cohort(self._frame(altered))[4]),
        )


class TestFinalCohortReconciliation(unittest.TestCase):
    """D39/R43: the two final arms and persistence share one paired cohort.

    Equal row counts are not pairing. These exercise the comparison that the close
    audit found missing from the final report.
    """

    @staticmethod
    def _cohort(rows):
        import pandas as pd

        return rr._normalise_cohort(pd.DataFrame(rows))

    def test_same_row_count_over_different_areas_is_detected(self) -> None:
        a = self._cohort([
            {"FEWSNET_admin_code": 1, "target_month": "2022-02",
             "target_label": 1, "persistence": 0},
            {"FEWSNET_admin_code": 2, "target_month": "2022-02",
             "target_label": 0, "persistence": 0},
        ])
        b = self._cohort([
            {"FEWSNET_admin_code": 1, "target_month": "2022-02",
             "target_label": 1, "persistence": 0},
            {"FEWSNET_admin_code": 3, "target_month": "2022-02",
             "target_label": 0, "persistence": 0},
        ])
        self.assertEqual(len(a), len(b))
        self.assertNotEqual(rr._cohort_signature(a), rr._cohort_signature(b))

    def test_same_keys_with_a_different_persistence_value_is_detected(self) -> None:
        base = [{"FEWSNET_admin_code": 1, "target_month": "2022-02",
                 "target_label": 1, "persistence": 0}]
        altered = [dict(base[0], persistence=1)]
        self.assertNotEqual(
            rr._cohort_signature(self._cohort(base)),
            rr._cohort_signature(self._cohort(altered)),
        )

    def test_row_order_does_not_affect_the_signature(self) -> None:
        rows = [
            {"FEWSNET_admin_code": 2, "target_month": "2022-02",
             "target_label": 0, "persistence": 0},
            {"FEWSNET_admin_code": 1, "target_month": "2022-02",
             "target_label": 1, "persistence": 0},
        ]
        self.assertEqual(
            rr._cohort_signature(self._cohort(rows)),
            rr._cohort_signature(self._cohort(list(reversed(rows)))),
        )

    def test_required_source_disclosures_are_present(self) -> None:
        """D33/D48/D49 made these disclosures a condition of retaining the predictors,
        so their absence from the report is a contract violation, not a style choice."""
        text = " ".join(rr.FINAL_LIMITATIONS).lower()
        for required in (
            "two-sided linear interpolation",   # D33 Gini
            "ungrouped forward filling",        # D33 nightlight SD
            "zero filling",                     # D33 nightlight mean/SD
            "original missingness is not recovered",  # D33 block E
            "prior-year snapshot",              # D48 population
            "2018 gpw",                         # D48 availability
            "market-access vintage is 2015",    # D49
        ):
            self.assertIn(required, text, f"missing required disclosure: {required!r}")


class TestRoutingLabels(unittest.TestCase):
    def test_pooled_routes_are_distinguishable(self) -> None:
        """"unassigned", "below the gate" and "never trained" mean different things and
        must never collapse into one label."""
        routes = {
            rp.ROUTE_LOCAL, rp.ROUTE_POOLED_UNASSIGNED, rp.ROUTE_POOLED_UNSUPPORTED,
            rp.ROUTE_POOLED_UNSEEN, rp.ROUTE_POOLED_NO_SPLIT,
        }
        self.assertEqual(len(routes), 5)


# --------------------------------------------------------------------------------------
# D37/D38: the up-only correction and its null-threshold outcome
# --------------------------------------------------------------------------------------


class TestCorrection(unittest.TestCase):
    def test_only_zero_to_one_is_expressible(self) -> None:
        persistence = np.array([0, 0, 1, 1])
        probability = np.array([0.9, 0.1, 0.9, 0.1])
        out = rr.apply_correction(persistence, probability, 0.5)
        self.assertEqual(list(out), [1, 0, 1, 1])

    def test_equality_does_not_flip(self) -> None:
        """D37 says strictly greater. p == tau must leave persistence unchanged."""
        out = rr.apply_correction(np.array([0]), np.array([0.5]), 0.5)
        self.assertEqual(int(out[0]), 0)

    def test_null_threshold_returns_persistence_unchanged(self) -> None:
        """A no-correction outcome is a valid frozen result, not a missing number, and
        must never be passed to the numeric override."""
        persistence = np.array([0, 1, 0])
        out = rr.apply_correction(persistence, np.array([0.99, 0.99, 0.99]), None)
        self.assertTrue(np.array_equal(out, persistence))

    def test_correction_never_reduces_predicted_positives(self) -> None:
        rng = np.random.default_rng(0)
        persistence = rng.integers(0, 2, 200)
        probability = rng.random(200)
        out = rr.apply_correction(persistence, probability, 0.4)
        self.assertGreaterEqual(int(out.sum()), int(persistence.sum()))


# --------------------------------------------------------------------------------------
# D22/D41: pooled confusion counts, not averaged F1
# --------------------------------------------------------------------------------------


class TestPooledF1(unittest.TestCase):
    def test_pooling_differs_from_averaging_month_f1(self) -> None:
        """The contract forbids averaging per-month F1. This fixture shows the two
        give different answers, so the prohibition is not cosmetic."""
        month_a = (np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0]))
        month_b = (np.array([1, 0, 0, 0]), np.array([0, 1, 1, 1]))
        pooled = rr.confusion(
            np.concatenate([month_a[0], month_b[0]]),
            np.concatenate([month_a[1], month_b[1]]),
        )
        pooled_f1 = rr.f1_from_counts(pooled)
        averaged = np.mean([
            rr.f1_from_counts(rr.confusion(*month_a)),
            rr.f1_from_counts(rr.confusion(*month_b)),
        ])
        self.assertNotAlmostEqual(pooled_f1, float(averaged), places=6)

    def test_zero_denominator_is_zero_not_nan(self) -> None:
        counts = rr.confusion(np.zeros(5, dtype=int), np.zeros(5, dtype=int))
        self.assertEqual(rr.f1_from_counts(counts), 0.0)

    def test_perfect_prediction_is_one(self) -> None:
        truth = np.array([1, 0, 1, 0])
        self.assertEqual(rr.f1_from_counts(rr.confusion(truth, truth)), 1.0)

    def test_counts_partition_the_rows(self) -> None:
        rng = np.random.default_rng(1)
        truth = rng.integers(0, 2, 50)
        predicted = rng.integers(0, 2, 50)
        counts = rr.confusion(truth, predicted)
        self.assertEqual(
            counts["tp"] + counts["fp"] + counts["fn"] + counts["tn"], counts["n"]
        )


# --------------------------------------------------------------------------------------
# D43/D44: adjudication logic
# --------------------------------------------------------------------------------------


class TestAdjudication(unittest.TestCase):
    @staticmethod
    def _bootstrap(interval, complete=True):
        return {"complete": complete, "interval_95": interval}

    @staticmethod
    def _loyo(positive, defined=True, years=rr.LEAVE_ONE_YEAR_OUT_YEARS):
        """Supplies real per-year results, not just summary flags.

        The flags alone must never decide the outcome: adjudicate recomputes from
        these four values, so a fixture that omitted them was testing nothing.
        """
        gain = 0.02 if positive else -0.02
        return {
            "all_strictly_positive": positive,
            "all_defined": defined,
            "by_excluded_year": {
                str(year): {"primary_gain": gain} for year in years
            },
        }

    def test_summary_flags_alone_cannot_produce_a_pass(self) -> None:
        """The flags claim success while the four results say otherwise."""
        lying = {
            "all_strictly_positive": True, "all_defined": True,
            "by_excluded_year": {
                str(year): {"primary_gain": -0.1}
                for year in rr.LEAVE_ONE_YEAR_OUT_YEARS
            },
        }
        result = rr.adjudicate(0.03, self._bootstrap([0.01, 0.05]), lying)
        self.assertEqual(result["d45_status"], "complete_fail")

    def test_a_missing_required_year_is_incomplete(self) -> None:
        partial = self._loyo(True, years=(2021, 2022, 2023))
        partial["all_defined"] = True
        result = rr.adjudicate(0.03, self._bootstrap([0.01, 0.05]), partial)
        self.assertEqual(result["d45_status"], "incomplete")

    def test_positive_interval_and_stable_years_passes(self) -> None:
        result = rr.adjudicate(0.03, self._bootstrap([0.01, 0.05]), self._loyo(True))
        self.assertEqual(result["d45_status"], "complete_pass")

    def test_interval_touching_zero_fails(self) -> None:
        """D43.1 requires the lower endpoint strictly above zero."""
        result = rr.adjudicate(0.03, self._bootstrap([0.0, 0.05]), self._loyo(True))
        self.assertFalse(result["condition_1_positive_with_interval_above_zero"])
        self.assertEqual(result["d45_status"], "complete_fail")

    def test_one_negative_year_fails(self) -> None:
        result = rr.adjudicate(0.03, self._bootstrap([0.01, 0.05]), self._loyo(False))
        self.assertEqual(result["d45_status"], "complete_fail")

    def test_missing_evidence_is_incomplete_not_a_null(self) -> None:
        """D45's third row: incomplete evidence must never be reported as a null."""
        result = rr.adjudicate(0.03, self._bootstrap(None, complete=False), self._loyo(True))
        self.assertEqual(result["d45_status"], "incomplete")
        self.assertIn("NOT a scientific null", result["interpretation"])

    def test_advisory_reference_is_not_a_gate(self) -> None:
        """D42: a gain below +0.01 can still pass, and one above it can still fail."""
        small = rr.adjudicate(0.004, self._bootstrap([0.001, 0.01]), self._loyo(True))
        self.assertFalse(small["meets_advisory_reference"])
        self.assertEqual(small["d45_status"], "complete_pass")
        large = rr.adjudicate(0.05, self._bootstrap([-0.01, 0.1]), self._loyo(True))
        self.assertTrue(large["meets_advisory_reference"])
        self.assertEqual(large["d45_status"], "complete_fail")

    def test_negative_gain_stays_negative(self) -> None:
        """D42 warns explicitly against taking abs(delta)."""
        result = rr.adjudicate(-0.02, self._bootstrap([-0.05, -0.001]), self._loyo(False))
        self.assertLess(result["point_estimate"], 0)
        self.assertEqual(result["d45_status"], "complete_fail")


class TestJointBootstrap(unittest.TestCase):
    """Exercises joint_bootstrap itself, not NumPy's seeding."""

    @staticmethod
    def _full_frame():
        import pandas as pd

        rows = []
        for horizon in pdat.HORIZONS:
            for year, month in pdat.final_target_dates(horizon):
                for index in range(6):
                    rows.append({
                        "horizon_months": horizon,
                        "target_month": f"{year:04d}-{month:02d}",
                        "FEWSNET_admin_code": index,
                        "target_label": index % 2,
                        "persistence": 0,
                        "p_raw": 0.9 if index % 2 else 0.1,
                        "p_calibrated": 0.9 if index % 2 else 0.1,
                    })
        return pd.DataFrame(rows)

    @staticmethod
    def _thresholds():
        return {
            f"{variant}_h{horizon}": {"tau": 0.5}
            for horizon in pdat.HORIZONS for variant in rp.PROBABILITY_VARIANTS
        }

    def test_draws_are_reproducible_end_to_end(self) -> None:
        frame, thresholds = self._full_frame(), self._thresholds()
        first = rr.joint_bootstrap(frame, thresholds)
        second = rr.joint_bootstrap(frame, thresholds)
        self.assertEqual(first["gains"], second["gains"])
        self.assertEqual(first["interval_95"], second["interval_95"])

    def test_the_protocol_constants_are_actually_used(self) -> None:
        result = rr.joint_bootstrap(self._full_frame(), self._thresholds())
        self.assertEqual(result["valid_draws"], rr.BOOTSTRAP_DRAWS)
        self.assertEqual(result["seed"], rr.BOOTSTRAP_SEED)
        self.assertEqual(result["n_target_dates"], 11)
        self.assertTrue(result["complete"])

    def test_every_draw_retains_its_shared_multiplicities(self) -> None:
        """D44's draw is shared across horizons; the ledger is what proves it and
        what lets the interval be recomputed independently."""
        result = rr.joint_bootstrap(self._full_frame(), self._thresholds())
        self.assertEqual(len(result["draw_ledger"]), result["attempts"])
        for entry in result["draw_ledger"][:20]:
            self.assertEqual(sum(entry["multiplicity"].values()), 11)
            self.assertEqual(set(entry["multiplicity"]), set(result["target_dates"]))

    def test_interval_is_the_linear_percentile_of_the_retained_gains(self) -> None:
        result = rr.joint_bootstrap(self._full_frame(), self._thresholds())
        expected = [
            float(np.percentile(result["gains"], 2.5, method="linear")),
            float(np.percentile(result["gains"], 97.5, method="linear")),
        ]
        self.assertEqual(result["interval_95"], expected)


if __name__ == "__main__":
    unittest.main()
