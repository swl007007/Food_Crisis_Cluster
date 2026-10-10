"""Focused discriminating contracts: truth/projection, availability/schedule, moments/routing, inference."""

import math
import unittest
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pandas as pd

from helpers import small_contract
from ipcch_alphaearth import aemoments, compact, descriptors, evaluation as ev, partition as pt, projection, schedule, targets
from ipcch_alphaearth.contract import load_contract
from ipcch_alphaearth.errors import ContractError, TechnicalError
from ipcch_alphaearth.months import parse_month


def D(*xs):
    return [None if x is None else Decimal(x) for x in xs]


class TruthAndProjection(unittest.TestCase):
    def test_inclusive_point_two_truth_boundary(self):
        v = targets.classify_row(D("0.50", "0.30", "0.15", "0.05", "0"), Decimal("100"))
        self.assertEqual(v.phase, 3)  # q3 = 0.20 exactly
        v = targets.classify_row(D("0.5001", "0.3", "0.1499", "0.05", "0"), Decimal("100"))
        self.assertEqual(v.phase, 2)

    def test_invalid_rows_and_p5_fill(self):
        self.assertEqual(targets.classify_row(D("0.5", "0.3", "0.2", None, "0"), Decimal(1)).reason, "missing_phase_1_to_4")
        self.assertEqual(targets.classify_row(D("0.5", "0.3", "0.2", "0", "0"), Decimal(0)).reason, "population_not_positive")
        self.assertEqual(targets.classify_row(D("0.5", "0.3", "0.2", "0", "0"), None).reason, "population_missing")
        self.assertEqual(targets.classify_row(D("1.2", "0", "0", "0", "0"), Decimal(1)).reason, "phase_share_out_of_bounds")
        self.assertEqual(targets.classify_row(D("0.5", "0.3", "0.05", "0.0", "0"), Decimal(1)).reason, "sum_out_of_bounds")
        v = targets.classify_row(D("0.5", "0.3", "0.2", "0", None), Decimal(1))
        self.assertTrue(v.valid and v.p5_filled)

    def test_prediction_decoding_inclusive(self):
        _, phase = projection.project_and_decode(np.array([[0.5, 0.2, 0.1, 0.0], [0.5, 0.19999999, 0.1, 0.0]]))
        self.assertEqual(phase.tolist(), [3, 2])

    def test_crossing_and_projection_before_clipping(self):
        star = projection.project(np.array([[0.6, 1.4, 0.2, 0.0]]))
        np.testing.assert_allclose(star, [[1.0, 1.0, 0.2, 0.0]])  # clipping first would give 0.8, 0.8
        star = projection.project(np.array([[0.3, 0.5, 0.1, -0.1]]))
        np.testing.assert_allclose(star, [[0.4, 0.4, 0.1, 0.0]])

    def test_absent_class_and_constant_truth_are_undefined(self):
        s = ev.exact_scores(np.array([1, 2, 3]), np.array([1, 2, 3]))
        self.assertIsNone(s["four_class.macro_f1"])  # class 4/5 absent from truth and prediction
        p = ev.panel(np.array([1, 2]), np.array([1, 2]), np.array([0.3, 0.3]), np.array([0.2, 0.4]), np.array([0.2, 0.4]))
        self.assertIsNone(p["q3_r2_projected"])
        g = ev.guard_check(s | {"q3_r2_projected": 0.5}, {"B1": s | {"q3_r2_projected": 0.5}, "B2": s | {"q3_r2_projected": 0.5},
                                                           "B0": s | {"q3_r2_projected": 0.5}}, load_contract())
        self.assertFalse(g["pass"])  # undefined macro-F1 never passes


class AvailabilityAndSchedule(unittest.TestCase):
    def test_label_at_fit_origin_allowed_later_refused(self):
        np.testing.assert_array_equal(schedule.decay_weights([100, 76], 100, 24), [1.0, 0.5])
        with self.assertRaises(ContractError):
            schedule.decay_weights([101], 100, 24)

    def test_features_cut_at_each_rows_own_origin_and_h1_is_not_h0(self):
        grid = compact.Grid(np.array([7]), parse_month("2019-01"), 36, {})
        x = np.arange(36, dtype=float)[None, :] + parse_month("2019-01")
        rows = pd.DataFrame({"area_id": [7, 7], "year": [2020, 2021], "month": [6, 3]})
        T = np.array([parse_month("2020-06"), parse_month("2021-03")])
        for h in (1, 3, 12):
            got = compact.gather(x, grid.index_of(rows["area_id"].to_numpy(), T - h))
            np.testing.assert_array_equal(got, T - h)  # value at origin is the source cell at O = T - H
        h1 = compact.gather(x, grid.index_of(rows["area_id"].to_numpy(), T - 1))
        h0 = compact.gather(x, grid.index_of(rows["area_id"].to_numpy(), T))
        self.assertFalse(np.array_equal(h1, h0))

    def test_trailing_window_ends_at_origin(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0, 100.0]])
        ma3 = compact.trailing(x, 3, 2, "mean")
        self.assertEqual(ma3[0, 3], 3.0)  # months 1..3 (indices), never the later 100

    def test_completed_season_boundary_is_first_day_of_next_month(self):
        seasons = pd.DataFrame({"admin_code": [5, 5], "season_year": [2020, 2020], "season": ["s1", "s2"],
                                "gs_start_date": ["2020-01-01", "2020-03-01"], "gs_end_date_exclusive": ["2020-04-01", "2020-04-02"],
                                **{m: [1.0, 2.0] for m in compact.SEASON_METRICS}})
        feats, _ = compact.completed_seasons(seasons, np.array([5]), np.array([parse_month("2020-03")]))
        self.assertEqual(feats[f"gs_last1__{compact.SEASON_METRICS[0]}__at_origin"].iloc[0], 1.0)  # ends exactly 2020-04-01
        self.assertTrue(np.isnan(feats[f"gs_last2__{compact.SEASON_METRICS[0]}__at_origin"].iloc[0]))  # 04-02 is after O+1

    def test_anchors_ignore_observed_labels_and_first_partial_year(self):
        c = load_contract()
        b = schedule.horizon_blocks(c, 1)
        self.assertEqual((b[0].target_lo, b[0].fit_origin), (parse_month("2018-02"), parse_month("2018-01")))
        main = [x for x in b if x.role == "main"][0]
        self.assertEqual(main.fit_origin, parse_month("2023-01"))
        keys = pd.DataFrame({"admin_code": [1, 1], "target_ord": [parse_month("2018-05"), parse_month("2019-03")]})
        enum = schedule.enumerate_requests(c, {h: keys for h in c["calendar"]["horizons_months"]})
        first = enum["requests"][enum["requests"]["request_id"] == "B1_h01_hist_2018"].iloc[0]
        self.assertEqual(first["fit_origin_month"], "2018-01")  # empty 2018-02..04 does not move the anchor
        self.assertEqual(int(first["predict_rows"]), 1)

    def test_fold_calendar_keeps_empty_folds(self):
        c = load_contract()
        keys = pd.DataFrame({"admin_code": [1], "target_ord": [parse_month("2024-05")]})
        f = schedule.fold_calendar(c, {h: keys for h in c["calendar"]["horizons_months"]})
        self.assertEqual(int((f["period"] == "main").sum()), 122)
        self.assertEqual(int((f["status"] == "no_valid_target").sum()), len(f) - 4)

    def test_bridge_and_provider_uniqueness(self):
        c = load_contract()
        self.assertEqual(schedule.check_fit_ceiling(c, schedule.all_blocks(c))["bridge_slots"], 22)
        b1 = schedule.Block("a", 1, "historical", 10, 20, 9, ("B1",), None)
        b2 = schedule.Block("b", 1, "historical", 20, 30, 19, ("B1",), None)
        with self.assertRaises(ContractError):
            schedule.assign_providers(pd.DataFrame({"target_ord": [20]}), [b1, b2])


def piece(pid, A, mean, var, status="complete"):
    return {"piece_id": pid, "status": status, "valid_area": A, "S": [A * m for m in mean], "Q": [A * (v + m * m) for m, v in zip(mean, var)]}


class MomentsAndRouting(unittest.TestCase):
    def test_merge_moments_not_average_sd(self):
        m1, m2 = [0.0] * 64, [1.0] * 64
        merged = aemoments.merge_pieces([piece("a", 3.0, m1, [0.0] * 64), piece("b", 1.0, m2, [0.0] * 64)], ["a", "b"])
        d = aemoments.descriptor(merged, 4.0)
        self.assertAlmostEqual(d["mean"][0], 0.25)
        self.assertAlmostEqual(d["sd"][0], math.sqrt(0.25 * 0.75))  # pooled, while the average piece SD is 0
        self.assertTrue(d["usable"])

    def test_coverage_threshold_inclusive_and_nan_not_zero(self):
        merged = aemoments.merge_pieces([piece("a", 4.0, [0.1] * 64, [0.01] * 64)], ["a"])
        self.assertTrue(aemoments.descriptor(merged, 5.0)["usable"])  # exactly 0.80
        low = aemoments.descriptor(aemoments.merge_pieces([piece("a", 3.999999, [0.1] * 64, [0.01] * 64)], ["a"]), 5.0)
        self.assertFalse(low["usable"])
        self.assertEqual(low["route_reason"], "low_coverage")
        self.assertTrue(all(math.isnan(v) for v in low["mean"] + low["sd"]))

    def test_no_data_versus_failed_export(self):
        zero = aemoments.descriptor(aemoments.merge_pieces([piece("a", 0.0, [0.0] * 64, [0.0] * 64)], ["a"]), 5.0)
        self.assertEqual(zero["route_reason"], "no_coverage")
        with self.assertRaises(aemoments.ExtractionFailure):
            aemoments.merge_pieces([piece("a", 1.0, [0.0] * 64, [0.0] * 64)], ["a", "b"])  # planned piece missing
        with self.assertRaises(aemoments.ExtractionFailure):
            aemoments.merge_pieces([piece("a", 1.0, [0.0] * 64, [0.0] * 64, status="FAILED")], ["a"])
        bad = piece("a", 1.0, [0.0] * 64, [0.0] * 64)
        bad["S"][3] = float("nan")
        with self.assertRaises(aemoments.ExtractionFailure):
            aemoments.merge_pieces([bad], ["a"])

    def test_variance_cancellation_tolerance(self):
        p = piece("a", 1.0, [0.5] * 64, [0.0] * 64)
        p["Q"][0] = 0.25 - 1e-13
        d = aemoments.descriptor(aemoments.merge_pieces([p], ["a"]), 1.0)
        self.assertEqual(d["negative_variance_zeroed"], 1)
        p["Q"][0] = 0.25 - 1e-9
        with self.assertRaises(aemoments.ExtractionFailure):
            aemoments.descriptor(aemoments.merge_pieces([p], ["a"]), 1.0)

    def test_descriptor_table_and_failed_admin(self):
        c = load_contract()
        cols = {"admin_code": [1, 2], "piece_id": ["1|x", "2|x"], "asset_id": ["x", "x"], "status": ["complete", "complete"], "valid_area": [5.0, 1.0]}
        for b in aemoments.BANDS:
            cols[f"S_{b}"] = [0.5, 0.1]
            cols[f"Q_{b}"] = [0.1, 0.02]
        pieces = pd.DataFrame(cols)
        plan = pd.DataFrame({"admin_code": [1, 2, 3], "piece_id": ["1|x", "2|x", None], "asset_id": ["x", "x", None]})
        polys = pd.DataFrame({"admin_code": [1, 2, 3], "polygon_area_m2": [5.0, 5.0, 7.0]})
        table, ledger = descriptors.descriptor_table(pieces, plan, polys, c)
        self.assertEqual(ledger["route_reason"].tolist(), ["usable", "low_coverage", "no_coverage"])
        Dr, routable = descriptors.row_descriptors(np.array([1, 2, 3, 1]), table, c)
        self.assertEqual(routable.tolist(), [True, False, False, True])
        self.assertTrue(np.isnan(Dr[1]).all() and not (Dr[1] == 0).any())
        with self.assertRaises(TechnicalError):
            descriptors.descriptor_table(pieces.iloc[:1], plan, polys, c)  # admin 2's planned piece missing

    def test_overlap_ownership_no_double_count(self):
        from shapely.geometry import box

        from ipcch_alphaearth.extraction import ownership_pieces

        admin = box(0, 0, 4, 1)
        fps = {"B_tile": box(1.5, -1, 5, 2), "A_tile": box(-1, -1, 2.5, 2)}
        pieces = ownership_pieces(admin, fps)
        self.assertEqual([p["asset_id"] for p in pieces], ["A_tile", "B_tile"])  # lexicographic ownership
        self.assertAlmostEqual(pieces[0]["geometry"].area, 2.5)
        self.assertAlmostEqual(pieces[1]["geometry"].area, 1.5)  # B minus A's overlap
        self.assertAlmostEqual(sum(p["geometry"].area for p in pieces), admin.area)

    def test_unseen_admin_follows_frozen_thresholds(self):
        pmap = pt.PartitionMap({"r": (0, 0.0), "r1": (1, 2.0)}, ["r0", "r10", "r11"])
        Dn = np.array([[-1.0, 9.0], [0.0, 9.0], [1.0, 2.0], [1.0, 2.5], [np.nan, np.nan]])
        self.assertEqual(list(pmap.route(Dn, np.isfinite(Dn).all(axis=1))), ["r0", "r0", "r10", "r11", None])


class Inference(unittest.TestCase):
    def frame(self, g, b, country):
        return pd.DataFrame({"country_key": country, "phase_truth": [3, 1, 3, 1, 3, 2] * 2, "G_phase": g, "B2_phase": b})

    def test_identical_arms_have_zero_delta_draws(self):
        c = small_contract(**{"evaluation.bootstrap.draws": 500})
        ph = [3, 1, 1, 1, 3, 2] * 2
        rec, draws = ev.paired_bootstrap(self.frame(ph, ph, list("AABBCC") * 2), "G_phase", "B2_phase", c)
        self.assertEqual(rec["lower_bound"], 0.0)
        self.assertTrue((draws["delta"] == 0).all())
        self.assertTrue((draws.filter(like="m::").sum(axis=1) == 3).all())  # K countries drawn K times

    def test_undefined_draws_never_dropped(self):
        c = small_contract(**{"evaluation.bootstrap.draws": 300})
        f = pd.DataFrame({"country_key": ["A", "B"], "phase_truth": [3, 1], "G_phase": [3, 1], "B2_phase": [1, 1]})
        rec, draws = ev.paired_bootstrap(f, "G_phase", "B2_phase", c)
        self.assertIsNone(rec["lower_bound"])
        self.assertIn("undefined replicate", rec["na_reason"])
        self.assertEqual(len(draws), 300)

    def test_single_country_inconclusive(self):
        c = small_contract(**{"evaluation.bootstrap.draws": 50})
        f = pd.DataFrame({"country_key": ["A"] * 3, "phase_truth": [3, 3, 1], "G_phase": [3, 3, 1], "B2_phase": [3, 1, 1]})
        rec, _ = ev.paired_bootstrap(f, "G_phase", "B2_phase", c)
        self.assertIsNone(rec["lower_bound"])
        self.assertIn("K < 2", rec["na_reason"])

    def test_selection_exact_ranking_and_fallback(self):
        def rec(rid, k, lam, g1, g2, leaves=2, guards=True):
            return {"recipe_id": rid, "leaf_budget": k, "lambda_region": lam,
                    "years": {2021: {"gain": g1, "guards_pass": guards, "n_leaves": leaves, "depth": 1},
                              2022: {"gain": g2, "guards_pass": guards, "n_leaves": leaves, "depth": 1}}}
        F = Fraction
        sel = ev.select_recipe([rec("a", 2, 1.0, F(1, 100), F(1, 100)), rec("b", 4, 0.1, F(1, 100), F(1, 100), leaves=4),
                                rec("c", 8, 1.0, F(-1, 1000), F(5, 100)), rec("d", 2, 0.01, F(1, 50), F(0), guards=False)])
        self.assertEqual(sel["selected"], "a")  # equal mean gain: fewer realized leaves; c has a negative year; d fails guards
        self.assertEqual(ev.select_recipe([rec("x", 2, 1.0, F(0), F(0))])["selected"], "B2")  # mean must be > 0
        self.assertEqual(ev.select_recipe([rec("x", 2, 1.0, None, F(1, 10))])["selected"], "B2")  # undefined never qualifies

    def test_per_h_decision_statuses(self):
        c = load_contract()
        good = {"binary.f1": Fraction(1, 2), **{m: Fraction(1, 2) for m in ev.GUARDED}, "q3_r2_projected": 0.3}
        better = dict(good, **{"binary.f1": Fraction(1, 2) + Fraction(1, 100)})
        panels = {"G": better, "B2": good, "B1": good, "B0": good}
        years = {2024: {"G": better, "B2": good}, 2025: {"G": better, "B2": good}}
        d = ev.horizon_decision(None, panels, years, {"lower_bound": 0.001}, c)
        self.assertEqual(d["status"], "protected_stable_benefit")
        d = ev.horizon_decision(None, panels, years, {"lower_bound": -0.001}, c)
        self.assertEqual(d["status"], "imprecise_bound_not_above_zero")
        small = dict(good, **{"binary.f1": Fraction(1, 2) + Fraction(1, 1000)})
        d = ev.horizon_decision(None, dict(panels, G=small), {2024: {"G": small, "B2": good}}, {"lower_bound": 0.0005}, c)
        self.assertEqual(d["status"], "gain_below_practical_floor")
        worse_recall = dict(better, **{"binary.recall": Fraction(1, 2) - Fraction(2, 100)})
        d = ev.horizon_decision(None, dict(panels, G=worse_recall), years, {"lower_bound": 0.001}, c)
        self.assertEqual(d["status"], "primary_met_guard_tradeoff")


if __name__ == "__main__":
    unittest.main()
