"""Focused four-class contract checks: python -B tests/test_baseline.py.

Hand-computable fixtures for every item implement.md section 4 lists.
"""
from contextlib import redirect_stdout
from fractions import Fraction
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from src.metrics import fourclass
from src.metrics.metrics import get_prf
from src.partition import partition_opt as opt
from src.partition import transformation as trans
from src.model.model_RF import RFmodel, MaxPlusImputer
from src.feature import fourclass_features as ff
from src.utils.split import group_aware_train_val_split
from scripts import compare_partitioned_vs_pooled_rf_k40_nc4 as compare
from scripts import report_fourclass as report
from scripts import prepare_fourclass as prep
from scripts.step4_similarity_matrix import compute_plan_weights, load_lat_lon
from scripts.step1_merge_results import MODEL_CONFIG, load_results_table

SCHEMA = prep.SCHEMA_PATH
from scripts import run_stage1 as stage1  # noqa: E402
from src.utils import run_identity as rid  # noqa: E402


def m(label):
    return prep.mi(label)


class PresetRF(RFmodel):
    """Deterministic labels from the row id (column 0), with production save/load."""

    def __init__(self, path, labeller):
        super().__init__(path, 2, num_class=4, n_jobs=1)
        self.labeller = labeller
        self.trained = []

    def train(self, X, y, branch_id='', **kwargs):
        self.trained.append((branch_id, len(X)))
        self.imputer = MaxPlusImputer().fit(X)
        labels = self.labeller(X[:, 0].astype(int), len(X))
        Xp = np.vstack([X, np.zeros((4, X.shape[1]))])
        self.model = DecisionTreeClassifier(random_state=0).fit(Xp, np.r_[labels, 0, 1, 2, 3])
        self.fit_record = {'fit_seq': len(self.fit_log), 'trained_under': branch_id,
                           'real_rows': len(X), 'imputer_sha256': self.imputer.digest()}
        self.fit_log.append(dict(self.fit_record))


class ClassesAndMetrics(unittest.TestCase):
    def test_phase_merge_and_missing(self):
        merged = fourclass.merge_phase([1, 2, 3, 4, 5, np.nan])
        np.testing.assert_array_equal(merged[:5], [1, 2, 3, 4, 4])
        self.assertTrue(np.isnan(merged[5]))
        self.assertEqual(fourclass.CLASS_LABELS[3], '4或5')
        with self.assertRaises(ValueError):
            fourclass.merge_phase([0])
        with self.assertRaises(ValueError):
            fourclass.merge_phase([2.5])

    def test_fixed_four_macro_absent_class_scores_zero(self):
        truth = np.array([0, 1, 2, 2])
        self.assertAlmostEqual(fourclass.macro_f1(truth, truth), 0.75)
        self.assertEqual(fourclass.macro_f1_exact(truth, truth), Fraction(3, 4))

    def test_wrong_class_counts_fp_and_fn(self):
        truth, pred = np.array([0, 0, 1]), np.array([0, 1, 1])
        tp, fp, fn = fourclass.class_counts(fourclass.confusion(truth, pred))
        np.testing.assert_array_equal(tp, [1, 1, 0, 0])
        np.testing.assert_array_equal(fp, [0, 1, 0, 0])
        np.testing.assert_array_equal(fn, [1, 0, 0, 0])
        expected = f1_score(truth, pred, labels=[0, 1, 2, 3], average='macro', zero_division=0)
        self.assertAlmostEqual(fourclass.macro_f1(truth, pred), expected)
        summary = fourclass.summary(truth, pred)
        self.assertEqual(summary['per_class']['3']['f1'], 0.0)
        self.assertAlmostEqual(summary['per_class']['2']['f1'], 2 / 3)
        self.assertAlmostEqual(summary['category_step_mae'], 1 / 3)

    def test_counts_aggregate_before_f1(self):
        # Two months: pooled counts, not the mean of monthly macro F1.
        t1, p1 = np.array([0, 0]), np.array([0, 0])
        t2, p2 = np.array([1]), np.array([0])
        pooled = fourclass.macro_f1(np.r_[t1, t2], np.r_[p1, p2])
        mean = (fourclass.macro_f1(t1, p1) + fourclass.macro_f1(t2, p2)) / 2
        self.assertAlmostEqual(pooled, (0.8 + 0) / 4)
        self.assertNotAlmostEqual(pooled, mean)

    def test_get_prf_has_no_mean_fill(self):
        pre, rec, f1, _ = get_prf(np.array([1, 0, 0, 0]), np.array([2, 0, 1, 0]), np.array([1, 2, 0, 0]))
        np.testing.assert_allclose(f1, [2 / 3, 0, 0, 0])
        self.assertTrue(np.isnan(pre[2]) and np.isnan(rec[1]))

    def test_probability_axis_alignment_and_ties(self):
        aligned = fourclass.align_probabilities(np.array([[.5, .5], [.2, .8]]), [1, 3])
        np.testing.assert_allclose(aligned, [[0, .5, 0, .5], [0, .2, 0, .8]])
        np.testing.assert_array_equal(fourclass.argmax_codes(aligned), [1, 3])
        with self.assertRaises(ValueError):
            fourclass.align_probabilities(np.ones((1, 2)), [0, 1, 2])


class ScanStatistics(unittest.TestCase):
    def test_parent_normalized_masses_and_zero_columns(self):
        truth = np.array([0, 0, 1, 1, 2])
        pred = np.array([0, 1, 1, 1, 0])
        groups = np.array([10, 10, 20, 20, 30])
        gids, Y, _, A = opt.get_class_wise_stat(truth, pred, groups)
        np.testing.assert_array_equal(gids, [10, 20, 30])
        # D per class: c0 = 2+1(FN)+1(FP) = 4; c1 = 4+1(FP) = 5; c2 = 1(FN); c3 = 0.
        np.testing.assert_allclose(Y[:, 0], np.array([3, 0, 1]) / 16)
        np.testing.assert_allclose(Y[:, 1], np.array([1, 4, 0]) / 20)
        np.testing.assert_allclose(Y[:, 2], np.array([0, 0, 1]) / 4)
        np.testing.assert_array_equal(Y[:, 3], 0)
        np.testing.assert_allclose(A[:, 0], np.array([2, 0, 0]) / 16)
        np.testing.assert_allclose(A[:, 1], np.array([0, 4, 0]) / 20)
        np.testing.assert_array_equal(A[:, 3], 0)
        # Total C = (exposed classes)/4 - macro F1: zero-exposure columns carry no mass.
        self.assertAlmostEqual(float((Y - A).sum()), 3 / 4 - fourclass.macro_f1(truth, pred))
        self.assertEqual(Y.dtype, float)
        C, B = opt.get_c_b(Y, A)
        with redirect_stdout(StringIO()):
            s0, s1, g, q = opt.scan(Y, A, 0, return_score=True)
        self.assertTrue(np.isfinite(g).all() and np.isfinite(q).all())
        self.assertEqual(set(s0) | set(s1), {0, 1, 2})

    def test_zero_exposure_scan_is_finite(self):
        with redirect_stdout(StringIO()):
            for Y in (np.zeros((3, 4)), np.full((3, 4), .1)):
                s0, s1, g, q = opt.scan(Y, Y.copy(), 0, return_score=True)
                self.assertTrue(np.isfinite(g).all() and np.isfinite(q).all())


class SplitGate(unittest.TestCase):
    def test_exact_boundary_ties_and_strict_gain(self):
        # 49 phase-3 rows; parent predicts phase 1 for all; child gets one right:
        # class-3 F1 rises 0 -> 2/(2+0+48) = 1/25, so macro gain is exactly 1/100.
        e = np.array([], dtype=int)
        truth = np.full(49, 2)
        parent = np.zeros(49, dtype=int)
        child = parent.copy()
        child[0] = 2
        accepted, _, _, base, best, _ = opt.select_macro_children(truth, e, parent, e, child, e)
        self.assertEqual(best - base, Fraction(1, 100))
        self.assertFalse(accepted)  # exactly .01 is rejected
        child[1] = 2
        self.assertTrue(opt.select_macro_children(truth, e, parent, e, child, e)[0])
        same = np.zeros(4, dtype=int)
        self.assertFalse(opt.select_macro_children(same, e, same, e, same, e)[0])

    def test_mixed_parent_child_combination(self):
        y0, y1 = np.array([0, 1]), np.array([2, 2])
        p0, p1 = np.array([0, 1]), np.array([0, 0])
        c0, c1 = np.array([1, 1]), np.array([2, 2])
        accepted, selected, preds, *_ = opt.select_macro_children(y0, y1, p0, p1, c0, c1)
        self.assertTrue(accepted)
        self.assertEqual(selected, (False, True))
        np.testing.assert_array_equal(np.concatenate(preds), [0, 1, 2, 2])

    def test_partition_root_rejection_and_inherited_checkpoint_bundle(self):
        rows = np.column_stack((np.arange(4), [10, 10, 10, 20])).astype(float)
        X = np.vstack((rows, rows))
        y = np.tile([0, 1, 1, 2], 2)
        groups = X[:, 1].astype(int)
        split = np.repeat([0, 1], 4)

        def proposal(d, a, *args, **kwargs):
            return np.array([0]), np.arange(1, len(d)), np.zeros(len(d)), np.ones(1)
        for reject in (True, False):
            def labeller(ids, n):
                if n == 4:  # root: wrong on row 3
                    return np.array([0, 1, 1, 0])[ids]
                if n == 1:  # child for group 20
                    return np.full(n, 0 if reject else 2)
                return np.array([0, 1, 1, 2])[ids]
            with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()), \
                    patch.object(trans, 'CONTIGUITY', False), \
                    patch.object(trans, 'scan', side_effect=proposal), \
                    patch.object(trans, 'generate_count_grid', return_value=(None, 0, 1)):
                model = PresetRF(tmp, labeller)
                assigned, table, _ = trans.partition(
                    model, X, y, groups, split, np.arange(len(X)), np.full(len(X), '', dtype='<U8'),
                    min_depth=1, max_depth=2, contiguity_type='polygon', model_dir=tmp, VIS_DEBUG_MODE=False)
                decisions = trans.partition.decisions
                if reject:
                    self.assertTrue((assigned == '').all())
                    self.assertEqual(decisions[0]['outcome'], 'rejected_gate')
                    self.assertFalse(table[:, 1:].any())
                else:
                    self.assertEqual(set(assigned), {'0', '1'})
                    self.assertEqual(decisions[0]['selected_children'], [False, True])
                    # branch '0' inherited the root forest AND the root imputer.
                    model.load('')
                    root_imputer = model.imputer.digest()
                    model.load('0')
                    self.assertEqual(model.imputer.digest(), root_imputer)
                    self.assertEqual(model.fit_record['trained_under'], '')
                    np.testing.assert_array_equal(model.predict(rows[:3]), [0, 1, 1])
                    model.load('1')
                    np.testing.assert_array_equal(model.predict(rows[3:]), [2])

    def test_zero_error_parent_gives_no_candidate(self):
        rows = np.column_stack((np.arange(4), [10, 10, 20, 20])).astype(float)
        X = np.vstack((rows, rows))
        y = np.tile([0, 1, 2, 3], 2)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()), \
                patch.object(trans, 'CONTIGUITY', False), \
                patch.object(trans, 'generate_count_grid', return_value=(None, 0, 1)):
            model = PresetRF(tmp, lambda ids, n: np.array([0, 1, 2, 3])[ids])
            assigned, _, _ = trans.partition(
                model, X, y, X[:, 1].astype(int), np.repeat([0, 1], 4), np.arange(8),
                np.full(8, '', dtype='<U8'), min_depth=1, max_depth=2, contiguity_type='polygon',
                model_dir=tmp, VIS_DEBUG_MODE=False)
        self.assertTrue((assigned == '').all())
        self.assertEqual(trans.partition.decisions[0]['outcome'], 'no_candidate_zero_error_mass')

    def test_no_class1_path_remains(self):
        with self.assertRaises(ValueError):
            opt.get_score(np.array([0]), np.array([0]))
        self.assertFalse(hasattr(opt, 'select_f1_children'))
        self.assertFalse(hasattr(opt, 'get_class_1_f1_score'))


class Imputation(unittest.TestCase):
    def test_max_plus_rules_and_fit_only_on_given_rows(self):
        fit = np.array([[1.0, 0.0, np.nan, -2.0], [3.0, np.nan, np.nan, -5.0], [np.inf, 0.0, np.nan, np.nan]])
        imputer = MaxPlusImputer().fit(fit)
        np.testing.assert_allclose(imputer.fill_, [300.0, 100.0, 0.0, -200.0])
        out = imputer.transform(np.array([[np.nan, np.nan, np.nan, np.nan], [99.0, 7.0, 1.0, np.inf]]))
        np.testing.assert_allclose(out, [[300, 100, 0, -200], [99, 7, 1, -200]])
        np.testing.assert_array_equal(imputer.missing_in_fit_, [1, 1, 3, 1])

    def test_pseudo_rows_after_imputation_and_bundle_roundtrip(self):
        X = np.array([[1.0, np.nan], [2.0, 5.0], [3.0, 7.0], [4.0, np.nan]])
        y = np.array([0, 1, 1, 0])
        seen = []
        real_fit = RandomForestClassifier.fit

        def capture(model, rows, labels, *args, **kwargs):
            seen.append((rows.copy(), labels.copy()))
            return real_fit(model, rows, labels, *args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()), \
                patch.object(RandomForestClassifier, 'fit', new=capture):
            rf = RFmodel(tmp, 3, n_jobs=1)
            rf.train(X, y, '')
            rows, labels = seen[-1]
            np.testing.assert_allclose(rows[:4, 1], [700, 5, 7, 700])  # fill from real rows only
            np.testing.assert_array_equal(rows[4:], np.zeros((4, 2)))
            np.testing.assert_array_equal(labels[4:], [0, 1, 2, 3])
            self.assertEqual(rf.fit_record['real_rows'], 4)
            self.assertEqual(rf.fit_record['pseudo_rows'], 4)
            before = rf.predict(X, prob=True)
            self.assertEqual(before.shape, (4, 4))
            rf.save('')
            fresh = RFmodel(tmp, 3, n_jobs=1)
            fresh.load('')
            np.testing.assert_array_equal(fresh.predict(X, prob=True), before)
            self.assertEqual(fresh.imputer.digest(), rf.imputer.digest())
            with self.assertRaises(ValueError):
                RFmodel(tmp, 3, use_smote=True)


class HistoryFeatures(unittest.TestCase):
    def feats(self, obs, keys):
        observations = pd.DataFrame(obs, columns=['area', 'month', 'phase'])
        areas = [a for a, _ in keys]
        origins = [o for _, o in keys]
        return ff.history_features(observations, areas, origins)

    def test_exact_offsets_windows_events_and_run(self):
        # Area 1 phases at Feb/Jun/Oct cadence; origin 2020-06.
        O = m('2020-06')
        obs = [(1, m('2019-02'), 1), (1, m('2019-06'), 3), (1, m('2019-10'), 3),
               (1, m('2020-02'), 2), (1, m('2020-06'), 4), (1, m('2020-10'), 1),  # future
               (2, m('2020-06'), 1)]
        f = self.feats(obs, [(1, O), (3, O)]).iloc[0]
        self.assertEqual(f['hist_phase_o00'], 4)
        self.assertEqual(f['hist_crisis_o00'], 1)
        self.assertEqual(f['hist_phase_o04'], 2)
        self.assertEqual(f['hist_phase_o08'], 3)
        self.assertEqual(f['hist_phase_o12'], 3)
        self.assertEqual(f['hist_delta_o00_o04'], 2)
        self.assertEqual(f['hist_direction_o04_o08'], -1)
        self.assertEqual(f['hist_delta_o00_o12'], 1)
        # W12 = [O-11, O]: 2019-10, 2020-02, 2020-06 -> phases 3, 2, 4.
        self.assertEqual(f['hist_w12_n_obs'], 3)
        self.assertEqual(f['hist_w12_n_pairs'], 2)
        self.assertAlmostEqual(f['hist_w12_frac3'], 1 / 3)
        self.assertEqual(f['hist_w12_frac1'], 0)
        self.assertEqual(f['hist_w12_min_phase'], 2)
        self.assertEqual(f['hist_w12_max_phase'], 4)
        self.assertEqual(f['hist_w12_up_rate'], 0.5)
        self.assertEqual(f['hist_w12_down_rate'], 0.5)
        self.assertEqual(f['hist_w12_span_months'], 8)
        self.assertEqual(f['hist_w12_latest_age'], 0)
        self.assertEqual(f['hist_w24_n_obs'], 5)
        self.assertEqual(f['hist_latest_observed_phase'], 4)
        self.assertEqual(f['hist_latest_observed_age'], 0)
        self.assertEqual(f['hist_crisis_age'], 0)
        self.assertEqual(f['hist_phase4or5_age'], 0)
        self.assertEqual(f['hist_change_age'], 0)
        self.assertEqual(f['hist_change_absent'], 0)
        self.assertEqual(f['hist_run_n'], 1)
        self.assertEqual(f['hist_run_span'], 0)
        self.assertEqual(f['hist_origin_phase4_x_w12_up_rate'], 0.5)
        self.assertEqual(f['hist_origin_phase1_x_w12_frac4or5'], 0)
        # Area 3 has no history; area 2's observation does not leak across areas.
        g = self.feats(obs, [(1, O), (3, O)]).iloc[1]
        self.assertEqual(g['hist_no_history'], 1)
        self.assertEqual(g['hist_w36_n_obs'], 0)
        self.assertTrue(np.isnan(g['hist_w36_frac1']))
        self.assertTrue(np.isnan(g['hist_run_n']))
        self.assertEqual(g['hist_crisis_absent'], 1)
        self.assertTrue(np.isnan(g['hist_crisis_age']))
        self.assertTrue(np.isnan(g['hist_origin_phase1_x_w12_up_rate']))

    def test_sparse_gap_missing_origin_and_window_edge(self):
        O = m('2021-06')
        obs = [(1, m('2020-06'), 2), (1, m('2020-07'), 2), (1, m('2021-02'), 2)]
        f = self.feats(obs, [(1, O)]).iloc[0]
        self.assertTrue(np.isnan(f['hist_phase_o00']))          # no exact origin record
        self.assertTrue(np.isnan(f['hist_crisis_o00']))
        self.assertTrue(np.isnan(f['hist_delta_o00_o04']))
        self.assertEqual(f['hist_phase_o04'], 2)
        self.assertEqual(f['hist_w12_n_obs'], 2)                # 2020-06 is O-12: outside [O-11, O]
        self.assertEqual(f['hist_run_n'], 3)
        self.assertEqual(f['hist_run_span'], 8)
        self.assertEqual(f['hist_change_absent'], 1)
        self.assertEqual(f['hist_latest_observed_age'], 4)
        self.assertTrue(np.isnan(f['hist_origin_phase2_x_w12_up_rate']))
        one = self.feats([(1, O, 3)], [(1, O)]).iloc[0]
        self.assertEqual(one['hist_w12_span_months'], 0)
        self.assertTrue(np.isnan(one['hist_w12_up_rate']))
        self.assertEqual(one['hist_w12_n_pairs'], 0)
        self.assertEqual(one['hist_run_n'], 1)

    def test_covariate_offsets_sums_and_area_isolation(self):
        months = pd.date_range('2019-01-01', '2020-12-01', freq='MS')
        panel = pd.DataFrame([(a, d) for a in (1, 2) for d in months], columns=['FEWSNET_admin_code', 'date'])
        idx = np.arange(len(months))
        panel['EVI'] = np.r_[idx, 1000 + idx].astype(float)
        panel['WFP_Price'] = np.r_[np.ones(len(months)), np.full(len(months), 5.0)]
        panel.loc[(panel.FEWSNET_admin_code == 1) & (panel.date == '2020-03-01'), 'WFP_Price'] = np.nan
        panel['nightlight'] = 1.0
        schema = {'static_sources': [], 'dynamic_sources_at_origin': ['EVI', 'WFP_Price', 'nightlight']}
        scaffold = ff.Scaffold(panel, ['EVI', 'WFP_Price', 'nightlight'])
        O = m('2020-06')
        out = ff.covariate_features(scaffold, schema, [1, 2, 1], [O + 4] * 3, [O, O, m('2019-03')])
        self.assertEqual(out.loc[0, 'EVI'], 17)                 # exact origin value
        self.assertEqual(out.loc[0, 'EVI_l1'], 16)
        self.assertEqual(out.loc[0, 'EVI_l12'], 5)
        self.assertEqual(out.loc[1, 'EVI_l12'], 1005)           # area 2's own series
        self.assertTrue(np.isnan(out.loc[0, 'WFP_Price_m4']))   # 2020-03 missing inside [O-4, O-1]
        self.assertEqual(out.loc[1, 'WFP_Price_m4'], 20)
        self.assertTrue(np.isnan(out.loc[2, 'EVI_l12']))        # before the scaffold
        self.assertTrue(np.isnan(out.loc[2, 'nightlight_m12']))
        self.assertAlmostEqual(out.loc[0, 'target_month_sin'], np.sin(2 * np.pi * 9 / 12))
        self.assertEqual(out.loc[0, 'target_year'], 2020)

    def test_schema_order_and_exclusions(self):
        schema = ff.load_schema(SCHEMA)
        names = schema['ordered_features']
        self.assertEqual(len(names), 162)
        for banned in ('FEWSNET_admin_code', 'fews_ipc', 'fews_ipc_crisis', 'fews_proj_near', 'fews_proj_med'):
            self.assertNotIn(banned, names)


class Stage3Routing(unittest.TestCase):
    def snapshot(self, n_areas=6, cluster_rows=None):
        rng = np.random.default_rng(0)
        schema = ff.load_schema(SCHEMA)
        features = schema['ordered_features']
        rows = []
        for area in range(n_areas):
            for month in range(m('2018-01'), m('2021-07')):
                rows.append((area, month))
        frame = pd.DataFrame(rows, columns=['area', 'target_month'])
        frame['horizon'] = 4
        frame['class_code'] = (frame['area'] + frame['target_month']) % 3
        X = rng.normal(size=(len(frame), len(features)))
        X[:, 0] = frame['class_code']
        X[rng.random(X.shape) < .05] = np.nan
        return pd.concat([frame, pd.DataFrame(X, columns=features)], axis=1), features

    def test_null_consensus_reuses_pooled_exactly(self):
        snap, features = self.snapshot()
        with patch.dict(compare.RF_PARAMS, {'n_estimators': 5}):
            preds, record, _ = compare.fit_fold(snap, features, pd.Period('2021-06', 'M'), 4, None)
        np.testing.assert_array_equal(preds['y_pred_pooled_code'], preds['y_pred_partitioned_code'])
        self.assertEqual(record['fits'], {'pooled': 1, 'local': 0, 'partitioned_arm_reuses_pooled': True})
        self.assertEqual(record['train_label_months'], ['2018-03', '2021-01'])
        self.assertEqual(record['pseudo_rows'], 0)

    def test_local_fallbacks_and_class_axis(self):
        snap, features = self.snapshot(n_areas=8)
        # 35 training rows per area: clusters 0/1/2 have 70 rows; cluster 3 has 35 (<50).
        clusters = {0: 0, 1: 0, 2: 1, 6: 1, 3: 2, 4: 2, 7: 3}  # area 5 unmapped
        snap.loc[snap['area'].isin([2, 6]), 'class_code'] = 1  # cluster 1 single-class
        with patch.dict(compare.RF_PARAMS, {'n_estimators': 5}):
            preds, record, extras = compare.fit_fold(snap, features, pd.Period('2021-06', 'M'), 4, clusters)
        routes = dict(zip(preds['area'], preds['partitioned_route']))
        self.assertEqual(routes[0], 'local_model')
        self.assertEqual(routes[2], 'pooled_fallback:single_class_local')
        self.assertEqual(routes[5], 'unmapped_area_pooled')
        self.assertEqual(routes[7], 'pooled_fallback:local_rows_below_50')
        pooled_by_area = dict(zip(preds['area'], preds['y_pred_pooled_code']))
        self.assertEqual(dict(zip(preds['area'], preds['y_pred_partitioned_code']))[5], pooled_by_area[5])
        local = RandomForestClassifier(**{**compare.RF_PARAMS, 'n_estimators': 5})
        self.assertEqual(sorted(extras['imputer_fills'].columns), ['local_0', 'local_2', 'pooled'])
        self.assertEqual(preds.filter(like='p_partitioned_').shape[1], 4)
        np.testing.assert_allclose(preds.filter(like='p_pooled_').sum(axis=1), 1)

    def test_saved_bundles_reproduce_probabilities(self):
        snap, features = self.snapshot(n_areas=8)
        clusters = {0: 0, 1: 0, 2: 1, 6: 1, 3: 2, 4: 2, 7: 3}
        with patch.dict(compare.RF_PARAMS, {'n_estimators': 5}):
            preds, record, extras = compare.fit_fold(snap, features, pd.Period('2021-06', 'M'), 4, clusters)
        self.assertEqual(sorted(extras['bundles']), ['local_0', 'local_1', 'local_2', 'pooled'])
        with tempfile.TemporaryDirectory() as tmp:
            for name, bundle in extras['bundles'].items():
                compare.save_bundle(Path(tmp) / f'{name}.pkl.xz', bundle)
            loaded = {n: compare.load_bundle(Path(tmp) / f'{n}.pkl.xz') for n in extras['bundles']}
        test = snap[snap.target_month == m('2021-06')]
        X = test[features].to_numpy(dtype=float)
        p = compare.bundle_proba(loaded['pooled'], X)
        np.testing.assert_array_equal(p, preds.filter(like='p_pooled_').to_numpy())
        with tempfile.TemporaryDirectory() as tmp:  # saved CSV round-trips exactly
            path = Path(tmp) / 'p.csv.gz'
            preds.to_csv(path, index=False, float_format='%.17g')
            back = pd.read_csv(path, float_precision='round_trip')
            np.testing.assert_array_equal(back.filter(like='p_pooled_').to_numpy(), p)
        for _ in range(3):  # single-threaded prediction is bit-stable across calls
            np.testing.assert_array_equal(compare.bundle_proba(loaded['pooled'], X), p)
        rows = (preds['cluster_id'] == 0).to_numpy()
        np.testing.assert_array_equal(compare.bundle_proba(loaded['local_0'], X[rows]),
                                      preds.loc[rows].filter(like='p_partitioned_').to_numpy())
        self.assertEqual(loaded['pooled']['identity']['train_keys_sha256'], record['train_keys_sha256'])
        self.assertEqual(loaded['local_0']['identity']['pool_train_keys_sha256'], record['train_keys_sha256'])
        train = snap[(snap.target_month >= m('2021-06') - 4 - 35) & (snap.target_month < m('2021-06') - 4)]
        local_keys = train[train['area'].map(clusters).eq(0)][['area', 'target_month']].to_numpy(np.int64)
        self.assertEqual(loaded['local_0']['identity']['train_keys_sha256'],
                         compare.sha256_bytes(np.ascontiguousarray(local_keys).tobytes()))
        self.assertEqual(loaded['local_0']['identity']['rows'], len(local_keys))
        self.assertEqual(loaded['pooled']['features'], features)

    def test_local_fit_uses_its_own_imputer(self):
        X = np.array([[1.0], [np.nan], [2.0], [np.nan]])
        y = np.array([0, 1, 0, 1])
        with patch.dict(compare.RF_PARAMS, {'n_estimators': 3}):
            keys = np.column_stack([np.arange(4), np.full(4, 100)])
            a = compare.FittedRF(X[:2], y[:2], keys[:2], n_jobs=1)
            b = compare.FittedRF(X, y, keys, n_jobs=1)
        self.assertNotEqual(a.train_keys_sha256, b.train_keys_sha256)
        self.assertEqual(a.imputer.fill_[0], 100.0)
        self.assertEqual(b.imputer.fill_[0], 200.0)

    def test_empty_training_pool_is_an_error(self):
        snap, features = self.snapshot()
        snap = snap[snap['target_month'] >= m('2021-06')]
        with self.assertRaises(RuntimeError):
            compare.fit_fold(snap, features, pd.Period('2021-06', 'M'), 4, None)


def make_prepared_identity(root):
    prepared = root / 'prepared'
    (prepared / 'manifests').mkdir(parents=True, exist_ok=True)
    rid.write_json_atomic(prepared / 'manifests' / 'outputs.json', rid.output_hashes(prepared))
    rid.write_json_atomic(prepared / 'manifests' / 'identity.json', {
        'stage': 'prepare', 'code': rid.code_identity(), 'runtime': rid.runtime_identity(),
        'outputs_sha256': rid.file_sha256(prepared / 'manifests' / 'outputs.json')})


class CommittedCode(unittest.TestCase):
    def test_schema_is_committed_and_identity_matches_git(self):
        import subprocess
        repo = Path(__file__).resolve().parents[2]
        listed = subprocess.run(['git', 'ls-files', 'FEWSNETFourClassBaseline/feature-schema.json'],
                                cwd=repo, capture_output=True, text=True).stdout.strip()
        self.assertEqual(listed, 'FEWSNETFourClassBaseline/feature-schema.json')
        self.assertEqual(rid.file_sha256(prep.SCHEMA_PATH), prep.APPROVED_SCHEMA_SHA256)
        head = rid.code_identity_at('HEAD')
        self.assertEqual(head['files'], rid.code_identity()['files'])
        self.assertNotIn('scripts/verify_fourclass.py', json.dumps(head))
        self.assertEqual(set(rid.verifier_identity()), {'scripts/verify_fourclass.py'})


class Continuation(unittest.TestCase):
    """Audit A02: no fold is accepted or skipped on a completion filename alone."""

    def completed_run(self, tmp, retain=False):
        run = Path(tmp) / 'run'
        (run / 'prepared' / 'manifests').mkdir(parents=True)
        (run / 'prepared' / 'snapshot.bin').write_bytes(b'snapshot')
        for rel in rid.REQUIRED_PREPARED:
            (run / 'prepared' / rel).parent.mkdir(parents=True, exist_ok=True)
            (run / 'prepared' / rel).write_text(rel)
        make_prepared_identity(run)
        prepared = rid.require_prepared(run)
        name, term = 'fs1_2018-02', '2018-02'
        paths = stage1.handoff_paths(run, 1, term)
        paths['archive'].mkdir(parents=True)
        (paths['archive'] / 'correspondence_table_2018-02.csv').write_text('a')
        paths['metrics'].write_text('m')
        paths['predictions'].write_text('p')
        fold = run / 'stage1' / 'folds' / name
        fold.mkdir(parents=True)
        for rel in rid.REQUIRED_STAGE1_FOLD:
            (fold / rel).write_text('x')
        (fold / 'candidate.json').write_text(json.dumps({'scope': 1, 'target_month': term}))
        if retain:
            for rel in ('checkpoints/rf_', 'space_partitions/s_branch.pkl'):
                (run / 'stage1' / 'retained' / name / rel).parent.mkdir(parents=True, exist_ok=True)
                (run / 'stage1' / 'retained' / name / rel).write_bytes(b'model')
        rid.write_json_atomic(fold / 'completion.json', {
            'fold': name, 'prepared': prepared['outputs_sha256'], 'code': rid.code_identity(),
            'runtime': rid.runtime_identity(), 'retain': retain,
            'outputs': stage1.fold_outputs(run, name, 1, term, retain)})
        return run, name, prepared

    def test_forged_marker_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            (run / 'stage1' / 'folds' / 'fs1_2018-02').mkdir(parents=True)
            (run / 'stage1' / 'folds' / 'fs1_2018-02' / 'candidate.json').write_text('{}')
            fold = {'scope': 1, 'target_month': '2018-02'}
            with self.assertRaises(FileExistsError):
                stage1.run_fold(run, fold, 'NO_EXECUTABLE', True, {'outputs_sha256': 'x'})
            with self.assertRaises(RuntimeError):
                stage1.verify_fold(run, 'fs1_2018-02', {'outputs_sha256': 'x'})
            with self.assertRaises(RuntimeError):
                rid.require_prepared(run)

    def test_complete_fold_verifies_and_every_breach_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, name, prepared = self.completed_run(tmp, retain=True)
            stage1.verify_fold(run, name, prepared)
            with self.assertRaises(FileExistsError):  # never continued, even when complete
                stage1.run_fold(run, {'scope': 1, 'target_month': '2018-02'}, 'X', True, prepared)
            with self.assertRaises(RuntimeError):     # different preparation identity
                stage1.verify_fold(run, name, {**prepared, 'outputs_sha256': 'other'})
            (run / 'stage1' / 'retained' / name / 'checkpoints' / 'rf_').unlink()
            with self.assertRaises(RuntimeError):     # missing retained checkpoint
                stage1.verify_fold(run, name, prepared)
        with tempfile.TemporaryDirectory() as tmp:
            run, name, prepared = self.completed_run(tmp)
            stage1.handoff_paths(run, 1, '2018-02')['metrics'].write_text('changed')
            with self.assertRaises(RuntimeError):     # changed handoff output
                stage1.verify_fold(run, name, prepared)
        with tempfile.TemporaryDirectory() as tmp:
            run, name, prepared = self.completed_run(tmp)
            (run / 'prepared' / 'snapshot.bin').write_bytes(b'other input')
            with self.assertRaises(RuntimeError):     # changed prepared input
                rid.require_prepared(run)
        with tempfile.TemporaryDirectory() as tmp:
            run, name, prepared = self.completed_run(tmp)
            record = json.loads((run / 'stage1' / 'folds' / name / 'completion.json').read_text())
            record['code'] = {'sha256': 'older', 'files': 1}
            rid.write_json_atomic(run / 'stage1' / 'folds' / name / 'completion.json', record)
            with self.assertRaises(RuntimeError):     # different package code
                stage1.verify_fold(run, name, prepared)

    def test_identity_matching_record_with_empty_or_partial_inventory_is_refused(self):
        for mutate in ('empty', 'drop_required', 'wrong_fold', 'retain_mismatch', 'no_inventory'):
            with tempfile.TemporaryDirectory() as tmp:
                run, name, prepared = self.completed_run(tmp, retain=True)
                path = run / 'stage1' / 'folds' / name / 'completion.json'
                record = json.loads(path.read_text())
                if mutate == 'empty':
                    record['outputs'] = {}
                elif mutate == 'drop_required':
                    record['outputs'].pop(f'folds/{name}/candidate.json')
                elif mutate == 'wrong_fold':
                    record['fold'] = 'fs1_2018-06'
                elif mutate == 'no_inventory':
                    record.pop('outputs')
                rid.write_json_atomic(path, record)
                with self.assertRaises(RuntimeError, msg=mutate):
                    stage1.verify_fold(run, name, prepared,
                                       retain_expected=False if mutate == 'retain_mismatch' else None)

    def test_prepared_record_needs_required_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            (run / 'prepared' / 'manifests').mkdir(parents=True)
            make_prepared_identity(run)  # identity matches, but no required outputs exist
            with self.assertRaises(RuntimeError):
                rid.require_prepared(run)

    def test_stage2_and_stage3_refuse_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'stage2').mkdir()
            with self.assertRaises(FileExistsError):
                rid.refuse_existing(Path(tmp) / 'stage2', 'Stage 2')


class Stage2AndReporting(unittest.TestCase):
    def test_macro_logit_weights_and_all_zero(self):
        frame = pd.DataFrame({'macro_f1': [.8, .4, .5], 'macro_f1_base': [.5, .5, .5]})
        np.testing.assert_allclose(compute_plan_weights(frame)['weight'], [np.log(4), 0, 0])
        zero = pd.DataFrame({'macro_f1': [.3, 1.0], 'macro_f1_base': [.5, 1.0]})
        self.assertEqual(float(compute_plan_weights(zero)['weight'].sum()), 0.0)
        with self.assertRaises(ValueError):
            compute_plan_weights(pd.DataFrame({'macro_f1': [np.nan], 'macro_f1_base': [.5]}))

    def test_step1_reads_explicit_macro_columns(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            out = Path(tmp)
            pd.DataFrame({'year': [2018], 'month': [2], 'macro_f1': [.6], 'macro_f1_base': [.5]}).to_csv(
                out / 'results_df_gp_fs1_2018_2018_m02.csv', index=False)
            table = load_results_table(out, MODEL_CONFIG['georf'])
            self.assertEqual(list(table.columns[-2:]), ['macro_f1', 'macro_f1_base'])

    def test_stage2_missing_candidate_is_not_null_consensus(self):
        from scripts import run_stage2
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'prepared' / 'manifests').mkdir(parents=True)
            (run / 'prepared' / 'manifests' / 'schedule.json').write_text(json.dumps({'stage1': [
                {'scope': 1, 'target_month': '2018-02', 'status': 'scheduled'},
                {'scope': 1, 'target_month': '2018-03', 'status': 'skipped_empty_target'}]}))
            with self.assertRaises(RuntimeError):
                run_stage2.candidate_ledger(run)

    def synthetic_run(self, root, weights_positive):
        areas = np.arange(60)
        (root / 'prepared' / 'manifests').mkdir(parents=True)
        (root / 'prepared' / 'geometry').mkdir(parents=True)
        pd.DataFrame({'FEWSNET_admin_code': areas, 'lat': areas // 10 * 1.0, 'lon': areas % 10 * 1.0}).to_csv(
            root / 'prepared' / 'geometry' / 'FEWSNET_admin_code_lat_lon.csv', index=False)
        folds = []
        for i, term in enumerate(('2018-02', '2018-06')):
            name = f'fs1_{term}'
            folds.append({'scope': 1, 'target_month': term, 'status': 'scheduled'})
            part = np.where(areas < 30 + 10 * i, '0', '1')
            corr = pd.DataFrame({'FEWSNET_admin_code': areas, 'partition_id': part})
            fold = root / 'stage1' / 'folds' / name
            fold.mkdir(parents=True)
            corr.to_csv(fold / 'correspondence_table.csv', index=False)
            f1 = .7 if weights_positive else .5
            (fold / 'candidate.json').write_text(json.dumps({
                'scope': 1, 'target_month': term,
                'scores': {'macro_f1': f1, 'macro_f1_base': .5}, 'partition': {'n_terminal': 2},
                'rows': {'heldout_target': 60}}))
            results = root / 'stage1' / 'GeoRFResults'
            archive = results / f'result_GeoRF_2018_fs1_{term}_visual'
            archive.mkdir(parents=True)
            corr.to_csv(archive / f'correspondence_table_{term}.csv', index=False)
            pd.DataFrame({'year': [2018], 'month': [int(term[-2:])], 'macro_f1': [f1], 'macro_f1_base': [.5]}).to_csv(
                results / f'results_df_gp_fs1_2018_2018_m{term[-2:]}.csv', index=False)
            (results / f'y_pred_test_gp_fs1_2018_2018_m{term[-2:]}.csv').write_text('y')
        (root / 'prepared' / 'manifests' / 'schedule.json').write_text(json.dumps({'stage1': folds}))
        for rel in rid.REQUIRED_PREPARED:
            if not (root / 'prepared' / rel).exists():
                (root / 'prepared' / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / 'prepared' / rel).write_text(rel)
        rid.write_json_atomic(root / 'stage1' / 'retain_plan.json', [])
        for fold in folds:
            d = root / 'stage1' / 'folds' / f"fs1_{fold['target_month']}"
            for rel in rid.REQUIRED_STAGE1_FOLD:
                if not (d / rel).exists():
                    (d / rel).write_text('x')
        make_prepared_identity(root)
        prepared = rid.require_prepared(root)
        for fold in folds:
            name = f"fs1_{fold['target_month']}"
            rid.write_json_atomic(root / 'stage1' / 'folds' / name / 'completion.json', {
                'fold': name, 'prepared': prepared['outputs_sha256'], 'code': rid.code_identity(),
                'runtime': rid.runtime_identity(), 'retain': False,
                'outputs': stage1.fold_outputs(root, name, 1, fold['target_month'], False)})

    def test_stage2_synthetic_integration_and_null_route(self):
        import subprocess
        for positive in (True, False):
            with tempfile.TemporaryDirectory() as tmp:
                run = Path(tmp) / 'run'
                self.synthetic_run(run, positive)
                result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve().parents[1] / 'scripts' / 'run_stage2.py'),
                                         '--run-dir', str(run)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr[-2000:])
                record = json.loads((run / 'stage2' / 'consensus.json').read_text())
                if positive:
                    self.assertEqual(record['route'], 'learned_map')
                    mapping = pd.read_csv(record['cluster_map'])
                    self.assertEqual(sorted(mapping['FEWSNET_admin_code']), list(range(60)))
                    self.assertEqual(record['actual_clusters'], record['recommended_clusters'])
                    consensus, clusters = compare.load_consensus(run / 'stage2' / 'consensus.json')
                    self.assertEqual(len(clusters), 60)
                else:
                    self.assertEqual(record['route'], 'null_consensus')
                    self.assertFalse((run / 'stage2' / 'experiment').exists())
                    self.assertEqual(compare.load_consensus(run / 'stage2' / 'consensus.json')[1], None)

    def test_country_bootstrap_matches_direct_recount(self):
        frame = pd.DataFrame({
            'country': ['A', 'A', 'B', 'C'], 'truth_code': [0, 1, 2, 2],
            'y_pred_partitioned_code': [0, 1, 2, 0], 'y_pred_pooled_code': [0, 0, 2, 2],
            'persistence_code': [0.0, 1.0, 1.0, 2.0]})
        arms = ('partitioned', 'pooled', 'persistence')
        mats = report.country_matrices(frame, arms, ['A', 'B', 'C'])
        mult = np.array([2, 0, 1])
        got = report.macro_from_matrices(np.tensordot(mult.astype(float), mats, axes=1))
        rep = pd.concat([frame[frame.country == 'A']] * 2 + [frame[frame.country == 'C']])
        for a, arm in enumerate(arms):
            self.assertAlmostEqual(got[a], fourclass.macro_f1(rep['truth_code'],
                                                              rep[report.ARM_COLUMN[arm]].astype(int)))

    def test_cohorts_share_keys_and_exclude_missing_baselines(self):
        base = pd.DataFrame({'horizon': [4, 4, 4, 12, 12], 'persistence_code': [0, np.nan, 1, 1, np.nan],
                             'expert_code': [0, 1, np.nan, np.nan, np.nan]})
        c = report.cohorts(base)
        self.assertEqual(len(c['main_h4'][0]), 1)
        self.assertEqual(len(c['supp_h4'][0]), 2)
        self.assertEqual(len(c['main_h12'][0]), 1)
        self.assertEqual(c['main_h12'][1], ('partitioned', 'pooled', 'persistence'))

    def test_baseline_calendar_join_and_no_backfill(self):
        panel = pd.DataFrame({'FEWSNET_admin_code': [1] * 3, 'date': pd.to_datetime(['2021-02-01', '2021-06-01', '2021-10-01']),
                              'fews_proj_near': [5.0, np.nan, 2.0], 'fews_proj_med': [3.0, 1.0, 1.0]})
        obs = pd.DataFrame({'area': [1, 1, 1], 'month': [m('2021-02'), m('2021-06'), m('2021-10')],
                            'country': 'X', 'raw_phase': [5.0, 2.0, 3.0], 'class_code': [3, 1, 2]})
        with patch.dict(prep.STAGE3_TARGETS, {4: ('2021-06', '2021-10'), 8: ('2021-10', '2021-10'), 12: ('2022-01', '2022-01')}):
            out = prep.build_baselines(panel, obs)
        h4 = out[out.horizon == 4].set_index('target_label')
        self.assertEqual(h4.loc['2021-06', 'persistence_code'], 3)   # raw 5 -> merged 4或5
        self.assertEqual(h4.loc['2021-06', 'expert_code'], 3)        # near(2021-02)=5 -> 4或5
        self.assertTrue(np.isnan(h4.loc['2021-10', 'expert_code']))  # near(2021-06) missing stays missing
        h8 = out[out.horizon == 8].set_index('target_label')
        self.assertEqual(h8.loc['2021-10', 'expert_code'], 2)        # med(2021-02)=3
        self.assertEqual(h8.loc['2021-10', 'persistence_code'], 3)
        self.assertEqual(len(out[out.horizon == 12]), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
