"""Focused contract checks for the GeoXGBoost fork: python -B tests/test_baseline.py.

Inherited fixed-four metric, scan, gate, feature, weight, bootstrap and calendar tests
are kept; the RF/imputer/pseudo-row tests are replaced by tests that drive the new
production paths (native continuation, partition(), Stage 3 gate, consensus routes,
time boundaries) with hand-checkable fixtures.
"""
from contextlib import redirect_stdout
from fractions import Fraction
from io import StringIO
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score

from src.experiment import plan
from src.experiment import stage3 as s3
from src.metrics import fourclass
from src.metrics.metrics import get_prf
from src.model import native_xgb as nx
from src.partition import partition_opt as opt
from src.partition import transformation as trans
from src.feature import fourclass_features as ff
from scripts import report_fourclass as report
from scripts import prepare_fourclass as prep
from scripts import run_stage2 as stage2
from scripts import run_experiment as rexp
from scripts import verify_fourclass as vf
from scripts.step4_similarity_matrix import compute_plan_weights
from scripts.step1_merge_results import MODEL_CONFIG, load_results_table
from src.utils import run_identity as rid

SCHEMA = prep.SCHEMA_PATH
SMALL_G = {g: {**c, "rounds": 6} for g, c in plan.G_CONFIGS.items()}


def m(label):
    return prep.mi(label)


def synthetic(n=1200, p=8, seed=0, classes=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[rng.random(X.shape) < .2] = np.nan
    X[:, 1] = 0.0  # true zeros stay zeros
    z = np.nan_to_num(X[:, 0]) + .5 * np.nan_to_num(X[:, 2])
    y = np.digitize(z, [-0.6, 0.3, 1.2])[:n] if classes == 4 else (z > 0).astype(int)
    return X, y.astype(int)


def tree_prefix(booster, n_trees):
    trees = json.loads(bytes(booster.save_raw('json')))['learner']['gradient_booster']['model']
    return [{k: v for k, v in t.items() if k != 'id'} for t in trees['trees'][:n_trees]], trees['tree_info'][:n_trees]


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
        gids, Y, _, A = opt.get_class_wise_stat(truth, pred, groups, endpoint='macro_f1_fourclass')
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

    def test_no_class1_path_remains(self):
        with self.assertRaises(ValueError):
            opt.get_score(np.array([0]), np.array([0]))
        self.assertFalse(hasattr(opt, 'select_f1_children'))
        self.assertFalse(hasattr(opt, 'get_class_1_f1_score'))



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

    def test_country_bootstrap_matches_direct_recount(self):
        frame = pd.DataFrame({
            'country': ['A', 'A', 'B', 'C'], 'truth_code': [0, 1, 2, 2],
            'y_pred_xgbmap_shared': [0, 1, 2, 0], 'y_pred_pooled': [0, 0, 2, 2],
            'persistence_code': [0.0, 1.0, 1.0, 2.0]})
        arms = ('partitioned', 'pooled', 'persistence')
        mats = report.country_matrices(frame, arms, ['A', 'B', 'C'])
        mult = np.array([2, 0, 1])
        got = report.macro_from_matrices(np.tensordot(mult.astype(float), mats, axes=1))
        rep = pd.concat([frame[frame.country == 'A']] * 2 + [frame[frame.country == 'C']])
        for a, arm in enumerate(arms):
            self.assertAlmostEqual(got[a], fourclass.macro_f1(rep['truth_code'],
                                                              rep[report.ARM_COLUMN[arm]].astype(int)))

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




class NativeContinuation(unittest.TestCase):
    def test_frozen_prefix_missing_class_reload_and_record(self):
        X, y = synthetic()
        g, grec = nx.fit_global(X, y, SMALL_G['G1'])
        before = nx.raw(g)
        sub = y != 3  # local subset without the rare class
        child, rec = nx.continue_booster(g, X[sub], y[sub], plan.L_CONFIGS['L1'])
        n0 = g.num_boosted_rounds()
        self.assertEqual(nx.raw(g), before)                       # parent untouched
        self.assertEqual(child.num_boosted_rounds(), n0 + 20)
        self.assertEqual(tree_prefix(child, 4 * n0), tree_prefix(g, 4 * n0))  # structure/leaves/default_left
        self.assertEqual(nx.base_score(child), nx.base_score(g))
        self.assertEqual(rec['child_prefix_structure_sha256'], grec['structure_sha256'])
        self.assertEqual(rec['parent_sha256'], grec['booster_sha256'])
        self.assertEqual(rec['class_counts'][3], 0)
        p = nx.proba(child, X)
        self.assertEqual(p.shape, (len(X), 4))
        self.assertTrue((p[:, 3] > 0).all())                    # class axis kept, no zero-filling
        np.testing.assert_allclose(p.sum(axis=1), 1, rtol=1e-6)
        self.assertEqual(rec['resolved_config']['learner']['learner_model_param']['num_class'], '4')
        with tempfile.TemporaryDirectory() as tmp:               # UBJ replay is exact
            path = Path(tmp) / 'c.ubj'
            path.write_bytes(nx.raw(child))
            np.testing.assert_array_equal(nx.proba(nx.from_raw(path.read_bytes()), X), p)
        self.assertEqual(nx.prefix_identity(nx.from_raw(nx.raw(child)), n0)['sha256'], grec['structure_sha256'])

    def test_refresh_or_base_score_override_is_refused_and_digests_differ(self):
        X, y = synthetic()
        g, _ = nx.fit_global(X, y, SMALL_G['G1'])
        for bad in ({'process_type': 'update', 'updater': 'refresh'}, {'base_score': .5}):
            with self.assertRaises(ValueError):
                nx.continue_booster(g, X, y, {**plan.L_CONFIGS['L1'], **bad})
        other, _ = nx.fit_global(X[::-1], y[::-1], SMALL_G['G1'])
        self.assertNotEqual(nx.prefix_identity(other)['sha256'], nx.prefix_identity(g)['sha256'])

    def test_inf_becomes_nan_and_true_zero_is_kept(self):
        X = np.array([[0.0, np.inf, -np.inf, np.nan, 2.0]])
        c = nx.clean(X)
        self.assertEqual(c[0, 0], 0.0)
        self.assertTrue(np.isnan(c[0, 1:4]).all())
        self.assertEqual(c[0, 4], 2.0)
        self.assertTrue(np.isinf(X[0, 1]))                       # input not modified
        with self.assertRaises(ValueError):
            nx.dmatrix(np.zeros((1, 2)), [4])


class PresetModel(nx.XGBmodel):
    """partition()'s checkpoint protocol with deterministic labels from column 0 (row id)."""

    def __init__(self, path, labeller, rounds=20):
        super().__init__(path, {**plan.L_CONFIGS['L1'], 'rounds': rounds})
        self.labeller = labeller
        self.trained = []
        self.labels = None

    def set_root(self, n):
        self.labels = ('root', n)
        self.fit_record = {'trained_under': None, 'path_rounds_added': 0}
        self.fit_log.append(dict(self.fit_record))
        self.save('')

    def train(self, X, y, branch_id, meta=None):
        if self.fit_record.get('loaded_as') != branch_id:
            raise RuntimeError('train without its parent loaded')
        ids = X[:, 0].astype(int)
        self.trained.append((branch_id, sorted(ids.tolist())))
        self.labels = ('child', tuple(sorted(ids.tolist())))
        self.fit_record = {'trained_under': branch_id, 'kind': 'continuation',
                           'path_rounds_added': self.fit_record['path_rounds_added'] + self.local_config['rounds'],
                           **(meta or {})}
        self.fit_log.append(dict(self.fit_record))

    def predict(self, X, prob=False):
        return self.labeller(self.labels, X[:, 0].astype(int))

    def save(self, branch_id):
        self._store[branch_id or ''] = (self.labels, {k: v for k, v in self.fit_record.items() if k != 'loaded_as'})

    def load(self, branch_id, fresh=True):
        self.labels, record = self._store[branch_id or '']
        self.fit_record = {**record, 'loaded_as': branch_id or ''}


FLOORS = dict(fit_support={'rows': 1, 'areas': 1, 'dates': 1, 'classes': 1},
              val_support={'rows': 1, 'areas': 1, 'dates': 1})


def run_partition(model, X, y, groups, split, months, threshold=Fraction(1, 100), proposal=None, cap=80,
                  max_depth=2, **floors):
    kwargs = {**FLOORS, **floors}
    patches = [patch.object(trans, 'CONTIGUITY', False),
               patch.object(trans, 'generate_count_grid', return_value=(None, 0, 1))]
    if proposal is not None:
        patches.append(patch.object(trans, 'scan', side_effect=proposal))
    with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
        for p in patches:
            p.start()
        try:
            out = trans.partition(model, X, y, groups, split, np.arange(len(X)), np.full(len(X), '', dtype='<U8'),
                                  min_depth=1, max_depth=max_depth, contiguity_type='polygon', model_dir=tmp,
                                  VIS_DEBUG_MODE=False, X_month=months, threshold=threshold,
                                  path_round_cap=cap, **kwargs)
        finally:
            for p in patches:
                p.stop()
    return out, trans.partition.decisions


class Stage1Partition(unittest.TestCase):
    def fixture(self):
        # Areas 10 and 20 have fitting+validation rows; area 30 is fitting-only.
        groups = np.array([10, 10, 20, 20, 30, 10, 20])
        split = np.array([0, 0, 0, 0, 0, 1, 1])
        y = np.array([0, 0, 2, 2, 0, 0, 2])
        X = np.column_stack([np.arange(7), groups]).astype(float)
        months = np.array([1, 2, 1, 2, 1, 3, 3])

        def labeller(labels, ids):
            if labels[0] == 'root':          # root wrong on the area-20 validation row
                return np.zeros(len(ids), dtype=int)
            return np.where(np.isin(ids, [2, 3, 6]), 2, 0)

        def proposal(Y, A, *args, **kwargs):  # validation groups ordered [10, 20]
            return np.array([0]), np.array([1]), np.zeros(2), np.ones(4)
        return X, y, groups, split, months, labeller, proposal

    def test_fitting_only_area_stays_on_parent_everywhere(self):
        X, y, groups, split, months, labeller, proposal = self.fixture()
        model = PresetModel('.', labeller)
        model.set_root(int((split == 0).sum()))
        (assigned, table, s_branch), decisions = run_partition(model, X, y, groups, split, months, proposal=proposal)
        self.assertEqual(decisions[0]['outcome'], 'accepted')
        # child 0 was fitted on area 10 rows only; area 30 (row 4) is in neither child.
        self.assertEqual(model.trained, [('', [0, 1]), ('', [2, 3])])
        self.assertEqual(assigned[4], '')
        self.assertEqual(decisions[0]['parent_kept_fitting'], {'rows': 1, 'areas': 1})
        e2 = pd.concat(trans.partition.e2_rows)  # keyed E2 evidence = complete parent validation rows
        self.assertEqual(sorted(e2['row_id']), [5, 6])
        self.assertEqual(sorted(e2.loc[e2.side == 1, 'y_child']), [2])
        from src.helper.helper import get_X_branch_id_by_group
        routed = get_X_branch_id_by_group(groups, s_branch)
        np.testing.assert_array_equal(routed, assigned)   # row branch == spatial routing for every row
        self.assertEqual(routed[4], '')

    def test_threshold_families_ties_and_eligibility(self):
        e = np.array([], dtype=int)
        truth, parent = np.full(49, 2), np.zeros(49, dtype=int)
        child = parent.copy(); child[0] = 2                # gain exactly 1/100
        self.assertFalse(opt.select_macro_children(truth, e, parent, e, child, e, min_improvement=Fraction(1, 100))[0])
        self.assertTrue(opt.select_macro_children(truth, e, parent, e, child, e, min_improvement=Fraction(0))[0])
        self.assertFalse(opt.select_macro_children(truth, e, parent, e, parent, e, min_improvement=Fraction(0))[0])
        y0, y1 = np.array([0, 1]), np.array([2, 2])
        p0, p1 = np.array([0, 1]), np.array([0, 0])
        c0, c1 = np.array([1, 1]), np.array([2, 2])
        accepted, sel, preds, *_ = opt.select_macro_children(y0, y1, p0, p1, c0, c1, eligible=(True, False))
        self.assertFalse(accepted)                          # only the ineligible side could help
        np.testing.assert_array_equal(np.concatenate(preds), np.concatenate((p0, p1)))

    def test_support_and_path_cap_fallbacks_and_single_root(self):
        X, y, groups, split, months, labeller, proposal = self.fixture()
        model = PresetModel('.', labeller)
        model.set_root(5)
        _, decisions = run_partition(model, X, y, groups, split, months, proposal=proposal,
                                     fit_support={'rows': 3, 'areas': 1, 'dates': 1, 'classes': 1})
        self.assertEqual(decisions[0]['outcome'], 'rejected_no_eligible_child')
        self.assertEqual(decisions[0]['fallback'], ['fit_support', 'fit_support'])
        self.assertEqual(model.trained, [])
        model = PresetModel('.', labeller, rounds=81)
        model.set_root(5)
        _, decisions = run_partition(model, X, y, groups, split, months, proposal=proposal)
        self.assertEqual(decisions[0]['fallback'], ['path_round_cap', 'path_round_cap'])
        twice = PresetModel('.', labeller)
        twice.set_root(5); twice.set_root(5)
        with self.assertRaises(RuntimeError):
            run_partition(twice, X, y, groups, split, months, proposal=proposal)

    def test_real_booster_partition_children_continue_the_root(self):
        rng = np.random.default_rng(3)
        groups = np.repeat(np.arange(40), 30)
        X = rng.normal(size=(len(groups), 4)); X[:, 3] = groups >= 20
        y = ((X[:, 0] + 3 * X[:, 3] * X[:, 1]) > 0).astype(int) * 2
        months = np.tile(np.arange(30), 40)
        from src.utils.split import group_aware_train_val_split
        split = group_aware_train_val_split(groups, .5, 1, 42, True)['X_set']
        root, rec = nx.fit_global(X[split == 0], y[split == 0], SMALL_G['G1'])
        with tempfile.TemporaryDirectory() as tmp:
            model = nx.XGBmodel(tmp, plan.L_CONFIGS['L1'])
            model.set_root(root, rec)
            run_partition(model, X, y, groups, split, months, threshold=Fraction(0))
            for entry in model.fit_log[1:]:
                self.assertEqual(entry['kind'], 'continuation')
                self.assertEqual(entry['parent_structure_sha256'], rec['structure_sha256'])
                self.assertEqual(entry['path_rounds_added'], 20)
            model.load('0')
            self.assertEqual(nx.prefix_identity(model.booster, root.num_boosted_rounds())['sha256'],
                             rec['structure_sha256'])


def write_panel(path, n_areas=120, first='2015-01', last='2021-06', seed=0, rare=True):
    features = ff.load_schema(SCHEMA)['ordered_features']
    rng = np.random.default_rng(seed)
    rows = [(a, mo) for a in range(n_areas) for mo in range(m(first), m(last) + 1)]
    frame = pd.DataFrame(rows, columns=['area', 'target_month'])
    frame['horizon'] = 4
    frame['country'] = np.where(frame['area'] < n_areas // 2, 'A', 'B')
    X = rng.normal(size=(len(frame), len(features)))
    signal = X[:, 0] + np.where(frame['area'] < n_areas // 2, 1.0, -1.0) * X[:, 1]
    frame['class_code'] = np.digitize(signal, [-1.0, 0.0, 1.2 if rare else 99])
    X[rng.random(X.shape) < .1] = np.nan
    snap = pd.concat([frame, pd.DataFrame(X, columns=features)], axis=1)
    snap.to_parquet(path, index=False)
    return features


class Stage3Engine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.features = write_panel(self.root / 'snap.parquet')
        self.panel = s3.Panel(self.root / 'snap.parquet', self.features, 4)
        self.store = s3.GlobalStore(self.root / 'globals')
        labels = sorted(set(self.panel.month.tolist()))
        self.fold = prep._fold(4, m('2021-06'), pd.Series(1, index=labels), labels, with_gate=True)
        self.patch = patch.dict(plan.G_CONFIGS, SMALL_G)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def specs(self, cluster_of, arm='shared'):
        return [{'arm': 'pooled', 'label': 'pooled', 'local': None, 'route': None},
                {'arm': arm, 'label': 'mapped', 'local': 'L1', 'route': 'learned_map', 'cluster_of': cluster_of}]

    def test_internal_origins_windows_gate_pairs_and_fallbacks(self):
        cluster_of = {a: (0 if a < 60 else 1) for a in range(119)}  # area 119 unmapped
        with redirect_stdout(StringIO()):
            res = s3.run_fold(self.panel, self.store, self.fold, 'G1', self.specs(cluster_of), keep_boosters=True)
        gates = [g['validation_month'] for g in self.fold['gate']]
        self.assertEqual(gates, ['2020-08', '2020-09', '2020-10', '2020-11', '2020-12', '2021-01'])
        for g in self.fold['gate']:  # each internal global is fitted on its OWN [V-59, V)
            _, rec = self.store.get(self.panel, m(g['internal_origin']), 'G1', fit_if_missing=False)
            self.assertEqual(rec['fit_label_months'], g['fit_label_months'])
            self.assertLess(m(rec['fit_label_months'][1]), m(g['internal_origin']))
        mapped, pooled = res['mapped'], res['pooled']
        pairs = mapped['gate_pairs']
        self.assertEqual(set(pairs['validation_month']), set(gates))
        self.assertTrue(pairs['local_fit_ok'].any() and (mapped['predictions']['route'] == 'local_model').any())
        for c in (0, 1):  # recompute the D13 decision from the saved pairs
            sub = pairs[pairs['cluster_id'] == c]
            gain = s3.exact_f1(sub['y_true'], sub['y_local_routed']) - s3.exact_f1(sub['y_true'], sub['y_global'])
            dec = next(d for d in mapped['gate'] if d['cluster_id'] == c)
            self.assertEqual(Fraction(dec['gain']), gain)
            self.assertEqual(dec['enabled'], gain > Fraction(1, 100) and dec['rows'] >= 100)
        preds = mapped['predictions']
        unmapped = preds['area'] == 119
        self.assertTrue((preds.loc[unmapped, 'route'] == 'unmapped_area_global').all())
        np.testing.assert_array_equal(preds.loc[unmapped].filter(like='p_').to_numpy(),
                                      pooled['predictions'].loc[unmapped].filter(like='p_').to_numpy())
        fallback = preds['route'].str.startswith('global_fallback')
        np.testing.assert_array_equal(preds.loc[fallback].filter(like='p_').to_numpy(),
                                      pooled['predictions'].loc[fallback].filter(like='p_').to_numpy())
        for name, payload in mapped['boosters'].items():  # saved local boosters replay exactly
            c = int(name.split('_')[1])
            rows = (preds['cluster_id'] == c).to_numpy()
            test = self.panel.at(m('2021-06'))
            np.testing.assert_array_equal(nx.proba(nx.from_raw(payload), self.panel.X[test[rows]]),
                                          preds.loc[rows].filter(like='p_').to_numpy())

    def test_failed_internal_fit_keeps_its_rows_with_global_predictions(self):
        cluster_of = {a: 0 for a in range(120)}
        with patch.dict(plan.FIT_SUPPORT, {'dates': 63}), redirect_stdout(StringIO()):  # >59 dates: never fits
            res = s3.run_fold(self.panel, self.store, self.fold, 'G1', self.specs(cluster_of))
        pairs = res['mapped']['gate_pairs']
        self.assertEqual(len(pairs), 6 * 120)             # every validation key retained
        self.assertFalse(pairs['local_fit_ok'].any())
        np.testing.assert_array_equal(pairs['y_local_routed'], pairs['y_global'])
        dec = res['mapped']['gate'][0]
        self.assertFalse(dec['enabled'])
        self.assertEqual(dec['reason'], 'gate_support:local_fit_dates')

    def test_gate_decision_exact_boundary_and_rare_class_local_fit(self):
        pairs = pd.DataFrame({'area': np.arange(100) % 25, 'validation_month': np.repeat(['a', 'b', 'c', 'd'], 25),
                              'local_fit_ok': True, 'y_true': np.r_[np.full(49, 2), np.zeros(51, int)],
                              'y_global': np.zeros(100, int)})
        pairs['y_local_routed'] = pairs['y_global']
        base = s3.exact_f1(pairs['y_true'], pairs['y_global'])
        for k in range(1, 6):
            pairs.loc[:k - 1, 'y_local_routed'] = 2
            gain = s3.exact_f1(pairs['y_true'], pairs['y_local_routed']) - base
            self.assertEqual(s3.gate_decision(pairs)['enabled'], gain > Fraction(1, 100))
        rows = self.panel.window(m('2021-02'))
        rows = rows[self.panel.y[rows] != 3]
        g, _ = self.store.get(self.panel, m('2021-02'), 'G1')
        booster, rec = s3.fit_local('shared', g, self.panel, rows, 'G1', 'L2')
        self.assertEqual(rec['fit_support']['class_counts'][3], 0)
        self.assertEqual(nx.proba(booster, self.panel.X[:5]).shape, (5, 4))
        ind, irec = s3.fit_local('independent', g, self.panel, rows, 'G1', 'L1')
        self.assertEqual(ind.num_boosted_rounds(), 6 + 20)
        self.assertNotEqual(irec['parent_sha256'], nx.sha(g))

    def test_fresh_store_reopens_a_stored_global_from_disk(self):
        _, rec = self.store.get(self.panel, m('2021-02'), 'G1')
        fresh = s3.GlobalStore(self.root / 'globals')
        booster, again = fresh.get(self.panel, m('2021-02'), 'G1', fit_if_missing=False)
        self.assertEqual(again['booster_sha256'], rec['booster_sha256'])
        self.assertEqual(again['g_config_params'], plan.G_CONFIGS['G1'])      # requested config (identity)
        self.assertEqual(again['params']['objective'], 'multi:softprob')          # resolved params (provenance)

    def test_no_map_and_null_routes_are_the_pooled_global(self):
        with redirect_stdout(StringIO()):
            res = s3.run_fold(self.panel, self.store, self.fold, 'G1',
                              [{'arm': 'pooled', 'label': 'pooled', 'local': None, 'route': None},
                               {'arm': 'shared', 'label': 'none', 'local': 'L1', 'route': 'no_prior_candidates',
                                'cluster_of': None}])
        np.testing.assert_array_equal(res['none']['predictions'].filter(like='p_').to_numpy(),
                                      res['pooled']['predictions'].filter(like='p_').to_numpy())
        self.assertTrue((res['none']['predictions']['route'] == 'no_prior_candidates_pooled').all())


class ConsensusAndBoundaries(unittest.TestCase):
    def candidates(self, tmp, positive=True):
        coords = pd.DataFrame({'FEWSNET_admin_code': range(60), 'lat': np.repeat(np.arange(6), 10) * 1.0,
                               'lon': np.tile(np.arange(10), 6) * 1.0})
        coords.to_csv(tmp / 'coords.csv', index=False)
        rows, paths = [], {}
        for i, (h, t) in enumerate([(4, '2018-02'), (8, '2018-06'), (12, '2018-10')]):
            name = f'h{h}_{t}_G1_L1_r80_s42_gt001'
            part = np.where(coords['lat'] < 3, '0', '1') if i < 2 else np.where(coords['lon'] < 5, '0', '1')
            path = tmp / f'{name}.csv'
            pd.DataFrame({'FEWSNET_admin_code': range(60), 'partition_id': part}).to_csv(path, index=False)
            rows.append({'name': name, 'horizon': h, 'target_month': t, 'macro_f1': .6 if positive else .4,
                         'macro_f1_base': .5, 'n_terminal': 2, 'correspondence_sha256': rid.file_sha256(path),
                         'source': 'test'})
            paths[name] = path
        return pd.DataFrame(rows), paths, tmp / 'coords.csv'

    def test_routes_diagnostics_and_forged_ledger_refused(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            frame, paths, coords = self.candidates(tmp)
            with redirect_stdout(StringIO()):
                empty = stage2.build_consensus(tmp / 'none', frame.iloc[:0], {}, coords, 'x')
                null = stage2.build_consensus(tmp / 'null', self.candidates(tmp, False)[0], paths, coords, 'x')
                learned = stage2.build_consensus(tmp / 'map', frame, paths, coords, 'x')
            self.assertEqual(empty['route'], 'no_prior_candidates')
            self.assertEqual(null['route'], 'null_consensus')
            self.assertEqual(learned['route'], 'learned_map')
            self.assertEqual(learned['diagnostics']['same_coverage_duplicate_partitions'], 1)
            self.assertEqual(learned['diagnostics']['positive_weight'], 3)
            self.assertAlmostEqual(learned['diagnostics']['weight_concentration_sum_sq'], 3.0)
            self.assertFalse((tmp / 'map' / 'experiment' / 'similarity_matrices' / 'similarity_matrices.npz').exists())
            self.assertEqual(len(stage2.cluster_map(tmp / 'map', learned)), 60)
            self.assertEqual(stage2.build_consensus(tmp / 'map', frame, paths, coords, 'x')['pool_identity'],
                             learned['pool_identity'])           # identical pool reuses, never rebuilds
            with self.assertRaises(RuntimeError):                 # a different expected pool is refused
                stage2.accept_consensus(tmp / 'map', frame.iloc[:2])
            ledger = tmp / 'map' / 'candidate_ledger.csv'
            original = ledger.read_bytes()
            pd.read_csv(ledger).assign(macro_f1=.99).to_csv(ledger, index=False)
            record = json.loads((tmp / 'map' / 'consensus.json').read_text()); record.pop('outputs')
            stage2._finish(tmp / 'map', record, 0)                # re-hashed, self-consistent forgery
            with self.assertRaises(RuntimeError):
                stage2.accept_consensus(tmp / 'map', frame)
            ledger.write_bytes(original)

    def test_map_pool_uses_only_scoring_targets_before_origin(self):
        names = [plan.candidate_name(h, t, 'G1', l, 'r80', 42, f) for h in (4, 12) for t in ('2018-02', '2018-06')
                 for l in ('L1', 'L2') for f in ('gt0', 'gt001')]
        frame = pd.DataFrame({'name': names, 'horizon': [int(n.split('_')[0][1:]) for n in names],
                              'target_month': [n.split('_')[1] for n in names]})
        scheme = {'l_vector': {4: 'L1', 8: 'L2', 12: 'L2'}, 'strategy': 'strict-only'}
        sub = rexp.scheme_pool(frame, scheme, m('2018-06'))
        self.assertEqual(sorted(sub['name']), ['h12_2018-02_G1_L2_r80_s42_gt001', 'h4_2018-02_G1_L1_r80_s42_gt001'])
        self.assertTrue(rexp.scheme_pool(frame, scheme, m('2018-02')).empty)
        merged = rexp.scheme_pool(frame, {**scheme, 'strategy': 'merged'}, m('2018-07'))
        self.assertEqual(len(merged), 8)

    def test_frozen_schedule_counts_and_selection_order(self):
        self.assertEqual(len(plan.schemes()), 24)
        self.assertEqual(sorted(plan.G_CONFIGS, key=plan.g_tiebreak_key), ['G1', 'G3', 'G2', 'G4'])
        obs = pd.DataFrame({'month': [m(t) for t in plan.STAGE1_TARGETS + ('2010-01', '2016-02', '2016-06',
                                                                          '2016-10', '2017-02', '2017-06', '2017-10')]
                                     + [m(f'{y}-{mo:02d}') for y in range(2021, 2025) for mo in (2, 6, 10)]})
        sched = prep.build_schedule(obs)
        self.assertEqual(sched['stage1_counts']['candidates'], 648)
        self.assertEqual(sched['stage1_counts']['roots'], 162)
        for fold in sched['development'] + [f for f in sched['stage3'] if f['status'] == 'scheduled']:
            self.assertTrue(all(m(g['validation_month']) < m(fold['origin_month']) for g in fold['gate']))
            self.assertEqual(m(fold['train_label_months'][1]) + 1 - m(fold['train_label_months'][0]), 59)
        self.assertEqual(sched['earliest_window_lower_bound'], '2010-03')  # 2019-02/H12 gate: V=2015-02


class ConsensusPlumbing(unittest.TestCase):
    def test_candidates_consensus_gated_fold_and_scores(self):
        """Plumbing fixture on a synthetic panel: Stage 1 GeoRF.fit from one installed root
        -> build_consensus -> gated Stage 3 fold -> keyed fixed-four scores. One candidate's
        E3 score is SET above its base so that a learned map exists; this is not producer
        evidence. The producer-created integration evidence is tests/integration_minirun.py
        (the full production chain incl. report and verifier on a real-data area subset)."""
        import os
        from src.model.GeoRF import GeoRF
        from src.utils.split import group_aware_train_val_split
        from src.helper.helper import get_X_branch_id_by_group
        with tempfile.TemporaryDirectory() as t, patch.dict(plan.G_CONFIGS, SMALL_G), \
                patch.object(trans, 'CONTIGUITY', False), redirect_stdout(StringIO()):
            tmp = Path(t)
            features = write_panel(tmp / 'snap.parquet')
            panel = s3.Panel(tmp / 'snap.parquet', features, 4)
            coords = pd.DataFrame({'FEWSNET_admin_code': range(120), 'lat': np.arange(120) // 12 * 1.0,
                                   'lon': np.arange(120) % 12 * 1.0})
            coords.to_csv(tmp / 'coords.csv', index=False)
            rows, paths = [], {}
            for seed in plan.SPLIT_SEEDS:  # three candidates of one root target
                target = m('2019-06')
                win = panel.window(target - 4)
                split = group_aware_train_val_split(panel.area[win], .5, 1, seed, True)['X_set']
                root = nx.fit_global(panel.X[win][split == 0], panel.y[win][split == 0], plan.G_CONFIGS['G1'])
                here = os.getcwd(); work = tmp / f's{seed}'; work.mkdir(); os.chdir(work)
                try:
                    model = GeoRF(min_model_depth=1, max_model_depth=3)
                    model.fit(panel.X[win], panel.y[win], panel.area[win], X_set=split, split={'X_set': split},
                              contiguity_type='polygon', polygon_contiguity_info=None, print_to_file=False,
                              VIS_DEBUG_MODE=False, root=root, local_config=plan.L_CONFIGS['L1'],
                              threshold=plan.THRESHOLD_FAMILIES['gt0'], X_month=panel.month[win])
                finally:
                    os.chdir(here)
                    import logging  # GeoRF.fit's file handler (production closes it the same way)
                    for handler in list(logging.getLogger().handlers):
                        logging.getLogger().removeHandler(handler)
                        handler.close()
                self.assertEqual(sum(e.get('kind') == 'fresh' for e in model.model.fit_log), 1)
                branch = get_X_branch_id_by_group(panel.area[win], model.s_branch)
                corr = pd.DataFrame({'FEWSNET_admin_code': panel.area[win],
                                     'partition_id': np.where(branch == '', 'root', branch)}).drop_duplicates()
                self.assertFalse(corr['FEWSNET_admin_code'].duplicated().any())
                path = tmp / f'c{seed}.csv'; corr.to_csv(path, index=False)
                test_rows = panel.at(target)
                part = model.model.predict_georf(panel.X[test_rows], panel.area[test_rows], model.s_branch)
                pooled = fourclass.argmax_codes(nx.proba(root[0], panel.X[test_rows]))
                name = plan.candidate_name(4, '2019-06', 'G1', 'L1', 'r50', seed, 'gt0')
                rows.append({'name': name, 'horizon': 4, 'target_month': '2019-06',
                             'macro_f1': fourclass.macro_f1(panel.y[test_rows], part),
                             'macro_f1_base': fourclass.macro_f1(panel.y[test_rows], pooled),
                             'n_terminal': corr['partition_id'].nunique(),
                             'correspondence_sha256': rid.file_sha256(path), 'source': 'test'})
                paths[name] = path
            frame = pd.DataFrame(rows)
            frame.loc[0, 'macro_f1'] = frame.loc[0, 'macro_f1_base'] + .05  # guarantee one positive weight
            record = stage2.build_consensus(tmp / 'map', frame, paths, tmp / 'coords.csv', 'e2e')
            self.assertEqual(record['route'], 'learned_map')
            cluster_of = stage2.cluster_map(tmp / 'map', record)
            labels = sorted(set(panel.month.tolist()))
            fold = prep._fold(4, m('2021-06'), pd.Series(1, index=labels), labels, with_gate=True)
            res = s3.run_fold(panel, s3.GlobalStore(tmp / 'g'), fold, 'G1',
                              [{'arm': 'pooled', 'label': 'pooled', 'local': None, 'route': None},
                               {'arm': 'shared', 'label': 'main', 'local': 'L1', 'route': 'learned_map',
                                'cluster_of': cluster_of}])
            truth = panel.y[panel.at(m('2021-06'))]
            for arm in ('pooled', 'main'):
                preds = res[arm]['predictions']
                self.assertEqual(sorted(preds['area']), sorted(panel.area[panel.at(m('2021-06'))]))
                np.testing.assert_array_equal(preds['y_true_code'], truth)
                np.testing.assert_array_equal(preds['y_pred_code'], preds.filter(like='p_').to_numpy().argmax(axis=1))
                self.assertAlmostEqual(fourclass.macro_f1(truth, preds['y_pred_code']),
                                       f1_score(truth, preds['y_pred_code'], labels=[0, 1, 2, 3], average='macro',
                                                zero_division=0))


class VerifierBoundaries(unittest.TestCase):
    """Follow-up repairs: the verifier must fail on these boundaries, not self-certify."""

    def test_branch_ids_keep_leading_zeros(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / 'p.csv'
            pd.DataFrame({'branch_id': ['00', '01', '10', '', 'root'], 'x': range(5)}).to_csv(path, index=False)
            self.assertEqual(list(vf.read(path, ('branch_id',)).branch_id), ['00', '01', '10', '', 'root'])

    def test_e2_rows_bound_to_actual_rows_and_sides(self):
        X, y, groups, split, months, labeller, proposal = Stage1Partition().fixture()
        model = PresetModel('.', labeller)
        model.set_root(5)
        (assigned, _, _), decisions = run_partition(model, X, y, groups, split, months, proposal=proposal)
        e2 = pd.concat(trans.partition.e2_rows).reset_index(drop=True)
        e2.insert(4, 'area', groups[e2.row_id]); e2.insert(5, 'target_month', [f'm{m}' for m in months[e2.row_id]])
        c = {'partition': {'decisions': decisions}, 'local_config': 'L1',
             'fits': {'fit_log': model.fit_log}}
        t = {'val_pos': np.where(split == 1)[0], 'fit_pos': np.where(split == 0)[0], 'area': groups,
             'month': months, 'month_label': np.array([f'm{m}' for m in months], dtype=object), 'y': y,
             'branch': assigned.astype(str)}
        with patch.dict(plan.FIT_SUPPORT, FLOORS['fit_support']), \
                patch.dict(plan.STAGE1_VAL_SUPPORT, FLOORS['val_support']):
            self.assertEqual(vf.e2_problems(c, e2, t, Fraction(1, 100)), [])
            bad_side = e2.copy(); bad_side.loc[bad_side.side == 1, 'side'] = 2
            self.assertTrue(vf.e2_problems(c, bad_side, t, Fraction(1, 100)))
            bad_truth = e2.copy(); bad_truth.loc[0, 'y_true'] = 3
            self.assertTrue(vf.e2_problems(c, bad_truth, t, Fraction(1, 100)))
            self.assertTrue(vf.e2_problems(c, e2, t, Fraction(2)))  # a threshold no gain can pass: outcome must flip

    def test_no_prior_route_needs_an_empty_expected_pool(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            frame, paths, coords = ConsensusAndBoundaries().candidates(tmp)
            one = frame.iloc[:1].assign(macro_f1=.4)  # a genuine single-candidate null pool
            with redirect_stdout(StringIO()):
                stage2.build_consensus(tmp / 'none', frame.iloc[:0], {}, coords, 'x')
                stage2.build_consensus(tmp / 'one', one, paths, coords, 'x')
            stage2.accept_consensus(tmp / 'none', frame.iloc[:0])
            stage2.accept_consensus(tmp / 'one', one)
            record = json.loads((tmp / 'one' / 'consensus.json').read_text()); record.pop('outputs')
            record['route'] = 'no_prior_candidates'      # forged route, pool identity still matches
            stage2._finish(tmp / 'one', record, 0)
            with self.assertRaises(RuntimeError):
                stage2.accept_consensus(tmp / 'one', one)

    def test_gate_population_and_local_records_are_exact(self):
        env = Stage3Engine(); env.setUp()
        try:
            prepared = env.root / 'prepared'; prepared.mkdir()
            import shutil
            shutil.copy2(env.root / 'snap.parquet', prepared / 'snapshot_h4.parquet')
            cluster_of = {a: (0 if a < 60 else 1) for a in range(119)}
            with redirect_stdout(StringIO()):
                res = s3.run_fold(env.panel, env.store, env.fold, 'G1', env.specs(cluster_of))
            fold_dir = env.root / 'fold'
            rexp.save_fold(fold_dir, res['mapped'], {'phase': 'test'}, False)
            ex = vf.Expected(env.root, 4)
            problems = []
            vf.verify_gate_dir(fold_dir, env.fold, cluster_of, ex, env.root, 'G1', problems)
            self.assertEqual(problems, [])
            pairs = res['mapped']['gate_pairs']
            extra = pairs.iloc[:30].assign(validation_month='2099-01')
            rexp.write_csv_gz(fold_dir / 'gate_pairs.csv.gz', pd.concat([pairs, extra]))
            problems = []
            vf.verify_gate_dir(fold_dir, env.fold, cluster_of, ex, env.root, 'G1', problems)
            self.assertTrue(any('population' in p for p in problems))
            rexp.write_csv_gz(fold_dir / 'gate_pairs.csv.gz', pairs)
            gate = json.loads((fold_dir / 'gate.json').read_text()); gate['locals'] = {}
            (fold_dir / 'gate.json').write_text(json.dumps(gate))
            problems = []
            vf.verify_gate_dir(fold_dir, env.fold, cluster_of, ex, env.root, 'G1', problems)
            routed = (res['mapped']['predictions'].route == 'local_model').any()
            self.assertEqual(any('local booster records' in p for p in problems), bool(routed))
        finally:
            env.tearDown()


class CrisisEndpoint(unittest.TestCase):
    """D26: four-class probabilities, binary crisis (IPC>=3) evaluation everywhere in Stage 1."""

    def test_crisis_collapse_and_exact_f1(self):
        truth, pred = np.array([0, 2, 3, 1]), np.array([2, 2, 0, 1])
        np.testing.assert_array_equal(fourclass.crisis(truth), [0, 1, 1, 0])
        self.assertEqual(fourclass.crisis_counts(truth, pred), {'tp': 1, 'fp': 1, 'fn': 1, 'tn': 1})
        self.assertEqual(fourclass.crisis_f1_exact(truth, pred), Fraction(1, 2))
        self.assertAlmostEqual(fourclass.crisis_f1(truth, pred),
                               f1_score(truth >= 2, pred >= 2, pos_label=True, zero_division=0))
        self.assertEqual(fourclass.crisis_f1_exact(np.zeros(3, int), np.zeros(3, int)), 0)  # undefined -> 0
        self.assertEqual(plan.ENDPOINT, 'crisis_f1')
        self.assertEqual(fourclass.endpoint_exact(truth, pred), Fraction(1, 2))

    def test_crisis_scan_masses_single_column_and_scan(self):
        truth = np.array([2, 2, 0, 3, 1, 0])
        pred = np.array([2, 0, 2, 3, 1, 0])
        groups = np.array([10, 10, 20, 20, 30, 30])
        gids, Y, _, A = opt.get_class_wise_stat(truth, pred, groups)
        np.testing.assert_array_equal(gids, [10, 20, 30])
        # crisis TP/FP/FN per group: 10: 1/0/1, 20: 1/1/0, 30: 0/0/0 -> D = 3, 3, 0 (total 6)
        np.testing.assert_allclose(Y[:, 0], [3 / 6, 3 / 6, 0])
        np.testing.assert_allclose(A[:, 0], [2 / 6, 2 / 6, 0])
        self.assertAlmostEqual(float((Y - A).sum()), 1 - float(fourclass.crisis_f1_exact(truth, pred)))
        with redirect_stdout(StringIO()):
            s0, s1, g, q = opt.scan(Y, A, 0, return_score=True)
        self.assertTrue(np.isfinite(g).all() and np.isfinite(q).all())

    def test_e2_crisis_gain_families(self):
        e = np.array([], dtype=int)
        truth = np.array([2, 2, 3, 0, 1, 0])
        parent = np.array([2, 0, 0, 0, 1, 2])          # crisis tp1 fp1 fn2 -> 2/5
        child = np.array([2, 3, 0, 0, 1, 2])           # tp2 fp1 fn1 -> 4/6
        gain = Fraction(4, 6) - Fraction(2, 5)
        for family, threshold in plan.THRESHOLD_FAMILIES.items():
            accepted, _, _, base, best, _ = opt.select_macro_children(
                truth, e, parent, e, child, e, min_improvement=threshold, score=fourclass.endpoint_exact)
            self.assertEqual((base, best - base), (Fraction(2, 5), gain))
            self.assertEqual(accepted, gain > threshold)
        # a four-class change inside the crisis group is NOT a crisis gain: parent wins ties
        recode = parent.copy(); recode[0] = 3
        self.assertFalse(opt.select_macro_children(truth, e, parent, e, recode, e, min_improvement=Fraction(0),
                                                   score=fourclass.endpoint_exact)[0])

    def test_gscreen_reuse_requires_identical_inputs_and_downstream_is_blocked(self):
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / 'a', Path(t) / 'b'
            for run, sha in ((a, 'x'), (b, 'y')):
                (run / 'prepared' / 'manifests').mkdir(parents=True)
                (run / 'prepared' / 'manifests' / 'outputs.json').write_text(json.dumps(
                    {f: sha for f in rexp.REUSED_PREPARED}))
            with self.assertRaises(RuntimeError):
                rexp.reused_gscreen_predictions(a, b)
        with patch.object(sys, 'argv', ['run_experiment.py', '--run-dir', '.', 'maps']), \
                self.assertRaises(SystemExit):
            rexp.main()


class TimeBlockContrast(unittest.TestCase):
    """D27 (experiment-plan A2): tb3 time-block split, six-root schedule, G-prediction reuse."""

    @staticmethod
    def mgf():
        from app import main_model_GF
        return main_model_GF

    @staticmethod
    def pool():
        # H4, T=2018-02, O=2017-10. Areas 1-3 every Feb/Jun/Oct 2015-02..2017-06; area 9 only
        # in the last block month (validation-only); area 8 only early (fitting-only).
        months = [m(f'{y}-{mo:02d}') for y in (2015, 2016, 2017) for mo in (2, 6, 10) if (y, mo) < (2017, 10)]
        rows = [(a, mo) for a in (1, 2, 3) for mo in months] + [(9, m('2017-06')), (8, m('2015-06'))]
        groups, mons = (np.array(c) for c in zip(*rows))
        return groups, mons

    def test_whole_months_strict_order_and_validation_only_area(self):
        from src.utils.split import time_block_split
        groups, months = self.pool()
        res = time_block_split(groups, months, m('2017-10'), 3, ('2016-10', '2017-02', '2017-06'))
        block = [m('2016-10'), m('2017-02'), m('2017-06')]
        np.testing.assert_array_equal(res['X_set'], np.isin(months, block).astype(int))   # whole months, all areas
        self.assertLess(months[res['X_set'] == 0].max(), months[res['X_set'] == 1].min())
        self.assertTrue((res['X_set'][groups == 9] == 1).all())                             # never moved to fitting
        self.assertNotIn(9, set(groups[res['X_set'] == 0]))
        self.assertEqual((res['validation_only_groups'], res['fitting_only_groups']), (1, 1))
        self.assertEqual(res['validation_months'], ['2016-10', '2017-02', '2017-06'])
        x_set, ratio, record = self.mgf().stage1_split('tb3', groups, months, m('2017-10'), 42, 4, '2018-02')
        np.testing.assert_array_equal(x_set, res['X_set'])
        self.assertIsNone(ratio)
        self.assertEqual((record['split_mode'], record['validation_months'][0], record['fitting_months'][-1]),
                         ('tb3', '2016-10', '2016-06'))

    def test_target_month_excluded_through_the_rolling_window(self):
        from config import TRAIN_WINDOW_MONTHS
        from src.customize.customize import train_test_split_rolling_window
        labelled = [m(f'{y}-{mo:02d}') for y in range(2012, 2019) for mo in (2, 6, 10) if m(f'{y}-{mo:02d}') <= m('2018-02')]
        areas = np.repeat(np.arange(6), len(labelled))
        months = np.tile(labelled, 6)
        dates = pd.to_datetime(pd.Series(ff.month_label(months)) + '-01')
        X = np.zeros((len(months), 2))
        split = train_test_split_rolling_window(X, np.zeros(len(months), int), X, areas, dates.dt.year.to_numpy(), dates,
                                                test_month=pd.Period('2018-02', freq='M'), active_lag=4,
                                                train_window_months=TRAIN_WINDOW_MONTHS, admin_codes=np.arange(len(months)))
        mtrain, mtest = months[split[8]], months[split[9]]
        self.assertEqual(set(mtest), {m('2018-02')})
        x_set, _, record = self.mgf().stage1_split('tb3', areas[split[8]], mtrain, m('2017-10'), 42, 4, '2018-02')
        self.assertNotIn(m('2018-02'), set(mtrain))
        self.assertEqual(record['validation_months'], list(plan.TB3_VALIDATION_MONTHS[(4, '2018-02')]))
        self.assertTrue(all(f < '2016-10' for f in record['fitting_months']))
        from src.utils.split import time_block_split
        with self.assertRaises(ValueError):          # a target-month row in the pool is refused
            time_block_split(areas, months, m('2017-10'), 3)

    def test_mismatch_and_incomplete_blocks_raise_without_random_fallback(self):
        from src.utils.split import time_block_split
        groups, months = self.pool()
        with self.assertRaises(ValueError):
            time_block_split(groups, months, m('2017-10'), 3, ('2016-06', '2016-10', '2017-02'))
        few = np.isin(months, [m('2016-10'), m('2017-02'), m('2017-06')])
        with self.assertRaises(ValueError):          # three months and no earlier fitting month
            time_block_split(groups[few], months[few], m('2017-10'), 3)
        with self.assertRaises(ValueError):
            time_block_split(groups[:2], months[:2], m('2017-10'), 3)
        mgf = self.mgf()
        with self.assertRaises(ValueError):          # the computed block differs from the A2 table for H8
            mgf.stage1_split('tb3', groups, months, m('2017-10'), 42, 8, '2018-02')
        with self.assertRaises(ValueError):
            mgf.stage1_split('tb3', groups, months, m('2017-10'), 43, 4, '2018-02')
        with self.assertRaises(ValueError):
            mgf.stage1_split('tb3', groups, months, m('2017-10'), 42, 4, '2019-02')

    def test_random_split_is_unchanged(self):
        from src.utils.split import group_aware_train_val_split
        groups, months = self.pool()
        for ratio, share in plan.SPLIT_RATIOS.items():
            x_set, val_ratio, record = self.mgf().stage1_split(ratio, groups, months, m('2017-10'), 43, 4, '2018-02')
            ref = group_aware_train_val_split(groups=groups, val_ratio=share, min_val_per_group=1, random_state=43,
                                              skip_singleton_groups=True)
            np.testing.assert_array_equal(x_set, ref['X_set'])
            self.assertEqual(val_ratio, share)
            self.assertEqual(set(record), {'rule', 'groups_with_validation', 'singleton_groups_train_only'})
            self.assertEqual(record['singleton_groups_train_only'], 2)

    def test_tb3_schedule_has_six_distinct_roots_and_candidates(self):
        from scripts import run_stage1
        obs = pd.DataFrame({'month': [m(t) for t in plan.STAGE1_TARGETS + ('2010-01', '2016-02', '2016-06',
                                                                          '2016-10', '2017-02', '2017-06', '2017-10')]
                                     + [m(f'{y}-{mo:02d}') for y in range(2021, 2025) for mo in (2, 6, 10)]})
        sched = prep.build_schedule(obs)
        self.assertEqual((sched['stage1_counts']['roots'], sched['stage1_counts']['candidates']), (162, 648))
        self.assertEqual((sched['stage1_tb3_counts']['roots'], sched['stage1_tb3_counts']['candidates']), (6, 6))
        g_of = {'4': 'G1', '8': 'G4', '12': 'G2'}
        tb3 = run_stage1.scheduled_roots(sched, g_of, 'tb3')
        tb3_c = run_stage1.scheduled_candidates(sched, g_of, 'tb3')
        self.assertEqual(len(tb3), 6)
        self.assertEqual(sorted(tb3_c), sorted(f'h{h}_{t}_{g_of[str(h)]}_L1_tb3_s42_gt0'
                                               for h in plan.HORIZONS for t in plan.TB3_TARGETS))
        self.assertFalse(set(tb3) & set(run_stage1.scheduled_roots(sched, g_of)))
        self.assertFalse(set(tb3_c) & set(run_stage1.scheduled_candidates(sched, g_of)))
        self.assertEqual({(e['horizon'], e['target_month']) for e in tb3.values()}, set(plan.TB3_VALIDATION_MONTHS))
        self.assertEqual(self.mgf().root_candidates(4, '2018-02', 'G1', 'tb3', 42), ['h4_2018-02_G1_L1_tb3_s42_gt0'])
        self.assertEqual(len(self.mgf().root_candidates(4, '2018-02', 'G1', 'r80', 42)), 4)

    def test_time_block_partition_runs_with_validation_only_areas(self):
        from src.utils.split import time_block_split
        rng = np.random.default_rng(5)
        groups = np.repeat(np.arange(40), 12)
        months = np.tile(np.arange(100, 112), 40)
        keep = ~((groups < 3) & (months < 109))          # areas 0-2 observed only in the block
        groups, months = groups[keep], months[keep]
        X = rng.normal(size=(len(groups), 4)); X[:, 3] = groups >= 20
        y = ((X[:, 0] + 3 * X[:, 3] * X[:, 1]) > 0).astype(int) * 2
        split = time_block_split(groups, months, 112, 3)['X_set']
        self.assertTrue((split[groups < 3] == 1).all())
        root, rec = nx.fit_global(X[split == 0], y[split == 0], SMALL_G['G1'])
        with tempfile.TemporaryDirectory() as tmp:
            model = nx.XGBmodel(tmp, plan.L_CONFIGS['L1'])
            model.set_root(root, rec)
            (assigned, _, s_branch), _ = run_partition(model, X, y, groups, split, months, threshold=Fraction(0))
            for entry in model.fit_log[1:]:
                self.assertLessEqual(entry['rows'], int((split == 0).sum()))
            from src.helper.helper import get_X_branch_id_by_group
            np.testing.assert_array_equal(get_X_branch_id_by_group(groups, s_branch), assigned)

    def reuse_fixture(self, root, dev_targets=plan.DEV_TARGETS, params=None, extra=None, reused=None):
        """A source/target run layout: preparation identity, schedule, G predictions and,
        per H x G, a stored global record with its effective params/rounds."""
        from src.utils.run_identity import file_sha256
        dev = [{'horizon': h, 'target_month': t, 'origin_month': ff.month_label([m(t) - h])[0]}
               for h in plan.HORIZONS for t in plan.DEV_TARGETS]
        manifests = root / 'prepared' / 'manifests'
        manifests.mkdir(parents=True)
        (manifests / 'schedule.json').write_text(json.dumps({'development': dev, **(extra or {})}))
        outputs = {f: 'same' for f in rexp.REUSED_PREPARED}
        outputs['manifests/schedule.json'] = file_sha256(manifests / 'schedule.json')
        (manifests / 'outputs.json').write_text(json.dumps(outputs))
        prepared = file_sha256(manifests / 'outputs.json')
        (manifests / 'identity.json').write_text(json.dumps({'stage': 'prepare', 'outputs_sha256': prepared}))
        rows = [{'area': 1, 'target_month': t, 'horizon': h, 'g_config': g, 'y_pred_code': 0}
                for h in plan.HORIZONS for t in dev_targets for g in plan.G_CONFIGS]
        (root / 'gscreen').mkdir()
        rexp.write_csv_gz(root / 'gscreen' / 'predictions.csv.gz', pd.DataFrame(rows))
        (root / 'gscreen' / 'selection.json').write_text(json.dumps({
            'prepared': prepared, 'code': {'sha256': 'c'}, 'reused_predictions': reused,
            'outputs': {'predictions.csv.gz': file_sha256(root / 'gscreen' / 'predictions.csv.gz')}}))
        for h in plan.HORIZONS:
            for g, config in plan.G_CONFIGS.items():
                p_, rounds = plan.booster_params(config)
                p_ = {**p_, **((params or {}).get(g, {}))}
                d = root / 'globals' / f'h{h}' / g
                d.mkdir(parents=True)
                (d / 'O2019-01.json').write_text(json.dumps({'params': p_, 'rounds_total': rounds}))

    def test_gscreen_reuse_identity_guards(self):
        def pair(t, **src_kwargs):
            run, src = Path(t) / 'run', Path(t) / 'src'
            self.reuse_fixture(run, extra={'stage1_tb3_roots': [1, 2]})     # tb3 lists may differ
            self.reuse_fixture(src, **src_kwargs)
            return run, src
        with tempfile.TemporaryDirectory() as t:
            run, src = pair(t)
            preds, record = rexp.reused_gscreen_predictions(run, src)
            self.assertEqual(len(preds), 3 * 4 * 6)
            self.assertEqual(record['matched_development_schedule_sha256'],
                             rexp.dev_schedule_digest(json.loads((run / 'prepared' / 'manifests' / 'schedule.json').read_text())))
        with tempfile.TemporaryDirectory() as t:                            # changed development schedule
            run, src = pair(t)
            sched = json.loads((src / 'prepared' / 'manifests' / 'schedule.json').read_text())
            sched['development'][0]['origin_month'] = '1999-01'
            (src / 'prepared' / 'manifests' / 'schedule.json').write_text(json.dumps(sched))
            with self.assertRaisesRegex(RuntimeError, 'development schedule differs'):
                rexp.reused_gscreen_predictions(run, src)
        with tempfile.TemporaryDirectory() as t:                            # one development target missing
            run, src = pair(t, dev_targets=plan.DEV_TARGETS[:-1])
            with self.assertRaisesRegex(RuntimeError, 'targets'):
                rexp.reused_gscreen_predictions(run, src)
        with tempfile.TemporaryDirectory() as t:                            # other effective XGB parameters
            run, src = pair(t, params={'G2': {'nthread': 8}})
            with self.assertRaisesRegex(RuntimeError, 'effective XGBoost parameters'):
                rexp.reused_gscreen_predictions(run, src)
        with tempfile.TemporaryDirectory() as t:                            # G screen not bound to its preparation
            run, src = pair(t)
            sel = json.loads((src / 'gscreen' / 'selection.json').read_text()); sel['prepared'] = 'other'
            (src / 'gscreen' / 'selection.json').write_text(json.dumps(sel))
            with self.assertRaisesRegex(RuntimeError, 'completed preparation'):
                rexp.reused_gscreen_predictions(run, src)
        with tempfile.TemporaryDirectory() as t:                            # a reuse of a reuse
            run, src = pair(t, reused={'source_run': 'x'})
            with self.assertRaisesRegex(RuntimeError, 'fitted the G predictions'):
                rexp.reused_gscreen_predictions(run, src)

    def test_compare_rebuilds_the_split_and_joins_by_key(self):
        from scripts import stage1_tb3_compare as cmp
        months = [m(x) for x in ('2016-02', '2016-06', '2016-10', '2017-02', '2017-06', '2017-10', '2018-02')]
        snap = pd.DataFrame([(a, mo) for a in (1, 2, 3) for mo in months], columns=['area', 'target_month'])
        roles = cmp.rederived_tb3_roles(snap, 4, '2018-02')
        self.assertEqual(sorted(roles.loc[roles.role == 'validation', 'target_month'].unique()),
                         ['2016-10', '2017-02', '2017-06'])
        self.assertTrue((roles.target_month < '2017-10').all())       # O = 2017-10 and the target excluded
        with tempfile.TemporaryDirectory() as t:
            stage = Path(t)
            (stage / 'roots' / 'r').mkdir(parents=True); (stage / 'candidates' / 'c').mkdir(parents=True)
            members = pd.concat([roles, pd.DataFrame({'area': [1, 2, 3], 'target_month': '2018-02',
                                                      'role': 'heldout_target'})])
            rexp.write_csv_gz(stage / 'roots' / 'r' / 'fold_membership.csv.gz', members.assign(class_code=0))
            self.assertEqual(cmp.split_problems(stage, 'r', snap, 4, '2018-02'), [])
            flipped = members.copy(); flipped.loc[flipped.target_month == '2016-06', 'role'] = 'validation'
            rexp.write_csv_gz(stage / 'roots' / 'r' / 'fold_membership.csv.gz', flipped.assign(class_code=0))
            self.assertTrue(cmp.split_problems(stage, 'r', snap, 4, '2018-02'))
            rexp.write_csv_gz(stage / 'roots' / 'r' / 'fold_membership.csv.gz', members.assign(class_code=[0] * (len(members) - 3) + [2, 0, 3]))
            pd.DataFrame({'FEWSNET_admin_code': [1, 2, 3], 'y_true_code': [2, 0, 3], 'y_pred_pooled_code': [2, 0, 0]}).to_csv(
                stage / 'roots' / 'r' / 'root_target_predictions.csv', index=False)
            local = pd.DataFrame({'FEWSNET_admin_code': [3, 1, 2], 'y_true_code': [3, 2, 0],
                                  'y_pred_pooled_code': [0, 2, 0], 'y_pred_partitioned_code': [3, 2, 0], 'branch_id': '0'})
            local.to_csv(stage / 'candidates' / 'c' / 'target_predictions.csv', index=False)
            keyed = cmp.keyed_target(stage, 'r', 'c', '2018-02')       # different row order, joined by key
            self.assertEqual(dict(zip(keyed.area, keyed.y_local)), {1: 2, 2: 0, 3: 3})
            local.assign(y_true_code=[3, 2, 1]).to_csv(stage / 'candidates' / 'c' / 'target_predictions.csv', index=False)
            with self.assertRaises(cmp.CompareError):
                cmp.keyed_target(stage, 'r', 'c', '2018-02')
            local.iloc[:2].to_csv(stage / 'candidates' / 'c' / 'target_predictions.csv', index=False)
            with self.assertRaises(cmp.CompareError):
                cmp.keyed_target(stage, 'r', 'c', '2018-02')

    def test_tb3_locks_the_d26_g_selection(self):
        from scripts import run_stage1 as s1
        s1.require_tb3_g({'4': 'G1', '8': 'G4', '12': 'G2'})
        with self.assertRaises(SystemExit):
            s1.require_tb3_g({'4': 'G3', '8': 'G2', '12': 'G2'})


class SharedRootIncrement(unittest.TestCase):
    """D28 / experiment-plan A3: children continue the shared root once."""

    def _model(self, tmp, X, y):
        g, grec = nx.fit_global(X, y, SMALL_G['G1'])
        model = nx.XGBmodel(tmp, plan.L_CONFIGS['L1'], increment_source='root')
        model.set_root(g, grec)
        return model, g, grec

    def test_second_level_child_starts_from_root_and_search_budget_accumulates(self):
        X, y = synthetic()
        with tempfile.TemporaryDirectory() as tmp:
            model, g, grec = self._model(tmp, X, y)
            n0 = g.num_boosted_rounds()
            model.load('')
            model.train(X[:600], y[:600], '')
            model.save('0')
            first = dict(model.fit_record)
            self.assertEqual((first['actual_local_rounds'], first['path_selection_rounds']), (20, 20))
            parent_bytes = model._store['0'][0]
            model.load('0')
            model.train(X[:300], y[:300], '0')
            model.save('00')
            child = model.fit_record
            self.assertEqual(model.booster.num_boosted_rounds(), n0 + 20)       # root + 20, not parent + 20
            self.assertEqual(child['parent_sha256'], grec['booster_sha256'])
            self.assertEqual(child['child_prefix_structure_sha256'], grec['structure_sha256'])
            self.assertEqual(child['shared_source'], grec['booster_sha256'])
            self.assertEqual(child['routing_parent']['branch_id'], '0')
            self.assertEqual(child['routing_parent']['booster_sha256'], first['booster_sha256'])
            self.assertEqual((child['actual_local_rounds'], child['path_selection_rounds']), (20, 40))
            self.assertEqual(model._store['0'][0], parent_bytes)                 # current parent unchanged
            self.assertEqual(model.path_rounds('00'), 40)
            # parent-route copy keeps the parent's booster, predictions and search count
            model.load('0'); model.save('01')
            self.assertEqual(model._store['01'][0], parent_bytes)
            self.assertEqual(model.path_rounds('01'), 20)
            model.load('01'); p01 = model.predict(X)
            model.load('0'); np.testing.assert_array_equal(p01, model.predict(X))

    def test_search_cap_uses_selection_rounds_not_actual_rounds(self):
        X, y = synthetic()
        with tempfile.TemporaryDirectory() as tmp:
            model, _, _ = self._model(tmp, X, y)
            model.load('')
            model.train(X[:600], y[:600], '')
            model.fit_record['path_selection_rounds'] = 80       # deep accepted chain
            model.save('0101')
            self.assertEqual(model.path_rounds('0101'), 80)
            self.assertEqual(model.fit_record['actual_local_rounds'], 20)
            self.assertFalse(model.path_rounds('0101') + 20 <= plan.PATH_ROUND_CAP)

    def test_real_partition_in_root_mode(self):
        rng = np.random.default_rng(3)
        groups = np.repeat(np.arange(40), 30)
        X = rng.normal(size=(len(groups), 4)); X[:, 3] = groups >= 20
        y = ((X[:, 0] + 3 * X[:, 3] * X[:, 1]) > 0).astype(int) * 2
        months = np.tile(np.arange(30), 40)
        from src.utils.split import group_aware_train_val_split
        split = group_aware_train_val_split(groups, .5, 1, 42, True)['X_set']
        root, rec = nx.fit_global(X[split == 0], y[split == 0], SMALL_G['G1'])
        with tempfile.TemporaryDirectory() as tmp:
            model = nx.XGBmodel(tmp, plan.L_CONFIGS['L1'], increment_source='root')
            model.set_root(root, rec)
            _, decisions = run_partition(model, X, y, groups, split, months, threshold=Fraction(0))
            fits = model.fit_log[1:]
            self.assertTrue(fits)
            for entry in fits:
                self.assertEqual(entry['parent_sha256'], rec['booster_sha256'])   # always the shared root
                self.assertEqual(entry['rounds_total'], root.num_boosted_rounds() + 20)
                self.assertEqual(entry['actual_local_rounds'], 20)
                self.assertLessEqual(entry['path_selection_rounds'], plan.PATH_ROUND_CAP)
            self.assertTrue(decisions)

    def test_parent_mode_is_default_and_unchanged(self):
        self.assertEqual(nx.XGBmodel('.', plan.L_CONFIGS['L1']).increment_source, 'parent')
        with self.assertRaises(ValueError):
            nx.XGBmodel('.', plan.L_CONFIGS['L1'], increment_source='other')

    def production_root_mode(self, cap):
        rng = np.random.default_rng(11)
        groups = np.repeat(np.arange(64), 30)
        X = rng.normal(size=(len(groups), 5)); X[:, 3] = (groups % 8) >= 4; X[:, 4] = groups >= 32
        y = ((X[:, 0] + 3 * X[:, 3] * X[:, 1] - 3 * X[:, 4] * X[:, 2]) > 0).astype(int) * 2
        months = np.tile(np.arange(30), 64)
        from src.utils.split import group_aware_train_val_split
        split = group_aware_train_val_split(groups, .5, 1, 42, True)['X_set']
        root, rec = nx.fit_global(X[split == 0], y[split == 0], SMALL_G['G1'])
        tmp = tempfile.mkdtemp()
        model = nx.XGBmodel(tmp, plan.L_CONFIGS['L1'], increment_source='root')
        model.set_root(root, rec)
        (assigned, _, s_branch), decisions = run_partition(model, X, y, groups, split, months,
                                                           threshold=Fraction(0), cap=cap, max_depth=3)
        return model, decisions, rec

    def test_production_non_root_parent_fallback_and_exhausted_budget(self):
        model, decisions, rec = self.production_root_mode(cap=80)
        seen_fallback = 0
        for d in decisions:
            if d['outcome'] != 'accepted' or len(d['branch_id']) < 1:
                continue
            b = d['branch_id']                           # a NON-root parent
            for side, used in zip('01', d['selected_children']):
                model.load(b + side)
                child_bytes, child_rec = nx.raw(model.booster), dict(model.fit_record)
                if used:                                 # fresh child: root + exactly 20 rounds
                    self.assertEqual(model.booster.num_boosted_rounds(), 6 + 20)
                    self.assertEqual(child_rec['actual_local_rounds'], 20)
                    continue
                model.load(b)                            # parent-route fallback copies the parent
                self.assertEqual(child_bytes, nx.raw(model.booster))
                self.assertEqual(child_rec['path_selection_rounds'], model.fit_record['path_selection_rounds'])
                seen_fallback += 1
        self.assertGreater(seen_fallback, 0, 'fixture must exercise a non-root parent-route fallback')
        model, decisions, _ = self.production_root_mode(cap=20)
        capped = [d for d in decisions if 'path_round_cap' in (d.get('fallback') or [])]
        self.assertTrue(capped, 'fixture must exercise an exhausted search budget')
        for d in capped:                                 # blocked although the parent model has only 20 rounds
            model.load(d['branch_id'])
            self.assertEqual(model.fit_record['path_selection_rounds'], 20)
            self.assertLessEqual(model.booster.num_boosted_rounds(), 6 + 20)

    def test_six_rootinc_schedule_entries_with_distinct_names(self):
        from scripts import run_stage1 as s1
        obs = pd.DataFrame({'month': [m(t) for t in plan.STAGE1_TARGETS + ('2010-01', '2016-02', '2016-06',
                                                                          '2016-10', '2017-02', '2017-06', '2017-10')]
                                     + [m(f'{y}-{mo:02d}') for y in range(2021, 2025) for mo in (2, 6, 10)]})
        sched = prep.build_schedule(obs)                      # the production schedule builder
        self.assertEqual(sched['stage1_rootinc_counts']['roots'], 6)
        self.assertEqual(len(sched['stage1_candidates']), 648)
        with self.assertRaises(SystemExit):                   # a preparation without the D28 lists is refused
            s1.rootinc_entries({'stage1_roots': sched['stage1_roots']})
        roots = s1.scheduled_roots(sched, plan.TB3_G, plan.ROOTINC)
        cands = s1.scheduled_candidates(sched, plan.TB3_G, plan.ROOTINC)
        self.assertEqual((len(roots), len(cands)), (6, 6))
        old = {plan.root_name(h, t, g, r, sd) for h in plan.HORIZONS for t in plan.STAGE1_TARGETS
               for g in plan.G_CONFIGS for r in list(plan.SPLIT_RATIOS) + [plan.TIME_BLOCK] for sd in plan.SPLIT_SEEDS}
        self.assertFalse(set(roots) & old)
        for name, c in cands.items():
            parts = name.split('_')
            self.assertEqual((parts[3], parts[-1]), ('L1', 'gt0'))
            self.assertEqual((c['ratio'], c['split_seed'], c['increment_source']), ('r80', 42, 'root'))
        with self.assertRaises(SystemExit):
            s1.rootinc_entries({'stage1_rootinc_roots': []})


class ConfirmationDiagnostic(unittest.TestCase):
    """D29 (experiment-plan A4): label-blind S/C split, C isolation from the search, schedule."""

    def test_deterministic_label_blind_split(self):
        from src.utils.split import confirmation_split
        # area 30: 3 rows (odd), area 7: 4 (even), area 100: 1 (singleton), area 2: 5 (odd); input unsorted
        rows = [(30, 5), (7, 1), (100, 9), (2, 4), (30, 1), (7, 3), (2, 1), (7, 2), (2, 2), (30, 3), (7, 9),
                (2, 3), (2, 9)]
        g, mo = (np.array(c) for c in zip(*rows))
        role = confirmation_split(g, mo, 42)
        np.testing.assert_array_equal(role, confirmation_split(g, mo, 42))        # deterministic
        perm = np.random.default_rng(0).permutation(len(g))                       # input order irrelevant
        np.testing.assert_array_equal(confirmation_split(g[perm], mo[perm], 42), role[perm])
        self.assertLessEqual(abs(int((role == 0).sum()) - int((role == 1).sum())), 1)
        for a, n in ((30, 3), (7, 4), (100, 1), (2, 5)):
            self.assertIn(int((role[g == a] == 0).sum()), {n // 2, n // 2 + 1})
        self.assertEqual(int((role[g == 7] == 0).sum()), 2)                       # even area: exact half
        odd = [a for a in (2, 30, 100)]
        extra = sum(int((role[g == a] == 0).sum()) - (int((g == a).sum()) // 2) for a in odd)
        self.assertEqual(extra, len(odd) // 2)                                    # floor(n_odd/2) S extras
        self.assertEqual(len(confirmation_split(np.array([], int), np.array([], int))), 0)
        with self.assertRaises(ValueError):
            confirmation_split(np.array([1, 1]), np.array([3, 3]))

    def test_split_partitions_original_validation_and_keeps_fitting(self):
        from src.utils.split import confirmation_split, group_aware_train_val_split
        groups = np.repeat(np.arange(25), np.arange(1, 26) % 7 + 1)
        months = np.concatenate([np.arange(n) for n in np.arange(1, 26) % 7 + 1])
        x_set = group_aware_train_val_split(groups, .2, 1, 42, True)['X_set']
        val = np.flatnonzero(x_set == 1)
        role = confirmation_split(groups[val], months[val], 42)
        s, c = set(val[role == 0]), set(val[role == 1])
        self.assertFalse(s & c)
        self.assertEqual(s | c, set(val))
        self.assertEqual(set(np.flatnonzero(x_set == 0)) & (s | c), set())
        np.testing.assert_array_equal(group_aware_train_val_split(groups, .2, 1, 42, True)['X_set'], x_set)

    def test_c_labels_cannot_change_the_frozen_candidate_and_every_c_key_is_predicted(self):
        mgf = TimeBlockContrast.mgf()
        from src.utils.split import confirmation_split, group_aware_train_val_split
        rng = np.random.default_rng(11)
        groups = np.repeat(np.arange(64), 30)
        X = rng.normal(size=(len(groups), 5))
        y = ((X[:, 0] * np.where(groups < 40, 1, -1)) > 0).astype(int) * 2   # sign flips in areas 40-63
        months = np.tile(np.arange(400, 430), 64)
        x_set = np.asarray(group_aware_train_val_split(groups, .5, 1, 42, True)['X_set'])
        val = np.flatnonzero(x_set == 1)
        conf = np.zeros(len(groups), bool)
        conf[val[confirmation_split(groups[val], months[val], 42) == 1]] = True
        conf[(groups == 63) & (x_set == 1)] = True     # area 63: every validation row in C (no S rows)
        keep = ~conf
        root = nx.fit_global(X[x_set == 0], y[x_set == 0], SMALL_G['G1'])
        Xt, yt, gt = X[:64], y[:64], groups[::30]
        y_pool = fourclass.argmax_codes(nx.proba(root[0], Xt))
        data = (X[keep], y[keep], groups[keep], months[keep], x_set[keep], Xt, yt, gt, y_pool)
        outs = []
        for trial, yc in enumerate((y[conf], rng.permutation(y[conf]))):
            with tempfile.TemporaryDirectory() as t, patch.object(trans, 'CONTIGUITY', False), \
                    patch.object(trans, 'generate_count_grid', return_value=(None, 0, 1)), \
                    patch.dict(plan.FIT_SUPPORT, FLOORS['fit_support']), \
                    patch.dict(plan.STAGE1_VAL_SUPPORT, FLOORS['val_support']), \
                    patch.object(mgf, 'MAX_DEPTH', 3), redirect_stdout(StringIO()):
                work, ck = Path(t) / 'w', Path(t) / 'ck'
                work.mkdir()
                rec = mgf.run_candidate('c', 'L1', 'gt0', root, data, work, ck, None, [f'f{i}' for i in range(5)],
                                        increment_source='root',
                                        confirmation=(X[conf], yc, groups[conf], months[conf]))
                cp = pd.read_csv(work / 'c' / 'confirmation_predictions.csv.gz', converters={'branch_id': str})
                self.assertEqual(len(cp), int(conf.sum()))
                self.assertFalse(cp.duplicated(['area', 'target_month']).any())
                self.assertEqual(set(zip(cp['area'], cp['target_month'])),
                                 set(zip(groups[conf], ff.month_label(months[conf]))))
                self.assertIn(63, set(cp['area']))
                outs.append((rec['confirmation']['frozen_digest_before_scoring'], rec['checkpoints']['sha256'],
                             (work / 'c' / 's_branch.pkl').read_bytes(),
                             (work / 'c' / 'correspondence_table.csv').read_text(),
                             rec['partition']['decisions']))
                self.assertGreater(rec['partition']['accepted_splits'], 0, f"fixture must split {rec['partition']['decisions']} {rec['fits']['child_fits']}")
        self.assertEqual(outs[0][:4], outs[1][:4])
        self.assertEqual(json.dumps(outs[0][4], default=str), json.dumps(outs[1][4], default=str))

    def test_six_rootconf_schedule_entries_with_distinct_names(self):
        from scripts import run_stage1 as s1
        obs = pd.DataFrame({'month': [m(t) for t in plan.STAGE1_TARGETS + ('2010-01', '2016-02', '2016-06',
                                                                          '2016-10', '2017-02', '2017-06', '2017-10')]
                                     + [m(f'{y}-{mo:02d}') for y in range(2021, 2025) for mo in (2, 6, 10)]})
        sched = prep.build_schedule(obs)
        self.assertEqual(sched['stage1_rootconf_counts']['roots'], 6)
        roots = s1.scheduled_roots(sched, plan.TB3_G, plan.ROOTCONF)
        cands = s1.scheduled_candidates(sched, plan.TB3_G, plan.ROOTCONF)
        self.assertEqual((len(roots), len(cands)), (6, 6))
        self.assertFalse(set(roots) & set(s1.scheduled_roots(sched, plan.TB3_G, plan.ROOTINC)))
        self.assertFalse(set(cands) & set(s1.scheduled_candidates(sched, plan.TB3_G, plan.ROOTINC)))
        for name, c in cands.items():
            parts = name.split('_')
            self.assertEqual((parts[3], parts[-1]), ('L1', 'gt0'))
            self.assertEqual((c['ratio'], c['split_seed'], c['increment_source'], c['confirmation_seed']),
                             ('r80', 42, 'root', 42))
        with self.assertRaises(SystemExit):
            s1.rootinc_entries({'stage1_rootinc_roots': sched['stage1_rootinc_roots']}, plan.ROOTCONF)


class RecentSearchContrast(unittest.TestCase):
    """D30 (experiment-plan A5): search S limited to the latest six original-validation months."""

    def test_recent_months_deterministic_and_incomplete_raises(self):
        from src.utils.split import recent_search_months
        months = np.array([9, 3, 3, 12, 15, 1, 20, 7, 20, 25, 7])
        np.testing.assert_array_equal(recent_search_months(months), [7, 9, 12, 15, 20, 25])
        np.testing.assert_array_equal(recent_search_months(months[::-1]), [7, 9, 12, 15, 20, 25])
        with self.assertRaises(ValueError):
            recent_search_months(np.array([1, 2, 3, 4, 5, 5]))

    def _fixture(self):
        from src.utils.split import confirmation_split, group_aware_train_val_split
        rng = np.random.default_rng(11)
        groups = np.repeat(np.arange(64), 30)
        X = rng.normal(size=(len(groups), 5))
        y = ((X[:, 0] * np.where(groups < 40, 1, -1)) > 0).astype(int) * 2
        months = np.tile(np.arange(400, 430), 64)
        x_set = np.asarray(group_aware_train_val_split(groups, .5, 1, 42, True)['X_set'])
        val = np.flatnonzero(x_set == 1)
        conf = np.zeros(len(groups), bool)
        conf[val[confirmation_split(groups[val], months[val], 42) == 1]] = True
        return rng, groups, X, y, months, x_set, conf

    def _run(self, mgf, root, data, conf_data, val_support=None):
        with tempfile.TemporaryDirectory() as t, patch.object(trans, 'CONTIGUITY', False), \
                patch.object(trans, 'generate_count_grid', return_value=(None, 0, 1)), \
                patch.dict(plan.FIT_SUPPORT, FLOORS['fit_support']), \
                patch.dict(plan.STAGE1_VAL_SUPPORT, val_support or FLOORS['val_support']), \
                patch.object(mgf, 'MAX_DEPTH', 3), redirect_stdout(StringIO()):
            work, ck = Path(t) / 'w', Path(t) / 'ck'
            work.mkdir()
            rec = mgf.run_candidate('c', 'L1', 'gt0', root, data, work, ck, None, [f'f{i}' for i in range(5)],
                                    increment_source='root', confirmation=conf_data)
            return rec, ((rec['confirmation']['frozen_digest_before_scoring'], rec['checkpoints']['sha256'],
                          (work / 'c' / 's_branch.pkl').read_bytes(),
                          (work / 'c' / 'correspondence_table.csv').read_text(),
                          json.dumps(rec['partition']['decisions'], default=str)))

    def test_unused_and_c_labels_cannot_change_the_candidate(self):
        mgf = TimeBlockContrast.mgf()
        rng, groups, X, y, months, x_set, conf = self._fixture()
        expected = tuple(ff.month_label(np.arange(424, 430)).tolist())
        recent, unused = mgf.recent_search_roles(x_set, conf, months, expected)
        np.testing.assert_array_equal(recent, np.arange(424, 430))
        search = (x_set == 1) & ~conf & ~unused
        s_orig = (x_set == 1) & ~conf
        self.assertTrue(unused.any() and search.any())
        self.assertFalse((search & unused).any())
        np.testing.assert_array_equal(search | unused, s_orig)                  # S_recent u unused == S
        self.assertFalse(((x_set == 0) & (unused | conf)).any())                # fitting untouched
        with self.assertRaises(ValueError):
            mgf.recent_search_roles(x_set, conf, months, expected[1:] + ('2099-01',))
        root = nx.fit_global(X[x_set == 0], y[x_set == 0], SMALL_G['G1'])
        Xt, yt, gt = X[:64], y[:64], groups[::30]
        y_pool = fourclass.argmax_codes(nx.proba(root[0], Xt))
        keep = ~conf & ~unused
        outs = []
        for y_unused, y_conf in ((y, y), (np.where(unused, rng.permutation(y), y), y),
                                 (y, np.where(conf, rng.permutation(y), y))):
            ys = np.where(unused, y_unused, np.where(conf, y_conf, y))
            data = (X[keep], ys[keep], groups[keep], months[keep], x_set[keep], Xt, yt, gt, y_pool)
            rec, out = self._run(mgf, root, data, (X[conf], ys[conf], groups[conf], months[conf]))
            self.assertGreater(rec['partition']['accepted_splits'], 0)
            outs.append(out)
        self.assertEqual(outs[0], outs[1])
        self.assertEqual(outs[0], outs[2])

    def test_insufficient_recent_support_keeps_the_parent(self):
        mgf = TimeBlockContrast.mgf()
        rng, groups, X, y, months, x_set, conf = self._fixture()
        root = nx.fit_global(X[x_set == 0], y[x_set == 0], SMALL_G['G1'])
        Xt, yt, gt = X[:64], y[:64], groups[::30]
        y_pool = fourclass.argmax_codes(nx.proba(root[0], Xt))
        # only one recent date survives: below the 3-date validation floor on every side
        unused = (x_set == 1) & ~conf & (months != 429)
        keep = ~conf & ~unused
        data = (X[keep], y[keep], groups[keep], months[keep], x_set[keep], Xt, yt, gt, y_pool)
        rec, _ = self._run(mgf, root, data, (X[conf], y[conf], groups[conf], months[conf]),
                           val_support={**FLOORS['val_support'], 'dates': 3})
        self.assertEqual(rec['partition']['accepted_splits'], 0)
        self.assertEqual(rec['fits']['child_fits'], 0)

    def test_six_recentsearch_schedule_entries_with_distinct_names(self):
        from scripts import run_stage1 as s1
        obs = pd.DataFrame({'month': [m(t) for t in plan.STAGE1_TARGETS + ('2010-01', '2016-02', '2016-06',
                                                                          '2016-10', '2017-02', '2017-06', '2017-10')]
                                     + [m(f'{y}-{mo:02d}') for y in range(2021, 2025) for mo in (2, 6, 10)]})
        sched = prep.build_schedule(obs)
        self.assertEqual(sched['stage1_recentsearch_counts']['roots'], 6)
        roots = s1.scheduled_roots(sched, plan.TB3_G, plan.RECENTSEARCH)
        cands = s1.scheduled_candidates(sched, plan.TB3_G, plan.RECENTSEARCH)
        self.assertEqual((len(roots), len(cands)), (6, 6))
        for other in (plan.ROOTINC, plan.ROOTCONF):
            self.assertFalse(set(roots) & set(s1.scheduled_roots(sched, plan.TB3_G, other)))
            self.assertFalse(set(cands) & set(s1.scheduled_candidates(sched, plan.TB3_G, other)))
        for name, c in cands.items():
            self.assertTrue(name.endswith('_recentsearch_gt0'))
            self.assertIn((c['horizon'], c['target_month']), plan.RECENT_SEARCH_DATES)
            self.assertEqual((c['ratio'], c['split_seed'], c['increment_source'], c['confirmation_seed'],
                              c['recent_search_months']), ('r80', 42, 'root', 42, 6))
        with self.assertRaises(SystemExit):
            s1.rootinc_entries({'stage1_rootconf_roots': sched['stage1_rootconf_roots']}, plan.RECENTSEARCH)


    def test_compare_recent_membership_matches_d29_and_rejects_moved_rows(self):
        """The D30 reporter accepts exactly D29 fitting/C/target with S split into S_recent +
        unused by the A5 months, and refuses an unused row moved into C or a non-A5 calendar."""
        from scripts import stage1_rootconf_compare as rc
        h, t = 4, "2018-02"
        recent = list(plan.RECENT_SEARCH_DATES[(h, t)])
        older = ["2014-06", "2014-10", "2015-02", "2015-06"]
        rows = []
        for area in range(6):
            for i, m in enumerate(older + recent):
                k = (area + i) % 3   # every month has fitting, S and C rows across areas
                role = ("fitting" if k == 0 else "confirmation" if k == 1 else "validation")
                rows.append({"area": area, "target_month": m, "role": role, "class_code": (area + i) % 4})
            rows.append({"area": area, "target_month": t, "role": "heldout_target", "class_code": area % 4})
        old = pd.DataFrame(rows)
        new = old.copy()
        new.loc[(new["role"] == "validation") & ~new["target_month"].isin(recent), "role"] = "unused_search_history"
        target = pd.DataFrame({"area": range(6), "target_month": t, "y_true": [a % 4 for a in range(6)],
                               "y_root": [0, 1, 2, 3, 0, 1]})
        with tempfile.TemporaryDirectory() as tmp:
            def stage(name, members):
                d = Path(tmp) / name / "roots" / "r"
                d.mkdir(parents=True)
                members.to_csv(d / "fold_membership.csv.gz", index=False)
                (d / "root.json").write_text(json.dumps({"root_booster_sha256": "same"}), encoding="utf-8")
                return Path(tmp) / name
            new_stage, old_stage = stage("new", new), stage("old", old)
            got = rc.same_roots_recent(new_stage, "r", old_stage, "r", target, target, h, t)
            self.assertEqual(int((got["role"] == "unused_search_history").sum()),
                             int(((old["role"] == "validation") & ~old["target_month"].isin(recent)).sum()))
            moved = new.copy()
            moved.loc[moved.index[moved["role"] == "unused_search_history"][0], "role"] = "confirmation"
            moved.to_csv(new_stage / "roots" / "r" / "fold_membership.csv.gz", index=False)
            with self.assertRaises(rc.CompareError):
                rc.same_roots_recent(new_stage, "r", old_stage, "r", target, target, h, t)
            new.to_csv(new_stage / "roots" / "r" / "fold_membership.csv.gz", index=False)
            with self.assertRaises(rc.CompareError):
                rc.same_roots_recent(new_stage, "r", old_stage, "r", target, target, 4, "2020-10")

class MatchedSearchControl(unittest.TestCase):
    """D31 (experiment-plan A6): per-area matched-size search drawn from all original S dates."""

    @staticmethod
    def _reference(groups, months, k_by_area, seed):
        import random
        rng = random.Random(seed)
        out = set()
        for area in sorted(set(groups.tolist())):
            rows = sorted(((int(mo), i) for i, (g, mo) in enumerate(zip(groups, months)) if g == area))
            keys = [(area, mo) for mo, _ in rows]
            rng.shuffle(keys)                       # consumed for every area, k_a = 0 included
            out.update(keys[:k_by_area.get(area, 0)])
        return out

    def test_sampler_deterministic_order_free_exact_counts(self):
        from src.utils.split import matched_size_sample
        rng = np.random.default_rng(3)
        groups, months = [], []
        for area in range(40):
            n = int(rng.integers(1, 5))
            groups += [area * 7] * n
            months += sorted(rng.choice(np.arange(100, 130), n, replace=False).tolist())
        groups, months = np.array(groups), np.array(months)
        n_a = dict(zip(*np.unique(groups, return_counts=True)))
        k = {int(a): int(rng.integers(0, n + 1)) for a, n in n_a.items()}
        k[0] = 0
        k[7] = n_a[7]
        self.assertTrue(any(0 < k[a] < n_a[a] for a in n_a))            # genuine sampling choices
        sel = matched_size_sample(groups, months, k, 101)
        np.testing.assert_array_equal(sel, matched_size_sample(groups, months, k, 101))
        for a in n_a:
            self.assertEqual(int(sel[groups == a].sum()), k[a])
        got = set(zip(groups[sel].tolist(), months[sel].tolist()))
        self.assertEqual(got, self._reference(groups, months, k, 101))
        perm = rng.permutation(len(groups))
        sel_p = matched_size_sample(groups[perm], months[perm], k, 101)
        self.assertEqual(set(zip(groups[perm][sel_p].tolist(), months[perm][sel_p].tolist())), got)
        draws = {frozenset(zip(groups[m].tolist(), months[m].tolist()))
                 for m in (matched_size_sample(groups, months, k, s) for s in plan.MATCHED_SEEDS)}
        self.assertEqual(len(draws), 3)
        with self.assertRaises(ValueError):
            matched_size_sample(groups, months, {**k, 7: n_a[7] + 1}, 101)
        with self.assertRaises(ValueError):
            matched_size_sample(np.r_[groups, groups[:1]], np.r_[months, months[:1]], k, 101)

    def test_unused_and_c_labels_cannot_change_the_candidate(self):
        mgf = TimeBlockContrast.mgf()
        rs = RecentSearchContrast()
        rng, groups, X, y, months, x_set, conf = rs._fixture()
        expected = tuple(ff.month_label(np.arange(424, 430)).tolist())
        recent, d30_unused = mgf.recent_search_roles(x_set, conf, months, expected)
        _, sampled = mgf.matched_search_roles(x_set, conf, groups, months, expected, 102)
        s_orig = (x_set == 1) & ~conf
        unused = s_orig & ~sampled
        self.assertFalse((sampled & ~s_orig).any())
        np.testing.assert_array_equal(sampled | unused, s_orig)                 # sample u unused == S
        recent_s = s_orig & ~d30_unused
        for a in np.unique(groups[s_orig]):
            self.assertEqual(int((sampled & (groups == a)).sum()), int((recent_s & (groups == a)).sum()))
        self.assertFalse(np.array_equal(sampled, recent_s))                     # earlier S dates are drawn
        self.assertFalse(((x_set == 0) & (unused | conf | sampled)).any())      # fitting untouched
        root = nx.fit_global(X[x_set == 0], y[x_set == 0], SMALL_G['G1'])
        Xt, yt, gt = X[:64], y[:64], groups[::30]
        y_pool = fourclass.argmax_codes(nx.proba(root[0], Xt))
        keep = ~conf & ~unused
        outs = []
        for y_unused, y_conf in ((y, y), (np.where(unused, rng.permutation(y), y), y),
                                 (y, np.where(conf, rng.permutation(y), y))):
            ys = np.where(unused, y_unused, np.where(conf, y_conf, y))
            data = (X[keep], ys[keep], groups[keep], months[keep], x_set[keep], Xt, yt, gt, y_pool)
            rec, out = rs._run(mgf, root, data, (X[conf], ys[conf], groups[conf], months[conf]))
            self.assertGreater(rec['partition']['accepted_splits'], 0)
            outs.append(out)
        self.assertEqual(outs[0], outs[1])
        self.assertEqual(outs[0], outs[2])

    def test_eighteen_matchedsize_schedule_entries_with_distinct_names(self):
        from scripts import run_stage1 as s1
        obs = pd.DataFrame({'month': [m(t) for t in plan.STAGE1_TARGETS + ('2010-01', '2016-02', '2016-06',
                                                                          '2016-10', '2017-02', '2017-06', '2017-10')]
                                     + [m(f'{y}-{mo:02d}') for y in range(2021, 2025) for mo in (2, 6, 10)]})
        sched = prep.build_schedule(obs)
        self.assertEqual(sched['stage1_matchedsize_counts']['roots'], 18)
        self.assertEqual(len(sched['stage1_recentsearch_roots']), 6)
        roots = s1.scheduled_roots(sched, plan.TB3_G, plan.MATCHEDSIZE)
        cands = s1.scheduled_candidates(sched, plan.TB3_G, plan.MATCHEDSIZE)
        self.assertEqual((len(roots), len(cands)), (18, 18))
        for other in (plan.ROOTINC, plan.ROOTCONF, plan.RECENTSEARCH):
            self.assertFalse(set(roots) & set(s1.scheduled_roots(sched, plan.TB3_G, other)))
            self.assertFalse(set(cands) & set(s1.scheduled_candidates(sched, plan.TB3_G, other)))
        for name, c in cands.items():
            parts = name.split('_')
            self.assertEqual((parts[3], parts[-1]), ('L1', 'gt0'))
            self.assertTrue(name.endswith(f"_matchedsize_m{c['matched_size_seed']}_gt0"))
            self.assertIn(c['matched_size_seed'], plan.MATCHED_SEEDS)
            self.assertEqual(c['root'], f"{plan.root_name(c['horizon'], c['target_month'], c['g_config'], 'r80', 42)}"
                                        f"_matchedsize_m{c['matched_size_seed']}")
            self.assertEqual((c['ratio'], c['split_seed'], c['increment_source'], c['confirmation_seed'],
                              c['recent_search_months']), ('r80', 42, 'root', 42, 6))
        with self.assertRaises(SystemExit):
            s1.rootinc_entries({'stage1_matchedsize_roots': sched['stage1_recentsearch_roots']}, plan.MATCHEDSIZE)
        with self.assertRaises(SystemExit):
            s1.rootinc_entries({'stage1_recentsearch_roots': sched['stage1_matchedsize_roots'][:6]},
                               plan.RECENTSEARCH)


class MatchedSearchReporter(unittest.TestCase):
    """D31 (A6) reporter: membership re-derivation and the pure row function, end to end
    on fixture frames (the D30 crash sat in row assembly, untested before)."""

    H, T, SEED = 4, "2018-02", 101

    def _memberships(self):
        from app import main_model_GF
        recent = list(plan.RECENT_SEARCH_DATES[(self.H, self.T)])
        months = ["2014-06", "2014-10", "2015-02", "2015-06"] + recent
        rows = []
        for area in range(10):
            for i, m in enumerate(months):
                k = (area + i) % 3
                rows.append({"area": area, "target_month": m, "class_code": (area * 3 + i) % 4,
                             "role": "fitting" if k == 0 else "confirmation" if k == 1 else "validation"})
            rows.append({"area": area, "target_month": self.T, "role": "heldout_target", "class_code": area % 4})
        d29 = pd.DataFrame(rows)
        d30 = d29.copy()
        d30.loc[(d30["role"] == "validation") & ~d30["target_month"].isin(recent), "role"] = "unused_search_history"
        # D31 roles through the PRODUCER helper (independent of the reporter's re-derivation)
        tr = d29[d29["role"] != "heldout_target"].reset_index(drop=True)
        idx = pd.PeriodIndex(tr["target_month"], freq="M")
        mtrain = (idx.year * 12 + idx.month - 1).to_numpy(np.int64)
        x_set = tr["role"].isin(["validation", "confirmation"]).to_numpy().astype(int)
        conf = (tr["role"] == "confirmation").to_numpy()
        _, sampled = main_model_GF.matched_search_roles(x_set, conf, tr["area"].to_numpy(np.int64), mtrain,
                                                        plan.RECENT_SEARCH_DATES[(self.H, self.T)], self.SEED)
        role = tr["role"].where(~((tr["role"] == "validation") & ~sampled), "unused_search_history")
        new = pd.concat([tr.assign(role=role), d29[d29["role"] == "heldout_target"]], ignore_index=True)
        return d29, d30, new

    def test_membership_rederivation_accepts_producer_draw_and_rejects_changes(self):
        from scripts import stage1_rootconf_compare as rc
        d29, d30, new = self._memberships()
        s = new[new["role"].isin(["validation", "unused_search_history"])]
        self.assertTrue(0 < (s["role"] == "validation").sum() < len(s))
        rc.same_roots_matched(new, d29, d30, self.H, self.T, self.SEED, "fixture")
        with self.assertRaises(rc.CompareError):     # another search seed draws other rows
            rc.same_roots_matched(new, d29, d30, self.H, self.T, 103, "fixture")
        moved = new.copy()
        i = moved.index[moved["role"] == "validation"][0]
        j = moved.index[(moved["role"] == "unused_search_history") & (moved["area"] == moved.at[i, "area"])]
        moved.loc[i, "role"] = "unused_search_history"
        if len(j):
            moved.loc[j[0], "role"] = "validation"     # same per-area count, different rows
        with self.assertRaises(rc.CompareError):
            rc.same_roots_matched(moved, d29, d30, self.H, self.T, self.SEED, "fixture")

    def test_matched_row_end_to_end_on_fixture_frames(self):
        from scripts import stage1_rootconf_compare as rc
        d29, d30, new = self._memberships()
        rng = np.random.default_rng(0)

        def preds(members, role):
            f = members[members["role"] == role][["area", "target_month", "class_code"]].rename(
                columns={"class_code": "y_true"}).reset_index(drop=True)
            f["y_root"] = (f["y_true"] + (f.index % 3 == 0)) % 4
            f["y_final"] = np.where(rng.random(len(f)) < 0.7, f["y_true"], f["y_root"])
            f["branch_id"] = np.where(f["area"] % 2 == 0, "1", "root")
            f["routing"] = "terminal_branch"
            return f

        conf_new = preds(new, "confirmation")
        target = pd.DataFrame({"area": range(10), "target_month": self.T, "y_true": [a % 4 for a in range(10)],
                               "y_root": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1], "y_local": [0, 1, 2, 2, 0, 1, 3, 3, 0, 1]})
        mat = fourclass.confusion(target["y_true"], target["y_root"])

        def run(members, e3, name, confirmation, validation=None):
            row = {"candidate": name, "e3_root_crisis_f1": 0.5, "e3_local_crisis_f1": 0.5 + float(e3),
                   "e3_local_minus_root": float(e3), "e3_local_minus_root_exact": str(e3), "e3_root_fourclass": 0.4,
                   "e3_local_fourclass": 0.41, "n_terminal": 3, "accepted_splits": 2,
                   "distinct_terminal_boosters": 3, "e4_weight": 0.0}
            out = {"row": row, "target": target, "conf": {"target_root": mat, "target_local": mat},
                   "partition": ((0, 1, 2), (0, 1, 1)), "members": members, "confirmation": confirmation}
            if validation is not None:
                out["validation"] = validation
            return out

        c_old = conf_new.assign(y_final=conf_new["y_root"])       # D30/D29 local == root on C
        data = {"seed": self.SEED,
                "new": dict(run(new, Fraction(1, 50), "new", conf_new, preds(new, "validation")),
                            rounds={"deployed_rounds_after_root_max": 20, "search_budget_rounds_max": 80},
                            root_booster="R", branch_booster={"1": "R", "root": "R"}),
                "d30": run(d30, Fraction(-1, 100), "d30", c_old),
                "d29": run(d29, Fraction(-1, 40), "d29", c_old, preds(d29, "validation"))}
        # D29 S predictions must carry the same root codes on the sample keys
        data["new"]["validation"] = data["new"]["validation"].drop(columns="y_root").merge(
            data["d29"]["validation"][["area", "target_month", "y_root"]], on=["area", "target_month"])
        row, mats, terms = rc.matched_row(data, self.H, self.T, "new")
        gain = (fourclass.crisis_f1_exact(conf_new["y_true"], conf_new["y_final"])
                - fourclass.crisis_f1_exact(conf_new["y_true"], conf_new["y_root"]))
        self.assertAlmostEqual(row["C_local_minus_root"], float(gain), places=12)
        self.assertEqual(row["d30_C_local_minus_root"], 0.0)
        self.assertAlmostEqual(row["C_minus_d30"], float(gain), places=12)
        self.assertAlmostEqual(row["e3_minus_d30_e3"], 0.03, places=12)
        self.assertEqual(row["S_sample_rows"], int((new["role"] == "validation").sum()))
        self.assertIn("d30_target_local", mats)
        self.assertEqual((row["deployed_rounds_after_root_max"], row["search_budget_rounds_max"]), (20, 80))
        self.assertEqual(row["C_rows_root_booster"], len(conf_new))     # branch "1" inherited the root booster
        self.assertEqual(row["C_rows_on_root_branch_name"], int((conf_new["branch_id"] == "root").sum()))
        self.assertLess(row["C_rows_on_root_branch_name"], row["C_rows_root_booster"])
        self.assertEqual({t["set"] for t in terms}, {"S_sample", "C", "C_recent6", "C_older"})
        self.assertEqual(sum(t["rows"] for t in terms if t["set"] == "C"), len(conf_new))
        bad = dict(data, d30=dict(data["d30"], confirmation=c_old.assign(y_true=(c_old["y_true"] + 1) % 4)))
        with self.assertRaises(rc.CompareError):
            rc.matched_row(bad, self.H, self.T, "new")

class AssignmentEvidence(unittest.TestCase):
    """D32/A7: Stage 1 assignment-evidence export (routing unchanged; spatial support = search rows)."""

    def _run(self, conf_labels_permuted=False, val_support=None):
        import hashlib
        mgf = TimeBlockContrast.mgf()
        rng, groups, X, y, months, x_set, conf = RecentSearchContrast()._fixture()
        conf[(groups == 63) & (x_set == 1)] = True       # area 63: fitting rows, no search rows
        root = nx.fit_global(X[x_set == 0], y[x_set == 0], SMALL_G['G1'])
        # target rows: every fixture area plus area 999 (target only)
        Xt = np.vstack([X[:64], X[:2]])
        yt = np.concatenate([y[:64], y[:2]])
        gt = np.concatenate([groups[::30], [999, 999]])
        y_pool = fourclass.argmax_codes(nx.proba(root[0], Xt))
        keep = ~conf
        yc = rng.permutation(y[conf]) if conf_labels_permuted else y[conf]
        data = (X[keep], y[keep], groups[keep], months[keep], x_set[keep], Xt, yt, gt, y_pool)
        with tempfile.TemporaryDirectory() as t, patch.object(trans, 'CONTIGUITY', False), \
                patch.object(trans, 'generate_count_grid', return_value=(None, 0, 1)), \
                patch.dict(plan.FIT_SUPPORT, FLOORS['fit_support']), \
                patch.dict(plan.STAGE1_VAL_SUPPORT, val_support or FLOORS['val_support']), \
                patch.object(mgf, 'MAX_DEPTH', 3), redirect_stdout(StringIO()):
            work, ck = Path(t) / 'w', Path(t) / 'ck'
            work.mkdir()
            rec = mgf.run_candidate('c', 'L1', 'gt0', root, data, work, ck, None, [f'f{i}' for i in range(5)],
                                    increment_source='root',
                                    confirmation=(X[conf], yc, groups[conf], months[conf]))
            raw = (work / 'c' / 'assignment_evidence.csv').read_bytes()
            ev = pd.read_csv(work / 'c' / 'assignment_evidence.csv',
                             converters={'prediction_branch_id': str, 'spatial_partition_id': str})
            corr = pd.read_csv(work / 'c' / 'correspondence_table.csv', converters={'partition_id': str})
            ckpt_sha = dict(rec['checkpoints']['sha256'])
        tr = (groups[keep], x_set[keep])
        return mgf, rec, raw, hashlib.sha256(raw).hexdigest(), ev, corr, tr, gt, groups[conf], root, ckpt_sha

    def _check_counts(self, ev, tr, gt, gc):
        g, xs = tr
        universe = sorted(set(g.tolist()) | set(gt.tolist()) | set(gc.tolist()))
        self.assertEqual(ev['FEWSNET_admin_code'].tolist(), universe)             # 1:1 universe, sorted
        for _, r in ev.iterrows():
            a = r['FEWSNET_admin_code']
            self.assertEqual(r['search_rows'], int(((g == a) & (xs == 1)).sum()))
            self.assertEqual(r['fitting_rows'], int(((g == a) & (xs == 0)).sum()))
            self.assertEqual(r['target_rows'], int((gt == a).sum()))
            self.assertEqual(r['confirmation_rows'], int((gc == a).sum()))
        self.assertTrue(((ev['spatial_partition_id'] == 's-1') == (ev['search_rows'] == 0)).all())
        searched = ev['search_rows'] >= 1
        self.assertTrue((ev.loc[searched, 'spatial_partition_id'] == ev.loc[searched, 'prediction_branch_id']).all())

    def test_split_candidate_statuses_counts_and_contracts(self):
        mgf, rec, raw, sha, ev, corr, tr, gt, gc, root, _ = self._run()
        self.assertGreater(rec['partition']['accepted_splits'], 0)
        self._check_counts(ev, tr, gt, gc)
        st = dict(zip(ev['FEWSNET_admin_code'], ev['assignment_status']))
        self.assertEqual(st[63], 'unsearched_fit_fallback')
        r63 = ev.set_index('FEWSNET_admin_code').loc[63]
        self.assertGreater(r63['fitting_rows'], 0)
        self.assertGreater(r63['target_rows'], 0)
        self.assertEqual(st[999], 'target_only_fallback')
        self.assertEqual(ev.set_index('FEWSNET_admin_code').loc[999, 'prediction_branch_id'], 'root')
        self.assertIn('searched_assigned', set(st.values()))
        # routing column equals the unchanged correspondence export on every training area
        lookup = dict(zip(corr['FEWSNET_admin_code'], corr['partition_id']))
        for a, b in zip(ev['FEWSNET_admin_code'], ev['prediction_branch_id']):
            if a in lookup:
                self.assertEqual(lookup[a], b)
        # candidate.json contracts
        self.assertEqual(rec['routing_export']['file'], 'correspondence_table.csv')
        ae = rec['assignment_evidence']
        self.assertEqual((ae['schema'], ae['file'], ae['sha256']), ('d32-v1', 'assignment_evidence.csv', sha))
        self.assertEqual(sum(ae['areas_by_status'].values()), len(ev))
        self.assertEqual(ae['areas_by_status']['target_only_fallback'], 1)
        self.assertEqual(sum(ae['target_rows_by_status'].values()), len(gt))
        # routed_booster_is_root: derived from the saved log, root terminals true
        last = {}
        for e in rec['fits']['saved_log']:
            last[e['saved_as']] = e['booster_sha256']
        for _, r in ev.iterrows():
            self.assertEqual(bool(r['routed_booster_is_root']),
                             r['prediction_branch_id'] == 'root'
                             or last.get(r['prediction_branch_id']) == root[1]['booster_sha256'])

    def test_c_labels_leave_the_evidence_file_identical(self):
        first = self._run()[2]
        self.assertEqual(first, self._run(conf_labels_permuted=True)[2])

    def test_unsplit_candidate_is_searched_root(self):
        _, rec, _, _, ev, _, tr, gt, gc, _, _ = self._run(val_support={**FLOORS['val_support'], 'rows': 10 ** 6})
        self.assertEqual(rec['partition']['accepted_splits'], 0)
        self._check_counts(ev, tr, gt, gc)
        st = set(ev.loc[ev['search_rows'] >= 1, 'assignment_status'])
        self.assertEqual(st, {'searched_root'})
        self.assertTrue(ev['routed_booster_is_root'].all())

    def test_pure_function_confirmation_only_and_root_booster_copy(self):
        mgf = TimeBlockContrast.mgf()
        gtrain = np.array([1, 1, 2, 2, 3])
        x_set = np.array([1, 0, 1, 0, 0])
        gtest = np.array([1, 4, 6])
        s_branch = pd.DataFrame({'0': [1.0], '1': [2.0]})          # columns = branch ids, values = groups
        ev = mgf.assignment_evidence(gtrain, x_set, gtest, s_branch, conf_groups=np.array([5, 5, 3, 6]),
                                     branch_booster={'0': 'ROOT', '1': 'other'}, root_booster_sha='ROOT')
        ev = ev.set_index('FEWSNET_admin_code')
        self.assertEqual(ev.loc[5, 'assignment_status'], 'confirmation_only_fallback')
        self.assertEqual(ev.loc[5, 'confirmation_rows'], 2)
        self.assertEqual(ev.loc[3, 'assignment_status'], 'unsearched_fit_fallback')
        self.assertEqual(ev.loc[4, 'assignment_status'], 'target_only_fallback')
        # mixed C + target, no search/fit: target-only relative to the search input, C rows still counted
        self.assertEqual(ev.loc[6, 'assignment_status'], 'target_only_fallback')
        self.assertEqual((ev.loc[6, 'confirmation_rows'], ev.loc[6, 'target_rows'], ev.loc[6, 'spatial_partition_id']), (1, 1, 's-1'))
        self.assertEqual(ev.loc[1, 'assignment_status'], 'searched_assigned')
        self.assertTrue(ev.loc[1, 'routed_booster_is_root'])          # named branch with a root booster copy
        self.assertFalse(ev.loc[2, 'routed_booster_is_root'])
        self.assertEqual(ev.loc[1, 'spatial_partition_id'], '0')
        self.assertEqual(ev.loc[3, 'spatial_partition_id'], 's-1')
        no_flag = mgf.assignment_evidence(gtrain, x_set, gtest, s_branch)
        self.assertNotIn('routed_booster_is_root', no_flag.columns)

    def test_completion_chain_requires_evidence_only_when_declared(self):
        from scripts import run_stage1 as s1
        old = {'candidate': 'c', 'checkpoints': {'sha256': {}}}
        self.assertNotIn('assignment_evidence.csv', s1.candidate_files(old, False))
        self.assertNotIn('assignment_evidence.csv', s1.candidate_files(old, True))
        self.assertEqual(s1.candidate_files(old, False), s1.CANDIDATE_FILES)
        new = {**old, 'assignment_evidence': {'schema': 'd32-v1'}}
        self.assertIn('assignment_evidence.csv', s1.candidate_files(new, True))
        self.assertNotIn('assignment_evidence.csv', s1.CANDIDATE_FILES)


class ShallowReplay(unittest.TestCase):
    """D33/A8: depth-1 truncated routing, root-decision save order, exact gate comparator."""

    def setUp(self):
        from scripts import stage1_shallow_replay as sr
        self.sr = sr

    def test_truncated_routing(self):
        s_branch = pd.DataFrame({'': [1, 2, 3, 4, 5, 6], '0': [1, 2, 3, np.nan, np.nan, np.nan],
                                 '1': [4, 5, np.nan, np.nan, np.nan, np.nan],
                                 '00': [1, 2, np.nan, np.nan, np.nan, np.nan], '10': [4, np.nan, np.nan, np.nan,
                                                                                      np.nan, np.nan]})
        routes = self.sr.depth1_routes(np.array([1, 2, 3, 4, 5, 6, 9]), s_branch)
        self.assertEqual(routes.tolist(), ['0', '0', '0', '1', '1', '', ''])
        full = self.sr.get_X_branch_id_by_group(np.array([1, 4]), s_branch)
        self.assertEqual(full.tolist(), ['00', '10'])

    def test_root_decision_save_order(self):
        decisions = [{'branch_id': '', 'outcome': 'accepted', 'selected_children': [True, False]}]
        log = [{'saved_as': 'root', 'booster_sha256': 'R'}, {'saved_as': '0', 'booster_sha256': 'A'},
               {'saved_as': '1', 'booster_sha256': 'R'}, {'saved_as': '00', 'booster_sha256': 'B'}]
        out = self.sr.check_root_decision(log, decisions, 'R')
        self.assertEqual(out['sha'], {'0': 'A', '1': 'R'})
        self.assertEqual(out['root_copy_sides'], ['1'])
        with self.assertRaises(self.sr.GateError):
            self.sr.check_root_decision(log + [{'saved_as': '1', 'booster_sha256': 'C'}], decisions, 'R')
        with self.assertRaises(self.sr.GateError):
            self.sr.check_root_decision(log, [{'branch_id': '', 'outcome': 'rejected_gate'}], 'R')
        with self.assertRaises(self.sr.GateError):      # root copy on side 1 but the decision kept both locals
            self.sr.check_root_decision(log, [{'branch_id': '', 'outcome': 'accepted',
                                                'selected_children': [True, True]}], 'R')
        self.assertEqual(self.sr.row_equal(np.array([[0.1, 0.2], [0.3, 0.4]]),
                                           np.array([[0.1, 0.2], [0.3, np.nextafter(0.4, 1)]])).tolist(), [True, False])

    def test_gate_comparator_exact(self):
        a = np.array([[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(self.sr.exact_mismatches(a, a.copy()), 0)
        b = a.copy()
        b[0, 1] = np.nextafter(b[0, 1], 1.0)
        self.assertEqual(self.sr.exact_mismatches(a, b), 1)
        self.assertEqual(self.sr.exact_mismatches(np.array([np.nan]), np.array([np.nan])), 1)
        self.assertGreater(self.sr.exact_mismatches(a, a[:1]), 0)


class E1BrierContrast(unittest.TestCase):
    """D34 (experiment-plan A9): E1 hard crisis F1 vs crisis Brier loss, one shared root."""

    def test_brier_masses_formula(self):
        rng = np.random.default_rng(3)
        y = rng.integers(0, 4, 200)
        p = rng.uniform(-.2, 1.2, 200)
        g = rng.integers(10, 25, 200)
        groups, Y, A = fourclass.brier_crisis_scan_masses(y, p, g)
        np.testing.assert_array_equal(groups, np.unique(g))
        self.assertEqual((Y.shape, A.shape), ((len(groups), 1), (len(groups), 1)))
        z = (y >= 2).astype(float)
        loss = (np.clip(p, 0, 1) - z) ** 2
        for k, grp in enumerate(groups):
            self.assertAlmostEqual(Y[k, 0], np.sum(g == grp) / 200)
            self.assertAlmostEqual(A[k, 0], (np.sum(g == grp) - loss[g == grp].sum()) / 200)
        self.assertTrue(np.all(Y >= A) and np.all(Y - A >= 0))
        self.assertAlmostEqual(Y.sum(), 1.0)
        self.assertAlmostEqual((Y - A).sum(), loss.sum() / 200)
        _, Y0, A0 = fourclass.brier_crisis_scan_masses(y, (y >= 2).astype(float), g)
        self.assertEqual(np.sum(Y0 - A0), 0.0)                 # perfect -> zero error mass

    def test_scan_tie_diagnostics(self):
        g = np.array([3., 1., 1., 1., 0., 0., 2.])
        c = np.array([[1.], [0.], [.5], [0.], [0.], [0.], [1.]])
        s0, s1 = np.array([0, 6, 1]), np.array([2, 3, 4, 5])
        d = trans.scan_tie_diagnostics(g, s0, s1, c, 'hard_f1')
        self.assertEqual((d['n_groups'], d['c_zero_groups'], d['g_zero_groups'], d['distinct_g']), (7, 4, 2, 4))
        self.assertEqual((d['boundary_tie_block'], d['cut_crosses_tie_block'], d['tie_block_in_s0'],
                          d['tie_block_in_s1']), (3, True, 1, 2))
        self.assertEqual((d['pre_refinement_s0_groups'], d['pre_refinement_s1_groups']), (3, 4))
        d = trans.scan_tie_diagnostics(g, np.array([0, 6]), np.array([1, 2, 3, 4, 5]), c, 'brier_crisis')
        self.assertEqual((d['boundary_tie_block'], d['cut_crosses_tie_block'], d['e1']), (1, False, 'brier_crisis'))

    def _setup(self, conf_perm=False):
        rng, groups, X, y, months, x_set, conf = RecentSearchContrast()._fixture()
        if conf_perm:
            y = np.where(conf, np.random.default_rng(5).permutation(y), y)
        root = nx.fit_global(X[x_set == 0], y[x_set == 0], SMALL_G['G1'])
        Xt, yt, gt = X[:64], y[:64], groups[::30]
        y_pool = fourclass.argmax_codes(nx.proba(root[0], Xt))
        keep = ~conf
        data = (X[keep], y[keep], groups[keep], months[keep], x_set[keep], Xt, yt, gt, y_pool)
        return root, data, (X[conf], y[conf], groups[conf], months[conf])

    def _run(self, root, data, conf_data, **kw):
        mgf = TimeBlockContrast.mgf()
        with tempfile.TemporaryDirectory() as t, patch.object(trans, 'CONTIGUITY', False), \
                patch.object(trans, 'generate_count_grid', return_value=(None, 0, 1)), \
                patch.dict(plan.FIT_SUPPORT, FLOORS['fit_support']), \
                patch.dict(plan.STAGE1_VAL_SUPPORT, FLOORS['val_support']), \
                patch.object(mgf, 'MAX_DEPTH', 3), redirect_stdout(StringIO()):
            work, ck = Path(t) / 'w', Path(t) / 'ck'
            work.mkdir()
            rec = mgf.run_candidate('c', 'L1', 'gt0', root, data, work, ck, None, [f'f{i}' for i in range(5)],
                                    increment_source='root', confirmation=conf_data, **kw)
            return rec, (rec['confirmation']['frozen_digest_before_scoring'], rec['checkpoints']['sha256'],
                         (work / 'c' / 's_branch.pkl').read_bytes(),
                         (work / 'c' / 'correspondence_table.csv').read_text(),
                         json.dumps(rec['partition']['decisions'], default=str))

    def test_hard_default_identical_and_brier_isolated_from_c(self):
        root, data, conf_data = self._setup()
        rec_d, key_d = self._run(root, data, conf_data)
        rec_h, key_h = self._run(root, data, conf_data, e1='hard_f1')
        self.assertEqual(key_d, key_h)                          # new-code-only evidence
        self.assertEqual((rec_d['e1'], rec_h['e1']), ('hard_f1', 'hard_f1'))
        rec_b, key_b = self._run(root, data, conf_data, e1='brier_crisis')
        self.assertEqual(rec_b['e1'], 'brier_crisis')
        diags = [d['scan_diagnostics'] for d in rec_b['partition']['decisions'] if 'scan_diagnostics' in d]
        self.assertTrue(diags and all(d['e1'] == 'brier_crisis' for d in diags))
        self.assertTrue(all('post_refinement_s0_groups' in d for d in diags
                            if d['pre_refinement_s0_groups'] and d['pre_refinement_s1_groups']))
        root_p, data_p, conf_p = self._setup(conf_perm=True)
        _, key_bp = self._run(root, data, conf_p, e1='brier_crisis')
        self.assertEqual(key_b, key_bp)

    def test_brier_zero_loss_records_no_candidate(self):
        root, data, conf_data = self._setup()

        original = fourclass.brier_crisis_scan_masses

        def perfect(y_true, p, grp):
            groups, Y, _ = original(y_true, p, grp)
            return groups, Y, Y.copy()
        with patch.object(trans.fourclass, 'brier_crisis_scan_masses', side_effect=perfect):
            rec, _ = self._run(root, data, conf_data, e1='brier_crisis')
        outcomes = [d['outcome'] for d in rec['partition']['decisions']]
        self.assertEqual(outcomes, ['no_candidate_zero_error_mass'])
        self.assertEqual(rec['partition']['n_terminal'], 1)

    def test_twentyone_e1pair_roots_with_42_distinct_names(self):
        from scripts import run_stage1 as s1
        obs = pd.DataFrame({'month': [m(t) for t in plan.STAGE1_TARGETS + ('2010-01', '2016-02', '2016-06',
                                                                          '2016-10', '2017-02', '2017-06', '2017-10')]
                                     + [m(f'{y}-{mo:02d}') for y in range(2021, 2025) for mo in (2, 6, 10)]})
        sched = prep.build_schedule(obs)
        self.assertEqual(sched['stage1_e1pair_counts'], {**sched['stage1_e1pair_counts'], 'roots': 21, 'candidates': 42})
        roots = s1.scheduled_roots(sched, plan.TB3_G, plan.E1PAIR)
        cands = s1.scheduled_candidates(sched, plan.TB3_G, plan.E1PAIR)
        self.assertEqual((len(roots), len(cands)), (21, 42))
        for other in (plan.ROOTINC, plan.ROOTCONF, plan.RECENTSEARCH, plan.MATCHEDSIZE):
            self.assertFalse(set(roots) & set(s1.scheduled_roots(sched, plan.TB3_G, other)))
            self.assertFalse(set(cands) & set(s1.scheduled_candidates(sched, plan.TB3_G, other)))
        self.assertEqual(sorted({(c['horizon'], c['target_month']) for c in cands.values()}),
                         sorted((h, t) for h in plan.HORIZONS for t in plan.E1PAIR_TARGETS))
        per_root = {}
        for name, c in cands.items():
            parts = name.split('_')
            self.assertEqual((parts[3], parts[-1]), ('L1', 'gt0'))
            self.assertTrue(name.endswith({'hard_f1': '_e1hard_gt0', 'brier_crisis': '_e1brier_gt0'}[c['e1']]))
            self.assertIn(c['root'], roots)
            self.assertTrue(c['root'].endswith('_r80_s42_e1pair'))
            per_root.setdefault(c['root'], set()).add(c['e1'])
            self.assertEqual((c['ratio'], c['split_seed'], c['increment_source'], c['confirmation_seed']),
                             ('r80', 42, 'root', 42))
        self.assertTrue(all(v == {'hard_f1', 'brier_crisis'} for v in per_root.values()))
        self.assertFalse(any(r['recent_search_months'] if 'recent_search_months' in r else 0
                             for r in roots.values()))
        with self.assertRaises(SystemExit):
            s1.scheduled_roots({'stage1_e1pair_roots': sched['stage1_e1pair_roots'][:20]}, plan.TB3_G, plan.E1PAIR)


class E1PairReporter(unittest.TestCase):
    """D34 reporter on a REAL paired fixture: both E1 variants through run_candidate on one shared
    root, root files as main() writes them, then the canonical-order completion check,
    load_e1pair (file reads) and e1pair_row."""

    def test_real_pair_through_loader_and_row(self):
        from scripts import stage1_rootconf_compare as rc
        from src.feature.fourclass_features import month_label
        mgf = TimeBlockContrast.mgf()
        rng, groups, X, y, months, x_set, conf = RecentSearchContrast()._fixture()
        root = nx.fit_global(X[x_set == 0], y[x_set == 0], SMALL_G['G1'])
        first = np.unique(groups, return_index=True)[1]
        Xt, yt, gt = X[first], y[first], groups[first]            # one target row per area (unique keys)
        y_pool = fourclass.argmax_codes(nx.proba(root[0], Xt))
        keep = ~conf
        data = (X[keep], y[keep], groups[keep], months[keep], x_set[keep], Xt, yt, gt, y_pool)
        h, t, g = 4, "2018-06", "G1"
        root_name = plan.e1pair_root_name(h, t, g)
        pairs = plan.e1pair_candidate_names(h, t, g)
        with tempfile.TemporaryDirectory() as tmp, patch.object(trans, 'CONTIGUITY', False), \
                patch.object(trans, 'generate_count_grid', return_value=(None, 0, 1)), \
                patch.dict(plan.FIT_SUPPORT, FLOORS['fit_support']), patch.dict(plan.STAGE1_VAL_SUPPORT, FLOORS['val_support']), \
                patch.object(mgf, 'MAX_DEPTH', 3), redirect_stdout(StringIO()):
            stage = Path(tmp) / 'stage1_e1pair'
            (stage / 'candidates').mkdir(parents=True)
            for cand, e1 in pairs:
                mgf.run_candidate(cand, 'L1', 'gt0', root, data, stage / 'candidates', stage / 'checkpoints', None,
                                  [f'f{i}' for i in range(5)], increment_source='root',
                                  confirmation=(X[conf], y[conf], groups[conf], months[conf]), e1=e1)
            rdir = stage / 'roots' / root_name
            rdir.mkdir(parents=True)
            (rdir / 'root.json').write_text(json.dumps({'root_booster_sha256': root[1]['booster_sha256']}), encoding='utf-8')
            role = np.where(conf, 'confirmation', np.where(x_set == 1, 'validation', 'fitting'))
            pd.concat([pd.DataFrame({'area': groups, 'target_month': month_label(months), 'role': role, 'class_code': y}),
                       pd.DataFrame({'area': gt, 'target_month': t, 'role': 'heldout_target', 'class_code': yt})]
                      ).to_csv(rdir / 'fold_membership.csv.gz', index=False)
            pd.DataFrame({'FEWSNET_admin_code': gt, 'y_true_code': yt, 'y_pred_pooled_code': y_pool}).to_csv(
                rdir / 'root_target_predictions.csv', index=False)
            entry = {'horizon': h, 'target_month': t, 'g_config': g}
            expected = rc.expected_candidates(plan.E1PAIR, entry)
            self.assertEqual(expected, [n for n, _ in pairs])                     # hard first, then Brier
            rc.check_completion_candidates(root_name, [n for n, _ in pairs], expected)
            with self.assertRaises(rc.CompareError):                              # alphabetical order is not producer order
                rc.check_completion_candidates(root_name, sorted(expected), expected)
            loaded = rc.load_e1pair(stage, root_name, {e1: n for n, e1 in pairs}, h, t)
            row, mats, diag = rc.e1pair_row(loaded, h, t)
            tp = {e1: pd.read_csv(stage / 'candidates' / n / 'target_predictions.csv') for n, e1 in pairs}
            f = lambda d: fourclass.crisis_f1_exact(d['y_true_code'].to_numpy(), d['y_pred_partitioned_code'].to_numpy())   # noqa: E731
            self.assertAlmostEqual(row['e3_brier_minus_hard'], float(f(tp['brier_crisis']) - f(tp['hard_f1'])), places=12)
            for tok in ('e1hard', 'e1brier'):
                pools = diag[tok]['assigned_pools']
                self.assertEqual(sum(v['fitting_rows'] for v in pools.values()), int((x_set[keep] == 0).sum()))
                self.assertIn(f'{tok}_assigned_fitting_rows_min', row)
                self.assertIn(f'{tok}_checkpoint_training_rows_max', row)
            self.assertNotIn('e4_weight', row)
            self.assertEqual(row['e1hard_root_scan_n_groups'] > 0, True)

class GlobalIncrementControl(unittest.TestCase):
    """D35/A10: fitting pool excludes S/C/target, held-out labels cannot move the +20 continuation."""

    def _run(self, tmp, permute_held=False):
        from scripts import stage1_shallow_replay as sr
        features = ff.load_schema(sr.SCHEMA)['ordered_features']
        rng = np.random.default_rng(0)
        areas, months = np.arange(40), np.arange(2013 * 12, 2019 * 12)   # 2013-01 .. 2018-12
        a, m = np.repeat(areas, len(months)), np.tile(months, len(areas))
        snap = pd.DataFrame({'area': a, 'target_month': m, 'horizon': 4,
                             'class_code': rng.integers(0, 4, len(a))})
        snap = pd.concat([snap, pd.DataFrame(rng.normal(size=(len(a), len(features))), columns=features)], axis=1)
        root = {'horizon': 4, 'target_month': '2018-06', 'ratio': 'r80', 'split_seed': 42}
        run = Path(tmp) / 'run'
        (run / 'prepared').mkdir(parents=True, exist_ok=True)
        path = run / 'prepared' / 'snapshot_h4.parquet'
        snap.to_parquet(path, index=False)
        data = sr.rebuild(run, root, max_month=2018 * 12 + 7, with_fitting=True)
        if permute_held:
            held = set()
            for part in ('S', 'C', 'E3'):
                held |= self.gi.key_set(data[part][2], data[part][3])
            keys = list(zip(snap['area'].tolist(), ff.month_label(snap['target_month'])))
            mask = np.array([k in held for k in keys])
            snap.loc[mask, 'class_code'] = (snap.loc[mask, 'class_code'] + 1) % 4
            snap.to_parquet(path, index=False)
            data = sr.rebuild(run, root, max_month=2018 * 12 + 7, with_fitting=True)
        return data

    def setUp(self):
        from scripts import stage1_global_increment as gi
        self.gi = gi

    def test_fitting_pool_and_label_permutation_and_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = self._run(tmp)
            Xf, yf, gf, mf = data['FIT']
            fit = self.gi.key_set(gf, ff.month_label(mf))
            for part in ('S', 'C', 'E3'):
                self.assertTrue(len(data[part][1]) > 0)
                self.assertFalse(fit & self.gi.key_set(data[part][2], data[part][3]), part)
            lo, hi, o = data['window']
            self.assertTrue(lo >= o - plan.WINDOW and hi < o)
            self.assertTrue((data['membership']['target_month'] <= '2018-08').all())
            root, _ = nx.fit_global(Xf, yf, SMALL_G['G1'])
            child, rec = self.gi.fit_global_plus20(root, data)
            self.assertEqual(rec['rounds_added'], 20)
            self.assertEqual(child.num_boosted_rounds(), root.num_boosted_rounds() + 20)
            self.assertEqual(rec['child_prefix_structure_sha256'], nx.prefix_identity(root)['sha256'])
            self.assertEqual(rec['rows'], len(yf))
        with tempfile.TemporaryDirectory() as tmp:
            data2 = self._run(tmp, permute_held=True)
            self.assertTrue((data2['FIT'][1] == yf).all())
            self.assertFalse((data2['C'][1] == data['C'][1]).all())        # held-out labels did change
            child2, _ = self.gi.fit_global_plus20(nx.from_raw(nx.raw(root)), data2)
            self.assertEqual(nx.raw(child2), nx.raw(child))

    def test_strata_scoring_with_empty_stratum(self):
        frame = pd.DataFrame({'truth': [2, 0, 3, 1], 'y_root': [0, 0, 3, 1], 'y_global20': [2, 0, 3, 2],
                              'y_hard_local': [2, 2, 3, 1], 'y_brier_local': [2, 0, 0, 1],
                              'search_rows': [1, 2, 1, 3]})
        frame['stratum'] = np.where(frame['search_rows'] > 0, 'search_rows>0', 'search_rows==0')
        out = self.gi.score_part(frame)
        self.assertEqual(out['strata']['search_rows==0'], {'n': 0, 'status': 'no_data'})
        self.assertEqual(out['all']['n'], 4)
        self.assertEqual(Fraction(out['all']['global20']['crisis_f1_exact']), Fraction(4, 5))
        self.assertEqual(Fraction(out['all']['root']['crisis_f1_exact']), Fraction(2, 3))
        self.assertEqual(Fraction(out['all']['global20_minus_root']['crisis_f1_delta_exact']), Fraction(2, 15))
        agg = self.gi.aggregate({'a': {'scores': {p: out for p in ('S', 'C', 'E3')}}}, lambda r: True)
        self.assertEqual(agg['C']['search_rows==0']['folds_with_data'], 0)
        self.assertEqual(agg['C']['all']['pooled']['global20']['crisis_f1_exact'], '4/5')
        self.assertAlmostEqual(agg['C']['all']['mean_fold_crisis_f1_delta']['global20_minus_root'], 2 / 15)


class RecencyRoot(unittest.TestCase):
    """D37/A11: optional fit_global weights, fitting-only mean-1 recency weights, persistence scoring."""

    def setUp(self):
        from scripts import stage1_recency_root as rr
        self.rr = rr
        rng = np.random.default_rng(3)
        self.X = rng.normal(size=(120, 5))
        self.y = np.tile(np.arange(4), 30)

    def test_default_path_unchanged_and_invalid_weights(self):
        import xgboost as xgb
        g, rec = nx.fit_global(self.X, self.y, SMALL_G['G1'])
        self.assertEqual(set(rec), {"kind", "rounds_total", "rounds_added", "params", "resolved_config",
                                    "base_score", "rows", "class_counts", "structure_sha256", "booster_sha256"})
        params, rounds = nx.booster_params(SMALL_G['G1'])
        ref = xgb.train(params, xgb.DMatrix(nx.clean(self.X), label=self.y, missing=np.nan, nthread=4),
                        num_boost_round=rounds)
        self.assertEqual(nx.raw(g), nx.raw(ref))
        for bad in (np.ones(5), np.r_[np.ones(119), np.nan], np.r_[np.ones(119), 0.0], np.r_[np.ones(119), -1.0],
                    np.r_[np.ones(119), np.inf]):
            with self.assertRaises(ValueError):
                nx.fit_global(self.X, self.y, SMALL_G['G1'], sample_weight=bad)
        w = np.linspace(0.5, 1.5, 120)
        gw, recw = nx.fit_global(self.X, self.y, SMALL_G['G1'], sample_weight=w)
        self.assertEqual(recw['sample_weight']['dtype'], 'float32')
        self.assertNotEqual(nx.raw(gw), nx.raw(g))

    def test_weight_formula(self):
        o = 2018 * 12 + 1
        months = np.array([o - 1, o - 25, o - 49, o - 59, o - 1])
        w = self.rr.recency_weights(months, o)
        u = 2.0 ** (-((o - 1) - months) / 24.0)
        np.testing.assert_array_equal(w['w64'], u / u.mean())
        self.assertAlmostEqual(w['w64'].mean(), 1.0, places=14)
        self.assertAlmostEqual(w['w64'][0] / w['w64'][1], 2.0, places=12)
        self.assertEqual(w['w32'].dtype, np.float32)
        self.assertEqual(w['record']['sha256'], hashlib.sha256(w['w64'].astype(np.float32).tobytes()).hexdigest())
        self.assertAlmostEqual(w['record']['kish_ess'], w['w32'].astype(float).sum() ** 2 / np.sum(w['w32'].astype(float) ** 2), places=12)
        for bad in (np.array([o]), np.array([o - 60])):
            with self.assertRaises(self.rr.GateError):
                self.rr.recency_weights(bad, o)

    def test_heldout_labels_do_not_move_weights_or_booster(self):
        helper = GlobalIncrementControl()
        helper.setUp()
        with tempfile.TemporaryDirectory() as tmp:
            data = helper._run(tmp)
            b1, r1, w1 = self.rr.fit_weighted_root(data, 'G1')
        with tempfile.TemporaryDirectory() as tmp:
            data2 = helper._run(tmp, permute_held=True)
            self.assertFalse((data2['C'][1] == data['C'][1]).all())
            b2, r2, w2 = self.rr.fit_weighted_root(data2, 'G1')
        np.testing.assert_array_equal(w1['w32'], w2['w32'])
        self.assertEqual(r1['rows'], len(data['FIT'][1]))
        self.assertEqual(r1['sample_weight']['sha256'], r2['sample_weight']['sha256'])
        self.assertEqual(nx.raw(b1), nx.raw(b2))

    def test_persistence_missing_and_matched_scoring(self):
        pers = self.rr.persistence_codes([1, 3, 5, np.nan, 4])
        np.testing.assert_array_equal(pers[[0, 1, 2, 4]], [0, 2, 3, 3])
        self.assertTrue(np.isnan(pers[3]))
        frame = pd.DataFrame({'truth': [2, 0, 3, 1], 'y_original': [0, 0, 3, 1], 'y_weighted': [2, 0, 3, 2],
                              'persistence_code': [2.0, 2.0, np.nan, 0.0]})
        for m in ('original', 'weighted'):
            for k, lab in enumerate(fourclass.CLASS_LABELS):
                frame[f'p_{m}_{lab}'] = 0.25
        out = self.rr.score_frame(frame)
        self.assertEqual(out['matched_persistence']['n'], 3)
        self.assertAlmostEqual(out['matched_persistence']['coverage'], 0.75)
        # matched keys rows 0,1,3: persistence crisis preds [1,1,0] vs truth [1,0,0] -> tp1 fp1 fn0 -> 2/3
        self.assertEqual(Fraction(out['matched_persistence']['persistence']['crisis_f1_exact']), Fraction(2, 3))
        self.assertEqual(Fraction(out['matched_persistence']['weighted']['crisis_f1_exact']), Fraction(2, 3))
        self.assertEqual(Fraction(out['all']['weighted']['crisis_f1_exact']), Fraction(4, 5))
        self.assertEqual(out['transition_groups_post_hoc']['missing']['n'], 1)
        self.assertEqual(out['transition_groups_post_hoc']['11']['corrected'], 1)
        self.assertEqual(out['transition_groups_post_hoc']['00']['fp_change'], 1)
        self.assertAlmostEqual(out['all']['original']['crisis_brier'], np.mean((0.5 - np.array([1, 0, 1, 0])) ** 2))


class PersistenceMarginRoot(unittest.TestCase):
    """D38/A12: optional fit_global/proba base margin, marker guards, margin formula, controls."""

    def setUp(self):
        from scripts import stage1_persistence_margin_root as pm
        self.pm = pm
        rng = np.random.default_rng(5)
        self.X = rng.normal(size=(160, 5))
        self.y = np.tile(np.arange(4), 40)
        self.phase = np.tile([1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 3.0, 1.0], 20)

    def test_default_record_and_bytes_unchanged(self):
        g, rec = nx.fit_global(self.X, self.y, SMALL_G['G1'])
        self.assertEqual(set(rec), {"kind", "rounds_total", "rounds_added", "params", "resolved_config",
                                    "base_score", "rows", "class_counts", "structure_sha256", "booster_sha256"})
        params, rounds = nx.booster_params(SMALL_G['G1'])
        ref = xgb.train(params, xgb.DMatrix(nx.clean(self.X), label=self.y, missing=np.nan, nthread=4),
                        num_boost_round=rounds)
        self.assertEqual(nx.raw(g), nx.raw(ref))
        self.assertFalse(nx.is_margin_marked(g))

    def test_margin_formula_axis_float32_missing(self):
        m = self.pm.persistence_margin(self.phase)
        self.assertEqual(m.dtype, np.float32)
        self.assertEqual(m.shape, (len(self.phase), 4))
        miss = np.isnan(self.phase)
        self.assertTrue((m[miss] == np.float32(0.5)).all())
        q = self.pm.persistence_prior(self.phase)
        np.testing.assert_array_equal(q[miss], 0.25)
        for phase, code in ((1.0, 0), (2.0, 1), (3.0, 2), (4.0, 3), (5.0, 3)):
            row = np.flatnonzero(self.phase == phase)[0]
            expect_q = np.full(4, 0.125)
            expect_q[code] = 0.625
            np.testing.assert_allclose(q[row], expect_q, rtol=0, atol=1e-15)
            lq = np.log(expect_q)
            np.testing.assert_array_equal(m[row], (0.5 + lq - lq.mean()).astype(np.float32))
            self.assertEqual(int(np.argmax(m[row])), code)
        np.testing.assert_allclose(m.astype(float).mean(axis=1), 0.5, atol=1e-6)
        with self.assertRaises(self.pm.GateError):
            self.pm.persistence_margin([1.0, 6.0])
        with self.assertRaises(self.pm.GateError):
            self.pm.persistence_margin([0.0])

    def test_marker_hash_reload_and_guards(self):
        m = self.pm.persistence_margin(self.phase)
        g, rec = nx.fit_global(self.X, self.y, SMALL_G['G1'], base_margin=m)
        self.assertEqual(g.attr(nx.MARGIN_ATTR), nx.MARGIN_MARKER)
        self.assertEqual(rec['booster_sha256'], nx.sha(g))   # marker was written before hashing
        self.assertEqual(rec['base_margin']['dtype'], 'float32')
        self.assertEqual(rec['base_margin']['sha256'], hashlib.sha256(m.tobytes()).hexdigest())
        self.assertEqual(rec['base_margin']['n'], len(self.y))
        p = nx.proba(g, self.X, base_margin=m)
        g2 = nx.from_raw(nx.raw(g))
        self.assertEqual(g2.attr(nx.MARGIN_ATTR), nx.MARGIN_MARKER)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'a.ubj'
            path.write_bytes(nx.raw(g))
            g3 = nx.from_raw(path.read_bytes())
            self.assertEqual(g3.attr(nx.MARGIN_ATTR), nx.MARGIN_MARKER)
            np.testing.assert_array_equal(nx.proba(g3, self.X, base_margin=m), p)
        with self.assertRaises(ValueError):
            nx.proba(g, self.X)                                   # marked without margin
        unmarked, _ = nx.fit_global(self.X, self.y, SMALL_G['G1'])
        with self.assertRaises(ValueError):
            nx.proba(unmarked, self.X, base_margin=m)             # unmarked with margin
        with self.assertRaises(ValueError):
            nx.continue_booster(g, self.X, self.y, plan.L_CONFIGS['L1'])
        for bad in (m[:-1], m.T, np.c_[m[:, :3]], np.where(np.arange(4) == 0, np.nan, m)):
            with self.assertRaises(ValueError):
                nx.fit_global(self.X, self.y, SMALL_G['G1'], base_margin=bad)
            with self.assertRaises(ValueError):
                nx.proba(g, self.X, base_margin=bad)

    def test_neutral_margin_training_equivalence(self):
        neutral = np.full((len(self.y), 4), 0.5, dtype=np.float32)
        gm, recm = nx.fit_global(self.X, self.y, SMALL_G['G1'], base_margin=neutral)
        g, rec = nx.fit_global(self.X, self.y, SMALL_G['G1'])
        self.assertEqual(nx.prefix_identity(gm)['sha256'], nx.prefix_identity(g)['sha256'])
        self.assertEqual(recm['structure_sha256'], rec['structure_sha256'])
        self.assertEqual(recm['base_score'], rec['base_score'])
        np.testing.assert_array_equal(nx.proba(gm, self.X, base_margin=neutral), nx.proba(g, self.X))
        self.assertNotEqual(nx.raw(gm), nx.raw(g))               # only the marker attribute differs

    def test_raw_margin_offset_and_transpose_detected(self):
        g, _ = nx.fit_global(self.X, self.y, SMALL_G['G1'])
        m = self.pm.persistence_margin(self.phase)
        dm = nx.dmatrix(self.X)
        dm.set_base_margin(m)
        diff = g.predict(dm, output_margin=True) - g.predict(nx.dmatrix(self.X), output_margin=True)
        np.testing.assert_allclose(diff, m.astype(float) - 0.5, atol=1e-5)
        wrong = np.ascontiguousarray(m.T).reshape(len(self.y), 4)   # column-major read of the same values
        dmw = nx.dmatrix(self.X)
        dmw.set_base_margin(wrong)
        diffw = g.predict(dmw, output_margin=True) - g.predict(nx.dmatrix(self.X), output_margin=True)
        self.assertFalse(np.allclose(diffw, m.astype(float) - 0.5, atol=1e-5))

    def test_posthoc_and_prior_only(self):
        rng = np.random.default_rng(1)
        p = rng.dirichlet(np.ones(4), size=len(self.phase)).astype(np.float32).astype(np.float64)
        post = self.pm.posthoc(p, self.phase)
        miss = np.isnan(self.phase)
        np.testing.assert_array_equal(post[miss], p[miss])
        q = self.pm.persistence_prior(self.phase)
        pq = p[~miss] * q[~miss]
        np.testing.assert_allclose(post[~miss], pq / pq.sum(axis=1, keepdims=True), rtol=0, atol=1e-15)
        np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=1e-12)
        y_prior = np.argmax(q, axis=1)
        self.assertTrue((y_prior[miss] == 0).all())
        np.testing.assert_array_equal(y_prior[~miss], self.pm.rr.persistence_codes(self.phase)[~miss])

    def test_heldout_labels_do_not_move_margin_or_fit(self):
        from scripts import stage1_shallow_replay as sr
        col = ff.load_schema(sr.SCHEMA)['ordered_features'].index('hist_phase_o00')
        helper = GlobalIncrementControl()
        helper.setUp()

        def fit(permute):
            with tempfile.TemporaryDirectory() as tmp:
                data = helper._run(tmp, permute_held=permute)
            X = data['FIT'][0]
            X[:, col] = np.where(data['FIT'][2] % 7 == 0, np.nan, data['FIT'][2] % 5 + 1)
            b, rec, m = self.pm.fit_anchored_root(data, 'G1', col)
            return data, b, rec, m

        d1, b1, r1, m1 = fit(False)
        d2, b2, r2, m2 = fit(True)
        self.assertFalse((d2['C'][1] == d1['C'][1]).all())
        np.testing.assert_array_equal(d1['FIT'][0], d2['FIT'][0])
        np.testing.assert_array_equal(m1, m2)
        self.assertEqual(r1['base_margin']['sha256'], r2['base_margin']['sha256'])
        self.assertEqual(nx.raw(b1), nx.raw(b2))
        self.assertEqual(r1['rows'], len(d1['FIT'][1]))

    def test_score_frame_models_and_persistence(self):
        phase = np.array([3.0, 3.0, np.nan, 1.0])
        frame = pd.DataFrame({'truth': [2, 0, 3, 1], 'persistence_code': self.pm.rr.persistence_codes(phase),
                              'y_original': [0, 0, 3, 1], 'y_anchored': [2, 0, 3, 2],
                              'y_prior_only': [2, 2, 0, 0], 'y_posthoc': [2, 2, 3, 1]})
        for mdl in self.pm.MODELS:
            for k, lab in enumerate(fourclass.CLASS_LABELS):
                frame[f'p_{mdl}_{lab}'] = 0.25
        out = self.pm.score_frame(frame)
        mp = out['matched_persistence']
        self.assertEqual(mp['n'], 3)
        self.assertTrue(mp['prior_only_argmax_equals_persistence'])
        self.assertEqual(Fraction(mp['persistence']['crisis_f1_exact']), Fraction(2, 3))
        # one-hot persistence Brier on matched keys: crisis preds [1,1,0] vs truth [1,0,0] -> 1/3
        self.assertAlmostEqual(mp['persistence']['crisis_brier'], 1 / 3)
        dep = out['departure_from_persistence']['anchored']
        # matched anchored crisis [1,0,1] vs persistence [1,1,0], truth [1,0,0]: row1 model right, row3 persistence
        self.assertEqual((dep['disagree'], dep['model_right'], dep['persistence_right']), (2, 1, 1))
        self.assertEqual(out['transition_groups_post_hoc']['anchored']['missing']['n'], 1)
        agg = self.pm.aggregate({'r': {'scores': {'C': out, 'E3': out}}}, lambda r: True)
        self.assertEqual(agg['E3']['folds_with_data'], 1)
        self.assertIn('anchored_minus_posthoc', agg['E3']['pooled_all'])


class MapTransfer(unittest.TestCase):
    """D42/A16: U < O schedule, D32 map parsing, common refit pools/routes, label invariance, save/reload."""

    FLOOR = {'rows': 40, 'areas': 10, 'dates': 3, 'classes': 2}

    def setUp(self):
        from scripts import stage1_map_transfer as mt
        self.mt = mt

    def test_schedule_rule_gives_the_twelve_spec_pairs(self):
        got = self.mt.schedule()
        self.assertEqual(got, [(h, t, u) for h, t, u, _, _ in self.mt.SPEC_PAIRS])
        self.assertEqual(len(got), 12)
        self.assertEqual(sum(o + c for *_, o, c in self.mt.SPEC_PAIRS), self.mt.MAX_FITS)
        self.assertNotIn('2018-06', [t for _, t, _ in got])          # no U < O for the earliest targets
        for h, t, u in got:
            o = self.mt._mi(t) - h
            self.assertLess(self.mt._mi(u), o)
            self.assertFalse(any(self.mt._mi(u) < self.mt._mi(x) < o for x in plan.E1PAIR_TARGETS))
        self.assertEqual(self.mt.check_schedule(), got)

    def _evidence(self, tmp, rows):
        path = Path(tmp) / 'assignment_evidence.csv'
        pd.DataFrame(rows, columns=['FEWSNET_admin_code', 'prediction_branch_id', 'spatial_partition_id',
                                    'search_rows']).to_csv(path, index=False)
        return path

    def test_parse_map_leading_zeros_and_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._evidence(tmp, [[5, '01', '01', 2], [7, '0', '0', 1], [9, 'root', 's-1', 0],
                                        [11, 'root', 'root', 1]])
            area_map = self.mt.parse_map(path, rid.file_sha256(path))
            self.assertEqual(area_map, {5: '01', 7: '0', 9: 's-1', 11: 'root'})
            with self.assertRaises(self.mt.GateError):
                self.mt.parse_map(path, '0' * 64)
            bad = self._evidence(tmp, [[5, '01', 's-1', 2]])
            with self.assertRaises(self.mt.GateError):
                self.mt.parse_map(bad)
            bad = self._evidence(tmp, [[5, '01', '01', 0]])
            with self.assertRaises(self.mt.GateError):
                self.mt.parse_map(bad)
            bad = self._evidence(tmp, [[5, '01', '01', 1], [5, '0', '0', 1]])
            with self.assertRaises(self.mt.GateError):
                self.mt.parse_map(bad)

    def _fit_pool(self):
        rng = np.random.default_rng(11)
        areas, months = np.arange(50), np.arange(2015 * 12, 2015 * 12 + 8)
        g, mm = np.repeat(areas, len(months)), np.tile(months, len(areas))
        order = rng.permutation(len(g))                       # non-sorted original order
        g, mm = g[order], mm[order]
        X = rng.normal(size=(len(g), 6))
        y = np.digitize(X[:, 0], [-0.6, 0.3, 1.2]).astype(np.int64)
        return X, y, g, mm

    def test_common_refit_pools_routes_and_save_reload(self):
        X, y, g, mm = self._fit_pool()
        area_map = {**{a: '01' for a in range(20)}, **{a: 'root' for a in range(20, 40)},
                    **{a: '1' for a in range(40, 45)}, **{a: 's-1' for a in range(45, 48)}, 70: '01'}
        root, _ = nx.fit_global(X, y, SMALL_G['G1'])
        before = nx.raw(root)
        models, recs = self.mt.common_refit(root, (X, y, g, mm), area_map, floor=self.FLOOR)
        self.assertEqual(nx.raw(root), before)
        self.assertEqual(set(models), {'01', 'root'})               # named root-copy region refitted too
        self.assertFalse(recs['1']['eligible'])
        self.assertEqual(recs['1']['support']['areas'], 5)
        self.assertEqual(recs['1']['support']['rows'], 40)
        for region in ('01', 'root'):
            mask = np.array([area_map.get(int(a)) == region for a in g])
            self.assertEqual(recs[region]['fitting_keys_sha256'], nx.keys_sha(g[mask], mm[mask]))
            self.assertEqual(recs[region]['continuation']['rows'], int(mask.sum()))
            self.assertEqual(recs[region]['continuation']['rounds_added'], 20)
            self.assertEqual(recs[region]['continuation']['child_prefix_structure_sha256'],
                             nx.prefix_identity(root)['sha256'])
            ref, _ = nx.continue_booster(root, X[mask], y[mask], plan.L_CONFIGS['L1'])
            self.assertEqual(nx.raw(models[region]), nx.raw(ref))
        self.assertEqual(recs['01']['member_areas'], list(range(20)) + [70])
        self.assertEqual(recs['01']['member_areas_with_fitting_rows'], 20)
        eval_areas = np.array([70, 3, 25, 41, 46, 49, 99])
        sid, reason = self.mt.route(eval_areas, area_map, recs)
        self.assertEqual(reason.tolist(), ['region', 'region', 'region', 'insufficient_support', 's-1',
                                           'missing', 'missing'])
        self.assertEqual(sid.tolist()[:3], ['01', '01', 'root'])
        Xe = np.random.default_rng(2).normal(size=(len(eval_areas), 6))
        p_root = nx.proba(root, Xe)
        p = self.mt.arm_proba(Xe, sid, reason, p_root, models)
        np.testing.assert_array_equal(p[3:], p_root[3:])
        np.testing.assert_array_equal(p[:2], nx.proba(models['01'], Xe[:2]))
        np.testing.assert_array_equal(p[2:3], nx.proba(models['root'], Xe[2:3]))
        with tempfile.TemporaryDirectory() as tmp:
            frozen = self.mt.save_arm(Path(tmp) / 'arm', models, recs, {'candidate': 'x'}, 'sha')
            self.assertEqual(set(frozen), {'01', 'root'})
            self.assertTrue((Path(tmp) / 'arm' / 'region_1.json').is_file())
            self.assertFalse((Path(tmp) / 'arm' / 'region_1.ubj').exists())
            np.testing.assert_array_equal(self.mt.arm_proba(Xe, sid, reason, p_root, frozen), p)

    def test_heldout_labels_do_not_move_pools_boosters_or_routes(self):
        helper = GlobalIncrementControl()
        helper.setUp()
        area_map = {**{a: '0' for a in range(18)}, **{a: '1' for a in range(18, 36)}, 36: 's-1', 37: 's-1'}

        def refit(permute):
            with tempfile.TemporaryDirectory() as tmp:
                data = helper._run(tmp, permute_held=permute)
            root, _ = nx.fit_global(data['FIT'][0], data['FIT'][1], SMALL_G['G1'])
            with patch.dict(plan.FIT_SUPPORT, self.FLOOR):
                models, recs = self.mt.common_refit(root, data['FIT'], area_map)
            routes = {p: self.mt.route(data[p][2], area_map, recs) for p in ('C', 'E3')}
            return data, models, recs, routes

        d1, m1, r1, rt1 = refit(False)
        d2, m2, r2, rt2 = refit(True)
        self.assertFalse((d2['C'][1] == d1['C'][1]).all())
        self.assertEqual(set(m1), {'0', '1'})
        self.assertEqual({k: nx.raw(v) for k, v in m1.items()}, {k: nx.raw(v) for k, v in m2.items()})
        self.assertEqual({k: v['fitting_keys_sha256'] for k, v in r1.items()},
                         {k: v['fitting_keys_sha256'] for k, v in r2.items()})
        for p in ('C', 'E3'):
            for a, b in zip(rt1[p], rt2[p]):
                self.assertEqual(a.tolist(), b.tolist())
        self.assertEqual({r for r in rt1['E3'][1].tolist()}, {'region', 's-1', 'missing'})   # all 40 areas in E3

    def test_failed_first_pair_gate_stops_before_second_pair(self):
        mt = self.mt
        names = {plan.e1pair_root_name(h, x, plan.TB3_G[str(h)]) for h, t, u in mt.schedule() for x in (t, u)}
        cands = {f'c_{r}': {'root': r, 'e1': 'brier_crisis'} for r in names}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run, d35, out = tmp / 'd34', tmp / 'd35', tmp / 'out'
            (run / 'prepared' / 'ledgers').mkdir(parents=True)
            (run / 'prepared' / 'ledgers' / 'dev_baselines.csv').write_text('horizon,target_label\n', encoding='utf-8')
            d35.mkdir()
            (d35 / 'identity.json').write_text(json.dumps({'stage': 'd35_global_increment',
                                                           'producer_rev': '7b2bf6f'}), encoding='utf-8')
            (d35 / 'completion.json').write_text('{}', encoding='utf-8')
            argv = ['stage1_map_transfer.py', '--d34-run', str(run), '--d35-run', str(d35), '--out', str(out)]
            with patch.object(sys, 'argv', argv), \
                    patch.object(mt.rid, 'code_identity_at', return_value={'sha256': 'x'}), \
                    patch.object(mt.rid, 'code_identity', return_value={'sha256': 'x'}), \
                    patch.object(mt.rid, 'file_sha256', return_value='0' * 64), \
                    patch.object(mt, 'accept_mode', return_value=({n: {} for n in names}, cands,
                                                                  {'stage': str(tmp / 'stage')})), \
                    patch.object(mt, 'run_pair', side_effect=mt.GateError('replay mismatch')) as rp, \
                    redirect_stdout(StringIO()):
                code = mt.main()
            self.assertEqual(code, 2)
            self.assertEqual(rp.call_count, 1)
            gate = json.loads((out / 'gate.json').read_text(encoding='utf-8'))
            self.assertFalse(gate['passed'])
            self.assertEqual(gate['fits'], 0)
            self.assertEqual(list(gate['pairs']), [gate['stopped_at']])
            self.assertTrue((out / 'identity.json').is_file())
            self.assertFalse((out / 'summary.json').exists())

    def test_score_frame_and_aggregate(self):
        frame = pd.DataFrame({'truth': [2, 0, 3, 1], 'persistence_code': [2.0, 2.0, np.nan, 0.0],
                              'y_root': [0, 0, 3, 1], 'y_global20': [2, 0, 3, 2], 'y_current_map_refit': [2, 0, 3, 1],
                              'y_old_map_refit': [0, 2, 3, 1],
                              'route_current': ['region', 's-1', 'region', 'missing'],
                              'route_old': ['region', 'region', 'insufficient_support', 'missing']})
        for arm in self.mt.ARMS:
            for lab in fourclass.CLASS_LABELS:
                frame[f'p_{arm}_{lab}'] = 0.25
        out = self.mt.score_frame(frame)
        self.assertEqual(Fraction(out['all']['current_map_refit']['crisis_f1_exact']), Fraction(1))
        self.assertEqual(Fraction(out['all']['current_map_refit_minus_root']['crisis_f1_delta_exact']),
                         Fraction(1, 3))
        self.assertEqual(out['matched_persistence']['n'], 3)
        self.assertEqual(out['both_covered_supplementary']['n'], 1)
        self.assertEqual(out['routing']['current_map_refit']['by_reason'],
                         {'region': 2, 's-1': 1, 'missing': 1, 'insufficient_support': 0})
        self.assertEqual(out['routing']['old_map_refit']['root_share'], 0.5)
        self.assertEqual(out['transition_groups_post_hoc']['current_map_refit']['11']['corrected'], 1)
        agg = self.mt.aggregate({'r': {'horizon': 4, 'scores': {'C': out, 'E3': out}}}, lambda r: True)
        self.assertEqual(agg['E3']['pooled_all']['old_map_refit_minus_current_map_refit']['crisis_f1_delta_exact'],
                         str(Fraction(1, 2) - 1))
        self.assertEqual(agg['E3']['routing']['current_map_refit']['root_share'], 0.5)


class TemporalMapRefit(unittest.TestCase):
    """D43/A17: search-month table, pool/role reconstruction, temporal search + common refit isolation."""

    MONTHS = ('2017-11', '2017-12', '2018-01')        # last three pool months of the D35 fixture (H4, 2018-06)

    def setUp(self):
        from scripts import stage1_temporal_map_refit as tm
        self.tm = tm

    def _data(self, tmp):
        return GlobalIncrementControl()._run(tmp)

    def test_date_table_and_pool_reconstruction(self):
        tm = self.tm
        self.assertEqual(sorted(tm.SEARCH_MONTHS), sorted(tm.SCHEDULE))
        self.assertEqual(len(tm.SCHEDULE), 21)
        for (h, t), months in tm.SEARCH_MONTHS.items():
            idx = [prep.mi(x) for x in months]
            self.assertEqual(idx, sorted(set(idx)))
            self.assertLess(idx[-1], prep.mi(t) - h)
        with tempfile.TemporaryDirectory() as tmp:
            data = self._data(tmp)
        self.assertEqual(tm.check_search_months(data['membership'], self.MONTHS), list(self.MONTHS))
        with self.assertRaises(tm.GateError):
            tm.check_search_months(data['membership'], ('2017-10', '2017-11', '2017-12'))
        pool = tm.reconstruct_pool(data)
        mem = data['membership'][data['membership']['role'] != 'heldout_target']
        np.testing.assert_array_equal(pool['g'], mem['area'].to_numpy())
        self.assertEqual(ff.month_label(pool['m']).tolist(), mem['target_month'].tolist())
        self.assertEqual(pool['role'].tolist(), mem['role'].tolist())
        for role, part in (('fitting', 'FIT'), ('validation', 'S'), ('confirmation', 'C')):
            np.testing.assert_array_equal(pool['X'][pool['role'] == role], data[part][0])
        split = tm.temporal_split(pool, self.MONTHS)
        np.testing.assert_array_equal(split['X_set'], np.isin(pool['m'], [prep.mi(x) for x in self.MONTHS]).astype(int))
        with self.assertRaises(tm.GateError):
            tm.temporal_split(pool, ('2017-10', '2017-11', '2017-12'))
        frame = tm.pool_membership_frame(pool, split['X_set'], data['E3'][2], '2018-06')
        self.assertEqual(frame['area'].tolist(), mem['area'].tolist() + data['E3'][2].tolist())
        self.assertEqual(frame['d34_role'].tolist(), mem['role'].tolist() + ['heldout_target'] * len(data['E3'][2]))
        self.assertEqual(frame['temporal_role'].tolist()[:len(mem)],
                         np.where(np.isin(mem['target_month'], self.MONTHS), 'S_tb', 'FIT_tb').tolist())
        self.assertTrue((frame['temporal_role'].iloc[len(mem):] == 'excluded_E3').all())
        bad = dict(data)
        Xs, ys, gs, ms = data['S']
        bad['S'] = (Xs[::-1], ys[::-1], gs[::-1], ms[::-1])        # moved rows -> keys out of membership order
        with self.assertRaises(tm.GateError):
            tm.reconstruct_pool(bad)

    def _searches(self, data, mutate_e3=False):
        tm = self.tm
        from app import main_model_GF as mgf
        pool = tm.reconstruct_pool(data)
        Xt, yt, gt, _ = data['E3']
        if mutate_e3:
            yt = (yt + 1) % 4
        f = pool['role'] == 'fitting'
        current = nx.fit_global(pool['X'][f], pool['y'][f], SMALL_G['G1'])
        x_tb = np.asarray(tm.temporal_split(pool, self.MONTHS)['X_set'], dtype=int)
        s_root = tm.fit_search_root(pool, x_tb, SMALL_G['G1'])
        features = ff.load_schema(SCHEMA)['ordered_features']
        out = {'current': current, 'search_root': s_root, 'pool': pool}
        with tempfile.TemporaryDirectory() as t, patch.object(trans, 'CONTIGUITY', False), \
                patch.object(trans, 'generate_count_grid', return_value=(None, 0, 1)), \
                patch.dict(plan.FIT_SUPPORT, FLOORS['fit_support']), \
                patch.dict(plan.STAGE1_VAL_SUPPORT, FLOORS['val_support']), \
                patch.object(mgf, 'MAX_DEPTH', 3), redirect_stdout(StringIO()):
            t = Path(t)
            data_p, conf = tm.production_inputs(pool, (Xt, yt, gt), current[0])
            (t / 'rand' / 'candidates').mkdir(parents=True)
            rec_r = tm.run_search('rnd', current, data_p, t / 'rand' / 'candidates', t / 'rand' / 'ck', None, features,
                                  confirmation=conf)
            data_t = (pool['X'], pool['y'], pool['g'], pool['m'], x_tb, Xt, yt, gt,
                      fourclass.argmax_codes(nx.proba(s_root[0], Xt)))
            (t / 'temp' / 'candidates').mkdir(parents=True)
            rec_t = tm.run_search('tmp', s_root, data_t, t / 'temp' / 'candidates', t / 'temp' / 'ck', None, features)
            self.assertEqual((rec_r['e1'], rec_t['e1'], rec_t['increment_source']), ('brier_crisis',) * 2 + ('root',))
            maps = {'random': mt_load(t / 'rand', 'rnd'), 'temporal': mt_load(t / 'temp', 'tmp')}
            for k, d in (('random', t / 'rand'), ('temporal', t / 'temp')):
                out[f'{k}_files'] = ((d / 'candidates' / ('rnd' if k == 'random' else 'tmp') / 'assignment_evidence.csv')
                                     .read_bytes(), (rec_r if k == 'random' else rec_t)['checkpoints']['sha256'])
            with patch.dict(plan.FIT_SUPPORT, MapTransfer.FLOOR):
                out['arms'] = {k: tm.common_arm(current[0], data['FIT'], maps[k][0]) for k in maps}
            out['maps'] = maps
            if not mutate_e3:                                 # kept as the 'saved' run for the comparator
                import shutil
                shutil.copytree(t / 'rand', Path(self._keep.name) / 'rand')
        return out

    def test_search_and_common_refit_isolated_from_e3_and_from_search_root(self):
        tm = self.tm
        self._keep = tempfile.TemporaryDirectory()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                data = self._data(tmp)
            a = self._searches(data)
            b = self._searches(data, mutate_e3=True)
            self.assertEqual(a['random_files'], b['random_files'])
            self.assertEqual(a['temporal_files'], b['temporal_files'])
            for k in ('random', 'temporal'):
                self.assertEqual({r: nx.raw(m) for r, m in a['arms'][k][0].items()},
                                 {r: nx.raw(m) for r, m in b['arms'][k][0].items()})
            fit_keys = set(zip(data['FIT'][2].tolist(), data['FIT'][3].tolist()))
            cur_sha, search_sha = nx.sha(a['current'][0]), nx.sha(a['search_root'][0])
            self.assertNotEqual(cur_sha, search_sha)
            self.assertNotEqual(a['search_root'][1]['fit_keys_sha256'], nx.keys_sha(data['FIT'][2], data['FIT'][3]))
            n_models = 0
            for k in ('random', 'temporal'):
                models, recs = a['arms'][k]
                for region, rec in recs.items():
                    rows = [x for x, s in zip(zip(data['FIT'][2].tolist(), data['FIT'][3].tolist()),
                                              mt_ids(data['FIT'][2], a['maps'][k][0])) if s == region]
                    self.assertTrue(set(rows) <= fit_keys)
                    self.assertEqual(rec['support']['rows'], len(rows))
                    if region in models:
                        n_models += 1
                        self.assertEqual(rec['continuation']['parent_sha256'], cur_sha)
                        self.assertEqual(rec['continuation']['child_prefix_structure_sha256'],
                                         nx.prefix_identity(a['current'][0])['sha256'])
            self.assertGreater(n_models, 0)
            # production comparator: a saved run vs a replay with mutated E3 truth -> map/UBJ equal, E3 differs
            saved = Path(self._keep.name) / 'rand'
            with tempfile.TemporaryDirectory() as tmp, patch.object(trans, 'CONTIGUITY', False), \
                    patch.object(trans, 'generate_count_grid', return_value=(None, 0, 1)), \
                    patch.dict(plan.FIT_SUPPORT, FLOORS['fit_support']), \
                    patch.dict(plan.STAGE1_VAL_SUPPORT, FLOORS['val_support']), \
                    patch.object(TimeBlockContrast.mgf(), 'MAX_DEPTH', 3), redirect_stdout(StringIO()):
                pool = a['pool']
                Xt, yt, gt, _ = data['E3']
                for mutate, expect in ((False, True), (True, False)):
                    d = Path(tmp) / str(mutate)
                    (d / 'candidates').mkdir(parents=True)
                    data_p, conf = tm.production_inputs(pool, (Xt, (yt + 1) % 4 if mutate else yt, gt), a['current'][0])
                    tm.run_search('rnd', a['current'], data_p, d / 'candidates', d / 'ck', None,
                                  ff.load_schema(SCHEMA)['ordered_features'], confirmation=conf)
                    res = tm.compare_production(saved / 'candidates' / 'rnd', saved / 'ck' / 'rnd',
                                                d / 'candidates' / 'rnd', d / 'ck' / 'rnd')
                    self.assertEqual(res['passed'], expect)
                    self.assertEqual((res['checks']['assignment_evidence_bytes'], res['checks']['routed_ubj_bytes'],
                                      res['checks']['decisions']), (0, 0, 0))
                    self.assertEqual(res['checks']['target_predictions.csv'], int(mutate))
        finally:
            self._keep.cleanup()

    def test_fallbacks_and_foreign_root_refused(self):
        tm = self.tm
        X, y, g, mm = MapTransfer()._fit_pool()
        area_map = {**{a: 'root' for a in range(40)}, **{a: '1' for a in range(40, 45)},
                    **{a: 's-1' for a in range(45, 50)}}          # unsplit searched root + small region + s-1
        root, _ = nx.fit_global(X, y, SMALL_G['G1'])
        with patch.dict(plan.FIT_SUPPORT, MapTransfer.FLOOR):
            models, recs = tm.common_arm(root, (X, y, g, mm), area_map)
        self.assertEqual(set(models), {'root'})
        self.assertFalse(recs['1']['eligible'])
        sid, reason = mt_route(np.array([3, 41, 47, 99]), area_map, recs)
        self.assertEqual(reason.tolist(), ['region', 'insufficient_support', 's-1', 'missing'])
        other, _ = nx.fit_global(X[::2], y[::2], SMALL_G['G1'])
        real = tm.mt.common_refit
        with patch.dict(plan.FIT_SUPPORT, MapTransfer.FLOOR), \
                patch.object(tm.mt, 'common_refit', side_effect=lambda r, fit, am: real(other, fit, am)):
            with self.assertRaises(RuntimeError):
                tm.common_arm(root, (X, y, g, mm), area_map)

    def test_failed_first_unit_stops_later_units(self):
        import pickle
        tm = self.tm
        names = {plan.e1pair_root_name(h, t, plan.TB3_G[str(h)]) for h, t in tm.SCHEDULE}
        for fail_equiv in (True, False):
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                run, d35, out = tmp / 'd34', tmp / 'd35', tmp / 'out'
                (run / 'prepared' / 'ledgers').mkdir(parents=True)
                (run / 'prepared' / 'geometry').mkdir(parents=True)
                (run / 'prepared' / 'ledgers' / 'dev_baselines.csv').write_text('horizon,target_label\n', encoding='utf-8')
                (run / 'prepared' / 'geometry' / 'polygon_contiguity_info.pkl').write_bytes(pickle.dumps(None))
                d35.mkdir()
                (d35 / 'identity.json').write_text(json.dumps({'stage': 'd35_global_increment', 'producer_rev': '7b2bf6f',
                                                               'script_commit': 'be5f4854e5'}), encoding='utf-8')
                argv = ['x', '--d34-run', str(run), '--d35-run', str(d35), '--out', str(out)]
                eq = {'side_effect': tm.GateError('replay mismatch')} if fail_equiv else \
                    {'return_value': {'passed': True, 'checks': {}}}
                with patch.object(sys, 'argv', argv), \
                        patch.object(tm.rid, 'code_identity_at', return_value={'sha256': 'x'}), \
                        patch.object(tm.rid, 'code_identity', return_value={'sha256': 'x'}), \
                        patch.object(tm.rid, 'file_sha256', return_value='0' * 64), \
                        patch.object(tm, 'accept_mode', return_value=({n: {} for n in names}, {},
                                                                      {'stage': str(tmp / 'stage')})), \
                        patch.object(tm, 'run_equivalence', **eq) as re_, \
                        patch.object(tm, 'run_pair', side_effect=tm.GateError('gate')) as rp, \
                        redirect_stdout(StringIO()):
                    code = tm.main()
                self.assertEqual(code, 2)
                self.assertEqual((re_.call_count, rp.call_count), (1, 0) if fail_equiv else (3, 1))
                gate = json.loads((out / 'gate.json').read_text(encoding='utf-8'))
                self.assertFalse(gate['passed'])
                self.assertEqual(gate['budget']['search_roots'], 0)
                self.assertEqual(list(gate['pairs'])[-1], gate['stopped_at'])
                self.assertTrue((out / 'identity.json').is_file())
                self.assertFalse((out / 'summary.json').exists() or (out / 'completion.json').exists())


def mt_load(stage, cand):
    from scripts import stage1_map_transfer as mt
    return mt.load_map(stage, cand)


def mt_ids(areas, area_map):
    from scripts import stage1_map_transfer as mt
    return mt.region_ids(areas, area_map)


def mt_route(areas, area_map, recs):
    from scripts import stage1_map_transfer as mt
    return mt.route(areas, area_map, recs)


class ClassWeightRoot(unittest.TestCase):
    """D46/A20: gate before fit, FIT-label-only 2:1 class weights, post-hoc x2 control."""

    def setUp(self):
        from scripts import stage1_class_weight_root as cw
        self.cw = cw

    def test_failed_gate_prevents_any_fit(self):
        cw = self.cw
        failed = ({'root': 'r1', 'checks': {'E3_p_pooled': {'mismatches': 1, 'n': 4}}, 'passed': False},
                  None, None, None)
        passed = ({'root': 'r1', 'checks': {}, 'passed': True}, None, None, None)
        cases = ((cw.GateError('membership mismatch'), 'G1'), (failed, 'G1'), (passed, 'G4'))  # last: H4 needs G1
        for effect, g in cases:
            with tempfile.TemporaryDirectory() as tmp:
                stage, out = Path(tmp) / 'stage', Path(tmp) / 'out'
                out.mkdir()
                for name in ('r1', 'r2'):
                    (stage / 'roots' / name).mkdir(parents=True)
                    (stage / 'roots' / name / 'root.json').write_text(json.dumps(
                        {'horizon': 4, 'g_config': g, 'target_month': '2018-06'}), encoding='utf-8')
                roots = {'r1': {'horizon': 4}, 'r2': {'horizon': 4}}
                cands = {f'{r}_{e}': {'root': r, 'e1': e} for r in roots for e in ('hard_f1', 'brier_crisis')}
                kw = {'side_effect': effect} if isinstance(effect, Exception) else {'return_value': effect}
                with patch.object(cw.rr, 'gate_root', **kw) as gate, \
                        patch.object(cw.nx, 'fit_global') as fit, redirect_stdout(StringIO()):
                    code, gates, per_root = cw.run_all(Path(tmp) / 'run', stage, out, roots, cands, 'rev', 0, None)
                self.assertEqual(code, 2)
                fit.assert_not_called()
                self.assertEqual(gate.call_count, 1)                    # stopped before r2
                self.assertEqual(per_root, {})
                self.assertFalse((out / 'r1').exists())
                failure = json.loads((out / 'failure.json').read_text(encoding='utf-8'))
                self.assertEqual((failure['failed_root'], failure['not_attempted']), ('r1', ['r2']))
                self.assertFalse(json.loads((out / 'gate.json').read_text(encoding='utf-8'))['passed'])

    def test_weights_use_fit_labels_only_and_fit_checks(self):
        cw = self.cw
        w = cw.crisis_weights([0, 1, 2, 3, 3])
        np.testing.assert_array_equal(w['w64'], np.array([1, 1, 2, 2, 2]) / 1.6)
        helper = GlobalIncrementControl()
        helper.setUp()
        runs = []
        with patch.dict(plan.G_CONFIGS, {'G1': SMALL_G['G1']}):
            for permute in (False, True):
                with tempfile.TemporaryDirectory() as tmp:
                    data = helper._run(tmp, permute_held=permute)
                b, rec, wt = cw.fit_weighted_root(data, 'G1')
                cw.check_fit(b, rec, wt, 'G1', len(data['FIT'][1]))
                runs.append((data, b, rec, wt))
            (d1, b1, r1, w1), (d2, b2, r2, w2) = runs
            self.assertFalse((d2['C'][1] == d1['C'][1]).all())          # held-out labels did change
            for k in range(4):
                np.testing.assert_array_equal(d1['FIT'][k], d2['FIT'][k])
            self.assertEqual(w1['w32'].tobytes(), w2['w32'].tobytes())
            self.assertEqual(nx.raw(b1), nx.raw(b2))
            y = d1['FIT'][1]
            self.assertAlmostEqual(w1['w64'].mean(), 1.0, places=14)
            cv = w1['class_values_float32']
            self.assertAlmostEqual(cv['crisis'] / cv['noncrisis'], 2.0, places=6)
            np.testing.assert_array_equal(w1['w32'][y >= 2], np.float32(cv['crisis']))
            self.assertEqual(r1['sample_weight']['sha256'], hashlib.sha256(w1['w32'].tobytes()).hexdigest())
            for bad in ({**r1, 'base_score': '6E-1'}, {**r1, 'rounds_total': 7}):
                with self.assertRaises(RuntimeError):
                    cw.check_fit(b1, bad, w1, 'G1', len(y))
            tampered = {**w1, 'w32': w1['w32'] * np.float32(1.5)}
            with self.assertRaises(RuntimeError):
                cw.check_fit(b1, r1, tampered, 'G1', len(y))
        with self.assertRaises(cw.GateError):
            cw.crisis_weights([0, 1, 1])

    def test_posthoc2x_ratios_and_ranking(self):
        cw = self.cw
        rng = np.random.default_rng(2)
        p = rng.dirichlet(np.ones(4), size=500).astype(np.float32).astype(np.float64)
        p[:5] = p[5]                                                    # exact ties survive
        q = cw.posthoc2x(p)
        np.testing.assert_allclose(q[:, 0] / q[:, 1], p[:, 0] / p[:, 1], rtol=1e-12)
        np.testing.assert_allclose(q[:, 2] / q[:, 3], p[:, 2] / p[:, 3], rtol=1e-12)
        np.testing.assert_allclose(q.sum(axis=1), 1.0, atol=1e-15)
        s = cw.crisis_score(p)
        np.testing.assert_allclose(cw.crisis_score(q), 2 * s / (1 + s), rtol=0, atol=1e-15)
        chk = cw.posthoc_rank_check(p, q)
        self.assertTrue(chk['rank_preserved'])
        self.assertLess(chk['max_abs_map_error'], 1e-15)
        bad = q.copy()
        bad[[10, 11]] = bad[[11, 10]]                                   # perturbed control: order broken
        with self.assertRaises(cw.GateError):
            cw.posthoc_rank_check(p, bad)
        truth = rng.integers(0, 4, 500)
        ro, rq = cw.ranking(truth, p), cw.ranking(truth, q)
        self.assertAlmostEqual(ro['auc'], rq['auc'], places=12)
        self.assertAlmostEqual(ro['ap'], rq['ap'], places=12)
        self.assertFalse(cw.ranking(np.zeros(3, int), p[:3])['eligible'])

    def test_score_part_aggregate_and_logloss_stop(self):
        cw = self.cw
        p = np.array([[.7, .1, .1, .1], [.1, .1, .7, .1], [.4, .1, .4, .1], [.1, .6, .2, .1]])
        p = p.astype(np.float32).astype(np.float64)                    # native proba is float32-exact
        frame = pd.DataFrame({'truth': [0, 2, 3, 1], 'persistence_code': [0.0, 2.0, np.nan, 2.0]})
        pw = p[[1, 1, 2, 3]]
        for arm, pr in (('original', p), ('weighted', pw), ('posthoc2x', cw.posthoc2x(p))):
            frame[f'y_{arm}'] = fourclass.argmax_codes(pr)
            for k, lab in enumerate(fourclass.CLASS_LABELS):
                frame[f'p_{arm}_{lab}'] = pr[:, k]
        out = cw.score_part(frame)
        # original argmax [0,2,0,1] vs truth [0,2,3,1]: crisis tp1 fp0 fn1 -> 2/3
        self.assertEqual(Fraction(out['all']['original']['crisis_f1_exact']), Fraction(2, 3))
        self.assertAlmostEqual(out['all']['original']['logloss_fourclass'],
                               -np.mean(np.log(p[[0, 1, 2, 3], [0, 2, 3, 1]])), places=14)
        mp = out['matched_persistence']
        self.assertEqual(mp['n'], 3)
        self.assertNotIn('logloss_fourclass', mp['persistence'])
        self.assertIn('crisis_brier', mp['persistence'])
        agg = cw.aggregate({'a': {'scores': {pt: out for pt in cw.PARTS}},
                            'b': {'scores': {pt: out for pt in cw.PARTS}}}, lambda r: True)
        e3 = agg['E3']
        self.assertEqual((e3['folds_with_data'], e3['rows']), (2, 8))
        self.assertEqual(e3['pooled_all']['original']['crisis_f1_exact'], '2/3')
        self.assertAlmostEqual(e3['pooled_all']['original']['logloss_fourclass'],
                               out['all']['original']['logloss_fourclass'], places=14)
        self.assertEqual(e3['ranking_mean_fold']['original']['eligible_folds'], 2)
        self.assertIn('weighted_minus_persistence', e3['matched_persistence']['pooled'])
        bad = p.copy()
        bad[0] = [0.0, .5, .25, .25]
        with self.assertRaises(cw.GateError):
            cw.log_loss_fourclass([0, 2, 3, 1], bad)


class StumpRoot(unittest.TestCase):
    """D47/A21: gate before fit; copied depth-1 config; saved trees are stumps."""

    def setUp(self):
        from scripts import stage1_stump_root as st
        self.st = st

    def test_failed_gate_prevents_any_fit(self):
        st = self.st
        failed = ({'root': 'r1', 'checks': {'E3_p_pooled': {'mismatches': 1, 'n': 4}}, 'passed': False},
                  None, None, None)
        with tempfile.TemporaryDirectory() as tmp:
            stage, out = Path(tmp) / 'stage', Path(tmp) / 'out'
            out.mkdir()
            for name in ('r1', 'r2'):
                (stage / 'roots' / name).mkdir(parents=True)
                (stage / 'roots' / name / 'root.json').write_text(json.dumps(
                    {'horizon': 4, 'g_config': 'G1', 'target_month': '2018-06'}), encoding='utf-8')
            roots = {'r1': {'horizon': 4}, 'r2': {'horizon': 4}}
            cands = {f'{r}_{e}': {'root': r, 'e1': e} for r in roots for e in ('hard_f1', 'brier_crisis')}
            with patch.object(st.rr, 'gate_root', return_value=failed) as gate, \
                    patch.object(st.nx, 'fit_global') as fit, redirect_stdout(StringIO()):
                code, gates, per_root = st.run_all(Path(tmp) / 'run', stage, out, roots, cands, 'rev', 0, None)
            self.assertEqual(code, 2)
            fit.assert_not_called()
            self.assertEqual(gate.call_count, 1)                        # stopped before r2
            self.assertEqual(per_root, {})
            self.assertFalse((out / 'r1').exists())
            failure = json.loads((out / 'failure.json').read_text(encoding='utf-8'))
            self.assertEqual((failure['failed_root'], failure['not_attempted']), ('r1', ['r2']))

    def test_config_copy_and_depth_checks(self):
        st = self.st
        before = (json.dumps(plan.G_CONFIGS, sort_keys=True), json.dumps(plan.XGB_BASE, sort_keys=True))
        snap = st.plan_defaults()
        for h, g in st.G_BY_H.items():
            c = st.stump_config(g)
            self.assertEqual(c['max_depth'], 1)
            self.assertEqual({k: v for k, v in c.items() if k != 'max_depth'},
                             {k: v for k, v in plan.G_CONFIGS[g].items() if k != 'max_depth'})
            self.assertGreater(plan.G_CONFIGS[g]['max_depth'], 1)
        self.assertEqual((json.dumps(plan.G_CONFIGS, sort_keys=True), json.dumps(plan.XGB_BASE, sort_keys=True)),
                         before)
        st.check_defaults(snap)
        small = {**plan.G_CONFIGS['G1'], 'rounds': 6}                  # local copy; plan untouched
        X, y = synthetic(n=600, p=6, seed=3)
        stump, rec = nx.fit_global(X, y, dict(small, max_depth=1))
        s = st.check_fit(stump, rec, small, len(y))
        self.assertEqual((s['trees'], s['trees_depth_gt_1']), (24, 0))
        self.assertLessEqual(s['max_depth'], 1)
        self.assertEqual(s['split_nodes_total'] + s['trees'], s['leaf_nodes_total'])
        deep, deep_rec = nx.fit_global(X, y, small)
        self.assertGreater(st.tree_structure(deep)['max_depth'], 1)
        with self.assertRaisesRegex(st.GateError, 'depth > 1'):        # only the tree depth is wrong here
            st.check_fit(deep, rec, small, len(y))
        with self.assertRaises(st.GateError):
            st.check_fit(deep, deep_rec, small, len(y))
        self.assertEqual(st.plan_defaults(), snap)


class WeightedContinuation(unittest.TestCase):
    """Interruption-augmentation weights through the local continuation; synthetic rows only."""

    def setUp(self):
        self.X, self.y = synthetic(n=600, seed=5)
        self.root, self.root_rec = nx.fit_global(self.X, self.y, SMALL_G['G1'])
        self.sub = np.arange(0, 600, 2)
        self.w = np.tile([1 / 3, 1 / 3, 1 / 3, 1.0, 0.5, 2.0], 50)  # nonuniform, length 300

    def _reference(self, weight=None):
        params, rounds = nx.booster_params(plan.L_CONFIGS['L1'])
        dm = xgb.DMatrix(nx.clean(self.X[self.sub]), label=self.y[self.sub], missing=np.nan, nthread=4)
        if weight is not None:
            dm.set_weight(np.asarray(weight, dtype=np.float32))
        return xgb.train(params, dm, num_boost_round=rounds, xgb_model=nx.from_raw(nx.raw(self.root)))

    def test_default_call_is_backward_compatible(self):
        child, rec = nx.continue_booster(self.root, self.X[self.sub], self.y[self.sub], plan.L_CONFIGS['L1'])
        self.assertNotIn('sample_weight', rec)
        self.assertEqual(set(rec), {"kind", "parent_sha256", "parent_rounds", "rounds_added", "rounds_total",
                                    "params", "resolved_config", "base_score", "rows", "class_counts",
                                    "parent_structure_sha256", "child_prefix_structure_sha256",
                                    "structure_sha256", "prefix_check", "booster_sha256"})
        self.assertEqual(nx.raw(child), nx.raw(self._reference()))

    def test_nonuniform_weights_match_direct_reference_and_keep_prefix(self):
        before = nx.raw(self.root)
        child, rec = nx.continue_booster(self.root, self.X[self.sub], self.y[self.sub], plan.L_CONFIGS['L1'],
                                         sample_weight=self.w)
        self.assertEqual(nx.raw(self.root), before)                          # parent bytes unchanged
        self.assertEqual(nx.raw(child), nx.raw(self._reference(self.w)))     # direct xgb.train reference
        unweighted, _ = nx.continue_booster(self.root, self.X[self.sub], self.y[self.sub], plan.L_CONFIGS['L1'])
        self.assertNotEqual(nx.raw(child), nx.raw(unweighted))
        n0 = self.root.num_boosted_rounds()
        self.assertEqual(rec['child_prefix_structure_sha256'], self.root_rec['structure_sha256'])
        self.assertEqual(nx.prefix_identity(child, n0)['sha256'], nx.prefix_identity(self.root)['sha256'])
        np.testing.assert_array_equal(
            child.predict(nx.dmatrix(self.X), output_margin=True, iteration_range=(0, n0)),
            self.root.predict(nx.dmatrix(self.X), output_margin=True))
        w32 = self.w.astype(np.float32)
        self.assertEqual(rec['sample_weight']['dtype'], 'float32')
        self.assertEqual(rec['sample_weight']['n'], 300)
        self.assertEqual(rec['sample_weight']['sha256'], hashlib.sha256(w32.tobytes()).hexdigest())
        self.assertAlmostEqual(rec['sample_weight']['sum'], float(w32.astype(np.float64).sum()))
        self.assertEqual(rec['rows'], 300)                                    # rows, not weighted support

    def test_malformed_weights_are_rejected_before_fitting(self):
        before = nx.raw(self.root)
        for bad in (np.ones(299), np.ones((300, 1)), np.r_[np.ones(299), np.nan], np.r_[np.ones(299), 0.0],
                    np.r_[np.ones(299), -1.0], np.r_[np.ones(299), np.inf], np.r_[np.ones(299), 1e300]):
            with self.assertRaises(ValueError):
                nx.continue_booster(self.root, self.X[self.sub], self.y[self.sub], plan.L_CONFIGS['L1'],
                                    sample_weight=bad)
        self.assertEqual(nx.raw(self.root), before)

    def test_xgbmodel_train_forwards_weights_in_root_and_parent_modes(self):
        weighted, _ = nx.continue_booster(self.root, self.X[self.sub], self.y[self.sub], plan.L_CONFIGS['L1'],
                                          sample_weight=self.w)
        plain, _ = nx.continue_booster(self.root, self.X[self.sub], self.y[self.sub], plan.L_CONFIGS['L1'])
        for source in nx.INCREMENT_SOURCES:
            with tempfile.TemporaryDirectory() as tmp:
                model = nx.XGBmodel(tmp, plan.L_CONFIGS['L1'], increment_source=source)
                model.set_root(nx.from_raw(nx.raw(self.root)), self.root_rec)
                model.load('')
                model.train(self.X[self.sub], self.y[self.sub], '', sample_weight=self.w)
                self.assertEqual(nx.raw(model.booster), nx.raw(weighted), source)
                self.assertEqual(model.fit_record['sample_weight']['n'], 300)
                model.load('')
                model.train(self.X[self.sub], self.y[self.sub], '')                 # default unchanged
                self.assertEqual(nx.raw(model.booster), nx.raw(plain), source)
                self.assertNotIn('sample_weight', model.fit_record)
                self.assertEqual(nx.raw(nx.from_raw(model._store[''][0])), nx.raw(self.root))  # root intact


class ReleaseAwareViews(unittest.TestCase):
    """Interruption design D1/G2/G4: release ledger, cycle masks and A/B views on synthetic inputs."""

    from src.experiment import availability as av

    CYCLES = [m(f'{y}-{mm:02d}') for y in range(2014, 2021) for mm in (2, 6, 10)]

    def setUp(self):
        self.av = type(self).av
        # Areas 1, 2 in AAA (released the month after the cycle), area 3 in BBB (same month).
        self.area_country = {1: 'AAA', 2: 'AAA', 3: 'BBB'}
        self.obs = pd.DataFrame([(a, c, self.area_country[a], (a + c // 4) % 4)
                                 for a in (1, 2, 3) for c in self.CYCLES],
                                columns=['area', 'month', 'country', 'class_code'])
        self.ledger_rows = pd.DataFrame(
            [(f'CS-{ff.month_label([c])[0]}', 'CS', ctry, ff.month_label([c])[0],
              f'{ff.month_label([c + lag])[0]}-{day:02d}', 'synthetic', 'unit-test fixture')
             for c in self.CYCLES for ctry, lag, day in (('AAA', 1, 15), ('BBB', 0, 28))],
            columns=list(self.av.LEDGER_COLUMNS))
        months = pd.date_range('2014-01-01', '2021-12-01', freq='MS')
        panel = pd.DataFrame([(a, d) for a in (1, 2, 3) for d in months], columns=['FEWSNET_admin_code', 'date'])
        idx = ff.month_index(panel['date'])
        panel['crop'] = panel['FEWSNET_admin_code'].astype(float)
        panel['EVI'] = (idx + 1000 * panel['FEWSNET_admin_code']).astype(float)
        panel['GDP'] = (idx // 12).astype(float)          # annual value repeated monthly
        self.scaffold = ff.Scaffold(panel, ['crop', 'EVI', 'GDP'])
        hist = list(ff.history_features(pd.DataFrame(columns=['area', 'month', 'phase']), [1], [m('2020-06')]))
        evi_lags = [f'EVI_l{k}' for k in range(1, 13)]
        self.schema = {'static_sources': ['crop'], 'dynamic_sources_at_origin': ['EVI', 'GDP'],
                       'legacy_covariate_derived': evi_lags, 'known_calendar': ['target_year', 'target_month_sin',
                                                                                'target_month_cos'],
                       'ordered_features': ['crop', 'EVI', 'GDP'] + evi_lags +
                       ['target_year', 'target_month_sin', 'target_month_cos'] + hist}
        self.alignment = {
            'crop': {'kind': 'static', 'status': 'synthetic', 'evidence': 'fixture'},
            'EVI': {'kind': 'monthly', 'lag': 1, 'status': 'synthetic', 'evidence': 'fixture'},
            'GDP': {'kind': 'annual', 'release_delay_years': 1, 'release_month': 7, 'value_month': 12,
                    'status': 'synthetic', 'evidence': 'fixture'}}

    def ctx(self, obs=None, ledger=None, alignment='default', horizon=4):
        led = self.av.ReleaseLedger(self.ledger_rows if ledger is None else ledger)
        obs = self.obs if obs is None else obs
        return self.av.Availability(obs, led, self.scaffold, self.schema, horizon,
                                    self.alignment if alignment == 'default' else alignment, truth=obs)

    def test_ledger_contract_and_real_run_refusals(self):
        rows = self.ledger_rows
        with self.assertRaises(ValueError):
            self.av.ReleaseLedger(pd.concat([rows, rows.iloc[:1]]))               # duplicate cycle row
        early = rows.copy()
        early.loc[0, 'release_date'] = '2013-12-01'                              # before its reference month
        with self.assertRaises(ValueError):
            self.av.ReleaseLedger(early)
        with self.assertRaises(ValueError):
            self.av.ReleaseLedger(rows, real=True)                               # synthetic rows refused
        recon = rows.assign(evidence='reconstructed').iloc[1:]                   # one (country, cycle) missing
        led = self.av.ReleaseLedger(recon, real=True)
        with self.assertRaises(ValueError):                                      # missing calendar blocks
            self.av.Availability(self.obs, led, self.scaffold, self.schema, 4,
                                 {k: {**v, 'status': 'reconstructed'} for k, v in self.alignment.items()})
        lenient = self.av.Availability(self.obs, self.av.ReleaseLedger(rows.iloc[1:]), self.scaffold,
                                       self.schema, 4, self.alignment)
        self.assertEqual(lenient.unreleased, 2)                                  # AAA areas 1, 2: never visible

    def test_cycle_masks_are_publication_identities(self):
        led = self.av.ReleaseLedger(self.ledger_rows)
        self.assertEqual(led.hidden(m('2020-06'), 0), frozenset())
        self.assertEqual(led.hidden(m('2020-06'), 1), {m('2020-06')})            # BBB released in June: due
        self.assertEqual(led.hidden(m('2020-06'), 2), {m('2020-02'), m('2020-06')})
        self.assertEqual(led.hidden(m('2020-05'), 1), {m('2020-02')})            # June not yet due
        with self.assertRaises(self.av.UnsupportedScenario):
            led.hidden(m('2014-02'), 2)                                          # never a smaller k

    def test_delayed_release_hidden_and_future_labels_and_persistence(self):
        c = self.ctx()
        T, O = m('2020-10'), m('2020-06')
        k0 = c.prediction_view([1, 3], [T, T], 0)
        self.assertTrue(np.isnan(k0.loc[0, 'hist_phase_o00']))                   # AAA June released in July
        self.assertEqual(k0.loc[1, 'hist_phase_o00'], (3 + O // 4) % 4 + 1)     # BBB June released by cutoff
        self.assertEqual(k0.loc[0, 'persistence_source_month'], m('2020-02'))   # prolonged-lag persistence
        self.assertEqual(k0.loc[0, 'persistence_age'], 4)
        self.assertEqual(k0.loc[1, 'persistence_age'], 0)
        k2 = c.prediction_view([1, 3], [T, T], 2)
        for col in ('hist_phase_o00', 'hist_phase_o04'):
            self.assertTrue(np.isnan(k2.loc[1, col]))                            # exact lags stay missing
        self.assertEqual(k2.loc[1, 'hist_phase_o08'], (3 + m('2019-10') // 4) % 4 + 1)
        self.assertEqual(k2.loc[1, 'hist_latest_observed_age'], 8)               # older source keeps its age
        self.assertEqual(k2.loc[1, 'persistence_source_month'], m('2019-10'))
        self.assertEqual(k2.loc[1, 'persistence_class_code'], (3 + m('2019-10') // 4) % 4)
        changed = self.obs.copy()
        hidden_or_future = changed['month'].isin([m('2020-02'), m('2020-06'), m('2020-10')])
        changed.loc[hidden_or_future, 'class_code'] = 3 - changed.loc[hidden_or_future, 'class_code']
        pd.testing.assert_frame_equal(self.ctx(obs=changed).prediction_view([1, 3], [T, T], 2), k2)
        older = self.obs.copy()
        older.loc[older['month'] == m('2019-10'), 'class_code'] = 3 - older.loc[older['month'] == m('2019-10'),
                                                                                 'class_code']
        self.assertFalse(self.ctx(obs=older).prediction_view([1, 3], [T, T], 2).equals(k2))

    def test_label_pool_outer_mask_and_delayed_label(self):
        c = self.ctx()
        X = m('2020-10')
        self.assertEqual(c.label_pool(X, c.ledger.hidden(X, 1))['month'].max(), m('2020-06'))
        pool2 = c.label_pool(X, c.ledger.hidden(X, 2))
        self.assertEqual(pool2['month'].max(), m('2020-02'))                     # outer mask removes June for all
        self.assertGreaterEqual(pool2['month'].min(), X - plan.WINDOW)
        late = self.ledger_rows.copy()
        late.loc[(late['country'] == 'AAA') & (late['reference_month'] == '2020-02'), 'release_date'] = '2020-08-01'
        pool = self.ctx(ledger=late).label_pool(m('2020-06'))
        feb = pool[pool['month'] == m('2020-02')]
        self.assertEqual(sorted(feb['area']), [3])                               # AAA Feb not out by June
        self.assertEqual(self.ctx().gate_dates(X, 1),
                         [m(t) for t in ('2018-10', '2019-02', '2019-06', '2019-10', '2020-02', '2020-06')])

    def test_strategy_b_groups_conserve_weight_and_original_support(self):
        c = self.ctx()
        X = m('2020-10')
        a, sup_a = c.fitting_view(X, 1, 'A')
        b, sup_b = c.fitting_view(X, 1, 'B')
        self.assertEqual(sup_a, sup_b)                                           # original keys only
        self.assertEqual(sup_b['rows'], a['orig_key'].nunique())
        self.assertEqual(len(b), 3 * len(a))
        np.testing.assert_allclose(b.groupby('orig_key')['weight'].sum(), 1.0)
        self.assertTrue((b.groupby('orig_key')['variant_k'].apply(sorted).map(tuple) == (0, 1, 2)).all())
        self.assertTrue((b.groupby('orig_key')[['area', 'target_month', 'class_code']].nunique() == 1).all().all())
        pd.testing.assert_frame_equal(b[b['variant_k'] == 0].drop(columns=['variant_k', 'weight']).reset_index(drop=True),
                                      a.drop(columns=['variant_k', 'weight']).reset_index(drop=True))
        key = b[(b['area'] == 3) & (b['target_month'] == m('2020-02'))].set_index('variant_k')  # origin 2019-10
        self.assertEqual(key.loc[0, 'hist_phase_o00'], (3 + m('2019-10') // 4) % 4 + 1)
        self.assertTrue(np.isnan(key.loc[1, 'hist_phase_o00']))                  # own latest cycle hidden
        self.assertTrue(np.isnan(key.loc[2, 'hist_phase_o04']))
        self.assertEqual(key.loc[2, 'hist_phase_o08'], (3 + m('2019-02') // 4) % 4 + 1)

    def test_inherited_exclusions_reach_labels_and_every_variant(self):
        c = self.ctx()
        V, gone = m('2020-06'), frozenset({m('2019-10')})                        # internal origin, outer cycle
        frame, _ = c.fitting_view(V, 0, 'B', excluded=gone)
        self.assertNotIn(m('2019-10'), set(frame['target_month']))               # label removed
        late_keys = frame[frame['origin_month'] >= m('2019-10')]
        self.assertTrue(len(late_keys))
        self.assertTrue((late_keys['hist_w12_n_obs'] <= 2).all())               # 2019-10 never in any history
        ref = c.key_features(late_keys['area'], late_keys['target_month'], late_keys['variant_k'], gone)
        pd.testing.assert_frame_equal(late_keys[c.features].reset_index(drop=True), ref)

    def test_monthly_lag_annual_eligible_year_and_release_evidence(self):
        O = np.array([m('2020-06'), m('2020-07')])
        out = ff.covariate_features(self.scaffold, self.schema, [3, 3], O + 4, O, self.alignment)
        np.testing.assert_array_equal(out['EVI'], 3000 + O - 1)                 # exact source month O - 1
        np.testing.assert_array_equal(out['EVI_l1'], 3000 + O - 2)
        np.testing.assert_array_equal(out['GDP'], [2018, 2019])                  # July release of Y-1
        delayed = {**self.alignment, 'GDP': {**self.alignment['GDP'], 'releases': {2018: '2019-07-01',
                                                                                  2019: '2020-09-15'}},
                   'EVI': {**self.alignment['EVI'], 'releases': {'2020-05': '2020-06-30', '2020-06': '2020-08-02'}}}
        ev = ff.covariate_features(self.scaffold, self.schema, [3, 3], O + 4, O, delayed)
        np.testing.assert_array_equal(ev['GDP'], [2018, 2018])                   # delayed 2019 not exposed
        self.assertEqual(ev.loc[0, 'EVI'], 3000 + O[0] - 1)
        self.assertTrue(np.isnan(ev.loc[1, 'EVI']))                              # June not out: NaN, no search back
        self.assertTrue(np.isnan(ev.loc[0, 'EVI_l1']))                           # no evidence for April
        prov = self.ctx(alignment=delayed).provenance(O)
        np.testing.assert_array_equal(prov['prov_GDP_ref_year'], [2018, 2018])
        np.testing.assert_array_equal(prov['prov_GDP_age_years'], [2, 2])
        np.testing.assert_array_equal(prov['prov_EVI_source_month'], O - 1)
        dropped = {**self.alignment, 'EVI': {'kind': 'excluded', 'status': 'synthetic', 'evidence': 'fixture'}}
        names = ff.aligned_feature_names(self.schema, dropped)
        self.assertFalse({'EVI', 'EVI_l1', 'EVI_l12'} & set(names))
        self.assertFalse({'EVI', 'EVI_l1'} & set(ff.covariate_features(self.scaffold, self.schema, [3], [O[0] + 4],
                                                                       [O[0]], dropped)))
        for bad in ({}, {**self.alignment, 'GDP': {'kind': 'annual', 'status': 'synthetic', 'evidence': 'x'}},
                    {**self.alignment, 'EVI': {**self.alignment['EVI'], 'evidence': ''}},
                    {**self.alignment, 'EVI': {**self.alignment['EVI'], 'releases': {'2020-05': '2020-04-01'}}}):
            with self.assertRaises(ValueError):
                ff.covariate_features(self.scaffold, self.schema, [3], [O[0] + 4], [O[0]], bad)
        with self.assertRaises(ValueError):
            ff.check_alignment(self.schema, self.alignment, real=True)

    def test_cycle_identity_order_and_real_alignment_refusals(self):
        rows = self.ledger_rows
        split = rows.copy()
        split.loc[split['country'] == 'BBB', 'cycle_id'] = split.loc[split['country'] == 'BBB', 'cycle_id'] + 'b'
        with self.assertRaises(ValueError):
            self.av.ReleaseLedger(split)                                         # two ids for one reference month
        merged = rows.copy()
        merged['cycle_id'] = 'CS-one'
        with self.assertRaises(ValueError):
            self.av.ReleaseLedger(merged)                                        # one id for many months
        swapped = rows.copy()
        swapped.loc[swapped['reference_month'] == '2019-06', 'release_date'] = '2019-12-01'
        with self.assertRaises(ValueError):                                      # publication order != reference order
            self.av.ReleaseLedger(swapped)
        led = self.av.ReleaseLedger(rows)
        self.assertEqual(led.due_rule, 'earliest_country_release')
        self.assertEqual(led.cycle_id[m('2020-06')], 'CS-2020-06')
        real = self.av.ReleaseLedger(rows.assign(evidence='reconstructed'), real=True)
        with self.assertRaises(ValueError):
            self.av.Availability(self.obs, real, self.scaffold, self.schema, 4, None)   # no legacy covariates

    def test_gate_dates_not_limited_to_fitting_window_and_empty_pools(self):
        old = self.obs[self.obs['month'] <= m('2015-10')]
        c = self.ctx(obs=old)
        O = m('2021-02')
        self.assertEqual(len(c.label_pool(O)), 0)                                # all labels older than O-59
        self.assertEqual(c.gate_dates(O, 0),                                    # lawful U < O, any age
                         [m(t) for t in ('2014-02', '2014-06', '2014-10', '2015-02', '2015-06', '2015-10')])
        frame, support = c.fitting_view(O, 0, 'B')
        self.assertEqual(len(frame), 0)
        self.assertEqual(support['rows'], 0)
        self.assertEqual(list(c.key_features([], [], 0).columns), c.features)

    def test_evaluator_truth_includes_labels_without_input_release(self):
        final = pd.DataFrame({'area': [1, 3], 'month': [m('2021-02')] * 2, 'country': ['AAA', 'BBB'],
                              'class_code': [2, 0]})                             # no ledger row for 2021-02
        truth = pd.concat([self.obs, final], ignore_index=True)
        c = self.av.Availability(self.obs, self.av.ReleaseLedger(self.ledger_rows), self.scaffold, self.schema,
                                 4, self.alignment, truth=truth)
        t = c.truth_view([1, 3], [m('2021-02')] * 2)
        self.assertEqual(list(t['truth_code']), [2, 0])
        self.assertEqual(list(t['truth_reason']), ['', ''])
        lawful = c.truth_view([1, 3], [m('2021-02')] * 2, lawful_at=m('2021-06'))
        self.assertEqual(list(lawful['truth_reason']), ['not_lawful_at_cutoff'] * 2)   # never an input
        self.assertNotIn(m('2021-02'), set(c.visible(m('2021-12'))['month']))

    def test_unlabelled_prediction_targets_keep_rows_and_truth_is_separate(self):
        c = self.ctx()
        T = m('2021-02')                                                         # no label exists
        view = c.prediction_view([1, 2, 3], [T] * 3, 1)
        self.assertEqual(len(view), 3)
        self.assertFalse({'class_code', 'truth_code'} & set(view.columns))
        truth = c.truth_view([1, 3], [T, m('2020-10')])
        self.assertEqual(list(truth['truth_reason']), ['no_label', ''])
        gate = c.truth_view([1, 3], [m('2020-06'), m('2020-06')], lawful_at=m('2020-06'))
        self.assertEqual(list(gate['truth_reason']), ['not_lawful_at_cutoff', ''])   # AAA June out in July
        masked = c.truth_view([3], [m('2020-06')], lawful_at=m('2020-10'), excluded={m('2020-06')})
        self.assertEqual(masked.loc[0, 'truth_reason'], 'not_lawful_at_cutoff')


def scenario_fixture(n_areas=40, truth=True, relabel=None, last_year=2020):
    """Synthetic release-aware Availability: AAA areas (< n/2) released M+1, BBB released M, one extra
    unlabelled prediction area, tri-annual cycles 2012-2020, two covariates (static, monthly lag 1)."""
    from src.experiment import availability as av
    cycles = [m(f'{y}-{mm:02d}') for y in range(2012, last_year + 1) for mm in (2, 6, 10)]
    country = {a: ('AAA' if a < n_areas // 2 else 'BBB') for a in range(n_areas)}
    rng = np.random.default_rng(7)
    obs = pd.DataFrame([(a, c, country[a], int((a // 10 + c // 4 + rng.integers(0, 2)) % 4))
                        for a in range(n_areas) for c in cycles], columns=['area', 'month', 'country', 'class_code'])
    if relabel is not None:                                                     # (area, month, class_code)
        obs.loc[(obs['area'] == relabel[0]) & (obs['month'] == relabel[1]), 'class_code'] = relabel[2]
    ledger = pd.DataFrame([(f'CS-{ff.month_label([c])[0]}', 'CS', ctry, ff.month_label([c])[0],
                            f'{ff.month_label([c + lag])[0]}-10', 'synthetic', 'unit-test fixture')
                           for c in cycles for ctry, lag in (('AAA', 1), ('BBB', 0))],
                          columns=list(av.LEDGER_COLUMNS))
    months = pd.date_range('2011-01-01', f'{max(last_year + 1, 2021)}-12-01', freq='MS')
    panel = pd.DataFrame([(a, d) for a in range(n_areas + 1) for d in months], columns=['FEWSNET_admin_code', 'date'])
    idx = ff.month_index(panel['date'])
    panel['crop'] = (panel['FEWSNET_admin_code'] % 7).astype(float)
    panel['EVI'] = np.sin(idx / 3 + panel['FEWSNET_admin_code'])
    scaffold = ff.Scaffold(panel, ['crop', 'EVI'])
    hist = list(ff.history_features(pd.DataFrame(columns=['area', 'month', 'phase']), [1], [m('2020-06')]))
    schema = {'static_sources': ['crop'], 'dynamic_sources_at_origin': ['EVI'], 'legacy_covariate_derived': [],
              'known_calendar': ['target_year', 'target_month_sin', 'target_month_cos'],
              'history_blocks': {'all': hist},
              'ordered_features': ['crop', 'EVI', 'target_year', 'target_month_sin', 'target_month_cos'] + hist}
    schema['proposed_feature_count'] = len(schema['ordered_features'])
    alignment = {'crop': {'kind': 'static', 'status': 'synthetic', 'evidence': 'fixture'},
                 'EVI': {'kind': 'monthly', 'lag': 1, 'status': 'synthetic', 'evidence': 'fixture'}}
    return av, av.Availability(obs, av.ReleaseLedger(ledger), scaffold, schema, 4, alignment,
                               truth=obs if truth else None), schema


class ScenarioStage3(unittest.TestCase):
    """Interruption task: Stage 3 engine on release-aware views (G2/G4), synthetic rows only."""

    def setUp(self):
        self.av, self.ctx, _ = scenario_fixture()
        self.cluster_of = {a: (0 if a < 20 else 1) for a in range(40)}
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [patch.dict(plan.G_CONFIGS, SMALL_G),
                        patch.dict(plan.FIT_SUPPORT, {'rows': 20, 'areas': 5, 'dates': 3, 'classes': 2}),
                        patch.dict(plan.STAGE3_GATE_SUPPORT, {'rows': 10, 'areas': 5, 'dates': 2, 'local_fit_dates': 2})]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def specs(self):
        return [{'arm': 'pooled', 'label': 'pooled', 'local': None, 'route': None},
                {'arm': 'shared', 'label': 'mapped', 'local': 'L1', 'route': 'learned_map',
                 'cluster_of': self.cluster_of}]

    def test_scenario_fold_lawful_pools_weights_gate_and_forecast_only_rows(self):
        panel = s3.ScenarioPanel(self.ctx, 1, 'B')
        store = s3.GlobalStore(Path(self.tmp.name) / 'g')
        O, T = m('2020-06'), m('2020-10')
        hidden = self.ctx.ledger.hidden(O, 1)
        res = s3.run_fold(panel, store, {'target_month': '2020-10', 'origin_month': '2020-06'}, 'G1', self.specs())
        preds = res['mapped']['predictions']
        self.assertEqual(sorted(preds['area']), list(range(41)))                # cohort incl. unlabelled area 40
        self.assertTrue(np.isnan(preds.loc[preds['area'] == 40, 'y_true_code']).all())
        self.assertEqual(preds.loc[preds['area'] == 40, 'route'].item(), 'unmapped_area_global')
        self.assertTrue((preds.loc[preds['area'] < 40, 'persistence_source_month'] <= O - 4).all())  # June hidden
        _, grec = store.get(panel, O, 'G1', fit_if_missing=False)
        pool = self.ctx.label_pool(O, hidden)
        self.assertEqual(grec['fit_support']['rows'], len(pool))                # original keys, not 3x rows
        self.assertEqual(grec['sample_weight']['n'], 3 * len(pool))
        self.assertAlmostEqual(grec['sample_weight']['sum'], len(pool), places=3)  # w/3 conserves total weight
        pairs = res['mapped']['gate_pairs']
        gates = self.ctx.gate_dates(O, 1)
        self.assertEqual(sorted(set(pairs['validation_month'])), [ml_ for ml_ in ff.month_label(gates)])
        for u in gates:                                                          # internal fits inherit the outer mask
            _, rec = store.get(panel, u - 4, 'G1', fit_if_missing=False, excluded=hidden, internal=True)
            self.assertEqual(rec['excluded_months'], list(ff.month_label(sorted(hidden))))
            lawful = self.ctx.lawful_labels(u, O, hidden)                       # input labels lawful at O
            used = pairs.loc[pairs['validation_month'] == ff.month_label([u])[0], 'area']
            self.assertEqual(sorted(used), sorted(lawful['area']))
        for dec in res['mapped']['gate']:
            self.assertEqual(dec['metric'], 'crisis_f1')
        pooled = res['pooled']['predictions']
        np.testing.assert_array_equal(pooled.filter(like='p_').to_numpy(),
                                      nx.proba(nx.from_raw((Path(self.tmp.name) / 'g').glob('h4/G1/O2020-06_*.ubj')
                                                           .__next__().read_bytes()),
                                               panel.target_rows(T)[0].X))

    def test_store_identity_separates_strategies_scenarios_and_masks(self):
        store = s3.GlobalStore(Path(self.tmp.name) / 'g')
        O = m('2020-06')
        got = {}
        for k, strategy in ((0, 'A'), (1, 'A'), (2, 'A'), (1, 'B')):
            got[(k, strategy)] = store.get(s3.ScenarioPanel(self.ctx, k, strategy), O, 'G1')
        shas = {key: rec['booster_sha256'] for key, (_, rec) in got.items()}
        self.assertEqual(len(store.memo), 4)                                    # one entry per identity
        # k=1 hides only the origin cycle itself (outside [O-59, O)): same bytes, separate identity
        self.assertEqual(shas[(0, 'A')], shas[(1, 'A')])
        self.assertEqual(got[(0, 'A')][1]['masked_months'], [])
        self.assertEqual(got[(1, 'A')][1]['masked_months'], ['2020-06'])
        self.assertEqual(got[(2, 'A')][1]['masked_months'], ['2020-02', '2020-06'])
        self.assertEqual(len({shas[(0, 'A')], shas[(2, 'A')], shas[(1, 'B')]}), 3)   # k=2 drops Feb labels
        self.assertNotIn('sample_weight', got[(2, 'A')][1])
        again = store.get(s3.ScenarioPanel(self.ctx, 1, 'B'), O, 'G1')
        self.assertIs(again[0], got[(1, 'B')][0])                                # exact identity reuse only
        path = next((Path(self.tmp.name) / 'g' / 'h4' / 'G1').glob('O2020-06_*.json'))
        record = json.loads(path.read_text())
        record['strategy'] = 'tampered'
        path.write_text(json.dumps(record))
        fresh = s3.GlobalStore(Path(self.tmp.name) / 'g')
        with self.assertRaises(RuntimeError):
            for k, strategy in ((0, 'A'), (1, 'A'), (2, 'A'), (1, 'B')):
                fresh.get(s3.ScenarioPanel(self.ctx, k, strategy), O, 'G1')

    def test_changed_fitting_label_never_reuses_a_stale_global(self):
        O = m('2020-06')
        _, other, _ = scenario_fixture(relabel=(3, m('2020-02'), 3 - int(self.ctx.obs.query('area == 3 and month == @m("2020-02")')
                                                                          ['class_code'].iloc[0])))
        root = Path(self.tmp.name) / 'g'
        store = s3.GlobalStore(root)
        a, b = s3.ScenarioPanel(self.ctx, 0, 'A'), s3.ScenarioPanel(other, 0, 'A')
        pa, ra = a.fit_pool(O)
        pb, rb = b.fit_pool(O)
        np.testing.assert_array_equal(pa.X, pb.X)                               # label is not in any history
        self.assertFalse(np.array_equal(pa.y, pb.y))
        _, rec_a = store.get(a, O, 'G1')
        _, rec_b = store.get(b, O, 'G1')
        self.assertNotEqual(rec_a['labels_sha256'], rec_b['labels_sha256'])
        self.assertNotEqual(rec_a['booster_sha256'], rec_b['booster_sha256'])  # recomputed, not reused
        _, rec_disk = s3.GlobalStore(root).get(b, O, 'G1', fit_if_missing=False)  # disk path keyed apart
        self.assertEqual(rec_disk['booster_sha256'], rec_b['booster_sha256'])

    def test_forecast_production_needs_no_evaluator_truth(self):
        _, prod, _ = scenario_fixture(truth=False)
        self.assertEqual(prod.inputs_sha256, self.ctx.inputs_sha256)             # truth never in input identity
        panel = s3.ScenarioPanel(prod, 1, 'A')
        src, rows = panel.target_rows(m('2020-10'))
        self.assertTrue(np.isnan(src.y.astype(float)).all())                     # no truth before its release
        ref, _ = s3.ScenarioPanel(self.ctx, 1, 'A').target_rows(m('2020-10'))
        np.testing.assert_array_equal(src.X, ref.X)
        ev, _ = panel.gate_rows(m('2019-10'), m('2020-06'), prod.ledger.hidden(m('2020-06'), 1))
        self.assertEqual(len(ev.y), 40)                                          # gates use lawful input labels

    def test_weighted_local_fit_uses_original_support(self):
        panel = s3.ScenarioPanel(self.ctx, 0, 'B')
        pool, rows = panel.fit_pool(m('2020-06'))
        store = s3.GlobalStore(Path(self.tmp.name) / 'g')
        g, _ = store.get(panel, m('2020-06'), 'G1')
        sub = rows[pool.area[rows] < 20]
        booster, rec = s3.fit_local('shared', g, pool, sub, 'G1', 'L1')
        self.assertEqual(rec['sample_weight']['n'], len(sub))
        self.assertEqual(rec['fit_support']['rows'], len(sub) // 3)
        _, ind = s3.fit_local('independent', g, pool, sub, 'G1', 'L1')
        self.assertEqual(ind['independent_prefix']['sample_weight']['n'], len(sub))

    def test_crisis_gate_exact_boundary_undefined_and_fold_gate_mismatch(self):
        n = 300
        pairs = pd.DataFrame({'area': np.arange(n) % 30, 'validation_month': np.repeat(['a', 'b', 'c'], 100),
                              'local_fit_ok': True, 'y_true': np.r_[np.full(100, 2), np.zeros(200, int)],
                              'y_global': np.zeros(n, int)})
        for fp, enabled in ((99, False), (98, True)):                          # 2/200 = .01 exactly vs 2/199
            local = np.zeros(n, int)
            local[0] = 2
            local[100:100 + fp] = 3
            dec = s3.gate_decision(pairs.assign(y_local_routed=local), 'crisis')
            self.assertEqual(dec['enabled'], enabled, fp)
            self.assertEqual(dec['crisis_f1_global'], 0.0)                      # defined zero: TP = 0, D > 0
        quiet = pairs.assign(y_true=0, y_local_routed=0)
        dec = s3.gate_decision(quiet, 'crisis')
        self.assertFalse(dec['enabled'])
        self.assertEqual(dec['reason'], 'gate_metric_undefined')
        self.assertIsNone(dec['crisis_f1_global'])
        panel = s3.ScenarioPanel(self.ctx, 0, 'A')
        with self.assertRaises(ValueError):
            s3.run_fold(panel, s3.GlobalStore(Path(self.tmp.name) / 'g'),
                        {'target_month': '2020-10', 'origin_month': '2020-06',
                         'gate': [{'validation_month': '2019-02'}]}, 'G1', self.specs())


class Stage1Variants(unittest.TestCase):
    """Interruption task G4: Stage 1 partition with grouped B variants (weights w/3, original-key support)."""

    def test_support_counts_original_keys_not_variant_copies(self):
        X, y, groups, split, months, labeller, proposal = Stage1Partition().fixture()
        fit = np.flatnonzero(split == 0)
        rows = np.r_[np.repeat(fit, 3), np.flatnonzero(split == 1)]          # three variants per fitting key
        key = rows.copy()
        Xv, yv, gv, sv, mv = X[rows], y[rows], groups[rows], split[rows], months[rows]
        floor = {'rows': 3, 'areas': 1, 'dates': 1, 'classes': 1}
        for keys, expected in ((None, 'accepted'), (key, 'rejected_no_eligible_child')):
            model = PresetModel('.', labeller)
            model.set_root(int((sv == 0).sum()))
            _, decisions = run_partition(model, Xv, yv, gv, sv, mv, proposal=proposal, fit_support=floor,
                                         X_key=keys)
            self.assertEqual(decisions[0]['outcome'], expected)                 # 6 copies vs 2 original keys
        self.assertEqual(decisions[0]['fit_support'][0]['rows'], 2)
        bad = PresetModel('.', labeller)
        bad.set_root(int((sv == 0).sum()))
        with self.assertRaises(ValueError):
            run_partition(bad, Xv, yv, gv, sv, mv, proposal=proposal, X_key=key[:-1])

    def test_real_children_receive_variant_weights(self):
        rng = np.random.default_rng(3)
        groups = np.repeat(np.arange(40), 30)
        X = rng.normal(size=(len(groups), 4)); X[:, 3] = groups >= 20
        y = ((X[:, 0] + 3 * X[:, 3] * X[:, 1]) > 0).astype(int) * 2
        months = np.tile(np.arange(30), 40)
        from src.utils.split import group_aware_train_val_split
        split = group_aware_train_val_split(groups, .5, 1, 42, True)['X_set']
        fit = np.flatnonzero(split == 0)
        rows = np.r_[np.repeat(fit, 3), np.flatnonzero(split == 1)]
        weight = np.where(split[rows] == 0, 1 / 3, 1.0)
        Xv, yv, gv, sv, mv = X[rows], y[rows], groups[rows], split[rows], months[rows]
        root, rec = nx.fit_global(Xv[sv == 0], yv[sv == 0], SMALL_G['G1'], sample_weight=weight[sv == 0])
        with tempfile.TemporaryDirectory() as tmp:
            model = nx.XGBmodel(tmp, plan.L_CONFIGS['L1'])
            model.set_root(root, rec)
            run_partition(model, Xv, yv, gv, sv, mv, threshold=Fraction(0), X_weight=weight, X_key=rows)
            children = [e for e in model.fit_log[1:] if e.get('kind') == 'continuation']
            self.assertTrue(children)
            for entry in children:
                self.assertEqual(entry['sample_weight']['n'], 3 * entry['fit_support']['rows'])
                self.assertAlmostEqual(entry['sample_weight']['sum'], entry['fit_support']['rows'], places=4)
                self.assertEqual(entry['parent_structure_sha256'], rec['structure_sha256'])


class ScenarioStage1(unittest.TestCase):
    """Interruption task G4: the 648 schedule, prepared root inputs and one real scenario root."""

    def setUp(self):
        self.av, self.ctx, self.schema = scenario_fixture()

    def test_schedule_is_the_frozen_648(self):
        sched = plan.scenario_stage1_schedule()
        self.assertEqual(len(sched), 648)
        per = pd.DataFrame(sched).groupby(['strategy', 'horizon']).size()
        self.assertTrue((per == 162).all())
        self.assertEqual({r['g_config'] for r in sched if r['horizon'] == 4}, {'G1'})
        self.assertEqual({r['g_config'] for r in sched if r['horizon'] == 8}, {'G4'})
        self.assertEqual({(r['local_config'], r['threshold_family']) for r in sched}, {('L1', 'gt0')})
        self.assertEqual(sorted({r['target_month'] for r in sched}), sorted(plan.STAGE1_TARGETS))

    def test_stage1_input_roles_eval_features_and_labelled_targets(self):
        T, k = m('2020-10'), 2
        frame = self.ctx.stage1_input(T, k, 'B')
        fit, ev, tgt = (frame[frame['role'] == r] for r in ('fit_variant', 'eval', 'target'))
        self.assertEqual(len(fit), 3 * len(ev))
        self.assertFalse(ev['orig_key'].duplicated().any())
        self.assertEqual(set(fit['orig_key']), set(ev['orig_key']))
        self.assertTrue((ev['variant_k'] == k).all() and (ev['weight'] == 1.0).all())
        outer = self.ctx.ledger.hidden(T - 4, k)
        ref = self.ctx.key_features(ev['area'], ev['target_month'], k, outer)
        pd.testing.assert_frame_equal(ev[self.ctx.features].reset_index(drop=True), ref)
        self.assertNotIn(40, set(tgt['area']))                                    # E3 needs a genuine label
        self.assertEqual(len(tgt), 40)
        self.assertFalse(set(ev['target_month']) & set(outer))                    # outer-hidden labels absent

    def test_prepared_inputs_and_runner_accept_only_the_frozen_648(self):
        from scripts import run_stage1 as s1
        ledger = self.ctx.ledger.frame[list(self.av.LEDGER_COLUMNS)]
        obs = self.ctx.obs[['area', 'month', 'country', 'class_code']]
        sched = plan.scenario_stage1_schedule()
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):                                    # synthetic evidence: real refuses
                prep.write_scenario_inputs(Path(t), obs, self.ctx.scaffold, self.schema, ledger, self.ctx.alignment)
            some = [dict(r) for r in sched[:6]] + [dict(sched[-1])]
            rows = prep.write_scenario_inputs(Path(t), obs, self.ctx.scaffold, self.schema, ledger,
                                              self.ctx.alignment, real=False, entries=some)
            names = sorted(p.name for p in (Path(t) / 'scenario').iterdir() if p.suffix == '.parquet')
            self.assertEqual(names, sorted({s1.scenario_input_name(r) for r in some}))   # shared per (s, H, T, k)
            pinned = json.loads((Path(t) / 'scenario' / 'features.json').read_text())['ordered_features']
            self.assertEqual(pinned, self.ctx.features)
            self.assertEqual([r['input'] for r in rows], [s1.scenario_input_name(r) for r in some])
            frame = pd.read_parquet(Path(t) / 'scenario' / rows[-1]['input'])
            self.assertEqual(set(frame['strategy']), {'B'})
            self.assertEqual(set(frame['scenario_k']), {2})
        with self.assertRaises(SystemExit):
            s1.scenario_entries({'stage1_scenario_roots': rows})                    # partial schedule refused
        full = [{**r, 'input': s1.scenario_input_name(r)} for r in sched]
        roots = s1.scheduled_roots({'stage1_scenario_roots': full}, {}, plan.SCENARIO)
        self.assertEqual(len(roots), 648)
        self.assertTrue(all(r['increment_source'] == 'root' for r in roots.values()))
        tampered = [dict(r) for r in full]
        tampered[0]['g_config'] = 'G2'
        with self.assertRaises(SystemExit):
            s1.scheduled_candidates({'stage1_scenario_roots': tampered}, {}, plan.SCENARIO)

    def test_no_target_root_is_recorded_without_fitting(self):
        from app import main_model_GF as mgf
        frame = self.ctx.stage1_input(m('2020-10'), 0, 'A')
        frame = frame[frame['role'] != 'target']
        with tempfile.TemporaryDirectory() as t:
            here = os.getcwd()
            os.chdir(t)
            try:
                rec = mgf.scenario_root(frame, 'r80', 42, 4, '2020-10', Path(t), Path(t) / 'ck', None,
                                        self.ctx.features)
            finally:
                os.chdir(here)
            self.assertEqual(rec['status'], 'no_e3_target_labels')
            self.assertEqual(rec['candidates'], [plan.scenario_candidate_name('A', 4, '2020-10', 0, 'r80', 42)])
            self.assertEqual(sorted(p.name for p in Path(t).iterdir()), ['root.json'])   # no fit, no checkpoints

    def test_cli_refuses_inputs_without_the_pinned_feature_order(self):
        import argparse
        from app import main_model_GF as mgf
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / 'scenario').mkdir()
            schema_path = t / 'schema.json'
            schema_path.write_text(json.dumps(self.schema))
            (t / 'scenario' / 'features.json').write_text(json.dumps({'ordered_features': self.ctx.features}))
            frame = self.ctx.stage1_input(m('2020-10'), 0, 'A').drop(columns=['EVI'])
            frame.to_parquet(t / 'scenario' / 'x.parquet', index=False)
            args = argparse.Namespace(forecasting_scope=1, desired_terms='2020-10', increment_source='root',
                                      g_config='G1', confirmation_split=False, e1_pair=False, recent_search=False,
                                      matched_size_seed=None, data=None, scenario_input=str(t / 'scenario' / 'x.parquet'),
                                      schema=str(schema_path), geometry_dir=str(t), ratio='r80', split_seed=42,
                                      checkpoint_dir=str(t / 'ck'))
            with self.assertRaises(ValueError):
                mgf.scenario_main(args)

    def test_undefined_e2_parent_metric_through_partition(self):
        X, y, groups, split, months, _, proposal = Stage1Partition().fixture()
        y = np.zeros_like(y)                                                       # no crisis anywhere

        def labeller(labels, ids):                                                 # children call crisis
            return np.zeros(len(ids), dtype=int) if labels[0] == 'root' else np.full(len(ids), 2)
        model = PresetModel('.', labeller)
        model.set_root(int((split == 0).sum()))
        # The E1 zero-mass guard already stops this node (exposure == the E2 parent denominator);
        # force nonzero E1 masses so the E2 call path itself is exercised (defence in depth).
        fake = (np.array([10, 20]), np.ones((2, 1)), np.array([10, 20]), np.zeros((2, 1)))
        with patch.object(trans, 'get_class_wise_stat', return_value=fake):
            _, decisions = run_partition(model, X, y, groups, split, months, proposal=proposal,
                                         threshold=Fraction(0), e2_score=fourclass.crisis_f1_exact_or_none)
        d = decisions[0]
        self.assertEqual(d['outcome'], 'rejected_undefined_parent_metric')
        self.assertIsNone(d['gain'])
        self.assertIsNone(d['parent_macro_f1'])
        self.assertIsNone(d['scores']['parent_parent'])
        self.assertIn('metric_undefined', d)

    def test_undefined_e2_parent_metric_cannot_accept(self):
        y = np.zeros(10, int)
        out = opt.select_macro_children(y[:5], y[5:], y[:5], y[5:], y[:5] + 2, y[5:], min_improvement=0,
                                        score=fourclass.crisis_f1_exact_or_none)
        self.assertFalse(out[0])
        self.assertIsNone(out[3])
        self.assertEqual(fourclass.crisis_f1_exact_or_none([2, 0], [0, 0]), 0)    # defined zero stays zero

    def test_real_scenario_root_splits_original_keys_and_weights_children(self):
        from app import main_model_GF as mgf
        T = '2020-10'
        roles, records = {}, {}
        for strategy in ('A', 'B'):
            frame = self.ctx.stage1_input(m(T), 1, strategy)
            with tempfile.TemporaryDirectory() as t, patch.object(trans, 'CONTIGUITY', False), \
                    patch.object(trans, 'generate_count_grid', return_value=(None, 0, 1)), \
                    patch.dict(plan.G_CONFIGS, SMALL_G), \
                    patch.dict(plan.FIT_SUPPORT, {'rows': 20, 'areas': 5, 'dates': 3, 'classes': 2}), \
                    patch.dict(plan.STAGE1_VAL_SUPPORT, {'rows': 5, 'areas': 3, 'dates': 2}), \
                    patch.object(mgf, 'MAX_DEPTH', 3), redirect_stdout(StringIO()):
                here = os.getcwd()
                os.chdir(t)
                try:
                    rec = mgf.scenario_root(frame, 'r50', 42, 4, T, Path(t), Path(t) / 'ck', None, self.ctx.features)
                    roles[strategy] = pd.read_csv('fold_membership.csv.gz')
                    diag = pd.read_csv(Path(t) / rec['candidates'][0] / 'fit_diagnostic_predictions.csv.gz')
                    self.assertEqual(len(diag), rec['original_keys']['fitting'])        # one row per original F key
                    self.assertFalse(diag.duplicated(['area', 'target_month']).any())
                    evidence = pd.read_csv(Path(t) / rec['candidates'][0] / 'assignment_evidence.csv')
                    self.assertTrue((evidence['fitting_variant_rows'] ==
                                     (3 if strategy == 'B' else 1) * evidence['fitting_rows']).all())
                    records[strategy] = (rec, json.loads((Path(t) / rec['candidates'][0] / 'candidate.json').read_text()))
                finally:
                    os.chdir(here)
        pd.testing.assert_frame_equal(roles['A'], roles['B'])                    # same label-blind key roles
        a, b = records['A'][0], records['B'][0]
        self.assertEqual(a['root_support'], b['root_support'])                   # original keys only
        self.assertEqual(b['fitting_rows_with_variants'], 3 * a['fitting_rows_with_variants'])
        self.assertNotIn('sample_weight', a['root_fit'])
        self.assertEqual(b['root_fit']['sample_weight']['n'], b['fitting_rows_with_variants'])
        self.assertEqual(b['root_fit']['fit_support']['rows'], b['original_keys']['fitting'])
        keys = len(self.ctx.label_pool(m(T) - 4, self.ctx.ledger.hidden(m(T) - 4, 1)))
        self.assertEqual(sum(b['original_keys'][r] for r in ('fitting', 'search_S', 'confirmation_C')), keys)
        cand = records['B'][1]
        self.assertIn('confirmation', cand)                                       # C scored after freeze only
        self.assertEqual(cand['fit_diagnostic']['support']['rows'], b['original_keys']['fitting'])
        self.assertEqual(cand['fit_diagnostic']['frozen_digest_before_scoring'],
                         cand['confirmation']['frozen_digest_before_scoring'])
        for block in (cand['scores']['partitioned_crisis'], cand['scores']['validation']['final']['crisis'],
                      cand['confirmation']['root']['crisis'], cand['fit_diagnostic']['final']['crisis']):
            self.assertIn('f1_na_reason', block)                                  # nullable reporting
        undefined = fourclass.nullable_crisis_summary([0, 0], [0, 0])
        self.assertEqual((undefined['f1'], undefined['f1_na_reason']), (None, '2TP+FP+FN=0'))
        self.assertGreater(cand['fits']['child_fits'], 0, 'fixture must fit children')
        for entry in cand['fits']['fit_log'][1:]:
            if entry.get('kind') == 'continuation':
                self.assertEqual(entry['parent_structure_sha256'], b['root_fit']['structure_sha256'])
                self.assertEqual(entry['sample_weight']['n'], 3 * entry['fit_support']['rows'])


class ScenarioStage2(unittest.TestCase):
    """Interruption task G4: matched E3 crisis E4 weights, legitimate NA and distinct routes."""

    def row(self, name, f, b, status='scored', reason='', strategy='A', target='2018-10', last=None):
        return {'name': name, 'strategy': strategy, 'horizon': 4, 'target_month': target, 'scenario_k': 0,
                'status': status, 'crisis_f1': f, 'crisis_f1_base': b, 'na_reason': reason, 'n_terminal': 2,
                'evidence_last_month': last or target, 'correspondence_sha256': 'x', 'source': 'test'}

    def test_crisis_weights_na_eligibility_and_corrupt_rows(self):
        led = pd.DataFrame([self.row('a', .6, .5), self.row('b', .4, .5),
                            self.row('c', np.nan, np.nan, 'e3_undefined', 'e3_crisis_f1_undefined:root'),
                            self.row('d', np.nan, np.nan, 'no_e3_target_labels', 'no_e3_target_labels')])
        w = stage2.crisis_plan_weights(led)['weight'].to_numpy()
        self.assertAlmostEqual(w[0], np.log(.6 / .4) - np.log(.5 / .5))
        self.assertEqual(w[1], 0.0)
        self.assertTrue(np.isnan(w[2]) and np.isnan(w[3]))
        for bad in (self.row('e', np.nan, .5), self.row('f', np.nan, np.nan, 'e3_undefined', ''),
                    self.row('g', .5, .5, 'e3_undefined', 'x'), self.row('h', .6, .5, 'scored', 'leftover')):
            with self.assertRaises(ValueError):
                stage2.crisis_plan_weights(pd.DataFrame([bad]))

    def stage(self, tmp, status='completed', truth=(2, 2, 0, 0), part=(2, 0, 0, 0), pool=(0, 0, 0, 0),
              counts=None):
        entry = {'candidate': 'scenA_h4_2018-10_G1_k0_r80_s42_L1_gt0', 'root': 'scenA_h4_2018-10_G1_k0_r80_s42',
                 'strategy': 'A', 'horizon': 4, 'target_month': '2018-10', 'scenario_k': 0}
        root_dir = tmp / 'roots' / entry['root']
        root_dir.mkdir(parents=True)
        (root_dir / 'root.json').write_text(json.dumps({'status': status, 'candidates': [entry['candidate']],
                                                         'root': entry['root'], 'strategy': 'A', 'horizon': 4,
                                                         'target_month': '2018-10', 'scenario_k': 0}))
        (root_dir / 'completion.json').write_text(json.dumps({'status': status}))
        pd.DataFrame({'area': [1, 2], 'target_month': ['2018-02', '2018-06'], 'role': ['fitting', 'validation'],
                      'class_code': [0, 2]}).to_csv(root_dir / 'fold_membership.csv.gz', index=False)
        if status == 'completed':
            cand = tmp / 'candidates' / entry['candidate']
            cand.mkdir(parents=True)
            truth, part, pool = map(np.array, (truth, part, pool))
            scores = {'partitioned_crisis': fourclass.crisis_counts(truth, part),
                      'pooled_crisis': fourclass.crisis_counts(truth, pool)}
            if counts:
                scores['partitioned_crisis'] = {**scores['partitioned_crisis'], **counts}
            (cand / 'candidate.json').write_text(json.dumps({'candidate': entry['candidate'], 'scores': scores,
                                                             'partition': {'n_terminal': 3}}))
            pd.DataFrame({'y_true_code': truth, 'y_pred_partitioned_code': part,
                          'y_pred_pooled_code': pool}).to_csv(cand / 'target_predictions.csv', index=False)
            pd.DataFrame({'FEWSNET_admin_code': [1], 'partition_id': ['0']}).to_csv(
                cand / 'correspondence_table.csv', index=False)
        return entry

    def test_candidate_rows_from_saved_matched_rows(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t) / 'ok'
            r = stage2.scenario_candidate_row(tmp, self.stage(tmp))
            self.assertEqual(r['status'], 'scored')
            self.assertAlmostEqual(r['crisis_f1'], 2 / 3)
            self.assertEqual(r['crisis_f1_base'], 0.0)                              # defined zero, not NA
            self.assertEqual(r['evidence_last_month'], '2018-10')
            tmp = Path(t) / 'undef'
            r = stage2.scenario_candidate_row(tmp, self.stage(tmp, truth=(0, 0, 0, 0), part=(0, 0, 0, 0)))
            self.assertEqual(r['status'], 'e3_undefined')
            self.assertTrue(np.isnan(r['crisis_f1']))
            for status in ('no_e3_target_labels', 'root_insufficient_support'):
                tmp = Path(t) / status
                r = stage2.scenario_candidate_row(tmp, self.stage(tmp, status=status))
                self.assertEqual((r['status'], r['na_reason']), (status, status))
            tmp = Path(t) / 'forged'
            with self.assertRaises(RuntimeError):                                    # rows vs record disagree
                stage2.scenario_candidate_row(tmp, self.stage(tmp, counts={'tp': 2}))
            tmp = Path(t) / 'missing'
            entry = self.stage(tmp)
            (tmp / 'roots' / entry['root'] / 'completion.json').unlink()
            with self.assertRaises(RuntimeError):                                    # missing never becomes NA
                stage2.scenario_candidate_row(tmp, entry)
            tmp = Path(t) / 'substituted'
            entry = self.stage(tmp)
            (tmp / 'roots' / entry['root'] / 'root.json').write_text(json.dumps(
                {'status': 'completed', 'candidates': [entry['candidate']], 'root': entry['root'], 'strategy': 'B',
                 'horizon': 4, 'target_month': '2018-10', 'scenario_k': 0}))
            with self.assertRaises(RuntimeError):                                    # another root's evidence
                stage2.scenario_candidate_row(tmp, entry)
            tmp = Path(t) / 'nofile'
            entry = self.stage(tmp)
            (tmp / 'candidates' / entry['candidate'] / 'target_predictions.csv').unlink()
            with self.assertRaises(FileNotFoundError):
                stage2.scenario_candidate_row(tmp, entry)

    def test_routes_and_acceptance_under_the_crisis_metric(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            frame, paths, coords = ConsensusAndBoundaries().candidates(tmp)
            scored = pd.DataFrame([{**self.row(n, .6, .5), 'horizon': h, 'target_month': tm,
                                    'correspondence_sha256': frame.loc[i, 'correspondence_sha256']}
                                   for i, (n, h, tm) in enumerate(zip(frame['name'], frame['horizon'],
                                                                      frame['target_month']))])
            na = pd.DataFrame([self.row('zz_na', np.nan, np.nan, 'e3_undefined', 'e3_crisis_f1_undefined:root')])
            with redirect_stdout(StringIO()):
                none = stage2.build_consensus(tmp / 'none', scored.iloc[:0], {}, coords, 'x', metric='crisis')
                noscore = stage2.build_consensus(tmp / 'noscore', na, {}, coords, 'x', metric='crisis')
                zero = stage2.build_consensus(tmp / 'zero', scored.assign(crisis_f1=.4), paths, coords, 'x',
                                              metric='crisis')
                mixed = pd.concat([scored, na], ignore_index=True)
                learned = stage2.build_consensus(tmp / 'map', mixed, paths, coords, 'x', metric='crisis')
            self.assertEqual([r['route'] for r in (none, noscore, zero, learned)],
                             ['no_prior_candidates', 'no_scorable_evidence', 'null_consensus', 'learned_map'])
            self.assertEqual(learned['eligible_candidates'], 3)
            self.assertEqual(learned['ineligible_reasons'], {'e3_crisis_f1_undefined:root': 1})
            saved = pd.read_csv(tmp / 'map' / 'plan_weights.csv')
            self.assertEqual(len(saved), 4)                                          # full ledger kept
            self.assertTrue(np.isnan(saved.loc[saved['name'] == 'zz_na', 'weight']).all())
            for out, cands in (('noscore', na), ('map', mixed)):
                stage2.accept_consensus(tmp / out, cands, 'crisis')
            with self.assertRaises(RuntimeError):
                stage2.accept_consensus(tmp / 'map', mixed, 'macro')                 # metric is part of identity
            self.assertEqual(stage2.build_consensus(tmp / 'map', mixed, paths, coords, 'x',
                                                    metric='crisis')['pool_identity'], learned['pool_identity'])

    def test_common_origin_legal_pool_uses_full_evidence_span(self):
        _, ctx, _ = scenario_fixture()
        led = ctx.ledger
        rows = pd.DataFrame([self.row('ok', .6, .5, target='2018-10'),
                             self.row('late_evidence', .6, .5, target='2018-10', last='2019-02'),
                             self.row('hidden_target', .6, .5, target='2019-02'),
                             self.row('other_strategy', .6, .5, strategy='B', target='2018-10'),
                             self.row('after_origin', .6, .5, target='2019-06')])
        O = m('2019-06')                                                             # hidden(O, 2) = Feb, Jun 2019
        self.assertEqual(led.hidden(O, 2), {m('2019-02'), m('2019-06')})
        dev = stage2.scenario_map_pool(rows, 'A', O, led, strict=True)
        self.assertEqual(list(dev['name']), ['ok'])
        final = pd.DataFrame([self.row('oct', .6, .5, target='2020-10')])
        self.assertEqual(len(stage2.scenario_map_pool(final, 'A', m('2020-12'), led, strict=False, k_max=0)), 1)
        self.assertEqual(len(stage2.scenario_map_pool(final, 'A', m('2020-10'), led, strict=False, k_max=0)), 0)


class ScenarioDevelopment(unittest.TestCase):
    """Interruption task D4/G4: 72 complete development folds and the A/B stop rule."""

    @staticmethod
    def counts(tp, fp, fn, tn=10):
        return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

    def folds(self, spec):
        """spec[(strategy, h, k)] = (model, model_matched, persistence) per-fold counts."""
        out = []
        for f in rexp.scenario_dev_plan():
            model, matched, pers = spec[(f['strategy'], f['horizon'], f['scenario_k'])]
            out.append({**f, 'model': model, 'model_matched': matched, 'persistence': pers})
        return out

    def base(self):
        c = self.counts
        return {(s, h, k): (c(5, 5, 5), c(5, 5, 5), c(5, 5, 5)) for s in 'AB' for h in (4, 8) for k in (0, 1, 2)}

    def test_plan_is_the_frozen_72(self):
        plan_ = rexp.scenario_dev_plan()
        self.assertEqual(len(plan_), 72)
        self.assertEqual(len({(f['strategy'], f['horizon'], f['scenario_k'], f['target_month']) for f in plan_}), 72)
        self.assertEqual({f['target_month'] for f in plan_}, set(plan.DEV_TARGETS))

    def test_selection_rule_ties_stop_and_undefined(self):
        c = self.counts
        tie = rexp.ab_select(self.folds(self.base()))
        self.assertEqual((tie['4']['winner'], tie['4']['tie']), ('A', True))       # exact tie favours A
        spec = self.base()
        spec[('B', 4, 1)] = (c(8, 2, 2), c(8, 2, 2), c(5, 5, 5))                   # B better under interruption
        self.assertEqual(rexp.ab_select(self.folds(spec))['4']['winner'], 'B')
        spec[('B', 4, 0)] = (c(5, 5, 5), c(1, 9, 9), c(5, 5, 5))                   # B fails normal parity
        out = rexp.ab_select(self.folds(spec))['4']
        self.assertEqual(out['winner'], 'A')
        self.assertEqual(out['strategies']['B']['unmet'], ['normal_parity_below_-0.02'])
        spec[('A', 4, 0)] = (c(5, 5, 5), c(5, 5, 5), c(0, 0, 0))                   # persistence F1 undefined
        out = rexp.ab_select(self.folds(spec))['4']
        self.assertIsNone(out['winner'])                                           # no automatic winner
        self.assertEqual(out['strategies']['A']['unmet'], ['normal_parity_undefined'])
        boundary = self.base()                                                     # parity exactly -0.02 qualifies
        boundary[('A', 8, 0)] = (c(5, 5, 5), c(48, 52, 52, 0), c(50, 50, 50, 0))   # 96/200 vs 100/200
        sel = rexp.ab_select(self.folds(boundary))['8']['strategies']['A']
        self.assertEqual(Fraction(sel['normal_parity']), Fraction(-2, 100))
        self.assertTrue(sel['qualifies'])
        with self.assertRaises(RuntimeError):
            rexp.ab_select(self.folds(self.base())[:-1])                           # incomplete folds
        dup = self.folds(self.base())
        dup[-1] = dict(dup[-2])                                                    # a duplicate replaces a month
        with self.assertRaises(RuntimeError):
            rexp.ab_select(dup)

    def test_complete_fold_keys_scores_and_persistence(self):
        _, ctx, _ = scenario_fixture()
        cluster_of = {a: (0 if a < 20 else 1) for a in range(40)}
        with tempfile.TemporaryDirectory() as t, patch.dict(plan.G_CONFIGS, SMALL_G), \
                patch.dict(plan.FIT_SUPPORT, {'rows': 20, 'areas': 5, 'dates': 3, 'classes': 2}), \
                patch.dict(plan.STAGE3_GATE_SUPPORT, {'rows': 10, 'areas': 5, 'dates': 2, 'local_fit_dates': 2}):
            res = rexp.scen_dev_fold(ctx, 'B', 4, 2, '2020-10', s3.GlobalStore(Path(t)), {'route': 'learned_map'},
                                     cluster_of)
            none = rexp.scen_dev_fold(ctx, 'A', 4, 0, '2020-10', s3.GlobalStore(Path(t)),
                                      {'route': 'no_prior_candidates'}, None)
        preds = res['system']['predictions']
        self.assertEqual(len(preds), 41)
        self.assertTrue((preds['scenario_k'] == 2).all())
        sc = rexp.scen_fold_scores(preds)
        self.assertEqual((sc['labelled'], sc['matched']), (40, 40))
        lab = preds[preds['y_true_code'].notna()]
        self.assertEqual(sc['model'], fourclass.crisis_counts(lab['y_true_code'].astype(int), lab['y_pred_code']))
        self.assertTrue((none['system']['predictions']['route'] == 'no_prior_candidates_pooled').all())


class ScenarioReporting(unittest.TestCase):
    """Interruption task G3/D5: crisis paired country-block uncertainty, studies and country tables."""

    def rows(self, spec):
        """spec: list of (country, truth, model, comparator)."""
        frame = pd.DataFrame(spec, columns=['country', 'truth_code', 'model', 'comp'])
        return frame.assign(area=np.arange(len(frame)), target_month='2021-10')

    def test_bootstrap_point_ci_shared_schedule_and_na(self):
        spec = []
        for c, (tp, fp, fn) in {'X': (3, 1, 1), 'Y': (2, 2, 1), 'Z': (4, 0, 2)}.items():
            spec += [(c, 2, 2, 2)] * tp + [(c, 0, 2, 0)] * fp + [(c, 2, 0, 0)] * fn + [(c, 0, 0, 0)] * 3
        frame = self.rows(spec)
        out = report.crisis_paired_bootstrap(frame, 'model', 'comp')
        self.assertAlmostEqual(out['model_f1'], float(fourclass.crisis_f1_exact(frame['truth_code'], frame['model'])))
        self.assertEqual((out['valid_draws'], out['undefined_draws'], out['ci_reason']), (2000, 0, ''))
        self.assertLessEqual(out['ci95_low'], out['ci95_high'])
        self.assertEqual(report.crisis_paired_bootstrap(frame, 'model', 'comp'), out)    # fixed seed, no state
        sparse = self.rows([('X', 2, 2, 0), ('X', 0, 0, 0), ('Y', 0, 0, 0), ('Z', 0, 0, 0)])
        na = report.crisis_paired_bootstrap(sparse, 'model', 'comp')
        self.assertEqual(na['valid_draws'] + na['undefined_draws'], 2000)               # never redrawn
        self.assertGreater(na['undefined_draws'], 0)
        self.assertEqual((na['ci95_low'], na['ci_reason']), (None, 'undefined bootstrap draws'))
        self.assertIsNotNone(na['delta'])                                               # defined point kept
        one = report.crisis_paired_bootstrap(self.rows([('X', 2, 2, 0), ('X', 0, 0, 2)]), 'model', 'comp')
        self.assertEqual(one['ci_reason'], 'fewer than two country blocks')
        empty = report.crisis_paired_bootstrap(frame.iloc[:0], 'model', 'comp')     # empty cohort (e.g. Study2)
        self.assertEqual((empty['n'], empty['ci_reason'], empty['valid_draws']), (0, 'no eligible rows', 0))
        quiet = report.crisis_paired_bootstrap(self.rows([('X', 0, 0, 0), ('Y', 0, 0, 0)]), 'model', 'comp')
        self.assertEqual((quiet['model_f1'], quiet['ci_reason']), (None, 'point F1 undefined'))
        with self.assertRaises(ValueError):
            report.crisis_paired_bootstrap(frame.assign(comp=np.nan), 'model', 'comp')

    def test_studies_onset_and_country_table(self):
        frame = self.rows([('X', 2, 2, 0), ('X', 0, 0, 0), ('X', np.nan, 2, 0), ('Y', 2, 0, 0), ('Y', 3, 3, 3)])
        origin = [0, 1, 0, np.nan, 2]
        st = report.study_rows(frame, origin)
        self.assertEqual(len(st['study1']), 4)                                          # unlabelled target out
        self.assertEqual(len(st['study2']), 2)                                          # origin non-crisis only
        self.assertEqual((st['study2_excluded_missing_origin'], st['study2_excluded_origin_crisis']), (1, 1))
        self.assertEqual(report.onset_recall(st['study2'], 'model'), {'onsets': 1, 'recall': 1.0, 'reason': ''})
        self.assertEqual(report.onset_recall(st['study2'], 'comp')['recall'], 0.0)       # persistence-like zero
        table = report.country_table(frame.assign(comp=[0, 0, 0, np.nan, 3]), 'model', 'comp').set_index('country')
        self.assertEqual(table.loc['X', 'cohort_keys'], 3)                       # unlabelled key still in cohort
        self.assertEqual(table.loc['X', 'labelled_keys'], 2)
        self.assertAlmostEqual(table.loc['Y', 'model_f1_labelled'], 2 / 3)        # standalone, incl. unmatched key
        self.assertEqual(table.loc['Y', 'comparator_matched_keys'], 1)
        wide = report.country_table(frame, 'model', {'persistence': 'comp', 'expert': 'missing_col'},
                                    origin_truth=origin).set_index('country')
        self.assertEqual(wide.loc['X', 'study2_eligible_keys'], 2)                  # risk-set coverage per country
        self.assertEqual(wide.loc['Y', 'study2_excluded_missing_origin'], 1)
        self.assertIn('expert: comparator unavailable', wide.loc['X', 'na_reasons'])
        self.assertEqual(table.loc['Y', 'crisis_events'], 2)


class ScenarioFinalPath(unittest.TestCase):
    """Interruption task D2/G2: historical calendar, actual availability contract, gate intensity."""

    def test_historical_calendar_requires_origin_and_both_cycles_after_freeze(self):
        _, ctx, _ = scenario_fixture(last_year=2022)
        h4 = rexp.historical_targets(ctx, 4)
        self.assertEqual(h4['targets'][0], '2021-10')                             # O 2021-06, hidden Feb/Jun 2021
        self.assertEqual([e['target_month'] for e in h4['excluded']], ['2021-02', '2021-06'])
        h8 = rexp.historical_targets(ctx, 8)
        self.assertEqual(h8['targets'][0], '2022-02')
        self.assertTrue(all(e['reason'] for e in h8['excluded']))

    def test_actual_availability_contract_refuses_unresolved_metadata(self):
        good = pd.DataFrame([('AAA', 'CS', '2025-06', 1, 'verified_vintage', 'doc'),
                             ('BBB', 'CS', '2025-06', 2, 'reconstructed', 'doc')], columns=rexp.ACTUAL_COLUMNS)
        self.assertEqual(rexp.actual_gate_intensity(good, '2025-06', ['AAA', 'BBB']), {'AAA': 1, 'BBB': 2})
        conflict = pd.concat([good, good.iloc[:1].assign(missed_cycles=2)], ignore_index=True)
        for table, countries in ((None, ['AAA']), (good.iloc[:1], ['AAA', 'BBB']), (conflict, ['AAA', 'BBB']),
                                 (good.assign(evidence='synthetic'), ['AAA', 'BBB']),
                                 (good.assign(missed_cycles=[1.5, 2]), ['AAA', 'BBB']),
                                 (good.drop(columns='source'), ['AAA'])):
            with self.assertRaises(SystemExit):
                rexp.actual_gate_intensity(table, '2025-06', countries)

    def test_mixed_union_keeps_global_exclusions_and_per_row_intensity(self):
        _, ctx, _ = scenario_fixture()
        mixed = ctx.union({'AAA': frozenset({m('2019-02')})}, frozenset({m('2020-02')}))
        self.assertEqual(mixed['BBB'], {m('2020-02')})                            # global stays global
        self.assertEqual(mixed['AAA'], {m('2019-02'), m('2020-02')})
        seen = ctx.visible(m('2020-10'), mixed)
        self.assertFalse(((seen['country'] == 'BBB') & (seen['month'] == m('2020-02'))).any())
        self.assertTrue(((seen['country'] == 'BBB') & (seen['month'] == m('2019-02'))).any())
        view = ctx.prediction_view([1, 30, 40], [m('2020-10')] * 3, {'AAA': 1, 'BBB': 2})
        self.assertEqual(list(view['scenario_k']), [1, 2, 0])                     # per row; no-history area 0
        self.assertTrue(np.isnan(view.loc[0, 'hist_phase_o00']))                  # AAA: June hidden at k=1
        self.assertEqual(view.loc[0, 'hist_latest_observed_age'], 4)
        self.assertEqual(view.loc[1, 'hist_latest_observed_age'], 8)              # BBB: two cycles hidden

    def test_expert_matching_requires_horizon_origin_validity_and_release(self):
        keys = pd.DataFrame({'area': [1, 1, 1, 2, 3], 'target_month': ['2021-10'] * 5,
                             'origin_month': ['2021-06'] * 5, 'horizon': [4, 8, 4, 4, 4]})
        row = dict(area=1, issue_month='2021-06', product='P', horizon=4, validity_start='2021-10',
                   validity_end='2022-01', class_code=2, release_date='2021-06-30', evidence='reconstructed',
                   source='doc')
        experts = pd.DataFrame([row, {**row, 'area': 2, 'release_date': '2021-07-05'},
                                {**row, 'area': 3, 'validity_start': '2021-06', 'validity_end': '2021-09'}])
        out = report.keyed_expert(keys, experts)
        self.assertEqual(list(out['expert_class_code'].fillna(-1)), [2, -1, 2, -1, -1])
        self.assertEqual(list(out['expert_reason']), ['', 'no_expert_for_horizon', '',
                                                      'expert_released_after_origin_cutoff', 'validity_excludes_target'])
        self.assertTrue((report.keyed_expert(keys, None)['expert_reason'] == 'no_documented_expert_table').all())
        for bad in (experts.assign(evidence='synthetic'), experts.assign(class_code=[2, 4, 1]),
                    experts.assign(class_code=[2.5, 1, 1]), experts.drop(columns='horizon')):
            with self.assertRaises(ValueError):
                report.keyed_expert(keys, bad)
        self.assertEqual(report.keyed_expert(keys, experts.assign(evidence='synthetic'), real=False)
                         ['expert_class_code'].notna().sum(), 2)

    def test_real_fold_with_per_country_gate_intensity(self):
        _, ctx, _ = scenario_fixture()
        gate_k = {'AAA': 1, 'BBB': 2}
        with tempfile.TemporaryDirectory() as t, patch.dict(plan.G_CONFIGS, SMALL_G), \
                patch.dict(plan.FIT_SUPPORT, {'rows': 20, 'areas': 5, 'dates': 3, 'classes': 2}), \
                patch.dict(plan.STAGE3_GATE_SUPPORT, {'rows': 10, 'areas': 5, 'dates': 2, 'local_fit_dates': 2}), \
                patch.object(plan, 'GATE_DATES', 2):
            store = s3.GlobalStore(Path(t))
            res = rexp.scen_dev_fold(ctx, 'B', 4, 0, '2020-10', store, {'route': 'learned_map'},
                                     {a: (0 if a < 20 else 1) for a in range(40)}, gate_k=gate_k)
            self.assertEqual(len(res['system']['predictions']), 41)
            internal = [r for r in (rec for _, rec in store.memo.values()) if r['intensity_k'] == gate_k]
            self.assertTrue(internal)                                              # internal globals at dict k
            masked = internal[0]['masked_months']
            self.assertEqual(len(masked['AAA']) + 1, len(masked['BBB']))          # 1 vs 2 cycles per country
            outer = [r for r in (rec for _, rec in store.memo.values()) if r['intensity_k'] == 0]
            self.assertEqual(outer[0]['masked_months'], [])                        # outer forecast: real ledger

    def test_outer_and_internal_gate_intensity_are_separate(self):
        _, ctx, _ = scenario_fixture()
        actual = s3.ScenarioPanel(ctx, 0, 'A', gate_k=2)
        O, V = m('2020-06'), m('2019-10')
        outer, _ = actual.fit_pool(O)
        normal, _ = s3.ScenarioPanel(ctx, 0, 'A').fit_pool(O)
        np.testing.assert_array_equal(outer.X, normal.X)                          # outer forecast: real ledger only
        inner, _ = actual.fit_pool(V, internal=True)
        replay, _ = s3.ScenarioPanel(ctx, 2, 'A').fit_pool(V)
        np.testing.assert_array_equal(inner.X, replay.X)                          # internal: verified intensity
        with tempfile.TemporaryDirectory() as t, patch.dict(plan.G_CONFIGS, SMALL_G):
            store = s3.GlobalStore(Path(t))
            _, rec = store.get(actual, V, 'G1', internal=True)
            self.assertEqual((rec['scenario_k'], rec['intensity_k']), (0, 2))
            self.assertEqual(rec['masked_months'], list(ff.month_label(sorted(ctx.ledger.hidden(V, 2)))))


def synthetic_scenario_run(root: Path, crisis: bool = True) -> dict:
    """A synthetic PREPARED interruption run on disk (hashed outputs, identity record, pinned panel,
    release ledger, alignment, the complete 648-root Stage 1 evidence set). Labels follow a
    learnable pattern; ``crisis=False`` keeps every label non-crisis (undefined crisis F1)."""
    import pickle as _pickle
    from scripts import run_stage1 as s1
    av, ctx, schema = scenario_fixture(last_year=2022)
    obs = ctx.obs[['area', 'month', 'country']].copy()
    obs['class_code'] = ((obs['area'] // 10 + obs['month'] // 4) % 4) if crisis else (obs['area'] % 2)
    prepared = root / 'prepared'
    for d in ('manifests', 'ledgers', 'geometry', 'scenario'):
        (prepared / d).mkdir(parents=True)
    months = pd.date_range('2011-01-01', '2023-12-01', freq='MS')
    panel = pd.DataFrame([(a, d) for a in range(41) for d in months], columns=['FEWSNET_admin_code', 'date'])
    idx = ff.month_index(panel['date'])
    panel['crop'] = (panel['FEWSNET_admin_code'] % 7).astype(float)
    panel['EVI'] = np.sin(idx / 3 + panel['FEWSNET_admin_code'])
    panel['month'] = panel['date'].dt.month
    panel_path = root / 'sources' / 'panel.csv'
    panel_path.parent.mkdir()
    panel.assign(date=panel['date'].dt.strftime('%Y-%m')).to_csv(panel_path, index=False)
    ext_months = pd.date_range('2023-12-01', '2025-12-01', freq='MS')
    ext = pd.DataFrame([(a, d) for a in range(41) for d in ext_months], columns=['FEWSNET_admin_code', 'date'])
    eidx = ff.month_index(ext['date'])
    ext['crop'] = (ext['FEWSNET_admin_code'] % 7).astype(float)
    ext['EVI'] = np.sin(eidx / 3 + ext['FEWSNET_admin_code'])
    ext_path = root / 'sources' / 'extension.csv'
    ext.assign(date=ext['date'].dt.strftime('%Y-%m')).to_csv(ext_path, index=False)
    (root / 'sources' / 'extension.json').write_text(json.dumps(
        {'path': str(ext_path), 'sha256': rid.file_sha256(ext_path), 'first_month': '2023-12',
         'last_month': '2025-12', 'overlap_months': ['2023-12'], 'source': 'unit-test fixture'}))
    (prepared / 'manifests' / 'sources.json').write_text(json.dumps(
        {'sources': {'panel': {'path': str(panel_path), 'sha256': rid.file_sha256(panel_path)}}}))
    ctx.ledger.frame[list(av.LEDGER_COLUMNS)].to_csv(prepared / 'manifests' / 'release_ledger.csv', index=False)
    (prepared / 'manifests' / 'alignment.json').write_text(json.dumps(ctx.alignment))
    obs.to_csv(prepared / 'ledgers' / 'observations.csv', index=False)
    coords = pd.DataFrame({'FEWSNET_admin_code': range(41), 'lat': np.arange(41) // 7 * 1.0,
                           'lon': np.arange(41) % 7 * 1.0})
    coords.to_csv(prepared / 'geometry' / 'FEWSNET_admin_code_lat_lon.csv', index=False)
    (prepared / 'geometry' / 'polygon_contiguity_info.pkl').write_bytes(_pickle.dumps(None))
    for f in ('manifests/runtime.json', 'manifests/preflight.json', 'manifests/features.json',
              'manifests/geometry.json', 'ledgers/baselines.csv', 'ledgers/dev_baselines.csv',
              'snapshot_h4.parquet', 'snapshot_h8.parquet', 'snapshot_h12.parquet'):
        (prepared / f).write_text('{}')
    (prepared / 'scenario' / 'features.json').write_text(json.dumps({'ordered_features': ctx.features}))
    entries = [{**r, 'input': s1.scenario_input_name(r)} for r in plan.scenario_stage1_schedule()]
    for name in {e['input'] for e in entries}:
        (prepared / 'scenario' / name).write_text('synthetic input placeholder')
    (prepared / 'manifests' / 'schedule.json').write_text(json.dumps({'stage1_scenario_roots': entries}))
    outputs = rid.output_hashes(prepared)
    (prepared / 'manifests' / 'outputs.json').write_text(json.dumps(outputs))
    (prepared / 'manifests' / 'identity.json').write_text(json.dumps(
        {'stage': 'prepare', 'code': rid.code_identity(), 'runtime': rid.runtime_identity(),
         'outputs_sha256': rid.file_sha256(prepared / 'manifests' / 'outputs.json')}))
    stage = root / 'stage1_scenario'
    scored_targets = ('2018-02', '2018-06', '2018-10', '2019-02')
    for e in entries:
        rdir = stage / 'roots' / e['root']
        rdir.mkdir(parents=True)
        done = e['target_month'] in scored_targets and e['ratio'] == 'r80' and e['split_seed'] == 42
        status = 'completed' if done else 'no_e3_target_labels'
        (rdir / 'root.json').write_text(json.dumps({'root': e['root'], 'status': status, 'candidates': [e['candidate']],
                                                    **{f: e[f] for f in ('strategy', 'horizon', 'target_month',
                                                                         'scenario_k', 'ratio', 'split_seed')}}))
        (rdir / 'command.json').write_text('{}')
        (rdir / 'run.log').write_text('')
        files = [f'roots/{e["root"]}/{f}' for f in ('root.json', 'command.json', 'run.log')]
        if done:
            pd.DataFrame({'area': [1], 'target_month': [e['target_month']], 'role': ['heldout_target'],
                          'class_code': [2]}).to_csv(rdir / 'fold_membership.csv.gz', index=False)
            files.append(f'roots/{e["root"]}/fold_membership.csv.gz')
            cdir = stage / 'candidates' / e['candidate']
            cdir.mkdir(parents=True)
            truth, part, pool = np.array([2, 2, 2, 0, 0]), np.array([2, 2, 0, 0, 2]), np.array([2, 0, 0, 2, 2])
            record = {'candidate': e['candidate'], 'partition': {'n_terminal': 2},
                      'scores': {'partitioned_crisis': fourclass.crisis_counts(truth, part),
                                 'pooled_crisis': fourclass.crisis_counts(truth, pool)}}
            (cdir / 'candidate.json').write_text(json.dumps(record))
            pd.DataFrame({'y_true_code': truth, 'y_pred_partitioned_code': part,
                          'y_pred_pooled_code': pool}).to_csv(cdir / 'target_predictions.csv', index=False)
            split = coords['lat'] < 3 if e['horizon'] == 4 else coords['lon'] < 3
            pd.DataFrame({'FEWSNET_admin_code': range(41), 'partition_id': np.where(split, '0', '1')}).to_csv(
                cdir / 'correspondence_table.csv', index=False)
            for f in s1.candidate_files(record, True):
                if not (cdir / f).exists():
                    (cdir / f).write_text('synthetic evidence placeholder')
            files += [f'candidates/{e["candidate"]}/{f}' for f in s1.candidate_files(record, True)]
        (rdir / 'completion.json').write_text(json.dumps(
            {'root': e['root'], 'status': status, 'candidates': [e['candidate']],
             'prepared': rid.file_sha256(prepared / 'manifests' / 'outputs.json'),
             'g_selection': 'fixed design capacities (G4 of 10-02)', 'code': rid.code_identity(),
             'runtime': rid.runtime_identity(), 'outputs': {f: rid.file_sha256(stage / f) for f in files}}))
    availability = pd.DataFrame([(c, 'CS', o, kk, 'reconstructed', 'unit-test fixture')
                                 for o in ('2025-06', '2025-02', '2024-10')
                                 for c, kk in (('AAA', 1), ('BBB', 2))], columns=rexp.ACTUAL_COLUMNS)
    availability.to_csv(root / 'sources' / 'availability.csv', index=False)
    return {'schema': schema, 'extension': root / 'sources' / 'extension.json',
            'availability': root / 'sources' / 'availability.csv'}


class ScenarioDriverSmoke(unittest.TestCase):
    """End-to-end drivers on a synthetic prepared run: scen-develop -> select -> freeze -> historical ->
    actual -> report -> evaluate (positive path) and the no-qualifier path. Only test-scale constants,
    the synthetic schema and real=False for the synthetic ledger are patched; the drivers run as is."""

    def patches(self, schema):
        from functools import partial
        return [patch.dict(plan.G_CONFIGS, SMALL_G),
                patch.dict(plan.FIT_SUPPORT, {'rows': 20, 'areas': 5, 'dates': 3, 'classes': 2}),
                patch.dict(plan.STAGE3_GATE_SUPPORT, {'rows': 10, 'areas': 5, 'dates': 2, 'local_fit_dates': 2}),
                patch.object(plan, 'DEV_TARGETS', ('2020-06', '2020-10')), patch.object(plan, 'GATE_DATES', 3),
                patch.object(rexp, 'scenario_context', partial(rexp.scenario_context, real=False, schema=schema)),
                patch.object(rexp, 'historical_targets', partial(rexp.historical_targets, last='2022-10'))]

    def run_drivers(self, run, info, positive):
        with redirect_stdout(StringIO()):
            rexp.scen_develop(run, 1)
            rexp.scen_select(run)
            rexp.scen_freeze(run)
            rexp.scen_historical(run)
            rexp.scen_actual(run, info['availability'], info['extension'])
            rexp.scen_report(run)

    def test_positive_full_driver_path(self):
        with tempfile.TemporaryDirectory() as t:
            run = Path(t)
            info = synthetic_scenario_run(run, crisis=True)
            ps = self.patches(info['schema'])
            for p_ in ps:
                p_.start()
            try:
                self.run_drivers(run, info, True)
                selection = json.loads((run / 'scenario_development' / 'selection.json').read_text())
                self.assertEqual(selection['folds'], 24)
                frozen = json.loads((run / 'scenario_final' / 'frozen.json').read_text())
                released = sorted(int(h) for h, e in frozen['recipe'].items() if e['released'])
                self.assertTrue(released, selection['decisions'])
                fold = next((run / 'scenario_historical').rglob('fold.json')).parent
                self.assertTrue((fold / 'pooled_predictions.csv.gz').is_file())    # same-input pooled kept
                frec = json.loads((fold / 'fold.json').read_text())
                self.assertIn('availability_inputs_sha256', frec)                   # lawful-input identity
                with redirect_stdout(StringIO()):                                  # actual entry refusals
                    with self.assertRaises(SystemExit):
                        rexp.scen_actual(run, None, info['extension'])
                    with self.assertRaises(SystemExit):
                        rexp.scen_actual(run, info['availability'], None)
                actual = json.loads((run / 'scenario_actual' / 'actual.json').read_text())
                cases = sorted(k for k, v in actual['cases'].items() if v['released'])
                self.assertEqual(cases, sorted(f'h{h}_{t_}' for t_, h in rexp.ACTUAL_CASES if h in released))
                arec = json.loads((run / 'scenario_actual' / f'h{released[0]}' / '2025-10' / 'fold.json').read_text())
                self.assertEqual(arec['gate_k'], {'AAA': 1, 'BBB': 2})              # per-country replay
                self.assertIn('extension_manifest_sha256', arec)
                preds = pd.read_csv(run / 'scenario_actual' / f'h{released[0]}' / '2025-10' / 'predictions.csv.gz')
                self.assertTrue(preds['y_true_code'].isna().all())                 # truth never loaded
                ctx = rexp.scenario_context(run, 4, development_truth=False, extension=info['extension'])
                evi = ctx.prediction_view([5], [m('2025-10')], 0)['EVI'].item()
                self.assertAlmostEqual(evi, np.sin(m('2025-05') / 3 + 5))            # admitted extension, O - 1
                base_ctx = rexp.scenario_context(run, 4, development_truth=False)
                self.assertTrue(np.isnan(base_ctx.prediction_view([5], [m('2025-10')], 0)['EVI'].item()))
                report_ = json.loads((run / 'scenario_report' / 'report.json').read_text())
                s1 = [c for c in report_['comparisons'] if c.get('study') == 'study1']
                self.assertTrue(s1 and all('vs_pooled_same_input' in c for c in s1))
                with self.assertRaises(FileExistsError):                           # historical is run once
                    rexp.scen_historical(run)
                rel = run / 'release'                                              # separate truth release
                rel.mkdir()
                truth = preds.loc[preds['area'] < 40, ['area', 'target_month']]
                truth = truth.assign(class_code=(truth['area'] // 10) % 4)
                truth.to_csv(rel / 'truth.csv', index=False)
                release = {'approved': True, 'approved_by': 'unit-test', 'crosswalk': 'identity (fixture)',
                           'truth_file': 'truth.csv', 'truth_sha256': rid.file_sha256(rel / 'truth.csv'),
                           'frozen_actual': rid.file_sha256(run / 'scenario_actual' / 'actual.json')}
                (rel / 'release.json').write_text(json.dumps({**release, 'frozen_actual': 'stale'}))
                with self.assertRaises(SystemExit):                                # must bind frozen predictions
                    rexp.scen_evaluate(run, rel)
                (rel / 'release.json').write_text(json.dumps(release))
                with redirect_stdout(StringIO()):
                    rexp.scen_evaluate(run, rel)
                ev = json.loads((run / 'scenario_evaluation' / 'evaluation.json').read_text())
                evaluated = sorted({r['case'] for r in ev['results'] if r.get('evaluable')})
                unevaluable = sorted({r['case'] for r in ev['results'] if r.get('evaluable') is False})
                self.assertEqual(evaluated, [c for c in cases if c.endswith('2025-10')])
                self.assertEqual(unevaluable, [c for c in cases if c.endswith('2025-06')])
                for c in unevaluable:                                               # coverage-only countries kept
                    table = pd.read_csv(run / 'scenario_evaluation' / f'country_{c}.csv')
                    self.assertEqual(table['labelled_keys'].sum(), 0)
                    self.assertGreater(table['cohort_keys'].sum(), 0)
            finally:
                for p_ in ps:
                    p_.stop()

    def test_no_qualifier_path_reports_unmet_without_failure(self):
        with tempfile.TemporaryDirectory() as t:
            run = Path(t)
            info = synthetic_scenario_run(run, crisis=False)
            ps = self.patches(info['schema'])
            for p_ in ps:
                p_.start()
            try:
                self.run_drivers(run, info, False)
                selection = json.loads((run / 'scenario_development' / 'selection.json').read_text())
                self.assertTrue(all(d['winner'] is None for d in selection['decisions'].values()))
                hist = json.loads((run / 'scenario_historical' / 'historical.json').read_text())
                self.assertTrue(all(not c['released'] and c['reason'] for c in hist['calendar'].values()))
                actual = json.loads((run / 'scenario_actual' / 'actual.json').read_text())
                self.assertTrue(all(not v['released'] for v in actual['cases'].values()))
                report_ = json.loads((run / 'scenario_report' / 'report.json').read_text())
                self.assertTrue(all(c.get('released') is False for c in report_['comparisons']))
            finally:
                for p_ in ps:
                    p_.stop()


class CommittedCode(unittest.TestCase):
    def test_schema_is_committed_and_identity_matches_git(self):
        import subprocess
        repo = Path(__file__).resolve().parents[2]
        listed = subprocess.run(['git', 'ls-files', 'FEWSNETGeoXGBExperiment/feature-schema.json'],
                                cwd=repo, capture_output=True, text=True).stdout.strip()
        self.assertEqual(listed, 'FEWSNETGeoXGBExperiment/feature-schema.json')
        self.assertEqual(rid.file_sha256(prep.SCHEMA_PATH), prep.APPROVED_SCHEMA_SHA256)
        self.assertEqual(rid.code_identity_at('HEAD')['files'], rid.code_identity()['files'])
        self.assertIn('xgboost', rid.runtime_identity())
        self.assertEqual(prep.PINNED_RUNTIME['xgboost'], '3.0.0')


if __name__ == '__main__':
    unittest.main(verbosity=2)
