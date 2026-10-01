"""Focused contract checks for the GeoXGBoost fork: python -B tests/test_baseline.py.

Inherited fixed-four metric, scan, gate, feature, weight, bootstrap and calendar tests
are kept; the RF/imputer/pseudo-row tests are replaced by tests that drive the new
production paths (native continuation, partition(), Stage 3 gate, consensus routes,
time boundaries) with hand-checkable fixtures.
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


def run_partition(model, X, y, groups, split, months, threshold=Fraction(1, 100), proposal=None, cap=80, **floors):
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
                                  min_depth=1, max_depth=2, contiguity_type='polygon', model_dir=tmp,
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
