"""Focused numerical and real-checkpoint checks: python tests/test_baseline.py."""
from contextlib import redirect_stdout
from io import StringIO
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

from src.partition import partition_opt as opt
from src.partition import transformation as trans
from src.model.model_RF import RFmodel
from src.utils.split import group_aware_train_val_split
from scripts import compare_partitioned_vs_pooled_rf_k40_nc4 as compare
from scripts.step4_similarity_matrix import compute_plan_weights, load_lat_lon
from scripts.run_stage1 import collect_stage1
from scripts.step1_merge_results import MODEL_CONFIG, load_results_table, load_correspondence_tables


class PresetRF(RFmodel):
    """Deterministic predictions with production save/load and branch dispatch."""

    def __init__(self, path, reject):
        super().__init__(path, 2, num_class=2, n_jobs=1)
        self.reject = reject
        self.trained = []

    def train(self, X, y, branch_id='', **kwargs):
        self.trained.append((branch_id, len(X)))
        ids = X[:, 0].astype(int)
        if len(X) == 4:
            labels = np.array([1, 0, 0, 1 if self.reject else 0])[ids]
        else:
            labels = np.ones(len(X), dtype=int)
        self.model = DecisionTreeClassifier(random_state=0).fit(X, labels)


class BaselineChecks(unittest.TestCase):
    def test_feature_order_contract(self):
        import os
        from src.feature.feature import prepare_features
        df = pd.DataFrame({
            'FEWSNET_admin_code': [10, 10, 20, 20],
            'date': pd.to_datetime(['2017-01-01', '2017-02-01'] * 2),
            'fews_ipc_crisis': [1, 0, 0, 1], 'years': [2017] * 4,
            'AEZ_group': [0] * 4, 'AEZ_country_group': [0] * 4,
            'ISO_encoded': [0] * 4, 'signal': [5, 6, 7, 8],
        })
        cwd = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
                os.chdir(tmp)
                features = prepare_features(df, np.array([10, 10, 20, 20]), np.zeros((4, 2)), 1)
                np.testing.assert_array_equal(features[1], df['fews_ipc_crisis'])
                np.testing.assert_array_equal(features[0][:, features[-1].index('FEWSNET_admin_code')], df['FEWSNET_admin_code'])
                with self.assertRaisesRegex(ValueError, 'sorted'):
                    prepare_features(df.iloc[[2, 0, 3, 1]], np.array([20, 10, 20, 10]), np.zeros((4, 2)), 1)
                os.chdir(cwd)
        finally:
            os.chdir(cwd)

    def test_nonroot_candidate_also_requires_improvement(self):
        rows = np.column_stack((np.arange(8), np.repeat([10, 20, 30, 40], 2))).astype(float)
        truth = np.tile([1, 0], 4)
        def train(model, X, y, branch_id='', **kwargs):
            model.trained.append((branch_id, len(X)))
            ids = X[:, 0].astype(int)
            labels = truth[ids].copy()
            if len(X) == 8:
                labels[-2] = 0
            elif len(X) == 2:
                labels[:] = 1  # Each grandchild adds a false positive.
            model.model = DecisionTreeClassifier(random_state=0).fit(X, labels)
        def proposal(d, a, *args, **kwargs):
            mid = len(d) // 2
            return np.arange(mid), np.arange(mid, len(d)), np.zeros(len(d)), np.ones(1)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()), \
                patch.object(PresetRF, 'train', new=train), \
                patch.object(trans, 'CONTIGUITY', False), \
                patch.object(trans, 'scan', side_effect=proposal), \
                patch.object(trans, 'generate_count_grid', return_value=(None, 0, 1)):
            model = PresetRF(tmp, False)
            assigned, table, _ = trans.partition(
                model, np.vstack((rows, rows)), np.tile(truth, 2),
                np.tile(rows[:, 1].astype(int), 2), np.repeat([0, 1], 8),
                np.arange(16), np.full(16, '', dtype='<U8'),
                min_depth=3, max_depth=3, contiguity_type='polygon',
                model_dir=tmp, VIS_DEBUG_MODE=False,
            )
            self.assertEqual(set(assigned), {'0', '1'})
            self.assertEqual(len(model.trained), 7)
            self.assertFalse(table[:, 2].any())

    def test_root_and_leading_zero_branch_labels_survive_export(self):
        from src.feature.feature import create_correspondence_table
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            archive = Path(tmp) / 'result_GeoRF_2018_fs1_2018-02_visual'
            archive.mkdir()
            dates = pd.to_datetime(['2017-01-01', '2017-02-01', '2018-02-01', '2018-02-01'])
            create_correspondence_table(
                pd.DataFrame({'FEWSNET_admin_code': [10, 20, 10, 20]}),
                dates.year.to_numpy(), dates, 2018, 2, np.array(['', '00']),
                str(archive), active_lag=4, train_window_months=36,
                X_group=np.array([10, 20, 10, 20]),
            )
            table = load_correspondence_tables(Path(tmp), 'GeoRF')[0]['dataframe']
            self.assertEqual(table['partition_id'].tolist(), ['root', '00'])

    def test_monthly_handoff_is_readable_by_stage2(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            run = Path(tmp) / 'run'
            model = run / 'result_GeoRF_0'
            model.mkdir(parents=True)
            pd.DataFrame({'FEWSNET_admin_code': [10, 20], 'partition_id': ['0', '1']}).to_csv(
                model / 'correspondence_table_2018-02.csv', index=False)
            pd.DataFrame({'year': [2018], 'month': [2], 'f1(1)': [.8], 'f1_base(1)': [.6]}).to_csv(
                run / 'results_df_gp_fs1_2018_2018.csv', index=False)
            pd.DataFrame({'y_true': [1], 'y_pred': [1]}).to_csv(
                run / 'y_pred_test_gp_fs1_2018_2018.csv', index=False)
            output = Path(tmp) / 'GeoRFResults'
            collect_stage1(run, output, 2018, 2, 1)
            self.assertEqual(len(load_results_table(output, MODEL_CONFIG['georf'])), 1)
            self.assertEqual(len(load_correspondence_tables(output, 'GeoRF')), 1)
            with self.assertRaises(FileExistsError):
                collect_stage1(run, output, 2018, 2, 1)

    def test_stage1_nan_filter_keeps_correspondence_rows_aligned(self):
        import os
        from app import main_model_GF as entry
        df = pd.DataFrame({'FEWSNET_admin_code': [10, 20, 30, 40]})
        X = np.array([[1.], [np.nan], [3.], [4.]])
        dates = pd.date_range('2017-01-01', periods=4, freq='MS')
        def evaluate(*args, **kwargs):
            np.testing.assert_array_equal(args[11]['FEWSNET_admin_code'], [10, 30, 40])
            self.assertEqual(len(args[0]), len(args[11]))
            return pd.DataFrame({'year': [2018], 'month': [2], 'f1(1)': [.8], 'f1_base(1)': [.6]}), pd.DataFrame()
        cwd = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()), \
                    patch.object(sys, 'argv', ['main_model_GF.py']), \
                    patch.object(entry, 'load_and_preprocess_data', return_value=df), \
                    patch.object(entry, 'setup_spatial_groups', return_value=(np.arange(4), np.zeros((4, 2)), None)), \
                    patch.object(entry, 'prepare_features', return_value=(X, np.ones(4, int), [], [], np.repeat(2017, 4), np.arange(4), dates, ['x'])), \
                    patch.object(entry, 'run_temporal_evaluation', side_effect=evaluate), \
                    patch.object(entry, 'save_results'):
                os.chdir(tmp)
                self.assertEqual(entry.main(), 0)
                os.chdir(cwd)
        finally:
            os.chdir(cwd)

    def test_f1_includes_false_positives(self):
        truth = np.array([1, 1, 0, 0, 0])
        parent = np.array([1, 0, 0, 0, 0])
        child = np.ones(5, dtype=int)
        self.assertGreater(np.mean(child[:2]), np.mean(parent[:2]))
        self.assertLess(opt.get_score(truth, child), opt.get_score(truth, parent))
        decision = opt.select_f1_children(truth, truth[:0], parent, parent[:0], child, child[:0])
        self.assertFalse(decision[0])
        self.assertEqual(decision[1], (False, False))
        with self.assertRaises(ValueError):
            opt.get_metric_score_array(truth, child, 'class_1_f1')

    def test_regional_q_and_zero_evidence(self):
        truth = np.array([1, 0, 0, 1, 0, 0])
        pred = np.array([1, 1, 1, 0, 0, 0])
        groups = np.array([10, 20, 20, 30, 40, 40])
        gids, D, _, A = opt.get_class_wise_stat(truth, pred, groups)
        np.testing.assert_array_equal(gids, [10, 20, 30, 40])
        np.testing.assert_array_equal(D.ravel(), [2, 2, 1, 0])
        np.testing.assert_array_equal(A.ravel(), [2, 0, 0, 0])
        C, B = opt.get_c_b(D, A)
        self.assertEqual(C[1, 0], 2)  # Region containing false positives only.
        for subset in ([0, 1], [1, 2], [0, 2]):
            mask = np.isin(groups, gids[subset])
            expected = (1 - f1_score(truth[mask], pred[mask])) / (1 - f1_score(truth, pred))
            self.assertAlmostEqual(float(C[subset].sum() / B[subset].sum()), expected)
        with redirect_stdout(StringIO()):
            for d, a in ((D, A), (np.zeros((4, 1)), np.zeros((4, 1))),
                         (np.ones((4, 1)) * 2, np.ones((4, 1)) * 2),
                         (np.ones((1, 1)), np.zeros((1, 1)))):
                s0, s1, score, q = opt.scan(d, a, 0, return_score=True)
                self.assertTrue(np.isfinite(score).all() and np.isfinite(q).all())
                self.assertEqual(set(s0) | set(s1), set(range(len(d))))
                self.assertFalse(set(s0) & set(s1))

    def test_aggregate_selection_and_exact_threshold(self):
        y0, y1 = np.array([1, 0, 0]), np.array([1])
        p0, p1 = np.array([1, 0, 0]), np.array([0])
        c0, c1 = np.array([1, 1, 1]), np.array([1])
        accept, selected, _, base, best = opt.select_f1_children(y0, y1, p0, p1, c0, c1)
        self.assertTrue(accept)
        self.assertEqual(selected, (False, True))
        self.assertAlmostEqual(base, 2/3)
        self.assertEqual(best, 1)
        # F1(parent)=0, F1(child)=2/(1+199)=.01 exactly: reject.
        truth = np.r_[1, np.zeros(199, dtype=int)]
        parent = np.zeros(200, dtype=int)
        child = np.r_[np.ones(199, dtype=int), 0]
        empty = truth[:0]
        self.assertFalse(opt.select_f1_children(truth, empty, parent, empty, child, empty)[0])
        child[-2] = 0
        self.assertTrue(opt.select_f1_children(truth, empty, parent, empty, child, empty)[0])
        for truth, pred in ((np.zeros(4, int), np.zeros(4, int)),
                            (np.ones(4, int), np.ones(4, int))):
            self.assertFalse(opt.select_f1_children(truth, empty, pred, empty, pred, empty)[0])

    def test_partition_gate_and_saved_checkpoint_predictions(self):
        rows = np.column_stack((np.arange(4), [10, 10, 10, 20])).astype(float)
        X = np.vstack((rows, rows))
        y = np.tile([1, 0, 0, 1], 2)
        groups = X[:, 1].astype(int)
        split = np.repeat([0, 1], 4)
        # Freeze the spatial proposal to isolate acceptance/persistence.
        def proposal(d, a, *args, **kwargs):
            return np.array([0]), np.arange(1, len(d)), np.zeros(len(d)), np.ones(1)
        for reject in (True, False):
            with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()), \
                    patch.object(trans, 'CONTIGUITY', False), \
                    patch.object(trans, 'scan', side_effect=proposal), \
                    patch.object(trans, 'generate_count_grid', return_value=(None, 0, 1)):
                model = PresetRF(tmp, reject)
                assignments, table, _, tracker = trans.partition(
                    model, X, y, groups, split, np.arange(len(X)),
                    np.full(len(X), '', dtype='<U8'), min_depth=3, max_depth=3,
                    contiguity_type='polygon', track_partition_metrics=True,
                    model_dir=tmp, VIS_DEBUG_MODE=False,
                )
                if reject:
                    self.assertTrue((assignments == '').all())
                    self.assertFalse(table[:, 1:].any())
                    self.assertEqual(len(model.trained), 3)
                else:
                    self.assertEqual(set(assignments), {'0', '1'})
                    model.load('0')
                    np.testing.assert_array_equal(model.predict(rows[:3]), [1, 0, 0])
                    model.load('1')
                    np.testing.assert_array_equal(model.predict(rows[3:]), [1])
                    self.assertTrue(list(Path(tmp).rglob('rf_*')))
                root_metrics = [m for m in tracker.all_metrics if m['partition_round'] == 0]
                self.assertTrue(root_metrics)
                self.assertTrue(all(m['f1_after'] == 1 for m in root_metrics))

    def test_no_smote_retains_only_inherited_stage1_pseudo_rows(self):
        X = np.arange(120).reshape(40, 3)
        y = np.r_[np.zeros(35, int), np.ones(5, int)]
        seen = []
        real_fit = RandomForestClassifier.fit
        def capture(model, rows, labels, *args, **kwargs):
            seen.append((rows.copy(), labels.copy()))
            return real_fit(model, rows, labels, *args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()), \
                patch.object(RandomForestClassifier, 'fit', new=capture), \
                patch.dict(compare.RF_PARAMS, {'n_estimators': 3, 'n_jobs': 1}):
            stage1 = RFmodel(tmp, 3, n_jobs=1, num_class=2)
            stage1.train(X, y, '')
            np.testing.assert_array_equal(seen[-1][0][:-2], X)
            np.testing.assert_array_equal(seen[-1][0][-2:], np.zeros((2, 3)))
            np.testing.assert_array_equal(seen[-1][1][-2:], [0, 1])
            with self.assertRaises(ValueError):
                RFmodel(tmp, 3, use_smote=True)
            pooled = compare.train_pooled_model(X, y)
            local = compare.train_partitioned_model(X, y, np.zeros(len(y), int), min_samples=1)[0]
            self.assertEqual([len(rows) for rows, _ in seen], [42, 40, 40])
            np.testing.assert_array_equal(pooled.predict_proba(X), local.predict_proba(X))

    def test_validation_membership_and_stage2_f1_weights(self):
        groups = np.array([10, 10, 20, 20, 20, 30])
        split = group_aware_train_val_split(groups, .2, random_state=42)
        for gid in (10, 20):
            self.assertEqual(set(split['X_set'][groups == gid]), {0, 1})
        self.assertEqual(split['X_set'][-1], 0)
        frame = pd.DataFrame({'f1(1)': [.8, .4, .5], 'f1_base(1)': [.5, .5, .5]})
        np.testing.assert_allclose(compute_plan_weights(frame)['weight'], [np.log(4), 0, 0])
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame({'FEWSNET_admin_code': [20, 10], 'lat': [2, 1], 'lon': [4, 3]}).to_csv(
                Path(tmp) / 'FEWSNET_admin_code_lat_lon.csv', index=False)
            np.testing.assert_array_equal(load_lat_lon(Path(tmp), np.array([10, 20])), [[1, 3], [2, 4]])


if __name__ == '__main__':
    unittest.main(verbosity=2)
