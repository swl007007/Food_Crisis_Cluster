"""D43 independent roles, raw XGB replay and scoring; no production imports or fits."""
import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

BASE = Path(r'C:\Users\swl00\geoxgb_runs')
PACKAGE = Path(r'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\2.source_code\Step5_Geo_RF_trial\Food_Crisis_Cluster\FEWSNETGeoXGBExperiment')
D34 = BASE / 'geoxgb-d34-e1-brier-20261002'
STAGE = D34 / 'stage1_e1pair'
D35 = BASE / 'geoxgb-d35-global-increment-20261002'
ARMS = ('root', 'global20', 'random_map_refit', 'temporal_map_refit')
LABELS = ('1', '2', '3', '4或5')
KEY = ['area', 'target_month']
parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
RUN = args.run
assert not args.out.exists()
expected = json.loads((BASE / 'd43_input_reference.json').read_text())
summary = json.loads((RUN / 'summary.json').read_text())
features = json.loads((PACKAGE / 'feature-schema.json').read_text())['ordered_features']
assert len(features) == 162
checks = models_seen = probability_rows = 0
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
mi = lambda s: int(s[:4]) * 12 + int(s[5:]) - 1


def check(ok, what):
    global checks
    checks += 1
    assert ok, what


def read(path):
    return pd.read_csv(path, float_precision='round_trip', keep_default_na=False,
                       dtype={'area': str, 'FEWSNET_admin_code': str,
                              'spatial_partition_id': str, 'region_random': str, 'region_temporal': str})


def booster(path):
    obj = xgb.Booster()
    obj.load_model(path)
    return obj


def structure(obj):
    j = json.loads(obj.save_raw(raw_format='json'))['learner']
    return j['learner_model_param'], j['gradient_booster']['model']


def predict(obj, X):
    X = np.array(X, dtype=float, copy=True)
    X[np.isinf(X)] = np.nan
    return obj.predict(xgb.DMatrix(X, missing=np.nan, nthread=4))


def digest(rows):
    keys = np.column_stack((rows.area.astype(np.int64), [mi(t) for t in rows.target_month]))
    return hashlib.sha256(np.ascontiguousarray(keys, dtype=np.int64).tobytes()).hexdigest()


def score(truth, p):
    p = np.asarray(p, dtype=np.float64)
    pred = p.argmax(1)
    mat = np.zeros((4, 4), dtype=np.int64)
    np.add.at(mat, (truth, pred), 1)
    tp, fp, fn = int(mat[2:, 2:].sum()), int(mat[:2, 2:].sum()), int(mat[2:, :2].sum())
    f = Fraction(2 * tp, 2 * tp + fp + fn) if tp + fp + fn else Fraction(0)
    macro = np.mean([2 * mat[i, i] / (mat[i].sum() + mat[:, i].sum())
                     if mat[i].sum() + mat[:, i].sum() else 0 for i in range(4)])
    return {'confusion_fourclass': mat.tolist(),
            'crisis': dict(tp=tp, fp=fp, fn=fn, tn=len(truth) - tp - fp - fn),
            'crisis_f1_exact': str(f), 'crisis_f1': float(f), 'macro_f1_fourclass': float(macro),
            'crisis_brier': float(np.mean((p[:, 2:].sum(1) - (truth >= 2)) ** 2))}


def compare(got, ref, where):
    for key, value in got.items():
        check(key in ref, (where, 'missing metric', key))
        if isinstance(value, (dict, list, str)):
            check(value == ref[key], (where, key, value, ref[key]))
        else:
            check(abs(value - ref[key]) < 1e-12, (where, key, value, ref[key]))


check(set(summary['per_pair']) == set(expected), 'all21 pairs')
check(summary['budget']['search_roots'] == 21 and summary['budget']['searches'] == 24, 'fit/search budget')
check(json.loads((RUN / 'gate.json').read_text())['passed'], 'run gates')
allrows = []
for h in (4, 8, 12):
    snap = pd.read_parquet(D34 / 'prepared' / f'snapshot_h{h}.parquet',
                           columns=KEY + ['class_code'] + features,
                           filters=[('target_month', '<=', 2020 * 12 + 11)])
    snap['area'] = snap.area.astype(str)
    snap['target_month'] = [f'{int(m)//12:04d}-{int(m)%12+1:02d}' for m in snap.target_month]
    check(not snap.duplicated(KEY).any(), ('snapshot keys', h))
    snap = snap.set_index(KEY)
    for name, exp in expected.items():
        if exp['horizon'] != h:
            continue
        pdir = RUN / 'pairs' / name
        meta = json.loads((pdir / 'pair.json').read_text())
        source_meta = json.loads((STAGE / 'roots' / name / 'root.json').read_text())
        membership = read(STAGE / 'roots' / name / 'fold_membership.csv.gz')
        check(sha(STAGE / 'roots' / name / 'fold_membership.csv.gz') == exp['source_sha256'], (name, 'source keys hash'))
        actual = read(pdir / 'legal_pool_membership.csv.gz')
        check(actual[KEY].equals(membership[KEY]), (name, 'membership exact order'))
        check(actual.d34_role.tolist() == membership.role.tolist(), (name, 'original roles'))
        temporal = np.where(membership.role == 'heldout_target', 'excluded_E3',
                            np.where(membership.target_month.isin(exp['search_months']), 'S_tb', 'FIT_tb'))
        check(np.array_equal(actual.temporal_role, temporal), (name, 'temporal roles'))
        check(meta['search_months'] == exp['search_months'], (name, 'date table'))
        check(meta['search_root']['fit_keys_sha256'] == exp['temporal_fit']['ordered_key_sha256'], (name, 'search fit digest'))
        check(meta['overlap']['S_tb_keys_sha256'] == exp['temporal_search']['ordered_key_sha256'], (name, 'search keys digest'))
        check(meta['overlap']['S_tb_overlap_rows_by_d34_role'] == exp['search_overlap_with_D34_roles'], (name, 'search/refit overlap'))
        fit = membership[membership.role == 'fitting'].copy()
        check(digest(fit) == exp['forecast_fit']['ordered_key_sha256'], (name, 'current fit digest'))
        g = {4: 'G1', 8: 'G4', 12: 'G2'}[h]
        cand = f'h{h}_{exp["target"]}_{g}_L1_r80_s42_e1brier_gt0'
        root_path = STAGE / 'checkpoints' / cand / 'xgb_root.ubj'
        root = booster(root_path)
        rp, rt = structure(root)
        nroot = root.num_boosted_rounds()
        check(sha(root_path) == meta['root_booster_sha256'] == source_meta['root_booster_sha256'], (name, 'forecast root'))
        sr = json.loads((pdir / 'search_root.json').read_text())
        check(sha(pdir / 'search_root.ubj') == sr['booster_sha256'], (name, 'search root artifact'))
        check(sr['fit_keys_sha256'] == exp['temporal_fit']['ordered_key_sha256'], (name, 'saved search fit identity'))
        global20 = booster(D35 / name / 'global_plus20.ubj')
        map_paths = {'random': STAGE / 'candidates' / cand / 'assignment_evidence.csv',
                     'temporal': pdir / 'temporal' / 'candidates' / meta['temporal_candidate'] / 'assignment_evidence.csv'}
        maps, models = {}, {}
        for kind, path in map_paths.items():
            amap = read(path)
            check(not amap.FEWSNET_admin_code.duplicated().any(), (name, kind, 'unique map keys'))
            check(np.array_equal(amap.spatial_partition_id == 's-1', amap.search_rows == 0), (name, kind, 'D32 mask'))
            maps[kind] = amap.set_index('FEWSNET_admin_code').spatial_partition_id.to_dict()
            named = set(maps[kind].values()) - {'s-1'}
            models[kind] = {}
            for region in sorted(named):
                rec_path = pdir / f'{kind}_map' / f'region_{region}.json'
                rec = json.loads(rec_path.read_text())
                members = sorted(int(a) for a, s in maps[kind].items() if s == region)
                sub = fit[fit.area.map(maps[kind]) == region]
                support = dict(rows=len(sub), areas=sub.area.nunique(), dates=sub.target_month.nunique(), classes=sub.class_code.nunique(),
                               class_counts=[int((sub.class_code == c).sum()) for c in range(4)])
                eligible = all(support[k] >= floor for k, floor in dict(rows=500, areas=50, dates=6, classes=2).items())
                check(rec['support'] == support and rec['eligible'] == eligible, (name, kind, region, 'support'))
                check(rec['member_areas'] == members and rec['fitting_keys_sha256'] == digest(sub), (name, kind, region, 'current pool'))
                check(rec['root_booster_sha256'] == sha(root_path), (name, kind, region, 'current parent'))
                if not eligible:
                    continue
                ub = rec_path.with_suffix('.ubj')
                check(sha(ub) == rec['ubj_sha256'], (name, kind, region, 'saved model'))
                b = booster(ub)
                bp, bt = structure(b)
                check(b.num_boosted_rounds() == nroot + 20, (name, kind, region, '20 rounds'))
                check(bp['base_score'] == rp['base_score'] and bt['trees'][:len(rt['trees'])] == rt['trees']
                      and bt['tree_info'][:len(rt['tree_info'])] == rt['tree_info'], (name, kind, region, 'shared prefix'))
                models[kind][region] = b
                models_seen += 1
            check(len(models[kind]) == meta['map_counts'][kind]['eligible'], (name, kind, 'model count'))
        f = read(pdir / 'rows_E3.csv.gz')
        check(not f.duplicated(KEY).any(), (name, 'prediction unique keys'))
        want = membership[membership.role == 'heldout_target']
        check(set(map(tuple, f[KEY].to_numpy())) == set(map(tuple, want[KEY].to_numpy())), (name, 'complete E3 keys'))
        s = snap.loc[pd.MultiIndex.from_frame(f[KEY])]
        truth = f.truth.to_numpy(int)
        check(np.array_equal(s.class_code, truth), (name, 'truth'))
        pc = np.minimum(s.hist_phase_o00.to_numpy(float), 4) - 1
        saved_pc = pd.to_numeric(f.persistence_code, errors='coerce').to_numpy(float)
        check(np.array_equal(pc, saved_pc, equal_nan=True), (name, 'persistence'))
        X = s[features].to_numpy(float)
        probs = {'root': predict(root, X), 'global20': predict(global20, X)}
        for kind in ('random', 'temporal'):
            sid = np.array([maps[kind].get(a, '') for a in f.area], dtype=object)
            route = np.array(['missing' if x == '' else 's-1' if x == 's-1' else 'region'
                              if x in models[kind] else 'insufficient_support' for x in sid])
            check(np.array_equal(sid, f[f'region_{kind}']) and np.array_equal(route, f[f'route_{kind}']), (name, kind, 'routes'))
            p = probs['root'].copy()
            for region, b in models[kind].items():
                mask = sid == region
                if mask.any():
                    p[mask] = predict(b, X[mask])
            probs[f'{kind}_map_refit'] = p
        known = np.isfinite(pc)
        refs = summary['per_pair'][name]['scores']['E3']
        for arm, p in probs.items():
            check(np.array_equal(p, f[[f'p_{arm}_{c}' for c in LABELS]].to_numpy(float)), (name, arm, 'raw replay'))
            check(np.array_equal(p.argmax(1), f[f'y_{arm}']), (name, arm, 'argmax'))
            compare(score(truth, p), refs['all'][arm], (name, arm, 'all'))
            compare(score(truth[known], p[known]), refs['matched_persistence'][arm], (name, arm, 'matched'))
            probability_rows += len(p)
        old = read(D35 / name / 'rows_E3.csv.gz').set_index(KEY).loc[pd.MultiIndex.from_frame(f[KEY])]
        for arm in ('root', 'global20'):
            check(np.array_equal(probs[arm], old[[f'p_{arm}_{c}' for c in LABELS]].to_numpy(float)), (name, arm, 'D35 control'))
        f['persistence_code'] = saved_pc
        allrows.append(f)
        print('checked', name, flush=True)
check(models_seen == summary['budget']['refits'], 'all saved refits')
f = pd.concat(allrows, ignore_index=True)
facts = {}
groups = [('overall_21', f, summary['overall_21'])]
groups += [(f'H{h}', f[f.horizon == h], summary['by_horizon'][f'H{h}']) for h in (4, 8, 12)]
groups += [(t, f[f.target_month == t], ref) for t, ref in summary['by_target'].items()]
for group, chunk, refs in groups:
    truth = chunk.truth.to_numpy(int)
    known = chunk.persistence_code.notna().to_numpy()
    facts[group] = {'all': {}, 'matched': {}}
    for arm in ARMS:
        p = chunk[[f'p_{arm}_{c}' for c in LABELS]].to_numpy(float)
        a, b = score(truth, p), score(truth[known], p[known])
        compare(a, refs['pooled_all'][arm], (group, arm, 'pooled'))
        compare(b, refs['matched_persistence'][arm], (group, arm, 'matched pooled'))
        facts[group]['all'][arm], facts[group]['matched'][arm] = a, b
    pscore = score(truth[known], np.eye(4)[chunk.persistence_code.to_numpy()[known].astype(int)])
    compare(pscore, refs['matched_persistence']['persistence'], (group, 'persistence'))
    facts[group]['matched']['persistence'] = pscore
result = dict(checks=checks, issues=[], models_replayed=models_seen, probability_rows_replayed=probability_rows,
              keyed_rows=len(f), facts=facts,
              scope='All21 keyed roles/support/current-root prefixes/E3 raw replay; per-pair/all/H/target all+matched scores; no fitting. Search algorithm not independently rerun.')
args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print('PASS', checks, 'checks,', models_seen, 'models,', len(f), 'keyed rows')
