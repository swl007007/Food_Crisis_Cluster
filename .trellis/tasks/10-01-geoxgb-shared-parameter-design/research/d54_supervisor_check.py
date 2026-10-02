"""Independent D54 verification; no producer imports, model loading, or fitting.

Run with C:\\Users\\swl00\\AppData\\Local\\Microsoft\\WindowsApps\\python3.12.exe.
Only this script's directory receives verification artifacts.
"""
import hashlib
import json
import math
import subprocess
import sys
import traceback
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = Path('C:/Users/swl00/geoxgb_runs')
REPO = Path('C:/users/swl00/ifpri dropbox/weilun shi/google fund/analysis/2.source_code/step5_geo_rf_trial/food_crisis_cluster')
RESEARCH = '.trellis/tasks/10-01-geoxgb-shared-parameter-design/research'
PIN = '6d5619b'
OUT = BASE / 'geoxgb-d54-fixed-policy-local-contrast-20261002'
D52 = BASE / 'geoxgb-d52-binary-root-20261002'
D38 = BASE / 'geoxgb-d38-persistence-margin-root-20261002'
KEY = ['root', 'part', 'area', 'target_month', 'horizon']
CLS = ['1', '2', '3', '4或5']
ARMS = ['raw_root', 'raw_full', 'post_root', 'post_full']
REPORT = {'status': 'running', 'failures': [], 'check_counts': {}, 'hashes': {},
          'runtime': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__},
          'producer_imported': False, 'model_loaded_or_fit': False, 'tolerance': 1e-12}
COUNTS = Counter()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def check(ok, label, group='integrity'):
    COUNTS[group] += 1
    if not bool(ok):
        REPORT['failures'].append(label)


def compare(actual, expected, label, group):
    if isinstance(expected, dict):
        check(isinstance(actual, dict) and set(actual) == set(expected), label + ': keys', group)
        for k, v in expected.items():
            if isinstance(actual, dict) and k in actual:
                compare(actual[k], v, label + '/' + str(k), group)
        return
    if isinstance(expected, (float, np.floating)):
        check(actual is not None and math.isfinite(float(actual)) and abs(float(actual) - float(expected)) <= 1e-12,
              label + ': actual=' + repr(actual) + ' expected=' + repr(float(expected)), group)
    else:
        check(actual == expected, label + ': actual=' + repr(actual) + ' expected=' + repr(expected), group)


def record(name):
    rel = RESEARCH + '/' + name
    pinned = subprocess.run(['git', 'show', PIN + ':' + rel], cwd=REPO, check=True, capture_output=True).stdout
    current = (REPO / rel).read_bytes()
    check(pinned == current, 'record differs from pinned commit: ' + name, 'record_pin')
    REPORT['hashes']['record/' + name] = hashlib.sha256(pinned).hexdigest()
    return json.loads(pinned)


def hashcheck(path, expected, name, group):
    got = sha(path)
    REPORT['hashes'][name] = got
    check(got == expected, 'sha256 mismatch: ' + name, group)


def load(path):
    return pd.read_csv(path, dtype={'area': str, 'target_month': str}, float_precision='round_trip')


def metric(indices, arm, y, calls, masses):
    pred = calls[arm][indices] >= 2 if arm != 'persistence' else calls[arm][indices]
    truth = y[indices]
    tn, fp, fn, tp = np.bincount(2 * truth.astype(int) + pred.astype(int), minlength=4).tolist()
    den = 2 * tp + fp + fn
    f = Fraction(2 * tp, den) if den else Fraction(0)
    score = masses[arm][indices]
    brier = float(np.dot(score - truth, score - truth) / len(indices)) if len(indices) else None
    return {'n': len(indices), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'f1': float(f), 'f1_exact': str(f.numerator) + '/' + str(f.denominator), 'brier': brier}


def main():
    check(np.__version__ == '2.2.6' and pd.__version__ == '2.2.3', 'frozen dependency versions')
    check(sys.version_info[:2] == (3, 12), 'frozen Python major/minor')
    REPORT['pinned_commit'] = subprocess.run(['git', 'rev-parse', PIN], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()
    d41rec = record('d41_summary.json')
    d52rec = record('d52_completion.json')
    d39rec = record('d39_probability_diagnostic.json')
    d40rec = record('d40_summary.json')
    completion = json.loads((OUT / 'completion.json').read_text())
    identity = json.loads((OUT / 'identity.json').read_text())
    summary = json.loads((OUT / 'summary.json').read_text())
    check(completion['status'] == 'complete', 'output completion status')
    check(set(completion['outputs']) == {'identity.json', 'per_root.csv', 'changes.csv', 'summary.json'}, 'output inventory')
    REPORT['hashes']['output/completion.json'] = sha(OUT / 'completion.json')
    for name, expected in completion['outputs'].items():
        hashcheck(OUT / name, expected, 'output/' + name, 'output_hash')
    check((D52 / 'completion.json').read_bytes() == (REPO / RESEARCH / 'd52_completion.json').read_bytes(), 'D52 completion byte equality')
    for name in ('d41_summary.json', 'd52_completion.json', 'd39_probability_diagnostic.json', 'd40_summary.json'):
        check(identity['records_sha256'].get(name) == REPORT['hashes']['record/' + name], 'identity record hash: ' + name, 'identity')
    producer_rel = RESEARCH + '/d54_fixed_policy_local_contrast.py'
    producer_bytes = subprocess.run(['git', 'show', PIN + ':' + producer_rel], cwd=REPO, check=True, capture_output=True).stdout
    producer_blob = subprocess.run(['git', 'rev-parse', PIN + ':' + producer_rel], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()
    check(identity['script']['head_commit'] == REPORT['pinned_commit'], 'producer commit identity', 'identity')
    check(identity['script']['git_blob'] == producer_blob, 'producer blob identity', 'identity')
    check(identity['script']['sha256'] == hashlib.sha256(producer_bytes).hexdigest(), 'producer script SHA256', 'identity')
    REPORT['producer_script_sha256'] = hashlib.sha256(producer_bytes).hexdigest()
    roots = sorted({k.split('/')[0] for k in d52rec['outputs'] if k.endswith('/rows_C.csv.gz')})
    check(len(roots) == 21, 'expected 21 roots')
    for h in (4, 8, 12):
        check(sum(r.startswith('h' + str(h) + '_') for r in roots) == 7, 'seven roots per H' + str(h))
    path41 = BASE / 'd41-local-shrinkage-20261002/rows.csv.gz'
    hashcheck(path41, d41rec['rows_sha256'], 'd41_rows', 'input_hash')
    d41 = load(path41).set_index(KEY).sort_index()
    pieces = []
    for root in roots:
        for part in ('C', 'E3'):
            name = root + '/rows_' + part + '.csv.gz'
            hashcheck(D52 / name, d52rec['outputs'][name], 'd52/' + name, 'input_hash')
            pieces.append(load(D52 / name).assign(root=root, part=part))
    d52 = pd.concat(pieces, ignore_index=True).set_index(KEY).sort_index()
    check(d41.index.is_unique and d52.index.is_unique, 'D41/D52 unique keys', 'alignment')
    check(d41.index.equals(d52.index), 'D41/D52 exact key sets and ordering', 'alignment')
    if not d41.index.equals(d52.index):
        raise ValueError('Cannot align D41/D52 rows')
    check(set(d41.index.get_level_values('part')) == {'C', 'E3'}, 'only C/E3', 'alignment')
    check(np.array_equal(d41['truth'], d52['truth']), 'D41/D52 truth equality', 'alignment')
    origin = d52['persistence_code'].to_numpy(float)
    known = np.isfinite(origin)
    check(np.array_equal(d41['persistence_code'].to_numpy(float), origin, equal_nan=True), 'D41/D52 origin equality and missingness', 'alignment')
    check(np.isin(origin[known], [0, 1, 2, 3]).all(), 'valid known origin codes', 'alignment')
    truth = d52['truth'].to_numpy(int)
    check(np.isin(truth, [0, 1, 2, 3]).all(), 'valid truth codes', 'alignment')
    y = truth >= 2
    check(np.array_equal(d52['truth_crisis'].to_numpy(int), y.astype(int)), 'truth crisis conversion', 'alignment')
    rootp = d41[['p_root_' + c for c in CLS]].to_numpy(float)
    fullp = d41[['p_full_' + c for c in CLS]].to_numpy(float)
    check(np.array_equal(rootp, d52[['p_original_' + c for c in CLS]].to_numpy(float)), 'D41 root = D52 original exact', 'probability')
    z = (d41['route_type'] == 'zero_increment').to_numpy()
    check(set(d41['route_type']) <= {'local', 'zero_increment'}, 'valid route values', 'alignment')
    check(np.array_equal(rootp[z], fullp[z]), 'zero increment raw identity', 'probability')
    p = {'raw_root': rootp, 'raw_full': fullp}
    REPORT['raw_probability_max_sum_deviation'] = {}
    for arm in ['raw_root', 'raw_full']:
        check(np.isfinite(p[arm]).all() and (p[arm] >= 0).all() and (p[arm] <= 1).all(), arm + ': finite bounded probabilities', 'probability')
        deviation = float(np.max(np.abs(p[arm].sum(axis=1) - 1)))
        REPORT['raw_probability_max_sum_deviation'][arm] = deviation
        check(deviation < 1e-5, arm + ': probability sum validation', 'probability')
        adjusted = p[arm].copy()
        positions = np.flatnonzero(known)
        adjusted[positions, origin[known].astype(int)] *= 5.0
        adjusted[known] /= adjusted[known].sum(axis=1, keepdims=True)
        p[arm.replace('raw', 'post')] = adjusted
        check(np.array_equal(adjusted[~known], p[arm][~known]), arm + ': missing origin unchanged', 'probability')
    check(np.array_equal(p['post_root'][z], p['post_full'][z]), 'zero increment post identity', 'probability')
    calls = {a: np.argmax(v, axis=1) for a, v in p.items()}
    masses = {a: v[:, 2:].sum(axis=1) / v.sum(axis=1) for a, v in p.items()}
    check(np.array_equal(calls['raw_root'], d41['y_root'].to_numpy(int)), 'raw root saved argmax', 'probability')
    check(np.array_equal(calls['raw_full'], d41['y_full'].to_numpy(int)), 'raw full saved argmax', 'probability')
    calls['persistence'] = origin >= 2
    masses['persistence'] = calls['persistence'].astype(float)
    frame = d41.reset_index()[KEY + ['route_type']]
    expected_cells, expected_changes = {}, {}
    missing_by_root = {}
    for (root, part), g in frame.groupby(['root', 'part'], sort=True):
        indices = g.index.to_numpy()
        hs = g['horizon'].unique()
        check(len(hs) == 1, root + '/' + part + ': one horizon', 'alignment')
        h = int(hs[0])
        missing = int((~known[indices]).sum())
        missing_by_root[(root, part)] = missing
        for keyset in ('matched', 'all'):
            selected = indices[known[indices]] if keyset == 'matched' else indices
            for arm in ARMS + (['persistence'] if keyset == 'matched' else []):
                expected_cells[(root, part, h, keyset, arm)] = {'missing_origin_n': missing, **metric(selected, arm, y, calls, masses)}
            for route, rg in frame.loc[selected].groupby('route_type'):
                ii = rg.index.to_numpy()
                for prefix in ('raw', 'post'):
                    a, b = prefix + '_full', prefix + '_root'
                    ma, mb = metric(ii, a, y, calls, masses), metric(ii, b, y, calls, masses)
                    expected_changes[(root, part, keyset, h, route, prefix + ':' + a + '-' + b)] = {
                        'n': len(ii), 'crisis_flips': int(np.count_nonzero((calls[a][ii] >= 2) != (calls[b][ii] >= 2))),
                        'fourclass_flips': int(np.count_nonzero(calls[a][ii] != calls[b][ii])),
                        **{'d_' + t: ma[t] - mb[t] for t in ('tp', 'fp', 'fn')}}
    saved = load(OUT / 'per_root.csv').set_index(['root', 'part', 'horizon', 'keyset', 'arm'])
    check(saved.index.is_unique and set(saved.index) == set(expected_cells), 'per_root cell keys exact', 'per_root')
    check(len(expected_cells) == 378, '378 independent per_root cells', 'per_root')
    for key, m in expected_cells.items():
        if key in saved.index:
            compare(saved.loc[key].to_dict(), m, str(key), 'per_root')
    changes = load(OUT / 'changes.csv').set_index(['root', 'part', 'keyset', 'horizon', 'route_type', 'pair'])
    check(changes.index.is_unique and set(changes.index) == set(expected_changes), 'changes keys exact', 'changes')
    for key, m in expected_changes.items():
        if key in changes.index:
            compare(changes.loc[key].to_dict(), m, str(key), 'changes')
    pools = {}
    for part in ('C', 'E3'):
        part_indices = frame.index[frame['part'] == part].to_numpy()
        for keyset in ('matched', 'all'):
            base_indices = part_indices[known[part_indices]] if keyset == 'matched' else part_indices
            arms = ARMS + (['persistence'] if keyset == 'matched' else [])
            for h in ('all', 4, 8, 12):
                ii = base_indices if h == 'all' else base_indices[frame.loc[base_indices, 'horizon'].to_numpy() == h]
                rs = sorted(frame.loc[ii, 'root'].unique())
                cell = {'n_roots': len(rs), 'missing_origin_n': int((~known[ii]).sum())}
                fractions = {}
                for arm in arms:
                    m = metric(ii, arm, y, calls, masses)
                    del m['f1_exact']
                    fractions[arm] = {r: Fraction(expected_cells[(r, part, int(frame.loc[frame['root'] == r, 'horizon'].iloc[0]), keyset, arm)]['f1_exact']) for r in rs}
                    m['mean_fold_f1'] = float(sum(fractions[arm].values()) / len(rs))
                    cell[arm] = m
                wins = {}
                for a, b in [('post_full', 'post_root'), ('post_full', 'persistence'), ('raw_full', 'raw_root')]:
                    if b in arms:
                        signs = [(fractions[a][r] > fractions[b][r]) - (fractions[a][r] < fractions[b][r]) for r in rs]
                        wins[a + '_vs_' + b] = {'wins': signs.count(1), 'ties': signs.count(0), 'losses': signs.count(-1)}
                cell['fold_wins'] = wins
                pools[part + '|' + keyset + '|h' + str(h)] = cell
    compare(summary['pooled'], pools, 'summary/pooled', 'pooled')
    totals = {}
    for (root, part, keyset, h, route, pair), m in expected_changes.items():
        t = totals.setdefault((part, keyset, route, pair), Counter())
        t.update(m)
    saved_totals = {(x['part'], x['keyset'], x['route_type'], x['pair']): {k: v for k, v in x.items() if k not in ('part', 'keyset', 'route_type', 'pair')} for x in summary['change_totals']}
    check(len(saved_totals) == len(summary['change_totals']), 'unique change_total keys', 'change_totals')
    compare(saved_totals, {k: dict(v) for k, v in totals.items()}, 'summary/change_totals', 'change_totals')
    for key, m in expected_changes.items():
        if key[4] == 'zero_increment':
            check(all(m[k] == 0 for k in ('crisis_flips', 'fourclass_flips', 'd_tp', 'd_fp', 'd_fn')), 'zero increment change: ' + str(key), 'zero_increment')
    compare(summary['rows'], {part: int((frame['part'] == part).sum()) for part in ('C', 'E3')}, 'summary/rows', 'metadata')
    compare(summary['missing_origin_rows'], {part: int((~known[frame['part'] == part]).sum()) for part in ('C', 'E3')}, 'summary/missing', 'metadata')
    compare(summary['q'], {'origin': .625, 'other': .125}, 'summary/q', 'metadata')
    check(summary['n_roots'] == 21, 'summary n_roots', 'metadata')
    check(d39rec['input_hashes'] == d40rec['input_hashes'] and set(d39rec['input_hashes']) == set(roots), 'D38 records hash/root agreement')
    d38pieces = []
    for r in roots:
        name = r + '/rows_E3.csv.gz'
        hashcheck(D38 / name, d39rec['input_hashes'][r], 'd38/' + name, 'input_hash')
        d38pieces.append(load(D38 / name).assign(root=r, part='E3'))
    d38 = pd.concat(d38pieces, ignore_index=True).set_index(KEY).sort_index()
    e3indices = frame.index[frame['part'] == 'E3'].to_numpy()
    check(d38.index.is_unique and d38.index.equals(d41.index[e3indices]), 'D38 exact unique E3 keys', 'd38')
    check(np.array_equal(d38['truth'].to_numpy(int), truth[e3indices]), 'D38 E3 truth equality', 'd38')
    maxdiff = float(np.max(np.abs(d38[['p_posthoc_' + c for c in CLS]].to_numpy(float) - p['post_root'][e3indices])))
    check(maxdiff <= 1e-12, 'D38 p_posthoc match 1e-12', 'd38')
    check(np.array_equal(d38['y_posthoc'].to_numpy(int), calls['post_root'][e3indices]), 'D38 posthoc argmax equality', 'd38')
    oldorigin = d38['persistence_code'].to_numpy(float)
    disagreed = int(np.count_nonzero(~((np.isnan(oldorigin) & ~known[e3indices]) | (oldorigin == origin[e3indices]))))
    d38root = {}
    for r in roots:
        ii = frame.index[(frame['root'] == r) & (frame['part'] == 'E3')].to_numpy()
        m = metric(ii, 'post_root', y, calls, masses)
        d38root[r] = {k: m[k] for k in ('n', 'tp', 'fp', 'fn', 'tn')}
        h = int(frame.loc[ii[0], 'horizon'])
        for keyset in ('all', 'matched'):
            expected = expected_cells[(r, 'E3', h, keyset, 'post_root')]
            old = d39rec['per_root'][r][keyset]['posthoc']['argmax_crisis']
            compare({k: old[k] for k in ('tp', 'fp', 'fn', 'tn')}, {k: expected[k] for k in ('tp', 'fp', 'fn', 'tn')}, 'D39 ' + r + '/' + keyset, 'd39')
    compare(summary['reconciliation']['d38_posthoc_E3'], {'max_abs_posthoc_diff': maxdiff, 'n': len(e3indices), 'per_root': d38root, 'persistence_code_disagreements_d38_vs_d52_recorded': disagreed}, 'summary/D38', 'reconciliation')
    check(summary['reconciliation']['d41_rows_sha256'] == REPORT['hashes']['d41_rows'], 'summary D41 row hash', 'reconciliation')
    for part in ('C', 'E3'):
        pi = frame.index[frame['part'] == part].to_numpy()
        groups = [('all', pi), ('matched', pi[known[pi]])]
        for name, field in [('per_root', 'root'), ('per_horizon', 'horizon'), ('routes', 'route_type')]:
            for val, g in frame.loc[pi].groupby(field):
                groups.append((name + '/' + str(val), g.index.to_numpy()))
        for name, ii in groups:
            rec = d41rec['parts'][part]
            for segment in name.split('/'):
                rec = rec[segment]
            for label, arm in [('root', 'raw_root'), ('full', 'raw_full')]:
                m = metric(ii, arm, y, calls, masses)
                compare({k: rec[label][k] for k in ('n', 'tp', 'fp', 'fn', 'tn')}, {k: m[k] for k in ('n', 'tp', 'fp', 'fn', 'tn')}, 'D41 ' + part + '/' + name + '/' + label, 'd41_reconciliation')
    expected_input_hashes = {k: v for k, v in REPORT['hashes'].items() if k == 'd41_rows' or k.startswith('d52/') or k.startswith('d38/')}
    compare(identity['inputs_sha256'], expected_input_hashes, 'identity input hashes', 'identity')
    REPORT.update({'row_counts': summary['rows'], 'missing_origin_rows': summary['missing_origin_rows'],
                   'per_root_cells': len(expected_cells), 'pooled_cells': len(pools),
                   'changes_cells': len(expected_changes), 'change_totals_cells': len(totals),
                   'd38_max_abs_probability_diff': maxdiff, 'd38_origin_disagreements': disagreed,
                   'e3_matched_summary': pools['E3|matched|hall'],
                   'input_files_verified': len(expected_input_hashes), 'output_files_verified': len(completion['outputs'])})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        REPORT['failures'].append(traceback.format_exc())
    REPORT['check_counts'] = dict(sorted(COUNTS.items()))
    REPORT['checks_total'] = sum(COUNTS.values())
    REPORT['verifier_sha256'] = sha(Path(__file__))
    REPORT['status'] = 'passed' if not REPORT['failures'] else 'failed'
    (HERE / 'verification.json').write_text(json.dumps(REPORT, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    log = json.dumps({k: REPORT.get(k) for k in ('status', 'checks_total', 'check_counts', 'row_counts', 'per_root_cells', 'pooled_cells', 'changes_cells', 'change_totals_cells', 'input_files_verified', 'output_files_verified', 'd38_max_abs_probability_diff', 'failures')}, indent=2, sort_keys=True)
    (HERE / 'verification.log').write_text(log + '\n', encoding='utf-8')
    print(log)
    sys.exit(0 if REPORT['status'] == 'passed' else 1)
