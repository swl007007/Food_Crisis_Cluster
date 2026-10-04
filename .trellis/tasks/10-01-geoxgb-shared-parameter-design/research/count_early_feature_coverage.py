"""Read-only source support counts. No snapshots or models are created."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('/mnt/c/users/swl00/ifpri dropbox/weilun shi/google fund/analysis/2.source_code/step5_geo_rf_trial/food_crisis_cluster')
SOURCE = REPO.parents[2] / '1.Source Data' / 'FEWSNET_forecast_unadjusted_bm.csv'
SCHEMA = json.loads((REPO / 'FEWSNETFourClassBaseline/feature-schema.json').read_text())
CALENDAR = json.loads((REPO / '.trellis/tasks/10-01-geoxgb-shared-parameter-design/research/calendar-support-counts.json').read_text())
spec = importlib.util.spec_from_file_location('ff', REPO / 'FEWSNETFourClassBaseline/src/feature/fourclass_features.py')
ff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ff)
columns = SCHEMA['static_sources'] + SCHEMA['dynamic_sources_at_origin']
usecols = ['FEWSNET_admin_code', 'date', 'ISO', 'fews_ipc'] + columns
counts = {}
label_chunks = []
early_chunks = []
first_finite = {}
last_finite = {}
rows = 0
for chunk in pd.read_csv(SOURCE, usecols=usecols, chunksize=25000):
    rows += len(chunk)
    for date, group in chunk.groupby('date'):
        values = group[columns].to_numpy(float)
        c = counts.setdefault(date, {'rows': 0, 'finite': np.zeros(len(columns), np.int64), 'nan': np.zeros(len(columns), np.int64), 'inf': np.zeros(len(columns), np.int64), 'areas': set()})
        c['rows'] += len(group)
        c['finite'] += np.isfinite(values).sum(axis=0)
        c['nan'] += np.isnan(values).sum(axis=0)
        c['inf'] += np.isinf(values).sum(axis=0)
        c['areas'].update(group['FEWSNET_admin_code'].astype(int))
        for name, n in zip(columns, np.isfinite(values).sum(axis=0)):
            if n:
                first_finite[name] = min(first_finite.get(name, date), date)
                last_finite[name] = max(last_finite.get(name, date), date)
    valid = chunk.loc[chunk['fews_ipc'].notna(), ['FEWSNET_admin_code', 'date', 'fews_ipc']].copy()
    label_chunks.append(valid)
    early = chunk.loc[chunk['date'] < '2014-01'].copy()
    if len(early):
        early_chunks.append(early)
print('read complete', rows, flush=True)
labels = pd.concat(label_chunks, ignore_index=True)
labels = labels.rename(columns={'FEWSNET_admin_code': 'area', 'fews_ipc': 'phase'})
labels['month'] = ff.month_index(pd.to_datetime(labels['date']))
labels['phase'] = np.minimum(labels['phase'], 4)
panel = pd.concat(early_chunks, ignore_index=True)
panel['date'] = pd.to_datetime(panel['date'])
scaffold = ff.Scaffold(panel, columns)
def mi(s):
    y, m = map(int, s.split('-'))
    return y * 12 + m - 1
def ml(i):
    return f'{i // 12:04d}-{i % 12 + 1:02d}'
def earliest_inner(h, floor):
    outer = min((r for r in CALENDAR['stage1'] if r['H'] == h and r['T'] >= floor), key=lambda r: r['T'])
    a = outer['internal']['last_six'][0]
    v = mi(a['V'])
    fit_dates = sorted(t for t in CALENDAR['source']['date_rows'] if v - 59 <= mi(t) < v)
    return {'outer_T': outer['T'], 'outer_O': outer['O'], 'U': a['U'], 'V': a['V'], 'fit_interval': [ml(v - 59), a['V']], 'earliest_fit_T': fit_dates[0], 'earliest_fit_O': ml(mi(fit_dates[0]) - h), 'earliest_derived_covariate_month': ml(mi(fit_dates[0]) - h - 12), 'earliest_36_month_history_boundary': ml(mi(fit_dates[0]) - h - 35)}
result = {'source': str(SOURCE), 'source_bytes': SOURCE.stat().st_size, 'reader': 'Linux pandas ' + pd.__version__ + '; numeric missingness counts only, no estimator/runtime reproduction', 'schema_counts': {k: len(SCHEMA[k]) for k in ['static_sources', 'dynamic_sources_at_origin', 'legacy_covariate_derived', 'known_calendar', 'ordered_features']}, 'rows': rows, 'dates': [min(counts), max(counts)], 'months': len(counts), 'monthly_rows_minmax': [min(c['rows'] for c in counts.values()), max(c['rows'] for c in counts.values())], 'monthly_areas_minmax': [min(len(c['areas']) for c in counts.values()), max(len(c['areas']) for c in counts.values())], 'source_coverage': {}, 'selected_months': {}, 'early_feature_sets': {}}
for name in columns:
    idx = columns.index(name)
    c0 = counts['2010-01']
    early_counts = [c for d, c in counts.items() if d < '2014-01']
    result['source_coverage'][name] = {'first_finite': first_finite.get(name), 'last_finite': last_finite.get(name), '2010-01_finite': int(c0['finite'][idx]), '2010-01_nan': int(c0['nan'][idx]), '2010-01_inf': int(c0['inf'][idx]), '2010-2013_finite': int(sum(c['finite'][idx] for c in early_counts)), '2010-2013_nan': int(sum(c['nan'][idx] for c in early_counts)), '2010-2013_inf': int(sum(c['inf'][idx] for c in early_counts))}
for date in ['2010-01','2010-03','2010-05','2011-03','2014-01','2014-04','2015-01']:
    c = counts[date]
    result['selected_months'][date] = {'rows': c['rows'], 'finite': dict(zip(columns, map(int,c['finite']))), 'nan': dict(zip(columns, map(int,c['nan']))), 'inf': {k: int(v) for k,v in zip(columns,c['inf']) if v}}
history_columns = [n for ns in SCHEMA['history_blocks'].values() for n in ns]
blocks = {'static': SCHEMA['static_sources'], 'dynamic_at_origin': SCHEMA['dynamic_sources_at_origin'], 'covariate_derived': SCHEMA['legacy_covariate_derived'], 'calendar': SCHEMA['known_calendar'], 'history': history_columns}
for h in [4,8,12]:
    stage1 = earliest_inner(h, '2018-01')
    development = earliest_inner(h, '2019-01')
    keys = labels.loc[labels['date'] < '2014-01'].copy()
    areas = keys['area'].to_numpy(int)
    targets = keys['month'].to_numpy(int)
    origins = targets - h
    cov = ff.covariate_features(scaffold, SCHEMA, areas, targets, origins)
    hist = ff.history_features(labels[['area','month','phase']], areas, origins)
    feats = pd.concat([cov, hist],axis=1)[SCHEMA['ordered_features']]
    by_h = {'earliest_stage1': stage1, 'earliest_development': development, 'cohorts': {}}
    for cohort, boundary in [('stage1',stage1['earliest_fit_T']),('development',development['earliest_fit_T'])]:
        pick = keys['date'].ge(boundary).to_numpy()
        f = feats.loc[pick]
        cohort_origins = origins[pick]
        chosen = keys.loc[pick]
        info = {'selection': f'real label keys in union of inner W=59 fitting windows, earliest T={boundary}, clipped to T<2014-01 for early support inspection', 'keys': len(f), 'first_target': chosen['date'].min(), 'last_target': chosen['date'].max(), 'first_origin': ml(cohort_origins.min()), 'last_origin': ml(cohort_origins.max()), 'origins_before_scaffold': int((cohort_origins < scaffold.first_month).sum()), 'derived_window_crosses_scaffold': int((cohort_origins-12 < scaffold.first_month).sum()), 'any_missing_feature_rows': int((~np.isfinite(f.to_numpy(float))).any(axis=1).sum()), 'all_finite_feature_rows': int(np.isfinite(f.to_numpy(float)).all(axis=1).sum()), 'blocks': {}, 'missing_cells_by_feature': {name: int(n) for name, n in zip(f.columns,(~np.isfinite(f.to_numpy(float))).sum(axis=0)) if n}}
        for block, names in blocks.items():
            finite = np.isfinite(f[names].to_numpy(float))
            info['blocks'][block] = {'columns': len(names), 'all_finite_rows': int(finite.all(axis=1).sum()), 'all_missing_rows': int((~finite).all(axis=1).sum()), 'nonfinite_cells': int((~finite).sum())}
        first = f.loc[chosen['date'].eq(boundary).to_numpy()]
        info['earliest_fitting_target_features'] = {'rows': len(first), 'origin': ml(mi(boundary)-h), 'finite_cells_per_row_minmax': [int(np.isfinite(first.to_numpy(float)).sum(axis=1).min()), int(np.isfinite(first.to_numpy(float)).sum(axis=1).max())], 'finite_features_for_all_rows': [name for name in first if np.isfinite(first[name]).all()], 'missing_features_for_all_rows': [name for name in first if (~np.isfinite(first[name])).all()]}
        by_h['cohorts'][cohort] = info
    result['early_feature_sets'][str(h)] = by_h
    print('feature counts complete', h, flush=True)
digest = hashlib.sha256()
with SOURCE.open('rb') as handle:
    for block in iter(lambda: handle.read(1 << 20), b''):
        digest.update(block)
result['source_sha256'] = digest.hexdigest()
Path('/tmp/geoxgb_early_feature_coverage.json').write_text(json.dumps(result,indent=2))
print('saved /tmp/geoxgb_early_feature_coverage.json', flush=True)
