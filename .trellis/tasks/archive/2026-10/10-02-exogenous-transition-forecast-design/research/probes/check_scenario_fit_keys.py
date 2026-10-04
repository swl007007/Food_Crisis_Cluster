"""Independent input-key reconciliation; no package imports, features, fits or RUN writes."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

run = Path(sys.argv[1])
output = Path(sys.argv[2])
mi = lambda s: int(s[:4]) * 12 + int(s[5:7]) - 1
digest = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
obs = pd.read_csv(run / 'prepared/ledgers/observations.csv',
                  usecols=['area', 'month', 'country', 'class_code'])
ledger = pd.read_csv(run / 'prepared/manifests/release_ledger.csv', dtype=str)
ledger = ledger[ledger['product'] == 'CS'].copy()
ledger['month'] = ledger.reference_month.map(mi)
ledger['release'] = ledger.release_date.map(mi)
assert not obs.duplicated(['area', 'month']).any()
assert not ledger.duplicated(['country', 'month']).any()
obs = obs.merge(ledger[['country', 'month', 'release']], on=['country', 'month'],
                how='left', validate='many_to_one').sort_values(['area', 'month'])
assert obs.release.notna().all()
due = ledger.groupby('month').release.min()
paths = sorted((run / 'scenario_globals').rglob('*.json'))
results, problems = [], []
for path in paths:
    raw = path.read_bytes()
    rec = json.loads(raw)
    origin, k = mi(rec['origin_month']), int(rec['intensity_k'])
    released_cycles = due[due <= origin].index.tolist()
    assert len(released_cycles) >= k
    own = set(released_cycles[-k:]) if k else set()
    excluded = set(map(mi, rec['excluded_months']))
    masked = own | excluded
    bad = []
    if masked != set(map(mi, rec['masked_months'])):
        bad.append('recorded_mask')
    fit = obs[(obs.month >= origin - 59) & (obs.month < origin)
              & (obs.release <= origin) & ~obs.month.isin(masked)]
    copies = 1 if rec['strategy'] == 'A' else 3
    assert rec['strategy'] in ('A', 'B')
    a = np.repeat(fit.area.to_numpy(dtype='<i8'), copies)
    m = np.repeat(fit.month.to_numpy(dtype='<i8'), copies)
    v = np.tile(np.arange(copies, dtype='<i8'), len(fit))
    keys = np.column_stack([a, m, v]).astype('<i8')
    y = np.repeat(fit.class_code.to_numpy(dtype='<i8'), copies)
    weights = None if copies == 1 else digest(np.full(len(y), 1 / 3, dtype='<f8'))
    wanted = {'fit_keys_sha256': digest(keys), 'labels_sha256': digest(y),
              'weights_sha256': weights, 'original_keys': len(fit), 'rows': len(y),
              'class_counts': np.bincount(y, minlength=4).tolist()}
    bad += [name for name, value in wanted.items() if rec[name] != value]
    support = {'rows': len(fit), 'areas': fit.area.nunique(), 'dates': fit.month.nunique(),
               'classes': fit.class_code.nunique(),
               'class_counts': np.bincount(fit.class_code.to_numpy(dtype=int), minlength=4).tolist()}
    if rec['fit_support'] != support:
        bad.append('original_support')
    results.append({'record': path.relative_to(run).as_posix(),
                    'record_sha256': hashlib.sha256(raw).hexdigest(),
                    'original_keys': len(fit), 'problems': bad})
    if bad:
        problems.append(results[-1])
report = {'scope': 'all global records present at probe start, including completed/in-flight historical cache',
          'method': 'independent explicit filters and ordered key/label/weight byte hashes; no package imports',
          'records': len(results), 'problems': problems, 'results': results,
          'limits': 'does not reconstruct covariates, prove booster internals or local per-key fitting provenance'}
output.write_text(json.dumps(report, indent=1), encoding='utf-8')
print(json.dumps({'records': len(results), 'problems': len(problems), 'first_problems': problems[:3]}))
if problems:
    raise SystemExit(1)
