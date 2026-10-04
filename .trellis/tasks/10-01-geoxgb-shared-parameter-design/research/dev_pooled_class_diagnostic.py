import json, numpy as np, pandas as pd
R = r'C:\Users\swl00\geoxgb_runs\geoxgb-v1-20261001'
sel = json.load(open(R + r'\gscreen\selection.json'))['selected']
p = pd.read_csv(R + r'\gscreen\predictions.csv.gz', float_precision='round_trip')
b = pd.read_csv(R + r'\prepared\ledgers\dev_baselines.csv', float_precision='round_trip'); b['target_month'] = b['target_label']
def conf(t, q): return np.bincount(t * 4 + q, minlength=16).reshape(4, 4)
def f1s(m):
    tp = np.diag(m); fp = m.sum(0) - tp; fn = m.sum(1) - tp; d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0)
for h in (4, 8, 12):
    k = b[b.horizon == h].merge(p[(p.horizon == h) & (p.g_config == sel[str(h)])], on=['area', 'target_month', 'horizon'])
    k = k[k.persistence_code.notna() & (k.expert_code.notna() if h != 12 else True)]
    t = k.truth_code.to_numpy(int)
    print(f'\n=== H{h} main cohort n={len(k)}  truth class counts {np.bincount(t, minlength=4).tolist()}')
    arms = {'pooled ' + sel[str(h)]: k.y_pred_code, 'persistence': k.persistence_code}
    if h != 12: arms['expert'] = k.expert_code
    for name, q in arms.items():
        m = conf(t, q.to_numpy(float).astype(int)); f = f1s(m)
        rec = np.diag(m) / np.maximum(m.sum(1), 1); share = m.sum(0) / m.sum()
        print(f'{name:12s} macroF1={f.mean():.4f}  F1={np.round(f,3).tolist()}  recall={np.round(rec,3).tolist()}  pred_share={np.round(share,3).tolist()}')
    # agreement of pooled with persistence and accuracy where they differ
    pp, pe = k.y_pred_code.to_numpy(int), k.persistence_code.to_numpy(float).astype(int)
    diff = pp != pe
    print(f'pooled != persistence on {diff.mean():.3f} of keys; there pooled correct {np.mean(pp[diff]==t[diff]):.3f}, persistence correct {np.mean(pe[diff]==t[diff]):.3f}')
    # per target month macro F1
    rows = []
    for tm, g in k.groupby('target_month'):
        tt = g.truth_code.to_numpy(int)
        rows.append((tm, len(g), round(f1s(conf(tt, g.y_pred_code.to_numpy(int))).mean(), 3), round(f1s(conf(tt, g.persistence_code.to_numpy(float).astype(int))).mean(), 3)))
    print('per month (T, n, pooled, persistence):', rows)
