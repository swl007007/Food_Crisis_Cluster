import json, pandas as pd, numpy as np
R = r'C:\Users\swl00\geoxgb_runs\geoxgb-d26-20261001\stage1_diagnostics'
c = pd.read_csv(R + r'\candidates.csv'); s = json.load(open(R + r'\summary.json'))
c = c[c.status == 'completed']
print('candidates', len(c))
def tab(by):
    g = c.groupby(by).agg(n=('candidate', 'size'), split=('n_terminal', lambda x: int((x > 1).sum())),
                          terminals=('n_terminal', 'mean'), fits=('child_fits', 'mean'),
                          e2_crisis=('e2_crisis_gain', 'mean'), e3_crisis=('e3_crisis_gain', 'mean'),
                          e3_pos=('e3_crisis_gain', lambda x: int((x > 0).sum())),
                          e2pos_e3neg=('e2_crisis_gain', 'size'),
                          e2_4=('e2_fourclass_gain', 'mean'), e3_4=('e3_fourclass_gain', 'mean'))
    g['e2pos_e3neg'] = c.groupby(by).apply(lambda d: int(((d.e2_crisis_gain > 0) & (d.e3_crisis_gain < 0)).sum()), include_groups=False)
    return g.round(4)
pd.set_option('display.width', 250)
for by in (['threshold_family', 'local_config'], ['ratio'], ['horizon'], ['target_month'], ['threshold_family', 'local_config', 'ratio']):
    print('\n', by); print(tab(by).to_string())
print('\noverall', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s['overall'].items()})
print('\nper-candidate (h, T, ratio, L, fam, terminals, E2, E3):')
for r in c.sort_values(['horizon', 'target_month', 'ratio', 'local_config', 'threshold_family']).itertuples():
    print(f'  h{r.horizon} {r.target_month} {r.ratio} {r.local_config} {r.threshold_family:5s} term={r.n_terminal:2d}  E2c={r.e2_crisis_gain:+.4f} E3c={r.e3_crisis_gain:+.4f}  root_E3c={r.e3_crisis_root:.4f}')
print('\npooled confusion (targets, gt0/L1 vs root):')
for row in s['pooled_confusion']:
    if row['which'] in ('target_part', 'target_root'):
        print(row['family'], row['local'], row['which'], 'crisis', row['crisis_[[nn,nc],[cn,cc]]'], 'crisisF1', round(row['crisis_f1'], 4), 'perclassF1', row['per_class_f1'])
