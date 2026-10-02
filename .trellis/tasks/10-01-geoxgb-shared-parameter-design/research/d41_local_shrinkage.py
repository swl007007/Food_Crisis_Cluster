"""D41: fixed-map saved-probability geometric shrinkage; no training or selection."""
import hashlib
import json
import platform
from pathlib import Path
import sys
import numpy as np
import pandas as pd

BASE = Path(r'C:\Users\swl00\geoxgb_runs')
D34 = BASE / 'geoxgb-d34-e1-brier-20261002'
STAGE = D34 / 'stage1_e1pair'
OUT = BASE / 'd41-local-shrinkage-20261002'
DATES = ('2018-06','2018-10','2019-02','2019-06','2019-10','2020-02','2020-06')
CLASSES = ('1','2','3','4或5')
ARMS = ('root','half','full')
KEY = ['area','target_month']
hashes = {}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def track(path):
    hashes[str(path)] = sha(path)
    return path

def read(path):
    return pd.read_csv(track(path), float_precision='round_trip', keep_default_na=False,
                       dtype={'branch_id': str})

def half_prob(root, local):
    """No labels or fitted parameters; preserve identical rows exactly."""
    a = np.sqrt(root * local)
    out = a / a.sum(axis=1, keepdims=True)
    same = np.all(root == local, axis=1)
    out[same] = root[same]
    return out

def softmax(m):
    e = np.exp(m - m.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def selfcheck():
    r = np.array([[2.,-1.,0.,3.], [0.,0.,0.,0.]])
    d = np.array([[-1.,2.,.5,0.], [1.,2.,-2.,3.]])
    p, q = softmax(r), softmax(r+d)
    assert np.allclose(half_prob(p,q),softmax(r+.5*d),rtol=1e-14,atol=1e-15)
    assert np.array_equal(half_prob(p,p),p)
    # The transform's only inputs are the two probability matrices.
    truth = np.array([0,3]); before = half_prob(p,q)
    truth[:] = 3-truth
    assert np.array_equal(before,half_prob(p,q))

def score(z, pred, pc):
    y = pred >= 2
    tp, fp, fn, tn = (int(a.sum()) for a in (z&y,~z&y,z&~y,~z&~y))
    return dict(n=len(z),tp=tp,fp=fp,fn=fn,tn=tn,
                f1=2*tp/(2*tp+fp+fn) if tp+fp+fn else 0.,
                brier=float(np.mean((pc-z)**2)) if len(z) else None)

def report(f):
    z = f.truth.to_numpy() >= 2
    result = {a:score(z,f['y_'+a].to_numpy(),f['pc_'+a].to_numpy()) for a in ARMS}
    old = f.y_root.to_numpy() >= 2
    for a in ('half','full'):
        y = f['y_'+a].to_numpy() >= 2
        result[a]['changes'] = dict(corrected=int(((y==z)&(old!=z)).sum()),
            spoiled=int(((y!=z)&(old==z)).sum()),new_tp=int((z&~old&y).sum()),
            lost_tp=int((z&old&~y).sum()),new_fp=int((~z&~old&y).sum()),
            removed_fp=int((~z&old&~y).sum()))
    result['half_vs_full'] = dict(binary_flips=int(((f.y_half>=2)!=(f.y_full>=2)).sum()),
        fourclass_flips=int((f.y_half!=f.y_full).sum()),
        mean_abs_crisis_probability_delta=float((f.pc_half-f.pc_full).abs().mean()),
        max_abs_crisis_probability_delta=float((f.pc_half-f.pc_full).abs().max()))
    if f.persistence_code.notna().all():
        result['persistence'] = score(z,f.persistence_code.to_numpy(),(f.persistence_code>=2).to_numpy(float))
    return result

selfcheck()
if '--selfcheck' in sys.argv:
    print('D41 selfchecks passed'); sys.exit(0)
assert (platform.python_version(),np.__version__,pd.__version__)==('3.12.10','2.2.6','2.2.3')
assert not OUT.exists(), 'Do not overwrite existing evidence'
frames = []; provenance = {}; snapshots = {}
for h in (4,8,12):
    path = D34/'prepared'/f'snapshot_h{h}.parquet'
    # Only pre-final rows and necessary columns are loaded.
    snap = pd.read_parquet(path,columns=['area','target_month','class_code','hist_phase_o00'],
                           filters=[('target_month','<=',2020*12+11)])
    assert not snap.duplicated(KEY).any()
    snap['target_month'] = [f'{int(m)//12:04d}-{int(m)%12+1:02d}' for m in snap.target_month]
    snapshots[h] = snap
    for target in DATES:
        g = {4:'G1',8:'G4',12:'G2'}[h]
        root = f'h{h}_{target}_{g}_r80_s42_e1pair'
        cand = f'h{h}_{target}_{g}_L1_r80_s42_e1brier_gt0'
        rd, cd, ck = STAGE/'roots'/root, STAGE/'candidates'/cand, STAGE/'checkpoints'/cand
        meta = json.loads(track(rd/'root.json').read_text())
        member = read(rd/'fold_membership.csv.gz')
        rootsha = meta['root_booster_sha256']; kinds = {}
        for part in ('C','E3'):
            if part=='C':
                f = read(cd/'confirmation_predictions.csv.gz').rename(columns={'y_true':'truth','y_final':'y_full'})
                f = f.rename(columns={f'p_final_{c}':f'p_full_{c}' for c in CLASSES})
            else:
                f = read(cd/'target_predictions.csv').rename(columns={'FEWSNET_admin_code':'area',
                     'y_true_code':'truth','y_pred_partitioned_code':'y_full','y_pred_pooled_code':'y_root_saved'})
                f['target_month'] = target
                r = read(rd/'root_target_predictions.csv').rename(columns={'FEWSNET_admin_code':'area',
                     'y_true_code':'root_truth','y_pred_pooled_code':'y_root'})
                assert set(f.area)==set(r.area)
                f = f.merge(r,on='area',validate='one_to_one')
                assert np.array_equal(f.truth,f.root_truth) and np.array_equal(f.y_root_saved,f.y_root)
                f = f.rename(columns={**{f'p_partitioned_{c}':f'p_full_{c}' for c in CLASSES},
                                      **{f'p_pooled_{c}':f'p_root_{c}' for c in CLASSES}})
            assert not f.duplicated(KEY).any() and (f.target_month<='2020-12').all()
            expected = member[member.role==('confirmation' if part=='C' else 'heldout_target')]
            assert set(map(tuple,f[KEY].to_numpy()))==set(map(tuple,expected[KEY].to_numpy()))
            v = f[KEY+['truth']].merge(expected[KEY+['class_code']],on=KEY,validate='one_to_one')
            assert np.array_equal(v.truth,v.class_code)
            for branch in f.branch_id.unique():
                if branch in kinds: continue
                if branch=='root': kinds[branch]='zero_increment'; continue
                j = json.loads(track(ck/f'xgb_{branch}.json').read_text())
                ub = sha(track(ck/f'xgb_{branch}.ubj'))
                assert ub==j['booster_sha256']
                if j['kind']=='continuation':
                    assert j['increment_source']=='root' and j['parent_sha256']==j['shared_source']==rootsha
                    assert j['rounds_added']==j['actual_local_rounds']==20
                    assert j['parent_rounds']==meta['root_fit']['rounds_total'] and j['rounds_total']==j['parent_rounds']+20
                    kinds[branch]='local'
                else:
                    assert j['kind']=='fresh' and j['actual_local_rounds']==0 and ub==rootsha
                    kinds[branch]='zero_increment'
            f['route_type'] = f.branch_id.map(kinds)
            p = f[[f'p_root_{c}' for c in CLASSES]].to_numpy(float)
            q = f[[f'p_full_{c}' for c in CLASSES]].to_numpy(float)
            for a in (p,q):
                assert np.isfinite(a).all() and (a>0).all() and np.max(np.abs(a.sum(1)-1))<1e-6
            assert np.array_equal(p.argmax(1),f.y_root) and np.array_equal(q.argmax(1),f.y_full)
            zero = f.route_type.eq('zero_increment').to_numpy()
            assert np.array_equal(p[zero],q[zero])
            half = half_prob(p,q)
            assert np.array_equal(half[zero],p[zero])
            for i,c in enumerate(CLASSES): f['p_half_'+c]=half[:,i]
            f['y_half']=half.argmax(1)
            f = f.merge(snap,on=KEY,how='left',validate='one_to_one',indicator=True)
            assert f['_merge'].eq('both').all() and np.array_equal(f.truth,f.class_code)
            f['persistence_code']=f.hist_phase_o00-1
            known=f.hist_phase_o00.notna(); per=f.hist_phase_o00.ge(3); z=f.truth.ge(2)
            f['transition']=np.where(~known,'missing',np.where(per,'1','0').astype(object)+np.where(z,'1','0'))
            f['part']=part; f['root']=root; f['horizon']=h
            for a in ARMS: f['pc_'+a]=f['p_'+a+'_3']+f['p_'+a+'_4或5']
            keep=['root','part','horizon',*KEY,'truth','branch_id','routing','route_type','persistence_code','transition']
            keep += [n for a in ARMS for n in [f'y_{a}',f'pc_{a}',*[f'p_{a}_{c}' for c in CLASSES]]]
            frames.append(f[keep])
        provenance[root]=kinds

rows=pd.concat(frames,ignore_index=True)
assert len(rows)==321047 and not rows.duplicated(['root','part',*KEY]).any()
summary={'definition':'D41 fixed alpha=.5 saved-probability geometric shrinkage; no fit or selection',
         'script_sha256':sha(Path(__file__)),'planning_commit':'15b1085','input_hashes':hashes,
         'provenance':provenance,'parts':{}}
reference=json.loads((BASE/'d36-transfer-diagnostic-20261002/summary.json').read_text())
for part,f in rows.groupby('part',sort=True):
    d={'all':report(f),'matched':report(f[f.persistence_code.notna()]),'per_root':{},'per_horizon':{},'transitions':{},'routes':{}}
    for col,tag in [('root','per_root'),('horizon','per_horizon'),('transition','transitions'),('route_type','routes')]:
        for key,chunk in f.groupby(col,sort=True):d[tag][str(key)]=report(chunk)
        for arm in ARMS:
            for count in ('n','tp','fp','fn','tn'):
                assert sum(v[arm][count] for v in d[tag].values())==d['all'][arm][count]
            if arm!='root':
                for count in d['all'][arm]['changes']:
                    assert sum(v[arm]['changes'][count] for v in d[tag].values())==d['all'][arm]['changes'][count]
    old=next(r for r in reference['all'] if r['part']==part)
    for arm,ref in [('root','root'),('full','brier_local')]:
        for count in ('tp','fp','fn','tn'): assert d['all'][arm][count]==old[f'{ref}_{count}']
        assert abs(d['all'][arm]['brier']-old[f'{ref}_brier_loss'])<1e-14
    d['fold_means']={a:{'f1_delta_root':float(np.mean([v[a]['f1']-v['root']['f1'] for v in d['per_root'].values()])),
        'brier_delta_root':float(np.mean([v[a]['brier']-v['root']['brier'] for v in d['per_root'].values()]))} for a in ('half','full')}
    summary['parts'][part]=d
OUT.mkdir()
rows.to_csv(OUT/'rows.csv.gz',index=False,float_format='%.17g')
summary['rows_sha256']=sha(OUT/'rows.csv.gz')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
for part,d in summary['parts'].items(): print(part,json.dumps({k:d[k] for k in ('all','matched','fold_means')}))
print('OK',OUT,len(rows),'rows')
