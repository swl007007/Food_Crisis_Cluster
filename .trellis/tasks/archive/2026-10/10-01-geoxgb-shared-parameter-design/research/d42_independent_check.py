"""Independent raw-XGB replay and keyed scoring of D42; no runner imports or fitting."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

BASE=Path(r'C:\Users\swl00\geoxgb_runs')
PACKAGE=Path(r'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\2.source_code\Step5_Geo_RF_trial\Food_Crisis_Cluster\FEWSNETGeoXGBExperiment')
RUN=BASE/'geoxgb-d42-map-transfer-20261002'
D34=BASE/'geoxgb-d34-e1-brier-20261002'
STAGE=D34/'stage1_e1pair'
D35=BASE/'geoxgb-d35-global-increment-20261002'
ARMS=('root','global20','current_map_refit','old_map_refit')
LABELS=('1','2','3','4或5')
KEY=['area','target_month']
expected=json.loads((BASE/'d42_input_check.json').read_text())
summary=json.loads((RUN/'summary.json').read_text())
features=json.loads((PACKAGE/'feature-schema.json').read_text())['ordered_features']
assert len(features)==162
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
mi=lambda s:int(s[:4])*12+int(s[5:])-1
checks=0

def check(ok,what):
    global checks
    checks+=1
    assert ok,what

def read(p):
    return pd.read_csv(p,float_precision='round_trip',keep_default_na=False,
        dtype={'area':str,'FEWSNET_admin_code':str,'region_current':str,'region_old':str,'spatial_partition_id':str})

def booster(p):
    b=xgb.Booster();b.load_model(p);return b

def structure(b):
    j=json.loads(b.save_raw(raw_format='json'))['learner']
    return j['learner_model_param'],j['gradient_booster']['model']

def predict(b,X):
    a=np.array(X,dtype=float,copy=True);a[np.isinf(a)]=np.nan
    return b.predict(xgb.DMatrix(a,missing=np.nan,nthread=4))

def score(truth,p):
    p=np.asarray(p,dtype=np.float64)  # report sums saved native float32 probabilities in float64
    pred=p.argmax(1);mat=np.zeros((4,4),dtype=np.int64)
    np.add.at(mat,(truth,pred),1)
    tp=int(mat[2:,2:].sum());fp=int(mat[:2,2:].sum());fn=int(mat[2:,:2].sum())
    f=Fraction(2*tp,2*tp+fp+fn) if tp+fp+fn else Fraction(0)
    macro=np.mean([2*mat[i,i]/(mat[i].sum()+mat[:,i].sum()) if mat[i].sum()+mat[:,i].sum() else 0 for i in range(4)])
    return {'confusion_fourclass':mat.tolist(),'crisis':dict(tp=tp,fp=fp,fn=fn,tn=len(truth)-tp-fp-fn),
            'crisis_f1_exact':str(f),'crisis_f1':float(f),'macro_f1_fourclass':float(macro),
            'crisis_brier':float(np.mean((p[:,2:].sum(1)-(truth>=2))**2))}

def compare(got,ref,where):
    for k,v in got.items():
        if k not in ref:continue  # pooled records omit Brier
        if isinstance(v,(dict,list,str)):check(v==ref[k],(where,k))
        else:check(abs(v-ref[k])<1e-12,(where,k,v,ref[k]))

check(set(summary['per_pair'])==set(expected['pairs']),'12 exact pairs')
check(summary['fits']==175,'175 fits')
allrows=[];models_seen=0;replayed_rows=0
for h in (4,8,12):
    snap=pd.read_parquet(D34/'prepared'/f'snapshot_h{h}.parquet',
        columns=['area','target_month','class_code']+features,filters=[('target_month','<=',2020*12+11)])
    snap['area']=snap.area.astype(str)
    snap['target_month']=[f'{int(m)//12:04d}-{int(m)%12+1:02d}' for m in snap.target_month]
    check(not snap.duplicated(KEY).any(),('snapshot keys',h))
    snap=snap.set_index(KEY)
    for name,e in expected['pairs'].items():
        if e['horizon']!=h:continue
        meta=json.loads((STAGE/'roots'/name/'root.json').read_text())
        g={4:'G1',8:'G4',12:'G2'}[h]
        cand=f'h{h}_{e["target"]}_{g}_L1_r80_s42_e1brier_gt0'
        root=booster(STAGE/'checkpoints'/cand/'xgb_root.ubj')
        rp,rt=structure(root);nroot=root.num_boosted_rounds()
        global20=booster(D35/name/'global_plus20.ubj')
        maps={};models={}
        for arm,date in [('current_map_refit',e['target']),('old_map_refit',e['source_target'])]:
            key='current' if arm=='current_map_refit' else 'old'
            path=STAGE/'candidates'/f'h{h}_{date}_{g}_L1_r80_s42_e1brier_gt0'/'assignment_evidence.csv'
            check(sha(path)==e['maps'][arm]['map_sha256'],('map hash',name,arm))
            amap=read(path).set_index('FEWSNET_admin_code').spatial_partition_id.to_dict();maps[arm]=amap
            models[arm]={}
            for region,rec in e['maps'][arm]['regions'].items():
                path=RUN/name/f'{key}_map'/f'region_{region}.json';j=json.loads(path.read_text())
                check(j['support']==rec['support'],('support',name,arm,region))
                check(j['fitting_keys_sha256']==rec['fit_keys_sha256'],('fitting digest',name,arm,region))
                check(j['eligible']==rec['eligible'],('eligible',name,arm,region))
                check(j['member_areas']==sorted(int(a) for a,s in amap.items() if s==region),('members',name,arm,region))
                check(j['root_booster_sha256']==meta['root_booster_sha256'],('parent identity',name,arm,region))
                if not rec['eligible']:continue
                ub=path.with_suffix('.ubj');check(sha(ub)==j['ubj_sha256'],('ubj hash',name,arm,region))
                b=booster(ub);bp,bt=structure(b)
                check(b.num_boosted_rounds()==nroot+20,('rounds',name,arm,region))
                check(bp['base_score']==rp['base_score'] and bt['trees'][:len(rt['trees'])]==rt['trees']
                      and bt['tree_info'][:len(rt['tree_info'])]==rt['tree_info'],('frozen root prefix',name,arm,region))
                models[arm][region]=b;models_seen+=1
        member=read(STAGE/'roots'/name/'fold_membership.csv.gz')
        for part,role in [('C','confirmation'),('E3','heldout_target')]:
            f=read(RUN/name/f'rows_{part}.csv.gz')
            check(not f.duplicated(KEY).any(),('output duplicates',name,part))
            want=member[member.role==role]
            check(set(map(tuple,f[KEY].to_numpy()))==set(map(tuple,want[KEY].to_numpy())),('complete keys',name,part))
            s=snap.loc[pd.MultiIndex.from_frame(f[KEY])]
            check(np.array_equal(s.class_code.to_numpy(),f.truth.to_numpy()),('truth',name,part))
            phase=s.hist_phase_o00.to_numpy(float);pc=np.minimum(phase,4)-1
            savedpc=pd.to_numeric(f.persistence_code,errors='coerce').to_numpy(float)
            check(np.array_equal(pc,savedpc,equal_nan=True),('persistence',name,part))
            X=s[features].to_numpy(float);P={'root':predict(root,X),'global20':predict(global20,X)}
            for arm in ('current_map_refit','old_map_refit'):
                key='current' if arm=='current_map_refit' else 'old'
                sid=np.array([maps[arm].get(a,'') for a in f.area],dtype=object)
                reasons=np.array(['missing' if x=='' else 's-1' if x=='s-1' else 'region' if x in models[arm] else 'insufficient_support' for x in sid])
                check(np.array_equal(sid,f[f'region_{key}'].to_numpy()) and np.array_equal(reasons,f[f'route_{key}'].to_numpy()),('route',name,part,arm))
                out=P['root'].copy()
                for region,b in models[arm].items():
                    mask=sid==region
                    if mask.any():out[mask]=predict(b,X[mask])
                P[arm]=out
            known=np.isfinite(pc);truth=f.truth.to_numpy(int)
            for arm,p in P.items():
                saved=f[[f'p_{arm}_{c}' for c in LABELS]].to_numpy(float)
                check(np.array_equal(p,saved),('raw probability replay',name,part,arm))
                check(np.array_equal(p.argmax(1),f['y_'+arm]),('argmax',name,part,arm))
                refs=summary['per_pair'][name]['scores'][part]
                compare(score(truth,p),refs['all'][arm],(name,part,arm,'all'))
                compare(score(truth[known],p[known]),refs['matched_persistence'][arm],(name,part,arm,'matched'))
                replayed_rows+=len(p)
            d35=read(D35/name/f'rows_{part}.csv.gz').set_index(KEY).loc[pd.MultiIndex.from_frame(f[KEY])]
            for arm in ('root','global20'):
                check(np.array_equal(P[arm],d35[[f'p_{arm}_{c}' for c in LABELS]].to_numpy()),('D35 control',name,part,arm))
            f['persistence_code']=savedpc;allrows.append(f)
        print('checked',name,flush=True)
check(models_seen==175,'175 saved models checked')
f=pd.concat(allrows,ignore_index=True);facts={}
for group,chunk in [('overall_12',f)]+[(f'H{h}',f[f.horizon==h]) for h in (4,8,12)]:
    refs=summary['overall_12'] if group=='overall_12' else summary['by_horizon'][group]
    facts[group]={}
    for part,partrows in chunk.groupby('part'):
        truth=partrows.truth.to_numpy(int);known=partrows.persistence_code.notna().to_numpy()
        facts[group][part]={'all':{},'matched':{}}
        for arm in ARMS:
            p=partrows[[f'p_{arm}_{c}' for c in LABELS]].to_numpy(float)
            a,b=score(truth,p),score(truth[known],p[known])
            compare(a,refs[part]['pooled_all'][arm],(group,part,arm,'pooled'))
            compare(b,refs[part]['pooled_matched_persistence'][arm],(group,part,arm,'matched pooled'))
            facts[group][part]['all'][arm]=a;facts[group][part]['matched'][arm]=b
        persistence=score(truth[known],np.eye(4)[partrows.persistence_code.to_numpy()[known].astype(int)])
        compare(persistence,refs[part]['pooled_matched_persistence']['persistence'],(group,part,'persistence'))
        facts[group][part]['matched']['persistence']=persistence
result={'checks':checks,'issues':[],'models_replayed':models_seen,'probability_rows_replayed':replayed_rows,
        'keyed_rows':len(f),'facts':facts,'scope':'all12 pairs,175 local prefixes/pools,raw C/E3 replay,per-pair and pooled all/matched scores; no fitting'}
out=BASE/'d42_independent_results.json';assert not out.exists();out.write_text(json.dumps(result,indent=2),encoding='utf-8')
print('PASS',checks,'checks,',models_seen,'models,',len(f),'keyed rows')
