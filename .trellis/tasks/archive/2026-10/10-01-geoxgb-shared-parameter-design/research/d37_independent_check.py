"""Independent D37 weight/row/model/metric replay. No refits and no production imports."""
import hashlib,json,sys
from pathlib import Path
from fractions import Fraction
import numpy as np
import pandas as pd
import xgboost as xgb

source,run,pkg=map(Path,sys.argv[1:4]);stage=source/'stage1_e1pair'
su=json.loads((run/'summary.json').read_text());features=json.loads((pkg/'feature-schema.json').read_text())['ordered_features']
expected={f'h{h}_{t}_{g}_r80_s42_e1pair' for h,g in [(4,'G1'),(8,'G4'),(12,'G2')]
 for t in ('2018-06','2018-10','2019-02','2019-06','2019-10','2020-02','2020-06')}
assert set(su['per_root'])==expected
assert {p.name for p in run.iterdir() if p.is_dir()}==expected
def read(p):return pd.read_csv(p,float_precision='round_trip')
def mon(s):return int(s[:4])*12+int(s[5:])-1
def confusion(t,p):
 m=np.zeros((4,4),dtype=np.int64);np.add.at(m,(np.asarray(t,dtype=int),np.asarray(p,dtype=int)),1);return m
def f1(m):
 tp=int(m[2:,2:].sum());fp=int(m[:2,2:].sum());fn=int(m[2:,:2].sum())
 return Fraction(2*tp,2*tp+fp+fn) if 2*tp+fp+fn else Fraction(0)
tot={p:{k:np.zeros((4,4),dtype=np.int64) for k in ('original','weighted')} for p in ('C','E3')}
loss={p:{k:[] for k in ('original','weighted')} for p in ('C','E3')};checks=0;npred=0;results=[]
for h in (4,8,12):
 snap=pd.read_parquet(source/'prepared'/f'snapshot_h{h}.parquet',columns=['area','target_month','class_code']+features,
 filters=[('target_month','<=',2020*12+11)]).set_index(['area','target_month'])
 assert not snap.index.duplicated().any()
 for name in sorted(n for n in expected if n.startswith(f'h{h}_')):
  rd=run/name;root=json.loads((stage/'roots'/name/'root.json').read_text());meta=json.loads((rd/'weighted_root.json').read_text())
  mem=read(stage/'roots'/name/'fold_membership.csv.gz');fit=mem[mem.role=='fitting'];fw=read(rd/'fitting_weights.csv.gz')
  assert list(zip(fw.area,fw.target_month))==list(zip(fit.area,fit.target_month))
  assert np.array_equal(fw.class_code,fit.class_code)
  m=np.array([mon(x) for x in fw.target_month],dtype=np.int64);o=mon(root['target_month'])-h
  assert np.all((m>=o-59)&(m<o))
  u=np.exp2(-((o-1)-m)/24.);w64=u/u.mean();w=w64.astype(np.float32)
  assert np.allclose(w64,fw.weight_float64,rtol=0,atol=1e-15)
  assert np.array_equal(w,fw.weight_float32.to_numpy(dtype=np.float32))
  rec=meta['fit_record']['sample_weight'];sha=hashlib.sha256(w.tobytes()).hexdigest()
  assert sha==rec['sha256'] and rec['dtype']=='float32' and rec['n']==len(fit)
  keys=np.column_stack([fw.area.to_numpy(dtype=np.int64),m])
  assert hashlib.sha256(keys.tobytes()).hexdigest()==root['fitting_keys_sha256']==meta['fitting_keys_sha256']
  assert meta['fitting_rows']==meta['fit_record']['rows']==len(fit)
  for k,v in root['config']['G'].items():
   if k!='rounds':assert meta['fit_record']['params'][k]==v,(name,k)
  saved=stage/'checkpoints'/su['per_root'][name].get('hard','not_stored')/'xgb_root.ubj'
  if not saved.exists():saved=stage/'checkpoints'/name.replace('_r80_s42_e1pair','_L1_r80_s42_e1hard_gt0')/'xgb_root.ubj'
  assert hashlib.sha256(saved.read_bytes()).hexdigest()==root['root_booster_sha256']==meta['original_root_sha256']
  assert hashlib.sha256((rd/'weighted_root.ubj').read_bytes()).hexdigest()==meta['weighted_root_sha256']
  boosters={k:xgb.Booster(model_file=str(p)) for k,p in [('original',saved),('weighted',rd/'weighted_root.ubj')]}
  for b in boosters.values():b.set_param({'nthread':4});assert b.num_boosted_rounds()==root['config']['G']['rounds']
  for part,role in [('C','confirmation'),('E3','heldout_target')]:
   f=read(rd/f'rows_{part}.csv.gz');wanted=mem[mem.role==role]
   assert list(zip(f.area,f.target_month))==list(zip(wanted.area,wanted.target_month))
   assert np.array_equal(f.truth,wanted.class_code)
   idx=list(zip(f.area,[mon(x) for x in f.target_month]));ss=snap.loc[idx]
   assert np.array_equal(ss.class_code,f.truth)
   X=ss[features].to_numpy(dtype=float);X[~np.isfinite(X)]=np.nan;dm=xgb.DMatrix(X,missing=np.nan)
   per=np.minimum(ss.hist_phase_o00.to_numpy(dtype=float),4)-1
   assert np.allclose(per,f.persistence_code,equal_nan=True)
   known=np.isfinite(per);blocks=su['per_root'][name]['scores'][part]
   assert blocks['n']==len(f) and blocks['matched_persistence']['n']==int(known.sum())
   for k,b in boosters.items():
    pr=b.predict(dm).astype(float);assert np.array_equal(pr,f[[f'p_{k}_{c}' for c in ('1','2','3','4或5')]].to_numpy())
    assert np.array_equal(pr.argmax(axis=1),f['y_'+k])
    for mask,block in [(np.ones(len(f),dtype=bool),blocks['all']),(known,blocks['matched_persistence'])]:
     mm=confusion(f.truth[mask],f['y_'+k][mask]);assert mm.tolist()==block[k]['confusion_fourclass']
     assert f1(mm)==Fraction(block[k]['crisis_f1_exact']);checks+=1
     lossval=float(np.mean((pr[mask,2:].sum(axis=1)-(f.truth.to_numpy()[mask]>=2))**2))
     assert abs(lossval-block[k]['crisis_brier'])<1e-14
    tot[part][k]+=confusion(f.truth,f['y_'+k]);loss[part][k].extend(((pr[:,2:].sum(axis=1)-(f.truth.to_numpy()>=2))**2).tolist())
   mm=confusion(f.truth[known],per[known]);assert mm.tolist()==blocks['matched_persistence']['persistence']['confusion_fourclass'];checks+=1
   assert sum(x['n'] for x in blocks['transition_groups_post_hoc'].values())==len(f)
   old=f.y_original.to_numpy()>=2;new=f.y_weighted.to_numpy()>=2;truth=f.truth.to_numpy()>=2
   groups=np.where(~known,'missing',np.where(per>=2,'1','0').astype(object)+np.where(truth,'1','0').astype(object))
   for g,block in blocks['transition_groups_post_hoc'].items():
    mask=groups==g;assert mask.sum()==block['n']
    assert int(((new==truth)&(old!=truth)&mask).sum())==block['corrected']
    assert int(((new!=truth)&(old==truth)&mask).sum())==block['spoiled']
   npred+=len(f)
  results.append({'root':name,'weights_sha256':sha,'fitting_rows':len(fit)})
aggregate={}
for p in ('C','E3'):
 aggregate[p]={}
 for k in ('original','weighted'):
  assert tot[p][k].tolist()==su['overall'][p]['pooled_all'][k]['confusion_fourclass']
  aggregate[p][k]={'f1':float(f1(tot[p][k])),'brier_loss':float(np.mean(loss[p][k]))}
out=run.parent/'d37_independent_results.json'
out.write_text(json.dumps({'roots':len(results),'confusion_score_checks':checks,'probability_rows_per_model':npred,
 'independent_no_refit':True,'aggregate':aggregate,'details':results},indent=2))
print(out);print('PASS',len(results),'roots',checks,'confusion checks',npred,'probability rows/model');print(json.dumps(aggregate))
