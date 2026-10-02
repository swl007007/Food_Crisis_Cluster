"""D36: read-only C/E3 decomposition, all keys <=2020, no fitting or threshold selection."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

base=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=True);assert not list(out.iterdir()), 'refuse nonempty output'
d34=base/'geoxgb-d34-e1-brier-20261002';d35=base/'geoxgb-d35-global-increment-20261002'
s35=json.loads((d35/'summary.json').read_text())
models=['root','global20','hard_local','brier_local']; labels=['1','2','3','4或5']
def read(p):return pd.read_csv(p,float_precision='round_trip',keep_default_na=False)
def metric(t,p):
 t=np.asarray(t,dtype=bool);p=np.asarray(p,dtype=bool)
 tp=int((t&p).sum());fp=int((~t&p).sum());fn=int((t&~p).sum());tn=int((~t&~p).sum())
 return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0}
frames=[]
for h in (4,8,12):
 snap=pd.read_parquet(d34/'prepared'/f'snapshot_h{h}.parquet',
      columns=['area','target_month','country','class_code','hist_phase_o00'],
      filters=[('target_month','<=',2020*12+11)])
 assert not snap.duplicated(['area','target_month']).any()
 snap['target_month']=[f'{int(m)//12:04d}-{int(m)%12+1:02d}' for m in snap.target_month]
 for name,meta in s35['per_root'].items():
  if meta['horizon']!=h:continue
  for part in ('C','E3'):
   f=read(d35/name/f'rows_{part}.csv.gz')
   assert f.target_month.max()<='2020-12'
   f=f.merge(snap,on=['area','target_month'],how='left',validate='one_to_one',indicator=True)
   assert (f['_merge']=='both').all() and np.array_equal(f.truth,f.class_code)
   f=f.drop(columns=['_merge','class_code']);f['part']=part;f['root_name']=name;f['fold_target']=meta['target_month']
   for model in ('root','global20'):
    f['prob_'+model]=f['p_'+model+'_3']+f['p_'+model+'_4或5']
   for model,cand in [('hard_local',meta['hard']),('brier_local',meta['brier'])]:
    fn='confirmation_predictions.csv.gz' if part=='C' else 'target_predictions.csv'
    old=read(d34/'stage1_e1pair/candidates'/cand/fn)
    if part=='E3':old=old.rename(columns={'FEWSNET_admin_code':'area'});old['target_month']=meta['target_month']
    prefix='p_final_' if part=='C' else 'p_partitioned_'
    old['prob_'+model]=old[prefix+'3']+old[prefix+'4或5']
    pred='y_final' if part=='C' else 'y_pred_partitioned_code'
    old=old[['area','target_month',pred,'prob_'+model]].rename(columns={pred:'saved_pred'})
    f=f.merge(old,on=['area','target_month'],how='left',validate='one_to_one')
    assert np.array_equal(f['y_'+model],f.saved_pred);f=f.drop(columns=['saved_pred'])
   valid=f.hist_phase_o00.notna();f['persistence_code']=f.hist_phase_o00-1
   prior=f.hist_phase_o00>=3;truth=f.truth>=2
   f['transition']=np.select([~valid,~prior&~truth,~prior&truth,prior&~truth,prior&truth],
                            ['missing_origin','00_stable_noncrisis','01_onset','10_recovery','11_stable_crisis'],default='missing_origin')
   for m in models:
    assert f['prob_'+m].between(0,1.000001).all()
    f['loss_'+m]=(np.clip(f['prob_'+m],0,1)-truth)**2
   frames.append(f)
allrows=pd.concat(frames,ignore_index=True)
assert not allrows.duplicated(['root_name','part','area','target_month']).any()

# Existing 15-fold exact-origin persistence ledgers cross-check, never final-period ledger.
ledger=pd.read_csv(d34/'prepared/ledgers/dev_baselines.csv')
ledger['target_month']=[f'{int(m)//12:04d}-{int(m)%12+1:02d}' for m in ledger.target_month]
e3=allrows[allrows.part=='E3']
check=e3.merge(ledger[['area','target_month','horizon','persistence_code']],on=['area','target_month','horizon'],
               how='inner',validate='one_to_one',suffixes=('','_ledger'))
assert len(check)==81321
assert np.allclose(check.persistence_code,check.persistence_code_ledger,equal_nan=True)

rows=[];matched=[]
groups=[('all',[]),('horizon',['horizon']),('target',['fold_target']),('country',['country']),
        ('transition',['transition']),('horizon_transition',['horizon','transition'])]
for part,partrows in allrows.groupby('part',sort=True):
 for grouping,keys in groups:
  iterable=[((),partrows)] if not keys else partrows.groupby(keys,sort=True,dropna=False)
  for values,f in iterable:
   values=values if isinstance(values,tuple) else (values,)
   tags={k:v for k,v in zip(keys,values)};truth=(f.truth>=2).to_numpy();r=(f.y_root>=2).to_numpy();b=(f.y_brier_local>=2).to_numpy()
   row={'part':part,'grouping':grouping,**tags,'n':len(f),'crisis_n':int(truth.sum()),
        'persistence_known_n':int(f.hist_phase_o00.notna().sum()),
        'corrected':int(((b==truth)&(r!=truth)).sum()),'spoiled':int(((b!=truth)&(r==truth)).sum()),
        'new_tp':int((truth&~r&b).sum()),'lost_tp':int((truth&r&~b).sum()),
        'new_fp':int((~truth&~r&b).sum()),'removed_fp':int((~truth&r&~b).sum())}
   for m in models:
    met=metric(truth,f['y_'+m]>=2)
    row.update({m+'_'+k:v for k,v in met.items()});row[m+'_brier_loss']=float(f['loss_'+m].mean())
   rows.append(row)
  if grouping=='all':
   ref=s35['overall'][part]['all']['pooled']
   for m in models:assert abs(rows[-1][m+'_f1']-ref[m]['crisis_f1'])<1e-14
 f=partrows[partrows.hist_phase_o00.notna()];t=f.truth>=2
 mrow={'part':part,'n':len(f),'total_n':len(partrows),'coverage':len(f)/len(partrows)}
 for m in models+['persistence']:
  pred=f.persistence_code>=2 if m=='persistence' else f['y_'+m]>=2
  mrow.update({m+'_'+k:v for k,v in metric(t,pred).items()})
 matched.append(mrow)
table=pd.DataFrame(rows)
for part in ('C','E3'):
 total=table[(table.part==part)&(table.grouping=='all')].iloc[0]
 for grouping in ('transition','country','horizon','target'):
  chunk=table[(table.part==part)&(table.grouping==grouping)]
  for k in ['n','crisis_n','corrected','spoiled','new_tp','lost_tp','new_fp','removed_fp']:
   assert chunk[k].sum()==total[k],(part,grouping,k)
table.to_csv(out/'decomposition.csv',index=False)
pd.DataFrame(matched).to_csv(out/'persistence_matched.csv',index=False)
audit={'rows':len(allrows),'unique_fold_part_keys':True,'ledger_matched_rows':len(check),
       'all_strata_counts_reconcile':True,'D35_pooled_F1_reproduced':True,
       'all':[r for r in rows if r['grouping']=='all'],'persistence_matched':matched,
       'scope':'C and E3 <=2020; transition groups are retrospective; no new fit or selection'}
(out/'summary.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
