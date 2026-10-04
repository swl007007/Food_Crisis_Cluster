from pathlib import Path
from fractions import Fraction
from collections import Counter,defaultdict
import csv,gzip,hashlib,json,math
r=Path('/mnt/c/Users/swl00/geoxgb_runs/scen-b43ef6a-v1'); d=r/'scenario_development'
s=json.loads((d/'selection.json').read_text()); counts={}; routes=Counter(); cohorts={}; maps={}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected={f'{a}_h{h}_k{k}_{y}-{m:02}' for a in 'AB' for h in (4,8) for k in range(3) for y in (2019,2020) for m in (2,6,10)}
assert set(s['fold_records'])==expected and s['folds']==72
actual={f'{p.parts[-5]}_{p.parts[-4]}_{p.parts[-3]}_{p.parts[-2]}' for p in d.glob('*/*/*/*/fold.json')};assert actual==expected
for rel,digest in s['outputs'].items(): assert sha(d/rel)==digest,rel
nrows=0;missing_truth=0
for ident in sorted(expected):
 a,hh,kk,t=ident.split('_');h=int(hh[1:]);k=int(kk[1:]);fdir=d/a/hh/kk/t
 f=json.loads((fdir/'fold.json').read_text());assert sha(fdir/'fold.json')==s['fold_records'][ident]
 assert (f['strategy'],f['horizon'],f['scenario_k'],f['target_month'])==(a,h,k,t)
 assert f['code']==s['code'] and f['runtime']==s['runtime']
 assert f['prepared_outputs_sha256']=='972902bbc76d626741befceb46345ed6cf3d19c2adabb873dc1576ad5d6f5161'
 for rel,digest in f['outputs'].items(): assert sha(fdir/rel)==digest,(ident,rel)
 routes[f['map_route']]+=1
 mapkey=(a,f['origin_month']); old=maps.setdefault(mapkey,f['map_id']);assert old==f['map_id']
 with gzip.open(fdir/'predictions.csv.gz','rt') as inp: rows=list(csv.DictReader(inp))
 with gzip.open(fdir/'pooled_predictions.csv.gz','rt') as inp: pooled=list(csv.DictReader(inp))
 assert len(rows)==f['rows']==len(pooled)==5718
 bykey={(q['area'],q['target_month']):q for q in pooled};assert len(bykey)==len(rows)
 assert len({(q['area'],q['target_month']) for q in rows})==len(rows)
 c={name:Counter(dict.fromkeys(('tp','fp','fn','tn'),0)) for name in ('model','model_matched','persistence')}
 def add(out,y,p):out['tp' if y and p else 'fn' if y else 'fp' if p else 'tn']+=1
 keys={}
 for q in rows:
  p=bykey[(q['area'],q['target_month'])];pred=int(q['y_pred_code']);assert 0<=pred<=3
  probs=[float(q[x]) for x in ('p_1','p_2','p_3','p_4或5')];assert all(math.isfinite(x) and 0<=x<=1 for x in probs)
  assert abs(sum(probs)-1)<1e-6 and pred==max(range(4),key=lambda j:probs[j])
  assert q['target_month']==t and int(q['horizon'])==h and int(q['scenario_k'])==k
  assert q['origin_month']==f['origin_month']
  assert q['y_true_code']==p['y_true_code']
  if q['route']!='local_model':assert all(q[x]==p[x] for x in ('y_pred_code','p_1','p_2','p_3','p_4或5'))
  keys[q['area']]=(q['y_true_code'],q['persistence_class_code'],q['persistence_source_month'])
  if not q['y_true_code']:
   missing_truth+=1;continue
  y=int(float(q['y_true_code']));assert 0<=y<=3
  add(c['model'],y>=2,pred>=2)
  if q['persistence_class_code']:
   z=int(float(q['persistence_class_code']));assert 0<=z<=3
   add(c['model_matched'],y>=2,pred>=2);add(c['persistence'],y>=2,z>=2)
 nrows+=len(rows)
 cohortkey=(h,k,t);assert cohorts.setdefault(cohortkey,keys)==keys
 for metric,cc in c.items():counts.setdefault((a,h,k,metric),Counter()).update(cc)
 if h==8 and t=='2019-02':assert f['map_route']=='no_prior_candidates'
 else:assert f['map_route']=='learned_map'
def f1(c):
 den=2*c['tp']+c['fp']+c['fn'];return Fraction(2*c['tp'],den) if den else None
summ=[]
for h in (4,8):
 eligible={}
 for a in 'AB':
  rec=s['decisions'][str(h)]['strategies'][a]
  for k in range(3):
   for metric in ('model','model_matched','persistence'):assert dict(counts[a,h,k,metric])==rec['counts'][f'k{k}'][metric],(a,h,k,metric)
  normal=f1(counts[a,h,0,'model_matched']);pers=f1(counts[a,h,0,'persistence']);delta=normal-pers
  one=f1(counts[a,h,1,'model']);two=f1(counts[a,h,2,'model']);mean=(one+two)/2
  vals={'normal_model_matched_f1':normal,'normal_persistence_f1':pers,'normal_parity':delta,'one_cycle_f1':one,'two_cycle_f1':two,'interruption_mean_f1':mean,'one_cycle_persistence_f1':f1(counts[a,h,1,'persistence']),'two_cycle_persistence_f1':f1(counts[a,h,2,'persistence'])}
  for name,v in vals.items():assert Fraction(rec[name])==v,(a,h,name)
  qualify=delta>=Fraction(-1,50);assert rec['qualifies']==qualify
  assert rec['unmet']==([] if qualify else ['normal_parity_below_-0.02'])
  if qualify:eligible[a]=mean
  summ.append({'strategy':a,'horizon':h,**{name:float(v) for name,v in vals.items()},'qualifies':qualify,'matched_interruption_deltas':{str(k):float(f1(counts[a,h,k,'model_matched'])-f1(counts[a,h,k,'persistence'])) for k in (1,2)}})
 winner=max(eligible,key=lambda a:(eligible[a],a=='A')) if eligible else None
 assert s['decisions'][str(h)]['winner']==winner
 assert s['decisions'][str(h)]['tie']==(len(eligible)==2 and eligible['A']==eligible['B'])
report={'selection_sha256':sha(d/'selection.json'),'verified_folds':72,'verified_selection_outputs':len(s['outputs']),'forecast_rows':nrows,'unlabelled_rows_retained':missing_truth,'routes':dict(routes),'metrics':summ,'verdict':'PASS bounded selection recount; not full-task audit'}
Path('/tmp/scenario_selection_review.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
