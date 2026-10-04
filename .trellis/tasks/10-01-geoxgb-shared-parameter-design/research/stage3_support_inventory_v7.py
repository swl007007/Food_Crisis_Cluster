#!/usr/bin/env python3
"""Read v7 actual fitting keys only; diagnostic counts, no estimator execution."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
R=Path('/mnt/c/users/swl00/ifpri dropbox/weilun shi/google fund/analysis/2.source_code/step5_geo_rf_trial/food_crisis_cluster/FEWSNETFourClassBaseline/runs/fourclass-v7-20260928')
OUT=Path('/tmp/stage3_support_inventory_v7');OUT.mkdir(exist_ok=True)
def month(i):return f'{int(i)//12:04d}-{int(i)%12+1:02d}'
def mi(s):y,m=map(int,s.split('-'));return y*12+m-1
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
obsfile=R/'prepared/ledgers/observations.csv';mapfile=R/'stage2/experiment/knn_sparsification_results/cluster_mapping_k40_nc13_general.csv'
obs=pd.read_csv(obsfile);cmap=pd.read_csv(mapfile)
assert not obs.duplicated(['area','month']).any()
assert not cmap.FEWSNET_admin_code.duplicated().any()
assert sorted(cmap.cluster_id.unique())==list(range(13))
assert obs.groupby('area').country.nunique().max()==1
cluster_of=cmap.set_index('FEWSNET_admin_code').cluster_id
areas=np.array(sorted(set(obs.area)|set(cmap.FEWSNET_admin_code)))
country_of=obs.drop_duplicates('area').set_index('area').country
ledger=obs[['area','month','class_code','country']].rename(columns={'month':'target_month','class_code':'ledger_class_code'})
sources=[{'path':str(p),'sha256':digest(p)} for p in (obsfile,mapfile)]
folds=[];supports=[];area_support=[];month_support=[];routes=[];pred_routes=[]
checks={'ledger_unique_area_month':True,'mapping_unique_area':True,'country_constant_per_area':True,'fitted_folds_checked':0,'key_rows_checked':0,'local_support_rows_checked':0,'skipped_without_training_keys':0}

def add_support(df,h,target,origin,win,lo,hi,local):
 base={'horizon':h,'fold_target_month':target,'origin_month':month(origin),'window':win,'window_lo_inclusive':month(lo),'window_hi_exclusive':month(hi)}
 for cid in [-2,-1]+list(range(13)):
  g=df if cid==-2 else df[df.cluster_id==cid]
  counts=g.class_code.value_counts().reindex(range(4),fill_value=0)
  c4=g[g.class_code==3]
  supports.append({**base,'cluster_id':cid,'pool_kind':'pooled' if cid==-2 else 'unmapped' if cid==-1 else 'local','n':len(g),'n_areas':g.area.nunique(),'n_countries':g.country.nunique(),'n_target_months':g.target_month.nunique(),'n_country_months':len(g[['country','target_month']].drop_duplicates()),'n_classes':int((counts>0).sum()),**{f'class{k+1}':int(counts[k]) for k in range(4)},'class4_n_areas':c4.area.nunique(),'class4_n_countries':c4.country.nunique(),'class4_n_months':c4.target_month.nunique(),'countries':'|'.join(sorted(g.country.unique())),'target_months':'|'.join(map(month,sorted(g.target_month.unique()))),'current_route':str(local.loc[cid,'route']) if win=='full' and cid in local.index else '', 'current_reason':str(local.loc[cid,'reason']) if win=='full' and cid in local.index else ''})
  for mon,sub in g.groupby('target_month'):
   cc=sub.class_code.value_counts().reindex(range(4),fill_value=0)
   month_support.append({**base,'cluster_id':cid,'target_month':month(mon),'n':len(sub),'n_areas':sub.area.nunique(),'n_countries':sub.country.nunique(),**{f'class{k+1}':int(cc[k]) for k in range(4)}})
 a=df.groupby('area').agg(n=('class_code','size'),n_months=('target_month','nunique')).reindex(areas,fill_value=0).reset_index()
 a['cluster_id']=a.area.map(cluster_of).fillna(-1).astype(int);a['country']=a.area.map(country_of).fillna('')
 for k,v in base.items():a[k]=v
 area_support.append(a)

for h in (4,8,12):
 mpath=R/f'stage3/h{h}/run_manifest.json';manifest=json.loads(mpath.read_text());sources.append({'path':str(mpath),'sha256':digest(mpath)})
 for f in manifest['folds']:
  target=f['target_month'];fd=R/f'stage3/h{h}/folds/{target}';fp=fd/'fold.json';record=json.loads(fp.read_text());o=mi(record['origin_month']);lo=o-35
  assert record['status']==f['status']
  folds.append({'horizon':h,'target_month':target,'origin_month':month(o),'status':record['status'],'history_lo':month(lo),'history_hi_exclusive':month(o),'prefix_hi_exclusive':month(o-12-h),'train_n_metadata':record.get('rows',{}).get('train',0)})
  if f['status']!='fitted':
   assert not (fd/'training_keys.csv.gz').exists();checks['skipped_without_training_keys']+=1;continue
  kp=fd/'training_keys.csv.gz';lp=fd/'local_support.csv';sources.extend({'path':str(p),'sha256':digest(p)} for p in (fp,kp,lp))
  keys=pd.read_csv(kp);assert not keys.duplicated(['area','target_month']).any();assert keys.class_code.isin(range(4)).all();assert ((keys.target_month>=lo)&(keys.target_month<o)).all()
  keyhash=hashlib.sha256(np.ascontiguousarray(keys[['area','target_month']].to_numpy(dtype=np.int64)).tobytes()).hexdigest();assert keyhash==record['train_keys_sha256']
  joined=keys.merge(ledger,on=['area','target_month'],how='left',validate='one_to_one',indicator=True)
  assert joined['_merge'].eq('both').all();assert joined.class_code.eq(joined.ledger_class_code).all()
  expected=ledger[(ledger.target_month>=lo)&(ledger.target_month<o)]
  assert len(expected)==len(keys)==record['rows']['train'];assert set(map(tuple,expected[['area','target_month']].to_numpy()))==set(map(tuple,keys[['area','target_month']].to_numpy()))
  assert keys.class_code.value_counts().reindex(range(4),fill_value=0).tolist()==record['train_class_counts']
  joined['cluster_id']=joined.area.map(cluster_of).fillna(-1).astype(int)
  local=pd.read_csv(lp).fillna('').set_index('cluster_id');assert not local.index.duplicated().any()
  for cid,row in local.iterrows():
   g=joined[joined.cluster_id==cid];counts=g.class_code.value_counts().reindex(range(4),fill_value=0)
   assert len(g)==row.train_rows;assert int((counts>0).sum())==row.observed_classes
   assert counts.tolist()==[row['train_class1'],row['train_class2'],row['train_class3'],row['train_class4或5']]
   routes.append({'horizon':h,'fold_target_month':target,'cluster_id':int(cid),'route':row.route,'reason':row.reason,'test_rows':int(row.test_rows)})
  for route,n in record['partitioned_route_counts'].items():pred_routes.append({'horizon':h,'fold_target_month':target,'route':route,'n_predictions':n})
  assert sum(record['partitioned_route_counts'].values())==record['rows']['test']
  windows={'full':(lo,o),'E1':(o-12,o-6),'E2':(o-6,o),'prefix':(lo,o-12-h)}
  for win,(l,u) in windows.items():add_support(joined[(joined.target_month>=l)&(joined.target_month<u)],h,target,o,win,l,u,local)
  checks['fitted_folds_checked']+=1;checks['key_rows_checked']+=len(keys);checks['local_support_rows_checked']+=len(local)

S=pd.DataFrame(supports);A=pd.concat(area_support,ignore_index=True);F=pd.DataFrame(folds);M=pd.DataFrame(month_support);L=pd.DataFrame(routes);P=pd.DataFrame(pred_routes)
assert len(S)==checks['fitted_folds_checked']*4*15
for (_,_,_),g in S.groupby(['horizon','fold_target_month','window']):
 assert g[g.cluster_id==-2].n.iloc[0]==g[g.cluster_id!=-2].n.sum()
 assert (g[['class1','class2','class3','class4']].sum(axis=1)==g.n).all()
assert (A.n==A.n_months).all()
for (_,_,_),g in A.groupby(['horizon','fold_target_month','window']):
 assert len(g)==len(areas)
summary={'reference_run':str(R),'scope':'actual fitted folds only; skipped metadata recorded separately','definitions':{'class4':'merged phase 4 or 5, class_code=3','full':'[O-35,O)','E1':'[O-12,O-6)','E2':'[O-6,O)','prefix':'[O-35,O-12-H), strict label cutoff before earliest E1 validation origin','quantiles':'linear empirical quantiles; min,p10,p50,p90,max','area_universe':'union of observed-ledger and Stage2-mapped areas; zero support included','cluster_fold_denominator':'13 consensus clusters x actual fitted folds; zero rows included','information':'row counts are observed area-month keys, not independent sample/effective n'},'ledger':{'rows':len(obs),'areas':obs.area.nunique(),'countries':sorted(obs.country.unique()),'months':obs.month.nunique(),'class_counts':obs.class_code.value_counts().reindex(range(4),fill_value=0).tolist(),'raw_phase5_n':int((obs.raw_phase==5).sum())},'mapping':{'areas':len(cmap),'clusters':13,'sizes':cmap.cluster_id.value_counts().sort_index().to_dict()},'area_universe_n':len(areas),'checks':checks,'groups':[],'area_summary':[],'fold_inventory':[],'current_cluster_routes':L.groupby(['horizon','route','reason'],dropna=False).agg(cluster_folds=('cluster_id','size'),target_rows=('test_rows','sum')).reset_index().to_dict('records'),'current_prediction_routes':P.groupby(['horizon','route']).n_predictions.sum().reset_index().to_dict('records')}

def stats(x):return dict(zip(['min','p10','median','p90','max'],map(float,x.quantile([0,.1,.5,.9,1]))))
for (h,win,kind),g in S.groupby(['horizon','window','pool_kind']):
 if kind=='unmapped':pass
 d=len(g);flags={'n_lt50':g.n<50,'n_lt100':g.n<100,'n_classes_lt2':g.n_classes<2,'any_class_zero':(g[['class1','class2','class3','class4']]==0).any(axis=1),'class4_zero':g.class4==0,'class4_1to5':g.class4.between(1,5),'class4_1to10':g.class4.between(1,10)}
 summary['groups'].append({'horizon':int(h),'window':win,'pool_kind':kind,'denominator':d,'distribution':{c:stats(g[c]) for c in ['n','n_areas','n_countries','n_target_months','n_country_months','n_classes','class1','class2','class3','class4','class4_n_areas','class4_n_months']},'frequencies':{k:{'count':int(v.sum()),'fraction':float(v.mean())} for k,v in flags.items()},'zero_class_by_class':{f'class{k}':{'count':int((g[f'class{k}']==0).sum()),'fraction':float((g[f'class{k}']==0).mean())} for k in range(1,5)}})
for (h,win),g in A.groupby(['horizon','window']):
 positive=g[g.n>0]
 summary['area_summary'].append({'horizon':int(h),'window':win,'area_fold_denominator':len(g),'zero_n':int((g.n==0).sum()),'zero_fraction':float((g.n==0).mean()),'n_observed_labels':stats(g.n),'n_unique_months':stats(g.n_months),'positive_only_denominator':len(positive),'positive_only_n_unique_months':stats(positive.n_months)})
for h,g in F.groupby('horizon'):summary['fold_inventory'].append({'horizon':int(h),'scheduled':len(g),'fitted':int(g.status.eq('fitted').sum()),'skipped':int(g.status.ne('fitted').sum()),'fitted_target_months':g.loc[g.status=='fitted','target_month'].tolist()})
S.to_csv(OUT/'cluster_fold_window_support.csv',index=False);A.to_csv(OUT/'area_fold_window_support.csv.gz',index=False,compression='gzip');M.to_csv(OUT/'cluster_window_month_support.csv',index=False);F.to_csv(OUT/'fold_inventory.csv',index=False);L.to_csv(OUT/'current_cluster_routes.csv',index=False);P.to_csv(OUT/'current_prediction_routes.csv',index=False)
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));(OUT/'source_hashes.json').write_text(json.dumps(sources,indent=2))
print(json.dumps({'output':str(OUT),'checks':checks,'fold_inventory':summary['fold_inventory'],'full_local':[{k:d[k] for k in ('horizon','window','denominator','frequencies')} for d in summary['groups'] if d['window']=='full' and d['pool_kind']=='local'],'area_summary_full':[d for d in summary['area_summary'] if d['window']=='full'],'routes':summary['current_prediction_routes']},ensure_ascii=False,indent=2))
