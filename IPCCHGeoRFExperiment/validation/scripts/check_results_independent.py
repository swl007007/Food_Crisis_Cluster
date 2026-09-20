from pathlib import Path
import pandas as pd, numpy as np, hashlib,json
root=Path(r"C:/Users/swl00/AppData/Local/Temp/ipcch-review-fwopav82")
run=Path(r"C:/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/2.source_code/Step5_Geo_RF_trial/Food_Crisis_Cluster/IPCCHGeoRFExperiment/runs/ipcch-v1-20260920d")
rep=run/'reports'
p=pd.read_csv(run/'stage3/predictions.csv.gz')
m=json.loads((rep/'report_manifest.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(run/'stage3/predictions.csv.gz')==m['source_sha256']
print('reporter hash',sha(root/'IPCCHGeoRFExperiment/report_results.py'),m['reporter_sha256'])
assert sorted(x.relative_to(rep).as_posix() for x in rep.rglob('*') if x.is_file() and x.name!='report_manifest.json')==m['files']
for f in m['files']:
 a,b=rep/f,root/'reconstructed-report'/f
 if f.endswith(('.csv','.csv.gz')):pd.testing.assert_frame_equal(pd.read_csv(a),pd.read_csv(b))
 elif f not in ['validation.json']:assert a.read_bytes()==b.read_bytes(),f
print('all 23 files accounted for; all CSV content equal; nonvalidation config/text byte identical')
cols={'partitioned_rf':'pred_partitioned_rf','pooled_rf':'pred_pooled_rf','xgb':'pred_xgb','persistence':'persistence_pred'}
counts=lambda d,c:np.array([((d.ipcch_food_crisis==1)&(d[c]==1)).sum(),((d.ipcch_food_crisis==0)&(d[c]==1)).sum(),((d.ipcch_food_crisis==1)&(d[c]==0)).sum(),((d.ipcch_food_crisis==0)&(d[c]==0)).sum()])
f1=lambda c:2*c[...,0]/(2*c[...,0]+c[...,1]+c[...,2])
allcounts={}
for period in ['main','partial_2026']:
 metrics=pd.read_csv(rep/period/'metrics.csv');deltas=pd.read_csv(rep/period/'deltas.csv')
 for row in metrics.itertuples():
  d=p[(p.horizon_months==row.horizon_months)&((p.target_month<'2026') if period=='main' else (p.target_month>='2026'))]
  if row.cohort=='E_persist':d=d[d.persistence_pred.notna()]
  cc=counts(d,cols[row.arm]);assert np.array_equal(cc,[row.tp,row.fp,row.fn,row.tn]);assert np.isclose(f1(cc),row.f1);assert len(d)==row.n_observations
  allcounts[period,row.horizon_months,row.cohort,row.arm]=f1(cc)
 for row in deltas.itertuples():assert np.isclose(row.delta_f1,allcounts[period,row.horizon_months,row.cohort,row.reference_arm]-allcounts[period,row.horizon_months,row.cohort,row.baseline_arm])
 print(period,'metrics',len(metrics),'deltas',len(deltas),'independently verified')
draws=pd.read_csv(rep/'main/bootstrap_draws.csv.gz');reps=pd.read_csv(rep/'main/bootstrap_replicates.csv.gz');summ=pd.read_csv(rep/'main/bootstrap_summary.csv')
for (h,co),dd in draws.groupby(['horizon_months','cohort']):
 d=p[(p.horizon_months==h)&(p.target_month<'2026')]
 if co=='E_persist':d=d[d.persistence_pred.notna()]
 country='country_id' if 'country_id' in d else 'country_en'
 countries=sorted(d[country].unique());mat=dd.pivot(index='draw_index',columns='country_id',values='multiplicity').reindex(columns=countries).fillna(0).to_numpy()
 rng=np.random.default_rng(42);idx=rng.integers(0,len(countries),size=(1000,len(countries)));expected=np.array([np.bincount(x,minlength=len(countries)) for x in idx]);assert np.array_equal(mat,expected)
 vals={}
 for arm,col in cols.items():
  if co=='E_all' and arm=='persistence':continue
  cc=np.array([counts(d[d[country]==c],col) for c in countries]);vals['f1__'+arm]=f1(mat@cc)
 for arm in cols:
  if arm=='partitioned_rf' or (co=='E_all' and arm=='persistence'):continue
  vals['delta_f1__partitioned_rf_minus_'+arm]=vals['f1__partitioned_rf']-vals['f1__'+arm]
 for stat,v in vals.items():
  rr=reps[(reps.horizon_months==h)&(reps.cohort==co)&(reps.statistic==stat)].sort_values('draw_index');assert np.allclose(v,rr.value,equal_nan=True)
  ss=summ[(summ.horizon_months==h)&(summ.cohort==co)&(summ.statistic==stat)].iloc[0];assert np.allclose(np.percentile(v[np.isfinite(v)],[2.5,97.5]),[ss.ci_lower,ss.ci_upper])
 print('bootstrap',h,co,'all 1000 draws,',len(vals),'statistics and CIs verified')
print('coverage',len(p),p.persistence_pred.notna().sum(),p.persistence_pred.isna().sum(),'fraction',p.persistence_pred.notna().mean())
a=p[(p.horizon_months==1)&p.persistence_pred.notna()];b=p[(p.horizon_months==12)&p.persistence_pred.notna()];j=a.merge(b,on=['admin_code','target_month']);print('horizon same source and value',len(j),(j.persistence_source_month_x==j.persistence_source_month_y).mean(),(j.persistence_pred_x==j.persistence_pred_y).mean())
print('PASS')
