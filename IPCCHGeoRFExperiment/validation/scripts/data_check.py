import sys,json,hashlib,warnings
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent/'IPCCHGeoRFExperiment'))
import prepare_data as p
r=Path(r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\2.source_code\Step5_Geo_RF_trial\Food_Crisis_Cluster\IPCCHGeoRFExperiment\runs\ipcch-v1-20260920d")
source=Path(json.loads((r/'manifest.json').read_text())['source']['csv'])
ledger=p.build_target_ledger(source)
v=ledger.valid().reset_index(drop=True)
saved=pd.read_csv(r/'data/target_ledger_valid.csv.gz',keep_default_na=False,dtype=str)
assert len(v)==42695 and v.ipcch_food_crisis.sum()==15206
assert np.array_equal(v.admin_code,saved.admin_code.astype(int))
assert np.array_equal(v.ipcch_food_crisis.astype(int),saved.ipcch_food_crisis.astype(int))
for c in ['target_invalid_reason','p5_missing_filled','phase_sum_S_str','normalized_p3plus_str',*p.NORMALIZED_PHASE_COLUMNS]:
 assert np.array_equal(v[c].astype(str),saved[c].astype(str)),c
print('PASS full source SHA/target rebuild 42695 valid,15206 positive',flush=True)
m=pd.read_csv(r/'data/feature_metadata.csv.gz',keep_default_na=False);x=np.load(r/'data/feature_values.npy');schema=json.loads((r/'data/feature_schema.json').read_text())['feature_columns'];assert schema==list(p.FEATURE_COLUMNS)
assert x.shape==(170780,93)
def ords(s):
 d=pd.to_datetime(s);return d.dt.year.to_numpy()*12+d.dt.month.to_numpy()-1
t=ords(m.target_month);o=ords(m.origin_month);h=x[:,-1].astype(int);a=m.admin_code.to_numpy();assert np.array_equal(t-o,h)
# Independent keyed raw lookup and derivatives, all saved rows.
raw=pd.read_csv(source,usecols=['admin_code','year','month',*p.RAW_FEATURE_COLUMNS],low_memory=False)
raw['ord']=raw.year*12+raw.month-1
raw=raw.set_index(['admin_code','ord'])
def get(cols,shift=0):
 vals=raw.reindex(pd.MultiIndex.from_arrays([a,o-shift]))[cols].to_numpy(float);vals[np.isinf(vals)]=np.nan;return vals
np.testing.assert_allclose(x[:,:70],get(list(p.RAW_FEATURE_COLUMNS)),equal_nan=True,rtol=1e-13)
for name,col,n in p.SUM_DERIVATIVES:
 vals=sum((get([col],k)[:,0] for k in range(n-1,-1,-1)))
 np.testing.assert_allclose(x[:,schema.index(name)],vals,equal_nan=True,rtol=1e-13)
for k in range(1,13):np.testing.assert_allclose(x[:,schema.index(f'EVI_mean_lag{k}_asof')],get(['EVI_mean'],k)[:,0],equal_nan=True,rtol=1e-13)
np.testing.assert_allclose(x[:,85],np.sin(2*np.pi*(t%12)/12),atol=1e-14);np.testing.assert_allclose(x[:,86],np.cos(2*np.pi*(t%12)/12),atol=1e-14)
# Independent per-area binary-search history, without implementation helper.
vo=ords(v.target_month);history={int(k):g.index.to_numpy() for k,g in v.groupby('admin_code')}
for area,idx in m.groupby('admin_code').groups.items():
 idx=np.asarray(idx);vi=history[area];dates=vo[vi];ys=v.ipcch_food_crisis.to_numpy(dtype=int)[vi]
 for positive,col,datecol in [(False,'last_observed_label','last_observed_label_month'),(True,'months_since_last_observed_crisis','last_observed_crisis_month')]:
  ds=dates[ys==1] if positive else dates; yy=ys[ys==1] if positive else ys;pos=np.searchsorted(ds,o[idx],side='right')-1;ok=pos>=0;expect=np.full(len(idx),np.nan); found=np.full(len(idx),'',dtype=object)
  if ok.any():
   expect[ok]=o[idx][ok]-ds[pos[ok]] if positive else yy[pos[ok]]; found[ok]=[f'{d//12:04d}-{d%12+1:02d}' for d in ds[pos[ok]]]
  np.testing.assert_equal(x[idx,schema.index(col)],expect);assert np.array_equal(m[datecol].to_numpy()[idx],found)
print('PASS all 170780 rows raw70/derived15/calendar/origin/history/recency',flush=True)
s=pd.read_csv(r/'stage1/split_outcomes.csv.gz');s.target_month=s.target_month.str[:7];pool=v[(v.year>=2014)&(v.year<=2022)];assert len(s)==len(pool)
assert set(zip(s.admin_code,s.target_month))==set(zip(pool.admin_code,pool.target_month.dt.strftime('%Y-%m')))
for _,g in s.groupby('admin_code'):
 g=g.sort_values('target_month');n=len(g);assert list(g.split_role)==(['singleton'] if n==1 else ['fit']*(n//2)+['validation']*(n-n//2))
roles=s.set_index(['admin_code','target_month']).split_role.reindex(pd.MultiIndex.from_arrays([a,m.target_month])).to_numpy();fit=roles=='fit'
def fills(mask):
 with warnings.catch_warnings():
  warnings.simplefilter('ignore'); mx=np.nanmax(x[mask],axis=0)
 return np.where(np.isnan(mx),0,np.where(mx==0,100,mx*100))
np.testing.assert_allclose(fills(fit),pd.read_csv(r/'stage1/imputer_fill_values.csv').fill_value,rtol=1e-13)
print('PASS original outcome half split and 93 Stage1 fitting-only imputer values',s.split_role.value_counts().to_dict(),flush=True)
folds=pd.read_csv(r/'stage3/folds.csv');keys=pd.read_csv(r/'stage3/fold_training_keys.csv.gz');ff=pd.read_csv(r/'stage3/fold_imputer_fill_values.csv.gz');checked=0
for f in folds.itertuples():
 oo=int(f.origin_month[:4])*12+int(f.origin_month[5:])-1; mask=(h==f.horizon_months)&(t>=oo-35)&(t<=oo)
 assert f.train_rows==mask.sum();assert f.train_window_start==f'{(oo-35)//12:04d}-{(oo-35)%12+1:02d}'
 if not f.fitted:continue
 kk=keys[keys.fold_id==f.fold_id];assert len(kk)==mask.sum();assert set(zip(kk.admin_code,kk.target_month))==set(zip(a[mask],m.target_month.to_numpy()[mask]))
 np.testing.assert_allclose(fills(mask),ff[ff.fold_id==f.fold_id].fill_value,rtol=1e-13);checked+=1
print('PASS all fold windows/training keys and 93 per-fold imputer values:',checked,flush=True)
