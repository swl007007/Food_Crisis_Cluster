import pathlib,json,pandas as pd,numpy as np,sys
p=pathlib.Path(sys.argv[1]); s=p/'stage3'
m=pd.read_csv(p/'data/feature_metadata.csv.gz'); schema=json.loads((p/'data/feature_schema.json').read_text()); X=np.load(p/'data/feature_values.npy'); cols=schema['feature_columns']
def ordinal(v):
 return v.str[:4].astype(int)*12+v.str[5:7].astype(int)-1
mt=ordinal(m.target_month); mo=ordinal(m.origin_month); mh=mt-mo
k=pd.read_csv(s/'fold_training_keys.csv.gz'); pred=pd.read_csv(s/'predictions.csv.gz'); f=pd.read_csv(s/'folds.csv'); fills=pd.read_csv(s/'fold_imputer_fill_values.csv.gz'); assignments=pd.read_csv(p/'stage1/area_assignments.csv').set_index('admin_code')
for row in f.itertuples():
 o=int(row.origin_month[:4])*12+int(row.origin_month[5:])-1
 assert row.train_window_months==36 and o>2022*12+11
 train=m[(mh==row.horizon_months)&(mt>=o-35)&(mt<=o)]
 test=m[(mh==row.horizon_months)&(m.target_month==row.target_month)]
 assert len(train)==row.train_rows and len(test)==row.test_rows
 if not row.fitted: assert len(test)==0; continue
 saved=k[k.fold_id==row.fold_id]; ps=pred[pred.fold_id==row.fold_id]
 key=['admin_code','target_month','origin_month','ipcch_food_crisis']
 assert set(map(tuple,train[key].values))==set(map(tuple,saved[key].values)) and len(saved)==len(train)
 assert set(map(tuple,test[key].values))==set(map(tuple,ps[key].values)) and len(ps)==len(test)
 assert (saved.partition_code.to_numpy()==assignments.loc[saved.admin_code,'partition_code'].to_numpy()).all()
 fs=fills[fills.fold_id==row.fold_id].sort_values('feature_index'); assert fs.feature_name.tolist()==cols
 values=X[train.index]; lo=np.nanmin(values,axis=0); hi=np.nanmax(values,axis=0)
 assert np.allclose(fs.fit_min,lo,equal_nan=True) and np.allclose(fs.fit_max,hi,equal_nan=True)
 assert np.allclose(fs.fill_value,np.where(np.isnan(hi),0,np.where(hi==0,100,hi*100)),equal_nan=True)
for arm in ['partitioned_rf','pooled_rf','xgb']:
 assert pred['prob_'+arm].between(0,1).all()
 assert (pred['pred_'+arm]==(pred['prob_'+arm]>.5).astype(int)).all()
fallback=pred.model_route=='pooled_rf'
assert (pred.loc[fallback,'prob_partitioned_rf']==pred.loc[fallback,'prob_pooled_rf']).all()
assert (pred.partition_code.to_numpy()==assignments.loc[pred.admin_code,'partition_code'].to_numpy()).all()
ledger=pd.read_csv(p/'data/target_ledger_valid.csv.gz'); hist={a:list(zip(g.target_month.str[:7],g.ipcch_food_crisis)) for a,g in ledger.groupby('admin_code')}
for row in pred.itertuples():
 eligible=[v for v in hist[row.admin_code] if v[0]<=row.origin_month]
 if not eligible: assert pd.isna(row.persistence_pred);continue
 month,label=max(eligible); assert row.persistence_pred==label and row.persistence_source_month==month
print(json.dumps(dict(folds=len(f),fitted=int(f.fitted.sum()),training_keys=len(k),predictions=len(pred),imputer_rows=len(fills),fallback_rows=int(fallback.sum()),unresolved_areas=int((assignments.partition_code<0).sum()),checks='all exact training/test keys; own origins; frozen assignments; train-only imputer extrema/fills; strict thresholds; pooled fallback probabilities; independent latest valid ledger persistence'),indent=2))
