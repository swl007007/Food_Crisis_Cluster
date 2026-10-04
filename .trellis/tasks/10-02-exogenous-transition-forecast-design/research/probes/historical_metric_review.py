"""Independent saved-row recount; no experiment imports, no fits, no 2025 reads."""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
R = Path(r'C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1')
problems=[]
def check(ok, label):
    if not ok: problems.append(label)
def eq(a,b,label):
    if a is None or pd.isna(a): check(b is None or pd.isna(b),label); return
    check(b is not None and not pd.isna(b) and bool(np.isclose(a,b,rtol=0,atol=1e-12)),label)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return pd.read_csv(p,float_precision='round_trip')
def month(s):
    y,m=map(int,s.split('-'));return 12*y+m-1
def counts(f,col):
    y=f.truth_code.to_numpy()>=2;p=f[col].to_numpy()>=2
    return np.array([(y&p).sum(),((~y)&p).sum(),(y&(~p)).sum(),((~y)&(~p)).sum()],dtype=np.int64)
def f1(c):
    den=2*c[...,0]+c[...,1]+c[...,2]
    return np.divide(2*c[...,0],den,out=np.full(np.shape(den),np.nan),where=den!=0)
def cohort(f):
    lab=f[f.truth_code.notna()]; risk=lab[lab.origin_truth.notna()&(lab.origin_truth<2)]
    return lab,risk,int(lab.origin_truth.isna().sum()),int((lab.origin_truth>=2).sum())
def bootstrap(f,col,saved,label):
    countries=sorted(f.country.astype(str).unique());C=len(countries)
    a=np.array([counts(f[f.country==c],'y_pred_code')[:3] for c in countries]).reshape(C,3)
    b=np.array([counts(f[f.country==c],col)[:3] for c in countries]).reshape(C,3)
    fa,fb=float(f1(a.sum(0))),float(f1(b.sum(0)))
    expected={'n':len(f),'countries':C,'model_f1':fa,'comparator_f1':fb,'delta':fa-fb,'crisis_events':int((f.truth_code>=2).sum()),'draws':2000,'seed':42}
    ev=a[:,0]+a[:,2];expected['max_country_event_share']=float(ev.max()/ev.sum()) if ev.sum() else None
    if C:
        # Sample whole blocks by index, then sum their confusion counts, independently of producer multiplicity code.
        draws=np.random.default_rng(42).integers(C,size=(2000,C))
        delta=f1(a[draws].sum(1))-f1(b[draws].sum(1));valid=np.isfinite(delta)
        expected.update(valid_draws=int(valid.sum()),undefined_draws=int((~valid).sum()))
        reason='point F1 undefined' if not np.isfinite(fa-fb) else 'fewer than two country blocks' if C<2 else 'undefined bootstrap draws' if not valid.all() else ''
        lo,hi=np.quantile(delta,[.025,.975],method='linear') if not reason else (None,None)
    else:
        reason='no eligible rows';lo=hi=None;expected.update(valid_draws=0,undefined_draws=0)
    expected.update(ci95_low=lo,ci95_high=hi)
    for k,v in expected.items():eq(v,saved[k],label+':'+k)
    check(saved['ci_reason']==reason,label+':ci_reason')
report=json.loads((R/'scenario_report/report.json').read_text())
hist=json.loads((R/'scenario_historical/historical.json').read_text())
frozen=R/'scenario_final/frozen.json'
check(hist['frozen_sha256']==sha(frozen),'frozen binding')
check(report['historical_sha256']==sha(R/'scenario_historical/historical.json'),'report binding')
for base,rec in [('scenario_historical',hist),('scenario_report',report)]:
    for path,digest in rec['outputs'].items():check(sha(R/base/path)==digest,base+':'+path)
obs=read(R/'prepared/ledgers/observations.csv').set_index(['area','month'])['class_code']
check(obs.index.is_unique,'unique observation keys')
summary=[];total=0;country_rows=0
for h,targets in [(4,['2021-10']+[f'{y}-{m:02}' for y in range(2022,2025) for m in (2,6,10)]),(8,[f'{y}-{m:02}' for y in range(2022,2025) for m in (2,6,10)])]:
    check(hist['calendar'][str(h)]['targets']==targets,f'H{h}calendar')
    for k in (0,1,2):
        tag=f'H{h}k{k}'; frames=[]; pooled=[]
        for t in targets:
            base=R/f'scenario_historical/A/h{h}/k{k}/{t}'
            frames.append(read(base/'predictions.csv.gz').rename(columns={'y_true_code':'truth_code'}))
            pooled.append(read(base/'pooled_predictions.csv.gz')[['area','target_month','y_pred_code']].rename(columns={'y_pred_code':'y_pred_pooled'}))
        raw=pd.concat(frames,ignore_index=True); df=read(R/f'scenario_report/keyed_h{h}_k{k}.csv.gz')
        cols=list(raw.columns)
        try: pd.testing.assert_frame_equal(df[cols],raw,check_dtype=False,check_exact=True)
        except AssertionError as e:problems.append(tag+': forecast/report row equality '+str(e)[:200])
        check(not df.duplicated(['area','target_month']).any(),tag+': unique keys')
        pool=pd.concat(pooled,ignore_index=True)
        check(df[['area','target_month']].equals(pool[['area','target_month']]),tag+': pooled keys')
        check(np.array_equal(df.y_pred_pooled,pool.y_pred_pooled),tag+': pooled predictions')
        for col,ms in [('truth_code',df.target_month.map(month)),('origin_truth',df.target_month.map(month)-h)]:
            actual=obs.reindex(pd.MultiIndex.from_arrays([df.area,ms])).to_numpy()
            check(np.array_equal(actual,df[col],equal_nan=True),tag+': independent '+col)
        check(np.array_equal(df.y_pred_code,df[['p_1','p_2','p_3','p_4或5']].to_numpy().argmax(1)),tag+': argmax rule')
        check(df.expert_class_code.isna().all() and df.expert_reason.eq('no_documented_expert_table').all(),tag+': unavailable expert')
        lab,risk,miss,crisis=cohort(df);total+=len(df)
        for study,rows in [('study1',lab),('study2',risk)]:
            e=next(x for x in report['comparisons'] if (x['horizon'],x['scenario_k'],x['study'])==(h,k,study));label=tag+study
            for field,val in [('cohort_keys',len(df)),('keys',len(rows)),('matched',int(rows.persistence_class_code.notna().sum()))]:eq(val,e[field],label+':'+field)
            c=counts(rows,'y_pred_code');s=e['model_standalone']
            for field,val in zip(['tp','fp','fn','tn'],c):eq(val,s[field],label+':'+field)
            eq(len(rows),s['n'],label+':n');eq(float(f1(c)),s['f1'],label+':f1')
            eq(c[0]/(c[0]+c[1]) if c[0]+c[1] else None,s['precision'],label+':precision')
            eq(c[0]/(c[0]+c[2]) if c[0]+c[2] else None,s['recall'],label+':recall')
            for col,name in [('persistence_class_code','vs_persistence'),('y_pred_pooled','vs_pooled_same_input'),('expert_class_code','vs_expert')]:bootstrap(rows[rows[col].notna()],col,e[name],label+name)
            check(e['expert_coverage']=={'no_documented_expert_table':len(rows)},label+':expert coverage')
            if study=='study2':
                eq(miss,e['excluded_missing_origin'],label+':missing origin');eq(crisis,e['excluded_origin_crisis'],label+':crisis origin')
                for col,name in [('y_pred_code','onset_model'),('persistence_class_code','onset_persistence')]:
                    onset=rows[(rows.truth_code>=2)&rows[col].notna()];eq(len(onset),e[name]['onsets'],label+name+'count');eq(float((onset[col]>=2).mean()) if len(onset) else None,e[name]['recall'],label+name+'recall')
            summary.append({'h':h,'k':k,'study':study,'n':len(rows),'matched':e['matched'],'model':e['vs_persistence']['model_f1'],'persistence':e['vs_persistence']['comparator_f1'],'delta':e['vs_persistence']['delta'],'ci':[e['vs_persistence']['ci95_low'],e['vs_persistence']['ci95_high']],'pooled_delta':e['vs_pooled_same_input']['delta']})
        country=read(R/f'scenario_report/country_h{h}_k{k}.csv').set_index('country');country_rows+=len(country)
        check(set(country.index)==set(df.country),tag+': all countries')
        for code,g in df.groupby('country'):
            row=country.loc[code];cl,cr,cm,cc=cohort(g);on=cr[cr.truth_code>=2]
            expected={'cohort_keys':len(g),'labelled_keys':len(cl),'regions':g.area.nunique(),'target_months':g.target_month.nunique(),'crisis_events':int((cl.truth_code>=2).sum()),'model_f1_labelled':float(f1(counts(cl,'y_pred_code'))),'study2_eligible_keys':len(cr),'study2_excluded_missing_origin':cm,'study2_excluded_origin_crisis':cc,'onsets':len(on),'onset_recall_model':float((on.y_pred_code>=2).mean()) if len(on) else None}
            for name,col in [('persistence','persistence_class_code'),('pooled','y_pred_pooled'),('expert','expert_class_code')]:
                mat=cl[cl[col].notna()];fm=float(f1(counts(mat,'y_pred_code')));fc=float(f1(counts(mat,col)))
                expected.update({name+'_matched_keys':len(mat),'model_f1_vs_'+name:fm,name+'_f1':fc,'delta_vs_'+name:fm-fc})
            for field,val in expected.items():eq(val,row[field],tag+code+field)
check(len(report['comparisons'])==12,'12 study comparisons')
out={'pass':not problems,'problems':problems,'forecast_rows':total,'country_rows':country_rows,'bootstrap_comparisons':36,'historical_sha256':sha(R/'scenario_historical/historical.json'),'report_sha256':sha(R/'scenario_report/report.json'),'probe_sha256':sha(Path(__file__)),'comparisons':summary,'scope':'Independent saved forecasts/report equality, observation truth and exact origin truth, argmax, matched counts/metrics, 36 bootstrap comparisons and every country numeric field; not model fitting lineage, runtime acceptance or 2025 evaluation.'}
p=R.with_name(R.name+'.historical_metric_review.json')
with p.open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
print(json.dumps(out,indent=2,allow_nan=False));raise SystemExit(0 if out['pass'] else 1)
