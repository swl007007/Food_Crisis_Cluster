"""Independent D48 raw-model replay, eligible FIT lineage and known-only metrics; zero fits."""
import hashlib, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
B=Path(r"C:\Users\swl00\geoxgb_runs")
D=B/"geoxgb-d34-e1-brier-20261002"
S=D/"stage1_e1pair"
R=B/"geoxgb-d48-known-origin-fit-20261002"
P=Path(r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\2.source_code\Step5_Geo_RF_trial\Food_Crisis_Cluster\FEWSNETGeoXGBExperiment")
OUT=B/"d48_supervisor_results.json"
assert not OUT.exists()
assert (platform.python_version(),np.__version__,pd.__version__,xgb.__version__)==("3.12.10","2.2.6","2.2.3","3.0.0")
DATES=("2018-06","2018-10","2019-02","2019-06","2019-10","2020-02","2020-06")
GS={4:"G1",8:"G4",12:"G2"}
LABELS=("1","2","3","4或5")
ARMS=("original","known")
FEATURES=json.loads((P/"feature-schema.json").read_text())["ordered_features"]
summary=json.loads((R/"summary.json").read_text())
identity=json.loads((R/"identity.json").read_text())
checks=0
cells=0
def check(ok,where):
    global checks
    checks+=1
    assert ok,where
def mi(t):return int(t[:4])*12+int(t[5:])-1
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return pd.read_csv(p,float_precision="round_trip")
def near(a,b,where):check(np.allclose(a,b,rtol=0,atol=1e-12),where)
def score(y,p,loss=True):
    q=p.argmax(1);z=y>=2;a=q>=2
    mat=np.bincount(y*4+q,minlength=16).reshape(4,4)
    tp,fp,fn=int((z&a).sum()),int((~z&a).sum()),int((z&~a).sum())
    den=mat.sum(1)+mat.sum(0)
    macro=np.divide(2*np.diag(mat),den,out=np.zeros(4),where=den!=0).mean()
    result=dict(n=len(y),crisis=dict(tp=tp,fp=fp,fn=fn),confusion_fourclass=mat.tolist(),crisis_f1=2*tp/(2*tp+fp+fn) if tp+fp+fn else 0.,macro_f1_fourclass=float(macro),crisis_brier=float(np.mean((p[:,2:].astype(float).sum(1)-z)**2)))
    if loss:
        truep=p[np.arange(len(y)),y].astype(float)
        check(np.all(truep>0),"positive true-class probabilities")
        result["logloss_fourclass"]=float(-np.log(truep).mean())
    result["crisis_call_share"]=float(a.mean())
    return result
def compare_score(m,s,where):
    global cells
    for key in ("crisis_f1","macro_f1_fourclass","crisis_brier","logloss_fourclass","crisis_call_share"):
        if key in m:near(m[key],s[key],(where,key))
    if "confusion_fourclass" in s:check(m["confusion_fourclass"]==s["confusion_fourclass"],(where,"confusion"))
    if "crisis" in s:
        for k in ("tp","fp","fn"):check(m["crisis"][k]==s["crisis"][k],(where,k))
    cells+=1
pairs={};totals={};nrows=0
expected={f"h{h}_{t}_{g}_r80_s42_e1pair" for h,g in GS.items() for t in DATES}
check(set(summary["per_root"])==expected,"21 expected roots")
for h,g in GS.items():
    snap=pd.read_parquet(D/"prepared"/f"snapshot_h{h}.parquet",columns=["area","target_month","class_code"]+FEATURES,filters=[("target_month","<=",mi("2020-12"))]).set_index(["area","target_month"])
    check(snap.index.is_unique,(h,"snapshot unique"))
    pool={part:[] for part in ("FIT","C","E3")}
    for t in DATES:
        name=f"h{h}_{t}_{g}_r80_s42_e1pair";rd=R/name;root=S/"roots"/name
        meta=json.loads((rd/"known_root.json").read_text());oldmeta=json.loads((root/"root.json").read_text())
        mem=read(root/"fold_membership.csv.gz");mem["month"]=mem.target_month.map(mi)
        check(not mem.duplicated(["area","month"]).any(),(name,"unique membership"))
        fm=mem[mem.role=="fitting"]
        keysha=hashlib.sha256(fm[["area","month"]].to_numpy(np.int64).tobytes()).hexdigest()
        check(keysha==meta["fitting_keys_sha256"]==oldmeta["fitting_keys_sha256"],(name,"fit keys"))
        expected_params=dict(oldmeta["root_fit"]["params"])
        check(meta["fit_record"]["params"]==expected_params,(name,"G config unchanged"))
        check(sha(D/"prepared"/f"snapshot_h{h}.parquet")==identity["inputs"][name]["snapshot"],(name,"snapshot identity"))
        fdata=snap.loc[list(zip(fm.area,fm.month))];fk=np.isfinite(fdata.hist_phase_o00.to_numpy(float));eligible=fm[fk]
        em=meta["eligibility"]
        keys=eligible[["area","month"]].sort_values(["area","month"]).to_numpy(np.int64)
        check(hashlib.sha256(keys.tobytes()).hexdigest()==em["eligible_keys_sha256_sorted"],(name,"eligible key hash"))
        check(len(eligible)==em["known"]==meta["fit_record"]["rows"] and len(fm)==em["total"],(name,"eligible rows"))
        near(em["fraction"],len(eligible)/len(fm),(name,"eligible fraction"))
        check(eligible.month.nunique()==em["label_dates"] and eligible.area.nunique()==em["areas"],(name,"eligible support"))
        check(em["class_counts"]=={lab:int((eligible.class_code==k).sum()) for k,lab in enumerate(LABELS)},(name,"eligible classes"))
        check("sample_weight" not in meta["fit_record"] and "base_margin" not in meta["fit_record"],(name,"unweighted no margin"))
        models={}
        for arm,path in (("original",S/"checkpoints"/f"h{h}_{t}_{g}_L1_r80_s42_e1brier_gt0"/"xgb_root.ubj"),("known",rd/"known_root.ubj")):
            check(sha(path)==meta[f"{arm}_root_sha256"],(name,arm,"hash"))
            model=xgb.Booster();model.load_model(path);models[arm]=model
            config=json.loads(model.save_config())["learner"]["learner_model_param"]
            check(float(config["base_score"])==.5 and int(config["num_class"])==4,(name,arm,"base class"))
            check(model.num_boosted_rounds()==(200 if h==4 else 400),(name,arm,"rounds"))
        out={}
        for role,part in (("fitting","FIT"),("confirmation","C"),("heldout_target","E3")):
            full=mem[mem.role==role];data=snap.loc[list(zip(full.area,full.month))];known=np.isfinite(data.hist_phase_o00.to_numpy(float));keys=full[known];data=data[known];y=data.class_code.to_numpy(int)
            cv=summary["per_root"][name]["scores"][part]["coverage"]
            check(cv["known"]==len(keys) and cv["total"]==len(full) and cv["excluded_missing_origin"]==int((~known).sum()),(name,part,"coverage"))
            near(cv["coverage"],known.mean(),(name,part,"coverage fraction"))
            check(np.array_equal(y,keys.class_code),(name,part,"snapshot truth"))
            X=data[FEATURES].to_numpy(float);X[np.isinf(X)]=np.nan
            preds={a:m.predict(xgb.DMatrix(X,missing=np.nan,nthread=4)).astype(float) for a,m in models.items()}
            rows=read(rd/f"rows_{part}_known.csv.gz");nrows+=len(rows)
            check(np.array_equal(rows.area,keys.area) and np.array_equal(rows.target_month,keys.target_month) and np.array_equal(rows.truth,y),(name,part,"persisted keys truth"))
            phase=data.hist_phase_o00.to_numpy(float);mask=np.isfinite(phase);per=np.minimum(phase[mask],4).astype(int)-1
            check(np.array_equal(rows.persistence_code.notna(),mask) and np.array_equal(rows.persistence_code[mask],per),(name,part,"persistence"))
            scores=summary["per_root"][name]["scores"][part];out[part]={}
            for a,p in preds.items():
                check(np.array_equal(rows[[f"p_{a}_{l}" for l in LABELS]].to_numpy(float),p),(name,part,a,"raw replay exact"))
                check(np.array_equal(rows[f"y_{a}"],p.argmax(1)),(name,part,a,"argmax"))
                m=score(y,p);compare_score(m,scores[a],(name,part,a,"all"))
                z=y>=2;s=p[:,2:].sum(1)/p.sum(1)
                rank={"auc":roc_auc_score(z,s),"ap":average_precision_score(z,s)}
                for k,v in rank.items():near(v,scores["ranking"][a][k],(name,part,a,k))
                out[part][a]={**m,**rank}
            compare_score(score(y[mask],np.eye(4)[per],False),scores["persistence"],(name,part,"persistence"))
            pool[part].append((y,preds,mask,per))
        pairs[name]=out
        print(name,"raw replay/eligible FIT/metrics PASS",flush=True)
    totals[h]={}
    for part,items in pool.items():
        target=summary["by_horizon"][f"H{h}"][part]
        y=np.concatenate([v[0] for v in items]);mask=np.concatenate([v[2] for v in items]);per=np.concatenate([v[3] for v in items]);out={}
        for a in ARMS:
            p=np.concatenate([v[1][a] for v in items]);m=score(y,p);mm=score(y[mask],p[mask])
            compare_score(m,target["pooled"][a],(h,part,a,"pooled"))
            fold=[pairs[n][part][a] for n in expected if n.startswith(f"h{h}_")]
            for k in ("crisis_f1","macro_f1_fourclass","crisis_brier","logloss_fourclass","crisis_call_share"):near(np.mean([v[k] for v in fold]),target["mean_fold"][a][k],(h,part,a,k,"mean-fold"))
            for k in ("auc","ap"):near(np.mean([v[k] for v in fold]),target["ranking_mean_fold"][a][k],(h,part,a,k,"mean-fold"))
            out[a]=m
        pm=score(y[mask],np.eye(4)[per],False);compare_score(pm,target["pooled"]["persistence"],(h,part,"pooled persistence"));out["persistence"]=pm;totals[h][part]=out
result=dict(passed=True,checks=checks,metric_cells=cells,replayed_rows=nrows,pairs=pairs,by_horizon=totals,summary_sha256=sha(R/"summary.json"),identity_sha256=sha(R/"identity.json"),script_sha256=sha(Path(__file__)),method="No producer imports or fits. Native raw UBJ replay from keyed frozen snapshots; independent eligible FIT subset, coverage and numpy/sklearn metrics; all scores on origin-known keys only.")
OUT.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps({k:result[k] for k in ("passed","checks","metric_cells","replayed_rows")}))
