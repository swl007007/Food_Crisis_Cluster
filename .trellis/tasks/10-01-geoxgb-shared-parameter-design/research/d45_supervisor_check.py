"""D45 independent saved-booster slicing replay and FIT/C/E3 metrics; no production imports/fits."""
import hashlib, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

B=Path(r"C:\Users\swl00\geoxgb_runs")
D=B/"geoxgb-d34-e1-brier-20261002"
S=D/"stage1_e1pair"
P=Path(r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\2.source_code\Step5_Geo_RF_trial\Food_Crisis_Cluster\FEWSNETGeoXGBExperiment")
OUT=B/"d45_supervisor_results.json"
assert not OUT.exists()
assert (platform.python_version(),np.__version__,pd.__version__,xgb.__version__)==("3.12.10","2.2.6","2.2.3","3.0.0")
DATES=("2018-06","2018-10","2019-02","2019-06","2019-10","2020-02","2020-06")
GS={4:"G1",8:"G4",12:"G2"}
LABELS=("1","2","3","4或5")
FEATURES=json.loads((P/"feature-schema.json").read_text())["ordered_features"]
checks=0
def check(ok,where):
    global checks
    checks+=1
    assert ok,where
def mi(t):
    return int(t[:4])*12+int(t[5:])-1
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    return pd.read_csv(p,float_precision="round_trip")
def score(y,p,months,loss=True):
    q=p.argmax(1);z=y>=2;a=q>=2
    mat=np.bincount(y*4+q,minlength=16).reshape(4,4)
    tp,fp,fn=int((z&a).sum()),int((~z&a).sum()),int((z&~a).sum())
    den=mat.sum(1)+mat.sum(0)
    macro=np.divide(2*np.diag(mat),den,out=np.zeros(4),where=den!=0).mean()
    result=dict(n=len(y),dates=sorted(set(int(m) for m in months)),class_counts=np.bincount(y,minlength=4).tolist(),
        tp=tp,fp=fp,fn=fn,confusion=mat.tolist(),crisis_f1=2*tp/(2*tp+fp+fn) if tp+fp+fn else 0.,
        macro_f1=float(macro),brier=float(np.mean((p[:,2:].astype(np.float64).sum(1)-z)**2)))
    if loss:
        truep=p[np.arange(len(y)),y].astype(np.float64)
        check(np.all(truep>0),"positive true-class probabilities")
        result["logloss"]=float(-np.log(truep).mean())
    return result
def changes(y,p,full):
    z=y>=2;a=p.argmax(1)>=2;b=full.argmax(1)>=2;change=a!=b
    return dict(changed=int(change.sum()),corrected=int((change&(a==z)).sum()),spoiled=int((change&(b==z)).sum()))

pairs={};identities={}
for h,g in GS.items():
    snap=pd.read_parquet(D/"prepared"/f"snapshot_h{h}.parquet",columns=["area","target_month","class_code"]+FEATURES,
        filters=[("target_month","<=",mi("2020-12"))]).set_index(["area","target_month"])
    check(snap.index.is_unique,("unique snapshot",h))
    for t in DATES:
        name=f"h{h}_{t}_{g}_r80_s42_e1pair";root=S/"roots"/name
        cand=f"h{h}_{t}_{g}_L1_r80_s42_e1brier_gt0"
        meta=json.loads((root/"root.json").read_text())
        mem=read(root/"fold_membership.csv.gz");mem["month"]=mem.target_month.map(mi)
        check(not mem.duplicated(["area","month"]).any(),(name,"unique membership"))
        fm=mem[mem.role=="fitting"][["area","month"]].to_numpy(dtype=np.int64)
        check(hashlib.sha256(fm.tobytes()).hexdigest()==meta["fitting_keys_sha256"],(name,"fit digest"))
        modelpath=S/"checkpoints"/cand/"xgb_root.ubj"
        check(sha(modelpath)==meta["root_booster_sha256"],(name,"model SHA"))
        model=xgb.Booster();model.load_model(modelpath)
        rounds=meta["root_fit"]["rounds_total"]
        check(rounds==(200 if h==4 else 400)==model.num_boosted_rounds(),(name,"rounds"))
        models={"quarter":model[:rounds//4],"half":model[:rounds//2],"full":model}
        inputs={};pred={}
        for role,label in (("fitting","FIT"),("confirmation","C"),("heldout_target","E3")):
            keys=mem[mem.role==role];data=snap.loc[list(zip(keys.area,keys.month))]
            y=data.class_code.to_numpy(dtype=int)
            check(np.array_equal(y,keys.class_code.to_numpy(dtype=int)),(name,label,"truth"))
            X=data[FEATURES].to_numpy(dtype=float);X[np.isinf(X)]=np.nan
            check(X.shape==(len(y),162),(name,label,"matrix shape"))
            inputs[label]=(y,keys.month.to_numpy(),data.hist_phase_o00.to_numpy(float))
            pred[label]={}
            for fraction,obj in models.items():
                p=obj.predict(xgb.DMatrix(X,missing=np.nan,nthread=4))
                check(p.shape==(len(y),4) and np.isfinite(p).all(),(name,label,fraction,"probabilities"))
                pred[label][fraction]=p
            if label=="C":
                saved=read(S/"candidates"/cand/"confirmation_predictions.csv.gz")
                saved["month"]=saved.target_month.map(mi)
                saved=saved.set_index(["area","month"]).loc[list(zip(keys.area,keys.month))]
                check(np.array_equal(pred[label]["full"],saved[["p_root_"+l for l in LABELS]].to_numpy(np.float32)),(name,"C full replay"))
            elif label=="E3":
                saved=read(root/"root_target_predictions.csv").set_index("FEWSNET_admin_code").loc[keys.area]
                check(np.array_equal(pred[label]["full"],saved[["p_pooled_"+l for l in LABELS]].to_numpy(np.float32)),(name,"E3 full replay"))
        out=dict(horizon=h,target=t,rounds={k:obj.num_boosted_rounds() for k,obj in models.items()},metrics={},changes={})
        for label in ("FIT","C","E3"):
            y,months,phase=inputs[label]
            cohorts=[(label,np.ones(len(y),dtype=bool))]
            if label=="E3":cohorts.append(("E3_matched",np.isfinite(phase)))
            for cohort,mask in cohorts:
                out["metrics"][cohort]={k:score(y[mask],p[mask],months[mask]) for k,p in pred[label].items()}
                out["changes"][cohort]={k:changes(y[mask],pred[label][k][mask],pred[label]["full"][mask]) for k in ("quarter","half")}
                if cohort=="E3_matched":
                    per=np.minimum(phase[mask],4).astype(int)-1
                    check(np.isin(per,[0,1,2,3]).all(),(name,"persistence codes"))
                    out["metrics"][cohort]["persistence"]=score(y[mask],np.eye(4)[per],months[mask],loss=False)
        pairs[name]=out
        identities[name]=dict(booster_sha256=sha(modelpath),membership_sha256=sha(root/"fold_membership.csv.gz"))
        print(name,"independent slicing/replay scored",flush=True)
result=dict(passed=True,checks=checks,issues=[],pairs=pairs,identities=identities,
    method="Native Booster slicing for quarter/half; default full; all21 saved roots, every FIT/C/E3 row scored, zero fits",script_sha256=sha(Path(__file__)))
OUT.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(dict(passed=True,checks=checks,roots=len(pairs),out=str(OUT))))
