"""Compare independently sliced D45 metrics and replay every persisted C/E3 prefix row."""
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
B=Path(r"C:\Users\swl00\geoxgb_runs")
RUN=B/"geoxgb-d45-root-prefix-20261002"
D=B/"geoxgb-d34-e1-brier-20261002"
P=Path(r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\2.source_code\Step5_Geo_RF_trial\Food_Crisis_Cluster\FEWSNETGeoXGBExperiment")
OUT=B/"d45_comparison_results.json"
assert not OUT.exists()
ref=json.loads((B/"d45_supervisor_results.json").read_text())
s=json.loads((RUN/"summary.json").read_text())
ident=json.loads((RUN/"identity.json").read_text())
checks=0
def ck(ok,where):
    global checks
    checks+=1
    assert ok,where
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def mi(t):
    return int(t[:4])*12+int(t[5:])-1
MAP={"n":"n","tp":"tp","fp":"fp","fn":"fn","crisis_f1":"crisis_f1","macro_f1":"macro_f1_fourclass","brier":"crisis_brier","logloss":"logloss_fourclass"}
def compare(a,b,where):
    for k,v in a.items():
        if k in MAP:ck(abs(v-b[MAP[k]])<1e-12,(where,k,v,b[MAP[k]]))
def pool(blocks):
    n=sum(b["n"] for b in blocks);mat=np.array([b["confusion"] for b in blocks]).sum(0)
    tp=sum(b["tp"] for b in blocks);fp=sum(b["fp"] for b in blocks);fn=sum(b["fn"] for b in blocks)
    den=mat.sum(0)+mat.sum(1)
    out=dict(n=n,tp=tp,fp=fp,fn=fn,crisis_f1=2*tp/(2*tp+fp+fn) if tp+fp+fn else 0.,
        macro_f1=float(np.divide(2*np.diag(mat),den,out=np.zeros(4),where=den!=0).mean()))
    for k in ("brier","logloss"):
        if k in blocks[0]:out[k]=sum(b[k]*b["n"] for b in blocks)/n
    return out
ck(set(ref["pairs"])==set(s["per_pair"]) and len(s["per_pair"])==21,"root set")
ck((s["models_loaded"],s["prefix_evaluations"],s["part_predictions"],s["fits"])==(21,63,189,0),"scope")
for name,r in ref["pairs"].items():
    a=s["per_pair"][name]
    ck(r["rounds"]==a["rounds"],(name,"rounds"))
    for role,rr in r["metrics"].items():
        aa=a["parts"]["E3"]["matched"] if role=="E3_matched" else a["parts"][role]
        for f,b in rr.items():
            target=aa["persistence_reference"] if f=="persistence" else aa["prefix"][f]
            compare(b,target,(name,role,f))
        for f,changes in r["changes"][role].items():
            for k,v in changes.items():ck(v==aa["decisions_vs_full"][f][k],(name,role,f,k))
            for k in ("tp","fp"):
                delta=rr[f][k]-rr["full"][k]
                ck(delta==aa["decisions_vs_full"][f][k+"_delta"],(name,role,f,k+"delta"))
        if role!="E3_matched":
            desc=aa["describe"];full=rr["full"]
            ck(desc["n"]==full["n"] and desc["label_dates"]==len(full["dates"]),(name,role,"support"))
            ck(np.allclose(desc["class_prevalence"],np.array(full["class_counts"])/full["n"],atol=0,rtol=0),(name,role,"prevalence"))
for h in (4,8,12):
    names=[n for n,r in ref["pairs"].items() if r["horizon"]==h]
    ah=s["by_horizon"][f"H{h}"]
    for role in ("FIT","C","E3","E3_matched"):
        a=ah["E3_persistence_matched" if role=="E3_matched" else role]
        sums={}
        for f in ("quarter","half","full"):
            blocks=[ref["pairs"][n]["metrics"][role][f] for n in names]
            sums[f]=pool(blocks)
            compare(sums[f],a[f]["pooled"],(h,role,f,"pooled"))
            means={k:sum(b[k] for b in blocks)/len(blocks) for k in ("crisis_f1","macro_f1","brier","logloss")}
            compare(means,a[f]["mean_of_pairs"],(h,role,f,"mean"))
        for f in ("quarter","half"):
            ad=a[f+"_minus_full"]
            compare({k:sums[f][k]-sums["full"][k] for k in ("crisis_f1","macro_f1","brier","logloss")},ad["pooled"],(h,role,f,"delta"))
            means={k:sum(ref["pairs"][n]["metrics"][role][f][k]-ref["pairs"][n]["metrics"][role]["full"][k] for n in names)/len(names) for k in ("crisis_f1","macro_f1","brier","logloss")}
            compare(means,ad["mean_of_pair_deltas"],(h,role,f,"mean delta"))
            for k in ("changed","corrected","spoiled"):
                ck(sum(ref["pairs"][n]["changes"][role][f][k] for n in names)==ad["decisions"][k],(h,role,f,k))
            for k in ("tp","fp"):ck(sums[f][k]-sums["full"][k]==ad["decisions"][k+"_delta"],(h,role,f,k))
        if role=="E3_matched":
            compare(pool([ref["pairs"][n]["metrics"][role]["persistence"] for n in names]),a["persistence_reference"],(h,"persistence"))
            ck("logloss_fourclass" not in a["persistence_reference"],(h,"no persistence logloss"))
rows=RUN/"rows_C_E3_prefix.csv.gz"
ck(sha(rows)==ident["rows_C_E3_prefix_sha256"],"rows SHA")
df=pd.read_csv(rows,float_precision="round_trip")
ck(not df.duplicated(["root","part","area","target_month"]).any(),"unique persisted keys")
expected=sum(r["metrics"][p]["full"]["n"] for r in ref["pairs"].values() for p in ("C","E3"))
ck(len(df)==expected,"complete persisted row count")
features=json.loads((P/"feature-schema.json").read_text())["ordered_features"]
for h in (4,8,12):
    snap=pd.read_parquet(D/"prepared"/f"snapshot_h{h}.parquet",columns=["area","target_month","class_code"]+features,
        filters=[("target_month","<=",mi("2020-12"))]).set_index(["area","target_month"])
    for name,r in ref["pairs"].items():
        if r["horizon"]!=h:continue
        mem=pd.read_csv(D/"stage1_e1pair/roots"/name/"fold_membership.csv.gz")
        b=xgb.Booster();b.load_model(ident["pairs"][name]["checkpoint"])
        for part,role in (("C","confirmation"),("E3","heldout_target")):
            sub=df[(df.root==name)&(df.part==part)];keys=list(zip(sub.area,sub.target_month.map(mi)))
            expectedkeys=list(zip(mem[mem.role==role].area,mem[mem.role==role].target_month.map(mi)))
            ck(set(keys)==set(expectedkeys),(name,part,"exact keys"))
            inp=snap.loc[keys]
            ck(np.array_equal(sub.truth.to_numpy(),inp.class_code.to_numpy()),(name,part,"truth"))
            X=inp[features].to_numpy(dtype=float);X[np.isinf(X)]=np.nan
            for f,rounds in r["rounds"].items():
                obj=b if f=="full" else b[:rounds]
                q=obj.predict(xgb.DMatrix(X,missing=np.nan,nthread=4))
                saved=sub[[f"p_{f}_{l}" for l in ("1","2","3","4或5")]].to_numpy(dtype=float)
                ck(np.array_equal(q.astype(float),saved),(name,part,f,"raw prefix exact"))
            if part=="E3":
                ph=inp.hist_phase_o00.to_numpy(float)
                per=np.where(np.isfinite(ph),np.minimum(ph,4)-1,np.nan)
                ck(np.array_equal(per,sub.persistence_code.to_numpy(float),equal_nan=True),(name,"persistence"))
result=dict(passed=True,checks=checks,issues=[],rows=len(df),scope="All21 roots, all189 FIT/C/E3 metric cells independently recomputed via sliced boosters; all perH aggregates/means/deltas and every persisted C/E3 prefix probability verified",
    summary_sha256=sha(RUN/"summary.json"),identity_sha256=sha(RUN/"identity.json"),script_sha256=sha(Path(__file__)))
OUT.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result))
