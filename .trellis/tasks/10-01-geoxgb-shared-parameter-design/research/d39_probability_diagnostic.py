"""D39: frozen saved-probability diagnostic, no fitting or threshold search."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

BASE=Path(r"C:\Users\swl00\geoxgb_runs")
RUN=BASE/"geoxgb-d38-persistence-margin-root-20261002"
OUT=BASE/"d39_probability_diagnostic.json"
assert not OUT.exists()
MODELS=("original","anchored","posthoc","prior_only")
LABELS=("1","2","3","4或5")
EXPECTED={f"h{h}_{t}_{g}_r80_s42_e1pair" for h,g in [(4,"G1"),(8,"G4"),(12,"G2")]
          for t in ("2018-06","2018-10","2019-02","2019-06","2019-10","2020-02","2020-06")}
summary=json.loads((RUN/"summary.json").read_text())
assert set(summary["per_root"])==EXPECTED
assert json.loads((RUN/"identity.json").read_text())["script_commit"]=="2d4fe4e3dac5bff68327c5a426d72f41c300c212"

def confusion(z,y):
    tp,fp,fn,tn=(int(a.sum()) for a in (z&y,~z&y,z&~y,~z&~y))
    return {"tp":tp,"fp":fp,"fn":fn,"tn":tn,"f1":2*tp/(2*tp+fp+fn) if tp+fp+fn else 0.}

def describe(f,model,bins=False):
    z=f.truth.to_numpy()>=2
    if model=="persistence":
        assert f.persistence_code.notna().all()
        p=(f.persistence_code.to_numpy()>=2).astype(float);y=p.astype(bool)
    else:
        p=f["pc_"+model].to_numpy();y=f["y_"+model].to_numpy()>=2
    n=len(z);positive=int(z.sum());classes=int(np.unique(z).size)
    out={"n":n,"positive":positive,"negative":n-positive,"prevalence":float(z.mean()) if n else None,
         "mean_p":float(p.mean()) if n else None,"crisis_call_rate":float(y.mean()) if n else None,
         "argmax_crisis":confusion(z,y),"brier":float(np.mean((p-z)**2)) if n else None,
         "roc_auc":float(roc_auc_score(z,p)) if classes==2 else None,
         "average_precision":float(average_precision_score(z,p)) if classes==2 else None,
         "ranking_status":"defined" if classes==2 else "undefined_single_or_empty_class"}
    if model!="persistence":
        b=p>=.5;table=np.zeros((2,2),dtype=np.int64);np.add.at(table,(y.astype(int),b.astype(int)),1)
        out["fixed_half_mass"]={"argmax_rows_mass_columns":table.tolist(),"crisis":confusion(z,b),
                                "corrected":int(((b==z)&(y!=z)).sum()),"spoiled":int(((b!=z)&(y==z)).sum())}
    if bins:
        index=np.clip(np.floor(p*10).astype(int),0,9)
        out["fixed_bins"]=[]
        for i in range(10):
            k=index==i;nn=int(k.sum())
            out["fixed_bins"].append({"index":i,"lower":i/10,"upper":(i+1)/10,"upper_closed":i==9,
                 "n":nn,"positive":int(z[k].sum()),"mean_p":float(p[k].mean()) if nn else None,
                 "prevalence":float(z[k].mean()) if nn else None,
                 "brier":float(np.mean((p[k]-z[k])**2)) if nn else None})
        assert sum(x["n"] for x in out["fixed_bins"])==n
        if n:assert abs(sum(x["n"]*x["brier"] for x in out["fixed_bins"] if x["n"])/n-out["brier"])<1e-14
    return out

def group(f,bins=False):
    known=f.persistence_code.notna();matched=f[known]
    result={"n_all":len(f),"n_matched":len(matched),"missing_origin_n":int((~known).sum()),
            "all":{m:describe(f,m,bins) for m in MODELS},
            "matched":{m:describe(matched,m,bins) for m in MODELS+("persistence",)},
            "missing":{m:describe(f[~known],m) for m in MODELS},"by_origin_crisis":{}}
    assert np.array_equal(matched.y_prior_only,matched.persistence_code)
    for c in (0,1):
        ff=matched[(matched.persistence_code>=2)==bool(c)]
        records={m:describe(ff,m,bins) for m in MODELS+("persistence",)}
        if records["persistence"]["ranking_status"]=="defined":
            assert records["persistence"]["roc_auc"]==.5
            assert abs(records["persistence"]["average_precision"]-records["persistence"]["prevalence"])<1e-15
        result["by_origin_crisis"][str(c)]=records
    return result

hashes={};frames=[];per_root={}
for name in sorted(EXPECTED):
    p=RUN/name/"rows_E3.csv.gz";hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    f=pd.read_csv(p,float_precision="round_trip")
    assert not f.duplicated(["area","target_month","horizon"]).any()
    assert f.target_month.nunique()==1 and f.target_month.iloc[0]==summary["per_root"][name]["target_month"]
    assert f.horizon.nunique()==1 and int(f.horizon.iloc[0])==summary["per_root"][name]["horizon"]
    assert (f.target_month<="2020-12").all() and f.truth.isin(range(4)).all()
    for m in MODELS:
        pr=f[[f"p_{m}_{l}" for l in LABELS]].to_numpy()
        assert np.isfinite(pr).all() and (pr>=0).all() and (pr<=1).all()
        assert np.allclose(pr.sum(1),1,rtol=0,atol=2e-7)
        assert np.array_equal(pr.argmax(1),f["y_"+m])
        f["pc_"+m]=pr[:,2:].sum(1)
    f["root"]=name
    per_root[name]=group(f)
    for m in MODELS:
        ref=summary["per_root"][name]["scores"]["E3"]["all"][m]
        assert abs(per_root[name]["all"][m]["argmax_crisis"]["f1"]-ref["crisis_f1"])<1e-15
        assert abs(per_root[name]["all"][m]["brier"]-ref["crisis_brier"])<1e-15
    frames.append(f)
frame=pd.concat(frames,ignore_index=True)
result={"rule":"D39 fixed diagnostic only; no fit, threshold search, calibration fit, final-period access, or endpoint adoption. Per-root/H + pooled exposed E3; within-origin ranking, fixed equal-width bins, fixed p>=.5 diagnostic.",
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"source_producer":"2d4fe4e",
        "input_hashes":hashes,"per_root":per_root,
        "by_horizon":{str(h):group(f,True) for h,f in frame.groupby("horizon")},"overall":group(frame,True)}
OUT.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(OUT)
for h,r in list(result["by_horizon"].items())+[("all",result["overall"])]:
    for stratum,key in [("matched",None),("origin0","0"),("origin1","1")]:
        d=r["matched"] if key is None else r["by_origin_crisis"][key]
        print(h,stratum)
        for m in MODELS+("persistence",):
            v=d[m];print(m,"n",v["n"],"prev",round(v["prevalence"],4),"meanp",round(v["mean_p"],4),
                "auc",round(v["roc_auc"],4),"ap",round(v["average_precision"],4),"f1",round(v["argmax_crisis"]["f1"],4),
                "massf1",round(v.get("fixed_half_mass",{}).get("crisis",{}).get("f1",0),4))
