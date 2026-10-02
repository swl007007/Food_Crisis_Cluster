"""Independent D38 artifact replay. No production imports and no refits."""
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

source, run, pkg = map(Path, sys.argv[1:4])
stage = source / "stage1_e1pair"
summary = json.loads((run / "summary.json").read_text())
features = json.loads((pkg / "feature-schema.json").read_text())["ordered_features"]
labels = ("1", "2", "3", "4或5")
models = ("original", "anchored", "prior_only", "posthoc")
expected = {f"h{h}_{t}_{g}_r80_s42_e1pair" for h,g in [(4,"G1"),(8,"G4"),(12,"G2")]
            for t in ("2018-06","2018-10","2019-02","2019-06","2019-10","2020-02","2020-06")}
assert set(summary["per_root"]) == expected
assert {p.name for p in run.iterdir() if p.is_dir()} == expected
assert json.loads((run / "gate.json").read_text())["passed"]

def mon(s): return int(s[:4])*12+int(s[5:])-1
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return pd.read_csv(p,float_precision="round_trip")
def priors(phase):
    k=np.isfinite(phase)
    assert np.isin(phase[k], [1,2,3,4]).all()
    q=np.full((len(phase),4),.25);q[k]=.125
    q[np.flatnonzero(k),phase[k].astype(int)-1]=.625
    m=np.full(q.shape,.5,dtype=np.float32)
    logs=np.log(q[k]);m[k]=(.5+logs-logs.mean(axis=1,keepdims=True)).astype(np.float32)
    return q,m,k
def confusion(t,y):
    a=np.zeros((4,4),dtype=np.int64);np.add.at(a,(np.asarray(t,int),np.asarray(y,int)),1);return a
def f1(a):
    tp,fp,fn=int(a[2:,2:].sum()),int(a[:2,2:].sum()),int(a[2:,:2].sum())
    return Fraction(2*tp,2*tp+fp+fn) if tp+fp+fn else Fraction(0)

tot={part:{m:np.zeros((4,4),dtype=np.int64) for m in models} for part in ("C","E3")}
loss={part:{m:0. for m in models} for part in ("C","E3")};ns={"C":0,"E3":0}
checks=0;roots=[]
for h in (4,8,12):
    snap=pd.read_parquet(source/"prepared"/f"snapshot_h{h}.parquet", columns=["area","target_month","class_code"]+features,
                         filters=[("target_month","<=",2020*12+11)]).set_index(["area","target_month"])
    assert not snap.index.duplicated().any()
    for name in sorted(n for n in expected if n.startswith(f"h{h}_")):
        rd=run/name; root=json.loads((stage/"roots"/name/"root.json").read_text())
        meta=json.loads((rd/"anchored_root.json").read_text())
        mem=read(stage/"roots"/name/"fold_membership.csv.gz");fit=mem[mem.role=="fitting"];savedfit=read(rd/"fitting_origin.csv.gz")
        assert list(zip(fit.area,fit.target_month))==list(zip(savedfit.area,savedfit.target_month))
        assert np.array_equal(fit.class_code,savedfit.class_code)
        fm=np.array([mon(s) for s in fit.target_month],dtype=np.int64);o=mon(root["target_month"])-h
        assert np.all((fm>=o-59)&(fm<o))
        keys=np.column_stack([fit.area.to_numpy(np.int64),fm])
        assert hashlib.sha256(keys.tobytes()).hexdigest()==root["fitting_keys_sha256"]==meta["fitting_keys_sha256"]
        fs=snap.loc[list(map(tuple,keys))];assert np.array_equal(fs.class_code,fit.class_code)
        q,m,k=priors(fs.hist_phase_o00.to_numpy(float))
        assert np.allclose(fs.hist_phase_o00.to_numpy(float)-1,savedfit.origin_code,equal_nan=True)
        assert np.array_equal(m,savedfit[["margin_"+l for l in labels]].to_numpy(np.float32))
        rec=meta["fit_record"];assert "sample_weight" not in rec
        assert hashlib.sha256(m.tobytes()).hexdigest()==rec["base_margin"]["sha256"]
        assert rec["base_margin"]["dtype"]=="float32" and rec["base_margin"]["shape"]==[len(fit),4]
        assert meta["origin_available_fraction"]["fitting"]==float(k.mean())
        for key,val in root["config"]["G"].items():
            if key!="rounds":assert rec["params"][key]==val,(name,key)
        assert rec["rows"]==meta["fitting_rows"]==len(fit)
        op=Path(meta["original_root_source"]);ap=rd/"anchored_root.ubj"
        assert sha(op)==root["root_booster_sha256"]==meta["original_root_sha256"]
        assert sha(ap)==meta["anchored_root_sha256"]==rec["booster_sha256"]
        original=xgb.Booster(model_file=str(op));anchored=xgb.Booster(model_file=str(ap))
        assert original.attr("geoxgb_base_margin") is None
        assert anchored.attr("geoxgb_base_margin")=="d38-persistence-lambda0.5-v1"
        for b in (original,anchored):
            b.set_param({"nthread":4});assert b.num_boosted_rounds()==root["config"]["G"]["rounds"]
        for part,role in [("C","confirmation"),("E3","heldout_target")]:
            f=read(rd/f"rows_{part}.csv.gz");wanted=mem[mem.role==role]
            assert list(zip(f.area,f.target_month))==list(zip(wanted.area,wanted.target_month))
            assert np.array_equal(f.truth,wanted.class_code)
            ss=snap.loc[list(zip(f.area,[mon(s) for s in f.target_month]))]
            assert np.array_equal(ss.class_code,f.truth)
            phase=ss.hist_phase_o00.to_numpy(float);q,m,k=priors(phase)
            assert np.allclose(f.persistence_code,phase-1,equal_nan=True)
            assert np.allclose(f.origin_phase,phase,equal_nan=True)
            X=ss[features].to_numpy(float);X[~np.isfinite(X)]=np.nan
            dm=xgb.DMatrix(X,missing=np.nan,nthread=4)
            po=original.predict(dm).astype(float)
            plain=anchored.predict(dm,output_margin=True)
            # A fresh DMatrix is required: changing base_margin after this Booster
            # has predicted on the same DMatrix leaves XGBoost's prediction cache stale.
            dm=xgb.DMatrix(X,missing=np.nan,nthread=4,base_margin=m)
            pa=anchored.predict(dm).astype(float)
            margin=anchored.predict(dm,output_margin=True)
            assert np.allclose(margin-plain,m-.5,rtol=0,atol=2e-5)
            pp=po.copy();pq=po[k]*q[k];pp[k]=pq/pq.sum(axis=1,keepdims=True)
            assert np.array_equal(pp[~k],po[~k])
            probs={"original":po,"anchored":pa,"prior_only":q,"posthoc":pp}
            block=summary["per_root"][name]["scores"][part]
            assert block["n"]==len(f) and block["matched_persistence"]["n"]==int(k.sum())
            for model,p in probs.items():
                assert np.array_equal(p,f[[f"p_{model}_{l}" for l in labels]].to_numpy()),(name,part,model)
                y=p.argmax(axis=1);assert np.array_equal(y,f["y_"+model])
                for mask,reported in [(np.ones(len(f),bool),block["all"]),(k,block["matched_persistence"])]:
                    a=confusion(f.truth[mask],y[mask]);assert a.tolist()==reported[model]["confusion_fourclass"]
                    assert f1(a)==Fraction(reported[model]["crisis_f1_exact"])
                    brier=float(np.mean((p[mask,2:].sum(1)-(f.truth.to_numpy()[mask]>=2))**2))
                    assert abs(brier-reported[model]["crisis_brier"])<1e-14;checks+=1
                tot[part][model]+=confusion(f.truth,y)
                loss[part][model]+=float(np.sum((p[:,2:].sum(1)-(f.truth.to_numpy()>=2))**2))
                if model in ("original","anchored","posthoc"):
                    z=f.truth.to_numpy()[k]>=2;a=y[k]>=2;per=phase[k]>=3;diff=a!=per
                    dep=block["departure_from_persistence"][model]
                    assert dep["disagree"]==int(diff.sum())
                    assert dep["model_right"]==int((diff&(a==z)).sum())
                    assert dep["persistence_right"]==int((diff&(per==z)).sum())
            assert np.array_equal(q[k].argmax(1),phase[k]-1)
            ns[part]+=len(f)
        roots.append({"root":name,"fit_margin_sha256":rec["base_margin"]["sha256"]})
aggregate={}
for part in ("C","E3"):
    aggregate[part]={}
    for model in models:
        assert tot[part][model].tolist()==summary["overall"][part]["pooled_all"][model]["confusion_fourclass"]
        aggregate[part][model]={"f1":float(f1(tot[part][model])),"brier":loss[part][model]/ns[part]}
out=run.parent/"d38_independent_results.json"
assert not out.exists()
out.write_text(json.dumps({"roots":len(roots),"confusion_score_checks":checks,"probability_rows_per_model":sum(ns.values()),
                          "no_refits_no_production_imports":True,"aggregate":aggregate,"details":roots},indent=2))
print(out);print("PASS",len(roots),"roots",checks,"score checks",sum(ns.values()),"probability rows/model")
print(json.dumps(aggregate,indent=2))
