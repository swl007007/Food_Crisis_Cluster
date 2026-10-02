"""D40 frozen forward decision policy. No model fits, calibration, or current-target threshold search."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(r"C:\Users\swl00\geoxgb_runs")
SOURCE=BASE/"geoxgb-d38-persistence-margin-root-20261002"
OUT=BASE/"d40-forward-decision-20261002"
ARMS=("original","anchored")
DATES=("2018-06","2018-10","2019-02","2019-06","2019-10","2020-02","2020-06")
EXPECTED_COUNTS={4:[0,0,1,2,3,4,5],8:[0,0,0,1,2,3,4],12:[0,0,0,0,1,2,3]}

def month(s):return int(s[:4])*12+int(s[5:])-1
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def choose_threshold(score,truth):
    """Source rows only. Group ties; exact F1; largest tau on ties."""
    score=np.asarray(score,dtype=float);truth=np.asarray(truth,dtype=bool)
    assert len(score)==len(truth) and truth.any() and (~truth).any() and np.isfinite(score).all()
    order=np.argsort(-score,kind="stable");s=score[order];z=truth[order]
    ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
    ct=np.cumsum(z,dtype=np.int64);positive=int(z.sum())
    best=Fraction(-1);tau=None;counts=None
    for ix in ends:
        tp=int(ct[ix]);fp=int(ix+1-tp);fn=positive-tp;value=Fraction(2*tp,2*tp+fp+fn)
        if value>best:best=value;tau=float(s[ix]);counts={"tp":tp,"fp":fp,"fn":fn}
    return {"tau":tau,"tau_hex":tau.hex(),"source_f1_exact":str(best),"source_f1":float(best),"source_best_confusion":counts}

def source_names(names,h,target):
    origin=month(target)-h
    return sorted(n for n in names if n.startswith(f"h{h}_") and month(n.split("_")[1])<origin)

def binary_score(truth,pred):
    z=np.asarray(truth,bool);y=np.asarray(pred,bool)
    tp,fp,fn,tn=(int(v.sum()) for v in (z&y,~z&y,z&~y,~z&~y))
    f=Fraction(2*tp,2*tp+fp+fn) if tp+fp+fn else Fraction(0)
    return {"n":len(z),"tp":tp,"fp":fp,"fn":fn,"tn":tn,"f1":float(f),"f1_exact":str(f)}

def report(f):
    z=f.truth.to_numpy()>=2;old=f.argmax_crisis.to_numpy(bool);new=f.policy_crisis.to_numpy(bool)
    k=f.persistence_code.notna().to_numpy();per=f.persistence_code.to_numpy(float)>=2
    groups=np.where(~k,"missing",np.where(per,"1","0").astype(object)+np.where(z,"1","0"))
    layers={}
    for g in ("00","01","10","11","missing"):
        m=groups==g
        layers[g]={"n":int(m.sum()),"corrected":int((m&(new==z)&(old!=z)).sum()),
                   "spoiled":int((m&(new!=z)&(old==z)).sum()),
                   "tp_change":int((m&new&z).sum()-(m&old&z).sum()),
                   "fp_change":int((m&new&~z).sum()-(m&old&~z).sum())}
    d=k&(new!=per)
    return {"all":{"argmax":binary_score(z,old),"policy":binary_score(z,new)},
            "matched":{"argmax":binary_score(z[k],old[k]),"policy":binary_score(z[k],new[k]),"persistence":binary_score(z[k],per[k])},
            "brier_unchanged":float(np.mean((f.p_crisis.to_numpy()-z)**2)),
            "origin_coverage":float(k.mean()),"transitions":layers,
            "departure":{"n":int(d.sum()),"policy_right":int((d&(new==z)).sum()),"persistence_right":int((d&(per==z)).sum())}}

def selfcheck():
    s=np.array([.9,.8,.7,.6]);z=np.array([1,0,0,1],bool)
    got=choose_threshold(s,z);assert got["tau"]==.9 and got["source_f1_exact"]=="2/3"
    s=np.array([.8,.8,.5,.4,.3]);z=np.array([1,0,0,1,0],bool)
    brute=max((Fraction(binary_score(z,s>=t)["f1_exact"]),float(t)) for t in np.unique(s))
    got=choose_threshold(s,z);assert (Fraction(got["source_f1_exact"]),got["tau"])==brute
    assert source_names(["h4_2019-02_G1","h4_2019-06_G1","h8_2018-06_G4"],4,"2019-10")==["h4_2019-02_G1"]

selfcheck();assert not OUT.exists(),"Preserve existing evidence"
reference=json.loads((BASE/"d39_probability_diagnostic.json").read_text())
data={};hashes={}
for name,expected_hash in reference["input_hashes"].items():
    p=SOURCE/name/"rows_E3.csv.gz";hashes[name]=sha(p);assert hashes[name]==expected_hash
    f=pd.read_csv(p,float_precision="round_trip")
    assert not f.duplicated(["area","target_month","horizon"]).any() and (f.target_month<="2020-12").all()
    for arm in ARMS:
        f["pc_"+arm]=f[[f"p_{arm}_{c}" for c in ("3","4或5")]].sum(axis=1)
    data[name]=f
assert len(data)==21
thresholds={}
for name,f in data.items():
    h=int(f.horizon.iloc[0]);target=f.target_month.iloc[0];sources=source_names(data,h,target)
    assert len(sources)==EXPECTED_COUNTS[h][DATES.index(target)]
    both=all(data[n].truth.ge(2).nunique()==2 for n in sources)
    eligible=len(sources)>=3 and both
    thresholds[name]={"horizon":h,"target":target,"origin_index":month(target)-h,"source_roots":sources,
                      "source_dates":[n.split("_")[1] for n in sources],"source_hashes":{n:hashes[n] for n in sources},
                      "eligible":eligible,"reason":"three_or_more_prior_dates" if eligible else "insufficient_prior_dates_or_classes","arms":{}}
    for arm in ARMS:
        if not eligible:thresholds[name]["arms"][arm]={"status":"fallback_argmax"};continue
        past=pd.concat([data[n] for n in sources],ignore_index=True)
        assert all(month(t)<month(target)-h for t in past.target_month.unique())
        scores=past["pc_"+arm].to_numpy();truth=past.truth.ge(2).to_numpy()
        decision=choose_threshold(scores,truth)
        thresholds[name]["arms"][arm]={"status":"source_threshold",**decision,"source_rows":len(past),
            "source_positive":int(truth.sum()),"source_negative":int((~truth).sum())}
        # Changing current/future labels cannot change selected source or its arrays.
        altered={n:d if n in sources else d.assign(truth=3-d.truth) for n,d in data.items()}
        assert source_names(altered,h,target)==sources
        alt=pd.concat([altered[n] for n in sources],ignore_index=True)
        assert choose_threshold(alt["pc_"+arm],alt.truth.ge(2))==decision
assert sum(x["eligible"] for x in thresholds.values())==6

# All decisions are frozen before current-target scoring.
OUT.mkdir()
(OUT/"thresholds.json").write_text(json.dumps(thresholds,indent=2),encoding="utf-8")
per_root={};rows=[]
for name,f in data.items():
    per_root[name]={"eligible":thresholds[name]["eligible"],"arms":{}}
    for arm in ARMS:
        cfg=thresholds[name]["arms"][arm];old=f["y_"+arm].ge(2).to_numpy()
        pred=f["pc_"+arm].ge(cfg["tau"]).to_numpy() if cfg["status"]=="source_threshold" else old.copy()
        if not thresholds[name]["eligible"]:assert np.array_equal(old,pred)
        out=pd.DataFrame({"root":name,"arm":arm,"area":f.area,"target_month":f.target_month,"horizon":f.horizon,
            "truth":f.truth,"persistence_code":f.persistence_code,"p_crisis":f["pc_"+arm],
            "argmax_crisis":old.astype(int),"policy_crisis":pred.astype(int),"eligible":thresholds[name]["eligible"]})
        scored=report(out)
        assert abs(scored["all"]["argmax"]["f1"]-reference["per_root"][name]["all"][arm]["argmax_crisis"]["f1"])<1e-15
        assert abs(scored["brier_unchanged"]-reference["per_root"][name]["all"][arm]["brier"])<1e-15
        per_root[name]["arms"][arm]=scored;rows.append(out)
frame=pd.concat(rows,ignore_index=True)
frame.to_csv(OUT/"policy_rows.csv.gz",index=False,float_format="%.17g")
summary={"rule":"D40 pre-origin-only source optimization; no model fits, probability changes or current-target threshold choice. 6 eligible roots and 15 unchanged fallback roots; conditional exposed development, not endpoint adoption.",
         "script_sha256":sha(Path(__file__)),"input_hashes":hashes,"thresholds_sha256":sha(OUT/"thresholds.json"),
         "policy_rows_sha256":sha(OUT/"policy_rows.csv.gz"),"eligible_roots":sum(v["eligible"] for v in thresholds.values()),
         "per_root":per_root,"all_21":{arm:report(f) for arm,f in frame.groupby("arm")},
         "eligible_6_descriptive":{arm:report(f) for arm,f in frame[frame.eligible].groupby("arm")}}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(OUT)
for name,t in thresholds.items():
    if t["eligible"]:
        print(name)
        for arm,d in t["arms"].items():
            r=per_root[name]["arms"][arm]["matched"]
            print(arm,"tau",d["tau"],"source_f1",d["source_f1"],"current",{k:round(v["f1"],6) for k,v in r.items()})
for name in ("all_21","eligible_6_descriptive"):
    print(name,{arm:{k:round(v["f1"],6) for k,v in r["matched"].items()} for arm,r in summary[name].items()})
