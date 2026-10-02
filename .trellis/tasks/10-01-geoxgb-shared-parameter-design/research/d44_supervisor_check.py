"""D44 independent original-file rescore and fitting provenance; zero fits."""
import csv, gzip, hashlib, json, math, platform
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

B = Path(r"C:\Users\swl00\geoxgb_runs")
V = B / "geoxgb-v1-20261001"
D = B / "geoxgb-d34-e1-brier-20261002"
P = Path(r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\2.source_code\Step5_Geo_RF_trial\Food_Crisis_Cluster\FEWSNETGeoXGBExperiment")
OUT = B / "d44_supervisor_results.json"
assert not OUT.exists()
DATES = ("2019-02", "2019-06", "2019-10", "2020-02", "2020-06")
GS = {4: "G1", 8: "G4", 12: "G2"}
LABELS = ("1", "2", "3", "4或5")
checks = 0

def check(ok, where):
    global checks
    checks += 1
    assert ok, where

def read(p):
    with (gzip.open(p, "rt", encoding="utf-8") if p.suffix == ".gz" else p.open(encoding="utf-8")) as f:
        return list(csv.DictReader(f))

def mi(t):
    return int(t[:4])*12 + int(t[5:])-1

def digest(keys):
    return hashlib.sha256(np.asarray(keys, dtype=np.int64).tobytes()).hexdigest()

def score(rows, arm):
    c = [[0]*4 for _ in range(4)]
    errors = []
    for r in rows:
        q = r[arm]
        pred = max(range(4), key=q.__getitem__)
        c[r["truth"]][pred] += 1
        errors.append((math.fsum(q[2:]) - (r["truth"] >= 2))**2)
    tp = sum(c[i][j] for i in (2,3) for j in (2,3))
    fp = sum(c[i][j] for i in (0,1) for j in (2,3))
    fn = sum(c[i][j] for i in (2,3) for j in (0,1))
    macro = []
    for i in range(4):
        den = sum(c[i]) + sum(c[j][i] for j in range(4))
        macro.append(2*c[i][i]/den if den else 0.)
    return dict(n=len(rows),tp=tp,fp=fp,fn=fn,crisis_f1=2*tp/(2*tp+fp+fn) if tp+fp+fn else 0.,macro_f1=sum(macro)/4,brier=math.fsum(errors)/len(rows))

def block(rows):
    out = {}
    for cohort, rr in (("all",rows),("persistence_available",[r for r in rows if r["persistence"] is not None])):
        arms = ("r80", "full") + (("persistence",) if cohort != "all" else ())
        scores = {a:score(rr,a) for a in arms}
        corrected = spoiled = 0
        for r in rr:
            a,b = [max(range(4),key=r[z].__getitem__) >= 2 for z in ("r80","full")]
            if a != b:
                corrected += b == (r["truth"] >= 2)
                spoiled += a == (r["truth"] >= 2)
        out[cohort] = dict(scores=scores,corrected=corrected,spoiled=spoiled)
    return out

vp = {(int(r["horizon"]),r["target_month"],int(r["area"])):r for r in read(V/"gscreen/predictions.csv.gz") if int(r["horizon"]) in GS and r["g_config"]==GS[int(r["horizon"])] and r["target_month"] in DATES}
base = {(int(r["horizon"]),r["target_label"],int(r["area"])):r for r in read(V/"prepared/ledgers/dev_baselines.csv") if r["target_label"] in DATES}
features = json.loads((P/"feature-schema.json").read_text())["ordered_features"]
allrows=[]; identities={}; replay_models=[]
for h,g in GS.items():
    snap=pd.read_parquet(D/"prepared"/f"snapshot_h{h}.parquet",columns=["area","target_month","class_code"]+features,filters=[("target_month","<=",mi("2020-12"))]).sort_values(["area","target_month"])
    for t in DATES:
        name=f"h{h}_{t}_{g}_r80_s42_e1pair"
        root=D/"stage1_e1pair/roots"/name
        meta=json.loads((root/"root.json").read_text())
        rm=read(root/"fold_membership.csv.gz")
        keys=[(int(r["area"]),mi(r["target_month"])) for r in rm if r["role"]=="fitting"]
        check(digest(keys)==meta["fitting_keys_sha256"],(name,"FIT digest"))
        hist=snap[(snap.target_month>=mi(t)-h-59)&(snap.target_month<mi(t)-h)]
        origin=meta["origin_month"]
        glob=V/"globals"/f"h{h}"/g/f"O{origin}"
        gm=json.loads(glob.with_suffix(".json").read_text())
        check(digest(hist[["area","target_month"]].to_numpy())==gm["fit_keys_sha256"],(name,"full digest"))
        legal={(int(r["area"]),mi(r["target_month"])) for r in rm if r["role"]!="heldout_target"}
        full=set(map(tuple,hist[["area","target_month"]].to_numpy()))
        check(legal <= full,(name,"subset"))
        check(gm["params"]==meta["root_fit"]["params"],(name,"params"))
        extra=full-legal
        identities[name]=dict(fit=len(keys),legal=len(legal),full=len(full),extra_rows=len(extra),extra_areas=sorted({int(a) for a,m in extra}),fit_fraction=len(keys)/len(legal))
        rp=read(root/"root_target_predictions.csv")
        check({int(r["FEWSNET_admin_code"]) for r in rp}=={a for hh,tt,a in vp if (hh,tt)==(h,t)},(name,"target exact set"))
        pair=[]
        for r in rp:
            area=int(r["FEWSNET_admin_code"]); key=(h,t,area); v=vp[key]; b=base[key]
            y=int(r["y_true_code"])
            check(y==int(v["y_true_code"])==int(b["truth_code"]),(key,"truth"))
            pp=None if not b["persistence_code"] else [float(i==int(float(b["persistence_code"]))) for i in range(4)]
            row=dict(h=h,t=t,area=area,truth=y,r80=[float(r["p_pooled_"+l]) for l in LABELS],full=[float(v["p_"+l]) for l in LABELS],persistence=pp)
            pair.append(row)
        allrows.extend(pair)
        if t=="2019-06":
            e3=snap[snap.target_month==mi(t)].set_index("area").loc[[r["area"] for r in pair]]
            X=e3[features].to_numpy(dtype=float);X[np.isinf(X)]=np.nan
            for arm,path,sha in (("full",glob.with_suffix(".ubj"),gm["booster_sha256"]),("r80",D/"stage1_e1pair/checkpoints"/f"h{h}_{t}_{g}_L1_r80_s42_e1brier_gt0"/"xgb_root.ubj",meta["root_booster_sha256"])):
                check(hashlib.sha256(path.read_bytes()).hexdigest()==sha,(name,arm,"UBJ hash"))
                booster=xgb.Booster();booster.load_model(path)
                pred=booster.predict(xgb.DMatrix(X,missing=np.nan,nthread=4))
                check(np.array_equal(pred,np.asarray([r[arm] for r in pair],dtype=np.float32)),(name,arm,"raw replay"))
                replay_models.append(str(path))
result=dict(checks=checks,rows=len(allrows),per_pair={f"h{h}_{t}":block([r for r in allrows if r["h"]==h and r["t"]==t]) for h in GS for t in DATES},per_h={str(h):block([r for r in allrows if r["h"]==h]) for h in GS},overall=block(allrows),identities=identities,raw_replayed=replay_models,runtime=dict(python=platform.python_version(),xgboost=xgb.__version__))
OUT.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(dict(checks=checks,rows=len(allrows),overall=result["overall"],out=str(OUT))))
