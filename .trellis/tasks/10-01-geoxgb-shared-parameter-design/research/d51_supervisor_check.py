"""Independent D51 raw-model replay, full-FIT lineage and78-column projection and metric verification; zero fits."""
import hashlib, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
B=Path(r"C:\Users\swl00\geoxgb_runs")
D=B/"geoxgb-d34-e1-brier-20261002"
S=D/"stage1_e1pair"
R=B/"geoxgb-d51-history-calendar-root-20261002"
P=Path(r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\2.source_code\Step5_Geo_RF_trial\Food_Crisis_Cluster\FEWSNETGeoXGBExperiment")
OUT=B/"d51_supervisor_results.json"
assert not OUT.exists()
assert (platform.python_version(),np.__version__,pd.__version__,xgb.__version__)==("3.12.10","2.2.6","2.2.3","3.0.0")
DATES=("2018-06","2018-10","2019-02","2019-06","2019-10","2020-02","2020-06")
GS={4:"G1",8:"G4",12:"G2"}
LABELS=("1","2","3","4或5")
ARMS=("original","h78")
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

from fractions import Fraction
import subprocess
schema=json.loads((P/"feature-schema.json").read_text())
selected=set(schema["known_calendar"]+[f for group in schema["history_blocks"].values() for f in group])
F78=[f for f in FEATURES if f in selected]; IDX=[FEATURES.index(f) for f in F78]
fmap=json.loads((R/"feature_map.json").read_text()); gate=json.loads((R/"gate.json").read_text())
check(len(F78)==78 and len(FEATURES)==162,"schema counts")
check(fmap["selected_names"]==F78 and fmap["selected_index"]==IDX,"exact selected order")
check(fmap["removed_names"]==schema["static_sources"]+schema["dynamic_sources_at_origin"]+schema["legacy_covariate_derived"],"exact removed84")
check(fmap["schema_sha256"]==identity["schema"]["sha256"]==sha(P/"feature-schema.json"),"schema identity")
check(identity["script"]["sha256"]==sha(Path.cwd()/identity["script"]["path"]),"script sha")
check(subprocess.check_output(["git","rev-parse",identity["repo_head"]+":"+identity["script"]["path"]],text=True).strip()==identity["script"]["git_blob"],"committed script")
check(gate["passed"],"all gates passed")
d50=json.loads((Path.cwd()/".trellis/tasks/10-01-geoxgb-shared-parameter-design/research/d50_summary.json").read_text())
pairs={};totals={};nrows=0;phase_counts={};expected={f"h{h}_{t}_{g}_r80_s42_e1pair" for h,g in GS.items() for t in DATES}
check(set(summary["per_root"])==set(gate["roots"])==expected,"21 expected roots")
for h,g in GS.items():
    snapfile=D/"prepared"/f"snapshot_h{h}.parquet"
    snap=pd.read_parquet(snapfile,columns=["area","target_month","class_code"]+FEATURES,filters=[("target_month","<=",mi("2020-12"))]).set_index(["area","target_month"])
    check(snap.index.is_unique,(h,"snapshot unique")); sh=sha(snapfile)
    pool={part:[] for part in ("FIT","C","E3")}
    for t in DATES:
        name=f"h{h}_{t}_{g}_r80_s42_e1pair";rd=R/name;root=S/"roots"/name
        meta=json.loads((rd/"h78_root.json").read_text());oldmeta=json.loads((root/"root.json").read_text())
        mem=read(root/"fold_membership.csv.gz");mem["month"]=mem.target_month.map(mi)
        check(not mem.duplicated(["area","month"]).any(),(name,"unique membership"))
        fm=mem[mem.role=="fitting"]
        keysha=hashlib.sha256(fm[["area","month"]].to_numpy(np.int64).tobytes()).hexdigest()
        check(keysha==meta["fitting_keys_sha256"]==oldmeta["fitting_keys_sha256"],(name,"all original fit keys/order"))
        check(meta["fitting_labels_sha256"]==hashlib.sha256(fm.class_code.to_numpy(np.int64).tobytes()).hexdigest(),(name,"all fit labels/order"))
        check(meta["fitting_rows"]==meta["fit_record"]["rows"]==len(fm),(name,"all FIT rows retained"))
        origin=mi(t)-h;check(fm.month.between(origin-59,origin-1).all(),(name,"FIT window"))
        check(meta["fit_record"]["params"]==oldmeta["root_fit"]["params"],(name,"G params unchanged"))
        check("sample_weight" not in meta["fit_record"] and "base_margin" not in meta["fit_record"],(name,"no weights/margins"))
        check(sh==identity["inputs"][name]["snapshot"],(name,"snapshot identity"))
        for fn in ("root.json","fold_membership.csv.gz","root_target_predictions.csv"):
            check(sha(root/fn)==identity["inputs"][name][fn],(name,fn,"source identity"))
        models={}
        for arm,path in (("original",S/"checkpoints"/f"h{h}_{t}_{g}_L1_r80_s42_e1brier_gt0"/"xgb_root.ubj"),("h78",rd/"h78_root.ubj")):
            check(sha(path)==meta[f"{arm}_root_sha256"],(name,arm,"hash"))
            model=xgb.Booster();model.load_model(path);models[arm]=model
            config=json.loads(model.save_config())["learner"]["learner_model_param"]
            check(float(config["base_score"])==.5 and int(config["num_class"])==4,(name,arm,"base/class"))
            check(model.num_boosted_rounds()==(200 if h==4 else 400),(name,arm,"rounds"))
            check(model.num_features()==(162 if arm=="original" else 78),(name,arm,"features"))
        out={}
        for role,part in (("fitting","FIT"),("confirmation","C"),("heldout_target","E3")):
            keys=mem[mem.role==role];data=snap.loc[list(zip(keys.area,keys.month))];y=data.class_code.to_numpy(int)
            check(np.array_equal(y,keys.class_code),(name,part,"snapshot truth"))
            X=data[FEATURES].to_numpy(float);X[np.isinf(X)]=np.nan
            X78=data[F78].to_numpy(float);X78[np.isinf(X78)]=np.nan
            check(np.array_equal(X78,X[:,IDX],equal_nan=True),(name,part,"projection by names vs indices"))
            preds={a:m.predict(xgb.DMatrix(X if a=="original" else X78,missing=np.nan,nthread=4)).astype(float) for a,m in models.items()}
            rows=read(rd/f"rows_{part}.csv.gz");nrows+=len(rows)
            check(np.array_equal(rows.area,keys.area) and np.array_equal(rows.target_month,keys.target_month) and np.array_equal(rows.truth,y),(name,part,"persisted keys/order/truth"))
            phase=data.hist_phase_o00.to_numpy(float);mask=np.isfinite(phase);per=np.minimum(phase[mask],4).astype(int)-1
            check(np.array_equal(rows.persistence_code.notna(),mask) and np.array_equal(rows.persistence_code[mask],per),(name,part,"persistence"))
            scores=summary["per_root"][name]["scores"][part];out[part]={}
            check(scores["n_all"]==len(y) and scores["n"]==int(mask.sum()) and scores["excluded_missing_origin"]==int((~mask).sum()),(name,part,"coverage"))
            for a,p in preds.items():
                check(np.array_equal(rows[[f"p_{a}_{l}" for l in LABELS]].to_numpy(float),p),(name,part,a,"raw replay exact"))
                check(np.array_equal(rows[f"y_{a}"],p.argmax(1)),(name,part,a,"argmax"))
                m=score(y,p);compare_score(m,scores["all_keys"][a],(name,part,a,"all"))
                mm=score(y[mask],p[mask]);compare_score(mm,scores[a],(name,part,a,"matched"))
                z=y[mask]>=2;sc=p[mask,2:].sum(1)/p[mask].sum(1)
                rank={"auc":roc_auc_score(z,sc),"ap":average_precision_score(z,sc)}
                for k,v in rank.items():near(v,scores["ranking"][a][k],(name,part,a,k))
                out[part][a]={"all":m,"matched":mm,**rank}
                if part=="E3":
                    for code in range(4):
                        k=per==code;zz=z[k];ss=sc[k];Pz=int(zz.sum());Nz=len(zz)-Pz
                        record=summary["per_root"][name]["d50_cells"][str(code)]
                        check((record["n"],record["P"],record["N"])==(len(zz),Pz,Nz),(name,a,code,"phase support"))
                        reason="empty" if len(zz)==0 else "no_positive" if Pz==0 else "no_negative" if Nz==0 else None
                        check(record["auc_null_reason"][a]==reason,(name,a,code,"null reason"))
                        if reason:
                            val=None;check(record["auc"][a] is None,(name,a,code,"null"))
                        else:
                            neg=np.sort(ss[~zz]);pos=ss[zz]
                            left=np.searchsorted(neg,pos,side="left");right=np.searchsorted(neg,pos,side="right")
                            val=float(Fraction(int((left+right).sum()),2*Pz*Nz))
                            near(val,record["auc"][a],(name,a,code,"independent pair AUC"))
                        phase_counts[(name,code,a)]=val
                        if a=="original":
                            ref=d50["per_root"][name]["cells"][str(code)]
                            check(ref["auc_null_reason"]["original"]==reason,(name,code,"D50 reason"))
                            if val is not None:near(val,ref["auc"]["original"],(name,code,"D50 AUC"))
            pm=score(y[mask],np.eye(4)[per],False);compare_score(pm,scores["persistence"],(name,part,"persistence"));out[part]["persistence"]={"matched":pm}
            pool[part].append((y,preds,mask,per))
        pairs[name]=out
        print(name,"full-FIT lineage/78-column raw replay/metrics PASS",flush=True)
    totals[h]={}
    for part,items in pool.items():
        target=summary["by_horizon"][f"H{h}"][part]
        y=np.concatenate([v[0] for v in items]);mask=np.concatenate([v[2] for v in items]);per=np.concatenate([v[3] for v in items]);out={}
        for a in ARMS:
            p=np.concatenate([v[1][a] for v in items]);m=score(y,p);mm=score(y[mask],p[mask])
            compare_score(m,target["all_keys_pooled"][a],(h,part,a,"pooled all"));compare_score(mm,target["pooled"][a],(h,part,a,"pooled matched"))
            fold=[pairs[n][part][a] for n in expected if n.startswith(f"h{h}_")]
            for k in ("crisis_f1","macro_f1_fourclass","crisis_brier","logloss_fourclass","crisis_call_share"):near(np.mean([v["matched"][k] for v in fold]),target["mean_fold"][a][k],(h,part,a,k,"mean-fold"))
            for k in ("auc","ap"):near(np.mean([v[k] for v in fold]),target["ranking_mean_fold"][a][k],(h,part,a,k,"mean-fold"))
            out[a]={"all":m,"matched":mm}
        pm=score(y[mask],np.eye(4)[per],False);compare_score(pm,target["pooled"]["persistence"],(h,part,"pooled persistence"));out["persistence"]=pm;totals[h][part]=out
        for a,b in (("h78","original"),("h78","persistence"),("original","persistence")):
            deltas=[]
            for name in expected:
                if not name.startswith(f"h{h}_"):continue
                def exact(arm):
                    c=pairs[name][part][arm]["matched"]["crisis"]
                    den=2*c["tp"]+c["fp"]+c["fn"]
                    return Fraction(2*c["tp"],den) if den else Fraction(0)
                d=exact(a)-exact(b);deltas.append(d)
                check(Fraction(summary["per_root"][name]["scores"][part][f"{a}_minus_{b}"]["crisis_f1_delta_exact"])==d,(name,part,a,b,"exact delta"))
            win=dict(positive=sum(d>0 for d in deltas),negative=sum(d<0 for d in deltas),zero=sum(d==0 for d in deltas),folds=len(deltas))
            check(win==target["fold_wins"][f"{a}_minus_{b}"],(h,part,a,b,"fold signs"))
    for code in range(4):
        for a in ARMS:
            vals=[v for (name,c,arm),v in phase_counts.items() if name.startswith(f"h{h}_") and c==code and arm==a and v is not None]
            obs=summary["by_horizon"][f"H{h}"]["d50_cells_valid_fold_mean"][str(code)]
            check(obs["n_valid"][a]==f"{len(vals)}/7",(h,code,a,"valid folds"))
            if vals:near(np.mean(vals),obs["mean_fold_auc"][a],(h,code,a,"phase mean"))
            else:check(obs["mean_fold_auc"][a] is None,(h,code,a,"null mean"))
result=dict(passed=True,checks=checks,metric_cells=cells,replayed_rows=nrows,pairs=pairs,by_horizon=totals,summary_sha256=sha(R/"summary.json"),identity_sha256=sha(R/"identity.json"),feature_map_sha256=sha(R/"feature_map.json"),script_sha256=sha(Path(__file__)),method="No producer imports/fits. Full-original FIT key/order/label identity, raw UBJ replay from keyed frozen snapshots with independent schema-name selection of78 columns; numpy/sklearn metrics and exact within-phase pair-count AUC.")
OUT.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps({k:result[k] for k in ("passed","checks","metric_cells","replayed_rows")}))
