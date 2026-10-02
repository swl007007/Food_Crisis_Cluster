"""Independent D52 snapshot lineage, UBJ replay and binary metric verification. No producer imports or fits."""
import hashlib, json, platform, subprocess, sys
from pathlib import Path
from fractions import Fraction
import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn
from sklearn.metrics import roc_auc_score, average_precision_score
B=Path(r"C:\Users\swl00\geoxgb_runs")
D=B/"geoxgb-d34-e1-brier-20261002"; S=D/"stage1_e1pair"
R=B/"geoxgb-d52-binary-root-20261002"; OUT=B/"d52_supervisor_results.json"
REPO=Path(r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\2.source_code\Step5_Geo_RF_trial\Food_Crisis_Cluster")
P=REPO/"FEWSNETGeoXGBExperiment"; TASK=REPO/".trellis/tasks/10-01-geoxgb-shared-parameter-design/research"
assert not OUT.exists() and sys.flags.optimize==0
assert (platform.python_version(),np.__version__,pd.__version__,xgb.__version__,sklearn.__version__)==("3.12.10","2.2.6","2.2.3","3.0.0","1.6.1")
DATES=("2018-06","2018-10","2019-02","2019-06","2019-10","2020-02","2020-06")
GS={4:"G1",8:"G4",12:"G2"}; LABELS=("1","2","3","4或5")
ARMS=("binary","original_mass","original_argmax"); ALL=ARMS+("persistence",)
CONTRASTS=(("binary","original_mass"),("binary","original_argmax"),("binary","persistence"),("original_argmax","persistence"))
FEATURES=json.loads((P/"feature-schema.json").read_text())["ordered_features"]
summary=json.loads((R/"summary.json").read_text()); identity=json.loads((R/"identity.json").read_text())
gate=json.loads((R/"gate.json").read_text()); completion=json.loads((R/"completion.json").read_text())
d49=json.loads((TASK/"d49_summary.json").read_text()); d50=json.loads((TASK/"d50_summary.json").read_text())
checks=0; cells=0; nrows=0

def check(ok,where):
    global checks
    checks+=1
    assert ok,where

def mi(t):return int(t[:4])*12+int(t[5:])-1

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def labelsha(y):return hashlib.sha256(np.asarray(y,np.int64).tobytes()).hexdigest()

def read(p):return pd.read_csv(p,float_precision="round_trip")

def near(a,b,where):
    if a is None or b is None:check(a is None and b is None,where)
    else:check(np.allclose(a,b,rtol=0,atol=1e-12),where)

def score(z,c,s=None):
    z=np.asarray(z,bool);c=np.asarray(c,bool)
    tp,fp,fn,tn=(int(v.sum()) for v in (z&c,~z&c,z&~c,~z&~c))
    exact=Fraction(2*tp,2*tp+fp+fn) if 2*tp+fp+fn else Fraction(0)
    out=dict(n=len(z),tp=tp,fp=fp,fn=fn,tn=tn,crisis_f1_exact=str(exact),crisis_f1=float(exact),crisis_call_share=float(c.mean()))
    if s is not None:
        eps=np.finfo(float).eps;clip=np.clip(s,eps,1-eps)
        out.update(crisis_brier=float(np.mean((s-z)**2)),logloss_binary=float(-np.mean(z*np.log(clip)+(~z)*np.log1p(-clip))),clipped_low=int((s<eps).sum()),clipped_high=int((s>1-eps).sum()),eps=float(eps))
    return out

def compare(m,s,where):
    global cells
    for k in ("n","tp","fp","fn","tn","crisis_f1_exact","clipped_low","clipped_high"):
        if k in s:check(m[k]==s[k],(where,k))
    for k in ("crisis_f1","crisis_call_share","crisis_brier","logloss_binary","eps"):
        if k in s:near(m[k],s[k],(where,k))
    cells+=1

def rank(z,s):
    if len(z)==0 or np.unique(z).size<2:return dict(eligible=False,auc=None,ap=None)
    return dict(eligible=True,auc=float(roc_auc_score(z,s)),ap=float(average_precision_score(z,s)))

def reference(y,p):
    mat=np.bincount(y*4+p.argmax(1),minlength=16).reshape(4,4); den=mat.sum(0)+mat.sum(1)
    return dict(macro_f1_fourclass=float(np.divide(2*np.diag(mat),den,out=np.zeros(4),where=den!=0).mean()),logloss_fourclass=float(-np.log(p[np.arange(len(y)),y]).mean()))

check(len(FEATURES)==162 and FEATURES[87]=="hist_phase_o00","full feature schema")
check(identity["max_month"]=="2020-12" and identity["threshold"]==.5,"cutoff and threshold")
check(identity["script"]["sha256"]==sha(REPO/identity["script"]["path"]),"producer bytes")
check(subprocess.check_output(["git","rev-parse",identity["repo_head"]+":"+identity["script"]["path"]],cwd=REPO,text=True).strip()==identity["script"]["git_blob"],"committed producer blob")
for d in ("d49","d50"):check(identity["consistency_sources"][d]==sha(TASK/(d+"_summary.json")),d+" identity")
check(gate["passed"] and completion["status"]=="completed","run completed")
expected={f"h{h}_{t}_{g}_r80_s42_e1pair" for h,g in GS.items() for t in DATES}
check(set(summary["per_root"])==set(gate["roots"])==expected,"21 expected roots")
expected_outputs={"gate.json","summary.json"}|{f"{n}/{f}" for n in expected for f in ("binary_root.json","binary_root.ubj","rows_FIT.csv.gz","rows_C.csv.gz","rows_E3.csv.gz")}
check(set(completion["outputs"])==expected_outputs,"complete output inventory")
check({p.relative_to(R).as_posix() for p in R.rglob("*") if p.is_file()}==expected_outputs|{"identity.json","completion.json"},"actual output inventory")
for rel,digest in completion["outputs"].items():check(sha(R/rel)==digest,(rel,"output hash"))
for name in expected:
    for phase in ("gate_pass","fit_pass"):
        gg=gate["roots"][name][phase]
        check(gg["passed"] and all(v["mismatches"]==0 for v in gg["checks"].values()),(name,phase,"all gate checks"))
pairs={};totals={};phase_counts={}
for h,g in GS.items():
    snapfile=D/"prepared"/f"snapshot_h{h}.parquet"
    snap=pd.read_parquet(snapfile,columns=["area","target_month","class_code"]+FEATURES,filters=[("target_month","<=",mi("2020-12"))]).set_index(["area","target_month"])
    check(snap.index.is_unique,(h,"snapshot unique"));sh=sha(snapfile);pool={p:[] for p in ("FIT","C","E3")}
    for t in DATES:
        name=f"h{h}_{t}_{g}_r80_s42_e1pair";rd=R/name;root=S/"roots"/name
        meta=json.loads((rd/"binary_root.json").read_text());old=json.loads((root/"root.json").read_text());rec=meta["fit_record"]
        mem=read(root/"fold_membership.csv.gz");mem["month"]=mem.target_month.map(mi)
        check(not mem.duplicated(["area","month"]).any(),(name,"unique membership"));fm=mem[mem.role=="fitting"]
        check(labelsha(fm[["area","month"]].to_numpy())==meta["fitting_keys_sha256"]==old["fitting_keys_sha256"],(name,"all FIT keys and order"))
        check(labelsha(fm.class_code)==meta["fitting_four_class_labels_sha256"]==rec["four_class_labels_sha256"],(name,"original labels"))
        check(labelsha((fm.class_code.to_numpy()>=2).astype(int))==meta["fitting_binary_labels_sha256"]==rec["binary_labels_sha256"],(name,"derived binary labels"))
        check(meta["fitting_rows"]==rec["rows"]==len(fm),(name,"all FIT rows"))
        check(rec["four_class_counts"]==[int((fm.class_code==k).sum()) for k in range(4)] and rec["binary_counts"]==[int((fm.class_code<2).sum()),int((fm.class_code>=2).sum())],(name,"fit class support"))
        origin=mi(t)-h;check(fm.month.between(origin-59,origin-1).all() and (mem.month<=mi("2020-12")).all(),(name,"time boundary"))
        params={k:v for k,v in old["root_fit"]["params"].items() if k not in ("num_class","multi_strategy")};params.update(objective="binary:logistic",base_score=.5)
        check(rec["params"]==params and rec["four_class_params"]==old["root_fit"]["params"],(name,"approved param changes only"))
        check("sample_weight" not in rec and "base_margin" not in rec,(name,"no weight or margin record"))
        check(sh==identity["inputs"][name]["snapshot"],(name,"snapshot hash"))
        for fn in ("root.json","fold_membership.csv.gz","root_target_predictions.csv"):check(sha(root/fn)==identity["inputs"][name][fn],(name,fn,"hash"))
        models={}
        for a,path in (("original",S/"checkpoints"/f"h{h}_{t}_{g}_L1_r80_s42_e1brier_gt0"/"xgb_root.ubj"),("binary",rd/"binary_root.ubj")):
            check(sha(path)==meta[a+"_root_sha256"],(name,a,"model hash"))
            m=xgb.Booster();m.load_model(path);models[a]=m;cfg=json.loads(m.save_config())["learner"]
            check(float(cfg["learner_model_param"]["base_score"])==.5 and int(cfg["learner_model_param"]["num_class"])==(4 if a=="original" else 0),(name,a,"base/classes"))
            check(cfg["objective"]["name"]==("multi:softprob" if a=="original" else "binary:logistic"),(name,a,"objective"))
            check(m.num_boosted_rounds()==(200 if h==4 else 400) and m.num_features()==162,(name,a,"rounds/features"))
            trees=json.loads(bytes(m.save_raw("json")))["learner"]["gradient_booster"]["model"]["trees"]
            check(len(trees)==(200 if h==4 else 400)*(4 if a=="original" else 1),(name,a,"tree capacity"))
        out={}
        for role,part in (("fitting","FIT"),("confirmation","C"),("heldout_target","E3")):
            keys=mem[mem.role==role];data=snap.loc[list(zip(keys.area,keys.month))];y=data.class_code.to_numpy(int);z=y>=2
            check(np.array_equal(y,keys.class_code),(name,part,"snapshot labels"));X=data[FEATURES].to_numpy(float);X[np.isinf(X)]=np.nan
            dm=xgb.DMatrix(X,missing=np.nan,nthread=4);preds={a:m.predict(dm).astype(float) for a,m in models.items()};p=preds["original"];b=preds["binary"]
            check(p.shape==(len(y),4) and b.shape==(len(y),) and all(np.isfinite(v).all() and (v>=0).all() and (v<=1).all() for v in preds.values()),(name,part,"probability shapes/ranges"))
            s=p[:,2:].sum(1)/p.sum(1);rows=read(rd/f"rows_{part}.csv.gz");nrows+=len(rows)
            check(np.array_equal(rows.area,keys.area) and np.array_equal(rows.target_month,keys.target_month) and np.array_equal(rows.truth,y) and np.array_equal(rows.truth_crisis,z),(name,part,"saved keys/order/truth"))
            check(np.array_equal(rows[[f"p_original_{lab}" for lab in LABELS]],p) and np.array_equal(rows.p_binary,b) and np.array_equal(rows.s_original,s),(name,part,"raw UBJ replay exact"))
            phase=data.hist_phase_o00.to_numpy(float);mask=np.isfinite(phase);check(np.isin(phase[mask],[1,2,3,4,5]).all(),(name,part,"phase domain"));per=np.minimum(phase[mask],4).astype(int)-1
            check(np.array_equal(rows.persistence_code.notna(),mask) and np.array_equal(rows.persistence_code[mask],per),(name,part,"persistence from original features"))
            calls={"binary":b>=.5,"original_mass":s>=.5,"original_argmax":p.argmax(1)>=2};scores={"binary":b,"original_mass":s,"original_argmax":s};obs=summary["per_root"][name]["scores"][part];out[part]={}
            check((obs["n_all"],obs["n"],obs["excluded_missing_origin"])==(len(y),int(mask.sum()),int((~mask).sum())),(name,part,"coverage"))
            for a in ARMS:
                check(np.array_equal(rows["call_"+a],calls[a]),(name,part,a,"hard call"));ma=score(z,calls[a],scores[a]);mm=score(z[mask],calls[a][mask],scores[a][mask]);compare(ma,obs["all_keys"][a],(name,part,a,"all"));compare(mm,obs[a],(name,part,a,"matched"));out[part][a]={"all":ma,"matched":mm}
            check(np.array_equal(rows.y_original_argmax,p.argmax(1)) and obs["binary"]["macro_f1_fourclass"] is None,(name,part,"fourclass reference scope"))
            for cohort,yy,pp in (("all_keys_original_reference",y,p),("original_reference",y[mask],p[mask])):
                for k,v in reference(yy,pp).items():near(v,obs[cohort][k],(name,part,cohort,k))
            out[part]["reference"]=reference(y[mask],p[mask])
            pm=score(z[mask],per>=2,(per>=2).astype(float));compare(pm,obs["persistence"],(name,part,"persistence"));check("logloss_binary" not in obs["persistence"],(name,part,"persistence Brier only"));out[part]["persistence"]={"matched":pm}
            out[part]["ranking"]={}
            for a,ss in (("original",s),("binary",b)):
                ranking=rank(z[mask],ss[mask]);out[part]["ranking"][a]=ranking;check(ranking["eligible"]==obs["ranking"][a]["eligible"],(name,part,a,"rank eligibility"))
                for k in ("auc","ap"):near(ranking[k],obs["ranking"][a][k],(name,part,a,k))
                if part=="E3":
                    for c in range(4):
                        k=per==c;zz=z[mask][k];sc=ss[mask][k];pos=int(zz.sum());neg=len(zz)-pos;cell=summary["per_root"][name]["d50_cells"][str(c)]
                        reason="empty" if len(zz)==0 else "no_positive" if pos==0 else "no_negative" if neg==0 else None
                        check((cell["n"],cell["P"],cell["N"])==(len(zz),pos,neg) and cell["auc_null_reason"][a]==reason,(name,a,c,"phase support/null"))
                        val=None
                        if reason is None:
                            ns=np.sort(sc[~zz]);ps=sc[zz];val=float(Fraction(int((np.searchsorted(ns,ps,side="left")+np.searchsorted(ns,ps,side="right")).sum()),2*pos*neg))
                        near(val,cell["auc"][a],(name,a,c,"pair-count phase AUC"));phase_counts[(name,c,a)]=val
                        if a=="original":near(val,d50["per_root"][name]["cells"][str(c)]["auc"][a],(name,c,"D50 consistency"))
            for a,bb in CONTRASTS:
                delta=Fraction(out[part][a]["matched"]["crisis_f1_exact"])-Fraction(out[part][bb]["matched"]["crisis_f1_exact"])
                check(str(delta)==obs[a+"_minus_"+bb]["crisis_f1_delta_exact"],(name,part,a,bb,"exact delta"))
            if part=="E3":
                for a,ref in (("original_argmax",d49["per_root"][name]["arms"]["original"]["argmax"]),("persistence",d49["per_root"][name]["persistence"])):
                    check(all(out[part][a]["matched"][k]==ref[k] for k in ("tp","fp","fn","tn")),(name,a,"D49 confusion"))
            pool[part].append((z,calls,scores,mask,per))
        pairs[name]=out;print(name,"full-FIT lineage/binary+original UBJ replay/metrics PASS",flush=True)
    totals[h]={}
    for part,items in pool.items():
        obs=summary["by_horizon"][f"H{h}"][part];zz=np.concatenate([x[0] for x in items]);mask=np.concatenate([x[3] for x in items]);per=np.concatenate([x[4] for x in items]);agg={}
        check(obs["rows"]==int(mask.sum()) and obs["coverage"]["n_all_pooled"]==len(zz) and obs["coverage"]["excluded_missing_origin_pooled"]==int((~mask).sum()),(h,part,"pooled coverage"))
        for a in ALL:
            if a=="persistence":mm=score(zz[mask],per>=2,(per>=2).astype(float))
            else:
                call=np.concatenate([x[1][a] for x in items]);sc=np.concatenate([x[2][a] for x in items]);ma=score(zz,call,sc);compare(ma,obs["all_keys_pooled"][a],(h,part,a,"pooled all"));mm=score(zz[mask],call[mask],sc[mask])
            compare(mm,obs["pooled_crisis_confusion"][a],(h,part,a,"pooled matched"));agg[a]=mm
            folds=[pairs[n][part][a]["matched"] for n in sorted(expected) if n.startswith(f"h{h}_")]
            for k in ("crisis_f1","crisis_call_share","crisis_brier","logloss_binary"):
                if a=="persistence" and k=="logloss_binary":check(obs["mean_fold"][a][k] is None,(h,part,a,k))
                else:near(np.mean([v[k] for v in folds]),obs["mean_fold"][a][k],(h,part,a,k,"mean-fold"))
        for a in ("binary","original"):
            ranks=[pairs[n][part]["ranking"][a] for n in expected if n.startswith(f"h{h}_") and pairs[n][part]["ranking"][a]["eligible"]]
            check(len(ranks)==obs["ranking_mean_fold"][a]["eligible_folds"],(h,part,a,"rank eligible folds"))
            for k in ("auc","ap"):near(float(np.mean([v[k] for v in ranks])) if ranks else None,obs["ranking_mean_fold"][a][k],(h,part,a,k,"mean-fold"))
        for k in ("macro_f1_fourclass","logloss_fourclass"):near(np.mean([pairs[n][part]["reference"][k] for n in expected if n.startswith(f"h{h}_")]),obs["mean_fold"]["original_reference"][k],(h,part,k,"reference mean"))
        for a,bb in CONTRASTS:
            ds=[Fraction(pairs[n][part][a]["matched"]["crisis_f1_exact"])-Fraction(pairs[n][part][bb]["matched"]["crisis_f1_exact"]) for n in expected if n.startswith(f"h{h}_")];key=a+"_minus_"+bb
            check(dict(wins=sum(d>0 for d in ds),ties=sum(d==0 for d in ds),losses=sum(d<0 for d in ds),folds=len(ds))==obs["fold_wins"][key],(h,part,key,"fold wins"))
            near(float(sum(ds)/len(ds)),obs["mean_fold_deltas"][key]["crisis_f1_delta"],(h,part,key,"mean delta"))
            near(float(Fraction(agg[a]["crisis_f1_exact"])-Fraction(agg[bb]["crisis_f1_exact"])),obs["pooled_deltas"][key]["crisis_f1_delta"],(h,part,key,"pooled delta"))
        totals[h][part]=agg
    for c in range(4):
        obs=summary["by_horizon"][f"H{h}"]["d50_cells_valid_fold_mean"][str(c)]
        for a in ("binary","original"):
            vals=[v for (n,cc,aa),v in phase_counts.items() if n.startswith(f"h{h}_") and cc==c and aa==a and v is not None]
            check(obs["n_valid"][a]==f"{len(vals)}/7",(h,c,a,"valid fold count"));near(float(np.mean(vals)) if vals else None,obs["mean_fold_auc"][a],(h,c,a,"phase mean"))
result=dict(passed=True,checks=checks,metric_cells=cells,replayed_rows=nrows,pairs=pairs,by_horizon=totals,summary_sha256=sha(R/"summary.json"),identity_sha256=sha(R/"identity.json"),script_sha256=sha(Path(__file__)),method="No producer imports/fits. Independent keyed snapshots (cutoff2020-12), full FIT key/order/label hashes, binary label mapping, params, both raw UBJs, all/matched binary metrics, exact F1 contrasts and within-phase pair-count AUC.")
OUT.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps({k:result[k] for k in ("passed","checks","metric_cells","replayed_rows")}))
