"""D50 independent direct positive-negative pair-count verification. No sklearn or producer imports."""
from pathlib import Path
from fractions import Fraction
import hashlib,json,math,sys,subprocess
import numpy as np
import pandas as pd
BASE=Path(r"C:\Users\swl00\geoxgb_runs")
OUT=BASE/"geoxgb-d50-origin-phase-ranking-20261002"
SRC=BASE/"geoxgb-d38-persistence-margin-root-20261002"
R=Path.cwd()/".trellis/tasks/10-01-geoxgb-shared-parameter-design/research"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text(encoding="utf-8"))
s=load(OUT/"summary.json"); ident=load(OUT/"identity.json"); d49=load(R/"d49_summary.json")
checks=0; pair_comparisons=0; counts=[]; nrows=0; missing=0
def ck(ok,msg):
    global checks
    checks+=1
    if not ok:raise AssertionError(msg)
def cm(z,y):
    return {k:int(a.sum()) for k,a in zip(("tp","fp","fn","tn"),(z&y,~z&y,z&~y,~z&~y))}
def pair_auc(scores,z):
    global pair_comparisons
    pos=scores[z]; neg=scores[~z]; den=2*len(pos)*len(neg)
    if not len(z):return None,"empty",0,0
    if not len(pos):return None,"no_positive",0,0
    if not len(neg):return None,"no_negative",0,0
    num=0
    for i in range(0,len(pos),128):
        p=pos[i:i+128,None]
        num+=2*int((p>neg).sum())+int((p==neg).sum())
    pair_comparisons+=len(pos)*len(neg)
    return float(Fraction(num,den)),None,num,den
ck(pair_auc(np.array([.5,.9,.5,.1]),np.array([1,1,0,0],bool))[:2]==(.875,None),"verifier tied check")
ck(pair_auc(np.array([.3]*4),np.array([1,0,1,0],bool))[:2]==(.5,None),"verifier constant check")
pair_comparisons=0
ck(sha(Path(ident["script"]))==ident["script_sha256"]==s["script_sha256"],"producer bytes")
ck(subprocess.check_output(["git","rev-parse",ident["head"]+":"+ident["script"]],text=True).strip()==ident["git_blob"],"producer at recorded SHA")
for fn in ("d39_probability_diagnostic.json","d40_summary.json","d49_identity.json"):
    ck(load(R/fn)["input_hashes"]==ident["input_hashes"],fn+" hashes")
ck(set(s["per_root"])==set(ident["input_hashes"])==set(d49["per_root"]) and len(s["per_root"])==21,"inventory")
records={}
for root,r in s["per_root"].items():
    f=SRC/root/"rows_E3.csv.gz"; ck(sha(f)==ident["input_hashes"][root],root+" hash")
    df=pd.read_csv(f,float_precision="round_trip"); nrows+=len(df)
    ck(not df.duplicated(["area","target_month","horizon"]).any(),root+" unique")
    ck(df.target_month.nunique()==1 and df.horizon.nunique()==1 and (df.target_month<="2020-12").all(),root+" single fold")
    ck(int(df.horizon.iloc[0])==r["horizon"] and df.target_month.iloc[0]==r["target_month"],root+" metadata")
    k=np.isfinite(df.persistence_code.to_numpy(float)); g=df[k]; pc=g.persistence_code.to_numpy(float); z=g.truth.to_numpy()>=2
    nx=int((~k).sum()); missing+=nx
    ck(nx==r["excluded_missing_origin"]==d49["per_root"][root]["excluded_missing_origin"] and len(g)==r["n"]==d49["per_root"][root]["n"],root+" cohort")
    ck(np.isin(pc,[0,1,2,3]).all() and np.array_equal(pc+1,g.origin_phase.to_numpy()),root+" exact origin")
    preds={"persistence":pc>=2}; scores={}
    for arm in ("original","anchored"):
        p=df[[f"p_{arm}_{v}" for v in ("1","2","3","4或5")]].to_numpy(float)
        ck(np.isfinite(p).all() and (p>=0).all() and (p.sum(axis=1)>0).all(),root+arm+" probabilities")
        ck(np.array_equal(p.argmax(axis=1),df[f"y_{arm}"].to_numpy()),root+arm+" argmax")
        preds[arm]=p[k].argmax(axis=1)>=2
        scores[arm]=(p[k,2]+p[k,3])/p[k].sum(axis=1)
    sums={m:{v:0 for v in ("tp","fp","fn","tn")} for m in preds}
    ck(set(r["cells"])=={"0","1","2","3"},root+" all cells")
    rec={}
    for code in range(4):
        m=pc==code; y=z[m]; cell=r["cells"][str(code)]; P=int(y.sum()); N=len(y)-P
        ck((cell["n"],cell["P"],cell["N"],cell["origin_phase"])==(len(y),P,N,code+1),root+str(code)+" support")
        vals={}
        for arm in ("original","anchored"):
            v,reason,num,den=pair_auc(scores[arm][m],y)
            ck(reason==cell["auc_null_reason"][arm],root+arm+" reason")
            ck(cell["auc"][arm] is None if v is None else cell["auc"][arm] is not None and abs(v-cell["auc"][arm])<=1e-12,root+arm+" auc")
            vals[arm]=v; counts.append(dict(root=root,code=code,arm=arm,numerator=num,denominator=den,reason=reason))
        for model,pred in preds.items():
            expected=cm(y,pred[m]); ck(expected==cell["confusion"][model],root+model+" confusion")
            for key in expected:sums[model][key]+=expected[key]
        rec[str(code)]=dict(P=P,N=N,auc=vals)
    records[root]=dict(horizon=r["horizon"],target_month=r["target_month"],cells=rec)
    for model in preds:
        ref=d49["per_root"][root]["persistence"] if model=="persistence" else d49["per_root"][root]["arms"][model]["argmax"]
        ck(all(sums[model][v]==ref[v] for v in sums[model]),root+model+" D49 addback")
ck(nrows==113508 and missing==713 and s["known_total"]==nrows-missing==112795 and s["excluded_missing_origin_total"]==missing,"total")
expected_by_date={}
for root,r in s["per_root"].items():expected_by_date.setdefault(r["target_month"],{})[root]={"horizon":r["horizon"],"cells":r["cells"]}
ck(expected_by_date==s["by_date"],"date view complete")
ck(set(s["by_horizon_mean_fold_valid_only"])=={"4","8","12"},"H inventory")
for h in (4,8,12):
    rs=sorted([r for r in records.values() if r["horizon"]==h],key=lambda r:r["target_month"])
    ck(len(rs)==7 and len({r["target_month"] for r in rs})==7,"7 dates/H")
    for code in range(4):
        key=str(code); obs=s["by_horizon_mean_fold_valid_only"][str(h)][key]
        supports=[dict(date=r["target_month"],P=r["cells"][key]["P"],N=r["cells"][key]["N"],valid=r["cells"][key]["P"]*r["cells"][key]["N"]>0) for r in rs]
        ck(supports==obs["supports"] and obs["n_folds"]==7 and obs["n_valid"]==sum(v["valid"] for v in supports) and obs["origin_phase"]==code+1,"H supports")
        for arm in ("original","anchored"):
            a=[r["cells"][key]["auc"][arm] for r in rs if r["cells"][key]["auc"][arm] is not None]
            expected=sum(a)/len(a) if a else None; actual=obs["mean_fold_auc"][arm]
            ck(actual is None if expected is None else math.isclose(actual,expected,abs_tol=1e-12,rel_tol=0),"H means")
report=dict(status="PASS",checks=checks,pair_comparisons=pair_comparisons,source_rows=nrows,matched_rows=nrows-missing,excluded=missing,root_cells=84,arm_cells=len(counts),producer_head=ident["head"],verifier_sha256=sha(Path(__file__)),method="direct positive-negative comparisons in128-row chunks, wins=1 ties=0.5; exact integer numerator; no sklearn/producer imports; zero fits",artifacts={f:sha(OUT/f) for f in ("summary.json","identity.json")},exact_pair_counts=counts,by_horizon_mean_fold_valid_only=s["by_horizon_mean_fold_valid_only"],runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__))
(BASE/"d50_supervisor_verification.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k not in ("exact_pair_counts","by_horizon_mean_fold_valid_only")},indent=2))
for h,phases in report["by_horizon_mean_fold_valid_only"].items():
    for code,r in phases.items():print(h,'phase',int(code)+1,'valid',r['n_valid'],r['mean_fold_auc'],'P/N',[(v['P'],v['N']) for v in r['supports']])
