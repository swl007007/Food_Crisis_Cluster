"""Independent D49 check: label-specific sorted-score lookup, no producer import."""
import hashlib, json, math, subprocess, sys
from fractions import Fraction
from pathlib import Path
import numpy as np
import pandas as pd
BASE=Path(r"C:\Users\swl00\geoxgb_runs")
OUT=BASE/"geoxgb-d49-ranking-headroom-20261002"
SRC=BASE/"geoxgb-d38-persistence-margin-root-20261002"
TASK=Path.cwd()/".trellis/tasks/10-01-geoxgb-shared-parameter-design"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text(encoding="utf-8"))
s=load(OUT/"summary.json"); ident=load(OUT/"identity.json")
d39=load(TASK/"research/d39_probability_diagnostic.json"); d40=load(TASK/"research/d40_summary.json")
checks=0
def ck(ok,msg):
    global checks
    checks+=1
    if not ok: raise AssertionError(msg)
def ff(tp,fp,fn):
    d=2*int(tp)+int(fp)+int(fn)
    return Fraction(2*int(tp),d) if d else Fraction(0)
def cf(z,y):
    tp=int(np.sum(z&y)); fp=int(np.sum(~z&y)); fn=int(np.sum(z&~y)); tn=int(np.sum(~z&~y))
    f=ff(tp,fp,fn)
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,f1_exact=str(f),f1=float(f))
ck(ident["head"].startswith("879c335"),"producer head")
ck(sha(Path(ident["script"]))==ident["script_sha256"]==s["script_sha256"],"producer bytes")
ck(ident["git_blob"]==subprocess.check_output(["git","hash-object",ident["script"]],text=True).strip(),"producer blob")
ck(ident["input_hashes"]==d39["input_hashes"]==d40["input_hashes"],"hash records")
ck(set(s["per_root"])==set(ident["input_hashes"]) and len(s["per_root"])==21,"root inventory")
ft=pd.read_csv(OUT/"frontier.csv.gz",float_precision="round_trip")
ck(not ft.duplicated(["root","arm","idx"]).any(),"frontier unique keys")
ck(set(zip(ft.root,ft.arm))=={(r,a) for r in s["per_root"] for a in ("original","anchored")},"frontier inventory")
agg={h:[] for h in (4,8,12)}; nrows=0; excluded=0; endpoints=0; ranges={}
for root, rec in s["per_root"].items():
    path=SRC/root/"rows_E3.csv.gz"
    ck(sha(path)==ident["input_hashes"][root],root+" input hash")
    df=pd.read_csv(path,float_precision="round_trip"); nrows+=len(df)
    ck(not df.duplicated(["area","target_month","horizon"]).any(),root+" unique")
    ck(df.target_month.nunique()==1 and df.horizon.nunique()==1 and (df.target_month<="2020-12").all(),root+" temporal")
    mask=np.isfinite(df.persistence_code.to_numpy(float)); g=df[mask]; nx=int((~mask).sum()); excluded+=nx
    ck(nx==rec["excluded_missing_origin"]==d39["per_root"][root]["missing_origin_n"],root+" exclusions")
    z=g.truth.to_numpy()>=2; per=cf(z,g.persistence_code.to_numpy()>=2)
    ck(per==rec["persistence"],root+" persistence")
    ck(rec["n"]==len(g) and rec["positives"]==int(z.sum()),root+" support")
    ck(rec["horizon"]==int(g.horizon.iloc[0]) and rec["target_month"]==g.target_month.iloc[0],root+" fold metadata")
    calc=dict(persistence=per,arms={}); agg[rec["horizon"]].append(calc)
    for arm, ar in rec["arms"].items():
        pp=df[[f"p_{arm}_{c}" for c in ("1","2","3","4或5")]].to_numpy(float)
        ck(np.isfinite(pp).all() and (pp>=0).all() and (pp.sum(axis=1)>0).all(),root+arm+" probabilities")
        ck(np.array_equal(pp.argmax(axis=1),df[f"y_{arm}"].to_numpy()),root+arm+" argmax saved")
        arg=cf(z,pp[mask].argmax(axis=1)>=2); ck(arg==ar["argmax"],root+arm+" argmax confusion")
        for who,c in ((arm,arg),("persistence",per)):
            ck(all(c[k]==d39["per_root"][root]["matched"][who]["argmax_crisis"][k] for k in ("tp","fp","fn","tn")),root+who+" D39 confusion")
        p=pp[mask]; score=(p[:,2]+p[:,3])/p.sum(axis=1)
        cuts=np.unique(score)[::-1]
        # Independent of producer cumulative tie-block sum: binary search each class separately.
        pos=np.sort(score[z]); neg=np.sort(score[~z])
        tp=np.r_[0,len(pos)-np.searchsorted(pos,cuts,side="left")]
        fp=np.r_[0,len(neg)-np.searchsorted(neg,cuts,side="left")]
        fn=len(pos)-tp; calls=tp+fp
        obs=ft[(ft.root==root)&(ft.arm==arm)].sort_values("idx").reset_index(drop=True)
        ck(len(obs)==len(cuts)+1==ar["n_endpoints"],root+arm+" endpoint count"); endpoints+=len(obs)
        for field,expected in (("idx",np.arange(len(obs))),("tp",tp),("fp",fp),("fn",fn),("calls",calls)):
            ck(np.array_equal(obs[field].to_numpy(),expected),root+arm+field)
        ck(obs.kind.iloc[0]=="none" and obs.cutoff.isna().iloc[0] and obs.cutoff_hex.isna().iloc[0],root+arm+" none")
        ck((obs.kind.iloc[1:]=="cutoff").all() and np.array_equal(obs.cutoff.iloc[1:].to_numpy(),cuts),root+arm+" cutoffs")
        ck(obs.cutoff_hex.iloc[1:].tolist()==[float(c).hex() for c in cuts],root+arm+" exact cutoffs")
        vals=[ff(t,f,n) for t,f,n in zip(tp,fp,fn)]; best=max(vals); winner=vals.index(best)
        def pt(i):
            return dict(kind="none" if i==0 else "cutoff",cutoff=None if i==0 else float(cuts[i-1]),cutoff_hex=None if i==0 else float(cuts[i-1]).hex(),calls=int(calls[i]),tp=int(tp[i]),fp=int(fp[i]),fn=int(fn[i]),f1_exact=str(vals[i]),f1=float(vals[i]))
        optimum=dict(f1_exact=str(best),f1=float(best),n_optimal=vals.count(best),winner=pt(winner))
        dom=np.flatnonzero((tp>=per["tp"])&(fp<=per["fp"])&((tp>per["tp"])|(fp<per["fp"])))
        dominance=dict(dominates=bool(len(dom)),count=len(dom),witness=pt(dom[0]) if len(dom) else None)
        kp=per["tp"]+per["fp"]; i=int(np.searchsorted(calls,kp))
        budget=dict(k_p=kp,status="exact",point=pt(i)) if calls[i]==kp else dict(k_p=kp,status="bracket",below=pt(i-1),above=pt(i))
        ck(optimum==ar["optimum"],root+arm+" optimum/ties")
        ck(dominance==ar["dominance"],root+arm+" dominance")
        ck(budget==ar["budget"],root+arm+" budget")
        calc["arms"][arm]=dict(argmax=arg,optimum=optimum,dominance=dominance,budget=budget)
        ranges.setdefault((rec["horizon"],arm),[]).append(float(cuts[winner-1]) if winner else None)
ck(excluded==713==s["excluded_missing_origin_total"],"total excluded")
for h,rs in agg.items():
    ck(len(rs)==7,"seven folds per H")
    mean=lambda xs:sum(xs)/len(xs) if xs else None
    expected=dict(n_folds=7,persistence_f1=mean([r["persistence"]["f1"] for r in rs]),arms={})
    for arm in ("original","anchored"):
        a=[r["arms"][arm] for r in rs]; pf=[r["persistence"]["f1"] for r in rs]
        ex=[x["budget"]["point"]["tp"]-r["persistence"]["tp"] for x,r in zip(a,rs) if x["budget"]["status"]=="exact"]
        expected["arms"][arm]=dict(argmax_f1=mean([x["argmax"]["f1"] for x in a]),hindsight_max_f1=mean([x["optimum"]["f1"] for x in a]),gap_optimum_minus_persistence=mean([x["optimum"]["f1"]-p for x,p in zip(a,pf)]),gap_optimum_minus_argmax=mean([x["optimum"]["f1"]-x["argmax"]["f1"] for x in a]),gap_argmax_minus_persistence=mean([x["argmax"]["f1"]-p for x,p in zip(a,pf)]),folds_optimum_gt_persistence=sum(Fraction(x["optimum"]["f1_exact"])>Fraction(r["persistence"]["f1_exact"]) for x,r in zip(a,rs)),folds_dominating_persistence=sum(x["dominance"]["dominates"] for x in a),budget_exact=len(ex),budget_bracket=7-len(ex),mean_budget_tp_minus_persistence_tp_exact=mean(ex))
    def compare(a,b):
        if isinstance(a,dict):
            ck(a.keys()==b.keys(),"aggregation fields")
            for k in a: compare(a[k],b[k])
        elif isinstance(a,float): ck(math.isclose(a,b,rel_tol=0,abs_tol=1e-14),"aggregation value")
        else: ck(a==b,"aggregation value")
    compare(expected,s["by_horizon_mean_fold"][str(h)])
report=dict(status="PASS",checks=checks,source_rows=nrows,matched_rows=nrows-excluded,excluded=excluded,frontier_endpoints=endpoints,producer_head=ident["head"],verifier_sha256=sha(Path(__file__)),method="independent positive/negative sorted-score searchsorted per unique cutoff; no producer imports; zero fits",artifacts={f:sha(OUT/f) for f in ("summary.json","identity.json","frontier.csv.gz")},by_horizon_mean_fold=s["by_horizon_mean_fold"],runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__))
(BASE/"d49_supervisor_verification.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
print(json.dumps(report,indent=2))
print("Optimal cutoff ranges (descriptive only):",{str(k):(min(v),max(v)) for k,v in ranges.items()})
