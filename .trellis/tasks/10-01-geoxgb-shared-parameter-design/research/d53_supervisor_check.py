"""D53 independent saved-row aggregation using csv/math.fsum; no producer imports or fits."""
import csv,gzip,hashlib,json,math,subprocess
from pathlib import Path
from collections import defaultdict
B=Path(r"C:\Users\swl00\geoxgb_runs");D=B/"geoxgb-d52-binary-root-20261002";R=B/"geoxgb-d53-state-probability-transfer-20261002";OUT=B/"d53_supervisor_results.json"
REPO=Path(r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\2.source_code\Step5_Geo_RF_trial\Food_Crisis_Cluster")
assert not OUT.exists()
DATES=("2018-06","2018-10","2019-02","2019-06","2019-10","2020-02","2020-06");GS={4:"G1",8:"G4",12:"G2"};ROLES=("FIT","C","E3");GROUPS=("0","1","2","3","missing");SCORES=("s_original","p_binary")
CONTRASTS=("E3_minus_FIT","E3_minus_C");METRICS=["delta_rate"]+[f"delta_{k}_{s}" for s in SCORES for k in ("mean","bias")]
checks=0;nrows=0;role_ref={};month_ref={};con_ref={}

def check(ok,where):
 global checks
 checks+=1
 assert ok,where

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def near(a,b,where):
 if a is None or b is None:check(a is None and b is None,where)
 else:check(math.isclose(a,b,rel_tol=0,abs_tol=1e-12),where)

def read(p):
 with open(p,encoding="utf-8",newline="") as f:return list(csv.DictReader(f))

def indexed(p,fields):
 rows=read(p);d={tuple(row[f] for f in fields):row for row in rows};check(len(d)==len(rows),(p.name,"unique cells"));return d

def stats(rows):
 n=len(rows);pos=sum(int(x["truth_crisis"]) for x in rows);months={x["target_month"] for x in rows}
 r=dict(n=n,positives=pos,negatives=n-pos,n_unique_areas=len({x["area"] for x in rows}),n_unique_label_months=len(months),min_label_month=min(months) if n else None,max_label_month=max(months) if n else None,rate=pos/n if n else None,support_flag="empty" if not n else "no_positive" if not pos else "no_negative" if pos==n else "both",null_reason=None if n else "empty")
 for s in SCORES:
  m=math.fsum(float(x[s]) for x in rows)/n if n else None
  r.update({f"mean_{s}":m,f"bias_{s}":m-pos/n if n else None,f"brier_{s}":math.fsum((float(x[s])-int(x["truth_crisis"]))**2 for x in rows)/n if n else None})
 return r

def compare(r,observed,where):
 for k,v in r.items():
  raw=observed[k]
  if v is None:check(raw=="",(where,k,"null"))
  elif isinstance(v,int):check(int(raw)==v,(where,k))
  elif isinstance(v,float):near(v,float(raw),(where,k))
  else:check(raw==v,(where,k))

identity=json.loads((R/"identity.json").read_text());completion=json.loads((R/"completion.json").read_text());summary=json.loads((R/"summary.json").read_text());source_record=json.loads((D/"completion.json").read_text())
check(identity["d52_record"]["sha256"]==sha(D/"completion.json"),"D52 completion identity")
check((D/"completion.json").read_bytes()==(REPO/".trellis/tasks/10-01-geoxgb-shared-parameter-design/research/d52_completion.json").read_bytes(),"committed D52 record bytes")
check(identity["script"]["sha256"]==sha(REPO/identity["script"]["path"]),"producer SHA")
check(subprocess.check_output(["git","rev-parse",identity["script"]["head_commit"]+":"+identity["script"]["path"]],cwd=REPO,text=True).strip()==identity["script"]["git_blob"],"producer Git blob")
expected={f"h{h}_{t}_{g}_r80_s42_e1pair/rows_{role}.csv.gz" for h,g in GS.items() for t in DATES for role in ROLES}
check(set(identity["inputs_sha256"])==expected,"63 inputs")
for rel in sorted(expected):check(sha(D/rel)==identity["inputs_sha256"][rel]==source_record["outputs"][rel],(rel,"source hash"))
check(set(completion["outputs"])=={"identity.json","role_cells.csv","month_cells.csv","contrasts.csv","summary.json"},"output inventory")
for rel,digest in completion["outputs"].items():check(sha(R/rel)==digest,(rel,"output SHA"))
role_obs=indexed(R/"role_cells.csv",("root","role","group"));month_obs=indexed(R/"month_cells.csv",("root","role","group","label_month"));con_obs=indexed(R/"contrasts.csv",("root","group","contrast"))
for rel in sorted(expected):
 root,filename=rel.split("/");role=filename.removeprefix("rows_").removesuffix(".csv.gz");h=int(root.split("_")[0][1:]);t=root.split("_")[1]
 with gzip.open(D/rel,"rt",encoding="utf-8",newline="") as f:rows=list(csv.DictReader(f))
 nrows+=len(rows);check(len({(x["area"],x["target_month"]) for x in rows})==len(rows),(rel,"source unique keys"));check(all(x["target_month"]<="2020-12" and int(x["horizon"])==h for x in rows),(rel,"cutoff/horizon"))
 groups=defaultdict(list);by_month=defaultdict(list);months=sorted({x["target_month"] for x in rows})
 for x in rows:
  group=str(int(float(x["persistence_code"]))) if x["persistence_code"] else "missing"
  groups[group].append(x);by_month[(group,x["target_month"])].append(x)
 for group in GROUPS:
  key=(root,role,group);rr=stats(groups[group]);role_ref[key]=rr;compare(rr,role_obs[key],key)
  check(int(role_obs[key]["horizon"])==h and role_obs[key]["target"]==t,(key,"metadata"))
  for month in months:
   mk=key+(month,);mr=stats(by_month[(group,month)]);month_ref[mk]=mr;compare(mr,month_obs[mk],mk)
  mm=[month_ref[key+(m,)] for m in months];check(sum(v["n"] for v in mm)==rr["n"] and sum(v["positives"] for v in mm)==rr["positives"],(key,"month add-back"))
  if rr["n"]:
   for m in ("rate",)+tuple(f"{k}_{s}" for s in SCORES for k in ("mean","bias","brier")):near(math.fsum(v["n"]*v[m] for v in mm if v["n"])/rr["n"],rr[m],(key,m,"weighted recomposition"))
 print(root,role,"independent row/month cells PASS",flush=True)
check(set(role_ref)==set(role_obs) and len(role_ref)==315,"all315 role cells")
check(set(month_ref)==set(month_obs),"all month cells")
roots=sorted({k[0] for k in role_ref})
for root in roots:
 for group in GROUPS[:4]:
  a=role_ref[(root,"E3",group)]
  for contrast in CONTRASTS:
   role=contrast.removeprefix("E3_minus_");b=role_ref[(root,role,group)];key=(root,group,contrast);obs=con_obs[key];empty=[r for r,v in (("E3",a),(role,b)) if not v["n"]]
   check(int(obs["n_E3"])==a["n"] and int(obs["n_reference"])==b["n"],(key,"support"))
   rr={m:None for m in METRICS}
   if empty:check(obs["null_reason"]=="empty_"+"_".join(empty),(key,"null reason"))
   else:
    check(obs["null_reason"]=="",(key,"valid contrast"));rr["delta_rate"]=a["rate"]-b["rate"]
    for s in SCORES:
     for k in ("mean","bias"):rr[f"delta_{k}_{s}"]=a[f"{k}_{s}"]-b[f"{k}_{s}"]
     near(rr[f"delta_bias_{s}"],rr[f"delta_mean_{s}"]-rr["delta_rate"],(key,s,"bias identity"))
   compare(rr,obs,key);con_ref[key]=rr
check(set(con_ref)==set(con_obs) and len(con_ref)==168,"all168 contrasts")
expected_summary={f"h{h}|state{g}|{c}|{m}" for h in GS for g in GROUPS[:4] for c in CONTRASTS for m in METRICS};check(set(summary["cells"])==expected_summary,"all120 summary cells")
for h in GS:
 for group in GROUPS[:4]:
  for contrast in CONTRASTS:
   for m in METRICS:
    k=f"h{h}|state{group}|{contrast}|{m}";obs=summary["cells"][k];valid=[r for r in roots if r.startswith(f"h{h}_") and con_ref[(r,group,contrast)][m] is not None]
    vv=[con_ref[(r,group,contrast)][m] for r in valid];raw=[float(con_obs[(r,group,contrast)][m]) for r in valid]
    near(math.fsum(vv)/len(vv) if vv else None,obs["mean_over_valid_roots"],(k,"independent mean"))
    check(obs["n_valid"]==len(vv) and obs["n_roots"]==7,(k,"valid roots"))
    check((obs["n_positive"],obs["n_negative"],obs["n_zero"])==(sum(v>0 for v in raw),sum(v<0 for v in raw),sum(v==0 for v in raw)),(k,"signs from unrounded recorded contrasts"))
result=dict(passed=True,checks=checks,source_rows=nrows,role_cells=len(role_ref),month_cells=len(month_ref),contrasts=len(con_ref),summary_cells=len(summary["cells"]),summary_sha256=sha(R/"summary.json"),script_sha256=sha(Path(__file__)),method="stdlib csv/gzip/math.fsum, no producer imports/fits; all63 input hashes, role and month cells incl empties, independent support/rate/mean/bias/Brier, weighted month recomposition, all paired contrasts and summary means; signs from validated saved raw deltas.")
OUT.write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result))
