"""Cross-check independent D44 original-file calculations against executor output."""
import json, math, hashlib, csv, gzip
from pathlib import Path
B=Path(r"C:\Users\swl00\geoxgb_runs")
D=B/"geoxgb-d44-full-pool-root-20261002"
ref=json.loads((B/"d44_supervisor_results.json").read_text())
s=json.loads((D/"summary.json").read_text())
idn=json.loads((D/"identity.json").read_text())
checks=0

def ck(ok,where):
 global checks
 checks+=1
 assert ok,where

def compare(a,b,where):
 for ca,cb in (("all","all"),("persistence_available","matched")):
  for aa,bb in (("r80","r80_fit_root"),("full","full_pool_root"),("persistence","persistence")):
   if aa not in a[ca]["scores"]:continue
   for ma,mb in (("n","n"),("tp","tp"),("fp","fp"),("fn","fn"),("crisis_f1","crisis_f1"),("macro_f1","macro_f1_fourclass"),("brier","crisis_brier")):
    ck(abs(a[ca]["scores"][aa][ma]-b[cb][bb][mb])<1e-12,(where,ca,aa,ma))
  delta=b[cb].get("delta",b[cb].get("pooled_delta_full_minus_r80",{}).get("decisions"))
  for key in ("corrected","spoiled"):
   ck(a[ca][key]==delta[key],(where,ca,key))

ck(s["fits"]==0 and s["models_replayed"]==30,"budget")
compare(ref["overall"],s["overall_15"],"overall")
for h,block in ref["per_h"].items():compare(block,s["by_horizon"]["H"+h],h)
for name,block in s["per_pair"].items():
 key=f'h{block["horizon"]}_{block["target_month"]}'
 compare(ref["per_pair"][key],block["scores"],name)
 r=ref["identities"][name]; fit=block["fit_fraction"]; extra=block["window_minus_legal"]
 ck((r["fit"],r["legal"],r["full"],r["extra_rows"],r["extra_areas"])==(fit["fit_rows"],fit["legal_rows"],fit["window_rows"],extra["rows"],extra["areas"]),(name,"pool counts"))
for group,actual in [(list(s["per_pair"]),s["overall_15"])]+[( [n for n,b in s["per_pair"].items() if b["horizon"]==h],s["by_horizon"][f"H{h}"]) for h in (4,8,12)]:
 for ca,cb in (("all","all"),("persistence_available","matched")):
  for ma,mb in (("crisis_f1","crisis_f1"),("macro_f1","macro_f1_fourclass"),("brier","crisis_brier")):
   vals=[]
   for n in group:
    p=s["per_pair"][n];r=ref["per_pair"][f'h{p["horizon"]}_{p["target_month"]}'][ca]["scores"]
    vals.append(r["full"][ma]-r["r80"][ma])
   ck(abs(math.fsum(vals)/len(vals)-actual[cb]["mean_of_pair_deltas_full_minus_r80"][mb])<1e-12,(group,ca,ma))
ck(hashlib.sha256((D/"rows_E3.csv.gz").read_bytes()).hexdigest()==idn["rows_E3_sha256"],"rows hash")
# Joined rows must reproduce both original saved prediction files exactly.
def rows(p):
 with (gzip.open(p,"rt",encoding="utf-8") if p.suffix==".gz" else p.open(encoding="utf-8")) as f:return list(csv.DictReader(f))
v={(r["horizon"],r["target_month"],r["area"]):r for r in rows(B/"geoxgb-v1-20261001/gscreen/predictions.csv.gz") if r["g_config"]=={"4":"G1","8":"G4","12":"G2"}[r["horizon"]]}
raw={n:{r["FEWSNET_admin_code"]:r for r in rows(B/"geoxgb-d34-e1-brier-20261002/stage1_e1pair/roots"/n/"root_target_predictions.csv")} for n in s["per_pair"]}
joined=rows(D/"rows_E3.csv.gz")
ck(len(joined)==ref["rows"]==len({(r["horizon"],r["target_month"],r["area"]) for r in joined}),"joined unique keys")
for r in joined:
 a=raw[r["pair"]][r["area"]];b=v[(r["horizon"],r["target_month"],r["area"])]
 ck(r["truth"]==a["y_true_code"]==b["y_true_code"],"joined truth")
 for l in ("1","2","3","4或5"):
  ck(float(r["p_full_pool_"+l])==float(b["p_"+l]) and float(r["p_r80_fit_"+l])==float(a["p_pooled_"+l]),"joined probabilities")
result=dict(passed=True,checks=checks,issues=[],scope="All15 pairs, perH/overall, both cohorts, all aggregate/mean metrics, pool counts and all joined probabilities; separate supervisor check replayed6 saved models",source_summary_sha256=hashlib.sha256((D/"summary.json").read_bytes()).hexdigest())
out=B/"d44_comparison_results.json"
assert not out.exists()
out.write_text(json.dumps(result,indent=2))
print(json.dumps(result))
