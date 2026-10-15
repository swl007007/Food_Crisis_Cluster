"""Read-only P3 reconciliation after predict: 61/244 inventory, provider ancestry, same-key cohorts, G==B2."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
R = Path(sys.argv[1]); out = {}
dev = pd.read_csv(R / "develop/request_ledger.csv.gz"); pre = pd.read_csv(R / "predict/request_ledger.csv.gz")
sched = pd.read_csv(R / "prepared/schedule_requests.csv.gz")
allf = pd.concat([dev, pre])
out["fits"] = {"develop": len(dev), "predict": len(pre), "total": len(allf), "unique_identities": int(allf.identity.nunique()),
               "scalar_fits": 4 * len(allf), "predict_by_role_arm": {f"{a}_{b}": int(n) for (a, b), n in pre.groupby(["role", "arm"]).size().items()},
               "actions": allf.action.value_counts().to_dict()}
key = lambda d: set(zip(d.arm, d.block_id))
out["ledger_equals_schedule"] = key(allf) == set(zip(sched.arm, sched.block_id)) and len(allf) == len(sched) == 61
out["fit_origin_matches_schedule"] = bool((allf.merge(sched[["arm", "block_id", "fit_origin"]], on=["arm", "block_id"], suffixes=("", "_s"))
                                          .eval("fit_origin == fit_origin_s")).all())
for h in (1, 3, 6, 12):
    p = pd.read_csv(R / f"predict/predictions_h{h:02d}.csv.gz", float_precision="round_trip")
    pv = pd.read_csv(R / f"predict/providers_all_h{h:02d}.csv.gz", float_precision="round_trip")
    keys = pd.read_csv(R / f"prepared/keys_h{h:02d}.csv.gz", usecols=["admin_code", "target_ord"])
    origin = keys["target_ord"].to_numpy() - h
    d = {"rows": len(p), "periods": p.period.value_counts().to_dict(), "duplicate_rows": int(p.row.duplicated().sum()),
         "routable": int(p.routable.astype(bool).sum())}
    arms = ["B0", "B1", "B2", "G"]
    cols = [f"{a}_{q}_{k}" for a in arms for q in ("q2", "q3", "q4", "q5") for k in ("raw", "star")] + [f"{a}_phase" for a in arms]
    d["nonfinite_or_missing_arm_outputs"] = int(p[cols].isna().sum().sum() + (~np.isfinite(p[[c for c in cols if not c.endswith('phase')]].to_numpy(float))).sum())
    d["G_equals_B2_all"] = all(p[f"G_{s}"].equals(p[f"B2_{s}"]) for s in [f"{q}_{k}" for q in ("q2", "q3", "q4", "q5") for k in ("raw", "star")] + ["phase"])
    d["G_route_values"] = p.G_route.astype(str).value_counts().to_dict()
    # provider ancestry: every provider used for a row was fit strictly before that row's origin
    po = pv.merge(p[["row", "target_ord"]], on=["row", "target_ord"], how="left", indicator=True)
    d["provider_rows"] = len(pv); d["provider_rows_unmatched"] = int((po._merge != "both").sum())
    row_origin = dict(zip(range(len(keys)), origin))
    pv_origin = pv.row.map(row_origin)
    d["provider_fit_origin_not_before_origin"] = int((pv.provider_fit_origin.to_numpy() > pv_origin.to_numpy()).sum())
    d["provider_identities_in_ledger"] = bool(set(pv.provider_identity) <= set(allf.identity))
    d["arm_identities_in_ledger"] = bool(set(p.B1_identity.dropna()) <= set(allf.identity) and set(p.B0_identity.dropna()) <= set(allf.identity))
    out[f"h{h}"] = d
print(json.dumps(out, indent=1, default=str))
