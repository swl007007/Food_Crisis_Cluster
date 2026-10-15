"""Read-only precision check before attach: plan read with default float parsing (as attach_descriptors does) vs
float_precision='round_trip'; compare numeric columns exactly and the descriptor table / coverage routes built from each."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from ipcch_alphaearth import descriptors
from ipcch_alphaearth.contract import load_contract

run = Path(sys.argv[1]); ext = run / "extraction"; c = load_contract()
dt = {"piece_id": str, "asset_id": str, "recipe": str}
a = pd.read_csv(ext / "plan.csv.gz", dtype=dt)
b = pd.read_csv(ext / "plan.csv.gz", dtype=dt, float_precision="round_trip")
num = [k for k in a.columns if pd.api.types.is_float_dtype(a[k])]
diff = {}
for k in num:
    x, y = a[k].to_numpy(), b[k].to_numpy()
    same = (x == y) | (np.isnan(x) & np.isnan(y))
    rel = np.where(same, 0.0, np.abs(x - y) / np.maximum(np.abs(y), 1e-300))
    diff[k] = {"n_differ": int((~same).sum()), "max_rel": float(np.nanmax(rel)) if len(rel) else 0.0}
pieces = pd.read_csv(ext / "pieces.csv.gz", float_precision="round_trip", dtype={**dt, "status": str})
out = {}
for name, plan in (("default", a), ("round_trip", b)):
    poly = plan.groupby("admin_code", as_index=False)["polygon_area_m2"].first()
    t, led = descriptors.descriptor_table(pieces, plan, poly, c, expected_admins=sorted(poly["admin_code"]))
    out[name] = (t, led)
ta, la = out["default"]; tb, lb = out["round_trip"]
cov = [k for k in la.columns if "cover" in k.lower()]
res = {"plan_rows": len(a), "float_columns": num, "plan_float_diffs": diff,
       "ledger_columns": list(la.columns), "coverage_columns": cov,
       "route_reason_equal": bool((la["route_reason"].astype(str).values == lb["route_reason"].astype(str).values).all()),
       "usable_equal": bool((la["usable"].values == lb["usable"].values).all()),
       "usable_count": int(lb["usable"].sum()),
       "descriptor_table_bitwise_equal": bool(ta.equals(tb)),
       "ledger_bitwise_equal": bool(la.equals(lb))}
for k in cov:
    x, y = la[k].to_numpy(float), lb[k].to_numpy(float)
    res[f"max_abs_diff_{k}"] = float(np.nanmax(np.abs(x - y)))
    res[f"n_near_80_{k}"] = int(((y > 0.79999) & (y < 0.80001)).sum())
    res[f"min_margin_to_80_{k}"] = float(np.nanmin(np.abs(y - 0.80)))
print(json.dumps(res, indent=1, default=str))
