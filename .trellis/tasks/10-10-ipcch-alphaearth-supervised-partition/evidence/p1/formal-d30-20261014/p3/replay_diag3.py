"""Read-only, zero-fit: re-predict the h03 supplement B1 provider block from its stored booster and compare every
column of predict/providers_all_h03.csv.gz for that block (identifies exactly which cells differ)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from ipcch_alphaearth import backbone, schedule
from ipcch_alphaearth.contract import load_contract
from ipcch_alphaearth.runinputs import load_run_inputs
from ipcch_alphaearth.pipeline import TARGETS
run = Path(sys.argv[1]); c = load_contract(); inp = load_run_inputs(run, c); hi = inp.horizons[3]
store = backbone.ModelStore(run / "models", read_only=True)
led = pd.read_csv(run / "predict/request_ledger.csv.gz"); row = led[(led.horizon == 3) & (led.arm == "B1") & (led.block_id == "h03_supp_2026")].iloc[0]
b = {x.block_id: x for x in schedule.horizon_blocks(c, 3)}["h03_supp_2026"]
bst, rec = store.load_by_digest(row.identity)
rows = np.flatnonzero((hi.target >= b.target_lo) & (hi.target <= b.target_hi))
raw = backbone.predict_quartet(bst, hi.X("B1")[rows], int(c["backbone"]["fixed"]["n_jobs"]))
exp = pd.DataFrame({"row": rows, "target_ord": hi.target[rows], "provider_block": b.block_id, "provider_fit_origin": b.fit_origin,
                    "provider_identity": row.identity, **{f"p_{q}": raw[:, k] for k, q in enumerate(TARGETS)}})
saved = pd.read_csv(run / "predict/providers_all_h03.csv.gz", float_precision="round_trip")
saved = saved[saved.provider_block == b.block_id].sort_values("row", kind="mergesort").reset_index(drop=True)
exp = exp.sort_values("row", kind="mergesort").reset_index(drop=True)
out = {"rows_saved": len(saved), "rows_expected": len(exp), "rows_equal": bool(np.array_equal(saved.row, exp.row))}
for col in exp.columns:
    a, e = saved[col].to_numpy(), exp[col].to_numpy()
    d = np.flatnonzero(a != e) if a.dtype.kind in "fiu" and e.dtype.kind in "fiu" else np.flatnonzero(a.astype(str) != e.astype(str))
    out[col] = {"n_differ": int(len(d)), "examples": [[int(saved.row[i]), repr(a[i]), repr(e[i])] for i in d[:3]]}
print(json.dumps(out, indent=1, default=str))
