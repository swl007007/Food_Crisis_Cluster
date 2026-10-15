"""Read-only: (1) every persistence_source_month mismatch is NaN-vs-'' only; (2) h03 combined providers: which columns/rows differ."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from ipcch_alphaearth.contract import load_contract
from ipcch_alphaearth.runinputs import load_run_inputs
from ipcch_alphaearth import replay_report as rr
run = Path(sys.argv[1]); c = load_contract(); inp = load_run_inputs(run, c); out = {"canonical": {}, "h03_providers": {}}
for h, hi in sorted(inp.horizons.items()):
    pred = pd.read_csv(run / "predict" / f"predictions_h{h:02d}.csv.gz", float_precision="round_trip")
    other = 0; nan_vs_empty = 0
    for cx in json.loads((run / "predict" / f"contexts_h{h:02d}.json").read_text()):
        bd = cx["block"]; part = pred[pred.block_id == bd["block_id"]].reset_index(drop=True)
        rows = part["row"].to_numpy(); y = hi.keys["persistence_source_month"].to_numpy()[rows]; x = part["persistence_source_month"].to_numpy()
        for a, b in zip(x, y):
            if (pd.isna(a) and pd.isna(b)) or str(a) == str(b): continue
            if pd.isna(a) and b == "": nan_vs_empty += 1
            else: other += 1
    out["canonical"][h] = {"nan_vs_empty": nan_vs_empty, "other_mismatches": other}
# h03 combined providers: saved vs develop historical + compare per column against itself structure
h = 3
saved = pd.read_csv(run / "predict" / "providers_all_h03.csv.gz", float_precision="round_trip").sort_values("row", kind="mergesort").reset_index(drop=True)
hist = pd.read_csv(run / "develop" / "providers_h03.csv.gz", float_precision="round_trip").sort_values("row", kind="mergesort").reset_index(drop=True)
out["h03_providers"]["saved_rows"], out["h03_providers"]["hist_rows"] = len(saved), len(hist)
out["h03_providers"]["saved_dup_rows"] = int(saved.row.duplicated().sum())
m = saved.merge(hist, on="row", how="inner", suffixes=("_s", "_h"))
out["h03_providers"]["hist_rows_in_combined"] = len(m)
for col in [c for c in hist.columns if c != "row"]:
    a, b = m[col + "_s"], m[col + "_h"]
    if a.dtype.kind in "fiu":
        out["h03_providers"][f"hist_vs_combined_{col}_ndiff"] = int((~np.isclose(a, b, rtol=0, atol=0)).sum())
    else:
        out["h03_providers"][f"hist_vs_combined_{col}_ndiff"] = int((a.astype(str) != b.astype(str)).sum())
out["h03_providers"]["blocks"] = saved.provider_block.value_counts().to_dict()
out["h03_providers"]["fit_origin_ge_target"] = int((saved.provider_fit_origin >= saved.target_ord).sum())
for hh in (1, 6, 12):
    s = pd.read_csv(run / "predict" / f"providers_all_h{hh:02d}.csv.gz")
    out[f"h{hh}_providers_dup_rows"] = int(s.row.duplicated().sum())
print(json.dumps(out, indent=1, default=str))
