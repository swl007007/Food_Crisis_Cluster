"""Read-only diagnosis of the failed replay checks (no fitting, no writes to the run)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from ipcch_alphaearth.contract import load_contract
from ipcch_alphaearth.runinputs import load_run_inputs
from ipcch_alphaearth import replay_report as rr
from ipcch_alphaearth.replay import _frames_equal
from ipcch_alphaearth import schedule
run = Path(sys.argv[1]); c = load_contract(); inp = load_run_inputs(run, c); out = {}
for h, hi in sorted(inp.horizons.items()):
    pred = pd.read_csv(run / "predict" / f"predictions_h{h:02d}.csv.gz", float_precision="round_trip")
    for cx in json.loads((run / "predict" / f"contexts_h{h:02d}.json").read_text()):
        bd = cx["block"]
        class B: pass
        b = B(); b.target_lo, b.target_hi, b.role, b.block_id = bd["target_lo"], bd["target_hi"], bd["role"], bd["block_id"]
        part = pred[pred.block_id == b.block_id].reset_index(drop=True); canon = rr.canonical_block(hi, b)
        diff = {}
        for col in rr.CANONICAL:
            if not _frames_equal(part, canon, [col]):
                x, y = part[col].to_numpy(), canon[col].to_numpy()
                bad = [i for i in range(min(len(x), len(y))) if not ((pd.isna(x[i]) and pd.isna(y[i])) or str(x[i]) == str(y[i]))]
                diff[col] = {"n": len(bad), "len": [len(x), len(y)], "dtypes": [str(x.dtype), str(y.dtype)],
                             "examples": [[repr(x[i]), repr(y[i])] for i in bad[:3]]}
        out[cx["context"]] = diff
print(json.dumps(out, indent=1, default=str))
