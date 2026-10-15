"""Read-only: per H, AlphaEarth main+supplement truth vs the archived P6 predictions (store artifacts, read only):
same keys? same phase per key? max |q3 difference| and count of differing q3 text."""
import json, sys, glob
from pathlib import Path
import numpy as np, pandas as pd
run, p6_root = Path(sys.argv[1]), sys.argv[2]
out = {}
for h in (1, 3, 6, 12):
    ae = pd.read_csv(run / f"predict/predictions_h{h:02d}.csv.gz", dtype=str, keep_default_na=False)[["admin_code", "target_ord", "phase_truth", "q3"]]
    path = glob.glob(f"{p6_root}/**/stage3/h{h:02d}/predictions.csv.gz", recursive=True)[0]
    p6 = pd.read_csv(path, dtype=str, keep_default_na=False)[["admin_code", "target_ord", "phase_truth", "q3_truth"]]
    m = ae.merge(p6, on=["admin_code", "target_ord"], how="outer", indicator=True, suffixes=("_ae", "_p6"))
    both = m[m._merge == "both"]
    d = np.abs(both.q3.astype(float) - both.q3_truth.astype(float))
    out[h] = {"p6_file": path.split("artifacts/")[-1], "ae_rows": len(ae), "p6_rows": len(p6), "both": len(both),
              "ae_only": int((m._merge == "left_only").sum()), "p6_only": int((m._merge == "right_only").sum()),
              "phase_differ": int((both.phase_truth_ae != both.phase_truth_p6).sum()),
              "q3_text_differ": int((both.q3 != both.q3_truth).sum()), "q3_max_abs_diff": float(d.max())}
print(json.dumps(out, indent=1))
