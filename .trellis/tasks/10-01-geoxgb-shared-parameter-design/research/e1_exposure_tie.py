"""Zero-fit E1 crisis-exposure support and root-split side assignment (stdlib + pandas for s_branch.pkl).
Usage: python3 e1_exposure_tie.py <D29 stage1_rootconf dir>"""
import csv, gzip, os, sys
from collections import defaultdict
import numpy as np, pandas as pd
stage = sys.argv[1]
for cand in sorted(os.listdir(f"{stage}/candidates")):
    a = defaultdict(lambda: [0, 0, 0, 0])     # tp, fp, fn, tn on S with the ROOT prediction (root E1 input)
    for r in csv.DictReader(gzip.open(f"{stage}/candidates/{cand}/validation_predictions.csv.gz", "rt")):
        y, p = int(r["y_true"]) >= 2, int(r["y_root"]) >= 2
        a[int(r["area"])][0 if y and p else 1 if p else 2 if y else 3] += 1
    ev = {k: v[0] + v[1] + v[2] for k, v in a.items()}          # rows with D-mass (TP/FP/FN)
    sb = pd.read_pickle(f"{stage}/candidates/{cand}/s_branch.pkl")
    s0 = {int(x) for x in sb["0"] if x >= 0}
    zero = sorted(k for k in a if ev[k] == 0)
    seq = [k in s0 for k in zero]
    print(cand, dict(areas=len(a), exposed=sum(e > 0 for e in ev.values()), one_event=sum(e == 1 for e in ev.values()),
                     zero_exposure=len(zero), mixed_tp_and_error=sum(v[0] > 0 and v[1] + v[2] > 0 for v in a.values()),
                     zero_on_side0=sum(seq), side_switches_along_code=sum(seq[i] != seq[i - 1] for i in range(1, len(seq))),
                     side0_zero_below_median_code=round(float(np.mean([k < np.median(zero) for k, s in zip(zero, seq) if s])), 3)))
