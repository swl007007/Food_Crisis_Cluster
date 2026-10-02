"""Predefined D38 fixed-prior and output-only control, independent and without fits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

base = Path(r"C:\Users\swl00\geoxgb_runs")
source = base / "geoxgb-d37-recency-root-20261002"
out = base / "d38_fixed_controls.json"
assert not out.exists()
summary = json.loads((source / "summary.json").read_text())
rows = []
for name in sorted(summary["per_root"]):
    f = pd.read_csv(source / name / "rows_E3.csv.gz", float_precision="round_trip")
    known = f.persistence_code.notna().to_numpy()
    q = np.full((len(f), 4), .25)
    q[known] = .125
    q[np.flatnonzero(known), f.persistence_code[known].to_numpy(int)] = .625
    p = f[[f"p_original_{c}" for c in ("1", "2", "3", "4或5")]].to_numpy()
    post = p.copy()
    shifted = p[known] * q[known]
    post[known] = shifted / shifted.sum(axis=1, keepdims=True)
    assert np.array_equal(post[~known], p[~known])
    assert np.array_equal(q[known].argmax(axis=1), f.persistence_code[known])
    f["y_prior"], f["y_post"] = q.argmax(axis=1), post.argmax(axis=1)
    f["pc_prior"], f["pc_post"], f["pc_original"] = q[:, 2:].sum(1), post[:, 2:].sum(1), p[:, 2:].sum(1)
    f["root"] = name
    rows.append(f)
frame = pd.concat(rows, ignore_index=True)

def scores(f):
    t = f.truth.to_numpy() >= 2
    r = {"n": len(f)}
    for m in ("original", "prior", "post"):
        y = f["y_" + m].to_numpy() >= 2
        tp, fp, fn = int((t & y).sum()), int((~t & y).sum()), int((t & ~y).sum())
        r[m] = {"tp": tp, "fp": fp, "fn": fn, "f1": 2*tp/(2*tp+fp+fn),
                "brier": float(np.mean((f["pc_"+m].to_numpy()-t)**2))}
    return r

result = {"rule": "Precommitted D38 lambda=.5; q=.625/.125 known and uniform missing; output-only p*q normalised on known keys, missing unchanged. Exposed E3 diagnostic, no tuning.",
          "overall_all": scores(frame), "overall_matched": scores(frame[frame.persistence_code.notna()]),
          "by_horizon": {str(h): scores(f) for h,f in frame.groupby("horizon")},
          "by_root": {name: scores(f) for name,f in frame.groupby("root")}}
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(out)
print(json.dumps({k:v for k,v in result.items() if k not in ("by_root", "rule")}, indent=2))
