"""Executor exact match of the supervisor's pre-computed D38 fixed controls (d38_fixed_controls.json)
against the D38 run's saved E3 rows (original / prior_only / posthoc). Read-only; no fits."""
import csv, glob, gzip, json
import numpy as np

B = "/mnt/c/Users/swl00/geoxgb_runs"
F = json.load(open(f"{B}/d38_fixed_controls.json"))
MP = {"original": "original", "prior": "prior_only", "post": "posthoc"}


def sc(rows, m):
    z = np.array([int(r["truth"]) >= 2 for r in rows]); a = np.array([int(r[f"y_{m}"]) >= 2 for r in rows])
    p = np.array([float(r[f"p_{m}_3"]) + float(r[f"p_{m}_4或5"]) for r in rows])
    return {"tp": int((z & a).sum()), "fp": int((~z & a).sum()), "fn": int((z & ~a).sum())}, float(((p - z) ** 2).mean())


def block(rows, ref):
    out = {}
    for k, m in MP.items():
        c, b = sc(rows, m)
        f1 = 2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"])
        out[k] = {"counts_equal": all(c[x] == ref[k][x] for x in c), "f1_equal": f1 == ref[k]["f1"],
                  "abs_brier_diff": abs(b - ref[k]["brier"])}
    return out


res, allrows = {"by_root": {}}, []
for d in sorted(glob.glob(f"{B}/geoxgb-d38-persistence-margin-root-20261002/h*")):
    name = d.rsplit("/", 1)[-1]
    rows = list(csv.DictReader(gzip.open(f"{d}/rows_E3.csv.gz", "rt", encoding="utf-8")))
    assert F["by_root"][name]["n"] == len(rows)
    allrows += rows
    res["by_root"][name] = block(rows, F["by_root"][name])
res["overall_all"] = block(allrows, F["overall_all"])
res["overall_matched"] = block([r for r in allrows if r["persistence_code"] != ""], F["overall_matched"])
res["by_horizon"] = {h: block([r for r in allrows if r["horizon"] == h], F["by_horizon"][h]) for h in ("4", "8", "12")}
json.dump(res, open(f"{B}/d38_fixed_controls_match.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "by_root"}, indent=1))
