"""D7 probe: historical label calendar and N (distinct labelled areas per fitting window).

Reads only admin_code/year/month and the NON-NULL PRESENCE of fews_ipc from the
historical FEWSNET.csv (no 2025 dates). No label value, distribution or score is
computed or printed.
"""
import json, sys
import pandas as pd

SRC = "/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/1.Source Data/Outcome/FEWSNET_IPC/FEWSNET.csv"
df = pd.read_csv(SRC, usecols=["country", "admin_code", "year", "month", "fews_ipc"])
df = df.dropna(subset=["admin_code", "year", "month"])
assert df["year"].max() <= 2024, "unexpected post-2024 rows"
df["m"] = df["year"].astype(int) * 12 + df["month"].astype(int) - 1
lab = df[df["fews_ipc"].notna()][["admin_code", "country", "m"]].drop_duplicates()
del df["fews_ipc"]

def lbl(m): return f"{m // 12}-{m % 12 + 1:02d}"
months = sorted(lab["m"].unique())
per_month = lab.groupby("m").agg(areas=("admin_code", "nunique"), countries=("country", "nunique"))
out = {"label_months": [lbl(m) for m in months],
       "areas_per_label_month": {lbl(m): int(r.areas) for m, r in per_month.iterrows()},
       "countries_per_label_month": {lbl(m): int(r.countries) for m, r in per_month.iterrows()},
       "distinct_labelled_areas_total": int(lab["admin_code"].nunique())}

# N over every origin the design can use: Stage1 internal origins back to 2010, through 2025-06.
lo, hi = 2010 * 12, 2025 * 12 + 5
best = (0, None)
per_origin = {}
for O in range(lo, hi + 1):
    w = lab[(lab["m"] >= O - 59) & (lab["m"] < O)]
    n = int(w["admin_code"].nunique())
    per_origin[lbl(O)] = n
    if n > best[0]:
        best = (n, lbl(O))
out["N_max_distinct_areas_any_window"] = best[0]
out["N_argmax_origin"] = best[1]
out["N_per_origin_sample"] = {k: per_origin[k] for k in ["2015-06", "2017-10", "2018-02", "2020-06", "2021-06", "2024-10", "2025-02", "2025-06"]}
N = best[0]
out["fit_ceiling"] = {"stage1": 40824, "floor_N_over_50": N // 50,
                      "total_bound": 40824 + 931 * (1 + N // 50)}
json.dump(out, sys.stdout, indent=1)
