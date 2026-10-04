import glob, os
import pandas as pd
D35 = r"C:\Users\swl00\geoxgb_runs\geoxgb-d35-global-increment-20261002"; D34 = r"C:\Users\swl00\geoxgb_runs\geoxgb-d34-e1-brier-20261002"
def mi(s): y, m = map(int, s.split("-")); return y * 12 + m - 1
rows = []
for h in (4, 8, 12):
    s = pd.read_parquet(fr"{D34}\prepared\snapshot_h{h}.parquet", columns=["area", "target_month", "hist_phase_o00"], filters=[("target_month", "<=", 2020 * 12 + 11)])
    for rdir in sorted(glob.glob(fr"{D35}\h{h}_*_e1pair")):
        r = pd.read_csv(fr"{rdir}\rows_C.csv.gz", usecols=["area", "target_month", "truth", "y_root", "y_brier_local"])
        r["mi"] = r["target_month"].map(mi)
        j = r.merge(s, left_on=["area", "mi"], right_on=["area", "target_month"], suffixes=("", "_s"))
        j["year"] = j["target_month"].str[:4]; j["month"] = j["target_month"].str[5:7]; j["h"] = h
        rows.append(j)
d = pd.concat(rows)
d["missing"] = d["hist_phase_o00"].isna()
t, r, b = d.truth >= 2, d.y_root >= 2, d.y_brier_local >= 2
d["net"] = ((r != t) & (b == t)).astype(int) - ((r == t) & (b != t)).astype(int)
print(d.groupby("year").agg(rows=("missing", "size"), missing_share=("missing", "mean"), net=("net", "sum")).round(3).to_string())
print(d.groupby(["h", "missing"]).agg(rows=("net", "size"), net=("net", "sum")).to_string())
print("C label months by calendar month (rows):", d.groupby("month").size().to_dict())
