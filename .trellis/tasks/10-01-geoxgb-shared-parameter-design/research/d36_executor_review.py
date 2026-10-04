import glob, os, json
import numpy as np, pandas as pd
D35 = r"C:\Users\swl00\geoxgb_runs\geoxgb-d35-global-increment-20261002"
D34 = r"C:\Users\swl00\geoxgb_runs\geoxgb-d34-e1-brier-20261002"
CUT = 2020 * 12 + 11
def mi(s): y, m = map(int, s.split("-")); return y * 12 + m - 1
snaps = {h: pd.read_parquet(fr"{D34}\prepared\snapshot_h{h}.parquet", columns=["area", "target_month", "country", "class_code", "hist_phase_o00"],
                            filters=[("target_month", "<=", CUT)]) for h in (4, 8, 12)}
parts = {"C": [], "E3": []}; cooc = []
for rdir in sorted(glob.glob(D35 + r"\h*_e1pair")):
    name = os.path.basename(rdir); h = int(name.split("_")[0][1:])
    s = snaps[h]
    for part in ("C", "E3"):
        r = pd.read_csv(fr"{rdir}\rows_{part}.csv.gz")
        r["mi"] = r["target_month"].map(mi)
        j = r.merge(s, left_on=["area", "mi"], right_on=["area", "target_month"], how="left", suffixes=("", "_s"), validate="one_to_one")
        assert j["class_code"].notna().all() and (j["class_code"] == j["truth"]).all(), name
        j["root_name"] = name
        parts[part].append(j)
    # C rows: does the same (country, month) have fitting rows in this root?
    mem = pd.read_csv(fr"{D34}\stage1_e1pair\roots\{name}\fold_membership.csv.gz")
    mem["mi"] = mem["target_month"].map(lambda x: mi(x) if isinstance(x, str) else x)
    country = dict(zip(s["area"], s["country"]))
    mem["country"] = mem["area"].map(country)
    fitcm = set(zip(mem.loc[mem.role == "fitting", "country"], mem.loc[mem.role == "fitting", "mi"]))
    c = mem[mem.role == "confirmation"]
    cooc.append((len(c), sum((a, b) in fitcm for a, b in zip(c["country"], c["mi"]))))
out = {}
for part, frames in parts.items():
    d = pd.concat(frames, ignore_index=True)
    t = d["truth"] >= 2; r = d["y_root"] >= 2; b = d["y_brier_local"] >= 2
    ph = d["hist_phase_o00"]; pc = np.where(ph.isna(), np.nan, np.minimum(ph - 1, 3))
    pers = pd.Series(np.where(np.isnan(pc), "missing", np.where(pc >= 2, "1", "0")), index=d.index)
    grp = pers.where(pers == "missing", pers + t.astype(int).astype(str))
    def counts(m):
        return dict(n=int(m.sum()), root_tp=int((t & r & m).sum()), root_fp=int((~t & r & m).sum()), root_fn=int((t & ~r & m).sum()),
                    brier_tp=int((t & b & m).sum()), brier_fp=int((~t & b & m).sum()), brier_fn=int((t & ~b & m).sum()),
                    corrected=int(((r != t) & (b == t) & m).sum()), spoiled=int(((r == t) & (b != t) & m).sum()))
    res = {"all": counts(pd.Series(True, index=d.index)), "groups": {g: counts(grp == g) for g in ["00", "01", "10", "11", "missing"]}}
    net = ((r != t) & (b == t)).astype(int) - ((r == t) & (b != t)).astype(int)
    bycountry = net.groupby(d["country"]).sum().sort_values()
    nz = bycountry[bycountry != 0]
    res["country"] = {"countries": int(d["country"].nunique()), "net_total": int(net.sum()),
                      "positive_countries": int((bycountry > 0).sum()), "negative_countries": int((bycountry < 0).sum()),
                      "top3_positive": {k: int(v) for k, v in bycountry.tail(3)[::-1].items()},
                      "top3_negative": {k: int(v) for k, v in bycountry.head(3).items()},
                      "share_abs_net_top3": round(float(nz.abs().sort_values(ascending=False).head(3).sum() / nz.abs().sum()), 3) if len(nz) else None}
    res["persistence_coverage"] = round(float(1 - (pers == "missing").mean()), 4)
    out[part] = res
out["C_same_country_month_has_fitting_rows"] = {"c_rows": sum(a for a, _ in cooc), "with_fitting_same_country_month": sum(b for _, b in cooc)}
json.dump(out, open(r"C:\Users\swl00\AppData\Local\Temp\d36_review.json", "w"), indent=1)
print(json.dumps(out, indent=1))
