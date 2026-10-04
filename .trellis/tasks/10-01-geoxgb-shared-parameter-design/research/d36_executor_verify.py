import glob, os, json
import numpy as np, pandas as pd
D35 = r"C:\Users\swl00\geoxgb_runs\geoxgb-d35-global-increment-20261002"; D34 = r"C:\Users\swl00\geoxgb_runs\geoxgb-d34-e1-brier-20261002"
V1 = r"C:\Users\swl00\geoxgb_runs\geoxgb-v1-20261001"
CUT = 2020 * 12 + 11
def mi(s): y, m = map(int, s.split("-")); return y * 12 + m - 1
L = ["1", "2", "3", "4或5"]
out = {}; frames = {"C": [], "E3": []}
for h in (4, 8, 12):
    s = pd.read_parquet(fr"{D34}\prepared\snapshot_h{h}.parquet", columns=["area", "target_month", "hist_phase_o00"], filters=[("target_month", "<=", CUT)])
    for rdir in sorted(glob.glob(fr"{D35}\h{h}_*_e1pair")):
        name = os.path.basename(rdir)
        g = name.split("_")[2]; t = name.split("_")[1]
        bcand = fr"{D34}\stage1_e1pair\candidates\h{h}_{t}_{g}_L1_r80_s42_e1brier_gt0"
        for part in ("C", "E3"):
            r = pd.read_csv(fr"{rdir}\rows_{part}.csv.gz")
            if part == "C":
                bp = pd.read_csv(fr"{bcand}\confirmation_predictions.csv.gz", float_precision="round_trip")
                bp = bp.rename(columns={f"p_final_{l}": f"pb_{l}" for l in L})[["area", "target_month"] + [f"pb_{l}" for l in L]]
                r = r.merge(bp, on=["area", "target_month"], validate="one_to_one")
            else:
                bp = pd.read_csv(fr"{bcand}\target_predictions.csv", float_precision="round_trip")
                bp = bp.rename(columns={"FEWSNET_admin_code": "area", **{f"p_partitioned_{l}": f"pb_{l}" for l in L}})[["area"] + [f"pb_{l}" for l in L]]
                r = r.merge(bp, on="area", validate="one_to_one")
            r["mi"] = r["target_month"].map(mi)
            r = r.merge(s, left_on=["area", "mi"], right_on=["area", "target_month"], suffixes=("", "_s"), validate="one_to_one")
            frames[part].append(r)
def f1(t, p):
    tp = int((t & p).sum()); fp = int((~t & p).sum()); fn = int((t & ~p).sum()); return 2 * tp / (2 * tp + fp + fn)
for part, fr in frames.items():
    d = pd.concat(fr, ignore_index=True)
    z = (d.truth >= 2).astype(float)
    pr = d["p_root_3"] + d["p_root_4或5"]; pb = d["pb_3"] + d["pb_4或5"]
    known = d.hist_phase_o00.notna()
    pers = np.minimum(d.hist_phase_o00 - 1, 3) >= 2
    t = d.truth >= 2
    out[part] = {"rows": len(d), "brier_root": float(((pr - z) ** 2).mean()), "brier_brierlocal": float(((pb - z) ** 2).mean()),
                 "known_share": float(known.mean()),
                 "matched_f1_root": f1(t[known], (d.y_root >= 2)[known]), "matched_f1_brier": f1(t[known], (d.y_brier_local >= 2)[known]),
                 "matched_f1_persistence": f1(t[known], pers[known])}
    if part == "E3":
        e3 = d
# 15-fold ledger cross-check (dev_baselines 2019-02..2020-10; D34 E3 targets 2019-02..2020-06)
base = pd.read_csv(fr"{V1}\prepared\ledgers\dev_baselines.csv", usecols=["area", "target_label", "horizon", "persistence_code"], low_memory=False)
m = e3.merge(base, left_on=["area", "target_month", "horizon"], right_on=["area", "target_label", "horizon"], how="inner")
mk = m.persistence_code.notna() & m.hist_phase_o00.notna()
out["ledger_check"] = {"e3_keys_in_ledger": int(len(m)), "both_known": int(mk.sum()),
                       "persistence_code_equal": int((np.minimum(m.hist_phase_o00 - 1, 3)[mk] == m.persistence_code[mk]).sum())}
print(json.dumps(out, indent=1))
