"""Executor independent D38 spot-check: origin label from the raw label table at (area, m-H), not hist_phase_o00.
No XGBoost, no production imports, no supervisor code. <=2020-12 only."""
import json, hashlib
from collections import Counter
import numpy as np, pandas as pd

B = r"C:\Users\swl00\geoxgb_runs"
D34, D37 = B + r"\geoxgb-d34-e1-brier-20261002", B + r"\geoxgb-d37-recency-root-20261002"
MAX = 2020 * 12 + 11
mi = lambda s: int(s[:4]) * 12 + int(s[5:7]) - 1
G = {4: "G1", 8: "G4", 12: "G2"}
T = ["2018-06", "2018-10", "2019-02", "2019-06", "2019-10", "2020-02", "2020-06"]
cols = ["area", "target_month", "class_code", "raw_phase", "hist_phase_o00", "hist_latest_observed_phase", "hist_latest_observed_age"]
snap = {h: pd.read_parquet(D34 + rf"\prepared\snapshot_h{h}.parquet", columns=cols, filters=[("target_month", "<=", MAX)]) for h in G}
lab = pd.concat([s[["area", "target_month", "class_code", "raw_phase"]] for s in snap.values()]).drop_duplicates()
assert not lab.duplicated(["area", "target_month"]).any(), "label disagreement across snapshots"
assert set(lab.raw_phase.dropna().unique()) <= {1, 2, 3, 4, 5}
L = lab.set_index(["area", "target_month"]).class_code
# label calendar: distinct label months per year
cal = {int(y): sorted(int(m % 12) + 1 for m in g.target_month.unique()) for y, g in lab.assign(y=lab.target_month // 12).groupby("y")}
out = {"label_calendar_months_by_year": cal, "roots": []}

def sc(t, y, p):
    z, a = t >= 2, y >= 2
    tp, fp, fn = int((z & a).sum()), int((~z & a).sum()), int((z & ~a).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * tp / (2 * tp + fp + fn), "brier": float(np.mean((p - z) ** 2))}

pool = []
for h, g in G.items():
    S = snap[h].set_index(["area", "target_month"])
    for t in T:
        name = f"h{h}_{t}_{g}_r80_s42_e1pair"
        rd = D34 + rf"\stage1_e1pair\roots\{name}"
        mem = pd.read_csv(rd + r"\fold_membership.csv.gz")
        fit = mem[mem.role == "fitting"]
        fm = fit.target_month.map(mi).to_numpy(np.int64)
        ar = fit.area.to_numpy(np.int64)
        key_sha = hashlib.sha256(np.column_stack([ar, fm]).tobytes()).hexdigest()
        assert key_sha == json.load(open(rd + r"\root.json"))["fitting_keys_sha256"]
        y = fit.class_code.to_numpy(int)
        oi = pd.MultiIndex.from_arrays([ar, fm - h])
        o = L.reindex(oi).to_numpy(float)                      # raw origin label
        f = S.reindex(pd.MultiIndex.from_arrays([ar, fm]))
        assert np.array_equal(f.class_code.to_numpy(int), y)
        hp = f.hist_phase_o00.to_numpy(float) - 1
        agree = bool(np.array_equal(np.isnan(o), np.isnan(hp)) and np.array_equal(o[~np.isnan(o)], hp[~np.isnan(hp)]))
        k = ~np.isnan(o)
        C = np.zeros((4, 4), int); np.add.at(C, (o[k].astype(int), y[k]), 1)
        q = (C + 1) / (C.sum(1, keepdims=True) + 4)
        fb = (np.bincount(y, minlength=4) + 1) / (len(y) + 4)
        oy = (fm[k] - h) // 12
        lo_age = f.hist_latest_observed_age.to_numpy(float)[~k]
        # E3
        e = pd.read_csv(D37 + rf"\{name}\rows_E3.csv.gz")
        em = e.target_month.map(mi).to_numpy(np.int64); ea = e.area.to_numpy(np.int64)
        eo = L.reindex(pd.MultiIndex.from_arrays([ea, em - h])).to_numpy(float)
        assert np.allclose(eo, e.persistence_code.to_numpy(float), equal_nan=True)
        assert np.array_equal(L.reindex(pd.MultiIndex.from_arrays([ea, em])).to_numpy(int), e.truth.to_numpy(int))
        P = np.tile(fb, (len(e), 1)); ke = ~np.isnan(eo); P[ke] = q[eo[ke].astype(int)]
        po = e[[c for c in e.columns if c.startswith("p_original_")]].to_numpy()
        pool.append(pd.DataFrame({"h": h, "truth": e.truth, "orig": e.y_original, "po": po[:, 2:].sum(1),
                                  "prior": P.argmax(1), "pp": P[:, 2:].sum(1), "per": eo}))
        out["roots"].append({"root": name, "keys_ok": True, "rawlabel_equals_hist_phase_o00": agree,
            "known_frac": float(k.mean()), "known_label_months": int(np.unique(fm[k]).size),
            "known_origin_year_le2016": float((oy <= 2016).mean()), "known_target_year_le2016": float(((fm[k]) // 12 <= 2016).mean()),
            "counts": C.tolist(), "argmax": q.argmax(1).tolist(), "p_crisis_by_origin": (q[:, 2] + q[:, 3]).round(4).tolist(),
            "origin3_rows": int(C[3].sum()),
            "missing_origin_latest_obs_age": {"nan": int(np.isnan(lo_age).sum()), **{str(a): int(c) for a, c in sorted(Counter(lo_age[~np.isnan(lo_age)].astype(int)).items())[:8]}}})
F = pd.concat(pool, ignore_index=True); M = F[F.per.notna()]
out["pooled_all"] = {"original": sc(F.truth.values, F.orig.values, F.po.values), "prior": sc(F.truth.values, F.prior.values, F.pp.values)}
out["pooled_matched"] = {"original": sc(M.truth.values, M.orig.values, M.po.values), "prior": sc(M.truth.values, M.prior.values, M.pp.values),
                         "persistence": sc(M.truth.values, M.per.values.astype(int), (M.per.values >= 2).astype(float))}
out["matched_brier_by_origin_class"] = {int(c): {"n": int(len(g)), "original": float(np.mean((g.po - (g.truth >= 2)) ** 2)),
                                        "prior": float(np.mean((g.pp - (g.truth >= 2)) ** 2)),
                                        "orig_crisis_call_rate": float((g.orig >= 2).mean()), "true_crisis_rate": float((g.truth >= 2).mean())}
                                        for c, g in M.groupby(M.per.astype(int))}
json.dump(out, open("d38_executor_spotcheck.json", "w"), indent=1)
r = out["roots"]
print("all keys ok; rawlabel==hist_phase_o00:", Counter(x["rawlabel_equals_hist_phase_o00"] for x in r))
print("argmax:", Counter(tuple(x["argmax"]) for x in r))
for x in r: print(x["root"][:11], round(x["known_frac"], 3), x["known_label_months"], round(x["known_origin_year_le2016"], 3), round(x["known_target_year_le2016"], 3), x["argmax"], x["p_crisis_by_origin"], x["origin3_rows"], x["missing_origin_latest_obs_age"])
print(json.dumps({k: out[k] for k in ("pooled_all", "pooled_matched", "matched_brier_by_origin_class")}, indent=1))
print({y: m for y, m in cal.items()})
