"""Executor independent D39 check from saved D38 E3 rows only. No sklearn, no supervisor code, no fits.
AUC: Mann-Whitney with average ranks (ties = 1/2 credit), O(n log n). AP: step AP over distinct
score thresholds, sum (R_k - R_{k-1}) * P_k. Bins: fixed literal edges .1...9 by comparison; [.9,1] closed."""
import csv, glob, gzip, hashlib, json
import numpy as np

RUN = "/mnt/c/Users/swl00/geoxgb_runs/geoxgb-d38-persistence-margin-root-20261002"
MAIN = json.load(open("/mnt/c/Users/swl00/geoxgb_runs/d39_probability_diagnostic.json"))
MODELS = ("original", "anchored", "posthoc", "prior_only")
EDGES = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9])


def auc(s, z):
    pos, neg = int(z.sum()), int((~z).sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s, kind="mergesort"); ss = s[order]
    ranks = np.empty(len(s)); i = 0
    while i < len(ss):                                   # average ranks over tie blocks
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return (ranks[z].sum() - pos * (pos + 1) / 2) / (pos * neg)


def ap(s, z):
    pos = int(z.sum())
    if pos == 0 or pos == len(z):
        return None
    u, inv = np.unique(-s, return_inverse=True)          # distinct thresholds, descending score
    tp = np.cumsum(np.bincount(inv, weights=z.astype(float)))
    n = np.cumsum(np.bincount(inv))
    r = tp / pos; p = tp / n
    return float(np.sum(np.diff(np.concatenate([[0.0], r])) * p))


def bins(s, z):
    idx = (np.clip(s, 0, 1)[:, None] >= EDGES[None, :]).sum(axis=1)
    return [{"n": int((idx == k).sum()), "positive": int(z[idx == k].sum()),
             "brier": float(np.mean((s[idx == k] - z[idx == k]) ** 2)) if (idx == k).any() else None} for k in range(10)]


def f1c(z, a):
    tp, fp, fn = int((z & a).sum()), int((~z & a).sum()), int((z & ~a).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * tp / (2 * tp + fp + fn)}


rows = {}
for d in sorted(glob.glob(f"{RUN}/h*")):
    name = d.rsplit("/", 1)[-1]
    raw = open(f"{d}/rows_E3.csv.gz", "rb").read()
    assert hashlib.sha256(raw).hexdigest() == MAIN["input_hashes"][name]
    rows[name] = list(csv.DictReader(gzip.open(f"{d}/rows_E3.csv.gz", "rt", encoding="utf-8")))


def arrays(rs):
    z = np.array([int(r["truth"]) >= 2 for r in rs]); known = np.array([r["persistence_code"] != "" for r in rs])
    per = np.array([float(r["persistence_code"]) if r["persistence_code"] else np.nan for r in rs])
    P = {m: np.array([float(r[f"p_{m}_3"]) + float(r[f"p_{m}_4或5"]) for r in rs]) for m in MODELS}
    Y = {m: np.array([int(r[f"y_{m}"]) >= 2 for r in rs]) for m in MODELS}
    P["persistence"] = (np.nan_to_num(per, nan=-1) >= 2).astype(float); Y["persistence"] = P["persistence"] >= 1
    return z, known, per, P, Y


def block(z, P, Y, mask, models):
    out = {}
    for m in models:
        s, zz = P[m][mask], z[mask]
        half = s >= .5; arg = Y[m][mask]
        out[m] = {"n": int(mask.sum()), "positive": int(zz.sum()), "mean_p": float(s.mean()), "brier": float(np.mean((s - zz) ** 2)),
                  "roc_auc": auc(s, zz), "average_precision": ap(s, zz), "argmax_crisis": f1c(zz, arg)}
        if m != "persistence":
            out[m]["fixed_half_mass"] = {"table": [[int((~arg & ~half).sum()), int((~arg & half).sum())], [int((arg & ~half).sum()), int((arg & half).sum())]],
                                         "crisis": f1c(zz, half), "corrected": int(((half == zz) & (arg != zz)).sum()), "spoiled": int(((half != zz) & (arg == zz)).sum())}
            out[m]["fixed_bins"] = bins(s, zz)
    return out


def scope(rs):
    z, known, per, P, Y = arrays(rs)
    oc = np.nan_to_num(per, nan=-1) >= 2
    return {"matched": block(z, P, Y, known, MODELS + ("persistence",)), "all": block(z, P, Y, np.ones(len(z), bool), MODELS),
            "missing": block(z, P, Y, ~known, MODELS) if (~known).any() else None,
            "by_origin_crisis": {"0": block(z, P, Y, known & ~oc, MODELS + ("persistence",)), "1": block(z, P, Y, known & oc, MODELS + ("persistence",))}}


res = {"scope": "overall; per-H (4/8/12); first and last root per H (2018-06, 2020-06); parts all/matched/missing/origin0/origin1; "
                "models original/anchored/posthoc/prior_only (+persistence on matched/strata)", "checked": {}}
targets = {"overall": [r for v in rows.values() for r in v]}
for h, g in (("4", "G1"), ("8", "G4"), ("12", "G2")):
    targets[f"H{h}"] = [r for k, v in rows.items() if k.startswith(f"h{h}_") for r in v]
    for t in ("2018-06", "2020-06"):
        targets[f"h{h}_{t}_{g}_r80_s42_e1pair"] = rows[f"h{h}_{t}_{g}_r80_s42_e1pair"]
for k, rs in targets.items():
    res["checked"][k] = scope(rs)

# compare with the main JSON
def ref_of(k):
    return MAIN["overall"] if k == "overall" else MAIN["by_horizon"][k[1:]] if k.startswith("H") else MAIN["per_root"][k]

TOL = 1e-12; issues = []; ncmp = 0
for k, got in res["checked"].items():
    ref = ref_of(k)
    for part in ("all", "matched", "missing"):
        if got[part] is None: continue
        for m, g in got[part].items():
            r = ref[part][m]
            for key in ("n", "positive"):
                ncmp += 1; issues += [] if g[key] == r[key] else [(k, part, m, key, g[key], r[key])]
            for key in ("mean_p", "brier", "roc_auc", "average_precision"):
                ncmp += 1
                if (g[key] is None) != (r.get(key) is None) or (g[key] is not None and abs(g[key] - r[key]) > TOL):
                    issues.append((k, part, m, key, g[key], r.get(key)))
            for key in ("tp", "fp", "fn"):
                ncmp += 1; issues += [] if g["argmax_crisis"][key] == r["argmax_crisis"][key] else [(k, part, m, "argmax_" + key)]
            if "fixed_half_mass" in g and r.get("fixed_half_mass"):
                ncmp += 4
                for key in ("corrected", "spoiled"):
                    issues += [] if g["fixed_half_mass"][key] == r["fixed_half_mass"][key] else [(k, part, m, "half_" + key)]
                issues += [] if g["fixed_half_mass"]["table"] == r["fixed_half_mass"]["argmax_rows_mass_columns"] else [(k, part, m, "half_table")]
                issues += [] if all(g["fixed_half_mass"]["crisis"][x] == r["fixed_half_mass"]["crisis"][x] for x in ("tp", "fp", "fn")) else [(k, part, m, "half_crisis")]
            if "fixed_bins" in g and r.get("fixed_bins"):
                for gb, rb in zip(g["fixed_bins"], r["fixed_bins"]):
                    ncmp += 1
                    if gb["n"] != rb["n"] or gb["positive"] != rb["positive"] or (gb["brier"] is not None and abs(gb["brier"] - rb["brier"]) > TOL):
                        issues.append((k, part, m, "bin", gb, rb))
    for s in ("0", "1"):
        for m, g in got["by_origin_crisis"][s].items():
            r = ref["by_origin_crisis"][s][m]
            for key in ("n", "positive"):
                ncmp += 1; issues += [] if g[key] == r[key] else [(k, s, m, key)]
            for key in ("roc_auc", "average_precision", "brier", "mean_p"):
                ncmp += 1
                if (g[key] is None) != (r.get(key) is None) or (g[key] is not None and abs(g[key] - r[key]) > TOL):
                    issues.append((k, s, m, key, g[key], r.get(key)))

# identity checks on all rows
z, known, per, P, Y = arrays(targets["overall"])
oc = np.nan_to_num(per, nan=-1) >= 2
ident = {
    "prior_only_missing_rows": int((~known).sum()),
    "prior_only_missing_all_exact_half": bool(np.all(P["prior_only"][~known] == 0.5)),
    "prior_only_missing_argmax_code0": bool(np.all(np.array([int(r["y_prior_only"]) for r in targets["overall"]])[~known] == 0)),
    "prior_only_known_values": sorted(set(np.round(P["prior_only"][known], 12).tolist())),
    "prior_only_argmax_eq_persistence_matched": bool(np.all(Y["prior_only"][known] == Y["persistence"][known])),
    "prior_only_half_vs_argmax_disagreements": int((Y["prior_only"] != (P["prior_only"] >= .5)).sum()),
    "constant_strata": {s: {m: {"auc": auc(P[m][msk], z[msk]), "ap": ap(P[m][msk], z[msk]), "prevalence": float(z[msk].mean())}
                            for m in ("prior_only", "persistence")} for s, msk in (("0", known & ~oc), ("1", known & oc))},
    "pooled_prior_only_auc_eq_persistence": auc(P["prior_only"][known], z[known]) == auc(P["persistence"][known], z[known]),
    "p_crisis_max": {m: float(P[m].max()) for m in MODELS}, "p_crisis_min": {m: float(P[m].min()) for m in MODELS},
}
res["identities"] = ident
res["comparisons"] = ncmp; res["issues"] = issues
json.dump(res, open("d39_executor_check.json", "w"), indent=1, default=str)
print("comparisons", ncmp, "issues", len(issues)); print(issues[:10]); print(json.dumps(ident, indent=1, default=str))
