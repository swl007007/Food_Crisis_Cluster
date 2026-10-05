"""Executor independent D40 check. Reads the 21 original D38 rows_E3 frames; does not import or rerun the
main script. Thresholds: exhaustive ascending scan of distinct source scores with searchsorted counts and
exact Fraction F1, ties -> largest tau (a different algorithm from a descending cumulative scan)."""
import csv, glob, gzip, hashlib, json
from fractions import Fraction
import numpy as np

B = "/mnt/c/Users/swl00/geoxgb_runs"
RUN = f"{B}/geoxgb-d38-persistence-margin-root-20261002"
OUT = f"{B}/d40-forward-decision-20261002"
S = json.load(open(f"{OUT}/summary.json")); TH = json.load(open(f"{OUT}/thresholds.json"))
D39 = json.load(open(f"{B}/d39_probability_diagnostic.json"))["input_hashes"]
ARMS = ("original", "anchored")
mi = lambda s: int(s[:4]) * 12 + int(s[5:7]) - 1
issues, n_checks = [], 0


def check(cond, *what):
    global n_checks
    n_checks += 1
    if not cond:
        issues.append(what)


frames = {}
for d in sorted(glob.glob(f"{RUN}/h*")):
    name = d.rsplit("/", 1)[-1]
    raw = open(f"{d}/rows_E3.csv.gz", "rb").read()
    h = hashlib.sha256(raw).hexdigest()
    check(h == D39[name] == S["input_hashes"][name], "input hash", name)
    rows = list(csv.DictReader(gzip.open(f"{d}/rows_E3.csv.gz", "rt", encoding="utf-8")))
    f = {"H": int(name.split("_")[0][1:]), "T": name.split("_")[1], "n": len(rows),
         "keys": [(r["area"], r["target_month"]) for r in rows],
         "z": np.array([int(r["truth"]) >= 2 for r in rows]),
         "known": np.array([r["persistence_code"] != "" for r in rows]),
         "per": np.array([r["persistence_code"] != "" and float(r["persistence_code"]) >= 2 for r in rows])}
    check(len(set(f["keys"])) == len(rows), "dup keys", name)
    check(all(r["target_month"] == f["T"] for r in rows), "target month", name)
    for a in ARMS:
        f[f"s_{a}"] = np.array([float(r[f"p_{a}_3"]) + float(r[f"p_{a}_4或5"]) for r in rows])
        f[f"y_{a}"] = np.array([int(r[f"y_{a}"]) >= 2 for r in rows])
        P = np.array([[float(r[f"p_{a}_{c}"]) for c in ("1", "2", "3", "4或5")] for r in rows])
        f[f"brier_{a}"] = float(np.mean((P[:, 2:].sum(axis=1) - f["z"]) ** 2))
    frames[name] = f


def best_tau(s, z):
    pos, neg = np.sort(s[z]), np.sort(s[~z]); P = len(pos)
    best = None
    for tau in np.unique(s):                                     # ascending distinct scores
        tp = P - np.searchsorted(pos, tau, side="left"); fp = len(neg) - np.searchsorted(neg, tau, side="left")
        fn = P - tp; f1 = Fraction(2 * int(tp), 2 * int(tp) + int(fp) + int(fn))
        if best is None or f1 > best[0] or (f1 == best[0] and tau > best[1]):
            best = (f1, float(tau), int(tp), int(fp), int(fn))
    return best


def conf(z, a):
    tp, fp, fn = int((z & a).sum()), int((~z & a).sum()), int((z & ~a).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": int(len(z) - tp - fp - fn), "f1_exact": str(Fraction(2 * tp, 2 * tp + fp + fn))}


res = {"thresholds": {}, "scores": {}}
eligible = []
pooled = {k: {a: {"z": [], "pol": [], "arg": [], "known": [], "per": []} for a in ARMS} for k in ("all21", "elig6")}
for name, f in frames.items():
    O = mi(f["T"]) - f["H"]
    src = sorted(k for k, g in frames.items() if g["H"] == f["H"] and mi(g["T"]) < O)
    ok = len({frames[k]["T"] for k in src}) >= 3 and all(frames[k]["z"].any() and (~frames[k]["z"]).any() for k in src)
    t = TH[name]
    check(t["origin_index"] == O, "origin index", name)
    check(sorted(t["source_roots"]) == src, "source list", name, src, t["source_roots"])
    check(sorted(t["source_dates"]) == sorted(frames[k]["T"] for k in src), "source dates", name)
    check(all(mi(frames[k]["T"]) < O for k in src), "strict U<O", name)
    check(all(t["source_hashes"].get(k) == D39[k] for k in src), "source hashes", name)
    check(bool(t["eligible"]) == ok, "eligibility", name)
    if ok:
        eligible.append(name)
    for a in ARMS:
        if ok:
            s = np.concatenate([frames[k][f"s_{a}"] for k in src]); z = np.concatenate([frames[k]["z"] for k in src])
            f1, tau, tp, fp, fn = best_tau(s, z)
            ta = t["arms"][a]
            check(ta["status"] == "source_threshold", "status", name, a)
            check(ta["tau"] == tau and ta["tau_hex"] == float.hex(tau), "tau", name, a, tau, ta["tau"])
            check(ta["source_f1_exact"] == str(f1), "source f1", name, a)
            check(ta["source_best_confusion"] == {"tp": tp, "fp": fp, "fn": fn}, "source conf", name, a)
            check(ta["source_rows"] == len(s) and ta["source_positive"] == int(z.sum()) and ta["source_negative"] == int((~z).sum()), "source counts", name, a)
            res["thresholds"][f"{name}|{a}"] = {"tau": tau, "tau_hex": float.hex(tau), "source_f1": str(f1), "source_rows": len(s), "sources": len(src)}
            pol = f[f"s_{a}"] >= tau
        else:
            check(t["arms"][a]["status"] == "fallback_argmax", "fallback status", name, a)
            pol = f[f"y_{a}"]
        arg = f[f"y_{a}"]; z = f["z"]; k = f["known"]
        sc = {"all": {"argmax": conf(z, arg), "policy": conf(z, pol)},
              "matched": {"argmax": conf(z[k], arg[k]), "policy": conf(z[k], pol[k]), "persistence": conf(z[k], f["per"][k])}}
        ref = S["per_root"][name]["arms"][a]
        for part in ("all", "matched"):
            for m, v in sc[part].items():
                r = ref[part][m]
                check(all(v[x] == r[x] for x in ("tp", "fp", "fn", "tn")) and v["f1_exact"] == r["f1_exact"], "score", name, a, part, m)
        check(abs(ref["brier_unchanged"] - f[f"brier_{a}"]) == 0, "brier", name, a)
        if not ok:
            check(np.array_equal(pol, arg), "fallback unchanged", name, a)
        res["scores"][f"{name}|{a}"] = sc
        for key in ("all21",) + (("elig6",) if ok else ()):
            for nm, arr in (("z", z), ("pol", pol), ("arg", arg), ("known", k), ("per", f["per"])):
                pooled[key][a][nm].append(arr)
        f[f"pol_{a}"] = pol

# pooled aggregates
for key, ref_key in (("all21", "all_21"), ("elig6", "eligible_6_descriptive")):
    for a in ARMS:
        c = {nm: np.concatenate(v) for nm, v in pooled[key][a].items()}
        k = c["known"]
        got = {"all": {"argmax": conf(c["z"], c["arg"]), "policy": conf(c["z"], c["pol"])},
               "matched": {"argmax": conf(c["z"][k], c["arg"][k]), "policy": conf(c["z"][k], c["pol"][k]), "persistence": conf(c["z"][k], c["per"][k])}}
        ref = S[ref_key][a]
        for part in got:
            for m, v in got[part].items():
                check(all(v[x] == ref[part][m][x] for x in ("tp", "fp", "fn", "tn")), "pooled", key, a, part, m)
        res[f"pooled_{key}_{a}"] = got

# policy_rows.csv.gz row-level
seen = 0
for r in csv.DictReader(gzip.open(f"{OUT}/policy_rows.csv.gz", "rt", encoding="utf-8")):
    f = frames[r["root"]]; a = r["arm"]
    i = seen_idx = None
    seen += 1
idx = {name: {key: i for i, key in enumerate(f["keys"])} for name, f in frames.items()}
rowcount = {(n, a): 0 for n in frames for a in ARMS}
for r in csv.DictReader(gzip.open(f"{OUT}/policy_rows.csv.gz", "rt", encoding="utf-8")):
    f = frames[r["root"]]; a = r["arm"]; i = idx[r["root"]][(r["area"], r["target_month"])]
    rowcount[(r["root"], a)] += 1
    ok = (float(r["p_crisis"]) == f[f"s_{a}"][i] and (r["policy_crisis"] == "1") == bool(f[f"pol_{a}"][i])
          and (r["argmax_crisis"] == "1") == bool(f[f"y_{a}"][i]) and (int(r["truth"]) >= 2) == bool(f["z"][i])
          and (r["eligible"] == "True") == (r["root"] in eligible))
    if not ok:
        issues.append(("row", r["root"], a, r["area"]))
check(all(rowcount[(n, a)] == frames[n]["n"] for n in frames for a in ARMS), "row counts")
res["row_level"] = {"rows": seen, "expected": 2 * sum(f["n"] for f in frames.values())}
res["eligible"] = eligible; res["n_checks"] = n_checks; res["issues"] = [list(map(str, x)) for x in issues]
res["scope"] = ("all 21 roots x 2 arms: strict U<O source lists/dates/hashes, eligibility, 12 thresholds by exhaustive "
                "distinct-score scan with exact Fraction ties, applied policy, all/matched scores incl. persistence, "
                "15 fallbacks unchanged per arm, Brier unchanged, pooled all-21 and eligible-6, every policy_rows row")
json.dump(res, open("d40_executor_check.json", "w"), indent=1, default=str)
print("eligible", eligible); print("checks", n_checks, "issues", len(issues), issues[:5]); print(res["row_level"])
for k, v in res["thresholds"].items(): print(k, v)
