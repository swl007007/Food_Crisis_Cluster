"""Executor independent D41 check. Rebuilds keyed sources from the D34 Brier candidates, roots,
memberships, checkpoints and <=2020-12 snapshots; half = softmax(mean(log p_root, log p_local)),
identity rows copy root. Does not import or rerun d41_local_shrinkage.py."""
import gzip, hashlib, json, math, os, sys
import numpy as np, pandas as pd

B = r"C:\Users\swl00\geoxgb_runs"
D34 = os.path.join(B, "geoxgb-d34-e1-brier-20261002", "stage1_e1pair")
OUT = os.path.join(B, "d41-local-shrinkage-20261002")
D36 = json.load(open(os.path.join(B, "d36-transfer-diagnostic-20261002", "summary.json")))
SUM = json.load(open(os.path.join(OUT, "summary.json")))
C4 = ("1", "2", "3", "4或5"); MAXM = 2020 * 12 + 11
issues, ncmp = [], [0]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
mi = lambda s: int(s[:4]) * 12 + int(s[5:7]) - 1


def chk(cond, *what):
    ncmp[0] += 1
    if not cond:
        issues.append([str(x) for x in what])


cands = sorted(c for c in os.listdir(os.path.join(D34, "candidates")) if "e1brier" in c)
chk(len(cands) == 21, "21 candidates")
snaps = {h: pd.read_parquet(os.path.join(B, "geoxgb-d34-e1-brier-20261002", "prepared", f"snapshot_h{h}.parquet"),
                            columns=["area", "target_month", "hist_phase_o00"], filters=[("target_month", "<=", MAXM)])
         .set_index(["area", "target_month"])["hist_phase_o00"] for h in (4, 8, 12)}
frames, prov = [], {}
for c in cands:
    rn = c.replace("_L1", "").replace("_e1brier_gt0", "_e1pair"); H = int(rn.split("_")[0][1:]); T = rn.split("_")[1]
    R = json.load(open(os.path.join(D34, "roots", rn, "root.json"))); rsha = R["root_booster_sha256"]
    mem = pd.read_csv(os.path.join(D34, "roots", rn, "fold_membership.csv.gz"), dtype={"area": str, "target_month": str})
    cf = pd.read_csv(os.path.join(D34, "candidates", c, "confirmation_predictions.csv.gz"), dtype={"area": str, "target_month": str, "branch_id": str}, keep_default_na=False, float_precision="round_trip")
    tg = pd.read_csv(os.path.join(D34, "candidates", c, "target_predictions.csv"), dtype={"FEWSNET_admin_code": str, "branch_id": str}, keep_default_na=False, float_precision="round_trip")
    rt = pd.read_csv(os.path.join(D34, "roots", rn, "root_target_predictions.csv"), dtype={"FEWSNET_admin_code": str}, float_precision="round_trip")
    memC = mem[mem.role == "confirmation"].set_index(["area", "target_month"]).class_code
    memE = mem[mem.role == "heldout_target"].set_index("area").class_code
    chk(not cf.duplicated(["area", "target_month"]).any() and set(zip(cf.area, cf.target_month)) == set(memC.index), "C keys", rn)
    chk((memC.loc[list(zip(cf.area, cf.target_month))].to_numpy() == cf.y_true.to_numpy()).all(), "C truth", rn)
    chk(not tg.FEWSNET_admin_code.duplicated().any() and not rt.FEWSNET_admin_code.duplicated().any()
        and set(tg.FEWSNET_admin_code) == set(rt.FEWSNET_admin_code) == set(memE.index), "E3 keys", rn)
    rtd = rt.set_index("FEWSNET_admin_code").loc[tg.FEWSNET_admin_code]
    chk((rtd.y_true_code.to_numpy() == tg.y_true_code.to_numpy()).all() and (memE.loc[tg.FEWSNET_admin_code].to_numpy() == tg.y_true_code.to_numpy()).all(), "E3 truth", rn)
    # provenance
    btype = {}
    for b in set(cf.branch_id) | set(tg.branch_id):
        if b == "root":
            btype[b] = "zero_increment"; continue
        j = json.load(open(os.path.join(D34, "checkpoints", c, f"xgb_{b}.json"))); u = sha(os.path.join(D34, "checkpoints", c, f"xgb_{b}.ubj"))
        if j["kind"] == "continuation":
            ok = (j["increment_source"] == "root" and j["rounds_added"] == 20 and j["actual_local_rounds"] == 20 and j["parent_sha256"] == rsha
                  and j["shared_source"] == rsha and j["rounds_total"] == j["parent_rounds"] + 20 and u == j["booster_sha256"])
            btype[b] = "local"
        else:
            ok = j["kind"] == "fresh" and j["actual_local_rounds"] == 0 and j["booster_sha256"] == rsha and u == rsha
            btype[b] = "zero_increment"
        chk(ok, "provenance", rn, b)
    prov[rn] = btype
    for part, keys, months, truth, pr, pf, br, ro, yr, yf in (
            ("C", cf.area, cf.target_month, cf.y_true, cf[[f"p_root_{k}" for k in C4]], cf[[f"p_final_{k}" for k in C4]], cf.branch_id, cf.routing, cf.y_root, cf.y_final),
            ("E3", tg.FEWSNET_admin_code, pd.Series([T] * len(tg)), tg.y_true_code, rtd[[f"p_pooled_{k}" for k in C4]], tg[[f"p_partitioned_{k}" for k in C4]], tg.branch_id, tg.routing, rtd.y_pred_pooled_code, tg.y_pred_partitioned_code)):
        Pr, Pf = pr.to_numpy(float), pf.to_numpy(float)
        for A in (Pr, Pf):
            chk(np.isfinite(A).all() and (A > 0).all() and np.abs(A.sum(1) - 1).max() <= 1e-6, "probabilities", rn, part)
        chk((Pr.argmax(1) == yr.to_numpy()).all() and (Pf.argmax(1) == yf.to_numpy()).all(), "saved argmax", rn, part)
        ph = snaps[H].reindex(pd.MultiIndex.from_arrays([keys.astype(int).to_numpy(), np.array([mi(m) for m in months])])).to_numpy(float)
        frames.append(pd.DataFrame({"root": rn, "part": part, "horizon": H, "area": keys.to_numpy(), "target_month": months.to_numpy(),
                                    "truth": truth.to_numpy(int), "branch_id": br.to_numpy(), "routing": ro.to_numpy(),
                                    "route_type": [btype[b] for b in br], "phase": ph,
                                    **{f"pr{i}": Pr[:, i] for i in range(4)}, **{f"pf{i}": Pf[:, i] for i in range(4)}}))
D = pd.concat(frames, ignore_index=True)
Pr = D[[f"pr{i}" for i in range(4)]].to_numpy(); Pf = D[[f"pf{i}" for i in range(4)]].to_numpy()
same = (Pr == Pf).all(1)
chk(np.array_equal(same, (D.route_type == "zero_increment").to_numpy()), "identity rows == zero-increment routes")
L = 0.5 * (np.log(Pr) + np.log(Pf)); L -= L.max(1, keepdims=True); Ph = np.exp(L); Ph /= Ph.sum(1, keepdims=True)
Ph[same] = Pr[same]
P = {"root": Pr, "half": Ph, "full": Pf}
Y = {k: v.argmax(1) for k, v in P.items()}
z = D.truth.to_numpy() >= 2
known = np.isfinite(D.phase.to_numpy()); per = np.where(known, D.phase.to_numpy() - 1, np.nan)
chk(set(np.unique(D.phase.dropna())) <= {1.0, 2.0, 3.0, 4.0}, "phase values")
trans = np.where(~known, "missing", np.where(per >= 2, "1", "0")) .astype(object) + np.where(~known, "", np.where(z, "1", "0"))
trans = np.where(~known, "missing", trans)
sortp = np.sort(Ph, 1); near_tie = int((sortp[:, -1] - sortp[:, -2] < 1e-9).sum())

# --- row-level comparison
rows = pd.read_csv(os.path.join(OUT, "rows.csv.gz"), dtype={"area": str, "target_month": str, "branch_id": str, "persistence_code": str}, keep_default_na=False, float_precision="round_trip")
chk(sha(os.path.join(OUT, "rows.csv.gz")) == SUM["rows_sha256"], "rows sha")
k_mine = D.root + "|" + D.part + "|" + D.area + "|" + D.target_month
k_them = rows.root + "|" + rows.part + "|" + rows.area + "|" + rows.target_month
chk(len(rows) == len(D) == 321047 and not k_them.duplicated().any() and set(k_them) == set(k_mine), "row keys unique+complete", len(rows))
pos = pd.Series(np.arange(len(D)), index=k_mine.to_numpy()).loc[k_them.to_numpy()].to_numpy()
cmpcol = lambda name, mine, theirs: chk(np.array_equal(np.asarray(mine), np.asarray(theirs)), "column", name)
cmpcol("truth", D.truth.to_numpy()[pos], rows.truth.to_numpy()); cmpcol("horizon", D.horizon.to_numpy()[pos], rows.horizon.to_numpy())
cmpcol("branch_id", D.branch_id.to_numpy()[pos], rows.branch_id.to_numpy()); cmpcol("routing", D.routing.to_numpy()[pos], rows.routing.to_numpy())
cmpcol("route_type", D.route_type.to_numpy()[pos], rows.route_type.to_numpy()); cmpcol("transition", trans[pos], rows.transition.to_numpy())
theirs_per = np.array([float(x) if x != "" else np.nan for x in rows.persistence_code])
chk(np.allclose(per[pos], theirs_per, equal_nan=True, rtol=0, atol=0), "persistence_code")
maxdiff = {}
for arm in P:
    Q = rows[[f"p_{arm}_{k}" for k in C4]].to_numpy(float)
    maxdiff[arm] = float(np.abs(Q - P[arm][pos]).max())
    if arm in ("root", "full"):
        chk(np.array_equal(Q, P[arm][pos]), "exact source probs", arm)
    chk(maxdiff[arm] <= 1e-12, "prob tolerance", arm, maxdiff[arm])
    chk(np.array_equal(rows[f"y_{arm}"].to_numpy(), Y[arm][pos]), "labels", arm, int((rows[f"y_{arm}"].to_numpy() != Y[arm][pos]).sum()))
    chk(float(np.abs(rows[f"pc_{arm}"].to_numpy(float) - P[arm][pos][:, 2:].sum(1)).max()) <= 1e-12, "pc", arm)

# --- recompute summary blocks
def conf(mask, arm):
    a = Y[arm][mask] >= 2; t = z[mask]; pc = P[arm][mask][:, 2:].sum(1)
    tp, fp, fn = int((t & a).sum()), int((~t & a).sum()), int((t & ~a).sum())
    return {"n": int(mask.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": int(mask.sum()) - tp - fp - fn,
            "f1": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0, "brier": float(np.mean((pc - t) ** 2)) if mask.any() else None}


def changes(mask, arm):
    a, r, t = (Y[arm][mask] >= 2), (Y["root"][mask] >= 2), z[mask]
    return {"corrected": int(((a == t) & (r != t)).sum()), "spoiled": int(((a != t) & (r == t)).sum()), "new_tp": int((a & ~r & t).sum()),
            "lost_tp": int((~a & r & t).sum()), "new_fp": int((a & ~r & ~t).sum()), "removed_fp": int((~a & r & ~t).sum())}


def hvf(mask):
    dh = np.abs(P["half"][mask][:, 2:].sum(1) - P["full"][mask][:, 2:].sum(1))
    return {"binary_flips": int(((Y["half"][mask] >= 2) != (Y["full"][mask] >= 2)).sum()), "fourclass_flips": int((Y["half"][mask] != Y["full"][mask]).sum()),
            "mean_abs_crisis_probability_delta": float(dh.mean()) if mask.any() else None, "max_abs_crisis_probability_delta": float(dh.max()) if mask.any() else None}


def block(mask):
    out = {"root": conf(mask, "root"), "half": {**conf(mask, "half"), "changes": changes(mask, "half")},
           "full": {**conf(mask, "full"), "changes": changes(mask, "full")}, "half_vs_full": hvf(mask)}
    return out


def compare(mine, theirs, path):
    if isinstance(mine, dict):
        for k, v in mine.items():
            if k not in theirs:
                issues.append(["missing in summary", path + "/" + k]); continue
            compare(v, theirs[k], path + "/" + k)
    elif isinstance(mine, (int, np.integer)) and not isinstance(mine, bool):
        chk(mine == theirs, "int", path, mine, theirs)
    elif mine is None:
        chk(theirs is None, "none", path)
    else:
        chk(theirs is not None and abs(mine - theirs) <= 1e-12, "float", path, mine, theirs)


mine = {}
for part in ("C", "E3"):
    pm = (D.part == part).to_numpy()
    m = {"all": block(pm), "matched": block(pm & known)}
    m["per_horizon"] = {str(h): block(pm & (D.horizon == h).to_numpy()) for h in (4, 8, 12)}
    m["per_root"] = {r: block(pm & (D.root == r).to_numpy()) for r in sorted(D.root.unique())}
    m["transitions"] = {g: block(pm & (trans == g)) for g in ("00", "01", "10", "11", "missing")}
    m["routes"] = {rt: block(pm & (D.route_type == rt).to_numpy()) for rt in ("local", "zero_increment")}
    fm = {}
    for arm in ("half", "full"):
        df = [m["per_root"][r][arm]["f1"] - m["per_root"][r]["root"]["f1"] for r in m["per_root"]]
        db = [m["per_root"][r][arm]["brier"] - m["per_root"][r]["root"]["brier"] for r in m["per_root"]]
        fm[arm] = {"f1_delta_root": float(np.mean(df)), "brier_delta_root": float(np.mean(db))}
    m["fold_means"] = fm
    mine[part] = m
    compare(m, SUM["parts"][part], part)
    # persistence blocks in matched/transitions if present
    for loc, mask in [("matched", pm & known)] + [(f"transitions/{g}", pm & (trans == g)) for g in ("00", "01", "10", "11")]:
        node = SUM["parts"][part]
        for k in loc.split("/"):
            node = node[k]
        if "persistence" in node:
            pa = per[mask] >= 2; t = z[mask]; tp, fp, fn = int((t & pa).sum()), int((~t & pa).sum()), int((t & ~pa).sum())
            for kk, vv in (("n", int(mask.sum())), ("tp", tp), ("fp", fp), ("fn", fn), ("tn", int(mask.sum()) - tp - fp - fn)):
                chk(node["persistence"][kk] == vv, "persistence", part, loc, kk)
            if "brier" in node["persistence"]:
                chk(abs(node["persistence"]["brier"] - float(np.mean((pa.astype(float) - t) ** 2))) <= 1e-12, "persistence brier", part, loc)
# provenance vs summary
for rn, bt in prov.items():
    chk(SUM["provenance"].get(rn) == bt, "summary provenance", rn)
# D36 reproduction (root / full = brier_local)
d36 = {(r["part"]): r for r in D36["all"] if r["grouping"] == "all"}; d36m = {r["part"]: r for r in D36["persistence_matched"]}
for part in ("C", "E3"):
    for arm, pre in (("root", "root"), ("full", "brier_local")):
        a = mine[part]["all"][arm]; chk(all(a[k] == d36[part][f"{pre}_{k}"] for k in ("tp", "fp", "fn", "tn")) and abs(a["brier"] - d36[part][f"{pre}_brier_loss"]) <= 1e-12, "D36 all", part, arm)
        b = mine[part]["matched"][arm]; chk(all(b[k] == d36m[part][f"{pre}_{k}"] for k in ("tp", "fp", "fn", "tn")), "D36 matched", part, arm)
res = {"scope": "all 21 Brier candidates x C/E3 (321047 rows): sources/keys/truth/provenance (local vs hash-equal zero increment), probabilities "
                "positive, identity rows, persistence via <=2020-12 snapshot hist_phase_o00, every row of rows.csv.gz (keys, columns, exact root/full, "
                "half via log-prob mean + softmax at 1e-12, exact labels), the root/half/full blocks (n, confusion, F1, Brier, "
                "changes vs root, half_vs_full) of all/matched/per_horizon/per_root/transitions/routes and fold_means; persistence "
                "counts/Brier only where present in matched and the four origin-known transition groups (other optional persistence "
                "fields not compared); D36 root/full reproduction",
       "comparisons": ncmp[0], "issues": issues, "max_abs_prob_diff": maxdiff, "near_ties_half_lt_1e-9": near_tie,
       "route_rows": {p: {rt: int(((D.part == p) & (D.route_type == rt)).sum()) for rt in ("local", "zero_increment")} for p in ("C", "E3")},
       "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
       "e3_all": {arm: {k: mine["E3"]["all"][arm][k] for k in ("tp", "fp", "fn", "f1", "brier")} for arm in ("root", "half", "full")}}
json.dump(res, open("d41_executor_check.json", "w"), indent=1, default=str)
print("comparisons", ncmp[0], "issues", len(issues)); print(issues[:10]); print(json.dumps({k: res[k] for k in ("max_abs_prob_diff", "near_ties_half_lt_1e-9", "route_rows", "versions", "e3_all")}, indent=1))
