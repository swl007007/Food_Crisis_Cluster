"""Independent LOCAL and GATE reconciliation of saved scenario folds (read-only; no fit, no package import).

Lawful pools are reconstructed with explicit filters from the prepared observations, the release
ledger and each fold's saved frozen map (byte format of the saved digests only; no producer
selection function), then compared with the saved records:

* OUTER deployed locals (gate.json ``locals``): ordered (area, month, variant) key digest of the
  outer lawful pool restricted to the cluster's map members, original-key support, variant rows.
* GATE evaluator keys (gate_pairs): the latest six lawful label months U < O after the outer mask,
  lawful gate areas per cluster, truth codes; INTERNAL local support at V = U - H (pool masked by
  outer + own-k cycles) and its support decision. Internal local key lists are NOT saved, so only
  support counts and the support decision are reconciled for them.
* Gate decisions: exact-fraction crisis F1 recount from the saved gate pairs, support floors,
  undefined metric, strict gain > 1/100, current outer support and the deployed route, and the
  per-cluster routes in predictions.csv.gz.

Usage: python local_gate_reconcile.py RUN_DIR PHASE_DIR   (PHASE_DIR e.g. scenario_development)
"""
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

run, phase = Path(sys.argv[1]), sys.argv[2]
root = run / phase
mi = lambda s: int(s[:4]) * 12 + int(s[5:7]) - 1   # noqa: E731
digest = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()   # noqa: E731
WINDOW, GATE_DATES, CRISIS = 59, 6, 2
FIT_SUPPORT = {"rows": 500, "areas": 50, "dates": 6, "classes": 2}
GATE_SUPPORT = {"rows": 100, "areas": 20, "dates": 3, "local_fit_dates": 3}
GAIN = Fraction(1, 100)

obs = pd.read_csv(run / "prepared/ledgers/observations.csv", usecols=["area", "month", "country", "class_code"])
led = pd.read_csv(run / "prepared/manifests/release_ledger.csv", dtype=str)
led = led[led["product"] == "CS"].copy()
led["month"], led["release"] = led.reference_month.map(mi), led.release_date.map(mi)
obs = obs.merge(led[["country", "month", "release"]], on=["country", "month"], how="left",
                validate="many_to_one").sort_values(["area", "month"]).reset_index(drop=True)
assert obs.release.notna().all() and not obs.duplicated(["area", "month"]).any()
due = led.groupby("month").release.min()
lab = lambda x: f"{x // 12}-{x % 12 + 1:02d}"   # noqa: E731
records = {}
for path in (run / "scenario_globals").rglob("*.json"):
    r = json.loads(path.read_text(encoding="utf-8"))
    records.setdefault(r["booster_sha256"], []).append(r)


def global_link(sha, origin, k, strategy, excluded, masked, fit_keys=None):
    """Any saved record with these booster bytes whose identity matches the independently derived
    origin/strategy/k/exclusions/masks (identical bytes may carry several identities)."""
    want = {"origin_month": lab(origin), "strategy": strategy, "scenario_k": k, "intensity_k": k,
            "excluded_months": [lab(x) for x in sorted(excluded)], "masked_months": [lab(x) for x in sorted(masked)]}
    for r in records.get(sha, []):
        if all(r.get(f) == v for f, v in want.items()) and (fit_keys is None or r["fit_keys_sha256"] == fit_keys):
            return True
    return False


def hidden(origin, k):
    cycles = due[due <= origin].index.tolist()
    assert len(cycles) >= k
    return set(cycles[-k:]) if k else set()


def pool(origin, masked):
    return obs[(obs.month >= origin - WINDOW) & (obs.month < origin) & (obs.release <= origin)
               & ~obs.month.isin(masked)]


def support(frame):
    y = frame.class_code.to_numpy(dtype=np.int64)
    counts = [int((y == c).sum()) for c in range(4)]
    return {"rows": int(len(y)), "areas": int(frame.area.nunique()), "dates": int(frame.month.nunique()),
            "classes": int(sum(c > 0 for c in counts)), "class_counts": counts}


meets = lambda rec, floor: all(rec[k] >= v for k, v in floor.items())   # noqa: E731


def keys_digest(frame, copies):
    a = np.repeat(frame.area.to_numpy(dtype="<i8"), copies)
    m = np.repeat(frame.month.to_numpy(dtype="<i8"), copies)
    v = np.tile(np.arange(copies, dtype="<i8"), len(frame))
    return digest(np.column_stack([a, m, v]).astype("<i8"))


def crisis_f1(y, p):
    t, q = (np.asarray(y) >= CRISIS), (np.asarray(p) >= CRISIS)
    tp, fp, fn = int((t & q).sum()), int((~t & q).sum()), int((t & ~q).sum())
    d = 2 * tp + fp + fn
    return Fraction(2 * tp, d) if d else None


out, problems = [], []
totals = dict(folds=0, gated_folds=0, outer_locals=0, outer_b_weight_blocks=0, regions=0, gate_blocks=0,
              internal_local_fits=0, prediction_rows=0, forecast_rows=0, persistence_rows=0,
              outer_globals_linked=0, internal_globals_linked=0)
for fold_json in sorted(root.rglob("fold.json")):
    base = fold_json.parent
    rec = json.loads(fold_json.read_text(encoding="utf-8"))
    s, h, k, t = rec["strategy"], int(rec["horizon"]), int(rec["scenario_k"]), rec["target_month"]
    O = mi(t) - h
    where = base.relative_to(root).as_posix()
    totals["folds"] += 1
    # historical fold records omit map_route: read it from the frozen map's own consensus record
    route = rec.get("map_route") or json.loads((run / "scenario_maps" / rec["map_id"] / "consensus.json")
                                               .read_text(encoding="utf-8"))["route"]
    rec = {**rec, "map_route": route}
    row = {"fold": where, "map_route": route, "problems": []}
    bad = row["problems"]
    h_o = hidden(O, k)
    gate = json.loads((base / "gate.json").read_text(encoding="utf-8"))
    # forecast truth and matched persistence (all folds) vs prepared observations + explicit filters
    pr = pd.read_csv(base / "predictions.csv.gz", usecols=["area", "target_month", "origin_month", "y_true_code",
                                                           "persistence_class_code", "persistence_source_month",
                                                           "persistence_age"])
    totals["forecast_rows"] += len(pr)
    truth = obs[obs.month == mi(t)].set_index("area").class_code
    exp_t = pr.area.map(truth)
    if not np.array_equal(exp_t.to_numpy(dtype=float), pr.y_true_code.to_numpy(dtype=float), equal_nan=True):
        bad.append("forecast truth differs from prepared observations at T")
    lawful_o = obs[(obs.month <= O) & (obs.release <= O) & ~obs.month.isin(h_o)]
    last = lawful_o.sort_values("month").groupby("area").tail(1).set_index("area")
    src_exp = pr.area.map(last.month).to_numpy(dtype=float)
    cls_exp = pr.area.map(last.class_code).to_numpy(dtype=float)
    age_exp = O - src_exp
    for name, want_v, got in (("persistence source month", src_exp, pr.persistence_source_month),
                              ("persistence class", cls_exp, pr.persistence_class_code),
                              ("persistence age", age_exp, pr.persistence_age)):
        if not np.array_equal(want_v, got.to_numpy(dtype=float), equal_nan=True):
            bad.append(f"{name} differs from the latest lawful label <= O")
    totals["persistence_rows"] += int(np.isfinite(src_exp).sum())
    # outer global linked to independently derived masks (all folds)
    g = gate["global"]
    if not global_link(g["booster_sha256"], O, k, s, set(), h_o, g["fit_keys_sha256"]):
        bad.append("outer global: no saved record with the derived identity/masks")
    totals["outer_globals_linked"] += 1
    if rec["map_route"] != "learned_map":
        row["note"] = "non-learned map route: pooled global, no gate/locals"
        if bad:
            problems.append({"fold": where, "problems": bad[:20], "n": len(bad)})
        out.append(row)
        continue
    totals["gated_folds"] += 1
    copies = 1 if s == "A" else 3
    mdir = run / "scenario_maps" / rec["map_id"]
    cons = json.loads((mdir / "consensus.json").read_text(encoding="utf-8"))
    cmap = pd.read_csv(mdir / cons["cluster_map"])
    assert not cmap.FEWSNET_admin_code.duplicated().any()
    members = {int(c): set(g.FEWSNET_admin_code.astype(int)) for c, g in cmap.groupby("cluster_id")}
    pairs = pd.read_csv(base / "gate_pairs.csv.gz")
    preds = pd.read_csv(base / "predictions.csv.gz", usecols=["area", "cluster_id", "route"])
    outer = pool(O, h_o)
    amap = dict(zip(cmap.FEWSNET_admin_code.astype(int), cmap.cluster_id.astype(int)))
    exp_cluster = preds.area.map(lambda a: amap.get(int(a), -1))
    totals["prediction_rows"] += len(preds)
    if (preds.cluster_id.to_numpy() != exp_cluster.to_numpy()).any():
        bad.append(f"prediction area->cluster differs from the frozen map on {(preds.cluster_id != exp_cluster).sum()} rows")
    mapped = set(exp_cluster[exp_cluster >= 0].astype(int))
    regions = {int(r["cluster_id"]) for r in gate["regions"]}
    if regions != mapped: bad.append(f"regions {sorted(regions ^ mapped)} differ from mapped prediction clusters")
    if not {int(c) for c in (gate.get("locals") or {})} <= mapped: bad.append("deployed local outside mapped clusters")
    if not set(pairs.cluster_id.astype(int)) <= mapped: bad.append("gate pairs for unexpected clusters")
    # outer deployed locals: key digest + original support + variant rows
    for c, loc in (gate.get("locals") or {}).items():
        totals["outer_locals"] += 1
        fr = outer[outer.area.isin(members[int(c)])]
        if loc["fit_keys_sha256"] != keys_digest(fr, copies): bad.append(f"local {c}: key digest")
        if loc["fit_support"] != support(fr): bad.append(f"local {c}: original support")
        if loc["rows"] != copies * len(fr): bad.append(f"local {c}: variant rows")
        sw = loc.get("sample_weight")
        if copies == 1:
            if sw is not None: bad.append(f"local {c}: A local carries a sample_weight block")
        else:
            totals["outer_b_weight_blocks"] += 1
            n = copies * len(fr)
            w32 = np.full(n, 1 / 3, dtype="<f4")
            want = {"dtype": "float32", "sha256": digest(w32), "n": n, "min": float(w32[0]), "max": float(w32[0])}
            if sw is None or any(sw.get(x) != v for x, v in want.items()) \
                    or abs(sw["sum"] - float(w32.astype(np.float64).sum())) > 1e-6:
                bad.append(f"local {c}: B sample_weight block vs repeated float32(1/3)")
    # gate evaluator keys and internal local support
    vis = obs[(obs.release <= O) & ~obs.month.isin(h_o)]
    months = sorted(m for m in vis.month.unique() if m < O)[-GATE_DATES:]
    saved_months = sorted(mi(u) for u in pairs.validation_month.unique())
    if saved_months != months: bad.append(f"gate months {saved_months} vs {months}")
    for u in months:
        lawful = vis[vis.month == u]
        v = u - h
        ipool = pool(v, h_o | hidden(v, k))
        for gsha in pairs.loc[pairs.validation_month == lab(u), "global_sha256"].unique():
            totals["internal_globals_linked"] += 1
            if not global_link(gsha, v, k, s, h_o, h_o | hidden(v, k)):
                bad.append(f"internal global {lab(v)}: no saved record with the derived identity/masks")
        for c, mem in members.items():
            exp = lawful[lawful.area.isin(mem)]
            got = pairs[(pairs.cluster_id == c) & (pairs.validation_month == f"{u // 12}-{u % 12 + 1:02d}")]
            if len(exp) == 0:
                if len(got): bad.append(f"gate c{c} {u}: rows without lawful keys")
                continue
            totals["gate_blocks"] += 1
            if sorted(got.area) != sorted(exp.area):
                bad.append(f"gate c{c} {u}: evaluator keys")
                continue
            truth = exp.set_index("area").class_code.reindex(got.area).to_numpy()
            if not np.array_equal(truth, got.y_true.to_numpy()): bad.append(f"gate c{c} {u}: truth")
            if (got.internal_origin != f"{v // 12}-{v % 12 + 1:02d}").any(): bad.append(f"gate c{c} {u}: internal origin")
            sup = support(ipool[ipool.area.isin(mem)])
            if any((got[f"local_fit_{x}"].astype(int) != sup[x]).any() for x in ("rows", "areas", "dates", "classes")):
                bad.append(f"gate c{c} {u}: internal local support (any row)")
            ok = meets(sup, FIT_SUPPORT)
            if set(got.local_fit_ok.astype(bool)) != {ok}: bad.append(f"gate c{c} {u}: local_fit_ok")
            if ok:
                totals["internal_local_fits"] += 1
                if got.local_sha256.isna().any(): bad.append(f"gate c{c} {u}: missing internal local hash")
            elif got.local_sha256.notna().any() or (got.y_local_routed != got.y_global).any():
                bad.append(f"gate c{c} {u}: unsupported local not routed to global")
    # gate decisions, current support and deployed routes
    for reg in gate["regions"]:
        totals["regions"] += 1
        c = int(reg["cluster_id"])
        p = pairs[pairs.cluster_id == c]
        rows = len(p)
        dated = p.groupby("validation_month").local_fit_ok.any() if rows else pd.Series(dtype=bool)
        cnt = {"rows": rows, "areas": int(p.area.nunique()) if rows else 0, "dates": int(len(dated)),
               "local_fit_dates": int(dated.sum()) if rows else 0}
        if any(reg[x] != cnt[x] for x in cnt): bad.append(f"region c{c}: support counts")
        fg = crisis_f1(p.y_true, p.y_global) if rows else None
        fl = crisis_f1(p.y_true, p.y_local_routed) if rows else None
        short = [x for x, f in GATE_SUPPORT.items() if cnt[x] < f]
        undefined = rows and (fg is None or fl is None)
        if not short and not undefined and rows:
            if reg.get("gain") != str(fl - fg): bad.append(f"region c{c}: gain {reg.get('gain')} vs {fl - fg}")
        enabled = not short and not undefined and rows > 0 and (fl - fg) > GAIN
        reason = (f"gate_support:{'+'.join(short)}" if short else "gate_metric_undefined" if undefined
                  else "" if enabled else "gate_gain_not_above_0.01")
        if bool(reg.get("enabled")) != bool(enabled): bad.append(f"region c{c}: enabled {reg.get('enabled')} vs {enabled}")
        cur = support(outer[outer.area.isin(members[c])])
        if reg["current_fit_support"] != cur: bad.append(f"region c{c}: current outer support")
        deployed = enabled and meets(cur, FIT_SUPPORT)
        route = "local_model" if deployed else "global_fallback"
        final_reason = "" if deployed else (reason if not enabled else "current_fit_support")
        if reg["route"] != route or (not deployed and reg["reason"] != final_reason):
            bad.append(f"region c{c}: route {reg['route']}/{reg['reason']} vs {route}/{final_reason}")
        if deployed != (str(c) in (gate.get("locals") or {})): bad.append(f"region c{c}: local presence")
        want = "local_model" if deployed else f"global_fallback:{final_reason}"
        pr = preds[preds.cluster_id == c]
        if len(pr) == 0 or (pr.route != want).any(): bad.append(f"region c{c}: prediction routes")
    unm = preds[preds.cluster_id == -1]
    if (unm.route != "unmapped_area_global").any() or set(unm.area) & set(cmap.FEWSNET_admin_code):
        bad.append("unmapped routes")
    if bad:
        problems.append({"fold": where, "problems": bad[:20], "n": len(bad)})
    out.append(row)
summary = {"phase": phase, "totals": totals, "problems": problems, "folds": out,
           "evidence_scope": {
               "outer_locals": "saved fit_keys_sha256 + fit_support + variant rows vs independently reconstructed "
                               "ordered keys (outer lawful pool x frozen map cluster); B native sample_weight block "
                               "(float32 sha/n/sum/min/max) vs repeated float32(1/3); A has no weight block",
               "internal_locals": "support counts and support decision only; internal local key lists are not saved",
               "gate_keys": "evaluator key sets and truth vs lawful labels at U under the outer mask",
               "gate_decisions": "exact-fraction recount from saved gate pairs (predictions themselves not refit)",
               "forecast_truth_persistence": "all folds: y_true at T, persistence class/source month/age = latest "
                                             "label <= O with release <= O outside hidden(O,k), from prepared observations",
               "global_links": "outer (gate global booster + fit-key sha) and internal (gate_pairs global sha) linked to a "
                               "saved record whose origin/strategy/k/excluded/masked equal masks derived from the fold "
                               "origin and k; record key digests are covered by the independent fit-key checker"}}
Path(__file__).with_name(f"local_gate_reconcile_{phase}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(json.dumps({"phase": phase, "totals": totals, "problem_folds": len(problems), "first": problems[:3]}, indent=1))
if problems:
    raise SystemExit(1)
