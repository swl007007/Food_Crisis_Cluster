#!/usr/bin/env python3
"""D42 / A16: old-vs-current spatial map common refit on 12 fixed (H, T, U) pairs.

python scripts/stage1_map_transfer.py --d34-run D34_RUN --d35-run D35_RUN --out NEW_DIR [--producer-rev 7b2bf6f]

Per current D34 root (sequential): the D37 rebuild (target_month <= 2020-12) and original-root
replay GATE (stage1_recency_root.gate_root) before any fit; the saved D35 global+20 booster of the
same root is replayed on the current C/E3 keys and must equal the D35 saved probabilities exactly.
Two spatial maps (D32 assignment_evidence.csv of the D34 Brier candidate): OLD = latest same-H
D34 target U < O = T - H, CURRENT = T. For each map every named region (spatial id != s-1) gets
the current FIT rows of its member areas (original order); a region meeting plan.FIT_SUPPORT gets
exactly ONE L1 (+20 rounds) continuation of the current root. C/E3 rows route to their area's
eligible region model; s-1 / missing area / insufficient-support region -> current root.
Four same-key arms: root, global20, current_map_refit, old_map_refit; exact-origin persistence
on matched non-missing keys. Engineering errors stop the study (failure.json); support fallback
is normal. No Stage 2 map, no E4 weights, no final-period ledger is read; at most 175 fits.
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import traceback
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from scripts import stage1_global_increment as gi  # noqa: E402
from scripts import stage1_recency_root as rr  # noqa: E402
from scripts import stage1_shallow_replay as sr  # noqa: E402
from scripts.stage1_rootconf_compare import accept_mode  # noqa: E402
from src.experiment import plan  # noqa: E402
from src.feature.fourclass_features import load_schema, month_label  # noqa: E402
from src.metrics import fourclass  # noqa: E402
from src.model import native_xgb as nx  # noqa: E402
from src.utils import run_identity as rid  # noqa: E402

MAX_MONTH = gi.MAX_MONTH
LABELS = fourclass.CLASS_LABELS
PARTS = rr.PARTS                                  # ("C", "E3")
GROUPS = rr.GROUPS
ARMS = ("root", "global20", "current_map_refit", "old_map_refit")
MAP_ARMS = {"current_map_refit": "current", "old_map_refit": "old"}
DELTAS = (("global20", "root"), ("current_map_refit", "root"), ("old_map_refit", "root"),
          ("current_map_refit", "global20"), ("old_map_refit", "global20"),
          ("old_map_refit", "current_map_refit"))
REASONS = ("region", "s-1", "missing", "insufficient_support")
S_NONE = "s-1"
MAX_FITS = 175
GateError = rr.GateError

#: Frozen D42 schedule: (H, current T, old-map U, old eligible regions, current eligible regions).
SPEC_PAIRS = (
    (4, "2019-02", "2018-06", 11, 3), (4, "2019-06", "2018-10", 8, 5), (4, "2019-10", "2019-02", 3, 6),
    (4, "2020-02", "2019-06", 5, 7), (4, "2020-06", "2019-10", 6, 11),
    (8, "2019-06", "2018-06", 9, 7), (8, "2019-10", "2018-10", 8, 12), (8, "2020-02", "2019-02", 10, 6),
    (8, "2020-06", "2019-06", 7, 3),
    (12, "2019-10", "2018-06", 6, 9), (12, "2020-02", "2018-10", 4, 12), (12, "2020-06", "2019-02", 8, 9),
)


# ---------------------------------------------------------------- pure helpers (tested)

def _mi(label: str) -> int:
    return int(label[:4]) * 12 + int(label[5:7]) - 1


def schedule(targets=plan.E1PAIR_TARGETS, horizons=plan.HORIZONS) -> list:
    """[(H, T, U)]: U = latest same-H target with U < O = T - H (dates only; no scores read)."""
    out = []
    for h in horizons:
        for t in targets:
            prior = [u for u in targets if _mi(u) < _mi(t) - int(h)]
            if prior:
                out.append((int(h), t, max(prior, key=_mi)))
    return out


def check_schedule() -> list:
    got = schedule()
    want = [(h, t, u) for h, t, u, _, _ in SPEC_PAIRS]
    if got != want:
        raise GateError(f"U < O schedule {got} differs from the frozen D42 table {want}")
    return got


def parse_map(path: Path, recorded_sha: str | None = None) -> dict:
    """D32 assignment evidence -> {area (int): spatial_partition_id (str, leading zeros kept)}."""
    if recorded_sha is not None and rid.file_sha256(path) != recorded_sha:
        raise GateError(f"{path}: sha differs from the candidate.json assignment_evidence record")
    ev = pd.read_csv(path, dtype={"FEWSNET_admin_code": str, "spatial_partition_id": str,
                                  "prediction_branch_id": str}, keep_default_na=False)
    codes = ev["FEWSNET_admin_code"].to_numpy()
    if not all(c.isdigit() for c in codes):
        raise GateError(f"{path}: non-numeric FEWSNET_admin_code")
    areas = [int(c) for c in codes]
    if len(set(areas)) != len(areas):
        raise GateError(f"{path}: duplicate areas")
    sp = ev["spatial_partition_id"].to_numpy()
    if (sp == "").any():
        raise GateError(f"{path}: empty spatial_partition_id")
    if not np.array_equal(sp == S_NONE, ev["search_rows"].astype(np.int64).to_numpy() == 0):
        raise GateError(f"{path}: spatial_partition_id == s-1 is not equivalent to search_rows == 0")
    return dict(zip(areas, sp.tolist()))


def region_ids(areas, area_map: dict) -> np.ndarray:
    """Spatial id per row; missing areas -> '' (never a region)."""
    return np.array([area_map.get(int(a), "") for a in np.asarray(areas)], dtype=object)


def common_refit(root, fit, area_map: dict, floor: dict | None = None) -> tuple[dict, dict]:
    """One map arm: per named region, the current FIT rows of its member areas (original order);
    eligible (FIT_SUPPORT) regions get exactly one L1 continuation of ``root``.

    Returns (models {region: booster}, records {region: {...}}). Labels outside ``fit`` are
    never read. A continuation error propagates (no silent root fallback)."""
    floor = plan.FIT_SUPPORT if floor is None else floor
    X, y, g, m = fit
    rid_fit = region_ids(g, area_map)
    named = sorted({s for s in area_map.values() if s != S_NONE})
    root_bytes = nx.raw(root)
    models, records = {}, {}
    for region in named:
        rows = np.flatnonzero(rid_fit == region)
        members = sorted(a for a, s in area_map.items() if s == region)
        sup = nx.support(y[rows], g[rows], m[rows])
        rec = {"region": region, "member_areas": members, "member_areas_with_fitting_rows":
               int(np.unique(g[rows]).size), "fitting_row_index_sha256": _sha(rows),
               "fitting_keys_sha256": nx.keys_sha(g[rows], m[rows]), "support": sup,
               "eligible": bool(nx.meets(sup, floor)), "floor": dict(floor)}
        if rec["eligible"]:
            child, cont = nx.continue_booster(root, X[rows], y[rows], plan.L_CONFIGS["L1"])
            if cont["rounds_added"] != 20 or cont["parent_rounds"] != root.num_boosted_rounds():
                raise RuntimeError(f"region {region}: continuation did not add exactly 20 rounds")
            models[region] = child
            rec["continuation"] = cont
        records[region] = rec
    if nx.raw(root) != root_bytes:
        raise RuntimeError("the shared root changed during the common refit")
    return models, records


def _sha(rows) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(np.asarray(rows, dtype=np.int64)).tobytes()).hexdigest()


def route(areas, area_map: dict, records: dict) -> tuple[np.ndarray, np.ndarray]:
    """(spatial id, reason) per row: region / s-1 / missing / insufficient_support."""
    sid = region_ids(areas, area_map)
    reason = np.empty(len(sid), dtype=object)
    for i, s in enumerate(sid):
        if s == "":
            reason[i] = "missing"
        elif s == S_NONE:
            reason[i] = "s-1"
        elif records[s]["eligible"]:
            reason[i] = "region"
        else:
            reason[i] = "insufficient_support"
    return sid, reason


def arm_proba(X, sid, reason, p_root, models: dict) -> np.ndarray:
    """Region-model probabilities on 'region' rows; the current root's elsewhere."""
    out = np.array(p_root, dtype=np.float64, copy=True)
    for region in sorted(set(sid[reason == "region"].tolist())):
        rows = np.flatnonzero((sid == region) & (reason == "region"))
        out[rows] = nx.proba(models[region], X[rows])
    return out


def transition_groups(frame: pd.DataFrame, model: str, base: str = "root") -> dict:
    pers = frame["persistence_code"].to_numpy(float)
    t = rr.crisis(frame["truth"].to_numpy())
    grp = np.where(~np.isfinite(pers), "missing",
                   np.where(pers >= 2, "1", "0").astype(object) + np.where(t, "1", "0").astype(object))
    yb, ym = rr.crisis(frame[f"y_{base}"].to_numpy()), rr.crisis(frame[f"y_{model}"].to_numpy())
    out = {}
    for gname in GROUPS:
        k = grp == gname
        okm, okb = ym[k] == t[k], yb[k] == t[k]
        out[gname] = {"n": int(k.sum()), "corrected": int(np.sum(okm & ~okb)), "spoiled": int(np.sum(~okm & okb)),
                      "tp_change": int(np.sum(ym[k] & t[k]) - np.sum(yb[k] & t[k])),
                      "fp_change": int(np.sum(ym[k] & ~t[k]) - np.sum(yb[k] & ~t[k]))}
    return out


def routing_counts(frame: pd.DataFrame) -> dict:
    out = {}
    for arm, key in MAP_ARMS.items():
        r = frame[f"route_{key}"].to_numpy()
        counts = {x: int(np.sum(r == x)) for x in REASONS}
        n = int(len(r))
        out[arm] = {"n": n, "by_reason": counts, "root_rows": n - counts["region"],
                    "root_share": (n - counts["region"]) / n if n else None}
    return out


def _arms_block(frame: pd.DataFrame, arms=ARMS) -> dict:
    t = frame["truth"].to_numpy()
    out = {a: rr.score_block(t, frame[f"y_{a}"].to_numpy(), frame[[f"p_{a}_{l}" for l in LABELS]].to_numpy(float))
           for a in arms}
    for a, b in DELTAS:
        out[f"{a}_minus_{b}"] = rr.delta(out[a], out[b])
    return out


def score_frame(frame: pd.DataFrame) -> dict:
    """All keys (primary), matched persistence keys, both-covered subset (supplementary only)."""
    if len(frame) == 0:
        return {"n": 0, "status": "no_data"}
    out = {"n": int(len(frame)), "all": _arms_block(frame)}
    pers = frame["persistence_code"].to_numpy(float)
    k = np.isfinite(pers)
    matched = {"n": int(k.sum()), "coverage": float(k.mean())}
    if k.any():
        sub = frame[k]
        t = sub["truth"].to_numpy()
        matched.update(_arms_block(sub))
        pc = pers[k].astype(np.int64)
        matched["persistence"] = rr.score_block(t, pc, np.eye(nx.N_CLASSES)[pc])
        for a in ARMS:
            matched[f"{a}_minus_persistence"] = rr.delta(matched[a], matched["persistence"])
    out["matched_persistence"] = matched
    both = (frame["route_current"] == "region") & (frame["route_old"] == "region")
    out["both_covered_supplementary"] = ({"n": int(both.sum()), **_arms_block(frame[both])} if both.any()
                                         else {"n": 0, "status": "no_data"})
    out["routing"] = routing_counts(frame)
    out["transition_groups_post_hoc"] = {a: transition_groups(frame, a) for a in ARMS[1:]}
    return out


def _fdelta(a: dict, b: dict) -> dict:
    d = Fraction(a["crisis_f1_exact"]) - Fraction(b["crisis_f1_exact"])
    return {"crisis_f1_delta_exact": str(d), "crisis_f1_delta": float(d)}


def aggregate(per_pair: dict, select) -> dict:
    """Pooled confusion scores and mean-fold deltas kept separate; 12 folds are not independent."""
    out = {}
    for part in PARTS:
        have = [r["scores"][part] for r in per_pair.values() if select(r) and r["scores"][part].get("n", 0)]
        if not have:
            out[part] = {"folds_with_data": 0, "status": "no_data"}
            continue
        allp = rr.pooled({a: np.sum([h["all"][a]["confusion_fourclass"] for h in have], axis=0) for a in ARMS})
        for a, b in DELTAS:
            allp[f"{a}_minus_{b}"] = _fdelta(allp[a], allp[b])
        res = {"folds_with_data": len(have), "rows": int(sum(h["n"] for h in have)), "pooled_all": allp,
               "mean_fold_all": {f"{a}_minus_{b}": {
                   "crisis_f1": float(np.mean([h["all"][f"{a}_minus_{b}"]["crisis_f1_delta"] for h in have])),
                   "crisis_brier": float(np.mean([h["all"][f"{a}_minus_{b}"]["crisis_brier_delta"] for h in have])),
                   "macro_f1_fourclass": float(np.mean([h["all"][f"{a}_minus_{b}"]["macro_f1_fourclass_delta"]
                                                        for h in have]))} for a, b in DELTAS},
               "mean_crisis_brier_all": {a: float(np.mean([h["all"][a]["crisis_brier"] for h in have])) for a in ARMS}}
        mh = [h["matched_persistence"] for h in have if h["matched_persistence"]["n"]]
        if mh:
            mp = rr.pooled({a: np.sum([h[a]["confusion_fourclass"] for h in mh], axis=0)
                            for a in ARMS + ("persistence",)})
            for a in ARMS:
                mp[f"{a}_minus_persistence"] = _fdelta(mp[a], mp["persistence"])
            res["pooled_matched_persistence"] = mp | {
                "rows": int(sum(h["n"] for h in mh)),
                "mean_fold": {f"{a}_minus_persistence": float(np.mean(
                    [h[f"{a}_minus_persistence"]["crisis_f1_delta"] for h in mh])) for a in ARMS}}
        res["routing"] = {arm: {"n": int(sum(h["routing"][arm]["n"] for h in have)),
                                "by_reason": {x: int(sum(h["routing"][arm]["by_reason"][x] for h in have))
                                              for x in REASONS}} for arm in MAP_ARMS}
        for arm in MAP_ARMS:
            r = res["routing"][arm]
            r["root_share"] = (r["n"] - r["by_reason"]["region"]) / r["n"] if r["n"] else None
        res["transition_groups_post_hoc"] = {
            a: {gname: {k: int(sum(h["transition_groups_post_hoc"][a][gname][k] for h in have))
                        for k in ("n", "corrected", "spoiled", "tp_change", "fp_change")} for gname in GROUPS}
            for a in ARMS[1:]}
        out[part] = res
    return out


# ---------------------------------------------------------------- per pair

def brier_candidate(cands: dict, root_name: str) -> str:
    found = [c for c, e in cands.items() if e["root"] == root_name and e["e1"] == "brier_crisis"]
    if len(found) != 1:
        raise GateError(f"{root_name}: expected exactly one D34 Brier candidate, got {found}")
    return found[0]


def load_map(stage: Path, cand: str) -> tuple[dict, dict]:
    cdir = stage / "candidates" / cand
    record = json.loads((cdir / "candidate.json").read_text(encoding="utf-8"))["assignment_evidence"]
    if record.get("schema") != "d32-v1" or record.get("file") != "assignment_evidence.csv":
        raise GateError(f"{cand}: candidate.json lacks the D32 assignment_evidence block")
    area_map = parse_map(cdir / "assignment_evidence.csv", record["sha256"])
    named = sorted({s for s in area_map.values() if s != S_NONE})
    return area_map, {"candidate": cand, "assignment_evidence_sha256": record["sha256"], "areas": len(area_map),
                      "named_regions": named, "s-1_areas": int(sum(s == S_NONE for s in area_map.values()))}


def check_global20(d35: Path, name: str, root: dict, data: dict, gate: dict):
    """Replay the saved D35 global+20 of this root on current C/E3; equal to D35's saved values."""
    def check(key, n_bad, n):
        gate["checks"][key] = {"mismatches": int(n_bad), "n": int(n)}

    completion = json.loads((d35 / "completion.json").read_text(encoding="utf-8"))
    recorded = completion.get("outputs", {})
    check("d35_completed", int(completion.get("status") != "completed"), 1)
    files = ("global_plus20.ubj", "continuation.json", "rows_C.csv.gz", "rows_E3.csv.gz")
    check("d35_file_hashes", sum(recorded.get(f"{name}/{f}") != rid.file_sha256(d35 / name / f) for f in files), 4)
    cont = json.loads((d35 / name / "continuation.json").read_text(encoding="utf-8"))
    check("d35_root_identity", int(cont.get("root") != name or cont.get("root_booster_sha256") !=
                                   root["root_booster_sha256"]), 1)
    check("d35_fitting_keys", int(cont.get("fitting_keys_sha256") != nx.keys_sha(data["FIT"][2], data["FIT"][3])), 1)
    check("d35_booster_sha", int(cont.get("global_plus20_sha256") != rid.file_sha256(d35 / name / "global_plus20.ubj")), 1)
    booster = nx.from_raw((d35 / name / "global_plus20.ubj").read_bytes())
    probs = {}
    for part in PARTS:
        X, y, g, m = data[part]
        p = nx.proba(booster, X)
        saved = gi.read_csv(d35 / name / f"rows_{part}.csv.gz")
        n = len(y)
        ok = len(saved) == n and (saved["area"].to_numpy() == g).all() and \
            (saved["target_month"].astype(str).to_numpy() == np.asarray(m).astype(str)).all()
        check(f"d35_{part}_keys", int(not ok), n)
        if ok:
            check(f"d35_{part}_truth", np.sum(saved["truth"].to_numpy() != y), n)
            check(f"d35_{part}_p_global20", sr.exact_mismatches(saved[[f"p_global20_{l}" for l in LABELS]], p), 4 * n)
        probs[part] = p
    return probs


def save_arm(adir: Path, models: dict, records: dict, map_meta: dict, root_sha: str) -> dict:
    """Write each region UBJ + record; return the RELOADED boosters (used for every prediction)."""
    adir.mkdir(parents=True)
    frozen = {}
    for region, rec in records.items():
        tag = f"region_{region}"
        payload = {**rec, "map": map_meta, "root_booster_sha256": root_sha, "local_config": "L1"}
        if region in models:
            path = adir / f"{tag}.ubj"
            path.write_bytes(nx.raw(models[region]))
            frozen[region] = nx.from_raw(path.read_bytes())
            payload["ubj_sha256"] = rid.file_sha256(path)
            if payload["ubj_sha256"] != rec["continuation"]["booster_sha256"]:
                raise RuntimeError(f"region {region}: saved UBJ differs from the continuation record")
        rid.write_json_atomic(adir / f"{tag}.json", payload)
    return frozen


def run_pair(run, stage, d35, out, h, t, u, roots, cands, phase_col, base, fit_counter):
    g_cfg = plan.TB3_G[str(h)]
    name, old_root = plan.e1pair_root_name(h, t, g_cfg), plan.e1pair_root_name(h, u, g_cfg)
    if name not in roots or old_root not in roots:
        raise GateError(f"pair H{h} {t}/{u}: root not in the accepted D34 schedule")
    root = json.loads((stage / "roots" / name / "root.json").read_text(encoding="utf-8"))
    pair = {cands[c]["e1"]: c for c in cands if cands[c]["root"] == name}
    gate, data, replay, root_ubj = rr.gate_root(run, stage, name, pair, root)
    gate["pair"] = {"horizon": h, "target_month": t, "old_map_target": u}
    if not gate["passed"]:
        return gate, None, None
    p20 = check_global20(d35, name, root, data, gate)
    maps = {"current": load_map(stage, brier_candidate(cands, name)),
            "old": load_map(stage, brier_candidate(cands, old_root))}
    spec = {(sh, st): (so, sc) for sh, st, _, so, sc in SPEC_PAIRS}[(h, t)]
    _, y_fit, g_fit, m_fit = data["FIT"]
    rid_fit = {k: region_ids(g_fit, maps[k][0]) for k in maps}
    pre = {k: sum(nx.meets(nx.support(y_fit[rid_fit[k] == s], g_fit[rid_fit[k] == s], m_fit[rid_fit[k] == s]),
                           plan.FIT_SUPPORT) for s in maps[k][1]["named_regions"]) for k in maps}
    gate["checks"]["eligible_regions_equal_table"] = {"mismatches": int((pre["old"], pre["current"]) != spec),
                                                      "n": 1, "expected_old_current": list(spec),
                                                      "found_old_current": [pre["old"], pre["current"]]}
    gate["passed"] = all(c["mismatches"] == 0 for c in gate["checks"].values())
    if not gate["passed"]:
        return gate, None, None

    booster = nx.from_raw(root_ubj.read_bytes())
    root_sha = root["root_booster_sha256"]
    pdir = out / name
    routes, frozen, records = {}, {}, {}
    for key in ("current", "old"):
        models, recs = common_refit(booster, data["FIT"], maps[key][0])
        fit_counter[0] += len(models)
        if fit_counter[0] > MAX_FITS:
            raise RuntimeError(f"fit budget exceeded: {fit_counter[0]} > {MAX_FITS}")
        frozen[key] = save_arm(pdir / f"{key}_map", models, recs, maps[key][1], root_sha)
        for region, b in models.items():          # save/reload replay must be identical
            X_chk = data["FIT"][0][rid_fit[key] == region][:200]
            if not np.array_equal(nx.proba(b, X_chk), nx.proba(frozen[key][region], X_chk)):
                raise RuntimeError(f"{key} map region {region}: reloaded booster predicts differently")
        records[key] = recs
    if rid.file_sha256(root_ubj) != root_sha or nx.sha(booster) != nx.sha(nx.from_raw(root_ubj.read_bytes())):
        raise RuntimeError("current root booster changed")

    rows = {}
    for part in PARTS:
        X, y, g, m = data[part]
        frame = pd.DataFrame({"root": name, "part": part, "area": g, "target_month": m, "horizon": h, "truth": y,
                              "persistence_code": rr.persistence_codes(X[:, phase_col])})
        probs = {"root": replay[part], "global20": p20[part]}
        for arm, key in MAP_ARMS.items():
            sid, reason = route(g, maps[key][0], records[key])
            frame[f"region_{key}"] = sid
            frame[f"route_{key}"] = reason
            probs[arm] = arm_proba(X, sid, reason, replay[part], frozen[key])
        for arm in ARMS:
            frame[f"y_{arm}"] = fourclass.argmax_codes(probs[arm])
            for k, lab in enumerate(LABELS):
                frame[f"p_{arm}_{lab}"] = probs[arm][:, k]
        rows[part] = frame
        with gzip.open(pdir / f"rows_{part}.csv.gz", "wt", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False, float_format="%.17g")
    dc = rr.dev_baseline_check(rows["E3"], base, h)
    if dc["status"] == "checked" and (dc["joined"] != dc["e3_rows"] or dc["persistence_mismatches"]
                                      or dc["truth_mismatches"]):
        gate = {**gate, "passed": False, "error": f"dev_baselines cross-check failed: {dc}"}
    meta = {"root": name, "horizon": h, "target_month": t, "old_map_target": u,
            "origin_month": month_label(np.array([data["window"][2]]))[0], "g_config": g_cfg,
            "months_T_minus_U": _mi(t) - _mi(u), "root_booster_sha256": root_sha,
            "maps": {k: maps[k][1] for k in maps},
            "eligible_regions": {k: sum(r["eligible"] for r in records[k].values()) for k in records},
            "regions": {k: {s: {"eligible": r["eligible"], "support": r["support"],
                                "members": len(r["member_areas"])} for s, r in records[k].items()} for k in records},
            "fits": {k: sum(r["eligible"] for r in records[k].values()) for k in records},
            "dev_baselines_check": dc}
    rid.write_json_atomic(pdir / "pair.json", meta)
    return gate, rows, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--d34-run", required=True, type=Path)
    parser.add_argument("--d35-run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--producer-rev", default="7b2bf6f")
    args = parser.parse_args()
    run, d35, out = args.d34_run.resolve(), args.d35_run.resolve(), args.out.resolve()
    if out.is_relative_to(run) or out.is_relative_to(d35):
        raise ValueError("--out must not be inside the read-only D34 / D35 runs")
    rid.refuse_existing(out, "D42 map transfer")
    producer = rid.code_identity_at(args.producer_rev)
    if rid.code_identity() != rid.code_identity_at("HEAD"):
        raise RuntimeError("D42 must run from committed package code (working tree differs from HEAD)")
    script_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PACKAGE, capture_output=True, text=True,
                                   check=True).stdout.strip()
    pairs = check_schedule()
    roots, cands, ident = accept_mode(run, plan.E1PAIR, producer, args.producer_rev)
    stage = Path(ident["stage"])
    d35_ident = json.loads((d35 / "identity.json").read_text(encoding="utf-8"))
    if d35_ident.get("stage") != "d35_global_increment" or d35_ident.get("producer_rev") != args.producer_rev:
        raise ValueError("--d35-run is not a D35 global-increment run of the same D34 producer")
    phase_col = load_schema(sr.SCHEMA)["ordered_features"].index("hist_phase_o00")
    base_path = run / "prepared" / "ledgers" / "dev_baselines.csv"   # dev targets only; never baselines.csv
    base = pd.read_csv(base_path)
    out.mkdir(parents=True)
    identity = {"stage": "d42_map_transfer", "d34_run": str(run), "d35_run": str(d35),
                "producer_rev": args.producer_rev, "producer_code": producer, "script_commit": script_commit,
                "script_code": rid.code_identity(), "runtime": rid.runtime_identity(), "max_month": "2020-12",
                "schedule": [list(p) for p in SPEC_PAIRS], "fit_budget": MAX_FITS,
                "fit_support": dict(plan.FIT_SUPPORT), "local_config": plan.L_CONFIGS["L1"],
                "dev_baselines": {"path": str(base_path), "sha256": rid.file_sha256(base_path)},
                "d35_identity_sha256": rid.file_sha256(d35 / "identity.json"),
                "d35_completion_sha256": rid.file_sha256(d35 / "completion.json"),
                "acceptance": {k: (str(v) if isinstance(v, Path) else v) for k, v in ident.items()},
                "inputs": {}}
    gates, per_pair, fit_counter = {}, {}, [0]
    try:
        for h, t, u in pairs:
            g_cfg = plan.TB3_G[str(h)]
            name, old_root = plan.e1pair_root_name(h, t, g_cfg), plan.e1pair_root_name(h, u, g_cfg)
            pair = {cands[c]["e1"]: c for c in cands if cands[c]["root"] == name}
            identity["inputs"][name] = {
                "snapshot": rid.file_sha256(run / "prepared" / f"snapshot_h{h}.parquet"),
                **{f: rid.file_sha256(stage / "roots" / name / f)
                   for f in ("root.json", "fold_membership.csv.gz", "root_target_predictions.csv")},
                **{f"{c}/{f}": rid.file_sha256(stage / "candidates" / c / f) for c in pair.values()
                   for f in ("candidate.json", "validation_predictions.csv.gz", "confirmation_predictions.csv.gz",
                             "target_predictions.csv", "assignment_evidence.csv")},
                **{f"{c}/xgb_root.ubj": rid.file_sha256(stage / "checkpoints" / c / "xgb_root.ubj")
                   for c in pair.values()},
                **{f"old_map/{brier_candidate(cands, old_root)}/{f}":
                   rid.file_sha256(stage / "candidates" / brier_candidate(cands, old_root) / f)
                   for f in ("candidate.json", "assignment_evidence.csv")},
                **{f"d35/{name}/{f}": rid.file_sha256(d35 / name / f)
                   for f in ("global_plus20.ubj", "continuation.json", "rows_C.csv.gz", "rows_E3.csv.gz")}}
            try:
                gate, rows, meta = run_pair(run, stage, d35, out, h, t, u, roots, cands, phase_col, base, fit_counter)
            except GateError as exc:
                gate, rows, meta = {"root": name, "passed": False, "error": str(exc)}, None, None
            gates[name] = gate
            print(f"{name}: gate {'passed' if gate['passed'] else 'FAILED'}; fits so far {fit_counter[0]}", flush=True)
            if not gate["passed"]:                 # identity / replay failure stops the study (D42)
                rid.write_json_atomic(out / "gate.json", {
                    "passed": False, "fits": fit_counter[0], "fit_budget": MAX_FITS, "stopped_at": name,
                    "rule": "a failed pair gate stops the study before any later pair is gated or fitted",
                    "pairs": gates})
                rid.write_json_atomic(out / "identity.json", identity)
                print(f"GATE FAILED at {name}: study stopped; see gate.json")
                return 2
            if rows is not None:
                per_pair[name] = {**meta, "scores": {p: score_frame(rows[p]) for p in PARTS}}
    except Exception as exc:                       # engineering failure: keep the evidence, then stop
        rid.write_json_atomic(out / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc(),
                                                     "fits_so_far": fit_counter[0], "gates": gates})
        rid.write_json_atomic(out / "identity.json", identity)
        raise
    passed = all(g["passed"] for g in gates.values())
    expected_fits = sum(r["fits"]["current"] + r["fits"]["old"] for r in per_pair.values())
    if fit_counter[0] != expected_fits or fit_counter[0] > MAX_FITS:
        passed = False
        gates["_fit_count"] = {"passed": False, "fits": fit_counter[0], "eligible_regions": expected_fits}
    rid.write_json_atomic(out / "gate.json", {"passed": passed, "fits": fit_counter[0], "fit_budget": MAX_FITS,
                                              "rule": "exact equality, round-trip parse; a pair failing a pre-fit "
                                                      "gate is not fitted", "pairs": gates})
    rid.write_json_atomic(out / "identity.json", identity)
    if not passed:
        print("GATE FAILED for at least one pair: no summary computed; see gate.json")
        return 2
    summary = {
        "per_pair": per_pair,
        "by_horizon": {f"H{h}": aggregate(per_pair, lambda r, h=h: r["horizon"] == h) for h in plan.HORIZONS},
        "overall_12": aggregate(per_pair, lambda r: True),
        "fits": fit_counter[0],
        "interpretation": ("Primary: same-key E3 for root / global20 / current_map_refit / old_map_refit and "
                           "versus exact-origin persistence on matched non-missing keys. C is descriptive only "
                           "(the old map may have seen current C labels). Each pair lists both maps' eligible "
                           "region counts and per-arm root-fallback shares. Both-covered subset is supplementary. "
                           "mean_fold averages per-pair deltas; pooled sums confusions. The two maps differ in "
                           "region count, membership and support: differences are not attributable to map age "
                           "alone. 12 overlapping, repeatedly developed folds; not comparable to the 21-root "
                           "aggregates; no significance test."),
    }
    rid.write_json_atomic(out / "summary.json", summary)
    rid.write_json_atomic(out / "completion.json", {"status": "completed", "outputs": rid.output_hashes(out)})
    print(f"D42 map transfer completed: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
