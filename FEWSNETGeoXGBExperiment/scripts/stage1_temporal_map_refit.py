#!/usr/bin/env python3
"""D43 / A17: temporal Brier map learning under common forecasting (21 fixed (H, T) pairs).

python scripts/stage1_temporal_map_refit.py --d34-run D34_RUN --d35-run D35_RUN --out NEW_DIR

Sequential. Phase 1 (before any temporal search): three production-equivalence searches at
T=2018-06 (H4/H8/H12): the reconstructed D34 FIT/S/C roles, the SAVED D34 root and its root-fit
record, the existing Brier/root/L1/gt0 run_candidate with the original C tuple; D32 assignment
evidence, routed native UBJs, predictions and scores must equal the saved D34 Brier candidate.
Phase 2, per pair: D34 acceptance + D37 gate_root, recomputed search-month table, the legal pool
FIT u S u C restored in membership order, D35 global20 exact replay; then ONE search root fitted on
the time-block fitting rows FIT_tb (latest three pool months = search S_tb) and ONE Brier candidate
(root increments, L1, gt0, no confirmation). Both the D34 random Brier map and the new temporal
map are refitted (D42 common_refit) from the CURRENT D34 root on the CURRENT D34 FIT rows. Four
same-key E3 arms: root, global20, random_map_refit, temporal_map_refit. The search root's E3
predictions are descriptive only. Any failed gate / identity / replay / fit stops the study.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from scripts import stage1_global_increment as gi  # noqa: E402
from scripts import stage1_map_transfer as mt  # noqa: E402
from scripts import stage1_recency_root as rr  # noqa: E402
from scripts import stage1_shallow_replay as sr  # noqa: E402
from scripts.stage1_rootconf_compare import accept_mode  # noqa: E402
from src.experiment import plan  # noqa: E402
from src.feature.fourclass_features import load_schema, month_label  # noqa: E402
from src.metrics import fourclass  # noqa: E402
from src.model import native_xgb as nx  # noqa: E402
from src.utils import run_identity as rid  # noqa: E402
from src.utils.split import time_block_split  # noqa: E402

D34_REV, D35_REV = "7b2bf6f", "be5f485"
LABELS = fourclass.CLASS_LABELS
ARMS = ("root", "global20", "random_map_refit", "temporal_map_refit")
MAP_ARMS = {"random_map_refit": "random", "temporal_map_refit": "temporal"}
DELTAS = (("global20", "root"), ("random_map_refit", "root"), ("temporal_map_refit", "root"),
          ("random_map_refit", "global20"), ("temporal_map_refit", "global20"),
          ("temporal_map_refit", "random_map_refit"))
GateError = rr.GateError
EQUIV_TARGET = "2018-06"
MAX_SEARCH_ROOTS, MAX_SEARCHES, MAX_CHILD_FITS, MAX_REFITS = 21, 24, 1488, 838
MAX_TERMINALS = 32

#: Frozen D43 search-month table: (H, T) -> the last three distinct non-heldout D34 pool months.
SEARCH_MONTHS = {}
for _t, _h4, _h8, _h12 in (
        ("2018-06", "2017-02,2017-06,2017-10", "2016-10,2017-02,2017-06", "2016-06,2016-10,2017-02"),
        ("2018-10", "2017-06,2017-10,2018-02", "2017-02,2017-06,2017-10", "2016-10,2017-02,2017-06"),
        ("2019-02", "2017-10,2018-02,2018-06", "2017-06,2017-10,2018-02", "2017-02,2017-06,2017-10"),
        ("2019-06", "2018-02,2018-06,2018-10", "2017-10,2018-02,2018-06", "2017-06,2017-10,2018-02"),
        ("2019-10", "2018-06,2018-10,2019-02", "2018-02,2018-06,2018-10", "2017-10,2018-02,2018-06"),
        ("2020-02", "2018-10,2019-02,2019-06", "2018-06,2018-10,2019-02", "2018-02,2018-06,2018-10"),
        ("2020-06", "2019-02,2019-06,2019-10", "2018-10,2019-02,2019-06", "2018-06,2018-10,2019-02")):
    for _h, _s in ((4, _h4), (8, _h8), (12, _h12)):
        SEARCH_MONTHS[(_h, _t)] = tuple(_s.split(","))
SCHEDULE = tuple((h, t) for h in plan.HORIZONS for t in plan.E1PAIR_TARGETS)


def temporal_candidate_name(h, t, g) -> str:
    return f"h{h}_{t}_{g}_{plan.ROOTINC_LOCAL}_tb3pool_s42_d43temporal_e1brier_{plan.ROOTINC_FAMILY}"


# ---------------------------------------------------------------- pure helpers (tested)

def membership_search_months(membership: pd.DataFrame, n: int = plan.TIME_BLOCK_MONTHS) -> list:
    """Last ``n`` distinct non-heldout label months (YYYY-MM) of a D34 fold membership."""
    months = membership.loc[membership["role"] != "heldout_target", "target_month"].astype(str)
    return sorted(set(months.tolist()), key=mt._mi)[-n:]


def check_search_months(membership: pd.DataFrame, expected) -> list:
    got = membership_search_months(membership)
    if got != list(expected):
        raise GateError(f"search months {got} differ from the frozen D43 table {list(expected)}")
    return got


def reconstruct_pool(data: dict) -> dict:
    """FIT u S u C of a sr.rebuild(..., with_fitting=True) result, restored to membership order.

    Returns X, y, g, m (int month index), role ('fitting' / 'validation' / 'confirmation')."""
    mem = data["membership"]
    mem = mem[mem["role"] != "heldout_target"].reset_index(drop=True)
    role = mem["role"].to_numpy().astype(str)
    parts = {"fitting": "FIT", "validation": "S", "confirmation": "C"}
    if set(role) - set(parts):
        raise GateError(f"unexpected D34 roles {sorted(set(role) - set(parts))}")
    n_feat = data["FIT"][0].shape[1]
    X = np.full((len(mem), n_feat), np.nan)
    y = np.zeros(len(mem), dtype=np.int64)
    g = np.zeros(len(mem), dtype=np.int64)
    mo = np.zeros(len(mem), dtype=np.int64)
    for r, key in parts.items():
        pos = np.flatnonzero(role == r)
        Xp, yp, gp, mp = data[key]
        mp = np.asarray([mt._mi(v) if isinstance(v, str) else int(v) for v in mp], dtype=np.int64)
        if len(pos) != len(yp):
            raise GateError(f"role {r}: {len(pos)} membership rows vs {len(yp)} rebuilt rows")
        X[pos], y[pos], g[pos], mo[pos] = Xp, yp, gp, mp
    if not (np.array_equal(g, mem["area"].to_numpy(dtype=np.int64))
            and month_label(mo).tolist() == mem["target_month"].astype(str).tolist()
            and np.array_equal(y, mem["class_code"].to_numpy(dtype=np.int64))):
        raise GateError("reconstructed pool keys / labels differ from the membership order")
    if len(set(zip(g.tolist(), mo.tolist()))) != len(g):
        raise GateError("duplicate (area, month) keys in the legal pool")
    for r, key in parts.items():
        k = role == r
        if not (np.array_equal(X[k], data[key][0], equal_nan=True)
                and nx.keys_sha(g[k], mo[k]) == nx.keys_sha(data[key][2], [mt._mi(v) if isinstance(v, str)
                                                                         else int(v) for v in data[key][3]])):
            raise GateError(f"role {r}: features / keys differ from the rebuilt part")
    lo, hi, o_index = data["window"]
    if mo.min() < o_index - plan.WINDOW or mo.max() >= o_index:
        raise GateError("legal pool outside [O-59, O)")
    return {"X": X, "y": y, "g": g, "m": mo, "role": role, "o_index": int(o_index)}


def production_inputs(pool: dict, test: tuple, root_booster):
    """D34 main() search data: S = x_set 1, FIT = 0, C removed; C passed as the confirmation tuple."""
    Xt, yt, gt = test
    conf = pool["role"] == "confirmation"
    keep = ~conf
    x_set = (pool["role"] == "validation").astype(int)
    y_pool = fourclass.argmax_codes(nx.proba(root_booster, Xt))
    data = (pool["X"][keep], pool["y"][keep], pool["g"][keep], pool["m"][keep], x_set[keep], Xt, yt, gt, y_pool)
    return data, (pool["X"][conf], pool["y"][conf], pool["g"][conf], pool["m"][conf])


def temporal_split(pool: dict, expected) -> dict:
    try:
        return time_block_split(pool["g"], pool["m"], pool["o_index"], plan.TIME_BLOCK_MONTHS, expected)
    except ValueError as exc:
        raise GateError(f"time block: {exc}") from exc


def fit_search_root(pool: dict, x_set_tb, g_config: dict):
    """The D43 search root on FIT_tb, recorded as main_model_GF.main records its root."""
    f = np.asarray(x_set_tb) == 0
    sup = nx.support(pool["y"][f], pool["g"][f], pool["m"][f])
    if sup["classes"] < 2:
        raise GateError("search-root fitting rows have fewer than two observed classes")
    booster, record = nx.fit_global(pool["X"][f], pool["y"][f], g_config)
    record.update(fit_keys_sha256=nx.keys_sha(pool["g"][f], pool["m"][f]), fit_support=sup)
    return booster, record


def run_search(name, root, data, work, ckpt, contiguity_info, features, confirmation=None):
    """The existing Brier / root-increment / L1 / gt0 candidate search (main_model_GF.run_candidate)."""
    from app import main_model_GF as mgf
    return mgf.run_candidate(name, plan.ROOTINC_LOCAL, plan.ROOTINC_FAMILY, root, data, work, ckpt,
                             contiguity_info, features, increment_source="root", confirmation=confirmation,
                             e1="brier_crisis")


def overlap_record(pool: dict, x_set_tb) -> dict:
    s = np.asarray(x_set_tb) == 1
    keys = lambda k: set(zip(pool["g"][k].tolist(), pool["m"][k].tolist()))   # noqa: E731
    s_keys = keys(s)
    by_role = {r: len(s_keys & keys(pool["role"] == r)) for r in ("fitting", "validation", "confirmation")}
    if sum(by_role.values()) != len(s_keys):
        raise GateError("S_tb keys are not covered exactly by D34 FIT/S/C")
    f = ~s
    return {"S_tb_rows": int(s.sum()), "S_tb_areas": int(np.unique(pool["g"][s]).size),
            "S_tb_dates": int(np.unique(pool["m"][s]).size), "S_tb_keys_sha256": nx.keys_sha(pool["g"][s], pool["m"][s]),
            "S_tb_overlap_rows_by_d34_role": by_role,
            "S_tb_share_by_d34_role": {r: v / len(s_keys) for r, v in by_role.items()} if s_keys else None,
            "FIT_tb_rows": int(f.sum()), "FIT_tb_last_label_month": month_label([pool["m"][f].max()])[0],
            "d34_fit_last_label_month": month_label([pool["m"][pool["role"] == "fitting"].max()])[0],
            "search_root_age_months_at_O": int(pool["o_index"] - pool["m"][f].max())}


def pool_membership_frame(pool: dict, x_set_tb, e3_areas, e3_month: str) -> pd.DataFrame:
    """Every recombined legal key in pool order with its D34 role and D43 temporal role; E3 keys
    appended last as excluded (never in the search / fitting pool)."""
    x = np.asarray(x_set_tb)
    legal = pd.DataFrame({"area": pool["g"], "target_month": month_label(pool["m"]), "d34_role": pool["role"],
                          "temporal_role": np.where(x == 1, "S_tb", "FIT_tb")})
    e3 = pd.DataFrame({"area": np.asarray(e3_areas, dtype=np.int64), "target_month": e3_month,
                       "d34_role": "heldout_target", "temporal_role": "excluded_E3"})
    return pd.concat([legal, e3], ignore_index=True)


def common_arm(root, fit, area_map: dict):
    """D42 common_refit plus the D43 guard: every refit continues the CURRENT root on CURRENT FIT rows."""
    models, recs = mt.common_refit(root, fit, area_map)
    root_sha = nx.sha(root)
    fit_keys = set(zip(np.asarray(fit[2]).tolist(), np.asarray(fit[3]).tolist()))
    for region, rec in recs.items():
        if region in models and rec["continuation"]["parent_sha256"] != root_sha:
            raise RuntimeError(f"region {region}: refit did not continue the current forecasting root")
        if rec["support"]["rows"] > len(fit_keys):
            raise RuntimeError(f"region {region}: refit pool larger than the current FIT pool")
    return models, recs


def _text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8")


def compare_production(saved_cdir: Path, saved_ckpt: Path, new_cdir: Path, new_ckpt: Path) -> dict:
    """Saved D34 candidate vs replayed search: bytes / decompressed text / scores (no paths, timings)."""
    checks = {}
    saved_rec = json.loads((saved_cdir / "candidate.json").read_text(encoding="utf-8"))
    new_rec = json.loads((new_cdir / "candidate.json").read_text(encoding="utf-8"))
    checks["assignment_evidence_bytes"] = int((saved_cdir / "assignment_evidence.csv").read_bytes()
                                              != (new_cdir / "assignment_evidence.csv").read_bytes())
    for f in ("correspondence_table.csv", "target_predictions.csv", "heldout_scores.csv",
              "validation_predictions.csv.gz", "confirmation_predictions.csv.gz", "e2_predictions.csv.gz"):
        checks[f] = int(_text(saved_cdir / f) != _text(new_cdir / f))
    terminals = saved_rec["partition"]["terminal_partitions"]
    checks["terminal_partitions"] = int(terminals != new_rec["partition"]["terminal_partitions"])
    checks["routed_ubj_bytes"] = sum((saved_ckpt / f"xgb_{b}.ubj").read_bytes() != (new_ckpt / f"xgb_{b}.ubj").read_bytes()
                                     for b in terminals)
    dump = lambda x: json.dumps(x, sort_keys=True, default=str)   # noqa: E731
    checks["decisions"] = int(dump(saved_rec["partition"]["decisions"]) != dump(new_rec["partition"]["decisions"]))
    checks["scores"] = int(dump(saved_rec["scores"]) != dump(new_rec["scores"]))
    conf = lambda r: {k: v for k, v in r.get("confirmation", {}).items() if k != "frozen_digest_before_scoring"}  # noqa: E731
    checks["confirmation_scores"] = int(dump(conf(saved_rec)) != dump(conf(new_rec)))
    info = {"all_checkpoint_sha_equal": saved_rec["checkpoints"]["sha256"] == new_rec["checkpoints"]["sha256"],
            "frozen_digest_equal": saved_rec.get("confirmation", {}).get("frozen_digest_before_scoring")
            == new_rec.get("confirmation", {}).get("frozen_digest_before_scoring"),
            "child_fits": new_rec["fits"]["child_fits"], "terminals": terminals}
    return {"checks": checks, "passed": all(v == 0 for v in checks.values()), "info": info}


# ---------------------------------------------------------------- scoring

def decision_changes(frame: pd.DataFrame, a: str, b: str) -> dict:
    t = rr.crisis(frame["truth"].to_numpy())
    ya, yb = rr.crisis(frame[f"y_{a}"].to_numpy()), rr.crisis(frame[f"y_{b}"].to_numpy())
    oka, okb = ya == t, yb == t
    return {"changed": int(np.sum(ya != yb)), "corrected": int(np.sum(oka & ~okb)), "spoiled": int(np.sum(~oka & okb)),
            "tp_change": int(np.sum(ya & t) - np.sum(yb & t)), "fp_change": int(np.sum(ya & ~t) - np.sum(yb & ~t))}


def routing_counts(frame: pd.DataFrame) -> dict:
    out = {}
    for arm, key in MAP_ARMS.items():
        r = frame[f"route_{key}"].to_numpy()
        counts = {x: int(np.sum(r == x)) for x in mt.REASONS}
        n = int(len(r))
        out[arm] = {"n": n, "by_reason": counts, "root_share": (n - counts["region"]) / n if n else None}
    return out


def _block(frame: pd.DataFrame) -> dict:
    t = frame["truth"].to_numpy()
    out = {a: rr.score_block(t, frame[f"y_{a}"].to_numpy(), frame[[f"p_{a}_{l}" for l in LABELS]].to_numpy(float))
           for a in ARMS}
    for a, b in DELTAS:
        out[f"{a}_minus_{b}"] = rr.delta(out[a], out[b]) | {"decisions": decision_changes(frame, a, b)}
    return out


def score_frame(frame: pd.DataFrame) -> dict:
    if len(frame) == 0:
        return {"n": 0, "status": "no_data"}
    out = {"n": int(len(frame)), "all": _block(frame)}
    pers = frame["persistence_code"].to_numpy(float)
    k = np.isfinite(pers)
    matched = {"n": int(k.sum()), "coverage": float(k.mean())}
    if k.any():
        sub = frame[k]
        matched.update(_block(sub))
        pc = pers[k].astype(np.int64)
        matched["persistence"] = rr.score_block(sub["truth"].to_numpy(), pc, np.eye(nx.N_CLASSES)[pc])
        for a in ARMS:
            matched[f"{a}_minus_persistence"] = rr.delta(matched[a], matched["persistence"])
    out["matched_persistence"] = matched
    out["routing"] = routing_counts(frame)
    out["transition_groups_post_hoc"] = {a: mt.transition_groups(frame, a) for a in ARMS[1:]}
    return out


def _pool_block(blocks: list, arms: tuple, deltas) -> dict:
    res = rr.pooled({a: np.sum([b[a]["confusion_fourclass"] for b in blocks], axis=0) for a in arms})
    n = sum(b[arms[0]]["n"] for b in blocks)
    for a in arms:
        res[a]["crisis_brier"] = (float(sum(b[a]["crisis_brier"] * b[a]["n"] for b in blocks) / n)
                                  if all(b[a].get("crisis_brier") is not None for b in blocks) else None)
    for a, b in deltas:
        res[f"{a}_minus_{b}"] = mt._fdelta(res[a], res[b]) | {
            "macro_f1_fourclass_delta": res[a]["macro_f1_fourclass"] - res[b]["macro_f1_fourclass"],
            "crisis_brier_delta": (None if res[a]["crisis_brier"] is None or res[b]["crisis_brier"] is None
                                   else res[a]["crisis_brier"] - res[b]["crisis_brier"]),
            "decisions": {k: int(sum(x[f"{a}_minus_{b}"]["decisions"][k] for x in blocks)) for k in
                          ("changed", "corrected", "spoiled", "tp_change", "fp_change")}
            if all("decisions" in x.get(f"{a}_minus_{b}", {}) for x in blocks) else None}
    res["rows"] = int(n)
    return res


def aggregate(per_pair: dict, select) -> dict:
    """E3 pooled confusion/row metrics and mean-fold deltas kept separate; 21 folds are correlated."""
    have = [r["scores"]["E3"] for r in per_pair.values() if select(r) and r["scores"]["E3"].get("n", 0)]
    if not have:
        return {"folds_with_data": 0, "status": "no_data"}
    allb = [h["all"] for h in have]
    res = {"folds_with_data": len(have), "pooled_all": _pool_block(allb, ARMS, DELTAS),
           "mean_fold_all": {f"{a}_minus_{b}": {
               k: float(np.mean([x[f"{a}_minus_{b}"][f"{k}_delta"] for x in allb]))
               for k in ("crisis_f1", "crisis_brier", "macro_f1_fourclass")} for a, b in DELTAS}}
    mh = [h["matched_persistence"] for h in have if h["matched_persistence"]["n"]]
    if mh:
        pd_ = tuple((a, "persistence") for a in ARMS) + DELTAS
        mp = _pool_block(mh, ARMS + ("persistence",), pd_)
        mp["mean_fold"] = {f"{a}_minus_{b}": {
            k: float(np.mean([x[f"{a}_minus_{b}"][f"{k}_delta"] for x in mh]))
            for k in ("crisis_f1", "crisis_brier", "macro_f1_fourclass")} for a, b in pd_}
        res["matched_persistence"] = mp
    res["routing"] = {arm: {"n": int(sum(h["routing"][arm]["n"] for h in have)),
                            "by_reason": {x: int(sum(h["routing"][arm]["by_reason"][x] for h in have))
                                          for x in mt.REASONS}} for arm in MAP_ARMS}
    for arm in MAP_ARMS:
        r = res["routing"][arm]
        r["root_share"] = (r["n"] - r["by_reason"]["region"]) / r["n"] if r["n"] else None
    res["regions"] = {k: {"named": int(sum(p["map_counts"][k]["named"] for p in per_pair.values() if select(p))),
                          "eligible": int(sum(p["map_counts"][k]["eligible"] for p in per_pair.values() if select(p)))}
                      for k in ("random", "temporal")}
    return res


# ---------------------------------------------------------------- per unit

class Budget:
    def __init__(self):
        self.search_roots = self.searches = self.child_fits = self.refits = 0

    def add(self, **kw):
        for k, v in kw.items():
            setattr(self, k, getattr(self, k) + int(v))
        limits = {"search_roots": MAX_SEARCH_ROOTS, "searches": MAX_SEARCHES, "child_fits": MAX_CHILD_FITS,
                  "refits": MAX_REFITS}
        for k, lim in limits.items():
            if getattr(self, k) > lim:
                raise RuntimeError(f"budget exceeded: {k} {getattr(self, k)} > {lim}")

    def record(self) -> dict:
        return {"search_roots": self.search_roots, "searches": self.searches, "child_fits": self.child_fits,
                "refits": self.refits, "limits": {"search_roots": MAX_SEARCH_ROOTS, "searches": MAX_SEARCHES,
                                                  "child_fits": MAX_CHILD_FITS, "refits": MAX_REFITS}}


def _gate(ctx, h, t):
    """D34 root identity + D37 gate_root + search-month table + pool reconstruction (no fits)."""
    g_cfg = plan.TB3_G[str(h)]
    name = plan.e1pair_root_name(h, t, g_cfg)
    if name not in ctx["roots"]:
        raise GateError(f"{name}: not in the accepted D34 schedule")
    stage = ctx["stage"]
    root = json.loads((stage / "roots" / name / "root.json").read_text(encoding="utf-8"))
    pair = {ctx["cands"][c]["e1"]: c for c in ctx["cands"] if ctx["cands"][c]["root"] == name}
    gate, data, replay, root_ubj = rr.gate_root(ctx["run"], stage, name, pair, root)
    gate["pair"] = {"horizon": h, "target_month": t}
    if not gate["passed"]:
        return gate, None
    months = check_search_months(gi.read_csv(stage / "roots" / name / "fold_membership.csv.gz"), SEARCH_MONTHS[(h, t)])
    pool = reconstruct_pool(data)
    gate["checks"]["pool_membership_order"] = {"mismatches": 0, "n": int(len(pool["y"]))}
    gate["search_months"] = months
    return gate, {"name": name, "g": g_cfg, "root": root, "pair": pair, "data": data, "replay": replay,
                  "root_ubj": root_ubj, "pool": pool, "brier": mt.brier_candidate(ctx["cands"], name)}


def run_equivalence(ctx, h, budget):
    gate, u = _gate(ctx, h, EQUIV_TARGET)
    if u is None:
        return gate
    booster = nx.from_raw(u["root_ubj"].read_bytes())
    root_rec = u["root"]["root_fit"]
    if nx.sha(booster) != u["root"]["root_booster_sha256"] or root_rec.get("booster_sha256") != nx.sha(booster):
        raise GateError(f"{u['name']}: saved root booster / root-fit record identity differs")
    Xt, yt, gt, _ = u["data"]["E3"]
    data, conf = production_inputs(u["pool"], (Xt, yt, gt), booster)
    edir = ctx["out"] / "equivalence" / u["name"]
    (edir / "candidates").mkdir(parents=True)
    budget.add(searches=1)
    rec = run_search(u["brier"], (booster, root_rec), data, edir / "candidates", edir / "checkpoints",
                     ctx["contiguity"], ctx["features"], confirmation=conf)
    budget.add(child_fits=rec["fits"]["child_fits"])
    cmp_ = compare_production(ctx["stage"] / "candidates" / u["brier"], ctx["stage"] / "checkpoints" / u["brier"],
                              edir / "candidates" / u["brier"], edir / "checkpoints" / u["brier"])
    for k, v in cmp_["checks"].items():
        gate["checks"][f"equivalence_{k}"] = {"mismatches": int(v), "n": 1}
    gate["equivalence_info"] = cmp_["info"]
    gate["passed"] = all(c["mismatches"] == 0 for c in gate["checks"].values())
    rid.write_json_atomic(edir / "equivalence.json", {**gate, "role": "three-case plumbing check, not a proof"})
    return gate


def _refit_maps(u, maps, booster, budget, pdir):
    _, y_fit, g_fit, m_fit = u["data"]["FIT"]
    counts = {}
    for k, (area_map, meta) in maps.items():          # frozen counts BEFORE any refit
        rid_fit = mt.region_ids(g_fit, area_map)
        named = meta["named_regions"]
        if k == "temporal" and len(named) > MAX_TERMINALS:
            raise RuntimeError(f"temporal map has {len(named)} > {MAX_TERMINALS} named regions")
        counts[k] = {"named": len(named), "eligible": int(sum(nx.meets(nx.support(
            y_fit[rid_fit == s], g_fit[rid_fit == s], m_fit[rid_fit == s]), plan.FIT_SUPPORT) for s in named)),
            "s-1_areas": meta["s-1_areas"], "areas": meta["areas"]}
    rid.write_json_atomic(pdir / "map_counts_before_refit.json", counts)
    planned = sum(c["eligible"] for c in counts.values())
    if budget.refits + planned > MAX_REFITS:          # stop before any refit is fitted
        raise RuntimeError(f"refit budget would be exceeded: {budget.refits} + {planned} > {MAX_REFITS}")
    frozen, records = {}, {}
    for k, (area_map, meta) in maps.items():
        models, recs = common_arm(booster, u["data"]["FIT"], area_map)
        if len(models) != counts[k]["eligible"]:
            raise RuntimeError(f"{k} map: {len(models)} refits vs {counts[k]['eligible']} eligible regions")
        budget.add(refits=len(models))
        frozen[k] = mt.save_arm(pdir / f"{k}_map", models, recs, meta, u["root"]["root_booster_sha256"])
        for region, b in models.items():
            X_chk = u["data"]["FIT"][0][mt.region_ids(g_fit, area_map) == region][:200]
            if not np.array_equal(nx.proba(b, X_chk), nx.proba(frozen[k][region], X_chk)):
                raise RuntimeError(f"{k} map region {region}: reloaded booster predicts differently")
        records[k] = recs
    return counts, frozen, records


def run_pair(ctx, h, t, budget):
    gate, u = _gate(ctx, h, t)
    if u is None:
        return gate, None
    name, pool = u["name"], u["pool"]
    p20 = mt.check_global20(ctx["d35"], name, u["root"], u["data"], gate)
    gate["passed"] = all(c["mismatches"] == 0 for c in gate["checks"].values())
    if not gate["passed"]:
        return gate, None
    pdir = ctx["out"] / "pairs" / name
    pdir.mkdir(parents=True)
    # ---- temporal map learning (search root on FIT_tb; never the forecasting root)
    split = temporal_split(pool, SEARCH_MONTHS[(h, t)])
    x_tb = np.asarray(split["X_set"], dtype=int)
    overlap = overlap_record(pool, x_tb)
    mem_path = pdir / "legal_pool_membership.csv.gz"
    with gzip.open(mem_path, "wt", encoding="utf-8", newline="") as handle:
        pool_membership_frame(pool, x_tb, u["data"]["E3"][2], t).to_csv(handle, index=False)
    membership_record = {"file": mem_path.name, "sha256": rid.file_sha256(mem_path),
                         "legal_rows": int(len(pool["y"])), "e3_rows": int(len(u["data"]["E3"][1]))}
    budget.add(search_roots=1)
    s_root, s_rec = fit_search_root(pool, x_tb, plan.G_CONFIGS[u["g"]])
    (pdir / "search_root.ubj").write_bytes(nx.raw(s_root))
    rid.write_json_atomic(pdir / "search_root.json", s_rec)
    Xt, yt, gt, mt3 = u["data"]["E3"]
    p_search = nx.proba(s_root, Xt)
    data = (pool["X"], pool["y"], pool["g"], pool["m"], x_tb, Xt, yt, gt, fourclass.argmax_codes(p_search))
    cand = temporal_candidate_name(h, t, u["g"])
    tdir = pdir / "temporal"
    (tdir / "candidates").mkdir(parents=True)
    budget.add(searches=1)
    rec = run_search(cand, (s_root, s_rec), data, tdir / "candidates", tdir / "checkpoints",
                     ctx["contiguity"], ctx["features"])
    budget.add(child_fits=rec["fits"]["child_fits"])
    maps = {"random": mt.load_map(ctx["stage"], u["brier"]), "temporal": mt.load_map(tdir, cand)}
    # ---- common forecast: CURRENT D34 root, CURRENT D34 FIT rows, both maps
    booster = nx.from_raw(u["root_ubj"].read_bytes())
    root_sha = u["root"]["root_booster_sha256"]
    counts, frozen, records = _refit_maps(u, maps, booster, budget, pdir)
    if rid.file_sha256(u["root_ubj"]) != root_sha or nx.sha(booster) != root_sha:
        raise RuntimeError("current root booster changed")
    frame = pd.DataFrame({"root": name, "part": "E3", "area": gt, "target_month": mt3, "horizon": h, "truth": yt,
                          "persistence_code": rr.persistence_codes(Xt[:, ctx["phase_col"]])})
    probs = {"root": u["replay"]["E3"], "global20": p20["E3"]}
    for arm, key in MAP_ARMS.items():
        sid, reason = mt.route(gt, maps[key][0], records[key])
        frame[f"region_{key}"], frame[f"route_{key}"] = sid, reason
        probs[arm] = mt.arm_proba(Xt, sid, reason, u["replay"]["E3"], frozen[key])
    for arm in ARMS:
        frame[f"y_{arm}"] = fourclass.argmax_codes(probs[arm])
        for k, lab in enumerate(LABELS):
            frame[f"p_{arm}_{lab}"] = probs[arm][:, k]
    with gzip.open(pdir / "rows_E3.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, float_format="%.17g")
    tp = pd.read_csv(tdir / "candidates" / cand / "target_predictions.csv", float_precision="round_trip")
    desc = pd.DataFrame({"area": gt, "target_month": mt3, "truth": yt, "y_search_root": fourclass.argmax_codes(p_search),
                         "y_search_candidate": tp["y_pred_partitioned_code"].to_numpy()})
    for k, lab in enumerate(LABELS):
        desc[f"p_search_root_{lab}"] = p_search[:, k]
    with gzip.open(pdir / "search_root_E3_descriptive.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        desc.to_csv(handle, index=False, float_format="%.17g")
    dc = rr.dev_baseline_check(frame, ctx["base"], h)
    if dc["status"] == "checked" and (dc["joined"] != dc["e3_rows"] or dc["persistence_mismatches"]
                                      or dc["truth_mismatches"]):
        gate = {**gate, "passed": False, "error": f"dev_baselines cross-check failed: {dc}"}
    meta = {"root": name, "horizon": h, "target_month": t, "g_config": u["g"], "root_booster_sha256": root_sha,
            "origin_month": month_label([pool["o_index"]])[0], "search_months": split["validation_months"],
            "temporal_candidate": cand, "search_root": {"booster_sha256": s_rec["booster_sha256"],
                                                        "fit_support": s_rec["fit_support"],
                                                        "fit_keys_sha256": s_rec["fit_keys_sha256"]},
            "search": {"child_fits": rec["fits"]["child_fits"], "n_terminal": rec["partition"]["n_terminal"],
                       "accepted_splits": rec["partition"]["accepted_splits"],
                       "assignment_evidence_sha256": rec["assignment_evidence"]["sha256"],
                       "S_tb_scores_descriptive": rec["scores"]["validation"]},
            "overlap": overlap, "legal_pool_membership": membership_record, "map_counts": counts,
            "maps": {k: maps[k][1] for k in maps},
            "regions": {k: {s: {"eligible": r["eligible"], "support": r["support"], "members": len(r["member_areas"])}
                            for s, r in records[k].items()} for k in records},
            "descriptive_search_root_E3": {"role": "descriptive only; stale search root, never a forecast arm",
                                           "search_root": rr.score_block(yt, desc["y_search_root"].to_numpy(), p_search),
                                           "search_candidate": rr.score_block(yt, desc["y_search_candidate"].to_numpy())},
            "dev_baselines_check": dc}
    rid.write_json_atomic(pdir / "pair.json", meta)
    return gate, {**meta, "scores": {"E3": score_frame(frame)}}


# ---------------------------------------------------------------- main

def _stop(out, gates, budget, identity, unit):
    rid.write_json_atomic(out / "gate.json", {
        "passed": False, "stopped_at": unit, "budget": budget.record(), "pairs": gates,
        "rule": "a failed gate / identity / replay / fit stops the study before any later unit"})
    rid.write_json_atomic(out / "identity.json", identity)
    print(f"GATE FAILED at {unit}: study stopped; see gate.json", flush=True)
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--d34-run", required=True, type=Path)
    parser.add_argument("--d35-run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    run, d35, out = args.d34_run.resolve(), args.d35_run.resolve(), args.out.resolve()
    if out.is_relative_to(run) or out.is_relative_to(d35):
        raise ValueError("--out must not be inside the read-only D34 / D35 runs")
    rid.refuse_existing(out, "D43 temporal map refit")
    producer = rid.code_identity_at(D34_REV)
    if rid.code_identity() != rid.code_identity_at("HEAD"):
        raise RuntimeError("D43 must run from committed package code (working tree differs from HEAD)")
    script_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PACKAGE, capture_output=True, text=True,
                                   check=True).stdout.strip()
    if sorted(SEARCH_MONTHS) != sorted(SCHEDULE) or len(SCHEDULE) != 21:
        raise GateError("frozen D43 schedule / search-month table mismatch")
    roots, cands, ident = accept_mode(run, plan.E1PAIR, producer, D34_REV)
    stage = Path(ident["stage"])
    d35_ident = json.loads((d35 / "identity.json").read_text(encoding="utf-8"))
    if d35_ident.get("stage") != "d35_global_increment" or d35_ident.get("producer_rev") != D34_REV \
            or not str(d35_ident.get("script_commit", "")).startswith(D35_REV):
        raise ValueError(f"--d35-run is not the D35 ({D35_REV}) global-increment run of D34 {D34_REV}")
    geometry = run / "prepared" / "geometry" / "polygon_contiguity_info.pkl"
    with open(geometry, "rb") as handle:
        contiguity = pickle.load(handle)
    features = load_schema(sr.SCHEMA)["ordered_features"]
    base_path = run / "prepared" / "ledgers" / "dev_baselines.csv"
    ctx = {"run": run, "d35": d35, "out": out, "stage": stage, "roots": roots, "cands": cands,
           "contiguity": contiguity, "features": features, "phase_col": features.index("hist_phase_o00"),
           "base": pd.read_csv(base_path)}
    out.mkdir(parents=True)
    identity = {"stage": "d43_temporal_map_refit", "d34_run": str(run), "d35_run": str(d35),
                "producer_rev": D34_REV, "d35_rev": D35_REV, "producer_code": producer,
                "script_commit": script_commit, "script_code": rid.code_identity(), "runtime": rid.runtime_identity(),
                "max_month": "2020-12", "schedule": [list(p) for p in SCHEDULE],
                "search_months": {f"h{h}_{t}": list(v) for (h, t), v in SEARCH_MONTHS.items()},
                "equivalence_target": EQUIV_TARGET, "fit_support": dict(plan.FIT_SUPPORT),
                "local_config": plan.L_CONFIGS["L1"], "g_configs": {k: plan.G_CONFIGS[v] for k, v in plan.TB3_G.items()},
                "geometry": {"path": str(geometry), "sha256": rid.file_sha256(geometry)},
                "schema_sha256": rid.file_sha256(sr.SCHEMA),
                "dev_baselines": {"path": str(base_path), "sha256": rid.file_sha256(base_path)},
                "d35_identity_sha256": rid.file_sha256(d35 / "identity.json"),
                "d35_completion_sha256": rid.file_sha256(d35 / "completion.json"),
                "acceptance": {k: (str(v) if isinstance(v, Path) else v) for k, v in ident.items()}, "inputs": {}}
    budget, gates, per_pair = Budget(), {}, {}
    unit = None
    try:
        for h in plan.HORIZONS:                       # phase 1: production equivalence, before any temporal fit
            unit = f"equivalence_h{h}_{EQUIV_TARGET}"
            try:
                gate = run_equivalence(ctx, h, budget)
            except GateError as exc:
                gate = {"passed": False, "error": str(exc)}
            gates[unit] = gate
            print(f"{unit}: {'passed' if gate['passed'] else 'FAILED'}", flush=True)
            if not gate["passed"]:
                return _stop(out, gates, budget, identity, unit)
        for h, t in SCHEDULE:                         # phase 2: the 21 pairs, sequential
            name = plan.e1pair_root_name(h, t, plan.TB3_G[str(h)])
            unit = name
            identity["inputs"][name] = {
                "snapshot": rid.file_sha256(run / "prepared" / f"snapshot_h{h}.parquet"),
                **{f: rid.file_sha256(stage / "roots" / name / f)
                   for f in ("root.json", "fold_membership.csv.gz", "root_target_predictions.csv")}}
            try:
                gate, res = run_pair(ctx, h, t, budget)
            except GateError as exc:
                gate, res = {"root": name, "passed": False, "error": str(exc)}, None
            gates[name] = gate
            print(f"{name}: {'passed' if gate['passed'] else 'FAILED'}; budget {budget.record()}", flush=True)
            if not gate["passed"]:
                return _stop(out, gates, budget, identity, name)
            per_pair[name] = res
            identity["inputs"][name]["legal_pool_membership_sha256"] = res["legal_pool_membership"]["sha256"]
    except Exception as exc:                          # engineering failure: keep the evidence, then stop
        rid.write_json_atomic(out / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc(),
                                                     "stopped_at": unit, "budget": budget.record(), "gates": gates})
        rid.write_json_atomic(out / "identity.json", identity)
        raise
    expected = {"search_roots": 21, "searches": 24}
    if any(getattr(budget, k) != v for k, v in expected.items()):
        gates["_budget"] = {"passed": False, **budget.record()}
        return _stop(out, gates, budget, identity, "_budget")
    rid.write_json_atomic(out / "gate.json", {"passed": True, "budget": budget.record(), "pairs": gates})
    rid.write_json_atomic(out / "identity.json", identity)
    summary = {
        "per_pair": per_pair,
        "by_horizon": {f"H{h}": aggregate(per_pair, lambda r, h=h: r["horizon"] == h) for h in plan.HORIZONS},
        "by_target": {t: aggregate(per_pair, lambda r, t=t: r["target_month"] == t) for t in plan.E1PAIR_TARGETS},
        "overall_21": aggregate(per_pair, lambda r: True),
        "budget": budget.record(),
        "interpretation": ("E3 only. Primary: same-key root / global20 / random_map_refit / temporal_map_refit, all "
                           "from the CURRENT D34 root and FIT rows, and versus exact-origin persistence on matched "
                           "non-missing keys. Pooled sums confusions; mean_fold averages per-pair deltas. Search "
                           "root E3 and S_tb scores are descriptive only. Search-root age, sample size, coverage "
                           "and reuse of search labels change together; region counts/support differ by pipeline. "
                           "21 correlated, repeatedly developed folds; no significance claim; not comparable with "
                           "D42's 12-pair subset."),
    }
    rid.write_json_atomic(out / "summary.json", summary)
    rid.write_json_atomic(out / "completion.json", {"status": "completed", "outputs": rid.output_hashes(out)})
    print(f"D43 temporal map refit completed: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
