"""Saved-model replay and independent recomputation (design section 10).

1. Re-execute develop, Stage3 and report with a read-only model store: every
   model is loaded (identity and canonical tensor digest checked, never
   refitted), every transform is rebuilt from its referenced fitting rows (its
   digest is part of each model identity), every residual target is recomputed
   from the loaded frozen B (its digest is part of each P/L identity), and every
   saved prediction/selection/gate/report artifact must be reproduced
   byte-for-byte (store counters excluded).
2. Independently recompute from the saved keyed artifacts: projection and
   phases from saved components, G routing, gate decisions from saved pairs,
   recipe selection from saved S predictions, and the fit inventory.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_mlp import develop, metrics, projection, report, stage3
from ipcch_mlp.artifacts import sha256_file, write_json
from ipcch_mlp.quartets import TARGETS, Engine
from ipcch_mlp.store import ModelStore

VOLATILE = {"store_counts", "new_scalar_fits", "stage3_summary_sha256"}


class Checker:
    def __init__(self):
        self.passed, self.failures = 0, []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        if ok:
            self.passed += 1
        else:
            self.failures.append(f"{name}: {detail}")


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in VOLATILE}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def compare_trees(c: Checker, original: Path, replayed: Path) -> None:
    files = sorted(p.relative_to(original) for p in original.rglob("*") if p.is_file())
    again = sorted(p.relative_to(replayed) for p in replayed.rglob("*") if p.is_file())
    c.check(f"inventory:{original.name}", files == again, f"{set(files) ^ set(again)}")
    for rel in files:
        a, b = original / rel, replayed / rel
        if not b.is_file():
            continue
        if rel.suffix == ".json":
            ja = json.loads(a.read_text(encoding="utf-8"))
            jb = json.loads(b.read_text(encoding="utf-8"))
            c.check(f"json:{original.name}/{rel}", _strip(ja) == _strip(jb), "content differs")
        else:
            c.check(f"bytes:{original.name}/{rel}", sha256_file(a) == sha256_file(b), "bytes differ")


def independent_predictions(c: Checker, path: Path) -> None:
    pred = report.read_predictions(path)
    if len(pred) == 0:
        return
    base = pred[[f"base_{q}_raw" for q in TARGETS]].to_numpy()
    pres = pred[[f"pres_{q}" for q in TARGETS]].to_numpy()
    lres = pred[[f"lres_{q}" for q in TARGETS]].to_numpy()
    pool = base + pres
    for arm, raw in (("base", base), ("pool", pool)):
        star, phase = projection.project_and_decode(raw)
        c.check(f"{path}:{arm}_star", np.array_equal(star, pred[[f"{arm}_{q}_star" for q in TARGETS]].to_numpy()))
        c.check(f"{path}:{arm}_phase", np.array_equal(phase, pred[f"{arm}_phase"].to_numpy()))
    has_l = pred["local_eligible"].to_numpy() == 1
    c.check(f"{path}:local_eligible", np.array_equal(has_l, ~np.isnan(lres).any(axis=1)))
    if has_l.any():
        lstar, lphase = projection.project_and_decode(base[has_l] + lres[has_l])
        c.check(f"{path}:local_phase", np.array_equal(lphase, pred["local_phase"].to_numpy()[has_l]))
        c.check(f"{path}:local_star", np.array_equal(lstar, pred[[f"local_{q}_star" for q in TARGETS]].to_numpy()[has_l]))
    use_l = (pred["route"] == "local").to_numpy()
    c.check(f"{path}:adopted_have_local", bool(has_l[use_l].all()))
    expect = np.where(use_l, pred["local_phase"].to_numpy(), pred["pool_phase"].to_numpy())
    c.check(f"{path}:geo_routing", np.array_equal(expect, pred["geo_phase"].to_numpy()))
    geo_raw = pred[[f"geo_{q}_raw" for q in TARGETS]].to_numpy()
    c.check(f"{path}:geo_pool_bitwise", np.array_equal(geo_raw[~use_l], pool[~use_l]))
    c.check(f"{path}:unmapped_pool", bool((pred.loc[pred["region"] == "", "route"] == "unmapped_area_pool").all()))


def independent_gates(c: Checker, hdir: Path, contract: dict) -> None:
    gates = [json.loads(x) for x in (hdir / "gate_decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    by_fold: dict = {}
    for g in gates:
        by_fold.setdefault(g["fold_id"], []).append(g)
    for fold_id, decisions in by_fold.items():
        path = hdir / f"pairs_{fold_id}.csv.gz"
        pairs = pd.read_csv(path, float_precision="round_trip", dtype={"region": str, "lres_provider": str}) \
            if path.is_file() else pd.DataFrame(columns=["region", "admin_code", "validation_month", "phase_truth",
                                                          "phase_pool", "phase_local_routed", "local_fit_ok"])
        if len(pairs):
            base = pairs[[f"base_{q}" for q in TARGETS]].to_numpy()
            pres = pairs[[f"pres_{q}" for q in TARGETS]].to_numpy()
            lres = pairs[[f"lres_{q}" for q in TARGETS]].to_numpy()
            ok = pairs["local_fit_ok"].to_numpy().astype(bool)
            _, pphase = projection.project_and_decode(base + pres)
            routed = np.where(ok[:, None], base + np.nan_to_num(lres), base + pres)
            _, rphase = projection.project_and_decode(routed)
            c.check(f"{path}:phase_pool", np.array_equal(pphase, pairs["phase_pool"].to_numpy()))
            c.check(f"{path}:phase_local_routed", np.array_equal(rphase, pairs["phase_local_routed"].to_numpy()))
            c.check(f"{path}:fallback_has_no_local", bool(np.isnan(lres[~ok]).all()))
        for d in decisions:
            again = stage3.gate_decision(pairs[pairs["region"] == d["region"]] if len(pairs) else pairs, contract)
            for k in ("keys", "areas", "target_months", "crisis_keys", "noncrisis_keys", "local_fit_dates",
                      "historical_support", "enabled", "reason"):
                c.check(f"{hdir.name}/{fold_id}/{d['region']}:{k}", again[k] == d[k], f"{again[k]} != {d[k]}")


def independent_selection(c: Checker, run_dir: Path, contract: dict) -> None:
    for hdir in sorted((run_dir / "develop").glob("h*")):
        sel = json.loads((hdir / "selection.json").read_text(encoding="utf-8"))
        entries = {cid: {} for cid in develop.CANDIDATES}
        for path in sorted(hdir.glob("S_rep*_*.csv.gz")):
            rep, cid = path.name[len("S_rep"):-len(".csv.gz")].split("_")
            frame = pd.read_csv(path, float_precision="round_trip")
            raw = frame[[f"base_{q}" for q in TARGETS]].to_numpy() + frame[[f"pres_{q}" for q in TARGETS]].to_numpy()
            _, phase = projection.project_and_decode(raw)
            c.check(f"{path.name}:phase", np.array_equal(phase, frame["pool_phase"].to_numpy()))
            f1 = metrics.exact_f1(metrics.crisis_counts(frame["phase_truth"], phase))
            entries[cid][int(rep)] = None if f1 is None else str(f1)
        again = develop.select({k: {r: v for r, v in sorted(e.items())} for k, e in entries.items()}, contract)
        c.check(f"{hdir.name}:winner", again["winner"] == sel["selection"]["winner"],
                f"{again['winner']} != {sel['selection']['winner']}")


def inventory(c: Checker, run_dir: Path, contract: dict) -> dict:
    fits: dict = {}
    for line in (run_dir / "model_requests.jsonl").read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e["status"] == "failed":
            c.check("ledger:no_failed", False, line)
        if e["status"] == "fit":
            key = ("develop" if e.get("stage") == "develop" else f"stage3_rep{e.get('replicate')}")
            fits.setdefault(key, set()).add(e["identity_sha256"])
    counts = {k: len(v) for k, v in sorted(fits.items())}
    exp = contract["expected"]
    c.check("inventory:develop", counts.get("develop", 0) == exp["development_scalar_fits"], str(counts))
    for rep in contract["replicates"]:
        c.check(f"inventory:stage3_rep{rep}", counts.get(f"stage3_rep{rep}", 0) == exp["stage3_scalar_fits_per_seed"],
                str(counts))
    stored = {p.name for p in (run_dir / "models" / "models").glob("*/*") if p.is_dir() and not p.name.startswith(".tmp")}
    referenced = set().union(*fits.values()) if fits else set()
    c.check("inventory:store_equals_ledger", stored == referenced, f"{len(stored)} stored vs {len(referenced)} fitted")
    c.check("inventory:total", sum(counts.values()) == exp["total_scalar_fits"], str(sum(counts.values())))
    return counts


def run_replay(run_dir: Path, contract: dict, env: dict, device: str, horizons: dict, split, calendar,
               expect_inventory: bool = True) -> dict:
    out = run_dir / "replay"
    out.mkdir()
    c = Checker()
    store = ModelStore(run_dir / "models", out / "model_requests.jsonl", readonly=True)
    engine = Engine(store, contract, env, device)
    develop.run_develop(out, engine, contract, horizons, split)
    compare_trees(c, run_dir / "develop", out / "develop")
    winners = develop.load_winners(out)
    stage3.run_stage3(out, engine, contract, horizons, calendar, winners)
    compare_trees(c, run_dir / "stage3", out / "stage3")
    report.run_report(out, contract)
    compare_trees(c, run_dir / "report", out / "report")
    for path in sorted((run_dir / "stage3").glob("rep*/h*/predictions.csv.gz")):
        independent_predictions(c, path)
        independent_gates(c, path.parent, contract)
    independent_selection(c, run_dir, contract)
    counts = inventory(c, run_dir, contract) if expect_inventory else {}
    result = {"status": "passed" if not c.failures else "failed", "checks_passed": c.passed,
              "failures": c.failures[:200], "n_failures": len(c.failures), "fit_inventory": counts,
              "replay_store_counts": dict(store.counts)}
    write_json(out / "replay.json", result)
    return result
