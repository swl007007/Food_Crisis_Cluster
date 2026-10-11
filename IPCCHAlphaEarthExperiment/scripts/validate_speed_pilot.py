"""Validate a finished D30 batch speed pilot (LOCAL ONLY; ipcch-geo interpreter; no EE/GCS call).

Integrity: every downloaded object (several per task possible) is parsed with ``parse_export_csv``; per task the rows
must equal the manifest's item ids exactly once, carry the D30 recipe and 129 finite values (a blank value is a
technical failure, never low coverage). Gates fixed in advance (SPEED-PILOT-PROPOSAL section 4):
  references (interactive whole-admin responses, pinned): |relative A,S,Q| <= 1e-9, descriptors <= 1e-9, coverage <= 1e-9;
  reuse validation (sum of pinned old windows): A <= 1e-7, S/Q <= 1e-6, descriptors <= 1e-6, coverage <= 1e-7, same route.
Timing (task status timestamps + local ledger/download records; fresh cells only in every rate):
  (a) paired RUNNING-wall rate, (b) submit-to-final-validation rate, (c) slower task's individual RUNNING rate, plus the
  observed overlap of the two RUNNING intervals, and remaining-work projections for each. No startup term is added.
Usage: PYTHONPATH=.:scripts python scripts/validate_speed_pilot.py --request M --out RUN_OUT --full-run DIR
       --interactive-run DIR --plan ADAPTIVE_PLAN --remaining-cells 3.9887e11
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_alphaearth import aemoments
from ipcch_alphaearth.contract import load_contract
from ipcch_alphaearth.errors import TechnicalError
from ipcch_alphaearth.extraction import check_piece_binding, parse_export_csv

B = aemoments.BANDS
VALS = ["valid_area", *[f"S_{b}" for b in B], *[f"Q_{b}" for b in B]]
GATES = {"reference": {"A": 1e-9, "SQ": 1e-9, "desc": 1e-9, "coverage": 1e-9},
         "reuse_validation": {"A": 1e-7, "SQ": 1e-6, "desc": 1e-6, "coverage": 1e-7}}


def read_rows(req: dict, downloads: list) -> pd.DataFrame:
    """All downloaded objects -> one frame; per task exactly the manifest's items, D30 recipe, all values finite."""
    frames = []
    for d in downloads:
        frames.append(parse_export_csv(Path(d["local"])).assign(task=d["task"]))
    got = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["piece_id", "task"])
    for t in req["tasks"]:
        want = sorted(it["piece_id"] for g in t["groups"] for it in g["items"])
        have = got.loc[got["task"] == t["name"], "piece_id"]
        if have.duplicated().any() or sorted(have) != want:
            raise TechnicalError(f"task {t['name']}: rows differ from its items ({len(have)} rows, {len(want)} items, "
                                 f"{int(have.duplicated().sum())} duplicated)")
    if set(got["recipe"]) != {req["recipe"]}:
        raise TechnicalError("rows not produced under the D30 recipe")
    vals = got[VALS].apply(pd.to_numeric, errors="coerce")
    bad = got.loc[~np.isfinite(vals.to_numpy()).all(axis=1), "piece_id"].tolist()
    if bad:
        raise TechnicalError(f"{len(bad)} rows with null/nonfinite values (technical failure, never coverage): {bad[:5]}")
    return got


def moments(rows: list) -> dict:
    return {"A": sum(r[0] for r in rows), "S": np.sum([r[1] for r in rows], axis=0), "Q": np.sum([r[2] for r in rows], axis=0)}


def descriptor(m: dict, area: float, contract: dict) -> dict:
    ae = contract["alphaearth"]
    merged = aemoments.merge_pieces([{"piece_id": "x", "status": "complete", "valid_area": m["A"], "S": list(m["S"]), "Q": list(m["Q"])}], ["x"])
    return aemoments.descriptor(merged, area, ae["coverage_threshold"], float(ae["variance_cancellation_tolerance"]))


def compare(new: dict, ref: dict, area: float, contract: dict, gate: dict) -> dict:
    rel = lambda x, y: float(np.max(np.abs(np.asarray(x) - np.asarray(y)) / np.maximum(np.abs(np.asarray(y)), 1e-300)))
    dn, dr = descriptor(new, area, contract), descriptor(ref, area, contract)
    out = {"A_rel": rel(new["A"], ref["A"]), "S_rel": rel(new["S"], ref["S"]), "Q_rel": rel(new["Q"], ref["Q"]),
           "mean_abs": float(np.max(np.abs(np.array(dn["mean"]) - np.array(dr["mean"])))),
           "sd_abs": float(np.max(np.abs(np.array(dn["sd"]) - np.array(dr["sd"])))),
           "coverage_abs": abs(dn["coverage"] - dr["coverage"]), "route_new": dn["route_reason"], "route_ref": dr["route_reason"]}
    out["pass"] = (out["A_rel"] <= gate["A"] and max(out["S_rel"], out["Q_rel"]) <= gate["SQ"]
                   and max(out["mean_abs"], out["sd_abs"]) <= gate["desc"] and out["coverage_abs"] <= gate["coverage"]
                   and out["route_new"] == out["route_ref"])
    return out


def row_moments(got: pd.DataFrame, admin: int) -> dict:
    sub = got[got["admin_code"].astype(int) == admin]
    return moments([(float(r["valid_area"]), np.array([float(r[f"S_{b}"]) for b in B]), np.array([float(r[f"Q_{b}"]) for b in B]))
                    for _, r in sub.iterrows()])


def json_moments(paths: list) -> dict:
    rows = []
    for p in paths:
        s = json.loads(Path(p).read_text())["response"]["sums"]
        rows.append((float(s["valid_area"]), np.array([s[f"S_{b}"] for b in B]), np.array([s[f"Q_{b}"] for b in B])))
    return moments(rows)


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timing(req: dict, out: Path, got: pd.DataFrame, validated_epoch: float, remaining: float) -> dict:
    ledger = [json.loads(x) for x in (out / "ledger.jsonl").read_text().splitlines()]
    sub = {r["task"]: r for r in ledger if r["state"] == "SUBMITTED"}
    last = {}
    for line in (out / "status.jsonl").read_text().splitlines():
        for s in json.loads(line)["statuses"]:
            last[s["id"]] = s
    fresh = {t["name"]: sum(it["expected_cells"] for g in t["groups"] for it in g["items"] if it["role"] == "fresh") for t in req["tasks"]}
    per, runs = {}, []
    for t in req["tasks"]:
        s = last[sub[t["name"]]["task_id"]]
        start, end = s.get("start_timestamp_ms", 0) / 1000.0, s.get("update_timestamp_ms", 0) / 1000.0
        per[t["name"]] = {"state": s["state"], "queue_s": start - sub[t["name"]]["submitted_epoch_s"], "running_s": end - start,
                          "fresh_cells": fresh[t["name"]], "rate": fresh[t["name"]] / (end - start) if end > start else None,
                          "batch_eecu_usage_seconds": s.get("batch_eecu_usage_seconds")}
        runs.append((start, end))
    overlap = max(0.0, min(e for _, e in runs) - max(s for s, _ in runs))
    paired_wall = max(e for _, e in runs) - min(s for s, _ in runs)
    total_fresh = sum(fresh.values())
    first_submit = min(r["submitted_epoch_s"] for r in sub.values())
    a = total_fresh / paired_wall
    b = total_fresh / (validated_epoch - first_submit)
    c = min(v["rate"] for v in per.values() if v["rate"])
    concurrency = (sum(v["running_s"] for v in per.values()) / paired_wall) if paired_wall > 0 else None
    proj = lambda r: remaining / r / 3600.0
    return {"per_task": per, "overlap_s": overlap, "paired_running_wall_s": paired_wall, "observed_concurrency": concurrency,
            "rate_a_paired_running": a, "rate_b_submit_to_validated": b, "rate_c_slower_individual": c,
            "projection_hours": {"a": proj(a), "b": proj(b), "c_times_observed_concurrency": proj(c * concurrency) if concurrency else None},
            "target": {"cells_per_s_24h": remaining / 86400, "cells_per_s_22h_compute": remaining / 79200}}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    for name in ("--request", "--out", "--full-run", "--interactive-run", "--plan"):
        p.add_argument(name, required=True)
    p.add_argument("--remaining-cells", type=float, required=True)
    a = p.parse_args(argv)
    req = json.loads(Path(a.request).read_text())
    out, contract = Path(a.out), load_contract()
    downloads = json.loads((out / "downloads.json").read_text())["objects"]
    for d in downloads:
        if sha(Path(d["local"])) != d["sha256"]:
            raise TechnicalError(f"downloaded object {d['local']} differs from its recorded sha256")
    got = read_rows(req, downloads)
    plan = pd.read_csv(a.plan, dtype={"piece_id": str, "asset_id": str, "recipe": str})
    check_piece_binding(got, plan[plan["piece_id"].isin(set(got["piece_id"]))])
    validated = time.time()
    area = plan.groupby("admin_code")["polygon_area_m2"].first()
    gates = {}
    for admin, pin in req["reference_pins"]["interactive_responses"].items():
        f = Path(a.interactive_run) / "out" / "responses" / pin["file"]
        if sha(f) != pin["sha256"]:
            raise TechnicalError(f"reference input {f} differs from its pin")
        gates[admin] = {"role": "reference", **compare(row_moments(got, int(admin)), json_moments([f]), float(area[int(admin)]), contract, GATES["reference"])}
    for admin, wins in req["reference_pins"]["old_windows"].items():
        files = [Path(a.full_run) / "out" / "pieces" / (hashlib.sha1(w["piece_id"].encode()).hexdigest() + ".json") for w in wins]
        for f, w in zip(files, wins):
            if sha(f) != w["file_sha256"]:
                raise TechnicalError(f"old window {w['piece_id']} differs from its pin")
        gates[admin] = {"role": "reuse_validation", **compare(row_moments(got, int(admin)), json_moments(files), float(area[int(admin)]), contract, GATES["reuse_validation"])}
    report = {"integrity": "PASS", "rows": int(len(got)), "gates": gates, "all_gates_pass": all(g["pass"] for g in gates.values()),
              "timing": timing(req, out, got, validated, a.remaining_cells)}
    (out / "validation.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(json.dumps(report, indent=1, default=float)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
