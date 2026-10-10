"""Local checkpoint/completeness check of a D30 production output directory (no EE call; any Python 3.12).

Usage: python scripts/check_c_outputs.py --request RELEASED_MANIFEST --out OUT_DIR [--json REPORT]

For every manifest piece: a present output must carry its exact binding (manifest sha, piece, admin, candidates, recipe,
response digest) and pass the numerical invariants (129 finite sums, valid cells <= lattice box, owned area by source
rank = valid area within 1e-9 relative). Reports done/pending pieces and cells, ledger statuses, cumulative call
reservations and invocations against their caps, and the observed throughput with a projection of the remaining
invocation wall time. Exit 0 = no violation (pending pieces are not violations); exit 1 = violation (stop; report).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ee_pilot_c as epc  # noqa: E402
from ee_metadata import Stop  # noqa: E402


def lines(path: Path) -> list:
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def check(request: Path, out: Path) -> dict:
    req = epc.load_released(request)
    lim = req["limits"]
    violations, done, done_cells, pending_cells, seconds = [], 0, 0.0, 0.0, []
    for r in req["requests"]:
        path = epc.piece_file(out, r["request_id"])
        try:
            ok = epc.valid_output(req, r, path)
        except Stop as stop:
            violations.append(str(stop))
            continue
        if not ok:
            pending_cells += r["expected_cells"]
            continue
        resp = json.loads(path.read_text())["response"]
        if not epc.numerically_complete(resp):
            violations.append(f"{r['request_id']}: null/nonfinite output")
            continue
        area = float(resp["sums"]["valid_area"])
        rank = sum(float(g["sum"]) for g in (resp.get("area_by_rank") or {}).get("groups", []))
        if (abs(rank - area) / area if area > 0 else abs(rank)) > 1e-9:
            violations.append(f"{r['request_id']}: rank-area discrepancy")
        if resp["count"]["valid_cells"] > r["bbox_cells"]:
            violations.append(f"{r['request_id']}: valid cells above the lattice box")
        done += 1
        done_cells += r["expected_cells"]
        seconds.append(json.loads(path.read_text())["seconds"])
    statuses = Counter(json.loads(x)["status"] for x in lines(out / "ledger.jsonl"))
    runs = [json.loads(x) for x in lines(out / "run_summaries.jsonl")]
    reservations = len(lines(out / "call_reservations.log"))
    invocations = len(lines(out / "invocations.log"))
    wall = sum(float(x.get("invocation_seconds") or 0) for x in runs)
    rate = done_cells / wall if wall > 0 else None
    remaining_s = pending_cells / rate if rate else None
    max_inv = int(lim.get("max_invocations", 1))
    report = {
        "manifest_sha256": req["request_sha256"], "release_id": req["release_id"], "pieces": len(req["requests"]),
        "done": done, "pending": len(req["requests"]) - done - len([v for v in violations if "output" in v]),
        "done_cells": done_cells, "pending_cells": pending_cells, "ledger_statuses": dict(statuses),
        "call_reservations": reservations, "call_cap": lim["max_high_level_calls"],
        "invocations": invocations, "max_invocations": max_inv, "summed_invocation_seconds": round(wall, 1),
        "observed_cells_per_wall_second": rate, "projected_remaining_wall_s": remaining_s,
        "projected_within_invocation_cap": (remaining_s is None or wall + remaining_s <= max_inv * float(lim["wall_budget_s"])),
        "piece_seconds_max": max(seconds) if seconds else None, "last_stop": runs[-1].get("stopped") if runs else None,
        "complete": done == len(req["requests"]) and not violations, "violations": violations[:50], "n_violations": len(violations),
    }
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--request", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--json")
    a = p.parse_args(argv)
    report = check(Path(a.request), Path(a.out))
    text = json.dumps(report, indent=1, default=str)
    if a.json:
        Path(a.json).write_text(text + "\n")
    print(text)
    return 1 if report["n_violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
