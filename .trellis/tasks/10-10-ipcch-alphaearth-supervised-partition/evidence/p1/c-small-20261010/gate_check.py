"""Gate check for one released small-live C stage (local; reads runner outputs only, no EE call).

Usage: python gate_check.py RUN_ROOT STAGE [MAX_WALL_S]
Checks: complete; zero failed/halted ledger entries; projection check <= 1e-3 m; every piece < 90 s; owned-area-by-rank
sum equals valid_area (relative <= 1e-9); valid cells <= lattice box; all 129 sums finite (no nulls); sub-cell pieces have
0 or 1 valid cell; optional invocation wall <= MAX_WALL_S. Prints a JSON summary with per-piece rows.
"""

import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

root, stage = Path(sys.argv[1]), sys.argv[2]
max_wall = float(sys.argv[3]) if len(sys.argv) > 3 else None
out = root / stage
manifest = json.loads((root / "inputs" / stage / f"{stage}_manifest.json").read_text())
with gzip.open(root / "inputs" / stage / manifest["pieces_file"], "rt") as h:
    pieces = [json.loads(x) for x in h if x.strip()]
runs = [json.loads(x) for x in (out / "run_summaries.jsonl").read_text().splitlines()]
ledger = [json.loads(x) for x in (out / "ledger.jsonl").read_text().splitlines()]
last = runs[-1]
fails = []
rows = []
for r in pieces:
    path = out / "pieces" / (hashlib.sha1(r["request_id"].encode()).hexdigest() + ".json")
    rec = json.loads(path.read_text())
    resp = rec["response"]
    sums, cells = resp["sums"], resp["count"].get("valid_cells")
    vals = [sums.get("valid_area")] + [sums.get(f"{p}_A{j:02d}") for p in "SQ" for j in range(64)]
    finite = all(v is not None and math.isfinite(float(v)) for v in vals) and len(vals) == 129
    groups = (resp.get("area_by_rank") or {}).get("groups", [])
    rank_sum = sum(float(g["sum"]) for g in groups)
    area = float(sums.get("valid_area") or 0.0)
    rank_rel = abs(rank_sum - area) / area if area > 0 else abs(rank_sum)
    row = {"piece_id": r["request_id"], "seconds": rec["seconds"], "batch_size": rec["batch_size"], "candidates": len(r["candidates"]),
           "expected_cells": round(r["expected_cells"], 2), "bbox_cells": r["bbox_cells"], "valid_cells": cells,
           "valid_area_m2": area, "valid_frac_of_window": (area / (r["expected_cells"] * 100.0)) if r["expected_cells"] > 0 else None,
           "rank_area": {str(g["source_rank"]): float(g["sum"]) for g in groups}, "rank_rel": rank_rel, "finite_129": finite}
    rows.append(row)
    if not finite:
        fails.append(f"{r['request_id']}: null/nonfinite sums")
    if rec["seconds"] >= 90:
        fails.append(f"{r['request_id']}: {rec['seconds']} s >= 90")
    if rank_rel > 1e-9:
        fails.append(f"{r['request_id']}: rank-area relative discrepancy {rank_rel}")
    if cells is None or cells > r["bbox_cells"]:
        fails.append(f"{r['request_id']}: valid cells {cells} vs bbox {r['bbox_cells']}")
    if r["expected_cells"] < 1 and cells not in (0, 1):
        fails.append(f"{r['request_id']}: sub-cell piece has {cells} valid cells")
bad_ledger = [x for x in ledger if x["status"] != "ok"]
if bad_ledger:
    fails.append(f"{len(bad_ledger)} failed/halted ledger entries")
if not last.get("complete"):
    fails.append("stage not complete")
if last.get("projection_check_max_abs_diff_m", 1) > 1e-3:
    fails.append(f"projection diff {last.get('projection_check_max_abs_diff_m')}")
if max_wall is not None and last.get("invocation_seconds", 1e9) > max_wall:
    fails.append(f"invocation wall {last.get('invocation_seconds')} > {max_wall}")
summary = {"stage": stage, "release_id": manifest["release_id"], "request_sha256": manifest["request_sha256"],
           "calls_used": last.get("calls_used"), "invocation_seconds": last.get("invocation_seconds"),
           "projection_diff_m": last.get("projection_check_max_abs_diff_m"), "stopped": last.get("stopped"),
           "complete": last.get("complete"), "ledger_entries": len(ledger), "gate_pass": not fails, "fails": fails, "pieces": rows}
print(json.dumps(summary, indent=1))
