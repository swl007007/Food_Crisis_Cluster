"""Bounded c8 probe manifest from the RELEASED full production inventory (no code change, no EE call).

Piece records are copied byte-for-byte from the released production pieces file. Selection is deterministic and
outcome-free: only plan pieces at plan position >= 50% (far beyond what full-run invocation 1 can reach), never S1/S2/S3
pieces; 6 per candidate-count group {1}, {2}, {3-4}, {5+}, at within-group cell quantiles .50/.65/.80/.90/.97/1.00
(nearest, ties by piece id). Manifest body mirrors write_production_manifest fields; sample=True; release_id None.
Usage: python3 build_c8_sample.py RELEASED_INPUTS_DIR SMALL_LIVE_INPUTS_DIR OUT_DIR
"""

import gzip
import hashlib
import json
import sys
from pathlib import Path

inp, small, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=False)
full = json.loads((inp / "production_manifest.json").read_text())
raw = gzip.decompress((inp / full["pieces_file"]).read_bytes()).decode().splitlines(keepends=True)
assert hashlib.sha256((inp / full["pieces_file"]).read_bytes()).hexdigest() == full["pieces_sha256"]
used = set()
for d in small.iterdir():
    m = json.loads((d / f"{d.name}_manifest.json").read_text())
    used |= {json.loads(x)["request_id"] for x in gzip.decompress((d / m["pieces_file"]).read_bytes()).decode().splitlines()}
recs = [(i, json.loads(line), line) for i, line in enumerate(raw)]
start = len(recs) // 2
pool = [t for t in recs if t[0] >= start and t[1]["request_id"] not in used]
groups = {"1": lambda k: k == 1, "2": lambda k: k == 2, "3-4": lambda k: 3 <= k <= 4, "5+": lambda k: k >= 5}
chosen = []
for name, test in groups.items():
    sub = sorted((t for t in pool if test(len(t[1]["candidates"]))), key=lambda t: (t[1]["expected_cells"], t[1]["request_id"]))
    cells = [t[1]["expected_cells"] for t in sub]
    taken = set()
    for q in (0.50, 0.65, 0.80, 0.90, 0.97, 1.00):
        target = cells[min(len(cells) - 1, round(q * (len(cells) - 1)))]
        cand = sorted((t for t in sub if t[1]["request_id"] not in taken), key=lambda t: (abs(t[1]["expected_cells"] - target), t[1]["request_id"]))
        taken.add(cand[0][1]["request_id"])
        chosen.append((name, q, cand[0]))
chosen_sorted = sorted(chosen, key=lambda c: c[2][0])  # plan order
pieces_path = out / "C8_probe_pieces.jsonl.gz"
with open(pieces_path, "wb") as rawf:
    with gzip.GzipFile(filename="", mode="wb", fileobj=rawf, mtime=0) as h:
        for _, _, t in chosen_sorted:
            h.write(t[2].encode())
reqs = [t[1] for _, _, t in chosen_sorted]
limits = {"tile_scale": 4, "deadline_ms": 120000, "sdk_retries": 0, "budget_retries": 1, "concurrency": 8, "batch_size": 1,
          "max_high_level_calls": 2 + 2 * len(reqs), "wall_budget_s": 600, "outer_timeout_s": 900,
          "max_consecutive_failures": 2, "max_failure_fraction": 0.25, "min_attempts_for_fraction": 4,
          "max_pixels": max(r["max_pixels_band_px"] for r in reqs)}
body = {k: full[k] for k in ("kind", "recipe", "grid", "grid_crs_wkt", "projection_check", "best_effort", "project", "collection", "year")}
body.update({"provenance": {**full["provenance"], "probe": "c8 bounded parallelism probe", "source_manifest_sha256": full["request_sha256"],
                            "source_pieces_sha256": full["pieces_sha256"], "selector": "build_c8_sample.py"},
             "sample": True, "pieces_file": pieces_path.name, "pieces_sha256": hashlib.sha256(pieces_path.read_bytes()).hexdigest(),
             "n_pieces": len(reqs), "expected_cells_total": sum(r["expected_cells"] for r in reqs), "limits": limits})
canon = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
manifest = {**body, "request_sha256": hashlib.sha256(canon.encode()).hexdigest(), "release_id": None}
(out / "C8_probe_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
sel = [{"group": g, "q": q, "plan_index": t[0], "piece_id": t[1]["request_id"], "candidates": len(t[1]["candidates"]),
        "expected_cells": t[1]["expected_cells"], "bbox_cells": t[1]["bbox_cells"],
        "vertices": sum(len(r) for poly in ([t[1]["geometry"]["coordinates"]] if t[1]["geometry"]["type"] == "Polygon" else t[1]["geometry"]["coordinates"]) for r in poly)}
       for g, q, t in chosen]
(out / "C8_probe_selection.json").write_text(json.dumps({"plan_position_floor": start, "excluded_small_live": len(used), "pieces": sel}, indent=1) + "\n")
print(json.dumps({"request_sha256": manifest["request_sha256"], "pieces_sha256": body["pieces_sha256"], "n": len(reqs),
                  "cells": body["expected_cells_total"], "bbox_cells": sum(r["bbox_cells"] for r in reqs), "limits": limits}, indent=1))
