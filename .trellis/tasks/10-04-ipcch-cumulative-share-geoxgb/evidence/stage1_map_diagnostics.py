"""P6 checkpoint: frozen-map diagnostics from saved Stage1 artifacts (read-only).

Run from IPCCHGeoXGBExperiment/:  python <this file> runs/<run-id>
"""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import pandas as pd


def f(x):
    return None if x is None else round(float(Fraction(x)), 6)


def main(run: Path) -> dict:
    s1 = run / "stage1"
    summary = json.loads((s1 / "stage1-summary.json").read_text())
    out = {"run": str(run), "base_identity": summary["base_identity"],
           "model_store": summary["model_store"], "elapsed_seconds": summary["elapsed_seconds"],
           "summary_sha256": hashlib.sha256((s1 / "stage1-summary.json").read_bytes()).hexdigest(),
           "request_ledger": {}, "horizons": {}}
    led = collections.Counter()
    for line in (s1 / "model_requests.jsonl").read_text().splitlines():
        r = json.loads(line)
        led[f"H{r['H']}|{r['purpose']}|{r['status']}"] += 1
    out["request_ledger"] = dict(sorted(led.items()))
    for h, entry in summary["horizons"].items():
        fr = entry["frozen"]
        hh = f"h{int(h):02d}"
        sel = json.loads((s1 / hh / "selection.json").read_text())
        fmap = pd.read_csv(s1 / f"frozen_map_{hh}.csv", dtype={"node_id": str})
        assert hashlib.sha256((s1 / f"frozen_map_{hh}.csv").read_bytes()).hexdigest() == fr["map_sha256"]
        dec = json.loads((s1 / hh / fr["candidate"] / "decisions.json").read_text())["decisions"]
        nodes = []
        for d in dec:
            g = d.get("gate", {})
            nodes.append({"node": d["node_id"], "depth": d["depth"], "areas": d["areas"],
                          "outcome": d["outcome"], "eligible": d.get("eligible"),
                          "selected": d.get("selected_children"),
                          "base_f1": f(g.get("base_f1")), "best_f1": f(g.get("best_f1")),
                          "sizes_after_smoothing": d.get("sizes_after_smoothing"),
                          "smoothing_switched": d.get("smoothing_switched")})
        regions = {}
        for n, c in fr["connectivity"].items():
            regions[n] = {k: c[k] for k in ("areas", "components", "largest_component", "isolated_areas")}
            regions[n]["map_rows"] = int((fmap["node_id"] == n).sum())
        out["horizons"][h] = {
            "winner": fr["candidate"], "G": fr["G"], "L": fr["L"],
            "f1_exact": fr["f1_exact"], "f1": f(fr["f1_exact"]),
            "accepted_split": fr["accepted_split"], "terminal_regions": fr["terminal_regions"],
            "learned_areas": fr["learned_areas"], "map_rows": len(fmap),
            "map_sha256": fr["map_sha256"], "selection_sha256": fr["selection_sha256"],
            "selection_status": sel["selection"]["status"],
            "ranking": sel["selection"]["ranking"], "undefined": sel["selection"]["undefined"],
            "candidates": [{"c": c["candidate"], "f1": f(c["f1_exact"]), "f1_exact": c["f1_exact"],
                            "terminal_regions": c["terminal_regions"],
                            "accepted_splits": c["accepted_splits"], "scans": c["scans"]}
                           for c in sel["candidates"]],
            "regions": regions,
            "nodes": nodes,
        }
    return out


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])), indent=1))
