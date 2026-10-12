"""Run the local D30 reducer on pinned references and compare with immutable EE responses (ipcch-geo; no EE call).
Usage: PYTHONPATH=.:scripts python scripts/aef_parity.py --refs REFS_JSON --cogs DIR --out OUT_JSON [--select VARIANT]"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import rasterio
from shapely.geometry import shape

import aef_local_d30 as L
from ipcch_alphaearth import aemoments
from ipcch_alphaearth.contract import load_contract
from ipcch_alphaearth.extraction import Lattice

GATE = {"A": 1e-7, "SQ": 1e-6, "desc": 1e-6, "coverage": 1e-7}
B = aemoments.BANDS


def descriptor(A, S, Q, area, ae):
    m = aemoments.merge_pieces([{"piece_id": "x", "status": "complete", "valid_area": A, "S": list(S), "Q": list(Q)}], ["x"])
    return aemoments.descriptor(m, area, ae["coverage_threshold"], float(ae["variance_cancellation_tolerance"]))


def nan_diff(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = np.where(np.isnan(x) & np.isnan(y), 0.0, np.abs(x - y))
    return float(np.max(np.where(np.isnan(d), np.inf, d)))


def compare(loc, ref, area, ae):
    rel = lambda x, y: float(np.max(np.abs(np.asarray(x) - np.asarray(y)) / np.maximum(np.abs(np.asarray(y)), 1e-300)))
    dl, dr = descriptor(loc["A"], loc["S"], loc["Q"], area, ae), descriptor(ref["A"], ref["S"], ref["Q"], area, ae)
    out = {"A_local": loc["A"], "A_ref": ref["A"], "A_rel": rel(loc["A"], ref["A"]), "S_rel": rel(loc["S"], ref["S"]),
           "S_abs": float(np.max(np.abs(np.array(loc["S"]) - ref["S"]))), "Q_rel": rel(loc["Q"], ref["Q"]),
           "mean_abs": nan_diff(dl["mean"], dr["mean"]), "sd_abs": nan_diff(dl["sd"], dr["sd"]),
           "coverage_local": dl["coverage"], "coverage_ref": dr["coverage"], "coverage_abs": abs(dl["coverage"] - dr["coverage"]),
           "route_local": dl["route_reason"], "route_ref": dr["route_reason"]}
    out["pass"] = (out["A_rel"] <= GATE["A"] and max(out["S_rel"], out["Q_rel"]) <= GATE["SQ"] and max(out["mean_abs"], out["sd_abs"]) <= GATE["desc"]
                   and out["coverage_abs"] <= GATE["coverage"] and out["route_local"] == out["route_ref"])
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--refs", required=True)
    p.add_argument("--cogs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--select")
    a = p.parse_args(argv)
    contract = load_contract()
    ae = contract["alphaearth"]
    lattice = Lattice.from_contract(contract)
    refs = json.loads(Path(a.refs).read_text())
    results = []
    for r in refs["references"]:
        sources = [(asset, [rasterio.open(Path(a.cogs) / Path(c).name) for c in r["cogs"][asset]]) for asset in sorted(r["cogs"])]
        t = time.monotonic()
        loc = L.reduce_geometry(shape(r["geometry"]), lattice, sources)
        wall = time.monotonic() - t
        for _, dss in sources:
            for ds in dss:
                ds.close()
        s = r["ref_sums"]
        ref = {"A": float(s["valid_area"]), "S": np.array([s[f"S_{b}"] for b in B]), "Q": np.array([s[f"Q_{b}"] for b in B])}
        cmp = {v: compare(loc["variants"][v], ref, float(r["polygon_area_m2"]), ae) for v in L.VARIANTS}
        results.append({"id": r["id"], "role": r["role"], "cells_bbox": loc["cells_bbox"], "valid_cells_centre": loc["valid_cells_centre"],
                        "ref_valid_cells": r.get("ref_valid_cells"), "seconds": loc["seconds"], "wall_s": round(wall, 3), "max_rss_mb": loc["max_rss_mb"],
                        "compare": cmp})
        print(json.dumps({"id": r["id"], "A_rel": {v: f"{c['A_rel']:.3e}" for v, c in cmp.items()}, "count": [loc["valid_cells_centre"], r.get("ref_valid_cells")], "wall_s": round(wall, 1)}), flush=True)
    sel = a.select or min(L.VARIANTS, key=lambda v: next(x for x in results if x["role"] == "selection")["compare"][v]["A_rel"]) if any(x["role"] == "selection" for x in results) else a.select
    verdict = {"selected_variant": sel, "all_pass": bool(sel) and all(x["compare"][sel]["pass"] for x in results)}
    Path(a.out).write_text(json.dumps({"verdict": verdict, "results": results}, indent=1, default=float) + "\n")
    print(json.dumps(verdict))
    return 0


if __name__ == "__main__":
    sys.exit(main())
