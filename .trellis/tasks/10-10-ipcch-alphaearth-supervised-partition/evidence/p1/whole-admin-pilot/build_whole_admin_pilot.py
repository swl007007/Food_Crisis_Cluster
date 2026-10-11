"""Whole-admin D30 pilot manifest (OFFLINE; ipcch-geo; no EE call; release_id null).

Three admins chosen from geometry/plan only (no outcomes): REF = an admin already fully extracted by the 10 m window
run (2-4 windows, cells nearest the all-admin 25th percentile; its whole-admin sums are compared with the sum of its
window outputs), MED = the not-yet-extracted admin nearest the median cells, LARGE = the not-yet-extracted admin nearest
the 75th percentile. Each request: the full frozen admin geometry (no windowing), the complete ascending candidate
union (= every footprint intersecting the admin, from the frozen plan), aggregations "sums" (one reduceRegion:
valid_area, S, Q), maxPixels = 129 x whole-admin lattice box x 1.1. Recipe/grid/WKT/projection check unchanged.
Usage: PYTHONPATH=<pkg>:<pkg>/scripts python build_whole_admin_pilot.py RUN_DIR GEOMETRY_SHP OUT_JSON
"""
import hashlib, json, math, os, sys
from pathlib import Path

import pandas as pd
from shapely.geometry import mapping

import ee_pilot_c as epc
from ipcch_alphaearth import extraction as x
from ipcch_alphaearth.contract import load_contract

run, shp, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
contract = load_contract()
lat = x.Lattice.from_contract(contract)
full = json.loads((run / "inputs" / "production_manifest.json").read_text())
plan = pd.read_csv(run / "inputs" / "production_plan.csv.gz", dtype={"piece_id": str, "asset_id": str})
real = plan[plan.piece_id.notna()].copy()
real["cells"] = real.window_area_m2 / 100
real["done"] = [os.path.exists(run / "out" / "pieces" / (hashlib.sha1(p.encode()).hexdigest() + ".json")) for p in real.piece_id]
adm = real.groupby("admin_code").agg(cells=("cells", "sum"), pieces=("piece_id", "count"), done=("done", "sum"))
adm["assets"] = real.groupby("admin_code").asset_id.apply(lambda s: sorted(set(",".join(s).split(","))))
q = adm.cells.quantile


def nearest(mask, target):
    sub = adm[mask].sort_index()
    return int((sub.cells - target).abs().sort_values(kind="mergesort").index[0])


sel = [("REF_completed_small", nearest((adm.done == adm.pieces) & adm.pieces.between(2, 4), q(0.25))),
       ("MED_median_fresh", nearest(adm.done == 0, q(0.50))),
       ("LARGE_q75_fresh", nearest(adm.done == 0, q(0.75)))]
geo = x.load_geometry(shp).set_index("admin_code")
reqs, info = [], []
for role, a in sel:
    g = geo.loc[a, "geometry"]
    minx, miny, maxx, maxy = g.bounds
    bbox = (lat.col(maxx, True) - lat.col(minx, False)) * (lat.row(miny, True) - lat.row(maxy, False))
    r = {"request_id": f"WA-{role}-{a}", "admin_code": a, "candidates": adm.assets[a], "geometry": mapping(g),
         "aggregations": "sums", "max_pixels_band_px": int(math.ceil(129 * bbox * 1.1))}
    reqs.append(r)
    info.append({"role": role, "admin_code": a, "window_cells_sum": float(adm.cells[a]), "lattice_bbox_cells": int(bbox),
                 "windows_in_plan": int(adm.pieces[a]), "windows_done": int(adm.done[a]), "candidates": len(adm.assets[a]),
                 "max_pixels_band_px": r["max_pixels_band_px"],
                 "window_piece_ids": sorted(real.piece_id[real.admin_code == a])})
limits = {"tile_scale": 4, "deadline_ms": 120000, "sdk_retries": 0, "budget_retries": 0, "concurrency": 1,
          "max_high_level_calls": 2 + len(reqs), "outer_timeout_s": 600, "max_pixels": max(r["max_pixels_band_px"] for r in reqs)}
body = {"kind": "whole_admin_pilot", **{k: full[k] for k in ("recipe", "grid", "grid_crs_wkt", "projection_check", "best_effort",
                                                             "project", "collection", "year")},
        "source_manifest_sha256": full["request_sha256"], "plan_sha256": full["provenance"]["plan_sha256"],
        "selection": info, "limits": limits, "requests": reqs}
manifest = {**body, "request_sha256": epc.digest(body), "release_id": None}
out.write_text(json.dumps(manifest, indent=1) + "\n")
print(json.dumps({"request_sha256": manifest["request_sha256"], "limits": limits, "selection": [{k: v for k, v in i.items() if k != "window_piece_ids"} for i in info]}, indent=1))
