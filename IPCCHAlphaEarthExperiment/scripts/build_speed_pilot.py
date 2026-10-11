"""Build the D30 batch speed-pilot manifest (LOCAL ONLY; ipcch-geo interpreter; no EE call; release_id null).

1. Adaptive plan for all frozen admins with the existing ``plan_c_requests(..., max_cells=16000**2)``: an admin that fits
   one 16,000-cell lattice square is one whole item; larger admins become side-16,000 lattice pieces (same lattice and
   origin as every earlier window; each cell in at most one item). Exact piece ids and geometries.
2. Reuse set = the complete admins of the stopped window run (immutable inventory CSV). Fresh pool = items of the other
   admins, excluding every admin already touched (partial window admin, interactive pilots, design example).
3. Deterministic, outcome-free selection: the fresh pool sorted by (geodesic cells, piece id), a systematic sample of n
   items with n = round(target_cells / mean item cells), alternated in that order into tasks A and B.
4. Non-fresh rows (reported separately): interactive whole-admin references and reuse-validation admins (all of their
   adaptive items), with every reference input pinned by sha256.
5. Per task, items grouped by ascending candidate list; one constant maxPixels per group = ceil(129 x 1.1 x max lattice box).
Usage: PYTHONPATH=.:scripts python scripts/build_speed_pilot.py --geometry SHP --list-images JSON --full-run DIR
       --interactive-run DIR --reuse-inventory CSV --out DIR [--target-cells 5e9]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

import ee_pilot_c as epc
from ipcch_alphaearth import extraction
from ipcch_alphaearth.artifacts import write_csv_gz, write_json
from ipcch_alphaearth.contract import load_contract

SPLIT_MAX_CELLS = 16000 * 16000
REFERENCES = {658: "A", 4711: "B"}  # interactive whole-admin pilot responses
REUSE_VALIDATION = {15: "A", 117: "A", 349: "B", 229: "B"}  # complete window admins at q25/q50/q75/q90 cells
TOUCHED_EXTRA = {1050, 4505}  # interactive LARGE timeout; batch-design example
SELECTORS = [*extraction.EXPORT_COLUMNS, extraction.RECIPE_COLUMN, "batch_id"]


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bbox_cells(row) -> int:
    c, r = (json.loads(v) if isinstance(v, str) else list(v) for v in (row.cols, row.rows))
    return (c[1] - c[0]) * (r[1] - r[0])


def select_fresh(items: pd.DataFrame, target_cells: float) -> pd.DataFrame:
    pool = items.sort_values(["cells", "piece_id"], kind="mergesort").reset_index(drop=True)
    n = max(2, int(round(target_cells / pool["cells"].mean())))
    step = len(pool) / n
    picked = pool.iloc[[int(step * (k + 0.5)) for k in range(n)]].copy()
    picked["task"] = ["A" if k % 2 == 0 else "B" for k in range(len(picked))]
    picked["role"] = "fresh"
    return picked


def group_task(rows: pd.DataFrame, geoms: dict) -> list:
    groups = []
    for cands, g in rows.groupby("asset_id", sort=True):
        ids = cands.split(",")
        groups.append({"candidates": ids, "max_pixels_band_px": int(math.ceil(129 * 1.1 * int(g["bbox_cells"].max()))),
                       "items": [{"piece_id": r.piece_id, "admin_code": int(r.admin_code), "role": r.role,
                                  "expected_cells": float(r.cells), "bbox_cells": int(r.bbox_cells),
                                  "geometry": geoms[r.piece_id]} for r in g.sort_values("piece_id").itertuples()]})
    return groups


def main(argv=None) -> int:
    from shapely.geometry import mapping  # noqa: PLC0415
    from shapely.geometry import shape  # noqa: PLC0415

    p = argparse.ArgumentParser()
    for name in ("--geometry", "--list-images", "--full-run", "--interactive-run", "--reuse-inventory", "--out"):
        p.add_argument(name, required=True)
    p.add_argument("--target-cells", type=float, default=5e9)
    p.add_argument("--prefix", default="ipcch-alphaearth/speed-pilot-20261011")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    contract = load_contract()
    full = json.loads((Path(a.full_run) / "inputs" / "production_manifest.json").read_text())
    fps = {i["id"].split("/")[-1]: shape(i["geometry"]) for i in json.loads(Path(a.list_images).read_text())}
    geometry = extraction.load_geometry(Path(a.geometry))
    plan, geoms = extraction.plan_c_requests(geometry, fps, contract, max_cells=SPLIT_MAX_CELLS)
    plan_sha = write_csv_gz(out / "adaptive_plan.csv.gz", plan)
    real = plan[plan["piece_id"].notna()].copy()
    real["cells"] = real["window_area_m2"] / 100.0
    real["bbox_cells"] = [bbox_cells(r) for r in real.itertuples()]
    gmap = {plan.iloc[i]["piece_id"]: mapping(geoms[i]) for i in range(len(plan)) if isinstance(plan.iloc[i]["piece_id"], str)}

    inv = pd.read_csv(a.reuse_inventory, dtype={"piece_id": str})
    reuse_admins = set(inv["admin_code"].astype(int))
    old_plan = pd.read_csv(Path(a.full_run) / "inputs" / "production_plan.csv.gz", dtype={"piece_id": str})
    old_done = {pid for pid in old_plan["piece_id"].dropna()
                if (Path(a.full_run) / "out" / "pieces" / (hashlib.sha1(pid.encode()).hexdigest() + ".json")).exists()}
    touched = set(old_plan.loc[old_plan["piece_id"].isin(old_done), "admin_code"].astype(int)) | set(REFERENCES) | TOUCHED_EXTRA
    fresh_pool = real[~real["admin_code"].isin(touched)]
    remaining = real[~real["admin_code"].isin(reuse_admins)]
    fresh = select_fresh(fresh_pool, a.target_cells)

    nonfresh = []
    for admin, task in {**REFERENCES, **REUSE_VALIDATION}.items():
        rows = real[real["admin_code"] == admin].copy()
        rows["task"], rows["role"] = task, ("reference" if admin in REFERENCES else "reuse_validation")
        nonfresh.append(rows)
    chosen = pd.concat([fresh, *nonfresh], ignore_index=True)
    if chosen["piece_id"].duplicated().any():
        raise SystemExit("an item was selected twice")

    pins = {"interactive_responses": {}, "old_windows": {}}
    for admin in REFERENCES:
        f = next((Path(a.interactive_run) / "out" / "responses").glob(f"WA-*-{admin}.json"))
        pins["interactive_responses"][str(admin)] = {"file": f.name, "sha256": sha(f)}
    inv_idx = inv.set_index("piece_id")
    for admin in REUSE_VALIDATION:
        ids = sorted(old_plan.loc[(old_plan["admin_code"] == admin) & old_plan["piece_id"].notna(), "piece_id"])
        pins["old_windows"][str(admin)] = [{"piece_id": i, "file_sha256": inv_idx.loc[i, "file_sha256"],
                                            "response_sha256": inv_idx.loc[i, "response_sha256"]} for i in ids]

    tasks = [{"name": t, "groups": group_task(chosen[chosen["task"] == t], gmap)} for t in ("A", "B")]
    # EE calls: submit (init + projection + 2 starts) + monitor (init + <=75 per-ID status reads + <=2 cancels)
    # + download (init) + 1 spare = 84; the monitor keeps the cancel reserve before every round.
    limits = {"tile_scale": 4, "deadline_ms": 120000, "http_timeout_s": 60, "sdk_retries": 0, "budget_retries": 0,
              "max_tasks": 2, "max_body_bytes": 8_000_000, "cancel_after_s": 3600, "status_read_interval_s": 120,
              "max_status_reads": 75, "max_cancels": 2, "max_failed_rounds": 3, "max_gcs_reads": 20, "max_ee_calls": 84,
              "max_pixels": max(g["max_pixels_band_px"] for t in tasks for g in t["groups"])}
    body = {"kind": "batch_speed_pilot", **{k: full[k] for k in ("recipe", "grid", "grid_crs_wkt", "projection_check", "best_effort",
                                                              "project", "collection", "year")},
            "destination": {"type": "gcs", "bucket": "manuel-remote-sensing-rasters-645415198431", "prefix": a.prefix},
            "selectors": SELECTORS, "split_max_cells": SPLIT_MAX_CELLS, "adaptive_plan_sha256": plan_sha,
            "inputs": {"geometry_shp_sha256": sha(Path(a.geometry)), "list_images_sha256": sha(Path(a.list_images)),
                       "old_manifest_body_sha256": full["request_sha256"], "old_plan_sha256": full["provenance"]["plan_sha256"],
                       "reuse_inventory_sha256": sha(Path(a.reuse_inventory))},
            "reference_pins": pins, "limits": limits,
            "remaining_workload": {"admins": int(remaining["admin_code"].nunique()), "items": int(len(remaining)),
                                   "cells": float(remaining["cells"].sum()), "adaptive_plan_sha256": plan_sha,
                                   "reuse_admins": len(reuse_admins)},
            "tasks": tasks}
    manifest = {**body, "request_sha256": epc.digest(body), "release_id": None}
    write_json(out / "speed_pilot_manifest.json", manifest)

    per_admin = real.groupby("admin_code").size()
    report = {"adaptive_plan": {"sha256": plan_sha, "admins": int(plan["admin_code"].nunique()), "items": int(len(real)),
                                "whole_admin_items": int((per_admin == 1).sum()), "split_admins": int((per_admin > 1).sum()),
                                "split_pieces": int(per_admin[per_admin > 1].sum()), "max_item_cells": float(real["cells"].max())},
              "reuse": {"admins": len(reuse_admins), "windows": int(len(inv))},
              "remaining": {"admins": int(remaining["admin_code"].nunique()), "items": int(len(remaining)), "cells": float(remaining["cells"].sum())},
              "fresh_pool_items": int(len(fresh_pool)),
              "manifest_body_sha256": manifest["request_sha256"], "prefix": f"gs://{body['destination']['bucket']}/{a.prefix}/"}
    for t in ("A", "B"):
        c = chosen[chosen["task"] == t]
        f = c[c["role"] == "fresh"]
        report[f"task_{t}"] = {"items": int(len(c)), "fresh_items": int(len(f)), "fresh_cells": float(f["cells"].sum()),
                               "nonfresh_cells": float(c.loc[c["role"] != "fresh", "cells"].sum()),
                               "max_item_cells": float(c["cells"].max()), "groups": int(c["asset_id"].nunique()),
                               "inline_geometry_bytes": int(sum(len(json.dumps(gmap[i], separators=(",", ":"))) for i in c["piece_id"]))}
    write_json(out / "speed_pilot_report.json", report)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
