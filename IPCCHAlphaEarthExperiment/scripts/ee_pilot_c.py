"""D30 analysis-grid AlphaEarth 2017 pilot (EE interpreter only; NOT RUN without an explicit release_id).

Recipe D30: one fixed analysis grid (crs/affine from the manifest, which must equal the frozen contract values).
For each request window (frozen geographic geometry cut on lattice lines) and its ascending candidate asset ids:
  - every candidate image = its 64 bands (float64) + a constant ``source_rank`` band, ALL masked by the asset's common
    64-band validity (all bands unmasked AND |x| < 1e300) BEFORE compositing -> a cell can never mix bands of assets;
  - composite = ImageCollection(images in DESCENDING id order).mosaic(): EE draws the LAST image on top, so the lowest
    id with a valid vector wins (first valid, ascending id);
  - stack [pixelArea, area*A_j, area*A_j**2] masked by the composite validity, reduced with Reducer.sum() on the
    analysis grid (crs + crsTransform), bestEffort=False; plus an unweighted valid-cell count and the owned area
    grouped by source_rank (empirical check of the priority order) in the SAME high-level call.
Outputs go to local JSON only. No export, no cloud write. Null sums are recorded, never normalized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ee_metadata import Budget, Stop  # noqa: E402

COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
BANDS = [f"A{j:02d}" for j in range(64)]
FINITE_BOUND = 1e300
RECIPE = "D30|EPSG:6933|10,0,0,0,-10,0|nearest|first-valid-ascending-id"
GRID = {"crs": "EPSG:6933", "crsTransform": [10, 0, 0, 0, -10, 0]}


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def load_released(path: Path) -> dict:
    req = json.loads(path.read_text())
    body = {k: v for k, v in req.items() if k not in ("request_sha256", "release_id")}
    if digest(body) != req["request_sha256"]:
        raise SystemExit("C pilot manifest differs from its frozen sha256")
    if not req.get("release_id"):
        raise SystemExit("C pilot manifest has no release_id: EE computation requires the explicit supervisor release")
    if req.get("recipe") != RECIPE or req.get("grid") != GRID or req.get("best_effort") is not False:
        raise SystemExit("C pilot manifest does not carry the frozen D30 recipe/grid/bestEffort=false")
    for r in req["requests"]:
        ids = r["candidates"]
        if ids != sorted(ids) or len(set(ids)) != len(ids) or not ids:
            raise SystemExit(f"{r['request_id']}: candidate ids must be non-empty, unique and ascending")
    return req


def composite(ee, candidates: list):
    """First valid whole 64-band vector in ascending candidate id (common mask applied before the mosaic)."""
    images = []
    for rank, aid in enumerate(candidates, start=1):
        img = ee.Image(f"{COLLECTION}/{aid}").select(BANDS).toDouble()
        valid = img.mask().reduce(ee.Reducer.min()).And(img.abs().lt(FINITE_BOUND).reduce(ee.Reducer.min()))
        images.append(img.addBands(ee.Image.constant(rank).toInt32().rename("source_rank")).updateMask(valid))
    return ee.ImageCollection(images[::-1]).mosaic()  # last on top -> rank 1 (lowest id) wins where valid


def reduction(ee, r: dict, limits: dict):
    comp = composite(ee, r["candidates"])
    owned = comp.select("source_rank").mask()
    area = ee.Image.pixelArea().updateMask(owned)
    vec = comp.select(BANDS)
    stack = ee.Image.cat([area.rename("valid_area"), vec.multiply(area).rename([f"S_{b}" for b in BANDS]),
                          vec.pow(2).multiply(area).rename([f"Q_{b}" for b in BANDS])])
    geom = ee.Geometry(r["geometry"], "EPSG:4326", False)
    max_pixels = float(r.get("max_pixels_band_px", limits["max_pixels"]))  # band-pixels (129-band stack); per-request cap
    if max_pixels > float(limits["max_pixels"]):
        raise ValueError(f"{r['request_id']}: per-request maxPixels exceeds the released ceiling")
    common = dict(geometry=geom, crs=GRID["crs"], crsTransform=GRID["crsTransform"], bestEffort=False,
                  maxPixels=max_pixels, tileScale=int(limits["tile_scale"]))
    sums = stack.reduceRegion(reducer=ee.Reducer.sum(), **common)
    count = owned.selfMask().rename("valid_cells").reduceRegion(reducer=ee.Reducer.count().unweighted(), **common)
    by_rank = ee.Image.cat([area.rename("area"), comp.select("source_rank")]).reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName="source_rank"), **common)
    return ee.Dictionary({"sums": sums, "count": count, "area_by_rank": by_rank}).getInfo()


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--request", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    req = load_released(Path(a.request))
    limits = req["limits"]
    out = Path(a.out)
    (out / "responses").mkdir(parents=True, exist_ok=False)
    budget = Budget(int(limits["max_high_level_calls"]), out / "request_log.jsonl")
    summary = {"release_id": req["release_id"], "request_sha256": req["request_sha256"], "recipe": RECIPE, "done": [], "failed": []}
    try:
        import ee

        ee.data.setMaxRetries(0)
        budget.call("initialize (may issue >1 HTTP request)", lambda: ee.Initialize(project=req["project"]))
        ee.data.setMaxRetries(0)
        ee.data.setDeadline(int(limits["deadline_ms"]))
        for r in req["requests"]:
            t0 = time.time()
            res = budget.call(r["request_id"], lambda r=r: reduction(ee, r, limits))
            (out / "responses" / f"{r['request_id'].replace(':', '__').replace('|', '__')}.json").write_text(
                json.dumps({"request": {k: v for k, v in r.items() if k != "geometry"}, "seconds": round(time.time() - t0, 2),
                            "response": res}) + "\n")
            summary["done"].append(r["request_id"])
    except Stop as stop:
        summary["stopped"] = str(stop)
    summary["calls_used"] = budget.used
    (out / "pilot_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({"calls_used": budget.used, "done": len(summary["done"]), "stopped": summary.get("stopped")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
