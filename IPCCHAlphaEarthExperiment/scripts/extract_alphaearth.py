"""AlphaEarth 2017 native-grid sufficient-statistic extraction (EE side; /opt/earthengine/bin/python only).

NOT RUN IN P0. Every command contacts Earth Engine and therefore requires the concrete P1 release:

    inventory --project P --bbox W S E N --out FILE      read-only 2017 asset/version/projection/footprint inventory
    export    --request FILE --ledger FILE               submit table exports for a released, hash-verified request
    status    --ledger FILE                              read-only task status poll

Contract (design section 5; D16/D25/D26): collection GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL, calendar 2017 only, bands
A00..A63 cast to float64; common mask = all 64 bands unmasked AND finite (|x| < 1e300, so NaN and +/-Inf fail);
per frozen piece (admin & footprint minus
earlier footprints, computed offline) reduce [pixelArea, area*A_j, area*A_j**2] with Reducer.sum() in the piece's own
asset projection (crs + crsTransform), bestEffort=False. EE's fractional boundary weights are accepted (approximate,
~1/256). Piece geometries are planar lon/lat polygons (geodesic=False), matching the offline shapely ownership.
Only sufficient-statistic tables are exported; no raster archive. A completed piece with zero valid area exports
A = S = Q = 0 (genuine no coverage); a failed task never becomes low coverage. Pilot check (not yet observed): an
all-masked reduceRegion may return null sums; until the pilot proves that a successful zero-valid-area reduction is
what produced them, null/missing sums stay technical failures downstream (never normalized here).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ipcch_alphaearth import aemoments  # noqa: E402  (pure Python, no numpy)

BANDS = list(aemoments.BANDS)
S_NAMES = [f"S_{b}" for b in BANDS]
Q_NAMES = [f"Q_{b}" for b in BANDS]
COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
FINITE_BOUND = 1e300  # embeddings are unit-norm components; NaN fails every comparison, +/-Inf fails |x| < bound
ACTIVE = ("SUBMITTED", "READY", "RUNNING")


def canonical_digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def load_released_request(path: Path) -> dict:
    req = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {k: v for k, v in req.items() if k not in ("request_sha256", "release_id")}
    if canonical_digest(body) != req.get("request_sha256"):
        raise SystemExit("request body differs from its frozen sha256; refusing")
    if not req.get("release_id"):
        raise SystemExit("request has no release_id: EE export requires the explicit P1 supervisor release")
    if req["collection"] != COLLECTION or int(req["year"]) != 2017 or req["best_effort"] is not False:
        raise SystemExit("request violates the frozen source/year/bestEffort contract")
    return req


def init(project: str):
    import ee

    ee.Initialize(project=project)
    return ee


def inventory(args) -> None:
    ee = init(args.project)
    region = ee.Geometry.Rectangle(args.bbox, proj="EPSG:4326", geodesic=False)
    coll = ee.ImageCollection(COLLECTION).filterDate("2017-01-01", "2018-01-01").filterBounds(region)
    props = ["system:index", "system:time_start", "system:time_end", "MODEL_VERSION", "DATASET_VERSION",
             "PROCESSING_SOFTWARE_VERSION", "UTM_ZONE"]

    def describe(img):
        proj = img.select("A00").projection()
        return ee.Feature(img.geometry(), img.toDictionary(props).combine(
            {"crs": proj.crs(), "transform": proj.transform(), "band_names": img.bandNames(),
             "band_types": img.bandTypes()}))

    info = ee.FeatureCollection(coll.map(describe)).getInfo()
    out = {"collection": COLLECTION, "year": 2017, "bbox": args.bbox, "queried_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "assets": sorted(info["features"], key=lambda f: f["properties"]["system:index"])}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"inventory: {len(out['assets'])} assets -> {args.out}")


def common_valid_mask(ee, bands):
    """1 where all 64 bands are unmasked and finite, else 0 (shared footprint for every band's moments)."""
    return bands.mask().reduce(ee.Reducer.min()).And(bands.abs().lt(FINITE_BOUND).reduce(ee.Reducer.min()))


def reduce_piece_factory(ee, max_pixels: float, tile_scale: int):
    def reduce_piece(f):
        img = ee.Image.load(ee.String(COLLECTION + "/").cat(f.get("asset_id")))
        bands = img.select(BANDS).toDouble()
        proj = img.select("A00").projection()
        valid = common_valid_mask(ee, bands)
        area = ee.Image.pixelArea()
        stack = ee.Image.cat([area.rename("valid_area"), bands.multiply(area).rename(S_NAMES),
                              bands.pow(2).multiply(area).rename(Q_NAMES)]).updateMask(valid)
        stats = stack.reduceRegion(reducer=ee.Reducer.sum(), geometry=f.geometry(), crs=proj, bestEffort=False,
                                   maxPixels=max_pixels, tileScale=tile_scale)
        return ee.Feature(None, stats).set({"admin_code": f.get("admin_code"), "piece_id": f.get("piece_id"),
                                            "asset_id": f.get("asset_id"), "status": "complete"})
    return reduce_piece


def wait_for_slot(ee, ledger: dict, max_concurrent: int, sleep, save) -> None:
    """Block until fewer than ``max_concurrent`` ledger tasks are active (refreshing states first, also after resume)."""
    refresh(ee, ledger)
    save()
    while sum(t["state"] in ACTIVE for t in ledger["tasks"]) >= max_concurrent:
        sleep(30)
        refresh(ee, ledger)
        save()


def export(args, ee=None, sleep=time.sleep) -> None:
    req = load_released_request(Path(args.request))
    ledger_path = Path(args.ledger)
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"request_sha256": req["request_sha256"], "tasks": []}
    if ledger["request_sha256"] != req["request_sha256"]:
        raise SystemExit("ledger belongs to a different request")

    def save():
        ledger_path.write_text(json.dumps(ledger, indent=1) + "\n", encoding="utf-8")

    ee = ee or init(req["project"])
    refresh(ee, ledger)
    done = {pid for t in ledger["tasks"] if t.get("state") == "COMPLETED" for pid in t["piece_ids"]}
    active = {pid for t in ledger["tasks"] if t.get("state") in ACTIVE for pid in t["piece_ids"]}
    todo = [p for p in req["pieces"] if p["piece_id"] not in done and p["piece_id"] not in active]
    limits = req["limits"]
    per_task, max_conc = int(limits["pieces_per_task"]), int(limits["max_concurrent"])
    if len(ledger["tasks"]) + -(-len(todo) // per_task) > int(limits["max_tasks"]):
        raise SystemExit("submission would exceed the released max_tasks limit")
    reducer = reduce_piece_factory(ee, float(limits["max_pixels"]), int(limits.get("tile_scale", 1)))
    dest = req["destination"]
    for start in range(0, len(todo), per_task):
        batch = todo[start:start + per_task]
        wait_for_slot(ee, ledger, max_conc, sleep, save)  # concurrency is enforced BEFORE every start
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry(p["geometry"], "EPSG:4326", False),
                                              {"admin_code": p["admin_code"], "piece_id": p["piece_id"], "asset_id": p["asset_id"]})
                                   for p in batch])
        desc = f"ae2017_{req['kind']}_{req['request_sha256'][:10]}_{len(ledger['tasks']):05d}"
        kwargs = dict(collection=fc.map(reducer), description=desc, fileFormat="CSV", selectors=req["export_columns"])
        if dest["type"] == "gcs":
            task = ee.batch.Export.table.toCloudStorage(bucket=dest["bucket"], fileNamePrefix=f"{dest['prefix']}/{desc}", **kwargs)
        elif dest["type"] == "drive":
            task = ee.batch.Export.table.toDrive(folder=dest["folder"], fileNamePrefix=desc, **kwargs)
        else:
            raise SystemExit(f"unknown destination type {dest['type']}")
        task.start()
        ledger["tasks"].append({"description": desc, "task_id": task.id, "state": "SUBMITTED",
                                "piece_ids": [p["piece_id"] for p in batch], "submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        save()
    print(f"submitted {len(todo)} pieces; ledger {ledger_path}")


def refresh(ee, ledger: dict) -> None:
    states = {s["id"]: s for s in ee.data.getTaskList()} if ledger["tasks"] else {}
    for t in ledger["tasks"]:
        s = states.get(t["task_id"])
        if s:
            t["state"] = s["state"]
            t["error_message"] = s.get("error_message")


def status(args) -> None:
    ledger = json.loads(Path(args.ledger).read_text())
    req_project = args.project
    ee = init(req_project)
    refresh(ee, ledger)
    Path(args.ledger).write_text(json.dumps(ledger, indent=1) + "\n", encoding="utf-8")
    counts = {}
    for t in ledger["tasks"]:
        counts[t["state"]] = counts.get(t["state"], 0) + 1
    print(json.dumps(counts))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("inventory")
    a.add_argument("--project", required=True)
    a.add_argument("--bbox", type=float, nargs=4, required=True)
    a.add_argument("--out", required=True)
    a = sub.add_parser("export")
    a.add_argument("--request", required=True)
    a.add_argument("--ledger", required=True)
    a = sub.add_parser("status")
    a.add_argument("--ledger", required=True)
    a.add_argument("--project", required=True)
    args = p.parse_args(argv)
    {"inventory": inventory, "export": export, "status": status}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
