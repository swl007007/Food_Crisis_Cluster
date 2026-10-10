"""Ownership mask diagnostic (EE interpreter only; NOT RUN without an explicit release_id in the manifest).

For ONE frozen geometry and two AlphaEarth 2017 assets (owner, donor), on each requested reference grid (an asset's
native 10 m projection incl. affine; no bestEffort), sum pixelArea over four mutually exclusive classes:
owner_only, donor_only, both, neither, where each tile's mask = all 64 bands unmasked AND |x| < 1e300, and pixels
that are invalid or outside the tile footprint count as false (unmask(0)). The non-reference tile's mask is resampled
to the reference grid with EE's default nearest-neighbour resampling — for this diagnostic only; AlphaEarth value
extraction and the ownership recipe are unchanged. Four area bands per request; maxPixels is in band-pixels.
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


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def valid_mask(ee, asset_id: str):
    bands = ee.Image(f"{COLLECTION}/{asset_id}").select(BANDS).toDouble()
    v = bands.mask().reduce(ee.Reducer.min()).And(bands.abs().lt(FINITE_BOUND).reduce(ee.Reducer.min()))
    return v.unmask(0, False)


def classes(ee, req: dict, ref_asset: str):
    ref_proj = ee.Image(f"{COLLECTION}/{ref_asset}").select("A00").projection()
    mo = valid_mask(ee, req["owner_asset"]).reproject(ref_proj)  # nearest neighbour (EE default) for the non-native one
    md = valid_mask(ee, req["donor_asset"]).reproject(ref_proj)
    area = ee.Image.pixelArea()
    stack = ee.Image.cat([area.updateMask(mo.And(md.Not())).unmask(0).rename("owner_only"),
                          area.updateMask(mo.Not().And(md)).unmask(0).rename("donor_only"),
                          area.updateMask(mo.And(md)).unmask(0).rename("both"),
                          area.updateMask(mo.Not().And(md.Not())).unmask(0).rename("neither")])
    geom = ee.Geometry(req["geometry"], "EPSG:4326", False)
    sums = stack.reduceRegion(reducer=ee.Reducer.sum(), geometry=geom, crs=ref_proj, bestEffort=False,
                              maxPixels=float(req["limits"]["max_pixels"]), tileScale=int(req["limits"]["tile_scale"]))
    return ee.Dictionary({"sums": sums, "reference_projection": ref_proj}).getInfo()


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--request", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    req = json.loads(Path(a.request).read_text())
    body = {k: v for k, v in req.items() if k not in ("request_sha256", "release_id")}
    if digest(body) != req["request_sha256"]:
        raise SystemExit("diagnostic manifest differs from its frozen sha256")
    if not req.get("release_id"):
        raise SystemExit("diagnostic manifest has no release_id: requires the explicit supervisor release")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    budget = Budget(int(req["limits"]["max_high_level_calls"]), out / "request_log.jsonl")
    summary = {"release_id": req["release_id"], "request_sha256": req["request_sha256"], "results": {}}
    try:
        import ee

        ee.data.setMaxRetries(0)
        budget.call("initialize (may issue >1 HTTP request)", lambda: ee.Initialize(project=req["project"]))
        ee.data.setMaxRetries(0)
        ee.data.setDeadline(int(req["limits"]["deadline_ms"]))
        for ref in req["reference_assets"]:
            t0 = time.time()
            res = budget.call(f"mask classes on reference grid {ref}", lambda ref=ref: classes(ee, req, ref))
            summary["results"][ref] = {"seconds": round(time.time() - t0, 2), "response": res}
            (out / f"mask_classes_ref_{ref}.json").write_text(json.dumps(summary["results"][ref]) + "\n")
    except Stop as stop:
        summary["stopped"] = str(stop)
    summary["calls_used"] = budget.used
    (out / "diagnostic_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({"calls_used": budget.used, "stopped": summary.get("stopped"), "refs_done": list(summary["results"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
