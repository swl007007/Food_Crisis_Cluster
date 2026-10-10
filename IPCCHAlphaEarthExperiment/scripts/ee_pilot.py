"""Staged AlphaEarth 2017 pilot: direct native-grid A/S/Q reductions returned to local files (EE interpreter only).

NOT RUN without an explicit supervisor release (manifest release_id). No export, no GCS/Drive write, no project/IAM
change. Each request reduces ONE frozen geometry on its OWN asset's native projection:
    stack = [pixelArea, area*A_j, area*A_j**2] masked by (all 64 bands unmasked AND |x| < 1e300)
    reduceRegion(Reducer.sum() combined with an unweighted valid-pixel count, crs=asset projection, bestEffort=False)
and also returns EE's geodesic area of the request geometry for consistency checks. Responses are written verbatim
to <out>/responses/<request_id>.json with timing; null/missing sums are recorded, never normalized here.

Staging: A (protocol) -> B only if every B request projects <= 25% of the deadline, where projection = fixed overhead
(the tiny all-masked request's time) + marginal seconds/pixel (slowest observed (t - overhead)/pixels)
-> C1 only if all B succeeded within projection -> C2 never automatic (requires --allow-c2 and a separate release).
Budget: high-level calls capped (manifest limits), sequential, SDK retries 0, one Budget retry, deadline after init,
stop on auth/permission/quota; two distinct requests failing twice on memory/timeout stops the pilot.
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


def canonical_digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def load_released(path: Path) -> dict:
    req = json.loads(path.read_text())
    body = {k: v for k, v in req.items() if k not in ("request_sha256", "release_id")}
    if canonical_digest(body) != req["request_sha256"]:
        raise SystemExit("pilot manifest differs from its frozen sha256")
    if not req.get("release_id"):
        raise SystemExit("pilot manifest has no release_id: EE reductions require the explicit supervisor pilot release")
    return req


def reduction(ee, r: dict, limits: dict):
    img = ee.Image(f"{COLLECTION}/{r['asset_id']}")
    bands = img.select(BANDS).toDouble()
    proj = img.select("A00").projection()
    valid = bands.mask().reduce(ee.Reducer.min()).And(bands.abs().lt(FINITE_BOUND).reduce(ee.Reducer.min()))
    area = ee.Image.pixelArea()
    stack = ee.Image.cat([area.rename("valid_area"), bands.multiply(area).rename([f"S_{b}" for b in BANDS]),
                          bands.pow(2).multiply(area).rename([f"Q_{b}" for b in BANDS])]).updateMask(valid)
    geom = ee.Geometry(r["geometry"], "EPSG:4326", False)
    sums = stack.reduceRegion(reducer=ee.Reducer.sum(), geometry=geom, crs=proj, bestEffort=False,
                              maxPixels=float(limits["max_pixels"]), tileScale=int(limits["tile_scale"]))
    count = valid.selfMask().rename("valid_pixels").reduceRegion(reducer=ee.Reducer.count().unweighted(), geometry=geom, crs=proj,
                                                                 bestEffort=False, maxPixels=float(limits["max_pixels"]),
                                                                 tileScale=int(limits["tile_scale"]))
    return ee.Dictionary({"sums": sums, "count": count, "ee_geodesic_area_m2": geom.area(1), "projection": proj}).getInfo()


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--request", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-c2", action="store_true", help="full stress tile; only under its own explicit release")
    a = p.parse_args(argv)
    req = load_released(Path(a.request))
    limits = req["limits"]
    out = Path(a.out)
    (out / "responses").mkdir(parents=True, exist_ok=True)
    budget = Budget(int(limits["max_high_level_calls"]), out / "request_log.jsonl")
    summary = {"release_id": req["release_id"], "request_sha256": req["request_sha256"], "done": [], "skipped": {}, "failed": []}
    rate, overhead = None, None
    try:
        import ee

        ee.data.setMaxRetries(0)
        budget.call("initialize (may issue >1 HTTP request)", lambda: ee.Initialize(project=req["project"]))
        ee.data.setMaxRetries(0)
        ee.data.setDeadline(int(limits["deadline_ms"]))
        deadline_s = int(limits["deadline_ms"]) / 1000
        memory_failures, timings = set(), []
        for stage in ("A", "B", "C1", "C2"):
            if stage != "A" and timings:
                overhead = overhead if overhead is not None else min(t for _, t in timings)
                marg = [(t - overhead) / px for px, t in timings if px > 1e5]
                rate = max(max(marg), 0.0) if marg else None  # slowest observed marginal seconds per pixel
            todo = [r for r in req["requests"] if r["stage"] == stage]
            if stage == "C2" and not a.allow_c2:
                summary["skipped"].update({r["request_id"]: "C2 requires --allow-c2 and its own release" for r in todo})
                continue
            if stage != "A":
                gate = 0.25 if stage in ("B", "C1") else 0.5
                if rate is None or summary["failed"]:
                    summary["skipped"].update({r["request_id"]: f"stage {stage} gated: no measured rate or earlier failure" for r in todo})
                    continue
                too_big = [r["request_id"] for r in todo if overhead + rate * r["local_geodesic_area_m2"] / 100 > gate * deadline_s]
                if too_big:
                    summary["skipped"].update({r["request_id"]: f"stage {stage} gated: projected > {gate:.0%} of deadline" for r in todo})
                    continue
            for r in todo:
                t0 = time.time()
                try:
                    res = budget.call(r["request_id"], lambda r=r: reduction(ee, r, limits))
                except Stop as stop:
                    if "fatal" in str(stop) or "cap" in str(stop):
                        raise
                    summary["failed"].append({"request_id": r["request_id"], "error": str(stop)[:300]})
                    memory_failures.add(r["request_id"])
                    if len(memory_failures) >= 2:
                        raise Stop("two distinct requests failed twice") from stop
                    continue
                seconds = time.time() - t0
                px = r["local_geodesic_area_m2"] / 100
                if r["kind"] == "all_masked_fixture":
                    overhead = seconds
                timings.append((px, seconds))
                (out / "responses" / f"{r['request_id'].replace(':', '__')}.json").write_text(
                    json.dumps({"request": {k: v for k, v in r.items() if k != "geometry"}, "seconds": round(seconds, 2), "response": res}) + "\n")
                summary["done"].append(r["request_id"])
    except Stop as stop:
        summary["stopped"] = str(stop)
    summary["seconds_per_pixel_conservative"] = rate
    summary["overhead_seconds"] = overhead
    summary["calls_used"] = budget.used
    (out / "pilot_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("calls_used", "seconds_per_pixel_conservative")} | {"done": len(summary["done"]),
                     "skipped": len(summary["skipped"]), "failed": len(summary["failed"]), "stopped": summary.get("stopped")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
