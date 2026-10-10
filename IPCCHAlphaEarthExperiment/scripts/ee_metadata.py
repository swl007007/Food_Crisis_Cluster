"""P1 read-only AlphaEarth 2017 metadata inventory (EE interpreter only; release P1-METADATA-20261010).

Read-only: collection/asset metadata, versions, bands/types, native projections and footprints. No raster reduction,
no export, no GCS/Drive write, no project/IAM/API change. Every remote call goes through ``Budget``: hard cap on
requests (default 20), strictly sequential, at most one retry per failed request, per-request deadline, and an
immediate stop on authentication / permission / quota / rate-limit errors. Each request is logged (time, purpose,
status, seconds, response bytes).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
FATAL = ("auth", "permission", "quota", "429", "403", "401", "too many requests", "rate limit", "not enabled", "billing")
#: deterministic request-size rejections: never retried (a retry cannot succeed), reported distinctly from service errors
NONRETRYABLE = ("too many pixels", "maxpixels", "could not be parsed")


class Stop(RuntimeError):
    pass


class Halt(Stop):
    """A shared stop condition (published fatal error, invocation wall budget) refused a new attempt before it was sent."""


class Budget:
    def __init__(self, cap: int, log_path: Path, guard=None, on_fatal=None):
        self.cap, self.used, self.log_path = cap, 0, log_path
        self._lock = threading.Lock()  # concurrent callers share one cap and one log
        # optional shared stop control, consulted at EVERY attempt reservation (first attempts and retries alike)
        self.guard, self.on_fatal = guard, on_fatal

    def call(self, purpose: str, fn):
        for attempt in (1, 2):
            with self._lock:
                reason = self.guard() if self.guard else None
                if reason:
                    raise Halt(f"halted before '{purpose}' attempt {attempt}: {reason}")
                if self.used >= self.cap:
                    raise Stop(f"request cap {self.cap} reached before '{purpose}'")
                self.used += 1
            t0 = time.time()
            try:
                out = fn()
                size = len(json.dumps(out, default=str)) if out is not None else 0
                self._log(purpose, attempt, "ok", time.time() - t0, size)
                return out
            except Exception as error:  # noqa: BLE001
                msg = str(error)
                self._log(purpose, attempt, "error", time.time() - t0, 0, msg[:500])
                if any(k in msg.lower() for k in FATAL):
                    stop = Stop(f"fatal service error on '{purpose}': {msg[:300]}")
                    if self.on_fatal:
                        self.on_fatal(str(stop))  # published before raising: no worker may start a new attempt after it
                    raise stop from error
                if any(k in msg.lower() for k in NONRETRYABLE):
                    raise Stop(f"non-retryable request-size rejection on '{purpose}': {msg[:300]}") from error
                if attempt == 2:
                    raise Stop(f"'{purpose}' failed twice: {msg[:300]}") from error

    def _log(self, purpose, attempt, status, seconds, size, error=None):
        rec = {"n": self.used, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "purpose": purpose, "attempt": attempt,
               "status": status, "seconds": round(seconds, 2), "response_bytes": size}
        if error:
            rec["error"] = error
        with self._lock, open(self.log_path, "a", encoding="utf-8") as h:
            h.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


def list_images(a, budget: Budget, out: Path, result: dict) -> int:
    """Paginated projects.assets.listImages (FULL view): id, times, properties, footprint and band grids per image."""
    import ee

    boxes = json.loads(Path(a.boxes).read_text())["boxes"]
    region = {"type": "MultiPolygon", "coordinates": [[[[b["bbox"][0], b["bbox"][1]], [b["bbox"][2], b["bbox"][1]],
                                                         [b["bbox"][2], b["bbox"][3]], [b["bbox"][0], b["bbox"][3]], [b["bbox"][0], b["bbox"][1]]]]
                                                      for b in boxes]}
    images, token = [], None
    try:
        ee.data.setMaxRetries(0)  # the only retry is Budget's single retry (setDeadline requires an initialized session)
        budget.call("initialize existing project (listing phase; may issue >1 HTTP request: token refresh + discovery)",
                    lambda: ee.Initialize(project=a.project))
        ee.data.setMaxRetries(0)
        ee.data.setDeadline(a.deadline_ms)
        state = ee.data._get_state()
        result["sdk_state"] = {"max_retries": getattr(state, "max_retries", None), "deadline_ms": getattr(state, "deadline_ms", None)}
        while True:
            params = {"parent": f"projects/earthengine-public/assets/{COLLECTION}", "pageSize": a.page, "view": "FULL",
                      # listAssets filters endTime < bound (exclusive) and the 2017 annual images end at 2018-01-01T00:00Z
                      "startTime": "2017-01-01T00:00:00Z", "endTime": "2018-01-01T00:00:01Z", "region": json.dumps(region)}
            if token:
                params["pageToken"] = token
            page = budget.call(f"listImages page {len(images)}", lambda params=params: ee.data.listImages(params))
            images += page.get("images", [])
            token = page.get("nextPageToken")
            if not token:
                break
        result["listing_complete"] = True
    except Stop as stop:
        result["stopped"] = str(stop)
        result["listing_complete"] = False
    result["images_listed"] = len(images)
    result["requests_used"] = budget.used
    (out / "list_images.json").write_text(json.dumps(images) + "\n", encoding="utf-8")
    (out / "listing.json").write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: result.get(k) for k in ("images_listed", "listing_complete", "requests_used", "stopped")}))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--boxes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-requests", type=int, default=20)
    p.add_argument("--deadline-ms", type=int, default=120000)
    p.add_argument("--page", type=int, default=400)
    p.add_argument("--already-used", type=int, default=0, help="requests already spent in this release round (cumulative cap)")
    p.add_argument("--list-images", action="store_true", help="metadata-only REST listing (no server computation)")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    budget = Budget(a.max_requests, out / "request_log.jsonl")
    budget.used = a.already_used
    result = {"collection": COLLECTION, "project": a.project, "year": 2017, "release": "P1-METADATA-20261010"}
    if a.list_images:
        return list_images(a, budget, out, result)
    try:
        import ee

        ee.data.setMaxRetries(0)
        budget.call("initialize existing project (may issue >1 HTTP request)", lambda: ee.Initialize(project=a.project))
        ee.data.setMaxRetries(0)
        ee.data.setDeadline(a.deadline_ms)
        boxes = json.loads(Path(a.boxes).read_text())["boxes"]
        region = ee.Geometry.MultiPolygon([[[[b["bbox"][0], b["bbox"][1]], [b["bbox"][2], b["bbox"][1]], [b["bbox"][2], b["bbox"][3]],
                                              [b["bbox"][0], b["bbox"][3]]]] for b in boxes], None, False)
        all17 = ee.ImageCollection(COLLECTION).filterDate("2017-01-01", "2018-01-01")
        coll = all17.filterBounds(region).sort("system:index")
        result["collection_asset"] = budget.call("collection asset metadata", lambda: ee.data.getAsset(COLLECTION))
        result["counts"] = budget.call("2017 image counts (global, study boxes)", lambda: ee.Dictionary(
            {"global_2017": all17.size(), "study_2017": coll.size()}).getInfo())
        props = ["MODEL_VERSION", "DATASET_VERSION", "PROCESSING_SOFTWARE_VERSION", "UTM_ZONE", "system:time_start", "system:time_end"]
        result["version_histograms"] = budget.call("version/time histograms", lambda: ee.Dictionary(
            {k: coll.aggregate_histogram(k) for k in props}).getInfo())
        result["first_image"] = budget.call("first image full description", lambda: ee.Image(coll.first()).getInfo())

        def proj_props(img):
            pr = img.select("A00").projection()
            return img.set({"p_crs": pr.crs(), "p_scale": pr.nominalScale(), "p_nbands": img.bandNames().size(),
                            "p_types": img.bandTypes().values().map(lambda t: ee.Dictionary(t).get("precision")).distinct()})
        cp = coll.map(proj_props)
        result["projection_histograms"] = budget.call("projection/scale/band histograms", lambda: ee.Dictionary(
            {"crs": cp.aggregate_histogram("p_crs"), "scale": cp.aggregate_histogram("p_scale"), "nbands": cp.aggregate_histogram("p_nbands"),
             "types": cp.aggregate_array("p_types").flatten().distinct()}).getInfo())
        n = int(result["counts"]["study_2017"])

        def describe(img):
            pr = img.select("A00").projection()
            return ee.Feature(img.geometry(), {"id": img.get("system:index"), "crs": pr.crs(), "transform": pr.transform(),
                                               "dataset_version": img.get("DATASET_VERSION"), "model_version": img.get("MODEL_VERSION"),
                                               "processing": img.get("PROCESSING_SOFTWARE_VERSION"), "utm_zone": img.get("UTM_ZONE"),
                                               "start": img.get("system:time_start"), "end": img.get("system:time_end")})
        feats = []
        lst = coll.map(describe).toList(n)
        for off in range(0, n, a.page):
            page = budget.call(f"footprint page offset {off}", lambda off=off: ee.FeatureCollection(lst.slice(off, min(off + a.page, n))).getInfo())
            feats += page["features"]
        result["footprints_complete"] = len(feats) == n
        (out / "footprints.json").write_text(json.dumps({"type": "FeatureCollection", "features": feats}) + "\n", encoding="utf-8")
    except Stop as stop:
        result["stopped"] = str(stop)
    result["requests_used"] = budget.used
    (out / "metadata.json").write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: result.get(k) for k in ("counts", "requests_used", "stopped", "footprints_complete")}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
