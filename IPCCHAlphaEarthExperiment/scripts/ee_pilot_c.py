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

Production mode (manifest kind ``c_production``; same reducer, composite and projection check): pieces come from a
sha-bound jsonl.gz file; each finished piece is written atomically to ``<out>/pieces/`` with its binding (manifest sha,
piece id, admin, candidates, recipe) and a response digest; ``<out>/ledger.jsonl`` is append-only; a rerun resumes only
pieces without a valid output (a corrupt/mismatched output stops the run and is never overwritten). ``batch_size`` > 1
evaluates several identical per-piece reductions in one getInfo (fewer calls, same numerical reducer). Bounded thread
concurrency; stops on fatal service errors, on ``max_consecutive_failures``, on a failure fraction above
``max_failure_fraction`` after ``min_attempts_for_fraction`` pieces, at the call cap or the invocation wall budget. Stop
conditions are checked at every attempt reservation (retries included) through one shared ``Control``; a fatal error is
published before it is raised. The call cap is cumulative over every resume of one output directory (reservation log),
so a rerun never gets a fresh budget. Invocation wall budget (``wall_budget_s``, from before init) and the outer command
timeout (``outer_timeout_s``, hard kill) are distinct; the manifest must state both.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ee_metadata import Budget, Halt, Stop  # noqa: E402

COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
BANDS = [f"A{j:02d}" for j in range(64)]
FINITE_BOUND = 1e300
RECIPE = "D30|EPSG:6933|10,0,0,0,-10,0|nearest|first-valid-ascending-id"
GRID = {"crs": "EPSG:6933", "crsTransform": [10, 0, 0, 0, -10, 0]}
#: EE does not parse the authority code "EPSG:6933" (pilot 2026-10-10: "The CRS of a map projection could not be parsed").
#: Transport the identical definition as WKT1 (pyproj EPSG:6933 WKT1_GDAL without the outer AUTHORITY tag; verified
#: locally to equal EPSG:6933 with identical transforms) and verify EE's interpretation before any reduction.
EE_CRS_WKT = 'PROJCS["WGS 84 / NSIDC EASE-Grid 2.0 Global",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Cylindrical_Equal_Area"],PARAMETER["standard_parallel_1",30],PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def load_released(path: Path) -> dict:
    req = json.loads(Path(path).read_text())
    body = {k: v for k, v in req.items() if k not in ("request_sha256", "release_id")}
    if digest(body) != req["request_sha256"]:
        raise SystemExit("C pilot manifest differs from its frozen sha256")
    if not req.get("release_id"):
        raise SystemExit("C pilot manifest has no release_id: EE computation requires the explicit supervisor release")
    if req.get("recipe") != RECIPE or req.get("grid") != GRID or req.get("best_effort") is not False:
        raise SystemExit("C pilot manifest does not carry the frozen D30 recipe/grid/bestEffort=false")
    if req.get("grid_crs_wkt") != EE_CRS_WKT or not req.get("projection_check"):
        raise SystemExit("C pilot manifest must pin the transport WKT and its projection-equivalence check points")
    if req.get("kind") == "c_production":
        pieces_path = Path(path).parent / req["pieces_file"]
        if not pieces_path.exists() or hashlib.sha256(pieces_path.read_bytes()).hexdigest() != req["pieces_sha256"]:
            raise SystemExit("production pieces file missing or differs from its pinned sha256")
        lim = req["limits"]
        if "outer_timeout_s" not in lim or float(lim["outer_timeout_s"]) < float(lim["wall_budget_s"]) + float(lim["deadline_ms"]) / 1000 + 60:
            raise SystemExit("production limits must state outer_timeout_s >= invocation wall_budget_s + deadline + 60 s")
        req["requests"] = load_pieces(pieces_path)
        if len(req["requests"]) != int(req["n_pieces"]):
            raise SystemExit("production pieces file count differs from the manifest")
    for r in req["requests"]:
        ids = r["candidates"]
        if ids != sorted(ids) or len(set(ids)) != len(ids) or not ids:
            raise SystemExit(f"{r['request_id']}: candidate ids must be non-empty, unique and ascending")
    if len({r["request_id"] for r in req["requests"]}) != len(req["requests"]):
        raise SystemExit("duplicate request ids in the manifest")
    return req


def load_pieces(path: Path) -> list:
    with gzip.open(path, "rt", encoding="utf-8") as h:
        return [json.loads(line) for line in h if line.strip()]


def composite(ee, candidates: list):
    """First valid whole 64-band vector in ascending candidate id (common mask applied before the mosaic)."""
    images = []
    for rank, aid in enumerate(candidates, start=1):
        img = ee.Image(f"{COLLECTION}/{aid}").select(BANDS).toDouble()
        valid = img.mask().reduce(ee.Reducer.min()).And(img.abs().lt(FINITE_BOUND).reduce(ee.Reducer.min()))
        images.append(img.addBands(ee.Image.constant(rank).toInt32().rename("source_rank")).updateMask(valid))
    return ee.ImageCollection(images[::-1]).mosaic()  # last on top -> rank 1 (lowest id) wins where valid


def reduction(ee, r: dict, limits: dict):
    return reduction_obj(ee, r, limits).getInfo()


def reduction_obj(ee, r: dict, limits: dict):
    """Unevaluated per-piece reduction (sums, valid-cell count, owned area by source rank)."""
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
    common = dict(geometry=geom, crs=EE_CRS_WKT, crsTransform=GRID["crsTransform"], bestEffort=False,
                  maxPixels=max_pixels, tileScale=int(limits["tile_scale"]))
    sums = stack.reduceRegion(reducer=ee.Reducer.sum(), **common)
    count = owned.selfMask().rename("valid_cells").reduceRegion(reducer=ee.Reducer.count().unweighted(), **common)
    by_rank = ee.Image.cat([area.rename("area"), comp.select("source_rank")]).reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName="source_rank"), **common)
    return ee.Dictionary({"sums": sums, "count": count, "area_by_rank": by_rank})


def projection_check(ee, points: list) -> dict:
    """EE's coordinates of reference lon/lat points in the transport WKT (compared with pinned pyproj values)."""
    proj = ee.Projection(EE_CRS_WKT)
    return ee.Dictionary({str(k): ee.Geometry.Point([p[0], p[1]]).transform(proj, 0.001).coordinates()
                          for k, p in enumerate(points)}).getInfo()


def piece_file(out: Path, piece_id: str) -> Path:
    return out / "pieces" / (hashlib.sha1(piece_id.encode()).hexdigest() + ".json")


def piece_record(req: dict, r: dict, response: dict, seconds: float, batch: int) -> dict:
    return {"manifest_sha256": req["request_sha256"], "release_id": req["release_id"], "recipe": RECIPE, "piece_id": r["request_id"],
            "admin_code": r.get("admin_code"), "candidates": r["candidates"], "seconds": round(seconds, 3), "batch_size": batch,
            "response": response, "response_sha256": digest(response)}


def valid_output(req: dict, r: dict, path: Path) -> bool:
    """True for a complete matching output; raises Stop for a present but corrupt or mismatched one."""
    if not path.exists():
        return False
    try:
        rec = json.loads(path.read_text())
    except ValueError as error:
        raise Stop(f"corrupt output for {r['request_id']} ({path.name}); not overwritten") from error
    ok = (rec.get("manifest_sha256") == req["request_sha256"] and rec.get("piece_id") == r["request_id"]
          and rec.get("admin_code") == r.get("admin_code") and rec.get("candidates") == r["candidates"] and rec.get("recipe") == RECIPE
          and rec.get("response_sha256") == digest(rec.get("response")))
    if not ok:
        raise Stop(f"output for {r['request_id']} does not match the manifest binding; not overwritten")
    return True


def write_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}.{threading.get_ident()}")
    tmp.write_text(json.dumps(payload) + "\n")
    os.replace(tmp, path)


class Control:
    """Shared stop state of one released invocation. ``wall_budget_s`` is the INVOCATION wall budget, measured from
    before init/projection and enforced at every attempt reservation (first attempts and retries): after it, or after
    a published fatal error / stop rule, no new attempt starts; already-issued requests may finish (at most one per
    worker, each bounded by ``deadline_ms``). The OUTER command timeout (``outer_timeout_s``, e.g. ``timeout``) is a
    separate hard kill and must exceed wall budget + deadline."""

    def __init__(self, wall_budget_s: float, t_start: float | None = None):
        self.t_start = time.time() if t_start is None else t_start
        self.wall_budget_s, self.stopped, self.lock = float(wall_budget_s), None, threading.Lock()

    def stop(self, reason: str) -> None:
        with self.lock:
            if not self.stopped:
                self.stopped = reason

    def reason(self):
        with self.lock:
            if not self.stopped and time.time() - self.t_start > self.wall_budget_s:
                self.stopped = f"invocation wall budget {self.wall_budget_s} s reached"
            return self.stopped


def run_production(req: dict, out: Path, ee, budget: Budget, sleep=time.sleep, control: Control | None = None) -> dict:
    limits = req["limits"]
    out = Path(out)
    control = control or Control(limits["wall_budget_s"])
    budget.guard, budget.on_fatal = control.reason, control.stop
    (out / "pieces").mkdir(parents=True, exist_ok=True)
    pin = out / "MANIFEST.sha256"
    if pin.exists() and pin.read_text().strip() != req["request_sha256"]:
        raise SystemExit("output directory belongs to a different production manifest")
    pin.write_text(req["request_sha256"] + "\n")
    todo = [r for r in req["requests"] if not valid_output(req, r, piece_file(out, r["request_id"]))]
    batch = max(1, int(limits.get("batch_size", 1)))
    groups = [todo[k:k + batch] for k in range(0, len(todo), batch)]
    state = {"attempted": 0, "failed": 0, "consecutive": 0, "done": 0, "halted": 0}
    lock = threading.Lock()

    def ledger(rec):
        with lock, open(out / "ledger.jsonl", "a", encoding="utf-8") as h:
            h.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **rec}) + "\n")

    def work(group):
        if control.reason():
            return
        t0 = time.time()
        label = group[0]["request_id"] if len(group) == 1 else f"batch[{group[0]['request_id']}..+{len(group) - 1}]"
        try:
            res = budget.call(label, lambda: ee.Dictionary({str(i): reduction_obj(ee, r, limits) for i, r in enumerate(group)}).getInfo())
        except Halt as halt:  # not sent (or retry not sent): the pieces stay pending for a released resume
            with lock:
                state["halted"] += len(group)
            for r in group:
                ledger({"piece_id": r["request_id"], "status": "halted", "error": str(halt)[:300]})
            return
        except Stop as stop:
            msg = str(stop)
            with lock:
                state["attempted"] += len(group)
                state["failed"] += len(group)
                state["consecutive"] += 1
                frac = state["failed"] / state["attempted"]
                if msg.startswith("fatal") or "cap" in msg:
                    control.stop(msg)
                elif state["consecutive"] >= int(limits["max_consecutive_failures"]):
                    control.stop(f"{state['consecutive']} consecutive failed calls")
                elif state["attempted"] >= int(limits["min_attempts_for_fraction"]) and frac > float(limits["max_failure_fraction"]):
                    control.stop(f"failure fraction {frac:.4f} above limit")
            for r in group:
                ledger({"piece_id": r["request_id"], "status": "failed", "error": msg[:300]})
            return
        seconds = time.time() - t0
        for i, r in enumerate(group):
            rec = piece_record(req, r, res[str(i)], seconds, len(group))
            write_atomic(piece_file(out, r["request_id"]), rec)
            ledger({"piece_id": r["request_id"], "status": "ok", "seconds": rec["seconds"], "response_sha256": rec["response_sha256"]})
        with lock:
            state["attempted"] += len(group)
            state["done"] += len(group)
            state["consecutive"] = 0

    with ThreadPoolExecutor(max_workers=max(1, int(limits["concurrency"]))) as pool:
        list(pool.map(work, groups))
    remaining = [r["request_id"] for r in req["requests"] if not valid_output(req, r, piece_file(out, r["request_id"]))]
    return {**state, "stopped": control.stopped, "pending_at_start": len(todo), "remaining": len(remaining),
            "complete": not remaining, "batches": len(groups), "calls_used": budget.used,
            "invocation_seconds": round(time.time() - control.t_start, 3)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--request", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    req = load_released(Path(a.request))
    limits = req["limits"]
    out = Path(a.out)
    if req.get("kind") == "c_production":
        return main_production(req, out)
    (out / "responses").mkdir(parents=True, exist_ok=False)
    budget = Budget(int(limits["max_high_level_calls"]), out / "request_log.jsonl")
    summary = {"release_id": req["release_id"], "request_sha256": req["request_sha256"], "recipe": RECIPE, "done": [], "failed": []}
    try:
        import ee

        ee.data.setMaxRetries(0)
        budget.call("initialize (may issue >1 HTTP request)", lambda: ee.Initialize(project=req["project"]))
        ee.data.setMaxRetries(0)
        ee.data.setDeadline(int(limits["deadline_ms"]))
        got = budget.call("projection equivalence check (transport WKT vs pinned pyproj EPSG:6933)",
                          lambda: projection_check(ee, req["projection_check"]))
        worst = max(max(abs(got[str(k)][0] - p[2]), abs(got[str(k)][1] - p[3])) for k, p in enumerate(req["projection_check"]))
        summary["projection_check"] = {"ee": got, "max_abs_diff_m": worst}
        if worst > 1e-3:
            raise Stop(f"EE interprets the transport WKT differently from EPSG:6933 (max diff {worst} m); no reduction run")
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


class CumulativeBudget(Budget):
    """One call cap per production output directory across every resume: each attempt appends a reservation line
    BEFORE it is sent, and a rerun starts from the number of reservations already made (no fresh budget on resume)."""

    def __init__(self, cap: int, out: Path):
        super().__init__(cap, out / "request_log.jsonl")
        self.reservations = out / "call_reservations.log"
        self.used = 0
        if self.reservations.exists():
            with open(self.reservations, encoding="utf-8") as h:
                self.used = sum(1 for _ in h)

    def call(self, purpose: str, fn):
        def reserved():
            with self._lock, open(self.reservations, "a", encoding="utf-8") as h:
                h.write(purpose[:120] + "\n")
            return fn()

        return super().call(purpose, reserved)


def main_production(req: dict, out: Path, ee=None) -> int:
    limits = req["limits"]
    out.mkdir(parents=True, exist_ok=True)
    pin = out / "MANIFEST.sha256"
    if pin.exists() and pin.read_text().strip() != req["request_sha256"]:
        raise SystemExit("output directory belongs to a different production manifest")  # before any EE call
    control = Control(limits["wall_budget_s"])  # invocation wall clock starts before init/projection
    budget = CumulativeBudget(int(limits["max_high_level_calls"]), out)
    budget.guard, budget.on_fatal = control.reason, control.stop
    summary = {"release_id": req["release_id"], "request_sha256": req["request_sha256"], "recipe": RECIPE}
    try:
        if ee is None:
            import ee as ee  # noqa: PLC0415,PLW0127

        ee.data.setMaxRetries(0)
        budget.call("initialize (may issue >1 HTTP request)", lambda: ee.Initialize(project=req["project"]))
        ee.data.setMaxRetries(0)
        ee.data.setDeadline(int(limits["deadline_ms"]))
        got = budget.call("projection equivalence check", lambda: projection_check(ee, req["projection_check"]))
        worst = max(max(abs(got[str(k)][0] - p[2]), abs(got[str(k)][1] - p[3])) for k, p in enumerate(req["projection_check"]))
        summary["projection_check_max_abs_diff_m"] = worst
        if worst > 1e-3:
            raise Stop(f"EE interprets the transport WKT differently (max diff {worst} m); no reduction run")
        summary.update(run_production(req, out, ee, budget, control=control))
    except Stop as stop:
        summary["stopped"] = str(stop)
    summary["calls_used"] = budget.used
    with open(out / "run_summaries.jsonl", "a", encoding="utf-8") as h:
        h.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **summary}) + "\n")
    print(json.dumps({k: summary.get(k) for k in ("calls_used", "done", "remaining", "complete", "stopped")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
