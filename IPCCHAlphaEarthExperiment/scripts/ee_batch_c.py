"""D30 batch-export speed pilot (EE interpreter only; NOT RUN without an explicit release_id).

Same science as ``ee_pilot_c``: the existing D30 ``composite`` (mask before the descending mosaic) and ONE
``reduceRegion(Reducer.sum())`` per item over [valid_area, S x64, Q x64] on the pinned EPSG:6933 10 m lattice (WKT
transport), bestEffort False, one constant maxPixels per candidate group. Items of a task are grouped by their ascending
candidate list; the grouped collections are flattened into one ``Export.table.toCloudStorage`` CSV task.

Subcommands (one released manifest, one output directory, one cumulative call cap):
  submit    measure every task's serialized body after Initialize and refuse ALL starts if any exceeds the cap; then per
            task write a ledger INTENT record BEFORE ``start()``; a ``start()`` error marks the task AMBIGUOUS and holds
            (never resubmitted; inspect with ``lookup``). A task with any ledger record is never started again.
  monitor   one grouped ``getTaskStatus([ids])`` read per interval (bounded); at ``cancel_after_s`` from the first
            submission, ``cancelTask`` only for this manifest's own nonterminal tasks (bounded); outcomes preserved.
  lookup    read-only ``getTaskList`` search by this manifest's task descriptions (resolves an AMBIGUOUS start by hand).
  download  bounded GCS JSON-API reads: list every object under each task's prefix (several objects possible) and
            download it with its sha256; never assumes a single CSV.
No client resubmission. EE's own server-side task retries are recorded by the status reads, not controlled here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ee_pilot_c as epc  # noqa: E402
from ee_metadata import Stop  # noqa: E402

KIND = "batch_speed_pilot"
TERMINAL = ("COMPLETED", "FAILED", "CANCELLED")
S_NAMES = [f"S_{b}" for b in epc.BANDS]
Q_NAMES = [f"Q_{b}" for b in epc.BANDS]


def load_released(path: Path) -> dict:
    req = json.loads(Path(path).read_text())
    body = {k: v for k, v in req.items() if k not in ("request_sha256", "release_id")}
    if epc.digest(body) != req["request_sha256"]:
        raise SystemExit("batch manifest differs from its frozen sha256")
    if not req.get("release_id"):
        raise SystemExit("batch manifest has no release_id: cloud writes require the explicit user/supervisor release")
    if req.get("kind") != KIND or req.get("recipe") != epc.RECIPE or req.get("grid") != epc.GRID or req.get("best_effort") is not False:
        raise SystemExit("batch manifest does not carry the frozen D30 recipe/grid/bestEffort=false")
    if req.get("grid_crs_wkt") != epc.EE_CRS_WKT or not req.get("projection_check"):
        raise SystemExit("batch manifest must pin the transport WKT and its projection-equivalence check points")
    ids = [it["piece_id"] for t in req["tasks"] for g in t["groups"] for it in g["items"]]
    if len(ids) != len(set(ids)):
        raise SystemExit("an item appears more than once in the batch manifest")
    for t in req["tasks"]:
        for g in t["groups"]:
            if g["candidates"] != sorted(g["candidates"]) or len(set(g["candidates"])) != len(g["candidates"]) or not g["candidates"]:
                raise SystemExit(f"{t['name']}: candidate ids must be non-empty, unique and ascending")
            if int(g["max_pixels_band_px"]) > int(req["limits"]["max_pixels"]):
                raise SystemExit(f"{t['name']}: group maxPixels exceeds the released ceiling")
    return req


def description(req: dict, task: dict) -> str:
    return f"ae2017d30_{req['request_sha256'][:10]}_{task['name']}_a1"


def task_prefix(req: dict, task: dict) -> str:
    return f"{req['destination']['prefix'].rstrip('/')}/{task['name']}/attempt-1"


def build_collection(ee, req: dict, task: dict):
    """One flattened FeatureCollection: per candidate group one D30 stack and one sums-only reduceRegion per item."""
    parts = []
    for g in task["groups"]:
        comp = epc.composite(ee, g["candidates"])
        owned = comp.select("source_rank").mask()
        area = ee.Image.pixelArea().updateMask(owned)
        vec = comp.select(epc.BANDS)
        stack = ee.Image.cat([area.rename("valid_area"), vec.multiply(area).rename(S_NAMES), vec.pow(2).multiply(area).rename(Q_NAMES)])
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry(it["geometry"], "EPSG:4326", False),
                                              {"admin_code": it["admin_code"], "piece_id": it["piece_id"],
                                               "asset_id": ",".join(g["candidates"])}) for it in g["items"]])
        max_pixels = int(g["max_pixels_band_px"])

        def reduce_item(f, stack=stack, max_pixels=max_pixels):
            sums = stack.reduceRegion(reducer=ee.Reducer.sum(), geometry=f.geometry(), crs=epc.EE_CRS_WKT,
                                      crsTransform=epc.GRID["crsTransform"], bestEffort=False, maxPixels=max_pixels,
                                      tileScale=int(req["limits"]["tile_scale"]))
            return (ee.Feature(None, sums).copyProperties(f, ["admin_code", "piece_id", "asset_id"])
                    .set({"status": "complete", "recipe": epc.RECIPE, "batch_id": task["name"]}))

        parts.append(fc.map(reduce_item))
    return ee.FeatureCollection(parts).flatten()


def export_task(ee, req: dict, task: dict, collection):
    return ee.batch.Export.table.toCloudStorage(collection=collection, description=description(req, task),
                                                bucket=req["destination"]["bucket"], fileNamePrefix=f"{task_prefix(req, task)}/rows",
                                                fileFormat="CSV", selectors=req["selectors"])


def body_bytes(ee, collection) -> int:
    return len(ee.serializer.toJSON(collection).encode("utf-8"))


def read_ledger(out: Path) -> list:
    path = out / "ledger.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def append_ledger(out: Path, rec: dict) -> None:
    with open(out / "ledger.jsonl", "a", encoding="utf-8") as h:
        h.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **rec}) + "\n")
        h.flush()


def budget_for(out: Path, req: dict):
    budget = epc.CumulativeBudget(int(req["limits"]["max_ee_calls"]), out)
    budget.attempts = 1  # no automatic retry of any client call
    return budget


def pin_output(out: Path, req: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pin = out / "MANIFEST.sha256"
    if pin.exists() and pin.read_text().strip() != req["request_sha256"]:
        raise SystemExit("output directory belongs to a different batch manifest")
    pin.write_text(req["request_sha256"] + "\n")


def init_ee(ee, req: dict, budget) -> dict:
    ee.data.setMaxRetries(0)
    budget.call("initialize (may issue >1 HTTP request)", lambda: ee.Initialize(project=req["project"]))
    ee.data.setMaxRetries(0)
    ee.data.setDeadline(int(req["limits"]["deadline_ms"]))
    got = budget.call("projection equivalence check", lambda: epc.projection_check(ee, req["projection_check"]))
    worst = max(max(abs(got[str(k)][0] - p[2]), abs(got[str(k)][1] - p[3])) for k, p in enumerate(req["projection_check"]))
    if worst > 1e-3:
        raise Stop(f"EE interprets the transport WKT differently (max diff {worst} m); nothing submitted")
    return {"projection_check_max_abs_diff_m": worst}


def submit(req: dict, out: Path, ee) -> dict:
    pin_output(out, req)
    ledger = read_ledger(out)
    seen = {r["task"] for r in ledger}
    if seen:
        raise Stop(f"ledger already holds records for {sorted(seen)}: no task is started twice (inspect/lookup instead)")
    budget = budget_for(out, req)
    summary = init_ee(ee, req, budget)
    built = []
    for task in req["tasks"]:  # every body is measured BEFORE any start
        coll = build_collection(ee, req, task)
        n = body_bytes(ee, coll)
        summary[f"body_bytes_{task['name']}"] = n
        if n > int(req["limits"]["max_body_bytes"]):
            raise Stop(f"task {task['name']} serialized body {n} bytes exceeds {req['limits']['max_body_bytes']}; nothing started")
        built.append((task, coll, n))
    for task, coll, n in built:
        desc = description(req, task)
        append_ledger(out, {"task": task["name"], "description": desc, "state": "INTENT", "body_bytes": n,
                            "prefix": task_prefix(req, task)})
        t = export_task(ee, req, task, coll)
        try:
            budget.call(f"start {desc}", t.start)
        except Stop as stop:
            append_ledger(out, {"task": task["name"], "description": desc, "state": "AMBIGUOUS", "error": str(stop)[:500]})
            raise Stop(f"start of {desc} failed or is ambiguous: HOLD; do not resubmit; inspect with lookup") from stop
        append_ledger(out, {"task": task["name"], "description": desc, "state": "SUBMITTED", "task_id": t.id,
                            "submitted_epoch_s": time.time()})
    summary["calls_used"] = budget.used
    return summary


def monitor(req: dict, out: Path, ee, sleep=time.sleep, now=time.time) -> dict:
    lim = req["limits"]
    ledger = read_ledger(out)
    if any(r["state"] == "AMBIGUOUS" for r in ledger):
        raise Stop("an AMBIGUOUS start is unresolved: HOLD (lookup first)")
    subs = {r["task"]: r for r in ledger if r["state"] == "SUBMITTED"}
    if len(subs) != len(req["tasks"]):
        raise Stop(f"not every task is SUBMITTED ({sorted(subs)})")
    ids = [subs[t["name"]]["task_id"] for t in req["tasks"]]
    first = min(r["submitted_epoch_s"] for r in subs.values())
    budget = budget_for(out, req)
    ee.data.setMaxRetries(0)
    budget.call("initialize (may issue >1 HTTP request)", lambda: ee.Initialize(project=req["project"]))
    reads = sum(1 for _ in open(out / "status.jsonl")) if (out / "status.jsonl").exists() else 0
    cancels = {r["task_id"] for r in ledger if r["state"] == "CANCEL_REQUESTED"}
    last = {}
    while reads < int(lim["max_status_reads"]):
        got = budget.call("grouped getTaskStatus", lambda: ee.data.getTaskStatus(ids))
        reads += 1
        last = {s["id"]: s for s in got}
        with open(out / "status.jsonl", "a", encoding="utf-8") as h:
            h.write(json.dumps({"epoch_s": now(), "statuses": got}) + "\n")
        if all(last.get(i, {}).get("state") in TERMINAL for i in ids):
            break
        if now() - first >= float(lim["cancel_after_s"]):
            for i in ids:
                if last.get(i, {}).get("state") not in TERMINAL and i not in cancels and len(cancels) < int(lim["max_cancels"]):
                    budget.call(f"cancelTask {i}", lambda i=i: ee.data.cancelTask(i))
                    cancels.add(i)
                    append_ledger(out, {"task": next(t for t, r in subs.items() if r["task_id"] == i), "task_id": i,
                                        "state": "CANCEL_REQUESTED", "after_s": round(now() - first, 1)})
        sleep(float(lim["status_read_interval_s"]))
    return {"reads": reads, "final_states": {i: last.get(i, {}).get("state") for i in ids}, "cancel_requested": sorted(cancels),
            "calls_used": budget.used}


def lookup(req: dict, out: Path, ee) -> list:
    budget = budget_for(out, req)
    ee.data.setMaxRetries(0)
    budget.call("initialize (may issue >1 HTTP request)", lambda: ee.Initialize(project=req["project"]))
    wanted = {description(req, t) for t in req["tasks"]}
    found = [s for s in budget.call("getTaskList", ee.data.getTaskList) if s.get("description") in wanted]
    (out / "lookup.json").write_text(json.dumps(found, indent=1) + "\n")
    return found


def download(req: dict, out: Path, storage) -> list:
    """``storage``: a googleapiclient Storage v1 service. Bounded reads; every object under each task prefix."""
    reads, rows = 0, []
    dest = out / "downloads"
    dest.mkdir(parents=True, exist_ok=True)
    for task in req["tasks"]:
        prefix = task_prefix(req, task) + "/"
        token = None
        while True:
            if reads >= int(req["limits"]["max_gcs_reads"]):
                raise Stop("GCS read cap reached")
            page = storage.objects().list(bucket=req["destination"]["bucket"], prefix=prefix, pageToken=token).execute()
            reads += 1
            for obj in page.get("items", []):
                if reads >= int(req["limits"]["max_gcs_reads"]):
                    raise Stop("GCS read cap reached")
                data = storage.objects().get_media(bucket=req["destination"]["bucket"], object=obj["name"]).execute()
                reads += 1
                local = dest / task["name"] / obj["name"][len(prefix):].replace("/", "__")
                local.parent.mkdir(parents=True, exist_ok=True)
                if local.exists():
                    raise Stop(f"{local} already downloaded; not overwritten")
                local.write_bytes(data)
                rows.append({"task": task["name"], "object": obj["name"], "generation": obj.get("generation"),
                             "size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "local": str(local)})
            token = page.get("nextPageToken")
            if not token:
                break
    (out / "downloads.json").write_text(json.dumps({"reads": reads, "objects": rows}, indent=1) + "\n")
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("submit", "monitor", "lookup", "download"))
    p.add_argument("--request", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    req = load_released(Path(a.request))
    out = Path(a.out)
    pin_output(out, req)
    import ee  # noqa: PLC0415

    try:
        if a.command == "submit":
            result = submit(req, out, ee)
        elif a.command == "monitor":
            result = monitor(req, out, ee)
        elif a.command == "lookup":
            result = lookup(req, out, ee)
        else:
            from googleapiclient.discovery import build  # noqa: PLC0415

            budget = budget_for(out, req)
            budget.call("initialize (may issue >1 HTTP request)", lambda: ee.Initialize(project=req["project"]))
            result = download(req, out, build("storage", "v1", credentials=ee.data.get_persistent_credentials(), cache_discovery=False))
    except Stop as stop:
        result = {"stopped": str(stop)}
    with open(out / "run_summaries.jsonl", "a", encoding="utf-8") as h:
        h.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "command": a.command, "result": result}, default=str) + "\n")
    print(json.dumps(result, default=str)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
