"""D30 batch-export speed pilot (EE interpreter only; NOT RUN without an explicit release_id).

Same science as ``ee_pilot_c``: the existing D30 ``composite`` (mask before the descending mosaic) and ONE
``reduceRegion(Reducer.sum())`` per item over [valid_area, S x64, Q x64] on the pinned EPSG:6933 10 m lattice (WKT
transport), bestEffort False, one constant maxPixels per candidate group. Items of a task are grouped by their ascending
candidate list; the grouped collections are flattened into one ``Export.table.toCloudStorage`` CSV task.

Subcommands (one released manifest, one output directory, one cumulative EE call cap):
  submit    measure every task's serialized collection DAG after Initialize and refuse ALL starts if any exceeds the cap
            (the export request adds a small wrapper; the 8 MB cap leaves headroom to EE's 10 MB limit); then per task
            write a ledger INTENT record BEFORE ``start()``. A ``start()`` error marks the task AMBIGUOUS: HOLD, never
            resubmitted; its id is unknown, so it is inspected and, if it exists, cancelled by hand (EE Tasks page).
  monitor   counted per-ID ``getTaskStatus(id)`` reads (one HTTP GET each; the SDK loops GETs for a list) every
            ``status_read_interval_s``. Before every round: past ``cancel_after_s`` from the first submission, cancel
            this manifest's own known nonterminal tasks (UNKNOWN counts as nonterminal); if the remaining read/call
            budget cannot cover another round plus the reserved cancels, or ``max_failed_rounds`` consecutive rounds
            fail, cancel own nonterminal tasks now and HOLD with a handoff record. Known ids are monitored even when
            another task is AMBIGUOUS.
  download  bounded, cumulatively counted GCS JSON-API reads: list every object under each task's prefix (several
            objects possible); each object is written to a temp file, its generation/size/sha256 recorded, then
            renamed; a rerun verifies recorded objects and skips them without overwriting.
Every EE and GCS client uses an explicit socket timeout (``http_timeout_s``; the SDK default deadline 0 means none);
SDK retries 0; no client resubmission. EE's own server-side task retries are recorded by the status reads only.
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
REQUIRED_LIMITS = ("tile_scale", "deadline_ms", "http_timeout_s", "max_tasks", "max_body_bytes", "cancel_after_s",
                   "status_read_interval_s", "max_status_reads", "max_cancels", "max_failed_rounds", "max_gcs_reads",
                   "max_ee_calls", "max_pixels")


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
    missing = [k for k in REQUIRED_LIMITS if k not in req.get("limits", {})]
    if missing:
        raise SystemExit(f"batch manifest limits lack {missing}")
    if not 1 <= len(req["tasks"]) <= int(req["limits"]["max_tasks"]):
        raise SystemExit("task count outside the released max_tasks")
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


def collection_dag_bytes(ee, collection) -> int:
    """Serialized collection DAG (the export request wraps it with a few hundred bytes of task configuration)."""
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


def transport(req: dict):
    """Socket-timeout transport for every EE/GCS client (the SDK's default deadline 0 means no timeout at all)."""
    import httplib2  # noqa: PLC0415

    return httplib2.Http(timeout=float(req["limits"]["http_timeout_s"]))


def connect(ee, req: dict, budget, make_transport=transport) -> None:
    ee.data.setMaxRetries(0)
    budget.call("initialize (may issue >1 HTTP request)",
                lambda: ee.Initialize(project=req["project"], http_transport=make_transport(req)))
    ee.data.setMaxRetries(0)
    ee.data.setDeadline(int(req["limits"]["deadline_ms"]))


def submit(req: dict, out: Path, ee, make_transport=transport) -> dict:
    pin_output(out, req)
    seen = {r["task"] for r in read_ledger(out)}
    if seen:
        raise Stop(f"ledger already holds records for {sorted(seen)}: no task is started twice")
    budget = budget_for(out, req)
    connect(ee, req, budget, make_transport)
    got = budget.call("projection equivalence check", lambda: epc.projection_check(ee, req["projection_check"]))
    worst = max(max(abs(got[str(k)][0] - p[2]), abs(got[str(k)][1] - p[3])) for k, p in enumerate(req["projection_check"]))
    if worst > 1e-3:
        raise Stop(f"EE interprets the transport WKT differently (max diff {worst} m); nothing submitted")
    summary = {"projection_check_max_abs_diff_m": worst}
    built = []
    for task in req["tasks"]:  # every DAG is measured BEFORE any start
        coll = build_collection(ee, req, task)
        n = collection_dag_bytes(ee, coll)
        summary[f"collection_dag_bytes_{task['name']}"] = n
        if n > int(req["limits"]["max_body_bytes"]):
            raise Stop(f"task {task['name']} serialized collection DAG {n} bytes exceeds {req['limits']['max_body_bytes']}; nothing started")
        built.append((task, coll, n))
    for task, coll, n in built:
        desc = description(req, task)
        append_ledger(out, {"task": task["name"], "description": desc, "state": "INTENT", "collection_dag_bytes": n,
                            "prefix": task_prefix(req, task)})
        t = export_task(ee, req, task, coll)
        try:
            budget.call(f"start {desc}", t.start)
        except Stop as stop:
            append_ledger(out, {"task": task["name"], "description": desc, "state": "AMBIGUOUS", "error": str(stop)[:500]})
            raise Stop(f"start of {desc} failed or is ambiguous: HOLD; never resubmitted; known started ids remain "
                       "monitored/cancellable by monitor; inspect this one by hand") from stop
        append_ledger(out, {"task": task["name"], "description": desc, "state": "SUBMITTED", "task_id": t.id,
                            "submitted_epoch_s": time.time()})
    summary["calls_used"] = budget.used
    return summary


def monitor(req: dict, out: Path, ee, sleep=time.sleep, now=time.time, make_transport=transport) -> dict:
    lim = req["limits"]
    ledger = read_ledger(out)
    known = {r["task_id"]: r for r in ledger if r["state"] == "SUBMITTED"}
    ambiguous = sorted(r["task"] for r in ledger if r["state"] == "AMBIGUOUS")
    if not known:
        raise Stop(f"no known submitted task id (ambiguous: {ambiguous}): HOLD; inspect/cancel by hand")
    ids = sorted(known)
    first = min(r["submitted_epoch_s"] for r in known.values())
    budget = budget_for(out, req)
    connect(ee, req, budget, make_transport)
    status_log = out / "status.jsonl"
    reads = sum(1 for _ in open(status_log)) if status_log.exists() else 0
    attempted = {r["task_id"] for r in ledger if r["state"] == "CANCEL_ATTEMPT"}  # durable: never reissued, even after restart
    cancels = {r["task_id"] for r in ledger if r["state"] == "CANCEL_REQUESTED"}  # attempts that succeeded
    state, failed_rounds, handoff = {}, 0, None

    def nonterminal():
        return [i for i in ids if state.get(i) not in TERMINAL]

    def cancel_own(reason):
        for i in nonterminal():
            if i in attempted or len(attempted) >= int(lim["max_cancels"]):
                continue
            attempted.add(i)  # reserved and recorded BEFORE the call; a failed/ambiguous cancel is never reissued
            append_ledger(out, {"task": known[i]["task"], "task_id": i, "state": "CANCEL_ATTEMPT",
                                "after_s": round(now() - first, 1), "reason": reason})
            try:
                budget.call(f"cancelTask {i}", lambda i=i: ee.data.cancelTask(i))
                cancels.add(i)
                append_ledger(out, {"task": known[i]["task"], "task_id": i, "state": "CANCEL_REQUESTED"})
            except Stop as stop:
                append_ledger(out, {"task": known[i]["task"], "task_id": i, "state": "CANCEL_FAILED", "error": str(stop)[:300]})

    def unresolved():
        return [i for i in nonterminal() if i in attempted and i not in cancels]

    while True:
        if now() - first >= float(lim["cancel_after_s"]):
            cancel_own("cancel_after_s reached")
        todo = nonterminal()
        if not todo:
            break
        if set(todo) <= set(unresolved()):  # only tasks whose cancel failed remain: polling cannot resolve them
            handoff = "cancellation failed or ambiguous for the remaining own tasks"
            break
        reserve = int(lim["max_cancels"]) - len(attempted)
        if int(lim["max_status_reads"]) - reads < len(todo) or budget.cap - budget.used - len(todo) < reserve:
            cancel_own("read/call budget cannot cover another round plus reserved cancels")
            handoff = "status budget exhausted before every own task was terminal"
            break
        round_ok = True
        for i in todo:
            try:
                got = budget.call(f"getTaskStatus {i}", lambda i=i: ee.data.getTaskStatus(i))[0]
                state[i] = got.get("state")
                rec = {"epoch_s": now(), "id": i, "status": got}
            except Stop as stop:
                round_ok = False
                rec = {"epoch_s": now(), "id": i, "error": str(stop)[:300]}
            reads += 1
            with open(status_log, "a", encoding="utf-8") as h:
                h.write(json.dumps(rec) + "\n")
        failed_rounds = 0 if round_ok else failed_rounds + 1
        if failed_rounds >= int(lim["max_failed_rounds"]):
            cancel_own(f"{failed_rounds} consecutive failed status rounds")
            handoff = "status reads failing; own nonterminal tasks cancel-requested"
            break
        if nonterminal():
            sleep(float(lim["status_read_interval_s"]))
    if handoff or ambiguous or unresolved():
        append_ledger(out, {"task": "*", "state": "HOLD_HANDOFF", "reason": handoff or "ambiguous start or failed cancellation unresolved",
                            "ambiguous": ambiguous, "nonterminal": nonterminal(), "cancel_attempted": sorted(attempted),
                            "cancel_requested": sorted(cancels), "cancel_unresolved": unresolved()})
    return {"reads": reads, "final_states": {i: state.get(i) for i in ids}, "cancel_attempted": sorted(attempted),
            "cancel_requested": sorted(cancels), "cancel_unresolved": unresolved(), "ambiguous": ambiguous, "handoff": handoff,
            "calls_used": budget.used}


def download(req: dict, out: Path, storage) -> list:
    """``storage``: a googleapiclient Storage v1 service on a socket-timeout transport. Reads are counted cumulatively
    in ``gcs_reads.log`` across reruns; each object's record is appended BEFORE its file is renamed into place, so a
    partial run never leaves a file without provenance and a rerun skips verified objects without overwriting."""
    dest = out / "downloads"
    dest.mkdir(parents=True, exist_ok=True)
    log, records_path = out / "gcs_reads.log", out / "downloads.jsonl"
    records = {r["object"]: r for r in (json.loads(x) for x in records_path.read_text().splitlines())} if records_path.exists() else {}

    def read(purpose, fn):
        used = sum(1 for _ in open(log)) if log.exists() else 0
        if used >= int(req["limits"]["max_gcs_reads"]):
            raise Stop("GCS read cap reached")
        with open(log, "a", encoding="utf-8") as h:
            h.write(purpose[:200] + "\n")
        return fn()

    rows = []
    for task in req["tasks"]:
        prefix = task_prefix(req, task) + "/"
        token = None
        while True:
            page = read(f"list {prefix}", lambda: storage.objects().list(bucket=req["destination"]["bucket"], prefix=prefix,
                                                                     pageToken=token).execute())
            for obj in page.get("items", []):
                local = dest / task["name"] / obj["name"][len(prefix):].replace("/", "__")
                tmp = local.with_name(local.name + ".part")
                rec = records.get(obj["name"])
                if rec and rec.get("generation") == obj.get("generation"):
                    if local.exists() and hashlib.sha256(local.read_bytes()).hexdigest() == rec["sha256"]:
                        rows.append(rec)
                        continue
                    if tmp.exists() and hashlib.sha256(tmp.read_bytes()).hexdigest() == rec["sha256"] and not local.exists():
                        tmp.replace(local)
                        rows.append(rec)
                        continue
                    raise Stop(f"{obj['name']}: recorded download missing or altered locally; not overwritten")
                if local.exists() or rec:
                    raise Stop(f"{local}: exists without a matching record, or a new generation appeared; not overwritten")
                data = read(f"get {obj['name']}", lambda: storage.objects().get_media(bucket=req["destination"]["bucket"],
                                                                                      object=obj["name"]).execute())
                local.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_bytes(data)
                rec = {"task": task["name"], "object": obj["name"], "generation": obj.get("generation"), "size": len(data),
                       "sha256": hashlib.sha256(data).hexdigest(), "local": str(local)}
                with open(records_path, "a", encoding="utf-8") as h:
                    h.write(json.dumps(rec) + "\n")
                    h.flush()
                tmp.replace(local)
                records[obj["name"]] = rec
                rows.append(rec)
            token = page.get("nextPageToken")
            if not token:
                break
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("submit", "monitor", "download"))
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
        else:
            import google_auth_httplib2  # noqa: PLC0415
            from googleapiclient.discovery import build  # noqa: PLC0415

            connect(ee, req, budget_for(out, req))
            http = google_auth_httplib2.AuthorizedHttp(ee.data.get_persistent_credentials(), http=transport(req))
            storage = build("storage", "v1", http=http, cache_discovery=False)  # discovery GET uses the same timeout
            result = {"objects": len(download(req, out, storage))}
    except Stop as stop:
        result = {"stopped": str(stop)}
    with open(out / "run_summaries.jsonl", "a", encoding="utf-8") as h:
        h.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "command": a.command, "result": result}, default=str) + "\n")
    print(json.dumps(result, default=str)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
