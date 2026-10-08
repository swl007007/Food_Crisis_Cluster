"""Finite, idempotent import of the six completed IPCCH runs into local MLflow.

  plan    read-only: hash every source file, extract evaluation views, write plans
  import  create/resume parent source_run + child evaluation_view records
  verify  read back every metric/param/tag and download+hash every artifact
  reconcile  one-time additive reconciliation of imported records to a changed plan
             (refuses value changes; superseded manifests/views kept; nothing deleted)

Records are keyed by a stable ``source_key`` tag; ``import_fingerprint`` binds
the exact inputs. Same fingerprint -> verified no-op; different -> conflict
(stop). An interrupted record (import_status=in_progress) resumes against the
same fingerprint. Originals are only read.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract  # noqa: E402
import inventory  # noqa: E402
from extract import SourceConflict  # noqa: E402

IMPORTER_VERSION = "ipcch-mlflow-import-v1"
DEFAULT_ROOT = Path("/home/swl007007/.local/share/ipcch-mlflow")
METRIC_NAME = re.compile(r"^[\w\-. /]{1,250}$")
CHUNK = 1 << 20


# ------------------------------------------------------------------ helpers

def canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode()


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def glob_re(pattern: str) -> re.Pattern:
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**", i):
            out += ".*"
            i += 2
        elif pattern[i] == "*":
            out += "[^/]*"
            i += 1
        elif pattern[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(pattern[i])
            i += 1
    return re.compile(out + r"\Z")


def matches(rel: str, patterns) -> str | None:
    for p in patterns:
        if glob_re(p).match(rel):
            return p
    return None


class Hasher:
    """sha256 with a (path, size, mtime_ns) cache; ``rehash`` ignores the cache."""

    def __init__(self, cache_path: Path | None, rehash: bool = False):
        self.path, self.rehash, self.hits, self.misses = cache_path, rehash, 0, 0
        self.cache = {}
        if cache_path and cache_path.is_file() and not rehash:
            self.cache = json.loads(cache_path.read_text())

    def __call__(self, p: Path) -> tuple[int, str]:
        st = p.stat()
        key = f"{p}|{st.st_size}|{st.st_mtime_ns}"
        if key in self.cache:
            self.hits += 1
            return st.st_size, self.cache[key]
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(CHUNK), b""):
                h.update(b)
        self.misses += 1
        self.cache[key] = h.hexdigest()
        return st.st_size, self.cache[key]

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.cache))
            os.replace(tmp, self.path)


def resolve(cfg: dict, s: str) -> Path:
    return Path(s.replace("{repo}", cfg["repo_root"]).replace("{temp}", cfg["temp_root"]))


# ------------------------------------------------------------------ planning

def plan_family(cfg: dict, fam: dict, hasher: Hasher, plans: dict) -> dict:
    root = resolve(cfg, fam["root"])
    if not root.is_dir():
        raise SourceConflict(f"{fam['family']}: source root missing: {root}")
    include, bundle, excluded, unlisted = [], {name: [] for name in fam.get("bundle", {})}, [], []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            p = Path(dirpath) / fn
            rel = p.relative_to(root).as_posix()
            ex = matches(rel, fam.get("exclude", {}))
            size, digest = hasher(p)
            rec = {"path": rel, "bytes": size, "sha256": digest}
            if ex:
                excluded.append({**rec, "reason": fam["exclude"][ex], "source": str(p)})
                continue
            for name, pat in fam.get("bundle", {}).items():
                if matches(rel, [pat]):
                    bundle[name].append(rec)
                    break
            else:
                if matches(rel, fam["include"]):
                    include.append(rec)
                else:
                    unlisted.append(rec)
    shared = []
    si = fam.get("shared_inputs")
    if si and si.get("match"):
        parent = plans.get(si["parent"])
        if parent is None:
            raise SourceConflict(f"{fam['family']}: shared-input parent {si['parent']} not planned first")
        by_sha = {}
        for r in parent["include"] + parent["excluded"]:
            by_sha.setdefault(r["sha256"], r["path"])
        keep = []
        for r in include + unlisted:
            if r["path"].startswith(si["match"]):
                if r["sha256"] not in by_sha:
                    raise SourceConflict(f"{fam['family']}: staged input {r['path']} has no identical file in {si['parent']}")
                shared.append({**r, "parent": si["parent"], "parent_path": by_sha[r["sha256"]]})
            else:
                keep.append(r)
        include = [r for r in keep if matches(r["path"], fam["include"])]
        unlisted = [r for r in keep if not matches(r["path"], fam["include"])]
    extras = []
    for ex in fam.get("extra", []):
        base = resolve(cfg, ex["base"])
        seen = set()
        for pat in ex["paths"]:
            anchor = base / pat.split("*")[0].rsplit("/", 1)[0]
            if not anchor.exists():
                raise SourceConflict(f"{fam['family']}: extra path anchor missing: {anchor}")
            for dirpath, _dirs, files in os.walk(anchor):
                if "__pycache__" in dirpath:
                    continue
                for fn in files:
                    p = Path(dirpath) / fn
                    rel = p.relative_to(base).as_posix()
                    if rel in seen or not matches(rel, [pat]):
                        continue
                    seen.add(rel)
                    size, digest = hasher(p)
                    extras.append({"path": f"{ex['prefix']}/{rel}", "bytes": size, "sha256": digest, "source": str(p)})
    inv = inventory.reconcile(fam, root, {"include": include, "bundle": bundle, "excluded": excluded,
                                          "shared": shared}, plans)
    children, parent_metrics, report_sha = extract.extract(fam, root)
    for c in children:
        for k in c.metrics:
            if not METRIC_NAME.match(k):
                raise SourceConflict(f"{c.key}: metric name not MLflow-safe: {k}")
    keys = [c.key for c in children]
    if len(set(keys)) != len(keys):
        raise SourceConflict(f"{fam['family']}: duplicate evaluation-view keys")
    views = [view_record(fam, c) for c in children]
    body = {
        "family": fam["family"], "source_run_id": fam["source_run_id"], "root": str(root),
        "config": fam, "report_sha256": report_sha,
        "include": sorted(include, key=lambda r: r["path"]),
        "bundle": {k: sorted(v, key=lambda r: r["path"]) for k, v in bundle.items()},
        "excluded": sorted(excluded, key=lambda r: r["path"]),
        "shared": sorted(shared, key=lambda r: r["path"]),
        "extras": sorted(extras, key=lambda r: r["path"]),
        "unlisted": sorted(unlisted, key=lambda r: r["path"]),
        "parent_metrics": parent_metrics,
        "inventory": inv,
        "children": [{"key": v["key"], "fingerprint": v["fingerprint"]} for v in views],
    }
    body["fingerprint"] = sha_bytes(canon({k: body[k] for k in (
        "family", "source_run_id", "config", "report_sha256", "include", "bundle", "excluded", "shared",
        "extras", "parent_metrics", "inventory", "children")} | {"importer": IMPORTER_VERSION}))
    body["views"] = views
    body["totals"] = {
        "include": [len(include), sum(r["bytes"] for r in include)],
        "bundle": {k: [len(v), sum(r["bytes"] for r in v)] for k, v in bundle.items()},
        "excluded": [len(excluded), sum(r["bytes"] for r in excluded)],
        "shared": [len(shared), sum(r["bytes"] for r in shared)],
        "extras": [len(extras), sum(r["bytes"] for r in extras)],
        "unlisted": [len(unlisted), sum(r["bytes"] for r in unlisted)],
        "children": len(children),
        "child_metrics": sum(len(v["metrics"]) for v in views),
        "child_na": sum(len(v["na"]) for v in views),
    }
    return body


def view_record(fam: dict, c) -> dict:
    tags = {**{f"source.{k}": str(v) for k, v in fam["tags"].items()}, **c.tags,
            "record_kind": "evaluation_view", "source_key": c.key,
            "parent_source_key": f"{fam['family']}/{fam['source_run_id']}",
            "family": fam["family"], "horizon": c.H, "arm": c.arm, "seed": c.seed, "arm_kind": c.arm_kind}
    params = {"family": fam["family"], "source_run_id": fam["source_run_id"], "horizon_months": c.H,
              "arm": c.arm, "seed": c.seed, "arm_kind": c.arm_kind}
    doc = {"key": c.key, "metrics": c.metrics, "provenance": c.provenance, "na": c.na, "panels": c.panels,
           "params": params, "tags": tags}
    view_json = json.dumps(doc, sort_keys=True, indent=1, default=str).encode()
    fp = sha_bytes(canon({"metrics": c.metrics, "params": params, "tags": tags, "na": c.na,
                          "view_sha256": sha_bytes(view_json), "importer": IMPORTER_VERSION}))
    return {"key": c.key, "H": c.H, "arm": c.arm, "seed": c.seed, "metrics": c.metrics, "na": c.na,
            "params": params, "tags": tags, "view_json": view_json.decode(), "fingerprint": fp}


def family_dependencies(fam: dict) -> list:
    """Families this one reads during planning: shared inputs and inventory reference parents."""
    deps = [(fam.get("shared_inputs") or {}).get("parent")] + list((fam.get("inventory") or {}).get("reference_parents", []))
    return [d for d in dict.fromkeys(deps) if d]


def planning_order(cfg: dict, families=None) -> list:
    """Selected families plus their transitive read dependencies, dependencies first.
    Dependencies are planned only; writes and verification stay limited to the selection."""
    by_name = {f["family"]: f for f in cfg["families"]}
    for name in families or []:
        if name not in by_name:
            raise SourceConflict(f"unknown family {name}")
    order, seen = [], set()

    def visit(name: str, stack: tuple) -> None:
        if name in stack:
            raise SourceConflict(f"dependency cycle: {' -> '.join(stack + (name,))}")
        if name in seen:
            return
        if name not in by_name:
            raise SourceConflict(f"{stack[-1]} depends on undeclared family {name}")
        for d in family_dependencies(by_name[name]):
            visit(d, stack + (name,))
        seen.add(name)
        order.append(by_name[name])

    for f in cfg["families"]:
        if not families or f["family"] in families:
            visit(f["family"], ())
    return order


def cmd_plan(cfg: dict, store: Path, evidence: Path | None, rehash: bool, families=None) -> dict:
    hasher = Hasher(store / "cache" / "hashes.json", rehash)
    plans = {}
    try:
        for fam in planning_order(cfg, families):
            t0 = time.time()
            plans[fam["family"]] = plan_family(cfg, fam, hasher, plans)
            print(f"planned {fam['family']}: {json.dumps(plans[fam['family']]['totals'])} ({time.time() - t0:.0f}s)", flush=True)
    finally:
        hasher.save()
    selected = [k for k in plans if not families or k in families]
    summary = {"importer": IMPORTER_VERSION, "families": {k: {"fingerprint": v["fingerprint"], "totals": v["totals"],
                                                              "role": "selected" if k in selected else "read dependency"}
                                                          for k, v in plans.items()},
               "records": sum(1 + plans[k]["totals"]["children"] for k in selected),
               "hash_cache": {"hits": hasher.hits, "misses": hasher.misses, "rehash": rehash}}
    full = store / "plans"
    full.mkdir(parents=True, exist_ok=True)
    for k, v in plans.items():
        slim = {kk: vv for kk, vv in v.items() if kk != "views"}
        slim["views"] = [{kk: vv for kk, vv in x.items() if kk != "view_json"} for x in v["views"]]
        (full / f"plan-{k}.json").write_text(json.dumps(slim, indent=1, sort_keys=True))
    (full / "plan-summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    if evidence:  # compact: totals, exclusions, shared references, unlisted files, per-view counts
        evidence.mkdir(parents=True, exist_ok=True)
        for k, v in plans.items():
            compact = {kk: v[kk] for kk in ("family", "source_run_id", "root", "report_sha256", "fingerprint",
                                             "totals", "excluded", "shared", "unlisted", "inventory")}
            compact["include_manifest_sha256"] = sha_bytes(canon(v["include"]))
            compact["bundle_manifest_sha256"] = {n: sha_bytes(canon(m)) for n, m in v["bundle"].items()}
            compact["extras_manifest_sha256"] = sha_bytes(canon(v["extras"]))
            compact["views"] = [{"key": x["key"], "fingerprint": x["fingerprint"], "metrics": len(x["metrics"]),
                                 "na": len(x["na"])} for x in v["views"]]
            (evidence / f"plan-{k}.json").write_text(json.dumps(compact, indent=1, sort_keys=True))
        (evidence / "plan-summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    return plans


# ------------------------------------------------------------------ tar bundle

class _HashingReader(io.RawIOBase):
    def __init__(self, f):
        self.f, self.h = f, hashlib.sha256()

    def readable(self):
        return True

    def readinto(self, b):
        n = self.f.readinto(b)
        if n:
            self.h.update(memoryview(b)[:n])
        return n


def build_tar(root: Path, members: list, out: Path) -> dict:
    with tarfile.open(out, "w", format=tarfile.PAX_FORMAT) as tar:
        for m in members:
            p = root / m["path"]
            ti = tarfile.TarInfo(m["path"])
            ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = m["bytes"], 0, 0o644, 0, 0, "", ""
            with open(p, "rb") as f:
                hr = _HashingReader(f)
                tar.addfile(ti, io.BufferedReader(hr, CHUNK))
            if hr.h.hexdigest() != m["sha256"]:
                raise SourceConflict(f"bundle member changed since planning: {p}")
    size = out.stat().st_size
    h = hashlib.sha256()
    with open(out, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return {"bytes": size, "sha256": h.hexdigest()}


def tar_member_check(path: Path, members: list) -> int:
    want = {m["path"]: m["sha256"] for m in members}
    seen = 0
    with tarfile.open(path, "r") as tar:
        for ti in tar:
            f = tar.extractfile(ti)
            h = hashlib.sha256()
            for b in iter(lambda: f.read(CHUNK), b""):
                h.update(b)
            if want.get(ti.name) != h.hexdigest():
                raise SourceConflict(f"{path}: tar member {ti.name} checksum mismatch")
            seen += 1
    if seen != len(want):
        raise SourceConflict(f"{path}: tar has {seen} members, manifest {len(want)}")
    return seen


# ------------------------------------------------------------------ MLflow writes

def _client(uri: str):
    from mlflow.tracking import MlflowClient
    return MlflowClient(tracking_uri=uri)


def get_experiment(client, name: str, artifact_location: str | None):
    exp = client.get_experiment_by_name(name)
    if exp is None:
        eid = client.create_experiment(name, artifact_location=artifact_location)
        exp = client.get_experiment(eid)
    return exp


def find_run(client, exp_id: str, source_key: str):
    runs = client.search_runs([exp_id], filter_string=f"tags.source_key = '{source_key}'", max_results=5)
    if len(runs) > 1:
        raise SourceConflict(f"{source_key}: {len(runs)} MLflow records share this source key")
    return runs[0] if runs else None


def log_values(client, run, metrics: dict, params: dict, tags: dict) -> int:
    from mlflow.entities import Metric, Param, RunTag
    have_m = run.data.metrics if run else {}
    have_p = run.data.params if run else {}
    for k, v in params.items():
        if k in have_p and have_p[k] != str(v):
            raise SourceConflict(f"param {k} already logged with a different value")
    ts = int(time.time() * 1000)
    ms = [Metric(k, float(v), ts, 0) for k, v in sorted(metrics.items()) if have_m.get(k) != float(v)]
    ps = [Param(k, str(v)) for k, v in sorted(params.items()) if k not in have_p]
    tg = [RunTag(k, str(v)) for k, v in sorted(tags.items())]
    rid = run.info.run_id
    for i in range(0, len(ms), 1000):
        client.log_batch(rid, metrics=ms[i:i + 1000])
    for i in range(0, len(ps), 100):
        client.log_batch(rid, params=ps[i:i + 100])
    for i in range(0, len(tg), 100):
        client.log_batch(rid, tags=tg[i:i + 100])
    return len(ms)


def existing_artifacts(client, run_id: str, path: str | None = None) -> dict:
    out = {}
    for fi in client.list_artifacts(run_id, path):
        if fi.is_dir:
            out.update(existing_artifacts(client, run_id, fi.path))
        else:
            out[fi.path] = fi.file_size
    return out


def upload(client, run_id: str, have: dict, art_path: str, data: bytes | None = None, src: Path | None = None,
           size: int | None = None) -> bool:
    if size is None:
        size = len(data) if data is not None else src.stat().st_size
    if have.get(art_path) == size:
        return False
    d, name = (art_path.rsplit("/", 1) + [""])[:2] if "/" in art_path else ("", art_path)
    with tempfile.TemporaryDirectory(prefix="ipcch-mlflow-up-") as tmp:
        local = Path(tmp) / name
        if data is not None:
            local.write_bytes(data)
        else:
            os.symlink(src, local)
        client.log_artifact(run_id, str(local), d or None)
    return True


class Interrupt(RuntimeError):
    pass


PROVENANCE_TAGS = {
    "execution": "historical import of a completed run; not executed by MLflow",
    "mlflow_timestamps": "MLflow start/metric times are import times, not source execution times",
}


def import_family(client, exp, plan: dict, store: Path, run_ids: dict, fail_after: str | None = None) -> dict:
    fam = plan["config"]
    root = Path(plan["root"])
    skey = f"{fam['family']}/{fam['source_run_id']}"
    stats = {"source_key": skey, "created": 0, "resumed": 0, "noop": 0, "metrics_logged": 0, "artifacts_uploaded": 0}
    run = find_run(client, exp.experiment_id, skey)
    if run is not None:
        fp = run.data.tags.get("import_fingerprint")
        if run.data.tags.get("import_status") == "reconciling":
            raise SourceConflict(f"{skey}: reconciliation to {run.data.tags.get('reconcile_target')} was interrupted; "
                                 "resume it with `reconcile` using that plan -- import stops")
        if fp != plan["fingerprint"]:
            raise SourceConflict(f"{skey}: existing record fingerprint {fp} != planned {plan['fingerprint']}; "
                                 "source or importer changed -- stop for reconciliation")
        if run.data.tags.get("import_status") == "complete":
            run_ids[fam["family"]] = run.info.run_id
            verify_parent(client, run, plan, deep=True)        # verified no-op: full content readback
            stats["noop"] += 1
            for v in plan["views"]:
                cr = find_run(client, exp.experiment_id, v["key"])
                if cr is None or cr.data.tags.get("import_status") != "complete":
                    raise SourceConflict(f"{v['key']}: parent complete but child missing/incomplete")
                verify_child(client, cr, v, run.info.run_id, deep=True)
                stats["noop"] += 1
            return stats
        stats["resumed"] += 1
    else:
        run = client.create_run(exp.experiment_id, run_name=f"{fam['family']}:{fam['source_run_id']}",
                                tags={"source_key": skey, "import_fingerprint": plan["fingerprint"],
                                      "import_status": "in_progress", "record_kind": "source_run"})
        run = client.get_run(run.info.run_id)
        stats["created"] += 1
    rid = run.info.run_id
    run_ids[fam["family"]] = rid
    stats["metrics_logged"] += log_values(client, run, plan["parent_metrics"], parent_params(plan),
                                          parent_tags(plan, run_ids))
    have = existing_artifacts(client, rid)
    for r in plan["include"]:
        stats["artifacts_uploaded"] += upload(client, rid, have, f"source/{r['path']}", src=root / r["path"], size=r["bytes"])
    for r in plan["extras"]:
        stats["artifacts_uploaded"] += upload(client, rid, have, r["path"], src=Path(r["source"]), size=r["bytes"])
    if fail_after == "parent-manifests":
        raise Interrupt("injected interruption after parent artifacts")
    tar_info = {}
    for name, members in plan["bundle"].items():
        tags = client.get_run(rid).data.tags
        if have.get(name) is not None and tags.get(f"bundle.{name}.sha256"):
            tar_info[name] = {"bytes": int(tags[f"bundle.{name}.bytes"]), "sha256": tags[f"bundle.{name}.sha256"]}
            continue
        stage = store / "staging"
        stage.mkdir(parents=True, exist_ok=True)
        tmp = stage / f"{rid}-{name}"
        info = build_tar(root, members, tmp)
        try:
            _log_renamed(client, rid, tmp, name)
        finally:
            tmp.unlink(missing_ok=True)
        stats["artifacts_uploaded"] += 1
        client.set_tag(rid, f"bundle.{name}.sha256", info["sha256"])
        client.set_tag(rid, f"bundle.{name}.bytes", str(info["bytes"]))
        tar_info[name] = info
    for path, obj in parent_manifests(plan, tar_info).items():
        data = json.dumps(obj, indent=1, sort_keys=True).encode()
        stats["artifacts_uploaded"] += upload(client, rid, have, path, data=data)
    verify_parent(client, client.get_run(rid), plan, deep=True)   # full readback before any completion
    for i, v in enumerate(plan["views"]):
        if fail_after == "child-3" and i == 3:
            raise Interrupt("injected interruption during children")
        cs = import_child(client, exp, v, rid)
        for k in ("created", "resumed", "noop", "metrics_logged", "artifacts_uploaded"):
            stats[k] += cs[k]
    client.set_tag(rid, "import_status", "complete")
    client.set_terminated(rid, "FINISHED")
    return stats


def parent_params(plan: dict) -> dict:
    fam, t = plan["config"], plan["totals"]
    p = {"family": fam["family"], "source_run_id": fam["source_run_id"], "source_root": plan["root"],
         "report_sha256": plan["report_sha256"], "evaluation_views": t["children"],
         "archived_files": t["include"][0], "archived_bytes": t["include"][1],
         "excluded_files": t["excluded"][0], "excluded_bytes": t["excluded"][1],
         "shared_files": t["shared"][0], "shared_bytes": t["shared"][1],
         "extras_files": t["extras"][0], "extras_bytes": t["extras"][1]}
    for name, (n, b) in t["bundle"].items():
        p[f"bundle.{name}.members"], p[f"bundle.{name}.member_bytes"] = n, b
    return p


def parent_tags(plan: dict, run_ids: dict) -> dict:
    fam = plan["config"]
    tags = {**{f"source.{k}": str(v) for k, v in fam["tags"].items()}, **PROVENANCE_TAGS,
            "source_key": f"{fam['family']}/{fam['source_run_id']}", "import_fingerprint": plan["fingerprint"],
            "import_status": "in_progress", "record_kind": "source_run", "family": fam["family"],
            "importer_version": IMPORTER_VERSION}
    si = fam.get("shared_inputs")
    if si:
        tags["shared_inputs_parent"] = si["parent"]
        tags["shared_inputs_parent_run_id"] = run_ids.get(si["parent"], "")
        tags["shared_inputs_note"] = si["note"]
    return tags


def parent_manifests(plan: dict, tar_info: dict) -> dict:
    m = {"manifests/include.json": plan["include"], "manifests/excluded.json": plan["excluded"],
         "manifests/shared.json": plan["shared"], "manifests/extras.json": plan["extras"],
         "manifests/unlisted.json": plan["unlisted"],
         "manifests/original-inventory-check.json": plan["inventory"],
         "manifests/plan-summary.json": {"fingerprint": plan["fingerprint"], "totals": plan["totals"],
                                         "report_sha256": plan["report_sha256"], "config": plan["config"]}}
    for name, members in plan["bundle"].items():
        m[f"manifests/{name}.members.json"] = members
        m[f"manifests/{name}.tar.json"] = tar_info[name]
    return m


def _log_renamed(client, rid: str, tmp: Path, name: str) -> None:
    with tempfile.TemporaryDirectory(prefix="ipcch-mlflow-tar-") as d:
        link = Path(d) / name
        os.symlink(tmp, link)
        client.log_artifact(rid, str(link), None)


def child_tags(v: dict, parent_rid: str, comparator_rid: str | None) -> dict:
    tags = {**v["tags"], **PROVENANCE_TAGS, "import_fingerprint": v["fingerprint"], "mlflow.parentRunId": parent_rid,
            "importer_version": IMPORTER_VERSION}
    if v["tags"].get("comparator_parent"):
        tags["comparator_parent_run_id"] = comparator_rid or ""
    return tags


def import_child(client, exp, v: dict, parent_rid: str) -> dict:
    st = {"created": 0, "resumed": 0, "noop": 0, "metrics_logged": 0, "artifacts_uploaded": 0}
    run = find_run(client, exp.experiment_id, v["key"])
    if run is not None:
        if run.data.tags.get("import_fingerprint") != v["fingerprint"]:
            raise SourceConflict(f"{v['key']}: existing child fingerprint differs -- stop for reconciliation")
        if run.data.tags.get("mlflow.parentRunId") != parent_rid:
            raise SourceConflict(f"{v['key']}: existing child is linked to a different parent")
        if run.data.tags.get("import_status") == "complete":
            verify_child(client, run, v, parent_rid, deep=True)
            st["noop"] = 1
            return st
        st["resumed"] = 1
    else:
        run = client.create_run(exp.experiment_id, run_name=f"{v['tags']['family']} H{v['H']} {v['arm']} seed{v['seed']}",
                                tags={"source_key": v["key"], "import_fingerprint": v["fingerprint"],
                                      "import_status": "in_progress", "mlflow.parentRunId": parent_rid,
                                      "record_kind": "evaluation_view"})
        run = client.get_run(run.info.run_id)
        st["created"] = 1
    rid = run.info.run_id
    comp = _run_id_for(client, exp, v["tags"]["comparator_parent"]) if v["tags"].get("comparator_parent") else None
    st["metrics_logged"] = log_values(client, run, v["metrics"], v["params"],
                                      {**child_tags(v, parent_rid, comp), "import_status": "in_progress"})
    have = existing_artifacts(client, rid)
    st["artifacts_uploaded"] += upload(client, rid, have, "view/evaluation_view.json", data=v["view_json"].encode())
    st["artifacts_uploaded"] += upload(client, rid, have, "view/na.json",
                                       data=json.dumps(v["na"], indent=1, sort_keys=True).encode())
    verify_child(client, client.get_run(rid), v, parent_rid, deep=True)   # readback before completion
    client.set_tag(rid, "import_status", "complete")
    client.set_terminated(rid, "FINISHED")
    return st


_RID_CACHE: dict = {}


def _run_id_for(client, exp, source_key: str) -> str:
    if source_key not in _RID_CACHE:
        r = find_run(client, exp.experiment_id, source_key)
        if r is None:
            return ""
        _RID_CACHE[source_key] = r.info.run_id
    return _RID_CACHE[source_key]


# ------------------------------------------------------------------ one-time reconciliation

RECONCILE_NOTE = "supervisor review of 8c88f48; fixes a9a2598 + ec2686e"


def _download_bytes(client, rid: str, path: str) -> bytes:
    with tempfile.TemporaryDirectory(prefix="ipcch-mlflow-recon-") as tmp:
        return Path(client.download_artifacts(rid, path, tmp)).read_bytes()


def _supersede(client, rid: str, have: dict, path: str, new: bytes, old16: str, prefix: str) -> dict | None:
    """Keep the current artifact under ``{prefix}superseded/`` before replacing it. Returns a record or None."""
    if path not in have:
        upload(client, rid, have, path, data=new)
        return {"path": path, "action": "added", "new_sha256": sha_bytes(new)}
    old = _download_bytes(client, rid, path)
    if old == new:
        return None
    name = PurePosixPath(path).name
    stem, _, ext = name.partition(".")
    keep = f"{prefix}superseded/{stem}.{old16}.{ext}"
    upload(client, rid, {}, keep, data=old)
    upload(client, rid, {}, path, data=new)
    return {"path": path, "action": "superseded", "kept_as": keep, "old_sha256": sha_bytes(old), "new_sha256": sha_bytes(new)}


def reconcile_family(client, exp, plan: dict, run_ids: dict, fail_after: str | None = None) -> dict:
    """One-time, additive reconciliation of already-imported records to a changed plan.

    Phase 1 is read-only: the parent and every child are validated against the plan and the
    whole family is refused (no record touched) if any logged metric, param or existing tag
    value would change, or an archived source/model artifact differs. Phase 2 writes: it first
    records ``reconcile_target`` (plan fingerprint) and ``import_fingerprint.previous`` on the
    parent, so an interruption leaves an explicit state that only the same plan may resume.
    Replaced manifests/view JSON are kept under ``superseded/``. Nothing is deleted."""
    fam = plan["config"]
    skey = f"{fam['family']}/{fam['source_run_id']}"
    out = {"source_key": skey, "parent": None, "children_reconciled": 0, "children_unchanged": 0,
           "children_not_imported": 0, "artifact_changes": []}
    run = find_run(client, exp.experiment_id, skey)
    if run is None:
        out["parent"] = "not imported"
        return out
    rid, tags = run.info.run_id, run.data.tags
    run_ids[fam["family"]] = rid
    status, cur_fp, target = tags.get("import_status"), tags.get("import_fingerprint"), tags.get("reconcile_target")

    # ---- phase 1: validate everything, write nothing
    if status == "reconciling":
        if target != plan["fingerprint"]:
            raise SourceConflict(f"{skey}: interrupted reconciliation targets {target}; only that plan may resume "
                                 f"(this plan is {plan['fingerprint']}) -- stop")
        old_fp, mode = tags.get("import_fingerprint.previous"), "resume"
    elif status == "complete" and cur_fp == plan["fingerprint"]:
        old_fp, mode = cur_fp, "unchanged"
    elif status == "in_progress" and not run.data.metrics and not run.data.params and not existing_artifacts(client, rid):
        old_fp, mode = cur_fp, "empty shell"
    elif status == "complete":
        old_fp, mode = cur_fp, "reconcile"
    else:
        raise SourceConflict(f"{skey}: parent in state {status} with different content -- stop")
    tar_info, have = {}, {}
    if mode in ("reconcile", "resume"):
        if run.data.metrics != plan["parent_metrics"]:
            raise SourceConflict(f"{skey}: parent metrics would change -- refuse")
        for k, v in parent_params(plan).items():
            if run.data.params.get(k) != str(v):
                raise SourceConflict(f"{skey}: param {k} would change -- refuse")
        for k, v in parent_tags(plan, run_ids).items():
            if k in ("import_status", "import_fingerprint", "shared_inputs_parent_run_id"):
                continue
            if k in tags and tags[k] != str(v):
                raise SourceConflict(f"{skey}: tag {k} would change -- refuse")
        have = existing_artifacts(client, rid)
        for r in plan["include"]:
            if have.get(f"source/{r['path']}") != r["bytes"]:
                raise SourceConflict(f"{skey}: archived source/{r['path']} differs -- refuse")
        for r in plan["extras"]:
            if have.get(r["path"]) != r["bytes"]:
                raise SourceConflict(f"{skey}: archived {r['path']} differs -- refuse")
        tar_info = {n: {"bytes": int(tags[f"bundle.{n}.bytes"]), "sha256": tags[f"bundle.{n}.sha256"]} for n in plan["bundle"]}
        for n, info in tar_info.items():
            if have.get(n) != info["bytes"]:
                raise SourceConflict(f"{skey}: {n} differs -- refuse")
    todo = []
    for v in plan["views"]:
        cr = find_run(client, exp.experiment_id, v["key"])
        if cr is None:
            out["children_not_imported"] += 1
            continue
        ct = cr.data.tags
        if mode == "empty shell":
            raise SourceConflict(f"{v['key']}: child exists under an empty parent shell -- stop")
        if ct.get("mlflow.parentRunId") != rid:
            raise SourceConflict(f"{v['key']}: child mislinked -- stop")
        if ct.get("import_fingerprint") == v["fingerprint"] and ct.get("import_status") == "complete":
            out["children_unchanged"] += 1
            continue
        if mode == "unchanged":
            raise SourceConflict(f"{v['key']}: child differs under an unchanged parent -- stop")
        if ct.get("import_status") not in ("complete", "reconciling"):
            raise SourceConflict(f"{v['key']}: child in state {ct.get('import_status')} -- stop")
        if cr.data.metrics != v["metrics"]:
            raise SourceConflict(f"{v['key']}: metrics would change -- refuse")
        for k, x in v["params"].items():
            if cr.data.params.get(k) != str(x):
                raise SourceConflict(f"{v['key']}: param {k} would change -- refuse")
        for k, x in v["tags"].items():
            if k in ct and ct[k] != str(x):
                raise SourceConflict(f"{v['key']}: tag {k} would change -- refuse")
        c_old = ct["import_fingerprint"] if ct["import_fingerprint"] != v["fingerprint"] else ct.get("import_fingerprint.previous")
        if not c_old:
            raise SourceConflict(f"{v['key']}: interrupted child lacks its previous fingerprint -- stop")
        todo.append((v, cr, c_old))

    # ---- phase 2: writes (only after the whole family validated)
    if mode == "unchanged":
        out["parent"] = "unchanged"
        return out
    if mode == "empty shell":
        client.set_tag(rid, "import_fingerprint.previous", old_fp)
        client.set_tag(rid, "import_fingerprint", plan["fingerprint"])
        client.set_tag(rid, "reconciliation", f"{RECONCILE_NOTE}: empty in_progress shell rebound; resumed by import")
        out["parent"] = "empty shell rebound"
        return out
    if mode == "reconcile":
        client.set_tag(rid, "import_fingerprint.previous", old_fp)
        client.set_tag(rid, "reconcile_target", plan["fingerprint"])
        client.set_tag(rid, "import_status", "reconciling")
    old16 = old_fp[:16]
    for path, obj in parent_manifests(plan, tar_info).items():
        rec = _supersede(client, rid, have, path, json.dumps(obj, indent=1, sort_keys=True).encode(), old16, "manifests/")
        if rec:
            out["artifact_changes"].append(rec)
    log = {"record": skey, "previous_fingerprint": old_fp, "fingerprint": plan["fingerprint"], "note": RECONCILE_NOTE,
           "changes": out["artifact_changes"]}
    upload(client, rid, {}, f"manifests/superseded/reconciliation-{old16}.json",
           data=json.dumps(log, indent=1, sort_keys=True).encode())
    client.set_tag(rid, "import_fingerprint", plan["fingerprint"])
    client.set_tag(rid, "reconciliation", RECONCILE_NOTE)
    out["parent"] = "reconciled" if mode == "reconcile" else "resumed"
    comp = {}
    for i, (v, cr, c_old) in enumerate(todo):
        if fail_after == "reconcile-child-2" and i == 2:
            raise Interrupt("injected interruption during child reconciliation")
        crid = cr.info.run_id
        client.set_tag(crid, "import_status", "reconciling")
        chave = existing_artifacts(client, crid)
        changes = [c for c in (
            _supersede(client, crid, chave, "view/evaluation_view.json", v["view_json"].encode(), c_old[:16], "view/"),
            _supersede(client, crid, chave, "view/na.json", json.dumps(v["na"], indent=1, sort_keys=True).encode(),
                       c_old[:16], "view/")) if c]
        log_path = f"view/superseded/reconciliation-{c_old[:16]}.json"
        if log_path not in chave:   # a resumed child keeps its first reconciliation log
            upload(client, crid, {}, log_path, data=json.dumps(
                {"record": v["key"], "previous_fingerprint": c_old, "fingerprint": v["fingerprint"], "note": RECONCILE_NOTE,
                 "tags_added": sorted(k for k in v["tags"] if k not in cr.data.tags), "changes": changes},
                indent=1, sort_keys=True).encode())
        cp = v["tags"].get("comparator_parent")
        if cp and cp not in comp:
            comp[cp] = _run_id_for(client, exp, cp)
        log_values(client, client.get_run(crid), {}, {}, {**child_tags(v, rid, comp.get(cp)),
                                                          "import_status": "reconciling",
                                                          "import_fingerprint.previous": c_old,
                                                          "reconciliation": RECONCILE_NOTE})
        verify_child(client, client.get_run(crid), v, rid, deep=True)
        client.set_tag(crid, "import_status", "complete")
        out["children_reconciled"] += 1
    verify_parent(client, client.get_run(rid), plan, deep=True)
    client.set_tag(rid, "import_status", "complete")
    return out


# ------------------------------------------------------------------ verify

def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def verify_parent(client, run, plan: dict, deep: bool) -> dict:
    fam = plan["config"]
    skey = f"{fam['family']}/{fam['source_run_id']}"
    rid = run.info.run_id
    out = {"metrics": 0, "artifacts": 0, "artifact_bytes": 0, "tar_members": 0, "downloaded_bytes": 0}
    if run.data.tags.get("import_fingerprint") != plan["fingerprint"]:
        raise SourceConflict(f"{skey}: fingerprint readback mismatch")
    for k, v in plan["parent_metrics"].items():
        if run.data.metrics.get(k) != v:
            raise SourceConflict(f"{skey}: metric {k} readback mismatch")
        out["metrics"] += 1
    for k, v in parent_params(plan).items():
        if run.data.params.get(k) != str(v):
            raise SourceConflict(f"{skey}: param {k} readback mismatch")
    for k, v in parent_tags(plan, {}).items():
        if k in ("import_status", "shared_inputs_parent_run_id"):
            continue
        if run.data.tags.get(k) != str(v):
            raise SourceConflict(f"{skey}: tag {k} readback mismatch")
    have = existing_artifacts(client, rid)
    want = {f"source/{r['path']}": r for r in plan["include"]} | {r["path"]: r for r in plan["extras"]}
    tar_info = {}
    for name, members in plan["bundle"].items():
        tar_info[name] = {"bytes": int(run.data.tags[f"bundle.{name}.bytes"]), "sha256": run.data.tags[f"bundle.{name}.sha256"]}
        want[name] = {**tar_info[name], "members": members}
    for path, obj in parent_manifests(plan, tar_info).items():
        data = json.dumps(obj, indent=1, sort_keys=True).encode()
        want[path] = {"bytes": len(data), "sha256": sha_bytes(data)}
    extra = set(have) - set(want)
    if run.data.tags.get("import_fingerprint.previous"):
        extra = {p for p in extra if not p.startswith("manifests/superseded/")}
    if set(want) - set(have) or extra:
        raise SourceConflict(f"{skey}: artifact set mismatch missing={sorted(set(want) - set(have))[:5]} "
                             f"extra={sorted(extra)[:5]}")
    with tempfile.TemporaryDirectory(prefix="ipcch-mlflow-verify-") as tmp:
        for path, r in sorted(want.items()):
            if have[path] != r["bytes"]:
                raise SourceConflict(f"{skey}: {path} size {have[path]} != {r['bytes']}")
            if deep:
                local = Path(client.download_artifacts(rid, path, tmp))
                if _sha_file(local) != r["sha256"]:
                    raise SourceConflict(f"{skey}: {path} checksum mismatch after download")
                if "members" in r:
                    out["tar_members"] += tar_member_check(local, r["members"])
                out["downloaded_bytes"] += r["bytes"]
                local.unlink()
            out["artifacts"] += 1
            out["artifact_bytes"] += r["bytes"]
    return out


def verify_child(client, cr, v: dict, parent_rid: str, deep: bool) -> int:
    if cr.data.tags.get("import_fingerprint") != v["fingerprint"] or cr.data.tags.get("mlflow.parentRunId") != parent_rid:
        raise SourceConflict(f"{v['key']}: child fingerprint/parent link readback mismatch")
    if set(cr.data.metrics) != set(v["metrics"]):
        raise SourceConflict(f"{v['key']}: metric key set differs (NA must not be logged)")
    for k, x in v["metrics"].items():
        if cr.data.metrics[k] != x:
            raise SourceConflict(f"{v['key']}: metric {k} readback {cr.data.metrics[k]} != {x}")
    for k, x in v["params"].items():
        if cr.data.params.get(k) != str(x):
            raise SourceConflict(f"{v['key']}: param {k} readback mismatch")
    for k, x in {**v["tags"], **PROVENANCE_TAGS}.items():
        if cr.data.tags.get(k) != str(x):
            raise SourceConflict(f"{v['key']}: tag {k} readback mismatch")
    have = existing_artifacts(client, cr.info.run_id)
    na_bytes = json.dumps(v["na"], indent=1, sort_keys=True).encode()
    want = {"view/evaluation_view.json": v["view_json"].encode(), "view/na.json": na_bytes}
    if cr.data.tags.get("import_fingerprint.previous"):
        have = {k: n for k, n in have.items() if not k.startswith("view/superseded/")}
    if {k: len(b) for k, b in want.items()} != have:
        raise SourceConflict(f"{v['key']}: child artifact set/sizes differ")
    if deep:
        with tempfile.TemporaryDirectory(prefix="ipcch-mlflow-verify-") as tmp:
            for path, b in want.items():
                if Path(client.download_artifacts(cr.info.run_id, path, tmp)).read_bytes() != b:
                    raise SourceConflict(f"{v['key']}: {path} readback differs")
    return len(v["metrics"])


def verify_family(client, exp, plan: dict, deep: bool = True) -> dict:
    fam = plan["config"]
    skey = f"{fam['family']}/{fam['source_run_id']}"
    run = find_run(client, exp.experiment_id, skey)
    if run is None or run.data.tags.get("import_status") != "complete":
        raise SourceConflict(f"{skey}: parent record missing or incomplete")
    out = {"source_key": skey, "run_id": run.info.run_id, "records": 1, **verify_parent(client, run, plan, deep)}
    out["child_metrics"] = 0
    for v in plan["views"]:
        cr = find_run(client, exp.experiment_id, v["key"])
        if cr is None or cr.data.tags.get("import_status") != "complete":
            raise SourceConflict(f"{v['key']}: child record missing or incomplete")
        out["child_metrics"] += verify_child(client, cr, v, run.info.run_id, deep)
        out["records"] += 1
        out["artifacts"] += 2
    return out


# ------------------------------------------------------------------ CLI

class Lock:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a+")

    def __enter__(self):
        try:
            fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another import/verify holds the lock; refusing concurrent run")
        return self

    def __exit__(self, *a):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "import", "verify", "reconcile"])
    ap.add_argument("--sources", default=str(Path(__file__).resolve().parent / "sources.json"))
    ap.add_argument("--store", default=str(DEFAULT_ROOT))
    ap.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    ap.add_argument("--artifact-location", default=None, help="only for scratch stores (tests)")
    ap.add_argument("--family", action="append", help="limit to these families (repeatable)")
    ap.add_argument("--evidence", default=None, help="also write plan files here")
    ap.add_argument("--rehash", action="store_true", help="ignore the stat-keyed hash cache")
    ap.add_argument("--shallow", action="store_true",
                    help="verify: metadata-only (values, tags, artifact names and sizes; no download/hash)")
    ap.add_argument("--out", default=None, help="write command result JSON here")
    ap.add_argument("--fail-after", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.sources).read_text())
    store = Path(a.store)
    t0 = time.time()
    with Lock(store / "import.lock"):
        plans = cmd_plan(cfg, store, Path(a.evidence) if a.evidence else None, a.rehash, a.family)
        result = {"command": a.command,
                  "records_planned": sum(1 + p["totals"]["children"] for k, p in plans.items()
                                         if not a.family or k in a.family),
                  "read_dependencies": [k for k in plans if a.family and k not in a.family]}
        if a.command != "plan":
            client = _client(a.tracking_uri)
            exp = get_experiment(client, cfg["experiment"], a.artifact_location)
            result["experiment_id"] = exp.experiment_id
            run_ids, fams = {}, []
            for fam in planning_order(cfg, a.family):
                if fam["family"] not in plans or (a.family and fam["family"] not in a.family):
                    if fam["family"] in plans:
                        r = find_run(client, exp.experiment_id, f"{fam['family']}/{fam['source_run_id']}")
                        if r:
                            run_ids[fam["family"]] = r.info.run_id
                    continue
                t1 = time.time()
                if a.command == "import":
                    s = import_family(client, exp, plans[fam["family"]], store, run_ids, a.fail_after)
                elif a.command == "reconcile":
                    s = reconcile_family(client, exp, plans[fam["family"]], run_ids, a.fail_after)
                else:
                    s = verify_family(client, exp, plans[fam["family"]], deep=not a.shallow)
                s["seconds"] = round(time.time() - t1, 1)
                print(json.dumps(s), flush=True)
                fams.append(s)
            result["families"] = fams
    result["seconds"] = round(time.time() - t0, 1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in result.items() if k != "families"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
