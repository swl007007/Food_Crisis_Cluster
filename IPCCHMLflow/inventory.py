"""Reconcile a source plan against the source run's own saved inventories.

Discovery only sees files that still exist; this check compares the digests,
paths and sizes the run recorded about itself (ledgers, summaries, manifests)
with the planned files, so a retained model or transform that disappeared is
caught. Every ``*sha256`` key found must have an explicit decision in the
family's ``inventory`` policy (sources.json):

  required_digest_keys  every value equals the SHA256 of a planned file (this
                        family's or a reference parent's)
  name_keys             value names a directory (``dir``) or file stem
                        (``stem``) inside the model bundle
  informational_keys    content/identity hashes with no single-file form
  exempt                scoped (key, JSON-path regex[, file regex]) exceptions with a reason

plus optional path inventories ([{path, bytes, sha256}]), name->hash maps and
ledger artifact-directory lists. Unclassified keys or unmatched required
values raise SourceConflict.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath

from extract import SourceConflict

HEX = re.compile(r"^[0-9a-f]{64}$")


def _leaves(obj, path="", key=None):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, f"{path}.{k}", k if str(k).endswith("sha256") else key)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _leaves(v, f"{path}[{i}]", key)
    elif key and isinstance(obj, str) and HEX.match(obj):
        yield key, path, obj


def _load(p: Path):
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return [json.loads(p.read_text())]


def reconcile(fam: dict, root: Path, files: dict, parents: dict) -> dict:
    """files: {'include','bundle','excluded','shared'} planned records; parents: {family: plan}."""
    pol = fam.get("inventory")
    if pol is None:
        raise SourceConflict(f"{fam['family']}: no inventory policy in sources.json")
    bundle = [m for v in files["bundle"].values() for m in v]
    own = files["include"] + bundle + files["excluded"] + files["shared"]
    by_sha = defaultdict(list)
    by_path = {r["path"]: r for r in own}
    for r in own:
        by_sha[r["sha256"]].append(r["path"])
    for pf in pol.get("reference_parents", []):
        if pf not in parents:
            raise SourceConflict(f"{fam['family']}: reference parent {pf} not planned")
        pp = parents[pf]
        for r in pp["include"] + [m for v in pp["bundle"].values() for m in v] + pp["excluded"] + pp["shared"]:
            by_sha[r["sha256"]].append(f"{pf}:{r['path']}")
    dirs = {PurePosixPath(m["path"]).parent.name for m in bundle}
    stems = {PurePosixPath(m["path"]).name.split(".")[0] for m in bundle}
    dir_files = defaultdict(int)
    for m in bundle:
        dir_files[PurePosixPath(m["path"]).parent.name] += 1
    required = set(pol.get("required_digest_keys", []))
    name_keys = pol.get("name_keys", {})
    info = pol.get("informational_keys", {})
    exempt = [(e["key"], re.compile(e["path"]), re.compile(e.get("file", "")), e["reason"]) for e in pol.get("exempt", [])]
    report = {"keys": {}, "path_inventories": {}, "name_hash_maps": {}, "ledger_dirs": {}, "sources_scanned": 0}
    failures = []
    inv_files = sorted(r["path"] for r in files["include"] + files["excluded"] + files["shared"]
                       if r["path"].endswith((".json", ".jsonl")))
    for rel in inv_files:
        report["sources_scanned"] += 1
        for obj in _load(root / rel):
            for key, path, val in _leaves(obj):
                k = report["keys"].setdefault(key, {"values": 0, "unique": set(), "matched": 0, "exempted": 0,
                                                    "decision": None, "files": set()})
                k["values"] += 1
                k["unique"].add(val)
                k["files"].add(rel.split("/")[0])
                ex = next((r for kk, rx, fx, r in exempt if kk == key and rx.search(path) and fx.search(rel)), None)
                if key in required:
                    k["decision"] = "required file digest"
                    if val in by_sha:
                        k["matched"] += 1
                    elif ex:
                        k["exempted"] += 1
                        k.setdefault("exempt_reasons", set()).add(ex)
                    else:
                        failures.append(f"{rel}{path}: {key} {val[:12]} matches no planned file")
                elif key in name_keys:
                    kind = name_keys[key]
                    k["decision"] = f"names a bundle {kind}"
                    ok = val in (dirs if kind == "dir" else stems)
                    if ok:
                        k["matched"] += 1
                    elif ex:
                        k["exempted"] += 1
                        k.setdefault("exempt_reasons", set()).add(ex)
                    else:
                        failures.append(f"{rel}{path}: {key} {val[:12]} has no bundle {kind}")
                elif key in info:
                    k["decision"] = f"informational: {info[key]}"
                    k["matched"] += val in by_sha
                else:
                    failures.append(f"{rel}{path}: digest key {key!r} has no inventory decision")
    for k in report["keys"].values():
        k["unique"] = len(k["unique"])
        k["files"] = sorted(k["files"])
        if "exempt_reasons" in k:
            k["exempt_reasons"] = sorted(k["exempt_reasons"])
    for spec in pol.get("path_inventories", []):
        if isinstance(spec, str):  # list of {path, bytes, sha256}
            rel, entries = spec, _load(root / spec)[0]
        else:                      # {"file", "field", "prefix"}: dict path -> {bytes, sha256}
            rel = spec["file"]
            entries = [{"path": spec.get("prefix", "") + k, **v} for k, v in _load(root / rel)[0][spec["field"]].items()]
        st = {"entries": len(entries), "matched": 0}
        for e in entries:
            p = e["path"].replace("\\", "/")
            r = by_path.get(p)
            if r is None or r["bytes"] != e["bytes"] or r["sha256"] != e["sha256"]:
                failures.append(f"{rel}: {p} missing or size/sha differs")
            else:
                st["matched"] += 1
        report["path_inventories"][rel] = st
    for rel in pol.get("name_hash_maps", []):
        m = _load(root / rel)[0]
        st = {"entries": len(m), "matched": 0}
        for name, sha in m.items():
            r = by_path.get(name.replace("\\", "/"))
            if r is None or r["sha256"] != sha:
                failures.append(f"{rel}: {name} missing or sha differs")
            else:
                st["matched"] += 1
        report["name_hash_maps"][rel] = st
    for spec in pol.get("ledger_dirs", []):
        entries = _load(root / spec["file"])[0]
        st = {"entries": len(entries), "matched": 0}
        for e in entries:
            d = PurePosixPath(e[spec["field"]].replace("\\", "/")).name
            if dir_files.get(d, 0) < spec.get("min_files", 1):
                failures.append(f"{spec['file']}: artifact dir {d} missing or has too few files")
            else:
                st["matched"] += 1
        report["ledger_dirs"][spec["file"]] = st
    for key, kind in name_keys.items():
        if kind == "dir" and key in report["keys"]:
            sizes = sorted({dir_files[d] for d in dirs})
            report["keys"][key]["bundle_dir_file_counts"] = sizes
    if failures:
        raise SourceConflict(f"{fam['family']}: original-inventory reconciliation failed "
                             f"({len(failures)}): " + "; ".join(failures[:8]))
    report["status"] = "reconciled"
    return report
