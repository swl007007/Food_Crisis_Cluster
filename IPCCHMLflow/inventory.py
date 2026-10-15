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

plus optional path inventories ([{path, bytes, sha256}]), name->hash maps,
ledger artifact-directory lists with exact members, and a per-identity model
contract read from the run's own ledgers (model_requests*.jsonl):

  xgb  each ledger identity has exactly one bundle directory holding record.json
       and, for every declared booster ``q``, ``q.ubj`` with the declared digest
       (its own file, not any file with that digest)
  mlp  each ledger identity has record.json and state.pt; the transform named by
       record.identity.transform_sha256 exists as <store>/transforms/<digest>.npz
       (path-level: MLP stores record no file digests for states/transforms)

Counts distinguish reference occurrences from unique values/identities.
Unclassified keys or unmatched required values raise SourceConflict.

Strict mode (``strict_bindings``: AlphaEarth only; legacy policies never take this path): every digest leaf of
every planned JSON file, bundle record JSON included, must match exactly one scoped rule (key + file regex + JSON-path
regex): ``own`` (the full SHA of the one intended planned file), ``external`` (the full SHA of the one intended
absolute file, optionally pinned), ``typed`` (a documented non-file identity) or ``historical`` (an upstream
reference pinned by exact path and value). Unclassified, ambiguous, wrong-target, unused-rule and leaf-count drift
all refuse. Contract ``alphaearth_xgb`` binds the native CSV fit ledgers one-to-one to the schedule and each
identity to its own bundle directory (recomputed record identity digest, own q2..q5 boosters).
"""

from __future__ import annotations

import csv
import gzip
import hashlib
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
    if pol.get("strict_bindings"):
        return reconcile_strict(fam, root, files, pol)
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
                k = report["keys"].setdefault(key, {"reference_occurrences": 0, "unique_values": set(), "matched": 0,
                                                    "exempted": 0, "decision": None, "files": set()})
                k["reference_occurrences"] += 1
                k["unique_values"].add(val)
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
        k["unique_values"] = len(k["unique_values"])
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
            members = {PurePosixPath(m["path"]).name for m in bundle if PurePosixPath(m["path"]).parent.name == d}
            need = set(spec.get("members", []))
            if not members or not need <= members:
                failures.append(f"{spec['file']}: artifact dir {d} missing or lacks {sorted(need - members)}")
            else:
                st["matched"] += 1
        report["ledger_dirs"][spec["file"]] = st
    for key, kind in name_keys.items():
        if kind == "dir" and key in report["keys"]:
            sizes = sorted({dir_files[d] for d in dirs})
            report["keys"][key]["bundle_dir_file_counts"] = sizes
    if pol.get("model_contract"):
        report["model_contract"] = model_contract(pol["model_contract"], root, inv_files, bundle, by_path, failures)
    if failures:
        raise SourceConflict(f"{fam['family']}: original-inventory reconciliation failed "
                             f"({len(failures)}): " + "; ".join(failures[:8]))
    report["status"] = "reconciled"
    return report


def model_contract(kind: str, root: Path, inv_files: list, bundle: list, by_path: dict, failures: list) -> dict:
    ledgers = [rel for rel in inv_files if PurePosixPath(rel).name.startswith("model_requests")
               and rel.endswith(".jsonl")]
    dirs = defaultdict(set)
    for m in bundle:
        parent = PurePosixPath(m["path"]).parent
        if HEX.match(parent.name):
            dirs[parent.name].add(str(parent))
    refs, boosters = 0, defaultdict(dict)
    for rel in ledgers:
        for line in _load(root / rel):
            ident = line.get("identity_sha256")
            if not ident:
                continue
            refs += 1
            for q, digest in (line.get("booster_sha256") or {}).items() if kind == "xgb" else ():
                prev = boosters[ident].get(q)
                if prev is not None and digest is not None and prev != digest:
                    failures.append(f"{rel}: identity {ident[:12]} declares two digests for {q}")
                if digest is not None:
                    boosters[ident][q] = digest
            boosters.setdefault(ident, {})
    st = {"kind": kind, "ledger_files": len(ledgers), "ledger_reference_occurrences": refs,
          "unique_identities": len(boosters), "identities_satisfied": 0}
    if kind == "xgb":
        st["unique_boosters_checked"] = 0
    else:
        st.update(records_read=0, states_present=0, unique_transforms_referenced=0, records_without_transform=0)
        transforms = set()
    for ident, qs in sorted(boosters.items()):
        paths = dirs.get(ident, set())
        if len(paths) != 1:
            failures.append(f"model identity {ident[:12]}: {len(paths)} bundle directories")
            continue
        d = next(iter(paths))
        ok = f"{d}/record.json" in by_path
        if not ok:
            failures.append(f"model {ident[:12]}: record.json missing")
        if kind == "xgb":
            for q, digest in qs.items():
                r = by_path.get(f"{d}/{q}.ubj")
                if r is None or r["sha256"] != digest:
                    failures.append(f"model {ident[:12]}: own booster {q}.ubj missing or digest differs")
                    ok = False
                else:
                    st["unique_boosters_checked"] += 1
        else:
            if f"{d}/state.pt" in by_path:
                st["states_present"] += 1
            else:
                failures.append(f"model {ident[:12]}: state.pt missing")
                ok = False
            if f"{d}/record.json" in by_path:
                rec = json.loads((root / f"{d}/record.json").read_text())
                st["records_read"] += 1
                t = (rec.get("identity") or {}).get("transform_sha256")
                if t is None:
                    st["records_without_transform"] += 1
                else:
                    transforms.add(t)
                    store = PurePosixPath(d).parents[2]
                    if f"{store}/transforms/{t}.npz" not in by_path:
                        failures.append(f"model {ident[:12]}: referenced transform {t[:12]} missing")
                        ok = False
        st["identities_satisfied"] += ok
    if kind == "mlp":
        st["unique_transforms_referenced"] = len(transforms)
    return st


# ------------------------------------------------------------------ strict bindings (AlphaEarth)

def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _target(rule: dict, m: re.Match, rel: str):
    groups = [int(g) if g is not None and g.isdigit() else g for g in m.groups()]
    if "target_map" in rule:
        return rule["target_map"].get(str(m.group(1)))
    return rule["target"].format(*groups, dir=str(PurePosixPath(rel).parent))


def reconcile_strict(fam: dict, root: Path, files: dict, pol: dict) -> dict:
    bundle = [m for v in files["bundle"].values() for m in v]
    own = files["include"] + bundle + files["excluded"] + files["shared"]
    by_path = {r["path"]: r for r in own}
    rules = []
    for i, r in enumerate(pol["bindings"]):
        if r["kind"] not in ("own", "external", "typed", "historical"):
            raise SourceConflict(f"{fam['family']}: binding {i} has unknown kind {r['kind']!r}")
        rules.append((i, r, re.compile(r["file"] + r"\Z"), re.compile(r["path"] + r"\Z")))
    scan = sorted(r["path"] for r in files["include"] + files["excluded"] + files["shared"]
                  if r["path"].endswith((".json", ".jsonl")))
    if pol.get("scan_bundle_json"):
        scan += sorted(m["path"] for m in bundle if m["path"].endswith(".json"))
    failures, used = [], defaultdict(int)
    report = {"mode": "strict_bindings", "files_scanned": len(scan), "leaves": 0, "by_kind": defaultdict(int),
              "external_files": {}}
    ext_cache = {}
    for rel in scan:
        for obj in _load(root / rel):
            for key, path, val in _leaves(obj):
                report["leaves"] += 1
                hits = [(i, r, m) for i, r, fx, px in rules if r["key"] == key and fx.match(rel)
                        for m in [px.match(path)] if m]
                if len(hits) != 1:
                    failures.append(f"{rel}{path}: {key} matches {len(hits)} bindings (need exactly one)")
                    continue
                i, r, m = hits[0]
                used[i] += 1
                report["by_kind"][r["kind"]] += 1
                if r["kind"] == "own":
                    t = _target(r, m, rel)
                    rec = by_path.get(t) if t else None
                    if rec is None or rec["sha256"] != val:
                        failures.append(f"{rel}{path}: {key} {val[:12]} is not the SHA of its own file {t}")
                elif r["kind"] == "external":
                    t = _target(r, m, rel)
                    pin = (r.get("sha256_map") or {}).get(str(m.group(1))) if "sha256_map" in r else r.get("sha256")
                    if t is None or not Path(t).is_file():
                        failures.append(f"{rel}{path}: external file {t} missing")
                        continue
                    if t not in ext_cache:
                        ext_cache[t] = _sha_file(Path(t))
                    if ext_cache[t] != val or (pin is not None and pin != val):
                        failures.append(f"{rel}{path}: external {t} sha {ext_cache[t][:12]} / pin {str(pin)[:12]} != {val[:12]}")
                    report["external_files"][t] = ext_cache[t]
                elif r["kind"] == "typed":
                    if "value" in r and r["value"] != val:
                        failures.append(f"{rel}{path}: typed {key} value {val[:12]} != pinned {r['value'][:12]}")
                else:
                    want = (r.get("values") or {}).get(path)
                    if want != val:
                        failures.append(f"{rel}{path}: historical {key} not pinned to this exact path/value")
    unused = [i for i, *_ in rules if used[i] == 0]
    if unused:
        failures.append(f"bindings never used (stale policy): {unused}")
    if pol.get("expected_leaves") is not None and report["leaves"] != pol["expected_leaves"]:
        failures.append(f"digest leaves {report['leaves']} != expected {pol['expected_leaves']}")
    report["by_kind"] = dict(report["by_kind"])
    report["rules_used"] = {str(i): used[i] for i, *_ in rules}
    if pol.get("model_contract") == "alphaearth_xgb":
        report["model_contract"] = alphaearth_model_contract(root, pol, bundle, by_path, failures)
    else:
        failures.append("strict policy needs model_contract alphaearth_xgb")
    if failures:
        raise SourceConflict(f"{fam['family']}: original-inventory reconciliation failed "
                             f"({len(failures)}): " + "; ".join(failures[:8]))
    report["status"] = "reconciled"
    return report


def _csv_rows(p: Path) -> list:
    if not p.is_file():
        raise SourceConflict(f"model ledger missing: {p}")
    with gzip.open(p, "rt", newline="") as f:
        return list(csv.DictReader(f))


def alphaearth_model_contract(root: Path, pol: dict, bundle: list, by_path: dict, failures: list) -> dict:
    """Native fit ledgers (CSV) <-> schedule <-> own bundle directory, one-to-one.

    Identity digest = sha256 of the record's ``identity`` in the AlphaEarth package's canonical JSON
    (``ipcch_alphaearth/artifacts.py`` canonical: sort_keys, separators (',', ':')); nothing is loaded or fitted."""
    keyf = ("arm", "horizon", "block_id", "role", "fit_origin")
    rows = [r for rel in pol["model_ledgers"] for r in _csv_rows(root / rel)]
    sched = _csv_rows(root / pol["model_schedule"])
    st = {"kind": "alphaearth_xgb", "ledger_rows": len(rows), "schedule_rows": len(sched), "unique_identities": 0,
          "records_verified": 0, "boosters_checked": 0}
    if not rows:
        failures.append("model ledgers are empty")
        return st
    ids = [r["identity"] for r in rows]
    st["unique_identities"] = len(set(ids))
    if len(set(ids)) != len(ids):
        failures.append("duplicate identity in the model ledgers")
    if pol.get("expected_identities") is not None and len(set(ids)) != pol["expected_identities"]:
        failures.append(f"{len(set(ids))} ledger identities != expected {pol['expected_identities']}")
    if any(r["action"] != "fit" for r in rows):
        failures.append("a model ledger row is not action=fit")
    stuples = [tuple(r[k] for k in keyf) for r in sched]
    if len(set(stuples)) != len(stuples) or sorted(stuples) != sorted(tuple(r[k] for k in keyf) for r in rows):
        failures.append("model ledgers are not one-to-one with the schedule (arm/horizon/block/role/fit_origin)")
    dirs = defaultdict(set)
    for m in bundle:
        pp = PurePosixPath(m["path"])
        if len(pp.parts) == 4 and HEX.match(pp.parts[2]):
            dirs[pp.parts[2]].add(pp.name)
        else:
            failures.append(f"bundle member outside an identity directory: {m['path']}")
    if set(dirs) != set(ids):
        failures.append(f"bundle identity dirs {len(dirs)} != ledger identities {len(set(ids))} "
                        f"(extra {sorted(set(dirs) - set(ids))[:2]}, missing {sorted(set(ids) - set(dirs))[:2]})")
    want = {"record.json", "q2.ubj", "q3.ubj", "q4.ubj", "q5.ubj"}
    for r in rows:
        ident = r["identity"]
        d = f"{pol.get('model_root', 'models')}/{ident[:2]}/{ident}"
        if dirs.get(ident) != want:
            failures.append(f"model {ident[:12]}: own directory {d} members {sorted(dirs.get(ident, []))} != {sorted(want)}")
            continue
        rec = json.loads((root / d / "record.json").read_text())
        body = json.dumps(rec.get("identity"), sort_keys=True, separators=(",", ":"), default=str)
        if rec.get("identity_digest") != ident or hashlib.sha256(body.encode()).hexdigest() != ident:
            failures.append(f"model {ident[:12]}: record identity digest does not recompute")
            continue
        idn = rec["identity"]
        if tuple(str(idn.get(k)) for k in keyf) != tuple(r[k] for k in keyf):
            failures.append(f"model {ident[:12]}: record identity fields differ from its ledger row")
            continue
        tg = rec.get("targets") or {}
        if sorted(tg) != ["q2", "q3", "q4", "q5"]:
            failures.append(f"model {ident[:12]}: record targets {sorted(tg)} != q2..q5")
            continue
        ok = True
        for q in ("q2", "q3", "q4", "q5"):
            own = by_path.get(f"{d}/{q}.ubj")
            if own is None or (tg[q] or {}).get("booster_sha256") != own["sha256"]:
                failures.append(f"model {ident[:12]}: own booster {q}.ubj missing or digest differs")
                ok = False
            else:
                st["boosters_checked"] += 1
        st["records_verified"] += ok
    if pol.get("expected_boosters") is not None and st["boosters_checked"] != pol["expected_boosters"]:
        failures.append(f"boosters checked {st['boosters_checked']} != expected {pol['expected_boosters']}")
    return st
