"""Read-only (r2): every *sha256 leaf in every JSON/JSONL file under the formal run, with its exact file + JSON path
and a typed role. File roles are resolved by hashing: an active run file, an archived run file, a file in the
extraction run, the logs/receipts dir, or the repo; otherwise the key's documented non-file construction."""
import hashlib, json, re, sys
from collections import defaultdict
from pathlib import Path
run = Path(sys.argv[1]); ext = Path(sys.argv[2]); logs = Path(sys.argv[3]); repo = Path(sys.argv[4])
fsha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
index = {}
def add(root, label, skip=()):
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and not any(s in str(p) for s in skip):
            index.setdefault(fsha(p), []).append(f"{label}:{p.relative_to(root)}")
add(run, "run"); add(ext, "extraction_run", skip=("/out/pieces/",)); add(logs, "logs")
for sub in ("IPCCHAlphaEarthExperiment/config", "IPCCHAlphaEarthExperiment/ipcch_alphaearth", "IPCCHAlphaEarthExperiment/scripts",
            "IPCCHAlphaEarthExperiment/tests"):
    add(repo / sub, f"repo:{sub}")
for extra in sys.argv[5:]:  # optional extra read-only roots "label=path"
    lab, path = extra.split("=", 1)
    add(Path(path), lab)
NONFILE = {  # documented constructions (IPCCHAlphaEarthExperiment source)
    "X_sha256": "array_digest(float64 X fitting slice): sha256 of dtype|shape| + C bytes (artifacts.py:40-46); in-memory ndarray, no file",
    "y_sha256": "array_digest per target q2..q5 of the fitting slice (backbone.py fit_identity); no file",
    "features_sha256": "artifacts.digest(ordered feature-name list) = canonical-JSON hash; no file",
    "compact296_list_sha256": "list_sha256(ordered compact296 names): newline-joined UTF-8; no file",
    "request_sha256": "canonical-JSON digest of the extraction manifest BODY (differs from the manifest file SHA)",
    "resolved_config_sha256": "digest of the resolved fitting configuration dict; no file",
    "sha256": "context-dependent; see path",
}
rows = []
def walk(o, path, rel):
    if isinstance(o, dict):
        for k, v in o.items():
            if k.endswith("sha256") and isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v):
                rows.append((rel, f"{path}.{k}", k, v))
            elif k.endswith("sha256") and isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, str) and re.fullmatch(r"[0-9a-f]{64}", vv):
                        rows.append((rel, f"{path}.{k}.{kk}", k, vv))
                    else:
                        walk(vv, f"{path}.{k}.{kk}", rel)
            else:
                walk(v, f"{path}.{k}", rel)
    elif isinstance(o, list):
        for i, x in enumerate(o):
            walk(x, f"{path}[{i}]", rel)
for p in sorted(run.rglob("*")):
    if p.suffix not in (".json", ".jsonl") or not p.is_file():
        continue
    rel = str(p.relative_to(run))
    objs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.suffix == ".jsonl" else [json.loads(p.read_text())]
    for o in objs:
        walk(o, "", rel)
out = defaultdict(lambda: {"occurrences": 0, "roles": defaultdict(int), "examples": []})
detail = []
for rel, path, key, val in rows:
    hits = index.get(val, [])
    if hits:
        kinds = sorted({h.split(":")[0] if not h.startswith("run:archive/") else "run_archived" for h in hits})
        role = "file:" + "+".join(kinds)
    else:
        role = "nonfile:" + NONFILE.get(key, "UNRESOLVED")
    pathclass = re.sub(r"\[\d+\]", "[]", path)
    filecls = re.sub(r"[0-9a-f]{64}", "<id>", re.sub(r"/[0-9a-f]{2}/", "/<p>/", rel))
    detail.append({"file": rel, "path": path, "key": key, "value": val, "role": role, "resolved": hits[:3]})
    o = out[(key, filecls, pathclass)]; o["occurrences"] += 1; o["roles"][role] += 1
    if len(o["examples"]) < 2: o["examples"].append({"file": rel, "value": val[:16], "resolved": hits[:2]})
summary = [{"key": k, "file_class": f, "json_path_class": p, "occurrences": v["occurrences"], "roles": dict(v["roles"]), "examples": v["examples"]}
           for (k, f, p), v in sorted(out.items())]
unresolved = [d for d in detail if d["role"].endswith("UNRESOLVED")]
print(json.dumps({"n_leaves": len(rows), "classes": summary, "unresolved": unresolved[:200], "n_unresolved": len(unresolved)}, indent=1))
