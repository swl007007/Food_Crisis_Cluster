"""Versioned delivery inventory v2 (read-only on all enumerated roots). Output is written OUTSIDE every enumerated
root, then copied into the task evidence; the two copy destinations are excluded by exact path. Attempt-1 inventory
(ARTIFACT_INVENTORY.json, 0d366606...) is preserved and enumerated as an ordinary completed file."""
import datetime as dt, hashlib, json, subprocess, sys
from pathlib import Path
H = Path.home() / "ipcch-alphaearth-runs"
ROOT = Path("/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/2.source_code/Step5_Geo_RF_trial/Food_Crisis_Cluster")
T = ROOT / ".trellis/tasks/10-10-ipcch-alphaearth-supervised-partition"
EV = T / "evidence/p1/formal-d30-20261014"
OUT = H / "p1-formal-d30-inventory"
SELF_COPIES = {"ARTIFACT_INVENTORY.v2.json", "ARTIFACT_INVENTORY.v2.sha256"}  # their evidence copies, by exact relative path
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def files(d, skip_prefix=(), skip_exact=()):
    return {str(p.relative_to(d)): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(Path(d).rglob("*"))
            if p.is_file() and not str(p.relative_to(d)).startswith(skip_prefix) and str(p.relative_to(d)) not in skip_exact}
t0 = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
inv = {"version": "inventory-v2", "snapshot_started_utc": t0,
       "scope": "formal run, its logs dir, the extraction run (pieces by ordered digest), the task evidence dir and the delivery docs, as they exist at this snapshot; files written later are out of scope",
       "excluded_self_paths": [str(EV / x) for x in sorted(SELF_COPIES)] + [str(OUT)],
       "preserved_attempt1": {"path": str(EV / "ARTIFACT_INVENTORY.json"), "sha256": sha(EV / "ARTIFACT_INVENTORY.json")}}
run = H / "p1-formal-d30-20261014"
inv["formal_run"] = {"path": str(run), "files": files(run)}
logs = H / "p1-formal-d30-logs"
inv["formal_logs"] = {"path": str(logs), "files": files(logs)}
ext = H / "p1-c-full-20261010"
pieces = sorted((ext / "out/pieces").iterdir()); h = hashlib.sha256()
for p in pieces:
    h.update(f"{p.name}\t{sha(p)}\n".encode())
inv["extraction_run"] = {"path": str(ext), "files_except_pieces": files(ext, skip_prefix=("out/pieces/",)),
                         "pieces": {"n": len(pieces), "bytes": sum(p.stat().st_size for p in pieces), "ordered_name_sha256_digest": h.hexdigest(),
                                    "digest_rule": "sha256 over 'filename<TAB>sha256(file)\\n' for out/pieces/* sorted by filename"}}
evf = files(EV, skip_exact=SELF_COPIES)
ign = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "--stdin"], input="\n".join(str(EV / k) for k in evf),
                     capture_output=True, text=True, timeout=300).stdout.split("\n")
ign = {str(Path(x).relative_to(EV)) if x.startswith("/") else str((ROOT / x).relative_to(EV)) for x in ign if x}
for k in evf:
    evf[k]["git_ignored_requires_force_add"] = k in ign
inv["task_evidence"] = {"path": str(EV), "files": evf, "n_ignored": len(ign)}
docs = [ROOT / "docs/notes/2026-10-14_组会讨论稿_IPCCH_AlphaEarth监督分区.md", ROOT / "IPCCHAlphaEarthExperiment/README.md",
        T / "PROGRESS.md", T / "prd.md", T / "implement.md", T / "mlflow-plan.md", *sorted((T / "mlflow-plan-history").glob("*"))]
inv["delivery_docs"] = {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in docs}
inv["counts"] = {"formal_run": len(inv["formal_run"]["files"]), "formal_logs": len(inv["formal_logs"]["files"]),
                 "extraction_files_except_pieces": len(inv["extraction_run"]["files_except_pieces"]), "pieces": len(pieces),
                 "task_evidence": len(evf), "task_evidence_git_ignored": len(ign), "delivery_docs": len(docs)}
inv["snapshot_finished_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
out = OUT / "ARTIFACT_INVENTORY.v2.json"
out.write_text(json.dumps(inv, indent=1) + "\n")
(OUT / "ARTIFACT_INVENTORY.v2.sha256").write_text(f"{sha(out)}  ARTIFACT_INVENTORY.v2.json\n")
print(json.dumps(inv["counts"]), sha(out))
