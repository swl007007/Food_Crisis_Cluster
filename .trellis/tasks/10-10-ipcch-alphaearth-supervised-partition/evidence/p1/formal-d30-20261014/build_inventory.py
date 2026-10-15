"""Delivery artifact inventory (read-only): per-file sha256/bytes for the formal run, logs and repo evidence; the
extraction run's inputs/logs/receipts individually and its 57,801 piece files via an ordered digest. Prints JSON."""
import hashlib, json, subprocess, sys
from pathlib import Path
H = Path.home() / "ipcch-alphaearth-runs"
ROOT = Path("/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/2.source_code/Step5_Geo_RF_trial/Food_Crisis_Cluster")
T = ROOT / ".trellis/tasks/10-10-ipcch-alphaearth-supervised-partition"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def files(d, skip=()):
    return {str(p.relative_to(d)): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(Path(d).rglob("*"))
            if p.is_file() and not any(str(p.relative_to(d)).startswith(s) for s in skip)}
inv = {"generated_utc": subprocess.run(["date", "-u", "+%FT%TZ"], capture_output=True, text=True).stdout.strip()}
run = H / "p1-formal-d30-20261014"
inv["formal_run"] = {"path": str(run), "location": "WSL ext4, outside Dropbox, not in git", "files": files(run)}
inv["formal_run"]["n_files"] = len(inv["formal_run"]["files"]); inv["formal_run"]["bytes"] = sum(v["bytes"] for v in inv["formal_run"]["files"].values())
logs = H / "p1-formal-d30-logs"
inv["formal_logs"] = {"path": str(logs), "files": files(logs)}
ext = H / "p1-c-full-20261010"
pieces = sorted((ext / "out/pieces").iterdir())
pd_ = hashlib.sha256()
for p in pieces:
    pd_.update(f"{p.name}\t{sha(p)}\n".encode())
inv["extraction_run"] = {"path": str(ext), "files_except_pieces": files(ext, skip=("out/pieces/",)),
                         "pieces": {"n": len(pieces), "bytes": sum(p.stat().st_size for p in pieces),
                                    "ordered_name_sha256_digest": pd_.hexdigest(),
                                    "digest_rule": "sha256 over lines 'filename<TAB>sha256(file)\\n' for out/pieces/* sorted by filename"}}
ev = T / "evidence/p1/formal-d30-20261014"
evf = files(ev)
paths = [str(ev / k) for k in evf]
ign = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "--stdin"], input="\n".join(paths), capture_output=True, text=True, timeout=300).stdout.split("\n")
ign = {str(Path(x).resolve().relative_to(ev)) if x.startswith("/") else str((ROOT / x).relative_to(ev)) for x in ign if x}
for k in evf:
    evf[k]["git_ignored_requires_force_add"] = k in ign
inv["repo_evidence"] = {"path": str(ev.relative_to(ROOT)), "files": evf, "n_files": len(evf), "n_ignored": len(ign)}
note = ROOT / "docs/notes/2026-10-14_组会讨论稿_IPCCH_AlphaEarth监督分区.md"
inv["delivery_docs"] = {str(p.relative_to(ROOT)): sha(p) for p in (note, ROOT / "IPCCHAlphaEarthExperiment/README.md", T / "PROGRESS.md")}
print(json.dumps(inv, indent=1))
