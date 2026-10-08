"""Backup and scratch-restore check for the local IPCCH MLflow store.

  backup   SQLite online backup API + byte copy of artifacts/ + sha256 manifest
  restore  copy a backup into a scratch root and check it against the manifest,
           then open the restored DB read-only with MlflowClient and count records

Hold the import lock while backing up so no importer writes mid-copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from import_runs import CHUNK, DEFAULT_ROOT, Lock  # noqa: E402


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def tree_manifest(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(root.rglob("*")) if p.is_file()}


def db_counts(db: Path) -> dict:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        q = lambda s: con.execute(s).fetchone()[0]  # noqa: E731
        return {"experiments": q("select count(*) from experiments"), "runs": q("select count(*) from runs"),
                "latest_metrics": q("select count(*) from latest_metrics"), "params": q("select count(*) from params"),
                "tags": q("select count(*) from tags"), "integrity": con.execute("pragma integrity_check").fetchone()[0]}
    finally:
        con.close()


def backup(store: Path, dest: Path) -> dict:
    if dest.exists():
        raise SystemExit(f"backup destination exists: {dest}")
    dest.mkdir(parents=True)
    t0 = time.time()
    with Lock(store / "import.lock"):
        src = sqlite3.connect(f"file:{store / 'mlflow.db'}?mode=ro", uri=True)
        dst = sqlite3.connect(dest / "mlflow.db")
        with dst:
            src.backup(dst)
        src.close()
        dst.close()
        shutil.copytree(store / "artifacts", dest / "artifacts")
    man = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source_store": str(store),
           "db": {"bytes": (dest / "mlflow.db").stat().st_size, "sha256": sha(dest / "mlflow.db"),
                  "counts": db_counts(dest / "mlflow.db")},
           "artifacts": tree_manifest(dest / "artifacts")}
    man["artifact_files"] = len(man["artifacts"])
    man["artifact_bytes"] = sum(v["bytes"] for v in man["artifacts"].values())
    man["seconds"] = round(time.time() - t0, 1)
    (dest / "backup-manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    return {k: v for k, v in man.items() if k != "artifacts"}


def restore_check(backup_dir: Path, scratch: Path) -> dict:
    if scratch.exists():
        raise SystemExit(f"scratch restore root exists: {scratch}")
    man = json.loads((backup_dir / "backup-manifest.json").read_text())
    scratch.mkdir(parents=True)
    shutil.copy2(backup_dir / "mlflow.db", scratch / "mlflow.db")
    shutil.copytree(backup_dir / "artifacts", scratch / "artifacts")
    if sha(scratch / "mlflow.db") != man["db"]["sha256"]:
        raise SystemExit("restored DB checksum differs from backup manifest")
    got = tree_manifest(scratch / "artifacts")
    if got != man["artifacts"]:
        diff = sorted(k for k in set(got) | set(man["artifacts"]) if got.get(k) != man["artifacts"].get(k))[:5]
        raise SystemExit(f"restored artifacts differ from backup manifest: {diff}")
    counts = db_counts(scratch / "mlflow.db")
    if counts != man["db"]["counts"]:
        raise SystemExit(f"restored DB counts {counts} != {man['db']['counts']}")
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=f"sqlite:///{scratch / 'mlflow.db'}")
    exp = client.get_experiment_by_name("IPCCH")
    runs = client.search_runs([exp.experiment_id], max_results=5000) if exp else []
    kinds = {}
    for r in runs:
        kinds[r.data.tags.get("record_kind", "?")] = kinds.get(r.data.tags.get("record_kind", "?"), 0) + 1
    return {"scratch": str(scratch), "db_counts": counts, "artifact_files": len(got),
            "artifact_bytes": sum(v["bytes"] for v in got.values()), "ipcch_runs_by_kind": kinds}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["backup", "restore-check"])
    ap.add_argument("--store", default=str(DEFAULT_ROOT))
    ap.add_argument("--dest", required=True, help="backup dir (backup) or scratch root (restore-check)")
    ap.add_argument("--backup", help="backup dir to restore from (restore-check)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = backup(Path(a.store), Path(a.dest)) if a.command == "backup" else restore_check(Path(a.backup), Path(a.dest))
    print(json.dumps(res, indent=1, sort_keys=True))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
