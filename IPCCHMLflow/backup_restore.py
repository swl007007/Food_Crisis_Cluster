"""Backup and scratch-restore check for the local IPCCH MLflow store.

  backup   SQLite online backup API + byte copy of artifacts/ + sha256 manifest
  restore-check  copy a backup into a new scratch root, check it against the manifest,
           start a temporary server on its own localhost port over the scratch DB and
           artifacts, download known artifacts through it and compare hashes, stop it.
           The live server is never used for this check.

Hold the import lock while backing up so no importer writes mid-copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from import_runs import CHUNK, DEFAULT_ROOT, Lock  # noqa: E402

MLFLOW = Path(sys.executable).with_name("mlflow")
LIVE_PORT = 5000


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


def scratch_server(scratch: Path, port: int):
    if port == LIVE_PORT:
        raise SystemExit("refusing to use the live server port for a scratch restore check")
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(f"port {port} is already in use")
    log = open(scratch / "scratch-server.log", "ab")
    proc = subprocess.Popen([str(MLFLOW), "server", "--backend-store-uri", f"sqlite:///{scratch / 'mlflow.db'}",
                             "--artifacts-destination", str(scratch / "artifacts"), "--serve-artifacts",
                             "--host", "127.0.0.1", "--port", str(port), "--workers", "1"],
                            stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"})
    for _ in range(120):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read().strip() == b"OK":
                return proc
        except OSError:
            pass
        if proc.poll() is not None:
            break
        time.sleep(1)
    proc.terminate()
    raise SystemExit(f"scratch server on port {port} did not become healthy (see {scratch / 'scratch-server.log'})")


def download_checks(uri: str, man: dict, experiment: str) -> dict:
    """Every parent's plan-summary manifest and the smallest model bundle, through the scratch server."""
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name(experiment)
    runs = client.search_runs([exp.experiment_id], max_results=5000) if exp else []
    kinds = {}
    for r in runs:
        k = r.data.tags.get("record_kind", "?")
        kinds[k] = kinds.get(k, 0) + 1
    targets = []
    for r in runs:
        if r.data.tags.get("record_kind") != "source_run":
            continue
        base = r.info.artifact_uri.split("mlflow-artifacts:/", 1)[-1].lstrip("/")
        targets.append((r, "manifests/plan-summary.json", f"{base}/manifests/plan-summary.json"))
        targets.append((r, "models.tar", f"{base}/models.tar"))
    tars = sorted((t for t in targets if t[1] == "models.tar"), key=lambda t: man["artifacts"].get(t[2], {}).get("bytes", 1 << 62))
    targets = [t for t in targets if t[1] != "models.tar"] + tars[:1]
    checked = []
    with tempfile.TemporaryDirectory(prefix="ipcch-mlflow-restore-dl-") as tmp:
        for r, art, store_rel in targets:
            want = man["artifacts"].get(store_rel)
            if want is None:
                raise SystemExit(f"{store_rel} not in backup manifest")
            local = Path(client.download_artifacts(r.info.run_id, art, tmp))
            if sha(local) != want["sha256"]:
                raise SystemExit(f"{art} of {r.data.tags.get('source_key')} differs after scratch download")
            checked.append({"source_key": r.data.tags.get("source_key"), "artifact": art, "bytes": want["bytes"],
                            "sha256": want["sha256"]})
            local.unlink()
    return {"ipcch_runs_by_kind": kinds, "downloaded": checked}


def restore_check(backup_dir: Path, scratch: Path, port: int = 5001, experiment: str = "IPCCH") -> dict:
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
    proc = scratch_server(scratch, port)
    try:
        dl = download_checks(f"http://127.0.0.1:{port}", man, experiment)
    finally:
        proc.terminate()
        proc.wait(timeout=60)
    return {"scratch": str(scratch), "scratch_server_port": port, "db_counts": counts, "artifact_files": len(got),
            "artifact_bytes": sum(v["bytes"] for v in got.values()), **dl}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["backup", "restore-check"])
    ap.add_argument("--store", default=str(DEFAULT_ROOT))
    ap.add_argument("--dest", required=True, help="backup dir (backup) or scratch root (restore-check)")
    ap.add_argument("--backup", help="backup dir to restore from (restore-check)")
    ap.add_argument("--port", type=int, default=5001, help="scratch server port (restore-check; never 5000)")
    ap.add_argument("--experiment", default="IPCCH")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = backup(Path(a.store), Path(a.dest)) if a.command == "backup" else \
        restore_check(Path(a.backup), Path(a.dest), a.port, a.experiment)
    print(json.dumps(res, indent=1, sort_keys=True))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
