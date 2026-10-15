"""Empty artifacts read back through an MLflow 3.17 HTTP artifact server (task 10-10 amendment).

``FileInfo.to_proto`` drops a zero ``file_size`` and ``HttpArtifactRepository.list_artifacts`` turns the missing
field into ``None``, so a 0-byte artifact lists with unknown size. ``verify_parent`` accepts that only after
downloading the file and checking size 0 and its planned digest; every other size rule is unchanged.

Run: /home/swl007007/.venvs/ipcch-mlflow/bin/python -m unittest discover -s IPCCHMLflow/tests -v
"""

import copy
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import backup_restore  # noqa: E402
import import_runs  # noqa: E402
from extract import SourceConflict  # noqa: E402
from mlflow.entities import FileInfo  # noqa: E402
from test_import import free_port, make_cfg, make_fixture  # noqa: E402

EMPTY = "task_evidence/logs/attach.out"
NONEMPTY = "task_evidence/logs/run.log"


class Tamper:
    """Pass-through client that can rewrite listed sizes, hide artifacts, replace or fail downloads."""

    def __init__(self, client, sizes=None, drop=(), content=None, fail=()):
        self.c, self.sizes, self.drop, self.content, self.fail = client, sizes or {}, set(drop), content or {}, set(fail)
        self.downloads = []

    def list_artifacts(self, run_id, path=None):
        out = []
        for fi in self.c.list_artifacts(run_id, path):
            if fi.path in self.drop:
                continue
            out.append(FileInfo(fi.path, fi.is_dir, self.sizes[fi.path]) if fi.path in self.sizes else fi)
        return out

    def download_artifacts(self, run_id, path, dst):
        self.downloads.append(path)
        if path in self.fail:
            raise OSError("injected download failure")
        local = Path(self.c.download_artifacts(run_id, path, dst))
        if path in self.content:
            local.write_bytes(self.content[path])
        return str(local)

    def __getattr__(self, name):
        return getattr(self.c, name)


class EmptyArtifactHTTPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="ipcch-mlflow-empty-test-")
        base = cls.base = Path(cls.tmp.name)
        make_fixture(base / "src")
        (base / "evid/logs").mkdir(parents=True)
        (base / "evid/logs/attach.out").write_bytes(b"")
        (base / "evid/logs/run.log").write_bytes(b"hello\n")
        cfg = make_cfg(base)
        cfg["families"][0]["extra"] = [{"base": "{temp}/evid", "paths": ["logs/*"], "prefix": "task_evidence"}]
        cls.sources = base / "sources.json"
        cls.sources.write_text(json.dumps(cfg))
        cls.store = base / "imp"
        server = base / "server"
        (server / "artifacts").mkdir(parents=True)
        port = free_port()
        cls.uri = f"http://127.0.0.1:{port}"
        cls.proc = subprocess.Popen([str(backup_restore.MLFLOW), "server", "--backend-store-uri",
                                     f"sqlite:///{server / 'mlflow.db'}", "--artifacts-destination",
                                     str(server / "artifacts"), "--serve-artifacts", "--host", "127.0.0.1",
                                     "--port", str(port), "--workers", "1"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(120):
            try:
                urllib.request.urlopen(f"{cls.uri}/health", timeout=2)
                break
            except OSError:
                time.sleep(1)
        cls.import_error = None
        try:
            cls.cmd("import")
        except Exception as e:          # recorded; asserted by test_import_and_verify_succeed
            cls.import_error = e
        cls.client = import_runs._client(cls.uri)
        exp = cls.client.get_experiment_by_name(cfg["experiment"])
        cls.parent = import_runs.find_run(cls.client, exp.experiment_id, "p6_geoxgb/fx-run")
        cls.plan = import_runs.cmd_plan(cfg, cls.store, None, False, None)["p6_geoxgb"]

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(timeout=60)
        cls.tmp.cleanup()

    @classmethod
    def cmd(cls, command, *extra):
        import_runs.main([command, "--sources", str(cls.sources), "--store", str(cls.store),
                          "--tracking-uri", cls.uri, *extra])

    def verify(self, client, deep, plan=None):
        return import_runs.verify_parent(client, self.client.get_run(self.parent.info.run_id), plan or self.plan, deep)

    def test_listing_reports_unknown_size_for_empty_artifact(self):
        arts = import_runs.existing_artifacts(self.client, self.parent.info.run_id)
        self.assertIsNone(arts[EMPTY])                    # installed serialization: 0 -> omitted -> None
        self.assertEqual(arts[NONEMPTY], 6)
        self.assertEqual({r["path"]: r["bytes"] for r in self.plan["extras"]}, {EMPTY: 0, NONEMPTY: 6})

    def test_import_and_verify_succeed(self):
        self.assertIsNone(self.import_error)
        self.cmd("verify")
        self.cmd("verify", "--shallow")
        self.cmd("import")                                # complete parent: verified no-op

    def test_unknown_empty_checked_by_download_only_where_needed(self):
        t = Tamper(self.client)
        self.verify(t, deep=False)
        self.assertEqual(t.downloads, [EMPTY])            # shallow: only the unknown-size empty file is read
        t = Tamper(self.client)
        out = self.verify(t, deep=True)
        self.assertEqual(len(t.downloads), len(set(t.downloads)))   # deep: each artifact downloaded once
        self.assertIn(EMPTY, t.downloads)
        self.assertEqual(out["artifacts"], len(t.downloads))

    def test_unknown_empty_with_nonempty_readback_refused(self):
        for deep in (False, True):
            with self.assertRaisesRegex(SourceConflict, "attach.out listed size unknown; readback 1 bytes"):
                self.verify(Tamper(self.client, content={EMPTY: b"x"}), deep)

    def test_unknown_empty_with_wrong_digest_refused(self):
        plan = copy.deepcopy(self.plan)
        next(r for r in plan["extras"] if r["path"] == EMPTY)["sha256"] = "0" * 64
        with self.assertRaisesRegex(SourceConflict, "attach.out checksum mismatch after download"):
            self.verify(self.client, False, plan)          # manifests are size-only here; the empty file is read
        with self.assertRaisesRegex(SourceConflict, "checksum mismatch after download"):
            self.verify(self.client, True, plan)           # deep reaches the derived manifests/extras.json first

    def test_missing_empty_artifact_refused(self):
        with self.assertRaisesRegex(SourceConflict, "artifact set mismatch"):
            self.verify(Tamper(self.client, drop={EMPTY}), deep=False)

    def test_download_failure_refused(self):
        for deep in (False, True):
            with self.assertRaisesRegex(OSError, "injected download failure"):
                self.verify(Tamper(self.client, fail={EMPTY}), deep)

    def test_unknown_size_for_planned_nonzero_refused(self):
        for deep in (False, True):
            t = Tamper(self.client, sizes={NONEMPTY: None})
            with self.assertRaisesRegex(SourceConflict, "run.log size None != 6"):
                self.verify(t, deep)
            self.assertNotIn(NONEMPTY, t.downloads)

    def test_known_size_mismatch_refused(self):
        for deep in (False, True):
            with self.assertRaisesRegex(SourceConflict, "run.log size 7 != 6"):
                self.verify(Tamper(self.client, sizes={NONEMPTY: 7}), deep)


if __name__ == "__main__":
    unittest.main()
