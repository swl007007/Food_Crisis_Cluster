"""assistant_patch.py: Gateway assistant self-calls go to the local URL with the read-only login (MLflow 3.17.0).

Run: /home/swl007007/.venvs/ipcch-mlflow/bin/python -m unittest discover -s IPCCHMLflow/tests -v
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
PROBE = r'''
import asyncio, base64, json, os, sys
sys.path.insert(0, sys.argv[1])
from mlflow.assistant.providers.mlflow_gateway import MlflowGatewayProvider
seen = {}
async def fake(self, prompt, tracking_uri, *a, **kw):
    seen["tracking_uri"] = tracking_uri
    yield None
MlflowGatewayProvider.astream = fake
MlflowGatewayProvider._ipcch_patched = False   # the venv .pth may already have applied the installed copy
import assistant_patch                                    # applies only if IPCCH_ASSISTANT_SELF_URL is set
p = MlflowGatewayProvider()
async def run():
    async for _ in p.astream("q", "https://mlflow.weilunshi.com"):
        pass
asyncio.run(run())
h = p._auth_headers(None)
print(json.dumps({"tracking_uri": seen["tracking_uri"], "headers": h,
                  "chat_url": p._chat_url_builder(None, seen["tracking_uri"])}))
'''


def probe(env_extra: dict) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("IPCCH_ASSISTANT", "MLFLOW_TRACKING_"))}
    env.update(env_extra, MLFLOW_DISABLE_AGENT_HINT="1")
    out = subprocess.run([sys.executable, "-c", PROBE, str(HERE)], env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


class AssistantPatchTest(unittest.TestCase):
    def test_inactive_without_env(self):
        r = probe({})
        self.assertEqual(r["tracking_uri"], "https://mlflow.weilunshi.com")
        self.assertEqual(r["headers"], {})

    def test_local_url_and_readonly_login(self):
        r = probe({"IPCCH_ASSISTANT_SELF_URL": "http://127.0.0.1:5000/", "MLFLOW_TRACKING_USERNAME": "assistant",
                   "MLFLOW_TRACKING_PASSWORD": "pw"})
        self.assertEqual(r["tracking_uri"], "http://127.0.0.1:5000")
        self.assertEqual(r["chat_url"], "http://127.0.0.1:5000/gateway/mlflow/v1/chat/completions")
        self.assertEqual(r["headers"], {"X-MLflow-Authorization": "Basic YXNzaXN0YW50OnB3"})   # assistant:pw

    def test_missing_thought_signature_gets_placeholder(self):
        import assistant_patch
        payload = {"contents": [
            {"role": "user", "parts": [{"text": "q"}]},
            {"role": "model", "parts": [{"functionCall": {"name": "Bash", "args": {}}},
                                        {"functionCall": {"name": "Bash", "args": {}}}]},
            {"role": "model", "parts": [{"functionCall": {"name": "Bash", "args": {}}, "thoughtSignature": "real"}]}]}
        out = assistant_patch.fill_thought_signatures(payload)["contents"]
        self.assertEqual(out[1]["parts"][0]["thoughtSignature"], assistant_patch.SKIP_SIGNATURE)
        self.assertNotIn("thoughtSignature", out[1]["parts"][1])            # only the first call of a turn
        self.assertEqual(out[2]["parts"][0]["thoughtSignature"], "real")      # real signatures are kept
        self.assertNotIn("thoughtSignature", out[0]["parts"][0])

    def test_read_only_tool_allowlist(self):
        import assistant_patch
        ok = ["mlflow experiments search --max-results 50", "mlflow runs list --experiment-id 2",
              "mlflow runs describe --run-id abc", "mlflow experiments get -x 2 --output json",
              "mlflow artifacts list --run-id abc --artifact-path source", "mlflow runs describe --run-id=abc", "mlflow --help"]
        bad = ["python3 -c 'print(1)'", "mlflow run https://github.com/x/y", "mlflow server --port 5002",
               "mlflow experiments delete --experiment-id 2", "mlflow artifacts download --run-id abc -d /tmp",
               "mlflow artifacts list --artifact-uri file:///home", "mlflow runs describe --run-id",
               "mlflow gc --backend-store-uri sqlite:///x.db", "mlflow runs describe --run-id abc; rm -rf /", "ls /"]
        ok += ['ipcch-model search --family geoxgb_reference --lead 03', 'ipcch-model get "AlphaEarth supervised partition | pooled | 1-month | seed 42"',
               'ipcch-model features m-123', 'ipcch-model registered "IPCCH x | y | 3-month"', "ipcch-model search --family=x"]
        bad += ["ipcch-model", "ipcch-model delete x", "ipcch-model get", "ipcch-model get a b", "ipcch-model search --evil x",
                "ipcch-model search --family"]
        for c in ok:
            self.assertIsNone(assistant_patch.tool_denial("Bash", {"command": c}), c)
        for c in bad:
            self.assertIsNotNone(assistant_patch.tool_denial("Bash", {"command": c}), c)
        for tool in ("Read", "Write", "Edit"):
            self.assertIsNotNone(assistant_patch.tool_denial(tool, {"file_path": "/etc/passwd"}))

    def test_catalog_tool_passes_the_executor_check(self):
        out = subprocess.run([sys.executable, "-c", (
            "import sys, json; sys.path.insert(0, sys.argv[1]); import assistant_patch; assistant_patch.apply('http://127.0.0.1:5000');"
            "import mlflow.assistant.providers.tool_executor as te; from mlflow.assistant.config import PermissionsConfig as P;"
            "print(json.dumps([te.static_permission_error('Bash', {'command': c}, P(allow_edit_files=False), None) for c in "
            "['ipcch-model get x', 'mlflow runs list', 'rm -rf /']]))"), str(HERE)],
            env={**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"}, capture_output=True, text=True, check=True)
        allowed = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertEqual(allowed[:2], [None, None])            # the executor runs ipcch-model (argv, no shell)
        self.assertIsNotNone(allowed[2])

    def test_no_credentials_no_header(self):
        r = probe({"IPCCH_ASSISTANT_SELF_URL": "http://127.0.0.1:5000"})
        self.assertEqual(r["headers"], {})


if __name__ == "__main__":
    unittest.main()
