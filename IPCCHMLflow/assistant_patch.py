"""MLflow 3.17.0 compatibility patch: Gateway-backed UI assistant behind basic-auth and a tunnel.

The assistant's Gateway provider calls this same server for every chat turn
(``<page base URL>/gateway/mlflow/v1/chat/completions``) and hands that base URL to its ``mlflow``
tool commands, without MLflow credentials. Under basic-auth that call is 401, and for a page opened
through the Cloudflare tunnel the base URL is the public host, so the call would leave the machine.

When ``IPCCH_ASSISTANT_SELF_URL`` is set (``manage.sh`` sets it only for the server it starts), the
Gateway provider instead (1) always uses that local URL for its chat call and tool commands and
(2) sends the server-side read-only credentials (``MLFLOW_TRACKING_USERNAME`` / ``_PASSWORD``, the
``assistant`` user) in ``X-MLflow-Authorization``, the header MLflow reads on /gateway/ routes.
It also fills Gemini 3's ``thoughtSignature`` on replayed tool calls: the assistant client keeps only a
tool call's name and arguments in its history, so the next request has no signature and Gemini 3
rejects it (400 "Function call is missing a thought_signature"). When the first function-call part of a
model turn has none, the Gateway's Gemini request gets Google's documented placeholder
``skip_thought_signature_validator`` (applies to this server's Gemini/Vertex Gateway calls only).
Tools are narrowed for a multi-user (tunnel) server: MLflow 3.17 runs a user-approved, otherwise-denied
command with full access (any shell command on this host), and even plain ``mlflow`` commands include
``mlflow run``/``server``. With the patch the assistant gets no approval prompts and its only tool is a
read-only ``mlflow`` allowlist (``READ_ONLY``, options checked per subcommand); everything else is denied
back to the model. File tools stay off. The read-only catalog tool ``ipcch-model`` (``ipcch_model.py``) is also
allowed, and the system prompt gets a short guide to where model cards, features and metrics live (``GUIDE``). ``mlflow runs describe`` (the main read tool) fails in 3.17 on runs
with logged-model inputs (``LoggedModelInput is not JSON serializable``); the patch makes its JSON output fall
back to ``str`` for such objects.
Any other MLflow version is left untouched. ``manage.sh`` copies this file into the venv as
``ipcch_mlflow_assistant_patch.py`` with a one-line ``.pth`` loader; delete both to remove it.
"""
import base64
import json
import logging
import os
import shlex

PATCHED_VERSION = "3.17.0"
SKIP_SIGNATURE = "skip_thought_signature_validator"
# (group, command) -> allowed options; True = the option takes a value
READ_ONLY = {
    ("experiments", "search"): {"-v": True, "--view": True, "--max-results": True},
    ("experiments", "get"): {"-x": True, "--experiment-id": True, "-n": True, "--experiment-name": True, "--output": True},
    ("runs", "list"): {"--experiment-id": True, "-v": True, "--view": True},
    ("runs", "describe"): {"--run-id": True},
    ("artifacts", "list"): {"--run-id": True, "--artifact-path": True},
}
# ipcch-model subcommand -> (allowed options, number of positional arguments)
CATALOG = {
    "search": ({"--experiment": True, "--family": True, "--arm": True, "--lead": True, "--feature-set": True,
                "--max-results": True}, 0),
    "get": ({}, 1), "registered": ({}, 1), "features": ({}, 1), "families": ({}, 0),
}
GUIDE = """

## IPCCH catalog guide (read this before answering)
This server is the IPCCH food-crisis research catalog. Experiments: 'IPCCH dashboard' (id 2) and 'IPCCH forecasting
dashboard' (id 4) hold one row and one external model per family x arm x lead (x seed or snapshot); 'IPCCH detailed
runs' (id 1) and 'IPCCH forecasting detailed runs' (id 3) are the source records with every saved metric.
For ANY question about a model's environment, settings/hyperparameters, input features, feature engineering or
architecture, FIRST run `ipcch-model get "<words from the model name>"` (e.g. `ipcch-model get "q3 optimization
share_history_residual 3-month"`) and answer from the card it prints; dashboard run descriptions do NOT contain these
details. If several models match, rerun with one exact name from the list. Other steps: `ipcch-model families` (all family / arm / lead keys), then
`ipcch-model search --family F [--arm A] [--lead 3]` (exact model names), then `ipcch-model get "<model name>"` (the full
card: identity, data, features, feature engineering, architecture, settings, environment, evaluation, sources); `ipcch-model features "<model name>"` prints
the full ordered input-feature list, `ipcch-model registered "<registered model name>"` prints the registered model.
Metrics and row tags: `mlflow runs list --experiment-id 2|4` and `mlflow runs describe --run-id ID`.
Only these read-only commands work (no python, no files, no writes). Quote the card; if it says "not recorded",
say so instead of guessing.
"""


def tool_denial(tool_name: str, tool_input) -> str | None:
    """None if the call is a read-only allowlisted ``mlflow`` command, else the denial message."""
    deny = ("Permission denied: this assistant may only run read-only commands: "
            + "; ".join(f"mlflow {g} {c}" for g, c in READ_ONLY) + "; "
            + "; ".join(f"ipcch-model {c}" for c in CATALOG) + ".")
    if tool_name != "Bash" or not isinstance(tool_input, dict) or not isinstance(tool_input.get("command"), str):
        return deny
    try:
        argv = shlex.split(tool_input["command"].strip())
    except ValueError:
        return deny
    if argv[:1] == ["ipcch-model"]:
        spec = CATALOG.get(argv[1]) if len(argv) > 1 else None
        if spec is None:
            return deny
        options, n_positional = spec
        rest, positional = argv[2:], []
        while rest:
            tok = rest.pop(0)
            name = tok.split("=", 1)[0]
            if not tok.startswith("-"):
                positional.append(tok)
            elif name in options and ("=" in tok or (rest and rest.pop(0) is not None)):
                continue
            else:
                return deny
        return None if len(positional) == n_positional else deny
    if argv[:1] != ["mlflow"]:
        return deny
    if argv[1:] in (["--help"], ["--version"]):
        return None
    allowed = READ_ONLY.get(tuple(argv[1:3]))
    if allowed is None:
        return deny
    rest = argv[3:]
    while rest:
        tok = rest.pop(0)
        name, has_value = tok.split("=", 1)[0], "=" in tok
        if name == "--help":
            continue
        if name not in allowed:
            return deny
        if allowed[name] and not has_value:
            if not rest:
                return deny
            rest.pop(0)
    return None
_log = logging.getLogger("ipcch_mlflow_assistant_patch")


def apply(self_url: str) -> bool:
    from importlib.metadata import version

    if version("mlflow") != PATCHED_VERSION:
        _log.warning("assistant patch skipped: mlflow %s != %s", version("mlflow"), PATCHED_VERSION)
        return False
    from mlflow.assistant.providers.mlflow_gateway import MlflowGatewayProvider

    if getattr(MlflowGatewayProvider, "_ipcch_patched", False):
        return True
    astream = MlflowGatewayProvider.astream

    def local_astream(self, prompt, tracking_uri, *args, **kwargs):
        return astream(self, prompt, self_url, *args, **kwargs)

    def auth_headers(self, api_key):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        user, password = os.environ.get("MLFLOW_TRACKING_USERNAME"), os.environ.get("MLFLOW_TRACKING_PASSWORD")
        if user and password:
            token = base64.b64encode(f"{user}:{password}".encode()).decode()
            headers["X-MLflow-Authorization"] = f"Basic {token}"
        return headers

    from mlflow.gateway.providers.gemini import GeminiAdapter

    chat_to_model = GeminiAdapter.chat_to_model.__func__

    def chat_to_model_with_signatures(cls, payload, config):
        return fill_thought_signatures(chat_to_model(cls, payload, config))

    GeminiAdapter.chat_to_model = classmethod(chat_to_model_with_signatures)
    import mlflow.assistant.providers.openai_compatible as oc
    from mlflow.assistant.config import PermissionsConfig

    execute_tool = oc.execute_tool

    async def read_only_execute_tool(tool_name, tool_input, cwd=None, tracking_uri=None, permissions=None):
        if (denial := tool_denial(tool_name, tool_input)) is not None:
            return denial, True
        return await execute_tool(tool_name, tool_input, cwd=None, tracking_uri=tracking_uri,
                                  permissions=PermissionsConfig(allow_edit_files=False, full_access=False))

    oc.ASSISTANT_SYSTEM_PROMPT = oc.ASSISTANT_SYSTEM_PROMPT + GUIDE.replace("{", "{{").replace("}", "}}")
    import mlflow.assistant.providers.tool_executor as te
    te._ALLOWED_BASH_COMMANDS.add("ipcch-model")                 # executor re-checks argv[0]; run without a shell
    oc.execute_tool = read_only_execute_tool                    # every tool call goes through the allowlist
    oc.static_permission_error = lambda *args, **kwargs: None   # so no call is gated behind an approval prompt
    import mlflow.runs

    class _Json:                                   # json module for mlflow.runs whose dumps tolerates entities
        def __getattr__(self, name):
            return getattr(json, name)

        @staticmethod
        def dumps(obj, **kwargs):
            kwargs.setdefault("default", str)
            return json.dumps(obj, **kwargs)

    mlflow.runs.json = _Json()
    MlflowGatewayProvider.astream = local_astream
    MlflowGatewayProvider._auth_headers = auth_headers
    MlflowGatewayProvider._ipcch_patched = True
    return True


def fill_thought_signatures(gemini_payload: dict) -> dict:
    for content in gemini_payload.get("contents") or []:
        calls = [part for part in content.get("parts") or [] if "functionCall" in part]
        if content.get("role") == "model" and calls and not any("thoughtSignature" in part for part in calls):
            calls[0]["thoughtSignature"] = SKIP_SIGNATURE
    return gemini_payload


if os.environ.get("IPCCH_ASSISTANT_SELF_URL"):
    try:
        apply(os.environ["IPCCH_ASSISTANT_SELF_URL"].rstrip("/"))
    except Exception:          # never break interpreter start-up
        _log.exception("assistant patch failed")
