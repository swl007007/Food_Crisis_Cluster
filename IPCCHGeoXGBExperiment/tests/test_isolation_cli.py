"""Import independence, CLI behaviour and run-directory discipline."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ipcch_geoxgb import PACKAGE_ROOT, RUNS_DIR
from ipcch_geoxgb.artifacts import new_run_dir, write_json
from ipcch_geoxgb.cli import PENDING_PHASES
from ipcch_geoxgb.errors import ContractError

PACKAGE_DIR = PACKAGE_ROOT / "ipcch_geoxgb"
REPO_ROOT = PACKAGE_ROOT.parent
ALLOWED_TOP_LEVEL = {"numpy", "pandas", "geopandas", "shapely", "pyogrio", "pyproj", "xgboost", "ipcch_geoxgb"}
FORBIDDEN_MODULES = {"config", "config_visual", "src", "prepare_data", "baseline_runtime", "run_pipeline", "report_results"}

PROBE = r"""
import importlib, json, pkgutil, sys
import ipcch_geoxgb
names = [m.name for m in pkgutil.iter_modules(ipcch_geoxgb.__path__, "ipcch_geoxgb.") if not m.name.endswith("__main__")]
for name in names:
    importlib.import_module(name)
files = {k: getattr(v, "__file__", None) for k, v in list(sys.modules.items())}
print(json.dumps({"loaded": names, "files": files, "path0": sys.path[0]}))
"""


def _run(args, cwd):
    env = dict(os.environ, PYTHONPATH=str(PACKAGE_ROOT))
    return subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=300)


def _external_files(files):
    """Loaded module files inside the repository but outside this package."""
    out = []
    for name, path in files.items():
        if not path:
            continue
        resolved = Path(path).resolve()
        if REPO_ROOT in resolved.parents and PACKAGE_ROOT not in resolved.parents:
            out.append((name, str(resolved)))
    return out


@pytest.mark.parametrize("where", ["repo_root", "unrelated"])
def test_import_probe_has_no_legacy_modules(tmp_path, where):
    cwd = REPO_ROOT if where == "repo_root" else tmp_path
    result = _run(["-c", PROBE], cwd)
    assert result.returncode == 0, result.stderr
    probe = json.loads(result.stdout)
    assert {"ipcch_geoxgb.cli", "ipcch_geoxgb.preflight", "ipcch_geoxgb.geography"} <= set(probe["loaded"])
    assert not FORBIDDEN_MODULES & set(probe["files"])
    assert _external_files(probe["files"]) == []
    # Importing every module (including preflight) never loads XGBoost.
    assert "xgboost" not in probe["files"]


def test_import_probe_detects_a_legacy_import():
    """Negative control: the probe is not vacuous; a bare repo-root import is caught."""
    result = _run(["-c", "import config\n" + PROBE], REPO_ROOT)
    assert result.returncode == 0, result.stderr
    files = json.loads(result.stdout.strip().splitlines()[-1])["files"]
    assert "config" in FORBIDDEN_MODULES & set(files)
    assert any(name == "config" for name, _ in _external_files(files))


@pytest.mark.parametrize("where", ["repo_root", "unrelated"])
def test_cli_validate_config_from_any_directory(tmp_path, where):
    cwd = REPO_ROOT if where == "repo_root" else tmp_path
    result = _run(["-m", "ipcch_geoxgb", "validate-config"], cwd)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "passed"


@pytest.mark.parametrize("command", sorted(PENDING_PHASES))
def test_unported_phase_fails_explicitly_and_writes_nothing(tmp_path, command):
    before = sorted(p.name for p in RUNS_DIR.iterdir()) if RUNS_DIR.exists() else []
    result = _run(["-m", "ipcch_geoxgb", command, "--run-id", "should-not-exist"], tmp_path)
    assert result.returncode == 3
    assert "NOT IMPLEMENTED" in result.stderr and result.stdout == ""
    after = sorted(p.name for p in RUNS_DIR.iterdir()) if RUNS_DIR.exists() else []
    assert after == before and list(tmp_path.iterdir()) == []


def test_strict_commands_reject_unknown_arguments(tmp_path):
    assert _run(["-m", "ipcch_geoxgb", "validate-config", "--fit"], tmp_path).returncode == 2


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise AssertionError(f"relative import in {path.name}")
            yield node.module


def test_static_imports_are_stdlib_pinned_or_package():
    stdlib = set(sys.stdlib_module_names)
    for path in PACKAGE_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "sys.path" not in text and "spec_from_file_location" not in text, path.name
        for module in _imports(path):
            top = module.split(".")[0]
            assert top in stdlib or top in ALLOWED_TOP_LEVEL, f"{path.name} imports {module}"
            assert top not in FORBIDDEN_MODULES, f"{path.name} imports {module}"


def test_run_dir_is_new_and_contained(tmp_path):
    made = new_run_dir("r1", tmp_path)
    assert made == (tmp_path / "r1").resolve()
    for bad in ("r1", "../escape", "a/b", "", ".hidden"):
        with pytest.raises(ContractError):
            new_run_dir(bad, tmp_path)


def test_write_json_never_overwrites(tmp_path):
    target = tmp_path / "x.json"
    assert len(write_json(target, {"a": 1})) == 64
    with pytest.raises(ContractError):
        write_json(target, {"a": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
