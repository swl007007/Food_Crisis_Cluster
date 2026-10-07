"""Runtime probe and environment identity (P6 lock; source inventory digests).

The original package's environment identity hashed only quartet.py. Here every
model identity binds the digest of all fit-defining modules and configs
(quartet, modelstore, sources, schedule, engine, contract and the three config
files); every run manifest additionally records the complete package source
inventory, so report/replay/CLI code is pinned per run without making a
report-only edit invalidate saved models.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from importlib import metadata
from pathlib import Path

from ipcch_yearly_xgb import CONFIG_DIR, PACKAGE_ROOT
from ipcch_yearly_xgb.contract import load_runtime_lock
from ipcch_yearly_xgb.errors import ContractError

PACKAGE_DIR = Path(__file__).resolve().parent


def probe(lock: dict | None = None) -> dict:
    lock = lock or load_runtime_lock()
    observed = {}
    for name in lock["packages"]:
        try:
            observed[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            observed[name] = None
    bad = [f"{n}: locked {w}, observed {observed[n]}" for n, w in lock["packages"].items() if observed[n] != w]
    if platform.python_version() != lock["python"]:
        bad.append(f"python {platform.python_version()} != {lock['python']}")
    if platform.system() != lock["platform_system"]:
        bad.append(f"platform {platform.system()} != {lock['platform_system']}")
    return {"lock_version": lock["lock_version"], "executable": sys.executable, "python": platform.python_version(),
            "platform": platform.platform(), "packages": observed, "matches_lock": not bad, "mismatches": bad}


def source_inventory() -> dict:
    """SHA256 of every package module and config file (sorted relative paths)."""
    files = sorted(PACKAGE_DIR.glob("*.py")) + sorted(CONFIG_DIR.glob("*.json"))
    return {str(p.relative_to(PACKAGE_ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files}


FIT_SOURCES = ("ipcch_yearly_xgb/quartet.py", "ipcch_yearly_xgb/modelstore.py", "ipcch_yearly_xgb/sources.py",
               "ipcch_yearly_xgb/schedule.py", "ipcch_yearly_xgb/engine.py", "ipcch_yearly_xgb/contract.py",
               "config/yearly-contract.json", "config/inputs.json", "config/runtime-lock.json")


def code_digest(subset=FIT_SOURCES) -> str:
    inv = source_inventory()
    if subset is not None:
        missing = [k for k in subset if k not in inv]
        if missing:
            raise ContractError(f"fit source files missing: {missing}")
        inv = {k: inv[k] for k in subset}
    return hashlib.sha256("\n".join(f"{k} {v}" for k, v in sorted(inv.items())).encode()).hexdigest()


def environment_identity(lock: dict | None = None) -> dict:
    lock = lock or load_runtime_lock()
    info = probe(lock)
    if not info["matches_lock"]:
        raise ContractError(f"runtime does not match the lock: {info['mismatches']}")
    return {"lock": lock["lock_version"], "python": info["python"], "xgboost": info["packages"]["xgboost"],
            "numpy": info["packages"]["numpy"], "pandas": info["packages"]["pandas"],
            "fit_source_sha256": code_digest()}
