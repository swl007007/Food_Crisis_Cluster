"""Runtime probe against ``config/runtime-lock.json``.

Versions come from installed distribution metadata, so the probe itself never
imports XGBoost or starts any numerical work.
"""

from __future__ import annotations

import platform
import sys
from importlib import metadata

from ipcch_geoxgb.contract import load_runtime_lock


def probe_runtime(lock: dict | None = None) -> dict:
    lock = lock or load_runtime_lock()
    observed = {}
    for name in lock["packages"]:
        try:
            observed[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            observed[name] = None
    python = platform.python_version()
    system = platform.system()
    mismatches = [
        f"{name}: locked {want}, observed {observed[name]}"
        for name, want in lock["packages"].items()
        if observed[name] != want
    ]
    if python != lock["python"]:
        mismatches.append(f"python: locked {lock['python']}, observed {python}")
    if system != lock["platform_system"]:
        mismatches.append(f"platform: locked {lock['platform_system']}, observed {system}")
    return {
        "lock_version": lock["lock_version"],
        "executable": sys.executable,
        "python": python,
        "platform": platform.platform(),
        "packages": observed,
        "matches_lock": not mismatches,
        "mismatches": mismatches,
    }
