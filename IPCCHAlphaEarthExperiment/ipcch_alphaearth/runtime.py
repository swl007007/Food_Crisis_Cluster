"""Runtime probe against config/runtime-lock.json (distribution metadata only; no numerical imports)."""

from __future__ import annotations

import os
import platform
import sys
from importlib import metadata

from ipcch_alphaearth.contract import load_runtime_lock


def probe_numerical(lock: dict | None = None) -> dict:
    lock = (lock or load_runtime_lock())["numerical"]
    observed = {}
    for name in lock["packages"]:
        try:
            observed[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            observed[name] = None
    python = platform.python_version()
    mismatches = [f"{n}: locked {v}, observed {observed[n]}" for n, v in lock["packages"].items() if observed[n] != v]
    if python != lock["python"]:
        mismatches.append(f"python: locked {lock['python']}, observed {python}")
    if platform.system() != lock["platform_system"]:
        mismatches.append(f"platform: locked {lock['platform_system']}, observed {platform.system()}")
    threads = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    return {"executable": sys.executable, "python": python, "python_build": sys.version, "platform": platform.platform(),
            "packages": observed, "thread_env": threads, "cpu_count": os.cpu_count(),
            "matches_lock": not mismatches, "mismatches": mismatches}


def environment_identity() -> dict:
    """Compact identity bound into every fit record."""
    probe = probe_numerical()
    return {"python": probe["python"], "packages": probe["packages"], "executable": probe["executable"]}
