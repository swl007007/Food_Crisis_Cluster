"""Numerical runtime: deterministic PyTorch settings, version probe and environment identity.

``CUBLAS_WORKSPACE_CONFIG`` is set here, before torch is imported, so it is in
place before any CUDA initialization. Every module that needs torch imports it
from this module.
"""

from __future__ import annotations

import hashlib
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ipcch_mlp.contract import load_runtime_lock  # noqa: E402
from ipcch_mlp.errors import ContractError  # noqa: E402

#: Modules whose code defines the numerical fit; their digest enters every model identity.
NUMERIC_MODULES = ("nets.py", "preprocess.py", "quartets.py", "seeds.py")
PACKAGE_DIR = Path(__file__).resolve().parent


def configure(device: str, threads: int = 4) -> None:
    """Apply the frozen deterministic settings (design section 9)."""
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ContractError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 before CUDA initialization")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(int(threads))
    if device == "cuda" and not torch.cuda.is_available():
        raise ContractError("device cuda requested but CUDA is unavailable")
    if device not in ("cpu", "cuda"):
        raise ContractError(f"unknown device {device!r}")


def code_digest() -> str:
    h = hashlib.sha256()
    for name in NUMERIC_MODULES:
        h.update(name.encode())
        h.update((PACKAGE_DIR / name).read_bytes())
    return h.hexdigest()


def probe(lock: dict | None = None) -> dict:
    lock = lock or load_runtime_lock()
    observed = {}
    for name in lock["packages"]:
        try:
            observed[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            observed[name] = None
    mismatches = [f"{n}: locked {w}, observed {observed[n]}" for n, w in lock["packages"].items() if observed[n] != w]
    if platform.python_version() != lock["python"]:
        mismatches.append(f"python: locked {lock['python']}, observed {platform.python_version()}")
    if platform.system() != lock["platform_system"]:
        mismatches.append(f"platform: locked {lock['platform_system']}, observed {platform.system()}")
    cuda = torch.cuda.is_available()
    return {
        "lock_version": lock["lock_version"],
        "executable": sys.executable,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": observed,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda,
        "cuda_device": torch.cuda.get_device_name(0) if cuda else None,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "numpy": np.__version__,
        "matches_lock": not mismatches,
        "mismatches": mismatches,
    }


def frozen_device(lock: dict | None = None) -> str:
    lock = lock or load_runtime_lock()
    if lock["device"] is None:
        raise ContractError("runtime-lock device is not frozen yet: run the P0 probe and record the checkpoint first")
    return lock["device"]


def environment_identity(device: str, lock: dict | None = None) -> dict:
    """Identity bound into every model: locked runtime, device and numeric code digest."""
    lock = lock or load_runtime_lock()
    info = probe(lock)
    if not info["matches_lock"]:
        raise ContractError(f"runtime does not match the lock: {info['mismatches']}")
    return {
        "lock": lock["lock_version"],
        "python": info["python"],
        "torch": info["packages"]["torch"],
        "numpy": info["packages"]["numpy"],
        "device": device,
        "cuda_device": info["cuda_device"] if device == "cuda" else None,
        "threads": lock["numerics"]["cpu_threads"],
        "numeric_code_sha256": code_digest(),
    }
