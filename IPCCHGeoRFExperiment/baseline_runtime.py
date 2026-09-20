"""Isolated extraction, patching and import of the pinned GeoRF baseline.

design.md "Source and runtime preflight" and "Stage1 integration": each run gets
its own extracted copy of `georf-baseline-v0.1.0.zip`. The archive and the
repository's own `src/`/`config.py` are never modified, and the run's imports are
asserted to resolve *into the extracted copy* so a root module cannot shadow it.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

RELEASE_SHA256 = "39a26138e3fafb0be2bbd22e9760095d6cdefa7d79b98b4f798cb3aa79b500a0"

#: The single approved local source patch (design.md "Stage1 integration").
#: `get_refined_partitions_all` is grid-only by its own docstring, yet it runs
#: after s_branch.pkl (:438), branch_table.npy (:439) and X_branch_id.npy (:447)
#: are written, and its result feeds build_terminal (:471). Under polygon
#: contiguity that rewrites assignments the saved state and the trained
#: checkpoints do not know about. Training's polygon refinement is untouched.
PATCH_TARGET = "src/model/GeoRF.py"
PATCH_OLD = """		if CONTIGUITY:
			# Pass vis_dir only if VIS_DEBUG_MODE is enabled for contiguity refinement visualization
			vis_dir_param = self.dir_vis if VIS_DEBUG_MODE else None
			X_branch_id = get_refined_partitions_all(X_branch_id, self.s_branch, X_group, dir = vis_dir_param, min_component_size = MIN_COMPONENT_SIZE, VIS_DEBUG_MODE=VIS_DEBUG_MODE)"""
PATCH_NEW = """		# IPCCH local patch: this refinement is grid-only (see the comment above and
		# partition_opt.get_refined_partitions_all). It runs AFTER s_branch/branch_table/
		# X_branch_id are saved and its output feeds build_terminal, so in polygon mode it
		# would desynchronise the correspondence table from the accepted state and the
		# trained checkpoints. Training's polygon scan/refinement is unaffected.
		if CONTIGUITY and contiguity_type != 'polygon':
			# Pass vis_dir only if VIS_DEBUG_MODE is enabled for contiguity refinement visualization
			vis_dir_param = self.dir_vis if VIS_DEBUG_MODE else None
			X_branch_id = get_refined_partitions_all(X_branch_id, self.s_branch, X_group, dir = vis_dir_param, min_component_size = MIN_COMPONENT_SIZE, VIS_DEBUG_MODE=VIS_DEBUG_MODE)"""


class BaselineError(RuntimeError):
    """Raised when the pinned baseline cannot be established exactly."""


@dataclass
class BaselineRuntime:
    """A verified, patched, run-local copy of the baseline."""

    root: Path
    release_sha256: str
    manifest_version: str
    manifest_source_commit: str
    payload_files_verified: int
    patch_applied: bool
    patch_diff: str
    pristine_target_sha256: str
    patched_target_sha256: str
    module_locations: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = {k: v for k, v in self.__dict__.items()}
        data["root"] = str(self.root)
        return data


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_baseline(zip_path: Path | str, destination: Path | str) -> BaselineRuntime:
    """Verify then extract the pinned release into a fresh directory.

    Checks, in order and all fatal: release SHA256, ZIP CRC of every member, a
    single top-level ``GeoRFBaseline/``, and every MANIFEST payload hash both
    inside the archive and again on disk after extraction.
    """
    zip_path = Path(zip_path)
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise BaselineError(f"{destination} already exists and is not empty")

    actual = _sha256_file(zip_path)
    if actual != RELEASE_SHA256:
        raise BaselineError(
            f"release hash mismatch: expected {RELEASE_SHA256}, got {actual}"
        )

    with zipfile.ZipFile(zip_path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise BaselineError(f"ZIP CRC failure at {corrupt}")

        tops = {name.split("/")[0] for name in archive.namelist()}
        if tops != {"GeoRFBaseline"}:
            raise BaselineError(f"expected one top-level GeoRFBaseline/, found {tops}")

        manifest = json.loads(archive.read("GeoRFBaseline/MANIFEST.json"))
        payload = manifest["files_sha256"]
        for relative, expected in payload.items():
            member = f"GeoRFBaseline/{relative}"
            try:
                data = archive.read(member)
            except KeyError as exc:
                raise BaselineError(f"manifest lists a missing member: {relative}") from exc
            if _sha256_bytes(data) != expected:
                raise BaselineError(f"in-archive payload hash mismatch: {relative}")

        destination.mkdir(parents=True, exist_ok=True)
        archive.extractall(destination)

    root = destination / "GeoRFBaseline"
    for relative, expected in payload.items():
        on_disk = root / relative
        if not on_disk.is_file():
            raise BaselineError(f"extraction lost {relative}")
        if _sha256_file(on_disk) != expected:
            raise BaselineError(f"post-extraction payload hash mismatch: {relative}")

    return BaselineRuntime(
        root=root,
        release_sha256=actual,
        manifest_version=manifest.get("version", ""),
        manifest_source_commit=manifest.get("source_commit", ""),
        payload_files_verified=len(payload),
        patch_applied=False,
        patch_diff="",
        pristine_target_sha256="",
        patched_target_sha256="",
    )


def apply_polygon_refinement_patch(runtime: BaselineRuntime) -> BaselineRuntime:
    """Apply the single approved local patch to the run-local copy.

    Matching the exact pristine block is the safety property: if the released
    text ever differs by one character this raises instead of silently editing
    something else.
    """
    target = runtime.root / PATCH_TARGET
    # A1 records *file identity*, so both hashes are taken over the real bytes on
    # disk. Hashing `read_text(...).encode()` instead would hash newline-normalised
    # content and never match the archive or the file a third party checks.
    runtime.pristine_target_sha256 = _sha256_file(target)
    original = target.read_text(encoding="utf-8")

    occurrences = original.count(PATCH_OLD)
    if occurrences != 1:
        raise BaselineError(
            f"expected exactly one pristine refinement block in {PATCH_TARGET}, "
            f"found {occurrences}; refusing to patch"
        )

    patched = original.replace(PATCH_OLD, PATCH_NEW, 1)
    target.write_text(patched, encoding="utf-8")

    runtime.patched_target_sha256 = _sha256_file(target)
    runtime.patch_applied = True
    runtime.patch_diff = (
        f"--- pristine {PATCH_TARGET}\n+++ patched {PATCH_TARGET}\n"
        f"-{PATCH_OLD.strip().splitlines()[0].strip()}\n"
        f"+{PATCH_NEW.strip().splitlines()[-4].strip()}\n"
    )
    return runtime


@contextmanager
def baseline_imports(runtime: BaselineRuntime, feature_drop_off: bool = True):
    """Import the baseline with the extracted copy first on ``sys.path``.

    Clears external ``PYTHONPATH`` influence for the duration and asserts that
    ``config`` and ``src.*`` actually resolved inside the extracted copy, so a
    repository-root module cannot shadow the pinned one (design.md).
    """
    root = str(runtime.root.resolve())
    saved_path = list(sys.path)
    saved_env = os.environ.get("PYTHONPATH")
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "config" or name == "config_visual" or name.split(".")[0] == "src"
    }
    for name in list(saved_modules):
        del sys.modules[name]
    os.environ.pop("PYTHONPATH", None)
    sys.path.insert(0, root)
    try:
        import config  # noqa: PLC0415 - deliberate late import from the pinned copy

        if feature_drop_off:
            # IPCCH already supplies the exact 93-column schema, so the baseline's
            # own drop list must not remove columns behind our back (design.md).
            # `config.py:254` defines a lowercase `feature_drop = FEATURE_DROP`
            # alias; missing it would leave a live drop list bound to the old value.
            for attribute in (
                "FEATURE_DROP",
                "feature_drop",
                "FEATURE_DROP_LIST",
                "FEATURES_DROP",
            ):
                if hasattr(config, attribute):
                    setattr(config, attribute, [])

        import src  # noqa: PLC0415

        # `src` has no __init__.py, so it is a *namespace* package: its __path__ is
        # assembled from every sys.path entry containing a `src/` directory. The
        # repository root has one, so any submodule ABSENT from the baseline would
        # silently fall through to the root copy (verified: src.model.GeoRF_XGB
        # does exactly that). Pinning __path__ to the baseline closes the hole, so
        # a missing submodule raises instead of resolving to unpinned code.
        baseline_src = str((Path(root) / "src").resolve())
        src.__path__ = [baseline_src]

        import src.model.GeoRF as georf_module  # noqa: PLC0415

        locations = {
            "config": getattr(config, "__file__", ""),
            "src": baseline_src,
            "src.model.GeoRF": getattr(georf_module, "__file__", ""),
        }
        # Check every module the import actually pulled in, not just the two we
        # name: GeoRF.py imports partition_opt, customize, model_RF and others,
        # and any one of them resolving outside the baseline breaks the pin.
        for name, module in list(sys.modules.items()):
            if name != "config" and name.split(".")[0] != "src":
                continue
            location = getattr(module, "__file__", None)
            if location:
                locations[name] = location
        for name, location in locations.items():
            if not location or not Path(location).resolve().is_relative_to(Path(root)):
                raise BaselineError(
                    f"module {name} resolved to {location!r}, outside the extracted "
                    f"baseline at {root}; a root module is shadowing the pinned copy"
                )
        runtime.module_locations = locations
        yield config, georf_module
    finally:
        sys.path[:] = saved_path
        if saved_env is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = saved_env
        for name in list(sys.modules):
            if name == "config" or name == "config_visual" or name.split(".")[0] == "src":
                del sys.modules[name]
        sys.modules.update(saved_modules)


@contextmanager
def working_directory(path: Path | str):
    """Run with ``cwd`` moved, restoring it even on failure.

    GeoRF ignores its constructor output directory in ``create_dir``, so its fit
    must run with cwd isolated under the run's stage1 folder or ``result_GeoRF/``
    and ``checkpoints/`` escape into the workspace (design.md).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(previous)
