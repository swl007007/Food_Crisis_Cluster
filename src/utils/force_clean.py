"""Utilities for aggressively cleaning GeoRF result directories.

The earlier checkpoint-management helpers lived in this module, but they were
removed because the checkpoint system is deprecated.  Only the forced cleanup
logic remains so callers can purge stale outputs before a fresh run.
"""

from __future__ import annotations

import gc
import glob
import os
import shutil
import time


def force_cleanup_directories(*, pipeline: str = "gf") -> None:
    """Remove result directories and common temporary artifacts.

    Parameters
    ----------
    pipeline
        ``"gf"`` for GeoRF, ``"xgb"`` for GeoXGB, or ``"dt"`` for GeoDT.
    """

    print("Performing force cleanup of result directories...")

    if pipeline == "xgb":
        result_dirs = glob.glob("result_GeoXGB*")
        dir_label = "result_GeoXGB*"
    elif pipeline == "dt":
        result_dirs = glob.glob("result_GeoDT*")
        dir_label = "result_GeoDT*"
    else:
        result_dirs = glob.glob("result_GeoRF*")
        dir_label = "result_GeoRF*"

    if not result_dirs:
        print(f"No {dir_label} directories found to clean up.")
        return

    cleaned_count = 0
    failed_count = 0

    for result_dir in result_dirs:
        try:
            print(f"Attempting to delete: {result_dir}")

            if os.path.exists(result_dir):
                for root, dirs, files in os.walk(result_dir):
                    for directory in dirs:
                        os.chmod(os.path.join(root, directory), 0o777)
                    for filename in files:
                        file_path = os.path.join(root, filename)
                        os.chmod(file_path, 0o777)

                shutil.rmtree(result_dir, ignore_errors=True)

                if not os.path.exists(result_dir):
                    print(f"[OK] Successfully deleted: {result_dir}")
                    cleaned_count += 1
                else:
                    print(f"[ERROR] Failed to delete: {result_dir} (still exists)")
                    failed_count += 1

        except Exception as exc:  # pragma: no cover - best-effort cleanup
            print(f"[ERROR] Error deleting {result_dir}: {exc}")
            failed_count += 1

        time.sleep(0.1)

    gc.collect()

    print(f"Cleanup completed: {cleaned_count} deleted, {failed_count} failed")

    temp_patterns = ["temp_*", "*.pkl", "__pycache__"]
    for pattern in temp_patterns:
        for temp_path in glob.glob(pattern):
            try:
                if os.path.isfile(temp_path):
                    os.remove(temp_path)
                elif os.path.isdir(temp_path):
                    shutil.rmtree(temp_path, ignore_errors=True)
                print(f"Cleaned up: {temp_path}")
            except Exception:
                pass
