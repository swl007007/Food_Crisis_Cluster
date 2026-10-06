"""IPCCH cumulative-share GeoXGB experiment -- climate feature perturbation.

Sibling copy of ``IPCCHGeoXGBExperiment`` (frozen at 6798df2) for Trellis task
``10-05-ipcch-climate-perturbation``. Only the feature construction (rich601:
the four original monthly climate columns replaced by IPCCH_shared_folder
monthly ensmean values plus as-of growing-season aggregates), schema, input
registry and an optional run-root override differ from the original.


Independent package (R50): runtime imports are package-qualified
(``ipcch_climate_geoxgb.*``) plus pinned external libraries only. Nothing here imports
a sibling experiment package, the old GeoRF ZIP backend, or bare repository
``config``/``src`` modules.

P0 status: configuration, runtime probe and read-only input preflight are
implemented. Scientific phases (prepare, learn-map, predict, report) are not
ported yet and fail explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path

__version__ = "0.1.0.dev0+p0"

#: ``IPCCHClimateGeoXGBExperiment/`` -- holds ``config/``, ``runs/`` and ``tests/``.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PACKAGE_ROOT / "config"
#: Run root; ``IPCCH_CLIMATE_RUNS_DIR`` moves scratch runs outside Dropbox.
RUNS_DIR = Path(os.environ["IPCCH_CLIMATE_RUNS_DIR"]) if os.environ.get("IPCCH_CLIMATE_RUNS_DIR") else PACKAGE_ROOT / "runs"
