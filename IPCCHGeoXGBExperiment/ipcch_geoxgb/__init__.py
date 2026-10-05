"""IPCCH cumulative-share GeoXGB experiment.

Independent package (R50): runtime imports are package-qualified
(``ipcch_geoxgb.*``) plus pinned external libraries only. Nothing here imports
a sibling experiment package, the old GeoRF ZIP backend, or bare repository
``config``/``src`` modules.

P0 status: configuration, runtime probe and read-only input preflight are
implemented. Scientific phases (prepare, learn-map, predict, report) are not
ported yet and fail explicitly.
"""

from __future__ import annotations

from pathlib import Path

__version__ = "0.1.0.dev0+p0"

#: ``IPCCHGeoXGBExperiment/`` -- holds ``config/``, ``runs/`` and ``tests/``.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PACKAGE_ROOT / "config"
RUNS_DIR = PACKAGE_ROOT / "runs"
