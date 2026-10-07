"""IPCCH fixed-map yearly GeoXGB experiment (Trellis task 10-07-ipcch-fixed-map-yearly-xgb, design v1.0).

Independent package: runtime imports are ``ipcch_yearly_xgb.*`` plus pinned
external libraries only. Reused P6 logic is copied with attribution; nothing
imports ``ipcch_geoxgb`` or another sibling experiment, and there is no
``sys.path`` injection.
"""

from __future__ import annotations

from pathlib import Path

__version__ = "1.0.0"

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PACKAGE_ROOT / "config"
