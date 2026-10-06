"""IPCCH fixed-map MLP residual-adaptation experiment (Trellis task 10-05-ipcch-mlp-spatial-adaptation).

Independent package: runtime imports are ``ipcch_mlp.*`` plus pinned external
libraries (numpy, pandas, torch). Nothing here imports ``ipcch_geoxgb`` or other
sibling experiment packages; reused logic is copied with attribution.
"""

from __future__ import annotations

from pathlib import Path

__version__ = "1.0.0"

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PACKAGE_ROOT / "config"
