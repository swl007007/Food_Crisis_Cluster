"""Persistence-correction 2-layer feasibility experiment (isolated package).

This package is a sibling of ``Step3ExpertCorrectionExperiment`` and reuses its
utilities **by import only**.  Nothing under ``Step3ExpertCorrectionExperiment/``,
``paper_reproducibility_package/``, ``archived/``, ``other_outputs/``, ``src/``,
``app/`` or any ``result_*`` directory may be created, modified or deleted here.

Importing this package makes ``step3correction`` importable without requiring the
caller to set ``PYTHONPATH``, mirroring the ``sys.path`` bootstrap that the Step 3
test module already uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Repository root (``.../Food_Crisis_Cluster``).
ROOT = Path(__file__).resolve().parents[2]
#: The Step 3 experiment directory, whose ``step3correction`` package we import.
STEP3_DIR = ROOT / "Step3ExpertCorrectionExperiment"

if STEP3_DIR.is_dir() and str(STEP3_DIR) not in sys.path:
    sys.path.insert(0, str(STEP3_DIR))

__all__ = ["ROOT", "STEP3_DIR"]
