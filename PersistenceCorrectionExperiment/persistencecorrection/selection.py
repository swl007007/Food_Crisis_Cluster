"""Phase 4: select and freeze the single override threshold.

PRD R13 as revised by DECISIONS_LOG C5: ``tau`` is selected on the **2020**
window only and frozen before any 2021-2024 row is scored (R24).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prf

from .override import apply_override


def crisis_f1(y_true, y_pred):
    """Crisis-class (class 1) F1 -- the objective throughout (R17)."""
    return float(
        prf(y_true, y_pred, labels=[1], average=None, zero_division=0)[2][0]
    )


def select_threshold(persistence, p_cal, y_true):
    """Sweep ``tau`` and return the argmax-F1 threshold.

    Ties resolve to the smallest ``tau`` (ascending sweep, first wins) --
    the Step 3 convention.
    """
    persistence = np.asarray(persistence).astype(int)
    p_cal = np.asarray(p_cal, dtype=float)
    y_true = np.asarray(y_true).astype(int)

    baseline = crisis_f1(y_true, persistence)
    candidates = np.unique(p_cal)
    best_tau, best_f1 = None, baseline
    trace = []
    for tau in candidates:  # np.unique returns ascending; first strict win holds
        f1 = crisis_f1(y_true, apply_override(persistence, p_cal, tau))
        trace.append((float(tau), f1))
        if f1 > best_f1:
            best_tau, best_f1 = float(tau), f1
    return {
        "tau": best_tau,
        "selected_f1": best_f1,
        "baseline_persistence_f1": baseline,
        "improves_on_baseline": best_tau is not None,
        "n_candidates": int(len(candidates)),
        "trace": trace,
    }


def freeze(selections, input_hashes, path):
    """Write the frozen thresholds plus their input hashes, refusing overwrite."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; frozen thresholds are immutable")
    payload = {
        "frozen_thresholds": selections,
        "selection_window": "2020",
        "selection_inputs_sha256": input_hashes,
        "contract": (
            "tau selected on 2020 only; frozen before any 2021-2024 row is scored "
            "(PRD R13 as revised by DECISIONS_LOG C5, and R24)"
        ),
    }
    body = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(body, encoding="utf-8")
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
