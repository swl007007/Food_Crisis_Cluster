"""Exact-count metrics (R8, R13, R16, R25-R27).

Adapted from ``FEWSNETGeoXGBExperiment/src/metrics/fourclass.py`` (fixed-axis
confusion, class counts, exact rational crisis F1, single-column crisis scan
masses) with the R27 undefined-value policy: every metric is computed from
pooled counts, is NA (``None`` plus a reason) exactly when its own denominator
is zero, and legitimate zeros and negative R² are kept. Gates use exact
fractions; an NA F1 never passes a gate.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from ipcch_geoxgb.errors import TechnicalError
from ipcch_geoxgb.targets import FOUR_CLASS_LABELS, four_class

N_CLASSES = 4


def _phases(values) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise TechnicalError("phase vectors must be one-dimensional")
    arr = arr.astype(np.int64)
    if arr.size and (arr.min() < 1 or arr.max() > 5):
        raise TechnicalError("phase outside 1..5")
    return arr


def crisis_counts(truth_phase, pred_phase) -> dict:
    """Binary crisis (phase >= 3) TP/FP/FN/TN over aligned rows."""
    t, p = _phases(truth_phase) >= 3, _phases(pred_phase) >= 3
    if t.shape != p.shape:
        raise TechnicalError("truth and prediction are not aligned")
    return {
        "tp": int(np.sum(t & p)),
        "fp": int(np.sum(~t & p)),
        "fn": int(np.sum(t & ~p)),
        "tn": int(np.sum(~t & ~p)),
    }


def add_counts(*counts: dict) -> dict:
    return {k: int(sum(c[k] for c in counts)) for k in ("tp", "fp", "fn", "tn")}


def exact_f1(counts: dict) -> Fraction | None:
    """Crisis F1 = 2TP / (2TP + FP + FN) as an exact fraction; None if undefined."""
    denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
    return Fraction(2 * counts["tp"], denominator) if denominator else None


def gain_passes(candidate: dict, base: dict, threshold: Fraction) -> tuple[bool, str]:
    """Strict exact gain ``F1(candidate) - F1(base) > threshold``; NA never passes."""
    f_c, f_b = exact_f1(candidate), exact_f1(base)
    if f_c is None or f_b is None:
        return False, "f1_undefined"
    if f_c - f_b > threshold:
        return True, "gain_above_threshold"
    return False, "gain_not_above_threshold"


def _ratio(numerator: int, denominator: int, name: str) -> tuple[float | None, str]:
    if denominator == 0:
        return None, f"{name}: zero denominator"
    return numerator / denominator, ""


def binary_metrics(counts: dict) -> dict:
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    out = {"counts": dict(counts), "na_reasons": {}}
    for name, num, den in (
        ("accuracy", tp + tn, tp + fp + fn + tn),
        ("precision", tp, tp + fp),
        ("recall", tp, tp + fn),
        ("f1", 2 * tp, 2 * tp + fp + fn),
        ("f2", 5 * tp, 5 * tp + 4 * fn + fp),
    ):
        value, reason = _ratio(num, den, name)
        out[name] = value
        if reason:
            out["na_reasons"][name] = reason
    return out


def four_class_confusion(truth_phase, pred_phase) -> np.ndarray:
    t, p = four_class(_phases(truth_phase)), four_class(_phases(pred_phase))
    if t.shape != p.shape:
        raise TechnicalError("truth and prediction are not aligned")
    return np.bincount(t * N_CLASSES + p, minlength=N_CLASSES**2).reshape(N_CLASSES, N_CLASSES)


def four_class_metrics(matrix: np.ndarray) -> dict:
    """Accuracy and macro-F1 on the fixed 1/2/3/4-5 axis (rows truth, cols prediction).

    A class absent from both truth and prediction has undefined F1, which makes
    macro-F1 NA; the axis is never shortened and NA is never averaged away.
    """
    matrix = np.asarray(matrix, dtype=np.int64)
    tp = np.diag(matrix)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    n = int(matrix.sum())
    per_class, reasons = {}, {}
    f1_values = []
    for k, label in enumerate(FOUR_CLASS_LABELS):
        den = int(2 * tp[k] + fp[k] + fn[k])
        f1 = (2 * int(tp[k]) / den) if den else None
        per_class[label] = {
            "support": int(matrix[k].sum()),
            "predicted": int(matrix[:, k].sum()),
            "tp": int(tp[k]),
            "fp": int(fp[k]),
            "fn": int(fn[k]),
            "f1": f1,
        }
        f1_values.append(f1)
    macro = None if any(v is None for v in f1_values) else float(np.mean(f1_values))
    if macro is None:
        absent = [FOUR_CLASS_LABELS[k] for k, v in enumerate(f1_values) if v is None]
        reasons["macro_f1"] = f"class absent from truth and prediction: {absent}"
    accuracy = (int(tp.sum()) / n) if n else None
    if n == 0:
        reasons["accuracy"] = "n = 0"
    return {
        "accuracy": accuracy,
        "macro_f1": macro,
        "per_class": per_class,
        "confusion_rows_truth_cols_pred": matrix.tolist(),
        "na_reasons": reasons,
    }


def r_squared(truth, prediction) -> tuple[float | None, str]:
    """1 - SSE/SST; NA when n < 2 or SST = 0; negative values are kept."""
    y = np.asarray(truth, dtype=np.float64)
    f = np.asarray(prediction, dtype=np.float64)
    if y.shape != f.shape:
        raise TechnicalError("q3 truth and prediction are not aligned")
    if y.size < 2:
        return None, "n < 2"
    sst = float(((y - y.mean()) ** 2).sum())
    if sst == 0.0:
        return None, "constant truth (SST = 0)"
    return 1.0 - float(((y - f) ** 2).sum()) / sst, ""


def metric_panel(truth_phase, pred_phase, q3_true=None, q3_star=None, q3_raw=None) -> dict:
    """The full R8 panel for one cohort of aligned keyed rows."""
    out = {
        "n": int(len(np.asarray(truth_phase))),
        "binary": binary_metrics(crisis_counts(truth_phase, pred_phase)),
        "four_class": four_class_metrics(four_class_confusion(truth_phase, pred_phase)),
    }
    if q3_true is not None:
        for label, values in (("q3_r2_projected", q3_star), ("q3_r2_raw", q3_raw)):
            if values is not None:
                value, reason = r_squared(q3_true, values)
                out[label] = value
                if reason:
                    out[f"{label}_na_reason"] = reason
    return out


def crisis_scan_masses(truth_phase, pred_phase, groups) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Single-column crisis-F1 scan inputs per spatial group (R32).

    D_g = 2TP_g + FP_g + FN_g, D = sum D_g; Y_g = D_g / D and A_g = 2TP_g / D
    (zeros when D = 0). TN contributes no mass. Groups come back in ascending
    numeric order with their raw counts.
    """
    t, p = _phases(truth_phase) >= 3, _phases(pred_phase) >= 3
    groups = np.asarray(groups, dtype=np.int64)
    unique, inverse = np.unique(groups, return_inverse=True)
    tp = np.bincount(inverse, weights=(t & p), minlength=len(unique))
    fp = np.bincount(inverse, weights=(~t & p), minlength=len(unique))
    fn = np.bincount(inverse, weights=(t & ~p), minlength=len(unique))
    exposure = 2 * tp + fp + fn
    total = exposure.sum()
    Y = exposure / total if total > 0 else np.zeros_like(exposure)
    A = 2 * tp / total if total > 0 else np.zeros_like(exposure)
    counts = {"tp": tp.astype(np.int64), "fp": fp.astype(np.int64), "fn": fn.astype(np.int64)}
    return unique, Y.astype(np.float64), A.astype(np.float64), counts
