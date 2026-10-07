"""Attributed copy of IPCCHGeoXGBExperiment/ipcch_geoxgb/metrics.py at 6798df2 (targets.four_class inlined; logic unchanged).

Exact-count metrics (R8, R13, R16, R25-R27).

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

from ipcch_yearly_xgb.errors import TechnicalError

FOUR_CLASS_LABELS = ("1", "2", "3", "4/5")


def four_class(phase: np.ndarray) -> np.ndarray:
    """Phase 1..5 -> class index 0..3 (phase 4 and 5 merge); copied from ipcch_geoxgb/targets.py."""
    phase = np.asarray(phase, dtype=np.int64)
    if phase.size and (phase.min() < 1 or phase.max() > 5):
        raise TechnicalError("phase outside 1..5")
    return np.minimum(phase, 4) - 1

N_CLASSES = 4


def _phases(values) -> np.ndarray:
    """Exact integer phases 1..5 (a 0/NA sentinel is never a scored phase)."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise TechnicalError("phase vectors must be one-dimensional")
    if arr.dtype.kind == "f":
        if not np.isfinite(arr).all() or not np.all(arr == np.round(arr)):
            raise TechnicalError("phase values must be finite integers")
    elif arr.size and arr.dtype.kind not in "iu":
        raise TechnicalError(f"phase values must be numeric integers, got dtype {arr.dtype}")
    arr = arr.astype(np.int64)
    if arr.size and (arr.min() < 1 or arr.max() > 5):
        raise TechnicalError("phase outside 1..5")
    return arr


def _aligned_phases(truth_phase, pred_phase) -> tuple[np.ndarray, np.ndarray]:
    t, p = _phases(truth_phase), _phases(pred_phase)
    if t.shape != p.shape:
        raise TechnicalError(f"truth ({t.shape}) and prediction ({p.shape}) are not aligned")
    return t, p


def _finite_vector(values, name: str, n: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise TechnicalError(f"{name} must be one-dimensional")
    if n is not None and arr.shape[0] != n:
        raise TechnicalError(f"{name} has {arr.shape[0]} rows, expected {n} (same cohort)")
    if not np.isfinite(arr).all():
        raise TechnicalError(f"{name} contains NaN/Inf")
    return arr


def crisis_counts(truth_phase, pred_phase) -> dict:
    """Binary crisis (phase >= 3) TP/FP/FN/TN over aligned rows."""
    t, p = (x >= 3 for x in _aligned_phases(truth_phase, pred_phase))
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
    t, p = (four_class(x) for x in _aligned_phases(truth_phase, pred_phase))
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
    """1 - SSE/SST; NA when n < 2 or the truth is constant; negative values are kept.

    Constancy is decided on the exact values BEFORE any mean is formed: a
    floating mean of identical values can differ from them in the last bit and
    turn SST into ~1e-33, which would produce a meaningless huge negative R².
    If a non-constant SST underflows to 0, or SST/SSE overflow, the ratio is
    not representable in float64 and R² is NA with that reason (never +/-inf).
    """
    y = _finite_vector(truth, "q3 truth")
    f = _finite_vector(prediction, "q3 prediction", n=y.shape[0])
    if y.size < 2:
        return None, "n < 2"
    if np.all(y == y[0]):
        return None, "constant truth (SST = 0)"
    with np.errstate(over="ignore", under="ignore"):
        sst = float(((y - y.mean()) ** 2).sum())
        sse = float(((y - f) ** 2).sum())
    if not (np.isfinite(sst) and np.isfinite(sse)) or sst == 0.0:
        return None, "SST/SSE not representable in float64"
    value = 1.0 - sse / sst
    if not np.isfinite(value):
        return None, "SST/SSE not representable in float64"
    return value, ""


def metric_panel(truth_phase, pred_phase, q3_true=None, q3_star=None, q3_raw=None) -> dict:
    """The full R8 panel for one cohort of aligned keyed rows."""
    truth_phase, pred_phase = _aligned_phases(truth_phase, pred_phase)
    n = truth_phase.shape[0]
    if q3_true is not None:
        _finite_vector(q3_true, "q3 truth", n=n)
        for name, values in (("q3_star", q3_star), ("q3_raw", q3_raw)):
            if values is not None:
                _finite_vector(values, name, n=n)
    out = {
        "n": int(n),
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
    t, p = (x >= 3 for x in _aligned_phases(truth_phase, pred_phase))
    groups = np.asarray(groups)
    if groups.ndim != 1 or groups.shape != t.shape or (groups.size and groups.dtype.kind not in "iu"):
        raise TechnicalError("scan groups must be a 1-D integer vector aligned with the phases")
    groups = groups.astype(np.int64)
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
