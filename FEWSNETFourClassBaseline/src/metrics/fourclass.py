"""Fixed four-class scoring shared by every stage (PRD R1, R2, R9, R10).

Classes are the merged IPC phases 1, 2, 3 and "4或5", encoded internally as
0..3. Missing is never a class. Per-class F1 is 2TP/(2TP+FP+FN), scored 0 when
the denominator is zero, and macro F1 always averages all four classes. Counts
are aggregated over the scoring cohort before any F1 is computed.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

N_CLASSES = 4
CLASS_CODES = np.arange(N_CLASSES)
#: Exported class labels; code k is phase k+1, and code 3 is the merged 4/5 class.
CLASS_LABELS = ("1", "2", "3", "4或5")


def merge_phase(raw_phase):
    """Map raw phases 1..5 to merged classes 1..4 (4 and 5 -> 4); NaN stays NaN."""
    values = np.asarray(raw_phase, dtype=float)
    finite = np.isfinite(values)
    if not np.isin(values[finite], (1.0, 2.0, 3.0, 4.0, 5.0)).all():
        raise ValueError("phases must be integral 1..5 or missing")
    return np.where(values == 5.0, 4.0, values)


def as_codes(labels):
    """Validate an integer code vector on the fixed 0..3 axis."""
    codes = np.asarray(labels)
    if codes.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if codes.size and (not np.isin(codes, CLASS_CODES).all()):
        raise ValueError("labels must be codes 0..3")
    return codes.astype(np.int64)


def confusion(y_true, y_pred, weights=None):
    """4x4 confusion counts, rows = truth, columns = prediction."""
    truth, pred = as_codes(y_true), as_codes(y_pred)
    if truth.shape != pred.shape:
        raise ValueError("truth and prediction must be aligned")
    cells = truth * N_CLASSES + pred
    counts = np.bincount(cells, weights=weights, minlength=N_CLASSES ** 2)
    return counts.reshape(N_CLASSES, N_CLASSES)


def class_counts(matrix):
    """TP, FP and FN per class from a confusion matrix."""
    matrix = np.asarray(matrix)
    tp = np.diag(matrix)
    return tp, matrix.sum(axis=0) - tp, matrix.sum(axis=1) - tp


def per_class_f1(matrix):
    tp, fp, fn = class_counts(matrix)
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.zeros(N_CLASSES, dtype=float),
                     where=denominator > 0)


def macro_f1_from_matrix(matrix):
    return float(np.mean(per_class_f1(matrix)))


def macro_f1(y_true, y_pred, weights=None):
    return macro_f1_from_matrix(confusion(y_true, y_pred, weights))


def macro_f1_exact(y_true, y_pred):
    """Rational macro F1, used where a strict boundary must be decided exactly."""
    tp, fp, fn = class_counts(confusion(y_true, y_pred))
    total = Fraction(0)
    for k in range(N_CLASSES):
        denominator = int(2 * tp[k] + fp[k] + fn[k])
        if denominator:
            total += Fraction(int(2 * tp[k]), denominator)
    return total / N_CLASSES


def summary(y_true, y_pred):
    """Every reported statistic for one arm on one cohort."""
    truth, pred = as_codes(y_true), as_codes(y_pred)
    matrix = confusion(truth, pred)
    tp, fp, fn = class_counts(matrix)
    f1 = per_class_f1(matrix)
    n = int(truth.size)
    return {
        "n": n,
        "macro_f1": float(np.mean(f1)),
        "accuracy": float(tp.sum() / n) if n else None,
        "category_step_mae": float(np.mean(np.abs(truth - pred))) if n else None,
        "per_class": {
            CLASS_LABELS[k]: {
                "support": int(matrix[k].sum()),
                "predicted": int(matrix[:, k].sum()),
                "tp": int(tp[k]), "fp": int(fp[k]), "fn": int(fn[k]),
                "f1": float(f1[k]),
                "precision": float(tp[k] / (tp[k] + fp[k])) if tp[k] + fp[k] else None,
                "recall": float(tp[k] / (tp[k] + fn[k])) if tp[k] + fn[k] else None,
            }
            for k in range(N_CLASSES)
        },
        "confusion_rows_truth_cols_pred": matrix.astype(int).tolist(),
    }


def deterministic_proba(forest, X):
    """predict_proba with one thread: sklearn adds per-tree probabilities in thread
    completion order, so multi-threaded sums can differ in the last bit between calls.
    Fitting parallelism is unaffected (forests are identical for any n_jobs)."""
    if not hasattr(forest, "n_jobs"):  # single estimators have no threaded summation
        return forest.predict_proba(X)
    saved = forest.n_jobs
    forest.n_jobs = 1
    try:
        return forest.predict_proba(X)
    finally:
        forest.n_jobs = saved


def align_probabilities(proba, classes):
    """Place predict_proba columns on the fixed 0..3 axis; absent classes get 0."""
    proba = np.asarray(proba, dtype=float)
    classes = as_codes(np.asarray(classes))
    if proba.ndim != 2 or proba.shape[1] != classes.size:
        raise ValueError("probability columns do not match fitted classes")
    aligned = np.zeros((proba.shape[0], N_CLASSES), dtype=float)
    aligned[:, classes] = proba
    return aligned


def argmax_codes(aligned):
    """Hard decision on the fixed axis; ties go to the first class like RF.predict."""
    return np.argmax(np.asarray(aligned), axis=1).astype(np.int64)


def scan_masses(group_counts):
    """D8 parent-normalized scan inputs from per-group, per-class integer counts.

    ``group_counts`` has shape (n_groups, 3, 4) holding TP, FP and FN per class.
    Returns (Y, A) of shape (n_groups, 4): Y = D_gk / (4 D_k) and
    A = 2 TP_gk / (4 D_k), with zero columns wherever D_k = 0.
    """
    counts = np.asarray(group_counts, dtype=np.int64)
    tp, fp, fn = counts[:, 0, :], counts[:, 1, :], counts[:, 2, :]
    exposure = 2 * tp + fp + fn
    total = exposure.sum(axis=0, keepdims=True)
    scale = (N_CLASSES * total).astype(float)
    Y = np.divide(exposure, scale, out=np.zeros(exposure.shape, dtype=float), where=total > 0)
    A = np.divide(2 * tp, scale, out=np.zeros(exposure.shape, dtype=float), where=total > 0)
    return Y, A
