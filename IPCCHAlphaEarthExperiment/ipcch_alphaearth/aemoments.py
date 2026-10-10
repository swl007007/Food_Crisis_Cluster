"""AlphaEarth sufficient-statistic merge and D25 coverage decision. Pure Python (no numpy) so the separate EE
interpreter (/opt/earthengine/bin/python) and the numerical environment share one implementation.

Per admin, completed native-grid pieces carry valid covered area A (m2, common finite-valid 64-band mask,
EE fractional boundary weights), first moments S_j = sum(area * e_j) and second moments Q_j = sum(area * e_j**2).
Pieces are merged by summing A/S/Q in piece-id order (never by averaging per-piece SDs). For A > 0:
mean_j = S_j / A, var_j = Q_j / A - mean_j**2, sd_j = sqrt(var_j); a negative variance within 1e-12 is float64
cancellation (zeroed and counted), anything more negative or nonfinite is a technical failure. A completed
extraction with A == 0 is genuine no coverage. Coverage = A / full frozen polygon area (water included),
compared unrounded with exact fractions; usable iff coverage >= 0.80 and all 128 moments are finite.
"""

from __future__ import annotations

import math
from fractions import Fraction

N_BANDS = 64
BANDS = tuple(f"A{j:02d}" for j in range(N_BANDS))


class ExtractionFailure(RuntimeError):
    """Missing/duplicated/corrupt piece or impossible moment: never a legitimate low-coverage route."""


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def merge_pieces(pieces: list[dict], planned_piece_ids: list[str]) -> dict:
    """Sum A/S/Q over one admin's completed pieces; every planned piece must be present exactly once."""
    seen = {}
    for p in pieces:
        pid = p["piece_id"]
        if pid in seen:
            raise ExtractionFailure(f"duplicate piece {pid}")
        seen[pid] = p
    missing = sorted(set(planned_piece_ids) - set(seen))
    extra = sorted(set(seen) - set(planned_piece_ids))
    if missing or extra:
        raise ExtractionFailure(f"piece set differs from plan: missing {missing[:5]}, unplanned {extra[:5]}")
    A, S, Q = 0.0, [0.0] * N_BANDS, [0.0] * N_BANDS
    for pid in sorted(seen):
        p = seen[pid]
        if p.get("status") != "complete":
            raise ExtractionFailure(f"piece {pid} status {p.get('status')!r}")
        a = p["valid_area"]
        if not _finite(a) or a < 0:
            raise ExtractionFailure(f"piece {pid} valid area {a!r} is not finite and nonnegative")
        if len(p["S"]) != N_BANDS or len(p["Q"]) != N_BANDS:
            raise ExtractionFailure(f"piece {pid} lacks 64 moments")
        A += float(a)
        for j in range(N_BANDS):
            s, q = p["S"][j], p["Q"][j]
            if not (_finite(s) and _finite(q)):
                raise ExtractionFailure(f"piece {pid} band {BANDS[j]} has a nonfinite moment")
            if a == 0 and (s != 0 or q != 0):
                raise ExtractionFailure(f"piece {pid} has moments without valid area")
            S[j] += float(s)
            Q[j] += float(q)
    return {"A": A, "S": S, "Q": Q, "pieces": len(seen)}


def descriptor(merged: dict, polygon_area: float, threshold: str = "0.80", var_tol: float = 1e-12) -> dict:
    """128 moments (or 128 NaN), coverage and the route reason for one admin."""
    if not _finite(polygon_area) or polygon_area <= 0:
        raise ExtractionFailure(f"full polygon area {polygon_area!r} is not finite and positive")
    A = merged["A"]
    nan = float("nan")
    coverage = A / polygon_area
    out = {"valid_area_m2": A, "polygon_area_m2": float(polygon_area), "coverage": coverage, "negative_variance_zeroed": 0}
    if A == 0:
        return {**out, "mean": [nan] * N_BANDS, "sd": [nan] * N_BANDS, "usable": False, "route_reason": "no_coverage"}
    mean, sd = [], []
    for j in range(N_BANDS):
        m = merged["S"][j] / A
        var = merged["Q"][j] / A - m * m
        if not math.isfinite(var):
            raise ExtractionFailure(f"band {BANDS[j]}: nonfinite variance")
        if var < 0:
            if var < -var_tol:
                raise ExtractionFailure(f"band {BANDS[j]}: variance {var!r} below the cancellation tolerance")
            var = 0.0
            out["negative_variance_zeroed"] += 1
        mean.append(m)
        sd.append(math.sqrt(var))
    finite = all(math.isfinite(v) for v in mean + sd)
    usable = finite and Fraction(A) / Fraction(polygon_area) >= Fraction(threshold)
    if not usable:
        reason = "low_coverage" if finite else "nonfinite_moment"
        return {**out, "mean": [nan] * N_BANDS, "sd": [nan] * N_BANDS, "usable": False, "route_reason": reason,
                "moments_if_usable": {"mean": mean, "sd": sd}}
    return {**out, "mean": mean, "sd": sd, "usable": True, "route_reason": "usable"}
