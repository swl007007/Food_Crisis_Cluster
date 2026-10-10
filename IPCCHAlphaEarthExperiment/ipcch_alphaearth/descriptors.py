"""AlphaEarth descriptor table, coverage/route ledger and the 424-column schema (design sections 4-5, D16/D25).

``descriptor_table`` reaggregates descriptors offline from saved sufficient statistics (the shared
``aemoments`` merge), so moments -> descriptors is reproducible without EE. ``attach_descriptors`` (P1) refuses
to run without a COMPLETE extraction manifest; P0 never contacts EE.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_alphaearth import aemoments
from ipcch_alphaearth.artifacts import require_stage
from ipcch_alphaearth.errors import ContractError, TechnicalError


def ae_names(contract: dict) -> list:
    f = contract["features"]
    n = int(f["alphaearth_bands"])
    return [f"{f['alphaearth_prefix_mean']}{j:02d}" for j in range(n)] + [f"{f['alphaearth_prefix_sd']}{j:02d}" for j in range(n)]


def descriptor_table(pieces: pd.DataFrame, plan: pd.DataFrame, polygons: pd.DataFrame, contract: dict,
                     expected_admins=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-admin descriptors (admin_code + 128 columns; NaN rows for low/no coverage) and the coverage ledger.

    ``pieces``: admin_code, piece_id, asset_id, status, valid_area, S_A00..S_A63, Q_A00..Q_A63 (completed export rows).
    ``plan``: admin_code, piece_id (every planned piece). ``polygons``: admin_code, polygon_area_m2.
    A planned admin without its full completed piece set is a technical failure, never low coverage.
    """
    from ipcch_alphaearth.extraction import check_piece_binding  # noqa: PLC0415

    ae = contract["alphaearth"]
    s_cols = [f"S_{b}" for b in aemoments.BANDS]
    q_cols = [f"Q_{b}" for b in aemoments.BANDS]
    if polygons["admin_code"].duplicated().any():
        raise TechnicalError("polygon table has duplicate admins")
    plan_admins, poly_admins = set(plan["admin_code"]), set(polygons["admin_code"])
    if plan_admins != poly_admins:
        raise TechnicalError(f"plan and polygon admin sets differ ({len(plan_admins - poly_admins)} plan-only, "
                             f"{len(poly_admins - plan_admins)} polygon-only): every declared geometry needs a plan entry")
    if expected_admins is not None and poly_admins != set(expected_admins):
        raise TechnicalError("descriptor table admins differ from the declared geometry admins")
    from ipcch_alphaearth.extraction import recipe_id  # noqa: PLC0415

    want = recipe_id(contract)
    if "recipe" not in plan.columns or set(plan["recipe"].dropna()) != {want}:
        raise TechnicalError(f"piece plan was not built under the frozen extraction recipe {want} (old-rule artifacts are refused)")
    check_piece_binding(pieces, plan)
    area = polygons.set_index("admin_code")["polygon_area_m2"]
    planned = plan.dropna(subset=["piece_id"]).groupby("admin_code")["piece_id"].apply(list)  # zero-piece admins plan []
    got = {a: g for a, g in pieces.groupby("admin_code")}
    rows, ledger = [], []
    for admin in sorted(area.index):
        plan_ids = planned.get(admin, [])
        frame = got.get(admin)
        recs = [] if frame is None else [
            {"piece_id": r.piece_id, "status": r.status, "valid_area": float(r.valid_area),
             "S": [float(getattr(r, c)) for c in s_cols], "Q": [float(getattr(r, c)) for c in q_cols]}
            for r in frame.itertuples(index=False)]
        try:
            merged = aemoments.merge_pieces(recs, list(plan_ids))
            d = aemoments.descriptor(merged, float(area.loc[admin]), ae["coverage_threshold"], float(ae["variance_cancellation_tolerance"]))
        except aemoments.ExtractionFailure as error:
            raise TechnicalError(f"admin {admin}: {error}") from error
        rows.append([admin, *d["mean"], *d["sd"]])
        ledger.append({"admin_code": int(admin), "pieces": merged["pieces"], "valid_area_m2": d["valid_area_m2"],
                       "polygon_area_m2": d["polygon_area_m2"], "coverage": d["coverage"], "usable": d["usable"],
                       "route_reason": d["route_reason"], "negative_variance_zeroed": d["negative_variance_zeroed"]})
    table = pd.DataFrame(rows, columns=["admin_code", *ae_names(contract)])
    return table, pd.DataFrame(ledger)


def row_descriptors(admin_codes: np.ndarray, table: pd.DataFrame, contract: dict) -> tuple[np.ndarray, np.ndarray]:
    """(n, 128) descriptors per row (NaN for non-routable admins) and the routable mask."""
    names = ae_names(contract)
    t = table.set_index("admin_code")
    missing = sorted(set(np.unique(admin_codes)) - set(t.index))
    if missing:
        raise TechnicalError(f"{len(missing)} cohort admins absent from the descriptor table (not legitimate low coverage)")
    D = t.loc[admin_codes, names].to_numpy(dtype=np.float64)
    finite = np.isfinite(D)
    all_finite, all_nan = finite.all(axis=1), (~finite).all(axis=1)
    if not (all_finite | all_nan).all():
        raise TechnicalError("descriptor rows must be all finite or all NaN together")
    return D, all_finite


def attach_descriptors(run_dir: Path, contract: dict) -> None:
    """P1: build X424 from the frozen descriptor table; requires a COMPLETE extraction stage."""
    require_stage(run_dir, "prepared")
    if not (Path(run_dir) / "extraction" / "COMPLETE.json").exists():
        raise ContractError("no COMPLETE extraction stage: AlphaEarth descriptors are produced only after the P1 cloud release")
    raise ContractError("attach_descriptors is implemented at the P1 checkpoint with the approved extraction manifest")
