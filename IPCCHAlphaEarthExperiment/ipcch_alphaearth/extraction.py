"""Offline extraction planning and result validation (design section 5, D16/D25/D26). Never contacts EE.

* Geometry: the staged frozen repaired shapefile (6,227 admins), exact int64 ids, EPSG:4326, valid, no rebuild.
* Denominator: full frozen polygon area in m2 (WGS84 geodesic, pyproj), water included.
* Ownership: deterministic and outcome-independent. For one admin and the actual 2017 asset footprints sorted
  lexicographically by asset id, piece_k = admin & footprint_k - (footprint_1 | ... | footprint_{k-1}). Overlapping
  full tiles are never summed twice; first-owner masked area stays masked (no fill from another source). Pieces are
  computed here (replayable offline) and sent to EE only for native-grid reduction of that asset.
* Requests: frozen JSON manifests (pilot or full) with exact piece geometries, asset ids, destination and limits;
  the EE script refuses a request without an explicit release id. Completed exports are validated against the plan
  (missing/duplicate/unplanned pieces are technical failures, never low coverage).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_alphaearth import aemoments
from ipcch_alphaearth.artifacts import digest, write_json
from ipcch_alphaearth.errors import ContractError, TechnicalError

#: |negative geodesic area| up to this is floating-point zero (1e-5 of a 10 m pixel); larger negatives stop.
SLIVER_TOLERANCE_M2 = 1e-3

EXPORT_COLUMNS = ["admin_code", "piece_id", "asset_id", "status", "valid_area",
                  *[f"S_{b}" for b in aemoments.BANDS], *[f"Q_{b}" for b in aemoments.BANDS]]


def load_geometry(shp: Path, expected_admins: int | None = 6227):
    import geopandas as gpd  # noqa: PLC0415

    frame = gpd.read_file(shp, engine="pyogrio")
    if frame.crs is None or frame.crs.to_epsg() != 4326:
        raise ContractError(f"geometry CRS {frame.crs} is not EPSG:4326")
    ids = frame["admin_code"].astype(str).str.strip()
    if not ids.str.fullmatch(r"-?\d+").all():
        raise ContractError("non-integer admin_code in geometry")
    frame["admin_code"] = ids.astype(np.int64)
    if frame["admin_code"].duplicated().any():
        raise ContractError("duplicate admin_code in geometry")
    if frame.geometry.isna().any() or frame.geometry.is_empty.any() or (~frame.geometry.is_valid).any():
        raise ContractError("missing/empty/invalid geometry (never repaired here)")
    if expected_admins is not None and len(frame) != expected_admins:
        raise ContractError(f"geometry has {len(frame)} admins, expected {expected_admins}")
    return frame.sort_values("admin_code", kind="mergesort").reset_index(drop=True)


def geodesic_area(geom) -> float:
    """WGS84 geodesic area in m2, independent of ring orientation: every polygon component is oriented (exterior
    counter-clockwise, holes clockwise) for the computation only and component areas are summed. The original
    geometry is never rewritten or repaired."""
    from pyproj import Geod  # noqa: PLC0415
    from shapely.geometry import MultiPolygon, Polygon  # noqa: PLC0415
    from shapely.geometry.polygon import orient  # noqa: PLC0415

    if isinstance(geom, Polygon):
        parts = [geom]
    elif isinstance(geom, MultiPolygon):
        parts = list(geom.geoms)
    else:
        raise ContractError(f"area of unsupported geometry type {geom.geom_type}")
    geod = Geod(ellps="WGS84")
    total = 0.0
    for part in parts:
        area, _ = geod.geometry_area_perimeter(orient(part, sign=1.0))
        if area < 0:
            # degenerate intersection slivers (planar area ~1e-18 deg2) can carry a floating-point negative geodesic sign
            if area < -SLIVER_TOLERANCE_M2:
                raise ContractError(f"oriented polygon has negative geodesic area {area} m2")
            area = 0.0
        total += float(area)
    return total


def ownership_pieces(admin_geom, footprints: dict) -> list[dict]:
    """Deterministic pieces for one admin: lexicographic asset order, each footprint minus earlier footprints."""
    from shapely import union_all  # noqa: PLC0415

    pieces, earlier = [], []
    for asset_id in sorted(footprints):
        fp = footprints[asset_id]
        if not fp.intersects(admin_geom):
            continue
        part = admin_geom.intersection(fp)
        if earlier:
            part = part.difference(union_all(earlier))
        earlier.append(fp)
        if part.is_empty or part.area == 0:
            continue
        pieces.append({"asset_id": asset_id, "geometry": part})
    return pieces


def plan_pieces(geometry, footprints: dict) -> tuple[pd.DataFrame, list]:
    """Plan table (admin_code, piece_id, asset_id, piece_area_m2, polygon_area_m2) and piece geometries."""
    rows, geoms = [], []
    for rec in geometry.itertuples(index=False):
        poly_area = geodesic_area(rec.geometry)
        if not np.isfinite(poly_area) or poly_area <= 0:
            raise ContractError(f"admin {rec.admin_code}: polygon area not finite/positive")
        pieces = ownership_pieces(rec.geometry, footprints)
        for p in pieces:
            pid = f"{rec.admin_code}|{p['asset_id']}"
            rows.append({"admin_code": int(rec.admin_code), "piece_id": pid, "asset_id": p["asset_id"],
                         "piece_area_m2": geodesic_area(p["geometry"]), "polygon_area_m2": poly_area})
            geoms.append(p["geometry"])
        if not pieces:  # no asset footprint intersects: a planned zero-piece admin (genuine no coverage, A = 0)
            rows.append({"admin_code": int(rec.admin_code), "piece_id": None, "asset_id": None, "piece_area_m2": 0.0,
                         "polygon_area_m2": poly_area})
            geoms.append(None)
    return pd.DataFrame(rows), geoms


def build_request(kind: str, plan: pd.DataFrame, geoms: list, contract: dict, *, project: str, destination: dict,
                  limits: dict, release_id: str | None = None, admins: list | None = None) -> dict:
    """Frozen request manifest (GeoJSON piece geometries). ``release_id`` stays None until the supervisor releases it."""
    from shapely.geometry import mapping  # noqa: PLC0415

    if kind not in ("pilot", "full"):
        raise ValueError(kind)
    ae = contract["alphaearth"]
    keep = plan["piece_id"].notna() & (plan["admin_code"].isin(admins) if admins is not None else True)
    pieces = [{"piece_id": r.piece_id, "admin_code": int(r.admin_code), "asset_id": r.asset_id,
               "geometry": mapping(geoms[i])} for i, r in enumerate(plan.itertuples(index=False)) if keep.iloc[i]]
    body = {"kind": kind, "collection": ae["collection"], "year": ae["year"], "bands": list(aemoments.BANDS),
            "best_effort": False, "scale": "native asset projection (crs + crsTransform)", "project": project,
            "destination": destination, "limits": limits, "export_columns": EXPORT_COLUMNS, "pieces": pieces}
    return {**body, "request_sha256": digest(body), "release_id": release_id}


def validate_export(tables: list[pd.DataFrame], plan: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Concatenate completed export tables; every planned piece exactly once, nothing unplanned, finite moments."""
    got = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=EXPORT_COLUMNS)
    missing_cols = [c for c in EXPORT_COLUMNS if c not in got.columns]
    if missing_cols:
        raise TechnicalError(f"export lacks columns {missing_cols[:5]}")
    check_piece_binding(got, plan)
    return got, {"pieces": len(got), "admins": int(got["admin_code"].nunique())}


def check_piece_binding(pieces: pd.DataFrame, plan: pd.DataFrame) -> None:
    """Every planned piece exactly once with its planned admin and asset; nothing unplanned (status/moments checked in merge)."""
    planned = plan.dropna(subset=["piece_id"])
    if planned["piece_id"].duplicated().any():
        raise TechnicalError("piece plan has duplicate piece ids")
    want = {r.piece_id: (int(r.admin_code), str(r.asset_id)) for r in planned.itertuples(index=False)}
    dup = pieces["piece_id"][pieces["piece_id"].duplicated()].tolist()
    present = set(pieces["piece_id"])
    missing, extra = sorted(set(want) - present), sorted(present - set(want))
    mismatched = [r.piece_id for r in pieces.itertuples(index=False)
                  if r.piece_id in want and (int(r.admin_code), str(r.asset_id)) != want[r.piece_id]]
    if dup or missing or extra or mismatched:
        raise TechnicalError(f"export binding failed: {len(missing)} missing, {len(dup)} duplicated, {len(extra)} unplanned, "
                             f"{len(mismatched)} with a different admin/asset than planned")


def write_plan(out: Path, plan: pd.DataFrame, footprints_meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    plan.to_csv(out / "piece_plan.csv", index=False)
    write_json(out / "footprint_inventory.json", footprints_meta)


def parse_export_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, float_precision="round_trip")
    return frame[[c for c in EXPORT_COLUMNS if c in frame.columns]]


def request_from_file(path: Path) -> dict:
    req = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {k: v for k, v in req.items() if k not in ("request_sha256", "release_id")}
    if digest(body) != req["request_sha256"]:
        raise ContractError("request manifest body differs from its frozen sha256")
    return req
