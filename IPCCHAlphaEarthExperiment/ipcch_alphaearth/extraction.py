"""Offline extraction planning and result validation (design section 5, D16/D25/D26). Never contacts EE.

* Geometry: the staged frozen repaired shapefile (6,227 admins), exact int64 ids, EPSG:4326, valid, no rebuild.
* Denominator: full frozen polygon area in m2 (WGS84 geodesic, pyproj), water included.
* D30 (current): one fixed analysis grid (EPSG:6933, affine [10,0,0,0,-10,0]); each cell takes the whole 64-band vector
  of the first valid candidate asset in ascending id (nearest sampling). Work is split only into grid-aligned windows
  (cuts on lattice lines = exact meridians/parallels for this cylindrical projection) cut from the frozen geographic
  geometry; plans bind the recipe id and the ordered candidate asset ids (``asset_id`` column = comma-joined ids).
  ``plan_pieces``/``ownership_pieces`` below implement the superseded first-footprint rule and are kept only for the
  native-grid pilot comparison evidence.
* Ownership (superseded by D30): deterministic and outcome-independent. For one admin and the actual 2017 asset footprints sorted
  lexicographically by asset id, piece_k = admin & footprint_k - (footprint_1 | ... | footprint_{k-1}). Overlapping
  full tiles are never summed twice; first-owner masked area stays masked (no fill from another source). Pieces are
  computed here (replayable offline) and sent to EE only for native-grid reduction of that asset.
* Requests: frozen JSON manifests (pilot or full) with exact piece geometries, asset ids, destination and limits;
  the EE script refuses a request without an explicit release id. Completed exports are validated against the plan
  (missing/duplicate/unplanned pieces are technical failures, never low coverage).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_alphaearth import aemoments
from ipcch_alphaearth.artifacts import digest, write_json
from ipcch_alphaearth.errors import ContractError, TechnicalError

#: |negative geodesic area| up to this is floating-point zero (1e-5 of a 10 m pixel); larger negatives stop.
SLIVER_TOLERANCE_M2 = 1e-3
#: D30 plans/exports carry the extraction recipe id in this column (checked by check_piece_binding/descriptor_table)
RECIPE_COLUMN = "recipe"

EXPORT_COLUMNS = ["admin_code", "piece_id", "asset_id", "status", "valid_area",
                  *[f"S_{b}" for b in aemoments.BANDS], *[f"Q_{b}" for b in aemoments.BANDS]]


def recipe_id(contract: dict) -> str:
    g = contract["alphaearth"]["analysis_grid"]
    return f"{g['amendment']}|{g['crs']}|{','.join(str(v) for v in g['affine'])}|{g['resampling']}|first-valid-ascending-id"


class Lattice:
    """Fixed analysis lattice; cylindrical projections only (x depends on lon, y on lat), no rotation."""

    def __init__(self, crs: str, affine):
        from pyproj import CRS, Transformer  # noqa: PLC0415

        a = [float(v) for v in affine]
        if a[1] != 0 or a[3] != 0 or a[0] <= 0 or a[4] >= 0:
            raise ContractError(f"analysis affine {affine} must be north-up without rotation")
        self.crs, self.affine = crs, a
        self.dx, self.x0, self.dy, self.y0 = a[0], a[2], a[4], a[5]
        self.fw = Transformer.from_crs("EPSG:4326", CRS.from_user_input(crs), always_xy=True)
        self.bw = Transformer.from_crs(CRS.from_user_input(crs), "EPSG:4326", always_xy=True)
        x_a, _ = self.fw.transform(10.0, -30.0)
        x_b, _ = self.fw.transform(10.0, 30.0)
        _, y_a = self.fw.transform(-80.0, 20.0)
        _, y_b = self.fw.transform(100.0, 20.0)
        if abs(x_a - x_b) > 1e-6 or abs(y_a - y_b) > 1e-6:
            raise ContractError(f"{crs} is not cylindrical: lattice lines would not be meridians/parallels")

    @classmethod
    def from_contract(cls, contract: dict) -> "Lattice":
        g = contract["alphaearth"]["analysis_grid"]
        return cls(g["crs"], g["affine"])

    def col(self, lon: float, up: bool) -> int:
        x, _ = self.fw.transform(lon, 0.0)
        v = (x - self.x0) / self.dx
        return math.ceil(v) if up else math.floor(v)

    def row(self, lat: float, up: bool) -> int:  # rows grow southwards (dy < 0)
        _, y = self.fw.transform(0.0, lat)
        v = (y - self.y0) / self.dy
        return math.ceil(v) if up else math.floor(v)

    def lon(self, col: int) -> float:
        return self.bw.transform(self.x0 + col * self.dx, 0.0)[0]

    def lat(self, row: int) -> float:
        return self.bw.transform(0.0, self.y0 + row * self.dy)[1]


def lattice_windows(geom, lattice: Lattice, max_cells: int) -> list[dict]:
    """Split a frozen geographic geometry into grid-aligned windows of at most ``max_cells`` lattice cells.

    Windows are the geometry intersected with lon/lat boxes whose edges are lattice lines (exact meridians/parallels
    for a cylindrical lattice), so every analysis cell lies in at most one window. The geometry itself is never
    reprojected or repaired."""
    from shapely.geometry import box  # noqa: PLC0415

    minx, miny, maxx, maxy = geom.bounds
    c0, c1 = lattice.col(minx, up=False), lattice.col(maxx, up=True)
    r0, r1 = lattice.row(maxy, up=False), lattice.row(miny, up=True)
    side = max(1, int(math.isqrt(int(max_cells))))
    out = []
    for i, ca in enumerate(range(c0, c1, side)):
        cb = min(ca + side, c1)
        for j, ra in enumerate(range(r0, r1, side)):
            rb = min(ra + side, r1)
            part = geom.intersection(box(lattice.lon(ca), lattice.lat(rb), lattice.lon(cb), lattice.lat(ra)))
            if part.is_empty or part.area == 0 or part.geom_type not in ("Polygon", "MultiPolygon", "GeometryCollection"):
                continue
            if part.geom_type == "GeometryCollection":
                from shapely.geometry import MultiPolygon  # noqa: PLC0415

                polys = [g for g in part.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
                if not polys:
                    continue
                part = MultiPolygon([q for g in polys for q in (g.geoms if g.geom_type == "MultiPolygon" else [g])])
            out.append({"window": f"w{i:03d}_{j:03d}", "cols": [ca, cb], "rows": [ra, rb], "geometry": part,
                        "lattice_cells_bbox": (cb - ca) * (rb - ra)})
    return out


def plan_c_requests(geometry, footprints: dict, contract: dict, max_cells: int) -> tuple[pd.DataFrame, list]:
    """D30 plan: one row per (admin, grid-aligned window) with the ordered candidate asset ids that can own its cells.

    Windows that no footprint reaches contribute genuine zeros and are not request pieces; each admin row records their
    count/area. Admins with no sourced window keep one explicit zero row (genuine no coverage). The denominator is
    always the whole frozen admin area. ``asset_id`` carries the comma-joined ascending candidate ids (the source
    binding checked on export); ``recipe`` carries the D30 recipe id."""
    lattice = Lattice.from_contract(contract)
    rid = recipe_id(contract)
    rows, geoms = [], []
    for rec in geometry.itertuples(index=False):
        poly_area = geodesic_area(rec.geometry)
        if not np.isfinite(poly_area) or poly_area <= 0:
            raise ContractError(f"admin {rec.admin_code}: polygon area not finite/positive")
        cand = sorted(a for a, fp in footprints.items() if fp.intersects(rec.geometry))
        wins = lattice_windows(rec.geometry, lattice, max_cells) if cand else []
        sourced, empty_n, empty_area = [], 0, 0.0
        for w in wins:
            wc = [a for a in cand if footprints[a].intersects(w["geometry"])]
            if wc:
                sourced.append((w, wc))
            else:  # no footprint reaches this window: genuine zero contribution, not a request piece
                empty_n += 1
                empty_area += geodesic_area(w["geometry"])
        for w, wc in sourced:
            rows.append({"admin_code": int(rec.admin_code), "piece_id": f"{rec.admin_code}|{w['window']}",
                         "asset_id": ",".join(wc), "recipe": rid, "candidates": len(wc), "cols": w["cols"], "rows": w["rows"],
                         "window_area_m2": geodesic_area(w["geometry"]), "polygon_area_m2": poly_area,
                         "no_source_windows": empty_n, "no_source_area_m2": empty_area})
            geoms.append(w["geometry"])
        if not sourced:  # no window has a source: one explicit zero-coverage row keeps the admin and its denominator
            rows.append({"admin_code": int(rec.admin_code), "piece_id": None, "asset_id": None, "recipe": rid, "candidates": 0,
                         "cols": None, "rows": None, "window_area_m2": 0.0, "polygon_area_m2": poly_area,
                         "no_source_windows": empty_n, "no_source_area_m2": empty_area})
            geoms.append(None)
    return pd.DataFrame(rows), geoms


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
    if "recipe" in planned.columns:
        recipes = set(planned["recipe"])
        if len(recipes) != 1:
            raise TechnicalError(f"piece plan mixes extraction recipes {sorted(recipes)}")
        if "recipe" not in pieces.columns or set(pieces["recipe"]) != recipes:
            raise TechnicalError("exported pieces were not produced under the plan's extraction recipe")


def write_plan(out: Path, plan: pd.DataFrame, footprints_meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    plan.to_csv(out / "piece_plan.csv", index=False)
    write_json(out / "footprint_inventory.json", footprints_meta)


def parse_export_csv(path: Path) -> pd.DataFrame:
    """Exported/downloaded piece table: frozen columns plus the D30 recipe binding (never dropped)."""
    frame = pd.read_csv(path, float_precision="round_trip", dtype={"piece_id": str, "asset_id": str, "recipe": str, "status": str})
    return frame[[c for c in (*EXPORT_COLUMNS, RECIPE_COLUMN) if c in frame.columns]]


def request_from_file(path: Path) -> dict:
    req = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {k: v for k, v in req.items() if k not in ("request_sha256", "release_id")}
    if digest(body) != req["request_sha256"]:
        raise ContractError("request manifest body differs from its frozen sha256")
    return req


# --------------------------------------------------------------------------- D30 production manifest and output parsing


def write_production_manifest(plan: pd.DataFrame, geoms: list, out_dir: Path, contract: dict, *, limits: dict,
                              grid_crs_wkt: str, projection_check: list, project: str, provenance: dict, select=None,
                              name: str = "production") -> dict:
    """Sha-bound production manifest: ``<name>_pieces.jsonl.gz`` (one request per planned piece) + ``<name>_manifest.json``.

    ``select`` (iterable of piece ids) writes a deterministic subset for a bounded live sample; release_id stays None.
    The per-piece maxPixels guard (129-band stack) is sized from the window's lattice box, never from its geodesic area,
    so sub-cell sliver windows are not refused for touching one whole cell.
    """
    import gzip  # noqa: PLC0415

    from shapely.geometry import mapping  # noqa: PLC0415

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = None if select is None else set(select)
    rows = []
    for i, r in enumerate(plan.itertuples(index=False)):
        if r.piece_id is None or (isinstance(r.piece_id, float) and np.isnan(r.piece_id)):
            continue
        if keep is not None and r.piece_id not in keep:
            continue
        cells = float(r.window_area_m2) / 100.0
        cols, rws = (json.loads(v) if isinstance(v, str) else list(v) for v in (r.cols, r.rows))
        bbox_cells = (cols[1] - cols[0]) * (rws[1] - rws[0])  # lattice box of the window: an upper bound on touched cells
        rows.append({"request_id": r.piece_id, "admin_code": int(r.admin_code), "candidates": str(r.asset_id).split(","),
                     "geometry": mapping(geoms[i]), "expected_cells": cells, "bbox_cells": int(bbox_cells),
                     "max_pixels_band_px": int(math.ceil(129 * bbox_cells * 1.1))})
    if keep is not None and len(rows) != len(keep):
        raise ContractError("selected piece ids are not all in the plan")
    pieces_path = out_dir / f"{name}_pieces.jsonl.gz"
    with open(pieces_path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as h:
            for row in rows:
                h.write((json.dumps(row, sort_keys=True) + "\n").encode())
    import hashlib  # noqa: PLC0415

    body = {"kind": "c_production", "recipe": recipe_id(contract), "grid": {"crs": contract["alphaearth"]["analysis_grid"]["crs"],
            "crsTransform": list(contract["alphaearth"]["analysis_grid"]["affine"])}, "grid_crs_wkt": grid_crs_wkt,
            "projection_check": projection_check, "best_effort": False, "project": project,
            "collection": contract["alphaearth"]["collection"], "year": contract["alphaearth"]["year"],
            "provenance": provenance, "sample": keep is not None, "pieces_file": pieces_path.name, "pieces_sha256": hashlib.sha256(pieces_path.read_bytes()).hexdigest(),
            "n_pieces": len(rows), "expected_cells_total": sum(r["expected_cells"] for r in rows),
            "limits": {**limits, "max_pixels": max([r["max_pixels_band_px"] for r in rows] + [0])}}
    manifest = {**body, "request_sha256": digest(body), "release_id": None}
    write_json(out_dir / f"{name}_manifest.json", manifest)
    return manifest


def load_production_manifest(path: Path) -> tuple[dict, list]:
    """Production manifest + its pieces, both re-verified (body sha, pieces-file sha, count, unique ascending ids)."""
    import gzip  # noqa: PLC0415
    import hashlib  # noqa: PLC0415

    manifest = request_from_file(path)
    if manifest.get("kind") != "c_production":
        raise ContractError("not a c_production manifest")
    pieces_path = Path(path).parent / manifest["pieces_file"]
    if hashlib.sha256(pieces_path.read_bytes()).hexdigest() != manifest["pieces_sha256"]:
        raise ContractError("production pieces file differs from its pinned sha256")
    with gzip.open(pieces_path, "rt", encoding="utf-8") as h:
        pieces = [json.loads(line) for line in h if line.strip()]
    if len(pieces) != int(manifest["n_pieces"]) or len({r["request_id"] for r in pieces}) != len(pieces):
        raise ContractError("production pieces count/uniqueness differs from the manifest")
    return manifest, pieces


def read_c_outputs(out_dir: Path, manifest: dict, pieces: list) -> pd.DataFrame:
    """Per-piece runner outputs -> export rows (frozen columns + recipe + diagnostics), binding re-checked per piece.

    Pieces without an output are simply absent (``validate_export`` reports them missing). Null or nonfinite sums are
    kept with status ``null_result``/``nonfinite_result`` so the merge raises a technical failure, never zero coverage."""
    import hashlib  # noqa: PLC0415

    out_dir = Path(out_dir)
    rows = []
    for r in pieces:
        path = out_dir / "pieces" / (hashlib.sha1(r["request_id"].encode()).hexdigest() + ".json")
        if not path.exists():
            continue
        rec = json.loads(path.read_text())
        if (rec.get("manifest_sha256") != manifest["request_sha256"] or rec.get("piece_id") != r["request_id"]
                or rec.get("admin_code") != r["admin_code"] or rec.get("candidates") != r["candidates"] or rec.get("recipe") != manifest["recipe"]
                or rec.get("response_sha256") != digest(rec.get("response"))):
            raise TechnicalError(f"output for {r['request_id']} does not match its manifest binding")
        s = rec["response"]["sums"]
        values = [s.get("valid_area")] + [s.get(f"S_{b}") for b in aemoments.BANDS] + [s.get(f"Q_{b}") for b in aemoments.BANDS]
        status = "complete"
        if any(v is None for v in values):
            status = "null_result"
        elif not all(np.isfinite(float(v)) for v in values):
            status = "nonfinite_result"
        row = {"admin_code": int(r["admin_code"]), "piece_id": r["request_id"], "asset_id": ",".join(r["candidates"]),
               "status": status, "valid_area": s.get("valid_area"), "recipe": rec["recipe"],
               "valid_cells": rec["response"]["count"].get("valid_cells"), "seconds": rec.get("seconds")}
        for b in aemoments.BANDS:
            row[f"S_{b}"], row[f"Q_{b}"] = s.get(f"S_{b}"), s.get(f"Q_{b}")
        rows.append(row)
    return pd.DataFrame(rows, columns=[*EXPORT_COLUMNS, RECIPE_COLUMN, "valid_cells", "seconds"])
