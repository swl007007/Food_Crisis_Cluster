"""Local D30 reducer for AlphaEarth 2017 COGs (ipcch-geo; PREFLIGHT ONLY: no production use, no EE call).

Mirrors the D30 recipe term by term:
  lattice    EPSG:6933, affine [10,0,0,0,-10,0] (``extraction.Lattice``); cell (c, r) has centre
             (x0 + (c + .5) dx, y0 + (r + .5) dy);
  sampling   nearest: each cell centre is transformed to the COG's UTM CRS and takes the native pixel containing it
             (base resolution only; overviews are never read);
  decoding   README: v = (raw / 127.5) ** 2 * sign(raw) in float64; raw -128 is NoData;
  ownership  a source is valid at a cell only if all 64 raw bands are valid; the FIRST valid source in ascending
             frozen asset id supplies the whole vector (cells never mix assets);
  area       pixelArea on EPSG:6933 = 100 m2 per cell (equal-area lattice; EE pilot C05: 4e6 cells -> 4e8 m2);
  weights    polygon/pixel coverage fraction f of each cell (the geometry's lon/lat edges densified, then projected).
             EE stores weights in 8 bits; variants exact / round255 / floor255 / ceil255 are all reported, and the
             parity protocol pre-selects ONE on the selection reference before any held-out comparison.
Outputs per variant: A = sum w*100, S_j = sum w*100*v_j, Q_j = sum w*100*v_j^2 (float64), plus the
centre-inclusion valid-cell count (EE's unweighted count semantics: centre inside the region and mask non-zero).
Processing is in row strips: no full-admin float mosaic is materialised.
"""

from __future__ import annotations

import math
import time

import numpy as np

BANDS = 64
NODATA = -128
PIXEL_AREA = 100.0
VARIANTS = ("exact", "round255", "floor255", "ceil255")


def decode(raw: np.ndarray) -> np.ndarray:
    v = raw.astype(np.float64)
    return (v / 127.5) ** 2 * np.sign(v)


def quantize(f: np.ndarray, variant: str) -> np.ndarray:
    if variant == "exact":
        return f
    k = {"round255": np.rint, "floor255": np.floor, "ceil255": np.ceil}[variant](f * 255.0)
    return np.clip(k, 0, 255) / 255.0


def project_polygon(geom4326, crs: str, densify_deg: float = 1e-4):
    import shapely
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    dense = shapely.segmentize(geom4326, densify_deg)
    return shapely.transform(dense, lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))


def lattice_range(poly6933, lattice):
    minx, miny, maxx, maxy = poly6933.bounds
    c0 = math.floor((minx - lattice.x0) / lattice.dx)
    c1 = math.ceil((maxx - lattice.x0) / lattice.dx)
    r0 = math.floor((maxy - lattice.y0) / lattice.dy)
    r1 = math.ceil((miny - lattice.y0) / lattice.dy)
    return c0, c1, r0, r1


def strip_weights(poly6933, lattice, c0, c1, r0, r1):
    """Coverage fraction and centre inclusion for cells [r0, r1) x [c0, c1)."""
    import shapely
    from rasterio import features
    from rasterio.transform import Affine

    h, w = r1 - r0, c1 - c0
    tf = Affine(lattice.dx, 0, lattice.x0 + c0 * lattice.dx, 0, lattice.dy, lattice.y0 + r0 * lattice.dy)
    centre = features.rasterize([(poly6933, 1)], out_shape=(h, w), transform=tf, fill=0, all_touched=False, dtype="uint8").astype(bool)
    touched = features.rasterize([(poly6933, 1)], out_shape=(h, w), transform=tf, fill=0, all_touched=True, dtype="uint8").astype(bool)
    edge = features.rasterize([(poly6933.boundary, 1)], out_shape=(h, w), transform=tf, fill=0, all_touched=True, dtype="uint8").astype(bool)
    frac = np.where(touched & ~edge, 1.0, 0.0)
    rr, cc = np.nonzero(edge)
    if len(rr):
        x0 = lattice.x0 + (c0 + cc) * lattice.dx
        y0 = lattice.y0 + (r0 + rr) * lattice.dy
        boxes = shapely.box(x0, y0 + lattice.dy, x0 + lattice.dx, y0)
        clip = shapely.clip_by_rect(poly6933, *(lambda b: (b[0] - 20, b[1] - 20, b[2] + 20, b[3] + 20))(shapely.total_bounds(boxes)))
        shapely.prepare(clip)
        frac[rr, cc] = np.clip(shapely.area(shapely.intersection(boxes, clip)) / (lattice.dx * -lattice.dy), 0.0, 1.0)
    return frac, centre


def strip_values(lattice, c0, c1, r0, r1, sources):
    """First-valid-ascending composite raw int8 (h, w, 64) and validity for cells [r0, r1) x [c0, c1).

    ``sources``: [(asset_id, [rasterio dataset, ...]), ...] in ascending asset id."""
    from pyproj import Transformer
    from rasterio.windows import Window

    h, w = r1 - r0, c1 - c0
    cx = lattice.x0 + (np.arange(c0, c1) + 0.5) * lattice.dx
    cy = lattice.y0 + (np.arange(r0, r1) + 0.5) * lattice.dy
    X, Y = np.meshgrid(cx, cy)
    out = np.zeros((h, w, BANDS), dtype=np.int8)
    have = np.zeros((h, w), dtype=bool)
    for _asset, dss in sources:
        for ds in dss:
            tr = Transformer.from_crs(lattice.crs, ds.crs.to_string(), always_xy=True)
            E, N = tr.transform(X, Y)
            inv = ~ds.transform
            col = np.floor((E - ds.transform.c) / ds.transform.a).astype(np.int64)
            row = np.floor((N - ds.transform.f) / ds.transform.e).astype(np.int64)
            inside = (col >= 0) & (col < ds.width) & (row >= 0) & (row < ds.height) & ~have
            if not inside.any():
                continue
            del inv
            cmin, cmax, rmin, rmax = col[inside].min(), col[inside].max(), row[inside].min(), row[inside].max()
            block = ds.read(window=Window(int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)))  # (64, hh, ww) base level
            raw = np.moveaxis(block, 0, -1)[row[inside] - rmin, col[inside] - cmin]  # (n, 64)
            ok = (raw != NODATA).all(axis=1)
            idx = np.flatnonzero(inside)[ok]
            flat_have, flat_out = have.reshape(-1), out.reshape(-1, BANDS)
            flat_out[idx] = raw[ok]
            flat_have[idx] = True
    return out, have


def reduce_geometry(geom4326, lattice, sources, strip_rows: int = 512) -> dict:
    """A/S/Q per weight variant plus centre-inclusion valid count, for one region."""
    import resource

    t0 = time.monotonic()
    poly = project_polygon(geom4326, lattice.crs)
    c0, c1, r0, r1 = lattice_range(poly, lattice)
    acc = {v: {"A": 0.0, "S": np.zeros(BANDS), "Q": np.zeros(BANDS)} for v in VARIANTS}
    count = 0
    t_w = t_v = t_s = 0.0
    for ra in range(r0, r1, strip_rows):
        rb = min(ra + strip_rows, r1)
        t = time.monotonic()
        frac, centre = strip_weights(poly, lattice, c0, c1, ra, rb)
        t_w += time.monotonic() - t
        if not (frac > 0).any() and not centre.any():
            continue
        t = time.monotonic()
        raw, have = strip_values(lattice, c0, c1, ra, rb, sources)
        t_v += time.monotonic() - t
        t = time.monotonic()
        count += int((centre & have).sum())
        m = have & (frac > 0)
        vals = decode(raw[m])  # (n, 64) float64
        f = frac[m]
        for v in VARIANTS:
            wa = quantize(f, v) * PIXEL_AREA
            acc[v]["A"] += float(wa.sum())
            acc[v]["S"] += wa @ vals
            acc[v]["Q"] += wa @ (vals * vals)
        t_s += time.monotonic() - t
    return {"cells_bbox": (c1 - c0) * (r1 - r0), "lattice_range": [c0, c1, r0, r1], "valid_cells_centre": count,
            "variants": {v: {"A": a["A"], "S": a["S"].tolist(), "Q": a["Q"].tolist()} for v, a in acc.items()},
            "seconds": {"weights": round(t_w, 3), "values": round(t_v, 3), "sums": round(t_s, 3), "total": round(time.monotonic() - t0, 3)},
            "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
