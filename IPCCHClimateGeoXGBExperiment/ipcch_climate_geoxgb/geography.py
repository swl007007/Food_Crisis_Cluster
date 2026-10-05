"""Read-only validation of the frozen R39 geography and its key tables.

Bounded local copies of the IPCCH GeoRF loaders (see config/source-provenance.json)
plus a direct structural check of the frozen adjacency cache. The old adjacency
helper, the GeoRF ZIP backend and the completed run's code are never loaded,
topology is never rebuilt, and nothing here writes next to an input.

Known source limitation (R39): topology repair does not establish
administrative identity; upstream nearest-neighbour matching and boundary
vintage remain unverified. That limitation never excuses a hash mismatch, key
conflict or corrupt artifact -- those raise ``ContractError``.
"""

from __future__ import annotations

import pickle
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_climate_geoxgb.errors import ContractError

GEOMETRY_ID_COLUMN = "admin_code"
AREA_ID_COLUMN = "area_id"
GEOMETRY_EPSG = 4326
#: Explicitly pinned geometry reader (runtime-lock.json); never the default engine.
GEOMETRY_ENGINE = "pyogrio"
COUNTRY_LOOKUP_COLUMNS = ("area_id", "iso3", "country", "country_code", "country_en")
CACHE_KEYS = frozenset(
    {
        "adjacency_dict",
        "area_ids",
        "component_sha256",
        "id_column",
        "polygon_centroids",
        "polygon_group_mapping",
        "polygon_id_mapping",
        "polygons",
    }
)
CENTROID_TOLERANCE = 1e-9


def normalize_area_ids(values, source: str = "area ids") -> np.ndarray:
    """Normalize any area-ID representation to canonical ``int64``.

    The DBF ``admin_code`` is a string, the CSVs read as integers, and a pandas
    round trip can yield ``101324.0``. Anything that is not exactly an integer
    is refused -- a silent ``int(float)`` truncation would fuse two areas.
    """
    out = np.empty(len(values), dtype=np.int64)
    for position, raw in enumerate(np.asarray(values, dtype=object)):
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            raise ContractError(f"{source}: blank area id at position {position}")
        text = str(raw).strip()
        if text == "":
            raise ContractError(f"{source}: blank area id at position {position}")
        try:
            number = Decimal(text)
        except InvalidOperation as error:
            raise ContractError(f"{source}: area id {raw!r} is not numeric") from error
        if number != number.to_integral_value():
            raise ContractError(f"{source}: area id {raw!r} is not an integer")
        out[position] = int(number)
    return out


def country_key(country_en: pd.Series, country: pd.Series) -> np.ndarray:
    """Reporting key: trimmed ``country_en``, else trimmed ``country``."""
    en = country_en.astype(str).str.strip()
    fallback = country.astype(str).str.strip()
    return np.where(en.ne(""), en, fallback)


def read_strings(path: Path, columns=None) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        usecols=list(columns) if columns is not None else None,
    )


def load_country_lookup(path: Path | str) -> tuple[pd.DataFrame, dict]:
    """Keyed country lookup; areas with a blank ISO3/code are kept, never repaired."""
    path = Path(path)
    raw = read_strings(path)
    missing = [name for name in COUNTRY_LOOKUP_COLUMNS if name not in raw.columns]
    if missing:
        raise ContractError(f"{path.name} is missing columns: {missing}")
    frame = raw[list(COUNTRY_LOOKUP_COLUMNS)].copy()
    frame[AREA_ID_COLUMN] = normalize_area_ids(frame[AREA_ID_COLUMN].to_numpy(), source=path.name)
    if frame[AREA_ID_COLUMN].duplicated().any():
        raise ContractError(f"{path.name}: duplicate {AREA_ID_COLUMN}")
    for column in ("iso3", "country", "country_code", "country_en"):
        frame[column] = frame[column].astype(str).str.strip()
    blank = frame["country_en"].eq("") & frame["country"].eq("")
    if blank.any():
        raise ContractError(f"{path.name}: {int(blank.sum())} areas have no country name")
    frame["country_key"] = country_key(frame["country_en"], frame["country"])
    frame = frame.sort_values(AREA_ID_COLUMN, kind="mergesort").reset_index(drop=True)
    audit = {
        "rows": int(len(frame)),
        "areas": int(frame[AREA_ID_COLUMN].nunique()),
        "countries": int(frame["country_key"].nunique()),
        "areas_missing_iso3": int(frame["iso3"].eq("").sum()),
        "areas_missing_country_code": int(frame["country_code"].eq("").sum()),
    }
    return frame, audit


def load_reference_coordinates(path: Path | str, lat: str = "lat", lon: str = "lon") -> pd.DataFrame:
    """Keyed reference points (not polygon centroids)."""
    path = Path(path)
    raw = read_strings(path)
    for column in (AREA_ID_COLUMN, lat, lon):
        if column not in raw.columns:
            raise ContractError(f"{path.name} is missing column '{column}'")
    frame = pd.DataFrame(
        {
            AREA_ID_COLUMN: normalize_area_ids(raw[AREA_ID_COLUMN].to_numpy(), source=path.name),
            "ref_lat": pd.to_numeric(raw[lat], errors="coerce").to_numpy(np.float64),
            "ref_lon": pd.to_numeric(raw[lon], errors="coerce").to_numpy(np.float64),
        }
    )
    if frame[AREA_ID_COLUMN].duplicated().any():
        raise ContractError(f"{path.name}: duplicate {AREA_ID_COLUMN}")
    coords = frame[["ref_lat", "ref_lon"]].to_numpy()
    if not np.isfinite(coords).all():
        raise ContractError(f"{path.name}: non-finite reference coordinates")
    if not ((np.abs(coords[:, 0]) <= 90).all() and (np.abs(coords[:, 1]) <= 180).all()):
        raise ContractError(f"{path.name}: reference coordinates outside lat/lon range")
    return frame.sort_values(AREA_ID_COLUMN, kind="mergesort").reset_index(drop=True)


def load_geometry(path: Path | str):
    """Read the repaired layer in file order with the pinned engine and check it.

    File order is kept because the frozen cache indexes polygons by the row
    position the old helper enumerated.
    """
    import geopandas as gpd  # noqa: PLC0415 - heavy dependency, geometry checks only

    path = Path(path)
    gdf = gpd.read_file(path, engine=GEOMETRY_ENGINE)
    if GEOMETRY_ID_COLUMN not in gdf.columns:
        raise ContractError(f"{path.name} has no '{GEOMETRY_ID_COLUMN}' column")
    if gdf.crs is None:
        raise ContractError(f"{path.name} has no CRS")
    epsg = gdf.crs.to_epsg()
    if epsg != GEOMETRY_EPSG or not gdf.crs.is_geographic:
        raise ContractError(f"{path.name} CRS is {gdf.crs.to_string()}, expected EPSG:{GEOMETRY_EPSG}")
    ids = normalize_area_ids(gdf[GEOMETRY_ID_COLUMN].to_numpy(), source=path.name)
    if len(set(ids.tolist())) != len(ids):
        raise ContractError(f"{path.name}: area ids carry more than one feature")
    if int(gdf.geometry.isna().sum()) or int(gdf.geometry.is_empty.sum()):
        raise ContractError(f"{path.name}: missing or empty geometries")
    if not np.isfinite(gdf.geometry.bounds.to_numpy(dtype=np.float64)).all():
        raise ContractError(f"{path.name}: non-finite geometry bounds")
    invalid = int((~gdf.geometry.is_valid).sum())
    if invalid:
        raise ContractError(f"{path.name}: {invalid} invalid geometries in the repaired layer")
    areal = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    if not areal.all():
        raise ContractError(f"{path.name}: {int((~areal).sum())} non-areal geometries")
    audit = {
        "engine": GEOMETRY_ENGINE,
        "features": int(len(gdf)),
        "epsg": int(epsg),
        "geometry_types": {str(k): int(v) for k, v in gdf.geometry.geom_type.value_counts().items()},
        "invalid": invalid,
    }
    return gdf, ids, audit


def load_adjacency_cache(path: Path | str) -> dict:
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ContractError(f"{Path(path).name}: cache payload is not a dict")
    return payload


def _exact_int_mapping(mapping: dict, name: str) -> dict:
    """Canonical int -> int copy; a fractional or non-numeric key/value is a stop.

    Checked before any comparison, so ``int()`` truncation (index + 0.25 -> index)
    can never make a corrupt mapping look like a valid bijection.
    """
    keys = normalize_area_ids(list(mapping.keys()), source=f"{name} keys")
    values = normalize_area_ids(list(mapping.values()), source=f"{name} values")
    canonical = dict(zip(keys.tolist(), values.tolist()))
    if len(canonical) != len(mapping):
        raise ContractError(f"{name}: distinct keys collapse to the same integer")
    return canonical


def validate_adjacency_cache(
    payload: dict,
    geometry_ids: np.ndarray,
    expected_components: dict,
    geometry_centroids: np.ndarray | None = None,
) -> dict:
    """Check the frozen cache against the repaired geometry, without the old helper.

    ``polygon_id_mapping`` is area ID -> polygon index and
    ``polygon_group_mapping`` is polygon index -> area ID; area IDs are not
    dense indices. ``geometry_ids[i]`` is the area at shapefile row ``i``.
    """
    keys = set(payload)
    if keys != CACHE_KEYS:
        raise ContractError(f"cache keys {sorted(keys)} != {sorted(CACHE_KEYS)}")
    if payload["id_column"] != GEOMETRY_ID_COLUMN:
        raise ContractError(f"cache id_column {payload['id_column']!r}")
    if dict(payload["component_sha256"]) != dict(expected_components):
        raise ContractError("cache component_sha256 does not match the repaired geometry components")

    geometry_ids = np.asarray(geometry_ids, dtype=np.int64)
    n = len(geometry_ids)
    if payload["polygons"] != n:
        raise ContractError(f"cache polygons {payload['polygons']} != geometry features {n}")

    group = _exact_int_mapping(payload["polygon_group_mapping"], "polygon_group_mapping")
    by_area = _exact_int_mapping(payload["polygon_id_mapping"], "polygon_id_mapping")
    if sorted(group) != list(range(n)):
        raise ContractError("polygon_group_mapping keys are not the polygon indices 0..n-1")
    if len(by_area) != n or set(by_area.values()) != set(range(n)):
        raise ContractError("polygon_id_mapping is not a bijection onto polygon indices")
    if set(by_area) != set(geometry_ids.tolist()):
        raise ContractError("polygon_id_mapping keys are not the geometry area IDs (mapping direction?)")
    for area, index in by_area.items():
        if group[index] != area:
            raise ContractError(f"area {area} -> index {index} is not inverted by polygon_group_mapping")
    by_position = np.array([group[i] for i in range(n)], dtype=np.int64)
    if not np.array_equal(by_position, geometry_ids):
        mismatch = int((by_position != geometry_ids).sum())
        raise ContractError(f"cache index -> area disagrees with shapefile row order at {mismatch} rows")
    area_ids = np.asarray(payload["area_ids"])
    if area_ids.dtype.kind != "i" or not np.array_equal(area_ids.astype(np.int64), by_position):
        raise ContractError("cache area_ids differ from polygon_group_mapping order")

    adjacency = payload["adjacency_dict"]
    keys = normalize_area_ids(list(adjacency), source="adjacency_dict keys")
    if sorted(keys.tolist()) != list(range(n)) or len(adjacency) != n:
        raise ContractError("adjacency_dict keys are not the polygon indices 0..n-1")
    neighbours = {}
    for index in range(n):
        values = np.asarray(adjacency[index])
        if values.size and values.dtype.kind not in "iu":
            raise ContractError(f"adjacency of polygon {index} has non-integer entries")
        values = values.astype(np.int64)
        if values.size and (values.min() < 0 or values.max() >= n):
            raise ContractError(f"adjacency of polygon {index} has an out-of-range index")
        as_set = set(values.tolist())
        if index in as_set:
            raise ContractError(f"polygon {index} lists itself as a neighbour")
        if len(as_set) != values.size:
            raise ContractError(f"polygon {index} has duplicate neighbour entries")
        neighbours[index] = as_set
    asymmetric = sum(1 for i, ns in neighbours.items() for j in ns if i not in neighbours[j])
    if asymmetric:
        raise ContractError(f"adjacency is not symmetric ({asymmetric} one-way entries)")
    degrees = np.array([len(neighbours[i]) for i in range(n)], dtype=np.int64)

    centroids = np.asarray(payload["polygon_centroids"], dtype=np.float64)
    if centroids.shape != (n, 2) or not np.isfinite(centroids).all():
        raise ContractError(f"polygon_centroids shape {centroids.shape} or non-finite values")
    centroid_check = None
    if geometry_centroids is not None:
        delta = np.abs(centroids - np.asarray(geometry_centroids, dtype=np.float64)).max(axis=1)
        bad = int((delta > CENTROID_TOLERANCE).sum())
        if bad:
            raise ContractError(f"{bad} cached centroids differ from the repaired geometry's centroids")
        centroid_check = {"max_abs_degrees": float(delta.max()), "tolerance": CENTROID_TOLERANCE}

    return {
        "polygons": n,
        "edges_undirected": int(degrees.sum() // 2),
        "directed_entries": int(degrees.sum()),
        "isolated": int((degrees == 0).sum()),
        "degree_max": int(degrees.max()) if n else 0,
        "symmetric": True,
        "mapping_direction": "polygon_id_mapping: area->index; polygon_group_mapping: index->area (verified inverse)",
        "row_order_matches_geometry": True,
        "centroid_vs_geometry": centroid_check,
    }


def geometry_centroids_latlon(gdf) -> np.ndarray:
    """Planar centroids as (lat, lon), the quantity stored in the frozen cache.

    Degree-space centroids are what the old helper stored; geopandas' warning
    about geographic CRS is expected here and silenced deliberately.
    """
    import warnings  # noqa: PLC0415

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*geographic CRS.*", category=UserWarning)
        points = gdf.geometry.centroid
    return np.column_stack([points.y.to_numpy(), points.x.to_numpy()])
