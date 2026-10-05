"""P0 read-only input preflight. No fitting, no feature or label construction.

Checks, in order, each stopping on the first violation:

1. Byte length and SHA256 of every pinned input (config/inputs.json).
2. Country lookup and reference coordinates: the frozen run's saved copies
   equal the source tables value-for-value under the canonical area ID.
3. Repaired geometry: pinned engine, EPSG:4326, unique IDs, valid areal
   features; shared area universe across geometry/lookup/coordinates.
4. Frozen adjacency cache: keys, component hashes, mapping direction, row
   order and centroids against the geometry, index range, symmetry.
5. Raw panel keys: complete, integer, unique (admin_code, year, month); the
   raw area set equals the universe (no outside or missing area -- a source
   identity check, not a per-month valid-target requirement); per-area country
   equals the lookup; per-row lat/lon within tolerance of the reference point.
6. Known diagnostics (missing ISO3/code, centroid-vs-reference differences)
   are recorded against the planning observation, never repaired.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_climate_geoxgb import geography as geo
from ipcch_climate_geoxgb.artifacts import sha256_file
from ipcch_climate_geoxgb.contract import GEOMETRY_COMPONENTS, input_path, load_inputs
from ipcch_climate_geoxgb.errors import ContractError

RAW_KEY_COLUMNS = ("admin_code", "year", "month")
RAW_REQUIRED_COLUMNS = RAW_KEY_COLUMNS + (
    "phase1_percent",
    "phase2_percent",
    "phase3_percent",
    "phase4_percent",
    "phase5_percent",
    "estimated_population",
    "overall_phase",
    "country",
    "country_en",
    "country_code",
    "ISO3",
    "lat",
    "lon",
)


def verify_identities(inputs: dict) -> dict:
    """Byte length then SHA256 of every input; a mismatch is a stop."""
    observed = {}
    for name, entry in inputs.items():
        path = input_path(entry)
        if not path.is_file():
            raise ContractError(f"{name}: input missing at {path}")
        size = path.stat().st_size
        if size != entry["bytes"]:
            raise ContractError(f"{name}: {size} bytes, pinned {entry['bytes']}")
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise ContractError(f"{name}: sha256 {digest} != pinned {entry['sha256']}")
        observed[name] = {"bytes": size, "sha256": digest, "match": True}
    return observed


def check_saved_tables(source_lookup, saved_lookup_path, source_ref, saved_ref_path) -> dict:
    saved_lookup = geo.read_strings(Path(saved_lookup_path))
    saved_lookup[geo.AREA_ID_COLUMN] = geo.normalize_area_ids(saved_lookup[geo.AREA_ID_COLUMN].to_numpy())
    saved_lookup = saved_lookup.sort_values(geo.AREA_ID_COLUMN, kind="mergesort").reset_index(drop=True)
    columns = list(geo.COUNTRY_LOOKUP_COLUMNS[1:]) + ["country_key"]
    if not np.array_equal(saved_lookup[geo.AREA_ID_COLUMN], source_lookup[geo.AREA_ID_COLUMN]):
        raise ContractError("saved country lookup area IDs differ from the source lookup")
    for column in columns:
        if not saved_lookup[column].astype(str).str.strip().equals(source_lookup[column]):
            raise ContractError(f"saved country lookup column {column} differs from the source")
    saved_ref = geo.load_reference_coordinates(saved_ref_path, lat="ref_lat", lon="ref_lon")
    if not np.array_equal(saved_ref[geo.AREA_ID_COLUMN], source_ref[geo.AREA_ID_COLUMN]):
        raise ContractError("saved reference coordinates cover different area IDs")
    equal = (saved_ref[["ref_lat", "ref_lon"]].to_numpy() == source_ref[["ref_lat", "ref_lon"]].to_numpy()).all()
    if not equal:
        raise ContractError("saved ref_lat/ref_lon differ from source lat/lon")
    return {"country_lookup_equal": True, "reference_coordinates_equal": True, "columns": columns}


def reconcile_universe(geometry_ids, lookup_ids, reference_ids, expected: int) -> np.ndarray:
    sets = {
        "geometry": set(np.asarray(geometry_ids).tolist()),
        "country_lookup": set(np.asarray(lookup_ids).tolist()),
        "reference_coordinates": set(np.asarray(reference_ids).tolist()),
    }
    union = set().union(*sets.values())
    problems = [f"{k} missing {len(union - v)} ids" for k, v in sets.items() if union - v]
    if len(union) != expected:
        problems.append(f"universe has {len(union)} ids, expected {expected}")
    if problems:
        raise ContractError("area universe mismatch: " + "; ".join(problems))
    return np.array(sorted(union), dtype=np.int64)


def check_raw_panel(path, universe, lookup: pd.DataFrame, reference: pd.DataFrame, tolerance: float) -> dict:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in header]
    if missing:
        raise ContractError(f"raw panel is missing columns {missing}")
    raw = geo.read_strings(Path(path), RAW_KEY_COLUMNS + ("country", "country_en", "lat", "lon"))
    for column in RAW_KEY_COLUMNS:
        if raw[column].str.strip().eq("").any():
            raise ContractError(f"raw panel column {column} has blank values")
    area = geo.normalize_area_ids(raw["admin_code"].to_numpy(), source="raw admin_code")
    year = geo.normalize_area_ids(raw["year"].to_numpy(), source="raw year")
    month = geo.normalize_area_ids(raw["month"].to_numpy(), source="raw month")
    if not ((month >= 1) & (month <= 12)).all():
        raise ContractError("raw month outside 1..12")
    keys = pd.DataFrame({"area": area, "year": year, "month": month})
    duplicates = int(keys.duplicated().sum())
    if duplicates:
        raise ContractError(f"raw panel has {duplicates} duplicate (admin_code, year, month) keys")
    outside = sorted(set(np.unique(area).tolist()) - set(universe.tolist()))
    if outside:
        raise ContractError(f"raw panel has {len(outside)} areas outside the geography universe")
    # Source identity, not label coverage: every declared area must have at
    # least one raw row; areas may still lack valid targets in any month/fold.
    absent = sorted(set(universe.tolist()) - set(np.unique(area).tolist()))
    if absent:
        raise ContractError(
            f"raw panel lacks rows for {len(absent)} universe areas (first: {absent[:5]})"
        )

    raw_key = pd.Series(geo.country_key(raw["country_en"], raw["country"]))
    per_area = pd.DataFrame({"area": area, "key": raw_key}).drop_duplicates()
    multi = int(per_area["area"].duplicated().sum())
    if multi:
        raise ContractError(f"{multi} areas carry more than one country in the raw panel")
    expected_key = lookup.set_index(geo.AREA_ID_COLUMN)["country_key"]
    disagree = int((per_area.set_index("area")["key"] != expected_key.loc[per_area["area"]]).sum())
    if disagree:
        raise ContractError(f"{disagree} areas disagree with the country lookup")

    lat = pd.to_numeric(raw["lat"], errors="coerce").to_numpy(np.float64)
    lon = pd.to_numeric(raw["lon"], errors="coerce").to_numpy(np.float64)
    if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        raise ContractError("raw panel has blank or non-finite lat/lon")
    ref = reference.set_index(geo.AREA_ID_COLUMN).loc[area]
    delta = np.maximum(np.abs(lat - ref["ref_lat"].to_numpy()), np.abs(lon - ref["ref_lon"].to_numpy()))
    over = int((delta > tolerance).sum())
    if over:
        raise ContractError(f"{over} raw rows have lat/lon more than {tolerance} deg from the reference point")
    years = np.unique(year)
    return {
        "rows": int(len(raw)),
        "areas": int(len(np.unique(area))),
        "universe_areas_without_rows": 0,
        "years": [int(years.min()), int(years.max())],
        "duplicate_keys": 0,
        "country_disagreements": 0,
        "coordinate_rows_not_identical": int((delta > 0).sum()),
        "coordinate_max_abs_degrees": float(delta.max()),
        "coordinate_tolerance_degrees": tolerance,
    }


def run_preflight(inputs_config: dict | None = None) -> dict:
    """Run every check and return the evidence report; raises on the first failure."""
    started = time.time()
    config = inputs_config or load_inputs()
    inputs, expect = config["inputs"], config["structural_expectations"]
    report = {"inputs_version": config["inputs_version"], "checks": {}}

    report["checks"]["identities"] = verify_identities(inputs)

    lookup, lookup_audit = geo.load_country_lookup(input_path(inputs["country_lookup_source"]))
    reference = geo.load_reference_coordinates(input_path(inputs["reference_coordinates_source"]))
    report["checks"]["saved_tables"] = check_saved_tables(
        lookup,
        input_path(inputs["saved_country_lookup"]),
        reference,
        input_path(inputs["saved_reference_coordinates"]),
    )

    gdf, geometry_ids, geometry_audit = geo.load_geometry(input_path(inputs["geometry_shp"]))
    universe = reconcile_universe(
        geometry_ids, lookup[geo.AREA_ID_COLUMN], reference[geo.AREA_ID_COLUMN], expect["areas"]
    )
    report["checks"]["geometry"] = geometry_audit
    report["checks"]["universe"] = {
        "areas": int(len(universe)),
        "id_min": int(universe.min()),
        "id_max": int(universe.max()),
        "dense_0_to_n": bool(np.array_equal(universe, np.arange(len(universe)))),
    }

    components = {
        input_path(inputs[name]).name: inputs[name]["sha256"] for name in GEOMETRY_COMPONENTS
    }
    centroids = geo.geometry_centroids_latlon(gdf)
    cache = geo.validate_adjacency_cache(
        geo.load_adjacency_cache(input_path(inputs["adjacency_cache"])),
        geometry_ids,
        components,
        geometry_centroids=centroids,
    )
    for field, key in (("edges_undirected", "adjacency_edges_undirected"), ("isolated", "adjacency_isolated")):
        if cache[field] != expect[key]:
            raise ContractError(f"cache {field} {cache[field]} != frozen audit {expect[key]}")
    report["checks"]["adjacency_cache"] = cache

    report["checks"]["raw_panel"] = check_raw_panel(
        input_path(inputs["raw_panel"]),
        universe,
        lookup,
        reference,
        expect["panel_coordinate_tolerance_degrees"],
    )

    # Centroid vs reference point: different objects; differences are allowed.
    ref = reference.set_index(geo.AREA_ID_COLUMN).loc[geometry_ids][["ref_lat", "ref_lon"]].to_numpy()
    over = int((np.abs(centroids - ref).max(axis=1) > 1e-6).sum())
    diagnostics = {
        "countries": lookup_audit["countries"],
        "areas_missing_iso3": lookup_audit["areas_missing_iso3"],
        "areas_missing_country_code": lookup_audit["areas_missing_country_code"],
        "centroid_reference_over_1e-6_degrees": over,
    }
    observed_vs_expected = {k: {"observed": v, "planning": expect[k]} for k, v in diagnostics.items()}
    drift = [k for k, v in observed_vs_expected.items() if v["observed"] != v["planning"]]
    if drift:
        raise ContractError(f"pinned-input diagnostics drifted from planning observation: {drift}")
    report["checks"]["known_diagnostics"] = {
        "values": observed_vs_expected,
        "policy": "allowed and preserved; never repaired",
        "polygons_checked": int(len(geometry_ids)),
    }
    report["status"] = "passed"
    report["elapsed_seconds"] = round(time.time() - started, 1)
    report["limitations"] = [
        config["known_limit"],
        "Topology was not rebuilt; adjacency was validated structurally against the frozen cache.",
        "Observation-month availability is a convention, not verified release timing.",
    ]
    return report
