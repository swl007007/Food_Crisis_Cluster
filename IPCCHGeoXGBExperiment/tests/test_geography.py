"""Area IDs, lookups, geometry reader and frozen-cache validation on synthetic data."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from ipcch_geoxgb import geography as geo
from ipcch_geoxgb.errors import ContractError

COMPONENTS = {"g.shp": "a" * 64, "g.dbf": "b" * 64}
#: Non-dense area IDs in shapefile row order.
ROW_IDS = np.array([10, 20, 35, 7], dtype=np.int64)


def _payload():
    # 10-20, 20-35 edges; 7 isolated
    return {
        "adjacency_dict": {0: np.array([1]), 1: np.array([0, 2]), 2: np.array([1]), 3: np.array([], dtype=np.int64)},
        "area_ids": ROW_IDS.copy(),
        "component_sha256": dict(COMPONENTS),
        "id_column": "admin_code",
        "polygon_centroids": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        "polygon_group_mapping": {i: int(a) for i, a in enumerate(ROW_IDS)},
        "polygon_id_mapping": {int(a): i for i, a in enumerate(ROW_IDS)},
        "polygons": 4,
    }


def test_normalize_area_ids_accepts_exact_integers():
    assert geo.normalize_area_ids(["101324", 101324.0, " 7 "]).tolist() == [101324, 101324, 7]


@pytest.mark.parametrize("bad", ["1.5", "", None, "abc", float("nan")])
def test_normalize_area_ids_rejects(bad):
    with pytest.raises(ContractError):
        geo.normalize_area_ids([bad])


def test_country_lookup_keeps_blank_iso_and_uses_fallback_key(tmp_path):
    path = tmp_path / "lookup.csv"
    path.write_text(
        "area_id,iso3,country,country_code,country_en\n"
        "2,,Somaliland, ,\n"
        "1,KEN,Kenya,KE, Kenya \n",
        encoding="utf-8",
    )
    frame, audit = geo.load_country_lookup(path)
    assert frame["area_id"].tolist() == [1, 2]
    assert frame["country_key"].tolist() == ["Kenya", "Somaliland"]
    assert audit["areas_missing_iso3"] == 1 and audit["areas_missing_country_code"] == 1


@pytest.mark.parametrize(
    "body",
    ["1,KEN,,KE,\n", "1,KEN,Kenya,KE,Kenya\n1,KEN,Kenya,KE,Kenya\n"],
    ids=["no-country-name", "duplicate-area"],
)
def test_country_lookup_rejects(tmp_path, body):
    path = tmp_path / "lookup.csv"
    path.write_text("area_id,iso3,country,country_code,country_en\n" + body, encoding="utf-8")
    with pytest.raises(ContractError):
        geo.load_country_lookup(path)


def test_valid_cache_passes_and_reports_graph():
    out = geo.validate_adjacency_cache(_payload(), ROW_IDS, COMPONENTS, geometry_centroids=_payload()["polygon_centroids"])
    assert out["edges_undirected"] == 2 and out["isolated"] == 1 and out["directed_entries"] == 4


def _mutations():
    def asym(p):
        p["adjacency_dict"][2] = np.array([], dtype=np.int64)

    def self_loop(p):
        p["adjacency_dict"][3] = np.array([3])

    def out_of_range(p):
        p["adjacency_dict"][3] = np.array([9])

    def duplicate_neighbour(p):
        p["adjacency_dict"][1] = np.array([0, 0, 2])

    def reversed_direction(p):
        # index->area stored where area->index belongs (the FEWS prepare bug class)
        p["polygon_id_mapping"] = {i: int(a) for i, a in enumerate(ROW_IDS)}

    def not_inverse(p):
        p["polygon_group_mapping"][0], p["polygon_group_mapping"][1] = 20, 10

    def area_ids_order(p):
        p["area_ids"] = np.sort(ROW_IDS)

    def hash_mismatch(p):
        p["component_sha256"]["g.dbf"] = "c" * 64

    def extra_key(p):
        p["learned_map"] = {}

    def wrong_id_column(p):
        p["id_column"] = "FEWSNET_ID"

    def centroid_shift(p):
        p["polygon_centroids"] = p["polygon_centroids"] + 1e-6

    return [asym, self_loop, out_of_range, duplicate_neighbour, reversed_direction, not_inverse,
            area_ids_order, hash_mismatch, extra_key, wrong_id_column, centroid_shift]


@pytest.mark.parametrize("mutate", _mutations(), ids=lambda f: f.__name__)
def test_corrupt_cache_is_rejected(mutate):
    payload = _payload()
    centroids = copy.deepcopy(payload["polygon_centroids"])
    mutate(payload)
    with pytest.raises(ContractError):
        geo.validate_adjacency_cache(payload, ROW_IDS, COMPONENTS, geometry_centroids=centroids)


def test_cache_row_order_must_match_geometry():
    with pytest.raises(ContractError):
        geo.validate_adjacency_cache(_payload(), ROW_IDS[::-1].copy(), COMPONENTS)


def _write_layer(path, ids, epsg=4326):
    import geopandas as gpd
    from shapely.geometry import box

    frame = gpd.GeoDataFrame(
        {"admin_code": [str(i) for i in ids]},
        geometry=[box(k, 0, k + 1, 1) for k in range(len(ids))],
        crs=f"EPSG:{epsg}",
    )
    frame.to_file(path, engine="pyogrio")


def test_geometry_reader_keeps_row_order_and_centroids(tmp_path):
    path = tmp_path / "layer.shp"
    _write_layer(path, [35, 7, 10])
    gdf, ids, audit = geo.load_geometry(path)
    assert ids.tolist() == [35, 7, 10] and audit["engine"] == "pyogrio"
    assert np.allclose(geo.geometry_centroids_latlon(gdf), [[0.5, 0.5], [0.5, 1.5], [0.5, 2.5]])


@pytest.mark.parametrize("ids, epsg", [([1, 1, 2], 4326), ([1, 2, 3], 3857)], ids=["duplicate-id", "wrong-crs"])
def test_geometry_reader_rejects(tmp_path, ids, epsg):
    path = tmp_path / "layer.shp"
    _write_layer(path, ids, epsg)
    with pytest.raises(ContractError):
        geo.load_geometry(path)
