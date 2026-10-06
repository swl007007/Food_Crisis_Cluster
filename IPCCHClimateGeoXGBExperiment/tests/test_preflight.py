"""Preflight stop conditions on synthetic inputs (the real run is P0 evidence)."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from ipcch_climate_geoxgb import geography as geo
from ipcch_climate_geoxgb import preflight as pf
from ipcch_climate_geoxgb.errors import ContractError

HEADER = ",".join(pf.RAW_REQUIRED_COLUMNS)


def _entry(path):
    data = path.read_bytes()
    return {"path_windows": str(path), "path_wsl": str(path), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "role": "test"}


def test_identity_match_and_stops(tmp_path):
    path = tmp_path / "x.csv"
    path.write_bytes(b"a,b\n1,2\n")
    good = _entry(path)
    assert pf.verify_identities({"x": good})["x"]["match"]
    for bad in ({**good, "sha256": "0" * 64}, {**good, "bytes": good["bytes"] + 1},
                {**good, "path_windows": str(tmp_path / "gone"), "path_wsl": str(tmp_path / "gone")}):
        with pytest.raises(ContractError):
            pf.verify_identities({"x": bad})


def _lookup_and_reference():
    lookup = pd.DataFrame({"area_id": [7, 10], "country_key": ["Kenya", "Somalia"]})
    reference = pd.DataFrame({"area_id": [7, 10], "ref_lat": [1.0, 2.0], "ref_lon": [30.0, 40.0]})
    return lookup, reference


def _row(area, year, month, country="Kenya", lat="1.0", lon="30.0"):
    values = {c: "" for c in pf.RAW_REQUIRED_COLUMNS}
    values.update(admin_code=str(area), year=str(year), month=str(month), country_en=country, lat=lat, lon=lon)
    return ",".join(values[c] for c in pf.RAW_REQUIRED_COLUMNS)


def _panel(tmp_path, rows):
    path = tmp_path / "panel.csv"
    path.write_text(HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_raw_panel_passes_and_counts_rounding(tmp_path):
    lookup, reference = _lookup_and_reference()
    rows = [_row(7, 2020, 1, lat="1.00000000004"), _row(7, 2020, 2), _row(10, 2020, 1, "Somalia", "2.0", "40.0")]
    out = pf.check_raw_panel(_panel(tmp_path, rows), np.array([7, 10]), lookup, reference, 1e-9)
    assert out["rows"] == 3 and out["coordinate_rows_not_identical"] == 1


#: A valid row for the second universe area, so each case trips only its own check.
OTHER = _row(10, 2020, 1, "Somalia", "2.0", "40.0")


@pytest.mark.parametrize(
    "rows, match",
    [
        ([_row(7, 2020, 1), _row(7, 2020, 1), OTHER], "duplicate"),
        ([_row(7, 2020, 1), _row(99, 2020, 1), OTHER], "outside the geography universe"),
        ([_row(7, 2020, 1)], "lacks rows for 1 universe areas"),
        ([_row(7, 2020, 1, "Somalia"), OTHER], "disagree with the country lookup"),
        ([_row(7, 2020, 1), _row(7, 2020, 2, "Uganda"), OTHER], "more than one country"),
        ([_row(7, 2020, 1, lat="1.001"), OTHER], "from the reference point"),
        ([_row(7, 2020, 13), OTHER], "month outside"),
        ([_row(7, 2020, 1, lat=""), OTHER], "non-finite lat/lon"),
        ([_row("7.5", 2020, 1), OTHER], "not an integer"),
    ],
    ids=["duplicate-key", "area-outside-universe", "universe-area-missing", "country-vs-lookup",
         "two-countries", "coordinate-off", "bad-month", "blank-lat", "non-integer-area"],
)
def test_raw_panel_stops(tmp_path, rows, match):
    lookup, reference = _lookup_and_reference()
    with pytest.raises(ContractError, match=match):
        pf.check_raw_panel(_panel(tmp_path, rows), np.array([7, 10]), lookup, reference, 1e-9)


def test_raw_panel_missing_required_column(tmp_path):
    path = tmp_path / "panel.csv"
    path.write_text("admin_code,year,month\n7,2020,1\n", encoding="utf-8")
    lookup, reference = _lookup_and_reference()
    with pytest.raises(ContractError):
        pf.check_raw_panel(path, np.array([7, 10]), lookup, reference, 1e-9)


def test_universe_mismatch_stops():
    assert pf.reconcile_universe([7, 10], [10, 7], [7, 10], 2).tolist() == [7, 10]
    with pytest.raises(ContractError):
        pf.reconcile_universe([7, 10], [7], [7, 10], 2)
    with pytest.raises(ContractError):
        pf.reconcile_universe([7, 10], [7, 10], [7, 10], 3)


def _saved_tables(tmp_path, saved_lat="1.0"):
    (tmp_path / "lookup.csv").write_text(
        "area_id,iso3,country,country_code,country_en\n7,KEN,Kenya,KE,Kenya\n", encoding="utf-8")
    (tmp_path / "saved_lookup.csv").write_text(
        "area_id,iso3,country,country_code,country_en,country_key\n7,KEN,Kenya,KE,Kenya,Kenya\n", encoding="utf-8")
    (tmp_path / "ref.csv").write_text("area_id,lat,lon\n7,1.0,30.0\n", encoding="utf-8")
    (tmp_path / "saved_ref.csv").write_text(f"area_id,ref_lat,ref_lon\n7,{saved_lat},30.0\n", encoding="utf-8")
    lookup, _ = geo.load_country_lookup(tmp_path / "lookup.csv")
    reference = geo.load_reference_coordinates(tmp_path / "ref.csv")
    return lookup, reference


def test_saved_tables_equal_and_stop(tmp_path):
    lookup, reference = _saved_tables(tmp_path)
    assert pf.check_saved_tables(lookup, tmp_path / "saved_lookup.csv", reference, tmp_path / "saved_ref.csv")[
        "reference_coordinates_equal"
    ]
    lookup, reference = _saved_tables(tmp_path, saved_lat="1.0000001")
    with pytest.raises(ContractError):
        pf.check_saved_tables(lookup, tmp_path / "saved_lookup.csv", reference, tmp_path / "saved_ref.csv")
