"""Prepare and execute the gated Ethiopia ERA5-Drought SPI campaign."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import stat
import sys
import zipfile
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_COHORT = (
    EXPERIMENT_DIR
    / "outputs"
    / "baseline_audit"
    / "fewsnet_eth_pre_georf_20260831"
    / "fewsnet_eth_pre_georf.csv.gz"
)
DEFAULT_MANIFEST = (
    EXPERIMENT_DIR / "manifests" / "ethiopia_spi_campaign_2010_2024_v1.json"
)
DEFAULT_CAMPAIGN_ROOT = (
    EXPERIMENT_DIR / "data" / "raw" / "era5_drought_spi" / "campaign_2010_2024_v1"
)
DEFAULT_OUTPUT_ROOT = EXPERIMENT_DIR / "data" / "interim" / "era5_drought_spi"

KEY = "FEWSNET_admin_code"
DATE = "date"
EXPECTED_ROWS = 187_200
EXPECTED_ADMINS = 1_040
EXPECTED_MONTHS = 180
START_MONTH = "2010-01"
END_MONTH = "2024-12"

CDS_DATASET = "derived-drought-historical-monthly"
PRODUCT = "ERA5-Drought/DRYFALL"
PRODUCT_VERSION = "1_0"
PRODUCT_TYPE = "reanalysis"
DATASET_TYPE = "consolidated_dataset"
REFERENCE_PERIOD = "1991-01-01/2020-12-31"
SPI_VARIABLE = "standardised_precipitation_index"
P0_VARIABLE = "probability_of_zero_precipitation_spi"
NORMALITY_VARIABLE = "test_for_normality_spi"
SPI_SCALES = (1, 3, 6, 12)
GRID_RESOLUTION_DEGREES = 0.25
P0_THRESHOLD = 0.66
MINIMUM_COVERAGE = 0.95


class EthiopiaSpiContractError(RuntimeError):
    """Raised when the approved Ethiopia SPI contract is violated."""


@dataclass(frozen=True)
class EthiopiaSpiUniverse:
    area_ids: tuple[str, ...]
    cohort_path: Path
    cohort_sha256: str
    cohort_key_sha256: str
    geometry_path: Path
    geometry_bundle: tuple[dict[str, Any], ...]
    geometry_bundle_sha256: str
    geometry_feature_count: int
    geometry_crs: str
    geometry_bounds_wsen: tuple[float, float, float, float]
    request_area_nwse: tuple[float, float, float, float]


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize stable JSON bytes for scientific identity hashes."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    """Hash a stable JSON representation."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def campaign_identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the manifest fields covered by its embedded hash."""
    identity = dict(manifest)
    identity.pop("manifest_sha256", None)
    return identity


def normalize_admin_codes(values: Sequence[object] | pd.Series) -> pd.Series:
    """Normalize numeric-looking FEWS NET codes without changing membership."""
    return (
        pd.Series(values, dtype="string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def _admin_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _cohort_key_sha256(frame: pd.DataFrame) -> str:
    keys = frame[[KEY, DATE]].copy()
    keys[KEY] = normalize_admin_codes(keys[KEY]).to_numpy()
    keys[DATE] = pd.to_datetime(keys[DATE], errors="raise").dt.strftime("%Y-%m-%d")
    keys = keys.sort_values([KEY, DATE], kind="stable")
    payload = "".join(
        f"{admin}|{date}\n"
        for admin, date in keys.itertuples(index=False, name=None)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _aligned_request_area(
    bounds_wsen: Sequence[float],
) -> tuple[float, float, float, float]:
    west, south, east, north = (float(value) for value in bounds_wsen)
    step = GRID_RESOLUTION_DEGREES
    area = (
        math.ceil(north / step) * step,
        math.floor(west / step) * step,
        math.floor(south / step) * step,
        math.ceil(east / step) * step,
    )
    if area[0] <= area[2] or area[3] <= area[1]:
        raise EthiopiaSpiContractError("Ethiopia geometry has an invalid extent")
    return area


def _shapefile_bundle(path: Path) -> tuple[dict[str, Any], ...]:
    required = (".shp", ".shx", ".dbf", ".prj")
    optional = (".cpg",)
    files: list[dict[str, Any]] = []
    for suffix in (*required, *optional):
        member = path.with_suffix(suffix)
        if suffix in required and not member.is_file():
            raise EthiopiaSpiContractError(f"Shapefile member is missing: {member}")
        if member.is_file():
            files.append(
                {
                    "name": member.name,
                    "sha256": file_sha256(member),
                    "size_bytes": member.stat().st_size,
                }
            )
    return tuple(files)


def derive_ethiopia_spi_universe(
    cohort_path: Path, geometry_path: Path
) -> EthiopiaSpiUniverse:
    """Validate the frozen cohort and its exact FEWS NET polygon universe."""
    import geopandas as gpd

    cohort = pd.read_csv(
        cohort_path, usecols=["ISO3", KEY, DATE], low_memory=False
    )
    if len(cohort) != EXPECTED_ROWS:
        raise EthiopiaSpiContractError(f"Unexpected cohort rows: {len(cohort)}")
    if set(cohort["ISO3"].dropna().unique()) != {"ETH"}:
        raise EthiopiaSpiContractError("Cohort is not exact ISO3 == 'ETH'")
    cohort[KEY] = normalize_admin_codes(cohort[KEY]).to_numpy()
    cohort[DATE] = pd.to_datetime(cohort[DATE], errors="raise")
    if cohort[[KEY, DATE]].isna().any().any() or cohort.duplicated([KEY, DATE]).any():
        raise EthiopiaSpiContractError("Cohort contains null or duplicate keys")
    periods = pd.PeriodIndex(cohort[DATE], freq="M")
    expected_periods = pd.period_range(START_MONTH, END_MONTH, freq="M")
    if (
        cohort[KEY].nunique() != EXPECTED_ADMINS
        or periods.nunique() != EXPECTED_MONTHS
        or not pd.Index(periods.unique()).sort_values().equals(expected_periods)
    ):
        raise EthiopiaSpiContractError("Cohort dimensions or calendar differ")
    counts = cohort.groupby(KEY, sort=False)[DATE].size()
    if not counts.eq(EXPECTED_MONTHS).all():
        raise EthiopiaSpiContractError("Cohort is not a complete admin-month grid")
    area_ids = tuple(sorted(cohort[KEY].unique(), key=_admin_sort_key))

    geometry = gpd.read_file(geometry_path)
    if "admin_code" not in geometry.columns:
        raise EthiopiaSpiContractError("Geometry is missing admin_code")
    geometry[KEY] = normalize_admin_codes(geometry["admin_code"]).to_numpy()
    if geometry[KEY].isna().any() or geometry[KEY].duplicated().any():
        raise EthiopiaSpiContractError("Geometry admin_code is null or duplicated")
    selected = geometry.loc[geometry[KEY].isin(area_ids), [KEY, "geometry"]].copy()
    missing = sorted(set(area_ids).difference(selected[KEY]), key=_admin_sort_key)
    if missing or len(selected) != EXPECTED_ADMINS:
        raise EthiopiaSpiContractError(
            f"Geometry reconciliation failed: missing={len(missing)} selected={len(selected)}"
        )
    if geometry.crs is None:
        raise EthiopiaSpiContractError("Geometry CRS is missing")
    selected = selected.to_crs(4326)
    if (
        selected.geometry.isna().any()
        or selected.geometry.is_empty.any()
        or not selected.geometry.is_valid.all()
    ):
        raise EthiopiaSpiContractError("Selected Ethiopia geometry is invalid")
    bounds = tuple(float(value) for value in selected.total_bounds)
    bundle = _shapefile_bundle(geometry_path)
    return EthiopiaSpiUniverse(
        area_ids=area_ids,
        cohort_path=cohort_path.resolve(),
        cohort_sha256=file_sha256(cohort_path),
        cohort_key_sha256=_cohort_key_sha256(cohort),
        geometry_path=geometry_path.resolve(),
        geometry_bundle=bundle,
        geometry_bundle_sha256=canonical_json_sha256(bundle),
        geometry_feature_count=len(selected),
        geometry_crs=str(selected.crs),
        geometry_bounds_wsen=bounds,
        request_area_nwse=_aligned_request_area(bounds),
    )


def _request_area_suffix(area: Sequence[float]) -> str:
    north, west, south, east = area
    return ".area-subset." + ".".join(
        f"{value:g}" for value in (north, east, south, west)
    ) + ".nc"


def _provider_request(
    *,
    scale: int,
    variables: Sequence[str],
    year: str,
    months: Sequence[str],
    area: Sequence[float],
) -> dict[str, Any]:
    return {
        "variable": list(variables),
        "accumulation_period": [str(scale)],
        "version": PRODUCT_VERSION,
        "product_type": PRODUCT_TYPE,
        "dataset_type": DATASET_TYPE,
        "year": [year],
        "month": list(months),
        "area": list(area),
        "data_format": "netcdf",
        "download_format": "zip",
    }


def runtime_environment() -> dict[str, Any]:
    """Capture stable runtime identity without timestamps or process IDs."""
    packages = {}
    for name in (
        "numpy",
        "pandas",
        "geopandas",
        "shapely",
        "pyproj",
        "exactextract",
        "netCDF4",
        "cdsapi",
    ):
        packages[name] = importlib.metadata.version(name)
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }


def build_campaign_manifest(
    universe: EthiopiaSpiUniverse,
    *,
    campaign_root: Path,
    campaign_id: str = "ethiopia-era5-drought-spi-2010-2024-v1",
) -> dict[str, Any]:
    """Build the exact disabled-by-default 64-request campaign."""
    months = tuple(f"{month:02d}" for month in range(1, 13))
    suffix = _request_area_suffix(universe.request_area_nwse)
    requests: list[dict[str, Any]] = []
    for scale in SPI_SCALES:
        for year in range(2010, 2025):
            payload = _provider_request(
                scale=scale,
                variables=(SPI_VARIABLE,),
                year=str(year),
                months=months,
                area=universe.request_area_nwse,
            )
            requests.append(
                {
                    "request_id": f"spi{scale}-{year}",
                    "scale": scale,
                    "kind": "spi",
                    "payload": payload,
                    "request_sha256": canonical_json_sha256(payload),
                    "target_relative_path": f"raw/spi{scale}-{year}.zip",
                    "expected_member_count": 12,
                    "expected_members": [
                        f"SPI{scale}_gamma_global_era5_moda_ref1991to2020_"
                        f"{year}{month}{suffix}"
                        for month in months
                    ],
                }
            )
        payload = _provider_request(
            scale=scale,
            variables=(P0_VARIABLE, NORMALITY_VARIABLE),
            year="2020",
            months=months,
            area=universe.request_area_nwse,
        )
        quality_members = [
            name
            for month in months
            for name in (
                f"SPI{scale}_spipzero_gamma_global_era5_moda_ref1991to2020_"
                f"{month}{suffix}",
                f"SPI{scale}_spisignificance_gamma_global_era5_moda_ref1991to2020_"
                f"{month}{suffix}",
            )
        ]
        requests.append(
            {
                "request_id": f"spi{scale}-quality-calendar",
                "scale": scale,
                "kind": "quality",
                "payload": payload,
                "request_sha256": canonical_json_sha256(payload),
                "target_relative_path": f"raw/spi{scale}-quality-calendar.zip",
                "expected_member_count": 24,
                "expected_members": quality_members,
            }
        )

    manifest = {
        "schema_version": "ethiopia-era5-drought-spi-campaign-v1",
        "campaign_id": campaign_id,
        "completion_status": "prepared",
        "submission_enabled_by_default": False,
        "cloud_operation_performed": False,
        "approval_status": "awaiting_exact_manifest_hash_approval",
        "campaign_root": str(campaign_root.resolve()),
        "provider": {
            "name": "Copernicus Climate Data Store",
            "endpoint": "https://cds.climate.copernicus.eu/api",
            "dataset": CDS_DATASET,
            "product": PRODUCT,
            "version": PRODUCT_VERSION,
            "product_type": PRODUCT_TYPE,
            "dataset_type": DATASET_TYPE,
            "reference_period": REFERENCE_PERIOD,
            "variables": [SPI_VARIABLE, P0_VARIABLE, NORMALITY_VARIABLE],
            "scales": list(SPI_SCALES),
            "native_grid": "0.25 degree EPSG:4326",
            "doi": "10.24381/9bea5e16",
            "license": "CC-BY-4.0",
        },
        "universe": {
            **asdict(universe),
            "area_ids": list(universe.area_ids),
            "area_ids_sha256": canonical_json_sha256(list(universe.area_ids)),
            "cohort_path": str(universe.cohort_path),
            "geometry_path": str(universe.geometry_path),
        },
        "calendar": {
            "start": START_MONTH,
            "end": END_MONTH,
            "month_count_per_scale": EXPECTED_MONTHS,
            "temporal_alignment": "source_calendar_month_no_lag",
        },
        "aggregation_contract": {
            "spatial_rule": "native_grid_fractional_intersection_spherical_cell_area_weighted_mean",
            "pixel_validity_rule": "finite_spi_and_finite_p0_and_p0_lt_0.66",
            "p0_threshold": P0_THRESHOLD,
            "normality_use": "diagnostic_only",
            "minimum_valid_coverage_fraction": MINIMUM_COVERAGE,
            "reprojection": False,
            "interpolation": False,
            "filling": False,
        },
        "output_contract": {
            "public_key": [KEY, DATE],
            "public_columns": ["SPI_1", "SPI_3", "SPI_6", "SPI_12"],
            "qa_key": [KEY, DATE, "spi_scale"],
            "qa_separate_from_model_table": True,
        },
        "execution_policy": {
            "requires_explicit_manifest_sha256": True,
            "maximum_concurrency": 2,
            "runner_concurrency": 1,
            "automatic_retry_policy": "none",
            "automatic_cleanup": False,
            "overwrite_policy": "refuse_existing_target",
            "failed_part_files_retained": True,
            "resume_requires_new_manifest_approval": True,
        },
        "expected_counts": {
            "area_count": EXPECTED_ADMINS,
            "public_area_month_rows": EXPECTED_ROWS,
            "qa_area_month_scale_rows": EXPECTED_ROWS * len(SPI_SCALES),
            "requests": len(requests),
            "spi_requests": 60,
            "quality_requests": 4,
            "returned_members": sum(
                request["expected_member_count"] for request in requests
            ),
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "runtime": runtime_environment(),
        "requests": requests,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(
        campaign_identity_payload(manifest)
    )
    return manifest


def validate_campaign_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed when any prepared campaign identity field drifts."""
    if manifest.get("schema_version") != "ethiopia-era5-drought-spi-campaign-v1":
        raise EthiopiaSpiContractError("Campaign schema version differs")
    provider = manifest.get("provider", {})
    expected_provider = {
        "dataset": CDS_DATASET,
        "product": PRODUCT,
        "version": PRODUCT_VERSION,
        "product_type": PRODUCT_TYPE,
        "dataset_type": DATASET_TYPE,
        "reference_period": REFERENCE_PERIOD,
    }
    for field, expected in expected_provider.items():
        if provider.get(field) != expected:
            raise EthiopiaSpiContractError(f"Provider {field} differs")
    if tuple(provider.get("scales", ())) != SPI_SCALES:
        raise EthiopiaSpiContractError("Campaign scales differ")
    if manifest.get("submission_enabled_by_default") is not False:
        raise EthiopiaSpiContractError("Campaign must be disabled by default")
    counts = manifest.get("expected_counts", {})
    expected_counts = {
        "area_count": EXPECTED_ADMINS,
        "public_area_month_rows": EXPECTED_ROWS,
        "qa_area_month_scale_rows": EXPECTED_ROWS * len(SPI_SCALES),
        "requests": 64,
        "spi_requests": 60,
        "quality_requests": 4,
        "returned_members": 816,
    }
    if any(counts.get(key) != value for key, value in expected_counts.items()):
        raise EthiopiaSpiContractError("Campaign expected counts differ")
    requests = manifest.get("requests", [])
    ids = [request.get("request_id") for request in requests]
    targets = [request.get("target_relative_path") for request in requests]
    if len(requests) != 64 or len(set(ids)) != 64 or len(set(targets)) != 64:
        raise EthiopiaSpiContractError("Campaign requests are duplicated or incomplete")
    observed_spi: set[tuple[int, int]] = set()
    observed_quality: set[int] = set()
    request_area = list(manifest["universe"]["request_area_nwse"])
    for request in requests:
        path = PurePosixPath(str(request["target_relative_path"]))
        if path.is_absolute() or ".." in path.parts or path.parts[:1] != ("raw",):
            raise EthiopiaSpiContractError("Request target path is unsafe")
        payload = request.get("payload", {})
        if canonical_json_sha256(payload) != request.get("request_sha256"):
            raise EthiopiaSpiContractError("Request payload hash differs")
        if payload.get("area") != request_area:
            raise EthiopiaSpiContractError("Request area differs")
        expected_members = request.get("expected_members", [])
        if len(expected_members) != request.get("expected_member_count"):
            raise EthiopiaSpiContractError("Expected member count differs")
        scale = int(request["scale"])
        if scale not in SPI_SCALES:
            raise EthiopiaSpiContractError("Request scale differs")
        if request["kind"] == "spi":
            year = int(payload["year"][0])
            observed_spi.add((scale, year))
            if payload["variable"] != [SPI_VARIABLE] or len(expected_members) != 12:
                raise EthiopiaSpiContractError("SPI request contract differs")
        elif request["kind"] == "quality":
            observed_quality.add(scale)
            if (
                payload["variable"] != [P0_VARIABLE, NORMALITY_VARIABLE]
                or payload["year"] != ["2020"]
                or len(expected_members) != 24
            ):
                raise EthiopiaSpiContractError("Quality request contract differs")
        else:
            raise EthiopiaSpiContractError("Unknown request kind")
    expected_spi = {
        (scale, year) for scale in SPI_SCALES for year in range(2010, 2025)
    }
    if observed_spi != expected_spi or observed_quality != set(SPI_SCALES):
        raise EthiopiaSpiContractError("Request calendar is incomplete")
    runner = manifest.get("runner", {})
    runner_path = Path(str(runner.get("path", "")))
    if not runner_path.is_file() or file_sha256(runner_path) != runner.get("sha256"):
        raise EthiopiaSpiContractError("Campaign runner hash differs")
    supplied = manifest.get("manifest_sha256")
    if supplied != canonical_json_sha256(campaign_identity_payload(manifest)):
        raise EthiopiaSpiContractError("Campaign manifest hash differs")


def write_campaign_manifest(manifest: Mapping[str, Any], output_path: Path) -> None:
    """Atomically create a prepared manifest without overwriting prior identity."""
    validate_campaign_manifest(manifest)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    temp.unlink(missing_ok=True)
    try:
        temp.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        observed = json.loads(temp.read_text(encoding="utf-8"))
        validate_campaign_manifest(observed)
        os.replace(temp, output_path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def _validate_credential_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EthiopiaSpiContractError(f"CDS credential file is missing: {path}")
    info = path.stat()
    mode = stat.S_IMODE(info.st_mode)
    if info.st_size <= 0:
        raise EthiopiaSpiContractError("CDS credential file is empty")
    return {
        "path": str(path.resolve()),
        "owner_uid": info.st_uid,
        "group_gid": info.st_gid,
        "mode_octal": oct(mode),
        "size_bytes": info.st_size,
    }


def download_campaign(
    manifest_path: Path,
    *,
    approved_manifest_sha256: str,
    credential_path: Path,
) -> Path:
    """Execute the gated CDS campaign serially with no retry or overwrite."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_campaign_manifest(manifest)
    if manifest["manifest_sha256"] != approved_manifest_sha256:
        raise EthiopiaSpiContractError("Approved manifest SHA-256 differs")
    credential_metadata = _validate_credential_metadata(credential_path)
    try:
        import cdsapi
    except ImportError as exc:
        raise EthiopiaSpiContractError("cdsapi is required for download") from exc
    campaign_root = Path(manifest["campaign_root"])
    inventory_path = campaign_root / "download_inventory.jsonl"
    if inventory_path.exists():
        raise EthiopiaSpiContractError("Download inventory already exists")
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    with inventory_path.open("x", encoding="utf-8") as inventory:
        inventory.write(
            json.dumps({"record_type": "credential_metadata", **credential_metadata})
            + "\n"
        )
        inventory.flush()
        for request in manifest["requests"]:
            target = campaign_root / request["target_relative_path"]
            part = target.with_suffix(target.suffix + ".part")
            if target.exists() or part.exists():
                raise EthiopiaSpiContractError(
                    f"Campaign target or part already exists: {target}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            client.retrieve(CDS_DATASET, request["payload"], str(part))
            os.replace(part, target)
            inventory.write(
                json.dumps(
                    {
                        "record_type": "download",
                        "request_id": request["request_id"],
                        "path": str(target),
                        "sha256": file_sha256(target),
                        "size_bytes": target.stat().st_size,
                    }
                )
                + "\n"
            )
            inventory.flush()
    return inventory_path


def _normalize_regular_grid(
    *, spi: Any, p0: Any, normality: Any, latitudes: Any, longitudes: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spi_array = np.asarray(spi, dtype="float64")
    p0_array = np.asarray(p0, dtype="float64")
    normality_array = np.asarray(normality, dtype="float64")
    latitude_array = np.asarray(latitudes, dtype="float64")
    longitude_array = np.asarray(longitudes, dtype="float64")
    if (
        spi_array.ndim != 2
        or p0_array.shape != spi_array.shape
        or normality_array.shape != spi_array.shape
    ):
        raise EthiopiaSpiContractError("SPI, P0, and normality grids differ")
    if (
        latitude_array.ndim != 1
        or longitude_array.ndim != 1
        or latitude_array.size != spi_array.shape[0]
        or longitude_array.size != spi_array.shape[1]
        or latitude_array.size < 2
        or longitude_array.size < 2
    ):
        raise EthiopiaSpiContractError("Grid coordinate dimensions differ")
    lat_diff = np.diff(latitude_array)
    lon_diff = np.diff(longitude_array)
    if not (np.all(lat_diff > 0) or np.all(lat_diff < 0)) or not (
        np.all(lon_diff > 0) or np.all(lon_diff < 0)
    ):
        raise EthiopiaSpiContractError("Grid coordinates are not monotonic")
    if not np.allclose(np.abs(lat_diff), abs(lat_diff[0]), rtol=0, atol=1e-10):
        raise EthiopiaSpiContractError("Latitude grid is irregular")
    if not np.allclose(np.abs(lon_diff), abs(lon_diff[0]), rtol=0, atol=1e-10):
        raise EthiopiaSpiContractError("Longitude grid is irregular")
    if lat_diff[0] > 0:
        latitude_array = latitude_array[::-1]
        spi_array = spi_array[::-1, :]
        p0_array = p0_array[::-1, :]
        normality_array = normality_array[::-1, :]
    if lon_diff[0] < 0:
        longitude_array = longitude_array[::-1]
        spi_array = spi_array[:, ::-1]
        p0_array = p0_array[:, ::-1]
        normality_array = normality_array[:, ::-1]
    return spi_array, p0_array, normality_array, latitude_array, longitude_array


def _spherical_cell_areas_m2(
    latitudes: np.ndarray, longitudes: np.ndarray
) -> np.ndarray:
    latitude_step = abs(float(latitudes[0] - latitudes[1]))
    longitude_step = abs(float(longitudes[1] - longitudes[0]))
    north_edges = latitudes + latitude_step / 2
    south_edges = latitudes - latitude_step / 2
    radius_m = 6_371_008.8
    row_areas = radius_m**2 * np.deg2rad(longitude_step) * np.abs(
        np.sin(np.deg2rad(north_edges)) - np.sin(np.deg2rad(south_edges))
    )
    return np.repeat(row_areas[:, None], len(longitudes), axis=1)


def aggregate_spi_month(
    geometry_frame: Any,
    *,
    spi: Any,
    p0: Any,
    normality: Any,
    latitudes: Any,
    longitudes: Any,
    scale: int,
    source_month: str,
    minimum_valid_coverage_fraction: float = MINIMUM_COVERAGE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the frozen Kenya polygon-weighted aggregation contract."""
    from exactextract import exact_extract
    from exactextract.feature import JSONFeatureSource
    from exactextract.raster import NumPyRasterSource
    from pyproj import CRS

    if scale not in SPI_SCALES:
        raise EthiopiaSpiContractError("Unsupported SPI scale")
    if not 0 <= minimum_valid_coverage_fraction <= 1:
        raise EthiopiaSpiContractError("Coverage threshold is outside [0, 1]")
    frame = geometry_frame.to_crs(4326).copy()
    if KEY not in frame or frame[KEY].isna().any() or frame[KEY].duplicated().any():
        raise EthiopiaSpiContractError("Geometry keys are invalid")
    frame[KEY] = normalize_admin_codes(frame[KEY]).to_numpy()
    features = json.loads(frame[[KEY, "geometry"]].to_json())["features"]
    spi_array, p0_array, normality_array, lat_array, lon_array = (
        _normalize_regular_grid(
            spi=spi,
            p0=p0,
            normality=normality,
            latitudes=latitudes,
            longitudes=longitudes,
        )
    )
    lat_step = abs(float(lat_array[0] - lat_array[1]))
    lon_step = abs(float(lon_array[1] - lon_array[0]))
    xmin = float(lon_array[0] - lon_step / 2)
    xmax = float(lon_array[-1] + lon_step / 2)
    ymin = float(lat_array[-1] - lat_step / 2)
    ymax = float(lat_array[0] + lat_step / 2)
    cell_areas = _spherical_cell_areas_m2(lat_array, lon_array)
    finite_spi = np.isfinite(spi_array)
    p0_eligible = np.isfinite(p0_array) & (p0_array < P0_THRESHOLD)
    final_valid = finite_spi & p0_eligible
    nodata = -3.4028234663852886e38
    srs_wkt = CRS.from_epsg(4326).to_wkt()

    def raster(values: Any, name: str, nodata_value: float | None = None) -> Any:
        return NumPyRasterSource(
            np.asarray(values, dtype="float64"),
            xmin,
            ymin,
            xmax,
            ymax,
            nodata=nodata_value,
            name=name,
            srs_wkt=srs_wkt,
        )

    vectors = JSONFeatureSource(features, srs_wkt=srs_wkt)
    area_raster = raster(cell_areas, "cell_area_m2")

    def extract(
        values: Any,
        operation: str,
        *,
        nodata_value: float | None = None,
        weights: Any = None,
    ) -> dict[str, Any]:
        output = exact_extract(
            raster(values, "value", nodata_value),
            vectors,
            [operation],
            weights=weights,
            include_cols=[KEY],
            output="geojson",
        )
        return {
            str(item["properties"][KEY]): item["properties"].get(operation)
            for item in output
        }

    total_area = extract(cell_areas, "sum")
    finite_area = extract(np.where(finite_spi, cell_areas, 0.0), "sum")
    p0_area = extract(np.where(p0_eligible, cell_areas, 0.0), "sum")
    valid_area = extract(np.where(final_valid, cell_areas, 0.0), "sum")
    valid_spi = np.where(final_valid, spi_array, nodata)
    raw_mean = extract(
        valid_spi, "weighted_mean", nodata_value=nodata, weights=area_raster
    )
    p0_mean = extract(
        np.where(np.isfinite(p0_array), p0_array, nodata),
        "weighted_mean",
        nodata_value=nodata,
        weights=area_raster,
    )
    normality_mean = extract(
        np.where(np.isfinite(normality_array), normality_array, nodata),
        "weighted_mean",
        nodata_value=nodata,
        weights=area_raster,
    )
    source_period = pd.Period(source_month, freq="M")
    public_rows: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []
    for admin_code in frame[KEY]:
        total = float(total_area.get(admin_code) or 0.0)
        finite = float(finite_area.get(admin_code) or 0.0)
        eligible = float(p0_area.get(admin_code) or 0.0)
        valid = float(valid_area.get(admin_code) or 0.0)
        coverage = valid / total if total > 0 else None
        mean = raw_mean.get(admin_code)
        mean = float(mean) if mean is not None and np.isfinite(float(mean)) else None
        accepted = (
            coverage is not None
            and coverage >= minimum_valid_coverage_fraction
            and mean is not None
        )
        if total <= 0:
            status, reason = "outside_grid", "outside_grid"
        elif valid <= 0 or mean is None:
            status, reason = "source_invalid", "source_invalid"
        elif not accepted:
            status, reason = "coverage_failure", "spi_coverage_failure"
        else:
            status, reason = "valid", None
        row_key = {KEY: admin_code, DATE: source_period.start_time}
        public_rows.append(
            {**row_key, f"SPI_{scale}": mean if accepted else None}
        )
        qa_rows.append(
            {
                **row_key,
                "spi_scale": scale,
                "status": status,
                "missing_reason": reason,
                "total_intersection_area_m2": total,
                "finite_spi_intersection_area_m2": finite,
                "p0_eligible_intersection_area_m2": eligible,
                "final_valid_intersection_area_m2": valid,
                "valid_coverage_fraction": coverage,
                "outside_grid": total <= 0,
                "spi_area_weighted_mean_before_coverage_gate": mean,
                "p0_area_weighted_mean": p0_mean.get(admin_code),
                "normality_raw_area_weighted_mean": normality_mean.get(admin_code),
                "minimum_valid_coverage_fraction": minimum_valid_coverage_fraction,
                "pixel_validity_rule": "finite_spi_and_finite_p0_and_p0_lt_0.66",
                "spatial_aggregation_rule": "native_grid_fractional_intersection_spherical_cell_area_weighted_mean",
            }
        )
    return pd.DataFrame(public_rows), pd.DataFrame(qa_rows)


def validate_netcdf_member(
    path: Path,
    *,
    scale: int,
    kind: Literal["spi", "p0", "normality"],
    expected_source_month: str,
    expected_request_area: Sequence[float],
) -> dict[str, Any]:
    """Validate one immutable provider NetCDF member before aggregation."""
    from netCDF4 import Dataset, num2date

    variable_name = f"SPI{scale}" if kind == "spi" else (
        "pzero" if kind == "p0" else "significance"
    )
    before = path.stat()
    admitted_hash = file_sha256(path)
    with Dataset(path, "r") as handle:
        if not {"lat", "lon", "time", variable_name}.issubset(handle.variables):
            raise EthiopiaSpiContractError("NetCDF variables are incomplete")
        latitudes = np.asarray(handle.variables["lat"][:], dtype=float)
        longitudes = np.asarray(handle.variables["lon"][:], dtype=float)
        variable = handle.variables[variable_name]
        if variable.dimensions != ("time", "lat", "lon") or variable.shape != (
            1,
            len(latitudes),
            len(longitudes),
        ):
            raise EthiopiaSpiContractError("NetCDF dimensions differ")
        if not np.allclose(
            np.abs(np.diff(latitudes)), GRID_RESOLUTION_DEGREES, rtol=0, atol=1e-8
        ) or not np.allclose(
            np.abs(np.diff(longitudes)), GRID_RESOLUTION_DEGREES, rtol=0, atol=1e-8
        ):
            raise EthiopiaSpiContractError("NetCDF grid resolution differs")
        north, west, south, east = (float(value) for value in expected_request_area)
        if not (
            np.isclose(latitudes.min(), south, atol=1e-8)
            and np.isclose(latitudes.max(), north, atol=1e-8)
            and np.isclose(longitudes.min(), west, atol=1e-8)
            and np.isclose(longitudes.max(), east, atol=1e-8)
        ):
            raise EthiopiaSpiContractError("NetCDF request area differs")
        time_variable = handle.variables["time"]
        observed = num2date(
            time_variable[0],
            units=time_variable.units,
            calendar=getattr(time_variable, "calendar", "standard"),
        )
        if f"{observed.year:04d}-{observed.month:02d}" != expected_source_month:
            raise EthiopiaSpiContractError("NetCDF source month differs")
        values = np.asarray(variable[0, :, :], dtype=float)
        finite = values[np.isfinite(values)]
        if kind == "p0" and finite.size and (finite.min() < 0 or finite.max() > 1):
            raise EthiopiaSpiContractError("P0 values are outside [0, 1]")
        if kind == "normality" and finite.size and not np.isin(
            finite, (0.0, 1.0)
        ).all():
            raise EthiopiaSpiContractError("Normality values are not raw flags")
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or admitted_hash != file_sha256(path)
    ):
        raise EthiopiaSpiContractError("NetCDF member changed during validation")
    return {
        "path": str(path),
        "sha256": admitted_hash,
        "kind": kind,
        "scale": scale,
        "variable": variable_name,
        "shape": list(variable.shape),
    }


def _read_netcdf_arrays(
    path: Path, variable_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from netCDF4 import Dataset

    with Dataset(path, "r") as handle:
        values = np.ma.asarray(handle.variables[variable_name][0, :, :])
        values = np.asarray(values.filled(np.nan), dtype="float64")
        values[~np.isfinite(values)] = np.nan
        latitudes = np.asarray(handle.variables["lat"][:], dtype="float64")
        longitudes = np.asarray(handle.variables["lon"][:], dtype="float64")
    return values, latitudes, longitudes


def _member_scientific_identity(
    name: str,
) -> tuple[str, tuple[float, float, float, float]]:
    marker = ".area-subset."
    if marker not in name or not name.endswith(".nc"):
        raise EthiopiaSpiContractError(f"Unexpected archive member: {name}")
    scientific, bbox_text = name[:-3].split(marker, 1)
    parts = bbox_text.split(".")
    values: list[float] = []
    index = 0
    while index < len(parts):
        token = parts[index]
        if (
            index + 1 < len(parts)
            and parts[index + 1].isdigit()
            and token.lstrip("-").isdigit()
        ):
            token += "." + parts[index + 1]
            index += 1
        try:
            values.append(float(token))
        except ValueError as exc:
            raise EthiopiaSpiContractError("Archive member bbox is malformed") from exc
        index += 1
    if len(values) != 4:
        raise EthiopiaSpiContractError("Archive member bbox is incomplete")
    return scientific, tuple(values)


def _load_selected_geometry(
    geometry_path: Path, area_ids: Sequence[str]
) -> Any:
    import geopandas as gpd

    geometry = gpd.read_file(geometry_path)
    if "admin_code" not in geometry:
        raise EthiopiaSpiContractError("Geometry is missing admin_code")
    geometry[KEY] = normalize_admin_codes(geometry["admin_code"]).to_numpy()
    selected = geometry.loc[geometry[KEY].isin(area_ids), [KEY, "geometry"]].copy()
    if len(selected) != len(area_ids) or selected[KEY].nunique() != len(area_ids):
        raise EthiopiaSpiContractError("Aggregation geometry does not reconcile")
    return selected.to_crs(4326)


def admit_and_aggregate_campaign(
    manifest_path: Path,
    *,
    geometry_path: Path,
    output_root: Path,
    completed_manifest_path: Path,
) -> dict[str, Any]:
    """Admit immutable ZIP members and build compact public and separate QA data."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_campaign_manifest(manifest)
    campaign_root = Path(manifest["campaign_root"])
    member_root = campaign_root / "members"
    if completed_manifest_path.exists():
        raise EthiopiaSpiContractError("Completed manifest already exists")
    public_path = output_root / "ethiopia_spi_monthly.csv"
    qa_path = output_root / "ethiopia_spi_qa.csv.gz"
    if public_path.exists() or qa_path.exists():
        raise EthiopiaSpiContractError("SPI release output already exists")
    member_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    geometry = _load_selected_geometry(
        geometry_path, manifest["universe"]["area_ids"]
    )

    returned_artifacts: list[dict[str, Any]] = []
    members_by_name: dict[str, Path] = {}
    for request in manifest["requests"]:
        zip_path = campaign_root / request["target_relative_path"]
        if not zip_path.is_file():
            raise EthiopiaSpiContractError(f"Downloaded ZIP is missing: {zip_path}")
        with zipfile.ZipFile(zip_path) as archive:
            observed = archive.namelist()
            if len(observed) != request["expected_member_count"]:
                raise EthiopiaSpiContractError("ZIP member count differs")
            expected_scientific = sorted(
                name.split(".area-subset.", 1)[0]
                for name in request["expected_members"]
            )
            identities = [_member_scientific_identity(name) for name in observed]
            observed_scientific = sorted(identity[0] for identity in identities)
            north, west, south, east = request["payload"]["area"]
            expected_bbox = np.asarray((north, east, south, west), dtype=float)
            if observed_scientific != expected_scientific or not all(
                np.allclose(identity[1], expected_bbox, rtol=0, atol=1e-10)
                for identity in identities
            ):
                raise EthiopiaSpiContractError("ZIP scientific identity differs")
            admitted_members = []
            for name in observed:
                if Path(name).name != name or name in members_by_name:
                    raise EthiopiaSpiContractError("ZIP member path is unsafe or duplicated")
                target = member_root / name
                info = archive.getinfo(name)
                if target.exists():
                    crc = 0
                    with target.open("rb") as admitted:
                        for chunk in iter(lambda: admitted.read(1024 * 1024), b""):
                            crc = zlib.crc32(chunk, crc)
                    if target.stat().st_size != info.file_size or (
                        crc & 0xFFFFFFFF
                    ) != info.CRC:
                        raise EthiopiaSpiContractError("Admitted member differs from ZIP")
                else:
                    with archive.open(name) as source, target.open("xb") as destination:
                        while chunk := source.read(1024 * 1024):
                            destination.write(chunk)
                members_by_name[name] = target
                admitted_members.append(
                    {
                        "name": name,
                        "sha256": file_sha256(target),
                        "size_bytes": target.stat().st_size,
                    }
                )
        returned_artifacts.append(
            {
                "request_id": request["request_id"],
                "zip_relative_path": str(zip_path.relative_to(campaign_root)),
                "zip_sha256": file_sha256(zip_path),
                "zip_size_bytes": zip_path.stat().st_size,
                "members": admitted_members,
            }
        )

    request_area = manifest["universe"]["request_area_nwse"]
    spi_pattern = re.compile(
        r"^SPI(?P<scale>1|3|6|12)_gamma_global_era5_moda_ref1991to2020_(?P<ym>\d{6})\."
    )
    quality_pattern = re.compile(
        r"^SPI(?P<scale>1|3|6|12)_spi(?P<kind>pzero|significance)_"
        r"gamma_global_era5_moda_ref1991to2020_(?P<month>\d{2})\."
    )
    public_scales: list[pd.DataFrame] = []
    qa_scales: list[pd.DataFrame] = []
    for scale in SPI_SCALES:
        p0_by_month: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, str]] = {}
        normality_by_month: dict[
            int, tuple[np.ndarray, np.ndarray, np.ndarray, str]
        ] = {}
        for name, path in members_by_name.items():
            match = quality_pattern.match(name)
            if not match or int(match.group("scale")) != scale:
                continue
            month = int(match.group("month"))
            kind = "p0" if match.group("kind") == "pzero" else "normality"
            validate_netcdf_member(
                path,
                scale=scale,
                kind=kind,
                expected_source_month=f"2020-{month:02d}",
                expected_request_area=request_area,
            )
            variable = "pzero" if kind == "p0" else "significance"
            arrays = (*_read_netcdf_arrays(path, variable), file_sha256(path))
            (p0_by_month if kind == "p0" else normality_by_month)[month] = arrays
        if set(p0_by_month) != set(range(1, 13)) or set(normality_by_month) != set(
            range(1, 13)
        ):
            raise EthiopiaSpiContractError("Quality calendar is incomplete")

        spi_members = []
        for name, path in members_by_name.items():
            match = spi_pattern.match(name)
            if match and int(match.group("scale")) == scale:
                spi_members.append((match.group("ym"), path))
        if len(spi_members) != EXPECTED_MONTHS:
            raise EthiopiaSpiContractError("SPI source month inventory is incomplete")
        public_parts: list[pd.DataFrame] = []
        qa_parts: list[pd.DataFrame] = []
        for year_month, path in sorted(spi_members):
            source_month = f"{year_month[:4]}-{year_month[4:]}"
            month = int(year_month[4:])
            validate_netcdf_member(
                path,
                scale=scale,
                kind="spi",
                expected_source_month=source_month,
                expected_request_area=request_area,
            )
            spi, latitudes, longitudes = _read_netcdf_arrays(path, f"SPI{scale}")
            p0, p0_lat, p0_lon, p0_sha = p0_by_month[month]
            normality, normality_lat, normality_lon, normality_sha = (
                normality_by_month[month]
            )
            if not (
                np.array_equal(latitudes, p0_lat)
                and np.array_equal(longitudes, p0_lon)
                and np.array_equal(latitudes, normality_lat)
                and np.array_equal(longitudes, normality_lon)
            ):
                raise EthiopiaSpiContractError("SPI and quality grids differ")
            public, qa = aggregate_spi_month(
                geometry,
                spi=spi,
                p0=p0,
                normality=normality,
                latitudes=latitudes,
                longitudes=longitudes,
                scale=scale,
                source_month=source_month,
            )
            qa["spi_source_sha256"] = file_sha256(path)
            qa["p0_source_sha256"] = p0_sha
            qa["normality_source_sha256"] = normality_sha
            public_parts.append(public)
            qa_parts.append(qa)
        public_scale = pd.concat(public_parts, ignore_index=True).sort_values(
            [KEY, DATE], kind="stable"
        )
        qa_scale = pd.concat(qa_parts, ignore_index=True).sort_values(
            [KEY, DATE, "spi_scale"], kind="stable"
        )
        if len(public_scale) != EXPECTED_ROWS or len(qa_scale) != EXPECTED_ROWS:
            raise EthiopiaSpiContractError("Scale release row count differs")
        public_scales.append(public_scale)
        qa_scales.append(qa_scale)

    public = public_scales[0]
    for scale_frame in public_scales[1:]:
        public = public.merge(scale_frame, on=[KEY, DATE], how="outer", validate="one_to_one")
    qa = pd.concat(qa_scales, ignore_index=True)
    if public.duplicated([KEY, DATE]).any() or qa.duplicated(
        [KEY, DATE, "spi_scale"]
    ).any():
        raise EthiopiaSpiContractError("Final SPI release keys are duplicated")
    if len(public) != EXPECTED_ROWS or len(qa) != EXPECTED_ROWS * len(SPI_SCALES):
        raise EthiopiaSpiContractError("Final SPI release dimensions differ")
    public.to_csv(public_path, index=False, date_format="%Y-%m-%d")
    qa.to_csv(qa_path, index=False, date_format="%Y-%m-%d", compression="gzip")
    completed = dict(manifest)
    completed["completion_status"] = "complete"
    completed["cloud_operation_performed"] = True
    completed["returned_artifacts"] = returned_artifacts
    completed["output_artifacts"] = {
        "public_csv": str(public_path),
        "public_sha256": file_sha256(public_path),
        "public_rows": len(public),
        "qa_csv_gz": str(qa_path),
        "qa_sha256": file_sha256(qa_path),
        "qa_rows": len(qa),
    }
    completed["manifest_sha256"] = canonical_json_sha256(
        campaign_identity_payload(completed)
    )
    completed_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    completed_manifest_path.write_text(
        json.dumps(completed, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return completed["output_artifacts"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Build the offline exact manifest")
    prepare.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    prepare.add_argument("--geometry", type=Path, required=True)
    prepare.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    prepare.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    validate = subparsers.add_parser("validate", help="Validate a prepared manifest")
    validate.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    download = subparsers.add_parser("download", help="Run the approved CDS requests")
    download.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    download.add_argument("--approved-manifest-sha256", required=True)
    download.add_argument(
        "--credential", type=Path, default=Path.home() / ".cdsapirc"
    )

    aggregate = subparsers.add_parser(
        "aggregate", help="Admit downloaded members and build SPI releases"
    )
    aggregate.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    aggregate.add_argument("--geometry", type=Path, required=True)
    aggregate.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    aggregate.add_argument("--completed-manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        universe = derive_ethiopia_spi_universe(
            args.cohort.resolve(), args.geometry.resolve()
        )
        manifest = build_campaign_manifest(
            universe, campaign_root=args.campaign_root.resolve()
        )
        write_campaign_manifest(manifest, args.manifest.resolve())
        print(f"manifest={args.manifest.resolve()}")
        print(f"manifest_sha256={manifest['manifest_sha256']}")
        print(f"request_count={manifest['expected_counts']['requests']}")
        print(f"expected_members={manifest['expected_counts']['returned_members']}")
        print(f"request_area_nwse={manifest['universe']['request_area_nwse']}")
    elif args.command == "validate":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        validate_campaign_manifest(manifest)
        print(f"manifest_sha256={manifest['manifest_sha256']}")
        print("validation_status=PASS")
    elif args.command == "download":
        inventory = download_campaign(
            args.manifest.resolve(),
            approved_manifest_sha256=args.approved_manifest_sha256,
            credential_path=args.credential.resolve(),
        )
        print(f"download_inventory={inventory}")
    else:
        outputs = admit_and_aggregate_campaign(
            args.manifest.resolve(),
            geometry_path=args.geometry.resolve(),
            output_root=args.output_root.resolve(),
            completed_manifest_path=args.completed_manifest.resolve(),
        )
        print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
