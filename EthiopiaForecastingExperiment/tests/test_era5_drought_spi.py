"""Focused offline checks for the Ethiopia ERA5-Drought SPI contract."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from era5_drought_spi import (  # noqa: E402
    EthiopiaSpiUniverse,
    _aligned_request_area,
    _spherical_cell_areas_m2,
    aggregate_spi_month,
    build_campaign_manifest,
    validate_campaign_manifest,
)


class EthiopiaSpiContractTests(unittest.TestCase):
    def test_manifest_has_exact_request_inventory(self) -> None:
        universe = EthiopiaSpiUniverse(
            area_ids=("1", "2"),
            cohort_path=Path("/tmp/cohort.csv"),
            cohort_sha256="cohort",
            cohort_key_sha256="keys",
            geometry_path=Path("/tmp/geometry.shp"),
            geometry_bundle=(),
            geometry_bundle_sha256="geometry",
            geometry_feature_count=2,
            geometry_crs="EPSG:4326",
            geometry_bounds_wsen=(33.0, 3.0, 48.0, 15.0),
            request_area_nwse=(15.0, 33.0, 3.0, 48.0),
        )
        manifest = build_campaign_manifest(universe, campaign_root=Path("/tmp/spi"))
        validate_campaign_manifest(manifest)
        self.assertEqual(manifest["expected_counts"]["requests"], 64)
        self.assertEqual(manifest["expected_counts"]["returned_members"], 816)
        self.assertFalse(manifest["submission_enabled_by_default"])

    def test_request_area_snaps_outward(self) -> None:
        self.assertEqual(
            _aligned_request_area((33.276, 3.751, 47.345, 14.534)),
            (14.75, 33.25, 3.75, 47.5),
        )

    def test_polygon_weighting_and_p0_gate(self) -> None:
        geometry = gpd.GeoDataFrame(
            {"FEWSNET_admin_code": ["1"]},
            geometry=[box(0.0, 0.0, 1.0, 1.0)],
            crs=4326,
        )
        latitudes = np.array([0.75, 0.25])
        longitudes = np.array([0.25, 0.75])
        spi = np.array([[1.0, 2.0], [3.0, 4.0]])
        p0 = np.zeros((2, 2))
        normality = np.ones((2, 2))
        public, qa = aggregate_spi_month(
            geometry,
            spi=spi,
            p0=p0,
            normality=normality,
            latitudes=latitudes,
            longitudes=longitudes,
            scale=3,
            source_month="2020-01",
        )
        expected = np.average(
            spi, weights=_spherical_cell_areas_m2(latitudes, longitudes)
        )
        self.assertAlmostEqual(public.loc[0, "SPI_3"], expected, places=12)
        self.assertEqual(qa.loc[0, "status"], "valid")

        p0[0, 0] = 0.9
        public, qa = aggregate_spi_month(
            geometry,
            spi=spi,
            p0=p0,
            normality=normality,
            latitudes=latitudes,
            longitudes=longitudes,
            scale=3,
            source_month="2020-01",
        )
        self.assertTrue(pd.isna(public.loc[0, "SPI_3"]))
        self.assertEqual(qa.loc[0, "status"], "coverage_failure")


if __name__ == "__main__":
    unittest.main()
