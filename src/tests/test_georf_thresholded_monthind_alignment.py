import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SCOPES = ("fs1", "fs2", "fs3")
MONTH_MAPS = {
    2: "cluster_mapping_k40_nc13_m2_refined_contig3.csv",
    6: "cluster_mapping_k40_nc11_m6_refined_contig3.csv",
    10: "cluster_mapping_k40_nc16_m10_refined_contig3.csv",
}


def _load_map(scope: str, month: int) -> pd.Series:
    path = REPO_ROOT / f"result_partition_k40_compare_GF_{scope}" / "refined" / MONTH_MAPS[month]
    mapping = pd.read_csv(path)
    return mapping.set_index("FEWSNET_admin_code")["cluster_id"].astype(int)


def _prediction_admin_month_rows(scope: str, provider: str) -> pd.DataFrame:
    path = REPO_ROOT / provider / "predictions_monthly.csv"
    df = pd.read_csv(path, usecols=["FEWSNET_admin_code", "month_start", "partition_id"])
    df["month_start"] = pd.to_datetime(df["month_start"])
    return df.drop_duplicates(["FEWSNET_admin_code", "month_start"]).copy()


def test_thresholded_georf_predictions_use_month_specific_partitions():
    for scope in SCOPES:
        predictions = _prediction_admin_month_rows(scope, f"result_partition_k40_compare_GF_thresholded_{scope}")
        for month, _filename in MONTH_MAPS.items():
            month_rows = predictions[predictions["month_start"].dt.month == month].copy()
            expected = _load_map(scope, month)
            month_rows["expected_partition_id"] = month_rows["FEWSNET_admin_code"].map(expected)
            covered = month_rows["expected_partition_id"].notna()

            assert covered.any(), f"{scope} month {month} has no covered polygons"
            mismatches = (
                month_rows.loc[covered, "partition_id"].astype(int)
                != month_rows.loc[covered, "expected_partition_id"].astype(int)
            )
            assert not mismatches.any(), f"{scope} month {month} has {int(mismatches.sum())} partition mismatches"


def test_thresholded_georf_manifest_records_month_ind_and_runtime_provenance():
    for scope in SCOPES:
        manifest_path = REPO_ROOT / f"result_partition_k40_compare_GF_thresholded_{scope}" / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert manifest["month_ind_enabled"] is True
        assert manifest["partition_map_m2_path"].endswith("cluster_mapping_k40_nc13_m2_refined_contig3.csv")
        assert manifest["partition_map_m6_path"].endswith("cluster_mapping_k40_nc11_m6_refined_contig3.csv")
        assert manifest["partition_map_m10_path"].endswith("cluster_mapping_k40_nc16_m10_refined_contig3.csv")
        assert len(manifest["partition_map_hashes"]["m2"]) == 64
        assert "python_executable" in manifest
        assert "smote_available" in manifest
