import argparse
import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "compare_partitioned_vs_pooled_rf_k40_nc4.py"
spec = importlib.util.spec_from_file_location("compare_partitioned_vs_pooled_rf_k40_nc4", SCRIPT_PATH)
stage3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage3)


def test_file_sha256_returns_stable_digest(tmp_path: Path):
    sample = tmp_path / "sample.csv"
    sample.write_text("a,b\n1,2\n", encoding="utf-8")

    digest = stage3.file_sha256(sample)

    assert len(digest) == 64
    assert digest == stage3.file_sha256(sample)


def test_partition_map_provenance_records_month_specific_maps(tmp_path: Path):
    general = tmp_path / "general.csv"
    m2 = tmp_path / "m2.csv"
    m6 = tmp_path / "m6.csv"
    m10 = tmp_path / "m10.csv"
    for path in (general, m2, m6, m10):
        path.write_text("FEWSNET_admin_code,cluster_id\n1,2\n", encoding="utf-8")

    args = argparse.Namespace(
        month_ind=True,
        partition_map=str(general),
        partition_map_m2=str(m2),
        partition_map_m6=str(m6),
        partition_map_m10=str(m10),
    )

    provenance = stage3.partition_map_provenance(args)

    assert provenance["month_ind_enabled"] is True
    assert provenance["partition_map_path"] == str(general)
    assert provenance["partition_map_m2_path"] == str(m2)
    assert provenance["partition_map_m6_path"] == str(m6)
    assert provenance["partition_map_m10_path"] == str(m10)
    assert len(provenance["partition_map_hashes"]["general"]) == 64
    assert len(provenance["partition_map_hashes"]["m2"]) == 64
    assert len(provenance["partition_map_hashes"]["m6"]) == 64
    assert len(provenance["partition_map_hashes"]["m10"]) == 64


def test_runtime_provenance_records_python_and_smote_fields():
    provenance = stage3.runtime_provenance()

    assert "python_executable" in provenance
    assert "python_version" in provenance
    assert "smote_available" in provenance
    assert "imblearn_version" in provenance


def test_build_run_manifest_merges_runtime_and_partition_provenance(tmp_path: Path):
    general = tmp_path / "general.csv"
    m2 = tmp_path / "m2.csv"
    m6 = tmp_path / "m6.csv"
    m10 = tmp_path / "m10.csv"
    for path in (general, m2, m6, m10):
        path.write_text("FEWSNET_admin_code,cluster_id\n1,2\n", encoding="utf-8")

    args = argparse.Namespace(
        data="clean.csv",
        partition_map=str(general),
        partition_map_m2=str(m2),
        partition_map_m6=str(m6),
        partition_map_m10=str(m10),
        start_month="2021-01",
        end_month="2024-12",
        train_window=36,
        forecasting_scope=1,
        lower_model="rf",
        visual=False,
        month_ind=True,
        enable_validation_threshold=True,
        threshold_validation_months=6,
        threshold_lower_bound=0.05,
        threshold_upper_bound=0.95,
    )

    manifest = stage3.build_run_manifest(
        args=args,
        active_lag=4,
        test_months=[1, 2, 3],
        metrics_month_count=12,
        predictions_count=62189,
        n_polygons=5713,
        n_partitions=17,
        model_label="RF",
    )

    assert manifest["month_ind_enabled"] is True
    assert manifest["partition_map_m2_path"] == str(m2)
    assert len(manifest["partition_map_hashes"]["m10"]) == 64
    assert manifest["smote_available"] in {True, False}
    assert manifest["validation_threshold_enabled"] is True
