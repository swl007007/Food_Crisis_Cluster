# Paper Reproducibility Release Manifest

## Release

- Tag: `v1.0-paper-reproducibility-20260624`
- Date: 2026-06-24
- Scope: GeoRF paper reproduction with GeoDT appendix artifact provenance

## Active Release Commands

Validate the lightweight package:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_paper_reproducibility_package.py
```

Verify the live result bundle:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_current_results_reproducibility.py
```

Run the full GeoRF regeneration path:

```batch
run_batches_2018_2020_partition_learning_visual_monthly.bat georf
spatial_weighted_consensus_clustering.bat georf
run_partition_k40_comparison_unified.bat georf --visual --month-ind
```

## Retained Paper Assets

- `paper_reproducibility_package/`
- `final_artifacts_in_paper_updated/`
- `GeoRFExperiment/`
- `GeoDTExperiment/` for appendix provenance
- `result_partition_k40_compare_GF_fs1/`
- `result_partition_k40_compare_GF_fs2/`
- `result_partition_k40_compare_GF_fs3/`
- `result_partition_k40_compare_DT_fs1/` for appendix provenance
- `result_partition_k40_compare_DT_fs2/` for appendix provenance
- `result_partition_k40_compare_DT_fs3/` for appendix provenance
- `main_ablation_exclude_updated_stage3_fixed_partitions/`
- `result_partition_k40_compare_GF_thresholded_fs1/`
- `result_partition_k40_compare_GF_thresholded_fs2/`
- `result_partition_k40_compare_GF_thresholded_fs3/`

## Archived Non-Paper Entry Points

Non-paper workflow entry points are preserved under:

```text
archived/release_20260624_nonpaper_pipelines/
```

The archive includes GeoXGB workflow entry points, fs0 launch guidance sources,
2026-2027 forward/scenario prediction launchers, legacy notebooks, regional
exploratory entry points, and legacy baseline/demo entry points. These files are
historical provenance and are not maintained as runnable workflows from the
archive path.

## Verification Commands

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest src/tests/test_release_cleanup_contract.py -q
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest src.tests.test_paper_reproducibility_package -v
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_paper_reproducibility_package.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_current_results_reproducibility.py
git diff --check
git status --short
```
