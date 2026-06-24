# FEWS NET GeoRF Paper Reproducibility Package

This package supports fast audit and replication of the current manuscript-facing
GeoRF no-leak workflow. It bundles frozen Stage 2 consensus maps, Stage 3 result
summaries, final paper artifacts, and lightweight ablation provenance.

## What This Package Is

- Paper scope: 22 FEWS NET monitored countries across Africa, the Middle East,
  Asia, and Latin America.
- Main paper model: GeoRF with fixed partitions learned from 2018-2020 and
  evaluated on 2021-2024.
- Forecast horizons: 4, 8, and 12 months (`fs1`, `fs2`, `fs3`).
- Evaluation months: February, June, and October releases in 2021-2024.
- Fast path: inspect packaged artifacts or rerun Stage 3 using the packaged
  Stage 2 maps, without rerunning Stage 1 partition learning.

## What This Package Is Not

- It does not include raw source data.
- It does not include the full 4.7 GB ablation output tree.
- It does not include experimental GeoXGB, fs0 lag-1, or 2026-2027 forward
  prediction/scenario workflows.
- It does not move or archive experimental scripts; release-version code
  migration is a separate future task.

## Quick Validation

Run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_paper_reproducibility_package.py
```

Expected result: all files listed in `MANIFEST.csv` and `SHA256SUMS.txt` exist
and match their SHA-256 checksums.

## Full Repo Verification

Run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_current_results_reproducibility.py
```

This checks the live Stage 2/Stage 3/final-artifact bundle outside this copied
package.
