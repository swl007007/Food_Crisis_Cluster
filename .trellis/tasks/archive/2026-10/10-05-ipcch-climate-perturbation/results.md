# Results: climate feature perturbation vs p6-formal-20261004b

Run `climate-20261005`, code `60ebd62` (package `IPCCHClimateGeoXGBExperiment`),
pinned Windows runtime (Python 3.12.10, XGBoost 3.0.0). Run directory outside
Dropbox: `C:\Users\swl00\AppData\Local\Temp\ipcch-climate-runs\climate-20261005`
(2.3 GB, 5,761 files, inventory in `evidence/run-inventory.tsv`). Exploratory:
the 2023–2026 evaluation period had already been inspected.

## Execution

| Stage | Result | Wall time |
|---|---|---|
| tests | 293 passed | 12.6 min |
| preflight | passed; 15 input hashes incl. both climate files | 20 s |
| prepare | exit 0; 601 columns; ledger, F/S split, calendar and key files byte-identical to P6 | 1.9 min |
| learn-map | exit 0; 451 fits, 0 failed | 15.5 min |
| predict | exit 0; 3,199 requests, 641 fits, 0 failed | 53.7 min |
| report | exit 0 | <1 min |
| replay | passed, 70,527 checks, 0 failures | 10.6 min |

Comparison script self-check (P6 vs P6) gave zero deltas and reproduced P6's
published Geo−pooled and Geo−persistence values and intervals.

Feature coverage: rows with origin before 2015-01 are 717 (H1/H3/H6) and 978
(H12) of 42,695 per H. Monthly `evi_anom` NaN ≈ 2%, `spei03` 6–10%;
growing-season block NaN 2.2–3.4%.

## Maps re-learned on the new features

| H | New recipe / terminal regions | P6 recipe / terminal regions | Stage1 S F1 new / P6 |
|---|---|---|---|
| 1 | G1L2 / 7 | G1L2 / 9 | 0.8430 / 0.8510 |
| 3 | G1L2 / 6 | G3L2 / 7 | 0.8444 / 0.8429 |
| 6 | G4L2 / 2 | G4L2 / 6 | 0.8340 / 0.8368 |
| 12 | G1L2 / 5 | G2L2 / 9 | 0.8262 / 0.8301 |

Stage1 S F1 is in-sample development evidence, not a test result.

## Main period 2023–2025, identical keys (E_all)

New − P6 crisis F1 with the original paired country bootstrap (2000 draws, seed 42, all draws defined).

| H | keys | Pooled F1 new / P6 | Δ pooled [95% CI] | GeoXGB F1 new / P6 | Δ GeoXGB [95% CI] | Geo−pooled new / P6 |
|---|---:|---|---|---|---|---|
| 1 | 17,322 | 0.7773 / 0.7780 | −0.0007 [−0.0059, +0.0054] | 0.7773 / 0.7777 | −0.0004 [−0.0054, +0.0059] | +0.0000 / −0.0003 |
| 3 | 16,919 | 0.7822 / 0.7759 | +0.0063 [−0.0018, +0.0171] | 0.7819 / 0.7756 | +0.0063 [−0.0013, +0.0164] | −0.0003 / −0.0003 |
| 6 | 16,413 | 0.7727 / 0.7699 | +0.0029 [−0.0043, +0.0148] | 0.7728 / 0.7699 | +0.0029 [−0.0044, +0.0147] | +0.0001 / +0.0001 |
| 12 | 14,087 | 0.7533 / 0.7564 | −0.0031 [−0.0135, +0.0089] | 0.7540 / 0.7561 | −0.0021 [−0.0123, +0.0099] | +0.0006 / −0.0003 |

Other pooled metrics, new / P6:

| H | 4-class macro-F1 | projected q3 R² | crisis labels changed |
|---|---|---|---:|
| 1 | 0.4603 / 0.4598 | 0.2996 / 0.3053 | 687 |
| 3 | 0.4679 / 0.4633 | 0.3842 / 0.3672 | 934 |
| 6 | 0.4661 / 0.4811 | 0.2690 / 0.2541 | 921 |
| 12 | 0.4307 / 0.4646 | 0.2411 / 0.2525 | 750 |

Local-routed rows, new / P6: 47 / 588, 151 / 580, 1,129 / 509, 976 / 111.

### Versus persistence (E_persist, GeoXGB − persistence crisis F1)

| H | keys | New [95% CI] | P6 [95% CI] | 4-class macro-F1 new / P6 / persistence |
|---|---:|---|---|---|
| 1 | 16,002 | +0.0055 [−0.0093, +0.0257] | +0.0050 [−0.0098, +0.0242] | 0.4676 / 0.4665 / 0.5567 |
| 3 | 15,907 | +0.0111 [−0.0038, +0.0332] | +0.0074 [−0.0098, +0.0326] | 0.4726 / 0.4694 / 0.5591 |
| 6 | 14,960 | +0.0057 [−0.0089, +0.0219] | +0.0027 [−0.0151, +0.0157] | 0.4794 / 0.4961 / 0.5661 |
| 12 | 12,404 | +0.0047 [−0.0131, +0.0318] | +0.0075 [−0.0073, +0.0305] | 0.4467 / 0.4821 / 0.5293 |

## Supplementary 2026 (4 target months, point only)

| H | Pooled F1 new / P6 | Δ pooled | GeoXGB − persistence new / P6 |
|---|---|---|---|
| 1 | 0.7946 / 0.7766 | +0.0180 | +0.0667 / +0.0482 |
| 3 | 0.7608 / 0.7825 | −0.0217 | +0.0362 / +0.0571 |
| 6 | 0.7693 / 0.7854 | −0.0161 | +0.0431 / +0.0606 |
| 12 | 0.7492 / 0.7747 | −0.0255 | −0.0533 / −0.0281 |

## Reading

1. Replacing the monthly climate inputs and adding growing-season features does not change main-period crisis F1 beyond country-sampling noise. Pooled Δ is −0.0007 to +0.0063 and every interval includes zero. The largest gain, at H3, is not distinguishable from zero.
2. The spatial layer still adds nothing: Geo−pooled stays within ±0.0006 at every H. The re-learned maps are coarser, and local routing moves between horizons without changing the totals.
3. Secondary metrics move in mixed directions. q3 R² rises at H3/H6 and falls at H1/H12. Four-class macro-F1 falls at H6 (−0.015) and H12 (−0.034). The crisis-F1 point estimates hide 687–934 changed crisis labels per H.
4. Versus persistence, the crisis-F1 gap is similar to P6 (+0.005 to +0.011), with all intervals including zero. Four-class macro-F1 remains well below persistence.
5. In 2026 the new recipe is worse at H3–H12 (−0.016 to −0.026), point only on four months. The audit of the climate file flags 2025–2026 source-composition breaks (Terra NDVI/EVI, CPC temperature, PERSIANN SPI), so this may be data drift, not a weaker recipe; it is not tested here.

## Limits

- GeoXGB differences mix the feature change with re-learned maps and recipes; the pooled arm uses the same pipeline and isolates the feature recipe more cleanly, but its G recipe also changed at H3 and H12 (selection is part of the pipeline).
- Climate values are assumed available at the end of their observation month (inherited availability convention); release lags of the gridded products are not modelled.
- Only `*_ensmean` columns are used; source-count changes inside the ensembles are not controlled.
- One seed, one run; intervals are descriptive, conditional on saved predictions, without training or map-selection uncertainty and without multiplicity adjustment.
