# P1 ownership mask diagnostic result (release ipcch-alphaearth-p1-mask-diagnostic-20261010)

Pre-run checks: HEAD 30649a2; ee_mask_diagnostic.py 4224a532…, ee_metadata.py d246a6c5…; released manifest body
766e531c… used unchanged; fresh `/home/swl007007/ipcch-alphaearth-runs/p1-mask-diagnostic-20261010`; `timeout 600`.
Calls 3 of 4 (init + 2 reference grids, no retry), 7.9 s wall. No value extraction, export, cloud write or fit.

Geometry: frozen admin-1722 owned piece (sha256 d5589c32…, local geodesic 223,546,435 m²). Owner x6bdj6bnb7dgiukef
(EPSG:32638), donor xt0jtdcoxb1sp5b7v (EPSG:32637). Mask = 64 bands unmasked AND |x| < 1e300; invalid/outside = false;
non-reference mask nearest-resampled (diagnostic only).

| class (m²) | reference = owner grid | reference = donor grid |
|---|---|---|
| owner_only | 132,002,722.71 (59.049%) | 131,978,724.33 (59.039%) |
| donor_only | 87,638,265.56 (39.204%) | 87,641,811.82 (39.205%) |
| both | 3,912,673.56 (1.750%) | 3,912,312.69 (1.750%) |
| neither | 0.00 | 4,612.82 (0.002%) |
| total | 223,553,661.84 (+0.0032% vs local area) | 223,537,461.64 (−0.0040%) |

Reconciliation: owner valid (owner_only+both) on its own grid = 135,915,396 m² = B1 owner valid area exactly; donor
valid on its own grid = 91,554,125 m² = B1 donor valid area exactly; cross-grid nearest resampling changes class
areas by ≤ 0.018%.

Conclusion (bounded): on this sampled piece the two tiles' valid sets are nearly disjoint (1.75% overlap); their union
covers ~100% while the frozen first-owner rule keeps 60.8% and discards 39.2% (≈ 87.6 km²) of otherwise valid
coverage. This proves diagnostic coverage loss on this piece only, not its extent over the cohort (1,867 admins span
>1 UTM zone; 3,121 have footprint overlaps). It meets implement.md's P1 trigger to resolve the ownership rule before
bulk extraction or model fitting; the executor has not changed the recipe.
