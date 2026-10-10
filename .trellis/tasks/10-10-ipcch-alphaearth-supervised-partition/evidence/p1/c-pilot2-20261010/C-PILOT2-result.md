# C pilot rev2 result (release ipcch-alphaearth-p1-C-pilot2-20261010)

Pre-run checks: HEAD 4ca701d (clean); ee_pilot_c.py 4f43bbf8…, ee_metadata.py 6521c981…; body 8ef74cf2… (10 requests);
fresh `/home/swl007007/ipcch-alphaearth-runs/p1-c-pilot2-20261010`; `timeout 1200`; old failed run untouched.
Calls 12 of 14 (init + projection check + 10 reductions; no retry); 35.1 s wall. No export/write/fit; nothing normalized.

**Projection check:** EE coordinates of the 3 pinned points in the transport WKT differ from pyproj EPSG:6933 by at most
2.8e-9 m (≤ 1 mm required) → reductions proceeded.

| request | s | valid cells | valid area m2 | nulls | owned area by source rank |
|---|---|---|---|---|---|
| C01 1722 overlap piece (6 candidates) | 7.10 | 2,235,245 | 223,545,528.2 | 0 | r1 x6bdj… 135,888,820; r4 xjlze… 15,200.8; r6 xt0jt… 87,641,507.5 |
| C02 3447 whole | 4.05 | 512,389 | 51,234,473.3 | 0 | r1 all |
| C03 3447 windows (6) | 1.43–3.05 | 512,387 (sum) | 51,234,473.7 (sum) | 0 | r1 all |
| C04 Red Sea fixture | 1.18 | 0 | 0 (all 129 sums exactly 0) | 0 | none |
| C05 3649 4e6-cell window | 8.06 | 4,000,000 | 400,000,000 | 0 | r1 all |

## Assessment

- **C01 (all 6 candidates):** valid area 99.9996% of the frozen 223.55 km² geometry (first-footprint rule: 60.8%; native
  mask-diagnostic union ~100%). Ownership: rank 1 (x6bdj…, the old owner) 135.889 km² ≈ native owner valid 135.915;
  rank 6 (xt0jt…) 87.642 km² ≈ native donor_only 87.638–87.642; rank 4 (xjlze…) 0.015 km²; ranks 2, 3, 5 (xbcoe…,
  xdgzj…, xl600…) own nothing. Σ owned-by-rank = valid area (relative 2e-16). The lowest valid id wins as specified.
- **C02 vs A2 (single-zone control, estimand change):** coverage 0.999983 vs 0.999984 (both usable); 128 descriptors:
  max |Δmean| 1.4e-4 (median 4.8e-5), max |ΔSD| 9.9e-5 (median 2.9e-5; ≤ 0.22% relative).
- **C03 vs C02 (batching on one lattice):** A rel 7.7e-9, S max rel 2.2e-7, Q max rel 3.2e-8; valid cells 512,387 vs
  512,389 (2 cells: outer-boundary weight quantization); merged descriptors equal to ~1e-9. Not float-exact, as expected.
- **C04:** completed reduction over fully masked area returns exact zeros (not nulls) and an empty rank histogram on the
  analysis grid → genuine no coverage.
- **C05 / timing:** 4.0e6 single-source cells in 8.06 s (2.0e-6 s/cell); C01 2.24e6 cells with 6 candidates in 7.10 s
  (~3.2e-6 s/cell); small windows 1.4–3.1 s (overhead ~1 s).
