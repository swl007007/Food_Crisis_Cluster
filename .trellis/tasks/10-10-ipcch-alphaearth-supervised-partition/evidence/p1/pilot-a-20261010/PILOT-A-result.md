# P1-PILOT-A-20261010 result (two-request release)

Executed exactly as released: manifest copied unchanged (sha256 of file bfdc19de…; body sha f1bddf08…), released
script hashes verified before the run (ee_pilot.py 1c406793…, ee_metadata.py 86904862…, extraction.py 4e5832a6…),
fresh output `~/ipcch-alphaearth-runs/p1-pilot-a-20261010`, `timeout 600`. No export, no cloud write, no fit.

| call | purpose | outcome |
|---|---|---|
| 1 | initialize (project manuel-remote-sensing) | ok, 0.67 s |
| 2 | A: admin 3447 piece (asset x7kfwxt1zhh4tk7kr), maxPixels 2,000,000 | rejected in 0.22 s: "Image.reduceRegion: Too many pixels in the region. Found 66040811, but maxPixels allows only 2000000." |
| — | A: open-Red-Sea all-masked fixture | not attempted (run stopped) |

Calls used 2 of 6; no retry; no response moments; nothing normalized. Total wall 1.5 s.

## Finding: maxPixels counts band-pixels of the reduced stack

EE's count 66,040,811 = 511,944 × 129 bands (stack = pixelArea + 64 first moments + 64 second moments). The local
geodesic estimate for this geometry is 512,354 native 10 m pixels → 66,093,621 band-pixels (EE count is 0.08% lower,
consistent with fractional boundary pixels). Thus a native-pixel cap must be multiplied by 129: maxPixels 2e6 admits
only ~15,500 native pixels (~1.5 km²). The released fixture (37,650 native px → 4.86e6 band-px) would also exceed 2e6.
Area/count/moments/projection/all-masked behaviour were therefore NOT observed in this run.

Implication for later stages (band-pixels = 129 × native): median piece 3.16e7 px → 4.1e9; largest tile piece
2.69e8 px → 3.5e10; all pieces 4.67e11 px → 6.0e13 band-pixels. Whether such values are accepted interactively is unknown.

## Script classification fix (offline, after the run)

The wrapper labelled the rejection "fatal service error" because its keyword "too many" (meant for rate limits)
matched "Too many pixels". The outcome (stop, preserve, no retry) was correct. Fixed: rate limits stay fatal
("too many requests"/429); "too many pixels"/maxPixels is a distinct non-retryable size rejection that still stops a
released pilot. Tests added (test_extraction_offline.BudgetClassification). New hashes: ee_metadata.py d246a6c5…,
ee_pilot.py 24aa9369…; a rerun needs a release bound to these hashes.

## Proposed re-release of the same two requests (supervisor decision; caps not enlarged by the executor)

Same manifest requests/recipe; `max_pixels` expressed in band-pixels with a small margin over the larger stage-A need:
7.0e7 (= 1.06 × 66.09e6 for admin 3447; fixture needs 4.86e6). Deadline 120 s, ≤ 6 calls, sequential, SDK retries 0,
one wrapper retry (memory/timeout only), outer timeout 600 s.
