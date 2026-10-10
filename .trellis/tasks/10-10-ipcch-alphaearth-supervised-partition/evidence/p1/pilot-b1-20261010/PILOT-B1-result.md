# P1-PILOT-B1-20261010 result (five-request manual subset)

Checks before execution: ee_pilot.py 24aa9369…, ee_metadata.py d246a6c5…, extraction.py 4e5832a6… (receipt
source_hashes), manifest body 08b55eed… copied unchanged into fresh `~/ipcch-alphaearth-runs/p1-pilot-b1-20261010`;
`timeout 900`; A1/A2 untouched. Calls 6 of 8 (init + 5, no retry); cumulative pilot calls A1 2 + A2 3 + B1 6 = 11.
33.6 s wall. No export, cloud write, fit or normalization. Raw responses in `responses/`.

| request | asset (CRS) | s | valid px | valid area m2 | share of request geometry | nulls |
|---|---|---|---|---|---|---|
| Cabo Verde 101208 piece | xg60a6tyje250ifyj (32627) | 11.62 | 1,490,025 | 148,851,386 | 0.999993 | 0 |
| Somalia 1722 owned piece | x6bdj6bnb7dgiukef (32638) owner | 10.01 | 1,362,323 | 135,915,396 | 0.607996 | 0 |
| same geometry | xdgzjg7396r55w0vs (32737) non-owner | 2.48 | 0 | 0 (all sums 0) | 0 | 0 |
| same geometry | xl600p8zytcdfq5u2 (32738) non-owner | 2.23 | 0 | 0 (all sums 0) | 0 | 0 |
| same geometry | xt0jtdcoxb1sp5b7v (32637) non-owner | 6.24 | 917,393 | 91,554,125 | 0.409553 | 0 |

All 129 sums finite in every response; no negative variance zeroed; Σ_j E[e_j²] = 1.0002 for every valid response
(unit-norm embeddings). Cabo Verde coverage of the full frozen polygon 0.999993 (usable); its 64-mean norm 0.908.

## First-owner ownership: possible coverage loss, NOT yet established (pending mask diagnostic)

Over the identical 223.55 km² geometry the lexicographic owner (EPSG:32638, zone 38N) is valid on 60.8% (135.92 km²)
and the EPSG:32637 (zone 37N) tile on 41.0% (91.55 km²); the southern-hemisphere tiles EPSG:32737/32738 (zones
37S/38S) are fully masked there. These sums alone do not establish extra coverage: the 37N tile's valid pixels could
lie partly or wholly inside the owner's valid set. They only bound the owner-masked/donor-valid area between 0 and
min(91.55, 223.55 − 135.92) = 87.6 km² (39% of the region). Whether the first-owner rule discards otherwise valid
coverage requires the owner-only / donor-only / both / neither common-mask areas over the same geometry (proposed
two-request mask-only diagnostic). No ownership change is made or proposed as adopted before that evidence.
Scale context: 1,867 admins span >1 UTM zone and 3,121 have footprint overlaps (local plan).

## Timing / resource implications (5 cold samples; still not a full envelope)

Approximate model: ~2.3 s per request for a ~2.2e6-px geometry with no valid pixels, plus 4–6e-6 s per valid pixel
(11.62 s at 1.49e6 valid; 10.01 s at 1.36e6; 6.24 s at 0.92e6). Projections at 6e-6 s/valid px: median piece
(3.16e7 px) ~190 s (> 120 s deadline → must be split); total 4.67e11 px ~2.8e6 s ≈ 32 days sequential; ~2 days at
16 concurrent requests if quotas permit (unknown). Keeping each request under ~60 s implies pieces ≲ 1e7 px, i.e.
roughly 5e4 requests for the full extraction (each ownership piece split into bounded subpieces whose moments merge
exactly). EECU cost, interactive concurrency/rate quotas and project tier remain unknown.
