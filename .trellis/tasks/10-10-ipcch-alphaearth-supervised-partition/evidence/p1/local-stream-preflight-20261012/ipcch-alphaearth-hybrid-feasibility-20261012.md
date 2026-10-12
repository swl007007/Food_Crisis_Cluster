# AlphaEarth local plus EE feasibility — 2026-10-12

This is a capacity and interface study, not a production switch or a local throughput result.

The live EE invocation remains unchanged. Snapshot: 2026-10-12T05:25:52.479005+00:00. Pending expected cells: 303371794935.28015.

## Tail inventory

Selection: whole admins with zero completed pieces, no reservation/request/ledger attempts in the snapshot, all source parents mapped, screened compressed input <=40 decimal GB per admin. Original frozen plan tail order. Source screening remains provisional.

| Admins | Pieces | Expected cells | Distinct compressed input GB | Largest four independent input sets GB | Pending cells removed if accepted |
|---:|---:|---:|---:|---:|---:|
| 8 | 196 | 2.032e+09 | 138.34 | 115.93 | 0.670% |
| 32 | 546 | 5.225e+09 | 341.46 | 140.88 | 1.722% |
| 128 | 1504 | 1.321e+10 | 776.65 | 153.16 | 4.354% |

Distinct bytes are a lower bound on cumulative downloads with perfect reuse, not a simultaneous storage requirement. Eviction under 160 GB may cause redownloads. Largest-four values concern compressed input only. Proposed storage admission: 160 GB shared input/cache plus 40 GB total scratch/in-flight/output reservation; no physical full float mosaic. Measure actual RSS and transient disk before raising local worker count.

## Potential gain

For illustration only, EE effective rate 1.568e6 cells/s and sustained accepted local rate L give time ratio 1.568e6/(1.568e6+L), assuming disjoint workload, enough work for both, no shared resource slowdown and negligible setup. L=0.5e6 gives 24.2% less time; L=1e6 gives 38.9% less. These are scenarios, not observations. Local rate must include download, decode, reprojection, weighted accumulation, validation and repeated downloads.

The 128-admin sample is only 4.354% of pending expected cells. At the illustrative EE rate it removes about 2.34 request-parallel wall hours of future EE work if completed and accepted before EE reaches it. Significant gains require a sustained succession of batches. At an unmeasured 100 MB/s, its 776.65 GB distinct inputs alone take at least 2.16 hours, excluding compute and redownloads.

## Minimum safe integration

1. Run a bounded local parity and throughput check using existing EE references; preserve D30 grid, complete-vector first-valid source, nearest sampling, decoding, pixel area and fractional polygon weights. ArcPy default mosaic and zoning do not establish equivalence. The existing 64-bit rasterio/GDAL environment is the preferred candidate.
2. Preserve local outputs and COG object generations/hashes in separate provenance. Reuse the original frozen piece boundaries for A/S/Q where possible, then merge per admin. Do not fabricate EE-shaped raw responses or diagnostic count/rank data.
3. At an invocation boundary, bind a disjoint tail allocation and adapt pending/completion checks and ingestion explicitly. The current runner snapshots todo once (ee_pilot_c.py:226), does not recheck outputs in workers (:236-242), and writes results with replacement (:275); injecting files into the live output directory will not prevent duplicate work.
4. Only validated accepted local results can satisfy final completeness. Failed/unstarted local batches return to EE through a reviewed boundary update. Exact-once original piece coverage and separate engine provenance must both survive ingestion.

Evidence: /tmp/aef_hybrid_tail_options.json; /tmp/aef_gcs_2017_object_sizes.json; /tmp/aef_admin_tile_sets.json. No TIFF downloaded or new cloud work started for this study. Local numerical parity and end-to-end throughput remain unmeasured.
