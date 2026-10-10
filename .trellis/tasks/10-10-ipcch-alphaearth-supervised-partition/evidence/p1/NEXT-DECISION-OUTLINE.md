# Outline for the next P1 decision (local only; nothing submitted)

## 1. Ownership (specification decision required; not executed)

Measured: first-owner discards 39.2% valid coverage on a sampled cross-zone piece. Options for review (none adopted):
(a) keep first-owner and accept biased/lowered coverage (D25 .80 would route more cross-zone admins to NaN);
(b) mask-aware deterministic priority: per pixel, the first asset in lexicographic order whose 64-band mask is valid
    — single counting kept, outcome independent; operationally each asset's piece = admin ∩ footprint_k ∩
    valid_k ∩ ¬(valid_1 ∪ … ∪ valid_{k−1}) evaluated on asset k's native grid with nearest-resampled earlier masks
    (the diagnostic above already exercises this machinery);
(c) quantify cohort extent first: the same 4-class mask query on a bounded sample of cross-zone admins.

## 2. Minimum batching/scaling check (proposal for a later release; caps illustrative)

Goal: measured throughput and merge correctness before any bulk plan. All geometries frozen; same A/S/Q recipe.
- S1 additivity: admin 3447 whole piece (already measured in A2) vs its 4 deterministic quadrants (4 calls,
  ≤ 7e7 band-px each); merged A/S/Q must equal A2 within float tolerance.
- S2 size scaling: three nested subpieces of the median piece (admin 3649, asset xwosouun6r3g2pjvh) at ~2e6, ~8e6,
  ~1.6e7 native px (3 calls; maxPixels 2.2e9 band-px; deadline 120 s; stop if a projection from the previous size
  exceeds 60% of the deadline).
- S3 batching: one ee.data.computeFeatures call on a FeatureCollection of the 4 S1 quadrants vs the 4 single calls
  (1 call); compare wall time and results.
- S4 concurrency (only if S1–S3 succeed): 2 then 4 simultaneous small requests (≤ 8 calls), watching rate/quota errors.
Envelope: ≤ 20 high-level calls total, outer timeout 1,800 s, SDK retries 0, one wrapper retry, stop on any
auth/quota/rate error or two distinct failures. Output: local files only.

## 3. Existing-bucket table-export alternative (requires explicit GCS-write authorization; not submitted)

Destination: gs://manuel-remote-sensing-rasters-645415198431/ipcch-alphaearth/<run-id>/ (existing US multi-region
bucket, prefix unused). Mechanism: Export.table.toCloudStorage of a FeatureCollection of frozen pieces mapped to the
same A/S/Q reduction (bestEffort false, native projection per piece), CSV with the frozen export columns; download
with read-only GCS calls; validate with extraction.validate_export / descriptors.descriptor_table. Advantages: batch
processing without the 120 s interactive deadline. Unknowns/prerequisites: write permission for
weilun.shi@cgiar.org on that bucket (not checked), explicit user authorization of a GCS write, batch EECU cost and
task-queue concurrency (not visible without billing access). A one-task pilot (e.g. the 16 pilot geometries) would
precede any bulk export.
