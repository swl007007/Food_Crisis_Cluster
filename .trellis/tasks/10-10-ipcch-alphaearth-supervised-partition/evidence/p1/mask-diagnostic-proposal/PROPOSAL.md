# Ownership mask diagnostic proposal (not run; release_id null)

Purpose: decide whether the frozen first-owner rule discards otherwise valid coverage, which the B1 area sums alone
cannot establish (owner valid 135.92 km², donor valid 91.55 km² over the same 223.55 km² geometry).

- Geometry: admin 1722 owned piece (frozen pilot proposal request `B:piece:1722:x6bdj6bnb7dgiukef:07`), geometry
  sha256 d5589c32d92aa9ce1e12eed87cde917286d5f6bf2a7fbcfd7249c5d3a5667907, local geodesic area 223,546,435 m².
- Assets: owner x6bdj6bnb7dgiukef (EPSG:32638, affine [10,0,8480,0,10,0]); donor xt0jtdcoxb1sp5b7v (EPSG:32637,
  affine [10,0,827680,0,10,0]). The other two overlapping tiles were fully masked in B1 and are not needed.
- Two requests: reference grid = owner native grid; reference grid = donor native grid. Each sums pixelArea for
  owner_only / donor_only / both / neither (4 bands) with mask = 64 bands unmasked AND |x| < 1e300, invalid or
  outside footprint = false; non-reference mask resampled by EE default nearest neighbour onto the reference native
  grid (diagnostic only); bestEffort false; native 10 m. Expected ~8.94e6 band-pixels per request.
- Proposed caps: maxPixels 12,000,000 band-pixels, deadline 120 s, ≤ 4 high-level calls incl. init and ≤ 1 retry,
  sequential, SDK retries 0, outer `timeout 600`. Output: fresh local directory (script refuses an existing one).
- Script `scripts/ee_mask_diagnostic.py` sha256 4224a532fe2c998618513c1c5dd19b7d04a815b78ec038d142e864a48adfdc78;
  manifest `mask_diagnostic_request.json` body sha256 766e531c5d682d9dbc85729c6b0febcd23aad56e1494e9345ee73d2f04b286dc.
- Command after release: `timeout 600 /opt/earthengine/bin/python scripts/ee_mask_diagnostic.py --request <released
  manifest> --out ~/ipcch-alphaearth-runs/p1-mask-diagnostic-<date>`.
- Reading: donor_only > 0 on the owner grid = valid coverage the first-owner rule discards; both > 0 = overlap of
  valid sets; the two grids bracket nearest-resampling effects. No ownership change follows automatically.

# Read-only GCS metadata (calls 15-17 of the 20-call metadata allowance; 17 used cumulatively)

Existing buckets in project manuel-remote-sensing (`gcs/buckets.json`): `-oct-delivery-debug` (US-CENTRAL1),
`-oct-delivery-requests` (US-CENTRAL1), `-rasters-645415198431` (US multi-region), `_cloudbuild` (US). The
delivery-debug bucket belongs to another delivery pipeline (prefixes assets/, deliveries/, runs/, staging/…), so it
is not a suitable destination. Candidate for a later, separately authorized EE table export:
`gs://manuel-remote-sensing-rasters-645415198431/ipcch-alphaearth/` (only existing top-level prefix `feature-008/`;
`ipcch-alphaearth/` unused). No object written, no write permission tested, no IAM/billing call. A table export
would need: write permission on that bucket for weilun.shi@cgiar.org (not checked) and explicit user authorization of
a GCS write, since the current pre-authorization covers queries/downloads only.
