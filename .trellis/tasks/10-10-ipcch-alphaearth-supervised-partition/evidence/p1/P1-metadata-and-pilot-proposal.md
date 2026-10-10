# P1 metadata results and staged pilot proposal (2026-10-10)

Release: P1-METADATA-20261010 (receipt retained). Cloud scope: user pre-authorization (EE query/extraction/download;
at most GCS query/download; no IAM/billing/API/project/bucket changes, no GCS/Drive writes). This round made **no
raster reductions, exports or writes, and no model fit**.

## Identity and configuration (read locally, no secrets printed)

- EE credentials `~/.config/earthengine/credentials`: refresh-token OAuth, scopes earthengine + cloud-platform; no project field.
- gcloud: account weilun.shi@cgiar.org, project `manuel-remote-sensing`; ADC quota_project_id `manuel-remote-sensing`.
- Used existing project `manuel-remote-sensing` (initialized successfully). Nothing created or modified.

## Request accounting (`request_log.jsonl`, 14 of 20 high-level calls)

| n | purpose | result |
|---|---|---|
| 1 | initialize | ok |
| 2 | collection asset metadata | ok |
| 3 | 2017 counts: global 10,562; study boxes 1,683 | ok |
| 4–5 | whole-collection version histograms (server aggregate) | "User memory limit exceeded" twice → technical stop as designed (not auth/quota) |
| 6–7 | initialize; listImages with `endTime=2018-01-01` | ok but 0 images: listAssets filters `endTime < bound` (exclusive) and the annual images end exactly 2018-01-01T00:00Z (local SDK inspection) |
| 8–14 | initialize; paged listImages (FULL, 300/page) with `endTime=2018-01-01T00:00:01Z` | ok, 1,683 images, 5.3 MB/page |

SDK: `setMaxRetries(0)` before Initialize; `setDeadline(120000)` after Initialize (EE 1.7.36 asserts a session in
setDeadline before init — a local pre-network constraint, not a service failure); state confirmed max_retries 0,
deadline 120000. Budget counts high-level calls; Initialize may issue >1 HTTP request (token refresh/discovery).
Version/projection distributions computed locally from the listing. Full listing (29.5 MB) stays outside Dropbox
(`~/ipcch-alphaearth-runs/p1-metadata-20261010/ee/list_images.json`, sha256 in `list_images.sha256`).

## AlphaEarth 2017 facts (study-area listing, 1,683 assets)

- time: all 2017-01-01T00:00Z → 2018-01-01T00:00Z; MODEL_VERSION 2.1, DATASET_VERSION 1.1, PROCESSING_SOFTWARE_VERSION 1.02 (all assets; matches the expected catalog versions; no mixing).
- bands A00–A63 on every asset, listed dataType DOUBLE, one grid per asset; native UTM, 39 zones (EPSG:326xx/327xx), 10 m, tile 16,384 × 16,384 px (163.84 km). Listed affine scaleY is +10 (to confirm in pilot projections).
- footprints: Polygon per asset; same-zone tiles do not overlap; cross-zone tiles overlap (UTM-zone boundaries), summed footprint area / union = 1.285. No tiles over deep ocean (footprints exist only for land/shallow-water areas).

## Local ownership plan (frozen recipe; `admin_ownership_summary.csv.gz`, `piece_plan_estimate.csv.gz`)

- 6,227 admins → 17,276 ownership pieces (lexicographic asset order, footprint minus earlier footprints). Every admin
  intersects at least one footprint; footprint-level coverage is 1.0 for all admins (actual valid-pixel coverage
  unknown until reduction). 1,867 admins span >1 UTM zone; 3,121 admins have footprint overlap that ownership prevents
  from being double counted.
- Pixels (geodesic piece area / 100 m²): total 4.67×10¹¹; per piece median 1.0×10⁷, p90 7.5×10⁷, p99 2.2×10⁸,
  max 2.69×10⁸ (tile-bounded). 1,104 pieces exceed 10⁸. The 6,227 admins overlap each other heavily (summed admin
  area 46.7M km²), so admin-wise reduction re-reads pixels; this drives cost.
- 67 near-degenerate intersection slivers (65 admins) remain in the inventory. A P0 helper boundary defect was found:
  `geodesic_area` raised on floating-point negative geodesic areas (−1e-7…−2e-6 m²) of these slivers; fixed by
  treating |negative| ≤ 1e-3 m² as numerical zero (larger negatives still stop) + regression test. No dropping
  threshold: every piece stays planned; a zero contribution follows the frozen mask/geometry contract.

## Proposed staged pilot (`pilot_request.json`, sha256 3ce8ed9e…86e3, release_id null)

Same D16/D25/D26 recipe: per request ONE frozen geometry on its OWN asset's native projection; stack [pixelArea,
area·A_j, area·A_j²] masked by (all 64 bands unmasked AND |x| < 1e300); `reduceRegion(Reducer.sum(), crs=asset
projection, bestEffort=False, maxPixels=3e8, tileScale=4)` + unweighted valid-pixel count + EE geodesic area of the
geometry; `getInfo()` → local JSON (direct EE computation; no export, no cloud object). Script `scripts/ee_pilot.py`.

| Stage | Requests (geometry-only, deterministic rules in `pilot_selection.json`) | px |
|---|---|---|
| A protocol | small single-piece admin 3447 (Yemen, 51 km²); all-masked fixture: 2×2 km open Red Sea square in asset xzki2wh0491tllh8g | 5.1e5; 3.8e4 |
| B representative | median admin 3649 (Sudan, 3,156 km²) + its two halves (batching invariance); Cabo Verde 101208 (island/mask edge); cross-zone Somalia 1722 (3 pieces, 3 zones) + its 223.55 km² owned piece re-measured in the 3 non-owner tiles (first-owner loss); multipolygon South Africa 2 (3 pieces) | 1.5e6–3.2e7 each |
| C1 stress (contingent) | centred ~1/16-tile subpiece of the largest ownership piece (admin 20, asset xuasuttrjby0u6nah) | 1.7e7 |
| C2 (not automatic) | the full 2.69e8-px tile piece — only with C1 success, projection ≤ 50% of deadline, `--allow-c2` and a separate release | 2.7e8 |

Gates: B runs only if every B request projects ≤ 25% of the 300 s deadline, projection = fixed overhead (all-masked
request time) + slowest observed marginal seconds/pixel; C1 only if all B succeeded. Limits: ≤ 22 high-level calls
(17 planned incl. init, C2 excluded), 1 concurrent, SDK retries 0, one Budget retry, deadline 300 s per call,
outer process timeout 3,600 s. Stop: any auth/permission/quota/rate error; two distinct requests failing twice
(memory/timeout); cap reached. Command (after release_id is set by the supervisor):

    /opt/earthengine/bin/python scripts/ee_pilot.py --request <released pilot_request.json> --out ~/ipcch-alphaearth-runs/p1-pilot-<date>

Local checks after the pilot (numerical env): merge A/S/Q per admin with `aemoments` (no SD averaging), coverage vs
.80 with exact fractions, batching invariance (median admin whole vs two halves: merged moments equal within float
tolerance), first-owner loss (owner vs non-owner valid area over the same region), all-masked response form (genuine
zero vs null sums — normalized to A=S=Q=0 only if proven), EE vs local geodesic area consistency, projection affine
(scaleY sign), variance-cancellation counts, per-request seconds vs pixels.

## Unknowns stated honestly

- Cost/quota: project tier (commercial vs noncommercial), EECU consumption per request, interactive concurrency and
  rate limits are not visible without billing/IAM access (not authorized); the pilot measures only wall time.
- Full-extraction envelope: 17,276 pieces, 4.67×10¹¹ pixels. With direct queries, time ≈ Σ(overhead + rate·px);
  e.g. overhead 2 s alone is ~9.6 h sequential; the per-pixel rate is unknown until the pilot. A full-run proposal
  (batching via computeFeatures of several small pieces per call, concurrency, splitting of >10⁸-px pieces into
  ownership-preserving sub-pieces) will follow the pilot with measured numbers.
- Whether cross-zone tiles carry valid data across their whole footprint (first-owner rule suitability) — measured in B.
- All-masked null-sum behaviour — measured in A.
