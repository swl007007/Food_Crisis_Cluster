# Whole-admin direct reduceRegion pilot: offline proposal (NOT run) — 2026-10-11

Context: on the user's emergency direction the 10 m window extraction was stopped (invocation 4 killed, exit 143 at
12:45:56Z). Preserved state: 4 starts, 8,310 reservations, 8,280 saved window outputs (14.6% of cells, 774 admins
complete); old resume authority revoked. The D30 estimand is unchanged:
- year 2017;
- EPSG:6933 10 m grid, nearest sampling;
- first valid whole 64-band vector by ascending ID, mask before the descending mosaic;
- valid area plus area-weighted S/Q;
- the whole frozen polygon area as denominator and the 80% rule.

## Pilot (manifest `whole_admin_pilot_manifest.json`, file e31f388e…, body 65efd14b…, release_id null)

| role | admin | cells (Σ geodesic windows) | whole-admin lattice box | windows in plan / done | candidates | maxPixels (band-px) |
|---|---|---|---|---|---|---|
| REF completed small | 658 | 1.024e7 | 2.732e7 | 2 / 2 | 2 | 3.88e9 |
| MED current median (fresh) | 4711 | 3.152e7 | 5.430e7 | 5 / 0 | 4 | 7.70e9 |
| LARGE q75 (fresh) | 1050 | 7.652e7 | 1.520e8 | 9 / 0 | 3 (MultiPolygon) | 2.16e10 |

- **Selection.** Deterministic, from geometry and plan only. REF is the fully extracted admin with 2–4 windows nearest
  the q25 cells. MED and LARGE are the not-yet-extracted admins nearest the median and q75.
- **Each request:**
  - the complete frozen admin geometry (no windows);
  - the complete ascending candidate union, i.e. every footprint intersecting the admin, from the frozen plan;
  - the same composite and grid;
  - **one aggregation**: `stack.reduceRegion(Reducer.sum())` over [valid_area, S×64, Q×64], with bestEffort false.
- **Diagnostics dropped, not redefined.** The unweighted valid-cell count and the area-by-source-rank histogram are left
  out. No descriptor uses them. Combining them into the same reduceRegion would change output naming and is not needed
  for the estimand.
- **Server aggregations:** 3 in total, 1 per admin. The window graph used 3 per window: 57,801 × 3 = 173,403 for the full
  plan, and 2 + 5 + 9 = 16 windows × 3 = 48 for these three admins.
- **Limits:**
  - sequential (concurrency 1);
  - budget_retries 0, i.e. **no automatic retry**, read timeouts included;
  - SDK retries 0;
  - deadline 120 s (unchanged);
  - calls ≤ 5 (init + projection check + 3);
  - outer timeout 600 s.
- **Order:** REF → MED → LARGE. The first failure stops the remaining requests, and every response or error is kept.
- **Writes:** local JSON responses only. No export or cloud write.

**Analysis (offline):**
- **REF:** whole-admin A, S and Q against the Σ of its two window outputs. Window batching on one lattice has a pilot2
  precedent of ≈ 1e-7 relative difference, so not float-exact; differences are reported unnormalized. Also descriptor and
  coverage agreement.
- **All three admins:** finiteness, coverage against the frozen area, and the seconds measured for the new one-aggregation
  graph.
- **Timing caveat:** the old 4.0 s + 1.65e-6 s/cell model came from the three-aggregation window graph. It is a
  hypothesis only and is not used to reject whole-admin execution. A read timeout on MED or LARGE is itself the finding.

## Code diff (tiny, opt-in; default behaviour unchanged)

- `scripts/ee_pilot_c.py` `reduction_obj`: `if r.get("aggregations") == "sums": return ee.Dictionary({"sums": sums})`.
- `scripts/ee_metadata.py` `Budget(..., attempts=2)`: the loop runs `attempts` times; 1 means no automatic retry.
- The pilot path of `main()` passes `attempts = 1 + limits["budget_retries"]`.
- The production path, the FATAL / NONRETRYABLE / SHARED_STOP rules and all released manifests are unchanged.
- **Tests:** the sums-only request issues exactly one reduceRegion with the same descending mosaic and grid; attempts=1
  makes 1 call while the default still allows one retry. Focused 50 OK.

## What the 57,801-window design bought, and whether its cost was justified

**What it bought:**
- every request bounded at ≤ 1.6e7 cells (lattice box), so each fit well inside the 120 s deadline (max observed 81.7 s);
- a bounded per-request pixel guard;
- fine-grained resume: a failure costs ≤ 1 window, and 8,280 outputs survived four invocations and two stops;
- deterministic cost accounting.

**What it cost:**
- 57,801 requests instead of 6,227, with a per-request overhead the window model put at about 4 s;
- 3 aggregations per request, where 2 are diagnostics carried over from the C pilot into production;
- a large orchestration surface (manifests, ledger, guards, four release cycles, two guard repairs).

**Assessment:**
- The window cap only matters for admins too large for one interactive request; 50% of admins have ≤ 3.15e7 cells.
- The diagnostic aggregations were never needed for the estimand.
- I applied windows uniformly and carried the diagnostics into production without asking whether simpler whole-admin
  requests would do. That is exactly the drift the user's standing instruction targets.
- Whether whole-admin requests are faster per pixel is unknown until measured.

## After the pilot (each step needs its own decision)

- **If MED and LARGE succeed:** propose a whole-admin plan for the remaining admins, sized by their measured behaviour.
  The open decision is whether the 774 completed admins keep their window outputs (same estimand, about 1e-7 batching
  difference) or are redone whole.
- **If LARGE times out:** discuss. Options are hybrid sizing (whole admin below a measured size, windows above) or
  another route; any cloud-write export would need new user authorization.
