# D30 whole-admin batch extraction: offline design proposal (NOT implemented, NOT submitted) — 2026-10-11

User decisions:
- keep exhaustive full-10 m statistics;
- design whole-admin batch extraction;
- "可以考虑较大行政区拆分，较小行政区批处理" (split larger admins, batch smaller ones);
- cloud writes need a later concrete authorization.

Constraints: the old 57,801-window bulk stays stopped; no blanket return to windows.

## 1. Four separate layers

| layer | proposal | changes science? |
|---|---|---|
| Scientific invariant | D30 unchanged (see below) | no |
| Per-item split size | an admin is one item if its whole-admin lattice box ≤ 2.56e8 cells; otherwise it is split by the existing `lattice_windows(geom, lattice, max_cells=2.56e8)` (side 16,000 cells on the same lattice and origin) | no (contract only requires grid-aligned windows, each cell in ≤ 1 window) |
| Task grouping | consecutive items (admin-code order) packed into one export task while serialized inline geometry ≤ 5 MB, items ≤ 1,000 and Σ cells ≤ 5e9 | no |
| Task-level concurrency | ≤ 2 of our tasks READY/RUNNING at once (the documented default average); no quota change; EE's batch worker parallelism inside a task is server-managed and not user-configurable | no |

**Scientific invariant:**
- year 2017; EPSG:6933 10 m lattice, nearest sampling;
- first valid whole 64-band vector by ascending ID, mask before the descending mosaic (`ee_pilot_c.composite`);
- `pixelArea` weights with EE fractional boundary weights;
- one `reduceRegion(Reducer.sum())` per item over [valid_area, S×64, Q×64], with crs = pinned WKT, crsTransform
  [10,0,0,0,-10,0], bestEffort false;
- merge raw A/S/Q across an admin's items and never average per-piece SDs (`aemoments.merge_pieces`);
- the whole frozen polygon area as denominator; the ≥ .80 rule; recipe id unchanged.

The contract (`config/experiment-contract.json` `analysis_grid`) needs **no change**.

## 2. Measured offline sizes (frozen geometry and plan; `offline-size-analysis.txt`)

- 6,227 admins; inline GeoJSON is 120.6 MB in total (median 3.1 KB per admin, q99 219 KB, max 1.34 MB). The documented
  10 MB per-query limit therefore needs at least about 25 tasks.
- 2,200 distinct candidate unions; candidates per admin median 2, q99 15, max 59.
- Split bound options:

  | split side (cells) | admins split | share of cells | pieces | items total |
  |---|---|---|---|---|
  | 8,000 | 3,166 | — | — | 20,149 |
  | **16,000 (proposed)** | 827 | 59% | 3,236 | **8,636** |
  | 32,000 | 120 | 24% | 362 | 6,469 |

  At 16,000 the largest whole admin is 1.79e8 cells and the largest piece 2.56e8. The 32,000 option halves item count
  but puts up to 1e9 cells in one aggregation, with unknown batch behaviour.
- Packing at (≤ 5 MB, ≤ 5e9 cells, ≤ 1,000 items) gives about 96 tasks. At ≤ 2e9 cells it gives about 240.
- These are inline-geometry bytes, not the measured serialized task payload. The DAG adds composite expressions per
  candidate group. The real payload is measured offline after `ee.Initialize` (`ee.serializer.encode` of the
  collection) before any start; that is the first pilot step.

## 3. Minimal EE construction (one export task)

```python
groups = {}  # items of this task grouped by their ascending candidate union
for it in task_items: groups.setdefault(tuple(it["candidates"]), []).append(it)
parts = []
for cands, its in groups.items():
    comp = epc.composite(ee, list(cands))                      # existing D30 composite (mask before descending mosaic)
    owned = comp.select("source_rank").mask()
    area = ee.Image.pixelArea().updateMask(owned)
    vec = comp.select(epc.BANDS)
    stack = ee.Image.cat([area.rename("valid_area"), vec.multiply(area).rename(S_NAMES), vec.pow(2).multiply(area).rename(Q_NAMES)])
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry(it["geometry"], "EPSG:4326", False),
                                          {"admin_code": it["admin_code"], "piece_id": it["piece_id"], "asset_id": ",".join(cands),
                                           "max_pixels": it["max_pixels_band_px"]}) for it in its])
    parts.append(fc.map(lambda f, stack=stack: ee.Feature(None, stack.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=f.geometry(), crs=epc.EE_CRS_WKT, crsTransform=epc.GRID["crsTransform"],
        bestEffort=False, maxPixels=f.get("max_pixels"), tileScale=4)).copyProperties(f, ["admin_code", "piece_id", "asset_id"])
        .set({"status": "complete", "recipe": RECIPE, "batch_id": batch_id})))
task = ee.batch.Export.table.toCloudStorage(collection=ee.FeatureCollection(parts).flatten(), description=batch_id,
    bucket=BUCKET, fileNamePrefix=f"{PREFIX}/{batch_id}/attempt-{k}/rows", fileFormat="CSV",
    selectors=EXPORT_COLUMNS + ["recipe", "batch_id"])
```

**Graph notes:**
- The stack is the same expression as `reduction_obj` sums-only.
- `status` is a constant meaning only that the row exists. Numerical completeness is checked downstream: a null band
  gives a blank CSV cell, which the existing merge refuses as a technical failure, never as low coverage. A null is
  possible for a piece covering less than about 1/256 of every pixel (documented weighting).
- The count and rank diagnostics are not computed and are not faked.
- To verify in the pilot: `maxPixels` taken from a feature property (fallback: the group's maximum as a constant).

## 4. Destination, bindings, validation

- **Bucket:** existing `manuel-remote-sensing-rasters-645415198431` (US multi-region; seen in the P1 read-only
  listing with only `feature-008/`). Exact prefix: `gs://manuel-remote-sensing-rasters-645415198431/ipcch-alphaearth/<run-id>/`.
- **Permissions:** no IAM change and no new bucket. The EE writer's permission on this bucket is unknown; a permission
  failure means stop and report.
- **Columns:** `admin_code, piece_id, asset_id, status, valid_area, S_A00..S_A63, Q_A00..Q_A63, recipe, batch_id`.
- **Bindings:**
  - plan file (new adaptive plan, sha-pinned, same `recipe` column);
  - batch manifest (body sha; per task: batch_id, ordered item IDs, candidates, max_pixels; source plan sha; geometry
    and list_images shas);
  - ledger (task_id, attempt, state, `start_timestamp_ms`/`update_timestamp_ms`, EECU field if exposed, error).
- **Status:** `ee.data.getTaskStatus` / `getTaskList` (EE read only).
- **Download and validation** (GCS read/download only):
  1. List objects under `…/<batch_id>/attempt-k/`. There may be several objects; never assume one CSV.
  2. Download and record sha256 per object.
  3. Parse with `parse_export_csv`, keeping the recipe column.
  4. Check per task that the rows equal the batch's item IDs exactly.
  5. Run `validate_export(tables, plan)` (exactly once overall), then `descriptor_table`, then descriptors COMPLETE.
- **Code change needed:** `ingest_extraction` is currently hard-wired to the JSON runner (`read_c_outputs`). It needs
  a CSV branch that feeds the same validate → descriptor_table chain, plus `recipe` added to the selectors.

## 5. Resume, retries, old results

- **Retries.** EE may retry a batch task server-side (up to 5 failures before FAILED, per the docs); that is not ours
  to control. Our client makes no automatic resubmission in the pilot.
- **Bulk resubmission.** In bulk, a FAILED task may be resubmitted only as the same batch under a new `attempt-k`
  prefix, within a released attempt cap. Nothing is overwritten, and exactly one COMPLETED attempt per batch is
  ingested (the first; extras are recorded and excluded). `validate_export` refuses duplicates.
- **Old results (recommendation): do not mix provenance.**
  - Re-extract all 6,227 admins in batch.
  - Use the old 8,280 window outputs (774 complete admins) as a free verification set: compare batch against interactive
    raw A/S/Q per admin, with differences reported.
  - Cost: about 14.6% more cells.
  - Reusing them instead would need an explicit old→new provenance decision. Only REF 658 has been checked directly.

## 6. Reused machinery and the smallest code diff

- **New script** `scripts/ee_batch_c.py` (about 150 lines), assembled from existing parts:
  - `epc.composite`, `EE_CRS_WKT`, `GRID`, `RECIPE`, `load_released`-style pin checks;
  - the legacy `extract_alphaearth.export` loop pattern: ledger, `wait_for_slot`, `max_tasks`, descriptions, GCS prefix
    and `refresh` (pattern only; the legacy native single-asset reducer is NOT reused);
  - `Budget` with `attempts=1` for the client calls.
- **New helper** `extraction.plan_adaptive(geometry, footprints, contract, split_max_cells)`: whole admin when the
  lattice box is within bound, else `lattice_windows`. Same output columns as `plan_c_requests`, plus `item_kind`.
- **Ingest:** `descriptors.ingest_extraction` gets a CSV source using `parse_export_csv`. The rest of the chain is
  unchanged.
- **Tests (offline, fake ee):** graph structure (one reduceRegion per item, same composite and grid, selectors),
  packing bounds, the ledger/attempt rule, CSV ingest (multi-object, exactly-once, null refusal).
- **No orchestration framework.**

## 7. Spec amendment (operational only; shown, not applied)

`implement.md` P1 delivery currently reads "direct EE computation returned locally". Amend to:

> "P1 bulk delivers the same D30 per-item sufficient statistics by EE batch table export
> (`Export.table.toCloudStorage`, CSV) to `gs://manuel-remote-sensing-rasters-645415198431/ipcch-alphaearth/<run-id>/`,
> downloaded and validated locally. Items are whole admins, or grid-aligned lattice pieces for admins whose
> whole-admin lattice box exceeds the released split bound. Cloud writes require explicit user authorization per
> release."

No decision, PRD science or contract text changes.

## 8. Unknowns (stated, not estimated)

- per-item and per-task batch runtime and EECU;
- whether batch tasks are subject to the per-user aggregation ceiling;
- actual concurrent-task allowance for this tier;
- the EE writer's permission on the bucket;
- CSV float formatting precision;
- whether a per-feature `maxPixels` property is accepted.

The full-bulk wall time and EECU are UNKNOWN until the pilot reports EECU and time per cell.

## 9. Pilot proposal (for later cloud-write approval; one task; not a bulk release)

**One export task, 10 items, about 22 KB inline geometry:**

| item | why |
|---|---|
| 658 whole (1.02e7 cells, 2 candidates) | interactive whole-admin reference (identical to its windows): checks CSV precision and server-path equality |
| 4711 whole (3.15e7, 4 candidates) | interactive reference (98 s) |
| 1050 whole (7.65e7, 3 candidates, MultiPolygon) | timed out at the 120 s client deadline interactively; tests batch completion |
| 4505 split into 7 pieces on the 16,000 lattice (cells 2.17e8, 1.80e8, 2.65e6, 1.86e8, 8.45e7, 2.65e7, 7.44e6; Σ 7.04e8; 6 candidates) | a genuinely large admin: tests piece size up to the bound, the raw-moment merge, and a small edge piece |

**Writes:** 1 task, with objects under `gs://manuel-remote-sensing-rasters-645415198431/ipcch-alphaearth/pilot-<date>/`.

**Calls:**
- EE: initialize; projection check; one `task.start()`; ≤ 40 status polls over ≤ 6 h.
- GCS: ≤ 5 list/download reads.
- No client resubmission.

**Success:**
- the task COMPLETED;
- rows exactly equal the 10 item IDs; all 129 values finite;
- 658 and 4711 match their interactive whole-admin sums (differences reported);
- 4505 merges to finite descriptors with coverage reported;
- task runtime and any EECU field recorded;
- measured serialized payload recorded.

**After the pilot:** propose split/packing bounds and an attempt cap from the measurements, and write a bulk envelope
for a separate cloud-write authorization.
