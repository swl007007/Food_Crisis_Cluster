# Speed pilot: offline implementation checkpoint (no EE/GCS call; nothing submitted; release_id null)

## Code (new files only; no existing symbol edited)

| file | sha256 | role |
|---|---|---|
| scripts/ee_batch_c.py | 2fa4102777e7b2eb762d64981981488aac74ed7b1eea588cb99ed7f75d813330 | EE interpreter: `submit` / `monitor` / `lookup` / `download` |
| scripts/build_speed_pilot.py | a39b336f6f3c989cee721a4805cac789948f63e1f586ed2594491308d252f626 | ipcch-geo: adaptive plan, exact selection, pins, manifest |
| scripts/validate_speed_pilot.py | 30329aa7857cb480ccdb26966ee4ac425d5d6b5af05a7e93351e2bae145f79e9 | ipcch-geo: exactly-once CSV ingest, fixed gates, timing and projection |
| tests/test_speed_pilot.py | bbfe2fd0ae97d28005e9d4a94093d4d47ec79829d5b132b33a19454cc6626f53 | 10 focused offline tests |

**Reused unchanged:**
- `ee_pilot_c` (c18513e2…): `composite`, WKT, grid, recipe, `CumulativeBudget`, `projection_check`;
- `ee_metadata` (da50b46d…): `Budget`/`Stop`;
- `extraction` (b7010e28…): `plan_c_requests`, `check_piece_binding`, `parse_export_csv`, `EXPORT_COLUMNS`;
- `aemoments` (efbbdaef…): raw-moment merge and descriptor;
- `config/experiment-contract.json` (b75a7106…): no change.

GitNexus: these symbols are not indexed (index stale at 50e1f22). The new files are not referenced by any existing code.

## Adaptive plan (exact, built from the frozen geometry and footprints)

- `plan_c_requests(..., max_cells = 16000²)`: an admin that fits one 16,000-cell lattice square is one whole item;
  otherwise it becomes side-16,000 lattice pieces on the same lattice and origin. Plan sha256
  ae2785fe7faa97db5600b6fee064cd0bc37611f67a082e2e27dd53e71f688ab5.
- 6,227 admins → **9,138 items**: 4,898 single-item admins plus 1,329 split admins (4,240 pieces). Max item 2.560e8
  cells.
- Reuse: 774 admins / 8,268 windows (supervisor inventory fb109fe8…).
- Remaining: **5,453 admins, 7,996 items, 3.98874e11 cells** (baseline 3.98870e11; the small delta comes from actual
  piece geometry).
- **Delta from the design estimate.**
  - Items: 9,138 against the estimated 8,636; split admins: 1,329 against 827.
  - Cause: the estimate used a lattice-box-area rule, whereas `plan_c_requests` splits whenever either side exceeds
    16,000 cells, so elongated admins split too.
  - No piece identity or cell count was invented: all numbers below come from the actual pieces.

## Pilot manifest

- **Manifest:** file 7131c83b…, body `request_sha256` 7b8145b688cb5e1be8d410bbda7dd289d4f137efb6b877014a97d2d7a5592c69.
  Its rebuild is byte-identical.
- **Task A:** 53 items; 50 fresh with **2.418e9** fresh cells; non-fresh 8.35e7; 52 candidate groups; max item 2.17e8.
- **Task B:** 54 items; 50 fresh with **2.562e9** fresh cells; non-fresh 2.452e8; 52 groups; max item 2.55e8.
- **Fresh total:** 100 items, 4.980e9 cells, 1.25% of remaining.
  - Design estimate: 95 items / 4.99e9 cells.
  - Selection is systematic over the actual 7,986-item fresh pool, with n = round(5e9 / mean item cells).
  - Purpose (size-representative, balanced A/B) and scale unchanged.
- **Non-fresh rows** (separate from every fresh-throughput numerator):
  - references 658 (A) and 4711 (B): interactive response files pinned by sha256;
  - reuse validation 15 and 117 (A), 349 and 229 (B): all of their old windows pinned (file and response sha256).
- **Destination:** exactly `gs://manuel-remote-sensing-rasters-645415198431/ipcch-alphaearth/speed-pilot-20261011/{A,B}/attempt-1/`.
  Descriptions `ae2017d30_7b8145b688_{A,B}_a1`.
- **Limits:**
  - ≤ 2 tasks; serialized body ≤ 8,000,000 bytes per task, refused before any start;
  - cancel at 3,600 s from the first submission (own nonterminal tasks, ≤ 2 cancels);
  - 1 grouped status read per 60 s, ≤ 75 reads;
  - ≤ 20 GCS reads;
  - ≤ 84 cumulative EE calls;
  - deadline 120 s; tile scale 4; no client retry;
  - group maxPixels ceiling 3.6326e10.

## Payload measurement strategy

Inline geometry is 0.687 MB (A) and 0.517 MB (B). That is NOT the payload. The payload is measured only as
`len(ee.serializer.toJSON(collection))` after `ee.Initialize`, for BOTH tasks before ANY `start()`. If either exceeds
8,000,000 bytes, nothing is started and no INTENT record is written (tested). That is the first, read-only step of a
later authorized pilot.

## Offline tests (fake ee; `unittest-speed-pilot.log`, `unittest-focused.log`)

`test_speed_pilot` (10 tests):
- one sums-only reduceRegion per item, with the same descending-mosaic composite, WKT grid, bestEffort false and
  tileScale 4;
- one constant maxPixels per group; CSV with the selectors and exact prefixes;
- INTENT recorded before each start;
- an oversized body means no start and no intent;
- an ambiguous `start()` means INTENT then AMBIGUOUS, no second task, rerun refused, monitor refused, read-only lookup;
- grouped two-ID status reads; cancel only own nonterminal tasks after 60 min, with reads continuing to terminal;
  both running means both cancelled once; read cap 75;
- multi-object download with sha256, no overwrite, read cap;
- exactly-once CSV ingest across objects (duplicate, missing and null rows refused);
- deterministic balanced selection and group maxPixels.

Focused total with `test_c_production`, `test_c_extraction` and `test_extraction_offline`: 60 OK.

## Later pilot commands (only after USER cloud-write authorization of this exact manifest with a release_id)

```bash
cd IPCCHAlphaEarthExperiment; R=~/ipcch-alphaearth-runs/p1-speed-pilot-<date>   # fresh; released manifest copied unchanged
/opt/earthengine/bin/python scripts/ee_batch_c.py submit   --request $R/inputs/speed_pilot_manifest.json --out $R/out
/opt/earthengine/bin/python scripts/ee_batch_c.py monitor  --request $R/inputs/speed_pilot_manifest.json --out $R/out
/opt/earthengine/bin/python scripts/ee_batch_c.py download --request $R/inputs/speed_pilot_manifest.json --out $R/out
PYTHONPATH=.:scripts /home/swl007007/.venvs/ipcch-geo/bin/python scripts/validate_speed_pilot.py --request $R/inputs/speed_pilot_manifest.json \
  --out $R/out --full-run ~/ipcch-alphaearth-runs/p1-c-full-20261010 --interactive-run ~/ipcch-alphaearth-runs/p1-whole-admin-pilot-20261011 \
  --plan $R/inputs/adaptive_plan.csv.gz --remaining-cells 3.98874e11
```
