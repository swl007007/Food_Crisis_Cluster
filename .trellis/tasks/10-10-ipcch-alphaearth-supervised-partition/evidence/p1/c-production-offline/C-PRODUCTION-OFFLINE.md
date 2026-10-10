# D30 production C connection (offline) and small-live proposal — 2026-10-10

Scope: offline only, under the existing implementation authorization. No EE/GCS call, no fit. The full production
manifest is a held DRAFT (`release_id` null). The four sample manifests below are proposed for one bounded small-live
release.

> **Hash correction (2026-10-10, after commit 96613c1; documentation only).** The first version of this file and the
> checkpoint message quoted `request_sha256` body hashes from a build made before `outer_timeout_s` was added to the
> limits:
> - full: 3d19a0a5…
> - S1: 66e9ed9f…
> - S2: b60bdad2…
> - S3_batching: 2890a322…
> - S3_single: 4324fdc7…
>
> The committed manifest files, the manifest file-sha column and `production_report.json` were always the current
> build. The values below are recomputed from those files and equal the supervisor release receipt
> `codex-C-small-20261010`. No geometry re-enumeration or code change was involved.

## Code (pinned; equal to supervisor receipt `ipcch-alphaearth-supervisor-c-production-fixed-check-20261010.json`)

| file | sha256 |
|---|---|
| scripts/ee_pilot_c.py | bfec9e1bd158e5f50e3be11ade32d26165e943da94e739de1697e09227a76e1c |
| scripts/ee_metadata.py | 982e3640d156acef8f69f0fce0fb494eac8656ff2e640ca6c9b3f633ff200bfc |
| scripts/build_c_production.py | b9007cfa2e0d6d5e69a8d885451b782ccbac0e4914cdbd9a89d07cf2047be715 |
| ipcch_alphaearth/extraction.py | b7010e2859a945cc511ef319ba5bf5bfff08fa917a1938e4bc2ff545d99a8b8b |
| ipcch_alphaearth/descriptors.py | 955f55fd6271f4cf53e40be528e942883c302c4f0f3980dbc683a836b9999e1a |
| ipcch_alphaearth/runinputs.py | 85db7c55d198850fd377174ff8b46ae2b20fe84660a72a2df34566dc0ecb1f0f |
| ipcch_alphaearth/cli.py | 4307845d63bb9ce47c13b15710e19d5abc7c5a6336d90f6b5021b09e683e8e7d |
| tests/test_c_production.py | 70c2f27688463ff1cef68978f302313e8375c2c449da6a202e778b30cfc6126e |

What was connected:
- **Same pinned reducer, composite, WKT transport and projection check.** `reduction_obj` is the pilot's reduction left
  unevaluated.
- **Durable per-piece output.** Each finished piece is written atomically to `pieces/<sha1(piece_id)>.json` with its
  binding: manifest sha, piece, admin, candidates, recipe and response digest.
- **Resume and ledger.** `ledger.jsonl` is append-only. A resume recomputes only pieces without a valid output. A
  corrupt or foreign output stops the run and is never overwritten. An output directory is pinned to one manifest.
- **Call cap is cumulative across resumes.** `call_reservations.log` gets one line per attempt, written before the
  attempt is sent.
- **One shared stop control.** It is checked at every attempt reservation, retries included:
  - a fatal error is published before it is raised;
  - the invocation wall budget starts before init and projection;
  - stop rules: consecutive failures, failure fraction, call cap;
  - already-issued requests may finish.
- **Two separate clocks.** The invocation wall budget (`wall_budget_s`) and the outer command timeout
  (`outer_timeout_s`) are distinct. The manifest must state both, with outer ≥ wall + deadline + 60 s.
- **Ingest checks, all before anything is written to `extraction/`.** `descriptors.ingest_extraction` refuses unless:
  - the manifest is released, full-plan and not a sample;
  - the supplied plan bytes equal the manifest's pinned `provenance.plan_sha256` (this pins frozen denominators and
    geometry provenance);
  - the piece set, admin and ordered candidates equal the plan;
  - every per-piece binding is valid (`validate_export`);
  - every piece is numerically complete.

  Refusals are recorded under `extraction_attempts/`. A valid all-zero A remains genuine no coverage.
- **Descriptors under fresh bindings.** `attach_descriptors` verifies the stored files and the plan pin, requires every
  prepared admin to be planned, runs `descriptor_table`, and writes `descriptors/` bound to the config identity,
  prepared manifest, extraction record and plan provenance.
- **Staleness check.** `load_run_inputs` refuses stale descriptor bindings.
- **CLI.** New `ingest-extraction` command. Non-local `prepare` runs `attach_descriptors`.
- **Sliver fix (found here, before any live run).** Per-piece maxPixels is now sized from the window's lattice box, not
  its geodesic area. Sub-cell sliver windows (8 pieces < 1 cell, minimum 0.09 cells) would otherwise get a cap of about
  15 band-pixels. That is below the 129 band-pixels of one touched cell, so EE would have refused them.

Tests: `tests/test_c_production.py` has 16 tests on a fake ee:
- interruption and resume, with done outputs unchanged;
- refusal of corrupt and foreign outputs and of a manifest mismatch;
- batch 3 and concurrency 4 give identical values with ⌈n/3⌉ calls, each piece submitted once;
- stop rules: consecutive failures, failure fraction, fatal, non-retryable;
- cumulative budget;
- end-to-end plan → manifest → run → read → ingest → descriptors;
- sample, unreleased, tampered and incomplete extractions refused;
- the supervisor fixed list:
  - a published fatal blocks other workers' retries (barrier makes it deterministic);
  - the wall budget applies at the retry reservation;
  - the invocation clock includes init;
  - plan-pin refusal (doubled denominators);
  - null and nonfinite pieces refused before COMPLETE, after which a corrected ingest proceeds;
  - all-zero piece stays no coverage;
  - outer-timeout statement required.

Logs:
- `unittest-focused.log`: 31 tests OK (C production + C extraction).
- `unittest-full-prebarrier.log`: 105 tests OK. Run after fixed-list items 1–3; the later barrier change touches the
  test fixture only.
- Supervisor logs: `…-tests-20261010.log` (earlier fixture race, 1 fail, retained) and `…-tests-fixed-20261010.log`
  (16 PASS).

Batching check: the runner's `batch_size` evaluates several identical per-piece `reduceRegion` expressions in one
`ee.Dictionary(...).getInfo()`. The numerical reducer does not change; only call overhead is shared. I did not add a
`computeFeatures` path:
- it would need `Image.reduceRegions` over a FeatureCollection with one shared composite, so it only works for pieces
  with identical candidate sets;
- `reduceRegions` has no `bestEffort` argument, only `maxPixelsPerRegion`, so it is a second invocation path with
  different guards.

Sample S3 below measures whether `batch_size` 4 reduces per-piece time. It also checks value identity against the same
pieces run singly.

## Production enumeration and completeness (local, deterministic: two builds byte-identical)

Builder: `PYTHONPATH=.:scripts python scripts/build_c_production.py --geometry <run b staged geometry .shp> --list-images
~/ipcch-alphaearth-runs/p1-metadata-20261010/ee/list_images.json --out ~/ipcch-alphaearth-runs/p1-c-production-offline-20261010`

- **Inputs.** Geometry .shp is 9df70f3a…; `list_images.json` is b233255b… (1,683 footprints). The window cap is
  1.6e7 lattice cells.
- **Plan.** `production_plan.csv.gz` bccf45c215d21955ea16b09bf17e40d0c738c583f8454fe538b39b1a01926dda. The rows are
  the same plan as the pilot2 copy; the bytes differ because of the deterministic writer.
- **Admins.** 6,227 admins = 6,227 geometry admins; all have pieces. There are 0 zero-source admins, 0 no-source
  windows and 0 km² of no-source area.
- **Pieces.**
  - 57,801 unique piece ids, 4.6711e11 cells (6.026e13 band-pixels).
  - Cells per piece: q10 4.8e5, median 7.8e6, q90/max 1.6e7. The lattice box is ≤ 1.6e7 for every piece.
  - Geodesic window area exceeds the lattice box by at most 1.7e-4 (edge convention).
- **Candidates per piece:**
  - 1: 24,395
  - 2: 22,718
  - 3: 3,121
  - 4: 5,814
  - 5–8: 1,647
  - 9–16: 106 (max 16)
- **Geometry complexity.** Pieces per admin: median 6, q99 62, max 459. Vertices per piece: median 12, q99 1,004, max
  33,387. There are 4,459 multipart pieces.
- **Small pieces.** 8 pieces are below 1 cell and 68 are below 100 cells.
- **Area closure.** Per admin, (Σ window geodesic area + no-source area) / frozen polygon area − 1 has median 2.0e-7
  and max 1.1%. This is the recorded edge-convention item. It affects only the diagnostic `expected_cells`. The
  descriptor denominator is always the whole frozen polygon area.

**Full manifest — DRAFT, held, unreleased:**
- `production_manifest.json` fb1b643f…: request_sha256 b6bf588612ac0e230e37bc0db38d16817332397aa5df5e779961f5e9a1207a87.
- Pieces: 9f22cb263e1f358175c364c9ebcb8937bb65285591bd9b3cf00e1a28f28baf7d (43.8 MB, outside Dropbox).
- max_pixels 2.2704e9.

Its limits are placeholders, to be set at release from the sample results:
- concurrency 4, batch 1;
- calls 60,694;
- invocation wall 4 h, outer timeout 4 h 10 min, resumable.

At the pilot rates (1 s + 2.0–3.2e-6 s per cell) the full plan totals about 1.0–1.6e6 s of request time.

## Proposed small-live release (S1 → S2 → S3; 36 distinct pieces; ≤ 84 calls total)

| stage | manifest sha256 (file) | request_sha256 | pieces sha256 | n | conc | batch | calls cap | wall s | outer s |
|---|---|---|---|---|---|---|---|---|---|
| S1_sequential | 84da17c4… | 5774339c45141ca233adc2f845a37bf76145d0b785f4d87d5b15c6b49183097d | 41a938fb2d33cfdfe73fd252c0256a41996c29d21649ac0fb112432a822538cf | 12 | 1 | 1 | 26 | 900 | 1200 |
| S2_concurrency4 | 4da38ac2… | 68e40655f2dbcc3930442760e9e581beb4659add0f3d023f8f4e237f870775cb | 42510e915e7e9ff2b3a3ea47962dba758e5cb1acd3dd0338ad747294b266b549 | 16 | 4 | 1 | 34 | 600 | 900 |
| S3_batching | dc418f2a… | eb580b24e36bdeb337da983d2c4241062554512cfdc0b7efcbe49cd90d01e512 | c16db67db0fe3796b653fe06b4ad6036d3188978d4d682b56c0f95d003697d46 | 8 | 1 | 4 | 6 | 300 | 600 |
| S3_single | ac4bab3f… | eae40c1cd1d28f516ef82b83e5d8b2cf03887c1c4a9a46dc21a5b8980b4abb6e | c16db67db0fe3796b653fe06b4ad6036d3188978d4d682b56c0f95d003697d46 | 8 | 1 | 1 | 18 | 300 | 600 |

All four stages share these settings:
- deadline 120 s;
- SDK retries 0, budget retry 1;
- tile scale 4;
- max_consecutive_failures 2;
- max_failure_fraction 0.25 after 4 attempts.

Each stage's cap is 2 + 2 × groups: init, projection check, and every group with its one retry. Expected calls are
about 46. Full manifest content is in `samples/` and the selection is in `production_report.json`.

**S1 — 12 deterministic representative pieces, run sequentially:**
- 1-candidate pieces at the cell-count q10, median and max (16,000,653 cells);
- 2–3-candidate pieces at the median and max (16,002,702 cells), plus the vertex maximum (33,387 vertices);
- the 16-candidate piece with the most cells;
- 8-candidate pieces at the median and max;
- the most complex multipart piece (8 parts, 29,821 vertices);
- the q99.9-vertex piece (4,230 vertices);
- the smallest sliver (0.09 cells).

**S2** — 16 distinct pieces at the cell-count 1/17…16/17 quantiles, at concurrency 4.

**S3** — 8 distinct overhead-dominated pieces (about 1.3e5 cells), run first in batches of 4 (2 calls) and then singly
(8 calls).

Gates (each stage runs only if the previous one passes; otherwise stop and report):
- **S1 → S2.** S1 must pass all of the following:
  - complete, with 0 failed or halted pieces;
  - projection check ≤ 1 mm;
  - every piece < 90 s;
  - Σ owned-by-rank = valid area (relative ≤ 1e-9);
  - valid cells ≤ lattice box;
  - finite sums, and the sliver piece completed (0 or 1 cell).
- **S2 → S3.** S2 must be complete with 0 failed or halted pieces, and wall time ≤ 600 s.
- **S3.** Both runs complete. I report per-piece seconds for batch vs single and value identity per piece (expected
  identical; any difference is reported, not normalized).

Commands. Run in this order; each stage gets a fresh output dir. The supervisor writes the released copy:

```bash
cd IPCCHAlphaEarthExperiment
for S in S1_sequential S2_concurrency4 S3_batching S3_single; do   # stop at the first failed gate
  R=~/ipcch-alphaearth-runs/p1-c-small-<release>/$S; mkdir -p $R/input
  cp ~/ipcch-alphaearth-runs/p1-c-production-offline-20261010/samples/${S}_pieces.jsonl.gz $R/input/
  # released copy: draft manifest with release_id set (body sha unchanged) -> $R/input/${S}_manifest.json
  timeout <outer_timeout_s> /opt/earthengine/bin/python scripts/ee_pilot_c.py --request $R/input/${S}_manifest.json --out $R/out
done
```

No GCS/Drive writes, no export, no fit. Outputs stay local outside Dropbox, and compact results go to evidence. The full
manifest stays held until the sample results and a separate release.
