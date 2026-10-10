# D30 full extraction envelope (concrete; HELD until supervisor release) — 2026-10-10

Basis: the released small-live evidence (`../c-small-20261010/C-SMALL-result.md`). The default is the measured
concurrency 4 with batch 1. Neither concurrency nor large-piece batching is enlarged on the strength of the small S3
run. Reducer, composite, CRS transport, ordered geometry/candidate pieces and scientific recipe are unchanged.

## Fixed inventory

| item | identity |
|---|---|
| implementation | this commit (the parent 9450c26 + the envelope delta) |
| scripts/ee_pilot_c.py | f5c59f9dbd4330f10b131d2787523c58312901d7d26613ffd2c391cb750f9c7e |
| scripts/ee_metadata.py | 982e3640d156acef8f69f0fce0fb494eac8656ff2e640ca6c9b3f633ff200bfc |
| scripts/build_c_production.py | 5ea13a301ca1a0cea67f1af5689c27c8c7f530082fc3d72d8b66be806d510af8 |
| scripts/check_c_outputs.py | 15af8d4b93f2073b82ee5e84b05af7c6a589595923084da3354375bcc5d991e5 |
| tests/test_c_production.py | a6dc4473c5b1717e9858969755ca9e2961b14693a299528a426f5c1cc23f658d |
| ipcch_alphaearth/extraction.py / descriptors.py / runinputs.py / cli.py | b7010e28… / 955f55fd… / 85db7c55… / 4307845d… (unchanged since 96613c1) |
| plan `production_plan.csv.gz` | bccf45c215d21955ea16b09bf17e40d0c738c583f8454fe538b39b1a01926dda |
| pieces `production_pieces.jsonl.gz` (43.8 MB) | 9f22cb263e1f358175c364c9ebcb8937bb65285591bd9b3cf00e1a28f28baf7d |
| draft manifest file `production_manifest.json` | e6d6d28215c243502894ad034471ab1723c7dcc19c5219ff87ee752d1347ff06 |
| draft manifest body (`request_sha256`) | 63d0c7abc11a02a974c3d5c0aca2c05b124505a6ff3d40081c8e25015f42496a |
| request inventory | 57,801 pieces = every planned piece; 6,227 admins; 4.6711e11 cells; candidates 1–16 |

- **Draft location.** The draft is in `~/ipcch-alphaearth-runs/p1-c-full-draft-20261010/` with `release_id` null. The
  rebuild is deterministic: plan, pieces and the four released sample manifests are byte-identical to the earlier
  build. The earlier draft directory `p1-c-production-offline-20261010/` is kept and is superseded only in its limits.
- **Tests.** `unittest-full.log`: 108 OK. Supervisor check `ipcch-alphaearth-supervisor-full-envelope-check-20261010.json`
  with its 19-test log (sha256 10567d98b13a472acdd6654291af4f28129d851a4c2d903ec898877d79e1fffd).

## Limits (inside the hashed manifest body)

| limit | value | meaning |
|---|---|---|
| concurrency / batch_size | 4 / 1 | measured S2 default |
| wall_budget_s | 21,600 (6 h) | invocation wall from before init; no new attempt starts after it |
| outer_timeout_s | 22,200 (6 h 10 min) | hard `timeout` kill per invocation (≥ wall + 120 s deadline + 60 s) |
| max_invocations | 20 | counted at invocation start in `invocations.log` (a killed invocation counts) |
| max_high_level_calls | 59,576 | cumulative over all invocations: 57,801 pieces + 1,735 (3%) re-requests + 2 × 20 init/projection |
| max_consecutive_failures | 5 | per invocation |
| max_failure_fraction | 0.02 after 200 attempts | per invocation |
| deadline_ms / SDK retries / budget retry | 120,000 / 0 / 1 | as in all pilots |
| max_pixels (per-piece band-px guard) | ≤ 2.2704e9 | 129 × lattice box × 1.1; box ≤ 1.6e7 |

**Hard runtime boundary:**
- One invocation runs at a time, never two in parallel on the output directory.
- Request starts fall inside at most 20 × 6 h = 120 h of invocation windows.
- With the per-invocation hard kill, the total runtime reservation is at most 20 × 22,200 s = 123 h 20 min.
- Requests already issued when a window closes may finish, each bounded by the 120 s deadline.

**Aggregate cells ceiling:**
- The hashed manifest fixes the cells at 4.6711e11.
- Re-requests are bounded by the call cap: at most 1,735 extra piece requests × ≤ 1.6e7 cells, so the aggregate is
  ≤ 4.95e11 requested cells.

**Expected:**
- S1+S2 fit: t ≈ 4.0 s + 1.65e-6 s/cell → about 1.0e6 request-seconds.
- At the S2 effective parallelism of 3.5 that is about 80 h, i.e. about 14 invocations. 20 invocations leave about 1.5×
  margin.
- Expected calls: about 57,801 + 2 per invocation.

## Location and bytes

Fresh directory outside Dropbox: `~/ipcch-alphaearth-runs/p1-c-full-<release date>/`.
- `inputs/`: the released manifest plus pieces copied unchanged, about 44 MB.
- `out/` holds:
  - 57,801 piece JSONs (measured mean 4.3 KB, max 4.9 KB): about 0.25–0.28 GB, or about 0.47 GB on disk with 8 KB
    blocks;
  - the ledger (about 11 MB), request log (about 9 MB) and reservations (about 1 MB);
  - `failed_responses/` (small).
- Ceiling: 1 GB. Free space: 649 GB.
- Compact summaries and checker reports go to task evidence. Raw outputs stay outside Dropbox.

## Run, stop and resume policy

Each invocation (the operator runs exactly one at a time):
```bash
cd IPCCHAlphaEarthExperiment
F=~/ipcch-alphaearth-runs/p1-c-full-<release date>
timeout 22200 /opt/earthengine/bin/python scripts/ee_pilot_c.py --request $F/inputs/production_manifest.json --out $F/out \
  >> $F/invocations.stdout.log 2>&1; echo "exit $? $(date -u +%FT%TZ)" >> $F/invocations.exit.log
python3 scripts/check_c_outputs.py --request $F/inputs/production_manifest.json --out $F/out --json $F/check-<n>.json
```

- **Resume.** The next invocation may start immediately under the same release only if all of these hold:
  - the previous invocation ended on its invocation wall budget (`stopped` = "invocation wall budget …");
  - the checker exits 0 (no binding or numerical violation);
  - the checker projects completion within the invocation cap;
  - reservations and invocations are below their caps.

  A resume recomputes only pending pieces: absent outputs, earlier failures, and null/nonfinite responses, which are
  kept under `failed_responses/` and never saved as done outputs. Done outputs are never rewritten.
- **Stop and report; no resume without supervisor review:**
  - fatal auth/permission/quota/429;
  - 5 consecutive failures, or failure fraction > 2%;
  - call or invocation cap reached;
  - non-retryable size rejection;
  - outer-timeout kill (exit 124);
  - any checker violation: identity or binding mismatch, null or nonfinite output, rank-area discrepancy, count above
    the lattice box;
  - projected completion beyond 20 invocations.

  Outputs are preserved as they are.
- No cap is enlarged and no fresh budget is used on a resume: reservations and invocations are cumulative per output
  directory.

## Partial checkpoint schedule

- **After invocation 1 (≤ 6 h 10 min):** report to the supervisor:
  - done and pending pieces and cells;
  - calls, failures and null results;
  - observed cells per wall-second against the model;
  - projected invocations;
  - the checker JSON.

  Continue without waiting if every resume condition holds.
- **After each later invocation:** a brief delta of the same fields, plus explicit milestones at ≥ 25 / 50 / 75 % of
  cells.
- **Completion:** the checker reports `complete` (57,801 done, 0 violations). I then report, and ingest follows below.

## Downstream ingest (local; no EE)

Run b predates the D30 contract block, so its config identity differs. A fresh run is required:
```bash
cd IPCCHAlphaEarthExperiment; PY=/home/swl007007/.venvs/ipcch-geo/bin/python; F=~/ipcch-alphaearth-runs/p1-c-full-<release date>
RUN=~/ipcch-alphaearth-runs/p1-descriptors-<date>          # fresh; --kind formal if it is to feed main results
$PY -m ipcch_alphaearth preflight --run $RUN --hash-inputs [--kind formal]
$PY -m ipcch_alphaearth prepare --run $RUN --local-only
$PY -m ipcch_alphaearth ingest-extraction --run $RUN --plan $F/inputs/production_plan.csv.gz \
    --manifest $F/inputs/production_manifest.json --outputs $F/out
$PY -m ipcch_alphaearth prepare --run $RUN                  # attach_descriptors -> descriptors/COMPLETE.json
```

The plan copy in `inputs/` must be the pinned plan (sha256 bccf45c2…). Ingest refuses other bytes.

**What ingest checks before anything is written to `extraction/`:**
- the manifest is released, full-plan and not a sample;
- plan bytes equal the manifest pin;
- the piece, admin and candidate set equals the plan;
- every binding is valid;
- every piece is numerically complete.

**Attach** writes the descriptor table and coverage ledger, bound to the config identity, prepared manifest, extraction
record and plan provenance. Project-data fitting remains a separate later release.
