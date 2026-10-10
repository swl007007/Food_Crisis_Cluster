# C small-live result (release codex-C-small-20261010)

Receipt: `ipcch-alphaearth-c-small-live-release-20261010.json`.

Pre-run checks:
- HEAD 96613c1, tree clean, all 8 source hashes equal the receipt.
- The four released manifests have unchanged bodies: their recomputed hashes equal the receipt and the local drafts,
  and only `release_id` was added.
- Manifests and pieces were copied byte-identical into `~/ipcch-alphaearth-runs/p1-c-small-20261010/inputs/`
  (`released_inputs.sha256`).
- Each stage used a fresh output directory and the EE interpreter `/opt/earthengine/bin/python`, under
  `timeout <outer_timeout_s>`.

Boundary: direct EE computation returned locally. No export, no GCS/Drive write, no IAM/billing change, no fit.

| stage | calls used / cap | pieces | invocation s | failed/halted | gate |
|---|---|---|---|---|---|
| S1_sequential | 14 / 26 | 12/12 | 239.0 | 0 | PASS |
| S2_concurrency4 | 18 / 34 | 16/16 | 75.1 | 0 | PASS |
| S3_batching | 4 / 6 | 8/8 | 8.9 | 0 | PASS |
| S3_single | 10 / 18 | 8/8 | 11.3 | 0 | PASS |

Totals: 46 of 84 calls, 36 distinct pieces, 44 result records. No retry was needed and no error was logged.

**Invariants (every stage, `gate_check.py`):**
- Projection check: 2.8e-9 m.
- Σ owned area by source rank = valid area, within ≤ 2.5e-16 relative.
- Valid cells ≤ lattice box.
- All 129 sums finite.
- Valid area / geodesic window area: 0.9995–1.0002 (the recorded edge convention).

**S1 (representative, sequential):** maximum 45.96 s, for the 16-candidate 1.6e7-cell piece, where 10 source ranks own
area.
- The 1.6e7-cell single-source piece took 27.0 s.
- The most complex multipart piece (8 parts, 29,821 vertices) took 15.6 s. The 33,387-vertex piece took 11.7 s.
- The sliver 1266|w001_000 (0.09 cells) returned 0 counted cells and 9.0 m² weighted valid area out of 9.1 m². The
  count is unweighted; the sums carry fractional overlap weights.

**S2 (concurrency 4):** 16 pieces in 75.1 s wall against Σ per-piece 263 s, an effective parallelism of 3.5. Per-piece
times were not inflated relative to S1.

**S3 (batch 4 vs single, the same 8 pieces of about 1.3e5 cells):**
- Valid-cell counts are identical.
- 4 of 8 responses are bit-identical.
- The other 4 differ in 90 sum fields and 2 rank-area values, all at ≤ 3.1e-16 relative: the last bit of summation
  order. Nothing was normalized.
- Disclosure (the supervisor's independent check `ipcch-alphaearth-supervisor-small-live-check-20261010.json`):
  - the floating-point reductions are not bit-exact; the largest absolute S/Q difference is 9.31e-10;
  - derived means differ by ≤ 5.55e-17 and SDs by ≤ 6.42e-16;
  - area and counts are identical.

  The raw outputs are retained unnormalized.
- Timing: batch calls 4.43 + 2.89 s (7.3 s for 8 pieces in 2 calls); singles 0.13–2.66 s (10.1 s in 8 calls).

The singles ran second, so their timing is cache- and order-conditioned. This is not an unbiased speed ratio.

**Rate model (descriptive):** a least-squares fit over the S1+S2 pieces gives t ≈ 4.0 s + 1.65e-6 s/cell. Over all 36
pieces it gives 2.6 s + 1.76e-6 s/cell. Applied to the full plan (57,801 pieces, 4.67e11 cells) that is ≈ 1.0e6
request-seconds ≈ 279 h. If the S2 parallelism holds, that is ≈ 80 h at concurrency 4.

This is an extrapolation from 36 pieces; quotas and EECU cost are not visible. The full extraction remains held.

Files:
- `runs/<stage>/`: ledger, request log, run summaries, reservations and per-piece outputs;
- `runs/*.gate.json` and `runs/S3_compare.json`;
- `gate_check.py` and `compare_s3.py`: local analysis only.
