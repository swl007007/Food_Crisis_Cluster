# Bounded c8 parallelism probe — proposal (NOT run; awaiting supervisor binding) — 2026-10-11

Requested by the user ("测试一下是否可以并行", relayed by the supervisor). This uses the same frozen reducer, the same
runner (`scripts/ee_pilot_c.py` f5c59f9d…, commit 736efa95) and the released production inventory. No code change.

## Manifest (draft, release_id null)

| item | value |
|---|---|
| `C8_probe_manifest.json` file sha256 | c7333a7394d0253410b3761f6c33ea57562bd9506e59fc6c34298e4831302af1 |
| body `request_sha256` | 5eab5d21a6ed256fed7c30afaebb1fb45e040ee83908f91b54eecf267ea099d4 |
| `C8_probe_pieces.jsonl.gz` sha256 | f3c17f46ad1823961337ffedecf5008e64a1ed496624da0aa90ec573ea5bf8d5 |
| `C8_probe_selection.json` sha256 | fed170944899200e42c806c4498d69fcec5be12e898098e2dc365edfb0daf69d |
| pieces | 24 distinct, records byte-identical to the released full pieces file (9f22cb26…) |
| cells | 3.3780e8 geodesic-estimate cells (Σ lattice box 3.7636e8) |
| limits | concurrency 8, batch 1, calls ≤ 50 (init + projection + 24 × 2), deadline 120 s, wall 600 s, outer 900 s, max_consecutive_failures 2, failure fraction 0.25 after 4, per-piece band-px guard ≤ 2.2704e9 |

Selection (`build_c8_sample.py`): deterministic and outcome-free.
- Pool: only plan positions ≥ 28,900 (the second half of the plan), far beyond what full invocation 1 can reach, with
  no S1/S2/S3 piece.
- 6 pieces per candidate-count group {1}, {2}, {3–4}, {5+}, at within-group cell quantiles .50/.65/.80/.90/.97/1.00.
- The result spans 6.2e6–1.6e7 cells, 1–8 candidates and 5–64 vertices; 15 of the 24 are compute-heavy windows of about
  1.6e7 cells.

The full plan, body and pieces are untouched; this is a separate sample manifest (`sample: true`).

## Comparable c4 measurements (unmatched; for context only)

- **S2 (16 pieces, c4):** 1.294e8 cells in 75.1 s, i.e. 1.72e6 cells/s; parallelism Σ piece-seconds / wall = 3.5.
- **Full invocation 1 early prefix (c4, first 30 pieces, plan admins 1–5):** 1.95e8 cells in 135 s including init, i.e.
  1.45e6 cells/s; parallelism 3.7.
- **Expectations for this c8 probe:**
  - at c4-like throughput, 3.378e8 cells would take about 195–235 s;
  - ideal c8 doubling would be about 100–120 s;
  - the 600 s wall leaves room for no speedup at all.

**Limits on interpretation:**
- The probe pieces differ from the S2 and full-prefix pieces (heavier on average, more candidates), so cells/s is not
  matched.
- They have never been requested before, so there is no result cache. Shared asset or tile caching and service load
  vary with time.
- One run gives one draw of 24 pieces: the outcome is a feasibility and throughput indication, not an unbiased speedup
  estimate.
- **Overlap.** If the probe runs while full invocation 1 (c4) is active, the project would carry up to 12 concurrent
  requests. That confounds both runs and is not the c8-alone condition. I recommend running it between full
  invocations (after invocation 1's wall stop, before invocation 2) or once the supervisor coordinates otherwise.

**What to report after the run:**
- wall time, Σ piece seconds and effective parallelism;
- cells/s compared with the two c4 figures above;
- per-piece seconds against the S1/S2 rate model, i.e. whether c8 slows individual requests;
- any queueing, rate or quota errors;
- the same numerical invariants as `gate_check.py`.

## Commands (after binding; fresh directory outside Dropbox)

```bash
R=~/ipcch-alphaearth-runs/p1-c8-probe-<date>; mkdir -p $R/inputs
cp <released C8_probe_manifest.json> <C8_probe_pieces.jsonl.gz> $R/inputs/      # body unchanged, release_id set
cd IPCCHAlphaEarthExperiment
timeout 900 /opt/earthengine/bin/python scripts/ee_pilot_c.py --request $R/inputs/C8_probe_manifest.json --out $R/out
python3 scripts/check_c_outputs.py --request $R/inputs/C8_probe_manifest.json --out $R/out --json $R/check.json
```
