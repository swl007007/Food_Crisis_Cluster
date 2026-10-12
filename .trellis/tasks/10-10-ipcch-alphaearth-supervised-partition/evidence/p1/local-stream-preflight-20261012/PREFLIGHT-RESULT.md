# Local streaming D30 preflight result: NOT VIABLE (pre-registered parity gate failed) → EE-only

**Scope and spend:**
- Stage 1 only: admin 3447 and one pinned COG, `gs://alphaearth_foundations/.../2017/38N/x7kfwxt1zhh4tk7kr-0000008192-0000000000.tiff`
  (generation 1763494271186515, 1,324,750,322 bytes, crc32c bcLC7Q== verified).
- Downloaded bytes: 1.325 GB of the 30 GB cap. The bucket is requester-pays, so that egress is billed to
  manuel-remote-sensing.
- Stages 2 and 3 were NOT downloaded (stop rule).
- No EE call was made, the live EE invocation 8 was untouched, and nothing was injected into it.

**Protocol:** `receipt/PREREGISTRATION.md`, written before computing.
- Selection reference: C02 (3447 whole). Held-out references: the six C03 windows.
- Gates: A ≤ 1e-7; S/Q ≤ 1e-6; descriptors ≤ 1e-6; coverage ≤ 1e-7.

**Result:** `out/parity_stage1.json`.

| reference | A_rel (exact / round255 / floor255 / ceil255) | centre count local vs EE |
|---|---|---|
| C02 3447 whole (selection) | 1.62e-5 / 1.63e-5 / 1.68e-5 / 4.89e-5 | 512,366 vs 512,389 (−23) |
| C03 windows (held out) | 7.4e-6 … 3.2e-5 (exact) | −14 … +16 |

- C02 under the selected variant (exact): A_rel 1.62e-5 (local − EE = +829 m²), S_rel 1.15e-4, Q_rel 4.75e-5,
  |Δmean| 1.78e-6, |ΔSD| 2.25e-6, coverage 1.62e-5.
- Every reference fails. The selection gap is about 160× the A gate.

**Diagnosis:**
- A post-hoc diagnostic (`out/posthoc_densify_diagnostic.json`, not a pass criterion) compared no densification,
  1e-4° densification and 1e-6° densification. The count gap stayed at −22/−23 and A_rel at about 1.6e-5, so polygon
  edge representation is not the cause.
- The residual is EE-internal region rasterization, boundary weighting and/or masking.
- EE documents its overlap weights only as 0..255 levels (256), not a full rasterization/rounding algorithm. That
  algorithm cannot be reproduced locally to 1e-7 from documented information, and the gates are not relaxed.
- What the scales mean:
  - A_rel 1.6e-5 corresponds to about 8 cells of area on 512k cells.
  - S_rel is larger on bands whose S is near zero; per-band absolute differences are recorded.
  - Mixing local and EE pieces would introduce estimator inconsistency at the ~1e-5 level, well above EE's own
    whole-vs-window decomposition difference (about 1e-8).

**Speed (single worker, stage 1; indicative only):**
- Download: 41.5 s for 1.32 GB (about 32 MB/s), whole-file.
- Local reduce: 2.2 s for C02 (512k cells): 0.22 s weights, the rest values and sums. Peak RSS 780 MB.
- End to end: about 1.2e4 cells/s, download-dominated, because a whole COG is needed for a small admin.
- No warm-cache or parallel run was attempted after the gate failure.
- Scaling cost: egress at the study's 7.22 TB distinct volume (55.5 TB without reuse) would be large on a
  requester-pays bucket.

**Conclusion:**
- The local route does not reproduce D30 at the fixed tolerances, so the hybrid is not released.
- Continue the main EE task only, under its existing releases and gates.
- The downloaded COG is retained outside Dropbox for audit (1.3 GB). It can be deleted on request.
