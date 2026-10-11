# Batch speed pilot result (release codex-D30-batch-speed-pilot-20261011): FAIL

**Submission:** one submit at 14:00:35–14:00:43Z. Both collection DAGs were measured before any start: A 1,608,003
bytes, B 1,324,521 bytes.

| task | id | queue (create→start) | running | outcome | batch_eecu_usage_seconds |
|---|---|---|---|---|---|
| A | PQDBYWUMO4D3GCK7ZR3JFK4M | 3.14 s | 1,650.75 s | FAILED, "Computation timed out.", attempt 1 | 512,434.8 |
| B | OM27XES4LLVSWUHW63SONSSP | 3.21 s | 1,247.06 s | FAILED, "Computation timed out.", attempt 1 | 171,940.7 |

- The RUNNING intervals overlapped for 1,247.06 s.
- The monitor exited normally: 27 per-id reads, 32/84 EE calls, no cancel, no handoff.
- No GCS list or download was made, and no validation was run (the gate requires both tasks COMPLETED). Whether any
  partial objects exist is unchecked.
- This is a server task timeout, not the 120 s client deadline.
- No pixel throughput is inferred from EECU, and no claim is made about batch designs in general.

**Supervisor receipts** (retained here): B-failed check, terminal check.

**User steering afterwards:** "我记得 ~/projects/manuel-remote-sensing 做过类似的全量EVI多时点提取，但那个是只有墨西哥的多时序，
可以参考一下。我们主要是全球的。实在不行的话就继续原来的流程。" (A similar full EVI multi-date extraction was done in
~/projects/manuel-remote-sensing, but only for Mexico; use it as a reference; ours is global; if nothing else works,
continue the original flow.)

**Supervisor read-only comparison:**
- That run was a selected 50.6M-pixel, 30 m, single-band seasonal EVI extraction, not whole Mexico.
- 98 of its 504 large exports failed and were rescued by 3,045 smaller exports (bbox ≤ 4.194M 30 m cells).
- It offers no proven shortcut for our global 64-band, 129-sum graph.

**Decision:** the user-authorized fallback to the original c4 small-window flow, with no further batch pilot.
