# 24 h feasibility speed pilot: offline proposal (NOT submitted; no cloud write authorized) — 2026-10-11

User priority: "关键还是看看能不能加速，3天实在有点多了，看看能不能缩短到一天" (the key is whether it can be sped up; 3 days
is too long; try to bring it down to one day). Science stays 10 m exhaustive D30. This updates only the operational
proposal; no source code is implemented.

## 1. Target, bound to the actual remaining workload

Offline baseline (`ipcch-alphaearth-24h-feasibility-baseline-20261011.json`, supervisor), matching
`speed-remaining-workload.txt`. Cells are geodesic-area estimates, not billed EECU.

- **Reusable:**
  - 774 complete admins = 6.8242719e10 cells (14.609%; 8,268 windows).
  - The 1 partial admin's 12 windows (0.017%) are preserved but not reused; that admin is redone whole.
  - All 8,280 windows together are 14.626%.
- **Remaining:** 5,453 admins = **3.9887004e11 cells**.
- **Required end-to-end rate:**
  - **4.617e6 cells/s over 24 h**, or **5.036e6 cells/s over 22 h** of compute, leaving 2 h for queue tail, download and
    validation;
  - about 2.9× / 3.2× the observed interactive c4 rate (1.568e6 cells/s, invocations 1–3).

## 2. Design corrections (supervisor review, applied)

- **Reuse by default.** The 774 complete admins are reused with explicit bindings:
  - old manifest body 63d0c7ab…, plan bccf45c2…;
  - per-window response sha256 from the old ledger and outputs.

  The final plan is the union of the old-complete subset (its window rows unchanged; no rebinding or relabelling) and
  the new adaptive subset for the 5,453 remaining admins. Each admin has its inputs from exactly one source. The partial
  admin's 12 windows stay preserved but unused.
- **Task-count arithmetic.**
  - 120.6 MB of inline geometry / 10 MB per query gives a lower bound of about 13 tasks; about 25 tasks follows only
    from a 5 MB packing choice.
  - Side 32,000 (6,469 items) has about 25% fewer items than side 16,000 (8,636), not half.
  - No task count is real until actual split geometry and the serialized DAG are measured.
- **maxPixels.** Use one constant per candidate group: 129 × the group's maximum lattice box × 1.1. This is the known
  API usage; no per-feature property.
- **Submission guard.** The serialized DAG/body size is measured after `ee.Initialize` and before each `start()`; a
  task whose serialized body exceeds 8 MB is refused.
- **Submission ambiguity.**
  1. Write a ledger intent record (deterministic description = batch_id/attempt) BEFORE `task.start()`.
  2. Record the task id after the call.
  3. If `start()` raises or times out: HOLD and look up `ee.data.getTaskList()` by description. Never resubmit
     automatically, never duplicate. No framework.

## 3. Speed pilot: two balanced batch tasks (the documented default of ≤ 2 READY/RUNNING), not more client reduceRegion concurrency

**Fresh work:**
- 95 items systematically sampled from the cell-size distribution of the remaining items.
- Excluded: every admin already touched (old windows, interactive pilots 658/4711/1050, the batch-design example 4505).
- Alternated in size order into two balanced tasks (`speed-pilot-items.csv`):
  - **A:** 48 items (31 whole, 17 pieces), 2.567e9 cells; largest item 2.55e8 (at the split bound).
  - **B:** 47 items (23 whole, 24 pieces), 2.422e9 cells.
  - Together 4.99e9 cells = 1.25% of remaining. Item cells q25 9.4e6 / median 3.1e7 / q75 7.8e7 / max 2.55e8.
  - Distinct candidate unions: 46 (A) and 47 (B).

**Small non-fresh rows:** 6 admins, 3.47e8 cells, reported separately from fresh throughput:
- interactive references 658 (A) and 4711 (B);
- reuse validation using old complete admins at the q25/q50/q75/q90 sizes: 15 (2.56e7 cells, 5 windows) and 117
  (4.77e7, 9) in A; 349 (7.75e7, 12) and 229 (1.36e8, 17) in B.

**Per item:** the D30 sums-only reduceRegion on the existing composite. The construction is as in
BATCH-DESIGN-PROPOSAL §3, except that maxPixels is the group constant.

**Destination:** `gs://manuel-remote-sensing-rasters-645415198431/ipcch-alphaearth/speed-pilot-<date>/{A,B}/attempt-1/`.

**Calls:**
- EE: initialize; projection check; 2 × `start()`; at most 250 status calls (1 per minute); `cancel` for both tasks if
  not COMPLETED by 4 h after the first submission.
- GCS: at most 20 list/download reads.
- No client resubmission. EE's own server-side retries (up to 5) are recorded, not controlled.

**What is measured (task-status timestamps plus local clock):**
- per task: submission → READY → RUNNING (`start_timestamp_ms`) → COMPLETED (`update_timestamp_ms`), giving queue time
  and execution time;
- the concurrent interval of A and B;
- download time and validation time (local);
- fresh cells per wall-second, end to end (first submission → last validated row), plus per-task execution cells/s;
- any EECU field reported, but no speed claim from EECU alone.

## 4. Success gates (numerical tolerances fixed in advance)

- **Integrity.**
  - Both tasks COMPLETED.
  - Every object under the prefix is listed and sha256-recorded.
  - Rows equal the planned item IDs exactly (per task, exactly once).
  - Every value of the 129 is finite (nulls are technical failures).
  - Recipe column correct.
- **References 658 / 4711** (same graph as the interactive whole-admin requests; tests CSV precision and the server
  path):
  - |A, S, Q| relative difference ≤ **1e-9** each;
  - descriptors |Δmean|, |ΔSD| ≤ **1e-9**;
  - coverage |Δ| ≤ 1e-9, with identical route.
- **Reuse validation 15 / 117 / 349 / 229** (whole-admin batch against the sum of old windows; precedent: pilot2
  windows vs whole gave A 7.7e-9, S 2.2e-7, Q 3.2e-8; REF 658 gave ≤ 4.4e-14):
  - A relative difference ≤ **1e-7**;
  - S, Q relative difference ≤ **1e-6**;
  - descriptors |Δmean|, |ΔSD| ≤ **1e-6**;
  - coverage |Δ| ≤ 1e-7, with identical route.

  Failing this gate puts reuse of the 774 back into discussion.
- **Throughput** (decides only whether bulk is discussed as a one-day run):

  | projected bulk wall (see §5) | what follows |
  |---|---|
  | ≤ 22 h at the SLOWER task's rate | propose a bulk envelope for discussion |
  | 22–48 h | discuss options |
  | > 48 h | discuss; the one-day goal is not supported |
  | a task not complete in 4 h | cancel; the measured rate is < 3.5e5 cells/s per task, far below target |

## 5. Projection model (explicit, with uncertainty)

Bulk wall ≈ queue_start + (N_tasks / 2) × per-task fixed overhead + R_remaining / (2 × r_task) + download/validation.

- **Measured inputs:** per-task fixed overhead (queue plus startup); per-task execution rate r_task, taken as the slower
  of A and B; and download/validation seconds per row.
- **Uncertainty reported:**
  - range [slower, faster task];
  - only 2 tasks, from one time of day and one geographic draw;
  - whether more or larger tasks get more workers is unknown (documented as server-managed);
  - queue variance under EE load;
  - largest-item risk: one item of at most 2.56e8 cells bounds the tail of its task; the pilot includes a 2.55e8-cell
    item.
- **Interpretation.** Two balanced tasks of about 2.5e9 cells measure the documented concurrency (2 tasks) directly. They
  do not prove bulk throughput, which also depends on queueing over about 100+ tasks.

## 6. Before any bulk

If the projection still forecasts multiple days, discuss before any bulk. Options that do not change science:
- larger tasks, if per-task overhead dominates;
- a higher tier batch concurrency (a quota/plan change, user/admin decision; not requested here).

Bulk itself would need a separate cloud-write authorization with its own envelope.
