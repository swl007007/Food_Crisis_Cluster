# Repair of the speed-pilot path after the narrow review of 8b00a88 (offline; no cloud call)

**Classification.**
- Items 1 and 2 could change a reported result: the validator accepted unbound or tampered inputs and CANCELLED tasks,
  and its clock stopped too early.
- Items 3–5 are operational safety and accounting.

Each class is fixed once below.

1. **Validator binding.** `validate_speed_pilot.main` now:
   - re-verifies the manifest with `ee_batch_c.load_released` (body sha256, release id, D30 pins);
   - requires `out/MANIFEST.sha256` = the body sha;
   - requires the plan bytes to equal `adaptive_plan_sha256`;
   - takes the remaining workload from the new manifest field `remaining_workload` (5,453 admins / 7,996 items /
     3.98874e11 cells, plan-bound). The `--remaining-cells` CLI argument is removed.

   Tested: plan tamper leads to FAIL with exit 1; a body tamper (fresh cells ×100 without a hash update) is refused.
2. **Task states and clock.**
   - Every expected task must be SUBMITTED with a known id and end COMPLETED, with 0 < creation ≤ start ≤ update, no
     unknown state or id, and no AMBIGUOUS / HOLD_HANDOFF / CANCEL record.
   - Each downloaded object must match its record and lie under its task prefix.
   - `validated = time.time()` now runs after every check and gate.
   - Exit 1 on an integrity or gate failure; the report is kept.

   Tested: CANCELLED, non-monotonic timestamps and a failed reference gate each give exit 1. NaN-vs-NaN
   (below-coverage) descriptors compare equal; NaN-vs-number counts as infinite.
3. **Timeouts and cancellation.**
   - Every EE client calls `Initialize(project, http_transport=httplib2.Http(timeout=http_timeout_s=60))`, then
     `setDeadline`.
   - GCS uses `google_auth_httplib2.AuthorizedHttp(creds, http=httplib2.Http(timeout=60))` for discovery and every
     read.
   - SDK evidence (1.7.36): the default deadline 0 means `timeout=None`; `Initialize` accepts `http_transport`;
     `build_cloud_resource` wraps it in `AuthorizedHttp`.
   - Before every status round the monitor:
     - cancels own nonterminal tasks once the 3,600 s cap is reached (UNKNOWN counts as nonterminal);
     - if the remaining reads or calls cannot cover another round plus the reserved cancels, cancels now and records a
       HOLD_HANDOFF;
     - after 3 consecutive failed rounds, cancels now and records a HOLD_HANDOFF.
   - With a partial or AMBIGUOUS submission, the known started ids are still monitored and cancelled at the cap; the
     ambiguous one stays HOLD for a manual check.
   - The pagination-unbounded `lookup` is removed.

   Tested: partial start, the cap, the cancel reserve under a tight budget, failing rounds, and UNKNOWN state.
4. **Status-read accounting.**
   - `getTaskStatus([A, B])` loops one HTTP GET per id (`data.py:1656`), so reads are now per id: one counted call =
     one GET.
   - Interval 120 s, ≤ 75 reads (about 37 rounds, about 74 min), ≤ 84 EE calls, with the cancel reserve kept. Init is
     labelled "may issue >1 HTTP request".
5. **Download resume.**
   - Each object goes to `*.part`, then a record (task, object, generation, size, sha256) is appended to
     `downloads.jsonl`, then the file is renamed. Reads are counted cumulatively in `gcs_reads.log`.
   - A rerun verifies records and bytes and skips them without re-reading; tampered or unrecorded files are refused,
     never overwritten.

   Tested: a cap failure after the first object, resume, cumulative counting, and tamper refusal.

**Minor fixes:**
- the loader enforces `max_tasks` (2) and all required limits;
- the size guard is labelled honestly as the serialized collection DAG (the export request adds a few hundred bytes;
  the 8 MB cap leaves headroom to the 10 MB limit).

**Hash delta.**
- The adaptive plan (ae2785fe…), the A/B task item lists and the reference pins are byte-identical.
- Manifest body 7b8145b6… is superseded by **f66a1be8854ef555a1fdc2abc7533da5fa51cfdf31c5c49b1ac5632bf6a57f3c**
  (file 13705eba…). Only the limits changed (`http_timeout_s` 60, `status_read_interval_s` 60 → 120,
  `max_failed_rounds` 3) and `remaining_workload` was added.
- Sources: ee_batch_c 99a05545…, build_speed_pilot 1af042e9…, validate_speed_pilot 140d8e36…, test_speed_pilot
  aa1df6c6…

**Tests:** test_speed_pilot 15 OK; focused 65 OK.
