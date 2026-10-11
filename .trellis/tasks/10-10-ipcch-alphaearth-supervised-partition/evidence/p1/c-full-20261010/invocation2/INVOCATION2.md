# Full invocation 2 (continuation receipt codex-D30-c4-guard-continuation-20261011)

**Timeline:**
- Launched 2026-10-11T06:16:50Z on the same manifest, run and output, at c4/b1.
- Run summary written 07:27:57Z; wrapper `exit 0` at 07:28:14Z; checker `exit 0` at 07:28:18Z.
- The invocation ended early: `stopped` = "'378|w001_014' failed twice: Earth Engine capacity exceeded."

**Counts:**
- Attempted 699, failed 1, done 698, halted 0; invocation 4,263.3 s.
- Cumulative: 2 starts / 20; 4,513 reservations / 59,576; 4,505 ok outputs; 1 failed piece (no output, pending);
  4 request errors; checker 0 violations.
- No reservation started after the failing attempt: the max request_log n is 4513. The 3 later successes were already
  in flight.

**Corrections to my live reports:**
- A live PID and in-flight successes were NOT evidence of continuation. The invocation had published its stop at the
  failure.
- The 1,735 re-request figure is not an allowance. Only the 59,576 cumulative reservations and the 20 starts bind.

**Cause** (execution-guard defect; no numerical or recipe effect):
- `ee_pilot_c.py` `run_production` matched `"cap" in msg`, which also matches "capacity".
- The same handler did not stop on the released non-retryable request-size rejection.
- Repaired in the commit that adds this file (see `../stop-prefix-repair/`).
