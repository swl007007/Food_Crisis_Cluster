# Cancel-attempt accounting repair (narrow delta review of 7a4ad15; offline)

**Defect, reproduced by the supervisor.** `cancel_own` added a task to its cancel set only after a successful
`cancelTask`. With both cancels raising a read timeout, 13 cancel calls were made despite `max_cancels`=2, until the EE
call cap ran out.

**Fix (`scripts/ee_batch_c.py` only; plus one state name in the validator):**
- **Reserve before the call.** Each cancel is reserved before it is made: the id goes into `attempted`, and a durable
  `CANCEL_ATTEMPT` ledger record is written. `max_cancels` and the cancel reserve count attempts, not successes.
- **Distinct outcomes.** Success is recorded as `CANCEL_REQUESTED`, failure as `CANCEL_FAILED`.
- **Never reissued.** A failed or ambiguous cancel is never reissued automatically, including after a monitor restart:
  `attempted` is reloaded from the ledger.
- **The other task is still attempted** within the cap.
- **Bounded stop.** When only unresolved tasks (attempted but not successfully requested) remain nonterminal, polling
  stops and a `HOLD_HANDOFF` is recorded listing `cancel_attempted`, `cancel_requested` and `cancel_unresolved`.
- **Validator.** It also treats `CANCEL_ATTEMPT` as not-a-success.

**Regression tests** (`CancelFailure`, 2 tests):
- T0's cancel fails and T1's succeeds: each `cancelTask` is called exactly once; attempted [T0, T1], requested [T1],
  unresolved [T0]; ledger 2 ATTEMPT / 1 REQUESTED / 1 FAILED, then `HOLD_HANDOFF`; calls ≤ 84. A monitor restart
  issues no new cancel.
- Both cancels fail: exactly 2 calls in total (the reproduced edge made 13), then a handoff.

**Results:** test_speed_pilot 17 OK; focused 67 OK.

**Not changed:** the manifest (body f66a1be8…), the plan, the selection, the caps, the science.

**Sources:** ee_batch_c 8ebe656f…, validate_speed_pilot b2dac368…, test_speed_pilot 8c3e4939…, build_speed_pilot
1af042e9… (unchanged).
