# P2/P3 supervisor review corrections (review pinned at 791b4fd; fixes after 97f5c85)

Source: `/tmp/ipcch-geoxgb-p2p3-supervisor-review-20261004.md`.

Classification: all five items are evidence/provenance/integrity gaps; none
changes a number produced by the frozen recipe. The numeric notes affect only
magnitudes beyond any observed or float32-reachable model output. Item 1 is a
second-round finding in the same class as the first P2 review's fit-record item
(see the report to the supervisor/user for the continue/stop question).

1. **Cache record integrity** (`modelstore.validate_fit_records`, run on write
   and on every load): identity must bind `params`, `rounds`, `n_rows` and
   per-target `y_sha256` (Stage1/Stage3 producers now supply `n_rows` and
   `target_digests(Y)`); each target record must carry all required fields with
   correct types; `params` must equal the identity's requested params; rounds
   (global total / local appended) equal identity rounds; `rows` equals
   `n_rows`; weights `unit`; `y_sha256` equals the identity digest; constant
   fields consistent; the fit-time `resolved_config` must match its recorded
   `resolved_config_sha256` (new, captured at fit time) and its actual tree /
   gbtree / generic / train fields (max_depth, eta, min_child_weight,
   reg_lambda, reg_alpha, subsample, colsample_bytree, gamma, max_delta_step,
   max_bin, grow_policy, tree_method, num_parallel_tree, seed, nthread, device,
   booster, objective) must equal the requested params (float32-exact); never
   compared with a UBJ reload (which reports max_depth 6, eta .3). Any
   violation → TechnicalError, fit not invoked.
2. **Rejected-child evidence**: Stage1 decisions record `child_ids` and full
   `child_members` for every split attempt; every fit request in the Stage1
   ledger records `node_id`, `parent_node_id`, `members` and the exact
   `prepared_rows` (positions in keys_hNN / X_rich561_hNN) plus the bound
   keys/X artifact digests (root requests record the F rows).
3. **Durable R41 record**: `artifacts.record_incomplete` writes
   `<stage>/INCOMPLETE.json` and `RUN_INCOMPLETE.json` (stage, context, error
   type/message, notes, traceback, retained partial files) for prepare,
   learn-map, predict and report; the store ledgers failed requests with their
   use context; quartet errors are annotated with the q target, Stage1 errors
   with the node, Stage3 errors with fold/gate month/region; the CLI refuses any
   further stage on a run marked incomplete.
4. **Prepared inventory**: `verify_prepared(prepared, horizons)` requires the
   exact contract inventory (6 fixed files + X/keys per H); omitted or
   unexpected entries stop before any read; used by learn-map, predict and
   replay.
5. **Component sizes**: connectivity diagnostics include `component_sizes`
   (descending) in the frozen record.

Numeric notes: `array_digest` refuses any dtype with `hasobject` (structured
object fields); projection block means fall back to `fsum(v/n)` when the sum
overflows (ordinary values bit-unchanged; `[1e308, 1.5e308, .9, .1]` →
`[1, 1, .9, .1]`); R² returns NA ("not representable in float64") when a
non-constant SST underflows or SST/SSE overflow; docstring claim corrected.

Evidence: `evidence/P2P3-review-pytest.log` — **244 passed** (26 new in
`tests/test_review_p2p3.py`: 11 cached-record tampers incl. redigested
resolved-config edits, clean-hit control, 4 missing identity fields,
rejected-child member/row resolution, full component sizes, Stage1 failure
record + failed ledger entry + CLI refusal, Stage3 failure notes/failed request,
predict failure fold context, inventory omission/extra, structured-object
digest, overflow projection, R² range NA). Supervisor probe re-run on the
corrected code (only `n_rows`/`y_sha256` added to its identity, damage cases
unchanged): `evidence/P2P3-review-supervisor-probe.log` — every damage case
TechnicalError, clean global/local entries still hit, local-as-global refused.
