# Round-2 repair log

Findings: repair close-audit 026f8908 (A01-A05, evidence gap) and spot re-audit
175302c2 of the original task (its A01 = our A03, its A02 = our A01).

- A01 run/code binding: run v3 was produced before verify_fourclass.py was edited, so its
  recorded code identity is not the committed code. Now preparation refuses to start
  unless the working-tree package code equals the committed HEAD (`code_identity_at`
  hashes Git blobs with the same algorithm) and records `git_head` and
  `code_equals_git_head`; the verifier checks run identity == HEAD blobs == working
  tree. The authoritative run is fresh (v5), started after all code was committed.
  v3/v4 are superseded and not committed; v4 holds only an aborted preparation from a
  killed start.
- A02 digests: each Stage 3 estimator records the SHA-256 of its OWN ordered fitting
  keys plus the pool digest; the verifier recomputes local digests from
  training_keys.csv.gz and the cluster map, and row counts.
- A03 schema: `FEWSNETFourClassBaseline/feature-schema.json` (approved hash) and all
  Trellis `task.json` records are now committed (root `*.json` ignore exempted).
  `.gitattributes` marks the package `-text` so a Windows checkout is byte-identical
  (the clean-clone check found CRLF conversion changed the schema hash). A clean
  Windows clone passes all 37 tests with code identity equal to HEAD.
- A04 inventories (whole class): required-output lists for preparation, Stage 1 folds
  (fold evidence, Stage 2 handoff, requested checkpoints) and Stage 3 folds (pooled
  and every local model); acceptance checks the record's own identity fields
  (fold/horizon/month), the run's retention plan, that every required file is recorded,
  and that every recorded file matches. Stage 2 consensus has a required list too.
  Tests: empty inventory, dropped required file, wrong fold, retention mismatch,
  missing inventory, prepared record without required outputs.
- A05 exactness: prediction is single-threaded (`fourclass.deterministic_proba`);
  fitting keeps its parallelism (forests are identical for any n_jobs). Replay now
  requires bit-identical probabilities; tolerance removed.
- `prepare_fourclass.py` imports from a relocated checkout (default source root only
  when the canonical layout exists).
- Run v5 (bound to 4ad1109) failed the exact replay by 1.1e-16. Cause: the verifier read
  predictions.csv.gz with pandas' default fast float parser, which is not exact for
  %.17g values; with float_precision="round_trip" the loaded bundles reproduce all
  probabilities bit for bit. Because this was a verifier-only fix, the verifier is now
  identified separately (verifier_identity) and excluded from the producer identity
  that binds a run, so checking-code fixes cannot misattribute a run again. That
  changed run_identity.py (producer code), so a fresh run v6 was produced after
  committing; v5 is superseded and not committed.
