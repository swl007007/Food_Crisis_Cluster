# Repair log — four-class audit fa38f19ad8fafc36567f90ac

- A01: `compare_partitioned_vs_pooled_rf_k40_nc4.py` saves every pooled/local estimator
  as `models/<name>.pkl.xz` (forest, own imputer, feature order, fit identity incl.
  training-key digest); `fold.json` is written last with SHA-256 of every fold output;
  `run_manifest.json` binds code, runtime, predictions and fold records.
  `verify_fourclass.py` now LOADS those bundles for first/last fitted fold per horizon
  (no refit). Labels are identical; probabilities differ by at most 1.1e-16 because
  sklearn sums per-tree probabilities across threads in nondeterministic order, so the
  check uses a 1e-12 tolerance and records the maximum difference.
- A02 (whole class, all stages): nothing continues into existing output. Preparation
  writes `identity.json` last (code, runtime, outputs digest); Stage 1 refuses any
  existing fold/handoff/retained directory, verifies the preparation, and writes
  `completion.json` last with identities, retention request and hashes of every fold,
  handoff and retained file; Stage 2 accepts folds only through `verify_fold`, and its
  `consensus.json` (written last) binds its outputs; Stage 3 verifies the preparation
  and the consensus record; the report verifies every Stage 3 record.
  `feature-schema.json` is now a package file (hash equal to the approved schema).
- Tests: 34 pass (new: forged marker, changed prepared input, changed code identity,
  changed handoff, missing retained checkpoint, refusal of complete folds, bundle
  round-trip reproducing probabilities).
- Fresh run `fourclass-v3-20260928`: report tables identical to v2; verification 34/34
  including saved-model Stage 3 replay (6 folds, 14 bundles each). Stage 3 bundles total
  ~460 MB and are committed.
