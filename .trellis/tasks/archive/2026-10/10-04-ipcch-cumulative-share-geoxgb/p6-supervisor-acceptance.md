# P6 actual-artifact supervisory acceptance

Scientific run: p6-formal-20261004b. Reviewed evidence HEAD:
500a838846f314989f262a84557921f6e37701e9. Implementation remains
6798df21ea87d4916c7f36fa6e0753c32bb3ef98.

This is the bounded final output verification agreed with the user, not a new
whole-code review or the controller's independent close-audit result.

## Verified

- Independent metric scout used canonical Windows Python and independent
  NumPy/pandas formulas: 2912 comparisons, tolerance 1e-12, zero differences.
  Complete main/supplementary panels, both cohorts, confusion/per-class values,
  projected/raw q3 R2, coverage and deltas agree with the saved report.
- Independent cohort scout verified all prediction keys against prepared keys
  and calendar, with zero missing/extra/duplicate keys, and independently
  reconstructed latest valid persistence from the 42695-row valid ledger.
  All persistence dates <= origin; all target-origin differences equal H.
- All eight country-bootstrap contrasts reproduced with seed 42, 2000 paired
  country draws each. All 16000 draws defined; multiplicities and both arm F1s
  agree, delta rounding differences <1e-16, and all CI endpoints match exactly.
- All 4900 UBJ hashes match all 1225 model records. Identity digests match
  directories; store identities equal unique Stage1+Stage3 fits. Stage1 429
  fits; Stage3 796 fits / 3130 hits / zero failures. Actual 3184 Stage3 scalar
  fits lie within the checkpoint 3136..3388 bound. All exact request counts
  match. Run b has no tmp/incomplete remnants; failed run a remains preserved.
- Frozen package Git tree unchanged (3ed52ea85555acabc118ef4f66a6710eb445a041).
  Report sha256 142d717dedaf8afb573b544743cc6856c0d33d18e1b79a82543bb2dc514691f0.
  Replay sha256 1195716881b8509c53efd47b1d22920b6cb5c6d3952185f9ce316953682e30e0:
  saved replay reports 91880 passed checks, zero failures.

The independent checks used prepared country metadata as authority; they did
not independently rebuild raw-source country lookup or rerun model replay.
Saved replay, prior checkpoint checks, and numerical recomputation are distinct
evidence sources. No scientific superiority claim is an acceptance condition.

## Scientific reading

Main GeoXGB crisis F1 by H1/3/6/12 is .777697/.775626/.769944/.756106.
Geo-minus-pooled F1 is -.000322/-.000298/+.000090/-.000270; all CIs include zero.
Paired Geo-minus-persistence F1 is +.005026/+.007434/+.002674/+.007521;
all CIs include zero. Local routing accounts for only 3.39/3.43/3.10/0.79%
of main rows. The spatial layer shows no demonstrated benefit over pooled.
Against persistence, recall/F2 and continuous q3 improve, while precision,
binary accuracy and four-class macro F1 fall. Do not describe general superiority.
2026 remains a separate point-estimate supplement; no retuning is authorized.

## Finite evidence completion and close release

The existing inventory is a pre-replay snapshot: report lists 25 files /
1109403 bytes, while final report has 26 files / 1120905 bytes. The difference
is exactly replay-independent.json (11502 bytes). This is evidence bookkeeping,
not a model/recipe failure. Preserve the original inventory; add a labeled
post-replay final inventory with run-relative path, size and SHA256 for actual
run files. Save a byte-identical replay JSON copy in task evidence and verify its
hash. Keep these manifests outside the scientific run to avoid self-reference.
Append a brief correction reference to P6-results.md and copy this note into
the task. Do not change scientific files, code, fit outputs, or old failed run.

After these finite evidence actions are verified and committed, the supervisor
accepts P6 execution/results and releases the bound Claude executor to run
trellis-audit close for ipcch-cumulative-share-geoxgb using the exact registered
lowercase repository path. No new execution permission is required. Verify
live session d148c921-36bd-4b42-9ff9-a4f16979e6b5, terminal term_65d044bdb69412,
run 7ced754ea36c48c0a6d24ba2a17addec and base 6c98f73c34272101ddc124cf20ed5ef338563646
before close. Do not reset/rebind/archive manually. Report commit, final
inventory hash, close result and queued job ID. Controller is currently running.
Queued/launched is not accepted. Preserve independent audit findings and let
the controller's normal workflow govern acceptance; no gate waiver is granted.
