# Saved development ledgers versus lawful keys, calendars and masks (implement §3 open item)

2026-10-03, Claude executor. Read-only; no fit, no RUN write, no product/test edit.

**Run.** Probe `research/probes/dev_ledger_reconcile.py` (sha ea3f05da…b6c7), pinned Windows stack, 17 s, over `C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1`. Output: `research/probes/dev_ledger_reconcile_summary.json` (sha f3e21605…8285), with one row per fold.

**Result.** 72/72 folds reconcile, **0 problems**.

Three earlier runs flagged problems that were probe errors, each corrected before this result:
- a missing-file case for folds without a gate;
- requiring *every* identity record sharing a booster hash to match, instead of *any*;
- persistence age measured from T instead of from O.

## What the saved records prove

**Expected values and their limits.** The expected calendars and masks are recomputed from the prepared inputs (committed release ledger and observations) by the package's own definitions: `ReleaseLedger.hidden`, `Availability.gate_dates` / `visible`, and `plan.WINDOW = 59`. Agreement therefore shows that the saved records are consistent with the frozen lawful definitions. It is not independent lineage.

| Check (per fold) | Saved evidence | Result |
|---|---|---|
| **Outer global identity.** Origin = O; strategy; scenario_k = intensity_k = k; `excluded_months` = [] ; `masked_months` = hidden(O, k); the **declared** label window `fit_label_months` = [O−59, O−1]; rows = original_keys (A) or 3 × original_keys (B); weights digest present iff B | `gate.json` global (booster + fit-key sha), matched to `scenario_globals/*.json` | 72/72 (identical boosters can carry 1–6 identity records; at least one matches exactly) |
| **Gate calendar.** Validation months = the latest six lawful label months U < O after the outer mask (`gate_dates`); internal origin = U − H; every U < O and outside hidden(O, k) | `gate_pairs.csv.gz` | 66/66 learned-map folds. The 6 `no_prior_candidates` folds (A and B × k0–2, H8 2019-02) have no gate by design: pooled global, no gate_pairs |
| **Internal gate globals.** Origin = internal origin; same strategy and k; `excluded_months` = hidden(O, k); `masked_months` = hidden(O, k) ∪ hidden(o_int, k); declared window [o_int−59, o_int−1] | gate_pairs `global_sha256` → store records | 396 internal globals across folds, all matched |
| **Locals.** Each enabled local continues the fold's outer global (parent sha) with exactly 20 added rounds | `gate.json` locals | 278 locals, all pass |
| **Prediction keys.** 5,718 unique areas, at T, O, H and k | `predictions.csv.gz` | 72/72 |
| **Matched persistence.** The source month equals the latest lawful label month ≤ O for the area under the outer mask; never in hidden(O, k); age = O − source | predictions persistence columns | 0 mismatches. Persistence is present for 5,370 areas (64 folds) or 5,511 (8 folds) |

**Stage 1 roles.** Already established in `research/stage1-diagnostics.md`: F/S/C/E3 are disjoint on original keys, with 0 within-role duplicates, across all 648 candidates. The input-level keys, masks and weights were checked pre-fit by `reconcile_prepared.py` (eval keys in [O−59, O), no label in the outer hidden cycles, E3 targets labelled at T, per-key variant count and weight sum).

## Independent global fitting-key check (coordinator): PASS

- **Artifacts.** Checker `research/probes/check_scenario_fit_keys.py` (sha eed66e0d68bbeacc9dad984375a562724e917bbb3aa91e3dc22507a28def695b) and report `research/scenario_fit_keys_review.json` (sha 80577d3f16594b0aeb6568bb6ce1efacf082222f0ff8d02dbdcb6c4019fc4bd3). Both are byte-exact copies of the coordinator's /tmp files.
- **Scope.** All **373/373** `scenario_globals` records present when the probe started, with **0 problems**.
- **Method.**
  - Lawful observations were reconstructed directly from the prepared observations and release ledger, with explicit filters: country release ≤ origin; label window [O−59, O); outer exclusions plus the record's own-k masks.
  - A/B variants were ordered independently.
  - Every saved fitting-key, label and weight digest matched, as did original-key support, class counts and row counts.
  - No package imports, no features, no fits, no RUN writes.
- **Proves.** The saved global input identities equal independently reconstructed ordered keys, labels and weights.
- **Does not prove:**
  - local fit keys;
  - feature values;
  - booster internals;
  - the formal whole-task audit.
- **Not covered.** Globals written after the probe started, by the still-running scen-historical, are outside these 373. Reconcile them at final completion.

## Independent local and gate reconciliation (executor probe): PASS

- **Artifacts.** Current probe `research/probes/local_gate_reconcile.py` (sha 26d6e2618ec55c5d1bc7da5d1731ead59b830459506593749261e32e30f18f9f). Output `research/probes/local_gate_reconcile_scenario_development.json` (sha 8acfd935d65259b6af665bde7385f6131f3f4aa42f75ede5ce28a0d714974390). The probe exits nonzero on any problem; it exited 0.
- **Accepted earlier checkpoint:** probe 4059c594…77d9, output e24e620e…96ea.
- **Revisions.** The first version (607b1869…) was strengthened after coordinator review with four checks:
  1. every prediction area → cluster against the frozen map, plus the region, local and gate-pair cluster sets;
  2. internal support fields on every row of each block;
  3. the recorded `enabled` flag against the recomputed gate decision, separately from the deployed route;
  4. the B native `sample_weight` blocks.
- **Method.** pandas/numpy only, 31 s, read-only; no package import, no fit, no RUN write. Lawful pools are rebuilt with explicit filters from the prepared observations, the release ledger and each fold's saved frozen map file:
  - release ≤ cutoff;
  - window [O−59, O);
  - outer and own-k masks.

  Only the byte format of the saved digests is reused.
- **Scope.** 72 development folds: 66 learned-map folds; the 6 no-prior folds have no gate or locals by design.

| Evidence class | What was compared | Count | Result |
|---|---|---|---|
| **Outer deployed locals: key-digest proof** | Saved `fit_keys_sha256` against the ordered (area, month, variant) digest of the outer lawful pool ∩ the frozen-map cluster; `fit_support` (original keys); variant rows = 1× (A) or 3× (B). For **B**, the saved native `sample_weight` block (dtype float32, sha256, n, sum, min, max) is checked against repeated float32(1/3) over the reconstructed rows. **A** locals carry no weight block, preserving the no-weight default | 278 locals (142 B weight blocks) | all match |
| **Gate evaluator keys** | Gate months = the latest six lawful label months U < O under the outer mask; per cluster, the evaluator key set = lawful labels at U in the cluster's members; truth codes; internal origin = U − H | 7,160 cluster × month blocks | all match |
| **Internal gate locals: support-only evidence** | Original-key support of the pool at V = U − H (masked by outer + own-k) ∩ cluster, against the saved `local_fit_rows/areas/dates/classes` on **every row** of the block; support decision (`local_fit_ok`, ≥ 500 rows / 50 areas / 6 dates / 2 classes); hash present iff fitted; unsupported blocks routed to the global | 7,160 blocks (6,754 fitted) | all match |
| **Gate decisions and routes** | Exact-fraction crisis-F1 recount from the saved gate pairs; gate support floors (100 / 20 / 3 / 3); undefined metric; strict gain > 1/100; saved gain string; recorded `enabled` against the recomputed gate decision; then, separately, current outer support, deployed route and reason, and local presence. Every prediction area's `cluster_id` is checked against the frozen map (unmapped = −1, routed `unmapped_area_global`). The regions equal the mapped prediction clusters. Deployed locals and gate-pair clusters lie inside them. Per-cluster prediction routes are checked too | 1,194 regions; 377,388 prediction rows | all match |

- **Negative controls.** Both temporary copies and their outputs were deleted.
  - Window set to 58 instead of 59: flags 66/66 gated folds.
  - B weights hashed as float64 instead of float32: flags all 33 B folds with deployed locals.

  So neither the key checks nor the weight check is vacuous.
- **Distinction.**
  - Outer deployed locals carry a **key-digest proof**: the saved digests equal independently reconstructed ordered keys.
  - Internal gate locals have **support-only evidence**. Their per-key fit lists and key digests are not saved, so only their support counts, support decision and routing are reconciled.
  - The gate *predictions* (y_global / y_local_routed) are taken from the saved pairs and recounted. They are not regenerated.


### Extension (F6/F7 follow-up): forecast truth/persistence and independent global-mask links

Same probe; still no package import. All 72 folds, **0 problems**.

| Check | Count | Result |
|---|---|---|
| Forecast truth `y_true_code` = the prepared observation at (area, T), NaN where unlabelled | 411,696 forecast rows | all match |
| Matched persistence: class, source month and age = the latest label ≤ O with release ≤ O outside hidden(O, k), derived from the prepared observations and release ledger with explicit filters; age = O − source | 387,768 persistence rows | all match |
| Outer global: the gate.json booster + fit-key sha links to a saved record whose origin = O, strategy, scenario_k = intensity_k = k, `excluded_months` = [] and `masked_months` = hidden(O, k), all derived independently from the fold origin and k. Any matching record is accepted where identical booster bytes carry several identities | 72 | all linked |
| Internal gate globals: each gate_pairs `global_sha256` at U links to a record with origin = U − H, `excluded_months` = hidden(O, k) and `masked_months` = hidden(O, k) ∪ hidden(U − H, k) | 396 | all linked |

- **Combined with the coordinator's fit-key checker.** That checker proved the record key/label/weight digests against independently reconstructed keys, but it read the inherited exclusion from the record itself. This extension derives those exclusions independently from each fold's origin and k. Together they close that dependence for the development globals.
- **Negative control** (temporary copy, deleted): persistence restricted to months < O, and internal masks without the outer exclusion. Exit 1, with 68 problem folds:
  - 24 k=0 folds fail persistence, where O itself is the latest lawful label;
  - 264 internal globals fail, i.e. every k>0 fold × 6 months.

  For k = 0 the outer exclusion is empty, so that half of the control cannot fire there.
- **Not covered:**
  - internal local per-key provenance (support-only);
  - feature values;
  - booster internals;
  - historical folds, until scen-historical finishes.

## Not proven by saved records (unavailable evidence)

- **Per-key fit lists.** Global and local fit records store digests only (`fit_keys_sha256`, `labels_sha256`, `features_sha256`, `weights_sha256`). `fit_label_months` is the **declared** window [O−59, O−1], written by `GlobalStore.get` (stage3.py:253) from the origin. It is not the empirical minimum/maximum of the fitted label months, so matching it proves only the recorded window definition, not the fitted keys. They do not store the key lists. The checks above therefore prove only the *recorded* masks and the declared windows. They do **not** independently prove that every individual fitting key avoided every masked month. The "neither endpoint masked" sub-check in the probe is vacuous for a declared window and is not evidence. **Global key identity is now verified independently** (section above).
  - For the global fits this gap is closed by the independent reconstruction above.
  - A reconstruction does not necessarily test the same code against itself. It does so only if it reuses the package's availability code. The coordinator's reconstruction uses explicit filters and no package imports.
- **Local fit keys (narrowed).** The outer deployed locals (278) are now key-digest verified (section above). Internal gate locals save no key list or key digest, only support counts and a booster hash, so their fitted-key identity stays unverified beyond support counts and support decisions. Booster internals and feature values are not verified for any model.
- **Historical folds.** scen-historical is still running (launcher PID 1439211). Once it finishes, rerun `dev_ledger_reconcile.py` (adapted to the historical calendar), `local_gate_reconcile.py RUN scenario_historical` and the coordinator's global fit-key checker on all records.
- **Actual folds.** Not run; blocked on the availability input.

## Post-run historical reconciliation (2026-10-03, after scen-historical + scen-report)

**Run.** scen-historical finished in 11,761 s and scen-report in 14 s. Launcher PID 1439211 exited at 20:29. The logs have no traceback or fatal output. `historical.json` sha 67153c534d382ed9f30feb0982d37c680d221d243ad819c1dfd8f2a7d5de9d79; `report.json` sha 4c0efad070a92accd06d65e0ceb514f0f230af0a40c91f55eb81fe78b6bf7b2b.

| Check | Artifact (sha256) | Result |
|---|---|---|
| **Identity and inventory** (explicit expectations, plus the saved acceptors): the fold inventory equals calendar × k with nothing missing or extra; the calendar equals the design (H4 10 targets, 2021-10..2024-10, excl. 2021-02/06; H8 9 targets, 2022-02..2024-10, excl. 2021-02/06/10; all with truth available); every fold has phase scenario_historical, strategy A and map 8965af6d6a724ba5d61d per frozen.json, prepared sha 972902bb…, availability digest equal to the same-H development folds, the selection's code (fd25e2f7…) and runtime, fitted, 5,718 rows, and all outputs hashed; the bindings selection → frozen → historical → report hold; `_accept_frozen`, `accept_record`(historical/report) and `accept_fold` 57/57 all pass | probe `probes/historical_identity_check.py` (f1269cf9…6455); summary `probes/historical_identity_check_summary.json` (4bce3b8c…124e) | **57/57, 0 problems** (exit 0) |
| **Independent global fit keys** (coordinator's checker, byte-unchanged eed66e0d…695b) over ALL records | `scenario_fit_keys_review_final.json` (1f02f807…ce10) | **654/654 records, 0 problems** (exit 0) |
| **Independent local/gate**: outer local key digests, gate keys and truth, internal support and decisions, exact-fraction gate decisions, deployed routes, forecast truth and persistence, and outer/internal global mask links | probe `probes/local_gate_reconcile.py` (a6924fa6…ac8f); output `probes/local_gate_reconcile_scenario_historical.json` (7751ff1e…4aa1) | **57/57 folds, 0 problems** (exit 0): 232 outer locals; 0 B weight blocks (A only); 969 regions; 5,764 gate blocks (5,764 fitted internal locals); 325,926 forecast rows; 317,612 persistence rows; 57 outer and 342 internal globals linked |

- **Probe compatibility change.** Historical fold records omit `map_route`, so the probe now reads the route from the frozen map's `consensus.json` when the field is absent. Rerunning on development with this probe reproduces the accepted output byte-for-byte (8acfd935…74390).
- **Limits unchanged:**
  - internal gate locals: support-only evidence (no saved key list or digest);
  - feature values and booster internals are not verified.
- **Metric recount.** The report metric recount (Study1/Study2, matched persistence and pooled comparators, country tables, bootstrap) is the coordinator's independent check. It is not duplicated here.
