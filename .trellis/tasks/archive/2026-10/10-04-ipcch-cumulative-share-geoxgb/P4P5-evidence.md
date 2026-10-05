# P4/P5 evidence — rolling Stage3, reporting and independent replay (2026-10-04)

Phase status: implemented and tested on synthetic data only (unit tests with
real XGBoost quartets; an end-to-end learn-map → predict → report → replay run
on a synthetic prepared run with a test-only small contract). No Stage3 or
report on project data: that is the formal P6 run after supervisor release.

## P4 — `stage3.py`, `predict.py`, CLI `predict`

- `HorizonContext`: rows by target month; inclusive window `[O-35, O]`;
  `global_quartet(origin)` (R40: empty required pool → ContractError stop) and
  `local_quartet(origin, region, global)` (R28 fit floors on region window rows,
  continuation of the same-origin global only); identities bind scope, H,
  targets, fitting origin/window, ordered row positions and (area, month) keys,
  the verified prepared X/keys artifact digests, recipe/params/seed, schema,
  availability, environment/code and the frozen map digest; locals add region
  node, member digest and the global identity/booster digests.
- `run_fold`: empty T → `no_valid_target` ledger entry and no fit request
  (R47); current global → matched pooled predictions for all keys (E_all);
  with an accepted split, R37 date-major replay over the latest ≤ 6 observed
  U < O with V = U - H, every region key kept (unsupported dates use that
  date's global), `gate_decision` = R28/R29 pooled support + ≥ 3 successful
  supported local dates + strict exact crisis-F1 gain > 1/100 (NA never
  passes); enabled regions with current support route to their local quartet,
  others to `global_fallback:<reason>`; unmapped areas `unmapped_area_global`;
  maps without an accepted split `global_only_no_accepted_split` and no local
  fits. Keyed predictions carry raw/projected quartets and phases for GeoXGB
  and pooled, truth, persistence (phase, q3, source month, age), route,
  region and provider identities. Gate decisions (JSONL) and gate pairs
  (per fold) are saved; gates are recomputed at every O.
- `predict.run_predict`: re-verifies prepared artifacts, accepts a frozen map
  only if its SHA256 equals the freeze record and node IDs validate, runs all
  scheduled main + 2026 folds per H, writes predictions (digest in summary),
  fold ledger and the model request ledger.

## P5 — `report.py`, `replay.py`, CLI `report` / `replay`

- `report.run_report` (from saved predictions only; digest checked): per H and
  period, E_all (GeoXGB vs matched pooled) and E_persist (GeoXGB vs persistence)
  full R8 panels with NA reasons, all deltas and coverage (persistence coverage,
  countries, local/unmapped keys); main-period R49 bootstrap for the two crisis
  F1 differences (sorted countries, default_rng(42) reset per H × cohort, 2000
  draws of K indices with replacement, multiplicities shared by both arms, every
  draw saved with per-arm F1 and delta, interval = linear 2.5/97.5 only if
  K ≥ 2, point defined and all 2000 deltas finite, else NA with reason);
  2026 and country/month diagnostics are point estimates only.
- `replay.replay_run` re-implements projection (plain PAVA), counting, gate
  dates/decisions, Stage1 selection and bootstrap arithmetic without calling
  the production functions, and checks: prediction/frozen-map digests; fold key
  sets equal prepared valid keys (no dropped/extra/duplicated rows); truth equals
  prepared; origin = T − H; persistence source ≤ O and age consistent;
  projection/decoding/monotone order for GeoXGB and pooled; region from frozen
  map, non-local rows identical to pooled, unmapped → global; gate dates and
  every gate decision from saved pairs; local routes only where the gate enabled
  them, with the recorded provider; every Stage3 model request's window end =
  fitting origin (= O for current, = U − H for gate), local prefix = same-origin
  global (digest, rounds, structure); fitting rows and per-target y digests
  rebuilt from prepared keys (target order); Stage1 per-candidate F1 and winner;
  report counts/coverage and bootstrap deltas, RNG multiplicities and intervals.

## Tests

`python -m pytest -q`: **218 passed**, `evidence/P4P5-pytest.log` (P4 12,
P5 end-to-end/tamper 8, plus updated CLI tests). P4: inclusive window cut at O;
six latest observed dates; empty fold requests no fit; every key kept and
unmapped routed to global; non-local rows bit-identical to pooled; gate pairs
pooled per region with validation months < O and internal origin U − H; local
adoption on a two-regime world; cache hits across folds sharing five dates;
no-split map global-only with a single global request; changed membership does
not collide; empty global pool stops; successful-date counting with fallback
keys retained; gain exactly .01 rejected. P5: complete synthetic run replays
with zero failures (8 folds × 80 keys, one ledger-only empty fold, K = 2 for both
bootstraps); replay catches a dropped row, swapped q2/q5 raw columns, a
persistence source month after O, a flipped gate decision, a re-parented local
prefix, a tampered report count and swapped per-target y digests in a model
record (each tamper refreshes the outer digest so the deeper check must fire).
The end-to-end world's gates legitimately kept global (pooled F1 ≈ .62–.64 vs
local ≈ .36–.61), which is a data property; local adoption itself is covered
by the P4 unit test.

The P0 "unported command" test became: learn-map/predict/report/replay on a run
without its prerequisite fail with exit 1 and write nothing; no phase remains
unported. `config/source-provenance.json` now has 20 copied components and no
pending sources.
