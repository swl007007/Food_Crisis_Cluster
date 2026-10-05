# P2 evidence — quartet, projection, metrics, model provenance (2026-10-04)

Phase status: implemented and tested on synthetic data only. No model was
fitted on project data; real fitting belongs to the formal P6 run.

## Implementation

| Module | Content | Provenance |
|---|---|---|
| `quartet.py` | contract-built G/L params (reg:squarederror, eta .05, seed 42, hist/CPU, nthread 4, gamma 0, max_delta_step 0, max_bin 256, depthwise; global base_score .5); fixed-round scalar global fit; local = fresh-copy continuation of the matching global appending exactly L rounds; checks: global bytes, rounds, base score, prefix tree-structure digest, prefix margins on probe rows; constant targets recorded and fitted; finite/shape prediction checks; atomic `Quartet`; key alignment check | FEWS native_xgb 101-270 adapted |
| `projection.py` | exact bounded decreasing isotonic projection (enumerate 8 contiguous block partitions, min SSE feasible, then clip [0,1]); unrounded `>=0.20` decoding | new |
| `metrics.py` | binary accuracy/precision/recall/F1/F2 and four-class accuracy/macro-F1 from pooled counts with R27 NA reasons; R² (NA n<2 or SST=0, negatives kept); exact `Fraction` strict gain gate (NA fails); single-column crisis scan masses | FEWS fourclass 39-61,170-190,214-227 adapted |
| `modelstore.py` | R48 exact-identity quartet store (atomic write, full identity + booster byte checks on hit, partial/corrupt/conflicting entry → TechnicalError, never refit), per-request JSONL ledger, fit/hit counters | new |
| `errors.TechnicalError` | R41 stop class | new |

Sibling IPCCH `forecasting_weight_decay.py` and `operational_contract.py` and FEWS
`plan.py` are recorded as reference-only (the old `cummax` decoder repair is
superseded by the R15 projection).

## Tests

`python -m pytest -q`: **131 passed** (70 P0 + 28 P1 + 33 P2), `evidence/P2-pytest.log`.
P2 covers: six known projection solutions (feasible, pooling, lower/upper bound,
reversed, bound+pooling); clip-before-isotonic counterexample ((.5,1.3,.1,0):
correct (.9,.9,.1,0) vs clip-first (.75,.75,.1,0) with higher SSE); agreement
with an independent PAVA on 2,000 random rows and no random feasible point with
lower SSE; inclusive unrounded decode; NaN/Inf/misshaped projection input stops;
frozen G4/L2 parameters; constant q5 target fitted and recorded; all-NaN feature
column; local = 200+20 rounds, parent digest = own target's global, prefix
structure and base score preserved, global bytes unchanged; reload reproduces
predictions bit-for-bit; partial quartet rejected; NaN/Inf/misshaped booster
output stops; infinite input and misaligned keys stop; empty global pool stops;
local cannot override base_score; store fits once then hits, reordered keys do
not collide, ledger lines; corrupt bytes / conflicting identity / missing target
file stop without refit; NA rules (all-TN, FP-only legal zeros, F2), fixed
four-class axis with absent class → macro NA, phase 4/5 merge; negative and NA
R²; exact gate (tie fails at >0, gain exactly .01 fails at >.01, NA fails);
scan masses including D = 0.

One P0 test was narrowed: the all-module import probe no longer asserts XGBoost
is unloaded (the model module legitimately imports it); a new test asserts the
preflight/CLI path alone never loads XGBoost.

## Supervisor review corrections (review of `8a83580`, fixed after P3 `0087683`)

Classification: items 1 and 3 can change a computed value (R² of a
constant-truth cohort; projection of extreme finite raw shares); items 2, 4, 5
and the integration obligations harden inputs/provenance without changing any
value produced by the frozen recipe. No scientific parameter changed.

1. `metrics.r_squared`: constant truth is detected on exact values before any
   mean (previously `r2(full(3,.2), full(3,.1))` ≈ -1.3e31); NA with reason for
   constant truth whatever the prediction; valid negative R² kept.
2. Metric/scan entrypoints: shared `_aligned_phases` (1-D, equal length, exact
   finite integer phases 1..5; a 0 sentinel is rejected) and `_finite_vector`
   (1-D, same cohort length, finite) used by `crisis_counts`,
   `four_class_confusion`, `r_squared`, `metric_panel`, `crisis_scan_masses`
   (groups must be 1-D integers aligned with the phases).
3. `projection.isotonic_decreasing`: PAVA with `math.fsum` block means on rows
   that violate the order (others unchanged), then clip. `[-1e8, 1e8, .9, .1]`
   → `[.3, .3, .3, .1]` (phase 4); clip-first counterexample and the 2,000-row
   reference comparison still pass.
4. `modelstore.validate_fit_records` (on write and on every load): per-target
   required fields; target/kind (kind must match the identity scope); byte
   digest; rounds, base score and structure digest recomputed from the booster;
   resolved-config objective and base score; for locals rounds = global +
   appended, prefix digest = parent structure, parent digest = the identity's
   global booster. Any violation → TechnicalError, never a refit.
5. `quartet.resolved_config`: fit-time `save_config()` stored in every global
   and local fit record (not re-derived after a reload).
6. Integration obligations: `continue_local_quartet` refuses any non-global
   input quartet (blocks 220→240 accumulation); `array_digest` refuses object
   arrays; Stage1 identities now also name the four targets and the F scope
   (they already bound H, G/L, params incl. seed, ordered F keys, X/Y, unit
   weights, schema, availability, prepared-manifest source, environment and
   quartet-code hash; locals bind region members and the global booster digests).

Tests: **195 passed**, `evidence/P2-review-pytest.log`. Red check: with the
`8a83580` versions of metrics/projection/quartet/modelstore the 20 selected
regression cases give 16 failures (4 cases were already rejected before);
all pass on the corrected code. Original P1/P2 dev evidence identities are
unchanged; no real-data run was repeated.
