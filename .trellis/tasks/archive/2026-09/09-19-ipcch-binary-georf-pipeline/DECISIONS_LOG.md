# Decisions and findings during implementation

Every scope change made after `task.py start`, and every finding that limits how
the results may be read. Requirement IDs refer to `prd.md` v1.0.

## D1 — Q8g extended: extract the areal component from a two-part GeometryCollection

**The Q8g stop triggered**, exactly as the plan anticipated. Trial `make_valid` on the
253 invalid geometries gives 0 still-invalid and 0 empty, but **217 GeometryCollection**
results, which PRD R2 and design.md require a stop on.

Evidence (`research/geometry-repair-trial.md`, read-only, nothing written): all 217
have the same composition — one areal part plus one linear part (136
MultiLineString+Polygon, 62 MultiLineString+MultiPolygon, 19 LineString+Polygon) — and
**every linear part has exactly zero area**. They are the degenerate spikes that made
the originals self-intersect.

The user was asked the one scope decision implement.md Phase 2 provides for, and on
2026-09-20 **authorized extracting the areal component**, narrowly:

> When a GeometryCollection consists of exactly one areal component plus only zero-area
> linear/point components, keep the areal component and discard the zero-area parts.
> Any other composition still stops the run.

Implemented as a guarded branch that refuses any other composition and asserts the
discarded parts have zero area. This is **not** a general "take the largest polygon"
rule and it does not authorize boundary replacement, area deletion or any other repair
escalation.

Rejected alternatives: building adjacency on the unrepaired geometry (shapely topology
operations on invalid input are undefined, so adjacency could be silently wrong, and
adjacency drives the partition); and repairing only the 36 areas that came back clean
(leaves 217 invalid geometries in the adjacency computation, i.e. does not solve it).

**Limitation to carry into the report:** 34 of the 217 change polygon area by more than
1% against their original, worst `admin_code=1425` at 29.21%. An invalid original's area
is diagnostic only, not ground truth. Topology repair does not establish administrative
identity.

## D2 — Five covariates have literally zero missingness; treat as a provenance limit

Measured on the assembled 170,780-row feature matrix, independently in the main session:

| field | NaN rate |
|---|---|
| `EVI_mean`, `nightlight_mean`, `nightlight_std`, `Rainf_f_tavg_mean`, `Tair_f_tavg_mean` | **0.000000** |
| `GPP_mean` (same approved family) | 0.313538 |
| `event_count_battles` / `distance_to_nearest_acled` | 0.140362 |
| `WFP_Price` | 0.419282 |
| `CPI` | 0.508186 |

Zero missingness across 170,780 origin-month reads spanning 2013-2026, while a sibling
field in the same six-member family misses 31%, is the fingerprint
`research/secondary-predictors.md` already flags: a candidate upstream producer applies
**ungrouped** forward-fill then zero-fill to EVI/nightlight.

Per R3 this is **not** acted on: no claim of contamination or cleanliness, no
reconstruction, substitution or dropping of fields. It is recorded because if that
forward-fill is temporal, these five fields could carry information from after their
nominal month, and the own-origin contract this pipeline enforces would not catch it —
our contract governs *which month we read*, not what upstream put in that month.

## D3 — The monthly scaffold has 624 gaps; harmless now, a forward risk

39 areas (Central African Republic, Honduras, Lesotho, Mozambique, Somalia, South Sudan)
end at 2024-12. **Corrected by the D8 review:** this does not mean a 2025 target loses all
covariates — a 2025 target can still have a covered 2024 origin. The risk is narrower
than first stated. They hold only 49 valid labels with a latest target of 2024-09, so
**zero** current feature rows are affected. But Stage3 main targets run to 2025-12: if
any of those areas acquires a 2025+ label, all 70 raw fields would be NaN at its origin.
The feature audit exposes `panel_grid_gaps` and `panel_areas_with_month_gaps` so this
cannot pass silently.

## D4 — Window sums accumulate directly rather than by cumulative difference

A cumulative-sum implementation of the 4/12-month windows was rejected during
implementation: `cum[end] - cum[start]` subtracts two area-lifetime totals, so its
rounding error scales with the whole history rather than the window. Direct oldest-first
accumulation over at most 12 terms is bit-reproducible and costs nothing (assembly runs
in 1.3 s). Related trap documented in code: `np.nan_to_num(x, nan=0.0)` also rewrites
infinity to 1.8e308, which would convert a poisoned window into a plausible finite
number instead of NaN; infinities are normalized to NaN before any arithmetic.

## D5 — Two plain repairs lose ~97-98% of their footprint

Found during geography implementation, after D1 was written. D1 profiled only the 217
GeometryCollection outcomes; the worst footprint changes are outside that set, among the
36 areas `make_valid` returned as plain polygons — which the **original** Q8g approval
already covered and which need no extension.

| admin_code | original | repaired | change |
|---|---|---|---|
| 162 | 20,013 km² | 303 km² | **-98.49%** |
| 133 | 12,689 km² | 443 km² | **-96.51%** |

Only 2 of 36 plain repairs exceed 1%, but these are ~3x worse than the worst collection
case (`1425`, 29.21%). Independently reproduced in the main session with
`pyproj.Geod(ellps="WGS84")`.

**No stop is triggered.** Q8g stops on remaining-invalid / empty / non-polygon /
ambiguous identity, and none applies; design.md forbids inventing an unapproved
percentage-tolerance gate. Recorded as a limitation instead: two administrative areas
shrink by ~97-98%, an invalid original's area is diagnostic only, and any spatial
partition learned on this geometry inherits that uncertainty.

Also corrected here: D1 and `research/geometry-repair-trial.md` said 34 of 217 exceed
1%; the correct count is **35**, confirmed geodesically and planar, with a clean gap
between 0.00884 and 0.01154.

## D6 — Two geography facts that constrain later stages

- **1,231 of 6,227 polygons (19.8%) have no adjacency neighbour.** Inherited polygon
  refinement operates on neighbourhoods, so a fifth of the universe cannot be refined by
  it. This is a property of the source geometry, not of the repair.
- **Reference coordinates and polygon centroids are materially different.** 735 areas
  differ by more than 1e-6 degrees, worst 0.677 degrees (~75 km). Against the 100 km Q8r
  donor cap that is decision-changing, so the donor path must use the keyed
  `unique_area_id_lat_lon.csv` reference coordinates and never the helper's centroids.
  The implementation keeps them as separate objects and asserts the distinction in a test.

## D7 — Specification drift to reconcile

PRD R2 and design.md still read "no silent component extraction" and "do not extract
polygon pieces out of GeometryCollections", with a mandatory stop. The 2026-09-20 user
approval (D1) supersedes that narrowly, but those documents were written by the planning
agent and have not been amended. An amendment note has been added to `prd.md` pointing
at D1 rather than rewriting R2, so the authoritative text is not silently edited while
the disagreement is still visible.

## D8 — Independent adversarial review (codex, 2026-09-20) and its disposition

The planning agent was dispatched via herdr as an independent adversarial reviewer with
the explicit instruction to disprove the implementation's claims. It ran 11 minutes,
verified the five headline claims against the real source with independent rational
arithmetic, and found **10 defects**. Its verdict on the claims:

| claim | reviewer's finding |
|---|---|
| R1 counts and step order | reproduced exactly, incl. 84 fills and 2,601 shares at .20; all 42,695 labels agreed under independent exact-rational arithmetic |
| 93 features / Q6d windows | passed whitelist, order, sparse-calendar, O-k lag and future-exclusion checks |
| Stage1 split | all six counts reproduced; original outcomes split before horizon expansion |
| single patch | the approved guard is the only AST change |
| import isolation | **FAILS** — see below |
| reporting counts-before-ratio, shared bootstrap draw | hold |
| `E_persist` as history-available subset | **FAILS** — see below |

### Fixed immediately in the main session (all three were my own code)

- **Recorded patch hashes were not file hashes.** `baseline_runtime.py` hashed
  newline-normalised text, so it recorded `e5858ed5…` where the real ZIP bytes hash to
  `be2c362b…`. A1 records file identity, so both hashes now read the bytes on disk.
  Verified: pristine now equals the ZIP member's byte hash and patched equals the disk
  byte hash, both matching the values the reviewer computed independently.
- **Import isolation had a namespace-package hole.** `src` has no `__init__.py`, so its
  `__path__` spans every `sys.path` entry with a `src/` directory. Submodules *present*
  in the baseline resolved correctly — which is why the first verification passed — but
  any submodule *absent* from the 45-file baseline fell through to the repository root;
  `src.model.GeoRF_XGB` was demonstrated to do exactly that. `src.__path__` is now pinned
  to the baseline, so an absent submodule raises instead of silently resolving to
  unpinned code, and the guard now checks **15** loaded modules rather than 2.
- **The lowercase `feature_drop` alias was missed.** `config.py:254` defines
  `feature_drop = FEATURE_DROP`; only the uppercase names were being cleared, leaving a
  live drop list bound to the old value. Both are now cleared.

### Dispatched for repair

Four reporter defects: `E_persist` being defined by the supplied predictions rather than
by verified as-of history (P1, violating Q4/Q4b/A5); malformed numeric tokens silently
changing cohort membership and area identity; empty approved horizons disappearing
instead of being reported as explicit empty cohorts (Q9a); and the K=1 bootstrap
cancelling all 1000 draws when Q9b only requires suppressing the *interval* — with an
existing test that had encoded the wrong behaviour.

Two `prepare_data.py` defects are queued until the Stage1 runner agent releases the file:
R1 preserves only `normalized_p3plus_str` where R1 requires all five normalized
components, and a valid non-polygon input (Point, LineString) bypasses the mandatory Q8g
stop because the polygon-type check runs only after repair. Neither affects the pinned
source, which contains polygons and whose labels were independently confirmed.

### Corrections to this log the reviewer identified

- **D3 overstated the scaffold-gap risk.** A 2025 target can still have a covered 2024
  origin, so the 39 areas ending at 2024-12 do not automatically lose all covariates.
- **D6 overstated the adjacency comparison.** The repaired layer has 1,231 isolated
  polygons; the original has 1,222 plus 38 areas whose comparison raises a topology
  error. The comparable subset shows no newly isolated polygon, but that does not
  establish that all 1,231 are unrelated to repair. Report the repaired count, the
  comparable-subset result and the 38 unavailable comparisons separately.
- **D1's "34" is superseded by D5's verified 35**, and D7's "have not been amended" is
  stale — `prd.md` now carries the amendment note.
- A P3 precision edge case stands: with phases `.8, 0, .20000000000000000000000000001, 0, 0`
  default Decimal precision rounds the normalized share to exactly `.2` and labels it 0,
  where exact arithmetic gives 1. No row in the pinned source is affected; queued with
  the other `prepare_data.py` repairs.

## D9 — Stage 1 learned NO partition; the partitioned-vs-pooled contrast is degenerate

Run `ipcch-stage1-20260920c`, independently verified in the main session from the
artifacts rather than from the runner's summary.

```
F1 performance gate: parent=0.803606, candidate=0.811040, accepted=False
```

The spatial scan proposed a 1,428 / 1,836 group split, polygon contiguity refinement ran
its three epochs on the true adjacency, both candidate children were trained — and the
strict gate rejected the split at a gain of **0.007434**, below the inherited
`MIN_CLASS_1_IMPROVEMENT_THRESHOLD = 0.01` (extracted `config.py:346`, unmodified). In
`f1_mode` a descendant requires an accepted parent, so levels 1-4 were never attempted.

Independent verification: `branch_table` is (64, 6) with `sum() == 1` and only `[0,0]`
nonzero; `s_branch` has a single column `''`; `branch_to_code` is `{"": 0}`;
`branch_table_accepted_nodes: 1`. The learned map is the root branch alone — 1 terminal
branch, depth 0, 3,264 genuinely-root areas, 0 placeholders.

**Consequence.** With one partition, the partitioned RF and the pooled RF are literally
the same model, so R5's partitioned-vs-pooled comparison is degenerate and any delta
between those two arms is exactly zero by construction. Stage 3 still yields a real
three-way comparison — pooled RF, binary XGB and persistence — and the degenerate fourth
arm is itself the finding.

**No protocol change was made.** PRD R4 preserves the released gate mathematics and
implement.md phase 6 says not to vary protocol based on results; a changed scientific
choice returns to grill. The threshold, depths and gate were left exactly as released.
The run is marked `stage1_outcome: "no_partition_learned"` and carries an explicit
limitation so nothing downstream reads a partition advantage into an identity.

Supporting numbers: donor completion over the full 6,227 universe gives 3,264 learned +
2,367 donor-completed + 596 unresolved, with completion distances p50 4.40 km, p95 82.09
km, max 99.978 km and no area exactly at the 100 km cap; unresolved nearest-eligible
distances start at 100.17 km. Singleton scoring (5,888 views / 1,472 areas) gives
class-1 F1 0.6296. The `nearest_donor` 0.653 versus `unresolved` 0.561 split reflects
sample composition only, since both route to the identical root model.

Note for readers: parent F1 0.8036 is **internal development validation** under per-area
chronological cutoffs, not a forward forecast score. It is not comparable to Stage 3.

## D10 — A GeoRF-module binding defeated the config-level FEATURE_DROP override

Found by the Stage 1 agent and independently fixed in `baseline_runtime.py` after the D8
review flagged the same root cause. `GeoRF._get_feature_drop_config` falls through to the
lowercase `feature_drop` alias, and `from config import *` had already copied that alias
into the **GeoRF module namespace**, so clearing it on the config module alone left a
live drop list bound to the old value. `baseline_runtime` now clears both names on the
config module, and `run_pipeline` additionally neutralises both in the GeoRF module
namespace; `drop_list_` came back `[]`. The manifest reports the config-module and
GeoRF-module values separately so the divergence stays visible rather than being
smoothed over.

## D11 — Reporter defects from the D8 review are fixed and independently confirmed

All four reporter findings are repaired; 28/28 checks pass under both interpreters, and
the main session independently reproduced the reviewer's P1 scenario and confirmed it is
now rejected.

- **`E_persist` is now derived from the label history, not from the runner's column.** A
  `LabelHistory` type with an inclusive as-of lookup reconciles supplied persistence
  availability, source month and value against the valid R1 history *before* any cohort
  exists. The reviewer's fixture — February truth=1, March truth=0, both persistence cells
  blank — is now a hard failure naming the offending key, and so is a March persistence
  sourced at February that contradicts February's saved truth. Two verification
  strengths are recorded in the manifest rather than assumed: exact reconciliation against
  the ledger when a run directory supplies one, and refutation-only when scoring loose
  predictions, which adds an explicit limitation line. R1 is imported read-only, never
  reimplemented.
- **Malformed numbers fail instead of masquerading as missing.** `persistence_pred="BROKEN"`
  silently dropped a row out of `E_persist`; `admin_code="2.7"` silently became area 2.
  Declared missing tokens are now distinguished from malformed values, and identifiers
  must be whole numbers. Infinities count as malformed.
- **Empty approved horizons stay in the report** as explicit zero-support cohorts with
  `f1_reason="empty_cohort"`, reconciled against R4's 35/33/30/24 schedule, so genuinely
  empty support is separable from missing predictions.
- **The K=1 bootstrap draws again.** Q9b suppresses the *interval* below two countries; it
  does not cancel sampling. The 1000 replicates and their multiplicities are retained and
  only the CI is withheld.

Two corrections the repair surfaced, both worth recording because they are the kind of
thing that hides a defect:

- **An existing test had encoded the bug.** `test_ci_is_suppressed_below_two_countries`
  asserted `replicates.empty and draws.empty` — i.e. it asserted the wrong behaviour and
  would have kept asserting it forever. It was rewritten rather than deleted, keeping every
  correct assertion.
- **A test fixture was itself an instance of finding 1.** A row for area 11 carried
  `persistence=None` while area 11 already had a 2023-02 truth of 1, so its 2023-04 origin
  *did* have available history. The fixture was moved to an area whose first valid label is
  its own target month; all downstream counts are unchanged.

Byte-for-byte determinism still holds, verified across both interpreters by per-file
SHA-256. The only cross-interpreter differences are CRLF in three text files from
`Path.write_text` and the recorded input path/hash in `validation.json`; no computed value
differs.

## D12 — `prepare_data.py` defects fixed; R1 gate and geometry outcomes unchanged

The three remaining D8 findings are repaired, and the two invariants that had to hold did
hold, independently re-verified in the main session.

- **R1 now preserves the five normalized components.** R1 requires "raw components, P5-fill
  flag, S, normalized components, share, validity reason and label"; only the P3+ share was
  being kept. Five exact-decimal columns were added, blank on invalid rows. Verified:
  42,695/42,695 valid rows carry all five, 0 invalid rows carry any, and an independent
  sample of 300 rows has the five components summing to 1 within 1e-90.
- **A valid non-polygon input now stops the run.** The areal-type check had lived only on
  the repair path, so `Point(0,0)` and a valid `LineString` returned `unchanged_valid` and
  would have entered adjacency. The check now runs on the input, before the valid-geometry
  shortcut.
- **The label no longer depends on a rounded quotient.** The label is read from R1's stated
  exact comparison `5*(P3+P4+P5) > S`, computed in a context sized from the operands with
  `Inexact` trapped, so a precision error stops the run instead of relabelling a row. The
  divided normalized values are kept for provenance at a declared precision and are
  explicitly not label-bearing.

**Both gates unchanged, as required of a provenance-only change:** valid 42,695 / positive
15,206 / negative 27,489 / areas 6,227 / P5-fills 84 / shares at .20 2,601, with identical
invalid-reason counts; and geometry 5,974 unchanged_valid + 36 repaired_polygonal + 217
repaired_collection_areal_extracted. Contract checks 30 -> 33.

**Worth recording, because it shows why the P3 finding mattered.** The reviewer noted no row
in the pinned source is affected, which makes the fix look cosmetic. When the main session
tried to confirm the fix with a quick one-liner, that one-liner used the ambient 28-digit
Decimal context — and therefore rounded `S` and `5*P3+` to the same value and concluded the
label should be 0, i.e. **the verification reproduced the very bug it was checking for**,
while the fixed code returned the correct 1. A defect whose natural verification method
shares the defect is exactly the kind that survives review, which is the argument for
fixing it despite zero current impact.

## D13 — Partitioned and pooled are NOT identical under a single partition, and why that matters

**Correcting D9 and a wrong instruction I gave.** D9 said that with one learned partition the
partitioned and pooled arms are "the same model" and their delta is "exactly zero by
construction", and the Stage 3 dispatch told the implementing agent to assert that identity.
**Both were wrong**, and the instruction was retracted to the running agent before it could
act on it. The user independently made the same point: nothing about an unaccepted split
implies the two arms coincide.

Measured on run `ipcch-v1-20260920a` (81,109 prediction rows):

| route | rows | probabilities identical to pooled |
|---|---|---|
| `partition:0` | 75,698 | ~6% |
| `pooled_rf` (unresolved fallback) | 5,411 | **100%** |

8,256 hard predictions differ. The mechanism is exactly as design.md specifies and as the
unresolved-fallback row confirms: the local model for partition 0 trains on the subset of
the common pool that the frozen map assigns to it, while the pooled RF trains on the whole
pool. The 596 areas with no eligible donor within 100 km are therefore **excluded from the
local model's training rows but included in pooled's**. Different training sets, different
forests.

### The exclusion is strongly spatially structured — this is the finding

Per area (6,013 areas appear in the Stage 3 support; the remaining ones of the 6,227
universe have no valid test target): 3,198 learned, 2,298 donor-completed, 517 unresolved.
Those 517 sit in 21 countries and are anything but scattered:

| country | unresolved / country total |
|---|---|
| Lebanon | 88 / 88 (**100%**) |
| Bangladesh | 44 / 44 (**100%**) |
| Ecuador | 24 / 24 (**100%**) |
| Timor-Leste | 14 / 14 (**100%**) |
| Palestine, State of | all (**100%**) |
| Angola | 12 / 14 (85.7%) |
| Dominican Republic | 22 / 32 (68.8%) |
| Ethiopia | 26 / 39 (66.7%) |
| Mozambique | 139 / 579 (24.0%) |
| Pakistan | 58 / 159 (36.5%) |

**Five entire countries have no learned coverage at all.** The top three countries hold
55.1% of all unresolved areas, against a much flatter learned distribution (45 countries,
top three 37.4%). These are the areas geographically disconnected from the labelled core —
IPC/CH labels concentrate in Africa and the Sahel, and Lebanon, Bangladesh, Ecuador,
Timor-Leste and Palestine have no eligible donor inside the 100 km cap.

### How this must be reported

The partitioned-vs-pooled contrast in this run does **not** isolate spatial heterogeneity,
because no split was accepted. What it actually contrasts is *a model trained with five
countries and parts of several others removed* against *a model trained on everything*. A
nonzero delta is therefore not evidence of spatial structure in the outcome; it is a
training-set composition effect whose composition happens to be geographic.

Per the user (2026-09-20): this is a useful result rather than a defect, and the spatial
distribution of validation coverage is itself informative and often under-used — but it
**must be labelled**, so that the delta is never read as a partition benefit. Deeper
discussion is deferred; the requirement here is that the run's outputs and the final
write-up carry this characterization, including the five fully-unresolved countries.

## D14 — Scope of this round is confirmed baseline-only; the next round is scoped

User direction, 2026-09-20, after seeing the no-partition result and D13:

**This round stays exactly as specified.** Produce the baselines, establish the rough
picture, change nothing. No threshold relaxation, no tuning, no feature engineering. This
matches PRD R4's requirement to preserve the released gate mathematics and implement.md
phase 6's "do not vary protocol based on results".

**Next round** is to relax the `MIN_CLASS_1_IMPROVEMENT_THRESHOLD = 0.01` gate, tune
parameters, and do feature engineering.

All three are *changed scientific choices*, which implement.md routes back to grill. The
successor therefore needs its **own task and its own fresh pre-registration** — the current
task's acceptance criteria were written for a frozen-protocol baseline run and cannot be
reused to adjudicate a tuned one on the same data.

Two things the successor should carry in from this round:

- The rejected split's actual margin is known: parent 0.803606 vs candidate 0.811040, a
  gain of **0.007434** against the 0.01 gate. Any relaxed threshold should be chosen and
  declared *before* the run, not set just below that number after the fact.
- D13's finding is independent of the threshold. Even if a relaxed gate accepts a split,
  the 596 areas with no eligible donor inside 100 km stay outside every local model's
  training set, so the partitioned-vs-pooled contrast keeps carrying a spatially structured
  training-composition component that must be reported separately from any partition effect.
