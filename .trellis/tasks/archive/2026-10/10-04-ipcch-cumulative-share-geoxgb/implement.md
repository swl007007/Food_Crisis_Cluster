# IPCCH cumulative-share GeoXGB — execution contract v1.0

Authority: user R1–R51 in `prd.md`, scientific specification `design.md`, and the
three source-grounded clarifications below. R51 authorizes infrastructure,
freeze, Claude execution and Codex supervision; no repeated permission request
is needed. Historical planning-only statements in research notes describe their
date of writing, not the current authorization. This contract does not authorize
changing the scientific choices after seeing evaluation results.

## Ownership, location and milestones

- Branch: `ipcch-cumulative-share-geoxgb`; same repository, no push required.
- New implementation and tests: `IPCCHGeoXGBExperiment/` only. Task documents,
  contexts and small review evidence: this task directory. Existing experiments,
  raw inputs and `pooled-onset-confirmatory` task/session pointer stay untouched.
- Sole executor: the verified live Claude session bound through `trellis-audit`.
  Codex prepares the baseline, supervises, checks evidence and releases milestones.
- Operational ledger: **this task's `PROGRESS.md`**, not the repository-root
  ledger. Update at commits, phase boundaries, failures and handoff; it records
  state and never supplies scientific approval.
- First commit the science/plan/context/manifests. Bind the live Claude, then
  Claude runs audit `start`. Its recorded `base_sha` must be that baseline.
- Claude performs P0 only, commits and returns evidence. Codex reviews P0 and
  records the accepted foundation SHA before releasing P1–P6 to the same Claude.
  This is the user's requested infrastructure freeze, not a second user gate.
- Freeze the finished implementation commit and prepared-data/map identities
  before formal Stage3 evaluation. Necessary bug fixes are documented, tested
  and versioned; affected artifacts are regenerated rather than mixed across
  identities. No science retuning or silent baseline rewrite.

## Source-grounded execution clarifications

1. **Stage1 complete-route selection.** Baseline is parent/parent on all parent
   S keys. Consider child/parent, parent/child, child/child in that order; skip
   combinations requiring an unsupported child. Compare exact rational crisis
   F1 and update only for strictly larger values. Therefore parent wins ties;
   among equally improved combinations the first in this fixed order wins.
   Undefined parent means no split; undefined candidates cannot win. Each side
   selects an entire quartet, never individual q models. This preserves the
   reference `select_macro_children` loop while replacing its class adapter.
2. **Membership is separate from prediction provider.** An accepted split
   creates child memberships, including a child that inherited the Stage1
   parent checkpoint. The frozen map records terminal membership; Stage1
   inherited providers are a separate artifact. In Stage3 each terminal region
   of an accepted split is eligible for its own same-root local fit/gate on its
   own members. No accepted split means global-only, without a spurious local
   fit on the root region. No ancestor-local accumulation.
3. **Feature infinities.** Preserve the inherited data-layer distinction:
   raw covariate and assembled original93 infinities become NaN with column
   counts/audit before use; engineered history468 infinities fail preparation.
   Final X must contain finite values or NaN. Do not let the generic FEWS model
   cleaner hide an unexpected infinity in the history block or final matrix.
   Nonfinite predictions always stop under R41. Source anchors:
   `IPCCHGeoRFExperiment/prepare_data.py:627-690,922-932,1017-1026`,
   `IPCCHPopulationHistoryExperiment/prepare_data.py:747-760`,
   `FEWSNETGeoXGBExperiment/src/partition/partition_opt.py:860-890`.

## P0 — independent package foundation; no model fit

Build the smallest usable package foundation with package-qualified imports,
local configuration, a documented module/CLI entrypoint, focused infrastructure
tests and an independent output directory. Suggested modules are data/features,
targets/metrics, models, partition, rolling, reporting and I/O; these are
responsibilities, not a requirement for a framework or a file per concept.

Required P0 deliverables:

1. Copy/adapt only necessary source components named in `source-manifest.json`.
   A file listed there is a reference boundary, not a direction to copy all its
   imports. Record old path/hash/commit, destination, symbols or line span and
   adaptation reason in a package provenance manifest. For each transitive
   legacy dependency, record removal, a minimal local rewrite, or a bounded
   copy with an additional pinned source hash before use. Never copy a whole
   dependency chain solely because a reference file imports it. Modules not yet ported
   may be listed as pending and cannot pretend to work. Preserve source notices.
2. Explicit input configuration derived from `input-manifest.json`, frozen
   candidate/config contract and the ordered 561-feature schema with a new
   >=20% semantics version. The old schema's approval/status text and candidate
   grid do not govern this experiment. Old labels/features/models are forbidden
   caches; original93 covariates come from the same pinned raw CSV.
3. Runtime lock/probe matching `environment-lock.json`: existing Windows Python
   3.12.10 and XGBoost 3.0.0. No installation, upgrade, Linux numerical substitute
   or optional SHAP/plotting dependency is needed. List only packages actually
   used by the new implementation; explicitly pin XGBoost and the chosen
   geometry I/O engine (pyogrio if using geopandas read_file). Linux Python may
   operate Git/JSON/documentation, not produce numerical experiment artifacts.
4. Read-only input preflight: verify frozen byte identities, unique and complete
   required country/area keys, geometry sidecar identities, area/polygon mapping
   bijection, cache area IDs/adjacency keys, index ranges, symmetry and cache
   component hashes against the repaired geometry. Check country/coordinate
   consistency under the inherited key mapping. Map `polygon_id_mapping` as
   area ID -> polygon index and `polygon_group_mapping` as the reverse; area
   IDs are not dense indices. Country identity uses trimmed `country_en`, with
   trimmed `country` as fallback; missing ISO3/country_code alone is allowed.
   Raw `lat/lon` must match saved `ref_lat/ref_lon` by area ID; polygon centroid
   coordinates are different objects and need not equal those reference points.
   Preserve the known 31 missing ISO3, 15 missing country_code and 735 centroid
   differences as diagnostics for this pinned input, without repairing them.
   Validate the frozen cache directly, without loading the old adjacency helper.
   Do not rebuild topology or
   promise unverified administrative-match provenance. Do not load an old backend.
5. Import and CLI smoke checks from the repository root and an unrelated working
   directory with an explicit package location. No live imports from sibling
   experiment packages, bare `config`/`src` resolution or sys.path hijacking.
   Preflight must not fit. Unimplemented prepare/learn-map/predict/report commands
   must fail explicitly, never emit success-shaped empty scientific artifacts.
6. Record the actual commands, exit status, versions, input results, package
   inventory and focused test evidence in `P0-evidence.md`. Commit P0 and stop at
   `awaiting_supervisor_foundation_freeze`; report commit SHA and evidence paths.

Codex release requires actual passing checks and inspected package/import/input
evidence, not file existence or a successful handoff. Referenced bytes were
hashed during planning; P0 validates their current identities and structural
consistency. Any mismatch stops rather than edits the expected hash.

## P1 — population targets, calendar and rich561

Implement bounded local copies of IPCCH QC/calendar/original93 and population
history468. Preserve exact source decimals for truth boundary decisions; build
q2–q5, five-phase truth and four/binary mappings with the same >=20% policy.
Retain original fields, reported phase, P5 fill and invalid reasons. Rebuild all
crisis-derived history fields. Do not import old runtime code or its old labels.

Save full QC/scaffold/coverage ledgers, ordered schema, per-H keyed matrices,
availability identity, source-month/origin evidence and F/S membership. F/S is
pooled 2014–2022 within-area halves; Stage3 uses [O−35,O], training-row features
at that row's own origin. Preserve no-history/unmapped evaluation keys. Export
all 122 main fold identities and the full 2026 coverage ledger.

Focused tests: exact .20, sum .90/1.10 boundaries, missing P1–P4 versus P5,
malformed/duplicate keys, no-history rows, sparse calendar lags, 561 names/order,
history-window edges, old strict-threshold aliases regenerated, and future-data
perturbations leaving earlier feature/fitting/persistence inputs unchanged.
Do not transplant old package exact cohort-count assertions as new science.

## P2 — quartet, projection and model provenance

Four scalar `reg:squarederror` boosters with fixed G/L recipes; native feature
NaN, unit original-key weights, constant targets fitted normally. Local starts
from its matching immutable global, appends exactly 20/40 rounds once and keeps
global base score/prefix. Save fitted boosters, complete parameters, feature
order, fitting keys, X/y/weight identity and prefix evidence. Cache full identity
under R48; a stale/corrupt declared cache is a failure, not a refit shortcut.

Implement bounded equal-weight least-squares decreasing isotonic projection,
unrounded decoding and exact confusion-count gates. Test known projection
solutions (including why clipping before isotonic is incorrect), .20, all-NaN X
columns, constants, quartet atomic routing, global immutability, exact appended
rounds, reload predictions, bad shape/key/nonfinite output failures, and R27 NA
metrics including absent four-class axis and negative R².

## P3 — direct Stage1 maps and frozen winners

Implement the crisis-F1 scan and deterministic size/tie/1000-iteration rule,
three synchronous induced-neighbor smoothing rounds, support checks, complete
route selection and membership-depth4 recursion. Preserve the exact scan math
from the pinned source subject to R32/R46; record deterministic area ordering.
Use numeric ascending canonical area ID ordering, without locale/string ordering.
No Stage2, retries, 80-round cap, donor assignment or forced connectivity.

Evaluate all 8 G/L pairs for every H on the same F/S; shared H/G/F global reuse
does not share the L searches. Rank absolute whole-S F1 with the frozen R44 tie
order; save all candidate scores/predictions/maps/routes and counts. Tests cover
support equality edges, zero scan mass, small infeasible group sizes, scan ties,
simultaneous smoothing, unsupported-side inheritance, mixed combination ties,
depth budget and root-only equivalence. Freeze the winning map/config per H;
all-NA selection or any technical failure leaves that required run incomplete.

## P4 — rolling Stage3 and paired baselines

For each nonempty planned fold, compute the shared latest up to six observed U<O,
historical origins V=U−H, independent 36-calendar-month fitting pools and current
fit. Keep all region keys across support fallback dates. Gate uses pooled counts,
100/20/3 support, 20/20 class support, >=3 successful supported local dates and
exact gain >.01 against corresponding fold-global. Recompute gates each O;
reuse models only by complete fit identity, never reuse a gate decision.

Produce GeoXGB and matched pooled on E_all, persistence from the latest lawful
actual observation with age/source month, and E_persist paired comparisons.
Empty truth months are ledger-only; mandatory global empty pool or technical
error stops. Test no-history/unmapped retention, date cutoff equality, gate
exact .01, successful-date counts, no-split equivalence and caches shared across
identical fit requests while changed inputs/prefix/member sets do not collide.

## P5 — reporting and independent replay

Recompute every specified metric from saved keyed predictions, per H and cohort,
with main 2023–2025 separate from observed 2026. Save fixed-axis confusion counts,
all metric values/NA reasons, coverage, per-country/month diagnostics and paired
deltas. R49 bootstrap uses sorted nonempty country keys, NumPy default_rng(42),
2000 independent draws of K country indices with replacement per H/cohort;
store every draw's country multiplicities and scores. This fixes the generator
mechanics for reproducibility without changing R49's sampling unit or policy.
No redraw or dropped undefined draw; CI only if all 2000 deltas are defined.

Replay tests reconstruct projection/decoding, routing, candidate selection,
metrics, paired-key intersections and bootstrap intervals independently of the
training entrypoint. Keyed evidence must expose a dropped row, wrong target
order, future feature, inherited local prefix or stale gate; aggregate counts
alone are insufficient. No requirement to beat persistence at every H.

## P6 — frozen scientific run and delivery

After P1–P5 tests pass, commit implementation and freeze the source/environment/
schema/input identities. Run preparation, Stage1 all candidates, freeze winning
maps, then the full declared Stage3/report/replay schedule. Record timing and
actual fits/requests/reuse; 84824 is a conservative pre-reuse upper bound, not a
runtime promise or a permission to truncate. Small tests do not certify the
full scientific run. On a real error preserve partial artifacts and explain
the failure; fix the implementation without scientific retuning, then regenerate
the affected identity chain. Do not silently skip required candidates or folds.

Keep outputs in run-specific new-package directories (ignored bulky artifacts
are allowed with committed path/hash inventories and local review access).
Avoid overwriting existing completed runs. Identical verified model reuse under
R48 is permitted; record each usage separately. Save a concise final scientific
report with all adopted metrics, comparisons, limits and negative findings.

At completion commit relevant nonignored code/tests/config/docs, evidence
inventories and task ledger. Claude alone runs audit `close` from the bound
session using the exact registered repo spelling. Codex verifies queued job,
pinned SHAs and eventual accepted result; queued/launched/waived != audit pass.
Do not close at P0, code-only completion or a partial scientific run.

## Required evidence inventory

| Layer | Durable evidence needed for acceptance |
|---|---|
| Foundation | input/env/source/adaptation manifests, import probe, geometry/key preflight, P0 test log and accepted commit |
| Data | QC/raw-to-normalized targets and exact phase; all country/area/scaffold keys; ordered 561 schema; availability and F/S; all folds and no-target ledger |
| Fits | stage/H/q/origin/member identity, fitting keys/X/y/weights hashes and recoverable input references; saved boosters/config/order; global prefix checks; constants/support; unique-fit and request/reuse ledger |
| Stage1 | all 32 candidates, scans/smoothing/support/counts, accepted/rejected route combinations, parent/provider vs membership identities, keyed raw/projected S predictions, ranking/ties and frozen maps |
| Stage3 | lawful historical dates and fitting pools, keyed raw local/global replay predictions, every gate/count/reason, current support, per-row quartet route and global fallback reason |
| Predictions | key=(area,target_month,H), origin, truth, q_raw/q_star/phase/class, matched pooled quartet, persistence phase/q3/source/age, cohort flags and route/model references |
| Reporting | full main and separate 2026 metric/delta/coverage tables, fixed confusion axes, NA reasons, 2000 bootstrap multiplicities/deltas/CI rules and independent recomputation |
| Release | exact code/foundation/run identity, complete artifact inventory with paths/hashes, commands/test/replay results, task progress, limitations, audit run/job/base/completion identities |

## Scope-specific guideline precedence

`design.md` exceeds the automatic context injection byte limit. Claude must
explicitly read the complete file before implementing; injected truncation is
not permission to ignore later sections. Read referenced research contracts
when implementing their corresponding stage.

Repository backend guidelines provide key/provenance/output discipline. Their
ETH fs1=4/88-feature/imputation/SMOTE and FEWS interruption/59-month/Stage2
contracts do **not** override this task's actual H1/3/6/12, rich561, native NaN,
36-month and no-Stage2 contracts. Missing old audit evidence is never reported
as acceptance of the new experiment. Do not copy a legacy runner wholesale.
