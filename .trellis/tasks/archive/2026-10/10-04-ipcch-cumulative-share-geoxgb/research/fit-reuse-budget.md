# Model fit reuse and total fit accounting — accepted R48

Planning only, 2026-10-04. The user accepted this policy as R48 / v0.42,
following R47. No model fits, cache writes, raw-data recount or experiments ran.

## Source facts and reuse boundary

`FEWSNETGeoXGBExperiment/src/experiment/stage3.py:230-284` implements a global
booster store. Identity includes H, origin, G recipe and parameters, source
snapshot, fitting-key digest and window, with panel-specific feature/label/weight
and scenario fields added via store_identity. Disk reuse checks requested
identity and the booster byte digest (`:265-269`). Its historical local path
fits eligible local models within each call (`:431-446`); its current local path
fits only after the gate is enabled and support passes (`:470-475`). These call
sites do not cache local fits. This is a classifier implementation; the inspected
store lacks an explicit q2/q3/q4/q5 target identity and is not a reusable quartet
implementation as-is. FEWS windows/scenarios are not adopted by this task.

Under the new fixed-map, fixed-recipe and static-source contracts, the same
(H, historical fitting origin V, target q, region) may be requested by several
outer origins O. Its training inputs can be identical even though each O has a
different historical adoption gate. Reusing the fitted model must not reuse the
older gate decision or its already routed final output as a raw local candidate.

## Accepted R48 first-version policy

1. Within the new experiment's fixed source/code/environment namespace, fit each
   fully identical requested model once and reuse its saved fitted artifact.
   This includes Stage3 global and eligible local models shared by historical
   requests and, when the identity matches, current prediction requests. Stage1
   may share a global quartet between L1 and L2 only for the same H/G/F fitting
   identity; both map searches still start independently from that immutable
   root. Do not share fitted boosters across q targets or H, and do not reuse
   models from completed reference packages or across Stage1/Stage3 fit scopes.
2. Identity must bind stage/fit scope, H, q target, fitting origin/window (or
   Stage1 F membership), exact ordered original fitting keys, training feature
   values/order/schema, target values, unit weights, source/availability policy,
   requested and resolved parameters, seed, and code/numerical environment.
   For local fits also bind the frozen map/region membership and the corresponding
   global prefix's model digest. Origin alone or identical hyperparameters alone
   is not sufficient. Global identity excludes L because L does not fit global.
3. A cache hit requires complete, successfully fitted models plus matching
   identity, feature/target schema and model bytes. An absent entry is fitted
   lawfully when requested; a conflicting or corrupt claimed entry stops under
   R41 rather than being silently overwritten or treated as global fallback.
   Reusing one scalar artifact does not permit partial-quartet prediction: all
   four corresponding models and their joint route must be available and valid.
   Local continuation starts from a fresh load/copy of its global prefix; reuse
   cannot mutate the global or accumulate another region's local increment.
4. At each current O, select R37's historical dates and recompute regional gate
   support, confusion counts and adoption using that O's allowed keys. Historical
   candidates use saved raw global/local models' predictions under their own V;
   an older final route is not the candidate. No carried-forward adoption flag,
   expanded time window, extra dates or missing-date substitution is allowed.
   Re-evaluating the gate needs predictions and arithmetic, not a new fit when
   every model identity is already available.
5. Maintain a model-artifact ledger and request-to-artifact references, separating
   requested uses, unique successful scalar fits, quartet requests, cache hits,
   support/gain fallbacks and technical failures. Save keys and identity evidence
   needed to reproduce each fit/use. A reused model counts once toward physical
   fitting work but each scientific use retains its H/date/region/gate provenance.
6. Execute only the finite R43-R47 recipe/tree/fold envelope in a later authorized
   run. Do not add seeds, maps, retries, candidate refits or folds to improve the
   result. Do not stop after an arbitrary first-N fit quota and call partial
   coverage complete. Before execution, enumerate the available fold/date keys;
   after maps are frozen, enumerate supported model requests and report the
   exact unique planned count. Actual support/gate outcomes can reduce that
   count; unexpected extra fitting requires explaining the discrepancy, not
   silently expanding the frozen design. No wall-clock or runtime claim is made.

## Derived ceilings, not empirical fit counts

All counts below refer to one completed scientific run with no extra retries,
parameter searches or model refits for uncertainty estimation. A quartet means
four independent scalar regressions. Saving, scoring, projection, support checks
and recomputing gate decisions are not fitting operations.

| Component | Conservative scalar fitting ceiling |
|---|---:|
| Stage1 unique global roots: 4 H × 4 G × 4 q, shared across L | 64 |
| Stage1 local search: 32 recipes × 15 parents × 2 children × 4 q | 3,840 |
| Stage1 subtotal with accepted root reuse | 3,904 |
| One nonempty current Stage3 fold: 7 dates × (1 global + 16 local) × 4 q | 476 |
| Main Stage3, before any Stage3 reuse: 122 folds × 476 | 58,072 |
| 2026 supplementary Stage3, before reuse: at most 48 folds × 476 | 22,848 |
| Stage3 total before reuse | 80,920 |
| Stage1 + Stage3 with root reuse, before any Stage3 reuse | 84,824 |

The 7 dates mean at most 6 historical replay dates plus the current date. R45
limits a frozen binary membership tree to 16 terminal regions; root-only maps
need no local models. An unsupported historical region uses that fold's global,
and a current region fits only if support and its gate allow adoption. Empty
current evaluation folds request no fits under R47. All these reduce work.
Stage3 model reuse further reduces the ceiling; no actual reduction percentage
has been measured. The 122/48 folds are totals across four H, not per-H counts.

Without even the Stage1 root reuse, the previous ceiling is 3968+80920=84888.
The new arithmetic does not supersede the unchanged R45 local-search envelope;
it merely separates repeated identical roots from physical fits. Neither number
is a requirement to use all fits or authorization to launch them. Future tests
or an explicitly approved numerical replay have their own declared verification
work; they cannot add scientific candidates to this run.
