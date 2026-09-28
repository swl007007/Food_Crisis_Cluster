# Four-class baseline design — approved v0.9

Approved 2026-09-28, including D1-D23 and exact feature/technical defaults.
Execution is handed to the bound Claude executor; see implement.md for lifecycle.

## Boundary and reuse

Use a new `FEWSNETFourClassBaseline/` package derived from the hash-verified
GeoRFBaseline v0.1.0 ZIP. Preserve its app/src/scripts layout and three-stage
architecture. Record release payload hashes and every changed file. Never import
root src/config or copy IPCCH/ETH model defaults into the isolated package.
Keep original package/data/results immutable. Use installed release dependencies
and pinned Windows runtime. No plugin framework, estimator registry or scheduler.

Use one small preparation entrypoint for aligned feature snapshots and outcome/
expert ledgers, and one reporting entrypoint for keyed comparisons/bootstrap.
Otherwise adapt the inherited entrypoints/helpers in place within the new package;
do not create a second parallel model implementation. Exact naming can follow
existing module conventions without changing the scientific contract.

## Preparation

Pin source identities from research/release-and-integration.md; verify complete
shapefile sidecars and canonical area/country mapping. Preserve full monthly
scaffold, validate keys and raw targets, then build feature-contract.md's manifest
at each row's own origin. Attach labels separately and preserve dated histories.
Aligned X is final: bypass old row shifts, automatic dynamic detection, whole-panel
imputation and generated dummy discovery. Current outcome never enters X.

Separate true source observations, engineered predictor values, fitting masks
and evaluation baseline masks. Synthetic rows are fitting-only. Metadata is never
selected by a numeric-column wildcard. Save exact feature names and provenance.

## Stage 1

Schedule all 2018-2020 months and fs1/fs2/fs3; empty target months are recorded
without fitting. Train [O-35,O), retaining the original within-area .20 validation
split (seed42, singleton stays in training), geometry and branch-depth defaults.
The inherited test-group restriction on fitting membership must be recorded.
Use four-class RFs with fixed probability axes and per-estimator transforms.

Apply D8's four-column parent-normalized scan; guard zero exposure/error and
preserve floating masses. Remove binary class1 support assumptions. Reject empty
train/validation candidate children; do not require all four classes in every
branch. Evaluate parent and three inherited child/parent combinations on identical
parent validation keys using exact fixed-four macro F1. Parent wins ties; D7's
strict >.01 gate applies at every depth, including root. Rational count arithmetic
can resolve boundary comparisons. No legacy class1 significance path may select
a multiclass branch. Save branch model/imputer lineage and actual decisions.

The separate monthly held-out scores exported by Stage1 feed Stage2. They are
NOT the within-window split-validation scores. Compute partitioned and pooled
macro F1 on identical monthly keys, all no later than2020-12. Ensure final
contiguity assignments, prediction routing and exported correspondence agree;
conflicting terminal assignments for one area fail instead of taking first match.

## Stage 2

Carry explicitly named macro_f1/macro_f1_base columns through merge, linked-table
and similarity consumers. D9 gives logit-gain weights. Use the general consensus
pool over fs1-3; month-specific consensus is not proposed for the initial run.
Retain release spatial weighting sigma5 degrees, k40, recommended cluster-count
selection and spectral seed42. Retain scoped-universe behavior and report actual
in/out-of-scope areas; validate source codes against release assumptions0..5717.
Do not silently import IPCCH geographic donor rules. Preserve inherited connected
component handling and disclose actual assignment routes.

Require a complete candidate ledger before consensus. Apply D15 only if all
successfully computed weights are zero. Save this null-consensus route explicitly;
do not manufacture weights or a learned partition map. Missing artifacts fail.

## Stage 3

Use D10 schedules, D11 fitting windows and the same aligned feature schema as
Stage1. With a learned consensus, refit pooled and eligible local RF models on
real labels irrespective of expert support. On D15's all-zero route, fit only
the pooled RF, then reuse its predictions for the partitioned arm; record reuse
explicitly and count no extra partitioned fit.
Retain release RF100 trees, depthNone, seed5, no class weights/SMOTE; no proposed
parameter search or threshold calibration. Hard prediction is fixed-axis argmax,
ties to the first class, matching ordinary RF predict behavior. All-missing
feature columns remain. A single-class pooled fit returns that class; an empty
required fitting pool with nonempty test is an incomplete run, not a later-data fit.

D17 local failures route to the same fold's pooled model. Unmapped-area coverage
must meet the inherited2% gate; permitted unmapped rows use pooled fallback.
D15's global fallback bypasses learned-map coverage because no map is claimed.
Record coverage and each routing reason; fewer local models is not hidden success.

Expert and persistence are calendar joins on the raw phase ledger under D1/D3/D5.
Do not inherit ETH alignment, record shifting or missing->0 conversions.
D13 defines main/supplementary cohorts; model failures cannot change those keys.

## Reporting and reproducibility

Report every horizon, every class, all confusion counts and support/coverage.
Report macroF1/accuracy/category-step MAE; per-country/year tables are descriptive.
Use seed42 and2000 paired country-bootstrap draws shared across contrasts and
horizons, sampled from the sorted country union. Recompute weighted confusion
counts and fixed-four macroF1. Empty required cohorts invalidate the whole draw;
save rejected draws/reasons, cap attempts at20000, use linear2.5/97.5 percentiles.
Failure to obtain2000 valid draws is incomplete interval evidence. Missing a
class is handled by D6, not itself a rejected draw. Disclose conditioning.

Per-horizon intervals are marginal/descriptive; do not promote a favorable cell
to an overall superiority claim or call intervals simultaneous. No +.01 final
success floor is proposed: .01 governs partition splitting only. Complete negative
results are valid. Previously inspected historical years are retrospective.

Use fresh runs/<id>/ with source/runtime/schema manifests, target/history ledgers,
fold membership, branch/consensus decisions, fitted models/transforms, keyed
predictions and bootstrap multiplicities. Recompute reports from saved predictions;
replay first/last supported folds per horizon under the pinned runtime. Promote
audit-accessible evidence, not only ignored local aggregates. No overwriting a
completed run; exact-input continuation must verify complete identity and outputs.
