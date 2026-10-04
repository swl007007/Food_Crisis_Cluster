# Fitting and prediction failures: evidence and accepted R41

Planning-only source inspection, 2026-10-04. R40 global support was accepted in
v0.34; the error policy below is accepted as R41 in v0.35. No fitting or experiments.

## Focused current-source evidence

Let E denote `FEWSNETGeoXGBExperiment/`.

- `E/src/model/train_branch.py:31-53` uses fit=False for an unsupported/capped
  child, preserving the parent checkpoint and prediction. Eligible-child train,
  predict and save calls have no local catch converting exceptions to fallback.
- `E/src/experiment/stage3.py:435-457,474-485` substitutes same-fold global when
  local support/adoption fails. The eligible local fit/predict calls in those
  branches likewise have no catch that turns a numerical exception into a normal
  gate rejection. This is a scoped observation of these call sites, not an audit
  of every possible outer command wrapper.
- `E/src/model/native_xgb.py:177-199,213-254` calls xgb.train directly and raises
  on booster round/axis or frozen-prefix inconsistencies. Its backend is still
  classification; its integer target casts and four-class assumptions must change
  for the new regressors, while preservation of each global prefix remains required.
- `native_xgb.py:157-174` checks predicted class-array shape but does not explicitly
  assert all returned entries are finite. New regression-output finite checks
  therefore cannot be claimed to exist merely by reusing this helper.
- `native_xgb.py:35-49` treats input-feature NaN as missing and normalizes feature
  infinity to NaN. This input policy must not be repurposed as output sanitization.
  R21 already permits missing feature history; missing inputs are distinct from
  nonfinite predicted shares.

## Accepted unified technical-error boundary

Normal model outcomes continue to follow accepted policies: insufficient local
support, failed/undefined F1 gain, or an expected unmapped area use parent/global;
valid constant targets fit; permitted feature NaN remains missing; undefined
reporting metrics remain NA. Finite raw shares outside [0,1] or out of cumulative
order still use the accepted bounded isotonic projection.

For Stage1 search/configuration evaluation and Stage3 historical/current fits,
stop the affected run and mark it incomplete when any attempted required
global/local fit or prediction raises, any q2..q5 raw prediction is NaN/Inf,
prediction shape/key alignment is invalid, projection fails, or a required model,
feature-schema or map artifact has conflicting identity/corruption. A detected
global-prefix preservation violation is also a technical error.

Record stage, H, origin/candidate/region where applicable, target regressor,
affected keys or key-artifact reference, and the failure reason. Keep completed
evidence as partial, not a completed run. Do not convert failed local computation
to a successful global fallback, mix individual local/global targets, fill output
NaN/Inf with constants, drop rows/dates/candidates to finish scoring, or continue
with changed seeds/capacity/window until a configuration happens to succeed.

This creates an explicit finite-output check before projection/decoding and
preserves the planned search/evaluation population. The trade-off is that a local
technical fault can interrupt the run rather than returning a global prediction;
the underlying problem must be resolved before that run can count as complete.
Retry/resume mechanics do not authorize changing the frozen scientific recipe.

This policy does not broaden source QC repair, prohibit already allowed feature
NaN or normal metric NA, or override R36 expected unmapped routing and R39's
disclosed upstream geometry uncertainty. No new model code is implemented here.
