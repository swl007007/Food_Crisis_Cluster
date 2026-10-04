# Crisis-F1 scan adaptation: evidence and accepted contract

Read-only source inspection during grill, 2026-10-04. No code changes or experiments.
The following source behavior was read by the parent. The user accepted the
regression scan adaptation in draft v0.26, R32. Implementation remains unapproved.

## Current source

- `FEWSNETGeoXGBExperiment/src/metrics/fourclass.py:214-226`,
  `crisis_scan_masses`: for each spatial group g, D_g=2TP_g+FP_g+FN_g.
  With D=sum_g D_g, it returns one column each of Y_g=D_g/D and A_g=2TP_g/D.
  Current inputs are integer four-class codes collapsed to crisis; the new
  share vectors cannot be passed directly to this classifier-specific adapter.
- `FEWSNETGeoXGBExperiment/src/partition/partition_opt.py:213-218`,
  `get_c_b`: c_g=Y_g-A_g; b_g=sum(c)*Y_g/sum(Y). Thus c is normalized
  FP+FN mass and b is its expectation under a common parent F1 error fraction.
- Same file `:959-1012`, `scan`: location score is
  c_g*log(rho)+b_g*(1-rho), with rho updated from selected observed/expected
  error mass. Here rho denotes the source's scan variable q, not a population
  cumulative share q2/q3/q4/q5. This is a candidate-search score, not a calibrated p-value.
- `FEWSNETGeoXGBExperiment/src/partition/transformation.py:386-392` skips
  candidate search if total exposure or total error mass is zero.

## User-accepted adaptation

1. On the lawful S search keys for the current parent and H, obtain four parent
   share predictions; apply the already approved bounded isotonic projection and
   >=20% decoder. Crisis truth/prediction is phase>=3.
2. Aggregate TP_g/FP_g/FN_g by spatial group using original keys once each.
   Supply the one-column Y/A statistics above to the scan. Since c_g/Y_g is
   1-F1_g where Y_g>0, the scan uses geographic variation in crisis-F1 error.
3. If D=0 or sum_g(FP_g+FN_g)=0, produce no scan candidate and retain parent.
   Groups with D_g=0 contribute zero scan mass, not invented positive support;
   geometry/routing policies still need their separate contract.
4. A candidate is only a proposal: fitting/validation support and the strict
   crisis-F1 gain gate decide adoption on the prescribed complete comparison keys.
   TN has no F1 scan mass but stays in support counts and evaluation rows.
5. Do not reinterpret q3 as event probability or feed continuous q values into
   the existing argmax/class-code adapter. No new significance claim follows.

The user has chosen crisis-F1 for both outer selection and this candidate-search
statistic. This does not freeze search dates, smoothing/connectivity, iteration
budget or model capacity. The user subsequently omitted consensus Stage2 (v0.28,
R34), so consensus weights are no longer a pending first-version decision.
