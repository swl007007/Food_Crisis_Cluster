# Historical local-adoption calendar: accepted contract

Read-only source inspection during grill. No history refits or gate computation.
Accepted by the user and recorded as R37 in draft v0.31. Implementation remains unapproved.

## Source facts

- `FEWSNETGeoXGBExperiment/src/experiment/plan.py:74-81` sets six historical dates
  and the local/gate support floors already adopted with IPCCH-specific changes.
- `src/experiment/availability.py:189-194` under that package chooses up to six
  latest globally observed target months U<O visible at current O; dates are not
  restricted to the current fitting window.
- `src/experiment/stage3.py:424-457` uses V=U-H for each historical refit, applies
  the current map's region members, and retains global predictions when local
  support fails. `:330-365` pools region gate keys and requires supported local
  fits on at least three validation dates.
- That source uses different horizons, release-aware availability and a 59-month
  fitting window. These do not supersede the adopted IPCCH H, month-end availability
  and inclusive 36-month fitting-window contracts.

## Accepted IPCCH calendar and score

1. For each current (H,O), select up to six latest distinct observed target months
   U<O from the complete valid population ledger. Selection is global across
   areas, not separately optimized for each region; no score-based date selection.
   These are observed months, not necessarily six consecutive calendar months.
2. For each U, set V=U-H. Fit same-source global and eligible local four-regressor
   combinations using target months [V-35,V], with features at every fitting row's
   own origin. Predict U with features as of V. Current O's later feature/history
   values cannot be used to rebuild these historical predictor rows.
3. Keep all matched region validation keys, including dates whose local support
   fails (local-routed prediction then equals that date's global prediction).
   Pool confusion counts across the selected dates; do not average monthly F1.
4. Apply the adopted structural, class and successful-local-date support floors,
   undefined-metric rules and strict gain>.01. Fewer than six dates may be used,
   but insufficient gate support keeps global. Do not search further back merely
   to make a region pass. There is no extra current-window cap on candidate U.
5. The map and selected configuration are the current frozen development versions.
   They may have been learned using outcomes after an early V/U. Therefore these
   are map/configuration-conditional historical refits, not independent evaluation
   of a complete pipeline available at V. This gate only guides adoption at O;
   current target T's truth is excluded, and O's information boundary is respected.
6. R40 subsequently fixed required global eligibility: nonempty legal pools fit;
   empty required global pools stop the run as incomplete, without dropping gate
   dates. R41 fixes numerical failures as technical errors that stop the affected
   run; they are not silently converted into better-scoring gate pools.

R37 freezes the historical calendar and pooled-F1 gate score; global support was
separately adopted under R40. Neither decision authorizes model training or new
hyperparameter searches on main-period outcomes.
