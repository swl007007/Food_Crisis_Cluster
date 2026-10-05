"""P6 checkpoint: exact Stage3 request enumeration and distinct-fit budget (no fitting).

Run from IPCCHGeoXGBExperiment/ with the pinned runtime:
    PYTHONPATH=. python <this file> runs/<run-id>

Mirrors ``ipcch_geoxgb.stage3.run_fold`` request logic on the frozen maps and
prepared keys, without fitting or predicting:

* every scored fold (>= 1 valid key at T) requests the current global (H, O);
* with an accepted split, each of the latest <= 6 observed U < O requests the
  historical global (H, V = U - H) and, for every region with validation keys
  at U, a historical local (H, V, region) iff R28 fit support on the region's
  [V-35, V] rows holds -- data-deterministic, hence EXACT;
* a current local (H, O, region) is requested only if the region has current
  keys, its gate is enabled and current fit support holds. Gate support
  (pooled validation floors on truth and >= 3 supported dates) is data-
  deterministic, but the strict F1 gain depends on predictions, so the
  current-local count is an UPPER BOUND.

Identities: a global quartet is determined by (H, fitting origin); a local by
(H, fitting origin, region). Counts are quartets; scalar fits = 4 x quartets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_geoxgb.contract import load_experiment_contract
from ipcch_geoxgb.schedule import historical_gate_dates
from ipcch_geoxgb.stage1 import meets, support


def enumerate_run(run: Path) -> dict:
    contract = load_experiment_contract()
    window = contract["calendar"]["rolling_window_calendar_months"]
    fit_floor = contract["support"]["local_fit"]
    val_floor = contract["support"]["validation"]
    min_dates = contract["support"]["stage3_min_successful_local_dates"]
    calendar = pd.read_csv(run / "prepared" / "fold_calendar.csv")
    out = {"run": str(run), "horizons": {}}
    totals = {"requests": 0, "distinct_global": 0, "distinct_local_exact": 0, "distinct_local_upper": 0}
    for h in contract["calendar"]["horizons_months"]:
        keys = pd.read_csv(run / "prepared" / f"keys_h{h:02d}.csv.gz")
        frozen = json.loads((run / "stage1" / f"frozen_h{h:02d}.json").read_text())
        fmap = pd.read_csv(run / "stage1" / f"frozen_map_h{h:02d}.csv", dtype={"node_id": str})
        regions = {n: set(g["admin_code"].astype(int)) for n, g in fmap.groupby("node_id")}
        months = keys["target_ord"].to_numpy()
        area = keys["admin_code"].to_numpy()
        observed = np.unique(months)

        def region_rows(origin, members):
            rows = np.flatnonzero((months >= origin - window + 1) & (months <= origin))
            return rows[np.isin(area[rows], list(members))]

        fit_ok_cache: dict = {}

        def local_fit_ok(origin, region):
            key = (origin, region)
            if key not in fit_ok_cache:
                fit_ok_cache[key] = meets(support(keys.iloc[region_rows(origin, regions[region])]), fit_floor)
            return fit_ok_cache[key]

        folds = calendar[calendar["horizon_months"] == h]
        req = {"current_global": 0, "historical_global": 0, "historical_local_exact": 0,
               "historical_local_unsupported_fallback": 0, "current_local_upper": 0}
        globals_, locals_exact, locals_upper = set(), set(), set()
        gate_support_pass = 0
        gate_region_folds = 0
        scored = 0
        for fold in folds.to_dict("records"):
            target, origin = int(fold["target_ord"]), int(fold["origin_ord"])
            eval_rows = np.flatnonzero(months == target)
            if len(eval_rows) == 0:
                continue
            scored += 1
            req["current_global"] += 1
            globals_.add(origin)
            if not frozen["accepted_split"]:
                continue
            dates = historical_gate_dates(observed, origin, contract["calendar"]["historical_gate_max_dates"])
            pooled = {r: [] for r in regions}
            ok_dates = {r: 0 for r in regions}
            for u in dates:
                v = int(u) - h
                req["historical_global"] += 1
                globals_.add(v)
                u_rows = np.flatnonzero(months == u)
                for r, members in regions.items():
                    val = u_rows[np.isin(area[u_rows], list(members))]
                    if len(val) == 0:
                        continue
                    pooled[r].append(val)
                    if local_fit_ok(v, r):
                        req["historical_local_exact"] += 1
                        locals_exact.add((v, r))
                        ok_dates[r] += 1
                    else:
                        req["historical_local_unsupported_fallback"] += 1
            test_areas = set(area[eval_rows])
            for r, members in regions.items():
                gate_region_folds += 1
                rows = np.concatenate(pooled[r]) if pooled[r] else np.zeros(0, dtype=np.int64)
                gate_ok = meets(support(keys.iloc[rows]), val_floor) and ok_dates[r] >= min_dates
                gate_support_pass += int(gate_ok)
                if gate_ok and (test_areas & members) and local_fit_ok(origin, r):
                    req["current_local_upper"] += 1
                    locals_upper.add((origin, r))
        distinct_upper_only = locals_upper - locals_exact
        out["horizons"][str(h)] = {
            "winner": frozen["candidate"], "accepted_split": frozen["accepted_split"],
            "terminal_regions": frozen["terminal_regions"], "scored_folds": scored,
            "scheduled_folds": int(len(folds)),
            "requests_quartets": req,
            "requests_total_upper": int(sum(v for k, v in req.items() if k != "historical_local_unsupported_fallback")),
            "gate_region_folds": gate_region_folds,
            "gate_region_folds_with_support": gate_support_pass,
            "distinct_global_quartets_exact": len(globals_),
            "distinct_local_quartets_exact": len(locals_exact),
            "distinct_current_local_quartets_upper_not_shared": len(distinct_upper_only),
            "distinct_scalar_fits_exact": 4 * (len(globals_) + len(locals_exact)),
            "distinct_scalar_fits_upper": 4 * (len(globals_) + len(locals_exact) + len(distinct_upper_only)),
        }
        totals["requests"] += out["horizons"][str(h)]["requests_total_upper"]
        totals["distinct_global"] += len(globals_)
        totals["distinct_local_exact"] += len(locals_exact)
        totals["distinct_local_upper"] += len(distinct_upper_only)
    totals["distinct_scalar_fits_exact"] = 4 * (totals["distinct_global"] + totals["distinct_local_exact"])
    totals["distinct_scalar_fits_upper"] = totals["distinct_scalar_fits_exact"] + 4 * totals["distinct_local_upper"]
    totals["r48_stage3_ceiling_before_reuse"] = contract["reporting"]["fit_ceiling_upper_bound"]["stage3"]
    out["totals"] = totals
    return out


if __name__ == "__main__":
    print(json.dumps(enumerate_run(Path(sys.argv[1])), indent=2))
