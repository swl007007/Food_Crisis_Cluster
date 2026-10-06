"""No-fit request enumeration (design section 9; research/fit-enumeration.json).

Reproduces, from the verified keys/calendar/maps alone, the unique global,
pooled-residual and regional quartets per H, request counts, global pool
sizes and the historical-support eligible main-period keys (design section 8).
Support rules: regional fit >= 500 keys / 50 areas / 6 target months on the
region's [V-35, V] pool; historical validation support over the pooled U keys
plus >= 3 dates with a supported regional fit. Nothing is fitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ipcch_mlp.sources import Horizon, historical_gate_dates, meets, support


def gate_support(pairs: list[tuple[np.ndarray, int, bool]], hz: Horizon, floor: dict, min_dates: int) -> tuple[bool, dict]:
    """Model-independent historical support of one region (copy of the stage3.gate_decision support part)."""
    if pairs:
        rows = np.concatenate([p[0] for p in pairs])
        months = {p[1] for p in pairs}
        dates_ok = sum(1 for p in pairs if p[2])
    else:
        rows, months, dates_ok = np.zeros(0, dtype=np.int64), set(), 0
    phase = hz.keys["phase_truth"].to_numpy()[rows]
    crisis = int((phase >= 3).sum())
    record = {"keys": int(len(rows)), "areas": int(len(np.unique(hz.area[rows]))), "target_months": len(months),
              "crisis_keys": crisis, "noncrisis_keys": int(len(rows) - crisis), "local_fit_dates": int(dates_ok)}
    short = [k for k, v in floor.items() if record[k] < v]
    if dates_ok < min_dates:
        short.append("local_fit_dates")
    return not short, {**record, "short": short}


def enumerate_horizon(hz: Horizon, folds: pd.DataFrame, contract: dict) -> dict:
    sup = contract["support"]
    win = contract["calendar"]["rolling_window_calendar_months"]
    max_dates = contract["calendar"]["historical_gate_max_dates"]
    globals_, hist_reg, cur_reg = set(), set(), set()
    req = {"current_global": 0, "historical_global": 0, "history_regional": 0, "current_regional": 0}
    eligible_main, main_keys, scored = 0, 0, 0
    fit_cache: dict = {}

    def regional_ok(origin: int, node: str) -> bool:
        key = (origin, node)
        if key not in fit_cache:
            rows = hz.region_rows(hz.window_rows(origin, win), node)
            fit_cache[key] = meets(support(hz.keys.iloc[rows]), sup["local_fit"])
        return fit_cache[key]

    for fold in folds.sort_values(["target_ord", "period"]).to_dict("records"):
        target = int(fold["target_ord"])
        origin = target - hz.h
        eval_rows = hz.rows_at(target)
        if len(eval_rows) == 0:
            continue
        scored += 1
        globals_.add(origin)
        req["current_global"] += 1
        pairs: dict = {n: [] for n in hz.regions}
        for u in historical_gate_dates(hz.observed_months, origin, max_dates):
            v = int(u) - hz.h
            globals_.add(v)
            req["historical_global"] += 1
            u_rows = hz.rows_at(int(u))
            for node in hz.regions:
                val = hz.region_rows(u_rows, node)
                if len(val) == 0:
                    continue
                ok = regional_ok(v, node)
                if ok:
                    hist_reg.add((v, node))
                    req["history_regional"] += 1
                pairs[node].append((val, int(u), ok))
        if fold["period"] == "main":
            main_keys += len(eval_rows)
        for node in hz.regions:
            rows_te = hz.region_rows(eval_rows, node)
            if len(rows_te) == 0:
                continue
            if regional_ok(origin, node):
                cur_reg.add((origin, node))
                req["current_regional"] += 1
            passed, _ = gate_support(pairs[node], hz, sup["validation"], sup["stage3_min_successful_local_dates"])
            if passed and fold["period"] == "main":
                eligible_main += len(rows_te)
    pools = [len(hz.window_rows(o, win)) for o in sorted(globals_)]
    regional = hist_reg | cur_reg
    return {
        "scored": scored,
        "unique_global_quartets": len(globals_),
        "unique_pooled_residual_quartets": len(globals_),
        "unique_historical_regional_quartets": len(hist_reg),
        "unique_current_regional_quartets": len(cur_reg),
        "unique_current_only_regional_quartets": len(cur_reg - hist_reg),
        "unique_regional_quartets": len(regional),
        "scalar_fits_per_seed": 4 * (2 * len(globals_) + len(regional)),
        "global_pool_min_rows": int(min(pools)), "global_pool_max_rows": int(max(pools)),
        "main_keys": main_keys, "main_historical_support_eligible_keys": eligible_main,
        "requests_quartets": req,
        "global_origins": sorted(int(o) for o in globals_),
        "global_pool_sizes": pools,
        "regional_pool_sizes": [len(hz.region_rows(hz.window_rows(o, win), n)) for o, n in sorted(regional)],
    }
