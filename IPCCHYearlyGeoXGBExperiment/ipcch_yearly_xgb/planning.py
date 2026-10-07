"""No-fit request inventory (design section 6), reproducing research/fit-enumeration.json.

Uses the same annual schedule and count-based support as the engine but never
fits or predicts. Output fields mirror the planning enumeration so preflight
can compare them exactly.
"""

from __future__ import annotations

import numpy as np

from ipcch_yearly_xgb import schedule
from ipcch_yearly_xgb.sources import Horizon, meets, support


def enumerate_horizon(hz: Horizon, calendar, contract: dict) -> dict:
    first = schedule.parse_month(contract["protocol"]["first_main_target"][str(hz.h)])
    floor = contract["support"]["local_fit"]
    vfloor = contract["support"]["validation"]
    cache: dict = {}

    def fit_support(origin: int, node: str | None):
        key = (origin, node)
        if key not in cache:
            rows = hz.pool_rows(origin)
            if node is not None:
                rows = rows[hz.region_mask(rows, node)]
            s = support(hz.keys.iloc[rows])
            cache[key] = s
        return cache[key]

    globals_, locals_, blocks, main_counts = set(), set(), [], None
    for block in schedule.blocks(calendar, hz.h, first):
        eval_rows = np.sort(np.concatenate([hz.rows_at(int(f["target_ord"])) for f in block.folds]))
        for f in block.folds:
            if len(hz.rows_at(int(f["target_ord"]))) != int(f["eval_keys"]):
                raise ValueError(f"fold {f['fold_id']} eval_keys mismatch")
        if len(eval_rows) == 0:
            continue
        O = block.origin
        globals_.add(O)
        nodes = hz.node[eval_rows]
        current_supported = set()
        for region in sorted(set(nodes) - {""}):
            if meets(fit_support(O, region), floor):
                locals_.add((O, region))
                current_supported.add(region)
        dates = schedule.gate_dates(hz.observed_months, O, contract["protocol"]["historical_gate_max_dates"])
        hist = sorted(int(u) for u in dates)
        pairs: dict = {}
        ok_dates: dict = {}
        for u in hist:
            v = schedule.fit_origin(hz.h, first, u)
            globals_.add(v)
            u_rows = hz.rows_at(u)
            for region in sorted(set(hz.node[u_rows]) - {""}):
                val = u_rows[hz.node[u_rows] == region]
                pairs.setdefault(region, []).append(val)
                if meets(fit_support(v, region), floor):
                    locals_.add((v, region))
                    ok_dates.setdefault(region, set()).add(u)
        gate_supported = set()
        for region, parts in pairs.items():
            rows = np.concatenate(parts)
            s = support(hz.keys.iloc[rows])
            s = {**s, "crisis_keys": int((hz.keys["phase_truth"].to_numpy()[rows] >= 3).sum())}
            s["noncrisis_keys"] = s["keys"] - s["crisis_keys"]
            if meets(s, vfloor) and len(ok_dates.get(region, ())) >= contract["support"]["min_successful_local_dates"]:
                gate_supported.add(region)
        blocks.append({"period": block.period, "year": block.year, "anchor": schedule.month_label(block.anchor),
                       "fit_origin": schedule.month_label(O), "eval_keys": int(len(eval_rows)),
                       "global_fit_keys": fit_support(O, None)["keys"],
                       "diagnostic_keys": int(sum(n in current_supported for n in nodes)),
                       "historical_dates": [schedule.month_label(u) for u in hist],
                       "historical_origins": sorted({schedule.month_label(schedule.fit_origin(hz.h, first, u))
                                                     for u in hist}),
                       "gate_support_regions": sorted(gate_supported),
                       "gate_support_current_keys": int(sum(n in (gate_supported & current_supported) for n in nodes))})
        if block.period == "main":
            main_counts = {"global_quartets": len(globals_), "local_quartets": len(locals_),
                           "scalar_fits": 4 * (len(globals_) + len(locals_))}
    unsupported = [{"fit_origin": schedule.month_label(o), "region": r,
                    "keys_areas_months": [s["keys"], s["areas"], s["target_months"]]}
                   for (o, r), s in sorted(cache.items(), key=lambda it: (it[0][0], it[0][1] or ""))
                   if r is not None and (o, r) not in locals_]
    return {"origins": sorted(schedule.month_label(o) for o in globals_), "global_quartets": len(globals_),
            "local_quartets": len(locals_), "scalar_fits": 4 * (len(globals_) + len(locals_)), "main": main_counts,
            "unsupported_local_requests": unsupported, "blocks": blocks,
            "global_pool_sizes": {schedule.month_label(o): fit_support(o, None)["keys"] for o in sorted(globals_)},
            "local_pool_sizes": {f"{schedule.month_label(o)}/{r}": cache[(o, r)]["keys"] for o, r in sorted(locals_)}}


def compare(result: dict, expected: dict) -> list[str]:
    keys = ("origins", "global_quartets", "local_quartets", "scalar_fits", "main", "unsupported_local_requests", "blocks")
    return [k for k in keys if result[k] != expected[k]]
