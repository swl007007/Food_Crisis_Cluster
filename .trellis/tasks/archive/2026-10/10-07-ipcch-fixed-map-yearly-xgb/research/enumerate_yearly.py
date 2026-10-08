"""Planning-only inventory; stdlib, no model imports, no fitting or prediction.

Run from the repository root. JSON goes to stdout; input files are read-only.
"""
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("IPCCHGeoXGBExperiment/runs/p6-formal-20261004b")


def read_csv(path):
    stream = gzip.open(path, "rt", newline="") if path.suffix == ".gz" else path.open(newline="")
    with stream as handle:
        return list(csv.DictReader(handle))


def month(value):
    return f"{value // 12:04d}-{value % 12 + 1:02d}"


def enumerate_horizon(horizon, calendar):
    keys = read_csv(ROOT / f"prepared/keys_h{horizon:02}.csv.gz")
    rows = [(int(r["admin_code"]), int(r["target_ord"]), int(r["phase_truth"]) >= 3) for r in keys]
    assert len(rows) == 42695 and len({(a, t) for a, t, _ in rows}) == len(rows)
    map_path = ROOT / f"stage1/frozen_map_h{horizon:02}.csv"
    mapping = {int(r["admin_code"]): r["node_id"] for r in read_csv(map_path)}
    assert len(mapping) == 3264
    frozen = json.loads((ROOT / f"stage1/frozen_h{horizon:02}.json").read_text())
    by_month = defaultdict(list)
    for row in rows:
        by_month[row[1]].append(row)
    observed = sorted(by_month)
    folds = [r for r in calendar if int(r["horizon_months"]) == horizon]
    first = min(int(r["target_ord"]) for r in folds if r["period"] == "main")

    def origin(target):
        return first - horizon if target // 12 == first // 12 and target >= first else target // 12 * 12 - horizon

    support_cache = {}

    def fit_support(fit_origin, region):
        identity = (fit_origin, region)
        if identity not in support_cache:
            subset = [r for r in rows if r[1] <= fit_origin and (region is None or mapping.get(r[0]) == region)]
            counts = [len(subset), len({r[0] for r in subset}), len({r[1] for r in subset})]
            support_cache[identity] = counts
        return support_cache[identity]

    global_ids, local_ids, blocks = set(), set(), []
    grouped = defaultdict(list)
    for fold in folds:
        grouped[(fold["period"], int(fold["target_ord"]) // 12)].append(fold)
    main_counts = None
    for (period, year), block_folds in sorted(grouped.items()):
        anchor = min(int(f["target_ord"]) for f in block_folds)
        cutoff = origin(anchor)
        eval_rows = []
        for fold in block_folds:
            selected = by_month.get(int(fold["target_ord"]), [])
            assert len(selected) == int(fold["eval_keys"])
            eval_rows.extend(selected)
        if not eval_rows:
            continue
        global_ids.add(cutoff)
        current_regions = {mapping[a] for a, _, _ in eval_rows if a in mapping}
        current_supported = set()
        for region in current_regions:
            if all(n >= floor for n, floor in zip(fit_support(cutoff, region), (500, 50, 6))):
                local_ids.add((cutoff, region))
                current_supported.add(region)
        history = observed[:]
        history = [u for u in history if u < cutoff][-6:]
        pairs, successful_dates = defaultdict(list), defaultdict(set)
        for target in history:
            historical_origin = origin(target)
            assert historical_origin < target < cutoff
            global_ids.add(historical_origin)
            for row in by_month[target]:
                region = mapping.get(row[0])
                if region is not None:
                    pairs[region].append(row)
            for region in {mapping[a] for a, _, _ in by_month[target] if a in mapping}:
                if all(n >= floor for n, floor in zip(fit_support(historical_origin, region), (500, 50, 6))):
                    local_ids.add((historical_origin, region))
                    successful_dates[region].add(target)
        gate_supported = set()
        for region, validation in pairs.items():
            n, areas, dates = len(validation), len({r[0] for r in validation}), len({r[1] for r in validation})
            crisis = sum(r[2] for r in validation)
            if n >= 100 and areas >= 20 and dates >= 3 and crisis >= 20 and n-crisis >= 20 and len(successful_dates[region]) >= 3:
                gate_supported.add(region)
        blocks.append({"period": period, "year": year, "anchor": month(anchor), "fit_origin": month(cutoff),
                       "eval_keys": len(eval_rows), "global_fit_keys": fit_support(cutoff, None)[0],
                       "diagnostic_keys": sum(mapping.get(a) in current_supported for a, _, _ in eval_rows),
                       "historical_dates": [month(u) for u in history],
                       "historical_origins": sorted({month(origin(u)) for u in history}),
                       "gate_support_regions": sorted(gate_supported),
                       "gate_support_current_keys": sum(mapping.get(a) in gate_supported & current_supported for a, _, _ in eval_rows)})
        if period == "main":
            main_counts = {"global_quartets": len(global_ids), "local_quartets": len(local_ids), "scalar_fits": 4*(len(global_ids)+len(local_ids))}
    unsupported = [{"fit_origin": month(o), "region": r, "keys_areas_months": s}
                   for (o, r), s in sorted(support_cache.items(), key=lambda item: (item[0][0], item[0][1] or ""))
                   if r is not None and (o, r) not in local_ids]
    return {"H": horizon, "map_sha256": hashlib.sha256(map_path.read_bytes()).hexdigest(),
            "candidate": frozen["candidate"], "origins": sorted(month(o) for o in global_ids),
            "global_quartets": len(global_ids), "local_quartets": len(local_ids),
            "scalar_fits": 4*(len(global_ids)+len(local_ids)), "main": main_counts,
            "unsupported_local_requests": unsupported, "blocks": blocks}


if __name__ == "__main__":
    calendar = read_csv(ROOT / "prepared/fold_calendar.csv")
    horizons = [enumerate_horizon(h, calendar) for h in (1, 3, 6, 12)]
    assert [r["scalar_fits"] for r in horizons] == [200, 160, 168, 196]
    result = {"method": "No-fit stdlib enumeration from frozen keys/calendar/maps; no gain estimated.",
              "source_run": str(ROOT), "horizons": horizons,
              "total_global_quartets": sum(r["global_quartets"] for r in horizons),
              "total_local_quartets": sum(r["local_quartets"] for r in horizons),
              "total_scalar_fits": sum(r["scalar_fits"] for r in horizons)}
    assert result["total_scalar_fits"] == 724
    print(json.dumps(result, indent=2))
