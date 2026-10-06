"""Markdown tables from a run's report/report.json (stdlib only; reads saved results, fits nothing).

Usage: python scripts/summarize_report.py <run-dir> > summary.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def f(x, nd=4, sign=False):
    if x is None:
        return "NA"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def ci(b):
    if b.get("interval"):
        lo, hi = b["interval"]
        return f"{f(b['point_delta'], sign=True)} [{f(lo, sign=True)}, {f(hi, sign=True)}]"
    return f"{f(b.get('point_delta'), sign=True)} (no CI: {b.get('na_reason')})"


def main(run_dir: Path) -> None:
    rep = json.loads((run_dir / "report" / "report.json").read_text(encoding="utf-8"))
    reps = sorted(rep["replicates"], key=int)
    horizons = sorted({h for r in reps for h in rep["replicates"][r]}, key=int)
    for period in ("main", "supplementary"):
        print(f"\n## {period}\n")
        print("### Crisis F1 on E_all (identical keys)\n")
        print("| H | seed | keys | B | P | G | GeoXGB | pooled XGB | G−P | P−B | B−pooled XGB |")
        print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for h in horizons:
            for r in reps:
                e = rep["replicates"][r][h][period]
                if "E_all" not in e:
                    continue
                p = e["E_all"]["panels"]
                d = e["E_all"]["deltas"]
                print(f"| {h} | {r} | {e['E_all']['n']:,} | " + " | ".join(f(p[a]["binary"]["f1"]) for a in ("base", "pool", "geo", "xgbgeo", "xgbpool"))
                      + f" | {f(d['geo_minus_pool']['binary.f1'], sign=True)} | {f(d['pool_minus_base']['binary.f1'], sign=True)}"
                      + f" | {f(d['base_minus_xgbpool']['binary.f1'], sign=True)} |")
        print("\n### Other E_all metrics (four-class macro-F1 / projected q3 R²)\n")
        print("| H | seed | B | P | G | GeoXGB | pooled XGB |")
        print("|---|---|---|---|---|---|---|")
        for h in horizons:
            for r in reps:
                e = rep["replicates"][r][h][period]
                if "E_all" not in e:
                    continue
                p = e["E_all"]["panels"]
                print(f"| {h} | {r} | " + " | ".join(f"{f(p[a]['four_class']['macro_f1'])} / {f(p[a].get('q3_r2_projected'))}"
                                                     for a in ("base", "pool", "geo", "xgbgeo", "xgbpool")) + " |")
        print("\n### Coverage and routing\n")
        print("| H | seed | keys | unmapped | hist-support rejected | gain rejected | current-support fallback | adopted | support-eligible | L-eligible | crisis flips G vs P |")
        print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for h in horizons:
            for r in reps:
                e = rep["replicates"][r][h][period]
                c = e["coverage"]
                ro = c.get("routes", {})
                if not c["E_all_keys"]:
                    continue
                print(f"| {h} | {r} | {c['E_all_keys']:,} | {ro.get('unmapped', 0):,} | {ro.get('historical_support_rejected', 0):,} | "
                      f"{ro.get('gain_rejected', 0):,} | {ro.get('current_fit_support', 0):,} | {ro.get('adopted', 0):,} | "
                      f"{c.get('historical_support_eligible_keys', 0):,} | {c['local_eligible_keys']:,} | {e['label_flips_geo_vs_pool']['crisis']:,} |")
        print("\n### Ungated regional diagnostic (L vs P on L-eligible keys, crisis F1)\n")
        print("| H | seed | keys | L | P | L−P | adopted L−P (keys) | gain-rejected L−P (keys) | support-rejected L−P (keys) |")
        print("|---|---|---:|---:|---:|---:|---|---|---|")
        for h in horizons:
            for r in reps:
                dg = rep["replicates"][r][h][period].get("ungated_local_diagnostic", {})
                if not dg.get("keys"):
                    continue
                p = dg["all"]["panels"]

                def part(name):
                    b = dg["by_gate"].get(name)
                    return "—" if not b else f"{f(b['local_minus_pool']['binary.f1'], sign=True)} ({b['n']:,})"
                print(f"| {h} | {r} | {dg['keys']:,} | {f(p['local']['binary']['f1'])} | {f(p['pool']['binary']['f1'])} | "
                      f"{f(dg['all']['local_minus_pool']['binary.f1'], sign=True)} | {part('adopted')} | {part('gain_rejected')} | "
                      f"{part('historical_support_rejected')} |")
        if period == "main":
            print("\n### Country bootstrap (main period; 2000 draws, seed 42)\n")
            print("| H | seed | G−P [95% CI] | G−persistence on E_persist [95% CI] | E_persist keys |")
            print("|---|---|---|---|---:|")
            for h in horizons:
                for r in reps:
                    e = rep["replicates"][r][h][period]
                    if "bootstrap" not in e:
                        continue
                    b = e["bootstrap"]
                    print(f"| {h} | {r} | {ci(b['geo_minus_pool_E_all'])} | {ci(b['geo_minus_persistence_E_persist'])} | "
                          f"{e['coverage']['E_persist_keys']:,} |")
        print("\n### Persistence comparison on E_persist (crisis F1 / four-class macro-F1)\n")
        print("| H | seed | B | P | G | GeoXGB | persistence |")
        print("|---|---|---|---|---|---|---|")
        for h in horizons:
            for r in reps:
                e = rep["replicates"][r][h][period]
                if "E_persist" not in e or e["E_persist"].get("n", 0) == 0:
                    continue
                p = e["E_persist"]["panels"]
                print(f"| {h} | {r} | " + " | ".join(f"{f(p[a]['binary']['f1'])} / {f(p[a]['four_class']['macro_f1'])}"
                                                     for a in ("base", "pool", "geo", "xgbgeo", "persistence")) + " |")
    print("\n## Seed mean / range (descriptive)\n")
    print("| period/H | metric | mean | min | max |")
    print("|---|---|---:|---:|---:|")
    for k, v in sorted(rep["seed_mean_range_descriptive"].items()):
        for m, s in v.items():
            print(f"| {k} | {m} | {f(s['mean'])} | {f(s['min'])} | {f(s['max'])} |")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
