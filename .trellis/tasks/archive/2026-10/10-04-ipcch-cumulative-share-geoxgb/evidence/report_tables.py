"""P6: render the final report.json as markdown tables (read-only).

Run from IPCCHGeoXGBExperiment/:  python <this file> runs/<run-id>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def f(x, d=4):
    return "NA" if x is None else f"{x:.{d}f}"


def delta(point, b):
    if not b:
        return f"{f(point)} (point only; no interval by R49)"
    ci = "NA" if not b.get("interval") else f"[{f(b['interval'][0])}, {f(b['interval'][1])}]"
    return f"{f(b['point_delta'])} {ci} ({b['defined_draws']}) {b.get('na_reason') or ''}".rstrip()


def arm(e, a):
    x = e[a]
    return x["binary"]["f1"], x["four_class"]["accuracy"], x["four_class"]["macro_f1"], x.get("q3_r2_projected")


def main(run: Path) -> str:
    r = json.loads((run / "report" / "report.json").read_text())
    out = []
    for period in ("main", "supplementary"):
        out.append(f"\n### {period}\n")
        out.append("| H | E_all keys / countries | GeoXGB F1 | Pooled F1 | Δ F1 geo−pool [95% CI] (defined/2000) "
                   "| GeoXGB 4-class acc / macroF1 | Pooled 4-class acc / macroF1 | q3 R² geo / pool (projected) |")
        out.append("|---|---|---|---|---|---|---|---|")
        rows2 = ["| H | E_persist keys (coverage) | GeoXGB F1 | Persistence F1 | Δ F1 geo−persistence [95% CI] (defined/2000) "
                 "| GeoXGB 4-class acc / macroF1 | Persistence 4-class acc / macroF1 | GeoXGB q3 R² |",
                 "|---|---|---|---|---|---|---|---|"]
        routes = ["| H | cohort rows | local rows | global fallback | unmapped-area global | gate region-folds enabled / adopted / decisions | cohort areas in map / unmapped |",
                  "|---|---|---|---|---|---|---|"]
        for h, hh in r["horizons"].items():
            p = hh[period]
            cov = p["coverage"]
            ea, ep, bs = p["E_all"], p["E_persist"], p.get("bootstrap")  # R49: main period only
            if ea["status"] == "scored":
                g, po = arm(ea, "geo"), arm(ea, "pool")
                out.append(f"| {h} | {ea['n']} / {cov['countries']} | {f(g[0])} | {f(po[0])} | "
                           f"{delta(ea['delta_geo_minus_pool']['binary.f1'], bs and bs['geo_vs_pool_E_all'])} | "
                           f"{f(g[1])} / {f(g[2])} | {f(po[1])} / {f(po[2])} | {f(g[3])} / {f(po[3])} |")
            else:
                out.append(f"| {h} | {ea['status']} | | | | | | |")
            if ep["status"] == "scored":
                g, pe = arm(ep, "geo"), arm(ep, "persistence")
                rows2.append(f"| {h} | {ep['n']} ({f(cov['persistence_coverage'], 3)}) | {f(g[0])} | {f(pe[0])} | "
                             f"{delta(ep['delta_geo_minus_persistence']['binary.f1'], bs and bs['geo_vs_persistence_E_persist'])} | "
                             f"{f(g[1])} / {f(g[2])} | {f(pe[1])} / {f(pe[2])} | {f(g[3])} |")
            else:
                rows2.append(f"| {h} | {ep['status']} | | | | | | |")
            ro = p["routes"]
            gr = ro["gate_region_folds"]
            routes.append(f"| {h} | {ro['rows']['denominator_cohort_rows']} | {ro['rows']['local']} | "
                          f"{ro['rows']['global_fallback']} | {ro['rows']['unmapped_area_global']} | "
                          f"{gr['enabled']} / {gr['adopted_local']} / {gr['denominator_decisions']} | "
                          f"{ro['areas']['in_learned_map']} / {ro['areas']['unmapped']} |")
        out.append("")
        out += rows2
        out.append("")
        out += routes
    out.append(f"\nInterpretation: {r['interpretation']}")
    return "\n".join(out)


if __name__ == "__main__":
    print(main(Path(sys.argv[1])))
