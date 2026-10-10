"""Report from saved keyed predictions only (design sections 11-12; D22-D24, D28). Nothing here fits.

Per H, on identical main keys (E_all): seven classification scores + projected/raw q3 R^2 for B0/B1/B2/G, the
contrasts G-B2 (primary), G-B1, G-B0, B1-B0, B2-B1, annual G-B2 signs, D23 guards and the D24 paired country
bootstrap (fixed four-H family even under B2 fallback). Persistence and every model are scored on E_persist
(lawful persistence available) with coverage shown. 2026 supplement, country/year and AlphaEarth-coverage subsets
are descriptive. Each H gets its own status; there is no all-H success gate.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_alphaearth import evaluation as ev
from ipcch_alphaearth.artifacts import read_json, record_incomplete, sha256_file, write_csv_gz, write_json
from ipcch_alphaearth.errors import ContractError

ARMS = ("B0", "B1", "B2", "G")
CONTRASTS = (("G", "B2"), ("G", "B1"), ("G", "B0"), ("B1", "B0"), ("B2", "B1"))


def frame_panel(frame: pd.DataFrame, arm: str) -> dict:
    if arm == "persistence":
        return ev.panel(frame["phase_truth"], frame["persistence_phase"], frame["q3"], frame["persistence_q3"], frame["persistence_q3"])
    return ev.panel(frame["phase_truth"], frame[f"{arm}_phase"], frame["q3"], frame[f"{arm}_q3_star"], frame[f"{arm}_q3_raw"])


def delta_row(a: dict, b: dict) -> dict:
    fa, fb = ev.flat(a), ev.flat(b)
    return {k: (None if fa[k] is None or fb[k] is None else fa[k] - fb[k]) for k in fa if k != "n"}


def run_report(ws_dir: Path, contract: dict, horizons: list) -> dict:
    out = Path(ws_dir) / "report"
    if (out / "COMPLETE.json").exists():
        raise ContractError("report is COMPLETE; outputs are immutable")
    if not (Path(ws_dir) / "predict" / "COMPLETE.json").exists():
        raise ContractError("prerequisite stage 'predict' is not COMPLETE")
    out.mkdir(exist_ok=True)
    try:
        return _report(Path(ws_dir), out, contract, horizons)
    except Exception as error:
        record_incomplete(ws_dir, "report", {"stage": "report"}, error)
        raise


def _report(ws_dir: Path, out: Path, contract: dict, horizons: list) -> dict:
    t0 = time.time()
    result = {"family_size": len(horizons), "horizons": {}, "metric_rows": None}
    if len(horizons) != int(contract["evaluation"]["bootstrap"]["family_size"]) and not contract.get("_scaled_test_contract"):
        raise ContractError("the D24 family must contain all four H")
    rows = []
    for h in horizons:
        pred = pd.read_csv(ws_dir / "predict" / f"predictions_h{h:02d}.csv.gz", float_precision="round_trip")
        main = pred[pred["period"] == "main"].reset_index(drop=True)
        supp = pred[pred["period"] == "supplement"].reset_index(drop=True)
        panels = {arm: frame_panel(main, arm) for arm in ARMS}
        years = {}
        for y, part in main.groupby(main["target_ord"] // 12):
            years[int(y)] = {arm: frame_panel(part, arm) for arm in ARMS}
        boot, draws = ev.paired_bootstrap(main, "G_phase", "B2_phase", contract)
        if draws is not None:
            write_csv_gz(out / f"bootstrap_draws_h{h:02d}.csv.gz", draws)
        decision = ev.horizon_decision(main, panels, years, boot, contract)
        persist = main[main["persistence_available"] == 1]
        p_panels = {arm: frame_panel(persist, arm) for arm in (*ARMS, "persistence")}
        hrec = {"main_keys": int(len(main)), "countries": int(main["country_key"].nunique()),
                "years": sorted(years), "decision": decision,
                "bootstrap": boot,
                "panels": {arm: ev.flat(p) for arm, p in panels.items()},
                "panel_details": {arm: ev.detail(p) for arm, p in panels.items()},
                "contrasts": {f"{a}-{b}": delta_row(panels[a], panels[b]) for a, b in CONTRASTS},
                "annual": {str(y): {arm: ev.flat(p) for arm, p in ps.items()} for y, ps in years.items()},
                "annual_details": {str(y): {arm: ev.detail(p) for arm, p in ps.items()} for y, ps in years.items()},
                "persistence": {"keys": int(len(persist)), "coverage": float(len(persist) / len(main)) if len(main) else None,
                                "panels": {arm: ev.flat(p) for arm, p in p_panels.items()},
                                "panel_details": {arm: ev.detail(p) for arm, p in p_panels.items()},
                                "model_minus_persistence": {arm: delta_row(p_panels[arm], p_panels["persistence"]) for arm in ARMS}},
                "supplement_2026": {"keys": int(len(supp)),
                                    "panels": {arm: ev.flat(frame_panel(supp, arm)) for arm in ARMS} if len(supp) else None,
                                    "panel_details": {arm: ev.detail(frame_panel(supp, arm)) for arm in ARMS} if len(supp) else None,
                                    "note": "descriptive point evidence; cannot rescue a main failure or tune the design"},
                "coverage_subsets": {name: {"keys": int(len(part)), "G-B2": delta_row(frame_panel(part, "G"), frame_panel(part, "B2"))}
                                     for name, part in (("routable", main[main["routable"] == 1]), ("not_routable", main[main["routable"] == 0]))
                                     if len(part)},
                "g_routes": main["G_route"].astype(str).str.split(":").str[0].value_counts().to_dict(),
                "concentration": {"top3_country_share": float(main["country_key"].value_counts().head(3).sum() / len(main)) if len(main) else None,
                                  "year_shares": (main["target_ord"] // 12).value_counts(normalize=True).sort_index().to_dict()}}
        country = []
        for cname, part in main.groupby("country_key"):
            g, b = frame_panel(part, "G"), frame_panel(part, "B2")
            country.append({"country_key": cname, "keys": len(part),
                            **{f"G_{k}": v for k, v in g["counts"].items()}, **{f"B2_{k}": v for k, v in b["counts"].items()},
                            "f1_G": ev.as_float(g["binary.f1"]), "f1_B2": ev.as_float(b["binary.f1"]),
                            "f1_G_minus_B2": ev.as_float(ev.f1_gain(g, b))})
        write_csv_gz(out / f"country_diagnostics_h{h:02d}.csv.gz", pd.DataFrame(country))
        result["horizons"][str(h)] = hrec
        for arm in ARMS:
            rows.append({"horizon": h, "cohort": "main_E_all", "arm": arm, **ev.flat(panels[arm])})
            rows.append({"horizon": h, "cohort": "main_E_persist", "arm": arm, **ev.flat(p_panels[arm])})
        rows.append({"horizon": h, "cohort": "main_E_persist", "arm": "persistence", **ev.flat(p_panels["persistence"])})
    metric_table = pd.DataFrame(rows)
    write_csv_gz(out / "metrics.csv.gz", metric_table)
    result["qualifying_horizons"] = [h for h, r in result["horizons"].items() if r["decision"]["status"] == "protected_stable_benefit"]
    result["statement"] = ("Each H is evaluated separately under the fixed four-H Bonferroni family; any qualifying subset is a "
                           "horizon-specific retrospective finding; nonqualifying H are reported with their failure mode.")
    result["elapsed_seconds"] = round(time.time() - t0, 1)
    write_json(out / "report.json", result)
    write_json(out / "COMPLETE.json", {"report_sha256": sha256_file(out / "report.json")})
    return result
