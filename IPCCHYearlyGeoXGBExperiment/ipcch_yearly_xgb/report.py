"""P2 reporting from saved keyed predictions only (PRD R12, R14; design section 7).

Metric panel, cohort and bootstrap helpers are attributed copies of
ipcch_geoxgb/report.py at 6798df2 (``arm_columns``, ``panel``, ``flat``,
``deltas``, ``country_counts``, ``_f1_rows``, ``bootstrap_delta``: 2000 draws,
default_rng(42) reset per cohort, no redraw/drop, interval only with K >= 2,
a defined point and all draws finite). Arms: annual pooled ``pool``, gated
GeoXGB ``geo``, ungated local ``local`` (eligible keys only), original P6
``p6pool``/``p6geo`` on identical keys and truth, persistence on E_persist.
Gate decisions are counted per region x annual block. Nothing is fitted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_yearly_xgb import metrics, sources
from ipcch_yearly_xgb.artifacts import sha256_file, write_json
from ipcch_yearly_xgb.errors import TechnicalError

DRAWS, SEED, PERCENTILES = 2000, 42, (2.5, 97.5)
GZ = {"method": "gzip", "mtime": 0}
KEY = ["admin_code", "target_ord", "horizon_months"]
SCALARS = (("binary", "accuracy"), ("binary", "precision"), ("binary", "recall"), ("binary", "f1"), ("binary", "f2"),
           ("four_class", "accuracy"), ("four_class", "macro_f1"))


def arm_columns(arm: str) -> dict:
    if arm == "persistence":
        return {"phase": "persistence_phase", "q3_star": "persistence_q3", "q3_raw": "persistence_q3"}
    return {"phase": f"{arm}_phase", "q3_star": f"{arm}_q3_star", "q3_raw": f"{arm}_q3_raw"}


def panel(frame: pd.DataFrame, arm: str) -> dict:
    cols = arm_columns(arm)
    return metrics.metric_panel(frame["phase_truth"].to_numpy(), frame[cols["phase"]].to_numpy(),
                                frame["q3_truth"].to_numpy(), frame[cols["q3_star"]].to_numpy(),
                                frame[cols["q3_raw"]].to_numpy())


def flat(p: dict) -> dict:
    out = {f"{g}.{n}": p[g][n] for g, n in SCALARS}
    out["q3_r2_projected"], out["q3_r2_raw"] = p.get("q3_r2_projected"), p.get("q3_r2_raw")
    return out


def deltas(a: dict, b: dict) -> dict:
    fa, fb = flat(a), flat(b)
    return {k: (None if fa[k] is None or fb[k] is None else fa[k] - fb[k]) for k in fa}


def country_counts(frame: pd.DataFrame, arm: str, countries: list) -> np.ndarray:
    phase = arm_columns(arm)["phase"]
    rows = []
    for c in countries:
        part = frame[frame["country_key"] == c]
        cnt = metrics.crisis_counts(part["phase_truth"].to_numpy(), part[phase].to_numpy())
        rows.append([cnt["tp"], cnt["fp"], cnt["fn"], cnt["tn"]])
    return np.array(rows, dtype=np.int64).reshape(len(countries), 4)


def _f1_rows(counts: np.ndarray) -> np.ndarray:
    tp, fp, fn = counts[:, 0], counts[:, 1], counts[:, 2]
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), np.nan)


def bootstrap_delta(frame: pd.DataFrame, arm_a: str, arm_b: str) -> tuple[dict, pd.DataFrame | None]:
    countries = sorted(frame["country_key"].unique().tolist())
    k = len(countries)
    record = {"countries": countries, "K": k, "draws": DRAWS, "seed": SEED, "rng": "numpy.random.default_rng",
              "percentile_method": "linear", "percentiles": list(PERCENTILES)}
    pa = metrics.exact_f1(metrics.crisis_counts(frame["phase_truth"], frame[arm_columns(arm_a)["phase"]]))
    pb = metrics.exact_f1(metrics.crisis_counts(frame["phase_truth"], frame[arm_columns(arm_b)["phase"]]))
    point = None if pa is None or pb is None else float(pa - pb)
    record["point_delta"] = point
    if k == 0:
        return {**record, "interval": None, "na_reason": "empty cohort"}, None
    ca, cb = country_counts(frame, arm_a, countries), country_counts(frame, arm_b, countries)
    rng = np.random.default_rng(SEED)
    mult = np.zeros((DRAWS, k), dtype=np.int64)
    for d in range(DRAWS):
        mult[d] = np.bincount(rng.integers(0, k, size=k), minlength=k)
    fa, fb = _f1_rows(mult @ ca), _f1_rows(mult @ cb)
    delta = fa - fb
    draws = pd.DataFrame(mult, columns=[f"m::{c}" for c in countries])
    draws.insert(0, "delta", delta)
    draws.insert(0, f"f1_{arm_b}", fb)
    draws.insert(0, f"f1_{arm_a}", fa)
    draws.insert(0, "draw", np.arange(DRAWS))
    defined = int(np.isfinite(delta).sum())
    record.update(defined_draws=defined, undefined_draws=DRAWS - defined,
                  country_counts={arm_a: ca.tolist(), arm_b: cb.tolist()})
    reasons = []
    if k < 2:
        reasons.append("K < 2")
    if point is None:
        reasons.append("point delta undefined")
    if defined < DRAWS:
        reasons.append(f"{DRAWS - defined} undefined replicate deltas")
    if reasons:
        return {**record, "interval": None, "na_reason": "; ".join(reasons)}, draws
    lo, hi = np.percentile(delta, PERCENTILES, method="linear")
    return {**record, "interval": [float(lo), float(hi)], "na_reason": ""}, draws


def cohort(frame: pd.DataFrame, arms: tuple) -> dict:
    if len(frame) == 0:
        return {"n": 0, "status": "empty_cohort"}
    return {"n": int(len(frame)), "areas": int(frame["admin_code"].nunique()), "status": "scored",
            "panels": {a: panel(frame, a) for a in arms}}


# ------------------------------------------------------------ yearly report

def read_predictions(path: Path) -> pd.DataFrame:
    pred = pd.read_csv(path, float_precision="round_trip", dtype={"region": str, "local_provider": str,
                                                                  "pool_provider": str})
    for col in ("region", "local_provider"):
        pred[col] = pred[col].fillna("")
    return pred


def attach_p6(pred: pd.DataFrame, p6: pd.DataFrame) -> pd.DataFrame:
    cols = KEY + ["geo_phase", "pool_phase", "geo_q3_star", "geo_q3_raw", "pool_q3_star", "pool_q3_raw",
                  "phase_truth", "q3_truth"]
    x = p6[cols].rename(columns={"geo_phase": "p6geo_phase", "pool_phase": "p6pool_phase",
                                 "geo_q3_star": "p6geo_q3_star", "geo_q3_raw": "p6geo_q3_raw",
                                 "pool_q3_star": "p6pool_q3_star", "pool_q3_raw": "p6pool_q3_raw",
                                 "phase_truth": "p6_phase_truth", "q3_truth": "p6_q3_truth"})
    m = pred.merge(x, on=KEY, how="left", validate="one_to_one", indicator=True)
    if (m["_merge"] != "both").any() or len(pred) != len(p6):
        raise TechnicalError("annual and P6 prediction keys differ")
    if not ((m["phase_truth"] == m["p6_phase_truth"]).all() and (m["q3_truth"] == m["p6_q3_truth"]).all()):
        raise TechnicalError("truth differs between annual and P6 predictions")
    return m.drop(columns=["_merge", "p6_phase_truth", "p6_q3_truth"])


def gate_category(route: pd.Series) -> np.ndarray:
    r = route.astype(str)
    return np.select([r == "local", r.str.startswith("pool_fallback:gate_support"),
                      r == "pool_fallback:current_fit_support", r.str.startswith("pool_fallback:"),
                      r == "unmapped_area_pool"],
                     ["adopted", "historical_support_rejected", "current_fit_support", "gain_rejected", "unmapped"],
                     default="other")


def horizon_period(pred: pd.DataFrame, gates: list, blocks: list, period: str) -> dict:
    e_all = pred[pred["period"] == period]
    e_persist = e_all[e_all["persistence_available"] == 1]
    cat = gate_category(e_all["route"])
    block_ids = {b["block_id"] for b in blocks if b["period"] == period}
    pg = [g for g in gates if g["block_id"] in block_ids]
    entry = {"coverage": {
        "E_all_keys": int(len(e_all)), "E_all_areas": int(e_all["admin_code"].nunique()),
        "E_persist_keys": int(len(e_persist)), "countries": int(e_all["country_key"].nunique()),
        "routes_keys": {k: int(v) for k, v in pd.Series(cat).value_counts().sort_index().items()},
        "routes_areas": {k: int(e_all.loc[cat == k, "admin_code"].nunique()) for k in sorted(set(cat))},
        "diagnostic_keys": int(e_all["local_eligible"].sum()),
        "decisions_region_block": {"total": len(pg), "with_current_keys": sum(1 for g in pg if g["test_keys"]),
                                   "historical_support": sum(1 for g in pg if g["historical_support"]),
                                   "enabled": sum(1 for g in pg if g["enabled"]),
                                   "adopted": sum(1 for g in pg if g["route"] == "local")},
        "blocks": [{k: b.get(k) for k in ("block_id", "anchor", "fit_origin", "eval_keys", "global_fit_rows",
                                           "weight_sum", "weight_ess", "adopted_regions", "diagnostic_keys")}
                   for b in blocks if b["period"] == period]}}
    if len(e_all) == 0:
        return {**entry, "status": "empty"}
    arms = ("pool", "geo", "p6pool", "p6geo")
    entry["E_all"] = cohort(e_all, arms)
    pa = entry["E_all"]["panels"]
    entry["E_all"]["deltas"] = {"geo_minus_pool": deltas(pa["geo"], pa["pool"]),
                                "pool_minus_p6pool": deltas(pa["pool"], pa["p6pool"]),
                                "geo_minus_p6geo": deltas(pa["geo"], pa["p6geo"])}
    entry["E_persist"] = cohort(e_persist, arms + ("persistence",))
    if len(e_persist):
        pp = entry["E_persist"]["panels"]
        entry["E_persist"]["deltas"] = {f"{a}_minus_persistence": deltas(pp[a], pp["persistence"]) for a in arms}
    entry["label_flips_geo_vs_pool"] = {"crisis": int(((e_all["geo_phase"] >= 3) != (e_all["pool_phase"] >= 3)).sum()),
                                        "phase": int((e_all["geo_phase"] != e_all["pool_phase"]).sum())}
    diag = e_all[e_all["local_eligible"] == 1]
    dentry = {"keys": int(len(diag)), "areas": int(diag["admin_code"].nunique()), "by_gate": {}}
    if len(diag):
        dp = cohort(diag, ("local", "pool", "geo", "p6pool", "p6geo"))
        dentry["all"] = {**dp, "local_minus_pool": deltas(dp["panels"]["local"], dp["panels"]["pool"])}
        dcat = gate_category(diag["route"])
        for name in sorted(set(dcat)):
            part = diag[dcat == name]
            cp = cohort(part, ("local", "pool"))
            dentry["by_gate"][name] = {**cp, "local_minus_pool": deltas(cp["panels"]["local"], cp["panels"]["pool"])}
    entry["ungated_local_diagnostic"] = dentry
    # local vs persistence on exactly the keys that are L-eligible AND persistence-available
    lp = e_all[(e_all["local_eligible"] == 1) & (e_all["persistence_available"] == 1)]
    lentry = {"keys": int(len(lp)), "areas": int(lp["admin_code"].nunique()),
              "share_of_diagnostic_keys": (len(lp) / len(diag)) if len(diag) else None}
    if len(lp):
        lc = cohort(lp, ("local", "pool", "geo", "persistence"))
        lentry.update(lc)
        pl = lc["panels"]
        lentry["deltas"] = {"local_minus_persistence": deltas(pl["local"], pl["persistence"]),
                            "pool_minus_persistence": deltas(pl["pool"], pl["persistence"]),
                            "local_minus_pool": deltas(pl["local"], pl["pool"])}
    entry["local_persistence_matched"] = lentry
    return entry


def run_report(run_dir: Path, contract: dict, p6_loader, predict_name: str = "predict",
               out_name: str = "report") -> dict:
    """``p6_loader(h)`` must read the run's staged P6 comparator predictions."""
    pdir = run_dir / predict_name
    summary = json.loads((pdir / "predict-summary.json").read_text(encoding="utf-8"))
    out = run_dir / out_name
    out.mkdir()
    report = {"stage": "report", "predict_summary_sha256": sha256_file(pdir / "predict-summary.json"),
              "interpretation": ("exploratory (evaluation years already viewed); bundled annual protocol change "
                                 "versus P6; historical gate replay is conditional on through-2022 maps/recipes; "
                                 "pointwise country-cluster bootstrap conditional on saved predictions"),
              "horizons": {}}
    for h, info in summary["horizons"].items():
        hdir = pdir / f"h{int(h):02d}"
        if sha256_file(hdir / "predictions.csv.gz") != info["predictions_sha256"]:
            raise TechnicalError(f"H{h} predictions differ from the summary digest")
        pred = attach_p6(read_predictions(hdir / "predictions.csv.gz"), p6_loader(int(h)))
        if pred.duplicated(KEY).any():
            raise TechnicalError(f"H{h}: duplicated prediction key")
        gates = [json.loads(x) for x in (hdir / "gate_decisions.jsonl").read_text(encoding="utf-8").splitlines()]
        blocks = json.loads((hdir / "block_ledger.json").read_text(encoding="utf-8"))
        hrep = {}
        for period in ("main", "supplementary"):
            entry = horizon_period(pred, gates, blocks, period)
            if period == "main" and "E_all" in entry:
                e_all = pred[pred["period"] == period]
                e_persist = e_all[e_all["persistence_available"] == 1]
                entry["bootstrap"] = {}
                for name, frame, other in (("geo_minus_pool_E_all", e_all, "pool"),
                                           ("geo_minus_persistence_E_persist", e_persist, "persistence")):
                    rec, draws = bootstrap_delta(frame, "geo", other)
                    if draws is not None:
                        path = out / f"bootstrap_h{int(h):02d}_{name}.csv.gz"
                        draws.to_csv(path, index=False, compression=GZ)
                        rec["draws_sha256"] = sha256_file(path)
                    entry["bootstrap"][name] = rec
            hrep[period] = entry
        hrep["predictions_sha256"] = info["predictions_sha256"]
        report["horizons"][h] = hrep
    digest = write_json(out / "report.json", report)
    return {"report_sha256": digest}
